"""What ran, where it ran, and whether the conditions changed.

Phase 1 made results durable. A durable result whose machine is unknown is
still not evidence: ROADMAP § 7.0's own FP16 table carries the caveat *"not
P-core-pinned (matching how the FP32 recording was made)"*, and the only reason
anyone knows that is that someone remembered to write it down. This module is
what remembers instead.

Three jobs, and they are deliberately separate:

:func:`discover_machine`
    Read everything knowable about this machine, once, before anything is
    measured. Every probe fails independently and records **why**: a snapshot
    with a missing key is indistinguishable from one taken before that key
    existed, and the difference decides whether a later comparison means
    anything.
:func:`apply_cpu_policy`
    Restrict the process to the cores a benchmark should run on, **and verify
    it took effect**. Asking is not the same as getting.
:func:`verify_environment`
    Compare the machine now against the snapshot taken at the start, and say
    what changed. A display mode change or an unplugged power adapter mid-run
    is not a detail; it is the reason two halves of a run disagree.

**What pinning does and does not buy.** Restricting to performance cores
removes one large source of scheduling variation -- on this i9-14900HX,
comparing the suite to *itself* reported verdicts up to `SLOWER 2.57x` unpinned
and none pinned. It does **not** make timings deterministic. Clock speeds still
move with thermal and power state, other processes still contend for cache and
memory bandwidth, and the GPU is shared with the compositor. Pinning narrows
the distribution; it does not collapse it, and nothing here should be read as
claiming otherwise.

**On frequency.** This module records the *nominal* maximum frequency where
Windows reports one, and nothing else. A single query cannot establish the
frequency a run actually sustained -- that needs repeated sampling across the
measured window, which is :mod:`telemetry`'s job, with an explicit unavailable
state when no provider can supply it. ``GetSystemTimes`` is sometimes reached
for here; it reports how CPU *time* was divided between idle, kernel and user,
and says nothing whatsoever about clock speed.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import dataclasses
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

SCHEMA_VERSION = 1

from ._paths import PACKAGE_PARENT, REPO  # noqa: E402

#: The checkout when there is one; site-packages in an installed wheel.
ROOT = REPO if REPO is not None else PACKAGE_PARENT
PACKAGE = PACKAGE_PARENT / "rapidshot"

IS_WINDOWS = sys.platform == "win32"


def _unavailable(reason) -> dict:
    """The explicit third state, used everywhere a probe can fail.

    Never an empty dict and never a plausible default. "Could not read the
    topology" and "the topology is uniform" lead to different decisions, and a
    snapshot that cannot tell them apart is worse than one with a hole in it,
    because the hole is visible.
    """
    if isinstance(reason, BaseException):
        reason = f"{type(reason).__name__}: {reason}"
    return {"available": False, "reason": str(reason)}


def _probe(fn, *args, **kwargs) -> dict:
    if not IS_WINDOWS:
        return _unavailable(f"{sys.platform} is not Windows")
    try:
        value = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - every probe is best-effort by design
        return _unavailable(exc)
    if value is None:
        return _unavailable("probe returned nothing")
    value.setdefault("available", True)
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CPU topology
# ---------------------------------------------------------------------------

_RELATION_PROCESSOR_CORE = 0
_RELATION_NUMA_NODE = 1
_RELATION_CACHE = 2
_RELATION_PROCESSOR_PACKAGE = 3
_RELATION_GROUP = 4
_RELATION_ALL = 0xFFFF

_CACHE_TYPE = {0: "unified", 1: "instruction", 2: "data", 3: "trace"}


def _logical_processor_information():
    """Parse ``GetLogicalProcessorInformationEx`` for every relation at once.

    Hand-parsed from the byte buffer rather than declared as ctypes structures:
    ``SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX`` is a variable-length record with
    a trailing array, which ctypes cannot express, and the layout is fixed by
    ABI contract.
    """
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetLogicalProcessorInformationEx.argtypes = [
        ctypes.c_int, ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
    length = wintypes.DWORD(0)
    k32.GetLogicalProcessorInformationEx(_RELATION_ALL, None, ctypes.byref(length))
    if not length.value:
        raise OSError("GetLogicalProcessorInformationEx reported no data")
    buffer = (ctypes.c_ubyte * length.value)()
    if not k32.GetLogicalProcessorInformationEx(_RELATION_ALL, buffer,
                                                ctypes.byref(length)):
        raise ctypes.WinError(ctypes.get_last_error())
    return bytes(buffer)


def _group_affinity(raw, offset):
    """GROUP_AFFINITY: KAFFINITY mask (pointer-sized), WORD group, 3 reserved."""
    width = ctypes.sizeof(ctypes.c_size_t)
    mask = int.from_bytes(raw[offset:offset + width], "little")
    group = int.from_bytes(raw[offset + width:offset + width + 2], "little")
    return {"group": group, "mask": mask}, offset + width + 8


def _parse_processor_information(raw):
    """Physical cores, SMT, groups, packages and caches, in one pass."""
    width = ctypes.sizeof(ctypes.c_size_t)
    cores, caches, groups, packages, numa = [], [], [], 0, []
    offset = 0
    while offset + 8 <= len(raw):
        relation = int.from_bytes(raw[offset:offset + 4], "little")
        size = int.from_bytes(raw[offset + 4:offset + 8], "little")
        if size == 0:
            break
        body = offset + 8
        if relation == _RELATION_PROCESSOR_CORE:
            # PROCESSOR_RELATIONSHIP: Flags@0, EfficiencyClass@1, Reserved[20]@2,
            # GroupCount@22, GroupMask[]@24. The 20 reserved bytes are easy to
            # miss and cost nothing visible -- the masks simply read as zero and
            # every core reports no logical processors.
            flags = raw[body]
            efficiency = raw[body + 1]
            count = int.from_bytes(raw[body + 22:body + 24], "little")
            masks, cursor = [], body + 24
            for _ in range(max(count, 1)):
                affinity, cursor = _group_affinity(raw, cursor)
                masks.append(affinity)
            cores.append({
                "efficiency_class": efficiency,
                # LTP_PC_SMT: this physical core presents more than one logical
                # processor. Recorded per core rather than as a machine-wide
                # flag because a hybrid CPU can have SMT on its P-cores and not
                # on its E-cores, which is exactly this machine.
                "smt": bool(flags & 1),
                "logical_processors": sum(bin(m["mask"]).count("1") for m in masks),
                "group_affinity": masks,
            })
        elif relation == _RELATION_CACHE:
            level = raw[body]
            associativity = raw[body + 1]
            line_size = int.from_bytes(raw[body + 2:body + 4], "little")
            cache_size = int.from_bytes(raw[body + 4:body + 8], "little")
            # CACHE_RELATIONSHIP: Type@8 is a 4-byte enum, then Reserved[18],
            # GroupCount@30, GroupMask@32.
            cache_type = raw[body + 8]
            affinity, _ = _group_affinity(raw, body + 32)
            caches.append({
                "level": level, "type": _CACHE_TYPE.get(cache_type, cache_type),
                "size_bytes": cache_size, "line_size": line_size,
                "associativity": None if associativity == 0xFF else associativity,
                "group_affinity": affinity,
            })
        elif relation == _RELATION_GROUP:
            # GROUP_RELATIONSHIP: MaximumGroupCount@0, ActiveGroupCount@2,
            # Reserved[20]@4, GroupInfo[]@24.
            active = int.from_bytes(raw[body + 2:body + 4], "little")
            cursor = body + 24
            for _ in range(active):
                # PROCESSOR_GROUP_INFO: MaximumProcessorCount, ActiveProcessorCount,
                # 38 reserved bytes, then the active mask.
                maximum, current = raw[cursor], raw[cursor + 1]
                mask = int.from_bytes(raw[cursor + 40:cursor + 40 + width], "little")
                groups.append({"maximum_processors": maximum,
                               "active_processors": current, "active_mask": mask})
                cursor += 40 + width
        elif relation == _RELATION_PROCESSOR_PACKAGE:
            packages += 1
        elif relation == _RELATION_NUMA_NODE:
            numa.append(int.from_bytes(raw[body:body + 4], "little"))
        offset += size
    return cores, caches, groups, packages, numa


def _cpu_sets():
    """``GetSystemCpuSetInformation``: efficiency class and scheduling state.

    Deliberately carries no frequency. ``SYSTEM_CPU_SET_INFORMATION`` has no
    frequency field -- an earlier draft here read one out of the bytes where
    ``AllocationTag`` lives and got zeros -- and even a real nominal figure
    would say nothing about what a core ran at during a benchmark. Nominal
    clocks come from :func:`_nominal_frequencies`, labelled as nominal;
    sustained frequency is :mod:`telemetry`'s problem and has an explicit
    unavailable state there.
    """
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.c_void_p, ctypes.c_ulong]
    me = k32.GetCurrentProcess()
    needed = ctypes.c_ulong(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(needed), me, 0)
    if not needed.value:
        raise OSError("GetSystemCpuSetInformation reported no data")
    buffer = (ctypes.c_ubyte * needed.value)()
    if not k32.GetSystemCpuSetInformation(buffer, needed.value, ctypes.byref(needed),
                                          me, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    raw, sets, offset = bytes(buffer), [], 0
    while offset + 32 <= len(raw):
        size = int.from_bytes(raw[offset:offset + 4], "little")
        if size == 0:
            break
        if int.from_bytes(raw[offset + 4:offset + 8], "little") == 0:  # CpuSetInformation
            sets.append({
                "id": int.from_bytes(raw[offset + 8:offset + 12], "little"),
                "group": int.from_bytes(raw[offset + 12:offset + 14], "little"),
                "logical_processor_index": raw[offset + 14],
                "core_index": raw[offset + 15],
                "last_level_cache_index": raw[offset + 16],
                "numa_node_index": raw[offset + 17],
                "efficiency_class": raw[offset + 18],
                # AllFlags bit order is Parked, Allocated, AllocatedToTarget,
                # RealTime -- not the order the field names suggest.
                "parked": bool(raw[offset + 19] & 0x01),
                "allocated": bool(raw[offset + 19] & 0x02),
                "allocated_to_target_process": bool(raw[offset + 19] & 0x04),
                "scheduling_class": raw[offset + 20],
            })
        offset += size
    return sets


def _nominal_frequencies() -> dict:
    r"""Per-logical-processor nominal clock, from the registry, in MHz.

    ``HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\<n>`` is what
    Windows recorded for each processor at boot. It is a **nominal** figure and
    is labelled as one everywhere it appears: it is not what any core ran at
    during a benchmark, and reading it a second time will not make it so.

    Recorded per processor index rather than once for the machine, because a
    hybrid CPU's classes have different clocks and an efficiency class is a
    scheduling hint rather than a guarantee that its members are identical.
    """
    try:
        import winreg
    except ImportError as exc:  # pragma: no cover - Windows only
        raise OSError("winreg unavailable") from exc
    frequencies = {}
    base = r"HARDWARE\DESCRIPTION\System\CentralProcessor"
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base) as root:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                with winreg.OpenKey(root, name) as key:
                    frequencies[int(name)] = winreg.QueryValueEx(key, "~MHz")[0]
            except (OSError, ValueError):
                continue
    return frequencies


def current_affinity() -> dict:
    """This process's affinity mask and the system's, as the OS reports them."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.GetProcessAffinityMask.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_size_t)]
    process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(k32.GetCurrentProcess(),
                                      ctypes.byref(process_mask),
                                      ctypes.byref(system_mask)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"process_mask": process_mask.value, "system_mask": system_mask.value,
            "process_mask_hex": hex(process_mask.value),
            "logical_processors_allowed": bin(process_mask.value).count("1"),
            # A process launched under an inherited restriction is already
            # measuring a subset of the machine, and without this the recording
            # would look like an unrestricted run.
            "restricted_before_policy": process_mask.value != system_mask.value}


def discover_cpu() -> dict:
    """Topology, counts, caches, groups and the affinity already in force."""
    raw = _logical_processor_information()
    cores, caches, groups, packages, numa = _parse_processor_information(raw)
    sets = _cpu_sets()

    # A registry read failing must not cost the whole topology: nominal clocks
    # are the least important thing here and the most likely to be absent.
    try:
        nominal, nominal_reason = _nominal_frequencies(), None
    except Exception as exc:  # noqa: BLE001
        nominal, nominal_reason = {}, f"{type(exc).__name__}: {exc}"
    by_class = {}
    for entry in sets:
        by_class.setdefault(entry["efficiency_class"], 0)
        by_class[entry["efficiency_class"]] |= 1 << entry["logical_processor_index"]

    if not by_class:
        topology = "unknown"
    elif len(by_class) > 1:
        topology = "hybrid"
    else:
        topology = "uniform"

    classes = []
    for efficiency in sorted(by_class):
        members = [entry for entry in sets if entry["efficiency_class"] == efficiency]
        frequencies = sorted({nominal.get(entry["logical_processor_index"])
                              for entry in members} - {None})
        core_entries = [core for core in cores if core["efficiency_class"] == efficiency]
        classes.append({
            "efficiency_class": efficiency,
            "logical_processors": len(members),
            "physical_cores": len(core_entries),
            "mask": by_class[efficiency],
            "mask_hex": hex(by_class[efficiency]),
            "smt": sorted({core["smt"] for core in core_entries}),
            "nominal_frequencies_mhz": frequencies,
            # An efficiency class is a scheduling hint, not a promise that the
            # cores in it are identical. Where Windows reports different
            # nominal clocks or different SMT within one class, say so rather
            # than letting the class name imply a uniformity nobody checked.
            "homogeneous": len(frequencies) <= 1
                           and len({core["smt"] for core in core_entries}) <= 1,
        })

    info = {
        "topology": topology,
        "processor": platform.processor(),
        "physical_cores": len(cores),
        "logical_processors": sum(core["logical_processors"] for core in cores),
        "smt_present": any(core["smt"] for core in cores),
        "packages": packages,
        "numa_nodes": sorted(set(numa)),
        "processor_groups": len(groups),
        "groups": groups,
        "efficiency_classes": classes,
        "caches": _summarise_caches(caches),
        "affinity": current_affinity(),
        "nominal_frequency_note": (
            nominal_reason or "nominal clocks recorded at boot; not what any "
            "core ran at during a measurement"),
    }
    if len(groups) > 1:
        # SetProcessAffinityMask operates within a single processor group, so a
        # machine with more than one cannot be pinned by the simple path below.
        # Recorded rather than silently half-applied.
        info["multi_group_warning"] = (
            f"{len(groups)} processor groups; a single affinity mask covers only "
            "group 0, so a pinning policy here is partial")
    return info


def _summarise_caches(caches) -> dict:
    levels = {}
    for cache in caches:
        key = f"L{cache['level']}_{cache['type']}"
        entry = levels.setdefault(key, {"instances": 0, "sizes_bytes": set(),
                                        "line_size": cache["line_size"]})
        entry["instances"] += 1
        entry["sizes_bytes"].add(cache["size_bytes"])
    return {key: {"instances": value["instances"],
                  "sizes_bytes": sorted(value["sizes_bytes"]),
                  "line_size": value["line_size"]}
            for key, value in sorted(levels.items())}


# ---------------------------------------------------------------------------
# CPU policy
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CpuPolicy:
    """What was asked for, what was granted, and whether those agree."""

    policy: str
    topology: str
    requested_mask: int = None
    effective_mask: int = None
    applied: bool = False
    reasons: list = dataclasses.field(default_factory=list)

    @property
    def verified(self) -> bool:
        return (self.applied and self.requested_mask is not None
                and self.requested_mask == self.effective_mask)

    def as_dict(self) -> dict:
        return {"policy": self.policy, "topology": self.topology,
                "requested_mask": self.requested_mask,
                "requested_mask_hex": (None if self.requested_mask is None
                                       else hex(self.requested_mask)),
                "effective_mask": self.effective_mask,
                "effective_mask_hex": (None if self.effective_mask is None
                                       else hex(self.effective_mask)),
                "applied": self.applied, "verified": self.verified,
                "reasons": list(self.reasons)}


def performance_core_mask(cpu=None):
    """``(mask, topology)`` for the fastest cores, or ``(None, topology)``.

    Higher efficiency class means faster. A uniform CPU needs no pinning and
    returns ``None`` with ``"uniform"``; a CPU whose topology could not be read
    returns ``None`` with ``"unknown"``, and the difference matters -- claiming
    uniformity that was never observed leaves a hybrid CPU silently unpinned
    while the recording says scheduling was known.
    """
    cpu = cpu if cpu is not None else _probe(discover_cpu)
    if not cpu.get("available", False):
        return None, "unknown"
    topology = cpu.get("topology", "unknown")
    if topology != "hybrid":
        return None, topology
    fastest = max(cpu["efficiency_classes"], key=lambda item: item["efficiency_class"])
    return fastest["mask"], topology


def native_loaded():
    """Which native extension this process actually imports, not which is installed.

    The distribution list a recording carries reports `rapidshot-native` 0.2.0
    even when the in-tree development build is what loads -- it takes
    precedence -- so a recording could credit a wheel that never ran. This asks
    `rapidshot.native` instead, whose `source` field names the route.
    """
    try:
        from rapidshot import native
        return native.build_info() if native.is_available() else None
    except Exception as exc:  # noqa: BLE001 - provenance must not stop a run
        return {"error": f"{type(exc).__name__}: {exc}"}


def apply_cpu_policy(policy: str = "performance", *, cpu=None) -> CpuPolicy:
    """Restrict this process to the chosen cores and verify it took effect.

    **Call this before anything starts a thread pool.** NumPy's BLAS, ONNX
    Runtime and CuPy all size their pools from the affinity they see at
    initialisation, so pinning afterwards leaves a pool built for the whole
    machine running on a subset of it -- which is slower than either choice
    made consistently, and is not what the recording will claim was measured.

    ``policy="none"`` is a first-class configuration, not an absence of one:
    ``--no-pin`` runs are labelled and kept separate rather than pooled with
    pinned ones.
    """
    cpu = cpu if cpu is not None else _probe(discover_cpu)
    topology = cpu.get("topology", "unknown") if cpu.get("available") else "unknown"
    result = CpuPolicy(policy=policy, topology=topology)

    if policy == "none":
        result.reasons.append("pinning disabled by policy; recorded as its own "
                              "configuration, not pooled with pinned runs")
        result.effective_mask = _effective_mask()
        return result

    if policy != "performance":
        result.reasons.append(f"unknown policy {policy!r}")
        result.effective_mask = _effective_mask()
        return result

    if cpu.get("available") and cpu.get("processor_groups", 1) > 1:
        result.reasons.append(cpu.get("multi_group_warning",
                                      "multiple processor groups; cannot pin wholly"))
        result.effective_mask = _effective_mask()
        return result

    mask, topology = performance_core_mask(cpu)
    result.topology = topology
    if mask is None:
        result.reasons.append(
            "uniform CPU; no pinning needed" if topology == "uniform"
            else "CPU topology could not be read; running unpinned and "
                 "recording that, rather than claiming the machine was uniform")
        result.effective_mask = _effective_mask()
        return result

    existing = _probe(current_affinity)
    if existing.get("available") and existing.get("restricted_before_policy"):
        # An inherited restriction is information, not an obstacle: intersect
        # rather than widen, because widening would hand the process cores the
        # caller deliberately withheld.
        narrowed = mask & existing["process_mask"]
        if narrowed != mask:
            result.reasons.append(
                f"process was already restricted to {existing['process_mask_hex']}; "
                f"intersected rather than widened")
            mask = narrowed
        if mask == 0:
            result.reasons.append("inherited affinity excludes every performance "
                                  "core; leaving affinity untouched")
            result.effective_mask = existing["process_mask"]
            return result

    result.requested_mask = mask
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
            raise ctypes.WinError(ctypes.get_last_error())
        result.applied = True
    except Exception as exc:  # noqa: BLE001
        result.reasons.append(f"SetProcessAffinityMask failed: "
                              f"{type(exc).__name__}: {exc}")

    # Read it back rather than trusting the call. A policy that was asked for
    # and not granted must not be recorded as a policy that was in force.
    result.effective_mask = _effective_mask()
    if result.applied and result.effective_mask != mask:
        result.reasons.append(
            f"requested {hex(mask)} but the OS reports "
            f"{hex(result.effective_mask or 0)}")
    return result


def _effective_mask():
    probe = _probe(current_affinity)
    return probe.get("process_mask") if probe.get("available") else None


def verify_affinity(expected_mask=None) -> dict:
    """What a *worker* calls to confirm it got the cores the parent intended.

    Affinity is inherited, so a worker normally needs to do nothing -- but
    "normally" is not a measurement. A worker that silently ran on the whole
    machine while the parent recorded a pinned configuration produces numbers
    that cannot be compared with anything, and nothing else would notice.
    """
    probe = _probe(current_affinity)
    if not probe.get("available"):
        return probe
    probe["expected_mask"] = expected_mask
    probe["matches_expected"] = (expected_mask is None
                                 or probe["process_mask"] == expected_mask)
    probe["pid"] = os.getpid()
    return probe


def thread_pool_sizes() -> dict:
    """How many threads the libraries in this process think they may use.

    Recorded because it is a property of the *run*, not of the machine: the
    same benchmark on the same box measures differently when BLAS decides it
    owns 24 cores and the process is pinned to 8. The environment variables are
    included whether or not they are set, so "left at the default" is visible.
    """
    info = {"cpu_count": os.cpu_count(),
            "process_cpu_count": getattr(os, "process_cpu_count", lambda: None)(),
            "environment": {name: os.environ.get(name)
                            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                         "OPENBLAS_NUM_THREADS",
                                         "NUMEXPR_NUM_THREADS",
                                         "ORT_NUM_THREADS")}}
    try:
        import cv2
        info["opencv_threads"] = cv2.getNumThreads()
    except Exception as exc:  # noqa: BLE001
        info["opencv_threads"] = _unavailable(exc)
    try:
        import onnxruntime
        info["onnxruntime_version"] = onnxruntime.__version__
    except Exception as exc:  # noqa: BLE001
        info["onnxruntime_version"] = _unavailable(exc)
    return info


# ---------------------------------------------------------------------------
# Displays
# ---------------------------------------------------------------------------

_QDC_ONLY_ACTIVE_PATHS = 0x00000002
_DEVICE_INFO_GET_SOURCE_NAME = 1
_DEVICE_INFO_GET_TARGET_NAME = 2
_DEVICE_INFO_GET_ADAPTER_NAME = 4

_ROTATION = {1: 0, 2: 90, 3: 180, 4: 270}
_SCALING = {1: "identity", 2: "centered", 3: "stretched",
            4: "aspect-ratio-centered-max", 5: "custom", 128: "preferred"}
_OUTPUT_TECHNOLOGY = {
    0: "other", 4: "dvi", 5: "hdmi", 10: "displayport-external",
    11: "displayport-embedded", 12: "udi-external", 13: "udi-embedded",
    15: "sdtvdongle", 16: "miracast", 0x80000000: "internal",
}


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

    def as_int(self) -> int:
        return (self.HighPart << 32) | self.LowPart


class _RATIONAL(ctypes.Structure):
    _fields_ = [("Numerator", wintypes.UINT), ("Denominator", wintypes.UINT)]


class _PATH_SOURCE_INFO(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("statusFlags", wintypes.UINT)]


class _PATH_TARGET_INFO(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("outputTechnology", wintypes.UINT),
                ("rotation", wintypes.UINT), ("scaling", wintypes.UINT),
                ("refreshRate", _RATIONAL), ("scanLineOrdering", wintypes.UINT),
                ("targetAvailable", wintypes.BOOL), ("statusFlags", wintypes.UINT)]


class _PATH_INFO(ctypes.Structure):
    _fields_ = [("sourceInfo", _PATH_SOURCE_INFO), ("targetInfo", _PATH_TARGET_INFO),
                ("flags", wintypes.UINT)]


class _2DREGION(ctypes.Structure):
    _fields_ = [("cx", wintypes.UINT), ("cy", wintypes.UINT)]


class _VIDEO_SIGNAL_INFO(ctypes.Structure):
    _fields_ = [("pixelRate", ctypes.c_uint64), ("hSyncFreq", _RATIONAL),
                ("vSyncFreq", _RATIONAL), ("activeSize", _2DREGION),
                ("totalSize", _2DREGION), ("videoStandard", wintypes.UINT),
                ("scanLineOrdering", wintypes.UINT)]


class _TARGET_MODE(ctypes.Structure):
    _fields_ = [("targetVideoSignalInfo", _VIDEO_SIGNAL_INFO)]


class _POINTL(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _SOURCE_MODE(ctypes.Structure):
    _fields_ = [("width", wintypes.UINT), ("height", wintypes.UINT),
                ("pixelFormat", wintypes.UINT), ("position", _POINTL)]


class _DESKTOP_IMAGE_INFO(ctypes.Structure):
    _fields_ = [("PathSourceSize", _POINTL), ("DesktopImageRegion", wintypes.RECT),
                ("DesktopImageClip", wintypes.RECT)]


class _MODE_UNION(ctypes.Union):
    _fields_ = [("targetMode", _TARGET_MODE), ("sourceMode", _SOURCE_MODE),
                ("desktopImageInfo", _DESKTOP_IMAGE_INFO)]


class _MODE_INFO(ctypes.Structure):
    _fields_ = [("infoType", wintypes.UINT), ("id", wintypes.UINT),
                ("adapterId", _LUID), ("mode", _MODE_UNION)]


class _DEVICE_INFO_HEADER(ctypes.Structure):
    _fields_ = [("type", wintypes.UINT), ("size", wintypes.UINT),
                ("adapterId", _LUID), ("id", wintypes.UINT)]


class _TARGET_DEVICE_NAME(ctypes.Structure):
    _fields_ = [("header", _DEVICE_INFO_HEADER), ("flags", wintypes.UINT),
                ("outputTechnology", wintypes.UINT),
                ("edidManufactureId", wintypes.USHORT),
                ("edidProductCodeId", wintypes.USHORT),
                ("connectorInstance", wintypes.UINT),
                ("monitorFriendlyDeviceName", wintypes.WCHAR * 64),
                ("monitorDevicePath", wintypes.WCHAR * 128)]


class _SOURCE_DEVICE_NAME(ctypes.Structure):
    _fields_ = [("header", _DEVICE_INFO_HEADER),
                ("viewGdiDeviceName", wintypes.WCHAR * 32)]


class _ADAPTER_NAME(ctypes.Structure):
    _fields_ = [("header", _DEVICE_INFO_HEADER),
                ("adapterDevicePath", wintypes.WCHAR * 128)]


def _query_display_config():
    """Active display paths, with exact refresh rates and adapter identity.

    ``EnumDisplaySettingsW`` -- what the harness uses today -- reports
    ``dmDisplayFrequency`` as a rounded integer: a 165 Hz panel reads 165 when
    the panel actually presents 164.917. Source pacing is compared against that
    number and the comparison decides whether a throughput figure describes the
    apparatus, so the rounding is not cosmetic. ``QueryDisplayConfig`` returns
    the rational the driver actually programmed.

    It also returns the **adapter LUID** that owns each output, which is what
    makes "capture is on the iGPU and CUDA is on the discrete GPU" a recorded
    fact rather than an inference from a device description string.
    """
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetDisplayConfigBufferSizes.argtypes = [
        wintypes.UINT, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(wintypes.UINT)]
    user32.QueryDisplayConfig.argtypes = [
        wintypes.UINT, ctypes.POINTER(wintypes.UINT), ctypes.POINTER(_PATH_INFO),
        ctypes.POINTER(wintypes.UINT), ctypes.POINTER(_MODE_INFO), ctypes.c_void_p]
    user32.DisplayConfigGetDeviceInfo.argtypes = [ctypes.c_void_p]

    path_count, mode_count = wintypes.UINT(), wintypes.UINT()
    status = user32.GetDisplayConfigBufferSizes(
        _QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_count), ctypes.byref(mode_count))
    if status:
        raise OSError(f"GetDisplayConfigBufferSizes failed with {status}")
    paths = (_PATH_INFO * path_count.value)()
    modes = (_MODE_INFO * mode_count.value)()
    status = user32.QueryDisplayConfig(
        _QDC_ONLY_ACTIVE_PATHS, ctypes.byref(path_count), paths,
        ctypes.byref(mode_count), modes, None)
    if status:
        raise OSError(f"QueryDisplayConfig failed with {status}")

    outputs = []
    for path in paths[:path_count.value]:
        entry = {
            "adapter_luid": path.targetInfo.adapterId.as_int(),
            "target_id": path.targetInfo.id,
            "source_id": path.sourceInfo.id,
            "rotation_degrees": _ROTATION.get(path.targetInfo.rotation,
                                              path.targetInfo.rotation),
            "scaling": _SCALING.get(path.targetInfo.scaling, path.targetInfo.scaling),
            "output_technology": _OUTPUT_TECHNOLOGY.get(
                path.targetInfo.outputTechnology, path.targetInfo.outputTechnology),
        }
        numerator = path.targetInfo.refreshRate.Numerator
        denominator = path.targetInfo.refreshRate.Denominator
        entry["refresh_rate"] = {"numerator": numerator, "denominator": denominator,
                                 "hz": (numerator / denominator) if denominator else None}
        _attach_mode(entry, path, modes, mode_count.value)
        _attach_names(entry, user32, path)
        outputs.append(entry)
    return outputs


def _attach_mode(entry, path, modes, count):
    source_index = path.sourceInfo.modeInfoIdx
    if source_index < count and modes[source_index].infoType == 1:      # SOURCE
        mode = modes[source_index].mode.sourceMode
        entry["source_mode"] = {
            "width": mode.width, "height": mode.height,
            "desktop_left": mode.position.x, "desktop_top": mode.position.y}
    target_index = path.targetInfo.modeInfoIdx
    if target_index < count and modes[target_index].infoType == 2:      # TARGET
        signal = modes[target_index].mode.targetMode.targetVideoSignalInfo
        entry["signal"] = {
            "active_width": signal.activeSize.cx, "active_height": signal.activeSize.cy,
            "total_width": signal.totalSize.cx, "total_height": signal.totalSize.cy,
            "pixel_rate_hz": signal.pixelRate,
            "vsync_hz": (signal.vSyncFreq.Numerator / signal.vSyncFreq.Denominator
                         if signal.vSyncFreq.Denominator else None)}


def _attach_names(entry, user32, path):
    target = _TARGET_DEVICE_NAME()
    target.header.type = _DEVICE_INFO_GET_TARGET_NAME
    target.header.size = ctypes.sizeof(_TARGET_DEVICE_NAME)
    target.header.adapterId = path.targetInfo.adapterId
    target.header.id = path.targetInfo.id
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(target)) == 0:
        entry["monitor_name"] = target.monitorFriendlyDeviceName
        # The device path is the stable identity across reboots and reconnects;
        # the friendly name is not unique when two identical panels are attached.
        entry["monitor_device_path"] = target.monitorDevicePath
        entry["edid_manufacturer_id"] = target.edidManufactureId
        entry["edid_product_code"] = target.edidProductCodeId

    source = _SOURCE_DEVICE_NAME()
    source.header.type = _DEVICE_INFO_GET_SOURCE_NAME
    source.header.size = ctypes.sizeof(_SOURCE_DEVICE_NAME)
    source.header.adapterId = path.sourceInfo.adapterId
    source.header.id = path.sourceInfo.id
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(source)) == 0:
        # The GDI name is what ties this back to EnumDisplaySettingsW and to
        # whichever display a capture backend was pointed at.
        entry["gdi_device_name"] = source.viewGdiDeviceName

    adapter = _ADAPTER_NAME()
    adapter.header.type = _DEVICE_INFO_GET_ADAPTER_NAME
    adapter.header.size = ctypes.sizeof(_ADAPTER_NAME)
    adapter.header.adapterId = path.targetInfo.adapterId
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(adapter)) == 0:
        entry["adapter_device_path"] = adapter.adapterDevicePath


def _dpi_awareness() -> dict:
    """What this process will be told the screen size is, and whether it is true.

    A DPI-unaware process on a scaled display is lied to by every coordinate
    API: ``GetSystemMetrics`` returns the *virtualised* size, so a capture
    region computed from it addresses the wrong pixels and a "1920x1080" run on
    a 2560x1600 panel at 125% is measuring a scaled copy of something else.
    Recorded and checked rather than assumed.
    """
    info = {}
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        user32.GetThreadDpiAwarenessContext.restype = ctypes.c_void_p
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [ctypes.c_void_p]
        context = user32.GetThreadDpiAwarenessContext()
        awareness = user32.GetAwarenessFromDpiAwarenessContext(context)
        info["thread_awareness"] = {0: "unaware", 1: "system", 2: "per-monitor",
                                    3: "per-monitor-v2",
                                    4: "unaware-gdi-scaled"}.get(awareness, awareness)
    except Exception as exc:  # noqa: BLE001
        info["thread_awareness"] = _unavailable(exc)
    try:
        user32.GetDpiForSystem.restype = wintypes.UINT
        info["system_dpi"] = user32.GetDpiForSystem()
    except Exception as exc:  # noqa: BLE001
        info["system_dpi"] = _unavailable(exc)
    return info


def _monitors() -> list:
    """Desktop rectangles and per-monitor DPI, in physical pixels where possible."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        shcore.GetDpiForMonitor.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(wintypes.UINT)]
    except OSError:
        shcore = None

    class MONITORINFOEXW(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32)]

    found = []
    proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.POINTER(wintypes.RECT), ctypes.c_longlong)

    def callback(handle, _hdc, _rect, _data):
        info = MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(MONITORINFOEXW)
        entry = {}
        if user32.GetMonitorInfoW(ctypes.c_void_p(handle), ctypes.byref(info)):
            entry = {
                "gdi_device_name": info.szDevice,
                "primary": bool(info.dwFlags & 1),
                "desktop_rect": [info.rcMonitor.left, info.rcMonitor.top,
                                 info.rcMonitor.right, info.rcMonitor.bottom],
                "work_rect": [info.rcWork.left, info.rcWork.top,
                              info.rcWork.right, info.rcWork.bottom],
            }
        if shcore is not None:
            dpi_x, dpi_y = wintypes.UINT(), wintypes.UINT()
            if shcore.GetDpiForMonitor(ctypes.c_void_p(handle), 0,
                                       ctypes.byref(dpi_x), ctypes.byref(dpi_y)) == 0:
                entry["effective_dpi"] = [dpi_x.value, dpi_y.value]
                entry["scale_percent"] = round(dpi_x.value / 96 * 100)
            if shcore.GetDpiForMonitor(ctypes.c_void_p(handle), 1,
                                       ctypes.byref(dpi_x), ctypes.byref(dpi_y)) == 0:
                entry["raw_dpi"] = [dpi_x.value, dpi_y.value]
        found.append(entry)
        return True

    user32.EnumDisplayMonitors.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                           proc_type, ctypes.c_longlong]
    if not user32.EnumDisplayMonitors(None, None, proc_type(callback), 0):
        raise ctypes.WinError(ctypes.get_last_error())
    return found


def discover_displays() -> dict:
    """Every active output, and whether this process sees real pixels."""
    outputs = _query_display_config()
    monitors = _monitors()
    awareness = _dpi_awareness()

    by_gdi = {m.get("gdi_device_name"): m for m in monitors if m.get("gdi_device_name")}
    for output in outputs:
        match = by_gdi.get(output.get("gdi_device_name"))
        if match:
            output["desktop_rect"] = match.get("desktop_rect")
            output["primary"] = match.get("primary")
            output["effective_dpi"] = match.get("effective_dpi")
            output["scale_percent"] = match.get("scale_percent")

    info = {"count": len(outputs), "outputs": outputs, "dpi": awareness,
            # More than one adapter here is the hybrid-laptop case the whole
            # cross-adapter path exists for, stated as a LUID rather than
            # guessed from a device description string.
            "adapters": sorted({output["adapter_luid"] for output in outputs}),
            "monitors_without_a_path": [m for m in monitors
                                        if m.get("gdi_device_name") not in
                                        {o.get("gdi_device_name") for o in outputs}]}

    # The physical-pixel check. A primary output whose desktop rectangle is
    # smaller than the mode the driver programmed means this process is being
    # handed virtualised coordinates, and every region computed from them
    # addresses the wrong pixels.
    primary = next((o for o in outputs if o.get("primary")), None)
    if primary and primary.get("source_mode") and primary.get("desktop_rect"):
        left, top, right, bottom = primary["desktop_rect"]
        info["physical_pixel_mapping"] = {
            "mode": [primary["source_mode"]["width"], primary["source_mode"]["height"]],
            "desktop": [right - left, bottom - top],
            "matches": (right - left == primary["source_mode"]["width"]
                        and bottom - top == primary["source_mode"]["height"])}
        if not info["physical_pixel_mapping"]["matches"]:
            info["physical_pixel_warning"] = (
                "the desktop rectangle does not match the programmed display mode; "
                "this process is seeing DPI-virtualised coordinates and any region "
                "computed from them addresses the wrong pixels")
    return info


def display_fingerprint(displays) -> str:
    """A stable digest of everything about the displays that can change a run.

    Monitor identity, geometry, rotation, exact refresh and scaling -- not
    handles or enumeration order, which move for reasons nobody cares about.
    Two snapshots with the same fingerprint were taken on the same screens in
    the same configuration, which is what makes a mid-run topology change
    detectable rather than merely suspected.
    """
    if not displays.get("available", True):
        return "unavailable"
    stable = []
    for output in displays.get("outputs", []):
        stable.append({
            "monitor": output.get("monitor_device_path") or output.get("monitor_name"),
            "adapter": output.get("adapter_luid"),
            "mode": output.get("source_mode"),
            "rotation": output.get("rotation_degrees"),
            "refresh": output.get("refresh_rate"),
            "scale": output.get("scale_percent"),
            "primary": output.get("primary"),
        })
    stable.sort(key=lambda item: json.dumps(item, sort_keys=True, default=str))
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Memory, power and the operating system
# ---------------------------------------------------------------------------

def discover_memory() -> dict:
    """Installed and currently available RAM, from the kernel rather than WMI."""
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not k32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    return {"total_physical_mb": round(status.ullTotalPhys / 1e6, 1),
            "available_physical_mb": round(status.ullAvailPhys / 1e6, 1),
            "memory_load_percent": status.dwMemoryLoad,
            "total_pagefile_mb": round(status.ullTotalPageFile / 1e6, 1)}


def discover_power() -> dict:
    """Mains or battery, and which power plan is active.

    A laptop that drops off mains mid-run throttles, and the second half of the
    run then measures a different machine from the first. Recording the state
    at the start is what makes :func:`verify_environment` able to notice.
    """
    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", wintypes.BYTE), ("BatteryFlag", wintypes.BYTE),
                    ("BatteryLifePercent", wintypes.BYTE),
                    ("SystemStatusFlag", wintypes.BYTE),
                    ("BatteryLifeTime", wintypes.DWORD),
                    ("BatteryFullLifeTime", wintypes.DWORD)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    status = SYSTEM_POWER_STATUS()
    info = {}
    if k32.GetSystemPowerStatus(ctypes.byref(status)):
        line = {0: "battery", 1: "ac", 255: "unknown"}.get(status.ACLineStatus & 0xFF,
                                                           "unknown")
        info["ac_line_status"] = line
        percent = status.BatteryLifePercent & 0xFF
        info["battery_percent"] = None if percent == 255 else percent
        info["battery_present"] = not bool(status.BatteryFlag & 0x80)
        # Windows sets this when a power-saving mode is engaged, which is
        # precisely the condition under which clocks are not what the nominal
        # figures suggest.
        info["power_saver_on"] = bool(status.SystemStatusFlag & 0x01)
    else:
        info["ac_line_status"] = _unavailable("GetSystemPowerStatus failed")["reason"]

    try:
        powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
        guid = ctypes.c_void_p()
        if powrprof.PowerGetActiveScheme(None, ctypes.byref(guid)) == 0:
            raw = ctypes.string_at(guid, 16)
            info["power_scheme_guid"] = str(_guid_from_bytes(raw))
            ctypes.WinDLL("kernel32").LocalFree(guid)
    except Exception as exc:  # noqa: BLE001
        info["power_scheme_guid"] = _unavailable(exc)
    return info


def _guid_from_bytes(raw: bytes) -> str:
    import uuid as _uuid
    return _uuid.UUID(bytes_le=raw)


_CIM_QUERY = (
    "$ErrorActionPreference='Stop';"
    "$gpu=@(Get-CimInstance Win32_VideoController | Select-Object Name,"
    "DriverVersion,DriverDate,AdapterRAM,PNPDeviceID,VideoModeDescription);"
    "$cs=Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,"
    "TotalPhysicalMemory;"
    "$os=Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,"
    "BuildNumber;"
    "$mem=@(Get-CimInstance Win32_PhysicalMemory | Select-Object Capacity,Speed,"
    "ConfiguredClockSpeed,Manufacturer);"
    "@{gpu=$gpu; system=$cs; os=$os; memory=$mem} | ConvertTo-Json -Depth 4 -Compress"
)


def discover_hardware_details(timeout: float = 30.0) -> dict:
    """GPU, driver, chassis and memory-module detail, in one CIM round trip.

    One PowerShell call rather than several: it costs about a second, and it is
    made once per run, before anything is measured. Nothing here is sampled
    during a benchmark.
    """
    result = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", _CIM_QUERY],
        capture_output=True, text=True, timeout=timeout,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError("CIM query failed: " + result.stderr[-400:])
    payload = json.loads(result.stdout)
    gpus = payload.get("gpu") or []
    if isinstance(gpus, dict):
        gpus = [gpus]
    modules = payload.get("memory") or []
    if isinstance(modules, dict):
        modules = [modules]
    return {
        "gpus": [{"name": g.get("Name"), "driver_version": g.get("DriverVersion"),
                  "driver_date": str(g.get("DriverDate")),
                  "pnp_device_id": g.get("PNPDeviceID"),
                  "video_mode": g.get("VideoModeDescription")} for g in gpus],
        "system": payload.get("system") or {},
        "os": payload.get("os") or {},
        "memory_modules": [{"capacity_mb": round((m.get("Capacity") or 0) / 1e6, 1),
                            "rated_speed_mhz": m.get("Speed"),
                            "configured_speed_mhz": m.get("ConfiguredClockSpeed"),
                            "manufacturer": m.get("Manufacturer")} for m in modules],
    }


# ---------------------------------------------------------------------------
# Provenance: what code actually ran
# ---------------------------------------------------------------------------

def _git(*args, cwd=None, timeout=20.0):
    result = subprocess.run(["git", *args], cwd=str(cwd or ROOT), capture_output=True,
                            text=True, timeout=timeout,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()[:200]}")
    return result.stdout.strip()


def _hash_tree(root: Path, patterns=("*.py",)) -> dict:
    """Digest every matching file under ``root``, plus the count and total size.

    This is the half of provenance that survives a copied folder with no
    ``.git`` and no git installed -- which Phase 3 requires, and which is also
    the normal state of a second benchmark machine.
    """
    files = sorted({path for pattern in patterns for path in root.rglob(pattern)
                    if path.is_file() and "__pycache__" not in path.parts})
    digest, total = hashlib.sha256(), 0
    for path in files:
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        data = path.read_bytes()
        total += len(data)
        digest.update(hashlib.sha256(data).digest())
    return {"sha256": digest.hexdigest(), "files": len(files), "bytes": total}


def discover_source() -> dict:
    """Git identity where it exists, and a content fingerprint either way.

    **Both, never one or the other.** A commit hash identifies the code only if
    the tree is clean, and a tree is clean surprisingly rarely on the machine
    where benchmarks get run. The fingerprint is what actually distinguishes
    two copies of this project whose sources differ, and it is computed from
    the files on disk rather than from anything git says about them.
    """
    info = {"root": str(ROOT)}
    try:
        if REPO is None:
            # An installed wheel. Running git in site-packages would report
            # whatever repository happens to contain the environment -- a venv
            # inside someone's project -- and record its commit as RapidShot's.
            raise RuntimeError("installed package, not a checkout")
        info["commit"] = _git("rev-parse", "HEAD")
        info["branch"] = _git("rev-parse", "--abbrev-ref", "HEAD")
        dirty = _git("status", "--porcelain")
        info["dirty"] = bool(dirty)
        if dirty:
            paths = sorted(line[3:].strip() for line in dirty.splitlines() if line[3:])
            info["dirty_files"] = len(paths)
            # The commit alone describes code that is not what ran. This digest
            # is what makes two differently-dirty trees distinguishable.
            info["dirty_fingerprint"] = hashlib.sha256(
                "\n".join(paths).encode()).hexdigest()[:16]
    except Exception as exc:  # noqa: BLE001
        # A copied folder with no .git, or no git on PATH. Expected, not an
        # error -- but it must be visible, because a snapshot with no commit
        # and no reason reads like one taken before commits were recorded.
        info["git"] = _unavailable(exc)

    info["package_fingerprint"] = _hash_tree(PACKAGE)
    # The harness moved into the package; this is the subset that measures.
    info["harness_fingerprint"] = _hash_tree(PACKAGE / "_bench")
    info["native_binary"] = _native_binary()
    return info


def _native_binary() -> dict:
    """The compiled extension that actually loaded, hashed, with its origin.

    ``baseline.json`` is recorded with the native extension and
    ``baseline-nonative.json`` without it, and pointing a comparison at the
    wrong one reports a 6-20x regression on every conversion row forever. A
    boolean "is it available" was never enough: two builds of the extension are
    also different machines to measure on.
    """
    try:
        from rapidshot import native
    except Exception as exc:  # noqa: BLE001
        return _unavailable(exc)
    info = {"available": True}
    try:
        info["is_available"] = bool(native.is_available())
    except Exception as exc:  # noqa: BLE001
        info["is_available"] = f"unknown ({type(exc).__name__})"

    module = getattr(native, "__file__", None)
    candidates = []
    for base in filter(None, [Path(module).parent if module else None,
                              PACKAGE / "_libs", PACKAGE]):
        if base.is_dir():
            candidates += sorted(base.rglob("*.pyd"))
    seen, binaries = set(), []
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        data = path.read_bytes()
        binaries.append({
            "path": str(path),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            # Where it came from decides whether a rebuild would change it
            # without anything else in the recording moving.
            "origin": ("site-packages" if "site-packages" in path.parts
                       else "source-tree" if REPO is not None and str(REPO) in str(path)
                       else "other"),
        })
    info["binaries"] = binaries
    return info


def discover_runtime() -> dict:
    """Python, the packages that decide what is being compared, and thread pools."""
    import importlib.metadata

    interesting = ("rapidshot", "numpy", "opencv-python", "opencv-python-headless",
                   "cupy-cuda12x", "cupy-cuda11x", "cupy", "onnxruntime",
                   "onnxruntime-gpu", "torch", "ultralytics", "dxcam", "bettercam",
                   "mss", "pillow", "psutil")
    versions = {}
    for name in interesting:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "packages": versions,
        "threads": thread_pool_sizes(),
    }


def discover_os() -> dict:
    return {"platform": platform.platform(), "release": platform.release(),
            "version": platform.version(), "machine": platform.machine(),
            "hostname_hash": hashlib.sha256(
                platform.node().encode()).hexdigest()[:16]}


# ---------------------------------------------------------------------------
# The snapshot
# ---------------------------------------------------------------------------

#: The two expensive sections. ``hardware`` costs a PowerShell CIM round trip
#: (about a second); ``source`` shells out to git three times and hashes every
#: Python file in the project. Both are worth paying once per run and neither
#: is worth paying between every case, so they can be left out -- explicitly,
#: and recorded as left out rather than as absent.
OPTIONAL_SECTIONS = ("hardware", "source", "runtime")


def discover_machine(*, include_cim: bool = True, include_source: bool = True,
                     include_runtime: bool = True) -> dict:
    """Everything knowable about this machine, before anything is measured.

    Every section is probed independently: one failure costs that section and
    records its reason, never the snapshot. A run is allowed to proceed with a
    partial snapshot -- and is not allowed to pretend the missing parts were
    something in particular.

    ``collected`` lists the sections actually gathered, so a later comparison
    can tell "this did not change" apart from "this was not looked at". Without
    it, a cheap re-check reads as though half the machine had vanished.
    """
    cpu = _probe(discover_cpu)
    displays = _probe(discover_displays)
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": _utc_now(),
        "cpu": cpu,
        "memory": _probe(discover_memory),
        "displays": displays,
        "display_fingerprint": display_fingerprint(displays),
        "power": _probe(discover_power),
        "os": discover_os(),
    }
    collected = ["cpu", "memory", "displays", "power", "os"]
    wanted = {"hardware": include_cim, "source": include_source,
              "runtime": include_runtime}
    for section, gather in (("hardware", discover_hardware_details),
                            ("source", discover_source),
                            ("runtime", discover_runtime)):
        if wanted[section]:
            snapshot[section] = _probe(gather)
            collected.append(section)
        else:
            snapshot[section] = _unavailable("not collected for this snapshot")
    snapshot["collected"] = collected
    snapshot["machine_id"] = machine_id(snapshot)
    return snapshot


def machine_id(snapshot: dict) -> str:
    """A stable identifier for this physical machine.

    Derived, not configured, so a copied project on a new box identifies itself
    without anyone remembering to name it. It is a digest of hardware identity
    only -- no hostname, no username, no serial number -- so it can be published
    without carrying anything personal with it.

    **Built only from fields that are always obtainable.** An earlier version
    included the GPU list from the CIM query, which meant a cheaper re-check
    with ``include_cim=False`` computed a *different* id for the same machine
    and `verify_environment` reported that the hardware had been swapped
    mid-run. Anything that can be unavailable must stay out of an identifier,
    or the identifier reports on the probe rather than on the machine.

    Deliberately excluded for a different reason: power state, available
    memory, affinity and display configuration all change between runs on one
    machine, and two runs on one machine must share an id or per-machine
    history cannot be kept apart the way ROADMAP section 3 requires. The GPU is
    recorded in ``hardware`` and its adapter identity in
    ``display_fingerprint``; neither belongs here.
    """
    cpu = snapshot.get("cpu", {})
    memory = snapshot.get("memory", {})
    stable = {
        "processor": cpu.get("processor"),
        "physical_cores": cpu.get("physical_cores"),
        "logical_processors": cpu.get("logical_processors"),
        "topology": cpu.get("topology"),
        "caches": cpu.get("caches"),
        "total_physical_mb": memory.get("total_physical_mb"),
        "machine": snapshot.get("os", {}).get("machine"),
    }
    return "m-" + hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str).encode()).hexdigest()[:16]


def add_policy_arguments(parser):
    """The pinning flag, spelled the same way in every runner."""
    parser.add_argument("--no-pin", action="store_true",
                        help="do not restrict to performance cores. Recorded as its "
                             "own configuration and never pooled with pinned runs.")
    return parser


def prepare_run(args, *, announce: bool = True):
    """Apply the CPU policy and take the environment snapshot, in that order.

    Order matters and is the whole reason this is one function. The policy must
    land before any library in this process or any child of it sizes a thread
    pool, and the snapshot must be taken *after* the policy so it records the
    affinity the run actually had rather than the one it started with.
    """
    policy = apply_cpu_policy("none" if getattr(args, "no_pin", False)
                              else "performance")
    environment = discover_machine()
    if announce:
        detail = (f" -> {hex(policy.effective_mask)}" if policy.effective_mask
                  else "")
        trailer = ("" if policy.verified or policy.policy == "none"
                   else f" [{'; '.join(policy.reasons) or 'not applied'}]")
        print(f"  cpu policy         {policy.policy} on a {policy.topology} CPU"
              f"{detail}{trailer}", flush=True)
        print(f"  machine            {environment['machine_id']} "
              f"(displays {environment['display_fingerprint']})", flush=True)
    return policy, environment


@dataclasses.dataclass
class EnvironmentCheck:
    """What changed between two snapshots, and whether it invalidates a verdict."""

    changed: list = dataclasses.field(default_factory=list)
    blocking: list = dataclasses.field(default_factory=list)
    #: Fields one side did not gather. Not silence, and not agreement.
    not_checked: list = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.changed

    @property
    def comparable(self) -> bool:
        """Whether an automatic regression verdict may still be issued.

        A changed display mode or a different machine is not a noisy result to
        be weighed; it is a different experiment. Nothing downstream is allowed
        to compare across one.
        """
        return not self.blocking

    def as_dict(self) -> dict:
        return {"ok": self.ok, "comparable": self.comparable,
                "changed": list(self.changed), "blocking": list(self.blocking),
                "not_checked": list(self.not_checked)}


#: Changes that make two measurements incomparable rather than merely different.
BLOCKING_CHANGES = ("machine_id", "display_fingerprint", "cpu.topology",
                    "cpu.logical_processors", "source.commit",
                    "source.package_fingerprint", "power.ac_line_status")


def _dig(snapshot, dotted):
    node = snapshot
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def verify_environment(baseline: dict, current: dict = None, *,
                       watch=BLOCKING_CHANGES) -> EnvironmentCheck:
    """Compare the machine now against the snapshot the run started with.

    Called between cases. A display mode that changed, a laptop that came off
    mains, a rebuilt extension -- each of these makes the cases on either side
    of it measurements of different things, and the only way that is ever
    noticed is by looking.
    """
    if current is None:
        # The cheap check by default. The expensive sections cannot change
        # between two cases of one run in any way this would catch usefully,
        # and paying a CIM round trip and three git invocations per case would
        # add more to a run than it tells anyone.
        current = discover_machine(include_cim=False, include_source=False,
                                   include_runtime=False)
    gathered = set(current.get("collected", ()))
    check = EnvironmentCheck()
    for field in watch:
        section = field.split(".")[0]
        if section in OPTIONAL_SECTIONS and section not in gathered:
            check.not_checked.append(field)
            continue
        before, after = _dig(baseline, field), _dig(current, field)
        if before is None and after is None:
            continue
        if before != after:
            entry = {"field": field, "before": before, "after": after}
            check.changed.append(entry)
            check.blocking.append(entry)
    # Non-blocking drift, recorded so it can be weighed rather than discovered.
    for field in ("power.battery_percent", "memory.available_physical_mb",
                  "memory.memory_load_percent"):
        before, after = _dig(baseline, field), _dig(current, field)
        if before != after:
            check.changed.append({"field": field, "before": before, "after": after})
    return check
