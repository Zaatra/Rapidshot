"""Which backend should capture a source, decided from measurements (ROADMAP § 7.3).

    plan = rapidshot.plan(rapidshot.Monitor(), consumer="torch", calibrate=True)
    print(plan)                # the decision, and the evidence behind it
    camera = rapidshot.create(source=rapidshot.Monitor())   # "auto" uses it

"WGC works, therefore use WGC" is a default with nothing behind it. What is
measurable is how old each backend's pixels are when they reach the caller,
how many frames it delivers and what each costs in CPU -- the benchmark
harness's pixel age (§ 7.0), from the same test source and the same frame-ID
marker. ``plan(calibrate=True)`` measures that for every backend this process
can run, picks the freshest, and caches the result per machine; ``backend=
"auto"`` then follows the cached choice. Nothing is measured implicitly:
without a cached measurement ``"auto"`` uses Desktop Duplication for a display
and says so, and WGC for a window, which nothing else can capture.

The cache lives in ``%LOCALAPPDATA%\\rapidshot\\plans.json`` (or
``$RAPIDSHOT_PLAN_CACHE``) and is keyed on what the measurement depends on --
the adapters and their drivers, the display mode, and the RapidShot and
rapidshot-native versions -- so a driver or library upgrade simply has no
cached answer until it is measured again.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["CapturePlan", "Measurement", "plan", "cached_backend", "cache_path"]

#: Consumers a plan can describe. "cpu" is NumPy through grab().
CONSUMERS = (None, "cpu", "numpy", "torch", "cupy")

#: Two pixel ages this close are a tie, broken by frame rate and then CPU.
#: Run-to-run spread of the pixel-age p50 on one machine is about 1 ms (§ 7.0).
TIE_MS = 1.0

#: Frame rates within this fraction of each other are a tie too: two
#: backends paced by the same 165 Hz display both measure "164 fps", and a
#: fraction of a frame per second is not a reason to choose.
TIE_FPS = 0.03

#: Frames a backend must deliver for its measurement to count.
MIN_FRAMES = 30

#: The frame-ID marker the test source draws at (0, 0): a capture must be at
#: least this big, anchored there, to carry it.
_MARKER_SIZE = (384, 16)


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Measurement:
    """One backend's calibration: pixel age is the time from the source's
    ``Present()`` to the frame reaching the caller, in milliseconds."""

    backend: str
    frames: int
    seconds: float
    fps: float
    age_p50_ms: float
    age_p95_ms: float
    age_p99_ms: float
    cpu_ms_per_frame: float
    measured_at: str
    capture: Tuple[int, int] = (0, 0)

    def summary(self) -> str:
        return (f"{self.backend} at {self.capture[0]}x{self.capture[1]}: "
                f"pixel age p50 {self.age_p50_ms:.2f} ms "
                f"(p95 {self.age_p95_ms:.2f}), {self.fps:.0f} fps, "
                f"{self.cpu_ms_per_frame:.2f} ms CPU/frame")


@dataclass
class CapturePlan:
    """A capture decision and its reasoning. ``str(plan)`` prints both."""

    source: object
    backend: str
    candidates: Tuple[str, ...]
    measured: bool
    reasons: List[str] = field(default_factory=list)
    evidence: Dict[str, Measurement] = field(default_factory=dict)
    display: str = ""
    capture_adapter: str = ""
    consumer: Optional[str] = None
    consumer_adapter: Optional[str] = None
    crosses_adapters: Optional[bool] = None
    preprocessing: str = ""
    size: Optional[Tuple[int, int]] = None
    hdr: bool = False
    key: str = ""
    #: What the evidence timed, in words; empty when nothing was measured.
    measured_path: str = ""

    def __str__(self) -> str:
        lines = [f"capture {self.source} through {self.backend!r}"
                 + ("" if self.measured else " (unmeasured)")]
        lines += [f"  because {reason}" for reason in self.reasons]
        if self.measured_path:
            lines.append(f"  what was measured: {self.measured_path}")
        for measurement in self.evidence.values():
            lines.append(f"  measured {measurement.summary()}")
        lines.append(f"  display: {self.display}{' (HDR)' if self.hdr else ''}")
        lines.append(f"  capture adapter: {self.capture_adapter}")
        if self.consumer:
            lines.append(f"  consumer: {self.consumer} on "
                         f"{self.consumer_adapter or 'an adapter this process cannot identify'}")
            if self.crosses_adapters is not None:
                lines.append("  frames cross adapters: "
                             + ("yes -- convert, then TensorTransfer" if self.crosses_adapters
                                else "no"))
        if self.preprocessing:
            lines.append(f"  preprocessing: {self.preprocessing}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def cache_path() -> Path:
    override = os.environ.get("RAPIDSHOT_PLAN_CACHE")
    if override:
        return Path(override)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    return Path(base) / "rapidshot" / "plans.json"


def _load_cache() -> dict:
    try:
        data = json.loads(cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data.get("plans", {}) if isinstance(data, dict) else {}


def _store(key: str, entry: dict) -> None:
    path = cache_path()
    plans = _load_cache()
    plans[key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    # Written beside and swapped in, so a reader never sees half a file.
    fd, temp = tempfile.mkstemp(prefix=".plans-", suffix=".json", dir=path.parent)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"version": 1, "plans": plans}, stream, indent=1)
    os.replace(temp, path)


def _driver_version(adapter) -> str:
    """The adapter's user-mode driver version, as Windows reports it, or "".

    ``IDXGIAdapter::CheckInterfaceSupport(IDXGIDevice)`` fills it in; the
    binding in ``_libs/dxgi.py`` declares no argument types, so this calls
    the vtable slot (IUnknown 3 + IDXGIObject 4 + EnumOutputs + GetDesc = 9).
    """
    try:
        import comtypes  # type: ignore[import-untyped]

        iid = comtypes.GUID("{54ec77fa-1377-44e6-8c32-88fd5f44c84c}")   # IDXGIDevice
        raw = ctypes.cast(adapter, ctypes.c_void_p).value
        vtable = ctypes.cast(raw, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        check = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(comtypes.GUID),
                                   ctypes.POINTER(ctypes.c_longlong))(vtable[9])
        version = ctypes.c_longlong()
        if check(raw, ctypes.byref(iid), ctypes.byref(version)) < 0:
            return ""
        v = version.value
        return f"{(v >> 48) & 0xFFFF}.{(v >> 32) & 0xFFFF}.{(v >> 16) & 0xFFFF}.{v & 0xFFFF}"
    except Exception:  # noqa: BLE001 -- best effort; the key just gets coarser
        return ""


def _topology(factory) -> List[str]:
    """One line per adapter: what it is and which driver it runs."""
    rows = []
    for device in getattr(factory, "all_devices", []):
        desc = getattr(device, "desc", None)
        name = str(getattr(desc, "Description", "") or "")
        ids = f"{getattr(desc, 'VendorId', 0):04x}:{getattr(desc, 'DeviceId', 0):04x}"
        rows.append(f"{name} [{ids}] {_driver_version(getattr(device, 'adapter', None))}")
    return rows


def _display_mode(output) -> Tuple[int, int, int]:
    """Width, height and refresh rate of a display, for the key and the source."""
    width, height = output.resolution
    try:
        user32 = ctypes.WinDLL("user32")   # private handle; see util/io.py

        class DEVMODEW(ctypes.Structure):
            _fields_ = [("name", ctypes.c_wchar * 32), ("spec", ctypes.c_ushort),
                        ("driver", ctypes.c_ushort), ("size", ctypes.c_ushort),
                        ("extra", ctypes.c_ushort), ("fields", ctypes.c_ulong),
                        ("display", ctypes.c_byte * 16), ("color", ctypes.c_short),
                        ("duplex", ctypes.c_short), ("yres", ctypes.c_short),
                        ("tt", ctypes.c_short), ("collate", ctypes.c_short),
                        ("form", ctypes.c_wchar * 32), ("logpixels", ctypes.c_ushort),
                        ("bits", ctypes.c_ulong), ("width", ctypes.c_ulong),
                        ("height", ctypes.c_ulong), ("flags", ctypes.c_ulong),
                        ("hz", ctypes.c_ulong), ("tail", ctypes.c_ulong * 8)]

        mode = DEVMODEW()
        mode.size = ctypes.sizeof(DEVMODEW)
        user32.EnumDisplaySettingsW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong,
                                                ctypes.POINTER(DEVMODEW)]
        if user32.EnumDisplaySettingsW(output.devicename, 0xFFFFFFFF, ctypes.byref(mode)):
            return width, height, int(mode.hz)
    except Exception:  # noqa: BLE001
        pass
    return width, height, 0


def _display_owner(factory, output) -> str:
    """The adapter that enumerates ``output`` -- the one driving the display.

    Part of the key because a MUX switch can change it without changing the
    adapters present or their drivers: on a hybrid laptop the iGPU drives the
    panel in hybrid mode and the dGPU in discrete mode, and which adapter
    composes the desktop is exactly what decides between the backends.
    """
    for d, outputs in enumerate(getattr(factory, "outputs", [])):
        if any(o is output for o in outputs):
            desc = getattr(factory.devices[d], "desc", None)
            luid = getattr(desc, "AdapterLuid", None)
            luid_text = (f"{int(luid.HighPart) & 0xFFFFFFFF:08x}{int(luid.LowPart):08x}"
                         if luid is not None else "")
            return f"{getattr(desc, 'Description', '')} {luid_text}".strip()
    return ""


def _power_source() -> str:
    """"ac", "battery" or "unknown". A laptop on battery may clock its GPUs and
    CPU down, which changes the measurement, so the two are kept apart."""

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    try:
        status = SYSTEM_POWER_STATUS()
        if ctypes.WinDLL("kernel32").GetSystemPowerStatus(ctypes.byref(status)):
            return {0: "battery", 1: "ac"}.get(status.ACLineStatus, "unknown")
    except (OSError, AttributeError):  # pragma: no cover - non-Windows
        pass
    return "unknown"


def _key(kind: str, factory, output, capture: Tuple[int, int]) -> str:
    """What a measurement depends on, hashed. Readable parts are kept in the
    entry itself; the key only has to be stable and change with any of them.

    ``capture`` is the size grab() reads back. It is part of the answer, not a
    detail: on Machine B the marker-sized capture put WGC 0.4 ms ahead and
    the full display put Desktop Duplication 2.3 ms ahead (2026-10-29).
    """
    from rapidshot import native
    from rapidshot._version import __version__

    width, height, hz = _display_mode(output)
    parts = [kind, f"{width}x{height}@{hz}", f"capture {capture[0]}x{capture[1]}",
             f"display on {_display_owner(factory, output)}", f"power {_power_source()}",
             *_topology(factory),
             f"rapidshot {__version__}", f"native {native.extension_version()}"]
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:24]


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------

def _find_source() -> Optional[Path]:
    """The test source: a local build first, then rapidshot-native's (as the
    benchmark chooses)."""
    built = Path(__file__).resolve().parents[1] / "native" / "target" / "release" / "latency_source.exe"
    if built.is_file():
        return built
    try:
        import rapidshot_native  # type: ignore[import-not-found]

        path = Path(rapidshot_native.latency_source_path())
        return path if path.is_file() else None
    except (ImportError, AttributeError, OSError):
        return None


class _Source:
    """latency_source.exe covering the primary display, stamping frame IDs."""

    def __init__(self, width: int, height: int, hz: int):
        exe = _find_source()
        if exe is None:
            raise RuntimeError("calibration needs the test source latency_source.exe, "
                               "which rapidshot-native >= 0.2.1 ships")
        self.directory = Path(tempfile.mkdtemp(prefix="rapidshot-plan-"))
        self.log_path = self.directory / "presents.jsonl"
        self.proc = subprocess.Popen(
            [str(exe), str(width), str(height), str(float(hz or 60)), "motion",
             str(self.log_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True)
        self.frequency = None
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("event") == "ready":
                self.frequency = int(event["qpc_frequency"])
                break
        if self.frequency is None:
            self.close()
            raise RuntimeError("the test source did not become ready")
        time.sleep(0.5)   # let it settle into presenting

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except Exception:  # noqa: BLE001
                self.proc.kill()


def _measure(backend: str, source: "_Source", seconds: float,
             capture: Tuple[int, int]) -> Measurement:
    """Time ``backend``'s grab() of a ``capture``-sized area of the primary
    display, anchored at (0, 0) so it carries the frame-ID marker.

    The size matters as much as the backend. Measured on Machine B,
    2026-10-29, twice each: cropped to the 384x16 marker WGC was 0.4 ms
    fresher, while reading the full 2560x1600 display back Desktop
    Duplication was 2.3 ms fresher with 12 more frames a second. A crop
    measures capture freshness; a full grab also pays for the readback, which
    is what a caller of that size actually waits for."""
    import numpy as np

    import rapidshot
    from rapidshot.util.pixel_age import PresentLog, decode_marker, percentiles, qpc_clock
    from rapidshot.sources import Monitor

    now, _ = qpc_clock()
    log = PresentLog(source.log_path, source.frequency)
    width, height = capture
    camera = rapidshot.create(source=Monitor(), backend=backend, output_color="BGRA",
                              region=(0, 0, width, height), pool_output=False)
    ages, frames = [], 0
    try:
        for _ in range(20):           # warm-up: first frames, pools, shaders
            camera.grab()
        cpu0, wall0 = time.process_time(), time.perf_counter()
        deadline = wall0 + seconds
        while time.perf_counter() < deadline:
            image = camera.grab()
            if image is None:
                continue
            ready = now()
            frames += 1
            frame_id = decode_marker(np.asarray(image), np)
            if frame_id is not None:
                try:
                    age = log.age(frame_id, ready)
                except ValueError:
                    age = None
                if age is not None:
                    ages.append(age)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
    finally:
        camera.release()
        log.close()
    stats = percentiles(ages)
    if stats is None or len(ages) < MIN_FRAMES:
        raise RuntimeError(f"{backend} delivered {len(ages)} timed frames in {seconds:.1f} s")
    return Measurement(
        backend=backend, capture=(width, height), frames=frames, seconds=round(wall, 3),
        fps=round(frames / wall, 1), age_p50_ms=round(stats["p50"], 3),
        age_p95_ms=round(stats["p95"], 3), age_p99_ms=round(stats["p99"], 3),
        cpu_ms_per_frame=round(cpu * 1000 / max(frames, 1), 3),
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))


def _choose(evidence: Dict[str, Measurement]) -> Tuple[str, List[str]]:
    """The freshest backend; within TIE_MS, the one delivering more frames;
    within TIE_FPS of that too, the one costing less CPU.

    Returns the choice and a sentence naming the criterion that decided it --
    only that one, so the reason never claims an advantage nobody checked.
    """
    ranked = sorted(evidence.values(), key=lambda m: m.age_p50_ms)
    best, runner_up = ranked[0], ranked[1]
    tied = [m for m in ranked if m.age_p50_ms - best.age_p50_ms <= TIE_MS]
    if len(tied) == 1:
        return best.backend, [
            f"its pixels are {runner_up.age_p50_ms - best.age_p50_ms:.1f} ms fresher at "
            f"p50 than {runner_up.backend}'s"]

    by_fps = sorted(tied, key=lambda m: -m.fps)
    fastest, second = by_fps[0], by_fps[1]
    if second.fps < fastest.fps * (1 - TIE_FPS):
        return fastest.backend, [
            f"pixel ages are within {TIE_MS:.0f} ms, and {fastest.backend} delivered "
            f"more frames ({fastest.fps:.0f} fps against {second.fps:.0f})"]

    frame_tied = [m for m in tied if m.fps >= fastest.fps * (1 - TIE_FPS)]
    cheapest = sorted(frame_tied, key=lambda m: m.cpu_ms_per_frame)
    winner, other = cheapest[0], cheapest[1]
    return winner.backend, [
        f"pixel ages are within {TIE_MS:.0f} ms and frame rates within "
        f"{TIE_FPS:.0%}, so CPU cost decided: {winner.backend} "
        f"{winner.cpu_ms_per_frame:.2f} ms/frame against {other.backend}'s "
        f"{other.cpu_ms_per_frame:.2f}"]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------

def _resolve(factory, source):
    """(kind, output, device) for a source: the display it is on and that
    display's adapter."""
    from rapidshot.sources import Window

    d, o, _region = factory._resolve_source(source, 0, None, None)
    kind = "window" if isinstance(source, Window) else "display"
    return kind, factory.outputs[d][o], factory.devices[d]


def _capture_size(source, output) -> Tuple[int, int]:
    """The size a camera for ``source`` reads back: the whole display for a
    Monitor, the rectangle for a Region."""
    from rapidshot.sources import Region

    if isinstance(source, Region):
        return (source.width, source.height)
    width, height = output.resolution
    return (int(width), int(height))


def _candidates(kind: str) -> Tuple[str, ...]:
    from rapidshot.core import wgc

    wgc_ok = wgc.is_available()
    if kind == "window":
        return ("wgc",) if wgc_ok else ()
    return ("dxgi", "wgc") if wgc_ok else ("dxgi",)


def _luid(device) -> bytes:
    luid = getattr(getattr(device, "desc", None), "AdapterLuid", None)
    if luid is None:
        return b""
    return (int(luid.LowPart) & 0xFFFFFFFF).to_bytes(4, "little") + \
        (int(luid.HighPart) & 0xFFFFFFFF).to_bytes(4, "little")


def _describe_consumer(result: CapturePlan, consumer, device, size) -> None:
    if consumer not in CONSUMERS:
        raise ValueError(f"consumer must be one of {CONSUMERS[1:]} or None, got {consumer!r}")
    result.consumer = consumer
    if consumer in (None, "cpu", "numpy"):
        result.consumer_adapter = "the CPU" if consumer else None
        result.crosses_adapters = None if consumer is None else False
        result.preprocessing = ("on the CPU: grab() converts colour; to_nchw() makes the tensor"
                                if consumer else "")
        return
    # A CUDA consumer: is a CUDA device on the capture adapter?
    target = f" to {size[0]}x{size[1]}" if size else ""
    try:
        import cupy  # type: ignore[import-not-found]

        from rapidshot.converter import CrossAdapterRequired, _device_for_adapter
    except ImportError:
        result.reasons.append("CuPy is not installed, so which adapter CUDA runs on is unknown")
        result.preprocessing = f"GpuConverter{target} on the capture adapter"
        return
    try:
        ordinal = _device_for_adapter(cupy, _luid(device))
    except CrossAdapterRequired:
        # The ordinary hybrid-laptop case: capture on the iGPU, CUDA elsewhere.
        result.consumer_adapter = "a CUDA device on another adapter"
        result.crosses_adapters = True
        result.preprocessing = (f"on the capture adapter: GpuConverter{target}, then "
                                "TensorTransfer carries the converted tensor across")
        return
    except Exception as e:  # noqa: BLE001 -- "could not tell" is not "elsewhere"
        result.reasons.append(f"could not tell which adapter CUDA runs on: {e}")
        result.preprocessing = f"GpuConverter{target} on the capture adapter"
        return
    result.consumer_adapter = f"CUDA device {ordinal}, the capture adapter"
    result.crosses_adapters = False
    result.preprocessing = (f"on the capture adapter: GpuConverter{target}, then "
                            f"{'to_torch()' if consumer == 'torch' else 'to_cupy()'}")


def plan(source=None, consumer: Optional[str] = None, size: Optional[Tuple[int, int]] = None,
         calibrate: bool = False, seconds: float = 2.0) -> CapturePlan:
    """Decide how to capture ``source`` (default: the primary display).

    Args:
        source: ``rapidshot.Monitor``, ``Window`` or ``Region``.
        consumer: Where frames go -- ``"cpu"``/``"numpy"``, ``"torch"``,
            ``"cupy"`` -- so the plan can say which adapter that is, whether
            frames cross between adapters, and where preprocessing belongs.
        size: Model input size, for describing the preprocessing.
        calibrate: Measure every available backend now (a few seconds each,
            with a test window covering the primary display) and cache the
            result for ``backend="auto"``. Without it, a cached measurement is
            used if there is one.
        seconds: How long to measure each backend.
    """
    import rapidshot
    from rapidshot.core import hdr
    from rapidshot.sources import Monitor

    source = source if source is not None else Monitor()
    factory = rapidshot.get_factory()
    kind, output, device = _resolve(factory, source)
    candidates = _candidates(kind)
    capture = _capture_size(source, output)
    key = _key(kind, factory, output, capture)
    result = CapturePlan(
        source=source, backend="", candidates=candidates, measured=False, key=key,
        display=f"{output.devicename} {output.resolution[0]}x{output.resolution[1]}",
        capture_adapter=str(getattr(device.desc, "Description", "")),
        hdr=hdr.display_color(output.devicename).hdr, size=tuple(size) if size else None)

    if not candidates:
        raise rapidshot.ConfigurationError(
            f"nothing can capture {source} here: a window needs Windows.Graphics.Capture "
            "(rapidshot-native >= 0.3.0)")

    if kind == "window":
        result.backend = "wgc"
        result.reasons.append("only Windows.Graphics.Capture can capture a single window")
    else:
        evidence = {}
        if calibrate and len(candidates) > 1:
            # The test source covers the primary display, so that is where
            # backends are compared; the choice is cached per topology.
            _kind, primary, _device = _resolve(factory, Monitor())
            width, height, hz = _display_mode(primary)
            if (capture[0] > width or capture[1] > height
                    or capture[0] < _MARKER_SIZE[0] or capture[1] < _MARKER_SIZE[1]):
                raise rapidshot.ConfigurationError(
                    f"cannot calibrate a {capture[0]}x{capture[1]} capture: it is measured "
                    f"on the primary display ({width}x{height}) and must be at least "
                    f"{_MARKER_SIZE[0]}x{_MARKER_SIZE[1]} to carry the frame-ID marker")
            src = _Source(width, height, hz)
            try:
                for backend in candidates:
                    try:
                        evidence[backend] = _measure(backend, src, seconds, capture)
                    except Exception as e:  # noqa: BLE001 -- a backend that cannot run is evidence too
                        result.reasons.append(f"{backend} could not be measured: {e}")
            finally:
                src.close()
            if len(evidence) > 1:
                choice, _why = _choose(evidence)
                _store(key, {"choice": choice, "kind": kind,
                             "evidence": {b: asdict(m) for b, m in evidence.items()}})
        else:
            entry = _load_cache().get(key)
            if entry:
                evidence = {b: Measurement(**{**m, "capture": tuple(m.get("capture", (0, 0)))})
                            for b, m in entry.get("evidence", {}).items() if b in candidates}
        if len(evidence) > 1:
            result.backend, why = _choose(evidence)
            result.reasons += why
            result.evidence = evidence
            result.measured = True
            result.measured_path = (
                f"grab() to NumPy (BGRA) of a {capture[0]}x{capture[1]} area of the "
                "primary display, timed from the test source's Present()")
            if consumer in ("torch", "cupy"):
                result.reasons.append(
                    "this measured the CPU path; a GPU consumer takes grab_frame() and "
                    "GpuConverter instead, which was not timed and may rank the "
                    "backends differently")
        else:
            result.backend = "dxgi"
            result.reasons.append(
                "nothing has been measured on this machine yet: Desktop Duplication is the "
                "default for a display (rapidshot.plan(calibrate=True) measures)"
                if len(candidates) > 1 else
                "Desktop Duplication is the only backend available to this process")
    _describe_consumer(result, consumer, device, size)
    return result


def cached_backend(factory, source) -> Optional[str]:
    """The measured choice for a display source, if one is cached; else None.

    What ``backend="auto"`` asks. Cheap -- no measuring, no capture.
    """
    try:
        kind, output, _device = _resolve(factory, source)
        if kind != "display":
            return None
        entry = _load_cache().get(_key(kind, factory, output, _capture_size(source, output)))
    except Exception:  # noqa: BLE001 -- a broken cache must never break create()
        return None
    choice = entry.get("choice") if entry else None
    return choice if choice in _candidates(kind) else None
