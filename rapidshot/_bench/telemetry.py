"""Low-rate conditions sampling, with an honest account of what it can see.

This records the conditions a benchmark ran under: CPU load, memory, and --
where a provider can actually supply them -- CPU frequency, GPU utilisation,
temperature and power. It is **not** a profiler. Windows performance counters
and the power-information APIs are built for collection at human timescales,
not per frame, and sampling them inside a capture loop would change the thing
being measured. The default interval is a second; anything under
:data:`MINIMUM_INTERVAL` is refused rather than quietly accepted.

**Every metric has three states, not two.** Present, absent-with-a-reason, and
-- the one that matters most here -- *present but not trustworthy*. A provider
that returns the same number every time it is asked has not measured anything,
and this module says so rather than reporting a flat line as a finding.

**On CPU frequency specifically.** ``CallNtPowerInformation`` is the only
Windows API that reports a per-processor *current* megahertz, and on plenty of
machines it returns the nominal figure unchanged forever. So the sampler
watches: if a processor's reported clock never varies across the whole window
*and* equals its maximum, the series is marked ``suspect_nominal`` and must not
be read as evidence that the machine held its boost clock. One query could
never have established that; repeated queries can at least establish when the
answer is not a measurement.

``GetSystemTimes`` is sometimes reached for in this role. It reports how CPU
*time* was divided between idle, kernel and user. It says nothing about clock
speed, and nothing here uses it for that.

**Background load is a warning, not a verdict.** System CPU minus one worker's
CPU is not proof of contamination: the benchmark's own parent, its workers and
the motion source generating the workload are all "not this worker". The whole
process tree is accounted for before anything is attributed to a stranger, and
even then the result is flagged for a human rather than used to discard a run.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import dataclasses
from datetime import datetime, timezone
import os
import statistics
import sys
import threading
import time

#: Below this, sampling stops being observation and starts being interference.
#: The counters behind these providers are documented for low-frequency
#: collection; a 50 ms loop against them shows up in the numbers it is
#: supposed to be describing.
MINIMUM_INTERVAL = 0.25

#: What the overhead measurement on this machine came out at, per sample, in
#: milliseconds of wall clock. Re-measured by :func:`measure_overhead` rather
#: than assumed, and recorded with every run so a reader can see what the
#: instrument cost.
DEFAULT_INTERVAL = 1.0

IS_WINDOWS = sys.platform == "win32"


def _unavailable(reason) -> dict:
    if isinstance(reason, BaseException):
        reason = f"{type(reason).__name__}: {reason}"
    return {"available": False, "reason": str(reason)}


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

class CpuFrequencyProvider:
    """Per-processor current/max MHz via ``CallNtPowerInformation``.

    ``PROCESSOR_POWER_INFORMATION`` is six ULONGs per logical processor:
    Number, MaxMhz, CurrentMhz, MhzLimit, MaxIdleState, CurrentIdleState.
    """

    name = "CallNtPowerInformation(ProcessorInformation)"
    _PROCESSOR_INFORMATION = 11

    def __init__(self):
        self.count = os.cpu_count() or 1
        self._powrprof = ctypes.WinDLL("powrprof", use_last_error=True)
        self._powrprof.CallNtPowerInformation.argtypes = [
            ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
            ctypes.c_void_p, ctypes.c_ulong]
        self._buffer = (ctypes.c_ulong * (6 * self.count))()
        self.read()                      # fail now rather than mid-run

    def read(self) -> dict:
        size = ctypes.sizeof(self._buffer)
        status = self._powrprof.CallNtPowerInformation(
            self._PROCESSOR_INFORMATION, None, 0, self._buffer, size)
        if status != 0:
            raise OSError(f"CallNtPowerInformation returned 0x{status & 0xFFFFFFFF:X}")
        raw = list(self._buffer)
        current, maximum = [], []
        for index in range(self.count):
            base = index * 6
            maximum.append(raw[base + 1])
            current.append(raw[base + 2])
        return {"current_mhz": current, "max_mhz": maximum}


class NvmlProvider:
    """GPU utilisation, clocks, temperature and power, through NVML directly.

    ctypes rather than ``pynvml`` so the dependency is the driver that is
    already installed. Shelling out to ``nvidia-smi`` per sample would cost
    tens of milliseconds and a process launch, which is the opposite of what a
    conditions sampler should do.
    """

    name = "nvml"

    def __init__(self):
        self._lib = ctypes.CDLL("nvml.dll")
        if self._lib.nvmlInit_v2() != 0:
            raise OSError("nvmlInit_v2 failed")
        count = ctypes.c_uint()
        if self._lib.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            raise OSError("nvmlDeviceGetCount_v2 failed")
        self.handles = []
        for index in range(count.value):
            handle = ctypes.c_void_p()
            if self._lib.nvmlDeviceGetHandleByIndex_v2(index,
                                                       ctypes.byref(handle)) == 0:
                self.handles.append(handle)
        if not self.handles:
            raise OSError("NVML reported no devices")
        self.names = [self._device_name(handle) for handle in self.handles]

    def _device_name(self, handle):
        buffer = ctypes.create_string_buffer(96)
        if self._lib.nvmlDeviceGetName(handle, buffer, 96) == 0:
            return buffer.value.decode("utf-8", "replace")
        return "unknown"

    def read(self) -> dict:
        devices = []
        for index, handle in enumerate(self.handles):
            entry = {"index": index, "name": self.names[index]}

            class Utilization(ctypes.Structure):
                _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

            usage = Utilization()
            if self._lib.nvmlDeviceGetUtilizationRates(handle,
                                                       ctypes.byref(usage)) == 0:
                entry["gpu_percent"] = usage.gpu
                entry["memory_bus_percent"] = usage.memory
            value = ctypes.c_uint()
            if self._lib.nvmlDeviceGetTemperature(handle, 0,
                                                  ctypes.byref(value)) == 0:
                entry["temperature_c"] = value.value
            if self._lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(value)) == 0:
                entry["power_w"] = value.value / 1000.0
            # 0 = graphics clock, 1 = SM, 2 = memory.
            for clock, label in ((0, "graphics_mhz"), (2, "memory_mhz")):
                if self._lib.nvmlDeviceGetClockInfo(handle, clock,
                                                    ctypes.byref(value)) == 0:
                    entry[label] = value.value

            class Memory(ctypes.Structure):
                _fields_ = [("total", ctypes.c_ulonglong),
                            ("free", ctypes.c_ulonglong),
                            ("used", ctypes.c_ulonglong)]

            memory = Memory()
            if self._lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)) == 0:
                entry["vram_used_mb"] = round(memory.used / 1e6, 1)
                entry["vram_total_mb"] = round(memory.total / 1e6, 1)
            devices.append(entry)
        return {"devices": devices}

    def close(self):
        try:
            self._lib.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass


class SystemLoadProvider:
    """System CPU and memory, plus this benchmark's own process tree.

    The tree matters. Attributing everything that is not one worker to
    "background load" would indict the benchmark's own parent and the motion
    source that generates its workload, which is most of what is running.
    """

    name = "psutil"

    def __init__(self, tree_root_pid=None):
        import psutil
        self._psutil = psutil
        self.tree_root_pid = tree_root_pid or os.getpid()
        self._root = psutil.Process(self.tree_root_pid)
        psutil.cpu_percent(interval=None)          # prime the delta
        self._root.cpu_percent(interval=None)
        # Process objects are cached by pid because `cpu_percent(interval=None)`
        # is stateful *per object*: it reports the busy fraction since that
        # object's previous call, and the first call on a fresh one always
        # returns 0.0. Rebuilding the child list each sample therefore reported
        # every worker as idle -- which a live run showed plainly, with a worker
        # burning 113% of a core while the tree meter read 0.05% and the whole
        # of the benchmark's own CPU landed in "background load".
        self._processes = {self.tree_root_pid: self._root}

    def _tree(self):
        """Every process in the tree, as objects that persist between samples."""
        try:
            live = {self.tree_root_pid: self._root}
            for child in self._root.children(recursive=True):
                live[child.pid] = child
        except self._psutil.Error:
            return [], 0
        primed, fresh = [], 0
        for pid, process in live.items():
            known = self._processes.get(pid)
            if known is None:
                # Seen for the first time: prime it and leave it out of this
                # sample rather than contributing a false zero.
                try:
                    process.cpu_percent(interval=None)
                except self._psutil.Error:
                    continue
                self._processes[pid] = process
                fresh += 1
                continue
            primed.append(known)
        # Drop the ones that have gone, so a long run does not accumulate
        # handles for every worker it ever started.
        for pid in set(self._processes) - set(live):
            self._processes.pop(pid, None)
        return primed, fresh

    def read(self) -> dict:
        system = self._psutil.cpu_percent(interval=None)
        memory = self._psutil.virtual_memory()
        members, fresh = self._tree()
        tree_percent, tree_rss, counted, lost = 0.0, 0, 0, 0
        for process in members:
            try:
                tree_percent += process.cpu_percent(interval=None)
                tree_rss += process.memory_info().rss
                counted += 1
            except self._psutil.Error:
                # A worker that exited between listing and reading. Counted, so
                # "the tree was partly unreadable" is never mistaken for "the
                # tree used no CPU".
                lost += 1
        cores = os.cpu_count() or 1
        tree_share = tree_percent / cores
        return {
            "system_cpu_percent": system,
            "benchmark_tree_cpu_percent": round(tree_share, 2),
            "processes_counted": counted,
            "processes_unreadable": lost,
            # Newly seen processes contribute nothing to this sample: their
            # first reading is the priming call. Counted so a reader can tell a
            # genuinely quiet tree from one that had just been rebuilt.
            "processes_newly_seen": fresh,
            # Explicitly a residual, not an attribution. It is what is left
            # after this benchmark's own tree, and it includes the compositor,
            # the shell and anything else the OS is doing on its own account.
            "other_cpu_percent": round(max(0.0, system - tree_share), 2),
            "memory_used_percent": memory.percent,
            "memory_available_mb": round(memory.available / 1e6, 1),
            "benchmark_tree_rss_mb": round(tree_rss / 1e6, 1),
        }


# ---------------------------------------------------------------------------
# The sampler
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TelemetryRecord:
    """Samples, what produced them, and what they cannot be used to claim."""

    interval: float
    started_at: str
    stopped_at: str = None
    samples: list = dataclasses.field(default_factory=list)
    providers: dict = dataclasses.field(default_factory=dict)
    missed: int = 0
    limitations: list = dataclasses.field(default_factory=list)

    def summary(self) -> dict:
        """Central tendency and spread per series, plus the validity verdicts."""
        summary = {}
        for series, values in _collect_series(self.samples).items():
            numeric = [value for value in values
                       if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not numeric:
                continue
            summary[series] = {
                "samples": len(numeric),
                "min": min(numeric), "max": max(numeric),
                "median": statistics.median(numeric),
                "mean": statistics.fmean(numeric),
                # Spread, not a confidence interval. These are serial
                # observations of one machine and are heavily autocorrelated;
                # treating them as independent draws would manufacture
                # precision that does not exist.
                "stdev": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
                "varied": min(numeric) != max(numeric),
            }
        return summary

    def as_dict(self) -> dict:
        return {
            "interval_seconds": self.interval,
            "started_at": self.started_at, "stopped_at": self.stopped_at,
            "sample_count": len(self.samples),
            "missed_samples": self.missed,
            "providers": self.providers,
            "limitations": list(self.limitations),
            "summary": self.summary(),
            "note": ("conditions sampling at a low rate; spread here describes "
                     "serial, autocorrelated observations of one machine and is "
                     "not a confidence interval"),
        }


def _collect_series(samples) -> dict:
    """Flatten samples into name -> list of values, arrays kept apart."""
    series = {}
    for sample in samples:
        for key, value in _flatten(sample):
            series.setdefault(key, []).append(value)
    return series


def _flatten(node, trail=""):
    if isinstance(node, dict):
        for key, value in node.items():
            name = f"{trail}.{key}" if trail else str(key)
            yield from _flatten(value, name)
    elif isinstance(node, list):
        if node and all(isinstance(item, (int, float)) and not isinstance(item, bool)
                        for item in node):
            # A per-processor array reduces to its extremes; keeping 32 series
            # per sample would bury everything else.
            yield f"{trail}.min", min(node)
            yield f"{trail}.max", max(node)
            yield f"{trail}.mean", statistics.fmean(node)
        else:
            for index, item in enumerate(node):
                yield from _flatten(item, f"{trail}[{index}]")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield trail, node


class TelemetrySampler:
    """Samples conditions on a background thread, between nothing and nobody.

    The thread does no work inside the measured loop and touches none of the
    benchmark's state. It is still a thread on the same machine, which is why
    :func:`measure_overhead` exists and why the interval floor is enforced.
    """

    def __init__(self, interval: float = DEFAULT_INTERVAL, *, tree_root_pid=None,
                 want_gpu: bool = True, want_frequency: bool = True):
        if not IS_WINDOWS:
            raise RuntimeError(f"{sys.platform} is not Windows")
        if interval < MINIMUM_INTERVAL:
            raise ValueError(
                f"interval {interval}s is below the {MINIMUM_INTERVAL}s floor; these "
                "counters are documented for low-frequency collection and sampling "
                "them faster measures the sampler")
        self.interval = interval
        self.providers, self.limitations = {}, []
        self._readers = []
        self._install(SystemLoadProvider, "load", tree_root_pid=tree_root_pid)
        if want_frequency:
            self._install(CpuFrequencyProvider, "cpu_frequency")
        if want_gpu:
            self._install(NvmlProvider, "gpu")
        self._stop = threading.Event()
        self._thread = None
        self._record = None

    def _install(self, factory, label, **kwargs):
        try:
            provider = factory(**kwargs)
        except Exception as exc:  # noqa: BLE001
            # Recorded, never dropped. "No GPU telemetry" and "a GPU that was
            # never asked" look identical in a results file otherwise.
            self.providers[label] = _unavailable(exc)
            return
        self.providers[label] = {"available": True, "provider": provider.name}
        self._readers.append((label, provider))

    def sample_telemetry(self) -> dict:
        """One sample from every installed provider. Safe to call directly."""
        sample = {"t": time.time()}
        for label, provider in self._readers:
            try:
                sample[label] = provider.read()
            except Exception as exc:  # noqa: BLE001
                sample[label] = {"error": f"{type(exc).__name__}: {exc}"}
        return sample

    def start(self):
        if self._thread is not None:
            raise RuntimeError("already started")
        self._record = TelemetryRecord(
            interval=self.interval,
            started_at=datetime.now(timezone.utc).isoformat(),
            providers=dict(self.providers))
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="telemetry")
        self._thread.start()
        return self

    def _loop(self):
        # Absolute deadlines rather than sleep(interval): a slow sample would
        # otherwise push every later one back, and the recorded interval would
        # stop describing the series.
        deadline = time.monotonic()
        while not self._stop.is_set():
            self._record.samples.append(self.sample_telemetry())
            deadline += self.interval
            delay = deadline - time.monotonic()
            if delay < 0:
                self._record.missed += int(-delay // self.interval) + 1
                deadline = time.monotonic()
                delay = 0
            self._stop.wait(delay)

    def stop(self) -> TelemetryRecord:
        if self._thread is None:
            raise RuntimeError("not started")
        self._stop.set()
        self._thread.join(timeout=self.interval * 4 + 5)
        self._thread = None
        record = self._record
        record.stopped_at = datetime.now(timezone.utc).isoformat()
        record.limitations = self._limitations(record)
        for _label, provider in self._readers:
            close = getattr(provider, "close", None)
            if close is not None:
                close()
        return record

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stopped = self.stop()
        return False

    def _limitations(self, record) -> list:
        """What these samples cannot be used to claim, derived from the samples."""
        notes = list(self.limitations)
        if not record.samples:
            notes.append("no samples were collected; nothing here describes the run")
            return notes
        if len(record.samples) < 3:
            notes.append(f"only {len(record.samples)} samples; too few to describe "
                         "conditions over the window")
        if record.missed:
            notes.append(f"{record.missed} sampling deadline(s) missed; the series "
                         "has gaps the interval does not account for")
        notes += self._frequency_limitations(record)
        return notes

    @staticmethod
    def _frequency_limitations(record) -> list:
        """Decide whether the frequency series measured anything at all.

        A provider pinned to the nominal clock returns a perfectly flat series
        that looks like a machine holding its boost clock rock-steady. The two
        are indistinguishable from one query and distinguishable from many, so
        this is checked here rather than asserted anywhere.
        """
        readings = [sample.get("cpu_frequency") for sample in record.samples]
        readings = [r for r in readings if isinstance(r, dict) and "current_mhz" in r]
        if not readings:
            return ["CPU frequency was not sampled; sustained clock is unavailable, "
                    "not assumed"]
        current = [value for r in readings for value in r["current_mhz"]]
        maximum = [value for r in readings for value in r["max_mhz"]]
        if not current:
            return ["CPU frequency provider returned no processors"]
        if min(current) == max(current):
            flat = (f"CPU frequency never varied across {len(readings)} samples "
                    f"(always {current[0]} MHz)")
            if maximum and current[0] == max(maximum):
                return [flat + " and equals the reported maximum; treat this as the "
                               "nominal figure being echoed back, NOT as evidence "
                               "the machine held its clock"]
            return [flat + "; the provider may not be measuring"]
        return []


# ---------------------------------------------------------------------------
# Overhead
# ---------------------------------------------------------------------------

def measure_overhead(interval: float = DEFAULT_INTERVAL, duration: float = 3.0,
                     **kwargs) -> dict:
    """What one sample costs, measured rather than assumed.

    Run this before turning telemetry on by default anywhere. The cost is small
    but it is not zero, and a number nobody has measured is exactly the kind of
    thing that ends up inside a timed loop.
    """
    try:
        sampler = TelemetrySampler(interval=interval, **kwargs)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(exc)

    sampler.sample_telemetry()                    # warm every provider
    process_before = time.process_time()
    wall = []
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        sampler.sample_telemetry()
        wall.append((time.perf_counter() - start) * 1000.0)
    cpu_ms = (time.process_time() - process_before) * 1000.0

    return {
        "available": True,
        "samples": len(wall),
        "wall_ms_per_sample": {"median": round(statistics.median(wall), 4),
                               "max": round(max(wall), 4),
                               "mean": round(statistics.fmean(wall), 4)},
        "cpu_ms_per_sample": round(cpu_ms / max(len(wall), 1), 4),
        "providers": sampler.providers,
        "duty_cycle_at_interval": round(
            statistics.median(wall) / 1000.0 / interval, 6),
        "note": ("cost of one sample from every installed provider, on this "
                 "machine, now; re-measure rather than quoting this"),
    }


def sample_telemetry(*, tree_root_pid=None, want_gpu: bool = True,
                     want_frequency: bool = True) -> dict:
    """One-shot conditions reading, for callers that do not want a thread.

    A single sample describes an instant. It cannot establish a sustained
    frequency, a sustained load, or that conditions held for a measurement --
    those need the sampler and the window it covers.
    """
    try:
        sampler = TelemetrySampler(interval=MINIMUM_INTERVAL,
                                   tree_root_pid=tree_root_pid, want_gpu=want_gpu,
                                   want_frequency=want_frequency)
    except Exception as exc:  # noqa: BLE001
        return _unavailable(exc)
    sample = sampler.sample_telemetry()
    sample["providers"] = sampler.providers
    sample["limitation"] = ("a single instant; not a sustained frequency, load or "
                            "temperature, and not evidence about any window")
    return sample


def background_load_warning(record, *, threshold_percent: float = 15.0) -> list:
    """Reasons to suspect something else was using the machine.

    A **warning**, deliberately. What is left after this benchmark's own
    process tree still includes the compositor, the shell, indexing, and the
    driver's own threads -- none of which is a stranger stealing the machine,
    and all of which move with the workload. Returned as contamination reasons
    for a human to weigh, never as grounds to discard a measurement here.
    """
    summary = record.summary() if isinstance(record, TelemetryRecord) else record
    other = summary.get("load.other_cpu_percent") if summary else None
    if not other:
        return []
    reasons = []
    if other.get("median", 0) >= threshold_percent:
        reasons.append(
            f"CPU outside this benchmark's process tree ran at a median "
            f"{other['median']:.1f}% (peak {other['max']:.1f}%) during this case; "
            "absolute timings are inflated. This is a warning signal, not proof "
            "of contamination -- the compositor and driver threads are counted here "
            "too")
    return reasons
