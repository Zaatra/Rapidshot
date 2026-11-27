"""Windows worker telemetry for the capture benchmarks."""

from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
import math
import re
from statistics import median
import sys
import threading
import time
import tracemalloc
import uuid


_SECONDS_NS = 1_000_000_000
_MIN_CPU_DURATION_NS = 10 * _SECONDS_NS
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_DXGI_ERROR_NOT_FOUND = 0x887A0002
_PDH_MORE_DATA = 0x800007D2
_PDH_FMT_DOUBLE_NO_CAP = 0x200 | 0x8000
_PDH_PATHS = {
    "gpu": r"\GPU Engine(*)\Utilization Percentage",
    "thermal": r"\Thermal Zone Information(*)\Temperature",
}
_GPU_ENGINE = re.compile(
    r"pid_(\d+)_luid_(0x[0-9a-f]+)_(0x[0-9a-f]+)_phys_(\d+)_eng_(\d+)_engtype_([^#]+)(?:#\d+)?",
    re.I,
)


class TelemetryError(RuntimeError):
    """A required metric could not be sampled."""


class _ProcessMemoryCounters(ct.Structure):
    _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
        (name, ct.c_size_t) for name in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage",
            "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage",
        )
    ]


class _ProcessorPower(ct.Structure):
    _fields_ = [(name, wt.ULONG) for name in (
        "Number", "MaxMhz", "CurrentMhz", "MhzLimit", "MaxIdleState", "CurrentIdleState",
    )]


def _bind(dll, name, result, *arguments):
    function = getattr(dll, name)
    function.restype = result
    function.argtypes = arguments
    return function


def _filetime_ns(value):
    return ((value.dwHighDateTime << 32) | value.dwLowDateTime) * 100


def _hex_status(status):
    return f"0x{status & 0xffffffff:08X}"


def _luid(value):
    if value is None:
        return None
    return (int(value, 0) if isinstance(value, str) else int(value)) & 0xffffffffffffffff


def _luid_label(value):
    return f"0x{value:016x}" if value is not None else None


@dataclass(frozen=True)
class _CpuSample:
    at_ns: int
    cycles: int
    user_ns: int
    kernel_ns: int
    host_busy_ns: int | None


class _ProcessCounters:
    def __init__(self, pid):
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        open_process = _bind(kernel, "OpenProcess", wt.HANDLE, wt.DWORD, wt.BOOL, wt.DWORD)
        self._close = _bind(kernel, "CloseHandle", wt.BOOL, wt.HANDLE)
        self._process_times = _bind(kernel, "GetProcessTimes", wt.BOOL, wt.HANDLE, *([ct.POINTER(wt.FILETIME)] * 4))
        self._cycles = _bind(kernel, "QueryProcessCycleTime", wt.BOOL, wt.HANDLE, ct.POINTER(ct.c_uint64))
        self._memory = _bind(kernel, "K32GetProcessMemoryInfo", wt.BOOL, wt.HANDLE,
                             ct.POINTER(_ProcessMemoryCounters), wt.DWORD)
        self._system_times = _bind(kernel, "GetSystemTimes", wt.BOOL, *([ct.POINTER(wt.FILETIME)] * 3))
        group_count = _bind(kernel, "GetActiveProcessorGroupCount", wt.WORD)
        self._single_group = group_count() == 1
        self.handle = open_process(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not self.handle:
            raise TelemetryError(f"OpenProcess({pid}): {ct.WinError(ct.get_last_error())}")

    def close(self):
        if self.handle:
            self._close(self.handle)
            self.handle = None

    def cpu(self):
        creation, exited, kernel, user = (wt.FILETIME() for _ in range(4))
        cycles = ct.c_uint64()
        before = time.perf_counter_ns()
        if not self._process_times(self.handle, ct.byref(creation), ct.byref(exited), ct.byref(kernel), ct.byref(user)):
            raise TelemetryError(f"GetProcessTimes: {ct.WinError(ct.get_last_error())}")
        if not self._cycles(self.handle, ct.byref(cycles)):
            raise TelemetryError(f"QueryProcessCycleTime: {ct.WinError(ct.get_last_error())}")
        host_busy = None
        if self._single_group:
            idle, system_kernel, system_user = (wt.FILETIME() for _ in range(3))
            if self._system_times(ct.byref(idle), ct.byref(system_kernel), ct.byref(system_user)):
                host_busy = _filetime_ns(system_kernel) + _filetime_ns(system_user) - _filetime_ns(idle)
        at_ns = (before + time.perf_counter_ns()) // 2
        return _CpuSample(at_ns, cycles.value, _filetime_ns(user), _filetime_ns(kernel), host_busy)

    def memory(self):
        data = _ProcessMemoryCounters(cb=ct.sizeof(_ProcessMemoryCounters))
        if not self._memory(self.handle, ct.byref(data), data.cb):
            raise TelemetryError(f"K32GetProcessMemoryInfo: {ct.WinError(ct.get_last_error())}")
        return {
            "at_ns": time.perf_counter_ns(), "working_set_bytes": int(data.WorkingSetSize),
            "private_bytes": int(data.PrivateUsage), "os_peak_working_set_bytes": int(data.PeakWorkingSetSize),
            "os_peak_private_bytes": int(data.PeakPagefileUsage),
        }


def _cpu_delta(start, end, delivered_frames):
    elapsed_ns = end.at_ns - start.at_ns
    cycles, user, kernel = end.cycles - start.cycles, end.user_ns - start.user_ns, end.kernel_ns - start.kernel_ns
    if elapsed_ns <= 0 or min(cycles, user, kernel) < 0:
        raise TelemetryError("Win32 process counters regressed")
    elapsed_s = elapsed_ns / _SECONDS_NS
    cpu_ns = user + kernel
    long_enough = elapsed_ns >= _MIN_CPU_DURATION_NS
    host_load = None
    if long_enough and start.host_busy_ns is not None and end.host_busy_ns is not None:
        host_ns = end.host_busy_ns - start.host_busy_ns
        if host_ns >= cpu_ns:
            host_load = {"busy_core_percent": 100 * host_ns / elapsed_ns,
                         "other_processes_and_system_core_percent": 100 * (host_ns - cpu_ns) / elapsed_ns}
    return {
        "elapsed_seconds": elapsed_s,
        "start_ns": start.at_ns, "end_ns": end.at_ns, "delivered_frames": delivered_frames,
        "cycles": cycles, "cycles_per_second": cycles / elapsed_s,
        "cycles_per_delivered_frame": cycles / delivered_frames if delivered_frames else None,
        "user_time_ns": user, "kernel_time_ns": kernel,
        "percent_of_one_core": 100 * cpu_ns / elapsed_ns if long_enough else None,
        "cpu_ms_per_second": cpu_ns / (1_000_000 * elapsed_s) if long_enough else None,
        "utilization_unavailable_reason": None if long_enough else "The process CPU window was shorter than 10 seconds.",
        "host_load": host_load,
    }


def _memory_summary(samples):
    if not samples:
        return {}
    middle = (samples[0]["at_ns"] + samples[-1]["at_ns"]) // 2
    tail = [s for s in samples if s["at_ns"] >= middle]
    result = {"sample_count": len(samples), "steady_state_sample_count": len(tail)}
    for field in ("working_set_bytes", "private_bytes"):
        result[f"steady_{field}"] = median(s[field] for s in tail)
        result[f"sampled_peak_{field}"] = max(s[field] for s in samples)
        result[f"os_peak_{field}"] = max(s[f"os_peak_{field}"] for s in samples)
    return result


class _CpuPower:
    def __init__(self):
        kernel = ct.WinDLL("kernel32")
        active_count = _bind(kernel, "GetActiveProcessorCount", wt.DWORD, wt.WORD)
        self.count = active_count(0xffff)  # ALL_PROCESSOR_GROUPS
        dll = ct.WinDLL("powrprof")
        self.query = _bind(dll, "CallNtPowerInformation", wt.LONG, ct.c_int, ct.c_void_p, wt.ULONG,
                           ct.c_void_p, wt.ULONG)

    def sample(self):
        output = (_ProcessorPower * self.count)()
        status = self.query(11, None, 0, ct.byref(output), ct.sizeof(output))  # ProcessorInformation
        if status != 0:
            raise TelemetryError(f"CallNtPowerInformation(ProcessorInformation): {_hex_status(status)}")
        return {"at_ns": time.perf_counter_ns(), "current_mhz": [p.CurrentMhz for p in output],
                "maximum_mhz": [p.MaxMhz for p in output], "thermal_limit_mhz": [p.MhzLimit for p in output]}


class _PdhCounterValue(ct.Structure):
    _fields_ = [("status", wt.DWORD), ("value", ct.c_double)]


class _PdhItem(ct.Structure):
    _fields_ = [("name", wt.LPWSTR), ("counter", _PdhCounterValue)]


class _PdhSampler:
    def __init__(self, pid, gpu_adapter_luid):
        self.pid = pid
        self.adapter_luid = gpu_adapter_luid
        self.samples = {"gpu": [], "thermal": []}
        self.failures = {}
        self.counters = {}
        self.previous_ns = None
        self.handle = wt.HANDLE()
        dll = ct.WinDLL("pdh")
        open_query = _bind(dll, "PdhOpenQueryW", wt.DWORD, wt.LPCWSTR, ct.c_size_t, ct.POINTER(wt.HANDLE))
        add_counter = _bind(dll, "PdhAddEnglishCounterW", wt.DWORD, wt.HANDLE, wt.LPCWSTR, ct.c_size_t,
                            ct.POINTER(wt.HANDLE))
        self._close = _bind(dll, "PdhCloseQuery", wt.DWORD, wt.HANDLE)
        self._collect = _bind(dll, "PdhCollectQueryData", wt.DWORD, wt.HANDLE)
        self._array = _bind(dll, "PdhGetFormattedCounterArrayW", wt.DWORD, wt.HANDLE, wt.DWORD,
                            ct.POINTER(wt.DWORD), ct.POINTER(wt.DWORD), ct.POINTER(_PdhItem))
        status = open_query(None, 0, ct.byref(self.handle))
        if status:
            self.failures.update({key: f"PdhOpenQueryW: {_hex_status(status)}" for key in _PDH_PATHS})
            return
        for key, path in _PDH_PATHS.items():
            counter = wt.HANDLE()
            status = add_counter(self.handle, path, 0, ct.byref(counter))
            if status:
                self.failures[key] = f"PdhAddEnglishCounterW {path}: {_hex_status(status)}"
            else:
                self.counters[key] = counter

    def close(self):
        if self.handle:
            self._close(self.handle)
            self.handle = None

    def _values(self, counter):
        size, count = wt.DWORD(), wt.DWORD()
        status = self._array(counter, _PDH_FMT_DOUBLE_NO_CAP, ct.byref(size), ct.byref(count), None)
        if status != _PDH_MORE_DATA:
            return None, status
        buffer = ct.create_string_buffer(size.value)
        items = ct.cast(buffer, ct.POINTER(_PdhItem))
        status = self._array(counter, _PDH_FMT_DOUBLE_NO_CAP, ct.byref(size), ct.byref(count), items)
        if status:
            return None, status
        return [(item.name, item.counter.value) for item in items[:count.value]
                if item.name and item.counter.status in (0, 1) and math.isfinite(item.counter.value)], 0

    def sample(self):
        if not self.counters:
            return
        status = self._collect(self.handle)
        now = time.perf_counter_ns()
        if status:
            self.failures.update({key: f"PdhCollectQueryData: {_hex_status(status)}" for key in self.counters})
            self.previous_ns = None
            return
        previous, self.previous_ns = self.previous_ns, now
        for key, counter in self.counters.items():
            if key == "gpu" and previous is None:
                continue
            values, status = self._values(counter)
            if status:
                self.failures[key] = f"PdhGetFormattedCounterArrayW: {_hex_status(status)}"
            elif key == "gpu":
                rows = [_engine_sample(name, value, self.pid, self.adapter_luid) for name, value in values]
                self.samples[key].append({"at_ns": now, "interval_seconds": (now - previous) / _SECONDS_NS,
                                          "engines": [row for row in rows if row is not None]})
            else:
                self.samples[key].append({"at_ns": now, "zones": [
                    {"zone": name, "temperature_celsius": value - 273.15} for name, value in values if value > 0
                ]})


def _engine_sample(name, value, pid, adapter_luid):
    match = _GPU_ENGINE.fullmatch(name)
    if match is None or int(match[1]) != pid or not 0 <= value <= 100:
        return None
    luid = (int(match[2], 16) << 32) | int(match[3], 16)
    if adapter_luid is not None and luid != adapter_luid:
        return None
    return {"adapter_luid": _luid_label(luid), "physical_adapter": int(match[4]),
            "engine": int(match[5]), "engine_type": match[6], "utilization_percent": value}


def _gpu_summary(samples, reason, elapsed_s, delivered_frames):
    observations = {}
    for snapshot in samples:
        interval_s = snapshot["interval_seconds"]
        if interval_s <= 0:
            continue
        for engine in snapshot["engines"]:
            key = (engine["adapter_luid"], engine["physical_adapter"], engine["engine"], engine["engine_type"])
            observations.setdefault(key, []).append((interval_s, engine["utilization_percent"]))
    engines = []
    for (luid, physical, index, name), rows in observations.items():
        total_s = sum(t for t, _ in rows)
        average = sum(t * u for t, u in rows) / total_s
        engines.append({"adapter_luid": luid, "physical_adapter": physical, "engine": index, "engine_type": name,
                        "sample_count": len(rows), "sampled_seconds": total_s,
                        "mean_busy_percent": average, "peak_busy_percent": max(u for _, u in rows),
                        "busy_ms_per_second": 10 * average,
                        "estimated_busy_ms_per_delivered_frame": 10 * average * elapsed_s / delivered_frames
                        if delivered_frames and total_s >= 0.9 * elapsed_s else None})
    return {"samples": samples, "engines": engines, "available": bool(engines),
            "reason": None if engines else reason or "No valid process GPU engine samples."}


class ProcessSampler:
    """Sample an existing worker from its parent. Call finish before worker teardown."""

    def __init__(self, pid, sample_seconds=0.1, pdh=True, gpu_adapter_luid=None):
        if not isinstance(pid, int) or pid <= 0 or not math.isfinite(sample_seconds) or sample_seconds <= 0:
            raise ValueError("Expected a positive PID and sampling interval")
        self.pid = pid
        self.sample_seconds = sample_seconds
        self.enable_pdh = pdh
        self.adapter_luid = _luid(gpu_adapter_luid)
        self._process = None
        self._initial_cpu = None
        self._thread = None
        self._stop = threading.Event()
        self._memory = []
        self._frequency = []
        self._pdh = None
        self._power = None
        self._issues = {}

    def prepare(self):
        """Open process and counter handles before entering the timed window."""
        if self._initial_cpu is not None or self._stop.is_set():
            raise TelemetryError("ProcessSampler instances measure a single window")
        if self._process is not None:
            return self
        try:
            self._process = _ProcessCounters(self.pid)
            try:
                self._power = _CpuPower()
            except (OSError, AttributeError) as exc:
                self._issues["cpu_frequency"] = str(exc)
            if self.enable_pdh:
                try:
                    self._pdh = _PdhSampler(self.pid, self.adapter_luid)
                except (OSError, AttributeError) as exc:
                    self._issues["pdh"] = str(exc)
        except BaseException:
            self.close()
            raise
        return self

    def start(self):
        self.prepare()
        try:
            self._initial_cpu = self._process.cpu()
            self._thread = threading.Thread(target=self._collect, name="benchmark telemetry", daemon=True)
            self._thread.start()
        except BaseException:
            self.close()
            raise
        return self

    def _collect(self):
        next_host_sample = 0
        while not self._stop.is_set():
            started = time.perf_counter()
            try:
                self._memory.append(self._process.memory())
            except TelemetryError as exc:
                self._issues["memory"] = str(exc)
            if started >= next_host_sample:
                next_host_sample = started + 1
                if self._power is not None:
                    try:
                        self._frequency.append(self._power.sample())
                    except TelemetryError as exc:
                        self._issues["cpu_frequency"] = str(exc)
                if self._pdh is not None:
                    self._pdh.sample()
            self._stop.wait(max(0, started + self.sample_seconds - time.perf_counter()))

    def finish(self, delivered_frames=None):
        if delivered_frames is not None and (not isinstance(delivered_frames, int) or delivered_frames < 0):
            raise ValueError("Expected a nonnegative integer delivery count")
        if self._process is None or self._initial_cpu is None:
            raise TelemetryError("ProcessSampler is not running")
        try:
            final = self._process.cpu()
            self._stop.set()
            final_memory = None
            try:
                final_memory = self._process.memory()
            except TelemetryError as exc:
                self._issues["memory"] = str(exc)
            self._thread.join()
            if final_memory is not None:
                self._memory.append(final_memory)
                self._memory.sort(key=lambda row: row["at_ns"])
            cpu = _cpu_delta(self._initial_cpu, final, delivered_frames)
            pdh = self._pdh
            reason = self._issues.get("pdh") if self.enable_pdh else "PDH collection disabled."
            temperatures = pdh.samples["thermal"] if pdh is not None else []
            has_temperatures = any(row["zones"] for row in temperatures)
            return {"pid": self.pid, "cpu": cpu, "memory": _memory_summary(self._memory),
                    "memory_samples": self._memory, "cpu_frequency_samples": self._frequency,
                    "thermal": {"samples": temperatures, "available": has_temperatures,
                                "reason": None if has_temperatures else (pdh.failures.get("thermal") if pdh else reason)
                                or "No valid ACPI thermal zone samples."},
                    "gpu_engines": _gpu_summary(pdh.samples["gpu"] if pdh else [],
                                                pdh.failures.get("gpu") if pdh else reason,
                                                cpu["elapsed_seconds"], delivered_frames),
                    "sampling_issues": self._issues}
        finally:
            self.close()

    def close(self):
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join()
        if self._pdh is not None:
            self._pdh.close()
        if self._process is not None:
            self._process.close()
            self._process = None


class _Luid(ct.Structure):
    _fields_ = [("low", wt.DWORD), ("high", wt.LONG)]


class _AdapterDescription(ct.Structure):
    _fields_ = [
        ("description", wt.WCHAR * 128), ("vendor", wt.UINT), ("device", wt.UINT),
        ("subsys", wt.UINT), ("revision", wt.UINT), ("video_memory", ct.c_size_t),
        ("system_memory", ct.c_size_t), ("shared_memory", ct.c_size_t), ("luid", _Luid), ("flags", wt.UINT),
    ]


class _VideoMemoryInfo(ct.Structure):
    _fields_ = [(key, ct.c_uint64) for key in (
        "budget_bytes", "current_usage_bytes", "available_for_reservation_bytes", "current_reservation_bytes",
    )]


def _guid(value):
    return (ct.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le)


def _com_call(pointer, slot, result, arguments, *values):
    table = ct.cast(pointer, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
    return ct.WINFUNCTYPE(result, ct.c_void_p, *arguments)(table[slot])(pointer, *values)


def _release(pointer):
    if pointer:
        _com_call(pointer, 2, wt.ULONG, ())  # IUnknown::Release


def _dxgi_adapter3(luid, index):
    # Bare DXGI avoids loading a competing capture library into the benchmark worker.
    factory_id = _guid("770aae78-f26f-4dba-a829-253c83d1b387")  # IDXGIFactory1
    adapter_id = _guid("645967a4-1392-4310-a798-8053ce3e93fd")  # IDXGIAdapter3
    dll = ct.WinDLL("dxgi")
    create = _bind(dll, "CreateDXGIFactory1", wt.LONG, ct.c_void_p, ct.POINTER(ct.c_void_p))
    factory = ct.c_void_p()
    status = create(ct.byref(factory_id), ct.byref(factory))
    if status < 0 or not factory:
        raise TelemetryError(f"CreateDXGIFactory1: {_hex_status(status)}")
    try:
        candidate_index = index if luid is None else 0
        while True:
            candidate = ct.c_void_p()
            status = _com_call(factory, 12, wt.LONG, (wt.UINT, ct.POINTER(ct.c_void_p)),  # EnumAdapters1
                               candidate_index, ct.byref(candidate))
            if status & 0xffffffff == _DXGI_ERROR_NOT_FOUND:
                raise TelemetryError("The requested DXGI adapter was not found")
            if status < 0 or not candidate:
                raise TelemetryError(f"IDXGIFactory1.EnumAdapters1: {_hex_status(status)}")
            try:
                desc = _AdapterDescription()
                status = _com_call(candidate, 10, wt.LONG, (ct.POINTER(_AdapterDescription),),  # GetDesc1
                                   ct.byref(desc))
                if status < 0:
                    raise TelemetryError(f"IDXGIAdapter1.GetDesc1: {_hex_status(status)}")
                found_luid = ((desc.luid.high & 0xffffffff) << 32) | desc.luid.low
                if luid is None or found_luid == luid:
                    adapter = ct.c_void_p()
                    status = _com_call(candidate, 0, wt.LONG, (ct.c_void_p, ct.POINTER(ct.c_void_p)),  # QueryInterface
                                       ct.byref(adapter_id), ct.byref(adapter))
                    if status < 0 or not adapter:
                        raise TelemetryError(f"QueryInterface(IDXGIAdapter3): {_hex_status(status)}")
                    return adapter, found_luid, desc.description
            finally:
                _release(candidate)
            candidate_index += 1
    finally:
        _release(factory)


class VideoMemory:
    """Read the calling worker's DXGI usage for the selected adapter and node."""

    def __init__(self, adapter_luid=None, adapter_index=0, node_index=0):
        if not all(isinstance(index, int) and 0 <= index <= 0xffffffff for index in (adapter_index, node_index)):
            raise ValueError("DXGI adapter and node indices must be unsigned 32-bit values")
        self.adapter_luid = _luid(adapter_luid)
        self.node_index = node_index
        self.adapter = None
        self.reason = None
        self._pointer = None
        try:
            self._pointer, self.adapter_luid, self.adapter = _dxgi_adapter3(self.adapter_luid, adapter_index)
        except (OSError, AttributeError, TelemetryError) as exc:
            self.reason = str(exc)

    def sample(self):
        result = {"at_ns": time.perf_counter_ns(), "adapter_luid": _luid_label(self.adapter_luid),
                  "adapter": self.adapter, "node_index": self.node_index,
                  "local": None, "non_local": None, "unavailable": {}}
        for index, name in enumerate(("local", "non_local")):
            if self._pointer is None:
                result["unavailable"][name] = self.reason or "VideoMemory is closed."
                continue
            info = _VideoMemoryInfo()
            # IDXGIAdapter3::QueryVideoMemoryInfo
            status = _com_call(self._pointer, 14, wt.LONG, (wt.UINT, wt.UINT, ct.POINTER(_VideoMemoryInfo)),
                               self.node_index, index, ct.byref(info))
            if status < 0:
                result["unavailable"][name] = f"QueryVideoMemoryInfo: {_hex_status(status)}"
            else:
                result[name] = {key: getattr(info, key) for key, _ in _VideoMemoryInfo._fields_}
        return result

    def close(self):
        if self._pointer is not None:
            _release(self._pointer)
            self._pointer = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _allocation_domains(snapshot, numpy_domain):
    output = {key: {"bytes": 0, "blocks": 0} for key in ("python", "numpy", "other_traced")}
    for trace in snapshot.traces:
        key = "python" if trace.domain == 0 else "numpy" if trace.domain == numpy_domain else "other_traced"
        output[key]["bytes"] += trace.size
        output[key]["blocks"] += 1
    return output


class AllocationProfiler:
    """Track live Python and NumPy allocations in a dedicated worker round."""

    def __init__(self, stack_depth=1):
        self.stack_depth = stack_depth
        self._baseline = None

    def start(self):
        if tracemalloc.is_tracing():
            raise TelemetryError("tracemalloc is already running")
        tracemalloc.start(self.stack_depth)
        try:
            self._baseline = tracemalloc.take_snapshot()
            tracemalloc.reset_peak()
        except BaseException:
            tracemalloc.stop()
            raise
        return self

    def finish(self):
        if self._baseline is None or not tracemalloc.is_tracing():
            raise TelemetryError("AllocationProfiler is not running")
        try:
            current, peak = tracemalloc.get_traced_memory()
            overhead = tracemalloc.get_tracemalloc_memory()
            snapshot = tracemalloc.take_snapshot()
        finally:
            tracemalloc.stop()
        numpy_domain = getattr(getattr(sys.modules.get("numpy"), "lib", None), "tracemalloc_domain", None)
        before = _allocation_domains(self._baseline, numpy_domain)
        after = _allocation_domains(snapshot, numpy_domain)
        self._baseline = None
        return {"traced_current_bytes": current, "traced_peak_bytes": peak, "tracer_overhead_bytes": overhead,
                "numpy_domain": numpy_domain, "baseline_live": before, "final_live": after,
                "net_live": {name: {unit: values[unit] - before[name][unit] for unit in values}
                             for name, values in after.items()}}
