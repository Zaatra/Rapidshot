"""What continuous capture's queue policies change, measured on the real queue.

    python benchmarks/freshness_policies.py                # 5 shuffled runs per case
    python benchmarks/freshness_policies.py --repeats 9 --width 2560 --height 1600
    python benchmarks/freshness_policies.py --seed 1234 --json out.json   # replay an order

RapidShot's own capture thread, high-resolution timer, queue, policies and
readers run unmodified. Only the screen is simulated, so this needs no GPU and
no display:

- **The screen presents at a fixed rate.** A grab between two presents finds
  nothing new and returns None, as Desktop Duplication does. A grab after
  several reports them as one frame with ``accumulated_frames`` > 1 and the
  newest present time, as ``DXGI_OUTDUPL_FRAME_INFO`` does.
- **A new frame costs what a converting colour mode costs** (RGB, the
  default): a full-frame write into an output-pool buffer, or, when every pool
  buffer is out, a freshly allocated array -- ``_checkout_output_buffer``'s
  fallback, whose first-touch page faults are the expensive part.

What it does not measure: the GPU copy and conversion, which cost the same
under every policy, and present-to-acquire latency, which is the screen's and
the backend's, not the queue's. Ages here therefore start at the simulated
present and measure only what the queue adds.

Each scenario is a consumer pattern a real program has. The numbers that
matter are the consumer's (how old, in what order, how many) and the
producer's (CPU per frame, memory held), side by side per policy.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import threading
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.frame import _qpc_freq, _qpc_now  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool, PoolExhaustedError  # noqa: E402


# --------------------------------------------------------------------------
# Process memory, from the OS rather than from Python's view of it
# --------------------------------------------------------------------------

class _PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_psapi = ctypes.WinDLL("psapi", use_last_error=True)
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.OpenThread.restype = wintypes.HANDLE
_kernel32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.GetThreadTimes.argtypes = [wintypes.HANDLE] + [ctypes.c_void_p] * 4
_psapi.GetProcessMemoryInfo.argtypes = [
    wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]


def _memory():
    counters = _PROCESS_MEMORY_COUNTERS_EX()
    counters.cb = ctypes.sizeof(counters)
    if not _psapi.GetProcessMemoryInfo(
            _kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PrivateUsage, counters.PageFaultCount


def _thread_cpu_seconds(handle) -> float:
    creation, exit_, kernel, user = (wintypes.FILETIME() for _ in range(4))
    _kernel32.GetThreadTimes(
        handle, ctypes.byref(creation), ctypes.byref(exit_),
        ctypes.byref(kernel), ctypes.byref(user))

    def seconds(ft):
        return ((ft.dwHighDateTime << 32) | ft.dwLowDateTime) / 1e7
    return seconds(kernel) + seconds(user)


# --------------------------------------------------------------------------
# The simulated screen, behind a real ScreenCapture
# --------------------------------------------------------------------------

class _Duplicator:
    last_present_time = 0
    accumulated_frames = 0


class SimulatedCamera:
    """A ScreenCapture whose grab() reads a simulated screen."""

    def __init__(self, width, height, screen_hz, pool_size=2, max_buffer_len=64):
        cam = ScreenCapture.__new__(ScreenCapture)
        cam._capture_lock = threading.Lock()
        cam._frame_available_event = threading.Event()
        cam._stop_capture_event = threading.Event()
        cam._last_dup_source = None
        cam._pooled_frames_deque = None
        cam.max_buffer_len = max_buffer_len
        cam.nvidia_gpu = False
        cam.memory_pool = None
        cam.region = (0, 0, width, height)
        cam._capture_permanently_failed = False
        cam._needs_reinit = False
        cam._last_capture_error_message = ""
        cam._frame_count = 0
        cam._timer_handle = None
        cam._capture_thread = None
        cam.is_capturing = False
        cam._duplicator = _Duplicator()
        cam._validate_region = lambda region: None
        cam._grab = self._grab
        self.cam = cam

        self.shape = (height, width, 3)
        self.frame_bytes = width * height * 3
        self.pool = NumpyMemoryPool(self.shape, np.uint8, pool_size)
        self.source = np.random.default_rng(0).integers(
            0, 255, self.shape, dtype=np.uint8)
        self.ticks_per_present = _qpc_freq() / screen_hz
        self.epoch = _qpc_now()
        self.last_present = 0
        self.grab_seconds = []
        self.allocations = 0
        self.producer_handle = None
        self.stall_every_s = None
        self.stall_for_s = 0.0
        self._next_stall = None

    def _grab(self, region):
        if self.producer_handle is None:
            self.producer_handle = _kernel32.OpenThread(
                0x0040, False, threading.get_native_id())  # QUERY_INFORMATION
        if self.stall_every_s is not None:
            now = time.monotonic()
            if self._next_stall is None:
                self._next_stall = now + self.stall_every_s
            elif now >= self._next_stall:
                # A capture that stops answering for a while -- a rebuild
                # after an access loss, a driver hiccup -- and then resumes.
                time.sleep(self.stall_for_s)
                self._next_stall = time.monotonic() + self.stall_every_s

        started = time.perf_counter()
        present = int((_qpc_now() - self.epoch) // self.ticks_per_present)
        if present <= self.last_present:
            return None                      # Nothing presented since last grab
        accumulated = present - self.last_present
        self.last_present = present
        dup = self.cam._duplicator
        dup.last_present_time = int(self.epoch + present * self.ticks_per_present)
        dup.accumulated_frames = accumulated

        try:
            frame = self.pool.checkout()
            target = frame.array
        except PoolExhaustedError:
            frame = target = np.empty(self.shape, np.uint8)
            self.allocations += 1
        np.copyto(target, self.source)       # The conversion's full-frame write
        target.reshape(-1)[:8] = np.frombuffer(
            np.int64(present).tobytes(), np.uint8)
        self.grab_seconds.append(time.perf_counter() - started)
        return frame


def _sequence(frame) -> int:
    array = np.asarray(getattr(frame, "array", frame))
    return int(np.frombuffer(array.reshape(-1)[:8].tobytes(), np.int64)[0])


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------

SCENARIOS = {
    "inference": dict(
        title="Model inference: screen and capture at 240 Hz, 16 ms of work per frame",
        screen_hz=240, target_fps=240, work_ms=16, reader="latest",
        cases=[("drop_oldest", None), ("latest", None)]),
    "bursty": dict(
        title="Agent with pauses: 120 Hz, 3 ms per frame, a 250 ms pause every second",
        screen_hz=120, target_fps=120, work_ms=3, pause_every_s=1.0, pause_ms=250,
        reader="latest",
        cases=[("drop_oldest", None), ("drop_oldest", 20), ("latest", None),
               ("latest", 20)]),
    "capture-stall": dict(
        title="Capture stalls 300 ms every 2 s (a rebuild), 120 Hz, 20 ms per frame",
        screen_hz=120, target_fps=120, work_ms=20, stall_every_s=2.0,
        stall_ms=300, reader="latest",
        cases=[("drop_oldest", None), ("latest", None), ("latest", 50)]),
    "idle": dict(
        title="Mostly idle screen: 2 updates a second, capture at 60 Hz, 5 ms per frame",
        screen_hz=2, target_fps=60, work_ms=5, reader="latest",
        cases=[("latest", None), ("latest", 20)]),
    "recording": dict(
        title="Recording a 60 Hz screen with a 30 fps writer, read in order",
        screen_hz=60, target_fps=60, work_ms=33, reader="next",
        cases=[("drop_oldest", None), ("block", None)]),
}


def run_case(scenario, policy, max_age_ms, seconds, width, height):
    sim = SimulatedCamera(width, height, scenario["screen_hz"])
    sim.stall_every_s = scenario.get("stall_every_s")
    sim.stall_for_s = scenario.get("stall_ms", 0) / 1000.0
    cam = sim.cam
    read = (cam.get_latest_frame_buffer if scenario["reader"] == "latest"
            else cam.get_next_frame_buffer)

    private_before, faults_before = _memory()
    peak = {"private": private_before, "queued": 0}
    queued_samples = []
    done = threading.Event()

    def sample():
        while not done.wait(0.005):
            private, _ = _memory()
            peak["private"] = max(peak["private"], private)
            with cam._capture_lock:
                queue = cam._pooled_frames_deque
                queued = len(queue) if queue is not None else 0
            queued_samples.append(queued)
            peak["queued"] = max(peak["queued"], queued)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()

    cam.start(target_fps=scenario["target_fps"], policy=policy,
              max_age_ms=max_age_ms)
    ages, sequences = [], []
    backwards = repeats = timeouts = 0
    work_s = scenario["work_ms"] / 1000.0
    pause_every = scenario.get("pause_every_s")
    next_pause = time.monotonic() + pause_every if pause_every else None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        frame = read()
        if frame is None:
            timeouts += 1
            continue
        age = cam.stream_stats["age_ms_last"]
        sequence = _sequence(frame)
        release = getattr(frame, "release", None)
        if release is not None:
            release()
        if sequences:
            if sequence < sequences[-1]:
                backwards += 1
            elif sequence == sequences[-1]:
                repeats += 1
        sequences.append(sequence)
        if age is not None:
            ages.append(age)
        time.sleep(work_s)                    # The model, the OCR, the encoder
        if next_pause is not None and time.monotonic() >= next_pause:
            time.sleep(scenario["pause_ms"] / 1000.0)
            next_pause = time.monotonic() + pause_every
    producer_cpu = (_thread_cpu_seconds(sim.producer_handle)
                    if sim.producer_handle else 0.0)
    cam.stop()
    done.set()
    sampler.join()
    _, faults_after = _memory()
    if sim.producer_handle:
        _kernel32.CloseHandle(sim.producer_handle)

    stats = cam.stream_stats
    captured = stats["frames_captured"]
    ages_sorted = sorted(ages)

    def pct(p):
        if not ages_sorted:
            return None
        return ages_sorted[min(len(ages_sorted) - 1, int(p * len(ages_sorted)))]

    return {
        "policy": policy,
        "max_age_ms": max_age_ms,
        "frames_consumed": len(sequences),
        "consumed_fps": len(sequences) / seconds,
        "captured_fps": captured / seconds,
        "age_ms_p50": pct(0.50),
        "age_ms_p95": pct(0.95),
        "age_ms_max": ages_sorted[-1] if ages_sorted else None,
        "older_than_previous": backwards,
        "same_as_previous": repeats,
        "reader_timeouts": timeouts,
        "producer_cpu_ms_per_frame": (producer_cpu * 1000.0 / captured
                                      if captured else None),
        "grab_ms_mean": (statistics.fmean(sim.grab_seconds) * 1000.0
                         if sim.grab_seconds else None),
        "fallback_allocations": sim.allocations,
        "page_faults": faults_after - faults_before,
        "queued_frames_peak": peak["queued"],
        "queued_mb_mean": (statistics.fmean(queued_samples) * sim.frame_bytes / 2**20
                           if queued_samples else 0.0),
        "private_mb_peak_over_start": (peak["private"] - private_before) / 2**20,
        "stream_stats": stats,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt(value, digits=1):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------
# Live: a real camera on the real primary display
# --------------------------------------------------------------------------
#
# The screen is latency_source.exe -- the test source `rapidshot benchmark`
# and the capture planner use -- covering the primary display and logging the
# QPC time of every Present(). Each frame carries its ID in a marker in the
# top-left corner, so a frame's true pixel age is the time from its own
# Present() to the moment the consumer holds it: no simulated clock.

LIVE_SCENARIOS = {
    "inference": dict(
        title="Model inference: capture at the display's rate, 16 ms of work per frame",
        work_ms=16, reader="latest",
        cases=[("drop_oldest", None), ("latest", None)]),
    "bursty": dict(
        title="Agent with pauses: 3 ms per frame, a 250 ms pause every second",
        work_ms=3, pause_every_s=1.0, pause_ms=250, reader="latest",
        cases=[("drop_oldest", None), ("drop_oldest", 20), ("latest", None),
               ("latest", 20)]),
    "recording": dict(
        title="Recording with a 30 fps writer, read in order",
        work_ms=33, reader="next",
        cases=[("drop_oldest", None), ("block", None)]),
    "idle": dict(
        title="Mostly idle screen: the source presents twice a second, 5 ms per frame",
        source_fps=2, work_ms=5, reader="latest",
        cases=[("latest", None), ("latest", 20)]),
}


def _primary_mode():
    import rapidshot
    from rapidshot.planner import _display_mode, _resolve
    from rapidshot.sources import Monitor

    _kind, primary, _device = _resolve(rapidshot.get_factory(), Monitor())
    return _display_mode(primary)


def run_live_case(scenario, policy, max_age_ms, seconds, log, hz, backend):
    """One timed run. ``log`` is the source's PresentLog, shared by every run
    on that source and caught up here before timing starts: reading a whole
    long log inside the window is what starved capture in the first live run."""
    import rapidshot
    from rapidshot.sources import Monitor
    from rapidshot.util.pixel_age import decode_marker, qpc_clock

    now, _ = qpc_clock()
    log.refresh()
    camera = rapidshot.create(source=Monitor(), backend=backend, output_color="RGB")
    read = (camera.get_latest_frame_buffer if scenario["reader"] == "latest"
            else camera.get_next_frame_buffer)
    thread_handle = None
    try:
        for _ in range(10):               # Warm-up: first frames, pools
            frame = camera.grab()
            release = getattr(frame, "release", None)
            if release is not None:
                release()
        private_before, faults_before = _memory()
        peak = {"private": private_before}
        queued_samples = []
        done = threading.Event()

        def sample():
            while not done.wait(0.005):
                private, _ = _memory()
                peak["private"] = max(peak["private"], private)
                with camera._capture_lock:
                    queue = camera._pooled_frames_deque
                    queued_samples.append(len(queue) if queue is not None else 0)

        sampler = threading.Thread(target=sample, daemon=True)
        sampler.start()
        log.refresh()
        first_present = max(log.frames, default=0)
        window_start = time.monotonic()
        camera.start(target_fps=hz, policy=policy, max_age_ms=max_age_ms)
        thread_handle = _kernel32.OpenThread(
            0x0040, False, camera._capture_thread.native_id)

        ages, ids = [], []
        backwards = repeats = timeouts = undecoded = 0
        frame_bytes = None
        work_s = scenario["work_ms"] / 1000.0
        pause_every = scenario.get("pause_every_s")
        next_pause = time.monotonic() + pause_every if pause_every else None
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            frame = read()
            if frame is None:
                timeouts += 1
                continue
            ready = now()
            array = np.asarray(getattr(frame, "array", frame))
            frame_bytes = array.nbytes
            frame_id = decode_marker(array, np)
            release = getattr(frame, "release", None)
            if release is not None:
                release()
            if frame_id is None:
                undecoded += 1
            else:
                try:
                    age = log.age(frame_id, ready)
                except ValueError:
                    age = None
                if age is not None:
                    ages.append(age)
                if ids:
                    if frame_id < ids[-1]:
                        backwards += 1
                    elif frame_id == ids[-1]:
                        repeats += 1
                ids.append(frame_id)
            time.sleep(work_s)
            if next_pause is not None and time.monotonic() >= next_pause:
                time.sleep(scenario["pause_ms"] / 1000.0)
                next_pause = time.monotonic() + pause_every
        capture_cpu = _thread_cpu_seconds(thread_handle)
        log.refresh()
        # The source's own count of what it presented in the window: a run
        # whose screen stopped changing measures the source, not the policy.
        source_fps = ((max(log.frames, default=0) - first_present)
                      / (time.monotonic() - window_start))
        camera.stop()
        done.set()
        sampler.join()
        _, faults_after = _memory()
        stats = camera.stream_stats
    finally:
        if thread_handle:
            _kernel32.CloseHandle(thread_handle)
        camera.release()

    captured = stats["frames_captured"]
    ages.sort()

    def pct(p):
        return ages[min(len(ages) - 1, int(p * len(ages)))] if ages else None

    consumed = stats["frames_consumed"]
    return {
        "policy": policy,
        "max_age_ms": max_age_ms,
        "frames_consumed": consumed,
        "consumed_fps": consumed / seconds,
        "captured_fps": captured / seconds,
        "age_ms_p50": pct(0.50),
        "age_ms_p95": pct(0.95),
        "age_ms_max": ages[-1] if ages else None,
        "older_than_previous": backwards,
        "same_as_previous": repeats,
        "reader_timeouts": timeouts,
        "undecoded_frames": undecoded,
        "source_fps": source_fps,
        "producer_cpu_ms_per_frame": (capture_cpu * 1000.0 / captured
                                      if captured else None),
        "grab_ms_mean": None,
        "fallback_allocations": None,
        "page_faults": faults_after - faults_before,
        "queued_frames_peak": max(queued_samples) if queued_samples else 0,
        "queued_mb_mean": (statistics.fmean(queued_samples) * (frame_bytes or 0) / 2**20
                           if queued_samples else 0.0),
        "private_mb_peak_over_start": (peak["private"] - private_before) / 2**20,
        "stream_stats": stats,
    }


def live_main(args):
    import gc
    import random
    from rapidshot.planner import _Source
    from rapidshot.util.pixel_age import PresentLog

    width, height, hz = _primary_mode()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    names = args.scenario or list(LIVE_SCENARIOS)
    rng = random.Random(seed)
    # One test source per presentation rate: replacing the source process is
    # what stalled it on an Intel desktop (latency_source.rs), so each rate
    # gets one source for all of its runs, shuffled among themselves.
    groups = {}
    for name in names:
        rate = LIVE_SCENARIOS[name].get("source_fps", hz)
        groups.setdefault(rate, []).extend(
            (name, policy, max_age_ms, repeat)
            for repeat in range(args.repeats)
            for policy, max_age_ms in LIVE_SCENARIOS[name]["cases"])
    total = sum(len(jobs) for jobs in groups.values())
    print(f"live, {width}x{height} @ {hz} Hz, backend {args.backend}: {total} runs, "
          f"shuffle seed {seed}", file=sys.stderr, flush=True)

    runs, index, unhealthy = {}, 0, []
    for rate, jobs in groups.items():
        rng.shuffle(jobs)
        source = _Source(width, height, rate)
        log = PresentLog(source.log_path, source.frequency)
        try:
            for name, policy, max_age_ms, repeat in jobs:
                index += 1
                print(f"[{index}/{total}] {name}: {policy}"
                      + (f", max_age_ms={max_age_ms}" if max_age_ms else "")
                      + f" (run {repeat + 1}, source {rate} fps)",
                      file=sys.stderr, flush=True)
                result = run_live_case(LIVE_SCENARIOS[name], policy, max_age_ms,
                                       args.seconds, log, hz, args.backend)
                result.update(run=repeat + 1, order=index)
                if result["source_fps"] < 0.9 * rate:
                    bad = (f"run {index}: the test source presented "
                           f"{result['source_fps']:.1f} fps of {rate}")
                    unhealthy.append(bad)
                    print("WARNING " + bad, file=sys.stderr, flush=True)
                runs.setdefault((name, policy, max_age_ms), []).append(result)
                gc.collect()
                time.sleep(args.pause)
        finally:
            log.close()
            source.close()

    results = {
        name: [aggregate(runs[(name, policy, max_age_ms)])
               for policy, max_age_ms in LIVE_SCENARIOS[name]["cases"]]
        for name in names}
    text = report(results, width, height, args.seconds, args.repeats, seed,
                  scenarios=LIVE_SCENARIOS,
                  heading=f"Live capture, {args.backend}, {width}x{height} @ {hz} Hz RGB")
    rates = [r["source_fps"] for rs in runs.values() for r in rs]
    text += (f"\nTest source: {min(rates):.1f}-{max(rates):.1f} presents a second "
             f"across runs.\n")
    if unhealthy:
        text += ("\n**Not trustworthy: the test source fell below 90% of its rate "
                 "in these runs.**\n\n" + "\n".join(f"- {u}" for u in unhealthy) + "\n")
    print(text)
    if args.json:
        args.json.write_text(json.dumps({
            "live": True, "backend": args.backend, "seed": seed,
            "unhealthy_runs": unhealthy,
            "repeats": args.repeats, "seconds": args.seconds,
            "width": width, "height": height, "hz": hz,
            "medians": results,
            "runs": {f"{n}|{p}|{a}": r for (n, p, a), r in runs.items()},
        }, indent=2), encoding="utf-8")
    return 0


#: Per-run numbers reduced to a median across repeats.
_METRICS = (
    "frames_consumed", "consumed_fps", "captured_fps", "age_ms_p50",
    "age_ms_p95", "age_ms_max", "older_than_previous", "same_as_previous",
    "reader_timeouts", "producer_cpu_ms_per_frame", "grab_ms_mean",
    "fallback_allocations", "page_faults", "queued_frames_peak",
    "queued_mb_mean", "private_mb_peak_over_start", "undecoded_frames",
)
_COUNTERS = (
    "frames_captured", "frames_consumed", "stale_frames_replaced",
    "consumer_drops", "frames_expired", "capture_drops",
)


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def aggregate(runs):
    """Medians across repeats of one case, with each metric's min and max."""
    out = {"policy": runs[0]["policy"], "max_age_ms": runs[0]["max_age_ms"],
           "runs": len(runs), "range": {}}
    for key in _METRICS:
        values = [r.get(key) for r in runs if r.get(key) is not None]
        out[key] = _median(values)
        out["range"][key] = (min(values), max(values)) if values else None
    out["stream_stats"] = {
        key: _median([r["stream_stats"][key] for r in runs]) for key in _COUNTERS}
    return out


def _spread(row, key, digits=1):
    """``median (min-max)`` when the runs disagree, else the median alone."""
    value, bounds = row[key], row["range"].get(key)
    if bounds is None or bounds[0] == bounds[1]:
        return _fmt(value, digits)
    return f"{_fmt(value, digits)} ({_fmt(bounds[0], digits)}-{_fmt(bounds[1], digits)})"


def _num(value):
    """A median of counts: whole when it is, one decimal when two runs split it."""
    if value is None:
        return "-"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def report(results, width, height, seconds, repeats, seed, scenarios=None,
           heading=None):
    scenarios = scenarios or SCENARIOS
    heading = heading or f"Queue policies, {width}x{height} RGB"
    lines = [f"# {heading}, {seconds:g} s per run",
             "",
             f"Each case ran {repeats} times, in shuffled order (seed {seed}). "
             "Cells are medians across runs, with the min-max range in brackets "
             "where the runs differed.",
             ""]
    for name, scenario in scenarios.items():
        rows = results.get(name)
        if not rows:
            continue
        reader = ("get_latest_frame()" if scenario["reader"] == "latest"
                  else "get_next_frame()")
        lines += [f"## {scenario['title']}", "", f"Reader: `{reader}`", "",
                  "| policy | consumed fps | age p50 ms | age p95 ms | age max ms "
                  "| older than previous | expired | dropped (consumer / replaced) "
                  "| capture_drops | capture CPU ms/frame | grab ms/frame "
                  "| fallback allocs | queued MB mean | private MB peak |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: "
                  "| ---: | ---: | ---: | ---: | ---: |"]
        for r in rows:
            label = r["policy"] + (f", max_age_ms={r['max_age_ms']:g}"
                                   if r["max_age_ms"] else "")
            s = r["stream_stats"]
            lines.append(
                f"| `{label}` | {_fmt(r['consumed_fps'])} "
                f"| {_spread(r, 'age_ms_p50')} | {_spread(r, 'age_ms_p95')} "
                f"| {_spread(r, 'age_ms_max')} "
                f"| {_spread(r, 'older_than_previous', 0)} | {_num(s['frames_expired'])} "
                f"| {_num(s['consumer_drops'])} / {_num(s['stale_frames_replaced'])} "
                f"| {_num(s['capture_drops'])} "
                f"| {_spread(r, 'producer_cpu_ms_per_frame', 2)} "
                f"| {_spread(r, 'grab_ms_mean', 2)} "
                f"| {_spread(r, 'fallback_allocations', 0)} "
                f"| {_spread(r, 'queued_mb_mean', 0)} "
                f"| {_spread(r, 'private_mb_peak_over_start', 0)} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), action="append")
    parser.add_argument("--repeats", type=int, default=5,
                        help="runs per case; the report gives medians (default 5)")
    parser.add_argument("--seed", type=int,
                        help="shuffle seed, to replay an order (default: random)")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds of idle between runs (default 1)")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--live", action="store_true",
                        help="a real camera on the primary display, under the test "
                             "source latency_source.exe, instead of a simulated screen")
    parser.add_argument("--backend", choices=("dxgi", "wgc"), default="dxgi",
                        help="capture backend for --live (default dxgi)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.live:
        if args.scenario and not set(args.scenario) <= set(LIVE_SCENARIOS):
            parser.error(f"--live scenarios: {', '.join(LIVE_SCENARIOS)}")
        return live_main(args)

    import gc
    import random
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    names = args.scenario or list(SCENARIOS)
    # Every run of every case in one random order, so neither a case's
    # position nor what ran before it -- a policy that left the heap large,
    # a warm cache, a CPU that has throttled -- lines up with the policy.
    jobs = [(name, policy, max_age_ms, repeat)
            for repeat in range(args.repeats)
            for name in names
            for policy, max_age_ms in SCENARIOS[name]["cases"]]
    random.Random(seed).shuffle(jobs)
    print(f"{len(jobs)} runs, shuffle seed {seed}", file=sys.stderr, flush=True)

    runs = {}
    for index, (name, policy, max_age_ms, repeat) in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {name}: {policy}"
              + (f", max_age_ms={max_age_ms}" if max_age_ms else "")
              + f" (run {repeat + 1})", file=sys.stderr, flush=True)
        result = run_case(SCENARIOS[name], policy, max_age_ms, args.seconds,
                          args.width, args.height)
        result.update(run=repeat + 1, order=index)
        runs.setdefault((name, policy, max_age_ms), []).append(result)
        gc.collect()
        time.sleep(args.pause)

    results = {
        name: [aggregate(runs[(name, policy, max_age_ms)])
               for policy, max_age_ms in SCENARIOS[name]["cases"]]
        for name in names}

    print(report(results, args.width, args.height, args.seconds,
                 args.repeats, seed))
    if args.json:
        args.json.write_text(json.dumps({
            "seed": seed, "repeats": args.repeats, "seconds": args.seconds,
            "width": args.width, "height": args.height,
            "medians": results,
            "runs": {f"{n}|{p}|{a}": r for (n, p, a), r in runs.items()},
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
