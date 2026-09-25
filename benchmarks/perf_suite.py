"""Rapidshot performance suite — reproducible numbers for every change.

Every roadmap ordering decision is a performance claim, so each change gets
measured against a stored baseline rather than asserted.

Usage
-----
Record a baseline before changing anything::

    python benchmarks/perf_suite.py --out baseline.json

Re-run after a change and compare::

    python benchmarks/perf_suite.py --out after.json --compare baseline.json

Compare against whichever committed baseline was recorded on this machine,
which is what a release should do -- naming one file gates on one machine and
silently stops gating everywhere else::

    python benchmarks/perf_suite.py --compare auto

Only the deterministic benchmarks (no desktop session needed)::

    python benchmarks/perf_suite.py --synthetic-only

Benchmark classes
-----------------
* **synthetic** — fixed inputs, no screen dependency. Stable enough to gate a
  regression on, and safe to run in CI.
* **live** — real DXGI capture. Depends on what is happening on screen, so
  treat these as indicative; a static desktop produces few frames by design
  (Desktop Duplication only reports changed content).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import machine_inventory
import result_store
import statistics_report
from result_store import CaseIdentity

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.getLogger("rapidshot").setLevel(logging.ERROR)

FRAME_W, FRAME_H = 1920, 1080
MODES = ("BGRA", "RGBA", "RGB", "BGR", "GRAY")

# A benchmark whose implementation is fixed forever. Any change in its timing
# between two runs is attributable to the machine, not the code, so it acts as
# a calibration reference for every other comparison.
CONTROL_BENCHMARK = "control.memcopy"

# Benchmarks faster than this are dominated by scheduler granularity rather than
# by the code under test, so a "regression" in them is not actionable. Measured:
# the sub-millisecond BGRA copies swing ~1.15x between back-to-back runs of
# identical code, while the millisecond-scale ones hold within 1.02x.
LOW_RESOLUTION_MS = 0.5

# Benchmarks whose *meaning* changed in a given release, and are therefore not
# comparable against any recording made before it. This is a different failure
# from noise: the numbers are both correct, they just measure different things,
# so no amount of repetition or drift correction reconciles them.
#
# ROADMAP.md section 10 records both cases. `pipeline.cpu_to_nchw` was a strawman
# reference arm rewritten in 2.2.0 and again in 2.3.0. `pipeline.gpu_plus_readback`
# was substantially a CPython allocator benchmark until 2.3.0 made `read_back`
# return bytes instead of a `Vec<f32>` that PyO3 turned into 1.2M Python floats
# per call -- 14.19 -> 2.24 ms, which is a change in the harness's own
# verification helper rather than in anything a consumer pays for.
#
# Without this, comparing a 2.3.0 run against the 2.1.0 `baseline.json` reports
# `gpu_plus_readback` as **FASTER 6.08x** and reads as a hardware result. It is
# not; it is a code change that had already landed, and the row it lands on is
# one of the two that Stage 6 was promoted on (section 11). A spurious
# improvement is the dangerous direction precisely because nobody investigates
# good news.
#
# Keyed by the version the row's definition changed *in*: a baseline recorded on
# an earlier version cannot be compared on it.
_REDEFINED_IN: Dict[str, str] = {
    "pipeline.cpu_to_nchw": "2.3.0",
    "pipeline.gpu_plus_readback": "2.3.0",
}


def _version_tuple(v: str) -> tuple:
    """Parse 'x.y.z' for ordering. Unparseable versions sort lowest.

    An absent or malformed version must compare as *older* than every known
    redefinition, so a recording that predates version stamping is treated as
    not comparable rather than silently compared.
    """
    try:
        return tuple(int(part) for part in str(v).split(".")[:3])
    except (TypeError, ValueError):
        return ()


def _redefined_since(name: str, baseline_version: str) -> Optional[str]:
    """The version that redefined `name`, if the baseline predates it."""
    changed_in = _REDEFINED_IN.get(name)
    if changed_in is None:
        return None
    if _version_tuple(baseline_version) < _version_tuple(changed_in):
        return changed_in
    return None


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

class Result:
    """One benchmark's timings, reduced to comparable statistics."""

    def __init__(self, name: str, kind: str, samples: List[float],
                 bytes_moved: Optional[int] = None, note: str = ""):
        self.name = name
        self.kind = kind
        self.note = note
        self.bytes_moved = bytes_moved
        self._samples_ms = [s * 1000.0 for s in samples]
        self._recompute()

    def _recompute(self) -> None:
        ms = sorted(self._samples_ms)
        self.n = len(ms)
        self.median_ms = statistics.median(ms)
        self.min_ms = ms[0]
        self.p95_ms = ms[min(int(len(ms) * 0.95), len(ms) - 1)]
        # Spread of the distribution; high values mean the measurement is noisy
        # and small deltas should not be trusted.
        self.stdev_ms = statistics.stdev(ms) if len(ms) > 1 else 0.0

    def _pool(self, other: "Result") -> None:
        """Absorb another round's samples for the same benchmark."""
        self._samples_ms.extend(other._samples_ms)
        self._recompute()

    @property
    def gb_per_s(self) -> Optional[float]:
        if not self.bytes_moved or self.median_ms <= 0:
            return None
        return self.bytes_moved / 1e9 / (self.median_ms / 1000.0)

    def to_dict(self) -> dict:
        # 6 decimals: sub-microsecond benchmarks (per-call COM overhead is
        # ~0.4 us) round to zero at 4 and then compare as meaningless ratios.
        d = {
            "name": self.name,
            "kind": self.kind,
            "samples": self.n,
            "median_ms": round(self.median_ms, 6),
            "min_ms": round(self.min_ms, 6),
            "p95_ms": round(self.p95_ms, 6),
            "stdev_ms": round(self.stdev_ms, 6),
        }
        if self.gb_per_s is not None:
            d["gb_per_s"] = round(self.gb_per_s, 3)
        if self.note:
            d["note"] = self.note
        return d


# Benchmarks measured as duty-cycle sensitive, name -> (paced_ms, sustained_ms).
# Populated by check_duty_cycle and drained by annotate_duty_cycle after the
# suite finishes, so the warning appears in one place rather than in every
# benchmark.
_DUTY_SENSITIVE: Dict[str, tuple] = {}

# Target frame period in milliseconds: one frame at 60 Hz.
#
# Reps are paced to this *period*, not to a fixed idle gap. The difference
# matters, because a benchmark's duty cycle in production is decided by how
# long it takes relative to a frame:
#
#     RGB    1.8 ms of a 16.7 ms frame   ~11% duty cycle, mostly idle
#     GRAY  15.9 ms of a 16.7 ms frame   ~95% duty cycle, effectively sustained
#
# Pacing to a period reproduces both from one rule. A fixed gap cannot: a 16 ms
# gap gives GRAY a 50% duty cycle, which matches nothing real and measured
# 9.16 ms — a number the code never delivers in a capture loop.
#
# This is not cosmetic. Sustained heavy vector work holds the CPU in a lower
# power state, and GRAY has two modes on this machine. Measured, identical code:
#
#     back-to-back                      16.27 ms
#     16 ms fixed gap, short run         9.16 ms
#     16 ms fixed gap, 125 reps         14.41 ms
#     3 reps + 200 ms gap                9.91 ms
#
# The fast mode is a transient the CPU sustains for a second or two. Capturing
# GRAY continuously never sees it, so the sustained figure is the honest one and
# a harness that reports 9.16 ms is flattering the code.
FRAME_PERIOD_MS = 1000.0 / 60.0

# Reps this cheap are dominated by loop overhead rather than by real work, and
# pacing them would make the suite far slower for no gain in fidelity.
PACE_THRESHOLD_MS = 1.0

DUTY_CYCLE_THRESHOLD = 1.25


def _time_once(fn: Callable[[], None]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def time_it(fn: Callable[[], None], reps: int, name: str = "",
            warmup: int = 3, period_ms: Optional[float] = None) -> List[float]:
    """Warm up, then sample one rep per frame period.

    Each rep starts a frame: after timing it, sleep whatever is left of
    `period_ms`. Work that fits comfortably in a frame is therefore measured
    mostly-idle, and work that fills a frame is measured under sustained load —
    matching what each actually does in a capture loop.

    `period_ms` of None selects the 60 Hz frame period for reps expensive enough
    to matter. Pass 0 to sample back-to-back.
    """
    for _ in range(warmup):
        fn()

    if period_ms is None:
        # Decide from observed cost rather than a hardcoded list, so a benchmark
        # that gets faster or slower is paced correctly without anyone
        # remembering to update a table.
        period_ms = (FRAME_PERIOD_MS
                     if _time_once(fn) * 1000.0 >= PACE_THRESHOLD_MS else 0.0)

    if period_ms:
        # Let the CPU leave whatever state the warm-up put it in.
        time.sleep(period_ms * 4 / 1000.0)

    samples = []
    for _ in range(reps):
        elapsed = _time_once(fn)
        samples.append(elapsed)
        if period_ms:
            # Sleep only the remainder of the frame. A rep that overruns its
            # frame gets no idle time at all, which is exactly the situation of
            # a conversion too slow to keep up with the display.
            remaining = period_ms / 1000.0 - elapsed
            if remaining > 0:
                time.sleep(remaining)

    if name and period_ms:
        check_duty_cycle(fn, name, min(samples))
    return samples


def check_duty_cycle(fn: Callable[[], None], name: str, paced_min: float,
                     reps: int = 12) -> None:
    """Record how much this benchmark slows down under sustained load.

    A benchmark whose number depends on how hard it is driven cannot be compared
    against a baseline recorded at a different duty cycle. Rather than guess
    which benchmarks those are, measure it: run the same function back-to-back
    and compare against the paced minimum already collected.

    For work that already fills a frame this reports ~1.0x, correctly: such a
    benchmark is *already* running sustained, so there is no discrepancy between
    how the suite drives it and how production does.
    """
    for _ in range(3):
        fn()
    sustained = min(_time_once(fn) for _ in range(reps))
    # This check is itself a burn loop. Without a recovery gap it would leave
    # the CPU throttled for whichever benchmark runs next, spreading the very
    # artefact it exists to detect.
    time.sleep(FRAME_PERIOD_MS * 8 / 1000.0)
    if paced_min > 0 and sustained / paced_min >= DUTY_CYCLE_THRESHOLD:
        _DUTY_SENSITIVE[name] = (paced_min * 1000.0, sustained * 1000.0)


def annotate_duty_cycle(results: List[Result]) -> List[str]:
    """Flag duty-cycle-sensitive results, and return warnings for the operator."""
    warnings = []
    for r in results:
        measured = _DUTY_SENSITIVE.get(r.name)
        if measured is None:
            continue
        paced, sustained = measured
        note = (f"duty-cycle sensitive: {paced:.2f} ms paced vs {sustained:.2f} ms "
                f"back-to-back ({sustained / paced:.2f}x)")
        r.note = f"{r.note}; {note}" if r.note else note
        warnings.append(f"  {r.name}: {note}")
    return warnings


def merge_rounds(rounds: List[List[Result]]) -> List[Result]:
    """
    Combine repeated runs of the whole suite by pooling all their samples.

    Running the suite once and taking the minimum is not enough on a busy
    machine: if the entire run lands in a noisy window, even its minimum is
    inflated. Interleaving whole rounds means each benchmark only needs *one*
    quiet moment somewhere in the session for its minimum to be representative.
    """
    by_name: Dict[str, Result] = {}
    for rnd in rounds:
        for r in rnd:
            prev = by_name.get(r.name)
            if prev is None:
                by_name[r.name] = r
            else:
                prev._pool(r)
    return list(by_name.values())


# ---------------------------------------------------------------------------
# synthetic fixtures
# ---------------------------------------------------------------------------

class FakeMappedRect:
    """A DXGI_MAPPED_RECT-alike over a ctypes buffer, for screen-free runs."""

    def __init__(self, bgra: np.ndarray, pitch: Optional[int] = None):
        h, w, _ = bgra.shape
        self.Pitch = w * 4 if pitch is None else pitch
        self._backing = (ctypes.c_ubyte * (self.Pitch * h))()
        view = np.ctypeslib.as_array(self._backing).reshape(h, self.Pitch)
        view[:, : w * 4] = bgra.reshape(h, w * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def make_frame(h: int = FRAME_H, w: int = FRAME_W) -> np.ndarray:
    rng = np.random.default_rng(20260727)
    return rng.integers(0, 256, (h, w, 4), dtype=np.uint8)


# ---------------------------------------------------------------------------
# synthetic benchmarks
# ---------------------------------------------------------------------------

def bench_color_conversion(reps: int) -> List[Result]:
    """Per-mode BGRA -> output conversion. The dominant CPU cost per frame."""
    from rapidshot.processor.numpy_processor import NumpyProcessor

    src = make_frame()
    out = []
    for mode in MODES:
        proc = NumpyProcessor(mode)
        channels = proc.output_channels
        dst = np.empty((FRAME_H, FRAME_W, channels), np.uint8)
        samples = time_it(lambda p=proc: p.convert_into(src, dst), reps,
                          name=f"convert.{mode}")
        out.append(Result(f"convert.{mode}", "synthetic", samples,
                          bytes_moved=src.nbytes))
    return out


def bench_shot_path(reps: int) -> List[Result]:
    """Direct-to-buffer capture path, including the destination size check."""
    from rapidshot.processor.numpy_processor import NumpyProcessor

    src = make_frame()
    rect = FakeMappedRect(src)
    out = []
    for mode in MODES:
        proc = NumpyProcessor(mode)
        dst = np.empty((FRAME_H, FRAME_W, proc.output_channels), np.uint8)
        samples = time_it(
            lambda p=proc, d=dst: p.shot(d, rect, FRAME_W, FRAME_H), reps,
            name=f"shot.{mode}")
        out.append(Result(f"shot.{mode}", "synthetic", samples,
                          bytes_moved=src.nbytes))
    return out


def bench_process_pipeline(reps: int) -> List[Result]:
    """Full processor.process(): staging read + conversion + rotation."""
    from rapidshot.processor.numpy_processor import NumpyProcessor

    src = make_frame()
    rect = FakeMappedRect(src)
    out = []
    for mode in ("BGRA", "RGB"):
        proc = NumpyProcessor(mode)
        buf = np.empty((FRAME_H, FRAME_W, 4), np.uint8)
        samples = time_it(
            lambda p=proc: p.process(rect, FRAME_W, FRAME_H,
                                     (0, 0, FRAME_W, FRAME_H), 0, buf), reps,
            name=f"process.{mode}")
        out.append(Result(f"process.{mode}", "synthetic", samples,
                          bytes_moved=src.nbytes))
    return out


def bench_staging_read(reps: int) -> List[Result]:
    """Bulk copy out of a mapped-surface-shaped buffer (RAM upper bound)."""
    src = make_frame()
    rect = FakeMappedRect(src)
    h, pitch = FRAME_H, rect.Pitch
    buf = (ctypes.c_ubyte * (pitch * h)).from_address(
        ctypes.cast(rect.pBits, ctypes.c_void_p).value)
    view = np.ctypeslib.as_array(buf).reshape(h, pitch)
    dst = np.empty((h, FRAME_W * 4), np.uint8)
    samples = time_it(lambda: dst.__setitem__(slice(None),
                                              view[:, : FRAME_W * 4]), reps,
                      name=CONTROL_BENCHMARK)
    return [Result(CONTROL_BENCHMARK, "control", samples,
                   bytes_moved=src.nbytes,
                   note="CONTROL: a plain memory copy whose implementation never "
                        "changes. Its movement between runs measures machine "
                        "drift (background load, thermal state), which is used to "
                        "normalise the other comparisons.")]


# ---------------------------------------------------------------------------
# live benchmarks
# ---------------------------------------------------------------------------

def bench_preprocess_pipeline(reps: int) -> List[Result]:
    """
    The full capture-to-model-input pipeline, CPU versus GPU.

    Both arms do the same work: 1920x1080 BGRA -> 640x640 NCHW float32 RGB,
    normalised. The CPU arm is what Rapidshot did before the GPU shader existed;
    the GPU arm is one compute dispatch that never leaves the device.

    Uses a synthetic texture rather than live capture, so this is deterministic
    and runs without a desktop session — unlike the live benchmarks, these
    numbers are comparable across runs.
    """
    out: List[Result] = []
    src = make_frame()  # 1920x1080 BGRA
    OUT_W = OUT_H = 640

    # --- CPU arm ---------------------------------------------------------
    src_h, src_w = src.shape[:2]
    ys = (np.arange(OUT_H) * src_h // OUT_H).clip(0, src_h - 1)
    xs = (np.arange(OUT_W) * src_w // OUT_W).clip(0, src_w - 1)

    # This arm is what the GPU path is judged against, so it has to be the best
    # CPU implementation rather than the first one. It was neither: the original
    # widened the 640x640x4 gather to float32 *before* scaling (6.55 MB of
    # traffic where 1.6 MB suffices), divided in a second pass, stacked the
    # channels into a fresh array in a third, and allocated ~11 MB per call.
    # Writing each channel once into a destination that already exists is 1.78x
    # faster for a bit-identical result -- and a strawman here would have
    # overstated the GPU win by exactly that much.
    #
    # Preallocating is the fair comparison: the GPU arm reuses its buffers, and
    # any real consumer in a capture loop would reuse this one.
    sample_ys, sample_xs = ys, xs
    nchw = np.empty((1, 3, OUT_H, OUT_W), np.float32)

    def cpu_pipeline():
        # Two sequential takes, not `src[np.ix_(ys, xs)]`. Byte-identical and
        # 2.5x faster: two-dimensional advanced indexing measured 0.88 GB/s,
        # ~1% of the memory ceiling, and was 73% of this row. See `_gather` in
        # `rapidshot/preprocess.py`, which this mirrors.
        sampled = src.take(sample_ys, axis=0).take(sample_xs, axis=1)
        np.divide(sampled[..., 2], 255.0, out=nchw[0, 0])   # R
        np.divide(sampled[..., 1], 255.0, out=nchw[0, 1])   # G
        np.divide(sampled[..., 0], 255.0, out=nchw[0, 2])   # B
        return nchw

    out.append(Result("pipeline.cpu_to_nchw", "synthetic",
                      time_it(cpu_pipeline, max(reps // 2, 5),
                              name="pipeline.cpu_to_nchw"),
                      note="resize + normalise + transpose to NCHW on the CPU; "
                           "reimplemented 2026-08-05, 1.78x faster and "
                           "bit-identical -- not comparable with recordings "
                           "before that date"))

    # --- GPU arm ---------------------------------------------------------
    try:
        from rapidshot import native
        if not native.is_available():
            return out
        ext = native.require()
        tex = ext.TestTexture(src_w, src_h, np.ascontiguousarray(src).tobytes())
        pre = ext.GpuPreprocessor(tex.pointer, OUT_W, OUT_H)

        # Submission cost. GPU execution overlaps with subsequent CPU work and
        # the tensor stays on the device, so this is what the caller actually
        # pays in a pipeline that consumes the result on the GPU.
        out.append(Result(
            "pipeline.gpu_dispatch", "synthetic",
            time_it(lambda: pre.process(tex.pointer, 1.0, 0.0, False), reps,
                    name="pipeline.gpu_dispatch"),
            note="one compute dispatch; result stays on the GPU"))

        # With a full readback, i.e. paying the CPU round-trip anyway. Included
        # so the comparison cannot be accused of hiding synchronisation cost.
        def gpu_with_readback():
            pre.process(tex.pointer, 1.0, 0.0, False)
            pre.read_back()

        out.append(Result(
            "pipeline.gpu_plus_readback", "synthetic",
            time_it(gpu_with_readback, max(reps // 2, 5),
                    name="pipeline.gpu_plus_readback"),
            note="dispatch + full GPU->CPU readback (the round-trip we avoid)"))
    except Exception as e:  # pragma: no cover
        print(f"[GPU preprocessing benchmark skipped: {type(e).__name__}: {e}]")
    return out


def bench_com_overhead(reps: int) -> List[Result]:
    """Per-call comtypes cost — the ceiling on what a native core could save."""
    import rapidshot
    from rapidshot._libs.dxgi import DXGI_OUTPUT_DESC

    cam = rapidshot.create(output_idx=0)
    try:
        desc = DXGI_OUTPUT_DESC()
        out_iface = cam._output.output
        samples = time_it(lambda: out_iface.GetDesc(ctypes.byref(desc)), reps,
                          name="com.get_desc_call")
    finally:
        cam.release()
    return [Result("com.get_desc_call", "live", samples,
                   note="trivial COM method; approximates pure binding overhead")]


def bench_live_grab(duration_s: float) -> List[Result]:
    """
    End-to-end grab() cost against the real desktop.

    Only calls that actually returned a frame are timed. Mixing in the calls
    that return None makes the statistic meaningless: those exit early after the
    acquire timeout and are ~20x cheaper, so on a static desktop they dominate
    the sample and the minimum becomes 'how fast can grab() do nothing'.
    """
    import rapidshot

    cam = rapidshot.create(output_idx=0)
    productive: List[float] = []
    empty = 0
    try:
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            f = cam.grab()
            dt = time.perf_counter() - t0
            if f is None:
                empty += 1
                continue
            productive.append(dt)
            if hasattr(f, "release"):
                f.release()
    finally:
        cam.release()

    if not productive:
        return []

    note = (f"{len(productive)} frame-producing calls ({empty} returned no new "
            f"content) in {duration_s:.0f}s. DDA only reports changed content, "
            f"so this depends on screen activity — compare across runs with "
            f"caution.")
    return [Result("live.grab_with_frame", "live", productive, note=note)]


def bench_live_grab_frame(duration_s: float) -> List[Result]:
    """
    GPU-resident capture via grab_frame(), for comparison against grab().

    Same acquisition, but the frame stays on the GPU: no staging read and no
    colour conversion. The gap between this and live.grab_with_frame is the CPU
    round-trip that Stage 6 exists to eliminate, measured on real hardware.
    """
    import rapidshot

    cam = rapidshot.create(output_idx=0)
    productive: List[float] = []
    empty = 0
    try:
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            frame = cam.grab_frame()
            if frame is None:
                empty += 1
                continue
            # Touch the texture so the measurement includes making it usable,
            # then release inside the timed region — holding it blocks capture.
            _ = frame.d3d11_texture
            frame.release()
            productive.append(time.perf_counter() - t0)
    finally:
        cam.release()

    if not productive:
        return []

    note = (f"{len(productive)} frame-producing calls ({empty} empty) in "
            f"{duration_s:.0f}s; GPU-resident, no CPU round-trip")
    return [Result("live.grab_frame_gpu", "live", productive, note=note)]


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def calibration_metrics(results) -> dict:
    """One run's benchmarks flattened to ``name.metric`` -> value.

    Only the figures a comparison would quote. Sample counts and notes are not
    measurements and would clutter every table with rows whose spread is zero.
    """
    flat = {}
    for result in results:
        row = result.to_dict()
        for metric in ("median_ms", "min_ms", "p95_ms", "stdev_ms", "gb_per_s"):
            value = row.get(metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                flat[f"{result.name}.{metric}"] = value
    return flat


def run_calibration(args, info, first_results, run_once):
    """Repeat the whole suite with unchanged code and record what moves.

    The first pass has already happened by the time this is called, so it is
    reused rather than thrown away -- it was taken under the same conditions as
    the rest and discarding it would cost a run for nothing.
    """
    wanted = max(args.calibrate, 1)
    if wanted < statistics_report.MINIMUM_CALIBRATION_RUNS:
        print(f"{chr(10)}WARNING: {wanted} run(s) requested. Below "
              f"{statistics_report.MINIMUM_CALIBRATION_RUNS} this is not a noise "
              "floor, and the calibration will say so.")
    print(f"{chr(10)}CALIBRATION: {wanted} runs of unchanged code.")
    observations = [calibration_metrics(first_results)]
    for index in range(1, wanted):
        print(f"  run {index + 1}/{wanted}...", end=chr(13), flush=True)
        observations.append(calibration_metrics(
            merge_rounds([run_once() for _ in range(max(1, args.rounds))])))
    print(" " * 40, end=chr(13))

    configuration = (f"reps{args.reps}-rounds{args.rounds}"
                     f"-{info.get('cpu_topology', 'unknown')}"
                     f"-{'unpinned' if args.no_pin else 'pinned'}")
    calibration = statistics_report.calibrate_noise(
        observations, configuration=configuration)
    print(statistics_report.render_calibration_markdown(calibration))

    target = args.calibration_out or (Path(__file__).resolve().parent / "build"
                                      / "calibration.json")
    payload = {"schema_version": statistics_report.SCHEMA_VERSION,
               "machine": info, "calibration": calibration.as_dict(),
               "observations": observations}
    statistics_report.write_report(target, payload)
    statistics_report.write_report(
        Path(str(target).rsplit(".", 1)[0] + ".md"),
        statistics_report.render_calibration_markdown(calibration))
    print(f"wrote {target}")
    return 0


def machine_info() -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "frame": f"{FRAME_W}x{FRAME_H}",
    }
    # Provenance, not decoration: a pinned and an unpinned recording are not
    # comparable on a hybrid CPU, and after the fact there is no way to tell
    # them apart from the numbers alone.
    try:
        mask, topology = performance_core_mask()
        if mask is None:
            info["cpu_topology"] = topology
        else:
            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            # Declare these. Without argtypes ctypes coerces the process
            # pseudo-handle to a C int and raises OverflowError, which the
            # `except` below would then hide -- costing a recording whose
            # provenance silently reads `None` instead of failing loudly.
            k32.GetProcessAffinityMask.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
            ]
            process_mask = ctypes.c_size_t()
            system_mask = ctypes.c_size_t()
            k32.GetProcessAffinityMask(
                k32.GetCurrentProcess(),
                ctypes.byref(process_mask), ctypes.byref(system_mask))
            info["cpu_topology"] = topology
            info["pinned_to_performance_cores"] = process_mask.value == mask
            info["affinity_mask"] = hex(process_mask.value)
            # Only when set, so a machine without an exclusion records exactly
            # what it always has.
            excluded = machine_inventory.excluded_cpu_mask()
            if excluded:
                info["excluded_cpus"] = hex(excluded)
    except Exception as e:
        # Record the failure rather than dropping the keys. A recording with no
        # affinity provenance is indistinguishable from one taken before this
        # existed, and on a hybrid CPU that difference decides whether the
        # numbers mean anything (§ 2).
        info["cpu_topology"] = f"unknown ({type(e).__name__})"
    try:
        import rapidshot
        info["rapidshot"] = rapidshot.__version__
        info["gpu"] = rapidshot.get_factory().devices[0].description
    except Exception:
        pass
    # Provenance for the same reason as the affinity mask above, and with the
    # same failure mode. `baseline.json` is recorded with the optional native
    # extension and `baseline-nonative.json` without it; pointed at the wrong
    # one, every conversion row reports a 6-20x regression on every run forever
    # (.github/workflows/ci.yml says so, and routes around it by hand). Nothing
    # in a recording said which side of that line it came from, so after the
    # fact the two were distinguishable only by filename -- and `--compare
    # auto` cannot pick a file on a naming convention.
    try:
        from rapidshot import native
        info["native_extension"] = native.is_available()
    except Exception as e:
        # Record the reason rather than dropping the key, so "could not tell"
        # is never mistaken for "was not built".
        info["native_extension"] = f"unknown ({type(e).__name__})"
    # Added alongside the keys above, never instead of them: `print_comparison`
    # reads `processor`, `platform`, `gpu`, `python`, `numpy`, `cpu_topology`
    # and `pinned_to_performance_cores` out of committed baselines, and
    # renaming any of them would invalidate every baseline in the repository.
    info["environment"] = machine_inventory.discover_machine()
    info["machine_id"] = info["environment"]["machine_id"]
    return info


def _fmt_ms(value: float) -> str:
    """Render a duration without collapsing sub-microsecond values to zero."""
    if value >= 0.01:
        return f"{value:.3f}m"
    return f"{value * 1000:.2f}u"


def print_table(results: List[Result]) -> None:
    print(f"\n{'benchmark':<34}{'median':>10}{'min':>10}{'p95':>10}"
          f"{'stdev':>10}{'GB/s':>9}")
    print("-" * 83)
    for r in results:
        gb = f"{r.gb_per_s:.2f}" if r.gb_per_s else "-"
        print(f"{r.name:<34}{_fmt_ms(r.median_ms):>10}{_fmt_ms(r.min_ms):>10}"
              f"{_fmt_ms(r.p95_ms):>10}{_fmt_ms(r.stdev_ms):>10}{gb:>9}")
    print("-" * 83)
    print_cpu_vs_gpu(results)


def print_cpu_vs_gpu(results: List[Result]) -> None:
    """Headline comparison: the CPU round-trip that Stage 6 eliminates."""
    by_name = {r.name: r for r in results}
    cpu = by_name.get("live.grab_with_frame")
    gpu = by_name.get("live.grab_frame_gpu")
    if not (cpu and gpu and gpu.min_ms > 0):
        return
    ratio = cpu.min_ms / gpu.min_ms
    saved = cpu.min_ms - gpu.min_ms
    print(f"\nCPU round-trip vs GPU-resident capture (per frame, minimum):")
    print(f"  grab()        CPU staging read + convert   {cpu.min_ms:8.3f} ms"
          f"   {1000 / cpu.min_ms:6.0f} FPS ceiling")
    print(f"  grab_frame()  texture stays on the GPU     {gpu.min_ms:8.3f} ms"
          f"   {1000 / gpu.min_ms:6.0f} FPS ceiling")
    print(f"  -> {ratio:.1f}x faster, {saved:.2f} ms/frame saved. This gap is the"
          f" CPU round-trip\n     that Stage 6 removes; it is already real today"
          f" for GPU consumers.")

    cpu_pre = by_name.get("pipeline.cpu_to_nchw")
    gpu_pre = by_name.get("pipeline.gpu_dispatch")
    gpu_rt = by_name.get("pipeline.gpu_plus_readback")
    if cpu_pre and gpu_pre and gpu_pre.min_ms > 0:
        print(f"\nPreprocessing to model input (1920x1080 -> 640x640 NCHW float32):")
        print(f"  CPU  resize + normalise + transpose   {cpu_pre.min_ms:8.3f} ms"
              f"   (blocking)")
        print(f"  GPU  compute dispatch, submission     {gpu_pre.min_ms:8.3f} ms"
              f"   (async)")
        if gpu_rt:
            print(f"  GPU  dispatch + forced readback       {gpu_rt.min_ms:8.3f} ms"
                  f"   (blocking)")
        print("\n  Read these carefully. The dispatch figure is what the calling")
        print("  thread pays to submit work; the GPU executes asynchronously, so")
        print("  it is not a measure of total work done. The readback row is the")
        print("  honest worst case, and it is SLOWER than the CPU arm -- which is")
        print("  the point: this path wins only when the tensor is consumed on the")
        print("  GPU. Pulling it back to the CPU gives up the entire advantage.")


def resolve_auto_baseline(info: dict, directory: Path) -> Path:
    """
    Pick the committed baseline recorded on *this* machine.

    `RELEASING.md` step 4 named `benchmarks/baseline.json` outright, and that is
    one specific machine. Run anywhere else, `print_comparison` detects the
    mismatch and declines to gate -- correctly -- so the release step printed a
    table in which every verdict was indicative and nothing could fail. A gate
    that always passes is worse than no gate, because it is quoted as evidence.
    Once more than one machine records baselines, choosing the file has to
    follow the host rather than the instructions.

    Matched on the identity `print_comparison` already refuses to gate across,
    plus `native_extension` -- pairing a native run against a `-nonative`
    recording is the single most misleading comparison this repository can
    produce, and it is invisible in the output because every row moves together.

    Ambiguity fails loudly. Silently picking one of two equally valid baselines
    would make the verdict depend on directory order, which is precisely the
    kind of invisible coupling this suite exists to eliminate.
    """
    considered, candidates = [], []
    for path in sorted(directory.glob("baseline*.json")):
        try:
            machine = json.loads(path.read_text()).get("machine", {})
        except (OSError, ValueError) as e:
            # An unreadable baseline must not drop silently out of the running
            # and hand the comparison to some other machine's file.
            considered.append(f"  {path.name}: unreadable ({type(e).__name__})")
            continue
        mismatched = [k for k in ("processor", "platform", "gpu")
                      if machine.get(k) != info.get(k)]
        # The same machine on a different set of cores is a different
        # apparatus: withholding a core (RAPIDSHOT_BENCH_EXCLUDE_CPUS)
        # must send the gate to a baseline recorded the same way, or to none.
        if (machine.get("affinity_mask") is not None
                and info.get("affinity_mask") is not None
                and machine["affinity_mask"] != info["affinity_mask"]):
            mismatched.append("affinity_mask")
        # Absent on recordings that predate the field. Unknown is not a
        # mismatch on its own -- it would disqualify every older baseline --
        # but it cannot break a tie either, which is handled below.
        if (machine.get("native_extension") is not None
                and info.get("native_extension") is not None
                and machine["native_extension"] != info["native_extension"]):
            mismatched.append("native_extension")
        if mismatched:
            considered.append(f"  {path.name}: differs on {', '.join(mismatched)}")
            continue
        candidates.append((path, machine))

    if not candidates:
        raise SystemExit(
            "--compare auto found no baseline recorded on this machine.\n"
            + "\n".join(considered)
            + f"\n\nThis host is {info.get('processor')} / {info.get('gpu')}"
            f", native_extension={info.get('native_extension')}."
            "\nRecord one with --out benchmarks/baseline-<machine>.json, or"
            " pass an explicit path to compare across machines (which will"
            " report indicative verdicts only)."
        )

    if len(candidates) > 1:
        # Newest recording of the same machine wins: an older one is more
        # likely to predate a redefinition and suppress rows.
        candidates.sort(
            key=lambda c: (_version_tuple(c[1].get("rapidshot", "")),
                           c[1].get("timestamp", "")),
            reverse=True)
        best, runner_up = candidates[0], candidates[1]
        tied = (_version_tuple(best[1].get("rapidshot", ""))
                == _version_tuple(runner_up[1].get("rapidshot", ""))
                and best[1].get("timestamp", "") == runner_up[1].get("timestamp", ""))
        if tied:
            raise SystemExit(
                "--compare auto matched more than one baseline for this machine"
                " and cannot choose between them:\n"
                + "\n".join(f"  {p.name}" for p, _ in candidates)
                + "\n\nPass the one you mean explicitly."
            )

    path, machine = candidates[0]
    print(f"host: {info.get('processor')} / {info.get('gpu')}"
          f" (native_extension={info.get('native_extension')})")
    print(f"matched {path.name} (rapidshot {machine.get('rapidshot', '?')})")
    return path


def print_comparison(current: List[Result], baseline_path: Path,
                     threshold: float = 1.30) -> int:
    """
    Compare against a baseline using the MINIMUM sample, not the median.

    Background load can only ever make a benchmark slower, never faster, so the
    minimum is the sample least contaminated by interference — it approximates
    the run where the machine happened to be quietest. Medians on a loaded
    machine drift enough to bury a genuine 2x change in apparent noise, which is
    exactly what happened when this suite was first run at 55% background CPU.
    """
    baseline = json.loads(baseline_path.read_text())
    base = {b["name"]: b for b in baseline["results"]}

    print(f"\nComparison vs {baseline_path.name} "
          f"(recorded {baseline['machine'].get('timestamp', '?')[:19]})")
    print("Using minimum-sample times (robust to background load).")

    # A recording made on other hardware cannot gate a change, and the control
    # benchmark is not enough to rescue it. `control.memcopy` measures memory
    # bandwidth and nothing else, so dividing by its movement only normalises
    # benchmarks that are *also* bandwidth-bound. `pipeline.cpu_to_nchw` is
    # float32 resize/normalise/transpose -- compute-bound, and sensitive to
    # vector width and NumPy version in ways memcpy is not. Normalising it by a
    # memcpy ratio produced a 1.34x "regression" against an untouched code path
    # on a CI runner, while simultaneously reporting every conversion row 1.4x
    # *faster* on a machine that was uniformly slower. Both directions were
    # artefacts of one control standing in for workloads it does not resemble;
    # ROADMAP.md section 2 records the same limitation for GRAY.
    base_machine = baseline.get("machine", {})
    now_machine = machine_info()
    # Used to suppress rows whose definition changed after this baseline was
    # recorded. Absent on recordings made before the field existed, which
    # _version_tuple deliberately sorts lowest.
    base_version = base_machine.get("rapidshot", "")

    def _differing(keys):
        # `is not None`, not truthiness: `pinned_to_performance_cores` is a
        # bool, and a falsy-but-present value is a difference worth reporting,
        # not a missing field. Testing truthiness here made an unpinned run
        # compare cleanly against a pinned baseline -- exactly the case the
        # field was added to catch.
        return [k for k in keys
                if base_machine.get(k) is not None
                and now_machine.get(k) is not None
                and base_machine[k] != now_machine[k]]

    hardware = _differing(("processor", "platform", "gpu"))
    environment = _differing(("python", "numpy"))

    # A pinned and an unpinned recording are not comparable even on identical
    # hardware: unpinned, this suite reported false regressions up to 2.57x
    # against its own output (ROADMAP.md section 2). The provenance is recorded
    # precisely so that difference is visible, so it has to be *read* here --
    # otherwise a pinned baseline silently gates verdicts against an unpinned
    # run and the metadata documents a hazard that nothing acts on.
    scheduling = _differing(("cpu_topology", "pinned_to_performance_cores",
                             "affinity_mask"))

    # A baseline recorded before this provenance existed carries neither field,
    # and `_differing` ignores a key unless both sides have it -- so on a hybrid
    # CPU the pinned-now-versus-unpinned-then comparison this is meant to reject
    # would sail straight through and gate verdicts. Absence is not agreement:
    # if we are hybrid and the baseline cannot say how it was scheduled, it is
    # not comparable.
    # Only when the baseline is identifiably *this* machine. A recording that
    # names no hardware at all says nothing about scheduling either, and
    # refusing to gate against it would exempt every synthetic baseline rather
    # than the real pre-provenance ones this is aimed at.
    same_hardware = any(
        base_machine.get(k) is not None and now_machine.get(k) is not None
        and base_machine[k] == now_machine[k]
        for k in ("processor", "platform", "gpu"))
    # "unknown" counts alongside "hybrid": if the topology could not be read,
    # neither can whether pinning mattered. The asymmetry is deliberate -- a
    # needless "indicative only" label costs nothing, a false regression costs
    # somebody an afternoon.
    #
    # `startswith`, not equality: `machine_info()` records the *reason* on a
    # failure, as `unknown (OSError)`. An exact match against "unknown" never
    # fired for the very case the branch was added to catch, which is what
    # happens when the producer and the consumer of a string are written in
    # separate passes and never compared.
    topology = now_machine.get("cpu_topology") or ""
    unknown_scheduling = (same_hardware
                          and (topology == "hybrid"
                               or topology.startswith("unknown"))
                          and base_machine.get("pinned_to_performance_cores")
                          is None)
    cross_machine = bool(hardware) or bool(scheduling) or unknown_scheduling

    if cross_machine or environment:
        print()
        for key in hardware + scheduling + environment:
            print(f"  {key}: baseline {base_machine[key]!r} vs now "
                  f"{now_machine[key]!r}")
    if cross_machine:
        if hardware:
            reason = "CROSS-MACHINE"
        elif unknown_scheduling:
            reason = "UNKNOWN BASELINE SCHEDULING"
            print("")
            print("  the baseline predates CPU-scheduling provenance, and this")
            print("  is a hybrid CPU — how it was pinned cannot be recovered")
        else:
            reason = "DIFFERENT CPU SCHEDULING"
        print("")
        print(f"{reason} COMPARISON: verdicts below are indicative only")
        print("and nothing here gates. Re-record a baseline on this machine,")
        print("pinned the same way, to compare code against code rather than")
        print("conditions against conditions.")

    # Calibrate against the control benchmark: its code is identical in both
    # runs, so any movement is the machine, not us.
    drift = 1.0
    ctrl_now = next((r for r in current if r.name == CONTROL_BENCHMARK), None)
    ctrl_base = base.get(CONTROL_BENCHMARK)
    if ctrl_now and ctrl_base:
        cb = ctrl_base.get("min_ms", ctrl_base.get("median_ms", 0))
        if cb > 0 and ctrl_now.min_ms > 0:
            drift = ctrl_now.min_ms / cb
            state = ("machine is SLOWER now" if drift > 1.05
                     else "machine is FASTER now" if drift < 0.95
                     else "machine state comparable")
            print(f"Control benchmark moved {drift:.2f}x — {state}.")
            if abs(drift - 1.0) > 0.05:
                print("'adjusted' divides out that drift; treat it as an estimate.")

    show_adj = abs(drift - 1.0) > 0.05
    hdr_adj = f"{'adjusted':>11}" if show_adj else ""
    print(f"\n{'benchmark':<30}{'before':>9}{'after':>9}{'raw':>11}{hdr_adj}"
          f"  verdict")
    print("-" * (61 + len(hdr_adj) + 20))

    regressions = 0
    redefined_rows: List[Tuple[str, str]] = []
    for r in current:
        b = base.get(r.name)
        if b is None:
            print(f"{r.name:<30}{'-':>9}{r.min_ms:>8.3f}m{'new':>11}")
            continue
        before, after = b.get("min_ms", b["median_ms"]), r.min_ms
        if before <= 0 or after <= 0:
            continue

        ratio = before / after
        adj_ratio = ratio * drift  # what the ratio would be at baseline speed
        judged = adj_ratio if show_adj else ratio

        # Checked before anything else: if the row measures something different
        # now than it did when the baseline was recorded, no verdict about it
        # means anything, in either direction.
        redefined = _redefined_since(r.name, base_version)
        if redefined:
            redefined_rows.append((r.name, redefined))

        # Why this row cannot be believed, if it cannot. Computed once and
        # applied to both directions.
        #
        # These qualifiers used to live only in the SLOWER branch, while the
        # FASTER branch carried a comment claiming they applied "in both
        # directions on purpose". They did not. A live row reporting FASTER
        # 8.70x therefore printed bare, exactly the spurious improvement the
        # comment said to guard against -- found 2026-08-22 by running the
        # verification pass section 2 requires after re-recording a baseline.
        if redefined:
            caveat = f" (redefined in {redefined}: NOT COMPARABLE)"
        elif r.kind == "live":
            # Live benchmarks depend on what is happening on screen, which is
            # not a controlled input: repeated runs of identical code have been
            # observed to swing 2.5x. Reported for information, never gating,
            # or the suite cries wolf and gets ignored.
            caveat = " (live: informational)"
        elif r.name in _DUTY_SENSITIVE or "duty-cycle" in b.get("note", ""):
            # This benchmark measures differently depending on how hard it is
            # driven, so its minimum estimates "did a sample land in the fast
            # mode" rather than the code's cost. GRAY produced four readings
            # from 8.75 to 15.70 ms on identical code this way.
            #
            # The baseline's own flag counts, not just this run's: the detector
            # only fires when a paced sample reached the fast mode, so a run
            # that stays slow throughout looks consistent and would then be
            # gated against a baseline that got lucky. Once either side has
            # seen two modes, the comparison is untrustworthy both ways.
            caveat = " (duty-cycle sensitive: informational)"
        elif max(before, after) < LOW_RESOLUTION_MS:
            # Below roughly half a millisecond the OS scheduler's granularity
            # dominates, and drift normalisation amplifies it further.
            caveat = " (sub-ms: informational)"
        elif cross_machine:
            caveat = " (cross-machine: indicative)"
        else:
            caveat = ""

        if r.name == CONTROL_BENCHMARK:
            verdict = "(calibration)"
        elif judged >= threshold:
            # An unexplained improvement is as much a measurement failure as an
            # unexplained regression, and nobody investigates good news.
            verdict = f"FASTER {judged:.2f}x{caveat}"
        elif judged <= 1.0 / threshold:
            verdict = f"SLOWER {1 / judged:.2f}x{caveat}"
            if not caveat:
                regressions += 1
        else:
            verdict = "~ same"

        adj_col = f"{adj_ratio:>10.2f}x" if show_adj else ""
        print(f"{r.name:<30}{before:>8.3f}m{after:>8.3f}m{ratio:>10.2f}x"
              f"{adj_col}  {verdict}")
    print("-" * (61 + len(hdr_adj) + 20))

    if redefined_rows:
        # Say this loudly and unconditionally. The row this was built for
        # reported FASTER 6.08x against a 2.1.0 baseline and reads as a
        # hardware win; it is a code change that had already landed. A
        # reader who sees only the table has no way to know that.
        print(f"\n{len(redefined_rows)} row(s) NOT COMPARABLE against this "
              f"baseline (recorded on rapidshot {base_version or 'unknown'}):")
        for name, changed_in in redefined_rows:
            print(f"  {name} -- what it measures changed in {changed_in}")
        print("These are excluded from the verdict in both directions. "
              "Re-record on this machine to compare them.")

    if cross_machine:
        print("\nNo verdict gated: the baseline came from different hardware.")
        return 0
    if regressions:
        # Quote the real threshold. This said "the 10% threshold" while the
        # default was 1.30, so anyone reading the failure was told a change had
        # to be 10% to count when it actually had to be 30%.
        print(f"\n{regressions} regression(s) beyond {threshold:.2f}x "
              f"({(threshold - 1) * 100:.0f}%).")
    return regressions


def performance_core_mask() -> Tuple[Optional[int], str]:
    """``(mask, topology)`` for the fastest cores. See `machine_inventory`.

    The detection lived here first and now lives in `machine_inventory`, so the
    capture and memory runners can apply the same policy this suite has had
    since ROADMAP section 2 -- they never could while it was private to this
    file, which is why the FP16 ingestion table carries the caveat that it was
    not pinned. The signature stays as it was because `machine_info` below
    feeds `baseline.json` comparison, and changing those keys would invalidate
    every committed baseline.
    """
    return machine_inventory.performance_core_mask()


def pin_to_performance_cores() -> None:
    """Restrict this process to the fastest cores, and say what happened.

    Pinning narrows the distribution; it does not make timings deterministic.
    Clocks, thermals and other processes still move them.
    """
    policy = machine_inventory.apply_cpu_policy("performance")
    if policy.verified:
        print(f"  affinity   pinned to "
              f"{bin(policy.requested_mask).count('1')} performance cores "
              f"(mask {hex(policy.requested_mask)}) -- hybrid CPU detected")
    elif policy.topology == "uniform":
        pass
    else:
        reason = "; ".join(policy.reasons) or "unknown reason"
        print(f"  affinity   not pinned ({reason}); results will be noisier "
              "(see ROADMAP section 2)")


def warn_if_machine_is_busy() -> None:
    """Loud warning if the machine is too loaded for trustworthy numbers."""
    try:
        import subprocess
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor | "
             "Measure-Object -Property LoadPercentage -Average).Average"],
            capture_output=True, text=True, timeout=15)
        load = int(out.stdout.strip())
    except Exception:
        return
    if load >= 25:
        print(f"\n  !! background CPU load is {load}% — absolute timings will be")
        print("     inflated. Comparisons use minimum samples and stay usable,")
        print("     but close the load for publication-quality numbers.")


#: What a microbenchmark row must contain to be a measurement.
MICROBENCH_REQUIRED = ("samples", "median_ms")


def write_json_atomic(path, payload):
    """tmp + fsync + replace, so an interrupted write cannot truncate a baseline."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def record_history(args, info, results):
    """Commit each benchmark as a case, with the samples its figures came from.

    **Committed after the run, not between benchmarks**, and that is a weaker
    guarantee than the other runners get. These are sub-second synthetic
    microbenchmarks: the whole suite finishes in less time than one section 7
    case takes to warm up, so the crash window this store exists to close is
    barely open here. Restructuring the round-pooling in `merge_rounds` to
    commit incrementally would change how the figures are computed, and this
    suite gates releases -- not a thing to disturb for a guarantee it needs
    least.

    The raw per-rep samples go with each case. A median without them cannot be
    recomputed, re-examined, or used to say how noisy the machine was.
    """
    store = result_store.open_for_runner(
        "perf_suite", root=args.history_root, resume=args.resume,
        disabled=args.no_history,
        metadata={"machine": info, "reps": args.reps, "rounds": args.rounds,
                  "self_test": bool(args.self_test), "argv": sys.argv,
                  "environment": info.get("environment"),
                  "machine_id": info.get("machine_id")})
    if store is None:
        return None
    print(f"{chr(10)}History: {store.run_dir}")
    # The pinning choice and the topology it was applied to change what these
    # numbers mean, so they are part of the identity rather than a footnote.
    configuration = (f"reps{args.reps}-rounds{args.rounds}"
                     f"-{info.get('cpu_topology', 'unknown')}"
                     f"-{'unpinned' if args.no_pin else 'pinned'}"
                     + ("-selftest" if args.self_test else ""))
    for result in results:
        identity = CaseIdentity(benchmark="perf_suite", path=result.name,
                                configuration=configuration,
                                workload=getattr(result, "kind", "synthetic") or "synthetic",
                                repeat=1)
        action, status = result_store.resume_decision(store, identity,
                                                      retry_failed=args.retry_failed)
        if action == "skip":
            continue
        with result_store.case_context(store, identity, retry=action == "retry",
                                       required=MICROBENCH_REQUIRED) as case:
            case.result = result.to_dict()
            case.samples = [{"index": index, "ms": value}
                            for index, value in enumerate(result._samples_ms)]
    store.rebuild_summary()
    return store


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write results as JSON")
    result_store.add_history_arguments(ap)
    ap.add_argument("--compare", type=Path,
                    help="compare against a baseline JSON, or 'auto' to use "
                         "the committed baseline recorded on this machine. "
                         "A named file from another machine still compares, "
                         "but reports indicative verdicts and gates nothing.")
    ap.add_argument("--synthetic-only", action="store_true",
                    help="skip benchmarks that need a desktop session")
    ap.add_argument("--reps", type=int, default=30, help="reps per benchmark")
    ap.add_argument("--rounds", type=int, default=3,
                    help="run the whole suite N times and pool the samples. "
                         "More rounds beat more reps on a busy machine: each "
                         "benchmark only needs one quiet moment in the session.")
    ap.add_argument("--threshold", type=float, default=1.30,
                    help="ratio a change must exceed to be called real. The "
                         "default is set above this machine's measured "
                         "run-to-run noise floor; verify with --self-test.")
    ap.add_argument("--calibrate", type=int, metavar="RUNS",
                    help="repeat the whole suite RUNS times with unchanged code and "
                         "record what each benchmark's own numbers do when nothing "
                         "changes. This is the noise floor every later comparison is "
                         "judged against; at least "
                         f"{statistics_report.MINIMUM_CALIBRATION_RUNS} runs are "
                         "needed for it to mean anything.")
    ap.add_argument("--calibration-out", type=Path,
                    help="where to write the calibration (JSON, plus .md beside it)")
    ap.add_argument("--self-test", action="store_true",
                    help="measure the noise floor by comparing the suite to "
                         "itself; any 'change' reported is pure measurement error")
    ap.add_argument("--live-seconds", type=float, default=3.0)
    ap.add_argument("--no-pin", action="store_true",
                    help="do not restrict the process to performance cores. "
                         "On a hybrid P-core/E-core CPU this reintroduces "
                         "2-3x false verdicts; see ROADMAP § 2.")
    args = ap.parse_args()

    print(f"Rapidshot performance suite")
    # Pin *before* reading machine info: the recording's affinity provenance
    # has to describe the run, and reading it first records the state the
    # process started in rather than the one it benchmarked in.
    if not args.no_pin:
        pin_to_performance_cores()
    info = machine_info()
    for k in ("timestamp", "platform", "gpu", "python", "numpy", "frame"):
        if k in info:
            print(f"  {k:<10} {info[k]}")
    warn_if_machine_is_busy()

    def run_once() -> List[Result]:
        rs: List[Result] = []
        rs += bench_color_conversion(args.reps)
        rs += bench_shot_path(args.reps)
        rs += bench_process_pipeline(args.reps)
        rs += bench_preprocess_pipeline(args.reps)
        rs += bench_staging_read(args.reps)
        return rs

    rounds = []
    for i in range(max(1, args.rounds)):
        print(f"  round {i + 1}/{args.rounds}...", end="\r", flush=True)
        rounds.append(run_once())
    results = merge_rounds(rounds)
    print(" " * 30, end="\r")

    if not args.synthetic_only:
        try:
            results += bench_com_overhead(max(args.reps * 200, 5000))
            results += bench_live_grab(args.live_seconds)
            results += bench_live_grab_frame(args.live_seconds)
        except Exception as e:
            print(f"\n[live benchmarks skipped: {type(e).__name__}: {e}]")

    print_table(results)
    record_history(args, info, results)

    sensitive = annotate_duty_cycle(results)
    if sensitive:
        print("\nNOTE: these benchmarks measure differently depending on how hard")
        print("they are driven. The suite paces them to a per-frame duty cycle,")
        print("which is how capture actually runs; a back-to-back burn loop")
        print("reports the throttled number instead. Compare only against a")
        print("baseline recorded with the same pacing:")
        for line in sensitive:
            print(line)

    if args.calibrate:
        return run_calibration(args, info, results, run_once)

    if args.self_test:
        print(f"{chr(10)}SELF-TEST: re-running the suite and comparing it to itself.")
        print("Everything below should read '~ same'. Anything that does not")
        print("is measurement error, and sets the floor for what this machine")
        print("can resolve.")
        second = merge_rounds([run_once() for _ in range(max(1, args.rounds))])
        tmp = Path(str(args.out or "selftest") + ".selftest.json")
        tmp.write_text(json.dumps(
            {"machine": info, "results": [r.to_dict() for r in results]}, indent=2))
        noise = print_comparison(second, tmp, args.threshold)
        tmp.unlink(missing_ok=True)
        print(f"{chr(10)}noise floor: {noise} benchmark(s) exceeded "
              f"{args.threshold:.2f}x with no code change.")
        # Two passes is a sanity check, not a calibration. It can say "this
        # machine is behaving"; it cannot say how small a difference this
        # machine can resolve, because two observations have no spread worth
        # the name. --calibrate is the same idea carried far enough to answer
        # that, and is what a later comparison is judged against.
        print(f"{chr(10)}This is a two-pass sanity check. For a per-metric noise "
              f"floor a comparison can be judged against, run --calibrate "
              f"{statistics_report.MINIMUM_CALIBRATION_RUNS}.")
        return 0

    if args.out:
        args.out.write_text(json.dumps(
            {"machine": info, "results": [r.to_dict() for r in results]}, indent=2))
        print(f"\nwrote {args.out}")

    if args.compare:
        # Resolved here rather than in the parser: it needs `info`, which
        # cannot be read until after the process has pinned itself.
        compare_path = args.compare
        if str(compare_path) == "auto":
            compare_path = resolve_auto_baseline(info, Path(__file__).resolve().parent)
        return 1 if print_comparison(results, compare_path, args.threshold) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
