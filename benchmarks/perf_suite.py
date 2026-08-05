"""Rapidshot performance suite — reproducible numbers for every change.

Every roadmap ordering decision is a performance claim, so each change gets
measured against a stored baseline rather than asserted.

Usage
-----
Record a baseline before changing anything::

    python benchmarks/perf_suite.py --out baseline.json

Re-run after a change and compare::

    python benchmarks/perf_suite.py --out after.json --compare baseline.json

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
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

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

    def cpu_pipeline():
        sampled = src[np.ix_(ys, xs)].astype(np.float32)
        sampled /= 255.0
        b, g, r = sampled[..., 0], sampled[..., 1], sampled[..., 2]
        return np.ascontiguousarray(np.stack((r, g, b), axis=0)[None])

    out.append(Result("pipeline.cpu_to_nchw", "synthetic",
                      time_it(cpu_pipeline, max(reps // 2, 5),
                              name="pipeline.cpu_to_nchw"),
                      note="resize + normalise + transpose to NCHW on the CPU"))

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

def machine_info() -> dict:
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "frame": f"{FRAME_W}x{FRAME_H}",
    }
    try:
        import rapidshot
        info["rapidshot"] = rapidshot.__version__
        info["gpu"] = rapidshot.get_factory().devices[0].description
    except Exception:
        pass
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

    def _differing(keys):
        return [k for k in keys
                if base_machine.get(k) and now_machine.get(k)
                and base_machine[k] != now_machine[k]]

    hardware = _differing(("processor", "platform", "gpu"))
    environment = _differing(("python", "numpy"))
    cross_machine = bool(hardware)

    if cross_machine or environment:
        print()
        for key in hardware + environment:
            print(f"  {key}: baseline {base_machine[key]!r} vs now "
                  f"{now_machine[key]!r}")
    if cross_machine:
        print("\nCROSS-MACHINE COMPARISON: verdicts below are indicative only and")
        print("nothing here gates. Re-record a baseline on this machine to compare")
        print("code against code rather than hardware against hardware.")

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

        if r.name == CONTROL_BENCHMARK:
            verdict = "(calibration)"
        elif judged >= threshold:
            verdict = f"FASTER {judged:.2f}x"
            # Flagged in both directions on purpose: a spurious improvement is
            # as misleading as a spurious regression, and harder to notice
            # because nobody investigates good news.
            if cross_machine:
                verdict += " (cross-machine: indicative)"
        elif judged <= 1.0 / threshold:
            verdict = f"SLOWER {1 / judged:.2f}x"
            # Live benchmarks depend on what is happening on screen, which is
            # not a controlled input: repeated runs of identical code have been
            # observed to swing 2.5x. They are reported for information but must
            # not gate a change, or the suite cries wolf and gets ignored.
            if r.kind == "live":
                verdict += " (live: informational)"
            elif r.name in _DUTY_SENSITIVE or "duty-cycle" in b.get("note", ""):
                # This benchmark measures differently depending on how hard it
                # is driven, so its minimum estimates "did a sample land in the
                # fast mode" rather than the code's cost. GRAY produced four
                # readings from 8.75 to 15.70 ms on identical code this way.
                #
                # The baseline's own flag counts, not just this run's. The
                # detector compares a paced sample against a sustained one, so
                # it only fires when the paced sample reached the fast mode —
                # meaning a run that stays slow throughout looks consistent and
                # goes unflagged, and would then be gated against a baseline
                # that got lucky. Once either side has seen two modes, the
                # comparison is untrustworthy in both directions.
                verdict += " (duty-cycle sensitive: informational)"
            elif max(before, after) < LOW_RESOLUTION_MS:
                # Below roughly half a millisecond the OS scheduler's
                # granularity dominates, and drift normalisation amplifies it
                # further. These are reported but do not gate.
                verdict += " (sub-ms: informational)"
            elif cross_machine:
                verdict += " (cross-machine: indicative)"
            else:
                regressions += 1
        else:
            verdict = "~ same"

        adj_col = f"{adj_ratio:>10.2f}x" if show_adj else ""
        print(f"{r.name:<30}{before:>8.3f}m{after:>8.3f}m{ratio:>10.2f}x"
              f"{adj_col}  {verdict}")
    print("-" * (61 + len(hdr_adj) + 20))

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="write results as JSON")
    ap.add_argument("--compare", type=Path, help="compare against a baseline JSON")
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
    ap.add_argument("--self-test", action="store_true",
                    help="measure the noise floor by comparing the suite to "
                         "itself; any 'change' reported is pure measurement error")
    ap.add_argument("--live-seconds", type=float, default=3.0)
    args = ap.parse_args()

    info = machine_info()
    print(f"Rapidshot performance suite")
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

    sensitive = annotate_duty_cycle(results)
    if sensitive:
        print("\nNOTE: these benchmarks measure differently depending on how hard")
        print("they are driven. The suite paces them to a per-frame duty cycle,")
        print("which is how capture actually runs; a back-to-back burn loop")
        print("reports the throttled number instead. Compare only against a")
        print("baseline recorded with the same pacing:")
        for line in sensitive:
            print(line)

    if args.self_test:
        print("\nSELF-TEST: re-running the suite and comparing it to itself.")
        print("Everything below should read '~ same'. Anything that does not")
        print("is measurement error, and sets the floor for what this machine")
        print("can resolve.")
        second = merge_rounds([run_once() for _ in range(max(1, args.rounds))])
        tmp = Path(str(args.out or "selftest") + ".selftest.json")
        tmp.write_text(json.dumps(
            {"machine": info, "results": [r.to_dict() for r in results]}, indent=2))
        noise = print_comparison(second, tmp, args.threshold)
        tmp.unlink(missing_ok=True)
        print(f"\nnoise floor: {noise} benchmark(s) exceeded {args.threshold:.2f}x "
              f"with no code change.")
        return 0

    if args.out:
        args.out.write_text(json.dumps(
            {"machine": info, "results": [r.to_dict() for r in results]}, indent=2))
        print(f"\nwrote {args.out}")

    if args.compare:
        return 1 if print_comparison(results, args.compare, args.threshold) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
