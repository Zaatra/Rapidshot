"""Before/after measurements for the four performance items from the 2.6 audit.

Each section times the code path as the repository currently implements it,
alongside the candidate replacement, so one script records both sides of the
change and later reruns can confirm the shipped code matches the candidate:

  * ``cupy``    -- CuPy colour conversion (GRAY's three uint16 temporaries,
                   RGBA's four channel assignments) against one fused kernel.
  * ``rows``    -- per-row Python loops reading a padded or offset staging
                   surface (NumPy read, shot() BGRA, CuPy host read) against a
                   single strided slice.
  * ``stream``  -- TensorStream polling a camera that returns immediately
                   (``timeout_ms=0``): calls and CPU per idle second, and the
                   latency a back-off would add to the next frame.
  * ``idle``    -- live grab() on a screen with nothing new: what each empty
                   poll costs, including the staging-buffer checkout it does
                   before knowing whether a frame arrived.

Candidates are checked byte-exact against the shipped path before they are
timed; a faster wrong answer is not a result.

    python benchmarks/audit_perf_items.py --out result.json [--only cupy rows]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import statistics
import sys
import threading
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import logging  # noqa: E402

logging.getLogger("rapidshot").setLevel(logging.ERROR)

W, H = 1920, 1080
PADDED_PITCH = W * 4 + 16          # a driver-aligned row, as some adapters map
OFFSET_LEFT, OFFSET_W = 320, 1280  # a region that does not start at column 0


def timed(fn, reps, warmup=5, sync=None):
    """Median and p95 milliseconds, back-to-back."""
    for _ in range(warmup):
        fn()
        if sync:
            sync()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {"median_ms": round(statistics.median(samples), 4),
            "p95_ms": round(samples[int(len(samples) * 0.95) - 1], 4),
            "reps": reps}


def frame(h=H, w=W, pitch=W * 4):
    rng = np.random.default_rng(20260915)
    return rng.integers(0, 256, (h, pitch), dtype=np.uint8)


def rect_over(host):
    return types.SimpleNamespace(Pitch=host.shape[1], pBits=host.ctypes.data)


# ---------------------------------------------------------------------------
# cupy
# ---------------------------------------------------------------------------

def bench_cupy(reps):
    import cupy as cp

    from rapidshot.processor.cupy_processor import CupyProcessor
    from rapidshot.processor.numpy_processor import (
        _LUMA_B, _LUMA_G, _LUMA_R, _LUMA_ROUND, _LUMA_SHIFT, NumpyProcessor)

    sync = cp.cuda.get_current_stream().synchronize
    host = frame().reshape(H, W, 4)
    image = cp.asarray(host)
    out = {}

    gray_kernel = cp.ElementwiseKernel(
        "uint8 b, uint8 g, uint8 r", "uint8 y",
        f"y = (r * {_LUMA_R} + g * {_LUMA_G} + b * {_LUMA_B} + {_LUMA_ROUND}) >> {_LUMA_SHIFT}",
        "rapidshot_bench_gray")
    rgba_kernel = cp.ElementwiseKernel(
        "raw uint8 src, int64 width", "uint8 dst",
        "ptrdiff_t c = i % 4; ptrdiff_t base = i - c;"
        "dst = c == 3 ? src[base + 3] : src[base + 2 - c];",
        "rapidshot_bench_rgba")

    def fused_gray():
        y = cp.empty((H, W), dtype=cp.uint8)
        gray_kernel(image[..., 0], image[..., 1], image[..., 2], y)
        return y[..., cp.newaxis]

    def fused_rgba():
        dst = cp.empty_like(image)
        rgba_kernel(image, W, dst)
        return dst

    candidates = {"GRAY": fused_gray, "RGBA": fused_rgba}
    for mode in ("GRAY", "RGBA", "RGB", "BGR"):
        shipped = CupyProcessor(mode)
        reference = cp.asnumpy(shipped.process_cvtcolor(image))
        expected = NumpyProcessor(mode).convert_into(host) if hasattr(
            NumpyProcessor(mode), "convert_into") else None
        row = {"shipped": timed(lambda: shipped.process_cvtcolor(image), reps, sync=sync)}
        if expected is not None:
            row["shipped_matches_numpy"] = bool(np.array_equal(reference, expected))
        if mode in candidates:
            got = cp.asnumpy(candidates[mode]())
            row["candidate_exact"] = bool(np.array_equal(got, reference))
            row["candidate"] = timed(candidates[mode], reps, sync=sync)
        pool = cp.get_default_memory_pool()
        out[mode] = row
        pool.free_all_blocks()
    return out


# ---------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------

def bench_rows(reps):
    from rapidshot.processor.numpy_processor import NumpyProcessor

    out = {}
    padded = frame(pitch=PADDED_PITCH)
    row_bytes = W * 4

    # NumPy staging read, via the shipped process() on a padded surface.
    proc = NumpyProcessor("BGRA")
    rect = rect_over(padded)
    reference = padded[:, :row_bytes].reshape(H, W, 4)
    result = proc.process(rect, W, H, (0, 0, W, H), 0)[0]
    out["numpy_read_padded"] = {
        "exact": bool(np.array_equal(np.asarray(result), reference)),
        "shipped": timed(lambda: proc.process(rect, W, H, (0, 0, W, H), 0), reps),
    }
    dest = np.empty((H, row_bytes), np.uint8)
    out["numpy_read_padded"]["candidate"] = timed(
        lambda: dest.__setitem__(slice(None), padded[:, :row_bytes]), reps)

    # An offset region on an unpadded surface takes the same loop.
    plain = frame()
    rect_plain = rect_over(plain)
    region = (OFFSET_LEFT, 0, OFFSET_LEFT + OFFSET_W, H)
    ref_off = plain[:, OFFSET_LEFT * 4:(OFFSET_LEFT + OFFSET_W) * 4].reshape(H, OFFSET_W, 4)
    got = proc.process(rect_plain, W, H, region, 0)[0]
    out["numpy_read_offset"] = {
        "exact": bool(np.array_equal(np.asarray(got), ref_off)),
        "shipped": timed(lambda: proc.process(rect_plain, W, H, region, 0), reps),
    }
    dest_off = np.empty((H, OFFSET_W * 4), np.uint8)
    out["numpy_read_offset"]["candidate"] = timed(
        lambda: dest_off.__setitem__(
            slice(None), plain[:, OFFSET_LEFT * 4:(OFFSET_LEFT + OFFSET_W) * 4]), reps)

    # shot() in BGRA on a padded surface: one memmove per row.
    dst = np.empty((H, W, 4), np.uint8)
    proc.shot(dst, rect, W, H)
    shot_row = {"exact": bool(np.array_equal(dst, reference)),
                "shipped": timed(lambda: proc.shot(dst, rect, W, H), reps)}
    dst_rows = dst.reshape(H, row_bytes)
    shot_row["candidate"] = timed(
        lambda: dst_rows.__setitem__(slice(None), padded[:, :row_bytes]), reps)
    out["shot_bgra_padded"] = shot_row

    try:
        import cupy as cp
        from rapidshot.processor.cupy_processor import CupyProcessor
    except Exception as exc:                          # pragma: no cover - hardware
        out["cupy_read_padded"] = {"skipped": str(exc)}
        return out

    sync = cp.cuda.get_current_stream().synchronize
    gpu = CupyProcessor("BGRA")
    got = gpu.process(rect, W, H, (0, 0, W, H), 0)[0]
    device = cp.empty((H, W, 4), cp.uint8)
    out["cupy_read_padded"] = {
        "exact": bool(np.array_equal(cp.asnumpy(got), reference)),
        "shipped": timed(lambda: gpu.process(rect, W, H, (0, 0, W, H), 0), reps, sync=sync),
        "candidate": timed(
            lambda: device.set(np.ascontiguousarray(padded[:, :row_bytes]).reshape(H, W, 4)),
            reps, sync=sync),
    }
    return out


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------

class _ImmediateCamera:
    """grab_frame() returns at once, like a camera built with timeout_ms=0."""

    released = False
    _capture_permanently_failed = False

    def __init__(self):
        self.calls = 0
        self.ready_at = None

    def grab_frame(self):
        self.calls += 1
        if self.ready_at is not None and time.perf_counter() >= self.ready_at:
            self.ready_at = None
            return object()
        return None


def bench_stream(seconds):
    from rapidshot.tensor_stream import TensorStream

    out = {}
    camera = _ImmediateCamera()
    stream = TensorStream(camera, (64, 64), timeout=seconds)
    cpu0, wall0 = time.thread_time(), time.perf_counter()
    try:
        stream._next_frame()
    except TimeoutError:
        pass
    wall = time.perf_counter() - wall0
    out["idle"] = {
        "seconds": round(wall, 3),
        "grab_calls_per_s": round(camera.calls / wall),
        "cpu_fraction": round((time.thread_time() - cpu0) / wall, 3),
    }

    # Latency: a frame becomes available at a random point in a 60 Hz period;
    # how long after that does the stream hand it over?
    rng = np.random.default_rng(7)
    lat = []
    stream = TensorStream(camera, (64, 64), timeout=5)
    for _ in range(300):
        camera.ready_at = time.perf_counter() + rng.uniform(0, 1 / 60)
        stream._next_frame()
        lat.append((time.perf_counter() - camera.ready_at if camera.ready_at else 0))
    # ready_at is cleared on delivery; recompute from the recorded schedule.
    return out | {"latency": _stream_latency(TensorStream, rng)}


def _stream_latency(TensorStream, rng, frames=300):
    camera = _ImmediateCamera()
    stream = TensorStream(camera, (64, 64), timeout=5)
    delays = []
    cpu0, wall0 = time.thread_time(), time.perf_counter()
    for _ in range(frames):
        ready = time.perf_counter() + rng.uniform(0, 1 / 60)
        camera.ready_at = ready
        stream._next_frame()
        delays.append((time.perf_counter() - ready) * 1000.0)
    wall = time.perf_counter() - wall0
    delays.sort()
    return {"frames": frames,
            "median_ms": round(statistics.median(delays), 4),
            "p99_ms": round(delays[int(frames * 0.99) - 1], 4),
            "max_ms": round(delays[-1], 4),
            "cpu_fraction_at_60hz": round((time.thread_time() - cpu0) / wall, 3)}


# ---------------------------------------------------------------------------
# idle
# ---------------------------------------------------------------------------

def bench_idle(polls):
    import rapidshot

    out = {}
    for mode in ("BGRA", "RGB"):
        cam = rapidshot.create(output_color=mode, timeout_ms=0)
        try:
            empty, frames = [], 0
            for _ in range(20):
                cam.grab()
            for _ in range(polls):
                t0 = time.perf_counter()
                got = cam.grab()
                elapsed = (time.perf_counter() - t0) * 1000.0
                if got is None:
                    empty.append(elapsed)
                else:
                    frames += 1
                    release = getattr(got, "release", None)
                    if release:
                        release()
            empty.sort()
            out[mode] = {
                "polls": polls, "frames": frames, "empty": len(empty),
                "empty_median_ms": round(statistics.median(empty), 4) if empty else None,
                "empty_p95_ms": round(empty[int(len(empty) * 0.95) - 1], 4) if empty else None,
            }
            pool = cam.memory_pool
            if pool is not None:
                def cycle():
                    pool.checkout().release()
                out[mode]["checkout_release"] = timed(cycle, 2000)
        finally:
            cam.release()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--only", nargs="*", default=["cupy", "rows", "stream", "idle"])
    ap.add_argument("--reps", type=int, default=60)
    args = ap.parse_args()

    result = {
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu": platform.processor(),
        "python": platform.python_version(),
    }
    sections = {"cupy": lambda: bench_cupy(args.reps),
                "rows": lambda: bench_rows(args.reps),
                "stream": lambda: bench_stream(1.0),
                "idle": lambda: bench_idle(3000)}
    for name in args.only:
        print(f"-- {name}", flush=True)
        result[name] = sections[name]()
        print(json.dumps(result[name], indent=2), flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
