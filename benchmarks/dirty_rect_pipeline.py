"""Would the accumulator design actually be faster end to end? (ROADMAP.md 6.3)

`dirty_rect_savings.py` showed that converting only the dirty rects is up to
168x faster than converting the whole frame. That is not the question that
decides whether to build it.

Converting part of a frame means the rest of the destination must already hold
the previous frame, and `grab()` hands out recycled pool buffers. So the design
needs a persistent accumulator plus a copy-out per frame — and the copy-out is
full-frame work that the current path does not pay at all. This measures both
complete pipelines, including that copy, so the comparison is like for like:

    today        staging read (whole surface) + convert (whole frame)
    accumulator  staging read (dirty rows) + convert (dirty rects) + copy out

The case that matters most is the *worst* one. A desktop showing video is 100%
dirty every frame, where the accumulator saves nothing and still pays the
copy-out. If that regression is large, the optimisation needs a fallback; if it
is small, it does not.

    python benchmarks/dirty_rect_pipeline.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_suite import time_it  # noqa: E402

from rapidshot.processor.numpy_processor import NumpyProcessor  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
FRAME_PIXELS = FRAME_W * FRAME_H
PITCH = FRAME_W * 4
REPS = 25


def ms(fn):
    return min(time_it(fn, REPS)) * 1000.0


def band_rects(fraction):
    """One full-width band covering `fraction` of the frame.

    Full-width on purpose: it is the shape that makes the row-limited staging
    read meaningful, and it is what a moving window or a scrolling region
    actually dirties once rows are coalesced.
    """
    rows = max(1, int(FRAME_H * fraction))
    top = (FRAME_H - rows) // 2
    return [(0, top, FRAME_W, top + rows)]


def main() -> int:
    rng = np.random.default_rng(7)
    source = rng.integers(0, 256, (FRAME_H, PITCH), dtype=np.uint8)

    proc = NumpyProcessor("RGB")
    staged = np.empty((FRAME_H, PITCH), np.uint8)          # staging destination
    accumulator = np.empty((FRAME_H, FRAME_W, 3), np.uint8)  # persistent
    out = np.empty((FRAME_H, FRAME_W, 3), np.uint8)          # pool buffer

    def today():
        """What grab() does now: read the whole surface, convert all of it."""
        staged[:] = source
        proc.convert_into(staged.reshape(FRAME_H, FRAME_W, 4), out)

    def accumulate(rects):
        """Read and convert only what changed, then hand out a whole frame."""
        for left, top, right, bottom in rects:
            staged[top:bottom] = source[top:bottom]
            proc.convert_into(
                staged.reshape(FRAME_H, FRAME_W, 4)[top:bottom, left:right],
                accumulator[top:bottom, left:right],
            )
        out[:] = accumulator          # the cost the current path never pays

    # Correctness: the accumulator must produce the same frame as the direct
    # path once every region has been touched.
    today()
    reference = out.copy()
    accumulator[:] = 0
    accumulate([(0, 0, FRAME_W, FRAME_H)])
    assert np.array_equal(out, reference), "accumulator path produces a different frame"
    print("correctness: accumulator output matches the current path exactly\n")

    baseline = ms(today)
    print(f"Current path (staging read + full conversion): {baseline:.3f} ms\n")

    print(f"  {'dirty':>7}  {'accumulator':>12}  {'vs today':>9}   verdict")
    print("  " + "-" * 52)
    rows = []
    for fraction in (0.008, 0.03, 0.10, 0.25, 0.50, 0.75, 1.00):
        rects = band_rects(fraction)
        elapsed = ms(lambda r=rects: accumulate(r))
        ratio = baseline / elapsed
        verdict = (f"{ratio:.1f}x faster" if ratio >= 1.02
                   else f"{1 / ratio:.2f}x SLOWER" if ratio < 0.98
                   else "no change")
        print(f"  {fraction * 100:6.1f}%  {elapsed:>10.3f} ms  {ratio:>7.2f}x   {verdict}")
        rows.append((fraction, elapsed, ratio))

    # The copy-out in isolation: the floor the accumulator can never beat.
    copy_out = ms(lambda: out.__setitem__(slice(None), accumulator))
    print(f"\n  copy-out alone: {copy_out:.3f} ms — the accumulator's floor, "
          f"{baseline / copy_out:.1f}x faster than today at best")

    losing = [(f, r) for f, _, r in rows if r < 0.98]
    print()
    if losing:
        worst_f, worst_r = min(losing, key=lambda x: x[1])
        crossover = next((f for f, _, r in rows if r < 1.0), None)
        print(f"Accumulator loses above about {crossover * 100:.0f}% dirty; "
              f"worst case {1 / worst_r:.2f}x slower at {worst_f * 100:.0f}%.")
        print("A frame that reports no dirty metadata, or more dirty area than")
        print("that crossover, should take the current path instead.")
    else:
        print("Accumulator is at least as fast at every fraction measured, "
              "including 100% dirty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
