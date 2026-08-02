"""Does converting only the dirty regions actually pay? (ROADMAP.md 6.3)

`frame.dirty_rects` says which parts of the frame changed. The obvious next step
is to convert only those, instead of the whole frame. Obvious is not the same as
profitable, so this measures it before anyone builds it.

Three things could eat the saving:

  * a sub-rectangle of a NumPy array is a *strided* view, and strided access is
    less cache-friendly per pixel than the contiguous full-frame path;
  * every rect costs fixed Python and NumPy overhead, which many small rects
    multiply;
  * narrow rects are worse than wide ones at equal area, because each row
    touches a new cache line for very few useful bytes.

Reps are paced to a frame period, matching perf_suite.py — see the duty-cycle
note in ROADMAP.md section 2.

    python benchmarks/dirty_rect_savings.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from perf_suite import time_it  # noqa: E402

from rapidshot.processor.numpy_processor import NumpyProcessor  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
FRAME_PIXELS = FRAME_W * FRAME_H
REPS = 25


def make_frame():
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (FRAME_H, FRAME_W, 4), dtype=np.uint8)


def tiled_rects(count, fraction, aspect="wide"):
    """`count` rects covering `fraction` of the frame, laid out without overlap.

    `aspect` controls the shape at constant area: "wide" rects span many columns
    and few rows, "tall" rects the reverse. Real dirty rects are usually
    window-shaped, but a scrolling list or a progress bar produces extremes of
    both, and the strided-access cost depends on which.
    """
    total_area = FRAME_PIXELS * fraction
    per_rect = total_area / count
    if aspect == "wide":
        w = min(FRAME_W, int(per_rect ** 0.5 * 4))
        h = max(1, int(per_rect / w))
    else:
        h = min(FRAME_H, int(per_rect ** 0.5 * 4))
        w = max(1, int(per_rect / h))

    rects, y = [], 0
    while len(rects) < count and y + h <= FRAME_H:
        x = 0
        while len(rects) < count and x + w <= FRAME_W:
            rects.append((x, y, x + w, y + h))
            x += w
        y += h
    return rects


def covered_fraction(rects):
    return sum((r - l) * (b - t) for l, t, r, b in rects) / FRAME_PIXELS


def ms(fn, name=""):
    """Minimum of REPS paced samples, in milliseconds."""
    return min(time_it(fn, REPS, name=name)) * 1000.0


def main() -> int:
    src = make_frame()
    proc = NumpyProcessor("RGB")
    dst = np.empty((FRAME_H, FRAME_W, 3), np.uint8)

    full_min = ms(lambda: proc.convert_into(src, dst), name="convert.full")
    print(f"Full-frame RGB conversion: {full_min:.3f} ms "
          f"({FRAME_W}x{FRAME_H})\n")

    def convert_rects(rects):
        for left, top, right, bottom in rects:
            proc.convert_into(src[top:bottom, left:right],
                              dst[top:bottom, left:right])

    # Correctness before speed. A region-limited conversion that is subtly
    # wrong is worse than no optimisation at all, and the failure mode here is
    # quiet: strided views make it easy to write the right bytes to the wrong
    # place, which no timing catches.
    rects = tiled_rects(8, 0.15, "wide")
    reference = np.empty_like(dst)
    proc.convert_into(src, reference)
    dst[:] = 0
    for left, top, right, bottom in rects:
        proc.convert_into(src[top:bottom, left:right], dst[top:bottom, left:right])
    for left, top, right, bottom in rects:
        assert np.array_equal(dst[top:bottom, left:right],
                              reference[top:bottom, left:right]), \
            f"region conversion wrong at {(left, top, right, bottom)}"
    outside = dst.copy()
    for left, top, right, bottom in rects:
        outside[top:bottom, left:right] = 0
    assert not outside.any(), "region conversion wrote outside its rects"
    print("correctness: region-limited output matches the full-frame conversion "
          "inside every rect,\n             and touches nothing outside them\n")

    print("Region-limited conversion, wide rects")
    print(f"  {'dirty':>7}  {'rects':>5}  {'time':>9}  {'vs full':>9}   verdict")
    print("  " + "-" * 58)

    results = []
    for fraction in (0.008, 0.05, 0.15, 0.30, 0.50, 0.80):
        for count in (1, 8, 64):
            rects = tiled_rects(count, fraction, "wide")
            if not rects:
                continue
            actual = covered_fraction(rects)
            elapsed = ms(lambda r=rects: convert_rects(r))
            ratio = full_min / elapsed if elapsed else 0.0
            verdict = (f"{ratio:.1f}x faster" if ratio >= 1.05
                       else f"{1 / ratio:.1f}x SLOWER" if ratio else "-")
            print(f"  {actual * 100:6.1f}%  {len(rects):>5}  {elapsed:>7.3f} ms  "
                  f"{ratio:>7.2f}x   {verdict}")
            results.append((actual, len(rects), elapsed, ratio))
        print()

    # Fixed cost per rect: many tiny rects covering almost nothing.
    print("Per-rect overhead (0.1% of the frame, split N ways)")
    print(f"  {'rects':>5}  {'time':>9}  {'per rect':>10}")
    print("  " + "-" * 30)
    for count in (1, 8, 64, 256):
        rects = tiled_rects(count, 0.001, "wide")
        if len(rects) < count:
            continue
        elapsed = ms(lambda r=rects: convert_rects(r))
        print(f"  {len(rects):>5}  {elapsed:>7.3f} ms  "
              f"{elapsed / len(rects) * 1000:>8.1f} us")

    # Shape sensitivity at constant area.
    print("\nShape sensitivity (15% of the frame, 8 rects)")
    print(f"  {'aspect':>6}  {'time':>9}  {'vs full':>9}")
    print("  " + "-" * 30)
    for aspect in ("wide", "tall"):
        rects = tiled_rects(8, 0.15, aspect)
        elapsed = ms(lambda r=rects: convert_rects(r))
        print(f"  {aspect:>6}  {elapsed:>7.3f} ms  {full_min / elapsed:>7.2f}x")

    # The staging read is the other half of the CPU path, and it is the larger
    # half. Conversion saving 1.8 ms is worth much less if a 2.3 ms bulk copy
    # of the whole surface still happens first, so measure whether that can be
    # limited to the dirty rows.
    print("\nStaging read: whole surface vs dirty rows only")
    pitch = FRAME_W * 4
    mapped = np.ascontiguousarray(src).reshape(FRAME_H, pitch)
    staged = np.empty((FRAME_H, pitch), np.uint8)
    full_read = ms(lambda: staged.__setitem__(slice(None), mapped))
    print(f"  {'dirty':>7}  {'rows':>6}  {'time':>9}  {'vs full':>9}")
    print("  " + "-" * 38)
    print(f"  {100.0:6.1f}%  {FRAME_H:>6}  {full_read:>7.3f} ms  {1.0:>7.2f}x")
    for fraction in (0.008, 0.05, 0.15, 0.50):
        rects = tiled_rects(1, fraction, "wide")
        top, bottom = rects[0][1], rects[0][3]
        rows = bottom - top
        elapsed = ms(lambda t=top, b=bottom:
                     staged.__setitem__(slice(t, b), mapped[t:b]))
        print(f"  {rows / FRAME_H * 100:6.1f}%  {rows:>6}  {elapsed:>7.3f} ms  "
              f"{full_read / elapsed:>7.2f}x")

    # Where it stops paying, from the single-rect series.
    single = [(f, e) for f, n, e, _ in results if n == 1]
    breakeven = next((f for f, e in single if e >= full_min), None)
    print()
    if breakeven is None:
        print(f"Single-rect conversion beat the full frame at every fraction "
              f"measured, up to {single[-1][0] * 100:.0f}%.")
    else:
        print(f"Single-rect conversion stops paying at about "
              f"{breakeven * 100:.0f}% of the frame dirty.")
    print(f"\nLive capture measured 0.7-0.8% dirty for an animated window on an")
    print(f"otherwise still desktop. Read the top row against that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
