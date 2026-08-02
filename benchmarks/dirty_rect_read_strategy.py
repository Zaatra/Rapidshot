"""Read dirty rects as whole rows, or only their own columns? (ROADMAP.md 6.3)

The accumulator reads each dirty rect as a band of whole rows: one vectorised
NumPy slice that copies columns nobody asked for. Live capture shows why that
might be wasteful — an animated window marks **0.8% of the frame dirty but 11.5%
of its rows**, because a tall narrow rect spans many rows while covering little
of each. Reading only the rect's own columns touches the smaller figure, at the
cost of one Python-level slice per row instead of one per band.

**Measured against a real mapped staging surface, with fixed rects.** Both
details are deliberate:

* Timing `grab()` over live frames does not work here. Each frame's cost depends
  on what happened to change on screen at that instant, so runs sample different
  workloads: the same comparison produced 2.26x, 1.56x and 0.87x on three
  consecutive attempts. That is measuring the desktop, not the code.
* Using an ordinary RAM buffer instead of the mapped surface is how the earlier
  pipeline benchmark reached a 12-15x projection that turned out to be 1.5x.
  Mapped memory is uncached and behaves nothing like RAM, and the read is
  precisely what this compares.

So: capture one real frame, keep its staging surface mapped, and drive the
conversion over it repeatedly with rect shapes chosen rather than observed.

    python benchmarks/dirty_rect_read_strategy.py
"""

import logging
import statistics
import time

import numpy as np

import rapidshot
from rapidshot.processor.numpy_processor import NumpyProcessor

REPS = 40

# Shapes worth distinguishing, all covering the same ~1% of a 1080p frame.
# The strategies only differ when a rect is narrow relative to its height.
SHAPES = (
    ("tall and narrow", 135, 124),    # the animated window live capture shows
    ("square-ish", 410, 410),
    ("wide and short", 1920, 11),     # a taskbar or a status line
)


def timed(fn, reps=REPS):
    fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return min(samples), statistics.median(samples)


def main() -> int:
    logging.getLogger("rapidshot").setLevel(logging.ERROR)
    camera = rapidshot.create(output_color="RGB")
    processor = camera._processor
    backend = processor.backend

    if not isinstance(backend, NumpyProcessor):
        print("this benchmark needs the NumPy backend")
        return 1

    # One real capture, purely to fill the staging surface with real content.
    for _ in range(4000):
        if camera.grab() is not None:
            break
        time.sleep(0.001)
    else:
        print("no frames captured — is anything moving on screen?")
        return 1

    width = camera._stagesurf.width
    height = camera._stagesurf.height
    mapped = camera._stagesurf.map()
    try:
        buffer = np.empty((height, width, 4), np.uint8)

        def convert(rects, strategy):
            backend._read_patch = strategy.__get__(backend, NumpyProcessor)
            return processor.process(mapped, width, height,
                                     (0, 0, width, height), 0, buffer,
                                     dirty_rects=rects)

        # Full conversion, for scale.
        backend.invalidate_accumulator()
        full_min, _ = timed(lambda: convert(None, NumpyProcessor._read_patch_rows))
        print(f"staging surface {width}x{height}, mapped (uncached) memory")
        print(f"full conversion: {full_min:.3f} ms\n")

        print(f"  {'rect shape':<18}{'area':>7}{'rows':>7}"
              f"{'rows read':>11}{'cols read':>11}{'columns win':>13}")
        print("  " + "-" * 67)

        for label, rect_w, rect_h in SHAPES:
            rect_w, rect_h = min(rect_w, width), min(rect_h, height)
            left = (width - rect_w) // 2
            top = (height - rect_h) // 2
            rects = [(left, top, left + rect_w, top + rect_h)]
            area = rect_w * rect_h / (width * height)
            rows = rect_h / height

            # Populate the accumulator so the patch path engages, then measure.
            results = {}
            for name, strategy in (("rows", NumpyProcessor._read_patch_rows),
                                   ("cols", NumpyProcessor._read_patch_columns)):
                backend.invalidate_accumulator()
                convert(rects, strategy)          # full conversion, seeds accumulator
                results[name], _ = timed(lambda r=rects, s=strategy: convert(r, s))

            ratio = results["rows"] / results["cols"]
            verdict = (f"{ratio:.2f}x" if ratio >= 1.02
                       else f"{1 / ratio:.2f}x SLOWER" if ratio <= 0.98
                       else "no change")
            print(f"  {label:<18}{area * 100:>6.1f}%{rows * 100:>6.1f}%"
                  f"{results['rows']:>10.3f}m{results['cols']:>10.3f}m{verdict:>13}")

        # Correctness: both strategies must produce identical frames.
        rects = [(100, 100, 235, 224)]
        backend.invalidate_accumulator()
        convert(rects, NumpyProcessor._read_patch_rows)
        by_rows, _ = convert(rects, NumpyProcessor._read_patch_rows)
        backend.invalidate_accumulator()
        convert(rects, NumpyProcessor._read_patch_columns)
        by_cols, _ = convert(rects, NumpyProcessor._read_patch_columns)
        assert np.array_equal(by_rows, by_cols), \
            "the two read strategies disagree about the frame"
        print("\ncorrectness: both strategies produce identical frames")
    finally:
        camera._stagesurf.unmap()
        backend._read_patch = NumpyProcessor._read_patch_rows.__get__(
            backend, NumpyProcessor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
