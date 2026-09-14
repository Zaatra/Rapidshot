"""Convert then transfer, or transfer then convert? With the kernels that
were missing. (ROADMAP.md § 6.1)

`cross_adapter_ordering_v2.py` established that **B — convert first — wins
unconditionally at every size and representation**, because B's total beat A's
transfer *alone*, which is a bound rather than an estimate. It carried one
stated caveat, and this script removes it:

    Convert cost is the FP32 NCHW path for every row - GpuPreprocessor12
    emits only that. A resize-only kernel producing BGRA8 or FP16 would
    cost less, so those rows are pessimistic on the convert column and
    exact on transfer.

2.6's `GpuConverter` emits all three, so each row can now be measured with the
kernel that would actually produce it. **This can only move the result further
in B's favour** — the v2 rows already won while paying FP32's conversion for a
BGRA8 payload. What is new here is the size of the margin, and whether the
cheap representations are cheap to *produce* as well as to move.

Two things this deliberately does not do:

* **It does not re-open the ordering question.** That is settled; ROADMAP § 4
  exists because this project keeps re-litigating measured questions. This
  quantifies a margin.
* **It does not include A's destination-side conversion**, which still cannot
  be timed from here — `GpuConverter` is built from the source texture and
  runs on that adapter. It does not need to be: A is already losing without
  it, and adding a cost to the losing side cannot change the verdict.

Sampling is **nearest** throughout, matching v2, so the convert column is
comparable across the two scripts. Bilinear is the `GpuConverter` default and
costs more; timing it here would confound a kernel change with a sampling
change, which is the § 7.2 warning about stating which sampling was used.
"""

import ctypes
import logging
import statistics
import sys
import time
from pathlib import Path

# Run from a checkout without installing, as perf_suite.py does. `v2` omits
# this and therefore only runs against an installed rapidshot, which is a
# difference worth not inheriting.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rapidshot
from rapidshot import native

OUT_SIZES = (320, 416, 640, 832, 1024, 1280)
REPS = 30

#: (label, dtype, layout, bytes-per-output-pixel). Smallest payload first.
REPRESENTATIONS = (
    ("BGRA8 resize", "uint8", "nhwc", 4),
    ("FP16 NCHW", "float16", "nchw", 3 * 2),
    ("FP32 NCHW", "float32", "nchw", 3 * 4),
)


def grab_texture(camera, attempts=400):
    for _ in range(attempts):
        frame = camera.grab_frame()
        if frame is not None:
            return frame, ctypes.cast(frame.d3d11_texture, ctypes.c_void_p).value
        time.sleep(0.005)
    return None, None


def time_ms(fn, reps=REPS):
    """Minimum of `reps` runs, in milliseconds. See ROADMAP.md § 2."""
    fn()
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return min(samples), statistics.median(samples)


def main() -> int:
    if not native.is_available():
        print("native extension not built; cannot measure")
        return 1
    logging.getLogger("rapidshot").setLevel(logging.ERROR)

    ext = native.require()
    camera = rapidshot.create(output_color="BGRA")

    frame, ptr = grab_texture(camera)
    if frame is None:
        print("no frames captured - an idle screen yields nothing. Move the mouse.")
        return 1

    try:
        transfer = ext.CrossAdapterTransfer(ptr)
        print(f"source      : {transfer.source}")
        print(f"destination : {transfer.destination}")
        if transfer.destination_is_software:
            print("note        : destination is WARP, so only the *source* side")
            print("              of these numbers is representative.")
        print(f"frame       : {transfer.width}x{transfer.height}, "
              f"{transfer.total_bytes / 1e6:.2f} MB")
        print("sampling    : nearest (matches v2; bilinear costs more)\n")

        # Ordering A, lower bound: the transfer alone. A's destination-side
        # conversion is extra and unmeasured, so true A is strictly worse.
        a_min, a_median = time_ms(lambda: transfer.transfer(ptr))
        print(f"A lower bound (full-frame transfer only): "
              f"{a_min:.2f} ms min, {a_median:.2f} ms median")
        print("  A's destination conversion is NOT included and cannot be "
              "negative,\n  so true A > this figure.\n")

        rows = []
        for size in OUT_SIZES:
            for label, dtype, layout, bpp in REPRESENTATIONS:
                nbytes = size * size * bpp
                converter = rapidshot.GpuConverter(
                    frame, (size, size), dtype=dtype, layout=layout,
                    sampling="nearest",
                )
                if converter.output_byte_size != nbytes:
                    print(f"payload mismatch at {size} {label}: "
                          f"{converter.output_byte_size} != {nbytes}")
                    return 1
                convert_min, _ = time_ms(lambda c=converter: c.process(frame))

                probe = ext.probe_cross_adapter_buffer(nbytes, 40)
                if not probe.get("supported"):
                    print(f"buffer probe failed at {size} {label}: "
                          f"{probe.get('reason')}")
                    return 1
                rows.append((size, label, nbytes, convert_min,
                             probe["copy_ms_min"]))
    finally:
        frame.release()
        camera.release()

    print(f"{'out':>6} {'payload':>13} {'MB':>7} {'convert':>9} "
          f"{'transfer':>9} {'B total':>9}   verdict")
    for size, label, nbytes, convert_ms, copy_ms in rows:
        total = convert_ms + copy_ms
        if total < a_min:
            verdict = f"B wins by >={a_min - total:.2f} ms"
        else:
            verdict = "A may win (needs its conversion cost)"
        print(f"{size:>6} {label:>13} {nbytes / 1e6:>7.2f} "
              f"{convert_ms:>8.2f}m {copy_ms:>8.2f}m {total:>8.2f}m   {verdict}")

    print("\n'B wins' is unconditional: it beats A's transfer alone, so no "
          "value of\nA's destination conversion changes it.")
    print("\nEach row's convert column is now measured with the kernel that "
          "would\nactually produce that payload, which is what v2 could not do.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
