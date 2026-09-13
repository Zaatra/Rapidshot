"""Convert then transfer, or transfer then convert? Corrected. (ROADMAP.md 6.1)

`cross_adapter_ordering.py` settled this in favour of transferring the frame.
Two things about that comparison are wrong, and both favour ordering A:

  1. **A's destination-side conversion was never counted.** A was timed as the
     transfer alone. Its docstring says the figures "understate A's advantage"
     because the consumer's GPU is faster -- but omitting one of A's costs
     entirely *overstates* A, whatever the speed of the GPU that would pay it.
     Relative speed and an uncounted term are different arguments.
  2. **B was only ever measured carrying FP32.** The tensor sizes are
     `out*out*3*4`. A 640-square frame does not have to cross as 4.92 MB: FP16
     is 2.46 MB and a plain BGRA8 resize is 1.64 MB. The cheap representations
     -- the whole point of shrinking before the bus -- were never on the table.

It was also measured at 1080p (8.29 MB). This machine captures 2560x1600
(16.38 MB), which doubles the side of the ledger A pays.

**How this gets a decisive answer without the missing number.** A's destination
conversion cannot be timed from here: `GpuPreprocessor12` is built from the
source texture and runs on that adapter, and nothing in the native API builds
one on the destination device. But it does not need to be timed. It is
positive, so:

    A_total  >  transfer(full frame)
    B_total  =  convert(iGPU) + transfer(payload)

If `transfer(full frame)` alone already exceeds `B_total`, then A loses even in
the impossible best case where its destination conversion is free -- and the
unmeasured term cannot rescue it. That is a bound, not an estimate.

What this still does not measure: a source-side *resize-only* path producing
BGRA8 or FP16. `GpuPreprocessor12` emits FP32 NCHW, so the conversion cost for
those rows is the FP32 path's cost, which is an over-estimate for a cheaper
output and an under-estimate for nothing. Their transfer costs are real.

    python benchmarks/cross_adapter_ordering_v2.py
"""

import ctypes
import logging
import statistics
import time

import rapidshot
from rapidshot import native

OUT_SIZES = (320, 416, 640, 832, 1024, 1280)
REPS = 30

#: Ways the same H x W frame could cross the bus, smallest first.
PAYLOADS = (
    ("BGRA8 resize", lambda n: n * n * 4),
    ("FP16 NCHW", lambda n: n * n * 3 * 2),
    ("FP32 NCHW", lambda n: n * n * 3 * 4),
)


def grab_texture(camera, attempts=400):
    for _ in range(attempts):
        frame = camera.grab_frame()
        if frame is not None:
            return frame, ctypes.cast(frame.d3d11_texture, ctypes.c_void_p).value
        time.sleep(0.005)
    return None, None


def time_ms(fn, reps=REPS):
    """Minimum of `reps` runs, in milliseconds. See ROADMAP.md section 2."""
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
              f"{transfer.total_bytes / 1e6:.2f} MB\n")

        # Ordering A, lower bound: the transfer alone. Destination conversion is
        # extra and unmeasured, so the real A is strictly worse than this.
        a_min, a_median = time_ms(lambda: transfer.transfer(ptr))
        print(f"A lower bound (full-frame transfer only): "
              f"{a_min:.2f} ms min, {a_median:.2f} ms median")
        print("  A's destination conversion is NOT included and cannot be "
              "negative,\n  so true A > this figure.\n")

        rows = []
        for size in OUT_SIZES:
            pre = ext.GpuPreprocessor12(ptr, size, size)
            convert_min, _ = time_ms(lambda p=pre: p.process(ptr, 1.0, 0.0, False))
            for label, nbytes_of in PAYLOADS:
                nbytes = nbytes_of(size)
                probe = ext.probe_cross_adapter_buffer(nbytes, 40)
                if not probe.get("supported"):
                    print(f"buffer probe failed at {size} {label}: "
                          f"{probe.get('reason')}")
                    return 1
                rows.append((size, label, nbytes, convert_min,
                             probe["copy_ms_min"]))
    finally:
        frame.release()

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
          "value of\nA's destination conversion changes it. 'A may win' is "
          "undecided here.")
    print("\nConvert cost is the FP32 NCHW path for every row - "
          "GpuPreprocessor12 emits\nonly that. A resize-only kernel producing "
          "BGRA8 or FP16 would cost less,\nso those rows are pessimistic on "
          "the convert column and exact on transfer.")

    camera.release()
    rapidshot.reset()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
