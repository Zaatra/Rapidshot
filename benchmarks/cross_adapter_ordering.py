"""Convert then transfer, or transfer then convert? (ROADMAP.md 6.1)

On a hybrid laptop the frame is captured on the iGPU and consumed on the dGPU.
Two orderings are available, and the byte counts alone do not decide between
them:

  A. transfer the BGRA frame (8.29 MB at 1080p), convert on the consumer
  B. convert on the capture adapter, transfer the tensor (out*out*3*4 bytes)

B moves fewer bytes below ~832 square, and more above it. But B also spends the
*weak* adapter's time on the conversion, while A lets the fast one do it. So the
question is really: how much iGPU time does converting cost, against how much
transfer time the smaller tensor saves?

This measures the capture-side cost of each, which is the part Rapidshot owns
and controls. What the consumer adapter pays to convert in ordering A is its own
device's cost and cannot be measured from here — but it is the *faster* GPU by
assumption, so any figure here understates A's advantage.

Needs live capture: the D3D12 conversion path requires a genuinely duplicated
surface (see the synthetic-texture note in ROADMAP.md section 2).

    python benchmarks/cross_adapter_ordering.py
"""

import ctypes
import logging
import statistics
import time

import rapidshot
from rapidshot import native

# Common detection model input sizes, spanning the break-even point.
OUT_SIZES = (320, 416, 640, 832, 1024, 1280)
REPS = 30


def tensor_bytes(size):
    """NCHW float32, 3 channels."""
    return size * size * 3 * 4


def grab_texture(camera, attempts=400):
    for _ in range(attempts):
        frame = camera.grab_frame()
        if frame is not None:
            return frame, ctypes.cast(frame.d3d11_texture, ctypes.c_void_p).value
        time.sleep(0.005)
    return None, None


def time_ms(fn, reps=REPS):
    """Minimum of `reps` runs, in milliseconds.

    Minimum rather than mean: background load can only make a sample slower, so
    the minimum is the least contaminated estimate. See ROADMAP.md section 2.
    """
    fn()  # warm up
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
        print("no frames captured — Desktop Duplication reports only changed")
        print("content, so an idle screen yields nothing. Move the mouse.")
        return 1

    try:
        transfer = ext.CrossAdapterTransfer(ptr)
        print(f"source      : {transfer.source}")
        print(f"destination : {transfer.destination}")
        if transfer.destination_is_software:
            print("note        : destination is WARP, so only the *source* side of")
            print("              these numbers is representative.")
        print(f"frame       : {transfer.width}x{transfer.height}, "
              f"{transfer.total_bytes / 1e6:.2f} MB\n")

        # Ordering A, capture side: move the whole frame.
        frame_min, frame_median = time_ms(lambda: transfer.transfer(ptr))

        # Ordering B, capture side: convert here, then move the tensor.
        rows = []
        for size in OUT_SIZES:
            pre = ext.GpuPreprocessor12(ptr, size, size)
            convert_min, _ = time_ms(lambda p=pre: p.process(ptr, 1.0, 0.0, False))

            nbytes = tensor_bytes(size)
            probe = ext.probe_cross_adapter_buffer(nbytes, 40)
            if not probe.get("supported"):
                print(f"buffer probe failed at {size}: {probe.get('reason')}")
                return 1
            rows.append((size, nbytes, convert_min, probe["copy_ms_min"]))
    finally:
        frame.release()

    print(f"Ordering A — transfer the frame, convert on the consumer")
    print(f"  capture-side cost           {frame_min:6.3f} ms "
          f"(median {frame_median:.3f})")
    print(f"  consumer-side cost          conversion, on the faster GPU (not measured here)\n")

    print("Ordering B — convert on the capture adapter, transfer the tensor")
    print(f"  {'model in':>9}  {'tensor':>9}  {'convert':>9}  {'transfer':>9}  "
          f"{'B total':>9}   verdict")
    print("  " + "-" * 72)
    for size, nbytes, convert, move in rows:
        total = convert + move
        margin = frame_min - total
        verdict = (f"B saves {margin:5.2f} ms" if margin > 0
                   else f"A saves {-margin:5.2f} ms")
        print(f"  {size:>6}^2  {nbytes / 1e6:>6.2f} MB  {convert:>7.3f} ms  "
              f"{move:>7.3f} ms  {total:>7.3f} ms   {verdict}")

    print()
    better = [r for r in rows if (r[2] + r[3]) < frame_min]
    if not better:
        print("Ordering A wins at every size measured: converting on the capture")
        print("adapter costs more than transferring the extra bytes saves.")
    elif len(better) == len(rows):
        print("Ordering B wins at every size measured.")
    else:
        crossover = better[-1][0]
        print(f"B wins up to {crossover}^2; A wins above it.")
    print("\nAnd this understates A: in ordering A the conversion runs on the")
    print("consumer's GPU, which on a hybrid system is the faster one by")
    print("assumption, while ordering B always spends iGPU time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
