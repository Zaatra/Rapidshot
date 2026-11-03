"""Turn one window into a model-ready tensor, without the pixels touching the CPU.

    python examples/window_to_tensor.py "Notepad"

The window is captured through Windows.Graphics.Capture and converted on the
capture GPU -- resize to 640x640, RGB, normalise, FP16, NCHW -- in one
dispatch; only the 2.46 MB tensor is left for the model. With PyTorch and CUDA
on the capture adapter it is handed over zero-copy through `to_torch()`.

Without PyTorch, or on a hybrid laptop where the window is captured on the
integrated GPU and CUDA sees only the discrete one (`CrossAdapterRequired`),
the script still runs and reads the tensor back with `numpy()` so you can see
the numbers -- a check, not a path to use. The hybrid answer is one more
object, `rapidshot.TensorTransfer`; see README, "Hybrid GPU laptops".

A window can change size, and a converter is built for one source size, so
this rebuilds it whenever `frame.generation` moves.

Requires: the RapidShot native extension (`rapidshot-native >= 0.3.0`).
"""
import sys
import time
from pathlib import Path

# Running `python examples/window_to_tensor.py` puts *examples/* on the path,
# not the repo root, so the import below fails on a source checkout even though
# the package is right there -- the same line the other scripts use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rapidshot  # noqa: E402

if rapidshot.GpuConverter is None:
    sys.exit("this example needs the native extension:  pip install rapidshot[native]")


def main(title: str, seconds: float = 5.0) -> int:
    try:
        window = rapidshot.Window.find(title)
    except LookupError as e:
        print(e)
        return 1

    camera = rapidshot.create(source=window)
    converter, generation, use_torch = None, None, True
    frames, deadline = 0, time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            frame = camera.grab_frame()        # None: nothing new, or minimised
            if frame is None:
                if camera.last_capture_error:
                    print(f"stopped: {camera.last_capture_error}")
                    break
                continue
            with frame:
                if frame.generation != generation:   # first frame, or resized
                    converter = rapidshot.GpuConverter(
                        frame, (640, 640), dtype="float16", layout="NCHW")
                    generation = frame.generation
                    print(f"{frame.width}x{frame.height} {frame.pixel_format} window "
                          f"-> {converter.shape} {converter.dtype}")
                tensor = converter.process(frame)
            frames += 1

            if use_torch:
                try:
                    x = tensor.to_torch()      # zero-copy; the next process() overwrites it
                except (ImportError, rapidshot.CrossAdapterRequired) as e:
                    print(f"no zero-copy consumer here ({type(e).__name__}); "
                          "reading back with numpy() instead")
                    use_torch = False
                else:
                    tensor.sync()
                    if frames == 1:
                        print(f"torch: {tuple(x.shape)} {x.dtype} on {x.device}")
            if not use_torch and frames == 1:
                x = tensor.numpy()
                print(f"numpy: {x.shape} {x.dtype}, mean {float(x.mean()):.3f}")
    finally:
        converter = None                       # before the device it was built on
        camera.release()
    print(f"{frames} tensors")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
