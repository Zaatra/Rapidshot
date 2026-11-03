"""Capture one window into NumPy, wherever it is and whatever covers it.

    python examples/window_to_numpy.py "Notepad"

The title is matched case-insensitively, as a substring; the topmost match
wins. Capture goes through Windows.Graphics.Capture, the only backend that can
take a single window, so this needs `rapidshot-native >= 0.3.0`.

What to expect, and what the camera does about it:

- The image is the window's frame bounds -- title bar and borders included --
  in its displayed orientation.
- Resize the window and the camera rebuilds at the new size;
  `camera.generation` counts the rebuilds.
- Minimise it and `grab()` waits its timeout and returns None, without
  spinning.
- Close it and capture stops: `grab()` returns None from then on and
  `camera.last_capture_error` says the window was closed.
"""
import sys
import time
from pathlib import Path

# Running `python examples/window_to_numpy.py` puts *examples/* on the path,
# not the repo root, so the import below fails on a source checkout even though
# the package is right there -- the same line the other scripts use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rapidshot  # noqa: E402


def main(title: str, seconds: float = 5.0) -> int:
    try:
        window = rapidshot.Window.find(title)
    except LookupError as e:
        print(e)
        return 1

    camera = rapidshot.create(source=window, output_color="RGB")
    print(f"capturing {window} through {camera.backend!r} for {seconds:.0f} s")
    frames, last_shape, deadline = 0, None, time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            image = camera.grab()              # None: nothing new, or minimised
            if image is None:
                if camera.last_capture_error:
                    print(f"stopped: {camera.last_capture_error}")
                    break
                continue
            frames += 1
            if image.shape != last_shape:
                print(f"  {image.shape[1]}x{image.shape[0]} "
                      f"(generation {camera.generation})")
                last_shape = image.shape
            if hasattr(image, "release"):      # a pooled buffer goes back
                image.release()
    finally:
        camera.release()
    print(f"{frames} new frames")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1]))
