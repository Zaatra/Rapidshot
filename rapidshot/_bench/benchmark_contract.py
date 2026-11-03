"""Pure section 7 contracts. Importing this module never touches the desktop."""
import ctypes
import hashlib
import json
import math
from pathlib import Path
import time

OUT = 640
SHAPE = (1, 3, OUT, OUT)
#: Largest per-channel difference, in RGB8 levels, that `pipeline_rgb` may show
#: against `canonical_rgb`: cv2's fixed-point bilinear rounds differently from
#: the exact contract but computes the same interpolation.
PIPELINE_TOLERANCE_RGB8 = 1
# Moved to rapidshot/util/pixel_age.py so library code can measure without
# importing the harness; re-exported here under the same names.
from rapidshot.util.pixel_age import (  # noqa: E402,F401
    CELL, MAGIC, MARKER_BITS, MARKER_HEIGHT, PresentLog, checksum, decode_marker,
    percentiles, qpc_clock,
)


def pipeline_rgb(bgra, xp, size=OUT):
    """The resize a deployment would actually run, per backend.

    NumPy inputs go through cv2's fixed-point bilinear; CuPy inputs use the
    exact contract, which is already ~1.2 ms there and needs no substitute.
    Same algorithm, same output size, same full-image extent -- only the
    rounding differs, and `canonical_rgb` remains the reference every path is
    verified against.

    Using `canonical_rgb` in the measured loop instead would charge NumPy paths
    76-86 ms for what cv2 does in 0.72 ms. That is not a fairer comparison; it
    is a slower one applied only to the competitors.
    """
    if getattr(xp, "__name__", "") == "numpy":
        import cv2

        if len(bgra.shape) != 3 or bgra.shape[2] != 4 or str(bgra.dtype) != "uint8":
            raise ValueError("expected uint8 HxWx4 BGRA")
        small = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_LINEAR)
        return xp.ascontiguousarray(small[:, :, 2::-1])
    return canonical_rgb(bgra, xp, size)


#: How this contract maps a source frame into the model's square input.
#: `canonical_rgb` resizes the **whole** frame to size x size with independent
#: x and y scales -- a stretch, not a letterbox. Declared here because the
#: detection postprocessing has to invert exactly this, and a live run showed
#: what happens when it does not: the model saw objects squashed 1.6x
#: vertically while boxes were restored with a letterbox inverse, and the frame
#: that should have held 8 detections reported up to 37.
TENSOR_GEOMETRY = "stretch"


def canonical_rgb(bgra, xp, size=OUT):
    """Exact rational half-pixel bilinear, round-half-up to RGB8.

    Integer arithmetic fixes CPU/GPU interpolation and rounding differences.
    No FMA or hardware texture-sampler precision enters this contract.
    """
    if len(bgra.shape) != 3 or bgra.shape[2] != 4 or str(bgra.dtype) != "uint8":
        raise ValueError("expected uint8 HxWx4 BGRA")
    h, w = bgra.shape[:2]
    if min(h, w, size) <= 0:
        raise ValueError("empty image or output")
    den = 2 * size
    yn = xp.maximum((2 * xp.arange(size, dtype=xp.int64) + 1) * h - size, 0)
    xn = xp.maximum((2 * xp.arange(size, dtype=xp.int64) + 1) * w - size, 0)
    y0, x0 = yn // den, xn // den
    y1, x1 = xp.minimum(y0 + 1, h - 1), xp.minimum(x0 + 1, w - 1)
    wy, wx = (yn % den)[:, None, None], (xn % den)[None, :, None]
    a = bgra[y0[:, None], x0[None, :], :3].astype(xp.int64)
    b = bgra[y0[:, None], x1[None, :], :3].astype(xp.int64)
    c = bgra[y1[:, None], x0[None, :], :3].astype(xp.int64)
    d = bgra[y1[:, None], x1[None, :], :3].astype(xp.int64)
    value = ((a * (den - wx) + b * wx) * (den - wy)
             + (c * (den - wx) + d * wx) * wy)
    rounded = ((value + den * den // 2) // (den * den)).astype(xp.uint8)
    return xp.ascontiguousarray(rounded[:, :, ::-1])


def normalized_tensor(rgb, xp):
    chw = xp.ascontiguousarray(rgb.transpose(2, 0, 1))
    return (chw.astype(xp.float32) / xp.float32(255)).astype(xp.float16)[None]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


#: A test source that has said it is ready must then present steadily. On the
#: Intel desktop about 15% of sources launched after an earlier one presented
#: 3 frames, after which the display stopped completing their flips: every
#: Present() still returned S_OK, 2 s late, with the window visible and
#: uncloaked, and it never recovered. The first launch of a run never did it
#: (0 of 57), and the trigger is unknown. A relaunch cleared it every time.
HEALTHY_PRESENTS = 20
HEALTHY_WINDOW_S = 1.0
SOURCE_LAUNCHES = 3


def count_presents(path) -> int:
    """Frames a source has presented, from its own log (truncated at each launch)."""
    try:
        with Path(path).open("rb") as log:
            return sum(1 for line in log if b'"present"' in line)
    except OSError:
        return 0


def presenting_steadily(path, fps=0.0, *, since=0, clock=time.monotonic, sleep=time.sleep):
    """(steady, presents seen) for a source that has just reported ready, or
    just switched workload: ``since`` is the count to measure from.

    Steady means HEALTHY_PRESENTS within HEALTHY_WINDOW_S, or half the source's
    rate over that window when it presents slower than that.
    """
    required = HEALTHY_PRESENTS
    if fps and fps > 0:
        required = max(2, min(HEALTHY_PRESENTS, int(fps * HEALTHY_WINDOW_S / 2)))
    deadline = clock() + HEALTHY_WINDOW_S
    while True:
        seen = count_presents(path) - since
        if seen >= required:
            return True, seen
        if clock() >= deadline:
            return False, seen
        sleep(0.02)


def open_mss():
    """An mss session on whichever API this mss has: ``mss.MSS`` from 10.2, which
    deprecates the ``mss.mss()`` factory for removal after the 10.x series, and
    that factory before it. The extra floors mss at 9, not at the newest."""
    import mss
    return (getattr(mss, "MSS", None) or mss.mss)()


def primary_monitor(monitors):
    """mss's entry for the primary display, which is where the test source draws.

    Not ``monitors[1]``: that is the first display Windows enumerates, which on a
    machine with two is as often the secondary. mss >= 10.2 marks the primary;
    older versions do not, but on Windows the primary is by definition the one
    at the desktop origin.
    """
    displays = list(monitors[1:])
    for test in (lambda m: m.get("is_primary"),
                 lambda m: (m.get("left"), m.get("top")) == (0, 0)):
        match = next((m for m in displays if test(m)), None)
        if match is not None:
            return match
    return displays[0]
