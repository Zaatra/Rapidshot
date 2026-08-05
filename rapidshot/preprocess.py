"""Turn a captured frame into model input, correctly and without three extra passes.

A model does not want a screenshot. It wants a fixed size, values as float32
rather than bytes, and the colour channels split into separate planes -- the
layout usually written ``NCHW``. That conversion is five lines of NumPy, and the
five lines people write first cost **6.64 ms** per 1080p frame while an
equivalent that produces bit-identical output costs **3.99 ms**. The difference
is entirely in avoiding work: the obvious version widens the whole sampled image
to float32 *before* scaling it, divides in a second pass, stacks the channels
into a fresh array in a third, and allocates ~11 MB every call.

Two ways to get this wrong that no error message will catch:

* **Channel order.** Feeding a model BGR while telling it RGB does not fail. The
  model runs, returns plausible output, and is quietly worse. `source_order` is
  required for exactly this reason -- an ``(H, W, 3)`` array cannot tell you
  whether it holds RGB or BGR, so guessing would be guessing on the caller's
  behalf about something they cannot see go wrong.
* **Resampling.** This decimates by index (nearest neighbour), which is fast and
  is what the capture benchmarks measure. It is *not* the bilinear or area
  resize most detection models were trained with. For a model that cares, resize
  with OpenCV instead and pass the result here only for the layout conversion.

Deliberately narrow: output size and channel order, nothing else. No mean/std
normalisation, no float16, no NHWC, no letterboxing. Those are all decided by
the model rather than chosen for speed, and their costs are within ~1.5x of each
other (2.5-5.1 ms measured), so there is nothing to gain by guessing which one a
caller wants. ROADMAP.md section 11 is the reason this stops here: RapidShot
produces frames, and preprocessing is the edge of what that means.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["to_nchw"]

# Which source index feeds each output plane, per (source_order, output_order).
_CHANNEL_MAPS = {
    ("RGB", "RGB"): (0, 1, 2),
    ("RGB", "BGR"): (2, 1, 0),
    ("BGR", "RGB"): (2, 1, 0),
    ("BGR", "BGR"): (0, 1, 2),
    ("RGBA", "RGB"): (0, 1, 2),
    ("RGBA", "BGR"): (2, 1, 0),
    ("BGRA", "RGB"): (2, 1, 0),
    ("BGRA", "BGR"): (0, 1, 2),
}

_ORDERS = tuple(sorted({o for pair in _CHANNEL_MAPS for o in pair[:1]}))


@lru_cache(maxsize=32)
def _sample_index(src_h: int, src_w: int, out_h: int, out_w: int):
    """Row/column indices that decimate ``src`` down to ``out``.

    Cached because the arrays depend only on the two shapes, and a capture loop
    calls this with the same pair every frame.

    The indices are deliberately *not* a uniform stride. 1080 -> 640 steps by
    1.6875, so they run 0, 1, 3, 5, 6, 8... Replacing this with a strided slice
    looks 7x faster and silently samples a different image -- measured, and the
    reason every variant in the benchmarks is checked against its reference
    output before its timing is believed.
    """
    ys = (np.arange(out_h) * src_h // out_h).clip(0, src_h - 1)
    xs = (np.arange(out_w) * src_w // out_w).clip(0, src_w - 1)
    return np.ix_(ys, xs)


def to_nchw(
    frame,
    size: Sequence[int],
    source_order: str,
    output_order: str = "RGB",
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert a captured frame to ``(1, 3, H, W)`` float32 scaled to 0-1.

    Args:
        frame: ``(H, W, 3)`` or ``(H, W, 4)`` uint8 image. A ``PooledBuffer``
            works directly; anything ``np.asarray`` accepts does.
        size: ``(width, height)`` of the model input, e.g. ``(640, 640)``.
        source_order: Channel order of ``frame`` -- ``"RGB"``, ``"BGR"``,
            ``"RGBA"`` or ``"BGRA"``. Pass whatever ``output_color`` the camera
            was created with. Required, because the array cannot tell you.
        output_order: Channel order the model expects. Defaults to ``"RGB"``.
        out: Optional ``(1, 3, H, W)`` float32 array to write into. Supplying
            one removes the per-call allocation, which is a large part of why
            this is faster than the naive version -- reuse it across frames.

    Returns:
        ``(1, 3, height, width)`` float32 with values in 0-1. This is ``out``
        when ``out`` was given.

    Raises:
        ValueError: If the frame, size, orders or ``out`` buffer are unusable.

    Example:
        >>> camera = rapidshot.create(output_color="RGB")   # doctest: +SKIP
        >>> tensor = rapidshot.to_nchw(camera.grab(), (640, 640), "RGB")
    """
    image = np.asarray(frame)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(
            f"frame must be (H, W, 3) or (H, W, 4); got shape {image.shape}"
        )
    if image.dtype != np.uint8:
        raise ValueError(f"frame must be uint8; got {image.dtype}")

    try:
        out_w, out_h = (int(v) for v in size)
    except (TypeError, ValueError):
        raise ValueError(f"size must be (width, height); got {size!r}") from None
    if out_w < 1 or out_h < 1:
        raise ValueError(f"size must be positive; got {(out_w, out_h)}")

    key = (source_order, output_order)
    if key not in _CHANNEL_MAPS:
        raise ValueError(
            f"unsupported channel orders {key}; source_order must be one of "
            f"{_ORDERS} and output_order one of ('RGB', 'BGR')"
        )
    mapping = _CHANNEL_MAPS[key]
    # Compare against the *name's* width, not the highest index it happens to
    # use. "BGRA" on a 3-channel frame indexes 0..2 and so never reads out of
    # bounds -- it quietly produces the right answer for the wrong reason, and
    # leaves a caller believing they have an alpha channel they do not.
    if len(source_order) > image.shape[2]:
        raise ValueError(
            f"source_order {source_order!r} describes {len(source_order)} "
            f"channels but the frame has {image.shape[2]}"
        )

    expected = (1, 3, out_h, out_w)
    if out is None:
        out = np.empty(expected, dtype=np.float32)
    else:
        if out.shape != expected:
            raise ValueError(f"out must have shape {expected}; got {out.shape}")
        if out.dtype != np.float32:
            raise ValueError(f"out must be float32; got {out.dtype}")

    src_h, src_w = image.shape[:2]
    sampled = image[_sample_index(src_h, src_w, out_h, out_w)]

    # One pass per output plane, straight into the destination. The sampled
    # gather stays uint8 -- widening it first would move four times the bytes
    # for no benefit, which is the single biggest cost in the naive version.
    for plane, source_channel in enumerate(mapping):
        np.divide(sampled[..., source_channel], 255.0, out=out[0, plane])

    return out
