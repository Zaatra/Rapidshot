"""Crop for `GpuConverter` (ROADMAP § 7.2: crop -> resize -> convert).

**The reference is a slice.** The whole frame is read back once at identity
size with nearest sampling, which is the captured bytes, and each crop is
checked against a NumPy slice of that. Identity-size crops must match
*exactly* under both samplings: bilinear at a texel centre returns the texel,
so any difference is a misplaced sample, not filtering.

**Crops go where the content is.** A crop over flat wallpaper matches almost
anything, so each test picks the busiest window on screen and skips if even
that is flat.

Region handling — a camera created with ``region=`` — needs a camera of its
own and lives in ``test_gpu_converter_region.py``.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


@pytest.fixture(scope="module")
def live_frame():
    camera = rapidshot.create(output_color="BGRA")
    frame = None
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        camera.release()
        pytest.skip("no frame captured — the screen must be changing")
    yield frame
    frame.release()
    camera.release()


@pytest.fixture(scope="module")
def surface(live_frame):
    """(H, W, 4) BGRA of the whole frame: the captured bytes."""
    return bgra(live_frame, (live_frame.width, live_frame.height), "nearest")


def bgra(frame, size, sampling, crop=None):
    converter = rapidshot.GpuConverter(
        frame, size, dtype="uint8", layout="nhwc", sampling=sampling
    )
    return converter.process(frame, crop=crop).numpy()[0]


def busy_crop(surface, width, height, odd=True):
    """(left, top, right, bottom) of the highest-variance window, or a skip.

    Odd offsets by default, so a half-texel or off-by-one error in the offset
    cannot hide behind an even stride.
    """
    h, w = surface.shape[:2]
    best, best_std = None, -1.0
    grey = surface[..., :3].astype(np.float32).mean(axis=-1)
    for top in range(1, h - height, max(1, height // 2)):
        for left in range(1, w - width, max(1, width // 2)):
            std = grey[top:top + height, left:left + width].std()
            if std > best_std:
                best, best_std = (left | 1, top | 1), std
    if best is None or best_std < 8.0:
        pytest.skip("screen content too flat to test crop placement")
    left, top = best
    left = min(left, w - width)
    top = min(top, h - height)
    return left, top, left + width, top + height


def texels(surface, crop):
    left, top, right, bottom = crop
    return surface[top:bottom, left:right]


# --------------------------------------------------------------------------
# placement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_identity_size_crop_is_exactly_the_slice(live_frame, surface, sampling):
    crop = busy_crop(surface, 96, 64)
    got = bgra(live_frame, (96, 64), sampling, crop)
    np.testing.assert_array_equal(got, texels(surface, crop))


def test_crop_offset_is_not_off_by_one(live_frame, surface):
    """Shifting the crop by one texel must shift the output by one texel.

    Guards the case where the slice comparison above passes because the
    content is locally periodic (a text line, a gradient).
    """
    left, top, right, bottom = busy_crop(surface, 96, 64)
    a = bgra(live_frame, (96, 64), "nearest", (left, top, right, bottom))
    b = bgra(live_frame, (96, 64), "nearest", (left + 1, top, right + 1, bottom))
    if np.array_equal(a, b):
        pytest.skip("content is horizontally uniform across the crop")
    np.testing.assert_array_equal(a[:, 1:], b[:, :-1])


def test_nearest_downscaled_crop_matches_decimation(live_frame, surface):
    crop = busy_crop(surface, 300, 200)
    out_w, out_h = 64, 48
    got = bgra(live_frame, (out_w, out_h), "nearest", crop)

    left, top, right, bottom = crop
    ys = top + np.arange(out_h) * (bottom - top) // out_h
    xs = left + np.arange(out_w) * (right - left) // out_w
    np.testing.assert_array_equal(got, surface[np.ix_(ys, xs)])


def test_full_frame_crop_equals_no_crop(live_frame):
    full = (0, 0, live_frame.width, live_frame.height)
    for sampling in ("nearest", "bilinear"):
        np.testing.assert_array_equal(
            bgra(live_frame, (64, 64), sampling, full),
            bgra(live_frame, (64, 64), sampling),
        )


# --------------------------------------------------------------------------
# the filter stays inside the crop
# --------------------------------------------------------------------------


def bilinear_reference(patch, out_w, out_h):
    """Crop-then-resize with half-pixel centres and clamp-to-edge, in float."""
    src = patch.astype(np.float64)
    h, w = src.shape[:2]
    xs = np.clip((np.arange(out_w) + 0.5) * w / out_w - 0.5, 0, w - 1)
    ys = np.clip((np.arange(out_h) + 0.5) * h / out_h - 0.5, 0, h - 1)
    x0 = np.floor(xs).astype(int)
    y0 = np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (xs - x0)[None, :, None]
    fy = (ys - y0)[:, None, None]
    top = src[y0][:, x0] * (1 - fx) + src[y0][:, x1] * fx
    bottom = src[y1][:, x0] * (1 - fx) + src[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def test_upscaled_crop_corners_are_the_crop_corners(live_frame, surface):
    """At 8x upscale every corner sample clamps onto a corner texel.

    Without the clamp to the crop's texel centres, each corner blends in the
    row and column *outside* the crop — a leak that shape, range and most of
    the image would never show.
    """
    crop = busy_crop(surface, 12, 12)
    patch = texels(surface, crop)
    got = bgra(live_frame, (96, 96), "bilinear", crop)
    for gy, py in ((0, 0), (-1, -1)):
        for gx, px in ((0, 0), (-1, -1)):
            np.testing.assert_array_equal(got[gy, gx], patch[py, px])


def test_upscaled_crop_matches_crop_then_resize(live_frame, surface):
    """Whole-image check against a float reference.

    Two codes of tolerance: D3D guarantees only 6 bits of sub-texel filter
    precision, so hardware weights can be 1/64 away from exact. A leak from
    outside the crop or a half-texel offset misses by far more on busy content.
    """
    crop = busy_crop(surface, 24, 16)
    got = bgra(live_frame, (96, 64), "bilinear", crop).astype(np.float64)
    expected = bilinear_reference(texels(surface, crop), 96, 64)
    assert np.abs(got - expected).max() <= 2.0 + 1e-9


# --------------------------------------------------------------------------
# every output honours it
# --------------------------------------------------------------------------


def test_float_output_honours_crop(live_frame, surface):
    crop = busy_crop(surface, 64, 48)
    got = rapidshot.GpuConverter(
        live_frame, (64, 48), dtype="float32", sampling="nearest"
    ).process(live_frame, crop=crop).numpy()[0]
    patch = texels(surface, crop).astype(np.float32) / 255.0
    expected = np.stack([patch[..., 2], patch[..., 1], patch[..., 0]])
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_nv12_output_honours_crop(live_frame, surface):
    crop = busy_crop(surface, 64, 48)
    got = rapidshot.GpuConverter(
        live_frame, (64, 48), pixel_format="nv12", sampling="nearest"
    ).process(live_frame, crop=crop).numpy()
    patch = texels(surface, crop).astype(np.float64) / 255.0
    luma = 0.2126 * patch[..., 2] + 0.7152 * patch[..., 1] + 0.0722 * patch[..., 0]
    expected = np.clip(np.floor(16 + 219 * luma + 0.5), 0, 255)
    assert np.abs(got[:48].astype(np.int64) - expected).max() <= 1


# --------------------------------------------------------------------------
# default and per-call
# --------------------------------------------------------------------------


def test_constructor_crop_is_the_default_and_process_overrides_it(live_frame, surface):
    a = busy_crop(surface, 32, 32)
    b = (0, 0, 32, 32)
    converter = rapidshot.GpuConverter(
        live_frame, (32, 32), dtype="uint8", layout="nhwc",
        sampling="nearest", crop=a,
    )
    assert converter.crop == a

    np.testing.assert_array_equal(converter.process(live_frame).numpy()[0], texels(surface, a))
    np.testing.assert_array_equal(
        converter.process(live_frame, crop=b).numpy()[0], texels(surface, b)
    )
    # An override is for that call only.
    np.testing.assert_array_equal(converter.process(live_frame).numpy()[0], texels(surface, a))


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "crop",
    [
        "outside-right",
        "outside-bottom",
        (-1, 0, 10, 10),
        (10, 10, 10, 20),   # empty width
        (10, 20, 20, 20),   # empty height
        (20, 10, 10, 20),   # inverted
        (0, 0, 10),         # arity
        "abcd",
    ],
)
def test_bad_crops_are_refused_not_clamped(live_frame, crop):
    if crop == "outside-right":
        crop = (0, 0, live_frame.width + 1, 10)
    elif crop == "outside-bottom":
        crop = (0, 0, 10, live_frame.height + 1)
    with pytest.raises(ValueError, match="crop"):
        rapidshot.GpuConverter(live_frame, (16, 16), crop=crop)
    converter = rapidshot.GpuConverter(live_frame, (16, 16))
    with pytest.raises(ValueError, match="crop"):
        converter.process(live_frame, crop=crop)


def test_native_layer_refuses_a_crop_outside_the_surface(live_frame):
    """The Python check is the one users hit; this is the one that protects
    the shader from reading past the surface if a caller goes around it."""
    converter = rapidshot.GpuConverter(live_frame, (16, 16))
    with pytest.raises(RuntimeError, match="does not fit"):
        converter._impl.process(
            native._texture_address(live_frame), 1.0, 0.0, False,
            source_id=live_frame.source_id,
            crop=(live_frame.width - 8, 0, 16, 16),
        )
