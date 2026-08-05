"""Tests for rapidshot.to_nchw (ROADMAP.md 10 -- model-input conversion).

The interesting failures here are all silent. A wrong channel order does not
raise, it just feeds the model BGR while claiming RGB and makes results quietly
worse; a wrong resample picks different pixels and still returns the right
shape. So most of what is pinned here is *which pixel ended up where*, not that
the call succeeded.
"""

import numpy as np
import pytest

from rapidshot.preprocess import to_nchw


def solid(colour, height=120, width=200):
    """An image where every pixel is `colour`, so channel routing is visible."""
    frame = np.zeros((height, width, len(colour)), np.uint8)
    frame[:] = colour
    return frame


# --------------------------------------------------------------------------
# shape, dtype, range
# --------------------------------------------------------------------------

def test_produces_nchw_float32_scaled_to_unit_range():
    out = to_nchw(solid((255, 128, 0)), (64, 32), "RGB")
    assert out.shape == (1, 3, 32, 64)     # (N, C, H, W) from (width, height)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_size_is_width_height_not_height_width():
    """A transposed size silently produces a differently-shaped tensor."""
    out = to_nchw(solid((1, 2, 3)), (640, 480), "RGB")
    assert out.shape == (1, 3, 480, 640)


# --------------------------------------------------------------------------
# channel routing -- the silent one
# --------------------------------------------------------------------------

@pytest.mark.parametrize("source_order,pixel,expected", [
    # A pixel that is unambiguous per channel: R=10, G=20, B=30 in each layout.
    ("RGB", (10, 20, 30), (10, 20, 30)),
    ("BGR", (30, 20, 10), (10, 20, 30)),
    ("RGBA", (10, 20, 30, 255), (10, 20, 30)),
    ("BGRA", (30, 20, 10, 255), (10, 20, 30)),
])
def test_source_order_routes_channels_correctly(source_order, pixel, expected):
    """Whatever the input layout, plane 0 must end up holding red."""
    out = to_nchw(solid(pixel), (8, 8), source_order)
    got = tuple(round(float(out[0, c, 0, 0]) * 255) for c in range(3))
    assert got == expected, f"{source_order}: planes hold {got}, expected {expected}"


def test_output_order_bgr_swaps_the_planes():
    out = to_nchw(solid((10, 20, 30)), (8, 8), "RGB", output_order="BGR")
    got = tuple(round(float(out[0, c, 0, 0]) * 255) for c in range(3))
    assert got == (30, 20, 10)


def test_alpha_is_dropped_not_included():
    """A 4-channel source must yield 3 planes, with alpha discarded."""
    out = to_nchw(solid((10, 20, 30, 77)), (8, 8), "RGBA")
    assert out.shape[1] == 3
    assert round(float(out[0, 0, 0, 0]) * 255) == 10


# --------------------------------------------------------------------------
# resampling -- picks the right pixels, not merely the right count
# --------------------------------------------------------------------------

def test_downsample_selects_spread_out_pixels_not_a_crop():
    """A strided slice would return the top-left corner and the same shape.

    The resize indices are not a uniform stride (1080/640 = 1.6875), and a
    variant that assumed they were measured 7x faster while silently sampling a
    different image. A horizontal ramp catches it: a correct decimation spans
    the full range, a crop does not.
    """
    width = 640
    ramp = np.zeros((16, width, 3), np.uint8)
    ramp[..., 0] = (np.arange(width) * 255 // (width - 1)).astype(np.uint8)

    out = to_nchw(ramp, (64, 8), "RGB")
    row = out[0, 0, 0] * 255.0
    assert row[0] < 10, row[0]
    assert row[-1] > 245, row[-1]          # reached the far edge, so not a crop
    assert np.all(np.diff(row) >= -1e-3)   # monotonic, so ordering preserved


def test_upsizing_is_allowed():
    out = to_nchw(solid((10, 20, 30), height=4, width=4), (16, 16), "RGB")
    assert out.shape == (1, 3, 16, 16)
    assert round(float(out[0, 0, 5, 5]) * 255) == 10


# --------------------------------------------------------------------------
# the `out` buffer, which is where the speed comes from
# --------------------------------------------------------------------------

def test_out_buffer_is_written_in_place_and_returned():
    buf = np.zeros((1, 3, 32, 64), np.float32)
    result = to_nchw(solid((255, 0, 0)), (64, 32), "RGB", out=buf)
    assert result is buf
    assert buf[0, 0].max() == 1.0


def test_reusing_out_allocates_nothing():
    """Reuse is the point: the naive version's ~11 MB per call is most of its cost."""
    frame = solid((10, 20, 30), height=480, width=640)
    buf = np.empty((1, 3, 64, 64), np.float32)
    to_nchw(frame, (64, 64), "RGB", out=buf)      # warm the cached index

    import tracemalloc
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(10):
            to_nchw(frame, (64, 64), "RGB", out=buf)
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    # The gather still allocates; what must not grow is the output side.
    assert after - before < 64 * 64 * 3 * 4, after - before


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frame,message", [
    (np.zeros((10, 10), np.uint8), "H, W"),
    (np.zeros((10, 10, 2), np.uint8), "H, W"),
    (np.zeros((10, 10, 3), np.float32), "uint8"),
])
def test_rejects_unusable_frames(frame, message):
    with pytest.raises(ValueError, match=message):
        to_nchw(frame, (8, 8), "RGB")


@pytest.mark.parametrize("size", [(0, 8), (8, 0), (-1, 8), "640", (8,)])
def test_rejects_bad_sizes(size):
    with pytest.raises(ValueError, match="size"):
        to_nchw(solid((1, 2, 3)), size, "RGB")


def test_rejects_unknown_channel_orders():
    with pytest.raises(ValueError, match="channel orders"):
        to_nchw(solid((1, 2, 3)), (8, 8), "YUV")
    with pytest.raises(ValueError, match="channel orders"):
        to_nchw(solid((1, 2, 3)), (8, 8), "RGB", output_order="GRAY")


def test_rejects_a_source_order_wider_than_the_frame():
    """Claiming BGRA for a 3-channel frame must fail, not read past the end."""
    with pytest.raises(ValueError, match="channels"):
        to_nchw(solid((1, 2, 3)), (8, 8), "BGRA")


@pytest.mark.parametrize("buf,message", [
    (np.empty((1, 3, 8, 9), np.float32), "shape"),
    (np.empty((1, 3, 8, 8), np.float64), "float32"),
])
def test_rejects_bad_out_buffers(buf, message):
    with pytest.raises(ValueError, match=message):
        to_nchw(solid((1, 2, 3)), (8, 8), "RGB", out=buf)


# --------------------------------------------------------------------------
# equivalence with the implementation the benchmarks measure
# --------------------------------------------------------------------------

def test_matches_the_reference_pipeline_bit_for_bit():
    """Same output as the hand-written version in benchmarks/perf_suite.py."""
    rng = np.random.default_rng(20260805)
    src = rng.integers(0, 256, (1080, 1920, 4), dtype=np.uint8)   # BGRA
    out_w = out_h = 640

    ys = (np.arange(out_h) * 1080 // out_h).clip(0, 1079)
    xs = (np.arange(out_w) * 1920 // out_w).clip(0, 1919)
    sampled = src[np.ix_(ys, xs)].astype(np.float32)
    sampled /= 255.0
    b, g, r = sampled[..., 0], sampled[..., 1], sampled[..., 2]
    reference = np.ascontiguousarray(np.stack((r, g, b), axis=0)[None])

    assert np.array_equal(to_nchw(src, (out_w, out_h), "BGRA"), reference)
