"""NHWC float output from `GpuConverter` (ROADMAP § 7.2).

**The reference is the NCHW output, transposed, and the bar is exact.** Both
layouts run the same `shade(fetch(...))` per pixel and differ only in where
the bytes go, so NHWC must equal ``nchw.transpose(0, 2, 3, 1)`` bit for bit.
NCHW is itself pinned against captured bytes and the old preprocessor by
``test_gpu_converter.py``.

**Why exact equality is enough to catch a packing error.** FP16 NHWC packs
two pixels' six halves into three 32-bit stores, ``[a0 a1] [a2 b0] [b1 b2]``.
A swap inside that — the likeliest mistake — moves a channel value into a
neighbouring pixel or channel, which changes the array on any frame whose
adjacent pixels or channels differ. The fixture checks that this one does.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

from test_gpu_converter_crop import busy_crop

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
def busy(live_frame):
    """A crop over varied content, so neighbouring values actually differ."""
    surface = rapidshot.GpuConverter(
        live_frame, (live_frame.width, live_frame.height),
        dtype="uint8", layout="nhwc", sampling="nearest",
    ).process(live_frame).numpy()[0]
    crop = busy_crop(surface, 128, 96)
    rgb = surface[crop[1]:crop[3], crop[0]:crop[2], :3]
    if np.array_equal(rgb[..., 0], rgb[..., 2]) or np.array_equal(rgb[:, 1:], rgb[:, :-1]):
        pytest.skip("content too uniform to expose channel or pixel transposition")
    return crop


def convert(frame, size, layout, crop=None, regions=None, **kwargs):
    converter = rapidshot.GpuConverter(frame, size, layout=layout, **kwargs)
    return converter, converter.process(frame, crop=crop, regions=regions).numpy()


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
@pytest.mark.parametrize(
    "options", [dict(), dict(bgr=True), dict(normalize=False)], ids=["rgb", "bgr", "0-255"]
)
def test_nhwc_is_exactly_nchw_transposed(live_frame, busy, dtype, sampling, options):
    kwargs = dict(dtype=dtype, sampling=sampling, crop=busy, **options)
    _, nchw = convert(live_frame, (64, 48), "nchw", **kwargs)
    converter, nhwc = convert(live_frame, (64, 48), "nhwc", **kwargs)

    assert converter.layout == "nhwc"
    assert nhwc.shape == (1, 48, 64, 3)
    assert nhwc.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(nhwc, nchw.transpose(0, 2, 3, 1))


@pytest.mark.parametrize("size", [(2, 2), (2, 1), (62, 30), (640, 360)])
@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_sizes_at_the_edges_of_the_pair_packing(live_frame, busy, size, dtype):
    """2x1 is a single pair; 62 wide is even but not a multiple of four or
    eight, so the last thread in each row and the dispatch rounding both
    matter."""
    kwargs = dict(dtype=dtype, sampling="nearest", crop=busy)
    _, nchw = convert(live_frame, size, "nchw", **kwargs)
    _, nhwc = convert(live_frame, size, "nhwc", **kwargs)
    np.testing.assert_array_equal(nhwc, nchw.transpose(0, 2, 3, 1))


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_nhwc_batch_matches_nchw_batch(live_frame, busy, dtype):
    regions = [busy, (0, 0, 200, 150), (live_frame.width - 97, 3, live_frame.width - 1, 75)]
    kwargs = dict(dtype=dtype, sampling="bilinear", batch=4, regions=regions)
    _, nchw = convert(live_frame, (64, 48), "nchw", **kwargs)
    converter, nhwc = convert(live_frame, (64, 48), "nhwc", **kwargs)

    assert nhwc.shape == (3, 48, 64, 3)
    np.testing.assert_array_equal(nhwc, nchw.transpose(0, 2, 3, 1))
    assert converter.output_byte_size == 4 * 64 * 48 * 3 * np.dtype(dtype).itemsize


def test_hwc_is_accepted_as_nhwc(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (32, 32), layout="hwc")
    assert converter.layout == "nhwc"
    assert converter.shape == (1, 32, 32, 3)


def test_odd_width_float16_nhwc_is_refused(live_frame):
    """Same packing constraint as FP16 NCHW: a pair must not straddle a row."""
    with pytest.raises(Exception, match="even width"):
        rapidshot.GpuConverter(live_frame, (33, 32), dtype="float16", layout="nhwc")


def test_odd_width_float32_nhwc_is_fine(live_frame, busy):
    kwargs = dict(dtype="float32", sampling="nearest", crop=busy)
    _, nchw = convert(live_frame, (33, 17), "nchw", **kwargs)
    _, nhwc = convert(live_frame, (33, 17), "nhwc", **kwargs)
    np.testing.assert_array_equal(nhwc, nchw.transpose(0, 2, 3, 1))


def test_native_layer_refuses_a_layout_its_dtype_does_not_have(live_frame):
    """Beyond Python's check: bgra8 has one layout, and a request for another
    must not be ignored."""
    with pytest.raises(ValueError, match="layout"):
        native.require().GpuConverter12(
            native._texture_address(live_frame), 16, 16, dtype="bgra8", layout="nchw"
        )


@pytest.mark.parametrize(
    "dtype,shape",
    [("float32", (1, 3, 16, 16)), ("float16", (1, 3, 16, 16)),
     ("bgra8", (1, 16, 16, 4)), ("nv12", (24, 16))],
)
def test_native_layer_without_layout_keeps_each_dtypes_own(live_frame, dtype, shape):
    """Callers that predate `layout` pass none. Found by this exact break: a
    default of "nchw" refused every `dtype="bgra8"` call that omitted it."""
    impl = native.require().GpuConverter12(
        native._texture_address(live_frame), 16, 16, dtype=dtype
    )
    assert tuple(impl.shape) == shape
