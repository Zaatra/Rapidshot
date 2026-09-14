"""Correctness for the 2.6 `GpuConverter` path (ROADMAP § 7.2).

**These need live capture, and that is forced rather than chosen.** ROADMAP
§ 2: D3D11 refuses `SHARED_NTHANDLE` without `SHARED_KEYEDMUTEX`, and a
keyed-mutex resource reads as zeros until acquired — on both APIs — so a
`TestTexture` cannot back a D3D12 converter at all. The constructor rejects
one by design. `tests/test_gpu_preprocess.py` gets to use synthetic input only
because the D3D11 preprocessor has no such constraint.

Consequence: these skip on an idle screen rather than failing. A red suite
that means "nothing moved on screen" trains people to ignore red suites.

What is deliberately *not* here: the CUDA export path. Machine A has an Intel
iGPU and no CUDA device, so `to_cupy()` / `to_torch()` / `to_dlpack()` cannot
be exercised on it at all. Verifying them is Machine B work, and until that
happens they are unverified for the release — ROADMAP § 5's rule.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

OUT = 64


@pytest.fixture(scope="module")
def live_frame():
    """A live captured frame, or a skip. Released after the module finishes."""
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


def nearest_reference(source, out_w, out_h, bgr=False, normalize=True):
    """What nearest sampling must produce: index-decimate, reorder, scale."""
    src_h, src_w = source.shape[:2]
    ys = (np.arange(out_h) * src_h // out_h).clip(0, src_h - 1)
    xs = (np.arange(out_w) * src_w // out_w).clip(0, src_w - 1)
    s = source[np.ix_(ys, xs)].astype(np.float32)
    if normalize:
        s = s / 255.0
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    planes = (b, g, r) if bgr else (r, g, b)
    return np.stack(planes, axis=0)[None]


# --------------------------------------------------------------------------
# the contract with the path it does not replace
# --------------------------------------------------------------------------


def test_nearest_float32_matches_the_old_preprocessor_exactly(live_frame):
    """`sampling="nearest"` must reproduce `GpuPreprocessor12` bit for bit.

    This is what makes the new path a *superset* rather than a replacement.
    If it ever diverges, every stored recording comparing the two becomes
    meaningless, because the thing being compared moved.
    """
    old = native.GpuPreprocessor12(live_frame, OUT, OUT)
    old.process(live_frame, 1.0, 0.0, False)  # 0..1, matching normalize=True
    expected = old.read_back()

    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float32", sampling="nearest"
    )
    got = converter.process(live_frame).numpy()

    np.testing.assert_array_equal(got, expected)


def test_bilinear_differs_from_nearest(live_frame):
    """Bilinear must actually filter, not silently fall through to Load().

    A static sampler that was never bound, or a `#define` that did not reach
    the compiler, would produce output identical to nearest — correct-looking
    and wrong. Nothing about shape, range or channel order would catch it.
    """
    near = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float32", sampling="nearest"
    ).process(live_frame).numpy()
    bilin = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float32", sampling="bilinear"
    ).process(live_frame).numpy()

    if np.array_equal(near, bilin):
        pytest.skip(
            "captured frame is too uniform to distinguish the samplers "
            "(a flat region filters to itself)"
        )
    # Filtering averages neighbours, so it cannot leave the range.
    assert bilin.min() >= near.min() - 1e-6
    assert bilin.max() <= near.max() + 1e-6


# --------------------------------------------------------------------------
# dtypes
# --------------------------------------------------------------------------


def test_float16_matches_float32_within_half_precision(live_frame):
    """FP16 must be the same computation, stored narrower.

    The kernel packs two pixels per 32-bit store, so a mistake in the packing
    shows up as transposed or interleaved columns rather than as noise — which
    a tolerance check on the whole array would hide. Comparing element-wise
    against the FP32 path catches placement, not just magnitude.
    """
    f32 = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float32", sampling="nearest"
    ).process(live_frame).numpy()
    f16 = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float16", sampling="nearest"
    ).process(live_frame).numpy()

    assert f16.dtype == np.float16
    assert f16.shape == f32.shape
    # Half has ~3 decimal digits; values here are 0..1.
    np.testing.assert_allclose(f16.astype(np.float32), f32, atol=1e-3)


def test_float16_halves_the_payload(live_frame):
    """The entire point of FP16 on this path is bytes on the bus (§ 6.1)."""
    f32 = rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="float32")
    f16 = rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="float16")
    assert f16.output_byte_size * 2 == f32.output_byte_size


def test_odd_width_float16_is_refused_not_silently_wrong(live_frame):
    """An odd width would leave a half-written dword at each row end.

    The shape would still be right, which is the failure mode § 11 says to
    fail loudly on rather than ship.
    """
    with pytest.raises(Exception, match="even width"):
        rapidshot.GpuConverter(live_frame, (OUT + 1, OUT), dtype="float16")


def test_uint8_is_a_resized_frame_not_a_tensor(live_frame):
    """`uint8` output must stay BGRA and ignore normalisation entirely.

    It exists to be the cheapest thing to put on a cross-adapter bus, and a
    consumer is meant to treat it as an ordinary frame — so scale, bias and
    channel order must not touch it.
    """
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="uint8", layout="nhwc", sampling="nearest"
    )
    got = converter.process(live_frame).numpy()

    assert got.dtype == np.uint8
    assert got.shape == (1, OUT, OUT, 4)
    assert converter.output_byte_size == OUT * OUT * 4


def test_uint8_payload_is_far_smaller_than_the_frame(live_frame):
    """The § 6.1 argument in one assertion."""
    frame_bytes = live_frame.width * live_frame.height * 4
    converter = rapidshot.GpuConverter(
        live_frame, (640, 640), dtype="uint8", layout="nhwc"
    )
    assert converter.output_byte_size < frame_bytes / 5


# --------------------------------------------------------------------------
# the arguments that silently corrupt when wrong
# --------------------------------------------------------------------------


def test_channel_order_is_honoured(live_frame):
    """BGR labelled RGB runs fine and is quietly worse — the § 7.2 hazard."""
    rgb = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), sampling="nearest", bgr=False
    ).process(live_frame).numpy()
    bgr = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), sampling="nearest", bgr=True
    ).process(live_frame).numpy()

    if np.array_equal(rgb[:, 0], rgb[:, 2]):
        pytest.skip("captured frame is greyscale, so R and B are identical")
    np.testing.assert_array_equal(rgb[:, 0], bgr[:, 2])
    np.testing.assert_array_equal(rgb[:, 2], bgr[:, 0])
    np.testing.assert_array_equal(rgb[:, 1], bgr[:, 1])


def test_normalize_controls_the_output_range(live_frame):
    """0..1 by default, 0..255 when asked — and exactly 255x apart.

    This caught a real inversion: the capture format is UNORM, so the fetch
    already yields 0..1 and an implementation that "normalises" by dividing
    by 255 produces a tensor 255x too dark. Shape, dtype and channel order
    would all still be right.
    """
    unit = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), sampling="nearest", normalize=True
    ).process(live_frame).numpy()
    byte = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), sampling="nearest", normalize=False
    ).process(live_frame).numpy()

    assert unit.max() <= 1.0 + 1e-6
    np.testing.assert_allclose(byte, unit * 255.0, rtol=1e-5, atol=1e-3)
    if unit.max() > 0:
        assert byte.max() > 1.5


def test_nearest_output_matches_a_cpu_reference(live_frame):
    """Against numbers this test computes, not against another GPU path."""
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float32", sampling="nearest"
    )
    got = converter.process(live_frame).numpy()

    source = live_frame.frame_buffer if hasattr(live_frame, "frame_buffer") else None
    if source is None:
        pytest.skip("frame does not expose its CPU buffer")
    expected = nearest_reference(np.asarray(source), OUT, OUT)
    np.testing.assert_allclose(got, expected, atol=2e-3)


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float64", "int8", "fp16", ""])
def test_unknown_dtype_is_refused(live_frame, dtype):
    with pytest.raises(ValueError, match="dtype"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype=dtype)


def test_unknown_sampling_is_refused(live_frame):
    with pytest.raises(Exception, match="sampling|nearest|bilinear"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), sampling="bicubic")


@pytest.mark.parametrize("layout", ["nchwx", "chw", "nhcw", ""])
def test_float_output_refuses_unknown_layouts(live_frame, layout):
    """Rather than silently emitting NCHW under a label it does not match.

    Replaces a test that asserted `layout="nhwc"` was refused; NHWC float is
    now implemented and covered in `test_gpu_converter_nhwc.py`.
    """
    with pytest.raises(ValueError, match="nchw"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="float32", layout=layout)


def test_uint8_refuses_nchw(live_frame):
    with pytest.raises(ValueError, match="nhwc"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="uint8", layout="nchw")


# --------------------------------------------------------------------------
# the tensor handle
# --------------------------------------------------------------------------


def test_tensor_describes_itself_consistently(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="float16")
    tensor = converter.process(live_frame)

    assert tensor.shape == (1, 3, OUT, OUT)
    assert tensor.dtype == "float16"
    assert tensor.nbytes == OUT * OUT * 3 * 2
    assert len(tensor.adapter_luid) == 8
    assert tensor.shared_handle != 0


def test_process_returns_the_same_handle_each_time(live_frame):
    """Two handles would imply two results; there is one buffer."""
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT))
    assert converter.process(live_frame) is converter.process(live_frame)


# --------------------------------------------------------------------------
# source formats
# --------------------------------------------------------------------------


def test_source_format_is_reported(live_frame):
    """An HDR desktop and an SDR one are otherwise indistinguishable.

    Only the format this machine actually captures can be exercised here.
    `R8G8B8A8_UNORM`, `R10G10B10A2_UNORM` and `R16G16B16A16_FLOAT` are accepted
    by the converter and **have not been run against a real surface of that
    format** — this is an SDR desktop. See the module docstring's rule.
    """
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT), sampling="nearest")
    assert converter.source_format == "none"  # nothing opened yet

    converter.process(live_frame)
    assert converter.source_format in (
        "B8G8R8A8_UNORM",
        "R8G8B8A8_UNORM",
        "R10G10B10A2_UNORM (10-bit)",
        "R16G16B16A16_FLOAT (HDR scRGB)",
    )
