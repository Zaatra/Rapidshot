"""NV12 / P010 output from `GpuConverter` (ROADMAP § 7.2).

**What the reference is built from.** The converter's own float32 output of the
same held frame, at the same size and sampling. That fixes the RGB the YUV
kernel sees, so a mismatch here is the colour conversion or the plane layout,
never the resampling — which `tests/test_gpu_converter.py` already pins.

**Why two independent checks rather than one.** The CPU reference below
re-implements the forward matrix, and a coefficient typed wrong in both places
would pass it. So the decode test uses the *published* inverse coefficients
(ITU-R BT.709 / BT.601), which nothing in the kernel shares.

**Tolerance, stated rather than hidden.** The GPU computes in float32 and
rounds; ties can land one code either side of a float64 reference. So each
comparison allows one code *and* requires that at least 99% of samples match
exactly — a systematic off-by-one (a 15 instead of a 16) fails the second
condition while staying inside the first.

**Not verified here:** an `R10G10B10A2_UNORM` source (this is an SDR desktop),
and the `R16G16B16A16_FLOAT` refusal, which needs an HDR desktop to reach.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

OUT = 128

COEFFS = {"bt709": (0.2126, 0.0722), "bt601": (0.299, 0.114)}
# Published inverses, written from the standards, not derived from COEFFS.
INVERSE = {
    "bt709": dict(r_pr=1.5748, g_pb=-0.1873, g_pr=-0.4681, b_pb=1.8556),
    "bt601": dict(r_pr=1.402, g_pb=-0.344136, g_pr=-0.714136, b_pb=1.772),
}


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


def rgb_of(frame, w, h, sampling):
    """(H, W, 3) float32 RGB in 0..1 — exactly what the YUV kernel fetches."""
    t = rapidshot.GpuConverter(
        frame, (w, h), dtype="float32", sampling=sampling
    ).process(frame).numpy()
    return np.clip(t[0].transpose(1, 2, 0).astype(np.float64), 0.0, 1.0)


def require_colour(rgb):
    """Skip on a grey frame: Cb = Cr = neutral hides every matrix error."""
    if np.abs(rgb[..., 0] - rgb[..., 2]).max() < 0.2:
        pytest.skip("captured frame has too little colour to test chroma")


def round_half_up(v, maxcode):
    return np.clip(np.floor(v + 0.5), 0, maxcode)


def yuv_reference(rgb, bits, matrix="bt709", full_range=False):
    """4:2:0 planar reference with the layout NV12/P010 define."""
    kr, kb = COEFFS[matrix]
    maxcode = (1 << bits) - 1
    ds = 1 << (bits - 8)

    def luma(c):
        return kr * c[..., 0] + (1 - kr - kb) * c[..., 1] + kb * c[..., 2]

    h, w = rgb.shape[:2]
    y = luma(rgb)
    block = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    yb = luma(block)
    pb = (block[..., 2] - yb) / (2 * (1 - kb))
    pr = (block[..., 0] - yb) / (2 * (1 - kr))

    if full_range:
        mid = (maxcode + 1) / 2
        yc = round_half_up(y * maxcode, maxcode)
        cb = round_half_up(mid + pb * maxcode, maxcode)
        cr = round_half_up(mid + pr * maxcode, maxcode)
    else:
        yc = round_half_up((16 + 219 * y) * ds, maxcode)
        cb = round_half_up((128 + 224 * pb) * ds, maxcode)
        cr = round_half_up((128 + 224 * pr) * ds, maxcode)

    chroma = np.stack([cb, cr], axis=-1).reshape(h // 2, w)
    planes = np.concatenate([yc, chroma], axis=0)
    if bits == 8:
        return planes.astype(np.uint8)
    return (planes.astype(np.uint16) << 6)


def assert_codes_match(got, expected, shift=0, min_exact=0.99):
    g = got.astype(np.int64) >> shift
    e = expected.astype(np.int64) >> shift
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    diff = np.abs(g - e)
    assert diff.max() <= 1, f"max code difference {diff.max()}"
    exact = (diff == 0).mean()
    assert exact >= min_exact, f"only {exact:.1%} of samples match exactly"


# --------------------------------------------------------------------------
# against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_nv12_matches_cpu_reference(live_frame, sampling):
    rgb = rgb_of(live_frame, OUT, OUT, sampling)
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="nv12", sampling=sampling
    )
    got = converter.process(live_frame).numpy()
    assert_codes_match(got, yuv_reference(rgb, 8))


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_p010_matches_cpu_reference(live_frame, sampling):
    rgb = rgb_of(live_frame, OUT, OUT, sampling)
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="p010", sampling=sampling
    )
    got = converter.process(live_frame).numpy()
    assert_codes_match(got, yuv_reference(rgb, 10), shift=6)


def test_p010_low_six_bits_are_zero(live_frame):
    """P010 means the value in the *high* ten bits. A kernel storing the raw
    10-bit code would decode 64x too dark and still look like a P010 buffer."""
    got = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="p010"
    ).process(live_frame).numpy()
    assert got.dtype == np.uint16
    assert not np.any(got & 0x3F)
    assert got.max() > 0x3F


@pytest.mark.parametrize("matrix", ["bt709", "bt601"])
@pytest.mark.parametrize("full_range", [False, True])
def test_matrix_and_range_options_match_reference(live_frame, matrix, full_range):
    rgb = rgb_of(live_frame, OUT, OUT, "nearest")
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="nv12", sampling="nearest",
        matrix=matrix, full_range=full_range,
    )
    assert converter.matrix == matrix
    assert converter.full_range is full_range
    got = converter.process(live_frame).numpy()
    assert_codes_match(got, yuv_reference(rgb, 8, matrix, full_range))


# --------------------------------------------------------------------------
# independent of the reference's own coefficients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("matrix", ["bt709", "bt601"])
@pytest.mark.parametrize("pixel_format,bits", [("nv12", 8), ("p010", 10)])
def test_decodes_back_with_the_published_inverse(live_frame, matrix, pixel_format, bits):
    """Decode with ITU-published inverse coefficients and recover the RGB.

    Chroma is per 2x2 block, so compare block averages: the matrix is linear,
    so the block-averaged luma plus the block's chroma decodes to the block's
    average RGB, up to quantisation. A swapped Cb/Cr, a wrong Kr/Kb, a chroma
    plane shifted by a row, or limited range labelled full all miss by far
    more than the tolerance on a frame with colour in it.
    """
    rgb = rgb_of(live_frame, OUT, OUT, "nearest")
    require_colour(rgb)
    got = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format=pixel_format,
        sampling="nearest", matrix=matrix,
    ).process(live_frame).numpy().astype(np.float64)
    if bits == 10:
        got = got / 64.0
    ds = 1 << (bits - 8)

    h, w = OUT, OUT
    y = (got[:h] / ds - 16) / 219
    chroma = got[h:].reshape(h // 2, w // 2, 2)
    pb = (chroma[..., 0] / ds - 128) / 224
    pr = (chroma[..., 1] / ds - 128) / 224
    yb = y.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    inv = INVERSE[matrix]
    decoded = np.stack(
        [
            yb + inv["r_pr"] * pr,
            yb + inv["g_pb"] * pb + inv["g_pr"] * pr,
            yb + inv["b_pb"] * pb,
        ],
        axis=-1,
    )
    expected = rgb.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    tolerance = 0.012 if bits == 8 else 0.004
    np.testing.assert_allclose(decoded, expected, atol=tolerance)


def test_matrices_actually_differ(live_frame):
    """A `#define` that never reached the compiler would make them identical."""
    rgb = rgb_of(live_frame, OUT, OUT, "nearest")
    require_colour(rgb)
    a, b = (
        rapidshot.GpuConverter(
            live_frame, (OUT, OUT), pixel_format="nv12", sampling="nearest", matrix=m
        ).process(live_frame).numpy()
        for m in ("bt709", "bt601")
    )
    assert not np.array_equal(a, b)


def test_nv12_and_p010_agree(live_frame):
    """Same computation at two depths: P010 is NV12 x4 within rounding."""
    nv12 = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="nv12"
    ).process(live_frame).numpy().astype(np.int64)
    p010 = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="p010"
    ).process(live_frame).numpy().astype(np.int64) >> 6
    assert np.abs(p010 - nv12 * 4).max() <= 2


# --------------------------------------------------------------------------
# layout at the edges of the dword packing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("w,h", [(6, 2), (2, 2), (10, 6), (640, 360)])
@pytest.mark.parametrize("pixel_format,bits", [("nv12", 8), ("p010", 10)])
def test_sizes_that_do_not_fill_whole_dwords(live_frame, w, h, pixel_format, bits):
    """The kernel writes 32 bits at a time and NV12 samples are 8.

    6x2 has a luma row that is not a multiple of four and a chroma plane of
    six bytes, so its last dword is half padding; 2x2 has a single chroma
    sample. A packing error puts bytes in the wrong row or past the payload,
    and the shape would still be right.
    """
    converter = rapidshot.GpuConverter(
        live_frame, (w, h), pixel_format=pixel_format, sampling="nearest"
    )
    bytes_per_sample = 1 if bits == 8 else 2
    assert converter.output_byte_size == w * h * 3 // 2 * bytes_per_sample
    got = converter.process(live_frame).numpy()
    assert got.shape == (h * 3 // 2, w)
    rgb = rgb_of(live_frame, w, h, "nearest")
    # A 99%-exact rule means nothing over six samples, so tiny sizes get the
    # one-code bound alone.
    assert_codes_match(
        got, yuv_reference(rgb, bits),
        shift=0 if bits == 8 else 6,
        min_exact=0.99 if w * h >= 64 else 0.0,
    )


def test_describes_itself(live_frame):
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="nv12")
    assert converter.pixel_format == "nv12"
    assert converter.dtype == "uint8"
    assert converter.shape == (OUT * 3 // 2, OUT)
    assert converter.output_byte_size == OUT * OUT * 3 // 2
    tensor = converter.process(live_frame)
    assert tensor.nbytes == OUT * OUT * 3 // 2

    p010 = rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="p010")
    assert p010.dtype == "uint16"
    assert p010.output_byte_size == OUT * OUT * 3

    tensor_converter = rapidshot.GpuConverter(live_frame, (OUT, OUT))
    assert tensor_converter.pixel_format is None


def test_payload_is_smaller_than_bgra(live_frame):
    """NV12 is 1.5 bytes/pixel against BGRA's 4 — the encoder's input and
    also the cheapest representation on a cross-adapter bus."""
    nv12 = rapidshot.GpuConverter(live_frame, (640, 640), pixel_format="nv12")
    bgra = rapidshot.GpuConverter(live_frame, (640, 640), dtype="uint8", layout="nhwc")
    assert nv12.output_byte_size * 8 == bgra.output_byte_size * 3


def test_nv12_crosses_adapters_byte_exact(live_frame):
    """Convert-first with the encoder payload. Destination may be WARP here."""
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="nv12")
    try:
        transfer = rapidshot.TensorTransfer(converter)
    except RuntimeError as error:
        if "only one adapter" in str(error):
            pytest.skip("single-adapter machine; nothing to transfer to")
        raise
    produced = converter.process(live_frame).numpy()
    transfer.transfer()
    np.testing.assert_array_equal(transfer.read_back_destination(), produced)


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(OUT + 1, OUT), (OUT, OUT + 1)])
@pytest.mark.parametrize("pixel_format", ["nv12", "p010"])
def test_odd_dimensions_are_refused(live_frame, size, pixel_format):
    with pytest.raises(ValueError, match="even"):
        rapidshot.GpuConverter(live_frame, size, pixel_format=pixel_format)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(dtype="float32"),
        dict(dtype="uint8"),
        dict(layout="nchw"),
        dict(normalize=False),
        dict(bgr=True),
    ],
)
def test_tensor_arguments_are_refused_with_pixel_format(live_frame, kwargs):
    """Ignoring them would let a caller believe the output honours them."""
    with pytest.raises(ValueError, match="does not apply"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="nv12", **kwargs)


def test_unknown_pixel_format_is_refused(live_frame):
    with pytest.raises(ValueError, match="pixel_format"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="yuy2")


def test_unknown_matrix_is_refused(live_frame):
    with pytest.raises(ValueError, match="matrix"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="nv12", matrix="bt2020")
