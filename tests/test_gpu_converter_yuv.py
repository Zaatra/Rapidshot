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

**On an HDR desktop** NV12 and P010 refuse the frame unless given a tone map,
so the conversion tests skip and :func:`test_an_hdr_frame_is_refused_without_a_tone_map`
checks the refusal on the live surface instead. **Not verified here:** an
`R10G10B10A2_UNORM` source, and NV12's tone map on a live HDR frame.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

from _hdr_desktop import YUV_WITHOUT_TONE_MAP, is_hdr, skip_on_an_hdr_desktop

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


def left_sited(plane):
    """Downsample to 4:2:0 positions the way H.264 assumes them by default.

    Horizontally a [1 2 1] / 4 filter centred on each even column (the column
    left of 0 repeats column 0); vertically the mean of the two rows. Works on
    (H, W) and (H, W, C) arrays alike.
    """
    left = np.concatenate([plane[:, :1], plane[:, :-1]], axis=1)
    right = plane[:, 1:]
    centre = plane[:, 0::2]
    rows = 0.25 * left[:, 0::2] + 0.5 * centre + 0.25 * right[:, 0::2]
    return 0.5 * (rows[0::2] + rows[1::2])


def centre_sited(plane):
    """What the converter did before 2.7.1: each 2x2 block averaged."""
    h, w = plane.shape[:2]
    return plane.reshape(h // 2, 2, w // 2, 2, *plane.shape[2:]).mean(axis=(1, 3))


def yuv_reference(rgb, bits, matrix="bt709", full_range=False, siting=left_sited,
                  return_ties=False):
    """4:2:0 planar reference with the layout NV12/P010 define.

    With ``return_ties`` it also returns where the unrounded value sits on an
    exact .5, which float32 on the GPU may round either way. Full range makes
    those structural: a saturated colour has Cb or Cr of exactly +-0.5, so
    ``mid +- 127.5`` is a tie, and a screen full of pure colour is mostly ties.
    """
    kr, kb = COEFFS[matrix]
    maxcode = (1 << bits) - 1
    ds = 1 << (bits - 8)

    def luma(c):
        return kr * c[..., 0] + (1 - kr - kb) * c[..., 1] + kb * c[..., 2]

    h, w = rgb.shape[:2]
    y = luma(rgb)
    block = siting(rgb)
    yb = luma(block)
    pb = (block[..., 2] - yb) / (2 * (1 - kb))
    pr = (block[..., 0] - yb) / (2 * (1 - kr))

    if full_range:
        mid = (maxcode + 1) / 2
        raw = (y * maxcode, mid + pb * maxcode, mid + pr * maxcode)
    else:
        raw = ((16 + 219 * y) * ds, (128 + 224 * pb) * ds, (128 + 224 * pr) * ds)
    yc, cb, cr = (round_half_up(v, maxcode) for v in raw)

    def interleave(luma_plane, cb_plane, cr_plane):
        chroma = np.stack([cb_plane, cr_plane], axis=-1).reshape(h // 2, w)
        return np.concatenate([luma_plane, chroma], axis=0)

    planes = interleave(yc, cb, cr)
    planes = planes.astype(np.uint8) if bits == 8 else planes.astype(np.uint16) << 6
    if not return_ties:
        return planes
    # Within 1e-4 of a code: the RGB itself arrives in float32 (a saturated 1.0
    # is 0.99999994), so a true tie lands a few 1e-5 off .5, never on it.
    ties = interleave(*(np.abs(v - np.floor(v) - 0.5) < 1e-4 for v in raw))
    return planes, ties


def assert_codes_match(got, expected, shift=0, min_exact=0.99, ties=None):
    """Every sample within one code; ``min_exact`` of them exact.

    ``ties`` marks samples whose true value is an exact .5: those may round
    either way, so they are held to the one-code bound but left out of the
    exact count, which would otherwise depend on how much pure colour is on
    screen rather than on the kernel.
    """
    g = got.astype(np.int64) >> shift
    e = expected.astype(np.int64) >> shift
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    diff = np.abs(g - e)
    assert diff.max() <= 1, f"max code difference {diff.max()}"
    if ties is not None:
        diff = diff[~ties]
    exact = (diff == 0).mean()
    assert exact >= min_exact, f"only {exact:.1%} of samples match exactly"


# --------------------------------------------------------------------------
# against the reference
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_nv12_matches_cpu_reference(live_frame, sampling):
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    rgb = rgb_of(live_frame, OUT, OUT, sampling)
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="nv12", sampling=sampling
    )
    got = converter.process(live_frame).numpy()
    assert_codes_match(got, yuv_reference(rgb, 8))


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
def test_p010_matches_cpu_reference(live_frame, sampling):
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    rgb = rgb_of(live_frame, OUT, OUT, sampling)
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="p010", sampling=sampling
    )
    got = converter.process(live_frame).numpy()
    assert_codes_match(got, yuv_reference(rgb, 10), shift=6)


def test_p010_low_six_bits_are_zero(live_frame):
    """P010 means the value in the *high* ten bits. A kernel storing the raw
    10-bit code would decode 64x too dark and still look like a P010 buffer."""
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    got = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="p010"
    ).process(live_frame).numpy()
    assert got.dtype == np.uint16
    assert not np.any(got & 0x3F)
    assert got.max() > 0x3F


@pytest.mark.parametrize("matrix", ["bt709", "bt601"])
@pytest.mark.parametrize("full_range", [False, True])
def test_matrix_and_range_options_match_reference(live_frame, matrix, full_range):
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    rgb = rgb_of(live_frame, OUT, OUT, "nearest")
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format="nv12", sampling="nearest",
        matrix=matrix, full_range=full_range,
    )
    assert converter.matrix == matrix
    assert converter.full_range is full_range
    got = converter.process(live_frame).numpy()
    expected, ties = yuv_reference(rgb, 8, matrix, full_range, return_ties=True)
    assert_codes_match(got, expected, ties=ties)


# --------------------------------------------------------------------------
# independent of the reference's own coefficients
# --------------------------------------------------------------------------


@pytest.mark.parametrize("matrix", ["bt709", "bt601"])
@pytest.mark.parametrize("pixel_format,bits", [("nv12", 8), ("p010", 10)])
def test_decodes_back_with_the_published_inverse(live_frame, matrix, pixel_format, bits):
    """Decode with ITU-published inverse coefficients and recover the RGB.

    Chroma is one sample per 2x2 block, so compare at chroma positions: the
    matrix is linear, so luma filtered the way chroma was sited, plus the
    chroma, decodes to the RGB filtered the same way, up to quantisation. A
    swapped Cb/Cr, a wrong Kr/Kb, a chroma plane shifted by a row, or limited
    range labelled full all miss by far more than the tolerance on a frame with
    colour in it.
    """
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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
    yb = left_sited(y)

    inv = INVERSE[matrix]
    decoded = np.stack(
        [
            yb + inv["r_pr"] * pr,
            yb + inv["g_pb"] * pb + inv["g_pr"] * pr,
            yb + inv["b_pb"] * pb,
        ],
        axis=-1,
    )
    expected = left_sited(rgb)
    tolerance = 0.012 if bits == 8 else 0.004
    np.testing.assert_allclose(decoded, expected, atol=tolerance)


@pytest.mark.parametrize("pixel_format,bits", [("nv12", 8), ("p010", 10)])
def test_chroma_is_left_sited(live_frame, pixel_format, bits):
    """H.264 and HEVC read chroma as left-sited unless told otherwise.

    The reference tests above would pass a centre-sited kernel against a
    centre-sited reference, so this one names the siting: the chroma plane
    must match the left-sited reference and must *not* match the centre-sited
    one. Before 2.7.1 the kernel was centre-sited and fails here.
    """
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    rgb = rgb_of(live_frame, OUT, OUT, "nearest")
    require_colour(rgb)
    left = yuv_reference(rgb, bits, siting=left_sited)[OUT:]
    centre = yuv_reference(rgb, bits, siting=centre_sited)[OUT:]
    shift = 0 if bits == 8 else 6
    differing = (left.astype(np.int64) >> shift) != (centre.astype(np.int64) >> shift)
    if differing.mean() < 0.02:
        pytest.skip("frame has too few horizontal colour edges to tell the sitings apart")
    got = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), pixel_format=pixel_format, sampling="nearest"
    ).process(live_frame).numpy()[OUT:]
    assert_codes_match(got, left, shift=shift)
    g = got.astype(np.int64) >> shift
    c = centre.astype(np.int64) >> shift
    assert (g[differing] != c[differing]).mean() > 0.5, "chroma still matches the centre-sited reference"


def test_matrices_actually_differ(live_frame):
    """A `#define` that never reached the compiler would make them identical."""
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
    nv12 = rapidshot.GpuConverter(live_frame, (640, 640), pixel_format="nv12")
    bgra = rapidshot.GpuConverter(live_frame, (640, 640), dtype="uint8", layout="nhwc")
    assert nv12.output_byte_size * 8 == bgra.output_byte_size * 3


def test_nv12_crosses_adapters_byte_exact(live_frame):
    """Convert-first with the encoder payload. Destination may be WARP here."""
    skip_on_an_hdr_desktop(live_frame, YUV_WITHOUT_TONE_MAP)
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


@pytest.mark.parametrize("pixel_format", ["nv12", "p010"])
def test_an_hdr_frame_is_refused_without_a_tone_map(live_frame, pixel_format):
    """The HDR half of the skips above, on the live surface: a Y'CbCr matrix
    expects gamma-encoded values, so linear scRGB is refused at construction
    rather than encoded into wrong colours -- the 10-bit case once went
    through with no error at all. The default tone map is ``"none"`` here."""
    if not is_hdr(live_frame):
        pytest.skip("the captured display is SDR; this is the HDR half")
    with pytest.raises(ValueError, match="NV12/P010 cannot encode as it is"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format=pixel_format)


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


@pytest.mark.parametrize("pixel_format", ["nv12", "p010"])
def test_a_centre_sited_extension_is_refused(live_frame, monkeypatch, pixel_format):
    """0.3.0 builds export the same converter and write centre-sited chroma.

    Accepting one would hand an encoder bytes the docs say it will not get.
    """
    monkeypatch.setattr(native, "extension_version", lambda: "0.3.0")
    with pytest.raises(RuntimeError, match=r"rapidshot-native >= 0\.3\.1"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format=pixel_format)


def test_unknown_matrix_is_refused(live_frame):
    with pytest.raises(ValueError, match="matrix"):
        rapidshot.GpuConverter(live_frame, (OUT, OUT), pixel_format="nv12", matrix="bt2020")
