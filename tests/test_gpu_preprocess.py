"""Correctness tests for the GPU preprocessing shader.

These use a synthetic texture with contents this test chose, so the shader's
output is compared against an exact reference. Verifying against live capture
was tried first and proved unreliable — Desktop Duplication only reports changed
content, so an idle screen yields no frames at all, and two captures taken
moments apart are not guaranteed identical.

A fast tensor that is subtly wrong is worse than no tensor: it silently corrupts
whatever model consumes it. Hence the emphasis on exactness here.
"""

import numpy as np
import pytest

from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


def make_pattern(width, height, seed=7):
    """Deterministic BGRA image with all channels distinct."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (height, width, 4), dtype=np.uint8)


def texture_from(pattern):
    ext = native.require()
    h, w = pattern.shape[:2]
    return ext.TestTexture(w, h, np.ascontiguousarray(pattern).tobytes())


def preprocess(pattern, out_w, out_h, scale=1.0, bias=0.0, bgr=False):
    """Run the shader and return the tensor as (1, 3, H, W)."""
    ext = native.require()
    tex = texture_from(pattern)
    pre = ext.GpuPreprocessor(tex.pointer, out_w, out_h)
    pre.process(tex.pointer, scale, bias, bgr)
    # read_back returns raw bytes, not a list of floats — a list cost one Python
    # object per element and swamped the benchmark that priced it.
    flat = np.frombuffer(pre.read_back(), dtype=np.float32)
    return flat.reshape(1, 3, out_h, out_w)


def reference(pattern, out_w, out_h, scale=1.0, bias=0.0, bgr=False):
    """Expected result: nearest-neighbour resize, normalise, NCHW."""
    src_h, src_w = pattern.shape[:2]
    ys = (np.arange(out_h) * src_h // out_h).clip(0, src_h - 1)
    xs = (np.arange(out_w) * src_w // out_w).clip(0, src_w - 1)
    s = pattern[np.ix_(ys, xs)].astype(np.float32) / 255.0
    b, g, r = s[..., 0], s[..., 1], s[..., 2]
    planes = (b, g, r) if bgr else (r, g, b)
    return np.stack(planes, axis=0)[None] * scale + bias


# --------------------------------------------------------------------------
# channel order — the bug that made this test suite necessary
# --------------------------------------------------------------------------

def test_rgb_channel_order_is_actually_rgb():
    """
    Regression guard for a silent correctness bug.

    The shader originally reversed the channels by hand, on the assumption that
    a BGRA texture presents blue in .x. It does not: the hardware swizzles
    DXGI_FORMAT_B8G8R8A8_UNORM so .x is red. The result was a tensor labelled
    RGB that actually contained BGR — which no test of speed or stability would
    have caught, and which quietly corrupts model input.
    """
    pattern = np.zeros((4, 4, 4), np.uint8)
    pattern[..., 0] = 10    # B
    pattern[..., 1] = 100   # G
    pattern[..., 2] = 200   # R
    pattern[..., 3] = 255   # A

    out = preprocess(pattern, 4, 4)
    assert out[0, 0].mean() == pytest.approx(200 / 255, abs=1e-3), "plane 0 must be RED"
    assert out[0, 1].mean() == pytest.approx(100 / 255, abs=1e-3), "plane 1 must be GREEN"
    assert out[0, 2].mean() == pytest.approx(10 / 255, abs=1e-3), "plane 2 must be BLUE"


def test_bgr_flag_reverses_the_channel_order():
    pattern = np.zeros((4, 4, 4), np.uint8)
    pattern[..., 0] = 10
    pattern[..., 1] = 100
    pattern[..., 2] = 200
    pattern[..., 3] = 255

    out = preprocess(pattern, 4, 4, bgr=True)
    assert out[0, 0].mean() == pytest.approx(10 / 255, abs=1e-3), "plane 0 must be BLUE"
    assert out[0, 2].mean() == pytest.approx(200 / 255, abs=1e-3), "plane 2 must be RED"


# --------------------------------------------------------------------------
# exact agreement with the CPU reference
# --------------------------------------------------------------------------

@pytest.mark.parametrize("size", [(8, 8), (64, 64), (32, 16)])
def test_matches_reference_without_resize(size):
    w, h = size
    pattern = make_pattern(w, h)
    got = preprocess(pattern, w, h)
    want = reference(pattern, w, h)
    assert np.abs(got - want).max() < 1e-5


@pytest.mark.parametrize("out_size", [(4, 4), (16, 16), (40, 24)])
def test_matches_reference_with_downscale(out_size):
    out_w, out_h = out_size
    pattern = make_pattern(64, 64)
    got = preprocess(pattern, out_w, out_h)
    want = reference(pattern, out_w, out_h)
    assert np.abs(got - want).max() < 1e-5


def test_matches_reference_with_upscale():
    pattern = make_pattern(8, 8)
    got = preprocess(pattern, 32, 32)
    want = reference(pattern, 32, 32)
    assert np.abs(got - want).max() < 1e-5


def test_non_square_source_and_target():
    """The 1920x1080 -> 640x640 case, which is the realistic one."""
    pattern = make_pattern(192, 108)
    got = preprocess(pattern, 64, 64)
    want = reference(pattern, 64, 64)
    assert np.abs(got - want).max() < 1e-5


# --------------------------------------------------------------------------
# crop — how a frame's region reaches the shader
# --------------------------------------------------------------------------
#
# A region camera's texture is the whole output; until 2026-09-14 this path
# sized its sampling from the texture and resized the entire monitor. The fix
# is a crop rectangle in texels. These run on a synthetic texture, so unlike
# the live region tests they need no screen and pin exact values.


def preprocess_crop(pattern, out_w, out_h, crop):
    ext = native.require()
    tex = texture_from(pattern)
    pre = ext.GpuPreprocessor(tex.pointer, out_w, out_h)
    pre.process(tex.pointer, 1.0, 0.0, False, crop=crop)
    return np.frombuffer(pre.read_back(), dtype=np.float32).reshape(1, 3, out_h, out_w)


def sliced(pattern, crop):
    x, y, w, h = crop
    return pattern[y:y + h, x:x + w]


@pytest.mark.parametrize(
    "crop,out_size",
    [
        ((5, 3, 20, 12), (20, 12)),    # identity size: the slice itself
        ((7, 9, 40, 30), (16, 12)),    # downscale, odd offsets
        ((11, 13, 6, 4), (24, 16)),    # upscale a small crop
        ((0, 0, 64, 48), (32, 24)),    # anchored at the origin
        ((33, 17, 31, 31), (31, 31)),  # touching the far corner
    ],
)
def test_crop_matches_the_reference_of_the_slice(crop, out_size):
    """Cropping then resizing must equal resizing the slice: same stride, same
    offset. An off-by-one or a stride taken from the texture fails exactly."""
    pattern = make_pattern(64, 48)
    out_w, out_h = out_size
    got = preprocess_crop(pattern, out_w, out_h, crop)
    np.testing.assert_array_equal(got, reference(sliced(pattern, crop), out_w, out_h))


def test_whole_texture_crop_is_bit_identical_to_no_crop():
    """The shader's crop reduces to the old expression; this pins that."""
    pattern = make_pattern(64, 48)
    np.testing.assert_array_equal(
        preprocess_crop(pattern, 40, 24, (0, 0, 64, 48)), preprocess(pattern, 40, 24)
    )


def test_crop_offset_moves_the_output_by_exactly_one_texel():
    pattern = make_pattern(64, 48)
    a = preprocess_crop(pattern, 20, 12, (5, 3, 20, 12))
    b = preprocess_crop(pattern, 20, 12, (6, 3, 20, 12))
    np.testing.assert_array_equal(a[..., 1:], b[..., :-1])


@pytest.mark.parametrize(
    "crop",
    [(60, 0, 8, 8), (0, 44, 8, 8), (0, 0, 0, 8), (0, 0, 8, 0), (64, 48, 1, 1)],
)
def test_crop_outside_the_texture_is_refused_not_clamped(crop):
    with pytest.raises(RuntimeError, match="does not fit"):
        preprocess_crop(make_pattern(64, 48), 8, 8, crop)


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

def test_default_output_is_zero_to_one():
    pattern = make_pattern(16, 16)
    out = preprocess(pattern, 16, 16)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_scale_and_bias_map_to_signed_range():
    pattern = make_pattern(16, 16)
    out = preprocess(pattern, 16, 16, scale=2.0, bias=-1.0)
    want = reference(pattern, 16, 16, scale=2.0, bias=-1.0)
    assert np.abs(out - want).max() < 1e-5
    assert out.min() >= -1.001 and out.max() <= 1.001


def test_extremes_map_exactly():
    """Pure black and pure white must land on 0.0 and 1.0."""
    pattern = np.zeros((2, 2, 4), np.uint8)
    pattern[0, 0] = [0, 0, 0, 255]
    pattern[0, 1] = [255, 255, 255, 255]
    out = preprocess(pattern, 2, 2)
    assert out[0, :, 0, 0].max() == pytest.approx(0.0, abs=1e-6)
    assert out[0, :, 0, 1].min() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# shape and determinism
# --------------------------------------------------------------------------

def test_reported_shape_matches_output():
    ext = native.require()
    tex = texture_from(make_pattern(16, 16))
    pre = ext.GpuPreprocessor(tex.pointer, 24, 12)
    assert tuple(pre.shape) == (1, 3, 12, 24)
    pre.process(tex.pointer, 1.0, 0.0, False)
    assert np.frombuffer(pre.read_back(), dtype=np.float32).size == 3 * 12 * 24


def test_repeated_dispatch_is_deterministic():
    ext = native.require()
    tex = texture_from(make_pattern(32, 32))
    pre = ext.GpuPreprocessor(tex.pointer, 16, 16)
    pre.process(tex.pointer, 1.0, 0.0, False)
    first = np.frombuffer(pre.read_back(), dtype=np.float32)
    for _ in range(5):
        pre.process(tex.pointer, 1.0, 0.0, False)
        assert np.array_equal(
            np.frombuffer(pre.read_back(), dtype=np.float32), first)


def test_output_buffer_has_a_gpu_address():
    ext = native.require()
    tex = texture_from(make_pattern(8, 8))
    pre = ext.GpuPreprocessor(tex.pointer, 8, 8)
    assert pre.output_buffer_address != 0


def test_zero_dimensions_rejected():
    ext = native.require()
    tex = texture_from(make_pattern(8, 8))
    with pytest.raises(ValueError, match="non-zero"):
        ext.GpuPreprocessor(tex.pointer, 0, 8)


def test_test_texture_validates_its_data_length():
    ext = native.require()
    with pytest.raises(ValueError, match="expected"):
        ext.TestTexture(4, 4, b"\x00" * 10)


# --------------------------------------------------------------------------
# D3D12 path
# --------------------------------------------------------------------------

def test_shared_nthandle_requires_keyed_mutex():
    """
    Documents a constraint that shaped the test strategy.

    D3D11 refuses SHARED_NTHANDLE on its own — it must be paired with
    SHARED_KEYEDMUTEX. But a keyed-mutex resource cannot be read until its mutex
    is acquired, which is why synthetic textures cannot exercise the D3D12 path:
    they would report zeros on *both* APIs. D3D12 correctness is therefore
    established against real captured frames, whose mutex DXGI manages, plus the
    exact agreement with this file's exhaustively-tested D3D11 path.
    """
    ext = native.require()
    pattern = make_pattern(8, 8)
    data = np.ascontiguousarray(pattern).tobytes()

    # Unshared: fine.
    ext.TestTexture(8, 8, data, shared=False)
    # Shared with the mutex: allowed to create.
    ext.TestTexture(8, 8, data, shared=True, keyed_mutex=True)
    # Shared without it: rejected by D3D11.
    with pytest.raises(RuntimeError, match="CreateTexture2D failed"):
        ext.TestTexture(8, 8, data, shared=True, keyed_mutex=False)


def test_d3d12_preprocessor_requires_a_shareable_texture():
    """An unshared texture cannot reach D3D12, and must fail loudly."""
    ext = native.require()
    tex = texture_from(make_pattern(16, 16))  # unshared
    with pytest.raises(RuntimeError):
        ext.GpuPreprocessor12(tex.pointer, 8, 8)


def test_d3d12_rejects_zero_dimensions():
    ext = native.require()
    tex = texture_from(make_pattern(8, 8))
    with pytest.raises(ValueError, match="non-zero"):
        ext.GpuPreprocessor12(tex.pointer, 8, 0)
