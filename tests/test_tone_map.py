"""The GPU tone map's arithmetic against ``rapidshot/core/hdr.py``. Headless.

``GpuConverter(tone_map=...)`` runs in a compute shader, which needs a GPU.
What does not is the arithmetic, so it is checked here: :func:`to_display` is
the shader's ``to_display()`` step for step, in float32 as the GPU computes,
and it must reproduce the bytes :func:`hdr.to_bgra8` -- what ``grab()``
returns -- for **every** FP16 value and every 10-bit code, at several SDR
white levels. The transcription is pinned to the HLSL text below, and the
Rust tests in ``native/src/converter12.rs`` (``warp_tests``) run the real
shader on WARP against the same bytes.
"""
from pathlib import Path

import numpy as np
import pytest

from rapidshot.core import hdr

SHADER_SOURCE = Path(__file__).resolve().parents[1] / "native" / "src" / "converter12.rs"
WHITES_NITS = [80.0, 200.0, 240.0, 250.0, 300.0, 480.0, 1000.0]

F = np.float32


def to_display(rgb, white, max_rgb):
    """The shader's ``to_display()`` for ``ToneMap != 0``, in float32.

    ``rgb`` is ``(..., 3)`` linear scRGB; returns sRGB-encoded 0..1.
    """
    c = np.clip(np.nan_to_num(np.asarray(rgb, F), nan=0.0, posinf=65504.0, neginf=0.0),
                F(0.0), F(65504.0)) / F(white)
    if max_rgb:
        peak = c.max(axis=-1, keepdims=True)
        c = np.where(peak > F(1.0), c / np.maximum(peak, F(1.0)), c)
    c = np.clip(c, F(0.0), F(1.0)).astype(F)
    return np.where(c <= F(0.0031308), c * F(12.92),
                    F(1.055) * np.power(c, F(1.0 / 2.4)) - F(0.055)).astype(F)


def to_byte(encoded):
    """The BGRA8 kernel's quantisation: ``(uint)(saturate(v) * 255 + 0.5)``."""
    return (np.clip(encoded, F(0.0), F(1.0)) * F(255.0) + F(0.5)).astype(np.uint32).astype(np.uint8)


def grey(values):
    values = np.asarray(values)
    return np.stack([values, values, values], axis=-1)


def scale_milli(nits):
    return max(1, int(round(hdr.DisplayColor(True, nits).sdr_white_scale * 1000)))


@pytest.mark.parametrize("nits", WHITES_NITS)
@pytest.mark.parametrize("max_rgb", [False, True], ids=["clip", "auto"])
def test_every_fp16_value_comes_back_as_the_byte_grab_returns(nits, max_rgb):
    """Grey pixels never exceed SDR white in one channel alone, so both maps
    must reproduce grab() here -- every one of the 65536 half bit patterns,
    NaN and infinities included."""
    milli = scale_milli(nits)
    halves = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    got = to_byte(to_display(grey(halves), milli / 1000.0, max_rgb))[..., 0]
    assert np.array_equal(got, hdr._lut_fp16(True, milli))


@pytest.mark.parametrize("nits", WHITES_NITS)
@pytest.mark.parametrize("max_rgb", [False, True], ids=["clip", "auto"])
def test_every_10_bit_code_comes_back_as_the_byte_grab_returns(nits, max_rgb):
    milli = scale_milli(nits)
    codes = np.arange(1024, dtype=np.float64) / 1023.0
    got = to_byte(to_display(grey(codes), milli / 1000.0, max_rgb))[..., 0]
    assert np.array_equal(got, hdr._lut_10bit(True, milli))


def test_through_to_bgra8_the_two_paths_agree_pixel_for_pixel():
    """The same comparison through hdr.to_bgra8 itself, on coloured pixels:
    channel order, alpha and the per-channel lookups included."""
    rng = np.random.default_rng(7)
    linear = rng.uniform(-0.5, 3.0, size=(16, 32, 3)).astype(np.float16)   # SDR white 3.0
    pixels = np.concatenate([linear, np.ones((16, 32, 1), np.float16)], axis=-1)
    rows = pixels.view(np.uint8).reshape(16, 32 * 8)
    cpu = hdr.to_bgra8(rows, 32, 16, hdr.DXGI_FORMAT_R16G16B16A16_FLOAT,
                       hdr.DisplayColor(True, 240.0))
    gpu = to_byte(to_display(linear, 3.0, max_rgb=True))
    assert np.array_equal(cpu[..., 2::-1], gpu)
    assert (cpu[..., 3] == 255).all()


def test_auto_is_clip_wherever_no_channel_passes_sdr_white():
    rng = np.random.default_rng(11)
    sdr = rng.uniform(0.0, 3.0, size=(4096, 3))
    assert np.array_equal(to_display(sdr, 3.0, True), to_display(sdr, 3.0, False))


def test_auto_keeps_a_highlights_hue_where_clip_shifts_it():
    """(6, 3, 0.75) at SDR white 3: clip gives (1, 1, 0.25), yellow; auto
    gives (1, 0.5, 0.125) -- the orange it was, at SDR white's brightness."""
    orange = np.array([6.0, 3.0, 0.75])
    clipped = to_display(orange, 3.0, max_rgb=False)
    kept = to_display(orange, 3.0, max_rgb=True)
    assert np.allclose(clipped, to_display([3.0, 3.0, 0.75], 3.0, False))
    assert np.allclose(kept, to_display([3.0, 1.5, 0.375], 3.0, False))
    # Ratios survive in linear light: 4 : 2 : 0.5 -> 1 : 0.5 : 0.125.
    assert kept[0] == pytest.approx(1.0) and kept[1] < kept[0] and kept[2] < kept[1]


def test_non_finite_and_negative_input_follow_grab():
    """NaN and -inf to 0, +inf to the largest half -- then white -- as
    hdr._lut_fp16 does."""
    values = np.array([[np.nan, np.inf, -np.inf], [-0.5, 0.0, np.inf]])
    for max_rgb in (False, True):
        assert np.array_equal(to_byte(to_display(values, 3.0, max_rgb)),
                              [[0, 255, 0], [0, 0, 255]])


@pytest.mark.skipif(not SHADER_SOURCE.exists(), reason="native source not in this checkout")
def test_the_transcription_matches_the_shader_text():
    """If the HLSL changes, this transcription must change with it -- or the
    tests above check a formula the GPU no longer runs."""
    shader = SHADER_SOURCE.read_text(encoding="utf-8")
    start = shader.index("float3 to_display(float4 texel)")
    body = shader[start:shader.index("\n}", start)]
    for fragment in (
        "if (ToneMap == 0)",
        "clamp(texel.xyz, 0.0f, 65504.0f) / SdrWhite",
        "if (ToneMap == 2)",
        "float peak = max(c.x, max(c.y, c.z));",
        "if (peak > 1.0f)",
        "c /= peak;",
        "c = saturate(c);",
        "(c <= 0.0031308f) ? c * 12.92f : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f",
    ):
        assert fragment in body, fragment
    assert "(uint)(saturate(rgb.x) * 255.0f + 0.5f)" in shader
