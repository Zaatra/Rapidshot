"""HDR desktops through the CPU capture paths. Headless: no device, no display.

On an HDR desktop ``DuplicateOutput1`` hands back linear scRGB in FP16 or, on
some platforms, R10G10B10A2 -- and the BGRA8 staging copy the CPU paths used
came back black. These check the conversion against an independent reference:
SDR bytes as Windows composes them on an HDR desktop,
``scRGB = EOTF_sRGB(v) x SDR white / 80``, must come back as the same bytes.
"""
import ctypes
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from rapidshot.core import hdr
from rapidshot.core.stagesurf import StageSurface

LEVELS = np.arange(256, dtype=np.float64)


def srgb_to_linear(v):
    c = v / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def fp16_rows(linear):
    pixels = np.zeros((1, len(linear), 4), np.float16)
    pixels[0, :, 0] = pixels[0, :, 1] = pixels[0, :, 2] = linear
    pixels[0, :, 3] = 1.0
    return pixels.view(np.uint8).reshape(1, len(linear) * 8)


def r10_rows(codes):
    codes = np.asarray(codes, np.uint32)
    packed = codes | (codes << 10) | (codes << 20) | (3 << 30)
    return packed.view(np.uint8).reshape(1, len(codes) * 4)


@pytest.mark.parametrize("nits", [80.0, 200.0, 240.0, 480.0])
def test_sdr_content_on_an_fp16_hdr_desktop_comes_back_byte_exact(nits):
    color = hdr.DisplayColor(True, nits)
    rows = fp16_rows(srgb_to_linear(LEVELS) * (nits / 80.0))
    out = hdr.to_bgra8(rows, 256, 1, hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, color)
    assert np.array_equal(out[0, :, 2], LEVELS.astype(np.uint8))
    assert np.array_equal(out[0, :, 0], out[0, :, 2]) and (out[..., 3] == 255).all()


def test_hdr_highlights_above_sdr_white_clip_to_white():
    color = hdr.DisplayColor(True, 240.0)
    out = hdr.to_bgra8(fp16_rows([3.0, 6.0, 12.5, 700.0]), 4, 1,
                       hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, color)
    assert (out[0, :, :3] == 255).all()


def test_negative_and_non_finite_scrgb_is_black_not_garbage():
    color = hdr.DisplayColor(True, 200.0)
    values = np.array([-0.5, np.nan, -np.inf], np.float16)
    out = hdr.to_bgra8(fp16_rows(values), 3, 1, hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, color)
    assert (out[0, :, :3] == 0).all()


def test_a_clipped_10bit_desktop_is_exact_up_to_the_clip_and_flat_after():
    """The Comet Lake desktop: linear scRGB in R10G10B10A2 clipped at 1.0 =
    80 nits. With SDR white at 240 nits the clip lands at level 156."""
    color = hdr.DisplayColor(True, 240.0)
    linear = srgb_to_linear(LEVELS) * 3.0
    codes = np.rint(np.clip(linear, 0, 1) * 1023)
    out = hdr.to_bgra8(r10_rows(codes), 256, 1, hdr.DXGI_FORMAT_R10G10B10A2_UNORM, color)
    below = linear <= 1.0
    assert np.abs(out[0, below, 2].astype(int) - LEVELS[below]).max() <= 1
    assert len(set(out[0, ~below, 2].tolist())) == 1
    assert hdr.clipped_at_nominal_white(hdr.DXGI_FORMAT_R10G10B10A2_UNORM, color)
    assert not hdr.clipped_at_nominal_white(hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, color)
    assert not hdr.clipped_at_nominal_white(hdr.DXGI_FORMAT_R10G10B10A2_UNORM,
                                            hdr.DisplayColor(False))


def test_with_hdr_off_10bit_and_fp16_are_gamma_values_and_only_rescaled():
    sdr = hdr.DisplayColor(False)
    out10 = hdr.to_bgra8(r10_rows([0, 512, 1023]), 3, 1, hdr.DXGI_FORMAT_R10G10B10A2_UNORM, sdr)
    assert out10[0, :, 2].tolist() == [0, 128, 255]
    out16 = hdr.to_bgra8(fp16_rows([0.0, 0.5, 1.0]), 3, 1,
                         hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, sdr)
    assert out16[0, :, 2].tolist() == [0, 128, 255]


def test_rgba8_is_swizzled_and_bgra8_passes_through_with_its_pitch():
    rgba = np.array([[[10, 20, 30, 40], [50, 60, 70, 80]]], np.uint8)
    padded = np.zeros((1, 16), np.uint8)
    padded[0, :8] = rgba.reshape(-1)
    out = hdr.to_bgra8(padded, 2, 1, hdr.DXGI_FORMAT_R8G8B8A8_UNORM)
    assert out[0, 0].tolist() == [30, 20, 10, 255]
    same = hdr.to_bgra8(padded, 2, 1, hdr.DXGI_FORMAT_B8G8R8A8_UNORM)
    assert same[0, 1, :3].tolist() == [50, 60, 70]


def test_an_unknown_format_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no conversion"):
        hdr.to_bgra8(np.zeros((1, 4), np.uint8), 1, 1, 2)


def test_a_display_that_cannot_be_asked_is_treated_as_sdr(monkeypatch):
    monkeypatch.setattr(hdr, "_query", lambda name: (_ for _ in ()).throw(OSError("no")))
    color = hdr.display_color(r"\\.\DISPLAY9")
    assert (color.hdr, color.sdr_white_nits) == (False, 80.0)


# -- the staging surface -------------------------------------------------------

def stage_over(buffer, fmt, color=None, width=None):
    """A StageSurface with its D3D11 Map/Unmap replaced by a NumPy buffer."""
    surface = object.__new__(StageSurface)
    surface.height, pitch = buffer.shape
    surface.width = width if width is not None else pitch // hdr.bytes_per_pixel(fmt)
    surface.dxgi_format, surface.color = fmt, color
    surface._converted, surface._mapped = None, False
    calls = []

    def map_raw():
        calls.append("map")
        rect = SimpleNamespace(Pitch=pitch, pBits=buffer.ctypes.data)
        return rect

    surface._map_raw = map_raw
    surface._unmap_raw = lambda: calls.append("unmap")
    return surface, calls


def read(rect, width, height):
    raw = ctypes.string_at(rect.pBits, rect.Pitch * height)
    return np.frombuffer(raw, np.uint8).reshape(height, rect.Pitch)[:, : width * 4] \
        .reshape(height, width, 4)


def test_an_hdr_surface_maps_as_bgra8_and_is_unmapped_straight_away():
    color = hdr.DisplayColor(True, 240.0)
    buffer = fp16_rows(srgb_to_linear(np.array([0.0, 128.0, 255.0])) * 3.0)
    surface, calls = stage_over(buffer, hdr.DXGI_FORMAT_R16G16B16A16_FLOAT, color)
    rect = surface.map()
    assert calls == ["map", "unmap"], "the real surface is released once converted"
    assert rect.Pitch == 3 * 4
    assert read(rect, 3, 1)[0, :, 2].tolist() == [0, 128, 255]
    surface.unmap()
    assert calls == ["map", "unmap"], "nothing left to unmap"


def test_a_bgra8_surface_is_mapped_as_it_is():
    buffer = np.arange(16, dtype=np.uint8).reshape(1, 16)
    surface, calls = stage_over(buffer, hdr.DXGI_FORMAT_B8G8R8A8_UNORM)
    rect = surface.map()
    assert rect.pBits == buffer.ctypes.data and calls == ["map"]
    surface.unmap()
    assert calls == ["map", "unmap"]


def test_the_surface_is_rebuilt_when_the_captured_format_changes():
    surface = object.__new__(StageSurface)
    surface.texture, surface.width, surface.height = object(), 64, 48
    surface.dxgi_format = hdr.DXGI_FORMAT_B8G8R8A8_UNORM
    rebuilt = []
    surface.release = lambda: setattr(surface, "texture", None)
    surface.rebuild = lambda output, device, dim: rebuilt.append((surface.dxgi_format, dim))
    surface.ensure(None, None, (64, 48), hdr.DXGI_FORMAT_B8G8R8A8_UNORM)
    assert rebuilt == []
    surface.ensure(None, None, (64, 48), hdr.DXGI_FORMAT_R10G10B10A2_UNORM)
    assert rebuilt == [(hdr.DXGI_FORMAT_R10G10B10A2_UNORM, (64, 48))]


# -- the capture's side --------------------------------------------------------

class Stage:
    def __init__(self, fmt):
        self.fmt, self.color, self.ensured = fmt, None, []

    def format_of(self, texture):
        return self.fmt

    def ensure(self, output, device, dim, source_format):
        self.ensured.append((dim, source_format))


def capture_with(fmt, color, monkeypatch):
    from rapidshot.capture import ScreenCapture
    capture = object.__new__(ScreenCapture)
    capture._stagesurf = Stage(fmt)
    capture._duplicator = SimpleNamespace(texture=object())
    capture._output = SimpleNamespace(devicename=r"\\.\DISPLAY1")
    capture._device = None
    capture._display_color, capture._display_color_at = None, 0.0
    capture._warned_hdr_clip = False
    asked = []
    monkeypatch.setattr(hdr, "display_color", lambda name: asked.append(name) or color)
    return capture, asked


def test_an_sdr_desktop_never_asks_about_hdr(monkeypatch):
    capture, asked = capture_with(hdr.DXGI_FORMAT_B8G8R8A8_UNORM, None, monkeypatch)
    capture._prepare_stage((64, 48))
    assert asked == [] and capture._stagesurf.ensured == [((64, 48), 87)]


def test_a_clipped_hdr_desktop_is_named_once(monkeypatch, caplog):
    color = hdr.DisplayColor(True, 240.0)
    capture, asked = capture_with(hdr.DXGI_FORMAT_R10G10B10A2_UNORM, color, monkeypatch)
    with caplog.at_level(logging.WARNING, logger="rapidshot.capture"):
        capture._prepare_stage((64, 48))
        capture._prepare_stage((64, 48))
    assert capture._stagesurf.color is color and asked == [r"\\.\DISPLAY1"]
    assert caplog.text.count("clipped at 80 nits") == 1


def test_a_format_the_cpu_paths_cannot_convert_is_an_error(monkeypatch):
    from rapidshot.util.errors import RapidShotError
    capture, _ = capture_with(2, None, monkeypatch)
    with pytest.raises(RapidShotError, match="DXGI format 2"):
        capture._prepare_stage((64, 48))
