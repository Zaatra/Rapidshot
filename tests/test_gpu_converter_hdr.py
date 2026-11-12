"""`GpuConverter(tone_map=...)` against live capture.

Two kinds of desktop, two kinds of test. On an **SDR** desktop the tone map
must do nothing at all, and that is checked on any machine with a capture. On
an **HDR** desktop the converter's sRGB must be what ``grab()`` returns: the
frame's own texture is read back through a D3D11 staging copy in *its* format
and converted by ``rapidshot.core.hdr.to_bgra8`` -- the CPU path, with no
D3D12 and no shader in between. Those skip unless Windows HDR is on for the
captured display.

The arithmetic itself is checked headlessly in ``tests/test_tone_map.py`` and
on WARP by the Rust ``warp_tests`` in ``native/core/src/converter12.rs``; this is
the end-to-end run on real hardware and a real HDR surface. It first passed on
an HDR panel on 2026-11-12 (Arc 130V, Samsung TU7100, FP16), which is what
made ``"auto"`` the default; the README claims no more than that run covered.
"""
import ctypes

import numpy as np
import pytest

import rapidshot
from rapidshot import native
from rapidshot.core import hdr

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


@pytest.fixture(scope="module")
def live_camera():
    camera = rapidshot.create(output_color="BGRA")
    yield camera
    camera.release()


@pytest.fixture(scope="module")
def live_frame(live_camera):
    frame = None
    for _ in range(600):
        frame = live_camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        pytest.skip("no frame captured — the screen must be changing")
    yield frame
    frame.release()


@pytest.fixture(scope="module")
def hdr_frame(live_frame):
    if live_frame.color_space != "scrgb":
        pytest.skip(f"the captured display is not HDR ({live_frame.pixel_format}, "
                    f"hdr={live_frame.hdr})")
    if live_frame.rotation_angle:
        pytest.skip(f"rotated display ({live_frame.rotation_angle}°)")
    return live_frame


def grab_reference(camera, frame):
    """What ``grab()`` makes of this frame: (H, W, 4) BGRA8 from the CPU path.

    The texture is copied to a staging surface of its own format -- FP16 or
    10-bit -- and handed to :func:`hdr.to_bgra8`, so the reference shares
    nothing with the converter but the captured surface.
    """
    from rapidshot._libs.d3d11 import (
        D3D11_CPU_ACCESS_READ,
        D3D11_TEXTURE2D_DESC,
        D3D11_USAGE_STAGING,
        ID3D11Texture2D,
    )
    from rapidshot._libs.dxgi import DXGI_MAPPED_RECT, IDXGISurface

    device = camera._device.device
    context = camera._device.im_context
    width, height = camera._output.surface_size

    desc = D3D11_TEXTURE2D_DESC()
    desc.Width, desc.Height = width, height
    desc.MipLevels = desc.ArraySize = 1
    desc.Format = frame.dxgi_format
    desc.SampleDesc.Count, desc.SampleDesc.Quality = 1, 0
    desc.Usage = D3D11_USAGE_STAGING
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
    desc.BindFlags = desc.MiscFlags = 0

    staging = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), None, ctypes.byref(staging))
    surface = staging.QueryInterface(IDXGISurface)
    context.CopyResource(staging, frame.d3d11_texture)
    mapped = DXGI_MAPPED_RECT()
    surface.Map(ctypes.byref(mapped), 1)
    try:
        buffer = (ctypes.c_ubyte * (mapped.Pitch * height)).from_address(mapped.pBits)
        rows = np.frombuffer(buffer, dtype=np.uint8).reshape(height, mapped.Pitch).copy()
    finally:
        surface.Unmap()

    color = hdr.DisplayColor(True, frame.sdr_white_nits)
    whole = hdr.to_bgra8(rows, width, height, frame.dxgi_format, color)
    if frame.region is not None:
        left, top, right, bottom = frame.region
        whole = whole[top:bottom, left:right]
    return whole


# --------------------------------------------------------------------------
# any desktop: the tone map is per frame, and an SDR frame is left alone
# --------------------------------------------------------------------------

@pytest.mark.parametrize("dtype,layout", [("float32", "nchw"), ("uint8", "nhwc")])
def test_an_sdr_frame_is_untouched_by_a_tone_map(live_frame, dtype, layout):
    if live_frame.color_space == "scrgb":
        pytest.skip("the captured display is HDR; this is the SDR half")
    plain = rapidshot.GpuConverter(live_frame, (64, 64), dtype=dtype, layout=layout)
    mapped = rapidshot.GpuConverter(live_frame, (64, 64), dtype=dtype, layout=layout,
                                    tone_map="auto")
    np.testing.assert_array_equal(mapped.process(live_frame).numpy(),
                                  plain.process(live_frame).numpy())


# --------------------------------------------------------------------------
# HDR desktop: the converter's sRGB is grab()'s sRGB
# --------------------------------------------------------------------------

def test_clip_reproduces_grab_on_an_hdr_desktop(live_camera, hdr_frame):
    """Full size, nearest, uint8: the converter's frame must be grab()'s.

    One code of slack per channel: hdr.py rounds in float64 and the GPU's
    ``pow`` is an approximation. On WARP every FP16 value and every 10-bit
    code agreed exactly; a hardware ``pow`` may land on the other side of a
    rounding boundary.
    """
    expected = grab_reference(live_camera, hdr_frame)
    converter = rapidshot.GpuConverter(
        hdr_frame, (hdr_frame.width, hdr_frame.height), dtype="uint8", layout="nhwc",
        sampling="nearest", tone_map="clip")
    got = converter.process(hdr_frame).numpy()[0]

    difference = np.abs(got.astype(np.int16) - expected.astype(np.int16))
    exact = float((difference[..., :3] == 0).mean())
    assert difference.max() <= 1, f"max difference {difference.max()}, {exact:.4%} exact"
    assert (got[..., 3] == 255).all(), "a tone-mapped frame is opaque, as grab()'s is"


def test_auto_equals_clip_wherever_nothing_passes_sdr_white(live_camera, hdr_frame):
    """Every pixel no brighter than SDR white -- all SDR content on an HDR
    desktop -- is converted identically by the two maps."""
    size = (hdr_frame.width, hdr_frame.height)
    kwargs = dict(dtype="uint8", layout="nhwc", sampling="nearest")
    clip = rapidshot.GpuConverter(hdr_frame, size, tone_map="clip", **kwargs)
    auto = rapidshot.GpuConverter(hdr_frame, size, tone_map="auto", **kwargs)
    clipped = clip.process(hdr_frame).numpy()[0]
    kept = auto.process(hdr_frame).numpy()[0]

    # A pixel at 255 in some channel may be a highlight, where the two differ
    # by design; everywhere else they must not differ at all.
    below_white = (clipped[..., :3] < 255).all(axis=-1)
    if not below_white.any():
        pytest.skip("every pixel touches SDR white")
    np.testing.assert_array_equal(kept[below_white], clipped[below_white])


def test_tone_mapped_tensors_have_an_sdr_frames_range(hdr_frame):
    """Bilinear, float32, default normalisation: 0..1, finite, no scRGB."""
    converter = rapidshot.GpuConverter(hdr_frame, (320, 320), tone_map="auto")
    tensor = converter.process(hdr_frame).numpy()
    assert np.isfinite(tensor).all()
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0
