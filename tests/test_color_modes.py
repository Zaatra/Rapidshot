"""Tests for output color conversion and the direct-to-buffer shot() path.

These exercise the processor layer directly with a synthetic mapped surface, so
they run without a GPU or a Windows desktop session.
"""

import ctypes

import numpy as np
import pytest

from rapidshot.processor.base import (
    COLOR_MODE_CHANNELS,
    channels_for_color_mode,
    validate_color_mode,
)
from rapidshot.processor.numpy_processor import NumpyProcessor, bgra_to_gray


class FakeMappedRect:
    """Stands in for DXGI_MAPPED_RECT over a ctypes-backed BGRA buffer."""

    def __init__(self, bgra: np.ndarray, pitch=None):
        height, width, _ = bgra.shape
        self.Pitch = width * 4 if pitch is None else pitch
        self._backing = (ctypes.c_ubyte * (self.Pitch * height))()
        view = np.ctypeslib.as_array(self._backing).reshape(height, self.Pitch)
        view[:, : width * 4] = bgra.reshape(height, width * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def make_bgra(height=4, width=6):
    """Deterministic BGRA test image with a distinct value per channel."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


# --------------------------------------------------------------------------
# Color mode metadata
# --------------------------------------------------------------------------

def test_validate_color_mode_rejects_unknown():
    with pytest.raises(ValueError, match="Unsupported color mode"):
        validate_color_mode("YUV")


def test_channels_treats_none_as_bgra():
    assert channels_for_color_mode(None) == 4


@pytest.mark.parametrize("mode,expected", sorted(COLOR_MODE_CHANNELS.items()))
def test_processor_reports_channel_count(mode, expected):
    assert NumpyProcessor(mode).output_channels == expected


def test_processor_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported color mode"):
        NumpyProcessor("CMYK")


# --------------------------------------------------------------------------
# GRAY actually converts (regression: used to return unconverted BGRA)
# --------------------------------------------------------------------------

def test_gray_produces_single_channel():
    bgra = make_bgra()
    gray = bgra_to_gray(bgra)
    assert gray.shape == (bgra.shape[0], bgra.shape[1], 1)
    assert gray.dtype == np.uint8


def test_gray_matches_rec601_luma():
    bgra = make_bgra(height=64, width=64)
    gray = bgra_to_gray(bgra)[..., 0].astype(np.float64)
    expected = (
        0.299 * bgra[..., 2].astype(np.float64)
        + 0.587 * bgra[..., 1].astype(np.float64)
        + 0.114 * bgra[..., 0].astype(np.float64)
    )
    # Q8 fixed point with round-to-nearest. The coefficients themselves are
    # rounded (77/256 = 0.3008 vs 0.299), so a deviation of one level is
    # expected; this is the same approximation OpenCV uses for 8-bit input.
    assert np.max(np.abs(gray - expected)) <= 1.0
    # And it must not be systematically biased in either direction.
    assert abs(np.mean(gray - expected)) < 0.25


def test_gray_of_pure_channels():
    # Pure blue/green/red should land on the Rec.601 weights.
    pure = np.array([[[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]]], np.uint8)
    gray = bgra_to_gray(pure)[0, :, 0]
    assert abs(int(gray[0]) - 29) <= 1   # blue  -> 0.114 * 255
    assert abs(int(gray[1]) - 150) <= 1  # green -> 0.587 * 255
    assert abs(int(gray[2]) - 76) <= 1   # red   -> 0.299 * 255


def test_gray_does_not_require_opencv(monkeypatch):
    """GRAY must work with OpenCV absent -- it used to silently pass through."""
    import builtins

    real_import = builtins.__import__

    def no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("cv2 is unavailable in this test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cv2)

    processor = NumpyProcessor("GRAY")
    converted = processor.process_cvtcolor(make_bgra())
    assert converted.shape[2] == 1


# --------------------------------------------------------------------------
# RGBA channel order (regression: used to return BGRA data mislabeled RGBA)
# --------------------------------------------------------------------------

def test_rgba_swaps_red_and_blue():
    bgra = make_bgra()
    rgba = NumpyProcessor("RGBA").process_cvtcolor(bgra)
    assert np.array_equal(rgba[..., 0], bgra[..., 2])  # R
    assert np.array_equal(rgba[..., 1], bgra[..., 1])  # G
    assert np.array_equal(rgba[..., 2], bgra[..., 0])  # B
    assert np.array_equal(rgba[..., 3], bgra[..., 3])  # A


def test_rgb_channel_order():
    bgra = make_bgra()
    rgb = NumpyProcessor("RGB").process_cvtcolor(bgra)
    assert rgb.shape[2] == 3
    assert np.array_equal(rgb[..., 0], bgra[..., 2])
    assert np.array_equal(rgb[..., 2], bgra[..., 0])


def test_bgr_channel_order():
    bgra = make_bgra()
    bgr = NumpyProcessor("BGR").process_cvtcolor(bgra)
    assert np.array_equal(bgr, bgra[..., :3])


# --------------------------------------------------------------------------
# shot(): honors color mode and validates the destination
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", sorted(COLOR_MODE_CHANNELS))
def test_shot_writes_requested_color_mode(mode):
    bgra = make_bgra()
    height, width, _ = bgra.shape
    channels = COLOR_MODE_CHANNELS[mode]
    processor = NumpyProcessor(mode)

    dest = np.zeros((height, width, channels), dtype=np.uint8)
    assert processor.shot(dest, FakeMappedRect(bgra), width, height) is True

    expected = processor.process_cvtcolor(bgra) if mode != "BGRA" else bgra
    assert np.array_equal(dest, expected.reshape(dest.shape))


def test_shot_handles_padded_pitch():
    """The staging surface pitch is often wider than a row of pixels."""
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("RGB")

    dest = np.zeros((height, width, 3), dtype=np.uint8)
    processor.shot(dest, FakeMappedRect(bgra, pitch=width * 4 + 64), width, height)
    assert np.array_equal(dest, bgra[..., 2::-1])


def test_shot_rejects_undersized_buffer():
    """Regression: a 3-channel buffer used to be overrun by a full extra channel."""
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")  # needs 4 channels

    too_small = np.zeros((height, width, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        processor.shot(too_small, FakeMappedRect(bgra), width, height)


def test_shot_rejects_unsized_raw_pointer():
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")

    backing = np.zeros((height, width, 4), dtype=np.uint8)
    raw = ctypes.c_void_p(backing.ctypes.data)
    with pytest.raises(ValueError, match="Cannot verify destination size"):
        processor.shot(raw, FakeMappedRect(bgra), width, height)


def test_shot_accepts_raw_pointer_with_explicit_size():
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")

    backing = np.zeros((height, width, 4), dtype=np.uint8)
    raw = ctypes.c_void_p(backing.ctypes.data)
    assert processor.shot(
        raw, FakeMappedRect(bgra), width, height, buffer_size=backing.nbytes
    )
    assert np.array_equal(backing, bgra)


def test_shot_rejects_lying_explicit_size():
    """An explicit size smaller than required must fail even if the array fits."""
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")

    dest = np.zeros((height, width, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="too small"):
        processor.shot(dest, FakeMappedRect(bgra), width, height, buffer_size=16)


def test_shot_accepts_bytearray_and_ctypes_array():
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")

    as_bytearray = bytearray(height * width * 4)
    processor.shot(as_bytearray, FakeMappedRect(bgra), width, height)
    assert np.array_equal(
        np.frombuffer(bytes(as_bytearray), np.uint8).reshape(height, width, 4), bgra
    )

    as_ctypes = (ctypes.c_ubyte * (height * width * 4))()
    processor.shot(as_ctypes, FakeMappedRect(bgra), width, height)
    assert np.array_equal(
        np.ctypeslib.as_array(as_ctypes).reshape(height, width, 4), bgra
    )


def test_shot_rejects_non_contiguous_destination():
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("BGRA")

    oversized = np.zeros((height, width * 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="C-contiguous"):
        processor.shot(oversized[:, :width], FakeMappedRect(bgra), width, height)


@pytest.mark.parametrize("mode", sorted(COLOR_MODE_CHANNELS))
def test_process_result_never_aliases_the_pooled_buffer(mode):
    """
    process() must not return a view into the pooled buffer.

    When it reports is_still_pooled_buffer=False, grab() checks that buffer
    straight back into the pool -- so a view into it would be silently
    overwritten by the next capture. This used to happen for RGB, BGR and GRAY,
    where the converter returned a lazy slice instead of a materialised array,
    and callers' frames mutated under them.
    """
    processor = NumpyProcessor(mode)
    pool_buffer = np.zeros((4, 4, 4), dtype=np.uint8)

    frame_a = np.full((4, 4, 4), 10, dtype=np.uint8)
    frame_b = np.full((4, 4, 4), 200, dtype=np.uint8)

    out, still_pooled = processor.process(
        FakeMappedRect(frame_a), 4, 4, (0, 0, 4, 4), 0, pool_buffer)
    captured = out.copy()

    if not still_pooled:
        assert not np.shares_memory(out, pool_buffer), (
            "result aliases a buffer the caller is about to recycle")

    # Simulate the pooled buffer being reused by the next capture.
    processor.process(FakeMappedRect(frame_b), 4, 4, (0, 0, 4, 4), 0, pool_buffer)

    if not still_pooled:
        assert np.array_equal(out, captured), (
            "the caller's frame changed when the pool buffer was reused")


@pytest.mark.parametrize("mode", sorted(COLOR_MODE_CHANNELS))
def test_process_returns_contiguous_owned_array(mode):
    """Downstream consumers (encoders, ONNX) need contiguous buffers."""
    processor = NumpyProcessor(mode)
    pool_buffer = np.zeros((4, 4, 4), dtype=np.uint8)
    out, _ = processor.process(
        FakeMappedRect(make_bgra(4, 4)), 4, 4, (0, 0, 4, 4), 0, pool_buffer)
    assert out.flags.c_contiguous
    assert out.shape[2] == COLOR_MODE_CHANNELS[mode]


def test_process_with_rotation_is_contiguous_and_independent():
    processor = NumpyProcessor("RGB")
    pool_buffer = np.zeros((4, 4, 4), dtype=np.uint8)
    out, still_pooled = processor.process(
        FakeMappedRect(make_bgra(4, 4)), 4, 4, (0, 0, 4, 4), 90, pool_buffer)
    assert still_pooled is False
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, pool_buffer)


def test_capture_validates_destination_before_capturing():
    """
    ScreenCapture.shot() must reject a bad buffer up front.

    Validating only inside the processor made the error fire exclusively on the
    calls that received new frame content, so on a static desktop an undersized
    buffer returned False for a while and only raised once the screen changed.
    """
    from rapidshot.capture import ScreenCapture

    cam = ScreenCapture.__new__(ScreenCapture)  # bypass DXGI initialisation
    cam.width, cam.height = 640, 480
    cam.region = (0, 0, 640, 480)
    cam.output_color = "RGB"
    cam._processor = NumpyProcessor("RGB")

    assert cam.bytes_per_frame((0, 0, 640, 480)) == 640 * 480 * 3

    with pytest.raises(ValueError, match="too small"):
        cam._validate_destination(
            np.zeros((480, 640, 2), np.uint8), (0, 0, 640, 480), None
        )

    with pytest.raises(ValueError, match="Cannot verify destination size"):
        cam._validate_destination(ctypes.c_void_p(1234), (0, 0, 640, 480), None)

    # Exactly-sized destination is accepted.
    assert cam._validate_destination(
        np.zeros((480, 640, 3), np.uint8), (0, 0, 640, 480), None
    ) == 640 * 480 * 3


def test_shot_does_not_write_past_the_destination():
    """Guard the exact overflow that produced the 0xC0000005 crash."""
    bgra = make_bgra()
    height, width, _ = bgra.shape
    processor = NumpyProcessor("RGB")  # 3 channels out

    # Allocate the exact destination plus a sentinel region behind it.
    total = height * width * 3
    backing = (ctypes.c_ubyte * (total + 256))()
    sentinel = 0xAB
    for i in range(total, total + 256):
        backing[i] = sentinel

    dest = np.ctypeslib.as_array(backing)[:total].reshape(height, width, 3)
    processor.shot(dest, FakeMappedRect(bgra), width, height)

    tail = np.ctypeslib.as_array(backing)[total:]
    assert np.all(tail == sentinel), "shot() wrote past the end of the destination"
