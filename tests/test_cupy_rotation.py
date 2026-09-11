"""Rotation on the CuPy path must not alias the buffer it rotated.

`cp.rot90` returns a view, and `process()` used to hand that view on in two
different ways, both wrong on a physically rotated display:

  * at 180 degrees the shape is unchanged, so the pooled buffer was assigned
    from a view of *itself* -- an overlapping copy, which tears; and
  * at 90 and 270 the view was returned with the pooled flag cleared, so
    `_grab()` checked the buffer back in while the caller still pointed into it.

CuPy is a drop-in NumPy API here, so these run against NumPy standing in for
`self.cp`: no GPU, no CUDA driver, and the aliasing is a property of the array
protocol rather than of the device. `test_cupy_processor.py` covers the real
device where one is present.
"""

import ctypes

import numpy as np
import pytest

from rapidshot.processor.cupy_processor import CupyProcessor
from rapidshot.processor.numpy_processor import NumpyProcessor


class FakeMappedRect:
    """Stands in for DXGI_MAPPED_RECT over a ctypes-backed BGRA buffer."""

    def __init__(self, bgra: np.ndarray, pitch=None):
        height, width, _ = bgra.shape
        self.Pitch = width * 4 if pitch is None else pitch
        self._backing = (ctypes.c_ubyte * (self.Pitch * height))()
        view = np.ctypeslib.as_array(self._backing).reshape(height, self.Pitch)
        view[:, : width * 4] = bgra.reshape(height, width * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def numpy_backed_processor(color_mode):
    """A CupyProcessor with NumPy as its array module.

    Built with `__new__` because `__init__` imports CuPy. Everything
    `process()` touches -- `empty`, `rot90`, `ascontiguousarray`, `empty_like`,
    `newaxis`, `uint16` -- NumPy provides under the same names.
    """
    processor = CupyProcessor.__new__(CupyProcessor)
    processor.cp = np
    processor.color_mode = None if color_mode == "BGRA" else color_mode
    processor.cvtcolor = None
    return processor


def bgra_pattern(height=6, width=10):
    """Distinct per-pixel values, so a torn or transposed frame cannot pass."""
    rng = np.random.default_rng(20260911)
    return rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)


def run(processor, pattern, rotation_angle):
    """process() over a pooled buffer, as _grab() calls it."""
    height, width, _ = pattern.shape
    pooled = np.zeros((height, width, 4), dtype=np.uint8)
    result, still_pooled = processor.process(
        FakeMappedRect(pattern), width, height,
        (0, 0, width, height), rotation_angle, pooled,
    )
    return result, still_pooled, pooled


ROTATIONS = [90, 180, 270]


@pytest.mark.parametrize("rotation_angle", ROTATIONS)
def test_rotated_frame_does_not_alias_the_pooled_buffer(rotation_angle):
    """The whole bug: _grab() releases the buffer whenever the flag is False."""
    pattern = bgra_pattern()
    processor = numpy_backed_processor("BGRA")

    result, still_pooled, pooled = run(processor, pattern, rotation_angle)

    assert still_pooled is False, (
        "a rotated frame is a new allocation, so the pooled buffer is free")
    assert not np.shares_memory(result, pooled), (
        f"{rotation_angle} degrees returned a view of the pooled buffer; the "
        "next capture would overwrite the caller's frame")


@pytest.mark.parametrize("rotation_angle", ROTATIONS)
def test_rotated_frame_holds_the_rotated_pixels(rotation_angle):
    """The rotation itself has to stay right, not merely stop aliasing.

    Note this one passes on the unfixed code too, and on NumPy it always would:
    `a[:] = view_of_a` is overlap-safe there, because NumPy buffers the right
    hand side when it detects the overlap. CuPy's elementwise kernel does not,
    which is where the torn 180-degree frame came from -- a difference only a
    real device can show. What the stand-in *does* pin down is the structural
    fault above: that the write went into a view of its own destination.
    """
    pattern = bgra_pattern()
    processor = numpy_backed_processor("BGRA")

    result, _, _ = run(processor, pattern, rotation_angle)

    expected = np.rot90(pattern, k=(rotation_angle // 90) % 4)
    assert result.shape == expected.shape
    assert np.array_equal(result, expected), (
        f"{rotation_angle} degrees did not produce the rotated frame")


@pytest.mark.parametrize("rotation_angle", ROTATIONS)
def test_rotated_frame_is_contiguous(rotation_angle):
    """A view of a transpose is not something a consumer can memcpy out of."""
    pattern = bgra_pattern()
    processor = numpy_backed_processor("BGRA")

    result, _, _ = run(processor, pattern, rotation_angle)

    assert result.flags["C_CONTIGUOUS"]


def test_single_pixel_region_at_180_still_copies():
    """rot90 of a 1x1 frame is contiguous *and* aliased -- both flips no-op.

    `ascontiguousarray` would hand that view straight back, so the buffer would
    be released under a caller still reading it. Only an unconditional copy
    closes this one.
    """
    pattern = bgra_pattern(height=1, width=1)
    processor = numpy_backed_processor("BGRA")

    result, still_pooled, pooled = run(processor, pattern, 180)

    assert still_pooled is False
    assert not np.shares_memory(result, pooled)
    assert np.array_equal(result, pattern)


@pytest.mark.parametrize("mode", ["RGB", "BGR", "RGBA", "GRAY"])
def test_converted_and_rotated_frame_matches_the_numpy_path(mode):
    """Rotation composes with colour conversion, and still owns its storage."""
    pattern = bgra_pattern()
    processor = numpy_backed_processor(mode)

    result, still_pooled, pooled = run(processor, pattern, 90)

    channels = NumpyProcessor(mode).output_channels
    converted = np.empty((*pattern.shape[:2], channels), dtype=np.uint8)
    NumpyProcessor(mode).convert_into(pattern, converted)
    expected = np.rot90(converted, k=1)

    assert still_pooled is False
    assert not np.shares_memory(result, pooled)
    assert np.array_equal(result, expected)


def test_unrotated_frame_still_hands_back_the_pooled_buffer():
    """The fix must not cost an allocation on the ordinary, unrotated path."""
    pattern = bgra_pattern()
    processor = numpy_backed_processor("BGRA")

    result, still_pooled, pooled = run(processor, pattern, 0)

    assert still_pooled is True
    assert result is pooled
    assert np.array_equal(result, pattern)
