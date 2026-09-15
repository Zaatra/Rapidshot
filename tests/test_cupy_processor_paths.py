"""CupyProcessor with NumPy standing in for CuPy -- no CUDA device.

``test_cupy_processor.py`` needs a real device and ``test_cupy_rotation.py``
covers rotation, so construction, validation and the staging read were only
ever run on a CUDA machine. CuPy's array API is NumPy's for everything
``process()`` touches, so a module that forwards to NumPy runs the same code.
"""
import ctypes
import logging
import sys
import types
import warnings

import numpy as np
import pytest

from rapidshot.processor.cupy_processor import CupyProcessor


class NumpyAsCupy(types.ModuleType):
    """A 'cupy' module whose attributes are NumPy's."""

    def __init__(self, version="14.1.1", missing=()):
        super().__init__("cupy")
        self.__version__ = version
        self._missing = set(missing)

    def __getattr__(self, name):
        if name in self._missing:
            raise AttributeError(name)
        return getattr(np, name)


class DeviceArray(np.ndarray):
    """An array with CuPy's host-to-device `set()`."""

    def set(self, host):
        self[...] = host


@pytest.fixture
def cupy(monkeypatch):
    def install(**kwargs):
        module = NumpyAsCupy(**kwargs)
        monkeypatch.setitem(sys.modules, "cupy", module)
        return module
    return install


class Mapped:
    def __init__(self, bgra, pitch=None):
        height, width, _ = bgra.shape
        self.Pitch = width * 4 if pitch is None else pitch
        self._buf = (ctypes.c_ubyte * (self.Pitch * height))()
        view = np.ctypeslib.as_array(self._buf).reshape(height, self.Pitch)
        view[:, :width * 4] = bgra.reshape(height, width * 4)
        self.pBits = ctypes.cast(self._buf, ctypes.c_void_p)


def image(height=6, width=8):
    return np.random.default_rng(3).integers(0, 256, (height, width, 4), dtype=np.uint8)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_construction_with_a_current_cupy(cupy):
    cupy()
    proc = CupyProcessor("BGRA")
    assert proc.color_mode is None


def test_an_old_cupy_warns_and_checks_its_features(cupy, caplog):
    cupy(version="9.6.0")
    with pytest.warns(RuntimeWarning, match="CuPy version 9.6.0"):
        CupyProcessor("RGB")


def test_an_old_cupy_missing_features_says_so(cupy):
    cupy(version="9.0.0", missing={"rot90"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CupyProcessor("RGB")
    assert any("missing some required features" in str(w.message) for w in caught)


def test_no_cupy_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match=r"rapidshot\[gpu_cuda13\]"):
        CupyProcessor("RGB")


def test_an_unsupported_mode_is_refused_at_construction(cupy):
    cupy()
    with pytest.raises(ValueError, match="Unsupported color mode: 'YUV'"):
        CupyProcessor("YUV")


def test_conversion_refuses_a_mode_that_bypassed_validation(cupy):
    cupy()
    proc = CupyProcessor("RGB")
    proc.color_mode = "HSV"
    with pytest.raises(ValueError, match="Unsupported color mode: 'HSV'"):
        proc.process_cvtcolor(image())


# --------------------------------------------------------------------------
# process
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pitch_extra,left", [(0, 0), (12, 0), (0, 2)])
def test_the_staging_read_handles_padding_and_offsets(cupy, pitch_extra, left):
    cupy()
    bgra = image()
    height, width = bgra.shape[:2]
    proc = CupyProcessor("BGRA")

    result, pooled = proc.process(Mapped(bgra, pitch=width * 4 + pitch_extra),
                                  width, height, (left, 1, width, height), 0)

    np.testing.assert_array_equal(result, bgra[1:, left:])
    assert pooled is False


def test_a_device_buffer_is_filled_with_set_and_handed_back(cupy):
    cupy()
    bgra = image()
    height, width = bgra.shape[:2]
    device_buffer = np.zeros((height, width, 4), np.uint8).view(DeviceArray)

    result, pooled = CupyProcessor("BGRA").process(Mapped(bgra), width, height,
                                                   (0, 0, width, height), 0, device_buffer)

    assert result is device_buffer and pooled is True
    np.testing.assert_array_equal(result, bgra)


def test_a_full_turn_is_no_turn(cupy):
    cupy()
    bgra = image()
    height, width = bgra.shape[:2]
    result, _ = CupyProcessor("BGRA").process(Mapped(bgra), width, height,
                                              (0, 0, width, height), 360)
    np.testing.assert_array_equal(result, bgra)


class NullRect:
    Pitch = 16
    pBits = None


class UnreadableRect:
    Pitch = 16
    pBits = object()


@pytest.mark.parametrize("rect,region,buffer,message", [
    (NullRect(), (0, 0, 4, 4), None, "Invalid rect or pBits"),
    (UnreadableRect(), (0, 0, 4, 4), None, "valid pointer"),
    (None, (0, 0, 5, 4), None, "outside of the frame"),
    (None, (0, 0, 4, 4), np.zeros((4, 4, 3), np.uint8), "does not match region shape"),
])
def test_process_raises_rather_than_returning_a_blank_frame(cupy, caplog, rect, region, buffer, message):
    cupy()
    rect = rect if rect is not None else Mapped(image(4, 4))

    with caplog.at_level(logging.ERROR, logger="rapidshot.processor.cupy_processor"):
        with pytest.raises(ValueError, match=message):
            CupyProcessor("RGB").process(rect, 4, 4, region, 0, buffer)

    assert "Error processing frame with CuPy" in caplog.text


def test_a_conversion_that_changes_the_frame_size_is_reported(cupy, caplog, monkeypatch):
    cupy()
    bgra = image()
    height, width = bgra.shape[:2]
    proc = CupyProcessor("RGB")
    monkeypatch.setattr(proc, "process_cvtcolor", lambda array: np.zeros((1, 1, 3), np.uint8))

    with caplog.at_level(logging.WARNING, logger="rapidshot.processor.cupy_processor"):
        result, _ = proc.process(Mapped(bgra), width, height, (0, 0, width, height), 0)

    assert result.shape == (1, 1, 3)
    assert "changed height/width" in caplog.text
