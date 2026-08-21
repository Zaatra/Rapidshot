"""CupyProcessor colour conversion, against the NumPy path it must match.

These exist because `CupyProcessor` had no test coverage at all while being
reachable from the public API via `create(nvidia_gpu=True)`. It was written
against OpenCV, which is not a RapidShot dependency, so on any machine without
`cv2` every non-BGRA mode failed — and `process()` swallowed the failure and
returned the unconverted BGRA buffer as though it had succeeded. A caller
asking for RGB got a 4-channel BGRA array and no exception.

The byte-exactness requirement is the point of the whole file: `nvidia_gpu` is
a performance switch, so turning it on must not change a single pixel.
"""
import numpy as np
import pytest

from rapidshot.processor.numpy_processor import NumpyProcessor

cp = pytest.importorskip("cupy", reason="CuPy not installed")

try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # pragma: no cover - driver present but unusable
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

from rapidshot.processor.cupy_processor import CupyProcessor  # noqa: E402

MODES = ["BGRA", "RGB", "BGR", "RGBA", "GRAY"]


def bgra_pattern(height=17, width=23):
    """Deliberately not a multiple of any vector width, and fully saturated.

    Includes 0 and 255 in every channel: the luma rounding term only shows up
    at the extremes, and a mid-grey pattern would agree either way.
    """
    rng = np.random.default_rng(20260806)
    pattern = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    pattern[0, 0] = [0, 0, 0, 0]
    pattern[0, 1] = [255, 255, 255, 255]
    pattern[1, 0] = [255, 0, 0, 255]
    pattern[1, 1] = [0, 255, 0, 255]
    pattern[1, 2] = [0, 0, 255, 255]
    return pattern


def numpy_reference(pattern, mode):
    processor = NumpyProcessor(mode)
    channels = 1 if mode == "GRAY" else (4 if mode in ("BGRA", "RGBA") else 3)
    out = np.empty((*pattern.shape[:2], channels), dtype=np.uint8)
    processor.convert_into(pattern, out)
    return out


@pytest.mark.parametrize("mode", MODES)
def test_matches_numpy_processor_exactly(mode):
    pattern = bgra_pattern()
    expected = numpy_reference(pattern, mode)

    processor = CupyProcessor(mode)
    got = cp.asnumpy(processor.process_cvtcolor(cp.asarray(pattern)))

    assert got.shape == expected.shape, f"{mode}: shape differs"
    assert got.dtype == expected.dtype, f"{mode}: dtype differs"
    assert np.array_equal(got, expected), (
        f"{mode}: pixels differ from the NumPy path — turning nvidia_gpu on "
        f"must not change any pixel")


@pytest.mark.parametrize("mode", MODES)
def test_output_channel_count(mode):
    expected = {"BGRA": 4, "RGBA": 4, "RGB": 3, "BGR": 3, "GRAY": 1}[mode]
    processor = CupyProcessor(mode)
    got = processor.process_cvtcolor(cp.asarray(bgra_pattern()))
    assert got.shape[2] == expected


def test_gray_rounding_is_not_truncation():
    """The +128 term. Without it every pixel biases dark, which averages out
    across a random image and hides in any mean-based comparison."""
    pattern = np.zeros((1, 4, 4), np.uint8)
    pattern[0, 0] = [1, 1, 1, 255]      # rounds up to 1, truncates to 0
    pattern[0, 1] = [128, 128, 128, 255]
    pattern[0, 2] = [255, 255, 255, 255]
    pattern[0, 3] = [0, 0, 0, 255]

    got = cp.asnumpy(CupyProcessor("GRAY").process_cvtcolor(cp.asarray(pattern)))
    assert np.array_equal(got, numpy_reference(pattern, "GRAY"))
    assert got[0, 2, 0] == 255, "white must stay white"
    assert got[0, 3, 0] == 0, "black must stay black"


def test_unsupported_mode_rejected_at_construction():
    """Not on the first frame that happens to arrive — a bad configuration on a
    static desktop would otherwise look fine until something moved."""
    with pytest.raises(ValueError, match="Unsupported color mode"):
        CupyProcessor("YUV")


def test_conversion_needs_no_opencv(monkeypatch):
    """OpenCV is not a RapidShot dependency, and this path must not need it.

    Guards the original defect directly: import cv2 is made to fail, and every
    mode must still convert.
    """
    import builtins

    real_import = builtins.__import__

    def no_cv2(name, *args, **kwargs):
        if name in ("cv2", "cucv", "cucv.cv2"):
            raise ImportError(f"{name} blocked by test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cv2)

    pattern = bgra_pattern()
    for mode in MODES:
        got = cp.asnumpy(CupyProcessor(mode).process_cvtcolor(cp.asarray(pattern)))
        assert np.array_equal(got, numpy_reference(pattern, mode)), mode
