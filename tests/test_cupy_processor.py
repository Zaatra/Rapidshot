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
import sys

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


# --------------------------------------------------------------------------
# frame aliasing, through the public API
# --------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform != "win32",
                    reason="captures the desktop; Windows-only")
@pytest.mark.parametrize("mode", ["RGBA", "RGB", "GRAY"])
def test_held_frames_do_not_share_storage(mode):
    """A frame the caller still holds must never be overwritten.

    Found by review, then reproduced: same-shape conversions (RGBA) copied the
    result back into the pooled staging buffer, so `grab()` returned a bare
    array pointing at storage the pool had already recycled. Holding six frames
    against a two-buffer pool gave two distinct allocations, and frame one had
    been overwritten by frame six.

    This is the frame-aliasing corruption ROADMAP section 5 records being fixed
    once on the NumPy path, reappearing on the CuPy one. It is invisible unless
    a consumer holds a frame for longer than the pool depth, which is exactly
    what a batching or async pipeline does — hence a deliberately shallow pool
    and more frames than it can serve.
    """
    import rapidshot

    camera = rapidshot.create(output_color=mode, nvidia_gpu=True,
                              pool_size_frames=2)
    held, pointers = [], []
    try:
        for _ in range(1500):
            frame = camera.grab()
            if frame is None:
                continue
            array = frame.array if hasattr(frame, "array") else frame
            held.append(frame)
            pointers.append(int(array.data.ptr))
            if len(held) >= 6:
                break
        if len(pointers) < 3:
            pytest.skip("not enough frames — the screen must be changing")

        assert len(set(pointers)) == len(pointers), (
            f"{mode}: {len(pointers) - len(set(pointers))} of {len(pointers)} "
            f"held frames share storage with an earlier one, so an earlier "
            f"frame was overwritten while the caller still held it")
    finally:
        for frame in held:
            release = getattr(frame, "release", None)
            if release:
                release()
        camera.release()
        rapidshot.reset()


# -- the fused GRAY kernel against the portable form ----------------------


@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (64, 64), (271, 373)])
def test_the_fused_gray_kernel_matches_the_portable_one(shape):
    """Two implementations of the same arithmetic now exist, and only because
    one of them has to run under NumPy so the class stays testable without a
    GPU. That duplication is only safe while they agree exactly -- the fused
    kernel is 6-10x faster, which is worth nothing if it shifts a level.

    Odd sizes included deliberately: an ElementwiseKernel over strided channel
    views is where a tail element would be missed.
    """
    from rapidshot.processor import cupy_processor as backend

    height, width = shape
    rng = np.random.default_rng(width * height)
    host = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    image = cp.asarray(host)

    kernel = backend._gray_kernel(cp)
    assert kernel is not None, "CuPy should provide ElementwiseKernel"
    fused = kernel(image[..., 0], image[..., 1], image[..., 2])[..., cp.newaxis]
    portable = backend._gray_chained(cp, image)

    assert fused.shape == portable.shape
    assert fused.dtype == portable.dtype
    assert cp.array_equal(fused, portable)


def test_numpy_standing_in_for_cupy_takes_the_portable_path():
    """The substitution `test_cupy_rotation` depends on: NumPy has no
    ElementwiseKernel, and asking for one must yield the fallback rather than
    an AttributeError from inside the conversion."""
    from rapidshot.processor import cupy_processor as backend

    assert backend._gray_kernel(np) is None
    host = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    assert backend._gray_chained(np, host).shape == (2, 3, 1)
