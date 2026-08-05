"""Tests for pooled converted-output buffers (ROADMAP.md 10).

Allocating the converted frame costs ~1.6 ms per 1080p frame — `np.empty` is
nearly free, but the page faults on first touch are not, and they exceed the
conversion they feed. Reusing a buffer removes that, measured at 1.98x on
`grab()`.

The cost is ownership: a reused buffer must not be recycled while a caller is
still reading it. That is the same hazard as the recycled pool buffers this
project already had to fix once, so the behaviour pinned here is mostly about
who owns what, and about the default staying exactly as it was.
"""

import ctypes

import numpy as np
import pytest

from rapidshot.processor.numpy_processor import NumpyProcessor

WIDTH, HEIGHT = 32, 24


class FakeMappedRect:
    def __init__(self, bgra):
        h, w, _ = bgra.shape
        self.Pitch = w * 4
        self._backing = (ctypes.c_ubyte * (self.Pitch * h))()
        view = np.ctypeslib.as_array(self._backing).reshape(h, self.Pitch)
        view[:, : w * 4] = bgra.reshape(h, w * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def frame(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)


def run(proc, bgra, target=None, dirty_rects=None, rotation=0):
    buffer = np.empty((HEIGHT, WIDTH, 4), np.uint8)
    return proc.process(
        FakeMappedRect(bgra), WIDTH, HEIGHT, (0, 0, WIDTH, HEIGHT),
        rotation, buffer, dirty_rects=dirty_rects, output_target=target,
    )


class TestOutputTarget:
    def test_the_target_is_filled_and_returned(self):
        proc = NumpyProcessor("RGB")
        target = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        out, pooled = run(proc, frame(1), target=target)

        assert out is target, "should hand back the buffer it was given"
        assert pooled is True, "the caller owns it and must release it"
        assert out.any()

    def test_target_and_allocation_agree(self):
        """Reusing a buffer must not change a single pixel."""
        proc = NumpyProcessor("RGB")
        source = frame(2)

        allocated, _ = run(proc, source)
        target = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
        filled, _ = run(proc, source, target=target)

        assert np.array_equal(allocated, filled)

    def test_a_wrong_shaped_target_is_refused(self):
        """Silently ignoring it would hand back a frame the caller cannot use."""
        proc = NumpyProcessor("RGB")
        with pytest.raises(ValueError, match="output_target shape"):
            run(proc, frame(1), target=np.zeros((HEIGHT, WIDTH, 4), np.uint8))

    def test_no_target_still_allocates(self):
        """The default path must be untouched."""
        proc = NumpyProcessor("RGB")
        out, pooled = run(proc, frame(1))
        assert isinstance(out, np.ndarray)
        assert pooled is False

    def test_rotation_declines_the_target(self):
        """Rotation produces a differently shaped array than the target holds."""
        proc = NumpyProcessor("RGB")
        target = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        out, pooled = run(proc, frame(1), target=target, rotation=90)

        assert out is not target
        assert pooled is False, "an unused target must not be reported as owned"

    def test_the_accumulator_path_also_fills_the_target(self):
        """Both routes out of process() have to honour it, or one leaks."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])   # seed
        target = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

        out, pooled = run(proc, frame(2), target=target, dirty_rects=[(0, 0, 8, 8)])

        assert out is target
        assert pooled is True

    def test_bgra_ignores_the_target(self):
        """BGRA already returns the staging buffer with no copy at all."""
        proc = NumpyProcessor("BGRA")
        source = frame(3)
        target = np.zeros((HEIGHT, WIDTH, 4), np.uint8)

        out, pooled = run(proc, source, target=target)

        assert np.array_equal(out, source)
        assert out is not target


class TestPooledBufferIsArrayLike:
    """Since 2.0 `grab()` returns one of these, so it has to behave like a frame.

    Without this, making pooling the default would force every consumer to learn
    about pooling before it could read a pixel.
    """

    def _buffer(self):
        from rapidshot.memory_pool import NumpyMemoryPool

        pool = NumpyMemoryPool((HEIGHT, WIDTH, 3), np.uint8, 2)
        wrapper = pool.checkout()
        wrapper.array[:] = np.arange(HEIGHT * WIDTH * 3, dtype=np.uint8).reshape(
            HEIGHT, WIDTH, 3)
        return wrapper

    def test_indexing_reads_the_frame(self):
        buf = self._buffer()
        assert buf[0, 0, 0] == buf.array[0, 0, 0]
        assert np.array_equal(buf[5:9], buf.array[5:9])

    def test_numpy_sees_it_as_an_array(self):
        """cv2, PIL and every ndarray-consuming API go through this."""
        buf = self._buffer()
        assert np.array_equal(np.asarray(buf), buf.array)
        assert np.asarray(buf).shape == (HEIGHT, WIDTH, 3)

    def test_asarray_is_a_view_not_a_copy(self):
        """Copying here would give back the cost pooling exists to avoid."""
        buf = self._buffer()
        assert np.asarray(buf).base is buf.array or np.asarray(buf) is buf.array

    def test_the_usual_attributes_are_present(self):
        buf = self._buffer()
        assert buf.shape == (HEIGHT, WIDTH, 3)
        assert buf.dtype == np.uint8
        assert buf.ndim == 3
        assert buf.size == HEIGHT * WIDTH * 3
        assert len(buf) == HEIGHT

    def test_copy_survives_release(self):
        buf = self._buffer()
        kept = buf.copy()
        expected = kept.copy()
        buf.release()
        assert np.array_equal(kept, expected)

    def test_use_after_release_is_refused(self):
        """The buffer now belongs to the next frame; reading it silently would
        hand back someone else's pixels."""
        from rapidshot.memory_pool import BufferReleasedError

        buf = self._buffer()
        buf.release()

        with pytest.raises(BufferReleasedError):
            buf[0, 0, 0]
        with pytest.raises(BufferReleasedError):
            np.asarray(buf)

    def test_the_error_says_what_to_do(self):
        from rapidshot.memory_pool import BufferReleasedError

        buf = self._buffer()
        buf.release()
        with pytest.raises(BufferReleasedError) as excinfo:
            len(buf)
        assert "copy" in str(excinfo.value).lower()


class TestCaptureIntegration:
    """The checkout logic in ScreenCapture, without needing a screen."""

    def _capture(self, pool_output=True, color="RGB", size=4):
        from rapidshot.capture import ScreenCapture
        from rapidshot.processor.base import Processor

        cam = ScreenCapture.__new__(ScreenCapture)
        cam._pool_output = pool_output
        cam._output_pool = None
        cam._output_pool_size = size
        cam._processor = Processor(backend=None, output_color=color)
        return cam

    def test_disabled_by_default_hands_out_nothing(self):
        cam = self._capture(pool_output=False)
        assert cam._checkout_output_buffer(WIDTH, HEIGHT) is None

    def test_enabled_hands_out_a_correctly_shaped_buffer(self):
        cam = self._capture()
        wrapper = cam._checkout_output_buffer(WIDTH, HEIGHT)
        assert wrapper is not None
        assert wrapper.array.shape == (HEIGHT, WIDTH, 3)

    def test_bgra_never_pools_the_output(self):
        """There is no conversion, so there is nothing to pool."""
        cam = self._capture(color="BGRA")
        assert cam._checkout_output_buffer(WIDTH, HEIGHT) is None

    def test_released_buffers_come_back(self):
        cam = self._capture(size=2)
        for _ in range(10):
            wrapper = cam._checkout_output_buffer(WIDTH, HEIGHT)
            assert wrapper is not None
            wrapper.release()

    def test_exhaustion_falls_back_to_allocating(self):
        """Blocking capture, or recycling a buffer in use, would both be worse."""
        cam = self._capture(size=2)
        held = [cam._checkout_output_buffer(WIDTH, HEIGHT) for _ in range(2)]
        assert all(h is not None for h in held)

        assert cam._checkout_output_buffer(WIDTH, HEIGHT) is None

        held[0].release()
        assert cam._checkout_output_buffer(WIDTH, HEIGHT) is not None

    def test_a_changed_frame_size_rebuilds_the_pool(self):
        """Buffers of the previous size can hold nothing useful."""
        cam = self._capture()
        first = cam._checkout_output_buffer(WIDTH, HEIGHT)
        first.release()

        second = cam._checkout_output_buffer(WIDTH * 2, HEIGHT * 2)

        assert second.array.shape == (HEIGHT * 2, WIDTH * 2, 3)


# --------------------------------------------------------------------------
# pool_size_frames is public and validated (ROADMAP.md 10 — memory footprint)
# --------------------------------------------------------------------------

def test_pool_size_frames_is_reachable_from_create():
    """It was settable on ScreenCapture but the factory never forwarded it.

    That made the largest tunable part of the process footprint unreachable
    without constructing the class by hand -- the same gap timeout_ms had.
    """
    import inspect

    import rapidshot

    for fn in (rapidshot.create, rapidshot.RapidshotFactory.create):
        params = inspect.signature(fn).parameters
        assert "pool_size_frames" in params, fn
        assert params["pool_size_frames"].default == 4


def test_pool_default_is_four_not_ten():
    """Measured: 10 buffers cost 173.6 MB per camera, 4 cost 113.6 MB.

    Dropping the default saved 60 MB for -1.8% frame rate, which is inside
    run-to-run noise. Pinned so the number is a decision rather than a drift.
    """
    import inspect

    from rapidshot.capture import ScreenCapture

    assert inspect.signature(ScreenCapture.__init__).parameters[
        "pool_size_frames"].default == 4


@pytest.mark.parametrize("bad", [0, -1, 2.5, "4", None, True])
def test_pool_size_frames_rejects_nonsense(bad):
    """Zero is rejected too: an empty pool is not a smaller pool, it is a
    permanent fallback to allocating, which looks like the setting was ignored.

    `True` is here for the same reason as in timeout_ms -- bool subclasses int,
    so a naive check accepts it and silently configures a 1-buffer pool.
    """
    from rapidshot.capture import ScreenCapture

    class FakeOutput:
        devicename = "FAKE"
        resolution = (1920, 1080)
        rotation_angle = 0

        def update_desc(self):
            pass

    with pytest.raises(ValueError, match="pool_size_frames"):
        ScreenCapture(output=FakeOutput(), device=None, pool_size_frames=bad)
