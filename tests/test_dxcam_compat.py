"""The DXcam compatibility layer, exercised against a fake camera.

No capture, no GPU, no desktop: the layer's whole job is translating between two
Python APIs, and that translation is where its bugs live. The interesting one is
buffer lifetime -- DXcam callers never release anything, RapidShot requires it,
and getting that wrong corrupts data in code that looks correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from rapidshot.dxcam_compat import DXCamera


class FakeBuffer:
    """Stands in for a RapidShot pooled buffer.

    Reading after release raises, exactly as the real one does, so a test that
    hands out a stale view fails here instead of silently passing.
    """

    def __init__(self, value: int, shape=(2, 2, 3)):
        self._array = np.full(shape, value, dtype=np.uint8)
        self.released = False

    def __array__(self, dtype=None, copy=None):
        if self.released:
            raise AssertionError("read after release")
        return self._array if dtype is None else self._array.astype(dtype)

    @property
    def shape(self):
        return self._array.shape

    def release(self):
        self.released = True


class FakeCamera:
    def __init__(self, frames=None):
        self.frames = list(frames or [])
        self.handed_out = []
        self.started = None
        self.stopped = False
        self.released = False
        self.width, self.height, self.channels = 2, 2, 3
        self.region = (0, 0, 2, 2)
        self.is_capturing = False
        self.last_present_time = 0

    def _next(self):
        if not self.frames:
            return None
        frame = self.frames.pop(0)
        if frame is not None:
            self.handed_out.append(frame)
        return frame

    def grab(self, region=None):
        return self._next()

    def get_latest_frame(self):
        return self._next()

    def start(self, **kwargs):
        self.started = kwargs

    def stop(self):
        self.stopped = True

    def release(self):
        self.released = True


class TestGrabCopies:
    def test_returns_a_plain_ndarray(self):
        camera = DXCamera(FakeCamera([FakeBuffer(7)]))
        frame = camera.grab()
        assert isinstance(frame, np.ndarray)
        assert frame.tolist() == np.full((2, 2, 3), 7, dtype=np.uint8).tolist()

    def test_releases_the_buffer_immediately(self):
        """DXcam code never calls release, so the shim must.

        Without this the pool starves after a handful of frames and capture
        falls back to allocating -- correct, but slower, and silently so.
        """
        buffer = FakeBuffer(1)
        camera = DXCamera(FakeCamera([buffer]))
        camera.grab()
        assert buffer.released is True

    def test_the_copy_survives_the_release(self):
        """The returned array must not alias a recycled buffer.

        This is the bug the copy exists to prevent: DXcam callers keep frames,
        and a pooled buffer's contents are overwritten by the next capture.
        """
        buffer = FakeBuffer(3)
        camera = DXCamera(FakeCamera([buffer]))
        frame = camera.grab()
        buffer._array[:] = 99          # simulate the pool reusing it
        assert frame.tolist() == np.full((2, 2, 3), 3, dtype=np.uint8).tolist()

    def test_none_passes_through(self):
        assert DXCamera(FakeCamera([None])).grab() is None

    def test_releases_even_if_conversion_fails(self):
        class Hostile(FakeBuffer):
            def __array__(self, dtype=None, copy=None):
                raise ValueError("no")

        buffer = Hostile(1)
        camera = DXCamera(FakeCamera([buffer]))
        with pytest.raises(ValueError):
            camera.grab()
        assert buffer.released is True


class TestGrabViewIsZeroCopy:
    def test_view_aliases_the_buffer(self):
        """grab_view must not copy; that is its entire purpose."""
        buffer = FakeBuffer(5)
        camera = DXCamera(FakeCamera([buffer]))
        view = camera.grab_view()
        assert buffer.released is False
        buffer._array[0, 0, 0] = 42
        assert view[0, 0, 0] == 42

    def test_next_grab_retires_the_previous_view(self):
        """DXcam's contract: a view is valid until the next grab."""
        first, second = FakeBuffer(1), FakeBuffer(2)
        camera = DXCamera(FakeCamera([first, second]))
        camera.grab_view()
        assert first.released is False
        camera.grab_view()
        assert first.released is True
        assert second.released is False

    def test_release_retires_the_outstanding_view(self):
        buffer = FakeBuffer(1)
        camera = DXCamera(FakeCamera([buffer]))
        camera.grab_view()
        camera.release()
        assert buffer.released is True
        assert camera.is_released is True

    def test_stop_retires_the_outstanding_view(self):
        buffer = FakeBuffer(1)
        camera = DXCamera(FakeCamera([buffer]))
        camera.grab_view()
        camera.stop()
        assert buffer.released is True

    def test_a_copying_grab_also_retires_a_view(self):
        """Mixing grab_view and grab must not leak the view's buffer."""
        view_buffer, copy_buffer = FakeBuffer(1), FakeBuffer(2)
        camera = DXCamera(FakeCamera([view_buffer, copy_buffer]))
        camera.grab_view()
        camera.grab()
        assert view_buffer.released is True


class TestPassThrough:
    def test_start_forwards_its_arguments(self):
        inner = FakeCamera()
        DXCamera(inner).start(target_fps=30, video_mode=True)
        assert inner.started == {"target_fps": 30, "video_mode": True}

    def test_region_is_only_forwarded_when_given(self):
        """RapidShot's create() and start() treat absent and None differently."""
        inner = FakeCamera()
        DXCamera(inner).start(target_fps=60)
        assert "region" not in inner.started

    def test_channel_size_maps_to_channels(self):
        assert DXCamera(FakeCamera()).channel_size == 3

    def test_unknown_attributes_forward_to_the_camera(self):
        inner = FakeCamera()
        inner.something_rapidshot_specific = 11
        assert DXCamera(inner).something_rapidshot_specific == 11

    def test_missing_attributes_still_raise(self):
        with pytest.raises(AttributeError):
            DXCamera(FakeCamera()).definitely_not_a_real_method

    def test_latest_frame_time_is_zero_without_a_timestamp(self):
        assert DXCamera(FakeCamera()).latest_frame_time == 0.0

    def test_the_wrapped_camera_stays_reachable(self):
        """Migration is incremental: callers need the real object back."""
        inner = FakeCamera()
        assert DXCamera(inner).rapidshot_camera is inner
