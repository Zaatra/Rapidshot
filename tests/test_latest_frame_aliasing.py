"""get_latest_frame() must hand back memory the pool cannot take away.

The continuous-mode queue holds pooled buffers. `get_latest_frame()` used to
return the newest one's array without taking it out of the queue, so the
producer was free to evict that entry, release it, and let the next capture
write over the frame the caller was still reading -- and `stop()` did the same
to every queued buffer at once. Nothing raised; the pixels simply changed.

These drive the camera's queue directly rather than a real display: the bug is
in the handover between producer and consumer, and needs no DXGI to show.
"""

import collections
import threading

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool  # noqa: E402

SHAPE = (4, 6, 4)


def _camera(pool_size=2, max_buffer_len=64):
    """A ScreenCapture with only the continuous-mode queue wired up."""
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._capture_lock = threading.Lock()
    cam._frame_available_event = threading.Event()
    cam._pooled_frames_deque = collections.deque(maxlen=max_buffer_len)
    cam._last_dup_source = None
    cam.max_buffer_len = max_buffer_len
    cam.nvidia_gpu = False
    cam.memory_pool = NumpyMemoryPool(SHAPE, np.uint8, pool_size)
    return cam


def _queue(cam, fill):
    """Check a buffer out, fill it, queue it -- what the producer does."""
    buffer = cam.memory_pool.checkout()
    buffer.array[...] = fill
    with cam._capture_lock:
        cam._pooled_frames_deque.append(buffer)
        cam._last_dup_source = buffer
    cam._frame_available_event.set()
    return buffer


# --------------------------------------------------------------------------
# The aliasing itself
# --------------------------------------------------------------------------

def test_returned_frame_survives_the_buffer_being_recycled():
    """The bug, end to end: capture reuses the buffer, the frame must not move."""
    cam = _camera(pool_size=1)
    _queue(cam, 11)

    frame = cam.get_latest_frame()
    assert np.all(frame == 11)

    # The pool handed that buffer straight back out, and the next capture
    # overwrote it -- which is exactly what a one-buffer pool guarantees.
    recycled = cam.memory_pool.checkout()
    recycled.array[...] = 222

    assert np.all(frame == 11), (
        "the caller's frame changed when the pool recycled its buffer")


def test_returned_frame_does_not_alias_any_pool_buffer():
    cam = _camera()
    queued = _queue(cam, 7)

    frame = cam.get_latest_frame()

    assert not np.shares_memory(frame, queued.array)


def test_two_calls_return_independent_frames():
    """Diffing consecutive frames is the obvious use, and used to be unsound."""
    cam = _camera(pool_size=2)
    _queue(cam, 1)
    first = cam.get_latest_frame()
    _queue(cam, 2)
    second = cam.get_latest_frame()

    assert not np.shares_memory(first, second)
    assert np.all(first == 1)
    assert np.all(second == 2)


def test_stop_does_not_disturb_an_already_returned_frame():
    """stop() returns every queued buffer to the pool.

    One buffer in the pool, so the one stop() releases is necessarily the one
    the next capture writes into.
    """
    cam = _camera(pool_size=1)
    _queue(cam, 5)
    frame = cam.get_latest_frame()

    cam.is_capturing = False
    cam._capture_thread = None
    cam._stop_capture_event = threading.Event()
    cam._timer_handle = None
    cam._frame_count = 0
    cam.stop()

    recycled = cam.memory_pool.checkout()
    recycled.array[...] = 99
    assert np.all(frame == 5)


# --------------------------------------------------------------------------
# Ownership: the frame leaves the queue, so the pool gets it back
# --------------------------------------------------------------------------

def test_reading_a_frame_returns_its_buffer_to_the_pool():
    """Without this a pool of N stalls capture after N frames."""
    cam = _camera(pool_size=2)
    _queue(cam, 1)
    assert cam.memory_pool.get_stats()["available"] == 1

    cam.get_latest_frame()

    assert cam.memory_pool.get_stats()["available"] == 2


def test_the_buffer_accessor_transfers_ownership():
    cam = _camera(pool_size=2)
    _queue(cam, 3)

    frame = cam.get_latest_frame_buffer()

    assert cam.memory_pool.get_stats()["available"] == 1, (
        "the buffer is the caller's until they release it")
    with cam._capture_lock:
        assert not cam._pooled_frames_deque, (
            "a frame handed out must leave the queue, or the producer will "
            "recycle it under the caller")
    assert np.all(np.asarray(frame) == 3)

    frame.release()
    assert cam.memory_pool.get_stats()["available"] == 2


def test_handing_out_the_newest_frame_clears_the_duplication_source():
    """video_mode copies from the newest frame; it must not copy from a frame
    it has given away, whose buffer the caller may already have released."""
    cam = _camera(pool_size=2)
    queued = _queue(cam, 4)
    assert cam._last_dup_source is queued

    cam.get_latest_frame_buffer()

    assert cam._last_dup_source is None


# --------------------------------------------------------------------------
# Behaviour that must not regress
# --------------------------------------------------------------------------

def test_empty_queue_times_out_and_returns_none():
    cam = _camera()
    assert cam.get_latest_frame() is None


def test_a_stale_event_with_an_empty_queue_returns_none():
    cam = _camera()
    cam._frame_available_event.set()

    assert cam.get_latest_frame() is None
    assert not cam._frame_available_event.is_set()


def test_draining_the_queue_clears_the_event():
    """So the next call waits for a new frame instead of re-reading the old."""
    cam = _camera(pool_size=2)
    _queue(cam, 8)

    cam.get_latest_frame()

    assert not cam._frame_available_event.is_set()


def test_older_queued_frames_are_left_alone():
    """Only the newest is handed out; the queue is not drained."""
    cam = _camera(pool_size=3)
    _queue(cam, 1)
    _queue(cam, 2)

    frame = cam.get_latest_frame()

    assert np.all(frame == 2)
    with cam._capture_lock:
        assert len(cam._pooled_frames_deque) == 1
    assert cam._frame_available_event.is_set()


def test_plain_arrays_are_returned_as_copies_too():
    """Non-BGRA modes can queue a plain array; it has no release() to call."""
    cam = _camera()
    plain = np.full(SHAPE, 6, dtype=np.uint8)
    with cam._capture_lock:
        cam._pooled_frames_deque.append(plain)
    cam._frame_available_event.set()

    frame = cam.get_latest_frame()

    assert np.all(frame == 6)
    assert not np.shares_memory(frame, plain)
