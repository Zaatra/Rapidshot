"""`pool_output=False` must actually return unpooled frames.

It is the documented escape hatch for code that needs a true ndarray, and it
worked for every converting mode because conversion allocates anyway. BGRA
converts nothing, so its frame *is* the staging buffer -- and this handed that
buffer over still pooled. The caller got a `PooledBuffer` where the docs
promised an `ndarray`, and, told no release was needed, never released it. BGRA
has no allocating fallback, so capture stopped after exactly `pool_size_frames`
frames and returned None from then on, with nothing raised.

Found by `benchmarks/compare_libraries.py`, whose unpooled row reported "no
frames returned" for fullscreen BGRA while the region row -- which misses the
pool and so never touches it -- was fine.
"""

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool, PooledBuffer  # noqa: E402

SHAPE = (4, 6, 4)


class _Processor:
    def __init__(self, converts):
        self.converts_output = converts


def _camera(pool_output, converts):
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._pool_output = pool_output
    cam._processor = _Processor(converts)
    cam.nvidia_gpu = False
    cam.memory_pool = NumpyMemoryPool(SHAPE, np.uint8, 2)
    return cam


def _hand_back(cam, fill=7):
    """The tail of _grab_locked: a valid pooled buffer on the can_use_pool path."""
    buffer = cam.memory_pool.checkout()
    buffer.array[...] = fill
    if not cam._pool_output:
        try:
            return cam._frame_array(buffer).copy()
        finally:
            buffer.release()
    return buffer


def test_unpooled_bgra_returns_a_plain_array():
    cam = _camera(pool_output=False, converts=False)

    frame = _hand_back(cam)

    assert isinstance(frame, np.ndarray)
    assert not isinstance(frame, PooledBuffer)


def test_unpooled_bgra_returns_the_buffer_to_the_pool():
    """The caller was told not to release, so the library must."""
    cam = _camera(pool_output=False, converts=False)
    assert cam.memory_pool.get_stats()["available"] == 2

    _hand_back(cam)

    assert cam.memory_pool.get_stats()["available"] == 2


def test_unpooled_bgra_survives_more_grabs_than_the_pool_has_buffers():
    """The bug exactly: four frames at pool_size_frames=4, then nothing."""
    cam = _camera(pool_output=False, converts=False)

    frames = [_hand_back(cam, fill=i) for i in range(10)]

    assert len(frames) == 10
    assert all(f is not None for f in frames)


def test_unpooled_frames_do_not_alias_each_other():
    """A two-buffer pool recycles immediately; copies must not share storage."""
    cam = _camera(pool_output=False, converts=False)

    first = _hand_back(cam, fill=1)
    second = _hand_back(cam, fill=2)

    assert not np.shares_memory(first, second)
    assert first[0, 0, 0] == 1 and second[0, 0, 0] == 2


def test_pooled_bgra_still_hands_over_the_buffer():
    """The default path must keep its zero-copy behaviour."""
    cam = _camera(pool_output=True, converts=False)

    frame = _hand_back(cam)

    assert isinstance(frame, PooledBuffer)
    assert cam.memory_pool.get_stats()["available"] == 1
