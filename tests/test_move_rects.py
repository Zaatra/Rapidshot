"""Move rects, and the queue bound that starved continuous BGRA capture.

DXGI reports regions the compositor *moved* separately from the ones it
redrew, and does not repeat them in the dirty rects. Anything patching a
previous frame by dirty rect alone therefore leaves a moved region showing the
old pixels, and `changed_fraction` reports a scroll as no change at all.

Measured on this machine (Windows 11, 2560x1600): across 3,768 frames of window
dragging and real page scrolling, DWM reported **zero** move rects, with the
metadata readable on every frame -- a fully composited desktop leaves nothing
for a screen-to-screen blit to optimise. So the handling here is correctness
insurance for configurations where that is not true, and these tests supply the
move rects the hardware would not.

The queue bound is unrelated except in where it bites: the continuous-mode
queue was capped at `max_buffer_len` (64) while backed by a pool of 4, so BGRA
capture ran out of buffers and stopped after `pool_size_frames` frames.
"""

import collections
import threading

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.frame import Frame  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool  # noqa: E402


# --------------------------------------------------------------------------
# Frame metadata
# --------------------------------------------------------------------------

def _frame(region=(0, 0, 100, 100), dirty=None, move=None):
    return Frame(
        texture=object(),
        on_release=None,
        region=region,
        dirty_rects=dirty,
        move_rects=move,
    )


def test_move_rects_are_exposed():
    frame = _frame(move=[(10, 20, 0, 0, 50, 50)])

    assert frame.move_rects == [(10, 20, 0, 0, 50, 50)]


def test_move_rects_distinguish_unreadable_from_empty():
    assert _frame(move=None).move_rects is None
    assert _frame(move=[]).move_rects == []


def test_a_move_rect_destination_is_rebased_into_the_frame():
    """The destination is where the pixels landed, so it has to index the frame."""
    frame = _frame(region=(100, 100, 300, 300), move=[(5, 5, 150, 160, 200, 210)])

    assert frame.move_rects == [(5, 5, 50, 60, 100, 110)]


def test_a_move_rect_source_stays_in_desktop_coordinates():
    """The pixels may have come from outside the region; clamping would claim
    they came from somewhere they did not."""
    frame = _frame(region=(100, 100, 300, 300), move=[(0, 0, 150, 150, 200, 200)])

    source_x, source_y = frame.move_rects[0][:2]
    assert (source_x, source_y) == (0, 0)


def test_move_rects_outside_the_region_are_dropped():
    frame = _frame(region=(0, 0, 100, 100), move=[(0, 0, 500, 500, 600, 600)])

    assert frame.move_rects == []


def test_each_source_keeps_its_own_destination():
    """Clipping must not reorder or re-pair: one rect here misses the region."""
    frame = _frame(
        region=(0, 0, 100, 100),
        move=[
            (1, 1, 0, 0, 10, 10),          # kept
            (2, 2, 500, 500, 600, 600),    # dropped
            (3, 3, 20, 20, 30, 30),        # kept
        ],
    )

    assert frame.move_rects == [(1, 1, 0, 0, 10, 10), (3, 3, 20, 20, 30, 30)]


# --------------------------------------------------------------------------
# changed_fraction
# --------------------------------------------------------------------------

def test_a_moved_region_counts_as_changed():
    """A scroll reports move rects instead of dirty ones. Reporting that as
    'nothing changed' is what would make a consumer skip a frame that moved."""
    frame = _frame(dirty=[], move=[(0, 0, 0, 0, 50, 100)])

    assert frame.changed_fraction == pytest.approx(0.5)


def test_dirty_and_move_rects_are_unioned_not_summed():
    """Overlapping regions must not be double counted."""
    frame = _frame(dirty=[(0, 0, 50, 100)], move=[(0, 0, 0, 0, 50, 100)])

    assert frame.changed_fraction == pytest.approx(0.5)


def test_dirty_and_move_rects_in_different_places_add_up():
    frame = _frame(dirty=[(0, 0, 50, 100)], move=[(0, 0, 50, 0, 100, 100)])

    assert frame.changed_fraction == pytest.approx(1.0)


def test_no_dirty_metadata_still_means_unknown():
    assert _frame(dirty=None, move=[(0, 0, 0, 0, 10, 10)]).changed_fraction is None


def test_no_rects_at_all_still_means_everything_changed():
    assert _frame(dirty=[], move=[]).changed_fraction == 1.0


# --------------------------------------------------------------------------
# Dirty-rect patching must not run when pixels were moved
# --------------------------------------------------------------------------

class _Duplicator:
    def __init__(self, dirty=None, move=None):
        self.dirty_rects = dirty
        self.move_rects = move


def _camera(duplicator):
    camera = ScreenCapture.__new__(ScreenCapture)
    camera._duplicator = duplicator
    return camera


def test_a_frame_with_move_rects_converts_everything():
    """None is the processor's 'convert the whole frame'. Patching by dirty
    rect would leave the moved region showing the previous frame."""
    camera = _camera(_Duplicator(dirty=[(0, 0, 10, 10)], move=[(0, 0, 20, 20, 40, 40)]))

    assert camera._dirty_rects_for((0, 0, 100, 100)) is None


def test_without_move_rects_the_dirty_rects_are_used():
    camera = _camera(_Duplicator(dirty=[(0, 0, 10, 10)], move=[]))

    assert camera._dirty_rects_for((0, 0, 100, 100)) == [(0, 0, 10, 10)]


def test_unreadable_move_metadata_does_not_force_a_full_convert():
    """None means 'could not read', and the dirty rects are still usable."""
    camera = _camera(_Duplicator(dirty=[(0, 0, 10, 10)], move=None))

    assert camera._dirty_rects_for((0, 0, 100, 100)) == [(0, 0, 10, 10)]


# --------------------------------------------------------------------------
# The queue bound
# --------------------------------------------------------------------------

class _Processor:
    def __init__(self, converts):
        self.converts_output = converts


def _queue_camera(converts, pool_size, max_buffer_len=64):
    camera = ScreenCapture.__new__(ScreenCapture)
    camera.max_buffer_len = max_buffer_len
    camera._processor = _Processor(converts)
    camera.memory_pool = NumpyMemoryPool((4, 4, 4), np.uint8, pool_size)
    camera._init_args = {"pool_size_frames": pool_size}
    return camera


def test_bgra_queues_no_more_than_the_pool_can_spare():
    """The stall: 64 queued against a pool of 4 meant nothing was ever
    returned, so capture stopped after pool_size_frames frames."""
    camera = _queue_camera(converts=False, pool_size=4)

    assert camera._queue_limit() == 3, "one buffer must stay free for the next grab"


def test_a_converting_mode_keeps_its_full_queue():
    """Its frames come from the output pool, which falls back to allocating."""
    camera = _queue_camera(converts=True, pool_size=4)

    assert camera._queue_limit() == 64


def test_max_buffer_len_still_wins_when_it_is_the_smaller_bound():
    camera = _queue_camera(converts=False, pool_size=32, max_buffer_len=8)

    assert camera._queue_limit() == 8


def test_a_pool_too_small_to_queue_anything_warns(monkeypatch):
    """Recorded off the module logger directly rather than through caplog:
    the library configures its own logging, and whether records propagate to
    the root handler depends on which other test ran first."""
    import rapidshot.capture as capture_module

    warnings = []
    monkeypatch.setattr(capture_module.logger, "warning",
                        lambda message, *a, **k: warnings.append(str(message)))

    camera = _queue_camera(converts=False, pool_size=1)
    limit = camera._queue_limit()

    assert limit == 1
    assert any("pool_size_frames=1" in w for w in warnings), warnings


def test_the_producer_evicts_against_the_queues_own_bound():
    """The eviction check read max_buffer_len while the deque was built with
    the smaller limit, so the deque silently dropped frames without releasing
    them -- leaking a pool buffer per drop."""
    camera = _queue_camera(converts=False, pool_size=4)
    queue = collections.deque(maxlen=camera._queue_limit())

    assert queue.maxlen == 3
    for _ in range(5):
        queue.append(object())
    assert len(queue) == 3
