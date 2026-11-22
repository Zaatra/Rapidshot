"""Continuous mode's queue policies, freshness bound and counters (ROADMAP § 7.4c).

`target_fps` describes the producer; a consumer cares how old its frame is. At
240 fps capture into 60 fps inference, `policy="latest"` must not queue four
stale images but replace them, `max_age_ms` must refuse a frame too old to act
on, and `stream_stats` must say which of those happened, separately.

These drive the capture thread with a scripted grab and a fake duplicator, as
`test_remaining_branches.py` does: the policies live in the handover between
producer and consumer, and need no display to show. The clock is QPC, patched
to tick in milliseconds so ages are exact.
"""

import collections
import threading
import time

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot import frame as frame_module  # noqa: E402
from rapidshot.capture import QUEUE_POLICIES, ScreenCapture  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool  # noqa: E402

SHAPE = (4, 6, 4)


class FakeDuplicator:
    last_present_time = 0
    accumulated_frames = 0


@pytest.fixture
def clock(monkeypatch):
    """QPC at 1000 ticks a second, so one tick is one millisecond."""
    now = {"qpc": 10_000}
    monkeypatch.setattr(frame_module, "_qpc_freq", lambda: 1000)
    monkeypatch.setattr(frame_module, "_qpc_now", lambda: now["qpc"])
    return now


def _camera(policy="drop_oldest", max_buffer_len=8, max_age_ms=None, pool_size=8):
    """A ScreenCapture with only continuous mode wired up, as start() leaves it."""
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._capture_lock = threading.Lock()
    cam._frame_available_event = threading.Event()
    cam._stop_capture_event = threading.Event()
    cam._last_dup_source = None
    cam.max_buffer_len = max_buffer_len
    cam.nvidia_gpu = False
    cam.memory_pool = NumpyMemoryPool(SHAPE, np.uint8, pool_size)
    cam.region = (0, 0, SHAPE[1], SHAPE[0])
    cam._capture_permanently_failed = False
    cam._needs_reinit = False
    cam._last_capture_error_message = ""
    cam._frame_count = 0
    cam._duplicator = FakeDuplicator()
    cam._queue_policy = policy
    cam._max_age_ms = max_age_ms
    cam._pooled_frames_deque = collections.deque(maxlen=cam._queue_limit())
    return cam


def _produce(cam, script, video_mode=False, clock=None):
    """Run the capture thread over `script`, then stop it.

    Each step is ``(fill, present_qpc, accumulated_frames)`` for a new frame,
    or None for a tick with nothing new. Returns how many times it grabbed.
    With ``clock``, each frame is captured the moment it was presented, so it
    goes stale as the clock is moved on afterwards -- the way a frame does once
    capture stops confirming it.
    """
    steps = iter(script)
    grabs = []

    def grab(region):
        try:
            step = next(steps)
        except StopIteration:
            cam._stop_capture_event.set()
            return None
        grabs.append(step)
        # An idle tick is a poll that saw the screen unchanged, as
        # _grab_locked reports it.
        cam._last_grab_unchanged = step is None
        if step is None:
            return None
        fill, present_qpc, accumulated = step
        if clock is not None and present_qpc:
            clock["qpc"] = present_qpc
        cam._duplicator.last_present_time = present_qpc
        cam._duplicator.accumulated_frames = accumulated
        buffer = cam.memory_pool.checkout()
        buffer.array[...] = fill
        return buffer

    cam._grab = grab
    cam._stop_capture_event.clear()
    cam._capture_thread_func(cam.region, target_fps=0, video_mode=video_mode)
    return len(grabs)


def _fills(cam):
    with cam._capture_lock:
        return [int(np.asarray(getattr(f, "array", f)).flat[0])
                for f in cam._pooled_frames_deque]


# --------------------------------------------------------------------------
# Arguments
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["newest", "", "LATEST", 1])  # None means the default
def test_an_unknown_policy_is_refused(bad):
    cam = ScreenCapture.__new__(ScreenCapture)
    with pytest.raises(ValueError, match="policy must be one of"):
        cam.start(policy=bad)


def test_policy_all_is_refused_with_the_reason_and_the_alternative():
    cam = ScreenCapture.__new__(ScreenCapture)
    with pytest.raises(ValueError, match="'block'"):
        cam.start(policy="all")


@pytest.mark.parametrize("bad", [0, -5, True, "10", float("nan")])
def test_max_age_ms_must_be_a_positive_number(bad):
    cam = ScreenCapture.__new__(ScreenCapture)
    with pytest.raises(ValueError, match="max_age_ms"):
        cam.start(max_age_ms=bad)


def test_latest_cannot_be_combined_with_video_mode():
    cam = ScreenCapture.__new__(ScreenCapture)
    with pytest.raises(ValueError, match="video_mode"):
        cam.start(policy="latest", video_mode=True)


def test_the_policies_are_the_ones_documented():
    assert QUEUE_POLICIES == ("drop_oldest", "latest", "block")


def _startable(monkeypatch):
    """A queue-only camera whose start() runs for real but whose capture
    thread does nothing."""
    cam = _camera()
    cam.is_capturing = False
    cam._capture_thread = None
    cam._validate_region = lambda region: None
    monkeypatch.setattr(cam, "_capture_thread_func", lambda *a: None)
    return cam


@pytest.mark.parametrize("video_mode, expected, maxlen", [
    (False, "latest", 1),
    (True, "drop_oldest", 8),
])
def test_start_without_a_policy_uses_the_default(monkeypatch, video_mode, expected, maxlen):
    """latest by default since 2.8; video_mode keeps drop_oldest, which it
    needs and latest refuses -- so start(video_mode=True), as the examples
    call it, still works."""
    cam = _startable(monkeypatch)

    cam.start(target_fps=60, video_mode=video_mode)
    cam._capture_thread.join(5)

    assert cam._queue_policy == expected
    assert cam._pooled_frames_deque.maxlen == maxlen
    assert cam.stream_stats["policy"] == expected


def test_an_explicit_policy_overrides_the_default(monkeypatch):
    cam = _startable(monkeypatch)

    cam.start(target_fps=60, policy="drop_oldest")
    cam._capture_thread.join(5)

    assert cam._queue_policy == "drop_oldest"
    assert cam._pooled_frames_deque.maxlen == 8


def test_the_dxcam_shim_gets_dxcams_behaviour(monkeypatch):
    """DXcam's get_latest_frame() waits for a newer frame rather than
    returning an older one; the shim passes no policy, so it now gets that."""
    from rapidshot.dxcam_compat import DXCamera

    cam = _startable(monkeypatch)
    DXCamera(cam).start(target_fps=60)
    cam._capture_thread.join(5)

    assert cam._queue_policy == "latest"


# --------------------------------------------------------------------------
# policy="latest"
# --------------------------------------------------------------------------

def test_latest_holds_one_frame_whatever_max_buffer_len_says():
    cam = _camera(policy="latest", max_buffer_len=64)
    assert cam._pooled_frames_deque.maxlen == 1


def test_latest_replaces_the_waiting_frame_and_returns_its_buffer(clock):
    cam = _camera(policy="latest", pool_size=3)

    _produce(cam, [(1, 9_990, 1), (2, 9_995, 1), (3, 9_999, 1)])

    assert _fills(cam) == [3]
    stats = cam.stream_stats
    assert stats["frames_captured"] == 3
    assert stats["stale_frames_replaced"] == 2
    assert stats["consumer_drops"] == 0
    assert cam.memory_pool.get_stats()["available"] == 2, (
        "a replaced frame's buffer must go back to the pool, or capture "
        "stalls once the pool runs dry")


def test_latest_never_hands_out_a_frame_older_than_the_last_one(clock):
    """The drop_oldest queue hands a fast consumer older frames after a newer
    one; latest must not."""
    cam = _camera(policy="latest")
    _produce(cam, [(1, 9_990, 1), (2, 9_995, 1)])

    assert int(cam.get_latest_frame().flat[0]) == 2
    assert not cam._frame_available_event.is_set(), (
        "the next call must wait for a new frame, not find an older one")


def test_drop_oldest_still_leaves_older_frames_for_the_next_call(clock):
    """The default is unchanged: the contrast with the test above."""
    cam = _camera()
    _produce(cam, [(1, 9_990, 1), (2, 9_995, 1)])

    assert int(cam.get_latest_frame().flat[0]) == 2
    assert int(cam.get_latest_frame().flat[0]) == 1


# --------------------------------------------------------------------------
# policy="drop_oldest", the default
# --------------------------------------------------------------------------

def test_a_full_queue_drops_its_oldest_frame_and_counts_it(clock):
    cam = _camera(max_buffer_len=2, pool_size=4)

    _produce(cam, [(1, 9_990, 1), (2, 9_992, 1), (3, 9_994, 1), (4, 9_996, 1)])

    assert _fills(cam) == [3, 4]
    stats = cam.stream_stats
    assert stats["consumer_drops"] == 2
    assert stats["stale_frames_replaced"] == 0
    assert stats["policy"] == "drop_oldest"


# --------------------------------------------------------------------------
# policy="block", read in order
# --------------------------------------------------------------------------

def test_block_stops_grabbing_while_the_queue_is_full(clock):
    """Nothing queued is dropped; capture waits for the consumer instead."""
    cam = _camera(policy="block", max_buffer_len=2, pool_size=4)
    steps = [(1, 9_990, 1), (2, 9_991, 1), (3, 9_995, 4)]
    grabs = []

    def grab(region):
        if not steps:
            cam._stop_capture_event.set()
            return None
        fill, present_qpc, accumulated = steps.pop(0)
        grabs.append(fill)
        cam._duplicator.last_present_time = present_qpc
        cam._duplicator.accumulated_frames = accumulated
        buffer = cam.memory_pool.checkout()
        buffer.array[...] = fill
        return buffer

    cam._grab = grab
    producer = threading.Thread(
        target=cam._capture_thread_func, args=(cam.region, 0, False), daemon=True)
    producer.start()
    try:
        deadline = time.monotonic() + 5
        while len(_fills(cam)) < 2 and time.monotonic() < deadline:
            time.sleep(0.001)
        time.sleep(0.05)  # Long enough for a non-blocking producer to evict
        assert grabs == [1, 2], "capture must not grab while the queue is full"
        assert _fills(cam) == [1, 2]

        assert int(cam.get_next_frame().flat[0]) == 1
        deadline = time.monotonic() + 5
        while grabs != [1, 2, 3] and time.monotonic() < deadline:
            time.sleep(0.001)
        assert grabs == [1, 2, 3]
    finally:
        cam._stop_capture_event.set()
        producer.join(5)
    assert not producer.is_alive()

    assert _fills(cam) == [2, 3]
    stats = cam.stream_stats
    assert stats["consumer_drops"] == 0
    assert stats["stale_frames_replaced"] == 0
    assert stats["capture_drops"] == 3, (
        "the updates Windows merged while capture was held back are counted "
        "from the frame that carried them")


def test_get_next_frame_reads_in_capture_order(clock):
    cam = _camera()
    _produce(cam, [(1, 9_990, 1), (2, 9_991, 1), (3, 9_992, 1)])

    assert [int(cam.get_next_frame().flat[0]) for _ in range(3)] == [1, 2, 3]
    assert cam.stream_stats["frames_consumed"] == 3
    assert not cam._frame_available_event.is_set()


def test_get_next_frame_buffer_transfers_ownership(clock):
    cam = _camera(pool_size=3)
    _produce(cam, [(5, 9_990, 1)])

    frame = cam.get_next_frame_buffer()

    assert cam.memory_pool.get_stats()["available"] == 2
    assert not cam._pooled_frames_deque
    frame.release()
    assert cam.memory_pool.get_stats()["available"] == 3


# --------------------------------------------------------------------------
# max_age_ms
# --------------------------------------------------------------------------

def test_latest_skips_a_queue_too_old_to_use_and_waits(clock):
    cam = _camera(max_age_ms=10)
    # Captured 100 and 50 ms ago, and not confirmed since.
    _produce(cam, [(1, 9_900, 1), (2, 9_950, 1)], clock=clock)
    clock["qpc"] = 10_000

    started = time.monotonic()
    assert cam.get_latest_frame() is None
    assert time.monotonic() - started >= 0.9, "it waits for a fresh frame"

    stats = cam.stream_stats
    assert stats["frames_expired"] == 2
    assert stats["frames_consumed"] == 0
    assert not cam._pooled_frames_deque
    assert cam.memory_pool.get_stats()["available"] == 8, (
        "expired frames go back to the pool")


def test_a_fresh_frame_arriving_while_waiting_is_handed_out(clock):
    cam = _camera(max_age_ms=10)
    _produce(cam, [(1, 9_900, 1)], clock=clock)
    clock["qpc"] = 10_000

    def late_frame():
        time.sleep(0.1)
        with cam._capture_lock:
            evicted = cam._queue_frame(np.full(SHAPE, 9, np.uint8), 9_998)
        assert evicted is None
        cam._frame_available_event.set()

    threading.Thread(target=late_frame, daemon=True).start()
    frame = cam.get_latest_frame()

    assert frame is not None and int(frame.flat[0]) == 9
    stats = cam.stream_stats
    assert stats["frames_expired"] == 1
    assert stats["age_ms_last"] == pytest.approx(2.0)


def test_get_next_frame_skips_expired_frames_at_the_front(clock):
    cam = _camera(max_age_ms=20)
    _produce(cam, [(1, 9_900, 1), (2, 9_970, 1), (3, 9_990, 1), (4, 9_995, 1)],
             clock=clock)
    clock["qpc"] = 10_000

    assert int(cam.get_next_frame().flat[0]) == 3
    assert int(cam.get_next_frame().flat[0]) == 4
    assert cam.stream_stats["frames_expired"] == 2


def test_an_idle_screen_keeps_its_frame_fresh(clock):
    """Desktop Duplication delivers a frame only when the screen changes.

    On an idle screen the newest frame was presented long ago and is still
    exactly what the screen shows: every poll that finds nothing new confirms
    it. max_age_ms must measure from that confirmation, or an agent watching a
    still screen is refused the one frame that is correct.
    """
    cam = _camera(max_age_ms=12)
    steps = iter([(1, 10_000, 1), None, None, None])

    def grab(region):
        clock["qpc"] += 100          # 100 ms between polls
        try:
            step = next(steps)
        except StopIteration:
            cam._stop_capture_event.set()
            return None
        cam._last_grab_unchanged = step is None
        if step is None:
            return None
        fill, present_qpc, accumulated = step
        cam._duplicator.last_present_time = present_qpc
        cam._duplicator.accumulated_frames = accumulated
        buffer = cam.memory_pool.checkout()
        buffer.array[...] = fill
        return buffer

    cam._grab = grab
    cam._capture_thread_func(cam.region, target_fps=0, video_mode=False)
    clock["qpc"] += 5                 # The consumer arrives 5 ms after the last poll

    frame = cam.get_latest_frame()

    assert frame is not None and int(frame.flat[0]) == 1
    stats = cam.stream_stats
    assert stats["frames_expired"] == 0
    assert stats["age_ms_last"] == pytest.approx(505.0), (
        "the reported age is still the pixels' age, from the present")


def test_a_frame_nobody_has_confirmed_lately_expires(clock):
    """The other side: when capture stops polling, the frame does go stale."""
    cam = _camera(max_age_ms=12)
    _produce(cam, [(1, 10_000, 1)])
    clock["qpc"] += 50                # No poll since: 50 ms unconfirmed

    started = time.monotonic()
    assert cam.get_latest_frame() is None
    assert time.monotonic() - started >= 0.9
    assert cam.stream_stats["frames_expired"] == 1


def _poll(cam, steps, clock, step_ms=100):
    """Drive the capture thread through polls, ``step_ms`` apart.

    A step is ``("frame", fill, present_qpc)`` for a new frame, ``"same"`` for a
    poll that saw the screen unchanged -- what `_grab_locked` reports by setting
    ``_last_grab_unchanged`` -- or ``"fail"`` for one that saw nothing at all
    (protected content, a DXGI error), which returns None the same way.
    """
    steps = iter(steps)

    def grab(region):
        try:
            step = next(steps)
        except StopIteration:
            # The tick that ends the script takes no time, so the last poll's
            # confirmation is what the reader measures against.
            cam._stop_capture_event.set()
            cam._last_grab_unchanged = False
            return None
        clock["qpc"] += step_ms
        if step in ("same", "fail"):
            cam._last_grab_unchanged = step == "same"
            return None
        _kind, fill, present_qpc = step
        cam._last_grab_unchanged = False
        cam._duplicator.last_present_time = present_qpc
        cam._duplicator.accumulated_frames = 1
        buffer = cam.memory_pool.checkout()
        buffer.array[...] = fill
        return buffer

    cam._grab = grab
    cam._stop_capture_event.clear()
    cam._capture_thread_func(cam.region, target_fps=0, video_mode=False)


def test_an_unchanged_screen_renews_only_the_frame_it_shows(clock):
    """drop_oldest keeps older frames after the newest is taken. An unchanged
    poll confirms what the screen shows -- the frame taken, not the one left
    behind -- so the older frame must still expire (PR #14 review)."""
    cam = _camera(policy="drop_oldest", max_age_ms=12)
    # 2 ms apart, so frame 2 is fresh when it is read.
    _poll(cam, [("frame", 1, clock["qpc"] + 2), ("frame", 2, clock["qpc"] + 4)], clock,
          step_ms=2)
    assert int(cam.get_latest_frame().flat[0]) == 2   # B taken; A still queued

    _poll(cam, ["same", "same", "same"], clock, step_ms=20)  # the screen still shows B
    clock["qpc"] += 5

    started = time.monotonic()
    assert cam.get_latest_frame() is None, (
        "frame 1 was renewed by polls that saw frame 2 on screen")
    assert time.monotonic() - started >= 0.9
    assert cam.stream_stats["frames_expired"] == 1


def test_a_failed_poll_does_not_renew_a_frame(clock):
    """A grab that could not see the screen returns None too. Only a poll that
    saw the screen unchanged confirms the queued frame (PR #14 review)."""
    cam = _camera(policy="latest", max_age_ms=12)
    _poll(cam, [("frame", 1, clock["qpc"] + 100), "fail", "fail", "fail"], clock)
    clock["qpc"] += 5

    assert cam.get_latest_frame() is None, (
        "a frame was confirmed by polls that never saw the screen")
    assert cam.stream_stats["frames_expired"] == 1


def test_frames_with_no_recorded_time_are_never_expired(clock):
    """A frame queued without a time has an unknown age, not an old one."""
    cam = _camera(max_age_ms=1)
    with cam._capture_lock:
        cam._pooled_frames_deque.append(np.full(SHAPE, 6, np.uint8))
    cam._frame_available_event.set()

    assert int(cam.get_latest_frame().flat[0]) == 6
    assert cam.stream_stats["age_ms_last"] is None


# --------------------------------------------------------------------------
# Counters and ages
# --------------------------------------------------------------------------

def test_ages_are_measured_from_windows_present_time(clock):
    cam = _camera()
    _produce(cam, [(1, 9_990, 1), (2, 9_996, 1)])
    clock["qpc"] = 10_000

    cam.get_next_frame()   # 10 ms old
    cam.get_next_frame()   # 4 ms old

    stats = cam.stream_stats
    assert stats["age_ms_last"] == pytest.approx(4.0)
    assert stats["age_ms_mean"] == pytest.approx(7.0)


def test_without_a_present_time_the_age_runs_from_queueing(clock):
    cam = _camera()
    _produce(cam, [(1, 0, 1)])          # the backend reported no present time
    clock["qpc"] += 3

    cam.get_latest_frame()

    assert cam.stream_stats["age_ms_last"] == pytest.approx(3.0)


def test_capture_drops_count_merged_updates_but_not_before_the_first_frame(clock):
    cam = _camera()

    _produce(cam, [(1, 9_990, 7), (2, 9_991, 1), (3, 9_992, 3)])

    assert cam.stream_stats["capture_drops"] == 2


def test_video_mode_repeats_carry_their_source_time_and_are_counted(clock):
    cam = _camera()
    # One idle tick, plus the idle tick that stops the script: two repeats.
    _produce(cam, [(1, 9_980, 1), None], video_mode=True)
    clock["qpc"] = 10_000

    assert _fills(cam) == [1, 1, 1]
    stats = cam.stream_stats
    assert stats["frames_captured"] == 1
    assert stats["frames_duplicated"] == 2

    cam.get_latest_frame()
    assert cam.stream_stats["age_ms_last"] == pytest.approx(20.0), (
        "a repeat's pixels are as old as the frame it copies")


def test_draining_the_queue_forgets_every_recorded_time(clock):
    cam = _camera()
    _produce(cam, [(1, 9_990, 1), (2, 9_991, 1)])
    assert len(cam._queued_qpc) == 2

    cam._drain_frame_queue()

    assert cam._queued_qpc == {}


def test_every_frame_taken_out_leaves_no_recorded_time_behind(clock):
    cam = _camera(policy="latest", max_age_ms=5)
    _produce(cam, [(1, 9_990, 1), (2, 9_999, 1)])
    cam.get_latest_frame()

    assert cam._queued_qpc == {}


def test_stats_before_any_frame():
    stats = _camera().stream_stats
    assert stats["frames_captured"] == 0
    assert stats["age_ms_last"] is None and stats["age_ms_mean"] is None
    assert stats["max_age_ms"] is None
