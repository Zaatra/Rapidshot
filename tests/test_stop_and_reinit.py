"""stop() against a capture thread that will not stop, and the reinit drain.

Two faults meet here. `stop()` waited ten seconds for the capture thread, then
carried on regardless -- closing the timer handle that the still-running thread
closes again on its way out, a second CloseHandle on a handle Windows may have
reissued to something else, and emptying the queue that thread was still
appending to. The same double close happened whenever `stop()` was called
*from* the capture thread, where the join is skipped entirely.

And `_initialize_resources()` cleared the continuous-mode queue only when
`continuous_mode` was True -- a flag nothing has ever set -- and did it after
the pool those buffers belong to had already been destroyed, so the check-ins
were refused and the buffers dropped rather than recycled.

These drive the camera's own state rather than real capture: both faults are in
the bookkeeping between stop(), start() and the pool, and need no DXGI.
"""

import collections
import threading

import numpy as np
import pytest

pytest.importorskip("comtypes")

import rapidshot.capture as capture_module  # noqa: E402
from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool  # noqa: E402

SHAPE = (4, 6, 4)
HANDLE = 0xDEADBEEF


@pytest.fixture
def timer_calls(monkeypatch):
    """Record every timer call stop() makes, instead of calling into Windows."""
    calls = []
    monkeypatch.setattr(capture_module, "cancel_timer",
                        lambda h: calls.append(("cancel", h)))
    monkeypatch.setattr(capture_module, "close_timer",
                        lambda h: calls.append(("close", h)))
    return calls


_MADE = []


@pytest.fixture(autouse=True)
def _neutralise_test_cameras():
    """Keep __del__ from running release() over these half-built doubles.

    The destructor calls release(), which calls stop(), which would close the
    fake timer handle for real -- outside the test, where the monkeypatch is
    gone -- and try to release stub objects. None of that is under test, and
    all of it lands as warnings in some later test's output.
    """
    yield
    while _MADE:
        cam = _MADE.pop()
        cam._released = True
        cam.is_capturing = False
        cam._timer_handle = None
        cam._capture_thread = None
        cam.memory_pool = None
        cam._duplicator = None
        cam._stagesurf = None
        cam._live_frame = None


def _camera(pool_size=2):
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._capture_lock = threading.Lock()
    cam._frame_available_event = threading.Event()
    cam._stop_capture_event = threading.Event()
    cam._pooled_frames_deque = collections.deque(maxlen=64)
    cam._last_dup_source = None
    cam.max_buffer_len = 64
    cam.nvidia_gpu = False
    cam.is_capturing = False
    cam._capture_thread = None
    cam._timer_handle = HANDLE
    cam._frame_count = 7
    cam.memory_pool = NumpyMemoryPool(SHAPE, np.uint8, pool_size)
    _MADE.append(cam)
    return cam


def _queue(cam, fill=1):
    buffer = cam.memory_pool.checkout()
    buffer.array[...] = fill
    with cam._capture_lock:
        cam._pooled_frames_deque.append(buffer)
        cam._last_dup_source = buffer
    cam._frame_available_event.set()
    return buffer


class _StuckThread:
    """A capture thread that ignores the stop event, as a wedged one would."""

    def __init__(self, cam):
        self._release = threading.Event()
        self.thread = threading.Thread(target=self._release.wait, daemon=True)
        self.thread.start()
        cam.is_capturing = True
        cam._capture_thread = self.thread
        cam._stop_join_timeout_s = 0.05

    def finish(self):
        self._release.set()
        self.thread.join(timeout=5)


@pytest.fixture
def stuck():
    made = []

    def make(cam):
        wedged = _StuckThread(cam)
        made.append(wedged)
        return wedged

    yield make
    for wedged in made:
        wedged.finish()


# --------------------------------------------------------------------------
# stop() against a thread that will not stop
# --------------------------------------------------------------------------

def test_stop_reports_a_thread_it_could_not_join(timer_calls, stuck):
    cam = _camera()
    stuck(cam)

    assert cam.stop() is False


def test_stop_leaves_the_timer_handle_to_a_surviving_thread(timer_calls, stuck):
    """The thread closes it itself at the end of _capture_thread_func."""
    cam = _camera()
    stuck(cam)

    cam.stop()

    assert timer_calls == [], (
        "stop() closed a handle the surviving capture thread will close again")
    assert cam._timer_handle == HANDLE


def test_stop_leaves_the_frame_queue_to_a_surviving_thread(timer_calls, stuck):
    """The thread is still appending; setting the queue to None faults it."""
    cam = _camera()
    _queue(cam)
    stuck(cam)

    cam.stop()

    assert cam._pooled_frames_deque is not None
    assert len(cam._pooled_frames_deque) == 1


def test_stop_from_inside_the_capture_thread_leaves_the_timer_alone(timer_calls):
    """No join happens on this path, so the thread is still on its way out."""
    cam = _camera()
    cam.is_capturing = True
    result = {}

    def body():
        cam._capture_thread = threading.current_thread()
        result["returned"] = cam.stop()

    thread = threading.Thread(target=body)
    thread.start()
    thread.join(timeout=5)

    assert result["returned"] is False
    assert timer_calls == []
    assert cam._timer_handle == HANDLE


def test_a_surviving_thread_blocks_a_restart(timer_calls, stuck):
    """Two threads on one duplicator is the race stop() used to allow."""
    cam = _camera()
    stuck(cam)
    cam.stop()

    with pytest.raises(RuntimeError, match="has not exited yet"):
        cam.start()


def test_a_restart_is_allowed_once_the_thread_ends(timer_calls, stuck):
    cam = _camera()
    wedged = stuck(cam)
    cam.stop()
    wedged.finish()

    # Far enough into start() to prove the refusal did not fire: the delay
    # check is the first thing start() does after it.
    with pytest.raises(ValueError, match="non-negative number of seconds"):
        cam.start(delay=-1)


# --------------------------------------------------------------------------
# stop() on the ordinary path must still tidy up
# --------------------------------------------------------------------------

def test_stop_closes_the_timer_once_the_thread_has_finished(timer_calls):
    cam = _camera()
    cam.is_capturing = True
    thread = threading.Thread(target=lambda: None)
    thread.start()
    thread.join()
    cam._capture_thread = thread

    assert cam.stop() is True
    assert timer_calls == [("cancel", HANDLE), ("close", HANDLE)]
    assert cam._timer_handle is None
    assert cam._capture_thread is None


def test_stop_returns_queued_buffers_to_the_pool(timer_calls):
    cam = _camera()
    _queue(cam)
    assert cam.memory_pool.get_stats()["available"] == 1

    assert cam.stop() is True

    assert cam.memory_pool.get_stats()["available"] == 2
    assert cam._pooled_frames_deque is None
    assert cam._last_dup_source is None


def test_stop_is_harmless_when_capture_never_started(timer_calls):
    cam = _camera()
    cam._pooled_frames_deque = None

    assert cam.stop() is True


# --------------------------------------------------------------------------
# The drain the re-initialisation path needs
# --------------------------------------------------------------------------

def test_draining_checks_buffers_back_in():
    cam = _camera(pool_size=3)
    _queue(cam, 1)
    _queue(cam, 2)
    assert cam.memory_pool.get_stats()["in_use"] == 2

    cam._drain_frame_queue()

    assert cam.memory_pool.get_stats()["in_use"] == 0
    assert not cam._pooled_frames_deque
    assert cam._last_dup_source is None
    assert not cam._frame_available_event.is_set()


def test_draining_without_a_queue_is_a_no_op():
    cam = _camera()
    cam._pooled_frames_deque = None

    cam._drain_frame_queue()   # must not raise

    assert cam._last_dup_source is None


def test_draining_a_destroyed_pool_would_drop_the_buffers():
    """Why the order matters: this is what the old code did.

    Less a test of RapidShot's behaviour than of the constraint it has to
    respect -- a check-in after destroy_pool() is refused, so the buffer never
    comes back.
    """
    cam = _camera()
    _queue(cam)
    cam.memory_pool.destroy_pool()

    cam._drain_frame_queue()   # swallowed, as _discard_frame does

    assert cam.memory_pool.get_stats()["available"] == 0


# --------------------------------------------------------------------------
# The re-initialisation path itself
# --------------------------------------------------------------------------

class _RecordingPool(NumpyMemoryPool):
    """A pool that logs check-ins and its own destruction, in order."""

    def __init__(self, *args, log, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = log

    def checkin(self, buffer_wrapper):
        self.log.append("checkin")
        super().checkin(buffer_wrapper)

    def destroy_pool(self):
        self.log.append("destroy")
        super().destroy_pool()


class _FakeOutput:
    devicename = "fake-output"
    resolution = (SHAPE[1], SHAPE[0])
    rotation_angle = 0

    def update_desc(self):
        pass


def _reinit_camera(monkeypatch, log):
    """A camera with just enough wired up to run _initialize_resources()."""
    cam = _camera()
    cam._output = _FakeOutput()
    cam._device = object()
    cam._display_idx = 0        # only read by a log line on the reinit path
    cam._output_idx = 0
    cam.width, cam.height = _FakeOutput.resolution
    cam.region = (0, 0, cam.width, cam.height)
    cam._region_set_by_user = False
    cam._is_initialized = True
    cam._needs_reinit = True
    cam._reinit_attempts = 1
    cam._generation = 1
    cam._recovery_count = 0
    cam._last_recovery_reason = "test"
    cam._duplicator = None
    cam._stagesurf = None
    cam._init_args = {
        "region": None,
        "output_color": "BGRA",
        "nvidia_gpu": False,
        "pool_size_frames": 2,
    }
    cam.memory_pool = _RecordingPool(SHAPE, np.uint8, 2, log=log)

    monkeypatch.setattr(cam, "_build_duplicator", lambda: object())
    monkeypatch.setattr(capture_module, "StageSurface", lambda **kw: object())
    monkeypatch.setattr(capture_module, "Processor", lambda **kw: object())
    return cam


def test_reinitialisation_drains_the_queue(monkeypatch):
    """It never did: the block was gated on a flag nothing ever set."""
    log = []
    cam = _reinit_camera(monkeypatch, log)
    cam.is_capturing = True
    _queue(cam)

    assert cam._initialize_resources(is_reinit=True) is True

    assert not cam._pooled_frames_deque, (
        "frames from before the rebuild stayed queued, and get_latest_frame() "
        "would hand them out as current")
    assert cam._last_dup_source is None


def test_reinitialisation_drains_before_destroying_the_pool(monkeypatch):
    """The whole point of the ordering."""
    log = []
    cam = _reinit_camera(monkeypatch, log)
    cam.is_capturing = True
    _queue(cam)

    cam._initialize_resources(is_reinit=True)

    assert log == ["checkin", "destroy"], (
        f"expected the buffer back before the pool died, got {log}")


def test_reinitialisation_without_a_queue_still_works(monkeypatch):
    """First-time initialisation has no deque at all."""
    log = []
    cam = _reinit_camera(monkeypatch, log)
    cam._pooled_frames_deque = None

    assert cam._initialize_resources(is_reinit=False) is True


def test_the_dead_continuous_mode_flag_is_gone():
    """It read False forever while continuous capture ran, and gating on it is
    what kept the drain from running."""
    cam = ScreenCapture.__new__(ScreenCapture)
    assert not hasattr(cam, "continuous_mode")
