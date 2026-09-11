"""Single-shot capture against the capture thread, and the duplication lock.

grab() was meant to redirect to get_latest_frame() while start() ran, but the
flag it checked was never set, so it raced the capture thread on one
duplicator -- as did shot() and grab_frame(), which had no check at all. And
nothing serialised duplicator use between threads: `_capture_lock` guards only
the continuous-mode deque.
"""

import threading
import time

import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture  # noqa: E402


def _camera():
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._live_frame = None
    cam.is_capturing = False
    cam._capture_thread = None
    return cam


@pytest.mark.parametrize("name, call", [
    ("grab()", lambda cam: cam.grab()),
    ("grab_frame()", lambda cam: cam.grab_frame()),
    ("shot()", lambda cam: cam.shot(bytearray(16))),
])
def test_single_shot_capture_is_refused_while_continuous_capture_runs(name, call):
    cam = _camera()
    cam.is_capturing = True
    cam._capture_thread = threading.Thread(target=lambda: None)
    try:
        with pytest.raises(RuntimeError, match="continuous capture is running") as err:
            call(cam)
        assert name in str(err.value)
    finally:
        cam.is_capturing = False   # the stub has no thread for release() to stop


def test_the_capture_thread_itself_is_not_refused():
    cam = _camera()
    cam.is_capturing = True
    cam._capture_thread = threading.current_thread()
    cam._refuse_while_capturing("grab()")   # must not raise
    cam.is_capturing = False


def _overlap_probe():
    """A slow stand-in for a capture call that records peak concurrency."""
    state = {"active": 0, "peak": 0}
    guard = threading.Lock()

    def slow(*_args, **_kwargs):
        with guard:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.03)
        with guard:
            state["active"] -= 1
        return None

    return slow, state


def _run_concurrently(*targets):
    threads = [threading.Thread(target=t) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)


def test_the_probe_does_see_overlap_without_the_lock():
    """Control: proves the test below can fail."""
    slow, state = _overlap_probe()
    _run_concurrently(*[slow] * 4)
    assert state["peak"] > 1


def test_grab_shot_and_rebuild_never_touch_the_duplicator_at_once(monkeypatch):
    cam = _camera()
    cam.region = (0, 0, 1, 1)
    slow, state = _overlap_probe()
    monkeypatch.setattr(cam, "_grab_locked", slow)
    monkeypatch.setattr(cam, "_shot_locked", slow)
    monkeypatch.setattr(cam, "_grab_frame_locked", slow)
    monkeypatch.setattr(cam, "_on_output_change_locked", slow)
    _run_concurrently(
        cam._grab, cam._grab,
        lambda: cam._shot(None, (0, 0, 1, 1)),
        cam._on_output_change,
        cam.grab_frame,
    )
    assert state["peak"] == 1


def test_release_waits_for_a_grab_in_flight(monkeypatch):
    cam = _camera()
    order = []
    started = threading.Event()

    def slow_grab(region=None):
        started.set()
        time.sleep(0.1)
        order.append("grab finished")

    class Duplicator:
        def release(self):
            order.append("duplicator released")

    monkeypatch.setattr(cam, "_grab_locked", slow_grab)
    cam._duplicator = Duplicator()
    worker = threading.Thread(target=cam._grab)
    worker.start()
    assert started.wait(2)
    cam.release()
    worker.join(timeout=5)
    assert order[:2] == ["grab finished", "duplicator released"]
