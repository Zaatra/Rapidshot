"""Motion-source lifecycle tests use a fake Tk implementation only."""
import importlib
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import motion_source as motion


def test_import_does_not_create_tk_or_parse_arguments(monkeypatch):
    def forbidden():
        raise AssertionError("Tk was constructed during import")
    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(Tk=forbidden))
    monkeypatch.setattr(sys, "argv", ["pytest", "not-a-duration"])
    importlib.reload(motion)


def test_parent_eof_requests_shutdown():
    stopped = threading.Event()
    motion.watch_parent(io.StringIO(""), stopped)
    assert stopped.is_set()


def test_parent_stop_requests_shutdown():
    stopped = threading.Event()
    motion.watch_parent(io.StringIO("stop\n"), stopped)
    assert stopped.is_set()


def test_frame_cap_accounts_for_drawing_time():
    assert motion.frame_delay(1, 100, 1.004) == pytest.approx(0.006)
    assert motion.frame_delay(1, 100, 1.02) == 0
    assert motion.frame_delay(1, 0, 1.004) == 0


@pytest.fixture
def fake_tk(monkeypatch):
    state = SimpleNamespace(now=0.0, calls=[], emitted=[], fail=False, rectangles=0)

    class Root:
        def title(self, value): pass
        def overrideredirect(self, value): pass
        def geometry(self, value): pass
        def attributes(self, *args): pass
        def protocol(self, *args): pass
        def update_idletasks(self): pass
        def update(self):
            if state.fail:
                raise RuntimeError("simulated Tk drawing failure")
            state.calls.append("draw")
            state.now += 0.01
        def destroy(self):
            state.calls.append("destroy")

    class Canvas:
        def __init__(self, *args, **kwargs): pass
        def pack(self): pass
        def create_rectangle(self, *args, **kwargs):
            state.rectangles += 1
            return state.rectangles
        def coords(self, *args): pass
        def itemconfig(self, *args, **kwargs): pass

    monkeypatch.setitem(sys.modules, "tkinter", SimpleNamespace(Tk=Root, Canvas=Canvas))
    monkeypatch.setattr(motion.time, "perf_counter", lambda: state.now)
    monkeypatch.setattr(motion, "emit", lambda event, **fields: state.emitted.append((event, fields)))
    return state


def test_ready_follows_first_draw_and_reports_rate(fake_tk):
    motion.animate(2.1)
    events = [event for event, _ in fake_tk.emitted]
    assert events[0:2] == ["starting", "ready"]
    assert events.count("ready") == 1
    assert "rate" in events
    rate = next(fields for event, fields in fake_tk.emitted if event == "rate")
    assert rate["updates_per_second"] == pytest.approx(100)
    assert fake_tk.rectangles == 28
    assert fake_tk.calls[-1] == "destroy"


def test_drawing_failure_is_not_silently_successful(fake_tk):
    fake_tk.fail = True
    assert motion.main(["1"]) == 1
    assert fake_tk.calls[-1] == "destroy"
    assert fake_tk.emitted[-1][0] == "error"
    assert "drawing failure" in fake_tk.emitted[-1][1]["error"]


def test_cap_is_interruptible_by_parent(fake_tk):
    class Stopped:
        stopped = False
        def is_set(self): return self.stopped
        def set(self): self.stopped = True
        def wait(self, seconds):
            assert seconds == pytest.approx(0.09)
            self.stopped = True
    motion.animate(None, fps=10, stopped=Stopped())
    assert fake_tk.calls == ["draw", "destroy"]


def test_parent_controlled_run_has_no_estimated_duration(monkeypatch):
    seen = []
    class Thread:
        def __init__(self, **kwargs): pass
        def start(self): pass
    monkeypatch.setattr(motion.threading, "Thread", Thread)
    monkeypatch.setattr(motion, "animate", lambda duration, *args: seen.append(duration))
    assert motion.main(["--parent-controlled"]) == 0
    assert seen == [None]


@pytest.mark.parametrize("args", [["0"], ["nan"], ["inf"], ["--fps", "-1"],
                                   ["--fps", "nan"]])
def test_invalid_options_rejected_before_tk(args):
    with pytest.raises(SystemExit) as exc:
        motion.main(args)
    assert exc.value.code == 2
