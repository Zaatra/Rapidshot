"""shot(region=...) must not adopt the region; start(delay=...) is in seconds."""

import pytest

pytest.importorskip("comtypes")

import rapidshot.capture as capture_module  # noqa: E402
from rapidshot.capture import ScreenCapture  # noqa: E402


def _camera(width=20, height=10):
    cam = ScreenCapture.__new__(ScreenCapture)
    cam.width, cam.height = width, height
    cam.region = (0, 0, width, height)
    cam._sourceRegion = None
    cam.shot_w, cam.shot_h = width, height
    cam._live_frame = None
    cam.is_capturing = False
    cam._capture_thread = None
    return cam


def test_shot_with_a_region_leaves_the_camera_region_alone(monkeypatch):
    """A one-off shot(region=...) used to become every later grab's region."""
    cam = _camera()
    seen = {}
    monkeypatch.setattr(cam, "_validate_destination", lambda ptr, region, size: None)
    monkeypatch.setattr(cam, "_shot", lambda ptr, region, size: seen.setdefault("region", region))
    cam.shot(bytearray(1), region=(2, 3, 10, 9))
    assert seen["region"] == (2, 3, 10, 9), "the shot itself uses the region"
    assert cam.region == (0, 0, 20, 10)
    assert (cam.shot_w, cam.shot_h) == (20, 10)


def test_shot_still_rejects_an_invalid_region(monkeypatch):
    cam = _camera()
    monkeypatch.setattr(cam, "_shot", lambda *a: pytest.fail("captured an invalid region"))
    with pytest.raises(ValueError):
        cam.shot(bytearray(1), region=(0, 0, 50, 5))


class _NoThread:
    def __init__(self, *args, **kwargs):
        pass

    daemon = False

    def start(self):
        pass


@pytest.mark.parametrize("delay, slept", [(0, None), (2, 2), (0.25, 0.25)])
def test_start_delay_is_seconds(monkeypatch, delay, slept):
    cam = _camera()
    sleeps, rebuilds = [], []
    monkeypatch.setattr(capture_module.time, "sleep", sleeps.append)
    monkeypatch.setattr(capture_module, "Thread", _NoThread)
    monkeypatch.setattr(cam, "_on_output_change", lambda: rebuilds.append(True))
    cam.max_buffer_len = 4
    cam._frame_available_event = capture_module.Event()
    cam._stop_capture_event = capture_module.Event()
    cam.start(delay=delay)
    cam.is_capturing = False          # nothing real to stop
    if slept is None:
        assert sleeps == [] and rebuilds == []
    else:
        assert sleeps == [slept], "time.sleep() takes seconds; so does delay"
        assert rebuilds == [True]


@pytest.mark.parametrize("delay", [-1, "1", True, None])
def test_start_rejects_a_bad_delay(delay):
    cam = _camera()
    with pytest.raises(ValueError, match="seconds"):
        cam.start(delay=delay)
