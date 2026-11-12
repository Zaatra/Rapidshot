"""The Windows.Graphics.Capture backend, headless (ROADMAP § 7.3).

The capture itself is native (native/src/wgc.rs) and is exercised live by
tests/test_wgc_live.py. Everything here runs on fakes: the factory's routing
of sources onto WGC, the camera's handling of what a WGC backend reports, and
WgcBackend's translation of the native session's results into the
CaptureBackend contract.
"""
import ctypes
import logging
import types

import numpy as np
import pytest

import rapidshot
from rapidshot import Monitor, Window

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.capture as capture_module  # noqa: E402
import rapidshot.core.wgc as wgc  # noqa: E402
from rapidshot.core.backend import CaptureBackend  # noqa: E402
from rapidshot.core.hdr import DisplayColor  # noqa: E402
from rapidshot.util.errors import (  # noqa: E402
    RapidShotConfigError,
    RapidShotReinitError,
    RapidShotSourceClosedError,
)

from test_capture_paths import FakeDuplicator, pipeline  # noqa: E402,F401
from test_sources import two_displays  # noqa: E402


# --------------------------------------------------------------------------
# the factory routes sources onto WGC
# --------------------------------------------------------------------------

class FakeWindowGeometry:
    def __init__(self, hwnd):
        self.hwnd = hwnd


class FakeMonitorGeometry:
    def __init__(self, output):
        self.output = output


@pytest.fixture
def wgc_available(monkeypatch):
    monkeypatch.setattr(wgc, "is_available", lambda: True)
    monkeypatch.setattr(wgc, "WindowGeometry", FakeWindowGeometry)
    monkeypatch.setattr(wgc, "WgcMonitorGeometry", FakeMonitorGeometry)
    monkeypatch.setattr(rapidshot, "_window_display_name", lambda hwnd: "DISPLAY2")


def test_a_window_is_captured_through_wgc_on_its_display(monkeypatch, wgc_available):
    factory, _, _, devices = two_displays(monkeypatch)

    camera = factory.create(source=Window(0x1234))

    assert camera.kwargs["backend"] == "wgc"
    assert camera.kwargs["source"] == Window(0x1234)
    assert isinstance(camera.kwargs["output"], FakeWindowGeometry)
    assert camera.kwargs["output"].hwnd == 0x1234
    assert camera.kwargs["device"] is devices[1]          # DISPLAY2's adapter


def test_a_window_on_no_known_display_uses_the_first(monkeypatch, wgc_available):
    monkeypatch.setattr(rapidshot, "_window_display_name", lambda hwnd: "")
    factory, _, _, devices = two_displays(monkeypatch)
    assert factory.create(source=Window(0x1234)).kwargs["device"] is devices[0]


def test_desktop_duplication_cannot_capture_a_window(monkeypatch, wgc_available):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="captures displays, not windows"):
        factory.create(source=Window(0x1234), backend="dxgi")


def test_each_window_gets_its_own_camera(monkeypatch, wgc_available):
    """Two windows on one display are two captures, not one cached camera."""
    factory, *_ = two_displays(monkeypatch)

    first = factory.create(source=Window(0x1234))
    second = factory.create(source=Window(0x5678))

    assert first is not second
    assert factory.create(source=Window(0x1234)) is first


def test_a_monitor_through_wgc_gets_wgc_geometry(monkeypatch, wgc_available):
    factory, _, right, _ = two_displays(monkeypatch)

    camera = factory.create(source=Monitor(1), backend="wgc")

    assert camera.kwargs["backend"] == "wgc"
    assert isinstance(camera.kwargs["output"], FakeMonitorGeometry)
    assert camera.kwargs["output"].output is right


def test_wgc_and_dxgi_cameras_can_share_a_monitor(monkeypatch, wgc_available):
    """One camera per output is a Desktop Duplication rule; WGC sessions are
    independent of it."""
    factory, *_ = two_displays(monkeypatch)
    dxgi_camera = factory.create(source=Monitor(1))
    wgc_camera = factory.create(source=Monitor(1), backend="wgc")
    assert dxgi_camera is not wgc_camera


# --------------------------------------------------------------------------
# the camera on a WGC backend
# --------------------------------------------------------------------------

@pytest.fixture
def wgc_pipeline(pipeline, monkeypatch):  # noqa: F811
    """test_capture_paths' fake pipeline, with FakeDuplicator standing in for
    WgcBackend -- both are built as Backend(output=, device=, timeout_ms=)."""
    monkeypatch.setattr(wgc, "WgcBackend", FakeDuplicator)
    return pipeline


def test_camera_builds_a_wgc_backend_on_its_first_candidate(wgc_pipeline):
    cam, _, _, device = wgc_pipeline(backend="wgc", output_color="BGRA",
                                     source=Window(0x1234))

    assert cam.backend == "wgc"
    assert FakeDuplicator.built[-1].device is device
    with cam.grab_frame() as frame:
        assert frame.backend == "wgc"
        assert frame.source == Window(0x1234)


def test_a_closed_source_stops_the_camera_instead_of_rebuilding(wgc_pipeline):
    """A closed window cannot come back, so the re-init loop -- five attempts
    with back-off -- would only delay the answer."""
    cam, *_ = wgc_pipeline(backend="wgc", output_color="BGRA",
                           script=["frame", RapidShotSourceClosedError("window closed")])
    with cam.grab_frame():
        pass

    assert cam.grab_frame() is None
    assert cam._capture_permanently_failed
    assert "window closed" in cam.last_capture_error
    calls = FakeDuplicator.built[-1].calls
    assert cam.grab() is None
    assert FakeDuplicator.built[-1].calls == calls      # nothing more was tried


def test_grab_on_a_closed_source_stops_the_camera(wgc_pipeline):
    cam, *_ = wgc_pipeline(backend="wgc",
                           script=[RapidShotSourceClosedError("window closed")])
    assert cam.grab() is None
    assert cam._capture_permanently_failed
    assert not cam._needs_reinit


def test_a_source_that_closes_during_a_rebuild_stops_it(wgc_pipeline):
    cam, output, *_ = wgc_pipeline(backend="wgc")

    def gone():
        raise RapidShotSourceClosedError("window closed")

    output.update_desc = gone
    assert cam._on_output_change() is False
    assert cam._capture_permanently_failed


def test_a_grab_that_rebuilds_captures_at_the_new_size(pipeline):  # noqa: F811
    """The grab that performs a rebuild used the region read *before* it, so
    the first frame after a resize -- WGC's resized window, or Desktop
    Duplication's new resolution -- came back cropped to the old size. A
    still window sends no second frame, so that stale crop was all a caller
    got. Plain DDA here: the bug and the fix are the camera's, not WGC's."""
    cam, output, desktop, _ = pipeline(pool_output=False,
                                       script=[RapidShotReinitError("resized"), "frame"])
    assert cam.grab() is None                       # flags the rebuild

    output._native = (80, 60)                       # the source grew
    desktop.refresh(1)
    image = cam.grab()                              # rebuilds, then grabs

    assert image.shape[:2] == (60, 80)
    np.testing.assert_array_equal(image, desktop.as_desktop())


def test_screen_capture_refuses_unknown_backends():
    with pytest.raises(RapidShotConfigError, match="'dxgi' or 'wgc'"):
        capture_module.ScreenCapture(output=None, device=None, backend="winrt")


# --------------------------------------------------------------------------
# WgcBackend over a fake native session
# --------------------------------------------------------------------------

class FakeSession:
    """Scripted native WgcSession: acquire() returns the next result."""

    script = []
    created = []

    def __init__(self, device_ptr, **kwargs):
        if kwargs.get("window") == -1:
            raise RuntimeError("WGC session setup failed: E_INVALIDARG")
        self.device_ptr = device_ptr
        self.kwargs = kwargs
        self.closed = 0
        self.unsupported = []
        self.pixel_format = kwargs["pixel_format"]
        FakeSession.created.append(self)

    def acquire(self, timeout_ms):
        step = FakeSession.script.pop(0) if FakeSession.script else {"status": "timeout"}
        if isinstance(step, BaseException):
            raise step
        return step

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_native(monkeypatch):
    FakeSession.script = []
    FakeSession.created = []
    ext = types.SimpleNamespace(WgcSession=FakeSession)
    monkeypatch.setattr(wgc, "_require", lambda: ext)
    monkeypatch.setattr(wgc, "_texture_from_address", lambda address: ("texture", address))
    monkeypatch.setattr(wgc.hdr, "display_color", lambda name: DisplayColor(hdr=False))
    return FakeSession


def monitor_geometry():
    return types.SimpleNamespace(hmonitor=0x10001, devicename="DISPLAY1")


def window_geometry(hwnd=0x2002):
    geometry = wgc.WindowGeometry.__new__(wgc.WindowGeometry)
    geometry.hwnd, geometry._size, geometry._devicename = hwnd, (800, 600), "DISPLAY1"
    return geometry


def device():
    return types.SimpleNamespace(device=ctypes.c_void_p(0x7000))


def frame(serial=1, skipped=0, qpc=123456, arrived=None):
    return {"status": "frame", "texture": 0xABC, "width": 800, "height": 600,
            "dxgi_format": 87, "arrived_qpc": qpc if arrived is None else arrived,
            "acquired_qpc": qpc, "system_relative_qpc": qpc + 34000,
            "serial": serial, "skipped": skipped}


def test_a_monitor_session_is_set_up_as_measured(fake_native):
    """No cursor in the image (as Desktop Duplication), 1 ms minimum update
    interval (the default held 165 Hz to 55 fps), BGRA on an SDR display."""
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device(), timeout_ms=7)

    session = fake_native.created[-1]
    assert session.device_ptr == 0x7000
    assert session.kwargs == {"monitor": 0x10001, "pixel_format": "bgra", "cursor": False,
                              "min_update_interval_us": 1000}
    assert backend.timeout_ms == 7
    backend.release()


def test_an_hdr_display_is_captured_as_fp16(fake_native, monkeypatch):
    monkeypatch.setattr(wgc.hdr, "display_color", lambda name: DisplayColor(hdr=True))
    wgc.WgcBackend(output=monitor_geometry(), device=device()).release()
    assert fake_native.created[-1].kwargs["pixel_format"] == "fp16"


def test_a_window_session_targets_the_window(fake_native):
    wgc.WgcBackend(output=window_geometry(0x2002), device=device()).release()
    assert fake_native.created[-1].kwargs["window"] == 0x2002
    assert "monitor" not in fake_native.created[-1].kwargs


def test_a_refused_session_is_a_config_error(fake_native):
    with pytest.raises(RapidShotConfigError, match="refused"):
        wgc.WgcBackend(output=window_geometry(-1), device=device())


class FakeKernel32:
    """GetModuleHandleExW, recorded; ``ok`` decides what it returns."""

    def __init__(self, ok=True):
        self.ok, self.calls = ok, []

    def GetModuleHandleExW(self, flags, name, module):
        self.calls.append((flags, name))
        return self.ok


@pytest.fixture
def kernel32(monkeypatch):
    fake = FakeKernel32()
    monkeypatch.setattr(wgc, "_kernel32", fake)
    monkeypatch.setattr(wgc, "_graphics_capture_pinned", False)
    return fake


def test_graphics_capture_is_pinned_once_a_session_exists(fake_native, kernel32):
    """comtypes' CoUninitialize at exit unloaded GraphicsCapture.dll under
    WGC's own teardown work, and the process crashed in the unloaded DLL; a
    pinned DLL cannot unload. Pinned with the flag that pins, by its name."""
    wgc.WgcBackend(output=monitor_geometry(), device=device()).release()
    wgc.WgcBackend(output=monitor_geometry(), device=device()).release()
    assert kernel32.calls == [(0x1, "GraphicsCapture.dll")], "once per process"


def test_a_refused_session_pins_nothing(fake_native, kernel32):
    with pytest.raises(RapidShotConfigError):
        wgc.WgcBackend(output=window_geometry(-1), device=device())
    assert kernel32.calls == []


def test_a_failed_pin_is_logged_not_raised_and_tried_again(fake_native, kernel32, caplog):
    kernel32.ok = False
    with caplog.at_level(logging.DEBUG, logger=wgc.logger.name):
        wgc.WgcBackend(output=monitor_geometry(), device=device()).release()
    assert "could not pin GraphicsCapture.dll" in caplog.text
    kernel32.ok = True
    wgc.WgcBackend(output=monitor_geometry(), device=device()).release()
    assert len(kernel32.calls) == 2 and wgc._graphics_capture_pinned


def test_a_timeout_holds_nothing(fake_native):
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    assert backend.update_frame() is True
    assert not backend.updated and not backend._frame_acquired
    backend.release()


def test_a_frame_fills_the_contract(fake_native):
    fake_native.script = [frame(serial=3, skipped=2, qpc=999, arrived=950)]
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())

    assert backend.update_frame() is True

    assert backend.updated and backend._frame_acquired
    assert backend.texture == ("texture", 0xABC)
    assert backend.frame_serial == 3
    assert backend.last_present_time == 950          # arrival, not acquisition or SRT
    assert backend.accumulated_frames == 3           # this one and the two skipped
    assert backend.dirty_rects is None and backend.move_rects is None
    assert backend.cursor is None and not backend.cursor_visible

    backend.release_frame()
    assert backend.texture is None and not backend._frame_acquired
    backend.release_frame()                          # idempotent
    backend.release()


def test_an_older_extension_without_arrival_times_uses_acquisition(fake_native):
    step = frame(qpc=777)
    del step["arrived_qpc"]
    fake_native.script = [step]
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    backend.update_frame()
    assert backend.last_present_time == 777
    backend.release()


def test_each_unsupported_setting_is_a_warning_that_says_what_it_costs(fake_native, monkeypatch):
    """A Windows build older than 24H2 refuses MinUpdateInterval and capture
    runs at WGC's default rate. That changes what the caller gets, so it is a
    warning naming the cost, not an informational line."""
    warned = []
    monkeypatch.setattr(wgc.logger, "warning", lambda msg, *args: warned.append(msg % args))
    original = FakeSession.__init__

    def refusing(self, device_ptr, **kwargs):
        original(self, device_ptr, **kwargs)
        self.unsupported = ["min_update_interval", "border"]

    monkeypatch.setattr(FakeSession, "__init__", refusing)
    wgc.WgcBackend(output=monitor_geometry(), device=device()).release()

    assert len(warned) == 2
    assert "55 fps" in warned[0] and "24H2" in warned[0]
    assert "yellow border" in warned[1]


@pytest.mark.parametrize("step, error", [
    ({"status": "resized", "width": 640, "height": 480}, RapidShotReinitError),
    ({"status": "closed"}, RapidShotSourceClosedError),
    (RuntimeError("WGC acquire failed: device removed"), RapidShotReinitError),
])
def test_acquire_outcomes_map_onto_the_camera_errors(fake_native, step, error):
    """Resize rebuilds at the new size; a closed window ends capture; a native
    failure is treated as one more reason to rebuild."""
    fake_native.script = [step]
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    with pytest.raises(error):
        backend.update_frame()
    assert not backend._frame_acquired
    backend.release()


def test_release_closes_the_session_once(fake_native):
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    session = fake_native.created[-1]

    backend.release()
    backend.release()

    assert session.closed == 1
    assert backend.update_frame() is False           # released: unhealthy, not an error


def test_open_sessions_are_closed_at_exit(fake_native):
    """A session alive past comtypes' CoUninitialize crashed the interpreter
    on exit (Machine B, 2026-10-22); the module's atexit hook runs first."""
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    session = fake_native.created[-1]

    wgc._close_open_sessions()

    assert session.closed == 1
    assert backend not in wgc._open_backends


def test_cached_capture_items_are_released_at_exit(monkeypatch):
    """The extension keeps one capture item per target -- each new item costs
    a handle Windows never returns -- and an item alive when COM goes crashed
    the interpreter on exit, like an open session did."""
    from rapidshot import native

    released = []
    monkeypatch.setattr(native, "_ext", types.SimpleNamespace(
        wgc_release_items=lambda: released.append(True) or 2))
    wgc._close_open_sessions()
    assert released == [True]


def test_wgc_backend_meets_the_capture_backend_contract(fake_native):
    backend = wgc.WgcBackend(output=monitor_geometry(), device=device())
    members = set(CaptureBackend.__annotations__) | {
        name for name, value in vars(CaptureBackend).items()
        if not name.startswith("__") and callable(value)}
    assert {name for name in members if not hasattr(backend, name)} == set()
    backend.release()


def test_without_the_extension_wgc_names_the_version(monkeypatch):
    from rapidshot import native

    monkeypatch.setattr(native, "_ext", None)
    assert wgc.is_available() is False
    with pytest.raises(RapidShotConfigError, match="rapidshot-native >= 0.3.0"):
        wgc._require()


def test_window_geometry_refuses_a_closed_window(monkeypatch):
    monkeypatch.setattr(wgc._user32, "IsWindow", lambda hwnd: False)
    with pytest.raises(RapidShotSourceClosedError, match="no longer exists"):
        wgc.WindowGeometry(0x2002)
