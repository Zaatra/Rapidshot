"""Windows.Graphics.Capture, live: monitors and real windows (ROADMAP § 7.3).

Skipped unless this process can capture through WGC -- rapidshot-native
>= 0.3.0 on a Windows build that supports it, with a desktop. The window
target is a Tk Toplevel this module paints, moves, resizes and destroys, so
every expectation is something the test itself drew.

The headless half, which runs everywhere, is tests/test_wgc_backend.py.
"""
import subprocess
import sys
import time

import numpy as np
import pytest

import rapidshot
from rapidshot import Monitor, Window

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

from rapidshot.core import wgc  # noqa: E402

from test_gpu_converter import cpu_readback, nearest_reference  # noqa: E402
from _hdr_desktop import BGRA8_REFERENCE, skip_on_an_hdr_desktop  # noqa: E402

def _why_not():
    """Why WGC cannot capture here, or None. Asked once, at collection.

    Availability alone is not enough: a CI runner can have the API and the
    extension but no interactive desktop to capture, and these tests must
    skip there, not fail.
    """
    if not wgc.is_available():
        return "Windows.Graphics.Capture not available"
    try:
        rapidshot.create(source=Monitor(), backend="wgc").release()
    except Exception as e:  # noqa: BLE001 -- any refusal means "not here"
        return f"cannot capture through WGC here: {e}"
    return None


_UNAVAILABLE = _why_not()
pytestmark = pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or "")

#: The colour the test window is painted, as RGB.
PAINT = (51, 102, 204)
PAINT_HEX = "#3366cc"


def _grab(camera, root=None, tries=200):
    """The next frame grab() returns, pumping Tk so the window paints."""
    for _ in range(tries):
        if root is not None:
            root.update()
        image = camera.grab()
        if image is not None:
            return np.array(image)
        time.sleep(0.01)
    return None


@pytest.fixture
def painted_window(tk_root):
    """A solid-colour top-level window, and its frame HWND."""
    import tkinter as tk

    window = tk.Toplevel(tk_root, background=PAINT_HEX)
    window.geometry("480x320+120+120")
    window.attributes("-topmost", True)
    window.title("rapidshot wgc test")
    tk_root.update()
    hwnd = int(window.wm_frame(), 16)
    yield window, hwnd
    try:
        window.destroy()
        tk_root.update()
    except tk.TclError:
        pass


# --------------------------------------------------------------------------
# monitors
# --------------------------------------------------------------------------

def test_a_monitor_captures_through_wgc():
    camera = rapidshot.create(source=Monitor(), backend="wgc", output_color="BGRA")
    try:
        assert camera.backend == "wgc"
        frame = None
        for _ in range(200):
            frame = camera.grab_frame()
            if frame is not None:
                break
            time.sleep(0.01)
        assert frame is not None, "WGC delivers a first frame without any screen change"
        with frame:
            assert frame.backend == "wgc"
            assert (frame.width, frame.height) == camera._output.resolution
            assert frame.dirty_rects is None and frame.move_rects is None
            info = rapidshot.native.texture_sharing_info(frame)
            assert info["shared_nthandle"], "the copy must be shareable with D3D12"
            # The frame's time is its arrival: never in the future, and on a
            # first frame taken straight away, well under a second old.
            assert 0 <= frame.age_ms < 1000, frame.age_ms
    finally:
        camera.release()


def test_gpu_converter_reads_a_wgc_frame_exactly():
    """The D3D12 path over WGC's copy, against an independent D3D11 readback
    of the same texture -- the capture-order fence is what makes this hold
    (7 stale reads in 150 without it, in the spike)."""
    camera = rapidshot.create(source=Monitor(), backend="wgc", output_color="BGRA")
    try:
        frame = None
        for _ in range(200):
            frame = camera.grab_frame()
            if frame is not None:
                break
            time.sleep(0.01)
        assert frame is not None
        with frame:
            skip_on_an_hdr_desktop(frame, BGRA8_REFERENCE)   # `with` releases it
            converter = rapidshot.GpuConverter(frame, (64, 64), dtype="float32",
                                               sampling="nearest")
            got = converter.process(frame).numpy()
            expected = nearest_reference(cpu_readback(camera, frame), 64, 64)
            np.testing.assert_allclose(got, expected, atol=2e-3)
            converter.close()
    finally:
        camera.release()


# --------------------------------------------------------------------------
# windows
# --------------------------------------------------------------------------

def test_a_window_is_captured_with_its_own_pixels(painted_window, tk_root):
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="RGB")
    try:
        assert camera.backend == "wgc"
        image = _grab(camera, tk_root)
        assert image is not None
        height, width = image.shape[:2]
        # The capture is the window's frame bounds: the client area plus the
        # title bar and borders, but never less than what was asked for.
        assert width >= 480 and height >= 320
        centre = image[height // 2, width // 2]
        assert np.abs(centre.astype(int) - PAINT).max() <= 2, centre
    finally:
        camera.release()


def test_a_resized_window_is_recaptured_at_its_new_size(painted_window, tk_root):
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="RGB")
    try:
        before = _grab(camera, tk_root)
        assert before is not None
        generation = camera.generation

        window.geometry("640x400+120+120")
        tk_root.update()
        after = None
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            after = _grab(camera, tk_root, tries=20)
            if after is not None and after.shape[:2] != before.shape[:2]:
                break
        assert after is not None and after.shape[:2] != before.shape[:2]
        assert camera.generation > generation, "a resize is a rebuild"
        assert after.shape[1] - before.shape[1] == 160
        assert after.shape[0] - before.shape[0] == 80
    finally:
        camera.release()


def test_a_closed_window_ends_capture(painted_window, tk_root):
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="RGB")
    try:
        assert _grab(camera, tk_root) is not None
        window.destroy()
        tk_root.update()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not camera._capture_permanently_failed:
            assert camera.grab() is None or not camera._capture_permanently_failed
        assert camera._capture_permanently_failed
        assert "closed" in camera.last_capture_error or "no longer exists" in \
            camera.last_capture_error
    finally:
        camera.release()


def test_a_minimised_window_blocks_instead_of_spinning(painted_window, tk_root):
    """No frames arrive while minimised; each grab waits its timeout rather
    than returning at once and spinning a capture loop."""
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="RGB", timeout_ms=50)
    try:
        assert _grab(camera, tk_root) is not None
        window.iconify()
        tk_root.update()
        time.sleep(0.2)
        while camera.grab() is not None:      # drain anything from before
            pass
        start = time.perf_counter()
        results = [camera.grab() for _ in range(5)]
        elapsed = time.perf_counter() - start
        assert results == [None] * 5
        assert elapsed >= 5 * 0.04, f"{elapsed:.3f}s: grabs returned without waiting"
        assert not camera._capture_permanently_failed
    finally:
        camera.release()


@pytest.mark.parametrize("grab", [True, False], ids=["after-a-grab", "no-grab"])
def test_an_open_wgc_camera_does_not_crash_interpreter_exit(grab):
    """A WGC session closed moments before the process exits crashed it:
    WGC finishes tearing a session down on its own threads after Close()
    returns. core/wgc.py closes open sessions in an atexit hook, close()
    waits for the capture device's reference count to settle (wgc.rs), and
    GraphicsCapture.dll is pinned so comtypes' CoUninitialize cannot unload
    it under WGC's remaining work (core/wgc.py, ``_pin_graphics_capture``)."""
    code = ("import rapidshot, time\n"
            "camera = rapidshot.create(source=rapidshot.Monitor(), backend='wgc')\n"
            "time.sleep(0.2)\n"
            + ("camera.grab()\n" if grab else "")
            + "print('left open')\n")
    # Three runs, not one: before close() waited for WGC's teardown, this
    # crashed 1 in 10 for a released camera and 8 in 10 for an open one; with
    # the wait and no pin, on an HDR desktop, 6 in 120 after a grab and 74 in
    # 120 with none -- the case that makes three runs a real check.
    for _ in range(3):
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=60)
        assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
        assert "left open" in result.stdout


# --------------------------------------------------------------------------
# lifetime: continuous capture, held frames, create and destroy
# --------------------------------------------------------------------------

def test_continuous_capture_starts_and_stops_repeatedly(painted_window, tk_root):
    """start() drives the session from the capture thread, not the thread
    that created it -- the reason WgcSession is not `unsendable`."""
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="RGB")
    try:
        for attempt in range(3):
            camera.start(target_fps=30, video_mode=True)
            try:
                deadline = time.monotonic() + 1.0
                while time.monotonic() < deadline:
                    window.configure(background=("#3366cc", "#cc6633")[attempt % 2])
                    tk_root.update()
                    time.sleep(0.02)
                image = camera.get_latest_frame()
            finally:
                camera.stop()
            assert image is not None, f"no frame on start {attempt + 1}"
            assert np.asarray(image).ndim == 3
            assert not camera.is_capturing
        assert not camera._capture_permanently_failed, camera.last_capture_error
    finally:
        camera.release()


def test_a_held_frame_does_not_change_while_the_source_does(painted_window, tk_root):
    """The session copies each frame into one texture, so the next copy must
    never land while a frame is held. The camera refuses to acquire while one
    is live; this is the evidence that the pixels hold still."""
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="BGRA")
    try:
        frame = None
        for _ in range(200):
            tk_root.update()
            frame = camera.grab_frame()
            if frame is not None:
                break
            time.sleep(0.01)
        assert frame is not None
        with frame:
            before = cpu_readback(camera, frame)
            for colour in ("#cc6633", "#33cc66", "#cc33cc"):
                window.configure(background=colour)
                tk_root.update()
                time.sleep(0.07)
            after = cpu_readback(camera, frame)
        np.testing.assert_array_equal(before, after)

        # ...and capture carried on: the next frame shows the new colour.
        image = _grab(camera, tk_root)
        assert image is not None
        centre = image[image.shape[0] // 2, image.shape[1] // 2, :3]
        assert np.abs(centre.astype(int) - (204, 51, 204)).max() <= 2, centre   # BGRA of #cc33cc
    finally:
        camera.release()


def test_creating_and_destroying_cameras_holds_no_gpu_memory(painted_window, tk_root):
    """25 WGC cameras made, used and released. Measured with DXGI's own
    accounting: Task Manager's per-process figure charges every opening of a
    shared surface and never takes it back (README, Known limitations)."""
    import gc

    from rapidshot.util.gpu_memory import process_gpu_memory_mb

    window, hwnd = painted_window

    def cycle():
        camera = rapidshot.create(source=Window(hwnd), output_color="BGRA")
        try:
            assert _grab(camera, tk_root) is not None
        finally:
            camera.release()

    cycle()                                     # first-use allocations
    gc.collect()
    before = process_gpu_memory_mb()
    for _ in range(25):
        cycle()
    gc.collect()
    after = process_gpu_memory_mb()

    growth = (after["local"] - before["local"]) + (after["non_local"] - before["non_local"])
    # One 484x367 window surface is under 1 MiB; 25 leaked would be ~17 MiB.
    assert growth < 4, (before, after)


def test_a_slow_consumer_still_gets_the_current_frame(painted_window, tk_root):
    """WGC drops new frames once every pool buffer holds one, and never
    delivers them later. With frames left in the pool until asked for, a
    window changed 3 times between grabs came back showing its 2nd colour
    and stayed that way (2026-10-29). The session now drains the pool as
    frames arrive, keeping only the newest."""
    window, hwnd = painted_window
    camera = rapidshot.create(source=Window(hwnd), output_color="BGRA")
    try:
        assert _grab(camera, tk_root) is not None
        colours = ("#ff0000", "#00ff00", "#0000ff", "#ffff00", "#00ffff")
        for colour in colours:                       # five changes, no grabs
            window.configure(background=colour)
            tk_root.update()
            time.sleep(0.08)
        time.sleep(0.3)
        image = None
        for _ in range(50):
            image = camera.grab()
            if image is not None:
                break
            time.sleep(0.01)
        assert image is not None
        image = np.asarray(image)
        centre = image[image.shape[0] // 2, image.shape[1] // 2, :3]
        assert np.abs(centre.astype(int) - (255, 255, 0)).max() <= 2, centre    # BGR of #00ffff
    finally:
        camera.release()
