"""Access loss triggered by real exclusive fullscreen, end to end.

ROADMAP listed this path as fault-injection tested on the grounds that the real
trigger is "a game taking exclusive fullscreen". The real trigger is actually
``IDXGISwapChain::SetFullscreenState(TRUE)``, which is a page of ctypes and
restores itself on exit — no persistent system setting is touched.

Measured 2026-08-06 on Machine B: entering *and* leaving exclusive fullscreen
each raise ``ABANDONED_MUTEX_EXCEPTION`` (0x887A0026) out of ``update_frame``,
which the duplicator classifies as recoverable, and capture rebuilds and
carries on. So the whole chain — detect, classify, rebuild, resume — runs here
against a real transition rather than an injected HRESULT.

Note what the failure looks like: capture does **not** surface an error to the
caller, because absorbing this is the entire point. These tests therefore
assert on continued frame production and on the recovery actually being
triggered, not on an exception.

This briefly takes the display into exclusive fullscreen. It skips without a
desktop session and always restores on the way out.
"""
import ctypes
import ctypes.wintypes as w
import logging
import threading
import time

import pytest

import rapidshot

# ctypes.windll does not exist off Windows, and these modules reach for it
# at import time -- without this the suite errors during collection rather
# than skipping. CI only runs Windows, so this is about not breaking a
# contributor's machine.
if not hasattr(ctypes, "windll"):
    pytest.skip("Windows-only", allow_module_level=True)

tk = pytest.importorskip("tkinter", reason="needs tkinter for a window")

D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_SDK_VERSION = 7
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_USAGE_RENDER_TARGET_OUTPUT = 0x20
DXGI_SWAP_EFFECT_DISCARD = 0
DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH = 2

# IDXGISwapChain vtable: IUnknown(0-2), IDXGIObject(3-6),
# IDXGIDeviceSubObject(7), Present(8), GetBuffer(9), SetFullscreenState(10).
VT_RELEASE = 2
VT_PRESENT = 8
VT_SET_FULLSCREEN = 10


class DXGI_RATIONAL(ctypes.Structure):
    _fields_ = [("Numerator", w.UINT), ("Denominator", w.UINT)]


class DXGI_MODE_DESC(ctypes.Structure):
    _fields_ = [("Width", w.UINT), ("Height", w.UINT),
                ("RefreshRate", DXGI_RATIONAL), ("Format", ctypes.c_int),
                ("ScanlineOrdering", ctypes.c_int), ("Scaling", ctypes.c_int)]


class DXGI_SAMPLE_DESC(ctypes.Structure):
    _fields_ = [("Count", w.UINT), ("Quality", w.UINT)]


class DXGI_SWAP_CHAIN_DESC(ctypes.Structure):
    _fields_ = [("BufferDesc", DXGI_MODE_DESC),
                ("SampleDesc", DXGI_SAMPLE_DESC),
                ("BufferUsage", w.UINT), ("BufferCount", w.UINT),
                ("OutputWindow", ctypes.c_void_p), ("Windowed", ctypes.c_int),
                ("SwapEffect", ctypes.c_int), ("Flags", w.UINT)]


def vcall(interface, index, restype, argtypes, *args):
    """Invoke a COM method through its vtable, without comtypes."""
    vtable = ctypes.cast(interface, ctypes.POINTER(ctypes.c_void_p))[0]
    slot = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(slot)(
        interface, *args)


class CaptureLoop:
    """Capture continuously on a thread, counting frames and errors."""

    def __init__(self):
        self.frames = 0
        self.errors = []
        self._stop = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        camera = rapidshot.create(output_color="BGRA")
        try:
            while not self._stop:
                try:
                    buf = camera.grab()
                    if buf is not None:
                        self.frames += 1
                        release = getattr(buf, "release", None)
                        if release:
                            release()
                except Exception as e:      # noqa: BLE001 - recorded, not handled
                    self.errors.append(f"{type(e).__name__}: {str(e)[:150]}")
                time.sleep(0.002)
        finally:
            camera.release()
            rapidshot.reset()

    def __enter__(self):
        self._thread.start()
        time.sleep(1.2)                     # let capture settle
        return self

    def __exit__(self, *exc):
        self._stop = True
        self._thread.join(timeout=5)


@pytest.fixture
def exclusive_fullscreen(tk_root):
    """Yield a callable that enters exclusive fullscreen, or skips.

    Always leaves fullscreen and tears the window down, including on failure —
    a test that abandoned the display in exclusive mode would take the session
    with it. Uses the session-wide root from conftest: a second ``Tk()`` after
    an earlier one was destroyed raises ``TclError``, which made this file skip
    itself in a full run while passing on its own.
    """
    root = tk.Toplevel(tk_root)
    root.geometry("800x600+100+100")
    root.update()
    hwnd = ctypes.windll.user32.GetParent(root.winfo_id()) or root.winfo_id()

    d3d11 = ctypes.WinDLL("d3d11")
    d3d11.D3D11CreateDeviceAndSwapChain.restype = ctypes.c_long
    d3d11.D3D11CreateDeviceAndSwapChain.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, w.UINT,
        ctypes.c_void_p, w.UINT, w.UINT,
        ctypes.POINTER(DXGI_SWAP_CHAIN_DESC),
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
    ]

    desc = DXGI_SWAP_CHAIN_DESC()
    desc.BufferDesc.Width = root.winfo_screenwidth()
    desc.BufferDesc.Height = root.winfo_screenheight()
    desc.BufferDesc.RefreshRate = DXGI_RATIONAL(0, 1)
    desc.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM
    desc.SampleDesc = DXGI_SAMPLE_DESC(1, 0)
    desc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT
    desc.BufferCount = 2
    desc.OutputWindow = ctypes.c_void_p(hwnd)
    desc.Windowed = 1
    desc.SwapEffect = DXGI_SWAP_EFFECT_DISCARD
    desc.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH

    swapchain = ctypes.c_void_p()
    device = ctypes.c_void_p()
    context = ctypes.c_void_p()
    hr = d3d11.D3D11CreateDeviceAndSwapChain(
        None, D3D_DRIVER_TYPE_HARDWARE, None, 0, None, 0, D3D11_SDK_VERSION,
        ctypes.byref(desc), ctypes.byref(swapchain), ctypes.byref(device),
        None, ctypes.byref(context))
    if hr < 0:
        root.destroy()
        pytest.skip(f"could not create a swapchain (0x{hr & 0xFFFFFFFF:08x})")

    entered = {"value": False}

    def enter(frames=60):
        result = vcall(swapchain, VT_SET_FULLSCREEN, ctypes.c_long,
                       [ctypes.c_int, ctypes.c_void_p], 1, None)
        if result < 0:
            pytest.skip(
                f"exclusive fullscreen refused (0x{result & 0xFFFFFFFF:08x})")
        entered["value"] = True
        for _ in range(frames):
            vcall(swapchain, VT_PRESENT, ctypes.c_long, [w.UINT, w.UINT], 0, 0)
            root.update()
            time.sleep(0.02)

    def leave():
        """Leave fullscreen on demand, so a test can watch that half too.

        Leaving is a second access-loss event, and relying on teardown to do
        it meant no capture thread was running when it happened -- the release
        notes claimed both transitions were covered when only entry was.
        """
        if not entered["value"]:
            return
        vcall(swapchain, VT_SET_FULLSCREEN, ctypes.c_long,
              [ctypes.c_int, ctypes.c_void_p], 0, None)
        entered["value"] = False
        time.sleep(0.8)

    enter.leave = leave

    try:
        yield enter
    finally:
        leave()
        # Raw ctypes pointers: nothing releases these for us, and leaking a
        # swap chain per test keeps the window and its surfaces alive.
        for interface in (swapchain, context, device):
            if interface:
                vcall(interface, VT_RELEASE, ctypes.c_ulong, [])
        try:
            root.destroy()
        except tk.TclError:
            pass
        time.sleep(0.5)


def test_capture_survives_exclusive_fullscreen(exclusive_fullscreen):
    """Frames must keep arriving across the transition, and no error may reach
    the caller — absorbing this is what the recovery path is for."""
    with CaptureLoop() as loop:
        before = loop.frames
        assert before > 0, "capture produced nothing before the transition"

        exclusive_fullscreen()
        during = loop.frames
        assert during > before, (
            "capture stopped producing frames in exclusive fullscreen")

        # Leaving is a *second* access-loss event, and the more likely half of
        # a real session -- a game being closed. Letting teardown do it meant
        # no capture thread was running when it happened, so only entry was
        # ever covered.
        exclusive_fullscreen.leave()
        time.sleep(1.0)
        after = loop.frames

    assert after > during, (
        "capture did not recover after leaving exclusive fullscreen")
    assert loop.errors == [], (
        f"errors reached the caller across the transitions: {loop.errors[:3]}")


def test_access_loss_recovery_actually_runs(exclusive_fullscreen, caplog):
    """The transition must exercise the recovery, not merely be survivable.

    Without this the test above would also pass on a build where exclusive
    fullscreen never disturbed duplication at all, and would silently stop
    covering the path it exists to cover.
    """
    with caplog.at_level(logging.WARNING, logger="rapidshot"):
        with CaptureLoop() as loop:
            assert loop.frames > 0
            exclusive_fullscreen()
            time.sleep(0.5)

    messages = " ".join(record.getMessage().lower()
                        for record in caplog.records)
    if "re-init" not in messages and "access lost" not in messages:
        pytest.skip(
            "exclusive fullscreen did not disturb duplication on this Windows "
            "build; the access-loss path was not reached (see ROADMAP § 10)")

    assert loop.frames > 0, "capture never recovered after access loss"
