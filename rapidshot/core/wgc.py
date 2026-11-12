"""Windows.Graphics.Capture as a capture backend (ROADMAP § 7.3).

``WgcBackend`` meets the same contract as Desktop Duplication's ``Duplicator``
(``core/backend.py``), so ``ScreenCapture`` drives it unchanged. The capture
itself is native (``rapidshot-native`` >= 0.3.0, ``native/src/wgc.rs``): WGC is
WinRT, which comtypes does not speak, and its frame-pool events belong off the
GIL.

What differs from Desktop Duplication, and why, is measured in
``build/spike-2.7/FINDINGS.md`` and summarised in ``wgc.rs``. The parts that
show here:

- Each acquire hands back the session's own shareable copy of the newest
  frame, at a stable address. The WGC frame itself is already back in its pool.
- No dirty or move rectangles and no cursor shape: ``None``, never guessed.
- The frame's time is the acquisition QPC (WGC's ``SystemRelativeTime`` lands
  after it and is not a capture time).
- A resized source ends the session with ``RapidShotReinitError``, so the
  camera rebuilds at the new size; a closed window raises
  ``RapidShotSourceClosedError``, which no rebuild can fix.
"""
import atexit
import ctypes
import logging
import weakref
from ctypes import wintypes
from typing import Optional, Tuple

from rapidshot._libs.d3d11 import ID3D11Texture2D
from rapidshot.core import hdr
from rapidshot.core.duplicator import _duplicator_ids
from rapidshot.util.errors import (
    RapidShotConfigError,
    RapidShotReinitError,
    RapidShotSourceClosedError,
)

logger = logging.getLogger(__name__)

__all__ = ["WgcBackend", "WgcMonitorGeometry", "WindowGeometry", "is_available"]

#: Shortest gap WGC may leave between frames, in microseconds. Its default
#: (16 ms on build 26300) held a 165 Hz display to 55 fps; 1 ms gave 164.
MIN_UPDATE_INTERVAL_US = 1000

#: What it costs when this Windows build refuses one of the session settings.
#: Each is optional -- capture runs without it -- but none is free.
_UNSUPPORTED_CONSEQUENCES = {
    "min_update_interval": "frames arrive at WGC's default rate, which held a 165 Hz "
                           "display to 55 fps (the setting needs Windows 11 24H2)",
    "border": "Windows draws a yellow border around the captured source on screen",
    "cursor": "the cursor may be drawn into the captured image",
}


def _native():
    from rapidshot import native

    return native


def is_available() -> bool:
    """Whether this process can capture through WGC: the extension has it and
    Windows supports it."""
    ext = getattr(_native(), "_ext", None)
    if ext is None or not hasattr(ext, "WgcSession"):
        return False
    try:
        return bool(ext.wgc_is_supported())
    except Exception:  # pragma: no cover - a broken extension
        return False


def _require():
    """The extension module, or a RapidShotConfigError naming what is missing."""
    native = _native()
    try:
        native.require_feature("WgcSession")
    except RuntimeError as e:
        raise RapidShotConfigError(
            f"backend='wgc' needs rapidshot-native >= 0.3.0. {e}") from e
    return native._ext


def _handle(value) -> int:
    """An HMONITOR/HWND as a plain int, whatever ctypes type it arrived as."""
    if isinstance(value, int):
        return value
    return int(ctypes.cast(value, ctypes.c_void_p).value or 0)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_user32.IsWindow.argtypes = [wintypes.HWND]
_user32.IsWindow.restype = wintypes.BOOL
_user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
_user32.MonitorFromWindow.restype = wintypes.HMONITOR
_MONITOR_DEFAULTTONEAREST = 2


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32)]


_user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFOEXW)]
_user32.GetMonitorInfoW.restype = wintypes.BOOL


def _monitor_device_name(hmonitor: int) -> str:
    info = _MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(info)
    if not _user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
        return ""
    return info.szDevice


class WgcMonitorGeometry:
    """A display as WGC captures it: already in its displayed orientation, so
    ``rotation_angle`` is 0 whatever Desktop Duplication would report."""

    rotation_angle = 0

    def __init__(self, output):
        self.output = output
        self._size: Tuple[int, int] = (0, 0)
        self.update_desc()

    @property
    def hmonitor(self) -> int:
        return _handle(self.output.hmonitor)

    def update_desc(self) -> None:
        self.output.update_desc()
        self._size = tuple(_require().wgc_item_size(monitor=self.hmonitor))

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._size

    @property
    def surface_size(self) -> Tuple[int, int]:
        return self._size

    @property
    def devicename(self) -> str:
        return self.output.devicename

    def __repr__(self) -> str:
        return f"<WgcMonitorGeometry {self.devicename} {self._size}>"


class WindowGeometry:
    """One window, at the size WGC captures it (its frame bounds, without the
    drop shadow), on whichever display it is on now."""

    rotation_angle = 0

    def __init__(self, hwnd: int):
        self.hwnd = int(hwnd)
        self._size: Tuple[int, int] = (0, 0)
        self._devicename = ""
        self.update_desc()

    def update_desc(self) -> None:
        if not _user32.IsWindow(self.hwnd):
            raise RapidShotSourceClosedError(f"Window 0x{self.hwnd:x} no longer exists.")
        self._size = tuple(_require().wgc_item_size(window=self.hwnd))
        if self._size[0] <= 0 or self._size[1] <= 0:
            raise RapidShotSourceClosedError(
                f"Window 0x{self.hwnd:x} has no content to capture.")
        monitor = _user32.MonitorFromWindow(self.hwnd, _MONITOR_DEFAULTTONEAREST)
        self._devicename = _monitor_device_name(monitor) if monitor else ""

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._size

    @property
    def surface_size(self) -> Tuple[int, int]:
        return self._size

    @property
    def devicename(self) -> str:
        """The display the window is on, for HDR state and SDR white."""
        return self._devicename

    def __repr__(self) -> str:
        return f"<WindowGeometry 0x{self.hwnd:x} {self._size} on {self._devicename}>"


# --------------------------------------------------------------------------
# the backend
# --------------------------------------------------------------------------

# Sessions still open when the interpreter exits are closed first. comtypes
# runs CoUninitialize from its own atexit handler, and WinRT references
# released after that crashed the process on exit (Machine B, 2026-10-22).
# atexit runs handlers last-registered-first, and comtypes registers when it
# is imported -- before this module -- so this one runs ahead of it.
_open_backends = weakref.WeakSet()


@atexit.register
def _close_open_sessions() -> None:
    for backend in list(_open_backends):
        try:
            backend.release()
        except Exception:  # pragma: no cover - best effort at exit
            pass
    # The extension also keeps one capture item per target, because each new
    # item costs a handle Windows never returns; those must go before COM
    # does, for the same reason as the sessions.
    ext = getattr(_native(), "_ext", None)
    release_items = getattr(ext, "wgc_release_items", None)
    if release_items is not None:
        try:
            release_items()
        except Exception:  # pragma: no cover - best effort at exit
            pass


# GraphicsCapture.dll stays loaded until the process ends, once a session has
# loaded it. Closing first (above) and waiting for the teardown (wgc.rs
# `close()`) were not enough: WGC finishes a session on its own threads, and
# comtypes' CoUninitialize at exit unloads in-process WinRT servers, so work
# WGC still had queued ran in an unmapped DLL. Every such crash Windows
# recorded (Event 1000) faulted in "GraphicsCapture.dll_unloaded" at one
# offset. On an HDR desktop (Arc 130V, 2026-11-12), 120 runs each: a camera
# left open after one grab crashed 6 times, one left open with no grab 74;
# with the DLL pinned, 0 and 0. Pinning only keeps mapped what is in use.
# The native session pins it as well (wgc.rs, `pin_graphics_capture`), which
# covers Rust callers; this covers extensions built before that.
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.GetModuleHandleExW.argtypes = [
    wintypes.DWORD, wintypes.LPCWSTR, ctypes.POINTER(wintypes.HMODULE)]
_kernel32.GetModuleHandleExW.restype = wintypes.BOOL
_GET_MODULE_HANDLE_EX_FLAG_PIN = 0x1
_GRAPHICS_CAPTURE_DLL = "GraphicsCapture.dll"
_graphics_capture_pinned = False


def _pin_graphics_capture() -> bool:
    """Keep GraphicsCapture.dll mapped until the process exits. Idempotent.

    Called once a session exists, which is what has loaded the DLL. Returns
    whether it is pinned; a failure is logged, not raised -- capture works
    without it, and only a process exiting with WGC work in flight is at risk.
    """
    global _graphics_capture_pinned
    if _graphics_capture_pinned:
        return True
    module = wintypes.HMODULE()
    if _kernel32.GetModuleHandleExW(_GET_MODULE_HANDLE_EX_FLAG_PIN, _GRAPHICS_CAPTURE_DLL,
                                    ctypes.byref(module)):
        _graphics_capture_pinned = True
    else:
        logger.debug("could not pin %s (error %d); a process that exits with a WGC "
                     "session still tearing down may crash", _GRAPHICS_CAPTURE_DLL,
                     ctypes.get_last_error())
    return _graphics_capture_pinned


def _texture_from_address(address: int):
    """A comtypes ``ID3D11Texture2D`` for a texture the session owns.

    The address is borrowed. comtypes releases what it wraps when the wrapper
    is collected, so the reference it will give back is taken here.
    """
    texture = ctypes.cast(address, ctypes.POINTER(ID3D11Texture2D))
    texture.AddRef()
    return texture


class WgcBackend:
    """One WGC capture session, satisfying ``core.backend.CaptureBackend``.

    Built the way ``Duplicator`` is -- ``WgcBackend(output=, device=,
    timeout_ms=)`` -- where ``output`` is a ``WgcMonitorGeometry`` or
    ``WindowGeometry``.
    """

    def __init__(self, output, device, timeout_ms: int = 10,
                 pixel_format: Optional[str] = None, cursor: bool = False):
        ext = _require()
        geometry = output
        self.geometry = geometry
        self.device = device
        self.timeout_ms = timeout_ms
        self.instance_id = next(_duplicator_ids)
        self.frame_serial = 0
        self.texture = None
        self.updated = False
        self._frame_acquired = False
        self.last_present_time = 0
        self.accumulated_frames = 0
        self.dirty_rects = None
        self.move_rects = None
        self.rects_coalesced = False
        self.protected_content_detected = False
        self.cursor_visible = False
        self.cursor = None
        self.last_error = ""

        if pixel_format is None:
            # Linear FP16 on an HDR display, as Desktop Duplication delivers
            # it; the CPU paths convert, and GpuConverter knows the colour
            # space from the frame.
            pixel_format = ("fp16" if hdr.display_color(geometry.devicename).hdr
                            else "bgra")
        target = ({"window": geometry.hwnd} if isinstance(geometry, WindowGeometry)
                  else {"monitor": geometry.hmonitor})
        device_ptr = ctypes.cast(device.device, ctypes.c_void_p).value
        try:
            self._session = ext.WgcSession(
                device_ptr, pixel_format=pixel_format, cursor=cursor,
                min_update_interval_us=MIN_UPDATE_INTERVAL_US, **target)
        except RuntimeError as e:
            raise RapidShotConfigError(f"Windows.Graphics.Capture refused {geometry}: {e}") from e
        _pin_graphics_capture()
        for setting in self._session.unsupported:
            logger.warning(
                "Windows.Graphics.Capture on this Windows build does not support the %s "
                "setting, so %s.", setting.replace("_", " "),
                _UNSUPPORTED_CONSEQUENCES.get(setting, "capture runs without it"))
        _open_backends.add(self)

    @property
    def pixel_format(self) -> str:
        return self._session.pixel_format if self._session is not None else ""

    def update_frame(self) -> bool:
        """``CaptureBackend.update_frame``: acquire the newest frame, if any."""
        self.updated = False
        self._frame_acquired = False
        if self._session is None:
            return False
        try:
            result = self._session.acquire(int(self.timeout_ms))
        except RuntimeError as e:
            self.last_error = str(e)
            raise RapidShotReinitError(f"WGC acquire failed: {e}") from e

        status = result["status"]
        if status == "timeout":
            return True
        if status == "resized":
            raise RapidShotReinitError(
                f"{self.geometry} changed size to {result['width']}x{result['height']}; "
                "rebuilding the capture at the new size.")
        if status == "closed":
            raise RapidShotSourceClosedError(f"{self.geometry} was closed.")

        self.texture = _texture_from_address(result["texture"])
        self.frame_serial = int(result["serial"])
        # When WGC signalled the frame, not when it was taken: the difference
        # is time spent waiting in the pool, which is part of its age.
        # Older extensions report only acquisition.
        self.last_present_time = int(result.get("arrived_qpc", result["acquired_qpc"]))
        self.accumulated_frames = 1 + int(result["skipped"])
        self.updated = True
        self._frame_acquired = True
        return True

    def release_frame(self) -> None:
        """``CaptureBackend.release_frame``. The WGC frame went back to its
        pool at acquisition; this only drops the reference to the copy."""
        self.texture = None
        self._frame_acquired = False

    def release(self) -> None:
        """``CaptureBackend.release``. Idempotent."""
        self.release_frame()
        session, self._session = getattr(self, "_session", None), None
        if session is not None:
            session.close()
        _open_backends.discard(self)

    def __repr__(self) -> str:
        return f"<WgcBackend {self.geometry} id={self.instance_id}>"
