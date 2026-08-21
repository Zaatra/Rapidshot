"""Which desktop is this thread on, and is it the one receiving input?

``DuplicateOutput`` refuses with a bare ``E_ACCESSDENIED`` for two unrelated
reasons: protected (HDCP/DRM) content, and the calling thread not being on the
input desktop. RapidShot used to report both as protected content, so a locked
workstation, an open UAC prompt, or a Session-0 service all told the user to
"close or move the protected player window" -- a window that does not exist.

Measured 2026-08-06: a thread moved to a fresh desktop with ``CreateDesktop``
gets exactly ``0x80070005`` from ``DuplicateOutput``, indistinguishable from the
protected-content refusal by HRESULT alone. It has to be told apart by asking
the window station directly, which is what this module does.

Only the *generic* ``E_ACCESSDENIED`` is ambiguous. ``DXGI_ERROR_ACCESS_DENIED``
and ``DXGI_ERROR_CANNOT_PROTECT_CONTENT`` are specific to protected content and
need none of this.
"""
from __future__ import annotations

import ctypes
from typing import Optional

__all__ = ["DesktopState", "describe_desktop_access"]

_UOI_NAME = 2
_GENERIC_READ = 0x80000000

try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.GetThreadDesktop.restype = ctypes.c_void_p
    _user32.GetThreadDesktop.argtypes = [ctypes.c_ulong]
    _user32.OpenInputDesktop.restype = ctypes.c_void_p
    _user32.OpenInputDesktop.argtypes = [ctypes.c_ulong, ctypes.c_int,
                                         ctypes.c_ulong]
    _user32.CloseDesktop.argtypes = [ctypes.c_void_p]
    _user32.GetUserObjectInformationW.restype = ctypes.c_int
    _user32.GetUserObjectInformationW.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
    ]
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.GetCurrentThreadId.restype = ctypes.c_ulong
except (OSError, AttributeError):  # pragma: no cover - non-Windows import
    _user32 = None
    _kernel32 = None


class DesktopState:
    """What the desktop situation is, and whether it explains a refusal.

    Attributes:
        thread_desktop: Name of the desktop this thread is attached to, or None.
        input_desktop: Name of the desktop currently receiving input, or None
            when it could not be opened -- which is itself meaningful, since
            the secure desktop refuses to be opened at all.
        is_input_desktop: True only when the two are known and equal.
        blocked_reason: A one-line explanation when capture cannot work from
            here, otherwise None.
    """

    __slots__ = ("thread_desktop", "input_desktop", "is_input_desktop",
                 "blocked_reason")

    def __init__(self, thread_desktop, input_desktop, is_input_desktop,
                 blocked_reason):
        self.thread_desktop = thread_desktop
        self.input_desktop = input_desktop
        self.is_input_desktop = is_input_desktop
        self.blocked_reason = blocked_reason

    def __repr__(self) -> str:
        return (f"<DesktopState thread={self.thread_desktop!r} "
                f"input={self.input_desktop!r} "
                f"is_input={self.is_input_desktop}>")


def _desktop_name(handle) -> Optional[str]:
    if not handle:
        return None
    needed = ctypes.c_ulong(0)
    _user32.GetUserObjectInformationW(handle, _UOI_NAME, None, 0,
                                      ctypes.byref(needed))
    if not needed.value:
        return None
    buf = ctypes.create_unicode_buffer(needed.value // 2 + 1)
    if not _user32.GetUserObjectInformationW(handle, _UOI_NAME, buf,
                                             needed.value,
                                             ctypes.byref(needed)):
        return None
    return buf.value


def describe_desktop_access() -> DesktopState:
    """Report whether this thread can see the desktop it is trying to capture.

    Never raises: this runs while building an error message, and a diagnostic
    that can itself fail would replace a misleading message with no message.
    """
    if _user32 is None or _kernel32 is None:  # pragma: no cover - non-Windows
        return DesktopState(None, None, None, None)

    thread_handle = None
    input_handle = None
    try:
        thread_handle = _user32.GetThreadDesktop(
            _kernel32.GetCurrentThreadId())
        thread_name = _desktop_name(thread_handle)

        # OpenInputDesktop fails outright when the input desktop is the secure
        # one (lock screen, UAC prompt, Ctrl+Alt+Del), which is exactly the
        # case worth naming.
        input_handle = _user32.OpenInputDesktop(0, False, _GENERIC_READ)
        input_name = _desktop_name(input_handle) if input_handle else None

        if not input_handle:
            return DesktopState(
                thread_name, None, False,
                "the input desktop cannot be opened, which means the secure "
                "desktop is active — the workstation is locked, a UAC prompt "
                "is open, or Ctrl+Alt+Del is showing. Desktop Duplication "
                "cannot capture the secure desktop.")

        if thread_name and input_name and thread_name != input_name:
            return DesktopState(
                thread_name, input_name, False,
                f"this thread is attached to desktop {thread_name!r} but "
                f"{input_name!r} is receiving input. Desktop Duplication only "
                f"works on the input desktop — a Session 0 service or a "
                f"thread moved with SetThreadDesktop cannot capture the user's "
                f"screen directly.")

        return DesktopState(thread_name, input_name, True, None)
    except Exception:  # pragma: no cover - diagnostic must never raise
        return DesktopState(None, None, None, None)
    finally:
        if input_handle:
            try:
                _user32.CloseDesktop(ctypes.c_void_p(input_handle))
            except Exception:
                pass
