"""Top-level window lookup, for :meth:`rapidshot.Window.find`.

A private ``user32`` handle with its own prototypes, as in ``util/desktop.py``:
setting ``argtypes`` on ``ctypes.windll.user32`` would change them for every
other caller in the process.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import List, Optional, Tuple

__all__ = ["find_window", "visible_windows"]

try:
    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _EnumWindowsProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    _user32.EnumWindows.argtypes = [_EnumWindowsProc, wintypes.LPARAM]
    _user32.EnumWindows.restype = wintypes.BOOL
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _user32.IsWindowVisible.restype = wintypes.BOOL
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextLengthW.restype = ctypes.c_int
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.GetWindowTextW.restype = ctypes.c_int
except (OSError, AttributeError):  # pragma: no cover - non-Windows import
    _user32 = None


def visible_windows() -> List[Tuple[int, str]]:
    """``(handle, title)`` of every visible top-level window with a title,
    in Z order (topmost first)."""
    if _user32 is None:
        raise OSError("Window enumeration needs Windows")
    found: List[Tuple[int, str]] = []

    def visit(hwnd, _lparam):
        if _user32.IsWindowVisible(hwnd):
            length = _user32.GetWindowTextLengthW(hwnd)
            if length > 0:
                buffer = ctypes.create_unicode_buffer(length + 1)
                _user32.GetWindowTextW(hwnd, buffer, length + 1)
                found.append((int(hwnd), buffer.value))
        return True

    _user32.EnumWindows(_EnumWindowsProc(visit), 0)
    return found


def find_window(title: str, exact: bool = False) -> Optional[int]:
    """Handle of the topmost visible window whose title matches, or None.

    Case-insensitive. ``exact`` compares the whole title; otherwise ``title``
    only has to appear in it.
    """
    wanted = title.casefold()
    for handle, text in visible_windows():
        text = text.casefold()
        if (text == wanted) if exact else (wanted in text):
            return handle
    return None
