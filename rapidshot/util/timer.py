"""The high-resolution waitable timer the capture thread paces itself with.

Every function here is declared before use. That is not only for consistency
with `util/desktop.py`: without a `restype`, ctypes assumes the return value is
a C `int`, which is 32 bits and signed. `WaitForSingleObject` returns a DWORD,
and its failure value `WAIT_FAILED` is 0xFFFFFFFF -- as a signed int that reads
back as -1, so the capture thread's `res == WAIT_FAILED` check could never be
true and a failing timer looked like a normal tick. `CreateWaitableTimerExW`
returns a HANDLE, which is pointer-sized; truncating it to 32 bits would hand
every later call a handle that is not the one that was opened.

The DLL is opened privately rather than through `ctypes.windll`, which caches
one shared object per DLL for the whole process. Setting `argtypes` on that
shared object changes it for every other library in the process that reaches
for the same function.
"""

import ctypes
from ctypes import wintypes
from ctypes.wintypes import LARGE_INTEGER


INFINITE = 0xFFFFFFFF
WAIT_FAILED = 0xFFFFFFFF
CREATE_WAITABLE_TIMER_HIGH_RESOLUTION = 0x00000002
TIMER_MODIFY_STATE = 0x0002
TIMER_ALL_ACCESS = 0x1F0003


__kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

__kernel32.CreateWaitableTimerExW.restype = wintypes.HANDLE
__kernel32.CreateWaitableTimerExW.argtypes = [
    wintypes.LPVOID,    # lpTimerAttributes
    wintypes.LPCWSTR,   # lpTimerName
    wintypes.DWORD,     # dwFlags
    wintypes.DWORD,     # dwDesiredAccess
]

__kernel32.SetWaitableTimer.restype = wintypes.BOOL
__kernel32.SetWaitableTimer.argtypes = [
    wintypes.HANDLE,                    # hTimer
    ctypes.POINTER(LARGE_INTEGER),      # lpDueTime
    wintypes.LONG,                      # lPeriod, milliseconds
    wintypes.LPVOID,                    # pfnCompletionRoutine
    wintypes.LPVOID,                    # lpArgToCompletionRoutine
    wintypes.BOOL,                      # fResume
]

__kernel32.WaitForSingleObject.restype = wintypes.DWORD
__kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]

__kernel32.CancelWaitableTimer.restype = wintypes.BOOL
__kernel32.CancelWaitableTimer.argtypes = [wintypes.HANDLE]

__kernel32.CloseHandle.restype = wintypes.BOOL
__kernel32.CloseHandle.argtypes = [wintypes.HANDLE]


def create_high_resolution_timer():
    handle = __kernel32.CreateWaitableTimerExW(
        None, None, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS
    )
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def set_periodic_timer(handle, period: int):
    res = __kernel32.SetWaitableTimer(
        handle,
        ctypes.byref(LARGE_INTEGER(0)),
        period,
        None,
        None,
        0,
    )
    if res == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return True


wait_for_timer = __kernel32.WaitForSingleObject
cancel_timer = __kernel32.CancelWaitableTimer


def close_timer(handle):
    if handle and __kernel32.CloseHandle(handle) == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    return True
