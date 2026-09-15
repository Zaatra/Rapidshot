import ctypes
import logging
from ctypes import wintypes
from typing import List
from collections import defaultdict
import comtypes  # type: ignore[import-untyped]
from rapidshot._libs.dxgi import (
    IDXGIFactory1,
    IDXGIAdapter1,
    IDXGIOutput1,
    DXGI_ERROR_NOT_FOUND,
)
from rapidshot._libs.user32 import (
    DISPLAY_DEVICE,
    DISPLAY_DEVICE_ACTIVE,
    DISPLAY_DEVICE_PRIMARY_DEVICE,
)

# Configure logging
logger = logging.getLogger(__name__)

# Declared once, on private DLL handles. `ctypes.windll` caches one shared
# object per DLL for the whole process, so setting `argtypes` there changes the
# function for every other library in the process that reaches for it -- and
# `enum_dxgi_adapters` was doing exactly that, on every call. Guarded so this
# module still imports where the DLLs do not exist; the functions below fail
# when called, as they did before.
try:
    _dxgi = ctypes.WinDLL("dxgi", use_last_error=True)
    _dxgi.CreateDXGIFactory1.argtypes = [
        ctypes.POINTER(comtypes.GUID),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _dxgi.CreateDXGIFactory1.restype = ctypes.c_int32

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _user32.EnumDisplayDevicesW.argtypes = [
        wintypes.LPCWSTR,                   # lpDevice
        wintypes.DWORD,                     # iDevNum
        ctypes.POINTER(DISPLAY_DEVICE),     # lpDisplayDevice
        wintypes.DWORD,                     # dwFlags
    ]
    _user32.EnumDisplayDevicesW.restype = wintypes.BOOL
except (OSError, AttributeError):  # pragma: no cover - non-Windows import
    _dxgi = None
    _user32 = None


def _create_dxgi_factory1():
    """A DXGI 1.1 factory, or a COM error explaining why not."""
    pfactory = ctypes.c_void_p(0)
    hresult = _dxgi.CreateDXGIFactory1(ctypes.byref(IDXGIFactory1._iid_), ctypes.byref(pfactory))
    # The return value used to be ignored, so a failure became a null factory
    # and a "NULL COM pointer access" at the first call on it.
    if hresult < 0 or not pfactory.value:
        raise comtypes.COMError(
            hresult, f"CreateDXGIFactory1 failed ({hresult & 0xFFFFFFFF:#010x})",
            (None, None, None, 0, None))
    return ctypes.POINTER(IDXGIFactory1)(pfactory.value)


def enum_dxgi_adapters() -> List[ctypes.POINTER(IDXGIAdapter1)]:
    dxgi_factory = _create_dxgi_factory1()
    i = 0
    p_adapters = list()
    while True:
        try:
            p_adapter = ctypes.POINTER(IDXGIAdapter1)()
            dxgi_factory.EnumAdapters1(i, ctypes.byref(p_adapter))
            p_adapters.append(p_adapter)
            i += 1
        except comtypes.COMError as ce:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == ce.args[0]:
                break
            else:
                raise ce
    return p_adapters


def enum_dxgi_outputs(
    dxgi_adapter: ctypes.POINTER(IDXGIAdapter1),
) -> List[ctypes.POINTER(IDXGIOutput1)]:
    i = 0
    p_outputs = list()
    while True:
        try:
            p_output = ctypes.POINTER(IDXGIOutput1)()
            dxgi_adapter.EnumOutputs(i, ctypes.byref(p_output))
            p_outputs.append(p_output)
            i += 1
        except comtypes.COMError as ce:
            if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == ce.args[0]:
                break
            else:
                raise ce
    return p_outputs


def get_output_metadata():
    mapping_adapter = defaultdict(list)
    adapter = DISPLAY_DEVICE()
    adapter.cb = ctypes.sizeof(adapter)
    i = 0
    # Enumerate all adapters
    while _user32.EnumDisplayDevicesW(None, i, ctypes.byref(adapter), 1):
        if adapter.StateFlags & DISPLAY_DEVICE_ACTIVE != 0:
            is_primary = bool(adapter.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE)
            mapping_adapter[adapter.DeviceName] = [adapter.DeviceString, is_primary, []]
            display = DISPLAY_DEVICE()
            display.cb = ctypes.sizeof(adapter)
            j = 0
            # Enumerate Monitors
            while _user32.EnumDisplayDevicesW(
                adapter.DeviceName, j, ctypes.byref(display), 0
            ):
                mapping_adapter[adapter.DeviceName][2].append(
                    (
                        display.DeviceName,
                        display.DeviceString,
                    )
                )
                j += 1
        i += 1
    return mapping_adapter
