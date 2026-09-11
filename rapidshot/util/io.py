import ctypes
import logging
from ctypes import wintypes
from typing import List
from collections import defaultdict
import comtypes  # type: ignore[import-untyped]
from rapidshot._libs.dxgi import (
    IDXGIFactory1,
    IDXGIFactory6,  # Added this import
    IDXGIAdapter1,
    IDXGIOutput1,
    DXGI_ERROR_NOT_FOUND,
    # Add missing GPU preference constants
    DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
    DXGI_GPU_PREFERENCE_UNSPECIFIED,
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
    _dxgi.CreateDXGIFactory1(ctypes.byref(IDXGIFactory1._iid_), ctypes.byref(pfactory))
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


def enum_dxgi_adapters_with_preference(gpu_preference=DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) -> List[ctypes.POINTER(IDXGIAdapter1)]:
    """
    Enumerate DXGI adapters with a preference for high performance or power efficiency.
    Falls back to standard enumeration if DXGI 1.6 is not available.
    
    Args:
        gpu_preference: DXGI_GPU_PREFERENCE value
        
    Returns:
        List of adapter pointers
    """
    # Try to create a DXGI 1.6 factory
    try:
        dxgi_factory = _create_dxgi_factory1()

        # Try to query for DXGI 1.6 factory
        try:
            dxgi_factory6 = dxgi_factory.QueryInterface(IDXGIFactory6)
            p_adapters = list()
            i = 0
            
            # Use GPU preference enumeration
            while True:
                try:
                    p_adapter = ctypes.POINTER(IDXGIAdapter1)()
                    dxgi_factory6.EnumAdapterByGpuPreference(
                        i, 
                        gpu_preference,
                        IDXGIAdapter1._iid_,
                        ctypes.byref(ctypes.cast(ctypes.byref(p_adapter), ctypes.POINTER(ctypes.c_void_p)))
                    )
                    p_adapters.append(p_adapter)
                    i += 1
                except comtypes.COMError as ce:
                    if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == ce.args[0]:
                        break
                    else:
                        raise ce
                        
            logger.info(f"Enumerated {len(p_adapters)} adapters using DXGI 1.6 EnumAdapterByGpuPreference")
            return p_adapters
            
        except comtypes.COMError:
            # DXGI 1.6 not available, fall back to standard enumeration
            logger.info("DXGI 1.6 not available, falling back to standard adapter enumeration")
            return enum_dxgi_adapters()
    except Exception as e:
        logger.error(f"Failed to enumerate adapters with preference: {e}")
        # Fall back to standard enumeration
        return enum_dxgi_adapters()


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
