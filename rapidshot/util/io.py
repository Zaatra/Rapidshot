import ctypes
import logging
from typing import List
from collections import defaultdict
import comtypes
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
    MONITORINFOEXW,
    DISPLAY_DEVICE_ACTIVE,
    DISPLAY_DEVICE_PRIMARY_DEVICE,
)

# Configure logging
logger = logging.getLogger(__name__)


def enum_dxgi_adapters() -> List[ctypes.POINTER(IDXGIAdapter1)]:
    create_dxgi_factory = ctypes.windll.dxgi.CreateDXGIFactory1
    create_dxgi_factory.argtypes = (comtypes.GUID, ctypes.POINTER(ctypes.c_void_p))
    create_dxgi_factory.restype = ctypes.c_int32
    pfactory = ctypes.c_void_p(0)
    create_dxgi_factory(IDXGIFactory1._iid_, ctypes.byref(pfactory))
    dxgi_factory = ctypes.POINTER(IDXGIFactory1)(pfactory.value)
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
        create_dxgi_factory = ctypes.windll.dxgi.CreateDXGIFactory1
        create_dxgi_factory.argtypes = (comtypes.GUID, ctypes.POINTER(ctypes.c_void_p))
        create_dxgi_factory.restype = ctypes.c_int32
        pfactory = ctypes.c_void_p(0)
        create_dxgi_factory(IDXGIFactory1._iid_, ctypes.byref(pfactory))
        dxgi_factory = ctypes.POINTER(IDXGIFactory1)(pfactory.value)
        
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
    while ctypes.windll.user32.EnumDisplayDevicesW(0, i, ctypes.byref(adapter), 1):
        if adapter.StateFlags & DISPLAY_DEVICE_ACTIVE != 0:
            is_primary = bool(adapter.StateFlags & DISPLAY_DEVICE_PRIMARY_DEVICE)
            mapping_adapter[adapter.DeviceName] = [adapter.DeviceString, is_primary, []]
            display = DISPLAY_DEVICE()
            display.cb = ctypes.sizeof(adapter)
            j = 0
            # Enumerate Monitors
            while ctypes.windll.user32.EnumDisplayDevicesW(
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


def get_monitor_name_by_handle(hmonitor):
    info = MONITORINFOEXW()
    info.cbSize = ctypes.sizeof(MONITORINFOEXW)
    if ctypes.windll.user32.GetMonitorInfoW(hmonitor, ctypes.byref(info)):
        return info
    return None