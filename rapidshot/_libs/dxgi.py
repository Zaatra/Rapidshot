import ctypes
import ctypes.wintypes as wintypes
import comtypes
import logging
from .d3d11 import ID3D11Device

# Set up logger
logger = logging.getLogger("rapidshot._libs.dxgi")

# DXGI error codes
DXGI_ERROR_ACCESS_LOST = 0x887A0026
DXGI_ERROR_NOT_FOUND = 0x887A0002
DXGI_ERROR_WAIT_TIMEOUT = 0x887A0027
ABANDONED_MUTEX_EXCEPTION = -0x7785ffda  # -2005270490

# Pointer shape type constants
DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME = 0x00000001
DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR = 0x00000002
DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR = 0x00000004

# DXGI 1.2-1.6 Definitions
DXGI_GPU_PREFERENCE_UNSPECIFIED = 0
DXGI_GPU_PREFERENCE_MINIMUM_POWER = 1
DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE = 2


class LUID(ctypes.Structure):
    """
    Locally unique identifier structure.
    """
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class DXGI_ADAPTER_DESC1(ctypes.Structure):
    """
    DXGI adapter description.
    """
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", wintypes.ULARGE_INTEGER),
        ("DedicatedSystemMemory", wintypes.ULARGE_INTEGER),
        ("SharedSystemMemory", wintypes.ULARGE_INTEGER),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
    ]


class DXGI_ADAPTER_DESC2(ctypes.Structure):
    """
    DXGI adapter description (DXGI 1.2).
    """
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", wintypes.ULARGE_INTEGER),
        ("DedicatedSystemMemory", wintypes.ULARGE_INTEGER),
        ("SharedSystemMemory", wintypes.ULARGE_INTEGER),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
        ("GraphicsPreemptionGranularity", wintypes.UINT),
        ("ComputePreemptionGranularity", wintypes.UINT),
    ]


class DXGI_ADAPTER_DESC3(ctypes.Structure):
    """
    DXGI adapter description (DXGI 1.6).
    """
    _fields_ = [
        ("Description", wintypes.WCHAR * 128),
        ("VendorId", wintypes.UINT),
        ("DeviceId", wintypes.UINT),
        ("SubSysId", wintypes.UINT),
        ("Revision", wintypes.UINT),
        ("DedicatedVideoMemory", wintypes.ULARGE_INTEGER),
        ("DedicatedSystemMemory", wintypes.ULARGE_INTEGER),
        ("SharedSystemMemory", wintypes.ULARGE_INTEGER),
        ("AdapterLuid", LUID),
        ("Flags", wintypes.UINT),
        ("GraphicsPreemptionGranularity", wintypes.UINT),
        ("ComputePreemptionGranularity", wintypes.UINT),
    ]


class DXGI_OUTPUT_DESC(ctypes.Structure):
    """
    DXGI output description.
    """
    _fields_ = [
        ("DeviceName", wintypes.WCHAR * 32),
        ("DesktopCoordinates", wintypes.RECT),
        ("AttachedToDesktop", wintypes.BOOL),
        ("Rotation", wintypes.UINT),
        ("Monitor", wintypes.HMONITOR),
    ]


class DXGI_OUTDUPL_POINTER_POSITION(ctypes.Structure):
    """
    DXGI output duplication pointer position.
    """
    _fields_ = [("Position", wintypes.POINT), ("Visible", wintypes.BOOL)]


class DXGI_OUTDUPL_POINTER_SHAPE_INFO(ctypes.Structure):
    """
    DXGI output duplication pointer shape info.
    """
    _fields_ = [
        ("Type", wintypes.UINT),
        ("Width", wintypes.UINT),
        ("Height", wintypes.UINT),
        ("Pitch", wintypes.UINT),
        ("HotSpot", wintypes.POINT),
    ]


class DXGI_OUTDUPL_FRAME_INFO(ctypes.Structure):
    """
    DXGI output duplication frame info.
    """
    _fields_ = [
        ("LastPresentTime", wintypes.LARGE_INTEGER),
        ("LastMouseUpdateTime", wintypes.LARGE_INTEGER),
        ("AccumulatedFrames", wintypes.UINT),
        ("RectsCoalesced", wintypes.BOOL),
        ("ProtectedContentMaskedOut", wintypes.BOOL),
        ("PointerPosition", DXGI_OUTDUPL_POINTER_POSITION),
        ("TotalMetadataBufferSize", wintypes.UINT),
        ("PointerShapeBufferSize", wintypes.UINT),
    ]


class DXGI_MAPPED_RECT(ctypes.Structure):
    """
    DXGI mapped rectangle.
    """
    _fields_ = [("Pitch", wintypes.INT), ("pBits", ctypes.POINTER(wintypes.FLOAT))]


class IDXGIObject(comtypes.IUnknown):
    """
    DXGI object interface.
    """
    _iid_ = comtypes.GUID("{aec22fb8-76f3-4639-9be0-28eb43a67a2e}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "SetPrivateData"),
        comtypes.STDMETHOD(comtypes.HRESULT, "SetPrivateDataInterface"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetPrivateData"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetParent"),
    ]


class IDXGIDeviceSubObject(IDXGIObject):
    """
    DXGI device sub-object interface.
    """
    _iid_ = comtypes.GUID("{3d3e0379-f9de-4d58-bb6c-18d62992f1a6}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDevice"),
    ]


class IDXGIResource(IDXGIDeviceSubObject):
    """
    DXGI resource interface.
    """
    _iid_ = comtypes.GUID("{035f3ab4-482e-4e50-b41f-8a7f8bd8960b}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "GetSharedHandle"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetUsage"),
        comtypes.STDMETHOD(comtypes.HRESULT, "SetEvictionPriority"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetEvictionPriority"),
    ]


class IDXGISurface(IDXGIDeviceSubObject):
    """
    DXGI surface interface.
    """
    _iid_ = comtypes.GUID("{cafcb56c-6ac3-4889-bf47-9e23bbd260ec}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDesc"),
        comtypes.STDMETHOD(
            comtypes.HRESULT, "Map", [ctypes.POINTER(DXGI_MAPPED_RECT), wintypes.UINT]
        ),
        comtypes.STDMETHOD(comtypes.HRESULT, "Unmap"),
    ]


class IDXGIOutputDuplication(IDXGIObject):
    """
    DXGI output duplication interface.
    """
    _iid_ = comtypes.GUID("{191cfac3-a341-470d-b26e-a864f428319c}")
    _methods_ = [
        comtypes.STDMETHOD(None, "GetDesc"),
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "AcquireNextFrame",
            [
                wintypes.UINT,
                ctypes.POINTER(DXGI_OUTDUPL_FRAME_INFO),
                ctypes.POINTER(ctypes.POINTER(IDXGIResource)),
            ],
        ),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetFrameDirtyRects"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetFrameMoveRects"),
        comtypes.STDMETHOD(
            comtypes.HRESULT, 
            "GetFramePointerShape", 
            [
                wintypes.UINT,
                ctypes.c_void_p,
                ctypes.POINTER(wintypes.UINT),
                ctypes.POINTER(DXGI_OUTDUPL_POINTER_SHAPE_INFO),
            ]
        ),
        comtypes.STDMETHOD(comtypes.HRESULT, "MapDesktopSurface"),
        comtypes.STDMETHOD(comtypes.HRESULT, "UnMapDesktopSurface"),
        comtypes.STDMETHOD(comtypes.HRESULT, "ReleaseFrame"),
    ]


class IDXGIOutput(IDXGIObject):
    """
    DXGI output interface.
    """
    _iid_ = comtypes.GUID("{ae02eedb-c735-4690-8d52-5a8dc20213aa}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT, "GetDesc", [ctypes.POINTER(DXGI_OUTPUT_DESC)]
        ),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDisplayModeList"),
        comtypes.STDMETHOD(comtypes.HRESULT, "FindClosestMatchingMode"),
        comtypes.STDMETHOD(comtypes.HRESULT, "WaitForVBlank"),
        comtypes.STDMETHOD(comtypes.HRESULT, "TakeOwnership"),
        comtypes.STDMETHOD(None, "ReleaseOwnership"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetGammaControlCapabilities"),
        comtypes.STDMETHOD(comtypes.HRESULT, "SetGammaControl"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetGammaControl"),
        comtypes.STDMETHOD(comtypes.HRESULT, "SetDisplaySurface"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDisplaySurfaceData"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetFrameStatistics"),
    ]


class IDXGIOutput1(IDXGIOutput):
    """
    DXGI output interface version 1.
    """
    _iid_ = comtypes.GUID("{00cddea8-939b-4b83-a340-a685226666cc}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDisplayModeList1"),
        comtypes.STDMETHOD(comtypes.HRESULT, "FindClosestMatchingMode1"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDisplaySurfaceData1"),
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "DuplicateOutput",
            [
                ctypes.POINTER(ID3D11Device),
                ctypes.POINTER(ctypes.POINTER(IDXGIOutputDuplication)),
            ],
        ),
    ]


class IDXGIAdapter(IDXGIObject):
    """
    DXGI adapter interface.
    """
    _iid_ = comtypes.GUID("{2411e7e1-12ac-4ccf-bd14-9798e8534dc0}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "EnumOutputs",
            [wintypes.UINT, ctypes.POINTER(ctypes.POINTER(IDXGIOutput))],
        ),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetDesc"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CheckInterfaceSupport"),
    ]


class IDXGIAdapter1(IDXGIAdapter):
    """
    DXGI adapter interface version 1.
    """
    _iid_ = comtypes.GUID("{29038f61-3839-4626-91fd-086879011a05}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT, "GetDesc1", [ctypes.POINTER(DXGI_ADAPTER_DESC1)]
        ),
    ]


class IDXGIAdapter2(IDXGIAdapter1):
    """
    DXGI adapter interface version 2 (DXGI 1.2).
    """
    _iid_ = comtypes.GUID("{0AA1AE0A-FA0E-4B84-8644-E05FF8E5ACB5}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT, "GetDesc2", [ctypes.POINTER(DXGI_ADAPTER_DESC2)]
        ),
    ]


class IDXGIAdapter3(IDXGIAdapter2):
    """
    DXGI adapter interface version 3 (DXGI 1.4).
    """
    _iid_ = comtypes.GUID("{645967A4-1392-4310-A798-8053CE3E93FD}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterHardwareContentProtectionTeardownStatusEvent"),
        comtypes.STDMETHOD(None, "UnregisterHardwareContentProtectionTeardownStatus"),
        comtypes.STDMETHOD(comtypes.HRESULT, "QueryVideoMemoryInfo"),
        comtypes.STDMETHOD(comtypes.HRESULT, "SetVideoMemoryReservation"),
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterVideoMemoryBudgetChangeNotificationEvent"),
        comtypes.STDMETHOD(None, "UnregisterVideoMemoryBudgetChangeNotification"),
    ]


class IDXGIAdapter4(IDXGIAdapter3):
    """
    DXGI adapter interface version 4 (DXGI 1.6).
    """
    _iid_ = comtypes.GUID("{3C8D99D1-4FBF-4181-A82C-AF66BF7BD24E}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT, "GetDesc3", [ctypes.POINTER(DXGI_ADAPTER_DESC3)]
        ),
    ]


class IDXGIFactory(IDXGIObject):
    """
    DXGI factory interface.
    """
    _iid_ = comtypes.GUID("{7b7166ec-21c7-44ae-b21a-c9ae321ae369}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "EnumAdapters"),
        comtypes.STDMETHOD(comtypes.HRESULT, "MakeWindowAssociation"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetWindowAssociation"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CreateSwapChain"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CreateSoftwareAdapter"),
    ]


class IDXGIFactory1(IDXGIFactory):
    """
    DXGI factory interface version 1.
    """
    _iid_ = comtypes.GUID("{770aae78-f26f-4dba-a829-253c83d1b387}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "EnumAdapters1",
            [ctypes.c_uint, ctypes.POINTER(ctypes.POINTER(IDXGIAdapter1))],
        ),
        comtypes.STDMETHOD(wintypes.BOOL, "IsCurrent"),
    ]


class IDXGIFactory2(IDXGIFactory1):
    """
    DXGI factory interface version 2 (DXGI 1.2).
    """
    _iid_ = comtypes.GUID("{50C83A1C-E072-4C48-87B0-3630FA36A6D0}")
    _methods_ = [
        comtypes.STDMETHOD(wintypes.BOOL, "IsWindowedStereoEnabled"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CreateSwapChainForHwnd"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CreateSwapChainForCoreWindow"),
        comtypes.STDMETHOD(comtypes.HRESULT, "GetSharedResourceAdapterLuid"),
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterStereoStatusWindow"),
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterStereoStatusEvent"),
        comtypes.STDMETHOD(None, "UnregisterStereoStatus"),
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterOcclusionStatusWindow"),
        comtypes.STDMETHOD(comtypes.HRESULT, "RegisterOcclusionStatusEvent"),
        comtypes.STDMETHOD(None, "UnregisterOcclusionStatus"),
        comtypes.STDMETHOD(comtypes.HRESULT, "CreateSwapChainForComposition"),
    ]


class IDXGIFactory3(IDXGIFactory2):
    """
    DXGI factory interface version 3 (DXGI 1.3).
    """
    _iid_ = comtypes.GUID("{25483823-CD46-4C7D-86CA-47AA95B837BD}")
    _methods_ = [
        comtypes.STDMETHOD(wintypes.UINT, "GetCreationFlags"),
    ]


class IDXGIFactory4(IDXGIFactory3):
    """
    DXGI factory interface version 4 (DXGI 1.4).
    """
    _iid_ = comtypes.GUID("{1BC6EA02-EF36-464F-BF0C-21CA39E5168A}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "EnumAdapterByLuid",
            [LUID, ctypes.POINTER(comtypes.GUID), ctypes.POINTER(ctypes.c_void_p)],
        ),
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "EnumWarpAdapter",
            [ctypes.POINTER(comtypes.GUID), ctypes.POINTER(ctypes.c_void_p)],
        ),
    ]


class IDXGIFactory5(IDXGIFactory4):
    """
    DXGI factory interface version 5 (DXGI 1.5).
    """
    _iid_ = comtypes.GUID("{7632E1F5-EE65-4DCA-87FD-84CD75F8838D}")
    _methods_ = [
        comtypes.STDMETHOD(comtypes.HRESULT, "CheckFeatureSupport"),
    ]


class IDXGIFactory6(IDXGIFactory5):
    """
    DXGI factory interface version 6 (DXGI 1.6).
    """
    _iid_ = comtypes.GUID("{C1B6694F-FF09-44A9-B03C-77900A0A1D17}")
    _methods_ = [
        comtypes.STDMETHOD(
            comtypes.HRESULT,
            "EnumAdapterByGpuPreference",
            [
                wintypes.UINT,
                wintypes.UINT,
                ctypes.POINTER(comtypes.GUID),
                ctypes.POINTER(ctypes.c_void_p),
            ],
        ),
    ]


# Create DXGI Factory function
try:
    _CreateDXGIFactory1 = ctypes.windll.dxgi.CreateDXGIFactory1
    _CreateDXGIFactory1.restype = comtypes.HRESULT
    _CreateDXGIFactory1.argtypes = [
        ctypes.POINTER(comtypes.GUID),
        ctypes.POINTER(ctypes.c_void_p)
    ]

    def CreateDXGIFactory1(riid, ppFactory):
        """
        Create a DXGI factory object.
        
        Args:
            riid: Reference to the factory interface ID
            ppFactory: Pointer to receive the created factory
            
        Returns:
            HRESULT value
        """
        return _CreateDXGIFactory1(riid, ppFactory)
except (AttributeError, WindowsError) as e:
    # Provide a fallback implementation or raise an informative error
    logger.error(f"Failed to load CreateDXGIFactory1: {e}")
    
    def CreateDXGIFactory1(riid, ppFactory):
        """
        Fallback implementation that raises an error.
        """
        raise RuntimeError(
            "CreateDXGIFactory1 is not available. This might indicate DirectX is not properly installed."
        )


# Function to create DXGI Factory with the latest available version
try:
    _CreateDXGIFactory6 = ctypes.windll.dxgi.CreateDXGIFactory6
    _CreateDXGIFactory6.restype = comtypes.HRESULT
    _CreateDXGIFactory6.argtypes = [
        ctypes.POINTER(comtypes.GUID),
        ctypes.POINTER(ctypes.c_void_p)
    ]
    
    def CreateDXGIFactory6(riid, ppFactory):
        """
        Create a DXGI factory object with DXGI 1.6.
        
        Args:
            riid: Reference to the factory interface ID
            ppFactory: Pointer to receive the created factory
            
        Returns:
            HRESULT value
        """
        return _CreateDXGIFactory6(riid, ppFactory)
except (AttributeError, WindowsError) as e:
    logger.info(f"CreateDXGIFactory6 not available, falling back to CreateDXGIFactory1: {e}")
    CreateDXGIFactory6 = None


# Create DXGI Factory function with version detection
def CreateLatestDXGIFactory(riid, ppFactory):
    """
    Create a DXGI factory with the latest available version.
    
    Args:
        riid: Reference to the factory interface ID
        ppFactory: Pointer to receive the created factory
        
    Returns:
        HRESULT value
    """
    try:
        if CreateDXGIFactory6 is not None:
            logger.info("Using DXGI 1.6 (CreateDXGIFactory6)")
            return CreateDXGIFactory6(riid, ppFactory)
        else:
            logger.info("Using DXGI 1.1 (CreateDXGIFactory1)")
            return CreateDXGIFactory1(riid, ppFactory)
    except Exception as e:
        logger.error(f"Failed to create DXGI Factory: {e}")
        raise