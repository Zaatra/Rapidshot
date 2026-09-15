import ctypes
import logging
from dataclasses import dataclass
from typing import List, Optional
import comtypes  # type: ignore[import-untyped]
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *

# Configure logging
logger = logging.getLogger("rapidshot.core.device")

# On a private handle: ctypes.windll shares one function object per DLL across
# the whole process, so declaring argtypes there changes D3D11CreateDevice for
# every other library that calls it. See util/io.py.
_D3D11CreateDevice = ctypes.WinDLL("d3d11").D3D11CreateDevice
_D3D11CreateDevice.restype = ctypes.c_long
_D3D11CreateDevice.argtypes = [
    ctypes.c_void_p,                     # pAdapter
    ctypes.c_uint,                       # DriverType
    ctypes.c_void_p,                     # Software
    ctypes.c_uint,                       # Flags
    ctypes.POINTER(ctypes.c_uint),       # pFeatureLevels
    ctypes.c_uint,                       # FeatureLevels
    ctypes.c_uint,                       # SDKVersion
    ctypes.POINTER(ctypes.c_void_p),     # ppDevice
    ctypes.POINTER(ctypes.c_uint),       # pFeatureLevel
    ctypes.POINTER(ctypes.c_void_p)      # ppImmediateContext
]

@dataclass
class Device:
    adapter: ctypes.POINTER(IDXGIAdapter1)
    device: ctypes.POINTER(ID3D11Device) = None
    context: ctypes.POINTER(ID3D11DeviceContext) = None
    im_context: ctypes.POINTER(ID3D11DeviceContext) = None
    desc: DXGI_ADAPTER_DESC1 = None
    feature_level: int = 0

    def __post_init__(self) -> None:
        """
        Initialize Direct3D device with robust feature level negotiation.
        """
        self.desc = DXGI_ADAPTER_DESC1()
        self.adapter.GetDesc1(ctypes.byref(self.desc))

        logger.info(f"Initializing Device for adapter: {self.desc.Description}")
        
        self._create_device()

    # Every attempt is made on this adapter. Creation used to fall back to
    # pAdapter=None -- the *default* adapter -- and then to WARP, REFERENCE and
    # SOFTWARE, while this object went on reporting this adapter's description.
    # That turned "this adapter will not open" into a device living somewhere
    # else under this adapter's name, and on a multi-adapter machine the factory
    # would offer it for duplication as if it were this adapter. Failing
    # instead lets RapidshotFactory record the adapter in device_failures and
    # name it in its error, which is the report that can actually be acted on.
    #
    # What does vary is only what the same adapter may legitimately refuse:
    # the debug layer, which exists only where the SDK layers are installed,
    # and feature level 11.1, which a runtime predating it rejects with
    # E_INVALIDARG rather than skipping.
    _FEATURE_LEVELS = (
        D3D_FEATURE_LEVEL_11_1,
        D3D_FEATURE_LEVEL_11_0,
        D3D_FEATURE_LEVEL_10_1,
        D3D_FEATURE_LEVEL_10_0,
        D3D_FEATURE_LEVEL_9_3,
        D3D_FEATURE_LEVEL_9_2,
        D3D_FEATURE_LEVEL_9_1,
    )

    def _creation_attempts(self):
        """(flags, feature levels) to try, most capable first, no duplicates."""
        flag_sets = [D3D11_CREATE_DEVICE_BGRA_SUPPORT]
        if logger.getEffectiveLevel() <= logging.DEBUG:
            flag_sets.insert(0, D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG)
        # No flags last: BGRA_SUPPORT is for Direct2D interop, and a driver that
        # refuses it can still copy the duplicated BGRA surface.
        flag_sets.append(0)
        level_sets = [self._FEATURE_LEVELS, self._FEATURE_LEVELS[1:]]
        return [(flags, levels) for levels in level_sets for flags in flag_sets]

    def _create_device(self) -> None:
        """Create the D3D11 device on this adapter, or raise naming why not."""
        last_error = None
        for flags, levels in self._creation_attempts():
            levels_array = (ctypes.c_uint * len(levels))(*levels)
            device_ptr = ctypes.c_void_p()
            feature_level = ctypes.c_uint(0)
            context_ptr = ctypes.c_void_p()
            logger.debug(f"D3D11CreateDevice flags={flags:#x} levels={len(levels)}")
            try:
                result = _D3D11CreateDevice(
                    self.adapter,
                    D3D_DRIVER_TYPE_UNKNOWN,   # required when an adapter is given
                    None,
                    flags,
                    levels_array,
                    len(levels),
                    D3D11_SDK_VERSION,
                    ctypes.byref(device_ptr),
                    ctypes.byref(feature_level),
                    ctypes.byref(context_ptr),
                )
                if result != 0:
                    raise comtypes.COMError(
                        result, None,
                        f"D3D11CreateDevice failed with code {result & 0xFFFFFFFF:#010x}")
                if not device_ptr.value or not context_ptr.value:
                    raise RuntimeError("D3D11CreateDevice succeeded but returned no device")

                self.device = ctypes.cast(device_ptr, ctypes.POINTER(ID3D11Device))
                self.context = ctypes.cast(context_ptr, ctypes.POINTER(ID3D11DeviceContext))
                im_context_ptr = ctypes.POINTER(ID3D11DeviceContext)()
                self.device.GetImmediateContext(ctypes.byref(im_context_ptr))
                if not bool(im_context_ptr):
                    raise RuntimeError("Failed to get immediate context")
                self.im_context = im_context_ptr
                self.feature_level = feature_level.value
                logger.info(
                    "Created device with feature level "
                    f"{self.feature_level_to_str(self.feature_level)}")
                return
            except Exception as e:
                last_error = e
                logger.debug(f"Device creation attempt failed: {e}")

        error_msg = (
            f"Failed to create a D3D11 device on {self.desc.Description}. "
            f"Last error: {last_error}")
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    def feature_level_to_str(self, feature_level):
        """Convert feature level to string representation"""
        major = (feature_level >> 12) & 0xF
        minor = (feature_level >> 8) & 0xF
        return f"{major}.{minor}"

    def enum_outputs(self) -> List[ctypes.POINTER(IDXGIOutput1)]:
        """
        Enumerate adapter outputs.
        
        Returns:
            List of adapter outputs
        """
        i = 0
        p_outputs = []
        
        # Check if adapter is valid
        if not bool(self.adapter):
            logger.error("Cannot enumerate outputs: adapter is null")
            return p_outputs
        
        while True:
            try:
                p_output = ctypes.POINTER(IDXGIOutput1)()
                self.adapter.EnumOutputs(i, ctypes.byref(p_output))
                p_outputs.append(p_output)
                i += 1
            except comtypes.COMError as ce:
                if ctypes.c_int32(DXGI_ERROR_NOT_FOUND).value == ce.args[0]:
                    break
                else:
                    logger.error(f"Error enumerating outputs: {ce}")
                    raise ce
                    
        logger.info(f"Found {len(p_outputs)} outputs")
        return p_outputs

    @property
    def description(self) -> str:
        # Safely access description
        if self.desc and hasattr(self.desc, 'Description'):
            return self.desc.Description
        return "Unknown"

    @property
    def vram_size(self) -> int:
        # Safely access vram
        if self.desc and hasattr(self.desc, 'DedicatedVideoMemory'):
            return self.desc.DedicatedVideoMemory
        return 0

    @property
    def vendor_id(self) -> int:
        # Safely access vendor id
        if self.desc and hasattr(self.desc, 'VendorId'):
            return self.desc.VendorId
        return 0

    def __repr__(self) -> str:
        return "<{} Name:{} Dedicated VRAM:{}Mb VendorId:{}>".format(
            self.__class__.__name__,
            self.description,
            self.vram_size // 1048576 if self.vram_size else 0,
            self.vendor_id,
        )
        
    def release(self):
        """
        Release DirectX resources.

        Dropping the reference is the release. Each of these is a comtypes COM
        pointer, and comtypes calls ``Release`` itself when the Python object
        goes away -- so calling it here as well decremented the refcount twice
        for one reference. Measured on this machine: an explicit ``Release()``
        followed by dropping the pointer took the count down by two, dropping
        it alone by one.

        Over-releasing a COM object frees it while other holders still have
        valid pointers; what happens next depends on who touches it first,
        which is why this survived as long as it did. The duplicator had the
        same bug on the intermediate ``IDXGIResource`` in ``update_frame()``,
        where it corrupted the desktop surface's refcount outright.
        """
        self.im_context = None
        self.context = None
        self.device = None
        self.adapter = None