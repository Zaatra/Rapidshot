import ctypes
import logging
import sys
from dataclasses import dataclass
from typing import List, Optional
import comtypes
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *

# Configure logging
logger = logging.getLogger("rapidshot.core.device")

# Define D3D11CreateDevice function with correct argument types
_D3D11CreateDevice = ctypes.windll.d3d11.D3D11CreateDevice
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
        
        # Try different feature levels with flexible error handling
        self._create_device_with_multiple_fallbacks()

    def _create_device_with_multiple_fallbacks(self) -> None:
        """
        Create D3D11 device with enhanced fallback support to maximize compatibility.
        """
        # Define feature levels to try in order
        feature_levels = [
            # Try higher feature levels first
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0,
            D3D_FEATURE_LEVEL_9_3,
            D3D_FEATURE_LEVEL_9_2,
            D3D_FEATURE_LEVEL_9_1,
        ]
        
        # Filter to only include actually defined feature levels
        available_feature_levels = []
        for level in feature_levels:
            if hasattr(sys.modules[__name__], f'D3D_FEATURE_LEVEL_{level >> 12}_{(level >> 8) & 0xF}'):
                available_feature_levels.append(level)
        
        if not available_feature_levels:
            # If no feature levels matched constants, use values directly (failsafe)
            available_feature_levels = feature_levels
            
        logger.debug(f"Using feature levels: {[hex(level) for level in available_feature_levels]}")
        
        # Convert feature levels to C array
        feature_levels_array = (ctypes.c_uint * len(available_feature_levels))(*available_feature_levels)
        
        # Try different driver types for more compatibility
        driver_types = [
            D3D_DRIVER_TYPE_UNKNOWN,
            D3D_DRIVER_TYPE_HARDWARE,
            D3D_DRIVER_TYPE_WARP,      # Software fallback
            D3D_DRIVER_TYPE_REFERENCE,
            D3D_DRIVER_TYPE_SOFTWARE
        ]
        
        # Device creation flags - start with basic flags
        base_flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT
        
        # Add debug flag in debug mode
        if logger.getEffectiveLevel() <= logging.DEBUG:
            # Try with debug flag, but it's optional
            debug_flag = D3D11_CREATE_DEVICE_DEBUG
        else:
            debug_flag = 0
            
        # Create combinations of flags to try
        flag_combinations = [
            base_flags,                # Standard flags
            base_flags | debug_flag,   # With debug flag (if applicable)
            0                          # No flags
        ]
        
        # Try combinations of driver types, feature levels, and flags
        created = False
        last_error = None
        
        for driver_type in driver_types:
            if created:
                break
                
            for flags in flag_combinations:
                if created:
                    break
                    
                try:
                    # Initialize output pointers - FIXED: Using void pointers for D3D11CreateDevice
                    device_ptr = ctypes.c_void_p()
                    feature_level = ctypes.c_uint(0)
                    context_ptr = ctypes.c_void_p()
                    
                    # Log the current attempt
                    logger.debug(f"Trying driver type {driver_type} with flags {flags}")
                    
                    # Create device with current parameters
                    adapter_ptr = self.adapter
                    if driver_type == D3D_DRIVER_TYPE_UNKNOWN and adapter_ptr:
                        # Make sure adapter is a valid pointer before using it
                        if not bool(adapter_ptr):
                            logger.warning("Adapter pointer is null, skipping this attempt")
                            continue
                            
                        # Use specific adapter
                        result = _D3D11CreateDevice(
                            adapter_ptr,
                            driver_type,
                            None,
                            flags,
                            feature_levels_array,
                            len(available_feature_levels),
                            D3D11_SDK_VERSION,
                            ctypes.byref(device_ptr),
                            ctypes.byref(feature_level),
                            ctypes.byref(context_ptr)
                        )
                        
                        # Check result code
                        if result != 0:  # non-zero = error
                            logger.debug(f"D3D11CreateDevice returned error code: {result:#x}")
                            # Convert to COMError to be caught below
                            raise comtypes.COMError(result, None, f"D3D11CreateDevice failed with code {result:#x}")
                    else:
                        # Use driver type
                        result = _D3D11CreateDevice(
                            None,
                            driver_type,
                            None,
                            flags,
                            feature_levels_array,
                            len(available_feature_levels),
                            D3D11_SDK_VERSION,
                            ctypes.byref(device_ptr),
                            ctypes.byref(feature_level),
                            ctypes.byref(context_ptr)
                        )
                        
                        # Check result code
                        if result != 0:  # non-zero = error
                            logger.debug(f"D3D11CreateDevice returned error code: {result:#x}")
                            # Convert to COMError to be caught below
                            raise comtypes.COMError(result, None, f"D3D11CreateDevice failed with code {result:#x}")
                    
                    # CRITICAL: Verify pointers are valid
                    if not device_ptr.value or not context_ptr.value:
                        logger.warning("Device or context pointer is null after D3D11CreateDevice")
                        continue
                    
                    # Convert void pointers to the correct interface types
                    self.device = ctypes.cast(device_ptr, ctypes.POINTER(ID3D11Device))
                    self.context = ctypes.cast(context_ptr, ctypes.POINTER(ID3D11DeviceContext))
                    
                    # Get immediate context
                    im_context_ptr = ctypes.POINTER(ID3D11DeviceContext)()
                    self.device.GetImmediateContext(ctypes.byref(im_context_ptr))
                    if not bool(im_context_ptr):
                        logger.warning("Failed to get immediate context")
                        raise RuntimeError("Failed to get immediate context")
                    self.im_context = im_context_ptr
                    
                    self.feature_level = feature_level.value
                    
                    # Log success
                    logger.info(f"Successfully created device with feature level {self.feature_level_to_str(self.feature_level)}")
                    created = True
                    break
                
                except comtypes.COMError as ce:
                    last_error = ce
                    logger.debug(f"Failed to create device with driver {driver_type}, flags {flags}: {ce}")
                    continue
                    
                except Exception as e:
                    last_error = e
                    logger.debug(f"Exception creating device: {e}")
                    continue
        
        # If all attempts failed
        if not created:
            error_msg = f"Failed to create D3D11 device after trying all options. Last error: {last_error}"
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
        
    @classmethod
    def create(cls, adapter_idx=0):
        """
        Create a new Device instance for the given adapter index.
        
        Args:
            adapter_idx: Index of the adapter to use
            
        Returns:
            New Device instance
        """
        from rapidshot.util.io import enum_dxgi_adapters
        
        adapters = enum_dxgi_adapters()
        if not adapters:
            logger.error("No DXGI adapters found")
            raise RuntimeError("No DXGI adapters found")
            
        if adapter_idx >= len(adapters):
            logger.error(f"Adapter index {adapter_idx} out of range, found {len(adapters)} adapters")
            raise IndexError(f"Adapter index {adapter_idx} out of range, found {len(adapters)} adapters")
            
        return cls(adapters[adapter_idx])
        
    def release(self):
        """
        Release DirectX resources.
        """
        if self.im_context:
            self.im_context.Release()
            self.im_context = None
            
        if self.context:
            self.context.Release()
            self.context = None
            
        if self.device:
            self.device.Release()
            self.device = None
            
        if self.adapter:
            self.adapter.Release()
            self.adapter = None