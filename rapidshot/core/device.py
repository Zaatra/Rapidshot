import ctypes
import logging
from dataclasses import dataclass
from typing import List, Optional
import comtypes
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *

# Configure logging
logger = logging.getLogger(__name__)

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
        
        # Convert feature levels to C array
        feature_levels_array = (ctypes.c_uint * len(feature_levels))(*feature_levels)
        
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
                    # Initialize output pointers
                    self.device = ctypes.POINTER(ID3D11Device)()
                    feature_level = ctypes.c_uint(0)
                    self.context = ctypes.POINTER(ID3D11DeviceContext)()
                    
                    # Log the current attempt
                    logger.debug(f"Trying driver type {driver_type} with flags {flags}")
                    
                    # Create device with current parameters
                    if driver_type == D3D_DRIVER_TYPE_UNKNOWN and self.adapter:
                        # Use specific adapter
                        ctypes.windll.d3d11.D3D11CreateDevice(
                            self.adapter,
                            driver_type,
                            None,
                            flags,
                            feature_levels_array,
                            len(feature_levels),
                            D3D11_SDK_VERSION,
                            ctypes.byref(self.device),
                            ctypes.byref(feature_level),
                            ctypes.byref(self.context)
                        )
                    else:
                        # Use driver type
                        ctypes.windll.d3d11.D3D11CreateDevice(
                            None,
                            driver_type,
                            None,
                            flags,
                            feature_levels_array,
                            len(feature_levels),
                            D3D11_SDK_VERSION,
                            ctypes.byref(self.device),
                            ctypes.byref(feature_level),
                            ctypes.byref(self.context)
                        )
                    
                    # Success - get immediate context
                    self.device.GetImmediateContext(ctypes.byref(self.im_context))
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
        return self.desc.Description

    @property
    def vram_size(self) -> int:
        return self.desc.DedicatedVideoMemory

    @property
    def vendor_id(self) -> int:
        return self.desc.VendorId

    def __repr__(self) -> str:
        return "<{} Name:{} Dedicated VRAM:{}Mb VendorId:{}>".format(
            self.__class__.__name__,
            self.desc.Description,
            self.desc.DedicatedVideoMemory // 1048576,
            self.desc.VendorId,
        )