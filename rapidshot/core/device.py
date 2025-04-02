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
    selected_feature_level: ctypes.c_uint = 0

    def __post_init__(self) -> None:
        """
        Initialize Direct3D device with robust feature level negotiation.
        """
        self.desc = DXGI_ADAPTER_DESC1()
        self.adapter.GetDesc1(ctypes.byref(self.desc))

        logger.info(f"Initializing Device for adapter: {self.desc.Description}")
        
        # Get function pointer to D3D11CreateDevice
        D3D11CreateDevice = ctypes.windll.d3d11.D3D11CreateDevice
        
        # Try creating device with progressively more compatibility options
        self._create_device_with_feature_level_fallbacks(D3D11CreateDevice)

    def _create_device_with_feature_level_fallbacks(self, D3D11CreateDevice) -> None:
        """
        Create D3D11 device with multiple fallback options for better compatibility.
        
        Args:
            D3D11CreateDevice: Function pointer to D3D11CreateDevice
        """
        # Try multiple feature levels, starting from the highest
        # Only use feature levels that are defined in the module
        feature_levels = []
        
        # Check which feature levels are defined
        for level_name in ["D3D_FEATURE_LEVEL_11_1", "D3D_FEATURE_LEVEL_11_0", 
                           "D3D_FEATURE_LEVEL_10_1", "D3D_FEATURE_LEVEL_10_0",
                           "D3D_FEATURE_LEVEL_9_3", "D3D_FEATURE_LEVEL_9_2", 
                           "D3D_FEATURE_LEVEL_9_1"]:
            if level_name in globals():
                feature_levels.append(globals()[level_name])
                logger.debug(f"Using feature level: {level_name}")
                
        if not feature_levels:
            error_msg = "No DirectX feature levels available"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Convert feature levels to C array
        feature_levels_array = (ctypes.c_uint * len(feature_levels))(*feature_levels)
        
        # Initialize output pointers
        self.device = ctypes.POINTER(ID3D11Device)()
        self.context = ctypes.POINTER(ID3D11DeviceContext)()
        self.im_context = ctypes.POINTER(ID3D11DeviceContext)()
        self.selected_feature_level = ctypes.c_uint(0)
        
        # Try different device creation configurations
        creation_attempts = [
            # 1. Standard configuration with specified adapter
            {
                'adapter': self.adapter,
                'driver_type': D3D_DRIVER_TYPE_UNKNOWN,
                'flags': 0,
                'msg': "Creating device with adapter"
            },
            # 2. Try without debug flag
            {
                'adapter': self.adapter,
                'driver_type': D3D_DRIVER_TYPE_UNKNOWN,
                'flags': D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                'msg': "Creating device with BGRA support"
            },
            # 3. Try with hardware driver type
            {
                'adapter': None,
                'driver_type': D3D_DRIVER_TYPE_HARDWARE,
                'flags': 0,
                'msg': "Creating device with hardware driver type"
            },
            # 4. Last resort: try WARP software rendering
            {
                'adapter': None,
                'driver_type': D3D_DRIVER_TYPE_WARP,
                'flags': 0,
                'msg': "Creating device with WARP software rendering"
            }
        ]
        
        created = False
        last_error = None
        
        # Try each configuration until one succeeds
        for attempt in creation_attempts:
            try:
                logger.debug(attempt['msg'])
                
                D3D11CreateDevice(
                    attempt['adapter'],
                    attempt['driver_type'],
                    None,
                    attempt['flags'],
                    feature_levels_array,
                    len(feature_levels),
                    D3D11_SDK_VERSION,
                    ctypes.byref(self.device),
                    ctypes.byref(self.selected_feature_level),
                    ctypes.byref(self.context),
                )
                
                # Get immediate context
                self.device.GetImmediateContext(ctypes.byref(self.im_context))
                
                # Log success
                feature_level_str = f"{(self.selected_feature_level.value >> 12) & 0xF}.{(self.selected_feature_level.value >> 8) & 0xF}"
                logger.info(f"Successfully created device with feature level {feature_level_str}")
                
                created = True
                break
            
            except comtypes.COMError as ce:
                last_error = ce
                logger.warning(f"Failed to create device with {attempt['msg']}: {ce}")
        
        if not created:
            error_msg = f"Failed to create DirectX device after multiple attempts: {last_error}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

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