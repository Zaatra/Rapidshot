import weakref
import time
from rapidshot.core import ScreenCapture
from rapidshot.core import Output, Device
from rapidshot.util.io import (
    enum_dxgi_adapters,
    get_output_metadata,
)

# Define explicitly what's exposed from this module
__all__ = [
    "create", "device_info", "output_info", 
    "clean_up", "reset", "ScreenCapture",
    "RapidshotError"
]

class RapidshotError(Exception):
    """Base exception for Rapidshot errors."""
    pass

class Singleton(type):
    """
    Singleton metaclass to ensure only one instance of RapidshotFactory exists.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        else:
            print(f"Only 1 instance of {cls.__name__} is allowed.")
        return cls._instances[cls]

class RapidshotFactory(metaclass=Singleton):
    """
    Factory class for creating ScreenCapture instances.
    Maintains a registry of created screencapture instances to avoid duplicates.
    """
    _screencapture_instances = weakref.WeakValueDictionary()

    def __init__(self) -> None:
        """
        Initialize the factory by enumerating all available devices and outputs.
        """
        p_adapters = enum_dxgi_adapters()
        self.devices, self.outputs = [], []
        
        for p_adapter in p_adapters:
            device = Device(p_adapter)
            p_outputs = device.enum_outputs()
            if len(p_outputs) != 0:
                self.devices.append(device)
                self.outputs.append([Output(p_output) for p_output in p_outputs])
                
        self.output_metadata = get_output_metadata()

    def create(
        self,
        device_idx: int = 0,
        output_idx: int = None,
        region: tuple = None,
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len: int = 64,
    ):
        """
        Create a ScreenCapture instance.
        
        Args:
            device_idx: Device index
            output_idx: Output index (None for primary)
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for capture
            
        Returns:
            ScreenCapture instance
        """
        # Validate device index
        if device_idx >= len(self.devices):
            raise RapidshotError(f"Invalid device index: {device_idx}, max index is {len(self.devices)-1}")
            
        device = self.devices[device_idx]
        
        # Auto-select primary output if not specified
        if output_idx is None:
            output_idx_list = []
            for idx, output in enumerate(self.outputs[device_idx]):
                metadata = self.output_metadata.get(output.devicename)
                if metadata and metadata[1]:  # Is primary
                    output_idx_list.append(idx)
            
            if not output_idx_list:
                # No primary monitor found, use the first one
                output_idx = 0
                print("No primary monitor found, using first available output.")
            else:
                output_idx = output_idx_list[0]
        elif output_idx >= len(self.outputs[device_idx]):
            raise RapidshotError(f"Invalid output index: {output_idx}, max index is {len(self.outputs[device_idx])-1}")
        
        # Check if instance already exists
        instance_key = (device_idx, output_idx)
        if instance_key in self._screencapture_instances:
            print(
                "".join(
                    (
                        f"You already created a ScreenCapture Instance for Device {device_idx}--Output {output_idx}!\n",
                        "Returning the existed instance...\n",
                        "To change capture parameters you can manually delete the old object using `del obj`.",
                    )
                )
            )
            return self._screencapture_instances[instance_key]

        # Create new instance
        output = self.outputs[device_idx][output_idx]
        output.update_desc()
        screencapture = ScreenCapture(
            output=output,
            device=device,
            region=region,
            output_color=output_color,
            nvidia_gpu=nvidia_gpu,
            max_buffer_len=max_buffer_len,
        )
        self._screencapture_instances[instance_key] = screencapture
        
        # Small delay to ensure initialization is complete
        time.sleep(0.1)
        return screencapture

    def device_info(self) -> str:
        """
        Get information about available devices.
        
        Returns:
            String with device information
        """
        ret = ""
        for idx, device in enumerate(self.devices):
            ret += f"Device[{idx}]:{device}\n"
        return ret

    def output_info(self) -> str:
        """
        Get information about available outputs.
        
        Returns:
            String with output information
        """
        ret = ""
        for didx, outputs in enumerate(self.outputs):
            for idx, output in enumerate(outputs):
                ret += f"Device[{didx}] Output[{idx}]: "
                ret += f"Resolution:{output.resolution} Rotation:{output.rotation_angle} "
                ret += f"Primary:{self.output_metadata.get(output.devicename)[1]}\n"
        return ret

    def clean_up(self):
        """
        Release all created screencapture instances.
        """
        for _, screencapture in self._screencapture_instances.items():
            screencapture.release()

    def reset(self):
        """
        Reset the factory, releasing all resources.
        """
        self.clean_up()
        self._screencapture_instances.clear()
        Singleton._instances.clear()

# Global factory instance
__factory = RapidshotFactory()

def create(
    device_idx: int = 0,
    output_idx: int = None,
    region: tuple = None,
    output_color: str = "RGB",
    nvidia_gpu: bool = False,
    max_buffer_len: int = 64,
):
    """
    Create a ScreenCapture instance.
    
    Args:
        device_idx: Device index
        output_idx: Output index (None for primary)
        region: Region to capture (left, top, right, bottom)
        output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        nvidia_gpu: Whether to use NVIDIA GPU acceleration
        max_buffer_len: Maximum buffer length for capture
        
    Returns:
        ScreenCapture instance
    """
    return __factory.create(
        device_idx=device_idx,
        output_idx=output_idx,
        region=region,
        output_color=output_color,
        nvidia_gpu=nvidia_gpu,
        max_buffer_len=max_buffer_len,
    )

def device_info():
    """
    Get information about available devices.
    
    Returns:
        String with device information
    """
    return __factory.device_info()

def output_info():
    """
    Get information about available outputs.
    
    Returns:
        String with output information
    """
    return __factory.output_info()

def clean_up():
    """
    Release all created screencapture instances.
    """
    __factory.clean_up()

def reset():
    """
    Reset the library, releasing all resources.
    """
    __factory.reset()

# Version information
__version__ = "1.0.0"
__author__ = "Rapidshot Contributors"
__description__ = "High-performance screencapture library for Windows using Desktop Duplication API"

# Expose key classes
from rapidshot.core import ScreenCapture