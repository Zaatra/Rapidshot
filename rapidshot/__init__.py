import weakref
import time
import logging
import platform
import sys
from typing import Optional, Tuple, Dict, Any
from rapidshot.core import ScreenCapture
from rapidshot.core import Output, Device
from rapidshot.util.io import (
    enum_dxgi_adapters,
    get_output_metadata,
)
from rapidshot.util.logging import setup_logging, get_logger

# Initialize logging
logger = get_logger("init")

# Define explicitly what's exposed from this module
__all__ = [
    "create", "device_info", "output_info", 
    "clean_up", "reset", "ScreenCapture",
    "RapidshotError", "get_version_info"
]

class RapidshotError(Exception):
    """Base exception for Rapidshot errors."""
    pass


class DeviceError(RapidshotError):
    """Exception raised for errors related to device operations."""
    pass


class OutputError(RapidshotError):
    """Exception raised for errors related to output operations."""
    pass


class ConfigurationError(RapidshotError):
    """Exception raised for errors related to configuration."""
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
            logger.debug(f"Using existing instance of {cls.__name__}")
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
        logger.info("Initializing RapidshotFactory")
        try:
            p_adapters = enum_dxgi_adapters()
            if not p_adapters:
                error_msg = "No DXGI adapters found. Make sure your system has a compatible graphics card."
                logger.error(error_msg)
                raise DeviceError(error_msg)
                
            self.devices, self.outputs = [], []
            
            for p_adapter in p_adapters:
                try:
                    device = Device(p_adapter)
                    p_outputs = device.enum_outputs()
                    if len(p_outputs) != 0:
                        self.devices.append(device)
                        self.outputs.append([Output(p_output) for p_output in p_outputs])
                except Exception as e:
                    logger.warning(f"Failed to initialize device: {e}")
                    
            if not self.devices:
                error_msg = "No usable graphics devices found. Check your display configuration."
                logger.error(error_msg)
                raise DeviceError(error_msg)
                
            self.output_metadata = get_output_metadata()
            logger.info(f"RapidshotFactory initialized with {len(self.devices)} devices")
        except Exception as e:
            error_msg = f"Failed to initialize RapidshotFactory: {e}"
            logger.error(error_msg)
            raise RapidshotError(error_msg) from e

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
        logger.debug(f"Creating ScreenCapture with device_idx={device_idx}, output_idx={output_idx}, nvidia_gpu={nvidia_gpu}")
        
        # Validate device index
        if device_idx >= len(self.devices):
            error_msg = f"Invalid device index: {device_idx}, max index is {len(self.devices)-1}"
            logger.error(error_msg)
            raise DeviceError(error_msg)
            
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
                logger.info("No primary monitor found, using first available output.")
            else:
                output_idx = output_idx_list[0]
                logger.info(f"Using primary monitor (output index {output_idx})")
        elif output_idx >= len(self.outputs[device_idx]):
            error_msg = f"Invalid output index: {output_idx}, max index is {len(self.outputs[device_idx])-1}"
            logger.error(error_msg)
            raise OutputError(error_msg)
        
        # Validate color format
        valid_color_formats = ["RGB", "RGBA", "BGR", "BGRA", "GRAY"]
        if output_color not in valid_color_formats:
            error_msg = f"Invalid color format: {output_color}. Must be one of {valid_color_formats}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        # Check if instance already exists
        instance_key = (device_idx, output_idx)
        if instance_key in self._screencapture_instances:
            logger.info(f"Found existing ScreenCapture instance for Device {device_idx}--Output {output_idx}")
            return self._screencapture_instances[instance_key]

        try:
            # Create new instance
            output = self.outputs[device_idx][output_idx]
            output.update_desc()
            
            if nvidia_gpu:
                # Check if NVIDIA GPU is requested but not available
                try:
                    import cupy
                    logger.info("Using NVIDIA GPU acceleration with CuPy")
                except ImportError:
                    nvidia_gpu = False
                    logger.warning("NVIDIA GPU acceleration requested but CuPy not available. Falling back to CPU mode.")
            
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
            logger.info(f"Created new ScreenCapture instance for Device {device_idx}--Output {output_idx}")
            return screencapture
        except Exception as e:
            error_msg = f"Failed to create ScreenCapture instance: {e}"
            logger.error(error_msg)
            raise RapidshotError(error_msg) from e

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
        logger.info("Cleaning up all ScreenCapture instances")
        for _, screencapture in self._screencapture_instances.items():
            try:
                screencapture.release()
            except Exception as e:
                logger.warning(f"Error releasing ScreenCapture instance: {e}")

    def reset(self):
        """
        Reset the factory, releasing all resources.
        """
        logger.info("Resetting RapidshotFactory")
        self.clean_up()
        self._screencapture_instances.clear()
        Singleton._instances.clear()

# Global factory instance
__factory = None

def get_factory():
    """
    Get the global factory instance, initializing it if necessary.
    
    Returns:
        RapidshotFactory instance
    """
    global __factory
    if __factory is None:
        try:
            __factory = RapidshotFactory()
        except Exception as e:
            logger.error(f"Failed to initialize RapidshotFactory: {e}")
            raise
    return __factory

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
    factory = get_factory()
    return factory.create(
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
    factory = get_factory()
    return factory.device_info()

def output_info():
    """
    Get information about available outputs.
    
    Returns:
        String with output information
    """
    factory = get_factory()
    return factory.output_info()

def clean_up():
    """
    Release all created screencapture instances.
    """
    global __factory
    if __factory is not None:
        __factory.clean_up()

def reset():
    """
    Reset the library, releasing all resources.
    """
    global __factory
    if __factory is not None:
        __factory.reset()
        __factory = None

def get_version_info() -> Dict[str, Any]:
    """
    Get version information about RapidShot and its dependencies.
    
    Returns:
        Dictionary with version information
    """
    info = {
        "rapidshot": {
            "version": __version__,
            "author": __author__,
            "description": __description__,
        },
        "system": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "dependencies": {}
    }
    
    # Check numpy
    try:
        import numpy
        info["dependencies"]["numpy"] = numpy.__version__
    except ImportError:
        info["dependencies"]["numpy"] = "not installed"
    
    # Check cupy
    try:
        import cupy
        info["dependencies"]["cupy"] = cupy.__version__
    except ImportError:
        info["dependencies"]["cupy"] = "not installed"
    
    # Check pillow
    try:
        from PIL import __version__ as pil_version
        info["dependencies"]["pillow"] = pil_version
    except ImportError:
        info["dependencies"]["pillow"] = "not installed"
    
    # Check opencv
    try:
        import cv2
        info["dependencies"]["opencv"] = cv2.__version__
    except ImportError:
        info["dependencies"]["opencv"] = "not installed"
    
    # Check comtypes
    try:
        import comtypes
        info["dependencies"]["comtypes"] = comtypes.__version__
    except (ImportError, AttributeError):
        info["dependencies"]["comtypes"] = "version unknown"
    
    return info

# Version information
__version__ = "1.0.0"
__author__ = "Rapidshot Contributors"
__description__ = "High-performance screencapture library for Windows using Desktop Duplication API"

# Expose key classes
from rapidshot.core import ScreenCapture

# Initialize factory on first import - lazy initialization
try:
    get_factory()
except Exception as e:
    logger.error(f"Failed to initialize RapidshotFactory on import: {e}")
    # Let the user handle the initialization error when they try to use the library