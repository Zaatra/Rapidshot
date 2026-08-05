import weakref
import time
from rapidshot.util.logging import get_logger
import platform
import sys
from typing import Optional, Tuple, Dict, Any
# Screen capture relies on Windows-specific COM technology. Attempt to import it lazily so
# that other modules (e.g., memory pools) remain usable on non-Windows platforms.
try:
    from rapidshot.capture import ScreenCapture  # Updated import: now from rapidshot.capture
    _capture_import_error = None
except ImportError as exc:
    ScreenCapture = None
    _capture_import_error = exc
# DXGI device discovery also relies on COM, so guard those imports as well.
try:
    from rapidshot.core import Output, Device
    _core_import_error = None
except ImportError as exc:
    Output = Device = None
    _core_import_error = exc

try:
    from rapidshot.util.io import (
        enum_dxgi_adapters,
        get_output_metadata,
    )
    _io_import_error = None
except ImportError as exc:
    enum_dxgi_adapters = get_output_metadata = None
    _io_import_error = exc
from rapidshot.util.logging import setup_logging, get_logger
from rapidshot.util.topology import (
    AdapterInfo,
    GpuTopology,
    classify,
    probe_topology,
)

# Initialize logging
logger = get_logger("init")

# Define explicitly what's exposed from this module
__all__ = [
    "create", "device_info", "output_info", "topology_info",
    "clean_up", "reset", "ScreenCapture",
    "RapidshotError", "HeadlessError", "get_version_info",
    "probe_topology", "GpuTopology", "AdapterInfo",
]

class RapidshotError(Exception):
    """Base exception for Rapidshot errors."""
    pass

class DeviceError(RapidshotError):
    """Exception raised for errors related to device operations."""
    pass

class HeadlessError(DeviceError):
    """Raised when no adapter drives a display, so there is nothing to duplicate.

    Carries the probed topology so a caller can report which adapters exist.
    Subclasses DeviceError: code that already handles "no usable device" keeps
    working, it just gets a message it can act on.
    """
    def __init__(self, message, topology=None):
        super().__init__(message)
        self.topology = topology


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
        if _core_import_error is not None or _io_import_error is not None:
            raise RapidshotError(
                "DXGI device enumeration is not available on this platform."
            ) from (_core_import_error or _io_import_error)
        try:
            # Probe the topology before opening any device. An adapter with no
            # outputs is dropped below, but it is exactly the signal that
            # distinguishes a headless machine from a hybrid GPU one — so the
            # classification has to happen while that information still exists.
            self.topology = probe_topology()

            p_adapters = enum_dxgi_adapters()
            if not p_adapters:
                logger.error("No DXGI adapters found")
                raise HeadlessError(self.topology.help_text(), topology=self.topology)

            self.devices, self.outputs = [], []
            self.device_failures = []

            for p_adapter in p_adapters:
                try:
                    device = Device(p_adapter)
                    p_outputs = device.enum_outputs()
                    if len(p_outputs) != 0:
                        self.devices.append(device)
                        self.outputs.append([Output(p_output) for p_output in p_outputs])
                except Exception as e:
                    logger.warning(f"Failed to initialize device: {e}")
                    self.device_failures.append(str(e))

            if not self.devices:
                # Every adapter either has no output or refused to open. The
                # topology says which, and only one of the two is fixable by
                # the user.
                logger.error(f"No capture-capable device ({self.topology.kind})")
                raise HeadlessError(
                    self._no_device_message(), topology=self.topology
                )

            self.output_metadata = get_output_metadata()
            if self.topology.is_hybrid:
                logger.warning(
                    "Hybrid GPU system: capture is bound to the display adapter. "
                    "See rapidshot.topology_info() for detail."
                )
            logger.info(f"RapidshotFactory initialized with {len(self.devices)} devices")
        except RapidshotError:
            raise
        except Exception as e:
            error_msg = f"Failed to initialize RapidshotFactory: {e}"
            logger.error(error_msg)
            raise RapidshotError(error_msg) from e

    def _no_device_message(self) -> str:
        """Explain why no adapter could be used, without guessing."""
        help_text = self.topology.help_text()
        if not help_text:
            # Outputs exist, so this is not a headless machine: every device
            # creation failed instead. Report that, not a display problem.
            help_text = (
                "No usable graphics device. Adapters with displays attached were "
                "found, but none could be opened as a Direct3D 11 device."
            )
        if self.device_failures:
            failures = "\n".join(f"  {f}" for f in self.device_failures)
            help_text += f"\n\nDevice creation errors:\n{failures}"
        return help_text

    def create(
        self,
        device_idx: int = 0,
        output_idx: int = None,
        region: tuple = None,
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len: int = 64,
        prefer_integrated: bool = False,  # New parameter to force integrated GPU
        pool_output: bool = True,
    ) -> "ScreenCapture":
        """
        Create a ScreenCapture instance.

        Args:
            device_idx: Device index
            output_idx: Output index (None for primary)
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for capture
            prefer_integrated: If True, will search for an integrated GPU (e.g., Intel) and select it.
            pool_output: Reuse converted-frame buffers instead of allocating one
                per frame (default since 2.0). grab() then returns a
                PooledBuffer the caller must release. Pass False for the
                pre-2.0 behaviour.

        Returns:
            ScreenCapture instance
        """
        if ScreenCapture is None:
            raise RapidshotError(
                "ScreenCapture is not available on this platform."
            ) from _capture_import_error
        logger.debug(f"Creating ScreenCapture with device_idx={device_idx}, output_idx={output_idx}, nvidia_gpu={nvidia_gpu}, prefer_integrated={prefer_integrated}")
        
        # If the user prefers an integrated GPU, try to find one automatically.
        if prefer_integrated:
            for idx, device in enumerate(self.devices):
                desc = device.desc.Description if device.desc and hasattr(device.desc, "Description") else ""
                if "intel" in desc.lower():
                    device_idx = idx
                    logger.info(f"Selecting integrated GPU: {desc} at index {idx}")
                    break
            else:
                logger.info("No integrated GPU found; using default device index.")
        
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
            output = self.outputs[device_idx][output_idx]
            output.update_desc()
            
            if nvidia_gpu:
                try:
                    import cupy  # type: ignore[import-not-found]
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
                pool_output=pool_output,
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
        # Devices only cover adapters that drive a display. Adapters that do
        # not are invisible here otherwise, which is what makes a hybrid
        # system look like a plain single-GPU one.
        ret += self.topology.describe() + "\n"
        return ret

    def topology_info(self) -> str:
        """
        Get the GPU/display topology: which adapters exist, which drive a
        display, and what that implies for capture.

        Returns:
            Multi-line string
        """
        return self.topology.describe()

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

    def clean_up(self) -> None:
        """
        Release all created screencapture instances.
        """
        logger.info("Cleaning up all ScreenCapture instances")
        for _, screencapture in self._screencapture_instances.items():
            try:
                screencapture.release()
            except Exception as e:
                logger.warning(f"Error releasing ScreenCapture instance: {e}")

    def reset(self) -> None:
        """
        Reset the factory, releasing all resources.
        """
        logger.info("Resetting RapidshotFactory")
        self.clean_up()
        self._screencapture_instances.clear()
        Singleton._instances.clear()


# Global factory instance
__factory = None

def get_factory() -> "RapidshotFactory":
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
    prefer_integrated: bool = False,  # New parameter passed to factory
    pool_output: bool = True,
) -> "ScreenCapture":
    """
    Create a ScreenCapture instance.
    
    Args:
        device_idx: Device index
        output_idx: Output index (None for primary)
        region: Region to capture (left, top, right, bottom)
        output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        nvidia_gpu: Whether to use NVIDIA GPU acceleration
        max_buffer_len: Maximum buffer length for capture
        prefer_integrated: If True, forces selection of an integrated GPU if available.
        pool_output: Reuse converted-frame buffers (default since 2.0); grab()
            then returns a PooledBuffer the caller must release. Pass False
            for the pre-2.0 behaviour of a freshly allocated array.
        
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
        prefer_integrated=prefer_integrated,
        pool_output=pool_output,
    )

def device_info() -> str:
    """
    Get information about available devices.
    
    Returns:
        String with device information
    """
    factory = get_factory()
    return factory.device_info()

def output_info() -> str:
    """
    Get information about available outputs.

    Returns:
        String with output information
    """
    factory = get_factory()
    return factory.output_info()

def topology_info() -> str:
    """
    Get the GPU/display topology.

    Unlike device_info(), this reports adapters that cannot capture too — a
    render-only dGPU on a hybrid laptop, or a software adapter. Safe to call on
    a machine where capture itself is unavailable: it probes DXGI directly
    rather than going through the factory.

    Returns:
        String describing the topology
    """
    global __factory
    if __factory is not None:
        return __factory.topology_info()
    return probe_topology().describe()

def clean_up() -> None:
    """
    Release all created screencapture instances.
    """
    global __factory
    if __factory is not None:
        __factory.clean_up()

def reset() -> None:
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
        import cupy  # type: ignore[import-not-found]
        info["dependencies"]["cupy"] = cupy.__version__
    except ImportError:
        info["dependencies"]["cupy"] = "not installed"
    
    # Check pillow
    try:
        from PIL import __version__ as pil_version  # type: ignore[import-not-found]
        info["dependencies"]["pillow"] = pil_version
    except ImportError:
        info["dependencies"]["pillow"] = "not installed"
    
    # Check opencv
    try:
        import cv2  # type: ignore[import-not-found]
        info["dependencies"]["opencv"] = cv2.__version__
    except ImportError:
        info["dependencies"]["opencv"] = "not installed"
    
    # Check comtypes
    try:
        import comtypes  # type: ignore[import-untyped]
        info["dependencies"]["comtypes"] = comtypes.__version__
    except (ImportError, AttributeError):
        info["dependencies"]["comtypes"] = "version unknown"
    
    return info

# Version information
__version__ = "2.1.0"
__author__ = "Rapidshot Contributors"
__description__ = "High-performance screencapture library for Windows using Desktop Duplication API"

