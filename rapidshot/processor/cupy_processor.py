import ctypes
import platform
import logging
import warnings
import sys
from rapidshot.processor.base import ProcessorBackends

# Configure logging
logger = logging.getLogger(__name__)

class CupyProcessor:
    """
    CUDA-accelerated processor using CuPy.
    """
    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.CUPY
    
    # Minimum required CuPy version
    MIN_CUPY_VERSION = "10.0.0"
    
    def __init__(self, color_mode):
        """
        Initialize the processor.
        
        Args:
            color_mode: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        """
        # Import CuPy in constructor to delay import until needed
        try:
            import cupy as cp
            self.cp = cp
            
            # Check version compatibility
            version = cp.__version__
            if version < self.MIN_CUPY_VERSION:
                warning_msg = (
                    f"Warning: Using CuPy version {version}. "
                    f"Version {self.MIN_CUPY_VERSION} or higher is recommended. "
                    f"Some functionality may be limited or unstable."
                )
                logger.warning(warning_msg)
                warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
                
                # Continue with available functionality
                self._check_for_critical_cupy_features()
                
        except ImportError as e:
            # Get platform-specific installation instructions
            install_cmd = self._get_platform_specific_cupy_install()
            error_msg = (
                f"CuPy is required for CUDA acceleration. Error: {e}\n\n"
                f"To install CuPy for your platform ({platform.system()}, {platform.machine()}):\n"
                f"{install_cmd}\n\n"
                f"If you don't need GPU acceleration, initialize without 'nvidia_gpu=True'."
            )
            logger.error(error_msg)
            raise ImportError(error_msg) from e
            
        self.cvtcolor = None
        self.color_mode = color_mode
        
        # Try importing cuCV now to give early warning
        try:
            import cucv.cv2
            self._has_cucv = True
            logger.info("Using cuCV for color conversion (GPU accelerated)")
        except ImportError:
            self._has_cucv = False
            logger.info("cuCV not found, falling back to regular OpenCV for color conversion")
            
        # Simplified processing for BGRA
        if self.color_mode == 'BGRA':
            self.color_mode = None
    
    def _check_for_critical_cupy_features(self):
        """
        Check for critical CuPy features needed by the processor.
        Will fall back to compatible functionality if needed.
        """
        try:
            # Test critical functions we'll use
            test_array = self.cp.zeros((10, 10, 3), dtype=self.cp.uint8)
            # Test rotation
            self.cp.rot90(test_array)
            # Test array copying
            self.cp.asarray(test_array)
            # Test memory allocation
            self.cp.frombuffer(b"test", dtype=self.cp.uint8)
            
            logger.debug("All required CuPy features are available")
        except AttributeError as e:
            warning_msg = (
                f"Your CuPy version is missing some required features: {e}. "
                f"Some functionality might be limited."
            )
            logger.warning(warning_msg)
            warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
    
    def _get_platform_specific_cupy_install(self):
        """
        Get platform-specific installation instructions for CuPy.
        
        Returns:
            String with installation instructions
        """
        system = platform.system()
        if system == "Windows":
            # Check Python version to recommend correct CUDA version
            py_ver = sys.version_info
            if py_ver.major == 3 and py_ver.minor >= 10:
                return (
                    "pip install cupy-cuda11x\n"
                    "# Make sure you have CUDA 11.0+ installed from https://developer.nvidia.com/cuda-downloads\n"
                    "# For more detailed instructions: https://docs.cupy.dev/en/stable/install.html"
                )
            else:
                return (
                    "pip install cupy-cuda11x  # For CUDA 11.0+\n"
                    "# or\n"
                    "pip install cupy-cuda10x  # For CUDA 10.0+\n"
                    "# Make sure you have matching CUDA version installed from https://developer.nvidia.com/cuda-downloads"
                )
        elif system == "Linux":
            return (
                "# Install CUDA first using your package manager\n"
                "# For Ubuntu: sudo apt install nvidia-cuda-toolkit\n"
                "pip install cupy-cuda11x  # Adjust version based on your CUDA installation"
            )
        elif system == "Darwin":  # macOS
            return (
                "# Note: CUDA support on macOS is limited\n"
                "# For Apple Silicon (M1/M2):\n"
                "pip install cupy\n"
                "# For Intel Macs with NVIDIA GPUs, first install CUDA, then:\n"
                "pip install cupy-cuda11x"
            )
        else:
            return "pip install cupy  # Please check https://docs.cupy.dev/en/stable/install.html for detailed instructions"

    def process_cvtcolor(self, image):
        """
        Convert color format using cuCV or OpenCV.
        
        Args:
            image: Image to convert
            
        Returns:
            Converted image
        """
        # Use the already imported cuCV if available, otherwise use regular OpenCV
        if self._has_cucv:
            try:
                import cucv.cv2 as cv2
            except ImportError as e:
                logger.warning(f"Failed to import cuCV, falling back to regular OpenCV: {e}")
                import cv2
        else:
            try:
                import cv2
            except ImportError as e:
                error_msg = (
                    f"OpenCV is required for color conversion. Error: {e}\n"
                    f"Install OpenCV: pip install opencv-python"
                )
                logger.error(error_msg)
                raise ImportError(error_msg) from e
            
        # Initialize color conversion function once
        if self.cvtcolor is None:
            try:
                color_mapping = {
                    "RGB": cv2.COLOR_BGRA2RGB,
                    "RGBA": cv2.COLOR_BGRA2RGBA,
                    "BGR": cv2.COLOR_BGRA2BGR,
                    "GRAY": cv2.COLOR_BGRA2GRAY
                }
                
                if self.color_mode not in color_mapping:
                    error_msg = f"Unsupported color mode: {self.color_mode}. Supported modes: {list(color_mapping.keys())}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                    
                cv2_code = color_mapping[self.color_mode]
                
                # Create appropriate converter function
                if cv2_code != cv2.COLOR_BGRA2GRAY:
                    self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)
                else:
                    # Add axis for grayscale to maintain shape consistency
                    self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)[..., self.cp.newaxis]
            except Exception as e:
                error_msg = f"Failed to initialize color conversion: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e
                
        try:
            return self.cvtcolor(image)
        except Exception as e:
            error_msg = f"Error during color conversion: {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e

    def process(self, rect, width, height, region, rotation_angle):
        """
        Process a frame using GPU acceleration.
        
        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle
            
        Returns:
            Processed frame as CuPy array
        """
        try:
            pitch = int(rect.Pitch)

            # Calculate memory offset for region
            if rotation_angle in (0, 180):
                offset = (region[1] if rotation_angle == 0 else height - region[3]) * pitch
                height = region[3] - region[1]
            else:
                offset = (region[0] if rotation_angle == 270 else width - region[2]) * pitch
                width = region[2] - region[0]

            # Calculate buffer size
            if rotation_angle in (0, 180):
                size = pitch * height
            else:
                size = pitch * width

            # Get buffer and create CuPy array
            buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
            pitch = pitch // 4
            
            # Create CuPy array from buffer with appropriate shape
            if rotation_angle in (0, 180):
                # Transfer CPU memory to GPU
                cpu_array = self.cp.frombuffer(buffer, dtype=self.cp.uint8).reshape(height, pitch, 4)
                image = self.cp.asarray(cpu_array)
            elif rotation_angle in (90, 270):
                cpu_array = self.cp.frombuffer(buffer, dtype=self.cp.uint8).reshape(width, pitch, 4)
                image = self.cp.asarray(cpu_array)
            else:
                raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")

            # Convert color format if needed
            if self.color_mode is not None:
                image = self.process_cvtcolor(image)

            # Apply rotation
            if rotation_angle == 90:
                image = self.cp.rot90(image, axes=(1, 0))
            elif rotation_angle == 180:
                image = self.cp.rot90(image, k=2, axes=(0, 1))
            elif rotation_angle == 270:
                image = self.cp.rot90(image, axes=(0, 1))

            # Crop to actual dimensions if needed
            if rotation_angle in (0, 180) and pitch != width:
                image = image[:, :width, :]
            elif rotation_angle in (90, 270) and pitch != height:
                image = image[:height, :, :]

            # Final region adjustment
            if region[3] - region[1] != image.shape[0]:
                image = image[region[1]:region[3], :, :]
            if region[2] - region[0] != image.shape[1]:
                image = image[:, region[0]:region[2], :]

            return image
            
        except Exception as e:
            error_msg = f"Error processing frame with CuPy: {e}"
            logger.error(error_msg)
            # Re-raise with more context
            raise RuntimeError(error_msg) from e