import ctypes
import platform
import logging  # Added missing import
from rapidshot.util.logging import get_logger
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
            rotation_angle: Rotation angle,
            output_buffer: Pre-allocated CuPy array to store the processed frame.
        """
        # Phase 1: Get data into the output buffer (no rotation, no color conversion yet)
        # Import numpy for ctypes bridge, cupy (self.cp) is already imported
        import numpy as np

        try:
            if not hasattr(rect, 'pBits') or not rect.pBits:
                logger.warning(f"Invalid rect or pBits, cannot process. Rect type: {type(rect)}")
                if output_buffer is not None:
                    output_buffer.fill(0)
                return

            # Original width and height of the texture from Duplicator
            # width, height are passed in, representing the full output dimensions

            if hasattr(rect, 'Pitch'):
                pitch = int(rect.Pitch)
            else:
                logger.debug(f"Rect of type {type(rect)} doesn't have Pitch attribute, estimating based on width")
                pitch = width * 4 # Assuming BGRA format (4 bytes per pixel)

            buffer_address = ctypes.addressof(rect.pBits.contents)
            
            # Create a NumPy array view of the entire source frame buffer from rect.pBits
            # This view is on CPU memory, pointing to the DDA buffer.
            source_image_np_view = np.ctypeslib.as_array(
                (ctypes.c_ubyte * (pitch * height)).from_address(buffer_address)
            ).reshape((height, pitch // 4, 4)) # height, actual_cols_with_pitch, channels

            # Validate and adjust region coordinates
            left, top, right, bottom = region
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            right = max(left + 1, min(right, width))
            bottom = max(top + 1, min(bottom, height))

            region_height = bottom - top
            region_width = right - left

            if output_buffer.shape[0] != region_height or \
               output_buffer.shape[1] != region_width or \
               output_buffer.shape[2] != 4: # Assuming BGRA for now
                logger.error(
                    f"Output buffer shape {output_buffer.shape} does not match "
                    f"region shape ({region_height}, {region_width}, 4)."
                )
                output_buffer.fill(0)
                return
            
            # Extract the specified region from the source_image_np_view (this is still a NumPy array/view)
            # This creates a view, or a copy if slicing is complex, but it's CPU-side.
            region_np_view = source_image_np_view[top:bottom, left:right, :]
            
            # Ensure region_np_view is C-contiguous for CuPy copy if it's not already.
            # This is important if the slicing above created a non-contiguous view.
            if not region_np_view.flags['C_CONTIGUOUS']:
                region_np_view = np.ascontiguousarray(region_np_view)

            # Copy data from the NumPy region view (CPU) to the CuPy output_buffer (GPU)
            output_buffer.set(region_np_view)

            
            # Phase 2: Color Conversion and Rotation
            current_array = output_buffer # Start with the pooled buffer (already has BGRA data)
            is_still_pooled_buffer = True

            # Color Conversion
            if self.color_mode is not None: # Not 'BGRA', so conversion is intended
                # process_cvtcolor expects a CuPy array if _has_cucv, or NumPy if falling back to cv2
                # Since current_array is CuPy, this is fine for cuCV.
                # For OpenCV fallback, process_cvtcolor would need a NumPy array.
                # Let's assume process_cvtcolor is adapted or handles CuPy array input.
                # For now, we pass current_array. If it's OpenCV, it might involve implicit DtoH copy.
                
                # If using OpenCV (non-cuCV path), it's better to convert from the CPU numpy array
                # *before* copying to output_buffer, or copy output_buffer to CPU, convert, copy back.
                # This logic assumes process_cvtcolor can handle a CuPy array and returns a CuPy array.
                
                temp_for_conversion = current_array
                # If not using cuCV, and process_cvtcolor expects NumPy, we need a DtoH copy
                if not self._has_cucv:
                    logger.debug("CupyProcessor: Using OpenCV for color conversion, involves DtoH copy.")
                    temp_for_conversion = self.cp.asnumpy(current_array)

                converted_array = self.process_cvtcolor(temp_for_conversion) # process_cvtcolor returns array

                # If OpenCV was used, converted_array is NumPy, convert back to CuPy
                if not self._has_cucv and isinstance(converted_array, np.ndarray):
                    converted_array = self.cp.asarray(converted_array)

                if converted_array.shape[0] == current_array.shape[0] and \
                   converted_array.shape[1] == current_array.shape[1]:
                    if converted_array.shape[2] != current_array.shape[2]: # Channel change
                        current_array = converted_array
                        is_still_pooled_buffer = False
                    elif converted_array.data.ptr != current_array.data.ptr: # Different memory block
                        if is_still_pooled_buffer:
                            current_array[:] = converted_array
                        # else current_array is already new, no need to copy to original output_buffer
                else: # Height/width changed
                    logger.warning("CuPy color conversion changed height/width, which is unexpected.")
                    current_array = converted_array
                    is_still_pooled_buffer = False
            
            # Rotation
            if rotation_angle != 0:
                k = (rotation_angle // 90) % 4
                if k != 0:
                    rotated_array = self.cp.rot90(current_array, k=k)
                    
                    if rotated_array.shape[0] != current_array.shape[0] or \
                       rotated_array.shape[1] != current_array.shape[1]:
                        current_array = rotated_array
                        is_still_pooled_buffer = False
                    elif is_still_pooled_buffer:
                        current_array[:] = rotated_array
                    else: # Shape is same, but current_array is already a new buffer
                        current_array = rotated_array 

            return current_array, is_still_pooled_buffer

        except Exception as e:
            error_msg = f"Error processing frame with CuPy: {e}"
            logger.error(error_msg)
            if output_buffer is not None and hasattr(output_buffer, 'fill'):
                try:
                    output_buffer.fill(0)
                except Exception as fill_e:
                    logger.error(f"Error filling CuPy output_buffer after another error: {fill_e}")
            return output_buffer, False # Indicate buffer might be invalid