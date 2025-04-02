import ctypes
import numpy as np
import logging
from numpy import rot90, ndarray, newaxis, uint8
from numpy.ctypeslib import as_array
from rapidshot.processor.base import ProcessorBackends

# Set up logger
logger = logging.getLogger(__name__)

class NumpyProcessor:
    """
    NumPy-based processor for image processing.
    """
    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.NUMPY
    
    def __init__(self, color_mode):
        """
        Initialize the processor.
        
        Args:
            color_mode: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        """
        self.cvtcolor = None
        self.color_mode = color_mode
        self.PBYTE = ctypes.POINTER(ctypes.c_ubyte)
        
        # Simplified processing for BGRA
        if self.color_mode == 'BGRA':
            self.color_mode = None

    def process_cvtcolor(self, image):
        """
        Convert color format with robust error handling.
        
        Args:
            image: Image to convert
            
        Returns:
            Converted image
        """
        # Fixed region handling patch applied
        # Skip color conversion if image is None or empty
        if image is None or image.size == 0:
            logger.warning("Received empty image for color conversion")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Ensure image has proper shape and type
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except Exception as e:
                logger.warning(f"Failed to convert image to numpy array: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
        # Handle images with no channels or wrong number of channels
        if len(image.shape) < 3 or image.shape[2] < 3:
            try:
                import cv2
                # Convert grayscale to BGR if needed
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return image
            except Exception as e:
                logger.warning(f"Failed to convert image format: {e}")
                return np.zeros((image.shape[0] if len(image.shape) > 0 else 480, 
                                image.shape[1] if len(image.shape) > 1 else 640, 3), dtype=np.uint8)
        
        try:
            import cv2

            # Initialize color conversion function once
            if self.cvtcolor is None:
                color_mapping = {
                    "RGB": cv2.COLOR_BGRA2RGB,
                    "RGBA": cv2.COLOR_BGRA2RGBA,
                    "BGR": cv2.COLOR_BGRA2BGR,
                    "GRAY": cv2.COLOR_BGRA2GRAY
                }
                
                if self.color_mode not in color_mapping:
                    logger.warning(f"Unsupported color mode: {self.color_mode}. Falling back to BGR.")
                    cv2_code = cv2.COLOR_BGRA2BGR
                else:
                    cv2_code = color_mapping[self.color_mode]
                
                # Create appropriate converter function
                if cv2_code != cv2.COLOR_BGRA2GRAY:
                    self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)
                else:
                    # Add axis for grayscale to maintain shape consistency
                    self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)[..., np.newaxis]
                    
            return self.cvtcolor(image)
        except Exception as e:
            logger.warning(f"Color conversion error: {e}")
            # Return original image as fallback
            if image.shape[2] >= 3:
                return image[:, :, :3]  # Just take first 3 channels
            return image

    def shot(self, image_ptr, rect, width, height):
        """
        Process directly to a provided memory buffer.
        
        Args:
            image_ptr: Pointer to image buffer
            rect: Mapped rectangle
            width: Width
            height: Height
        """
        try:
            # Direct memory copy for maximum performance
            ctypes.memmove(image_ptr, rect.pBits, height * width * 4)
        except Exception as e:
            logger.error(f"Memory copy error: {e}")

    def process(self, rect, width, height, region, rotation_angle):
        """
        Process a frame with robust error handling.
        
        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle
            
        Returns:
            Processed frame
        """
        # Fixed region handling patch applied
        try:
            pitch = int(rect.Pitch)
            
            # Validate region bounds and clip to valid values
            region = list(region)
            region[0] = max(0, min(region[0], width-1))
            region[1] = max(0, min(region[1], height-1))
            region[2] = max(region[0]+1, min(region[2], width))
            region[3] = max(region[1]+1, min(region[3], height))

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

            # Use direct memory access for efficiency
            try:
                buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
                pitch = pitch // 4
                
                # Create NumPy array from buffer with appropriate shape
                if rotation_angle in (0, 180):
                    image = np.ndarray((height, pitch, 4), dtype=np.uint8, buffer=buffer)
                elif rotation_angle in (90, 270):
                    image = np.ndarray((width, pitch, 4), dtype=np.uint8, buffer=buffer)
                else:
                    raise RuntimeError(f"Invalid rotation angle: {rotation_angle}")
            except Exception as e:
                logger.error(f"Buffer access error: {e}")
                # Create an empty frame as fallback
                image = np.zeros((height if rotation_angle in (0, 180) else width, 
                                pitch, 4), dtype=np.uint8)

            # Convert color format if needed
            if self.color_mode is not None and image.size > 0:
                try:
                    image = self.process_cvtcolor(image)
                except Exception as e:
                    logger.error(f"Color conversion error: {e}")
                    # If color conversion fails, just return the original image
                    pass

            # Apply rotation safely
            try:
                if rotation_angle == 90:
                    image = np.rot90(image, axes=(1, 0))
                elif rotation_angle == 180:
                    image = np.rot90(image, k=2, axes=(0, 1))
                elif rotation_angle == 270:
                    image = np.rot90(image, axes=(0, 1))
            except Exception as e:
                logger.error(f"Rotation error: {e}")
                # Continue with unrotated image

            # Safe cropping with bounds checking
            try:
                # Crop to actual dimensions if needed
                if rotation_angle in (0, 180) and pitch != width:
                    image = image[:, :min(width, image.shape[1]), :]
                elif rotation_angle in (90, 270) and pitch != height:
                    image = image[:min(height, image.shape[0]), :, :]
            except Exception as e:
                logger.error(f"Dimension cropping error: {e}")
                # Continue with uncropped image

            # Final region adjustment with safe bounds checking
            try:
                h, w = image.shape[:2]
                if region[3] - region[1] != h and region[1] < h and region[3] <= h:
                    image = image[region[1]:region[3], :, :]
                if region[2] - region[0] != w and region[0] < w and region[2] <= w:
                    image = image[:, region[0]:region[2], :]
            except Exception as e:
                logger.error(f"Region adjustment error: {e}")
                # Continue with unadjusted image

            return image
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Return an empty frame in case of error
            return np.zeros((480, 640, 3), dtype=np.uint8)