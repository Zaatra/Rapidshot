import ctypes
import numpy as np
from numpy import rot90, ndarray, newaxis, uint8
from numpy.ctypeslib import as_array
from rapidshot.processor.base import ProcessorBackends


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
        Convert color format.
        
        Args:
            image: Image to convert
            
        Returns:
            Converted image
        """
        import cv2

        # Initialize color conversion function once
        if self.cvtcolor is None:
            color_mapping = {
                "RGB": cv2.COLOR_BGRA2RGB,
                "RGBA": cv2.COLOR_BGRA2RGBA,
                "BGR": cv2.COLOR_BGRA2BGR,
                "GRAY": cv2.COLOR_BGRA2GRAY
            }
            cv2_code = color_mapping[self.color_mode]
            
            # Create appropriate converter function
            if cv2_code != cv2.COLOR_BGRA2GRAY:
                self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)
            else:
                # Add axis for grayscale to maintain shape consistency
                self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)[..., np.newaxis]
                
        return self.cvtcolor(image)

    def shot(self, image_ptr, rect, width, height):
        """
        Process directly to a provided memory buffer.
        
        Args:
            image_ptr: Pointer to image buffer
            rect: Mapped rectangle
            width: Width
            height: Height
        """
        # Direct memory copy for maximum performance
        ctypes.memmove(image_ptr, rect.pBits, height * width * 4)

    def process(self, rect, width, height, region, rotation_angle):
        """
        Process a frame.
        
        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle
            
        Returns:
            Processed frame
        """
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

        # Use direct memory access for efficiency
        buffer = (ctypes.c_char * size).from_address(ctypes.addressof(rect.pBits.contents) + offset)
        pitch = pitch // 4
        
        # Create NumPy array from buffer with appropriate shape
        if rotation_angle in (0, 180):
            image = np.ndarray((height, pitch, 4), dtype=np.uint8, buffer=buffer)
        elif rotation_angle in (90, 270):
            image = np.ndarray((width, pitch, 4), dtype=np.uint8, buffer=buffer)

        # Convert color format if needed
        if self.color_mode is not None:
            image = self.process_cvtcolor(image)

        # Apply rotation
        if rotation_angle == 90:
            image = np.rot90(image, axes=(1, 0))
        elif rotation_angle == 180:
            image = np.rot90(image, k=2, axes=(0, 1))
        elif rotation_angle == 270:
            image = np.rot90(image, axes=(0, 1))

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