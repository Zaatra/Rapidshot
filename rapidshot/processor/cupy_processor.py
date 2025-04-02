import ctypes
from rapidshot.processor.base import ProcessorBackends


class CupyProcessor:
    """
    CUDA-accelerated processor using CuPy.
    """
    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.CUPY
    
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
            if version < "10.0.0":
                print(f"Warning: Using CuPy version {version}. Version 10.0.0 or higher is recommended.")
        except ImportError:
            raise ImportError("CuPy is required for CUDA acceleration. Install with 'pip install cupy-cuda11x'")
            
        self.cvtcolor = None
        self.color_mode = color_mode
        
        # Try importing cuCV now to give early warning
        try:
            import cucv.cv2
            self._has_cucv = True
        except ImportError:
            self._has_cucv = False
            
        # Simplified processing for BGRA
        if self.color_mode == 'BGRA':
            self.color_mode = None

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
            import cucv.cv2 as cv2
        else:
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
                self.cvtcolor = lambda img: cv2.cvtColor(img, cv2_code)[..., self.cp.newaxis]
                
        return self.cvtcolor(image)

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