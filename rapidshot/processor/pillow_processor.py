import ctypes
from PIL import Image
from rapidshot.processor.base import ProcessorBackends


class PillowProcessor:
    """
    PIL-based processor for image processing.
    """
    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.PIL
    
    def __init__(self, color_mode: str = "RGB"):
        """
        Initialize the processor.
        
        Args:
            color_mode: Color format (RGB, RGBA, BGR, BGRA, GRAY)
        """
        self.color_mode = color_mode
        
        # Handle BGR formats by setting a flag for channel reordering
        self.bgr_mode = color_mode in ("BGR", "BGRA")
        if self.bgr_mode:
            # Convert BGR to RGB or BGRA to RGBA for PIL processing
            self.pil_mode = "RGB" if color_mode == "BGR" else "RGBA"
        else:
            self.pil_mode = color_mode

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
            Processed frame as numpy array
        """
        pitch = int(rect.Pitch)

        # Calculate buffer size based on rotation
        if rotation_angle in (0, 180):
            size = pitch * height
        else:
            size = pitch * width

        # Get raw buffer data
        buffer = ctypes.string_at(rect.pBits, size)
        pitch //= 4
        
        # Create PIL image with appropriate dimensions
        if rotation_angle in (0, 180):
            image = Image.frombuffer("RGBA", (pitch, height), buffer)
        elif rotation_angle in (90, 270):
            image = Image.frombuffer("RGBA", (width, pitch), buffer)
        else:
            raise RuntimeError("Invalid rotation angle")
        
        # Fix channel order (BGRA to RGBA)
        if self.color_mode != "BGRA":
            blue, green, red, alpha = image.split()
            if self.color_mode == "RGB":
                image = Image.merge("RGB", (red, green, blue))
            elif self.color_mode == "RGBA":
                image = Image.merge("RGBA", (red, green, blue, alpha))
            elif self.color_mode == "BGR":
                image = Image.merge("RGB", (blue, green, red))
            elif self.color_mode == "GRAY":
                # PIL doesn't have a direct BGRA to GRAY mode, convert to RGB first
                image = Image.merge("RGB", (red, green, blue)).convert("L")
        
        # Apply rotation - support both newer and older PIL versions
        try:
            if rotation_angle == 90:
                image = image.transpose(Image.ROTATE_90)
            elif rotation_angle == 180:
                image = image.transpose(Image.ROTATE_180)
            elif rotation_angle == 270:
                image = image.transpose(Image.ROTATE_270)
        except AttributeError:
            # Fall back to older PIL version constants
            if rotation_angle == 90:
                image = image.transpose(Image.Transpose.ROTATE_90)
            elif rotation_angle == 180:
                image = image.transpose(Image.Transpose.ROTATE_180)
            elif rotation_angle == 270:
                image = image.transpose(Image.Transpose.ROTATE_270)

        # Crop to correct dimensions if needed
        if rotation_angle in (0, 180) and pitch != width:
            image = image.crop((0, 0, width, image.height))
        elif rotation_angle in (90, 270) and pitch != height:
            image = image.crop((0, 0, image.width, height))

        # Apply region cropping
        if region[2] - region[0] != width or region[3] - region[1] != height:
            image = image.crop((region[0], region[1], region[2], region[3]))

        # Convert to numpy array for compatibility
        import numpy as np
        return np.array(image)