import enum
from typing import Any, Optional


class ProcessorBackends(enum.Enum):
    """
    Enumeration of available processor backends.
    """
    PIL = 0
    NUMPY = 1
    CUPY = 2


class Processor:
    """
    Base processor class that delegates processing to the selected backend.
    """
    def __init__(
        self, 
        backend: Optional[ProcessorBackends] = None, 
        output_color: str = "RGB", 
        nvidia_gpu: bool = False
    ):
        """
        Initialize the processor.
        
        Args:
            backend: Processor backend to use (auto-selected if None)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
        """
        self.color_mode = output_color
        
        # Auto-select backend
        if backend is None:
            if nvidia_gpu:
                backend = ProcessorBackends.CUPY
            else:
                # Use NumPy by default
                backend = ProcessorBackends.NUMPY
        
        # Store the selected backend type
        self._active_backend_type = backend
        
        # Initialize the selected backend
        self.backend = self._initialize_backend(backend)
        
        # Check dependencies versions for critical libraries
        self._check_dependencies()

    @property
    def active_backend(self):
        """
        Returns the currently active processor backend type.
        
        Returns:
            ProcessorBackends: The active backend type
        """
        return self._active_backend_type

    def _check_dependencies(self):
        """Check versions of critical dependencies and warn if needed."""
        try:
            import numpy as np
            version = np.__version__
            if version < "1.20.0":
                print(f"Warning: Using NumPy version {version}. Version 1.20.0 or higher is recommended.")
        except ImportError:
            pass
            
        try:
            from PIL import Image, __version__ as pil_version
            if pil_version < "9.0.0":
                print(f"Warning: Using PIL version {pil_version}. Version 9.0.0 or higher is recommended.")
        except (ImportError, AttributeError):
            pass
            
        try:
            import cv2
            version = cv2.__version__
            if version < "4.5.0":
                print(f"Warning: Using OpenCV version {version}. Version 4.5.0 or higher is recommended.")
        except ImportError:
            pass

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
        return self.backend.process(rect, width, height, region, rotation_angle)
    
    def process2(self, image_ptr, rect, width, height):
        """
        Process directly to a provided memory buffer.
        
        Args:
            image_ptr: Pointer to image buffer
            rect: Mapped rectangle
            width: Width
            height: Height
        """
        if hasattr(self.backend, 'shot'):
            self.backend.shot(image_ptr, rect, width, height)
        else:
            raise NotImplementedError("Direct buffer processing not supported by this backend")

    def _initialize_backend(self, backend: ProcessorBackends):
        """
        Initialize the processor backend.
        
        Args:
            backend: Backend to initialize
            
        Returns:
            Initialized backend
        """
        if backend == ProcessorBackends.NUMPY:
            try:
                from rapidshot.processor.numpy_processor import NumpyProcessor
                return NumpyProcessor(self.color_mode)
            except ImportError:
                print("NumPy backend not available, falling back to PIL")
                backend = ProcessorBackends.PIL
                self._active_backend_type = backend
        
        if backend == ProcessorBackends.CUPY:
            try:
                from rapidshot.processor.cupy_processor import CupyProcessor
                return CupyProcessor(self.color_mode)
            except ImportError:
                print("CuPy backend not available, falling back to NumPy")
                from rapidshot.processor.numpy_processor import NumpyProcessor
                backend = ProcessorBackends.NUMPY
                self._active_backend_type = backend
                return NumpyProcessor(self.color_mode)
        
        if backend == ProcessorBackends.PIL:
            try:
                from rapidshot.processor.pillow_processor import PillowProcessor
                return PillowProcessor(self.color_mode)
            except ImportError:
                raise ImportError("No available backend. Please install either NumPy or PIL.")
        
        raise ValueError(f"Unknown backend: {backend}")