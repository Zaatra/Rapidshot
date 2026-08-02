import enum
from typing import Any, Optional


class ProcessorBackends(enum.Enum):
    """
    Enumeration of available processor backends.
    """
    PIL = 0
    NUMPY = 1
    CUPY = 2


# Number of 8-bit channels each output color mode produces. The captured
# desktop surface is always BGRA (4 channels); these are the shapes callers
# actually receive, and what buffer-size checks are computed from.
COLOR_MODE_CHANNELS = {
    "BGRA": 4,
    "RGBA": 4,
    "RGB": 3,
    "BGR": 3,
    "GRAY": 1,
}


def validate_color_mode(color_mode: str) -> str:
    """
    Validate an output color mode, raising immediately on an unknown value.

    Catching this at construction time matters: an unsupported mode used to be
    swallowed deep in the conversion path and silently yielded unconverted BGRA
    frames, so callers got the wrong pixel format with no error anywhere.

    Args:
        color_mode: Requested output color format

    Returns:
        The validated color mode

    Raises:
        ValueError: If the color mode is not supported
    """
    if color_mode not in COLOR_MODE_CHANNELS:
        raise ValueError(
            f"Unsupported color mode: {color_mode!r}. "
            f"Supported modes: {sorted(COLOR_MODE_CHANNELS)}"
        )
    return color_mode


def channels_for_color_mode(color_mode: Optional[str]) -> int:
    """
    Channel count for a color mode, accepting the internal ``None`` sentinel.

    Backends set ``self.color_mode = None`` for BGRA to mark "no conversion
    needed", so this treats None as BGRA.
    """
    if color_mode is None:
        return COLOR_MODE_CHANNELS["BGRA"]
    return COLOR_MODE_CHANNELS[validate_color_mode(color_mode)]


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

        Raises:
            ValueError: If output_color is not a supported color mode
        """
        self.color_mode = validate_color_mode(output_color)

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
            import cv2  # type: ignore[import-not-found]
            version = cv2.__version__
            if version < "4.5.0":
                print(f"Warning: Using OpenCV version {version}. Version 4.5.0 or higher is recommended.")
        except ImportError:
            pass

    def process(self, rect, width, height, region, rotation_angle,
                output_buffer: Optional[Any] = None, dirty_rects=None,
                output_target=None):
        """
        Process a frame.

        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle
            output_buffer: Pre-allocated destination
            dirty_rects: Regions that changed, in region-relative coordinates.
                Backends that cannot use them ignore the argument.

        Returns:
            Processed frame
        """
        # Passed through only to backends that accept them. CuPy and Pillow have
        # their own process() signatures, and forwarding an argument they do not
        # take would raise a TypeError that _grab()'s catch-all turns into a
        # silent None -- capture stops with no usable diagnostic.
        if self.backend_supports_dirty_rects:
            extra = {}
            if dirty_rects is not None:
                extra["dirty_rects"] = dirty_rects
            if output_target is not None:
                extra["output_target"] = output_target
            if extra:
                return self.backend.process(rect, width, height, region,
                                            rotation_angle, output_buffer, **extra)
        return self.backend.process(rect, width, height, region, rotation_angle, output_buffer)

    @property
    def converts_output(self) -> bool:
        """Whether frames go through a colour conversion at all.

        BGRA does not: the staging buffer *is* the frame, returned with no copy.
        Note the asymmetry this guards against — a backend sets its own
        ``color_mode`` to None to mean "no conversion", while this wrapper keeps
        the validated string, so ``color_mode is None`` is true on the backend
        and never here.
        """
        return self.color_mode not in (None, "BGRA")

    @property
    def backend_supports_dirty_rects(self) -> bool:
        """Whether this backend can convert only the changed regions."""
        return hasattr(self.backend, "invalidate_accumulator")

    def invalidate_accumulator(self) -> None:
        """Forward accumulator invalidation to backends that keep one."""
        invalidate = getattr(self.backend, "invalidate_accumulator", None)
        if invalidate is not None:
            invalidate()


    @property
    def output_channels(self) -> int:
        """Number of channels frames from this processor carry."""
        return channels_for_color_mode(self.color_mode)

    def bytes_required(self, width: int, height: int) -> int:
        """Destination buffer size, in bytes, that ``process2`` will write."""
        return width * height * self.output_channels

    def process2(self, image_ptr, rect, width, height, buffer_size: Optional[int] = None):
        """
        Process directly to a provided memory buffer.

        Args:
            image_ptr: Pointer to (or object exposing) the destination buffer
            rect: Mapped rectangle
            width: Width
            height: Height
            buffer_size: Size of the destination buffer in bytes. Required
                whenever the size cannot be derived from image_ptr itself --
                without it there is no way to detect an undersized buffer
                before writing to it.
        """
        if hasattr(self.backend, 'shot'):
            return self.backend.shot(image_ptr, rect, width, height, buffer_size)
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