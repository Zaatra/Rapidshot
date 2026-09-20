import ctypes
import platform
import logging  # Added missing import
from rapidshot.util.logging import get_logger
import warnings
from rapidshot.processor.base import (
    ProcessorBackends, check_surface_pitch, version_below)
from rapidshot.util.ctypes_helpers import pointer_to_address

# Configure logging
logger = logging.getLogger(__name__)

# Q8 luma coefficients, identical to `numpy_processor`. Duplicated rather than
# imported so the two paths cannot drift apart silently through a refactor of
# the other module -- the tests assert they agree.
_LUMA_R, _LUMA_G, _LUMA_B = 77, 150, 29
_LUMA_ROUND, _LUMA_SHIFT = 128, 8

_SUPPORTED_MODES = {"BGRA", "RGB", "BGR", "RGBA", "GRAY"}

#: Built on first use, because CuPy is imported lazily; CuPy caches the
#: compiled kernel thereafter.
_GRAY_KERNEL = None


def _gray_chained(cp, image):
    """Q8 luma as array expressions, for any NumPy-compatible module.

    The slower form, kept only because it is the portable one: it runs on
    anything with the NumPy API, which is what makes `CupyProcessor` testable
    without a GPU. `test_cupy_processor` asserts it and the fused kernel agree
    byte for byte on a real device, so the two cannot drift apart.

    The intermediate stays in uint16 because 255*(77+150+29) + 128 = 65408,
    just inside the limit; the +128 is round-to-nearest, and dropping it biases
    every pixel dark. One accumulator rather than one per channel.
    """
    acc = image[..., 2].astype(cp.uint16)
    acc *= cp.uint16(_LUMA_R)
    acc += cp.uint16(_LUMA_ROUND)
    acc += image[..., 1].astype(cp.uint16) * cp.uint16(_LUMA_G)
    acc += image[..., 0].astype(cp.uint16) * cp.uint16(_LUMA_B)
    acc >>= _LUMA_SHIFT
    return acc.astype(cp.uint8)[..., cp.newaxis]


def _gray_kernel(cp):
    """One pass over the frame for Q8 luma, instead of six.

    The chained expression this replaces launched a kernel per line and
    materialised a full-size uint16 temporary for each -- roughly six passes
    over an 8 MB frame to do arithmetic that needs one. Measured 6.4-9.6x
    faster from 1080p to 4K, and **byte-identical**, which is the part that
    matters: GRAY is asserted equal to `NumpyProcessor`'s output, so a kernel
    that is faster and differs by one level is a bug rather than a win.

    The intermediate is unsigned 16-bit for the same reason the chained form
    was: 255*(77+150+29) + 128 = 65408, just inside the limit, and the +128 is
    round-to-nearest -- dropping it biases every pixel dark.
    """
    if not hasattr(cp, "ElementwiseKernel"):
        # This class is deliberately NumPy-substitutable: `test_cupy_rotation`
        # runs the whole of it with NumPy standing in for `self.cp`, so it
        # needs no GPU and no CUDA driver. NumPy has no `ElementwiseKernel`,
        # so that stand-in takes `_gray_chained` instead.
        return None
    global _GRAY_KERNEL
    if _GRAY_KERNEL is None:
        _GRAY_KERNEL = cp.ElementwiseKernel(
            "uint8 b, uint8 g, uint8 r", "uint8 y",
            f"y = (unsigned short)((r * {_LUMA_R} + g * {_LUMA_G}"
            f" + b * {_LUMA_B} + {_LUMA_ROUND}) >> {_LUMA_SHIFT})",
            "rapidshot_bgra_to_gray")
    return _GRAY_KERNEL

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
            import cupy as cp  # type: ignore[import-not-found]
            self.cp = cp
            
            # Check version compatibility
            version = cp.__version__
            if version_below(version, self.MIN_CUPY_VERSION):
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
            
        # Reject an unsupported mode here rather than on the first frame that
        # happens to arrive. The same reasoning as `_validate_destination` in
        # the `shot()` fix (ROADMAP § 10): deferring the check to the processor
        # made it fire only on calls that received new content, so a bad
        # configuration looked fine on a static desktop and blew up later, when
        # something happened to move.
        if color_mode is not None and color_mode not in _SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported color mode: {color_mode!r}. "
                f"Supported modes: {sorted(_SUPPORTED_MODES)}")

        self.cvtcolor = None
        self.color_mode = color_mode

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
        """How to install CuPy for this package.

        This used to recommend ``cupy-cuda11x`` and ``cupy-cuda10x`` -- the
        latter long gone -- with Linux and macOS variants of a Windows-only
        library, and never mentioned the extras this package defines for it.
        The wheels are CUDA-major-specific and mutually exclusive, so the right
        one depends on the installed driver; ``nvidia-smi`` reports it.
        """
        return (
            "Check the CUDA version with `nvidia-smi`, then install the matching extra:\n"
            "  pip install rapidshot[gpu_cuda13]   # CUDA 13\n"
            "  pip install rapidshot[gpu_cuda12]   # CUDA 12\n"
            "  pip install rapidshot[gpu]          # CUDA 11"
        )

    def process_cvtcolor(self, image):
        """
        Convert a BGRA CuPy array to this processor's colour mode, on the GPU.

        Every mode is expressed in CuPy, so this needs no OpenCV and no cuCV.
        That is not only a dependency saving: the OpenCV path this replaced
        copied the frame **off** the GPU with ``cp.asnumpy``, converted it on the
        CPU, and copied it back — three PCIe crossings for a frame that was
        already resident, on a code path whose entire premise is GPU residency.

        The arithmetic is byte-for-byte the same as
        :meth:`NumpyProcessor.convert_into`. That matters more than it looks:
        it is what lets a caller switch ``nvidia_gpu`` on or off without any
        pixel changing, and OpenCV could not have offered it — its luma rounds
        differently and is off by up to 1 LSB (ROADMAP § 10).

        Args:
            image: (H, W, 4) uint8 BGRA CuPy array.

        Returns:
            A new CuPy array in the configured mode.
        """
        cp = self.cp
        mode = self.color_mode

        if mode is None or mode == "BGRA":
            return image
        if mode == "RGB":
            return cp.ascontiguousarray(image[..., 2::-1])
        if mode == "BGR":
            return cp.ascontiguousarray(image[..., :3])
        if mode == "RGBA":
            out = cp.empty_like(image)
            out[..., 0] = image[..., 2]
            out[..., 1] = image[..., 1]
            out[..., 2] = image[..., 0]
            out[..., 3] = image[..., 3]
            return out
        if mode == "GRAY":
            # Q8 luma, identical to the NumPy path -- see `_gray_kernel`.
            # Trailing axis kept so GRAY frames index like every other mode.
            kernel = _gray_kernel(cp)
            if kernel is None:
                return _gray_chained(cp, image)
            return kernel(
                image[..., 0], image[..., 1], image[..., 2])[..., cp.newaxis]

        raise ValueError(
            f"Unsupported color mode: {mode!r}. "
            f"Supported modes: {sorted(_SUPPORTED_MODES)}")

    def process(self, rect, width, height, region, rotation_angle, output_buffer=None):
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
                raise ValueError(f"Invalid rect or pBits, cannot process. Rect type: {type(rect)}")

            pitch = int(rect.Pitch)
            src_address = pointer_to_address(rect.pBits)
            if not src_address:
                raise ValueError("Mapped rect does not contain a valid pointer")

            # As in `NumpyProcessor.process`, in both directions.
            check_surface_pitch(pitch, width)

            left, top, right, bottom = region
            if not (0 <= left < right <= width) or not (0 <= top < bottom <= height):
                raise ValueError(f"Region {region} is outside of the frame dimensions {(width, height)}")

            region_height = bottom - top
            region_width = right - left

            if output_buffer is None:
                output_buffer = self.cp.empty((region_height, region_width, 4), dtype=self.cp.uint8)
                is_pooled_buffer = False
            else:
                is_pooled_buffer = True
                if output_buffer.shape[:2] != (region_height, region_width) or output_buffer.shape[2] != 4:
                    raise ValueError(
                        f"Output buffer shape {output_buffer.shape} does not match region shape "
                        f"({region_height}, {region_width}, 4)."
                    )

            row_bytes = region_width * 4
            total_pitch_bytes = pitch * region_height
            src_buffer = (ctypes.c_ubyte * total_pitch_bytes).from_address(src_address + top * pitch)
            src_view = np.ctypeslib.as_array(src_buffer).reshape(region_height, pitch)

            if pitch == row_bytes and left == 0:
                cpu_region = src_view[:, :row_bytes]
            else:
                # One strided copy rather than a Python loop over rows. Still a
                # copy rather than a view, because `.set()` uploads from
                # contiguous host memory -- but `ascontiguousarray` does in one
                # vectorised pass what the loop did a row at a time, and
                # allocates the same single buffer `np.empty` did.
                #
                # The branch is reached whenever the pitch is padded *or* the
                # region's left edge is non-zero, so every region camera took
                # the loop regardless of pitch.
                start = left * 4
                cpu_region = np.ascontiguousarray(
                    src_view[:, start:start + row_bytes])

            cpu_region = cpu_region.reshape(region_height, region_width, 4)

            if hasattr(output_buffer, "set"):
                output_buffer.set(cpu_region)
            else:
                output_buffer[...] = cpu_region


            # Phase 2: Color Conversion and Rotation
            current_array = output_buffer # Start with the pooled buffer (already has BGRA data)
            is_still_pooled_buffer = is_pooled_buffer

            # Color Conversion — entirely on the device. The array stays a CuPy
            # array from here to the caller; there is no host round-trip.
            if self.color_mode is not None: # Not 'BGRA', so conversion is intended
                converted_array = self.process_cvtcolor(current_array)

                # Never copy the result back into the pooled staging buffer.
                #
                # Doing that for same-shape conversions (RGBA) made the frame
                # alias pooled storage the pool had already recycled: holding
                # six frames against a two-buffer pool yielded two distinct
                # allocations, and frame one had been overwritten by frame six
                # while the caller still held it. That is the frame-aliasing
                # corruption ROADMAP section 5 records being fixed once already
                # on the NumPy path, arriving here by another route -- and it
                # stays invisible until a consumer holds a frame for longer
                # than the pool depth.
                #
                # The converted array owns its storage, so returning it costs
                # one allocation and cannot alias anything.
                if converted_array.shape[:2] != current_array.shape[:2]:
                    logger.warning(
                        "CuPy color conversion changed height/width, which is "
                        "unexpected.")
                current_array = converted_array
                is_still_pooled_buffer = False
            
            # Rotation
            if rotation_angle != 0:
                k = (rotation_angle // 90) % 4
                if k != 0:
                    # `rot90` returns a *view*, so neither branch this used to
                    # take was safe on a rotated display.
                    #
                    # At 180 degrees the shape is unchanged, so the pooled
                    # buffer was assigned from a view of itself -- an
                    # overlapping device-to-device copy. The elementwise kernel
                    # reads and writes that memory at the same time, and the
                    # frame comes back torn.
                    #
                    # At 90 and 270 the shape differs, so the view was returned
                    # with the pooled flag cleared. `_grab()` reads that flag as
                    # "the buffer is free", checks it back in, and the caller is
                    # left holding a view of storage the next capture
                    # overwrites.
                    #
                    # Copy, as the NumPy path does: one allocation that owns its
                    # storage and aliases nothing. `.copy()` rather than
                    # `ascontiguousarray`, which hands back its argument
                    # unchanged when the view is already contiguous -- true of a
                    # 1x1 region, where both flips are no-ops.
                    #
                    # Clockwise (negated k), matching NumpyProcessor -- see the
                    # note there on DXGI_MODE_ROTATION_ROTATE90.
                    current_array = self.cp.rot90(current_array, k=-k).copy()
                    is_still_pooled_buffer = False

            return current_array, is_still_pooled_buffer

        except Exception as e:
            # Raise, do not return the buffer.
            #
            # This used to log the error, zero the buffer and hand it back. The
            # result was that `create(output_color="RGB", nvidia_gpu=True)`
            # returned a **4-channel BGRA array** and reported success whenever
            # conversion failed — wrong shape, wrong channel order, no
            # exception. A caller feeding that to a model got silent garbage,
            # and only a log line said otherwise.
            #
            # ROADMAP § 11: a fast wrong answer is worthless. A frame that
            # cannot be produced in the requested format is an error, not a
            # frame.
            logger.error(f"Error processing frame with CuPy: {e}")
            raise