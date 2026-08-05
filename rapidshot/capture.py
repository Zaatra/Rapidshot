import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock, current_thread
import comtypes  # type: ignore[import-untyped]
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from rapidshot.memory_pool import (
    NumpyMemoryPool,
    CupyMemoryPool,
    PooledBuffer,
    PoolExhaustedError,
)
from rapidshot.util.errors import ( # Added for Phase 2
    RapidShotError,
    RapidShotDXGIError,
    RapidShotReinitError,
    RapidShotDeviceError,
    RapidShotConfigError,
    RapidShotProtectedContentError,
)
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.core.stagesurf import StageSurface
from rapidshot.core.duplicator import Duplicator
from rapidshot._libs.d3d11 import D3D11_BOX
from rapidshot.processor import Processor
from rapidshot.util.ctypes_helpers import describe_destination
import collections # Added for deque
from rapidshot.util.timer import (
    create_high_resolution_timer,
    set_periodic_timer,
    wait_for_timer,
    cancel_timer,
    close_timer,
    INFINITE,
    WAIT_FAILED,
)

# Set up logger
logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp  # type: ignore[import-not-found]
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

class ScreenCapture:
    def __init__(
        self,
        output: Output,
        device: Device,
        region: Optional[Tuple[int, int, int, int]] = None,
        output_color: str = "RGB",
        nvidia_gpu: bool = False,
        max_buffer_len: int = 64, # This is for the continuous mode ring buffer
        pool_size_frames: int = 10, # New parameter for memory pool
        pool_output: bool = True,
    ) -> None:
        """
        Initialize a ScreenCapture instance.

        Args:
            output: Output device to capture from
            device: Device interface
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for continuous mode capture
            pool_size_frames: Number of buffers in the memory pool for grab()
            pool_output: Reuse buffers for the converted frame instead of
                allocating one per frame. Saves ~1.6 ms on a 1080p RGB frame --
                the page faults on first touch cost more than the conversion.
                ``grab()`` then returns a ``PooledBuffer``, which behaves like
                the array for indexing and ``np.asarray`` but **must** be
                released when done. Pass False for the pre-2.0 behaviour of
                returning a freshly allocated array that needs no release.
        """
        # Initialize basic attributes first to prevent errors during cleanup if initialization fails
        self._output = output
        self._device = device
        self._duplicator = None
        self._stagesurf = None
        self._processor = None
        self._pool_output = pool_output
        self._output_pool = None
        self._output_pool_size = pool_size_frames
        # Outstanding GPU frame handed out by grab_frame(). Capture cannot
        # acquire again until it is released, so this is tracked explicitly to
        # give a clear error instead of an opaque DXGI_ERROR_INVALID_CALL.
        self._live_frame = None
        self.is_capturing = False
        self._capture_thread = None 
        self._capture_lock = Lock() 
        self._stop_capture_event = Event() 
        self._frame_available_event = Event() 
        
        # For continuous mode buffer using PooledBuffer wrappers
        self._pooled_frames_deque: Optional[collections.deque] = None 
        self.max_buffer_len = max_buffer_len 

        self._timer_handle = None 
        self._frame_count = 0 
        self._capture_start_time = 0 
        self.rotation_angle = 0
        self.width = 0
        self.height = 0
        self.region = None
        self._region_set_by_user = False
        self._sourceRegion = None
        self.shot_w = 0
        self.shot_h = 0
        self.max_buffer_len = max_buffer_len
        self.continuous_mode = False
        self.buffer = False
        self._buffer_lock = Lock() 
        self.cursor = False
        self.memory_pool = None 
        
        # Phase 2: Re-initialization state variables
        self._is_initialized = False
        self._needs_reinit = False
        self._reinit_attempts = 0
        self._max_reinit_attempts = 5
        self._reinit_backoff_seconds = [0.5, 1.0, 2.0, 3.0, 5.0] # Or generate dynamically
        # Exclusive-fullscreen / mode-switch transitions can refuse duplication
        # for a few hundred ms. Bound the retries so a permanent refusal fails
        # loudly instead of spinning forever.
        self._max_output_change_retries = 12
        self._capture_permanently_failed = False
        self._last_capture_error_message = ""
        
        # "Nothing is arriving" is a question about elapsed time, not about how
        # many times we asked. Counting consecutive empty acquires made the
        # answer depend on `timeout_ms`: with the 10 ms default, 100 misses mean
        # a second of genuinely still screen, but with a polling timeout of 0 the
        # same 100 misses take under 20 ms. That fired the warning seven times
        # while capture was running at 117 fps -- a message that is not merely
        # noisy but the opposite of true.
        self._last_frame_time = None            # set on the first successful grab
        self._quiet_warning_after_s = 2.0
        self._last_quiet_warning = 0.0
        
        # Store initial constructor arguments for re-initialization
        self._init_args = {
            "output": output, # This is an object, direct use might be tricky if it becomes invalid
            "device": device, # Same as output
            "region": region, # Value type, safe
            "output_color": output_color, # Value type, safe
            "nvidia_gpu": nvidia_gpu, # Value type, safe
            "pool_size_frames": pool_size_frames # Value type, safe
        }
        # For re-creating device and output, we might need display_idx/output_idx if original objects become stale.
        # This part needs careful thought if Device/Output objects themselves can become invalid.
        # Assuming for now that the passed device/output objects are stable or re-creatable from stored indices.
        # Storing original indices if available from device/output objects:
        self._display_idx = device.display_idx if hasattr(device, 'display_idx') else 0 # Example
        self._output_idx = output.output_idx if hasattr(output, 'output_idx') else 0 # Example
        
        try:
            if not self._initialize_resources():
                # _initialize_resources logs errors, raise a generic one if it fails on first try
                raise RapidShotError("Initial resource initialization failed. Check logs for details.")

        except Exception as e: # Catch errors from _initialize_resources or other __init__ steps
            logger.error(f"Critical error during ScreenCapture __init__: {e}")
            # Ensure cleanup of any partially initialized resources
            self.release() # Call release to clean up whatever was set up
            raise # Re-raise the exception to signal construction failure
            
    def _initialize_resources(self, is_reinit=False) -> bool:
        """
        Initializes or re-initializes DXGI/D3D resources (Device, Output, Duplicator).
        Also re-initializes StageSurface, Processor, and MemoryPool if needed.
        """
        logger.info(f"{'Re-initializing' if is_reinit else 'Initializing'} capture resources...")
        
        # 1. Clean up existing resources (if any)
        if hasattr(self, '_duplicator') and self._duplicator:
            self._duplicator.release()
            self._duplicator = None
        if hasattr(self, '_stagesurf') and self._stagesurf:
            self._stagesurf.release()
            self._stagesurf = None
        # Device and Output are more complex. If they are passed in, re-getting them might be needed.
        # For now, assume self._device and self._output are either still valid or are re-created.
        # If they are from initial args, and can become stale, this needs more robust handling
        # (e.g. re-calling rapidshot.get_device, rapidshot.get_output based on stored indices).
        
        # For simplicity in this phase, let's assume self._device and self._output are either:
        # a) The initially provided valid objects (if not is_reinit)
        # b) Re-acquired if is_reinit (this part is complex if original handles are stale)
        # Let's simulate re-acquiring for reinit, assuming we have stored indices.
        if is_reinit:
            try:
                logger.debug(f"Re-creating device and output for display {self._display_idx}, output {self._output_idx}")
                # These get_device/get_output calls might not exist in this class directly.
                # This implies ScreenCapture needs access to the global factory functions.
                # For now, this is a placeholder for how Device/Output might be refreshed.
                # If the original device/output objects are stateful and become invalid,
                # they MUST be recreated.
                # Let's assume for now the stored self._device and self._output are updated externally or are robust.
                # If not, this is a major point of failure for re-initialization.
                # For now, we'll proceed assuming self._device and self._output are valid/refreshed.
                # This part of re-initialization (Device/Output) might need to live higher up,
                # e.g. in a factory that creates ScreenCapture, or ScreenCapture needs display_idx.
                
                # A pragmatic approach for now: if re-init, we trust the existing self._device, self._output
                # have been externally managed/updated or are somehow still valid for re-creating Duplicator.
                # This is a known simplification.
                self._output.update_desc() # Try to update the existing output object
                self.width, self.height = self._output.resolution
                logger.info(f"Output description updated. New resolution: {self.width}x{self.height}")

            except Exception as e:
                logger.error(f"Failed to re-acquire/update device/output during re-initialization: {e}")
                self._is_initialized = False
                return False

        try:
            # Use init_args for properties that don't change or are value types
            current_region = self._init_args['region']
            output_color = self._init_args['output_color']
            nvidia_gpu = self._init_args['nvidia_gpu'] # self.nvidia_gpu should be set from this
            pool_size_frames = self._init_args['pool_size_frames']

            self.nvidia_gpu = nvidia_gpu # Ensure it's set before processor/pool

            # Check if GPU acceleration is requested but CuPy is not available
            if self.nvidia_gpu and not CUPY_AVAILABLE:
                logger.warning("NVIDIA GPU acceleration requested but CuPy is not available. Falling back to CPU mode for re-init.")
                self.nvidia_gpu = False # Fallback for this attempt

            self.width, self.height = self._output.resolution # Get current resolution
            
            # Validate region against current width/height
            self._region_set_by_user = current_region is not None
            self.region = current_region
            if self.region is None:
                self.region = (0, 0, self.width, self.height)
            self._validate_region(self.region) # This updates self.region and shot_w, shot_h

            logger.debug(f"Creating Duplicator for output: {self._output.devicename}")
            self._duplicator = Duplicator(output=self._output, device=self._device)
            
            logger.debug(f"Creating StageSurface for output: {self._output.devicename}")
            self._stagesurf = StageSurface(output=self._output, device=self._device)
            
            logger.debug(f"Creating Processor with color: {output_color}, GPU: {self.nvidia_gpu}")
            self._processor = Processor(output_color=output_color, nvidia_gpu=self.nvidia_gpu)
            
            self._sourceRegion = D3D11_BOX(
                left=0, top=0, right=self.width, bottom=self.height, front=0, back=1
            )
            self.rotation_angle = self._output.rotation_angle
            self.output_color = output_color 

            # Re-initialize Memory Pool
            if self.memory_pool: # Destroy existing pool before creating a new one
                logger.debug("Destroying existing memory pool before re-initialization.")
                self.memory_pool.destroy_pool()
            
            region_height = self.region[3] - self.region[1]
            region_width = self.region[2] - self.region[0]
            buffer_shape = (region_height, region_width, 4) # BGRA
            dtype = np.uint8

            logger.debug(f"Initializing new memory pool with shape {buffer_shape}, {pool_size_frames} buffers.")
            if self.nvidia_gpu:
                self.memory_pool = CupyMemoryPool(buffer_shape, dtype, pool_size_frames)
            else:
                self.memory_pool = NumpyMemoryPool(buffer_shape, dtype, pool_size_frames)
            
            # If continuous capture was running, its buffer needs to be reset
            if self.is_capturing and self.continuous_mode:
                if self._pooled_frames_deque is not None:
                    logger.debug("Clearing continuous mode frame deque due to re-initialization.")
                    # Buffers must go back to the old pool before it is destroyed,
                    # otherwise their release() targets a pool that no longer exists.
                    with self._capture_lock:
                        stale_frames = list(self._pooled_frames_deque)
                        self._pooled_frames_deque = collections.deque(maxlen=self.max_buffer_len)
                    for frame in stale_frames:
                        self._discard_frame(frame)
                self._frame_available_event.clear()


            self._is_initialized = True
            self._needs_reinit = False # Successfully re-initialized (or initialized)
            if is_reinit: # Only reset attempts if this was a re-initialization
                self._reinit_attempts = 0
            logger.info("Capture resources successfully initialized.")
            return True

        except (RapidShotConfigError, RapidShotDeviceError, RapidShotDXGIError, RapidShotError) as e:
            logger.error(f"Failed to {'re-initialize' if is_reinit else 'initialize'} resources: {e}")
            self._is_initialized = False
            # self.release() # Clean up anything that might have been created
            return False
        except Exception as e: # Catch any other unexpected error
            logger.error(f"Unexpected error during resource {'re-initialization' if is_reinit else 'initialization'}: {e}")
            self._is_initialized = False
            # self.release()
            return False

    def _attempt_reinitialization(self) -> bool:
        """
        Attempts to re-initialize capture resources after a recoverable error.
        Manages retries and backoff periods.
        """
        if self._capture_permanently_failed:
            logger.warning("Re-initialization attempt skipped: Capture is permanently failed.")
            return False

        self._reinit_attempts += 1
        logger.warning(f"Re-initialization attempt {self._reinit_attempts} of {self._max_reinit_attempts} scheduled.")

        if self._reinit_attempts > self._max_reinit_attempts:
            self._capture_permanently_failed = True
            self._last_capture_error_message = f"Max re-initialization attempts ({self._max_reinit_attempts}) reached."
            logger.error(self._last_capture_error_message)
            return False

        backoff_idx = min(self._reinit_attempts - 1, len(self._reinit_backoff_seconds) - 1)
        wait_time = self._reinit_backoff_seconds[backoff_idx]
        logger.info(f"Waiting for {wait_time:.1f} seconds before re-initialization attempt...")
        time.sleep(wait_time)

        logger.info(f"Attempting re-initialization (attempt {self._reinit_attempts}/{self._max_reinit_attempts})...")
        if self._initialize_resources(is_reinit=True):
            logger.info("Re-initialization successful.")
            self._needs_reinit = False # Clear the flag as we succeeded
            return True
        else:
            logger.warning(f"Re-initialization attempt {self._reinit_attempts} failed.")
            # If this was the last attempt, mark as permanently failed
            if self._reinit_attempts == self._max_reinit_attempts:
                self._capture_permanently_failed = True
                self._last_capture_error_message = f"Re-initialization failed after {self._max_reinit_attempts} attempts."
                logger.error(self._last_capture_error_message)
            return False

    def region_to_memory_region(self, region: Tuple[int, int, int, int], rotation_angle: int, output: Output):
        """
        Convert a screen region to memory region based on rotation angle.
        
        Args:
            region: Region to convert (left, top, right, bottom)
            rotation_angle: Rotation angle (0, 90, 180, 270)
            output: Output device
            
        Returns:
            Converted region
        """
        left, top, right, bottom = region

        if rotation_angle != output.rotation_angle:
            raise AssertionError(
                f"Rotation mismatch: capture reports {rotation_angle} but output is {output.rotation_angle}"
            )

        surface_width, surface_height = output.surface_size

        if rotation_angle == 0:
            return (left, top, right, bottom)
        if rotation_angle == 90:
            return (top, surface_width - right, bottom, surface_width - left)
        if rotation_angle == 180:
            return (
                surface_width - right,
                surface_height - bottom,
                surface_width - left,
                surface_height - top,
            )
        if rotation_angle == 270:
            return (
                surface_height - bottom,
                left,
                surface_height - top,
                right,
            )

        raise ValueError(f"Invalid rotation angle: {rotation_angle}. Must be 0, 90, 180, or 270.")

    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Union[np.ndarray, Any]]: # Any can be PooledBuffer
        """
        Grab a single frame from the screen.
        Uses the memory pool if the requested region matches the pool's buffer configuration.
        
        Args:
            region: Region to capture (left, top, right, bottom). 
                    If None, uses self.region.
            
        Returns:
            A PooledBuffer wrapper (if pool was used and buffer is valid),
            a NumPy/CuPy array (if pool was bypassed or buffer became invalid), 
            or None if no update or error.
        """
        # Checked here rather than inside _grab(): that method has a catch-all
        # handler which would swallow this into a None return, hiding a caller
        # bug that stalls capture.
        self._ensure_no_live_frame("grab()")

        # Continuous mode grabbing is handled by __capture thread and get_latest_frame
        if self.is_capturing and self.continuous_mode:
             logger.warning("grab() called in continuous mode. Use get_latest_frame() instead.")
             return self.get_latest_frame() # Or return None, or raise error

        current_region_tuple: Tuple[int, int, int, int]
        if region is None:
            current_region_tuple = self.region
        else:
            current_region_tuple = self._normalize_region(region)

        return self._grab(current_region_tuple)

    def _checkout_output_buffer(self, width: int, height: int):
        """A pooled buffer for the converted frame, or None if not in use.

        On by default since 2.0. The buffer must be released by whoever receives
        it; ``rapidshot.create(pool_output=False)`` restores the pre-2.0
        behaviour of allocating a fresh array per frame, which costs ~1.6 ms on
        a 1080p RGB frame but needs no release.
        """
        if not self._pool_output or not self._processor.converts_output:
            return None

        shape = (height, width, self._processor.output_channels)
        pool = self._output_pool
        if pool is None or tuple(pool.buffer_shape) != shape:
            if pool is not None:
                pool.destroy_pool()
            from rapidshot.memory_pool import NumpyMemoryPool
            pool = NumpyMemoryPool(shape, np.uint8, self._output_pool_size)
            self._output_pool = pool
        try:
            return pool.checkout()
        except PoolExhaustedError:
            # Every buffer is still out with a caller. Fall back to allocating,
            # which is slower but always correct -- far better than blocking
            # capture or recycling a buffer somebody is still reading.
            logger.debug("Output pool exhausted; allocating for this frame.")
            return None

    def _sync_accumulator_region(self, memory_region) -> None:
        """Drop the accumulated frame when the captured region moves.

        Shape alone is not identity. Alternating between two same-sized regions
        would otherwise patch one region's dirty rects onto the other region's
        pixels — the accumulator would look the right size and hold a blend of
        two places on screen.
        """
        if getattr(self, "_accumulator_region", None) != memory_region:
            invalidate = getattr(self._processor, "invalidate_accumulator", None)
            if invalidate is not None:
                invalidate()
            self._accumulator_region = memory_region

    def _dirty_rects_for(self, memory_region) -> Optional[list]:
        """This frame's dirty rects, translated into the staging surface.

        DXGI reports them in desktop coordinates, but the staging surface holds
        only ``memory_region`` — ``CopySubresourceRegion`` already cropped it.
        So the rects have to be clipped to that region and rebased to its
        top-left, or they would address the wrong pixels whenever a region is in
        use, and outside the buffer entirely when it is off-origin.

        Returns None when the metadata is unavailable, which the processor
        reads as "convert everything".
        """
        rects = getattr(self._duplicator, "dirty_rects", None)
        if not rects:
            return rects if rects is None else []

        left, top, right, bottom = memory_region
        clipped = []
        for rl, rt, rr, rb in rects:
            nl, nt = max(rl, left), max(rt, top)
            nr, nb = min(rr, right), min(rb, bottom)
            if nl < nr and nt < nb:
                clipped.append((nl - left, nt - top, nr - left, nb - top))
        return clipped

    def _ensure_no_live_frame(self, caller: str) -> None:
        """
        Refuse to start a capture while a Frame still holds the desktop texture.

        DXGI would fail the acquire with DXGI_ERROR_INVALID_CALL, which gives no
        hint about the actual cause. Failing here names the problem instead.
        """
        live = self._live_frame
        if live is not None and not live.released:
            raise RuntimeError(
                f"{caller} cannot start: a Frame from grab_frame() has not been "
                "released. DXGI cannot acquire the next frame while a reference "
                "to the previous desktop surface is outstanding. Use `with "
                "camera.grab_frame() as frame:` so release happens automatically."
            )
        self._live_frame = None

    def grab_frame(self, region: Optional[Tuple[int, int, int, int]] = None):
        """
        Capture a frame and hand back its GPU texture, skipping the CPU copy.

        This is the GPU-resident path. Unlike :meth:`grab`, no staging read and
        no color conversion happen — the caller receives the ``ID3D11Texture2D``
        DXGI produced, ready to pass to a GPU consumer (inference runtime,
        hardware encoder).

        The returned :class:`~rapidshot.frame.Frame` owns that texture for a
        bounded window and **must be released**. DXGI cannot acquire the next
        frame while a reference to the previous surface is outstanding, so a
        frame that is never released stalls capture completely. Use it as a
        context manager::

            with camera.grab_frame() as frame:
                do_gpu_work(frame.d3d11_texture)

        Args:
            region: Region metadata for the frame (left, top, right, bottom).
                Note the texture is the full desktop surface; the region is
                recorded on the frame rather than applied to the texture, since
                cropping would require a GPU copy this path exists to avoid.

        Returns:
            A Frame, or None if no new content was available.

        Raises:
            RuntimeError: If a previous Frame has not been released yet.
        """
        from rapidshot.frame import Frame

        self._ensure_no_live_frame("grab_frame()")

        if region is None:
            region = self.region
        else:
            region = self._normalize_region(region)

        if self._capture_permanently_failed:
            logger.error(f"Capture permanently failed: {self._last_capture_error_message}")
            return None

        if self._needs_reinit and not self._attempt_reinitialization():
            return None

        if not self._is_initialized or self._duplicator is None:
            logger.error("grab_frame() called but capture resources are not initialized.")
            self._needs_reinit = True
            return None

        try:
            self._duplicator.update_frame()
        except RapidShotProtectedContentError as e:
            logger.error(f"Protected content blocks capture: {e}")
            self._last_capture_error_message = str(e)
            return None
        except (RapidShotReinitError, RapidShotDeviceError) as e:
            logger.warning(f"grab_frame(): {e}. Flagging for re-initialization.")
            self._needs_reinit = True
            return None
        except RapidShotError as e:
            logger.error(f"grab_frame(): {e}")
            return None

        if not self._duplicator.updated:
            # No new content. Any frame that was acquired still has to go back.
            if self._duplicator._frame_acquired:
                self._duplicator.release_frame()
            return None

        duplicator = self._duplicator
        frame = Frame(
            texture=duplicator.texture,
            on_release=duplicator.release_frame,
            region=region,
            rotation_angle=self.rotation_angle,
            present_time_qpc=duplicator.last_present_time,
            accumulated_frames=duplicator.accumulated_frames,
            protected_content=duplicator.protected_content_detected,
            cursor_visible=duplicator.cursor_visible,
            dirty_rects=duplicator.dirty_rects,
            rects_coalesced=duplicator.rects_coalesced,
        )
        self._live_frame = frame
        return frame

    def grab_cursor(self):
        """
        Get cursor information.
        
        Returns:
            Cursor information
        """
        return self._duplicator.cursor

    def shot(
        self,
        image_ptr: Any,
        region: Optional[Tuple[int, int, int, int]] = None,
        buffer_size: Optional[int] = None,
    ) -> bool:
        """
        Capture directly into a caller-provided memory buffer.

        The buffer receives pixels in this instance's ``output_color`` format,
        matching what :meth:`grab` returns. Size it as
        ``width * height * channels`` where ``channels`` is 4 for BGRA/RGBA,
        3 for RGB/BGR and 1 for GRAY -- :attr:`bytes_per_frame` computes this
        for you.

        Prefer passing a NumPy array (or any sized buffer object): its size is
        read directly and validated before anything is written. A bare pointer
        carries no size, so one is only accepted together with ``buffer_size``.

        Args:
            image_ptr: Destination buffer -- a NumPy array, ctypes array,
                bytearray/memoryview, or a raw pointer plus ``buffer_size``
            region: Region to capture (left, top, right, bottom)
            buffer_size: Destination size in bytes; required only for raw pointers

        Returns:
            True if a new frame was written, False if there was no new content
            or the capture failed

        Raises:
            ValueError: If the destination is too small or its size is unknowable
        """
        if image_ptr is None:
            raise ValueError("image_ptr cannot be None")

        if region is None:
            region = self.region
        else:
            self._validate_region(region)

        # Validate the destination up front, before any capture work. Deferring
        # this to the processor would make it fire only on the calls that
        # actually receive new frame content -- so an undersized buffer would
        # quietly return False on a static desktop and only raise later, once
        # something on screen happened to change.
        self._validate_destination(image_ptr, region, buffer_size)

        return self._shot(image_ptr, region, buffer_size)

    def _validate_destination(self, image_ptr, region, buffer_size) -> int:
        """
        Check that a shot() destination is large enough, before capturing.

        Args:
            image_ptr: Destination buffer or pointer
            region: Region that will be captured
            buffer_size: Caller-declared size in bytes, if any

        Returns:
            The required size in bytes

        Raises:
            ValueError: If the destination is too small, or its size is unknown
        """
        required = self.bytes_per_frame(region)
        _, detected_size = describe_destination(image_ptr)

        known_sizes = [s for s in (detected_size, buffer_size) if s is not None]
        if not known_sizes:
            raise ValueError(
                f"Cannot verify destination size for shot(): {region[2] - region[0]}x"
                f"{region[3] - region[1]} in {self.output_color} needs {required} "
                "bytes. Pass a NumPy array (or any sized buffer), or supply "
                "buffer_size explicitly -- writing through an unsized pointer "
                "risks corrupting memory."
            )

        smallest = min(known_sizes)
        if smallest < required:
            raise ValueError(
                f"Destination buffer is too small for shot(): {smallest} bytes "
                f"provided, {required} needed for {region[2] - region[0]}x"
                f"{region[3] - region[1]} in {self.output_color} "
                f"({self.channels} channel(s))."
            )
        return required

    @property
    def channels(self) -> int:
        """Number of channels frames from this instance carry."""
        return self._processor.output_channels

    def bytes_per_frame(self, region: Optional[Tuple[int, int, int, int]] = None) -> int:
        """
        Size in bytes that :meth:`shot` writes for *region*.

        Use this to allocate a destination buffer that is guaranteed to fit.

        Args:
            region: Region to measure; defaults to this instance's region

        Returns:
            Required buffer size in bytes
        """
        if region is None:
            region = self.region
        else:
            region = self._normalize_region(region)
        width = region[2] - region[0]
        height = region[3] - region[1]
        return width * height * self.channels

    def _shot(
        self,
        image_ptr,
        region: Tuple[int, int, int, int],
        buffer_size: Optional[int] = None,
    ) -> bool:
        """
        Internal implementation of shot.

        Args:
            image_ptr: Destination buffer or pointer to one
            region: Region to capture (left, top, right, bottom)
            buffer_size: Destination size in bytes, if known

        Returns:
            True if successful, False otherwise
        """
        self._ensure_no_live_frame("shot()")

        try:
            duplication_healthy = self._duplicator.update_frame()
        except (RapidShotReinitError, RapidShotDeviceError) as e:
            # Access lost / device reset: rebuild, then let the caller retry.
            logger.warning(f"shot(): {e}. Rebuilding capture resources.")
            self._on_output_change()
            return False

        if duplication_healthy:
            frame_needs_release = self._duplicator._frame_acquired
            mapped_rect = None
            try:
                if not self._duplicator.updated:
                    # No new content within the acquire timeout. That is the
                    # normal state of a static desktop, not an output change --
                    # rebuilding here used to stall every idle shot() call.
                    return False

                _region = self.region_to_memory_region(region, self.rotation_angle, self._output)
                _width = _region[2] - _region[0]
                _height = _region[3] - _region[1]

                if self._stagesurf.width != _width or self._stagesurf.height != _height:
                    self._stagesurf.release()
                    self._stagesurf.rebuild(output=self._output, device=self._device, dim=(_width, _height))

                source_region = D3D11_BOX(
                    left=_region[0],
                    top=_region[1],
                    right=_region[2],
                    bottom=_region[3],
                    front=0,
                    back=1,
                )

                self._device.im_context.CopySubresourceRegion(
                    self._stagesurf.texture,
                    0,
                    0,
                    0,
                    0,
                    self._duplicator.texture,
                    0,
                    ctypes.byref(source_region),
                )

                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                    frame_needs_release = False

                mapped_rect = self._stagesurf.map()
                try:
                    self._processor.process2(
                        image_ptr,
                        mapped_rect,
                        _width,
                        _height,
                        buffer_size,
                    )
                finally:
                    self._stagesurf.unmap()
                return True
            finally:
                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
        else:
            self._on_output_change()
            return False

    def _grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Grab a frame with a specific region with improved error handling.

        Args:
            region: Region to capture (left, top, right, bottom)

        Returns:
            ndarray: Captured frame
        """
        try:
            if self._capture_permanently_failed:
                logger.error(f"Capture is permanently failed: {self._last_capture_error_message}")
                return None

            if self._needs_reinit:
                if not self._attempt_reinitialization():
                    logger.error(
                        "Re-initialization failed, current grab cannot proceed. "
                        f"Permanent failure: {self._capture_permanently_failed}"
                    )
                    return None

            if not self._is_initialized or self._duplicator is None:
                logger.error("Attempted to grab frame but capture resources are not initialized.")
                self._needs_reinit = True
                return None

            pooled_buffer_wrapper = None
            output_wrapper = None
            output_array_for_region = None
            can_use_pool = False

            if self.memory_pool:
                region_h = region[3] - region[1]
                region_w = region[2] - region[0]
                if (
                    self.memory_pool.buffer_shape[0] == region_h
                    and self.memory_pool.buffer_shape[1] == region_w
                    and self.memory_pool.buffer_shape[2] == 4
                ):
                    can_use_pool = True

            if can_use_pool:
                pooled_buffer_wrapper = self.memory_pool.checkout()
                output_array_for_region = pooled_buffer_wrapper.array
            else:
                logger.debug(
                    f"Region {region} not matching pool config. Using temporary buffer for this grab."
                )
                temp_region_h = region[3] - region[1]
                temp_region_w = region[2] - region[0]
                temp_shape = (temp_region_h, temp_region_w, 4)
                if self.nvidia_gpu:
                    output_array_for_region = cp.empty(temp_shape, dtype=cp.uint8)
                else:
                    output_array_for_region = np.empty(temp_shape, dtype=np.uint8)

            try:
                self._duplicator.update_frame()
            except RapidShotReinitError as e:
                logger.warning(f"DXGI Re-init error during update_frame: {e}. Flagging for re-initialization.")
                self._needs_reinit = True
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotDeviceError as e:
                logger.error(f"DXGI Device error during update_frame: {e}. Flagging for re-initialization.")
                self._needs_reinit = True
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotProtectedContentError as e:
                # Not recoverable by retrying: the OS is refusing while the
                # protected surface is on screen. Do not enter the re-init loop.
                logger.error(f"Protected content blocks capture: {e}")
                self._last_capture_error_message = str(e)
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotDXGIError as e:
                logger.error(f"DXGI error during update_frame: {e}")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None
            except RapidShotError as e:
                logger.error(f"RapidShot error during update_frame: {e}")
                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            now = time.perf_counter()
            if self._duplicator.updated:
                self._last_frame_time = now
            else:
                # Warn about a screen that has actually been still for a while,
                # and say so in seconds. A run of empty acquires is normal at any
                # timeout and says nothing on its own.
                if self._last_frame_time is None:
                    self._last_frame_time = now
                quiet_for = now - self._last_frame_time
                if (quiet_for >= self._quiet_warning_after_s
                        and now - self._last_quiet_warning >= self._quiet_warning_after_s):
                    logger.warning(
                        f"No screen updates for {quiet_for:.1f}s. Desktop "
                        "Duplication only reports changed content, so a still "
                        "screen produces no frames by design."
                    )
                    self._last_quiet_warning = now

                if self._duplicator._frame_acquired:
                    self._duplicator.release_frame()

                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            frame_needs_release = self._duplicator._frame_acquired
            mapped_rect = None
            try:
                memory_region = self.region_to_memory_region(region, self.rotation_angle, self._output)
                region_width = memory_region[2] - memory_region[0]
                region_height = memory_region[3] - memory_region[1]

                if (
                    self._stagesurf.width != region_width
                    or self._stagesurf.height != region_height
                ):
                    self._stagesurf.release()
                    self._stagesurf.rebuild(
                        output=self._output,
                        device=self._device,
                        dim=(region_width, region_height),
                    )

                source_region = D3D11_BOX(
                    left=memory_region[0],
                    top=memory_region[1],
                    right=memory_region[2],
                    bottom=memory_region[3],
                    front=0,
                    back=1,
                )

                self._device.im_context.CopySubresourceRegion(
                    self._stagesurf.texture,
                    0,
                    0,
                    0,
                    0,
                    self._duplicator.texture,
                    0,
                    ctypes.byref(source_region),
                )

                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()
                    frame_needs_release = False

                self._sync_accumulator_region(memory_region)
                output_wrapper = self._checkout_output_buffer(region_width, region_height)
                mapped_rect = self._stagesurf.map()
                final_array, is_pooled_buffer_still_valid = self._processor.process(
                    mapped_rect,
                    region_width,
                    region_height,
                    (0, 0, region_width, region_height),
                    self.rotation_angle,
                    output_array_for_region,
                    dirty_rects=self._dirty_rects_for(memory_region),
                    output_target=None if output_wrapper is None else output_wrapper.array,
                )
            finally:
                if mapped_rect is not None:
                    self._stagesurf.unmap()
                if frame_needs_release and self._duplicator._frame_acquired:
                    self._duplicator.release_frame()

            if output_wrapper is not None:
                # The converted frame went into a pooled output buffer. The BGRA
                # staging buffer is finished with either way; the output buffer
                # is the caller's until they release it.
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                if is_pooled_buffer_still_valid and final_array is output_wrapper.array:
                    return output_wrapper
                # The processor declined the target (rotation, or a shape it
                # would not write). Hand the buffer straight back rather than
                # leaking it, and return whatever was actually produced.
                output_wrapper.release()
                return final_array

            if can_use_pool and pooled_buffer_wrapper:
                if is_pooled_buffer_still_valid:
                    return pooled_buffer_wrapper
                pooled_buffer_wrapper.release()
                return final_array

            return final_array

        except PoolExhaustedError:
            logger.warning("Memory pool exhausted during grab. Consider increasing pool_size_frames.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in _grab: {e}")
            import traceback

            logger.error(traceback.format_exc())
            if 'pooled_buffer_wrapper' in locals() and pooled_buffer_wrapper:
                try:
                    pooled_buffer_wrapper.release()
                except Exception as rel_e:
                    logger.error(f"Error releasing buffer during exception handling in _grab: {rel_e}")
            self._needs_reinit = True
            self._last_capture_error_message = f"Unexpected error in _grab: {str(e)}"
            return None

    def _on_output_change(self) -> bool:
        """
        Rebuild duplication after a display mode change or access loss.

        This is the exclusive-fullscreen path: when a game takes or releases
        exclusive fullscreen, DXGI invalidates the duplication object and
        refuses to hand out a new one until the mode switch settles. Retrying
        immediately in a tight loop (the previous behaviour) either spins the
        CPU or hangs the caller forever, and the surviving stale stage surface
        is what surfaced as the "black screen in fullscreen" symptom.

        Returns:
            True if duplication was rebuilt, False if it could not be within
            the retry budget (caller should treat capture as degraded).
        """
        time.sleep(0.1)  # Wait for Display mode change (Access Lost)

        if self._duplicator is not None:
            self._duplicator.release()
            self._duplicator = None
        if self._stagesurf is not None:
            # Must be released, not just rebuilt: after a mode switch the old
            # staging texture is still sized for the previous resolution, and
            # StageSurface.rebuild() keeps an existing texture as-is.
            self._stagesurf.release()

        self._output.update_desc()
        self.width, self.height = self._output.resolution
        if self.region is None or not self._region_set_by_user:
            self.region = (0, 0, self.width, self.height)
        self._validate_region(self.region)
        self.rotation_angle = self._output.rotation_angle
        if self.is_capturing:
            self._rebuild_frame_buffer(self.region)

        for attempt in range(self._max_output_change_retries):
            try:
                self._stagesurf.rebuild(output=self._output, device=self._device)
                self._duplicator = Duplicator(output=self._output, device=self._device)
                logger.info(
                    f"Duplication rebuilt after output change "
                    f"(attempt {attempt + 1}, resolution {self.width}x{self.height})."
                )
                return True
            except RapidShotProtectedContentError as e:
                # Retrying cannot help while the protected surface is on screen.
                logger.error(f"Cannot rebuild duplication: {e}")
                self._last_capture_error_message = str(e)
                return False
            except (comtypes.COMError, RapidShotError) as e:
                # DXGI commonly reports UNSUPPORTED/ACCESS_DENIED for a short
                # window while the mode switch is in flight. Back off instead of
                # busy-waiting, and give up rather than hang if it persists.
                wait = min(0.05 * (2 ** attempt), 1.0)
                logger.debug(
                    f"Duplication rebuild attempt {attempt + 1} failed ({e}); "
                    f"retrying in {wait:.2f}s."
                )
                time.sleep(wait)

        self._last_capture_error_message = (
            f"Failed to rebuild duplication after {self._max_output_change_retries} "
            "attempts following an output change."
        )
        logger.error(self._last_capture_error_message)
        self._needs_reinit = True
        return False

    def start(
        self,
        region: Optional[Tuple[int, int, int, int]] = None,
        target_fps: int = 60,
        video_mode: bool = False,
        delay: int = 0,
    ):
        """
        Start capturing frames.

        Args:
            region: Region to capture (left, top, right, bottom)
            target_fps: Target frame rate
            video_mode: Whether to operate in video mode
            delay: Delay before starting capture (ms)
        """
        if self.is_capturing:
            logger.debug("start() called while capture is already active; ignoring request.")
            return

        if delay != 0:
            time.sleep(delay)
            self._on_output_change()
        if region is None:
            region = self.region
        self._validate_region(region)
        self.is_capturing = True
        
        # Phase 4: Initialize deque for continuous mode
        self._pooled_frames_deque = collections.deque(maxlen=self.max_buffer_len)
        self._frame_available_event.clear() # Clear before starting
        self._stop_capture_event.clear()

        # region is already validated and is self.region if None was passed
        # The capture thread will use self.region by default if grab is called with None
        
        self._capture_thread = Thread( # Renamed from self.__thread
            target=self._capture_thread_func, # Renamed from __capture
            name="ScreenCaptureThread", # More descriptive name
            args=(region, target_fps, video_mode),
        )
        self._capture_thread.daemon = True
        self._capture_thread.start()

    def stop(self):
        """
        Stop capturing frames.
        """
        if getattr(self, 'is_capturing', False):
            self._stop_capture_event.set() # Use renamed event
            if getattr(self, '_capture_thread', None) is not None:
                if current_thread() is not self._capture_thread:
                    self._capture_thread.join(timeout=10) # Wait for thread to finish
                self._capture_thread = None

        self.is_capturing = False
        self._frame_count = 0
        self._frame_available_event.clear()
        # self._stop_capture_event is already set, clear if restartable, but usually not needed

        if self._timer_handle:
            try:
                cancel_timer(self._timer_handle)
            except Exception as timer_error:
                logger.warning(f"Failed to cancel timer during stop(): {timer_error}")
            finally:
                try:
                    close_timer(self._timer_handle)
                except Exception as close_error:
                    logger.warning(f"Failed to close timer handle during stop(): {close_error}")
                self._timer_handle = None
        
        # Phase 4/5: Release any remaining buffers in the deque
        if hasattr(self, '_pooled_frames_deque') and self._pooled_frames_deque is not None:
            with self._capture_lock:
                # Copy then clear under the lock; the actual pool check-ins
                # happen outside it so a slow pool cannot block the producer.
                temp_deque_copy = list(self._pooled_frames_deque)
                self._pooled_frames_deque.clear()
            for buffer_wrapper in temp_deque_copy:
                self._discard_frame(buffer_wrapper)
            self._pooled_frames_deque = None
        
    def get_latest_frame(self, as_numpy: bool = True):
        """
        Get the latest captured frame.
        
        Args:
            as_numpy: If True, always return NumPy array even when using GPU acceleration.
                     If False and using GPU acceleration, return CuPy array for better performance.
        
        Returns:
            Latest captured frame as numpy or cupy array
        """
        # Phase 4: Get from deque
        if not self._frame_available_event.wait(timeout=1.0): # Wait for a short duration
            logger.debug("get_latest_frame timed out waiting for frame_available_event.")
            return None # No frame available or timeout
        
        with self._capture_lock: # Protect access to deque
            if not self._pooled_frames_deque:
                self._frame_available_event.clear() # Clear if deque is empty after wait
                return None

            # Get the most recent frame (without removing it). Entries are
            # PooledBuffer wrappers for BGRA output and plain arrays otherwise.
            frame_array = self._frame_array(self._pooled_frames_deque[-1])

            # self._frame_available_event.clear() # Do not clear here, new frames might arrive.
            # Event should be cleared only if no frames are in buffer after waiting.
            # Or, it's a signal that *at least one* frame is ready.

        # Convert to numpy if requested and if data is on GPU
        if self.nvidia_gpu and CUPY_AVAILABLE and isinstance(frame_array, cp.ndarray):
            if as_numpy:
                return cp.asnumpy(frame_array)
            else:
                return frame_array # Return CuPy array directly
        elif isinstance(frame_array, np.ndarray): # Already a NumPy array
            return frame_array
        else: # Should not happen if pool stores np or cp arrays
            logger.error(f"Unexpected array type in deque: {type(frame_array)}")
            return None

    def _capture_thread_func( # Renamed from __capture
        self, region: Tuple[int, int, int, int], target_fps: int = 60, video_mode: bool = False
    ):
        """
        Internal capture thread implementation for continuous mode.
        
        Args:
            region: Region to capture (left, top, right, bottom). This is the default region.
            target_fps: Target frame rate.
            video_mode: Whether to operate in video mode (duplicates last frame if no new one).
        """
        if target_fps > 0: # Allow target_fps = 0 for max speed
            period_ms = 1000 // target_fps
            self._timer_handle = create_high_resolution_timer()
            set_periodic_timer(self._timer_handle, period_ms)
        else: # Running at max speed
            self._timer_handle = None


        self._capture_start_time = time.perf_counter()
        capture_error = None
        last_successful_pooled_buffer = None # For video_mode duplication

        while not self._stop_capture_event.is_set():
            if self._timer_handle:
                res = wait_for_timer(self._timer_handle, INFINITE)
                if res == WAIT_FAILED: # Timer error
                    self._stop_capture_event.set()
                    capture_error = ctypes.WinError()
                    logger.error(f"High-resolution timer wait failed: {capture_error}")
                    continue
            
            grab_result = None
            try:
                if self._capture_permanently_failed: # Check before each grab attempt in loop
                    logger.error(f"Capture permanently failed. Stopping capture thread. Last error: {self._last_capture_error_message}")
                    self._stop_capture_event.set() # Signal thread to stop
                    break # Exit while loop

                # Use self.region for continuous capture, which was set during start()
                # _grab will handle _needs_reinit flag internally.
                grab_result = self._grab(self.region) 

                if grab_result is not None:
                    self._frame_count += 1
                    # grab_result is a PooledBuffer only when the processor could
                    # write the result in place, i.e. BGRA output. Every other
                    # color mode changes the channel count and yields a freshly
                    # allocated array -- which is equally valid to queue, and
                    # rejecting it (as this used to) left the deque permanently
                    # empty for RGB/BGR/RGBA/GRAY consumers.
                    evicted_buffer = None
                    with self._capture_lock:
                        if len(self._pooled_frames_deque) == self.max_buffer_len:
                            evicted_buffer = self._pooled_frames_deque[0]
                        self._pooled_frames_deque.append(grab_result)
                        last_successful_pooled_buffer = grab_result
                    # Check the evicted buffer back in outside the capture
                    # lock: release() takes the pool's own lock, and holding
                    # both here stalls every get_latest_frame() consumer for
                    # the duration of the pool round-trip.
                    self._discard_frame(evicted_buffer)
                    self._frame_available_event.set()
                
                elif self._needs_reinit: # _grab returned None and might have set _needs_reinit
                    logger.info("Continuous mode: Grab failed, re-initialization pending or in progress.")
                    # Optional: Short sleep before next attempt if re-init is happening via _grab
                    time.sleep(0.1) # Avoid tight loop if _grab keeps failing due to re-init
                    continue # Try again, _grab will attempt re-init

                elif video_mode and last_successful_pooled_buffer is not None:
                    # No new content this tick: re-queue a copy of the last frame
                    # so the output stream keeps a constant frame rate.
                    duplicate_frame = None
                    try:
                        source_array = self._frame_array(last_successful_pooled_buffer)
                        if isinstance(last_successful_pooled_buffer, PooledBuffer):
                            if self.memory_pool is None:
                                logger.warning("Video_mode: Memory pool not available for duplicating frame.")
                                raise PoolExhaustedError("no pool")
                            duplicate_frame = self.memory_pool.checkout()
                            if self.nvidia_gpu: # cp array
                                duplicate_frame.array[:] = source_array
                            else: # np array
                                np.copyto(duplicate_frame.array, source_array)
                        else:
                            # Plain array (non-BGRA output): copy directly, the
                            # pool's BGRA buffers are the wrong shape for it.
                            duplicate_frame = source_array.copy()

                        evicted_buffer = None
                        with self._capture_lock:
                            if len(self._pooled_frames_deque) == self.max_buffer_len:
                                evicted_buffer = self._pooled_frames_deque[0]
                            self._pooled_frames_deque.append(duplicate_frame)
                        self._discard_frame(evicted_buffer)  # Outside the lock, see above
                        self._frame_available_event.set()
                        self._frame_count += 1
                    except PoolExhaustedError:
                        logger.warning("Video_mode: Pool exhausted, cannot duplicate frame.")
                        self._discard_frame(duplicate_frame)
                    except Exception as dup_e:
                        logger.error(f"Video_mode: Error duplicating frame: {dup_e}")
                        self._discard_frame(duplicate_frame)
            
            except RapidShotReinitError as e: # Should be caught by _grab now
                logger.warning(f"Capture thread: Re-init error caught: {e}. _needs_reinit should be True.")
            except RapidShotDeviceError as e: # Should be caught by _grab now
                logger.error(f"Capture thread: Device error caught: {e}. _needs_reinit should be True.")
            except Exception as e: 
                import traceback
                logger.error(f"Error in capture thread: {e}\n{traceback.format_exc()}")
                self._last_capture_error_message = f"Runtime error in capture thread: {str(e)}"
                self._capture_permanently_failed = True # Assume critical error
                self._stop_capture_event.set() 
                capture_error = e
                
        # Clean up timer
        if self._timer_handle:
            try:
                cancel_timer(self._timer_handle)
            except Exception as timer_error:
                logger.warning(f"Failed to cancel capture timer: {timer_error}")
            finally:
                try:
                    close_timer(self._timer_handle)
                except Exception as close_error:
                    logger.warning(f"Failed to close capture timer handle: {close_error}")
                self._timer_handle = None
        
        if capture_error is not None or self._capture_permanently_failed:
            logger.error(f"Capture thread terminated. Error: {capture_error}. Permanent failure: {self._capture_permanently_failed}. Last message: {self._last_capture_error_message}")
            
        capture_duration = time.perf_counter() - self._capture_start_time
        if capture_duration > 0 and self._frame_count > 0: 
            actual_fps = self._frame_count / capture_duration
            logger.info(f"ScreenCapture continuous mode stopped. Captured {self._frame_count} frames in {capture_duration:.2f}s (FPS: {actual_fps:.2f}).")
        else:
            logger.info(f"ScreenCapture continuous mode stopped. No frames captured or capture time was zero.")

    @staticmethod
    def _discard_frame(frame) -> None:
        """
        Drop a queued frame, returning it to the pool if it came from one.

        The continuous-mode deque holds PooledBuffer wrappers for BGRA output
        and plain arrays for every other color mode, so callers must not assume
        a ``release()`` method exists.
        """
        if frame is None:
            return
        release = getattr(frame, "release", None)
        if release is None:
            return  # Plain array: ordinary garbage collection owns it
        try:
            release()
        except Exception as e:
            logger.debug(f"Ignoring error releasing pooled frame: {e}")

    @staticmethod
    def _frame_array(frame):
        """Return the underlying array for a queued frame (pooled or plain)."""
        return getattr(frame, "array", frame)

    def _rebuild_frame_buffer(self, region: Tuple[int, int, int, int]):
        """
        Rebuild the continuous-mode frame buffer after a resolution change.

        Drops every buffer still queued (they are sized for the old resolution)
        and rebuilds the memory pool to the new region shape.

        Args:
            region: Region to capture (left, top, right, bottom)
        """
        if region is None:
            region = self.region

        frame_shape = (region[3] - region[1], region[2] - region[0], 4)  # BGRA

        # Return queued buffers to the pool before it is torn down, otherwise
        # the wrappers outlive their pool and their release() targets a dead one.
        with self._capture_lock:
            stale_buffers = list(self._pooled_frames_deque or ())
            if self._pooled_frames_deque is not None:
                self._pooled_frames_deque.clear()
        for buffer_wrapper in stale_buffers:
            self._discard_frame(buffer_wrapper)
        self._frame_available_event.clear()

        pool_size = self._init_args.get("pool_size_frames", 10)
        if self.memory_pool is not None:
            if tuple(self.memory_pool.buffer_shape) == frame_shape:
                return  # Shape unchanged, existing pool is still correct
            self.memory_pool.destroy_pool()
            self.memory_pool = None

        logger.debug(f"Rebuilding memory pool for new frame shape {frame_shape}.")
        if self.nvidia_gpu and CUPY_AVAILABLE:
            self.memory_pool = CupyMemoryPool(frame_shape, np.uint8, pool_size)
        else:
            self.memory_pool = NumpyMemoryPool(frame_shape, np.uint8, pool_size)

    def _normalize_region(self, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Validate *region* without mutating capture state."""
        if region is None:
            raise ValueError("Region cannot be None")

        if not hasattr(self, 'width') or not hasattr(self, 'height'):
            raise ValueError("Capture dimensions are not initialized")

        try:
            l, t, r, b = map(int, region)
        except (TypeError, ValueError) as conversion_error:
            raise ValueError(f"Region must be a tuple of four integers: {conversion_error}") from conversion_error

        if l < 0 or t < 0:
            raise ValueError(f"Region start ({l}, {t}) is outside the capture bounds (0, 0)")

        if r > self.width or b > self.height:
            raise ValueError(
                f"Region end ({r}, {b}) exceeds capture bounds ({self.width}, {self.height})"
            )

        if l >= r or t >= b:
            raise ValueError(f"Region coordinates must form a positive area, got {region}")

        return (l, t, r, b)

    def _validate_region(self, region: Tuple[int, int, int, int]):
        """
        Validate region coordinates.

        Args:
            region: Region to validate (left, top, right, bottom)

        Raises:
            ValueError: If region is invalid
        """
        validated_region = self._normalize_region(region)
        l, t, r, b = validated_region
        self.region = validated_region

        if hasattr(self, '_sourceRegion') and self._sourceRegion is not None:
            self._sourceRegion.left = l
            self._sourceRegion.top = t
            self._sourceRegion.right = r
            self._sourceRegion.bottom = b

        self.shot_w, self.shot_h = r - l, b - t

    def release(self):
        """
        Release all resources.
        """
        try:
            if hasattr(self, 'is_capturing') and self.is_capturing: # Check is_capturing before calling stop
                self.stop()

            # A Frame still holding the desktop texture would keep DXGI's
            # surface pinned past the duplicator's own teardown.
            live = getattr(self, '_live_frame', None)
            if live is not None and not live.released:
                live.release()
            self._live_frame = None

            if hasattr(self, '_duplicator') and self._duplicator:
                self._duplicator.release()
                
            if hasattr(self, '_stagesurf') and self._stagesurf:
                self._stagesurf.release()

            # Phase 5: Destroy memory pool
            if hasattr(self, 'memory_pool') and self.memory_pool:
                logger.info("Destroying memory pool.")
                self.memory_pool.destroy_pool()
                self.memory_pool = None

        except Exception as e:
            logger.warning(f"Error during release: {e}")

    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        try:
            self.release()
        except Exception as e:
            logger.warning(f"Error during destruction: {e}")

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation of the ScreenCapture instance
        """
        try:
            return "<{}:\n\t{},\n\t{},\n\t{},\n\t{}\n>".format(
                "ScreenCapture",
                self._device if hasattr(self, '_device') else "No device",
                self._output if hasattr(self, '_output') else "No output",
                self._stagesurf if hasattr(self, '_stagesurf') else "No stage surface",
                self._duplicator if hasattr(self, '_duplicator') else "No duplicator",
            )
        except Exception:
            return "<ScreenCapture: initialization incomplete>"
