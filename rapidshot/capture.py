import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock
import comtypes
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from rapidshot.memory_pool import NumpyMemoryPool, CupyMemoryPool, PoolExhaustedError
from rapidshot.util.errors import ( # Added for Phase 2
    RapidShotError,
    RapidShotDXGIError,
    RapidShotReinitError,
    RapidShotDeviceError,
    RapidShotConfigError
)
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.core.stagesurf import StageSurface
from rapidshot.core.duplicator import Duplicator
from rapidshot._libs.d3d11 import D3D11_BOX
from rapidshot.processor import Processor
import collections # Added for deque
from rapidshot.util.timer import (
    create_high_resolution_timer,
    set_periodic_timer,
    wait_for_timer,
    cancel_timer,
    INFINITE,
    WAIT_FAILED,
)

# Set up logger
logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
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
        """
        # Initialize basic attributes first to prevent errors during cleanup if initialization fails
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
        self._capture_permanently_failed = False
        self._last_capture_error_message = ""
        
        # For timeout warnings (Phase 1 of timeout handling)
        self._consecutive_timeouts = 0
        self._timeout_warning_threshold = 100 # Warn after this many consecutive timeouts
        
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
                    # Buffers should be released by memory_pool.destroy_pool if they were checked out.
                    # If they are still in the deque, they need explicit release.
                    with self._capture_lock:
                        while self._pooled_frames_deque:
                            buf_wrapper = self._pooled_frames_deque.popleft()
                            buf_wrapper.release() # Ensure they are returned to the (old) pool before it's gone
                    self._pooled_frames_deque = collections.deque(maxlen=self.max_buffer_len)
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
        # Extract region coordinates
        left, top, right, bottom = region
        
        # Get surface dimensions
        width, height = output.surface_size
        
        # Convert based on rotation angle
        if rotation_angle == 0:
            # No rotation
            return (left, top, right, bottom)
        elif rotation_angle == 90:
            # 90-degree rotation (clockwise)
            # In 90-degree rotation, x becomes y, and y becomes (width - x)
            return (top, width - right, bottom, width - left)
        elif rotation_angle == 180:
            # 180-degree rotation
            # In 180-degree rotation, x becomes (width - x), and y becomes (height - y)
            return (width - right, height - bottom, width - left, height - top)
        elif rotation_angle == 270:
            # 270-degree rotation (clockwise)
            # In 270-degree rotation, x becomes (height - y), and y becomes x
            return (height - bottom, left, height - top, right)
        else:
            # Invalid rotation angle
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
        # Continuous mode grabbing is handled by __capture thread and get_latest_frame
        if self.is_capturing and self.continuous_mode:
             logger.warning("grab() called in continuous mode. Use get_latest_frame() instead.")
             return self.get_latest_frame() # Or return None, or raise error

        current_region_tuple: Tuple[int, int, int, int]
        if region is None:
            current_region_tuple = self.region
        else:
            # Validate and potentially update self.region if this new region should be default
            # For now, just use it as a temporary region for this grab.
            # _validate_region updates self.region, which might not be desired for one-off grabs.
            # Create a validated tuple without altering self.region.
            l, t, r, b = region
            l = max(0, min(l, self.width - 1))
            t = max(0, min(t, self.height - 1))
            r = max(l + 1, min(r, self.width))
            b = max(t + 1, min(b, self.height))
            current_region_tuple = (l, t, r, b)
            
        return self._grab(current_region_tuple)

    def grab_cursor(self):
        """
        Get cursor information.
        
        Returns:
            Cursor information
        """
        return self._duplicator.cursor

    def shot(self, image_ptr: Any, region: Optional[Tuple[int, int, int, int]] = None) -> bool:
        """
        Capture directly to a provided memory buffer.
        
        Args:
            image_ptr: Pointer to image buffer (must be properly sized for the region)
            region: Region to capture (left, top, right, bottom)
            
        Returns:
            True if successful, False otherwise
        """
        if image_ptr is None:
            raise ValueError("image_ptr cannot be None")
            
        if region is None:
            region = self.region
        else:
            self._validate_region(region)
            
        return self._shot(image_ptr, region)

    def _shot(self, image_ptr, region: Tuple[int, int, int, int]) -> bool:
        """
        Internal implementation of shot.
        
        Args:
            image_ptr: Pointer to image buffer
            region: Region to capture (left, top, right, bottom)
            
        Returns:
            True if successful, False otherwise
        """
        if self._duplicator.update_frame():
            if not self._duplicator.updated:
                return False

            _region = self.region_to_memory_region(region, self.rotation_angle, self._output)
            _width = _region[2] - _region[0]
            _height = _region[3] - _region[1]

            if self._stagesurf.width != _width or self._stagesurf.height != _height:
                self._stagesurf.release()
                self._stagesurf.rebuild(output=self._output, device=self._device, dim=(_width, _height))

            # Create a source-specific region object with the transformed coordinates
            source_region = D3D11_BOX(
                left=_region[0], top=_region[1], right=_region[2], bottom=_region[3], front=0, back=1
            )

            # Copy with region support
            self._device.im_context.CopySubresourceRegion(
                self._stagesurf.texture, 0, 0, 0, 0, self._duplicator.texture, 0, ctypes.byref(source_region)
            )
            self._duplicator.release_frame()
            rect = self._stagesurf.map()
            self._processor.process2(image_ptr, rect, self.shot_w, self.shot_h)
            self._stagesurf.unmap()
            return True
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
        Args:
            region: Validated region tuple (left, top, right, bottom) to capture.
            
        Returns:
            PooledBuffer, NumPy/CuPy array, or None.
        """
        # Phase 3 & Phase 2 (re-init integration) _grab modification
        
        if self._capture_permanently_failed:
            # Optional: could raise an exception here to be more explicit to the caller.
            logger.error(f"Capture is permanently failed: {self._last_capture_error_message}")
            return None 
        
        if self._needs_reinit:
            if not self._attempt_reinitialization():
                # Re-initialization attempt failed or max attempts reached
                logger.error(f"Re-initialization failed, current grab cannot proceed. Permanent failure: {self._capture_permanently_failed}")
                return None # Current grab fails
            # If re-initialization succeeded, _needs_reinit is now False.

        if not self._is_initialized or self._duplicator is None:
            logger.error("Attempted to grab frame but capture resources are not initialized.")
            self._needs_reinit = True # Flag for next attempt
            return None

        pooled_buffer_wrapper = None
        output_array_for_region = None
        can_use_pool = False

        if self.memory_pool:
            region_h = region[3] - region[1]
            region_w = region[2] - region[0]
            if self.memory_pool.buffer_shape[0] == region_h and \
               self.memory_pool.buffer_shape[1] == region_w and \
               self.memory_pool.buffer_shape[2] == 4: # BGRA
                can_use_pool = True
        
        try:
            if can_use_pool:
                pooled_buffer_wrapper = self.memory_pool.checkout()
                output_array_for_region = pooled_buffer_wrapper.array
            else: # Fallback for non-pooled or mismatched region
                logger.debug(f"Region {region} not matching pool config. Using temporary buffer for this grab.")
                temp_region_h = region[3] - region[1]
                temp_region_w = region[2] - region[0]
                temp_shape = (temp_region_h, temp_region_w, 4) # BGRA
                if self.nvidia_gpu:
                    output_array_for_region = cp.empty(temp_shape, dtype=cp.uint8)
                else:
                    output_array_for_region = np.empty(temp_shape, dtype=np.uint8)

            # Core frame acquisition and processing logic
            try:
                self._duplicator.update_frame() # This now returns None and raises on error
            except RapidShotReinitError as e:
                logger.warning(f"DXGI Re-init error during update_frame: {e}. Flagging for re-initialization.")
                self._needs_reinit = True
                if pooled_buffer_wrapper: pooled_buffer_wrapper.release()
                return None
            except RapidShotDeviceError as e:
                logger.error(f"DXGI Device error during update_frame: {e}. Flagging for re-initialization.")
                self._needs_reinit = True # Device errors also trigger re-init attempts
                if pooled_buffer_wrapper: pooled_buffer_wrapper.release()
                return None
            except RapidShotDXGIError as e: # Other DXGI errors not requiring re-init
                logger.error(f"DXGI error during update_frame: {e}")
                if pooled_buffer_wrapper: pooled_buffer_wrapper.release()
                return None
            except RapidShotError as e: # Other RapidShot errors
                logger.error(f"RapidShot error during update_frame: {e}")
                if pooled_buffer_wrapper: pooled_buffer_wrapper.release()
                return None

            # Phase 2 of timeout handling: Check duplicator.updated and count timeouts
            if self._duplicator.updated:
                self._consecutive_timeouts = 0 # Reset on a successful frame update
            else: # Duplicator.update_frame() succeeded but no new frame (timeout)
                self._consecutive_timeouts += 1
                if self._consecutive_timeouts >= self._timeout_warning_threshold:
                    logger.warning(
                        f"No screen updates received for {self._consecutive_timeouts} consecutive attempts "
                        f"(timeout: {self._duplicator.timeout_ms}ms per attempt). "
                        f"The screen may be static or not updating frequently."
                    )
                    self._consecutive_timeouts = 0 # Reset after warning to avoid log spam
                
                if pooled_buffer_wrapper: pooled_buffer_wrapper.release()
                return None # No new frame

            texture_to_process = self._duplicator.texture
            
            final_array, is_pooled_buffer_still_valid = self._processor.process(
                texture_to_process, self.width, self.height, region, 
                self.rotation_angle, output_array_for_region
            )
            
            # Duplicator.ReleaseFrame() is called inside Duplicator.update_frame's finally block.

            if can_use_pool and pooled_buffer_wrapper:
                if is_pooled_buffer_still_valid:
                    return pooled_buffer_wrapper
                else:
                    pooled_buffer_wrapper.release()
                    return final_array 
            else: 
                return final_array 

        except PoolExhaustedError: # Catch this if checkout fails
            logger.warning("Memory pool exhausted during grab. Consider increasing pool_size_frames.")
            return None
        except Exception as e: # Catch-all for other unexpected errors during _grab
            logger.error(f"Unexpected error in _grab: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if pooled_buffer_wrapper: # Ensure buffer is released
                try:
                    pooled_buffer_wrapper.release()
                except Exception as rel_e:
                    logger.error(f"Error releasing buffer during exception handling in _grab: {rel_e}")
            self._needs_reinit = True # Potentially an error that requires re-init
            self._last_capture_error_message = f"Unexpected error in _grab: {str(e)}"
            return None

    def _on_output_change(self):
        """
        Handle display mode changes.
        """
        time.sleep(0.1)  # Wait for Display mode change (Access Lost)
        self._duplicator.release()
        self._stagesurf.release()
        self._output.update_desc()
        self.width, self.height = self._output.resolution
        if self.region is None or not self._region_set_by_user:
            self.region = (0, 0, self.width, self.height)
        self._validate_region(self.region)
        if self.is_capturing:
            self._rebuild_frame_buffer(self.region)
        self.rotation_angle = self._output.rotation_angle
        while True:
            try:
                self._stagesurf.rebuild(output=self._output, device=self._device)
                self._duplicator = Duplicator(output=self._output, device=self._device)
                break
            except comtypes.COMError:
                continue

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
        if hasattr(self, 'is_capturing') and self.is_capturing:
            self._stop_capture_event.set() # Use renamed event
            if hasattr(self, '_capture_thread') and self._capture_thread is not None:
                self._capture_thread.join(timeout=10) # Wait for thread to finish
        
        self.is_capturing = False
        self._frame_count = 0
        self._frame_available_event.clear()
        # self._stop_capture_event is already set, clear if restartable, but usually not needed
        
        # Phase 4/5: Release any remaining buffers in the deque
        if hasattr(self, '_pooled_frames_deque') and self._pooled_frames_deque is not None:
            with self._capture_lock: 
                # Iterating and releasing like this is safer if deque operations are complex inside loop
                temp_deque_copy = list(self._pooled_frames_deque) # Copy pointers/references
                self._pooled_frames_deque.clear() # Clear original deque
                for buffer_wrapper in temp_deque_copy:
                    try:
                        buffer_wrapper.release()
                    except Exception as e:
                        logger.warning(f"Error releasing buffer from deque during stop: {e}")
        
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
            
            # Get the most recent PooledBuffer wrapper (without removing it)
            latest_pooled_buffer = self._pooled_frames_deque[-1]
            frame_array = latest_pooled_buffer.array
            
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
                    if isinstance(grab_result, PooledBuffer): 
                        with self._capture_lock:
                            if len(self._pooled_frames_deque) == self.max_buffer_len:
                                oldest_buffer = self._pooled_frames_deque[0] 
                                oldest_buffer.release()
                            self._pooled_frames_deque.append(grab_result)
                            last_successful_pooled_buffer = grab_result 
                        self._frame_available_event.set()
                    else: 
                        logger.warning("Continuous mode: Frame processed but not pool-compatible, cannot add to deque.")
                
                elif self._needs_reinit: # _grab returned None and might have set _needs_reinit
                    logger.info("Continuous mode: Grab failed, re-initialization pending or in progress.")
                    # Optional: Short sleep before next attempt if re-init is happening via _grab
                    time.sleep(0.1) # Avoid tight loop if _grab keeps failing due to re-init
                    continue # Try again, _grab will attempt re-init

                elif video_mode and last_successful_pooled_buffer:
                    new_pooled_buffer_for_duplicate = None
                    try:
                        if self.memory_pool: # Ensure pool exists
                            new_pooled_buffer_for_duplicate = self.memory_pool.checkout()
                            if self.nvidia_gpu: # cp array
                                new_pooled_buffer_for_duplicate.array[:] = last_successful_pooled_buffer.array 
                            else: # np array
                                np.copyto(new_pooled_buffer_for_duplicate.array, last_successful_pooled_buffer.array)
                            
                            with self._capture_lock:
                                if len(self._pooled_frames_deque) == self.max_buffer_len:
                                    oldest_buffer = self._pooled_frames_deque[0]
                                    oldest_buffer.release()
                                self._pooled_frames_deque.append(new_pooled_buffer_for_duplicate)
                            self._frame_available_event.set()
                            self._frame_count += 1
                        else:
                            logger.warning("Video_mode: Memory pool not available for duplicating frame.")
                    except PoolExhaustedError:
                        logger.warning("Video_mode: Pool exhausted, cannot duplicate frame.")
                        if new_pooled_buffer_for_duplicate: 
                            new_pooled_buffer_for_duplicate.release()
                    except Exception as dup_e:
                        logger.error(f"Video_mode: Error duplicating frame: {dup_e}")
                        if new_pooled_buffer_for_duplicate:
                            new_pooled_buffer_for_duplicate.release()
            
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
            cancel_timer(self._timer_handle)
            self._timer_handle = None
        
        if capture_error is not None or self._capture_permanently_failed:
            logger.error(f"Capture thread terminated. Error: {capture_error}. Permanent failure: {self._capture_permanently_failed}. Last message: {self._last_capture_error_message}")
            
        capture_duration = time.perf_counter() - self._capture_start_time
        if capture_duration > 0 and self._frame_count > 0: 
            actual_fps = self._frame_count / capture_duration
            logger.info(f"ScreenCapture continuous mode stopped. Captured {self._frame_count} frames in {capture_duration:.2f}s (FPS: {actual_fps:.2f}).")
        else:
            logger.info(f"ScreenCapture continuous mode stopped. No frames captured or capture time was zero.")

    def _rebuild_frame_buffer(self, region: Tuple[int, int, int, int]):
        """
        Rebuild the frame buffer, e.g., after resolution change.
        
        Args:
            region: Region to capture (left, top, right, bottom)
        """
        if region is None:
            region = self.region
        frame_shape = (
            region[3] - region[1],
            region[2] - region[0],
            self.channel_size,
        )
        with self.__lock:
            if self.nvidia_gpu and CUPY_AVAILABLE:
                self.__frame_buffer = cp.ndarray(
                    (self.max_buffer_len, *frame_shape), dtype=cp.uint8
                )
            else:
                self.__frame_buffer = np.ndarray(
                    (self.max_buffer_len, *frame_shape), dtype=np.uint8
                )
            self.__head = 0
            self.__tail = 0
            self.__full = False
            self.__has_frame = False  # Reset frame status

    def _validate_region(self, region: Tuple[int, int, int, int]):
        """
        Validate region coordinates.
        
        Args:
            region: Region to validate (left, top, right, bottom)
            
        Raises:
            ValueError: If region is invalid
        """
        try:
            l, t, r, b = region
            # Apply bounds checking
            l = max(0, min(l, self.width - 1))
            t = max(0, min(t, self.height - 1))
            r = max(l + 1, min(r, self.width))
            b = max(t + 1, min(b, self.height))
            
            # Update region with validated values
            region = (l, t, r, b)
            
            self.region = region
            
            # Update the source region with the new coordinates
            if hasattr(self, '_sourceRegion') and self._sourceRegion is not None:
                self._sourceRegion.left = region[0]
                self._sourceRegion.top = region[1]
                self._sourceRegion.right = region[2]
                self._sourceRegion.bottom = region[3]
            self.shot_w, self.shot_h = region[2]-region[0], region[3]-region[1]
        except Exception as e:
            # If validation fails, use safe default
            logger.error(f"Region validation error: {e}")
            if hasattr(self, 'width') and hasattr(self, 'height'):
                self.region = (0, 0, self.width, self.height)
                self.shot_w, self.shot_h = self.width, self.height

    def release(self):
        """
        Release all resources.
        """
        try:
            if hasattr(self, 'is_capturing') and self.is_capturing: # Check is_capturing before calling stop
                self.stop()
            
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