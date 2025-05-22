import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock
import comtypes
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from rapidshot.memory_pool import NumpyMemoryPool, CupyMemoryPool, PoolExhaustedError # Added
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
        self._buffer_lock = Lock() # This lock might be reused or consolidated with _capture_lock
        # self._latest_frame = None # Will be derived from the deque
        self.cursor = False
        self.memory_pool = None # Initialize memory_pool attribute
        
        try:
            # Check if GPU acceleration is requested but CuPy is not available
            if nvidia_gpu and not CUPY_AVAILABLE:
                logger.warning("NVIDIA GPU acceleration requested but CuPy is not available. Falling back to CPU mode.")
                nvidia_gpu = False
            
            self.nvidia_gpu = nvidia_gpu # Set this early for pool creation
            self._output = output
            self._device = device

            # Initialize with all fields for completeness
            self.width, self.height = self._output.resolution
            
            # Initialize Memory Pool (Phase 1, adjusted for region-specific buffer sizes)
            # Finalize self.region before determining pool buffer_shape
            self._region_set_by_user = region is not None
            self.region = region
            if self.region is None:
                self.region = (0, 0, self.width, self.height)
            self._validate_region(self.region) # This finalizes self.region dimensions

            region_height = self.region[3] - self.region[1]
            region_width = self.region[2] - self.region[0]
            buffer_shape = (region_height, region_width, 4) # BGRA
            dtype = np.uint8 # Universal dtype for buffer data

            if self.nvidia_gpu:
                self.memory_pool = CupyMemoryPool(buffer_shape, dtype, pool_size_frames)
            else:
                self.memory_pool = NumpyMemoryPool(buffer_shape, dtype, pool_size_frames)
            logger.info(f"Initialized {self.memory_pool.__class__.__name__} with {pool_size_frames} buffers of shape {buffer_shape}.")

            self._stagesurf = StageSurface(
                output=self._output, device=self._device
            )
            self._duplicator = Duplicator(
                output=self._output, device=self._device
            )
            self._processor = Processor(output_color=output_color, nvidia_gpu=self.nvidia_gpu) # Use self.nvidia_gpu
            
            self._sourceRegion = D3D11_BOX(
                left=0, top=0, right=self.width, bottom=self.height, front=0, back=1
            )
            
            self.shot_w, self.shot_h = self.width, self.height
            self.channel_size = 4 # Pooled buffers store BGRA initially, processor handles final format.
            
            self.rotation_angle = self._output.rotation_angle
            self.output_color = output_color # For the processor to know the target format

            # self.region is already set and validated above before pool init.
        except Exception as e:
            logger.error(f"Error initializing ScreenCapture: {e}")
            # Ensure partial resources are cleaned up if pool was created
            if self.memory_pool:
                self.memory_pool.destroy_pool()
            raise
    
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
        # Phase 3: _grab modification
        # Note: Continuous mode logic is handled by the __capture thread calling this _grab.
        # This method focuses on a single grab operation using the pool if possible.

        pooled_buffer_wrapper = None
        output_array_for_region = None
        can_use_pool = False

        # Determine if the requested region matches the pool's buffer configuration
        if self.memory_pool:
            region_h = region[3] - region[1]
            region_w = region[2] - region[0]
            # Pool stores BGRA (4 channels)
            if self.memory_pool.buffer_shape[0] == region_h and \
               self.memory_pool.buffer_shape[1] == region_w and \
               self.memory_pool.buffer_shape[2] == 4:
                can_use_pool = True
        
        try:
            if can_use_pool:
                try:
                    pooled_buffer_wrapper = self.memory_pool.checkout()
                    output_array_for_region = pooled_buffer_wrapper.array
                except PoolExhaustedError:
                    logger.warning("Memory pool exhausted. Consider increasing pool_size_frames or reducing capture rate.")
                    return None # Or implement a fallback to non-pooled allocation if desired
            else:
                logger.warning(f"Requested region {region} size differs from pool buffer shape {self.memory_pool.buffer_shape if self.memory_pool else 'N/A'}. Bypassing pool for this grab.")
                # Fallback: Create a temporary buffer for this specific grab
                # This path won't return a PooledBuffer, just the raw array.
                temp_region_h = region[3] - region[1]
                temp_region_w = region[2] - region[0]
                temp_shape = (temp_region_h, temp_region_w, 4) # BGRA
                if self.nvidia_gpu:
                    output_array_for_region = cp.empty(temp_shape, dtype=cp.uint8)
                else:
                    output_array_for_region = np.empty(temp_shape, dtype=np.uint8)

            if not self._duplicator.update_frame():
                if self._duplicator.last_error == "DXGI_ERROR_ACCESS_LOST": # Check specific error if available
                    self._on_output_change()
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            if not self._duplicator.updated:
                if pooled_buffer_wrapper:
                    pooled_buffer_wrapper.release()
                return None

            texture_to_process = self._duplicator.texture # This is the ID3D11Texture2D
            
            # The processor now returns (final_array, is_pooled_buffer_still_valid)
            final_array, is_pooled_buffer_still_valid = self._processor.process(
                texture_to_process,
                self.width, # Full output width
                self.height, # Full output height
                region, # Specific region to capture
                self.rotation_angle, # Use self.rotation_angle (updated by _on_output_change)
                output_array_for_region # The buffer (pooled or temporary) for the region
            )
            
            self._duplicator.release_frame() # Release DDA frame after processing

            if can_use_pool and pooled_buffer_wrapper:
                if is_pooled_buffer_still_valid:
                    return pooled_buffer_wrapper # Return the wrapper
                else:
                    pooled_buffer_wrapper.release() # Original pooled buffer is not valid for the result
                    return final_array # Return the new array (e.g. from shape-changing rotation)
            else: # Was not using pool or checkout failed (though checkout failing returns None above)
                return final_array # Return the directly processed array (non-pooled)

        except Exception as e:
            logger.error(f"Error in _grab: {e}")
            import traceback
            logger.error(traceback.format_exc())
            if pooled_buffer_wrapper: # Ensure buffer is released if an error occurs after checkout
                try:
                    pooled_buffer_wrapper.release()
                except Exception as rel_e:
                    logger.error(f"Error releasing buffer during exception handling: {rel_e}")
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
            with self._capture_lock: # Protect access to deque
                while self._pooled_frames_deque:
                    buffer_wrapper = self._pooled_frames_deque.popleft()
                    try:
                        buffer_wrapper.release()
                    except Exception as e:
                        logger.warning(f"Error releasing buffer from deque during stop: {e}")
                # self._pooled_frames_deque.clear() # Already cleared by popleft loop
        
        # Old buffer cleanup (if any parts of old logic were missed for __frame_buffer)
        # self.__frame_buffer = None # Ensure old buffer is cleared if it was used

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
                # Use self.region for continuous capture, which was set during start()
                grab_result = self._grab(self.region) 

                if grab_result is not None:
                    self._frame_count += 1
                    if isinstance(grab_result, PooledBuffer): # Check if it's a PooledBuffer wrapper
                        with self._capture_lock:
                            # If deque is full, append will pop the oldest, which needs release
                            if len(self._pooled_frames_deque) == self.max_buffer_len:
                                oldest_buffer = self._pooled_frames_deque[0] # Peek before it's popped
                                oldest_buffer.release()
                            self._pooled_frames_deque.append(grab_result)
                            last_successful_pooled_buffer = grab_result # Keep for video_mode
                        self._frame_available_event.set()
                    else: 
                        # Raw array returned (e.g. pool bypassed or shape changed)
                        # Cannot add to _pooled_frames_deque as it needs a wrapper for release.
                        # This frame is effectively dropped for continuous mode unless handled differently.
                        logger.warning("Continuous mode: Frame processed but not pool-compatible, cannot add to deque.")
                        # If we wanted to keep it, we'd need a different strategy for self._latest_frame
                        # or make _pooled_frames_deque handle raw arrays too (with no release for them).

                elif video_mode and last_successful_pooled_buffer:
                    # Duplicate last successful pooled frame content into a new pooled buffer
                    new_pooled_buffer_for_duplicate = None
                    try:
                        new_pooled_buffer_for_duplicate = self.memory_pool.checkout()
                        # Copy content from last_successful_pooled_buffer.array to new_pooled_buffer_for_duplicate.array
                        if self.nvidia_gpu:
                            new_pooled_buffer_for_duplicate.array[:] = last_successful_pooled_buffer.array # cp array copy
                        else:
                            np.copyto(new_pooled_buffer_for_duplicate.array, last_successful_pooled_buffer.array) # np array copy
                        
                        with self._capture_lock:
                            if len(self._pooled_frames_deque) == self.max_buffer_len:
                                oldest_buffer = self._pooled_frames_deque[0]
                                oldest_buffer.release()
                            self._pooled_frames_deque.append(new_pooled_buffer_for_duplicate)
                        self._frame_available_event.set()
                        self._frame_count += 1
                    except PoolExhaustedError:
                        logger.warning("Video_mode: Pool exhausted, cannot duplicate frame.")
                        if new_pooled_buffer_for_duplicate: # Should not happen if checkout failed
                            new_pooled_buffer_for_duplicate.release()
                    except Exception as dup_e:
                        logger.error(f"Video_mode: Error duplicating frame: {dup_e}")
                        if new_pooled_buffer_for_duplicate:
                            new_pooled_buffer_for_duplicate.release()
                # If grab_result is None and not video_mode, or video_mode but no last_successful_frame, do nothing.

            except Exception as e: # Catch errors from _grab() or within this loop
                import traceback
                logger.error(f"Error in capture thread: {e}\n{traceback.format_exc()}")
                self._stop_capture_event.set() # Stop capture on error
                capture_error = e
                # No continue here, loop will exit due to _stop_capture_event.set()
                
        # Clean up timer
        if self._timer_handle:
            cancel_timer(self._timer_handle)
            self._timer_handle = None
        
        if capture_error is not None:
            # self.stop() # Already called implicitly or explicitly by setting event
            logger.error(f"Capture thread terminated due to error: {capture_error}")
            # Propagate error to main thread? For now, just logs. It will stop the capture.
            
        # Report capture statistics
        capture_duration = time.perf_counter() - self._capture_start_time
        if capture_duration > 0 and self._frame_count > 0: # Avoid division by zero if no frames/time
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