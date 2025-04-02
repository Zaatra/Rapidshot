import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock
import comtypes
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.core.stagesurf import StageSurface
from rapidshot.core.duplicator import Duplicator
from rapidshot._libs.d3d11 import D3D11_BOX
from rapidshot.processor import Processor
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
        max_buffer_len: int = 64,
    ) -> None:
        """
        Initialize a ScreenCapture instance.
        
        Args:
            output: Output device to capture from
            device: Device interface
            region: Region to capture (left, top, right, bottom)
            output_color: Color format (RGB, RGBA, BGR, BGRA, GRAY)
            nvidia_gpu: Whether to use NVIDIA GPU acceleration
            max_buffer_len: Maximum buffer length for capture
        """
        # Initialize basic attributes first to prevent errors during cleanup if initialization fails
        self.is_capturing = False
        self.__thread = None
        self.__lock = Lock()
        self.__stop_capture = Event()
        self.__frame_available = Event()
        self.__frame_buffer = None
        self.__head = 0
        self.__tail = 0
        self.__full = False
        self.__has_frame = False
        self.__timer_handle = None
        self.__frame_count = 0
        self.__capture_start_time = 0
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
        self._latest_frame = None
        self.cursor = False
        
        try:
            # Check if GPU acceleration is requested but CuPy is not available
            if nvidia_gpu and not CUPY_AVAILABLE:
                logger.warning("NVIDIA GPU acceleration requested but CuPy is not available. Falling back to CPU mode.")
                nvidia_gpu = False
                
            self._output = output
            self._device = device
            self._stagesurf = StageSurface(
                output=self._output, device=self._device
            )
            self._duplicator = Duplicator(
                output=self._output, device=self._device
            )
            self._processor = Processor(output_color=output_color, nvidia_gpu=nvidia_gpu)
            
            # Initialize with all fields for completeness
            self.width, self.height = self._output.resolution
            self._sourceRegion = D3D11_BOX(
                left=0, top=0, right=self.width, bottom=self.height, front=0, back=1
            )
            
            self.nvidia_gpu = nvidia_gpu
            self.shot_w, self.shot_h = self.width, self.height
            self.channel_size = len(output_color) if output_color != "GRAY" else 1
            self.rotation_angle = self._output.rotation_angle
            self.output_color = output_color

            self._region_set_by_user = region is not None
            self.region = region
            if self.region is None:
                self.region = (0, 0, self.width, self.height)
            self._validate_region(self.region)
        except Exception as e:
            logger.error(f"Error initializing ScreenCapture: {e}")
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

    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        """
        Grab a single frame from the screen.
        
        Args:
            region: Region to capture (left, top, right, bottom)
            
        Returns:
            Captured frame as numpy array, or None if no update
        """
        if region is None:
            region = self.region
        else:
            self._validate_region(region)
        return self._grab(region)
    
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
            # First, check if we're in continuous mode
            if self.continuous_mode and self.buffer:
                with self._buffer_lock:
                    if not self.buffer:
                        return None
                    # Return the latest frame
                    frame = self._latest_frame
                    return frame
            
            # Determine which API to use - try get_frame first, then update_frame
            try:
                # Check if get_frame method exists
                if hasattr(self._duplicator, 'get_frame'):
                    # Use get_frame API
                    frame_info = self._duplicator.get_frame()
                    if frame_info is None:
                        return None
                        
                    # Make sure we have valid region
                    effective_region = region or self.region
                    if effective_region is None:
                        output_dims = self._duplicator.get_output_dimensions()
                        effective_region = (0, 0, output_dims[0], output_dims[1])
                    
                    # Validate region bounds
                    left, top, right, bottom = effective_region
                    output_width, output_height = self._duplicator.get_output_dimensions()
                    
                    left = max(0, min(left, output_width - 1))
                    top = max(0, min(top, output_height - 1))
                    right = max(left + 1, min(right, output_width))
                    bottom = max(top + 1, min(bottom, output_height))
                    
                    effective_region = (left, top, right, bottom)
                    
                    # Process the frame
                    frame = self._processor.process(
                        frame_info.rect,
                        frame_info.width,
                        frame_info.height,
                        effective_region,
                        self._duplicator.get_rotation_angle()
                    )
                    
                    # Handle cursor overlay if needed
                    if self.cursor and frame_info.cursor_visible and hasattr(self, '_cursor_processor'):
                        self._cursor_processor.draw_cursor(frame, frame_info)
                    
                    # Release the frame once processed
                    self._duplicator.release_frame()
                    return frame
                
                # Fall back to update_frame API
                elif hasattr(self._duplicator, 'update_frame'):
                    if self._duplicator.update_frame():
                        if not self._duplicator.updated:
                            return None
                            
                        # Process using regular method
                        _region = self.region_to_memory_region(region or self.region, self.rotation_angle, self._output)
                        _width = _region[2] - _region[0]
                        _height = _region[3] - _region[1]
                        
                        # Rebuild surface if needed
                        if self._stagesurf.width != _width or self._stagesurf.height != _height:
                            self._stagesurf.release()
                            self._stagesurf.rebuild(output=self._output, device=self._device, dim=(_width, _height))
                        
                        # Create source region
                        source_region = D3D11_BOX(
                            left=_region[0], top=_region[1], right=_region[2], bottom=_region[3], 
                            front=0, back=1
                        )
                        
                        # Copy and process
                        self._device.im_context.CopySubresourceRegion(
                            self._stagesurf.texture, 0, 0, 0, 0, 
                            self._duplicator.texture, 0, ctypes.byref(source_region)
                        )
                        self._duplicator.release_frame()
                        rect = self._stagesurf.map()
                        frame = self._processor.process(
                            rect, self.shot_w, self.shot_h, region or self.region, self.rotation_angle
                        )
                        self._stagesurf.unmap()
                        return frame
                    else:
                        # Handle output change
                        self._on_output_change()
                        return None
                else:
                    logger.error("Duplicator has neither get_frame nor update_frame methods")
                    return None
                    
            except AttributeError as e:
                logger.error(f"API method error: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Error grabbing frame: {e}")
            import traceback
            logger.error(traceback.format_exc())
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
        frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
        
        # Initialize frame buffer
        if self.nvidia_gpu and CUPY_AVAILABLE:
            self.__frame_buffer = cp.ndarray(
                (self.max_buffer_len, *frame_shape), dtype=cp.uint8
            )
        else:
            self.__frame_buffer = np.ndarray(
                (self.max_buffer_len, *frame_shape), dtype=np.uint8
            )
            
        self.__thread = Thread(
            target=self.__capture,
            name="ScreenCapture",
            args=(region, target_fps, video_mode),
        )
        self.__thread.daemon = True
        self.__thread.start()

    def stop(self):
        """
        Stop capturing frames.
        """
        if hasattr(self, 'is_capturing') and self.is_capturing:
            if hasattr(self, '__frame_available'):
                self.__frame_available.set()
            if hasattr(self, '__stop_capture'):
                self.__stop_capture.set()
            if hasattr(self, '__thread') and self.__thread is not None:
                self.__thread.join(timeout=10)
        self.is_capturing = False
        self.__frame_buffer = None
        self.__frame_count = 0
        if hasattr(self, '__frame_available'):
            self.__frame_available.clear()
        if hasattr(self, '__stop_capture'):
            self.__stop_capture.clear()
        self.__has_frame = False  # Reset frame status

    def get_latest_frame(self, as_numpy: bool = True):
        """
        Get the latest captured frame.
        
        Args:
            as_numpy: If True, always return NumPy array even when using GPU acceleration.
                     If False and using GPU acceleration, return CuPy array for better performance.
        
        Returns:
            Latest captured frame as numpy or cupy array
        """
        # Wait until a frame is available
        self.__frame_available.wait()
        
        # Lock to ensure thread safety
        with self.__lock:
            # Get the most recent frame
            ret = self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
            # Clear the event to indicate frame has been consumed
            self.__frame_available.clear()
        
        # Convert to numpy if requested
        if self.nvidia_gpu and CUPY_AVAILABLE:
            if as_numpy:
                return cp.asnumpy(ret)
            else:
                return ret  # Return CuPy array directly
        else:
            return np.array(ret)

    def __capture(
        self, region: Tuple[int, int, int, int], target_fps: int = 60, video_mode: bool = False
    ):
        """
        Internal capture thread implementation.
        
        Args:
            region: Region to capture (left, top, right, bottom)
            target_fps: Target frame rate
            video_mode: Whether to operate in video mode
        """
        if target_fps != 0:
            period_ms = 1000 // target_fps  # milliseconds for periodic timer
            self.__timer_handle = create_high_resolution_timer()
            set_periodic_timer(self.__timer_handle, period_ms)

        self.__capture_start_time = time.perf_counter()
        capture_error = None

        while not self.__stop_capture.is_set():
            if self.__timer_handle:
                res = wait_for_timer(self.__timer_handle, INFINITE)
                if res == WAIT_FAILED:
                    self.__stop_capture.set()
                    capture_error = ctypes.WinError()
                    continue
            try:
                frame = self._grab(region)
                if frame is not None:
                    with self.__lock:
                        # Check if frame dimensions match our buffer
                        current_shape = frame.shape
                        expected_shape = self.__frame_buffer[0].shape
                        
                        # Rebuild buffer if needed (e.g., resolution change)
                        if current_shape != expected_shape:
                            self.width, self.height = frame.shape[1], frame.shape[0]
                            region = (0, 0, frame.shape[1], frame.shape[0])
                            frame_shape = (region[3] - region[1], region[2] - region[0], self.channel_size)
                            
                            if self.nvidia_gpu and CUPY_AVAILABLE:
                                self.__frame_buffer = cp.ndarray(
                                    (self.max_buffer_len, *frame_shape), dtype=cp.uint8
                                )
                            else:
                                self.__frame_buffer = np.ndarray(
                                    (self.max_buffer_len, *frame_shape), dtype=np.uint8
                                )
                        
                        # Store frame and update buffer state
                        self.__frame_buffer[self.__head] = frame
                        if self.__full:
                            self.__tail = (self.__tail + 1) % self.max_buffer_len
                        self.__head = (self.__head + 1) % self.max_buffer_len
                        self.__frame_available.set()
                        self.__frame_count += 1
                        self.__full = self.__head == self.__tail
                        self.__has_frame = True  # We now have at least one frame
                elif video_mode and self.__has_frame:  # Only duplicate in video mode if we have at least one frame
                    with self.__lock:
                        # Copy last frame for video mode
                        if self.nvidia_gpu and CUPY_AVAILABLE:
                            self.__frame_buffer[self.__head] = cp.array(
                                self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
                            )
                        else:
                            self.__frame_buffer[self.__head] = np.array(
                                self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
                            )
                            
                        if self.__full:
                            self.__tail = (self.__tail + 1) % self.max_buffer_len
                        self.__head = (self.__head + 1) % self.max_buffer_len
                        self.__frame_available.set()
                        self.__frame_count += 1
                        self.__full = self.__head == self.__tail
            except Exception as e:
                import traceback
                logger.error(f"Error in capture thread: {e}\n{traceback.format_exc()}")
                self.__stop_capture.set()
                capture_error = e
                continue
                
        # Clean up
        if hasattr(self, '__timer_handle') and self.__timer_handle:
            cancel_timer(self.__timer_handle)
            self.__timer_handle = None
        if capture_error is not None:
            self.stop()
            raise capture_error
            
        # Report capture statistics
        capture_time = time.perf_counter() - self.__capture_start_time
        if capture_time > 0:
            logger.info(f"Screencapture FPS: {int(self.__frame_count/capture_time)}")

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
            if hasattr(self, 'is_capturing'):
                self.stop()
            
            if hasattr(self, '_duplicator'):
                self._duplicator.release()
                
            if hasattr(self, '_stagesurf'):
                self._stagesurf.release()
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