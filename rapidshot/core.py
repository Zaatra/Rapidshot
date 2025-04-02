import time
import ctypes
from typing import Tuple, Optional, Union, List, Any
from threading import Thread, Event, Lock
import comtypes
import numpy as np
# Fix 1: Direct imports from specific modules instead of circular import
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
        # Check if GPU acceleration is requested but CuPy is not available
        if nvidia_gpu and not CUPY_AVAILABLE:
            print("Warning: NVIDIA GPU acceleration requested but CuPy is not available. Falling back to CPU mode.")
            nvidia_gpu = False
            
        self._output: Output = output
        self._device: Device = device
        self._stagesurf: StageSurface = StageSurface(
            output=self._output, device=self._device
        )
        self._duplicator: Duplicator = Duplicator(
            output=self._output, device=self._device
        )
        self._processor: Processor = Processor(output_color=output_color, nvidia_gpu=nvidia_gpu)
        
        # Initialize with all fields for completeness
        self.width, self.height = self._output.resolution
        self._sourceRegion = D3D11_BOX(
            left=0, top=0, right=self.width, bottom=self.height, front=0, back=1
        )
        
        self.nvidia_gpu = nvidia_gpu
        self.shot_w, self.shot_h = self.width, self.height
        self.channel_size = len(output_color) if output_color != "GRAY" else 1
        self.rotation_angle: int = self._output.rotation_angle
        self.output_color = output_color

        self._region_set_by_user = region is not None
        self.region: Tuple[int, int, int, int] = region
        if self.region is None:
            self.region = (0, 0, self.width, self.height)
        self._validate_region(self.region)

        self.max_buffer_len = max_buffer_len
        self.is_capturing = False

        self.__thread = None
        self.__lock = Lock()
        self.__stop_capture = Event()

        self.__frame_available = Event()
        self.__frame_buffer = None
        self.__head = 0
        self.__tail = 0
        self.__full = False

        self.__timer_handle = None

        self.__frame_count = 0
        self.__capture_start_time = 0
    
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
        if rotation_angle == 0:
            return region
        elif rotation_angle == 90:  # Axes (X,Y) -> (-Y,X)
            return (region[1], output.surface_size[1]-region[2], region[3], output.surface_size[1]-region[0])
        elif rotation_angle == 180:  # Axes (X,Y) -> (-X,-Y)
            return (output.surface_size[0]-region[2], output.surface_size[1]-region[3], output.surface_size[0]-region[0], output.surface_size[1]-region[1])
        else:  # rotation_angle == 270 Axes (X,Y) -> (Y,-X)
            return (output.surface_size[0]-region[3], region[0], output.surface_size[0]-region[1], region[2])

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

            # Fix 2: Create a source-specific region object with the transformed coordinates
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

    def _grab(self, region: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Internal implementation of grab.
        
        Args:
            region: Region to capture (left, top, right, bottom)
            
        Returns:
            Captured frame as numpy array, or None if no update
        """
        if self._duplicator.update_frame():
            if not self._duplicator.updated:
                return None

            _region = self.region_to_memory_region(region, self.rotation_angle, self._output)
            _width = _region[2] - _region[0]
            _height = _region[3] - _region[1]

            # Rebuild surface if needed
            if self._stagesurf.width != _width or self._stagesurf.height != _height:
                self._stagesurf.release()
                self._stagesurf.rebuild(output=self._output, device=self._device, dim=(_width, _height))

            # Fix 2: Create a source-specific region object with the transformed coordinates
            source_region = D3D11_BOX(
                left=_region[0], top=_region[1], right=_region[2], bottom=_region[3], front=0, back=1
            )

            # Copy the frame
            self._device.im_context.CopySubresourceRegion(
                self._stagesurf.texture, 0, 0, 0, 0, self._duplicator.texture, 0, ctypes.byref(source_region)
            )
            self._duplicator.release_frame()
            rect = self._stagesurf.map()
            frame = self._processor.process(
                rect, self.shot_w, self.shot_h, region, self.rotation_angle
            )
            self._stagesurf.unmap()
            return frame
        else:
            self._on_output_change()
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
        if self.is_capturing:
            self.__frame_available.set()
            self.__stop_capture.set()
            if self.__thread is not None:
                self.__thread.join(timeout=10)
        self.is_capturing = False
        self.__frame_buffer = None
        self.__frame_count = 0
        self.__frame_available.clear()
        self.__stop_capture.clear()

    def get_latest_frame(self, as_numpy: bool = True):
        """
        Get the latest captured frame.
        
        Args:
            as_numpy: If True, always return NumPy array even when using GPU acceleration.
                     If False and using GPU acceleration, return CuPy array for better performance.
        
        Returns:
            Latest captured frame as numpy or cupy array
        """
        self.__frame_available.wait()
        with self.__lock:
            ret = self.__frame_buffer[(self.__head - 1) % self.max_buffer_len]
            self.__frame_available.clear()
        
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
                elif video_mode:
                    with self.__lock:
                        # Make sure we have at least one frame before trying to copy
                        if self.__frame_count > 0:
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
                print(traceback.format_exc())
                self.__stop_capture.set()
                capture_error = e
                continue
                
        # Clean up
        if self.__timer_handle:
            cancel_timer(self.__timer_handle)
            self.__timer_handle = None
        if capture_error is not None:
            self.stop()
            raise capture_error
            
        # Report capture statistics
        capture_time = time.perf_counter() - self.__capture_start_time
        if capture_time > 0:
            print(f"Screencapture FPS: {int(self.__frame_count/capture_time)}")

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

    def _validate_region(self, region: Tuple[int, int, int, int]):
        """
        Validate region coordinates.
        
        Args:
            region: Region to validate (left, top, right, bottom)
            
        Raises:
            ValueError: If region is invalid
        """
        l, t, r, b = region
        if not (self.width >= r > l >= 0 and self.height >= b > t >= 0):
            raise ValueError(
                f"Invalid Region: Region should be in {self.width}x{self.height}"
            )
        self.region = region
        
        # Update the source region with the new coordinates
        self._sourceRegion.left = region[0]
        self._sourceRegion.top = region[1]
        self._sourceRegion.right = region[2]
        self._sourceRegion.bottom = region[3]
        self.shot_w, self.shot_h = region[2]-region[0], region[3]-region[1]

    def release(self):
        """
        Release all resources.
        """
        self.stop()
        self._duplicator.release()
        self._stagesurf.release()

    def __del__(self):
        """
        Destructor to ensure resources are released.
        """
        self.release()

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation of the ScreenCapture instance
        """
        return "<{}:\n\t{},\n\t{},\n\t{},\n\t{}\n>".format(
            "ScreenCapture",
            self._device,
            self._output,
            self._stagesurf,
            self._duplicator,
        )