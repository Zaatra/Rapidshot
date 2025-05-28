import ctypes
from rapidshot.util.logging import get_logger
from time import sleep
from dataclasses import dataclass, InitVar
from typing import Tuple, Optional, Union
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import (
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_RESET,
    DXGI_ERROR_INVALID_CALL,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_NOT_FOUND, # For cursor shape
    ABANDONED_MUTEX_EXCEPTION, # Already used
    IDXGIOutputDuplication,
    DXGI_OUTDUPL_POINTER_POSITION,
    DXGI_OUTDUPL_POINTER_SHAPE_INFO,
    DXGI_OUTDUPL_FRAME_INFO,
    IDXGIResource,
    ID3D11Texture2D,
)
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.util.errors import (
    RapidShotDXGIError,
    RapidShotReinitError,
    RapidShotDeviceError,
    RapidShotConfigError,
    RapidShotTimeoutError # Though timeout is handled locally, good to have if needed
)

# Configure logging
logger = logging.getLogger(__name__)

# Error constants for better reporting
CURSOR_ERRORS = {
    "NO_SHAPE": "No cursor shape available",
    "SHAPE_BUFFER_EMPTY": "Cursor shape buffer is empty",
    "BUFFER_TOO_SMALL": "Provided buffer is too small for cursor shape",
    "QUERY_FAILED": "Failed to query cursor shape information",
    "INTERFACE_ERROR": "Failed to access cursor interface"
}

@dataclass
class Cursor:
    """
    Dataclass for cursor information.
    """
    PointerPositionInfo: DXGI_OUTDUPL_POINTER_POSITION = DXGI_OUTDUPL_POINTER_POSITION()
    PointerShapeInfo: DXGI_OUTDUPL_POINTER_SHAPE_INFO = DXGI_OUTDUPL_POINTER_SHAPE_INFO()
    Shape: bytes = None
    

@dataclass
class Duplicator:
    """
    Desktop Duplicator implementation.
    Handles frame and cursor acquisition from the Desktop Duplication API.
    """
    texture: ctypes.POINTER(ID3D11Texture2D) = ctypes.POINTER(ID3D11Texture2D)()
    duplicator: ctypes.POINTER(IDXGIOutputDuplication) = None
    updated: bool = False
    output: InitVar[Output] = None
    device: InitVar[Device] = None
    timeout_ms: int = 10  # Timeout for AcquireNextFrame in milliseconds
    cursor: Cursor = Cursor()
    last_error: str = ""
    cursor_visible: bool = False

    def __post_init__(self, output: Output, device: Device) -> None:
        """
        Initialize the duplicator.
        
        Args:
            output: Output to duplicate
            device: Device to use
        """
        try:
            self.output = output
            self.device = device
            self.duplicator = ctypes.POINTER(IDXGIOutputDuplication)()
            output.output.DuplicateOutput(device.device, ctypes.byref(self.duplicator))
            logger.info(f"Duplicator initialized for output: {output.devicename}")
            
            # Store output dimensions and rotation
            self._output_width, self._output_height = self.output.resolution
            self._rotation_angle = self.output.rotation_angle
            
        except comtypes.COMError as ce:
            error_msg = f"Failed to initialize duplicator: {ce}"
            logger.error(error_msg)
            self.last_error = error_msg
            # Map COMError to custom exceptions
            hresult = ce.args[0] if ce.args else None
            if hresult in (DXGI_ERROR_INVALID_CALL, DXGI_ERROR_UNSUPPORTED):
                raise RapidShotConfigError(f"Failed to initialize duplicator due to configuration or invalid call: {ce}", hresult=hresult) from ce
            elif hresult in (DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_DEVICE_RESET):
                raise RapidShotDeviceError(f"Failed to initialize duplicator due to device error: {ce}", hresult=hresult) from ce
            else:
                raise RapidShotDXGIError(f"Failed to initialize duplicator: {ce}", hresult=hresult) from ce

    def update_frame(self) -> None:
        """
        Update the frame and cursor state.
        Sets self.updated to True if a new frame is available, False otherwise.
        Raises exceptions for critical errors.
        """
        # Reset state for this update attempt
        self.updated = False
        self.last_error = ""
        
        info = DXGI_OUTDUPL_FRAME_INFO()
        res = ctypes.POINTER(IDXGIResource)()
        frame_acquired = False
        
        try:
            # Acquire the next frame with a short timeout
            self.duplicator.AcquireNextFrame(
                self.timeout_ms,
                ctypes.byref(info),
                ctypes.byref(res),
            )
            frame_acquired = True
            logger.debug("Frame acquired successfully")
            
            # FIX: Handle both LARGE_INTEGER and int types for LastMouseUpdateTime
            # Get the mouse update time safely
            if hasattr(info.LastMouseUpdateTime, 'QuadPart'):
                mouse_update_time = info.LastMouseUpdateTime.QuadPart
            else:
                # Handle case where LastMouseUpdateTime is already an integer
                mouse_update_time = info.LastMouseUpdateTime
            
            # Update cursor information if available
            if mouse_update_time > 0:
                cursor_result = self.get_frame_pointer_shape(info)
                if isinstance(cursor_result, tuple) and len(cursor_result) == 3:
                    new_pointer_info, new_pointer_shape, error_msg = cursor_result
                    if new_pointer_shape is not False:
                        self.cursor.Shape = new_pointer_shape
                        self.cursor.PointerShapeInfo = new_pointer_info
                    elif error_msg:
                        logger.debug(f"Cursor shape not updated: {error_msg}")
                self.cursor.PointerPositionInfo = info.PointerPosition
                self.cursor_visible = info.PointerPosition.Visible
            
            # FIX: Handle both LARGE_INTEGER and int types for LastPresentTime
            # Get the last present time safely
            if hasattr(info.LastPresentTime, 'QuadPart'):
                last_present_time = info.LastPresentTime.QuadPart
            else:
                # Handle case where LastPresentTime is already an integer
                last_present_time = info.LastPresentTime
                
            # No new frames
            if last_present_time == 0: 
                logger.debug("No new frame content")
                self.updated = False
                return True
       
            # Process the frame
            try:
                self.texture = res.QueryInterface(ID3D11Texture2D)
                self.updated = True
                return True
            except comtypes.COMError as ce:
                error_msg = f"Failed to query texture interface: {ce}"
                logger.warning(error_msg)
                self.last_error = error_msg
                self.updated = False
                return True
                
        except comtypes.COMError as ce:
            hresult = ce.args[0] if ce.args else None
            self.last_error = f"COMError in update_frame: {ce} (HRESULT: {hresult:#010x if isinstance(hresult, int) else hresult})"
            logger.warning(self.last_error)

            if hresult == DXGI_ERROR_WAIT_TIMEOUT:
                logger.debug("Frame acquisition timed out.")
                self.updated = False # No new frame
                # Do not return here, finally block must execute
            elif hresult == DXGI_ERROR_ACCESS_LOST or hresult == ABANDONED_MUTEX_EXCEPTION:
                # ABANDONED_MUTEX_EXCEPTION (0x000002E8) can also indicate a state requiring reinitialization.
                # self.release() # Release current resources, might be part of reinit logic higher up
                raise RapidShotReinitError(f"Access lost, re-initialization needed: {ce}", hresult=hresult) from ce
            elif hresult == DXGI_ERROR_DEVICE_REMOVED or hresult == DXGI_ERROR_DEVICE_RESET:
                # self.release() # Release current resources
                raise RapidShotDeviceError(f"Device error, re-initialization needed: {ce}", hresult=hresult) from ce
            else:
                # Other COM errors
                raise RapidShotDXGIError(f"Unexpected DXGI error in update_frame: {ce}", hresult=hresult) from ce
        
        except Exception as e:
            # Catch any other unexpected Python exceptions to ensure cleanup
            self.last_error = f"Python exception in update_frame: {e}"
            logger.error(self.last_error)
            self.updated = False # Ensure updated is False on other exceptions
            raise RapidShotError(f"Unhandled Python exception in update_frame: {e}") from e # Wrap in RapidShotError
        
        finally:
            # Always release the DDA frame if it was acquired by AcquireNextFrame
            if frame_acquired: # frame_acquired is True only if AcquireNextFrame succeeded
                try:
                    self.duplicator.ReleaseFrame()
                except Exception as e:
                    logger.warning(f"Failed to release frame: {e}")
                
            # If we have a resource pointer but failed to get the texture,
            # ensure it's properly released
            if frame_acquired and res and not self.texture:
                try:
                    res.Release()
                except Exception as e:
                    logger.warning(f"Failed to release resource: {e}")

    # Add this method to provide compatibility with capture.py
    def get_frame(self):
        """
        Get the current frame - wrapper for update_frame for API compatibility
        
        Returns:
            Frame information or None if no update
        """
        if self.update_frame():
            if not self.updated:
                return None
                
            # Create a simple frame information object with expected attributes
            class FrameInfo:
                def __init__(self, rect, width, height, cursor_visible=False):
                    self.rect = rect
                    self.width = width
                    self.height = height
                    self.cursor_visible = cursor_visible
                    
            # Return frame info object with the expected data
            return FrameInfo(
                rect=self.texture,  # Use texture directly as rect
                width=self._output_width,
                height=self._output_height,
                cursor_visible=self.cursor_visible
            )
        return None
        
    def get_output_dimensions(self):
        """
        Get the dimensions of the output device
        
        Returns:
            Tuple of (width, height)
        """
        return (self._output_width, self._output_height)
        
    def get_rotation_angle(self):
        """
        Get the rotation angle of the output device
        
        Returns:
            Rotation angle in degrees (0, 90, 180, or 270)
        """
        return self._rotation_angle

    def release_frame(self) -> None:
        """
        Release the current frame.
        """
        # Release frame warning fix applied
        if self.duplicator is not None:
            try:
                self.duplicator.ReleaseFrame()
                logger.debug("Frame released")
            except comtypes.COMError as ce:
                # Don't log as warning for specific known error code
                if ce.args and ce.args[0] == -2005270527:
                    logger.debug(f"Frame already released: {ce}")
                else:
                    logger.warning(f"Failed to release frame: {ce} (HRESULT: {ce.args[0]:#010x if ce.args else 'N/A'})")
                    # Not raising custom error here as it's a cleanup step, but logging is important.
                    # If specific HRESULTs here are critical, they could be mapped.
                    self.last_error = f"Failed to release frame: {ce}" # Keep last_error for simple errors
            except Exception as e: # Catch non-COM errors during ReleaseFrame
                logger.warning(f"Unexpected Python error releasing frame: {e}")
                self.last_error = f"Unexpected Python error releasing frame: {e}"


    def release(self) -> None:
        """
        Release all duplicator resources.
        """
        if self.duplicator is not None:
            try:
                self.duplicator.Release()
                logger.info("Duplicator resources released.")
            except comtypes.COMError as ce:
                hresult = ce.args[0] if ce.args else None
                error_msg = f"Failed to release duplicator: {ce} (HRESULT: {hresult:#010x if isinstance(hresult, int) else hresult})"
                logger.warning(error_msg)
                # Set last_error but don't necessarily raise; this is a cleanup.
                # If this fails, often the parent (ScreenCapture) will try to release Device too.
                self.last_error = error_msg 
            except Exception as e: # Catch non-COM errors
                error_msg = f"Unexpected Python error releasing duplicator: {e}"
                logger.warning(error_msg)
                self.last_error = error_msg
            finally: # Ensure self.duplicator is set to None even if Release() fails somehow
                self.duplicator = None

    def get_frame_pointer_shape(self, frame_info) -> Union[Tuple[DXGI_OUTDUPL_POINTER_SHAPE_INFO, bytes, str], Tuple[bool, bool, str]]:
        """
        Get pointer shape information from the current frame.
        
        Args:
            frame_info: Frame information
            
        Returns:
            Tuple of (pointer shape info, pointer shape buffer, error_message) or (False, False, error_message) if error
        """
        # Skip if no pointer shape
        if frame_info.PointerShapeBufferSize == 0:
            return False, False, CURSOR_ERRORS["NO_SHAPE"]
            
        # Allocate buffer for pointer shape
        pointer_shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO()  
        buffer_size_required = ctypes.c_uint()
        
        try:
            # Verify buffer size
            if frame_info.PointerShapeBufferSize <= 0:
                return False, False, CURSOR_ERRORS["SHAPE_BUFFER_EMPTY"]
                
            # Allocate buffer
            pointer_shape_buffer = (ctypes.c_byte * frame_info.PointerShapeBufferSize)()
            
            # Get pointer shape
            hr = self.duplicator.GetFramePointerShape(
                frame_info.PointerShapeBufferSize, 
                ctypes.byref(pointer_shape_buffer), 
                ctypes.byref(buffer_size_required), 
                ctypes.byref(pointer_shape_info)
            ) 
            
            if hr >= 0:  # Success
                logger.debug(f"Cursor shape acquired: {pointer_shape_info.Width}x{pointer_shape_info.Height}, Type: {pointer_shape_info.Type}")
                return pointer_shape_info, pointer_shape_buffer, ""
            else:
                error_msg = f"GetFramePointerShape returned error code: {hr}"
                logger.warning(error_msg)
                self.last_error = error_msg
                return False, False, error_msg
                
        except comtypes.COMError as ce:
            hresult = ce.args[0] if ce.args else None
            self.last_error = f"COMError in get_frame_pointer_shape: {ce} (HRESULT: {hresult:#010x if isinstance(hresult, int) else hresult})"
            logger.warning(self.last_error)

            if hresult == DXGI_ERROR_ACCESS_LOST:
                # This specific error should propagate as it requires re-initialization.
                # Caller (update_frame) will handle this by raising RapidShotReinitError.
                # For now, let this COMError propagate up to update_frame's handler.
                raise # Re-raise to be caught by update_frame's COMError handler
            elif hresult == DXGI_ERROR_NOT_FOUND:
                # This is a common case, not necessarily a critical error for the duplicator itself.
                return False, False, f"Cursor shape not found (HRESULT: {hresult:#010x})"
            # Other errors are logged and returned as failure.
            return False, False, self.last_error
            
        except Exception as e:
            # Handle any other Python exceptions
            self.last_error = f"Python exception in get_frame_pointer_shape: {e}"
            logger.warning(self.last_error)
            return False, False, self.last_error # Return error message

    def get_last_error(self) -> str:
        """
        Get the last error message.
        
        Returns:
            Last error message
        """
        return self.last_error

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation
        """
        cursor_status = "not available" if self.cursor.Shape is None else "available"
        return "<{} Initialized:{} Cursor:{}>".format(
            self.__class__.__name__,
            self.duplicator is not None,
            cursor_status
        )