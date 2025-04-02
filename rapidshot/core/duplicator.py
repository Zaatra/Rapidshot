import ctypes
from time import sleep
from dataclasses import dataclass, InitVar
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *
from rapidshot.core.device import Device
from rapidshot.core.output import Output


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
    cursor: Cursor = Cursor()

    def __post_init__(self, output: Output, device: Device) -> None:
        """
        Initialize the duplicator.
        
        Args:
            output: Output to duplicate
            device: Device to use
        """
        self.output = output
        self.device = device
        self.duplicator = ctypes.POINTER(IDXGIOutputDuplication)()
        output.output.DuplicateOutput(device.device, ctypes.byref(self.duplicator))

    def update_frame(self):
        """
        Update the frame and cursor state.
        
        Returns:
            True if successful, False if output has changed
        """
        info = DXGI_OUTDUPL_FRAME_INFO()
        res = ctypes.POINTER(IDXGIResource)()
        frame_acquired = False
        
        try:
            # Acquire the next frame with a short timeout
            self.duplicator.AcquireNextFrame(
                10,  # 10ms timeout
                ctypes.byref(info),
                ctypes.byref(res),
            )
            frame_acquired = True
            
            # Update cursor information if available
            if info.LastMouseUpdateTime.QuadPart > 0:
                new_pointer_info, new_pointer_shape = self.get_frame_pointer_shape(info)
                if new_pointer_shape is not False:
                    self.cursor.Shape = new_pointer_shape
                    self.cursor.PointerShapeInfo = new_pointer_info
                self.cursor.PointerPositionInfo = info.PointerPosition
                
            # No new frames
            if info.LastPresentTime.QuadPart == 0: 
                self.updated = False
                return True
       
            # Process the frame
            try:
                self.texture = res.QueryInterface(ID3D11Texture2D)
                self.updated = True
                return True
            except comtypes.COMError:
                self.updated = False
                return True
                
        except comtypes.COMError as ce:
            # Handle access lost (e.g., display mode change)
            if (ctypes.c_int32(DXGI_ERROR_ACCESS_LOST).value == ce.args[0] or 
                ctypes.c_int32(ABANDONED_MUTEX_EXCEPTION).value == ce.args[0]):
                self.release()  # Release resources before reinitializing
                sleep(0.1)
                # Re-initialize (will be picked up by _on_output_change)
                return False
                
            # Handle timeout
            if ctypes.c_int32(DXGI_ERROR_WAIT_TIMEOUT).value == ce.args[0]:
                self.updated = False
                return True
                
            # Other unexpected errors
            raise ce
        except Exception:
            # Catch any other unexpected exceptions to ensure cleanup
            self.updated = False
            raise
        finally:
            # Always release the frame if it was acquired
            if frame_acquired:
                self.duplicator.ReleaseFrame()
                
            # If we have a resource pointer but failed to get the texture,
            # ensure it's properly released
            if frame_acquired and res and not self.texture:
                res.Release()

    def release_frame(self):
        """
        Release the current frame.
        """
        if self.duplicator is not None:
            try:
                self.duplicator.ReleaseFrame()
            except (comtypes.COMError, Exception):
                # If ReleaseFrame fails, don't crash
                pass

    def release(self):
        """
        Release all resources.
        """
        if self.duplicator is not None:
            try:
                self.duplicator.Release()
            except (comtypes.COMError, Exception):
                # If Release fails, don't crash
                pass
            finally:
                self.duplicator = None

    def get_frame_pointer_shape(self, frame_info):
        """
        Get pointer shape information from the current frame.
        
        Args:
            frame_info: Frame information
            
        Returns:
            Tuple of (pointer shape info, pointer shape buffer) or (False, False) if no shape
        """
        # Skip if no pointer shape
        if frame_info.PointerShapeBufferSize == 0:
            return False, False
            
        # Allocate buffer for pointer shape
        pointer_shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO()  
        buffer_size_required = ctypes.c_uint()
        pointer_shape_buffer = (ctypes.c_byte * frame_info.PointerShapeBufferSize)()
        
        try:
            # Get pointer shape
            hr = self.duplicator.GetFramePointerShape(
                frame_info.PointerShapeBufferSize, 
                ctypes.byref(pointer_shape_buffer), 
                ctypes.byref(buffer_size_required), 
                ctypes.byref(pointer_shape_info)
            ) 
            
            if hr >= 0:  # Success
                return pointer_shape_info, pointer_shape_buffer
        except (comtypes.COMError, Exception):
            # Handle any exceptions getting the pointer shape
            pass
            
        return False, False

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation
        """
        return "<{} Initialized:{} Cursor:{}available>".format(
            self.__class__.__name__,
            self.duplicator is not None,
            "" if self.cursor.Shape is None else " "
        )