import ctypes
import logging
import os
import comtypes  # type: ignore[import-untyped]
from rapidshot.util.logging import get_logger
from time import sleep
from dataclasses import dataclass, field, InitVar
from typing import List, Tuple, Optional, Union
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import (
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_MORE_DATA,
    DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_RESET,
    DXGI_ERROR_INVALID_CALL,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_NOT_FOUND, # For cursor shape
    DXGI_ERROR_SESSION_DISCONNECTED,
    DXGI_RECOVERABLE_ERRORS,
    DXGI_DEVICE_ERRORS,
    DXGI_PROTECTED_CONTENT_ERRORS,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
    ABANDONED_MUTEX_EXCEPTION, # Already used
    IDXGIOutputDuplication,
    IDXGIOutput5,
    DXGI_OUTDUPL_POINTER_POSITION,
    DXGI_OUTDUPL_POINTER_SHAPE_INFO,
    DXGI_OUTDUPL_FRAME_INFO,
    IDXGIResource,
    RECT,
)
from rapidshot.core.device import Device
from rapidshot.core.output import Output
from rapidshot.util.errors import (
    RapidShotError,
    RapidShotDXGIError,
    RapidShotReinitError,
    RapidShotDeviceError,
    RapidShotConfigError,
    RapidShotProtectedContentError,
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

# Formats offered to DuplicateOutput1, in preference order. The desktop is
# normally BGRA8; the wider formats are listed so an HDR or 10-bit desktop
# duplicates instead of failing outright.
DUPLICATE_OUTPUT1_FORMATS = (
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
)

# Env var escape hatch: set RAPIDSHOT_DUPLICATE_OUTPUT=legacy to force the
# pre-1.5 DuplicateOutput path if a driver misbehaves on DuplicateOutput1.
_DUPLICATE_OUTPUT_ENV = "RAPIDSHOT_DUPLICATE_OUTPUT"


def _format_hresult(hresult) -> str:
    """Render an HRESULT as 0x-prefixed hex, tolerating non-integer values."""
    if isinstance(hresult, int):
        return f"{hresult & 0xFFFFFFFF:#010x}"
    return str(hresult)


def _legacy_duplication_forced() -> bool:
    """True if the env var asks for the pre-DuplicateOutput1 code path."""
    return os.environ.get(_DUPLICATE_OUTPUT_ENV, "").strip().lower() in (
        "legacy",
        "0",
        "duplicateoutput",
    )


@dataclass
class Cursor:
    """
    Dataclass for cursor information.
    """
    PointerPositionInfo: DXGI_OUTDUPL_POINTER_POSITION = field(
        default_factory=DXGI_OUTDUPL_POINTER_POSITION
    )
    PointerShapeInfo: DXGI_OUTDUPL_POINTER_SHAPE_INFO = field(
        default_factory=DXGI_OUTDUPL_POINTER_SHAPE_INFO
    )
    Shape: bytes = None


@dataclass
class Duplicator:
    """
    Desktop Duplicator implementation.
    Handles frame and cursor acquisition from the Desktop Duplication API.
    """
    texture: ctypes.POINTER(ID3D11Texture2D) = None
    duplicator: ctypes.POINTER(IDXGIOutputDuplication) = None
    updated: bool = False
    output: InitVar[Output] = None
    device: InitVar[Device] = None
    # How long AcquireNextFrame blocks waiting for the compositor to present.
    #
    # This one number is the whole difference in behaviour between this library
    # and the poll-based ones. Measured on the dev machine's 100 Hz output
    # against a source presenting at ~610 updates/s:
    #
    #     timeout   fps    hit%     calls     cpu%
    #           0  127.8    2.4%    26,224    68.6
    #           1  119.2   74.5%       801    19.5
    #           2  124.2   86.4%       719    16.9
    #           5  118.2   95.3%       621    18.9
    #          10  118.9  100.0%       595    15.7      <- default
    #          16  116.9  100.0%       585    16.9
    #
    # Throughput barely moves; CPU moves by 4x. Polling at 0 buys ~7% more
    # frames for ~4.4x the CPU, which is what DXcam does -- it calls
    # AcquireNextFrame(0) and reached 134 fps at 66% CPU in the same
    # comparison. 10 ms is the default because a capture stage in a pipeline
    # usually wants its cores back more than it wants the last 7% of frames.
    #
    # Exposed publicly as `ScreenCapture.timeout_ms`; see
    # `benchmarks/compare_libraries.py`.
    timeout_ms: int = 10
    cursor: Cursor = field(default_factory=Cursor)
    last_error: str = ""
    cursor_visible: bool = False
    protected_content_detected: bool = False
    used_duplicate_output1: bool = False
    # QPC timestamp of the most recent frame with new content, and how many
    # display updates DXGI coalesced into it. Both come straight from
    # DXGI_OUTDUPL_FRAME_INFO and are what Stage 3's Frame metadata is built on.
    last_present_time: int = 0
    accumulated_frames: int = 0
    # Regions the compositor redrew, in desktop coordinates. None means the
    # metadata could not be read, which is not the same as an empty list.
    dirty_rects: Optional[List[Tuple[int, int, int, int]]] = None
    # True when the driver merged rects rather than reporting them individually,
    # so the regions are an over-estimate of what actually changed.
    rects_coalesced: bool = False
    _frame_acquired: bool = False

    def __post_init__(self, output: Output, device: Device) -> None:
        """
        Initialize the duplicator.

        Args:
            output: Output to duplicate
            device: Device to use
        """
        self.output = output
        self.device = device
        self.texture = ctypes.POINTER(ID3D11Texture2D)()

        try:
            self.duplicator, self.used_duplicate_output1 = self._create_duplication(
                output, device
            )
            logger.info(
                f"Duplicator initialized for output: {output.devicename} "
                f"(DuplicateOutput1={self.used_duplicate_output1})"
            )

            # Store output dimensions and rotation
            self._output_width, self._output_height = self.output.resolution
            self._rotation_angle = self.output.rotation_angle

        except comtypes.COMError as ce:
            error_msg = f"Failed to initialize duplicator: {ce}"
            logger.error(error_msg)
            self.last_error = error_msg
            raise self._map_com_error(ce, "Failed to initialize duplicator") from ce

    def _create_duplication(
        self, output: Output, device: Device
    ) -> Tuple[ctypes.POINTER(IDXGIOutputDuplication), bool]:
        """
        Create the output duplication object.

        Prefers ``IDXGIOutput5.DuplicateOutput1``, which accepts an explicit
        list of formats the caller can consume (needed for HDR/10-bit desktops)
        and reports protected-content refusals distinguishably. Falls back to
        the legacy ``IDXGIOutput1.DuplicateOutput`` when IDXGIOutput5 is not
        available (pre-Windows 10 1607 / older drivers) or when
        ``RAPIDSHOT_DUPLICATE_OUTPUT=legacy`` is set.

        Returns:
            Tuple of (duplication interface, whether DuplicateOutput1 was used)
        """
        if _legacy_duplication_forced():
            logger.info(
                f"{_DUPLICATE_OUTPUT_ENV} requests the legacy path; "
                "skipping DuplicateOutput1."
            )
        else:
            try:
                output5 = output.output.QueryInterface(IDXGIOutput5)
            except comtypes.COMError as ce:
                logger.info(
                    "IDXGIOutput5 unavailable "
                    f"(HRESULT {_format_hresult(ce.args[0] if ce.args else None)}); "
                    "falling back to legacy DuplicateOutput."
                )
            else:
                formats = (ctypes.c_uint * len(DUPLICATE_OUTPUT1_FORMATS))(
                    *DUPLICATE_OUTPUT1_FORMATS
                )
                duplicator = ctypes.POINTER(IDXGIOutputDuplication)()
                try:
                    output5.DuplicateOutput1(
                        device.device,
                        0,  # Flags: reserved, must be zero
                        len(DUPLICATE_OUTPUT1_FORMATS),
                        formats,
                        ctypes.byref(duplicator),
                    )
                    return duplicator, True
                except comtypes.COMError as ce:
                    hresult = ce.args[0] if ce.args else None
                    if hresult in DXGI_PROTECTED_CONTENT_ERRORS:
                        # Protected surfaces are a permanent refusal for this
                        # output, not something the legacy path can work around.
                        self.protected_content_detected = True
                        raise RapidShotProtectedContentError(
                            "Desktop duplication was denied because protected "
                            "(HDCP/DRM) content is on screen. Close or move the "
                            "protected player window and retry.",
                            hresult=hresult,
                        ) from ce
                    logger.warning(
                        "DuplicateOutput1 failed (HRESULT "
                        f"{_format_hresult(hresult)}); falling back to legacy "
                        "DuplicateOutput."
                    )

        duplicator = ctypes.POINTER(IDXGIOutputDuplication)()
        output.output.DuplicateOutput(device.device, ctypes.byref(duplicator))
        return duplicator, False

    def _map_com_error(self, ce: "comtypes.COMError", context: str) -> RapidShotError:
        """
        Translate a COMError into the matching RapidShot exception type.

        Centralised so every DXGI entry point classifies the same HRESULT the
        same way: recoverable (rebuild duplication), device (rebuild device),
        protected content, or configuration.
        """
        hresult = ce.args[0] if ce.args else None
        detail = f"{context}: {ce}"

        if hresult in DXGI_PROTECTED_CONTENT_ERRORS:
            self.protected_content_detected = True
            return RapidShotProtectedContentError(
                f"{detail} (protected/HDCP content is blocking duplication)",
                hresult=hresult,
            )
        if hresult in DXGI_DEVICE_ERRORS:
            return RapidShotDeviceError(detail, hresult=hresult)
        if hresult in DXGI_RECOVERABLE_ERRORS:
            return RapidShotReinitError(detail, hresult=hresult)
        if hresult in (DXGI_ERROR_INVALID_CALL, DXGI_ERROR_UNSUPPORTED):
            return RapidShotConfigError(detail)
        return RapidShotDXGIError(detail, hresult=hresult)

    def update_frame(self) -> bool:
        """
        Update the frame and cursor state.

        Sets ``self.updated`` to True if new frame content is available, False
        otherwise (timeout, or an acquire that carried only a cursor move).

        Returns:
            True if the duplication object is still healthy — including the
            timeout case, which is normal on a static desktop. Callers must read
            ``self.updated`` to know whether a frame is actually present; a
            False return means duplication is degraded and the frame path should
            be skipped.

        Raises:
            RapidShotReinitError: duplication must be rebuilt (access lost,
                session disconnected, display mode change).
            RapidShotDeviceError: the D3D device is gone and must be recreated.
            RapidShotProtectedContentError: protected content blocks capture.
            RapidShotDXGIError: any other unexpected DXGI failure.
        """
        # Reset state for this update attempt
        self.updated = False
        self.last_error = ""
        self._frame_acquired = False

        info = DXGI_OUTDUPL_FRAME_INFO()
        res = ctypes.POINTER(IDXGIResource)()
        frame_acquired = False

        if self.duplicator is None:
            self.last_error = "update_frame called on a released duplicator"
            logger.debug(self.last_error)
            return False

        try:
            # Acquire the next frame with a short timeout
            self.duplicator.AcquireNextFrame(
                self.timeout_ms,
                ctypes.byref(info),
                ctypes.byref(res),
            )
            frame_acquired = True
            self._frame_acquired = True
            logger.debug("Frame acquired successfully")

            # Protected (HDCP/DRM) content is blanked out by the compositor
            # rather than failing the acquire. Surface it once so callers can
            # tell "black frame" apart from "protected content was masked out".
            if info.ProtectedContentMaskedOut:
                if not self.protected_content_detected:
                    logger.warning(
                        "Protected (HDCP/DRM) content is on screen; the affected "
                        "region is blanked out by the OS in captured frames."
                    )
                self.protected_content_detected = True
            else:
                self.protected_content_detected = False

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

            self.last_present_time = last_present_time
            self.accumulated_frames = info.AccumulatedFrames
            # Read while the frame is still acquired: the metadata belongs to
            # this frame and is gone after ReleaseFrame.
            self.dirty_rects = self.get_frame_dirty_rects(info)
            self.rects_coalesced = bool(info.RectsCoalesced)

            # Process the frame
            try:
                # Drop the previous desktop image before taking a new reference.
                # DXGI refuses the next AcquireNextFrame with
                # DXGI_ERROR_INVALID_CALL while any reference to the prior
                # frame's surface is still outstanding, so a stale self.texture
                # stalls capture after the first couple of frames.
                self.texture = None
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
            self.last_error = (
                f"COMError in update_frame: {ce} "
                f"(HRESULT: {_format_hresult(hresult)})"
            )

            if hresult == DXGI_ERROR_WAIT_TIMEOUT:
                # Normal on a static desktop, not an error worth warning about.
                logger.debug("Frame acquisition timed out.")
                self.updated = False  # No new frame
                return True  # finally: still runs and releases the resource

            logger.warning(self.last_error)

            if hresult in DXGI_RECOVERABLE_ERRORS:
                # Access lost / session disconnected / mode change in progress.
                # The duplication object is dead: drop it here so the caller
                # cannot keep issuing calls against a stale interface, and so a
                # rebuild starts from a clean slate.
                if hresult == DXGI_ERROR_SESSION_DISCONNECTED:
                    reason = "Session disconnected (RDP/fast user switch)"
                else:
                    reason = "Access lost"
                self._release_duplication()
                raise RapidShotReinitError(
                    f"{reason}, re-initialization needed: {ce}", hresult=hresult
                ) from ce

            if hresult in DXGI_DEVICE_ERRORS:
                self._release_duplication()
                raise RapidShotDeviceError(
                    f"Device error, re-initialization needed: {ce}", hresult=hresult
                ) from ce

            raise self._map_com_error(ce, "Unexpected DXGI error in update_frame") from ce

        except Exception as e:
            # Catch any other unexpected Python exceptions to ensure cleanup
            self.last_error = f"Python exception in update_frame: {e}"
            logger.error(self.last_error)
            self.updated = False # Ensure updated is False on other exceptions
            raise RapidShotError(f"Unhandled Python exception in update_frame: {e}") from e # Wrap in RapidShotError
        
        finally:
            # The intermediate IDXGIResource is a comtypes COM pointer, so its
            # reference is dropped automatically when `res` goes out of scope.
            # Calling Release() by hand here (as this used to) decremented the
            # count a second time and corrupted the desktop surface's refcount.
            res = None

    # Add this method to provide compatibility with capture.py
    def get_frame(self):
        """
        Get the current frame - wrapper for update_frame for API compatibility
        
        Returns:
            Frame information or None if no update
        """
        if self.update_frame():
            try:
                if not self.updated:
                    return None

                # Create a simple frame information object with expected attributes
                class FrameInfo:
                    def __init__(self, rect, width, height, cursor_visible=False):
                        self.rect = rect
                        self.width = width
                        self.height = height
                        self.cursor_visible = cursor_visible

                return FrameInfo(
                    rect=self.texture,  # Use texture directly as rect
                    width=self._output_width,
                    height=self._output_height,
                    cursor_visible=self.cursor_visible
                )
            finally:
                if self._frame_acquired:
                    self.release_frame()
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
        if not self._frame_acquired:
            logger.debug("ReleaseFrame called with no active frame")
            return

        # The acquired texture is only valid between AcquireNextFrame and
        # ReleaseFrame. Drop our reference first so DXGI sees no outstanding
        # references to the desktop image; comtypes issues the COM Release.
        self.texture = None

        if self.duplicator is None:
            self._frame_acquired = False
            return

        try:
            self.duplicator.ReleaseFrame()
            logger.debug("Frame released")
        except comtypes.COMError as ce:
            hresult = ce.args[0] if ce.args else None
            # Don't log as warning for specific known error code
            if hresult == DXGI_ERROR_INVALID_CALL:
                logger.debug(f"Frame already released: {ce}")
            else:
                logger.warning(
                    f"Failed to release frame: {ce} "
                    f"(HRESULT: {_format_hresult(hresult)})"
                )
                # Not raising custom error here as it's a cleanup step, but logging is important.
                # If specific HRESULTs here are critical, they could be mapped.
                self.last_error = f"Failed to release frame: {ce}" # Keep last_error for simple errors
        except Exception as e: # Catch non-COM errors during ReleaseFrame
            logger.warning(f"Unexpected Python error releasing frame: {e}")
            self.last_error = f"Unexpected Python error releasing frame: {e}"
        finally:
            # Always clear the flag: if ReleaseFrame failed we must not retry it
            # against the same frame, and holding the flag would make every
            # later acquire look like a leak.
            self._frame_acquired = False

    def _release_duplication(self) -> None:
        """
        Drop the duplication interface without touching device/output state.

        Used on access-lost style failures so no further calls are issued
        against an interface DXGI has already invalidated.
        """
        duplicator, self.duplicator = self.duplicator, None
        self._frame_acquired = False
        self.updated = False
        self.texture = None
        if duplicator is None:
            return
        try:
            duplicator.Release()
        except Exception as e:
            logger.debug(f"Ignoring error releasing invalidated duplication: {e}")


    def release(self) -> None:
        """
        Release all duplicator resources.
        """
        if self.duplicator is not None:
            # Drop any frame still held, otherwise DXGI can keep the desktop
            # surface pinned after the duplication object goes away.
            if self._frame_acquired:
                self.release_frame()
            try:
                self.duplicator.Release()
                logger.info("Duplicator resources released.")
            except comtypes.COMError as ce:
                hresult = ce.args[0] if ce.args else None
                error_msg = (
                    f"Failed to release duplicator: {ce} "
                    f"(HRESULT: {_format_hresult(hresult)})"
                )
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
                self._frame_acquired = False

    def get_frame_dirty_rects(self, frame_info) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        Regions the compositor redrew in this frame, in desktop coordinates.

        Must be called while the frame is acquired — between AcquireNextFrame
        and ReleaseFrame — because the metadata belongs to that frame.

        ``TotalMetadataBufferSize`` is an upper bound covering move rects *and*
        dirty rects together, so sizing the buffer from it is always safe; DXGI
        reports how much it actually used.

        Returns:
            A list of ``(left, top, right, bottom)`` tuples. Empty means the
            frame carried no dirty-rect metadata, which is not the same as
            "nothing changed": a mode change or a driver that coalesces
            aggressively can report none while the whole screen differs. None is
            returned when the metadata could not be read at all, so callers can
            tell "no rects" from "unknown".
        """
        if self.duplicator is None or not self._frame_acquired:
            return None

        capacity = getattr(frame_info, "TotalMetadataBufferSize", 0)
        if not capacity:
            return []

        rect_size = ctypes.sizeof(RECT)

        try:
            for _ in range(2):
                count = max(1, capacity // rect_size)
                buffer = (RECT * count)()
                used = ctypes.c_uint(0)
                try:
                    self.duplicator.GetFrameDirtyRects(
                        count * rect_size,
                        ctypes.cast(buffer, ctypes.POINTER(RECT)),
                        ctypes.byref(used),
                    )
                except comtypes.COMError as ce:
                    hresult = ce.args[0] if ce.args else None
                    if hresult == DXGI_ERROR_MORE_DATA:
                        # DXGI wrote nothing and told us the size it needs.
                        # Retry once at that size rather than looping blindly.
                        capacity = used.value
                        if capacity == 0:
                            return None
                        continue
                    raise

                return [
                    (r.left, r.top, r.right, r.bottom)
                    for r in buffer[: used.value // rect_size]
                ]
            return None
        except comtypes.COMError as ce:
            hresult = ce.args[0] if ce.args else None
            self.last_error = (
                f"COMError in get_frame_dirty_rects: {ce} "
                f"(HRESULT: {_format_hresult(hresult)})"
            )
            logger.debug(self.last_error)
            if hresult in DXGI_RECOVERABLE_ERRORS or hresult in DXGI_DEVICE_ERRORS:
                # These mean the duplicator is dead; update_frame's handler owns
                # the recovery, so do not swallow them here.
                raise
            return None

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
            self.last_error = (
                f"COMError in get_frame_pointer_shape: {ce} "
                f"(HRESULT: {_format_hresult(hresult)})"
            )
            logger.warning(self.last_error)

            if hresult == DXGI_ERROR_ACCESS_LOST:
                # This specific error should propagate as it requires re-initialization.
                # Caller (update_frame) will handle this by raising RapidShotReinitError.
                # For now, let this COMError propagate up to update_frame's handler.
                raise # Re-raise to be caught by update_frame's COMError handler
            elif hresult == DXGI_ERROR_NOT_FOUND:
                # This is a common case, not necessarily a critical error for the duplicator itself.
                return False, False, (
                    f"Cursor shape not found (HRESULT: {_format_hresult(hresult)})"
                )
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