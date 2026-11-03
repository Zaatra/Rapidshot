"""Desktop Duplication through the native extension (``rapidshot-native`` >= 0.4.0).

``NativeDuplicator`` meets the same contract as ``Duplicator``
(``core/backend.py``) and is built the same way, so ``ScreenCapture`` drives
it unchanged. The duplication itself is ``native/core/src/duplicator.rs``, a
behaviour-for-behaviour port of ``core/duplicator.py``: the same formats
offered to ``DuplicateOutput1``, the same fallback, the same HRESULT classes,
the same lock-screen / UAC / Session 0 diagnosis of ``E_ACCESSDENIED``, the
same metadata limits. It exists so that the Rust remote-support helper and
this package capture through one implementation.

**It is opt-in while the two are compared.** ``RAPIDSHOT_DUPLICATOR=native``
makes ``ScreenCapture`` build this instead of ``Duplicator``; nothing else
changes which one runs. Whether it replaces the Python class is a separate
decision -- doing so would make the native wheel necessary for a plain
screenshot.

What differs from the Python class, and only this:

- A COM error reads the way windows-rs renders it ("text (0x887A0026)"),
  not comtypes' tuple. Exception classes and HRESULTs are the same.
- Notes the native side records during construction are lost if
  construction fails; the exception carries the cause.
"""
import ctypes
import logging
from typing import Optional

from rapidshot._libs.d3d11 import ID3D11Texture2D
from rapidshot._libs.dxgi import (
    DXGI_OUTDUPL_POINTER_POSITION,
    DXGI_OUTDUPL_POINTER_SHAPE_INFO,
)
from rapidshot.core.duplicator import Cursor, _duplicator_ids
from rapidshot.util.errors import (
    RapidShotConfigError,
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotError,
    RapidShotProtectedContentError,
    RapidShotReinitError,
)

# The Python class's logger, so a log reads the same whichever class wrote it.
logger = logging.getLogger("rapidshot.core.duplicator")

__all__ = ["NativeDuplicator", "is_available", "NATIVE_DUPLICATOR_ENV"]

#: Set to ``native`` to have ``ScreenCapture`` duplicate through the extension.
NATIVE_DUPLICATOR_ENV = "RAPIDSHOT_DUPLICATOR"

_ERRORS = {
    "reinit": RapidShotReinitError,
    "device": RapidShotDeviceError,
    "protected_content": RapidShotProtectedContentError,
    "config": RapidShotConfigError,
    "dxgi": RapidShotDXGIError,
}

_LEVELS = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING}


def _ext():
    from rapidshot import native

    return native._ext


def is_available() -> bool:
    """Whether the loaded extension has the native Desktop Duplication."""
    ext = _ext()
    return ext is not None and hasattr(ext, "DesktopDuplicator")


def _address(pointer) -> int:
    return int(ctypes.cast(pointer, ctypes.c_void_p).value or 0)


def _texture_from_address(address: int):
    """A comtypes texture for the held frame. The address is borrowed, and
    comtypes releases what it wraps, so the reference it gives back is taken
    here."""
    texture = ctypes.cast(address, ctypes.POINTER(ID3D11Texture2D))
    texture.AddRef()
    return texture


def _translate(error: Exception) -> RapidShotError:
    """A ``DuplicationError(kind, hresult, message)`` as the class it names."""
    kind, hresult, message = error.args
    return _ERRORS.get(kind, RapidShotDXGIError)(message, hresult=hresult)


class NativeDuplicator:
    """One native Desktop Duplication stream, satisfying ``CaptureBackend``.

    ``NativeDuplicator(output=, device=, timeout_ms=)``, as ``Duplicator``.
    """

    def __init__(self, output=None, device=None, timeout_ms: int = 10):
        ext = _ext()
        if ext is None or not hasattr(ext, "DesktopDuplicator"):
            from rapidshot import native

            raise RapidShotConfigError(
                f"{NATIVE_DUPLICATOR_ENV}=native needs rapidshot-native >= 0.4.0; "
                f"the loaded extension is {native.extension_version() or 'absent'}.")
        self.output = output
        self.device = device
        self.instance_id = next(_duplicator_ids)
        self.texture = None
        self.updated = False
        self._frame_acquired = False
        self.cursor = Cursor()
        self.last_error = ""
        try:
            self._native = ext.DesktopDuplicator(
                _address(output.output), _address(device.device), int(timeout_ms))
        except ext.DuplicationError as e:
            error = _translate(e)
            # The message already names the step, as Duplicator's does.
            logger.error(str(error))
            self.last_error = str(error)
            raise error from e
        self._drain_log()
        logger.info(
            f"Duplicator initialized for output: {output.devicename} "
            f"(DuplicateOutput1={self.used_duplicate_output1}, native)")

    # -- CaptureBackend -----------------------------------------------------

    @property
    def timeout_ms(self) -> int:
        return self._native.timeout_ms

    @timeout_ms.setter
    def timeout_ms(self, value: int) -> None:
        self._native.timeout_ms = int(value)

    def update_frame(self) -> bool:
        """``CaptureBackend.update_frame``; see ``Duplicator.update_frame``."""
        self.updated = False
        self._frame_acquired = False
        try:
            healthy = self._native.update_frame()
        except _ext().DuplicationError as e:
            self.texture = None
            self.last_error = self._native.last_error
            raise _translate(e) from e
        finally:
            self._drain_log()
        self.last_error = self._native.last_error
        self._frame_acquired = self._native.frame_acquired
        self.updated = self._native.updated
        address = self._native.texture_address if self.updated else 0
        self.texture = _texture_from_address(address) if address else None
        if self._frame_acquired:
            self._refresh_cursor()
        return healthy

    def release_frame(self) -> None:
        """``CaptureBackend.release_frame``. Idempotent."""
        # Our reference first: DXGI refuses the next acquire while any
        # reference to the released surface is outstanding.
        self.texture = None
        native = getattr(self, "_native", None)
        if native is not None:
            native.release_frame()
            self._drain_log()
        self._frame_acquired = False

    def release(self) -> None:
        """``CaptureBackend.release``. Idempotent."""
        self.texture = None
        native = getattr(self, "_native", None)
        if native is not None:
            native.release()
            self._drain_log()
        self._frame_acquired = False

    # -- state the contract and ScreenCapture read ---------------------------

    @property
    def duplicator(self):
        """Truthy while the duplication is alive, as ``Duplicator.duplicator``."""
        return None if self._native.released else self._native

    @property
    def frame_serial(self) -> int:
        return self._native.frame_serial

    @property
    def last_present_time(self) -> int:
        return self._native.last_present_time

    @property
    def accumulated_frames(self) -> int:
        return self._native.accumulated_frames

    @property
    def dirty_rects(self):
        return self._native.dirty_rects

    @property
    def move_rects(self):
        return self._native.move_rects

    @property
    def rects_coalesced(self) -> bool:
        return self._native.rects_coalesced

    @property
    def protected_content_detected(self) -> bool:
        return self._native.protected_content_detected

    @property
    def cursor_visible(self) -> bool:
        return self._native.cursor_visible

    @property
    def used_duplicate_output1(self) -> bool:
        return self._native.used_duplicate_output1

    def get_last_error(self) -> str:
        return self.last_error

    # -- internals -----------------------------------------------------------

    def _drain_log(self) -> None:
        native = getattr(self, "_native", None)
        if native is None:
            return
        for level, message in native.take_log():
            logger.log(_LEVELS.get(level, logging.DEBUG), message)

    def _refresh_cursor(self) -> None:
        """Rebuild ``cursor`` in the ctypes shapes ``Duplicator`` holds."""
        x, y, visible = self._native.pointer_position
        position = DXGI_OUTDUPL_POINTER_POSITION()
        position.Position.x, position.Position.y = x, y
        position.Visible = bool(visible)
        shape = self._native.pointer_shape
        if shape is None:
            self.cursor = Cursor(PointerPositionInfo=position)
            return
        kind, width, height, pitch, hot_x, hot_y, data = shape
        info = DXGI_OUTDUPL_POINTER_SHAPE_INFO()
        info.Type, info.Width, info.Height, info.Pitch = kind, width, height, pitch
        info.HotSpot.x, info.HotSpot.y = hot_x, hot_y
        buffer = (ctypes.c_byte * len(data)).from_buffer_copy(data) if data else None
        self.cursor = Cursor(PointerPositionInfo=position, PointerShapeInfo=info, Shape=buffer)

    def __repr__(self) -> str:
        alive = getattr(self, "_native", None) is not None and not self._native.released
        return f"<NativeDuplicator Initialized:{alive} id={self.instance_id}>"


def selected() -> bool:
    """True when the environment asks ``ScreenCapture`` for this class."""
    import os

    return os.environ.get(NATIVE_DUPLICATOR_ENV, "").strip().lower() == "native"
