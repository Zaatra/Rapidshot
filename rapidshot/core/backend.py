"""The contract between ``ScreenCapture`` and whatever delivers its pixels.

``ScreenCapture`` was written against Desktop Duplication, and this module
writes down exactly what it relies on, so a second backend
(Windows.Graphics.Capture, ROADMAP § 7.3) has something to meet other than
"behave like ``Duplicator``". ``Duplicator`` satisfies both protocols as it
stands; ``tests/test_capture_backend.py`` keeps it that way.

Nothing here is new behaviour. It is the existing contract, stated.
"""
from typing import Any, List, Optional, Protocol, Tuple, runtime_checkable

__all__ = ["CaptureBackend", "SourceGeometry", "BACKENDS"]

#: Capture backends a caller can name, plus ``"auto"`` for the planner's pick.
BACKENDS = ("auto", "dxgi", "wgc")


@runtime_checkable
class SourceGeometry(Protocol):
    """Where the captured texture sits and how it is oriented.

    ``core.output.Output`` is the monitor implementation. Regions are given in
    source coordinates, ``(0, 0)`` to ``resolution``; the texture may be stored
    rotated (``surface_size``, ``rotation_angle``), which the CPU paths undo.
    """

    @property
    def resolution(self) -> Tuple[int, int]:
        """Width and height in source coordinates."""

    @property
    def surface_size(self) -> Tuple[int, int]:
        """Width and height of the texture as stored (swapped at 90/270)."""

    @property
    def rotation_angle(self) -> int:
        """0, 90, 180 or 270: how the texture is rotated from the source."""

    @property
    def devicename(self) -> str:
        """GDI name of the display the source is on (``\\\\.\\DISPLAY1``), which
        is how HDR state and SDR white are looked up (``core.hdr``)."""

    def update_desc(self) -> None:
        """Refresh the above after a mode change, resize or move."""


@runtime_checkable
class CaptureBackend(Protocol):
    """One acquisition stream: acquire a texture, hold it, give it back.

    **Lifetime.** ``update_frame()`` may acquire a frame; while one is held
    (``_frame_acquired``), ``texture`` is valid and nothing else may be
    acquired. ``release_frame()`` gives it back and invalidates ``texture``. It
    is idempotent and safe after ``release()``, because it doubles as
    ``Frame.on_release`` and can run after a rebuild has replaced the backend.

    **Errors** from ``update_frame()`` keep the meanings ``ScreenCapture``
    acts on: ``RapidShotReinitError`` (access lost: rebuild),
    ``RapidShotDeviceError`` (device removed: rebuild),
    ``RapidShotProtectedContentError`` (nothing to retry while it is on screen),
    ``RapidShotConfigError`` (setup refused; an ``hresult`` of
    ``DXGI_ERROR_UNSUPPORTED`` or ``DXGI_ERROR_INVALID_CALL`` at construction
    means "try the next adapter"), and ``RapidShotError`` for anything else.

    **Identity.** ``instance_id`` is unique per backend object in the process;
    it becomes ``Frame.source_id``, which GPU consumers pair with the texture
    address because COM addresses are recycled. ``frame_serial`` counts
    new-content frames, so ``(instance_id, frame_serial)`` names a frame and
    consecutive serials mean nothing was skipped.

    **Metadata** a backend cannot provide is ``None``, never a guess:
    ``dirty_rects``/``move_rects`` of ``None`` mean "unknown" (forcing a full
    conversion) where ``[]`` means "nothing changed".
    """

    # Acquisition state
    texture: Any
    updated: bool
    _frame_acquired: bool
    timeout_ms: int

    # Identity
    instance_id: int
    frame_serial: int

    # Per-frame metadata
    last_present_time: int
    accumulated_frames: int
    dirty_rects: Optional[List[Tuple[int, int, int, int]]]
    move_rects: Optional[List[Tuple[int, int, int, int, int, int]]]
    rects_coalesced: bool
    protected_content_detected: bool
    cursor_visible: bool
    cursor: Any

    def update_frame(self) -> bool:
        """Try to acquire the next frame.

        Resets ``updated`` and ``_frame_acquired`` first. Returns True while
        healthy -- a timeout included, with ``updated`` False -- and False once
        the backend is released or dead. ``updated`` True means new content in
        ``texture``; an acquire can also hold a frame with ``updated`` False
        (cursor-only), which still has to be released.
        """

    def release_frame(self) -> None:
        """Give back the held frame, if any. Idempotent."""

    def release(self) -> None:
        """Release the held frame and everything else. Idempotent."""
