"""The Frame object: a captured frame with an explicit GPU texture lifetime.

Rapidshot's CPU path (:meth:`ScreenCapture.grab`) hands back a NumPy array that
the caller owns indefinitely. That is the right shape for screenshots, but it
forces the frame through a CPU round-trip -- read the mapped staging surface,
convert pixels -- which measures around 7.6 ms at 1920x1080 and is the single
largest per-frame cost in the library.

A ``Frame`` skips that. It hands back the GPU texture DXGI already produced, so
callers who are going to hand the pixels to a GPU consumer (an inference
runtime, a hardware encoder) never pay to bring them down to the CPU and back.

The catch, and the reason this type exists at all
------------------------------------------------
The duplicated desktop surface is only valid between ``AcquireNextFrame`` and
``ReleaseFrame``. DXGI refuses the *next* acquire with
``DXGI_ERROR_INVALID_CALL`` while any reference to the previous surface is still
outstanding -- this is not a leak that degrades gracefully, it stalls capture
completely after one frame. (Rapidshot shipped exactly this bug before Stage 1:
a stale texture reference stopped capture dead after two frames.)

So the texture lifetime cannot be left to the garbage collector. ``Frame`` makes
it explicit and, ideally, scoped::

    with camera.grab_frame() as frame:
        texture = frame.d3d11_texture      # valid only in here
        upload_to_model(texture)
    # released on exit; the next capture can proceed

Using a ``Frame`` outside its window raises :class:`FrameReleasedError` with a
clear message rather than letting DXGI fail opaquely later.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Any, List, Optional, Tuple

from rapidshot.util.errors import RapidShotError

logger = logging.getLogger(__name__)

# QPC ticks per second, for converting LastPresentTime to seconds.
_qpc_frequency: Optional[int] = None


def _qpc_freq() -> int:
    global _qpc_frequency
    if _qpc_frequency is None:
        freq = ctypes.c_longlong()
        ctypes.windll.kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        _qpc_frequency = freq.value or 1
    return _qpc_frequency


class FrameReleasedError(RapidShotError):
    """
    Raised when a Frame's GPU resources are touched after release.

    Holding the desktop texture past its acquire window stalls capture entirely,
    so this is reported as an error rather than tolerated.
    """


class FrameQuarantinedError(RapidShotError):
    """Raised when releasing a frame could invalidate untracked GPU work."""


class Frame:
    """
    A captured frame whose GPU texture is valid for a bounded window.

    Prefer the context-manager form; call :meth:`release` directly only when the
    lifetime genuinely cannot be scoped.

    Attributes are metadata and stay readable after release; only
    :attr:`d3d11_texture` and GPU-side operations become invalid.
    """

    __slots__ = (
        "_texture", "_on_release", "_released", "_region", "_rotation_angle",
        "_present_time_qpc", "_accumulated_frames", "_protected_content",
        "_cursor_visible", "_width", "_height", "_dirty_rects",
        "_rects_coalesced", "_source_id", "_release_drains",
        "_release_quarantine",
    )

    def __init__(
        self,
        texture,
        on_release,
        region: Tuple[int, int, int, int],
        rotation_angle: int = 0,
        present_time_qpc: int = 0,
        accumulated_frames: int = 0,
        protected_content: bool = False,
        cursor_visible: bool = False,
        dirty_rects: Optional[List[Tuple[int, int, int, int]]] = None,
        rects_coalesced: bool = False,
        source_id: int = 0,
    ) -> None:
        self._texture = texture
        self._on_release = on_release
        self._released = False
        # Work that must finish before the surface goes back to DXGI.
        # See defer_release_until().
        self._release_drains = []
        self._release_quarantine = None
        self._region = region
        self._rotation_angle = rotation_angle
        self._present_time_qpc = present_time_qpc
        self._accumulated_frames = accumulated_frames
        self._protected_content = protected_content
        self._cursor_visible = cursor_visible
        self._width = region[2] - region[0]
        self._height = region[3] - region[1]
        self._dirty_rects = self._clip_to_region(dirty_rects)
        self._rects_coalesced = rects_coalesced
        self._source_id = source_id

    def _clip_to_region(self, rects):
        """Translate desktop-coordinate rects into this frame's coordinates.

        DXGI reports dirty rects relative to the whole duplicated output, but a
        Frame may cover only a region of it. Handing back raw desktop
        coordinates would make ``frame.dirty_rects`` index outside the frame
        whenever a region is in use — an easy bug to write and a hard one to
        see, since it only misbehaves off-origin.

        Rects that miss the region entirely are dropped; rects that straddle its
        edge are clipped to it.
        """
        if rects is None:
            return None
        left, top, right, bottom = self._region
        clipped = []
        for rl, rt, rr, rb in rects:
            nl, nt = max(rl, left), max(rt, top)
            nr, nb = min(rr, right), min(rb, bottom)
            if nl < nr and nt < nb:
                clipped.append((nl - left, nt - top, nr - left, nb - top))
        return clipped

    # -- GPU resource ------------------------------------------------------

    @property
    def d3d11_texture(self):
        """
        The ``ID3D11Texture2D`` holding this frame, as a comtypes pointer.

        Valid only until :meth:`release`. Do not store it beyond that -- the
        next capture cannot start while a reference is outstanding.
        """
        self._check_live("d3d11_texture")
        return self._texture

    @property
    def released(self) -> bool:
        """True once the GPU texture has been handed back to DXGI."""
        return self._released

    def _check_live(self, what: str) -> None:
        if self._released:
            raise FrameReleasedError(
                f"Frame.{what} was accessed after the frame was released. The "
                "desktop texture is only valid between acquire and release; "
                "capture cannot proceed while a reference is held, so it is "
                "released as soon as the frame's scope ends. Copy what you need "
                "inside the `with` block, or call to_numpy() to take an owned "
                "CPU copy."
            )

    # -- metadata (remains valid after release) ----------------------------

    @property
    def region(self) -> Tuple[int, int, int, int]:
        """Captured region as (left, top, right, bottom)."""
        return self._region

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def rotation_angle(self) -> int:
        """Display rotation in degrees (0, 90, 180, 270)."""
        return self._rotation_angle

    @property
    def timestamp_qpc(self) -> int:
        """
        Raw ``LastPresentTime`` (QueryPerformanceCounter ticks).

        This is when the compositor presented the frame, not when it was
        captured, so it is the right clock for measuring capture latency.
        """
        return self._present_time_qpc

    @property
    def timestamp(self) -> float:
        """:attr:`timestamp_qpc` converted to seconds."""
        return self._present_time_qpc / _qpc_freq()

    @property
    def accumulated_frames(self) -> int:
        """
        Display updates DXGI coalesced into this frame.

        Greater than 1 means the consumer is falling behind the display and
        intermediate frames were dropped by the OS.
        """
        return self._accumulated_frames

    @property
    def source_id(self) -> int:
        """Which duplicator produced this frame's texture.

        Consumers caching anything keyed on the texture address need this too:
        COM addresses are recycled, so a released surface and a later unrelated
        one can share a pointer. The pair is a sound identity; the pointer
        alone is not.
        """
        return self._source_id

    @property
    def protected_content(self) -> bool:
        """True if HDCP/DRM content was blanked out of this frame by the OS."""
        return self._protected_content

    @property
    def cursor_visible(self) -> bool:
        return self._cursor_visible

    @property
    def dirty_rects(self) -> Optional[List[Tuple[int, int, int, int]]]:
        """Regions the compositor redrew, as ``(left, top, right, bottom)``.

        Coordinates are relative to this frame, not to the desktop, so they
        index straight into the captured image even when a region is in use.

        **An empty list does not mean nothing changed.** It means the frame
        carried no dirty-rect metadata: a mode change, a driver that declines to
        report them, or a full-surface update can all produce that while the
        image differs completely. Treat empty as "assume everything changed".
        ``None`` means the metadata could not be read at all.

        See :attr:`rects_coalesced` before using these to skip work.
        """
        return self._dirty_rects

    @property
    def rects_coalesced(self) -> bool:
        """True if the driver merged dirty rects instead of listing them.

        The regions are then an over-estimate: correct to redraw, but they may
        cover more than actually changed, so they are a weaker basis for
        skipping work.
        """
        return self._rects_coalesced

    # -- lifetime ----------------------------------------------------------

    def defer_release_until(self, drain, *, quarantine_on_failure=False) -> None:
        """Register work that must complete before the surface is handed back.

        The duplicated surface is only valid between ``AcquireNextFrame`` and
        ``ReleaseFrame``, and a GPU copy reading it does not stop when Python
        leaves the ``with`` block. An asynchronous consumer -- such as
        :meth:`CrossAdapterTransfer.transfer_async` -- therefore registers a
        drain here, and :meth:`release` runs it first.

        Without this, ``with camera.grab_frame() as frame:`` around an async
        submit releases the surface mid-copy. DXGI is then free to recycle it,
        and the copy lands a blend of two frames. Nothing raises; the pixels
        are simply wrong, which is the failure mode this project treats as the
        worst kind.

        `drain` is called at most once, before the surface is released, and
        must be idempotent-safe to call after the work already finished.

        ``quarantine_on_failure`` is for drains that prove submitted GPU work
        no longer reads this surface. If such a drain fails, releasing would
        invalidate the texture while that work may still be live, so the frame
        remains acquired and release raises :class:`FrameQuarantinedError`.
        Ordinary cleanup callbacks keep the historical log-and-release policy.
        """
        self._release_drains.append((drain, bool(quarantine_on_failure)))

    def _quarantine_release(self, reason: str) -> None:
        """Keep this DXGI surface acquired because submitted work is untracked.

        This is narrower than an ordinary failing drain. A drain failure still
        releases so capture does not stall; here D3D12 accepted a command list
        and both completion signals failed while the device remained live.
        DXGI says the surface becomes invalid after ReleaseFrame, so releasing
        would trade a visible capture stall for silent corruption or device
        removal. The native transfer intentionally preserves its resources for
        the same reason.
        """
        if not self._released:
            self._release_quarantine = str(reason)

    def release(self) -> None:
        """
        Hand the texture back to DXGI. Idempotent.

        Until this runs, the next capture cannot acquire a frame. Any drains
        registered by :meth:`defer_release_until` run first -- the surface must
        not go back while a GPU copy is still reading it.
        """
        if self._released:
            return
        if self._release_quarantine is not None:
            raise FrameQuarantinedError(
                "Frame cannot be released because a GPU submission could not "
                "be tracked to completion. Capture is intentionally stopped "
                "to preserve the DXGI surface; restart the process. "
                f"Native failure: {self._release_quarantine}"
            )
        # Before the flag flips: a drain that raises must not leave the frame
        # marked released while the surface is still held.
        drains, self._release_drains = self._release_drains, []
        for drain, quarantine_on_failure in drains:
            try:
                drain()
            except Exception as exc:
                if quarantine_on_failure:
                    self._quarantine_release(str(exc))
                    raise FrameQuarantinedError(
                        "Frame cannot be released because its asynchronous GPU "
                        "copy could not be drained. Capture is intentionally "
                        "stopped to preserve the DXGI surface; restart the "
                        f"process. Native failure: {exc}"
                    ) from exc
                # A generic cleanup failure is not evidence that native GPU
                # work still references the duplication surface.
                logger.warning(
                    "A release drain failed; releasing the surface anyway. "
                    "A GPU copy may still have been reading it: %s", exc
                )
        self._released = True
        self._texture = None
        on_release, self._on_release = self._on_release, None
        if on_release is not None:
            on_release()

    def __enter__(self) -> "Frame":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False

    def __del__(self):
        # A safety net, not the intended path: by the time the collector runs,
        # capture has already been stalled for an unbounded period.
        if not self._released:
            logger.warning(
                "Frame was garbage-collected without being released. Capture is "
                "blocked until a frame is released, so use `with "
                "camera.grab_frame() as frame:` or call frame.release()."
            )
            try:
                self.release()
            except Exception:
                pass

    def __repr__(self) -> str:
        state = "released" if self._released else "live"
        return (f"<Frame {self._width}x{self._height} {state} "
                f"accumulated={self._accumulated_frames}"
                f"{' protected' if self._protected_content else ''}>")
