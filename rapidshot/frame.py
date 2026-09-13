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
from threading import RLock
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


def _qpc_now() -> int:
    """Current QueryPerformanceCounter value, on the same clock as
    ``LastPresentTime``.

    Used by :attr:`Frame.age_ms`. QPC is a high-resolution interval timer, not a
    clock synchronised to anything external, so differences against a present
    time are meaningful while absolute values are not.
    """
    value = ctypes.c_longlong()
    ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(value))
    return value.value


class FrameReleasedError(RapidShotError):
    """
    Raised when a Frame's GPU resources are touched after release.

    Holding the desktop texture past its acquire window stalls capture entirely,
    so this is reported as an error rather than tolerated.
    """


class FrameQuarantinedError(RapidShotError):
    """Raised when releasing a frame could invalidate untracked GPU work."""


class CursorInfo:
    """Where the cursor is and what it looks like, as DXGI reported it.

    Desktop Duplication already carried all of this; only :attr:`visible` was
    ever surfaced, so callers wanting to draw or track a cursor had to reach
    into `camera.grab_cursor()` and the raw COM structures themselves.

    :attr:`shape` is the raw pointer image exactly as DXGI supplied it, in one
    of three encodings named by :attr:`shape_type` -- monochrome (a 1-bit AND
    mask stacked above a 1-bit XOR mask, so its height is twice the cursor's),
    colour (BGRA), or masked colour. Compositing correctly means handling all
    three, which is why this exposes the buffer and does not attempt to blend
    it for you.
    """

    __slots__ = ("visible", "position", "hotspot", "shape", "shape_type",
                 "shape_size", "shape_pitch")

    def __init__(self, visible=False, position=None, hotspot=None, shape=None,
                 shape_type=0, shape_size=None, shape_pitch=0):
        self.visible = bool(visible)
        #: ``(x, y)`` in desktop coordinates, or None if not reported.
        self.position = position
        #: ``(x, y)`` offset of the click point within the shape.
        self.hotspot = hotspot
        #: Raw pointer-shape bytes, or None when no shape has been sent.
        self.shape = shape
        #: DXGI_OUTDUPL_POINTER_SHAPE_TYPE: 1 monochrome, 2 colour, 4 masked.
        self.shape_type = int(shape_type)
        #: ``(width, height)`` of the shape buffer.
        self.shape_size = shape_size
        #: Row stride of :attr:`shape` in bytes.
        self.shape_pitch = int(shape_pitch)

    def __repr__(self) -> str:
        kind = {1: "monochrome", 2: "color", 4: "masked"}.get(self.shape_type, "none")
        return (f"CursorInfo(visible={self.visible}, position={self.position}, "
                f"hotspot={self.hotspot}, shape={kind}, size={self.shape_size})")


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
        "_release_quarantine", "_sequence", "_generation", "_cursor", "_lock",
        "_move_rects",
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
        move_rects: Optional[List[Tuple[int, int, int, int, int, int]]] = None,
        rects_coalesced: bool = False,
        source_id: int = 0,
        sequence: int = 0,
        generation: int = 0,
        cursor: "Optional[CursorInfo]" = None,
    ) -> None:
        # First, so __del__ can rely on it even if a later assignment raises.
        # Reentrant: __del__ and __exit__ both route through release(), and a
        # drain callback is arbitrary caller code.
        self._lock = RLock()
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
        self._move_rects = self._clip_move_rects(move_rects)
        self._rects_coalesced = rects_coalesced
        self._source_id = source_id
        self._sequence = sequence
        self._generation = generation
        self._cursor = cursor

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

    def _clip_move_rects(self, move_rects):
        """Clip move rects to this frame, keeping each source paired with its own
        destination.

        Clipping through :meth:`_clip_to_region` and zipping the result back
        against the input does not work: that helper drops rects which miss the
        region, so the two lists go out of step and a source point gets attached
        to some other rectangle's destination. Done here in one pass instead.

        Only the destination is translated into frame coordinates. The source
        point stays in desktop coordinates, because the pixels may have come
        from somewhere outside this frame's region entirely -- and clamping it
        would silently claim they came from somewhere they did not.
        """
        if move_rects is None:
            return None
        left, top, right, bottom = self._region
        clipped = []
        for source_x, source_y, rl, rt, rr, rb in move_rects:
            nl, nt = max(rl, left), max(rt, top)
            nr, nb = min(rr, right), min(rb, bottom)
            if nl < nr and nt < nb:
                clipped.append((source_x, source_y,
                                nl - left, nt - top, nr - left, nb - top))
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
    def sequence(self) -> int:
        """Monotonic index of this frame within its camera, from 1.

        Counts frames this camera *returned*, not frames the display presented,
        so gaps do not appear here when the compositor outruns the consumer --
        :attr:`accumulated_frames` is what reports that. Its use is correlating
        a frame with logs and with :attr:`generation` after a recovery.

        Continues across recoveries rather than restarting, so a sequence number
        identifies one frame for the life of the camera.
        """
        return self._sequence

    @property
    def generation(self) -> int:
        """How many times capture had rebuilt itself when this frame was taken.

        Starts at 0 and increments on every successful recovery -- access loss,
        a mode change, a monitor coming or going, a device reset. Two frames
        with different generations came from different duplicator instances and
        may differ in size, rotation or format, so anything cached from a frame
        (a resize table, a preprocessor, a cross-adapter transfer) must be
        rebuilt when this changes.

        `camera.recovery_count` and `camera.last_recovery_reason` say how often
        and why; this pins each frame to one side of that boundary.
        """
        return self._generation

    @property
    def move_rects(self) -> Optional[List[Tuple[int, int, int, int, int, int]]]:
        """Regions the compositor moved rather than redrew.

        Each entry is ``(source_x, source_y, left, top, right, bottom)``: where
        the pixels came from, in desktop coordinates, and the rectangle they now
        occupy, in this frame's coordinates. DXGI does not repeat these in
        :attr:`dirty_rects`, so anything patching a previous frame by dirty rect
        alone would leave these regions stale.

        Usually empty. Measured on Windows 11 across 3,768 frames of window
        dragging and page scrolling: zero move rects, with the metadata readable
        every time. A fully composited desktop has nothing left for a
        screen-to-screen blit to optimise. None means the metadata could not be
        read, the same distinction :attr:`dirty_rects` draws.
        """
        return self._move_rects

    @property
    def changed_fraction(self) -> Optional[float]:
        """Fraction of this frame's area that is new, 0.0 to 1.0.

        None when :attr:`dirty_rects` is None -- metadata could not be read, so
        nothing can be inferred. An **empty** rect list gives ``1.0``, not
        ``0.0``: no rects means no information, and the safe reading is that
        everything changed (see :attr:`dirty_rects`).

        Counts :attr:`move_rects` as well. A moved region carries content that
        was not there in the previous frame, and DXGI reports it *instead of* a
        dirty rect, so a consumer deciding how much work to do would otherwise
        be told a scroll changed nothing.

        Overlapping rects are counted once. The driver may still have merged
        regions before reporting them, in which case this over-estimates --
        check :attr:`rects_coalesced` before using it to skip work.
        """
        if self._dirty_rects is None:
            return None
        regions = list(self._dirty_rects)
        regions.extend(rect[2:] for rect in (self._move_rects or ()))
        if not regions:
            return 1.0
        total = self._width * self._height
        if total <= 0:
            return 0.0
        # Union by row spans, so overlapping rects are not double counted.
        events = []
        for left, top, right, bottom in regions:
            if right > left and bottom > top:
                events.append((top, 1, left, right))
                events.append((bottom, -1, left, right))
        if not events:
            return 0.0
        events.sort()
        active, area, previous = [], 0, events[0][0]
        for y, delta, left, right in events:
            if y > previous and active:
                spans = sorted(active)
                covered, end = 0, None
                start = None
                for s, e in spans:
                    if start is None:
                        start, end = s, e
                    elif s > end:
                        covered += end - start
                        start, end = s, e
                    else:
                        end = max(end, e)
                if start is not None:
                    covered += end - start
                area += covered * (y - previous)
            previous = y
            if delta > 0:
                active.append((left, right))
            else:
                active.remove((left, right))
        return min(area / total, 1.0)

    @property
    def age_ms(self) -> float:
        """Milliseconds since the compositor presented this frame.

        Measured against :attr:`timestamp_qpc`, so it is how old the *pixels*
        are right now -- not how long any call took. Read it late (just before
        handing the frame to a consumer) rather than at capture, since it grows
        while you hold the frame.

        Returns 0.0 when no present time was reported, which is not the same as
        a zero-age frame; check :attr:`timestamp_qpc` if the difference matters.
        """
        if not self._present_time_qpc:
            return 0.0
        return max(0.0, (_qpc_now() - self._present_time_qpc) * 1000.0 / _qpc_freq())

    @property
    def cursor(self) -> "CursorInfo":
        """Cursor position, hotspot and shape as of this frame.

        Always present; when capture reported nothing, its :attr:`visible` is
        False and the rest are None. :attr:`cursor_visible` remains as a
        shorthand for ``frame.cursor.visible``.
        """
        if self._cursor is None:
            self._cursor = CursorInfo(visible=self._cursor_visible)
        return self._cursor

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

        Registering is serialised against :meth:`release`, which takes the
        drain list away wholesale: without that, a drain registered on one
        thread while another released could be dropped without ever running,
        and the surface would go back to DXGI mid-copy.

        ``quarantine_on_failure`` is for drains that prove submitted GPU work
        no longer reads this surface. If such a drain fails, releasing would
        invalidate the texture while that work may still be live, so the frame
        remains acquired and release raises :class:`FrameQuarantinedError`.
        Ordinary cleanup callbacks keep the historical log-and-release policy.
        """
        with self._lock:
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

        Serialised, so two threads releasing the same frame cannot both get
        past the released check. That used to run every drain twice and call
        ``on_release`` twice, which hands the same DXGI frame back to
        ``ReleaseFrame`` twice.
        """
        with self._lock:
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
        # Outside the lock: on_release re-enters capture, which takes locks of
        # its own, and holding this one across it would order the two.
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
