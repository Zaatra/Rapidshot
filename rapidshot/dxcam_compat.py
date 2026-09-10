"""Run DXcam code against RapidShot by changing one import.

    import rapidshot.dxcam_compat as dxcam     # was: import dxcam

    camera = dxcam.create(output_color="BGR")
    frame = camera.grab()

Section 7.1 ranks this as the highest adoption return in the roadmap, and it is
also the dullest engineering in it: no new capability, just the API an existing
project already calls.

**The one semantic difference, handled here.** DXcam's ``grab()`` returns a plain
``numpy.ndarray``. RapidShot's returns a pooled buffer that must be released, and
reading one after release raises rather than returning stale pixels. Code written
against DXcam has no ``release()`` calls, so this layer copies each frame out and
releases immediately. That copy is the compatibility tax -- roughly the cost of
one frame memcpy per grab -- and it is why this is a migration aid rather than
the way to use RapidShot. Drop the shim and add ``release()`` to get the
performance back; ``benchmarks/ai_ingestion.py`` measures what that is worth.

Not emulated, because nothing sensible could be: DXcam internals such as
``camera._duplicator``. Anything reaching into those is not portable and this
layer does not pretend otherwise.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

import rapidshot as _rapidshot

__all__ = ["create", "device_info", "output_info", "reset", "clean_up", "DXCamera"]


class DXCamera:
    """A DXcam-shaped view of a RapidShot :class:`ScreenCapture`.

    Wraps rather than subclasses, so the RapidShot object stays reachable as
    :attr:`rapidshot_camera` for anything migrating incrementally -- the point
    of this layer is to be removable one call site at a time.
    """

    def __init__(self, camera) -> None:
        self.rapidshot_camera = camera
        # The frame handed out by the most recent *_view call. DXcam's views
        # stay valid until the next grab, so this is released exactly then --
        # which is also what returns the buffer to RapidShot's pool.
        self._outstanding = None
        self._released = False

    def _retire_view(self) -> None:
        frame, self._outstanding = self._outstanding, None
        if frame is not None:
            release = getattr(frame, "release", None)
            if release is not None:
                try:
                    release()
                except Exception:
                    # A view already invalidated by a rebuild is not an error
                    # worth propagating out of the next grab.
                    pass

    # -- capture ----------------------------------------------------------
    def grab(self, region: Optional[Tuple[int, int, int, int]] = None):
        """Capture one frame as a plain ndarray, or None if nothing changed.

        Returns a copy. RapidShot hands back a pooled buffer whose contents the
        next capture overwrites, and DXcam callers keep frames around, so
        handing the buffer through unchanged would corrupt data in code that
        looks correct.
        """
        self._retire_view()
        frame = (self.rapidshot_camera.grab(region=region) if region is not None
                 else self.rapidshot_camera.grab())
        if frame is None:
            return None
        try:
            # `np.asarray(...).copy()`, not `np.array(..., copy=True)`.
            # The latter delegates the decision to __array__, and any
            # buffer that accepts `copy` without honouring it then hands
            # back a view. Copying the view explicitly cannot be got
            # wrong by an implementation this layer does not control.
            return np.asarray(frame).copy()
        finally:
            release = getattr(frame, "release", None)
            if release is not None:
                release()

    def shot(self, image_ptr, region: Optional[Tuple[int, int, int, int]] = None):
        """DXcam's write-into-your-own-buffer call."""
        if region is not None:
            return self.rapidshot_camera.shot(image_ptr, region=region)
        return self.rapidshot_camera.shot(image_ptr)

    # -- continuous capture ----------------------------------------------
    def start(self, region: Optional[Tuple[int, int, int, int]] = None,
              target_fps: int = 60, video_mode: bool = False,
              delay: int = 0) -> None:
        kwargs: dict = {"target_fps": target_fps, "video_mode": video_mode}
        if region is not None:
            kwargs["region"] = region
        if delay:
            kwargs["delay"] = delay
        self.rapidshot_camera.start(**kwargs)

    def get_latest_frame(self):
        """Latest frame from continuous capture, as a plain ndarray.

        Copied for the same reason as :meth:`grab`.
        """
        self._retire_view()
        frame = self.rapidshot_camera.get_latest_frame()
        if frame is None:
            return None
        try:
            # `np.asarray(...).copy()`, not `np.array(..., copy=True)`.
            # The latter delegates the decision to __array__, and any
            # buffer that accepts `copy` without honouring it then hands
            # back a view. Copying the view explicitly cannot be got
            # wrong by an implementation this layer does not control.
            return np.asarray(frame).copy()
        finally:
            release = getattr(frame, "release", None)
            if release is not None:
                release()

    def stop(self) -> None:
        self._retire_view()
        self.rapidshot_camera.stop()

    def release(self) -> None:
        self._retire_view()
        self._released = True
        self.rapidshot_camera.release()

    def grab_view(self, region=None):
        """Zero-copy view of the frame, valid until the next grab.

        DXcam's contract, and RapidShot's pooled buffer has the same lifetime,
        so no copy is needed. The previous view is released here, which is what
        invalidates it -- reading a retired view raises `BufferReleasedError`
        rather than returning another frame's pixels, which is stricter than
        DXcam and the difference is worth having.
        """
        self._retire_view()
        frame = (self.rapidshot_camera.grab(region=region) if region is not None
                 else self.rapidshot_camera.grab())
        if frame is None:
            return None
        self._outstanding = frame
        return np.asarray(frame)

    def get_latest_frame_view(self):
        """Continuous-capture counterpart of :meth:`grab_view`."""
        self._retire_view()
        frame = self.rapidshot_camera.get_latest_frame()
        if frame is None:
            return None
        self._outstanding = frame
        return np.asarray(frame)

    @property
    def is_released(self) -> bool:
        return self._released

    @property
    def latest_frame_ticks(self) -> int:
        """QPC ticks when the compositor presented the latest frame, or 0."""
        return int(getattr(self.rapidshot_camera, "last_present_time", 0) or 0)

    @property
    def latest_frame_time(self) -> float:
        """:attr:`latest_frame_ticks` in seconds, or 0.0."""
        ticks = self.latest_frame_ticks
        if not ticks:
            return 0.0
        from rapidshot.frame import _qpc_freq

        return ticks / _qpc_freq()

    # -- attributes DXcam code reads --------------------------------------
    @property
    def width(self) -> int:
        return self.rapidshot_camera.width

    @property
    def height(self) -> int:
        return self.rapidshot_camera.height

    @property
    def channel_size(self) -> int:
        """DXcam's name for the channel count."""
        return self.rapidshot_camera.channels

    @property
    def region(self):
        return self.rapidshot_camera.region

    @property
    def is_capturing(self) -> bool:
        return bool(getattr(self.rapidshot_camera, "is_capturing", False))

    def __getattr__(self, name: str) -> Any:
        """Forward anything unrecognised to the wrapped camera.

        Deliberate: this layer cannot enumerate everything DXcam code touches,
        and forwarding fails with RapidShot's own AttributeError naming the real
        object, which is more useful than one naming the shim.
        """
        if name in ("_outstanding", "_released", "rapidshot_camera"):
            # Reached only if __init__ has not run yet (unpickling, a subclass
            # calling a method early). Forwarding would recurse forever.
            raise AttributeError(name)
        return getattr(self.rapidshot_camera, name)

    def __repr__(self) -> str:
        return f"<DXCamera (rapidshot compat) {self.rapidshot_camera!r}>"


def create(device_idx: int = 0, output_idx: Optional[int] = None,
           region: Optional[Tuple[int, int, int, int]] = None,
           output_color: str = "RGB", max_buffer_len: int = 64,
           **kwargs) -> DXCamera:
    """DXcam's ``create()``.

    ``max_buffer_len`` maps to RapidShot's ``max_buffer_len``. Unknown keywords
    are forwarded rather than rejected, so DXcam-specific arguments RapidShot
    also accepts (``nvidia_gpu``) keep working, and ones it does not fail with
    RapidShot's own error naming the argument.
    """
    options: dict = {"device_idx": device_idx, "output_color": output_color,
                     "max_buffer_len": max_buffer_len}
    if output_idx is not None:
        options["output_idx"] = output_idx
    if region is not None:
        options["region"] = region
    options.update(kwargs)
    return DXCamera(_rapidshot.create(**options))


def device_info() -> str:
    return _rapidshot.device_info()


def output_info() -> str:
    return _rapidshot.output_info()


def reset() -> None:
    _rapidshot.reset()


def clean_up() -> None:
    _rapidshot.clean_up()
