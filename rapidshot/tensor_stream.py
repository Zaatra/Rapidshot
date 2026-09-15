"""`TensorStream` — capture straight to model input, as an iterator (ROADMAP § 7.2).

::

    stream = rapidshot.TensorStream(camera, size=(640, 640),
                                    dtype="float16", layout="NCHW")
    for tensor in stream:
        model(tensor.to_torch())

This is a convenience layer, and the value of one is in the hazards it closes
rather than the lines it saves. Written out by hand, the loop above has four
ways to be quietly wrong, and each is handled here:

1. **The frame must go back to DXGI before the next capture.** A frame held
   across iterations stalls capture completely. The stream releases each frame
   before yielding; the conversion has finished by then, because
   ``GpuConverter.process`` waits on the D3D12 fence.
2. **The buffer is reused, so CUDA work on the last tensor must finish before
   the next conversion overwrites it.** Nothing faults if it does not; the
   model reads half of one frame and half of the next. The stream calls
   :meth:`GpuTensor.sync` before each conversion (``sync=True``).
3. **``grab_frame()`` returns ``None`` for "nothing changed" and for
   "capture has permanently failed" alike.** A loop that waits for a frame
   hangs forever on the second. The stream checks the camera and raises.
4. **A rebuilt capture can land on a new adapter**, where the converter's
   device cannot open the surface. The stream rebuilds the converter once for
   a frame from a new duplicator, and only then.

**What it does not do.** It never owns the camera — releasing that stays the
caller's job — and it does not invent frames: Desktop Duplication reports only
*changed* content, so an idle desktop yields nothing, and ``timeout`` decides
how long that is allowed to last.
"""

from __future__ import annotations

import time
from typing import Any, List, Optional, Sequence, Tuple

from . import converter as _converter

#: A `grab_frame()` that returns faster than this did not block, so the wait
#: loop is spinning rather than being paced by the camera. The blocking default
#: is `timeout_ms=10`, an order of magnitude above it.
_BLOCKING_THRESHOLD_S = 0.001

#: Windows rounds any non-zero sleep up to roughly half a millisecond, so this
#: asks for the smallest thing that still yields the core. `sleep(0)` returns
#: in 0.2 us and leaves the loop burning a full core, which is the problem
#: rather than the fix.
_SPIN_YIELD_S = 0.00005

__all__ = ["TensorStream"]


class TensorStream:
    """Iterate over captured frames as converted GPU tensors.

    Args:
        camera: A camera from :func:`rapidshot.create`. Not owned: closing the
            stream leaves it open. Must not be in continuous capture
            (``camera.start()``), which owns the duplicator.
        size: ``(width, height)`` of each converted region.
        regions: optional ``(left, top, right, bottom)`` rectangles in frame
            coordinates, converted together each frame into an ``(N, ...)``
            batch. Settable between iterations via :attr:`regions`, e.g. to
            follow moving windows; the count may vary up to ``batch``.
        timeout: seconds to wait for a *changed* frame before raising
            :class:`TimeoutError`. ``None`` (default) waits indefinitely.
        sync: call :meth:`GpuTensor.sync` before each conversion (default
            ``True``). A no-op until the tensor has been exported to CUDA.
            Turn off only when you synchronise yourself.
        **options: passed to :class:`GpuConverter` — ``dtype``, ``layout``,
            ``sampling``, ``normalize``, ``bgr``, ``crop``, ``batch``,
            ``pixel_format``, ``matrix``, ``full_range``. ``batch`` defaults
            to ``len(regions)``.

    Each yielded :class:`GpuTensor` is valid until the next iteration. It is the
    same object every time — unless the converter had to be rebuilt, which
    :attr:`rebuilds` counts; exports such as ``to_torch()`` must then be taken
    again from the new tensor.
    """

    def __init__(
        self,
        camera,
        size: Tuple[int, int],
        *,
        regions: Optional[Sequence[Tuple[int, int, int, int]]] = None,
        timeout: Optional[float] = None,
        sync: bool = True,
        **options: Any,
    ) -> None:
        if timeout is not None and not timeout > 0:
            raise ValueError(f"timeout must be positive seconds or None, got {timeout!r}")
        if regions is not None:
            regions = list(regions)
            if not regions:
                raise ValueError("regions must not be empty; pass None for the whole frame")
            options.setdefault("batch", len(regions))

        self._camera = camera
        self._size = (int(size[0]), int(size[1]))
        self._options = options
        self._timeout = timeout
        self._sync = bool(sync)
        self.regions: Optional[List[Tuple[int, int, int, int]]] = regions

        self._converter: Optional[_converter.GpuConverter] = None
        self._built_for_source: Optional[int] = None
        self._frame = None
        self._frames = 0
        self._rebuilds = 0
        self._closed = False

    # -- iteration --------------------------------------------------------

    def __iter__(self) -> "TensorStream":
        return self

    def __next__(self) -> "_converter.GpuTensor":
        if self._closed:
            raise StopIteration
        # Before the buffer is overwritten, not after it is filled: the CUDA
        # work to wait for is whatever the caller queued on the last tensor.
        if self._sync and self._converter is not None:
            self._converter._tensor.sync()

        frame = self._next_frame()
        with frame:
            tensor = self._convert(frame)
        # Released above, deliberately before the caller sees the tensor, so
        # holding the tensor across iterations cannot stall capture. A released
        # Frame's metadata stays readable; its texture does not.
        self._frame = frame
        self._frames += 1
        return tensor

    def _next_frame(self):
        deadline = None if self._timeout is None else time.monotonic() + self._timeout
        while True:
            if getattr(self._camera, "released", False):
                raise RuntimeError("the camera was released; the stream cannot continue")
            # Package-internal: grab_frame() reports permanent failure only by
            # returning None, the same value as "nothing changed on screen".
            if getattr(self._camera, "_capture_permanently_failed", False):
                reason = getattr(self._camera, "_last_capture_error_message", "") or "unknown"
                raise RuntimeError(f"capture has permanently failed: {reason}")

            started = time.perf_counter()
            frame = self._camera.grab_frame()
            if frame is not None:
                return frame
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    f"no changed frame in {self._timeout} s. Desktop Duplication "
                    "reports only changed content, so an idle screen yields "
                    "nothing; protected content also blanks capture."
                )
            # What paces this loop is the camera blocking inside
            # AcquireNextFrame. With `timeout_ms=0` it does not block, and this
            # measured 10.3 million calls a second against a still screen -- a
            # core burned re-asking a question whose answer had not changed.
            # Yield only when the call came back instantly, so the blocking
            # default keeps its latency and pays nothing for this.
            if time.perf_counter() - started < _BLOCKING_THRESHOLD_S:
                time.sleep(_SPIN_YIELD_S)

    def _convert(self, frame) -> "_converter.GpuTensor":
        source = getattr(frame, "source_id", None)
        if self._converter is None:
            self._build(frame)
        try:
            return self._process(frame)
        except RuntimeError:
            # Rebuild only for a frame from a *different* duplicator — the one
            # case where the converter's device may no longer reach the
            # surface. Any other failure is re-raised: retrying it would hide a
            # real bug behind a rebuild per frame.
            if source == self._built_for_source:
                raise
            self._build(frame)
            self._rebuilds += 1
            return self._process(frame)

    def _build(self, frame) -> None:
        self._converter = _converter.GpuConverter(frame, self._size, **self._options)
        self._built_for_source = getattr(frame, "source_id", None)

    def _process(self, frame):
        if self.regions is not None:
            return self._converter.process(frame, regions=self.regions)
        return self._converter.process(frame)

    # -- lifetime ---------------------------------------------------------

    def close(self) -> None:
        """Stop the stream and drop the converter. The camera stays open."""
        self._closed = True
        self._converter = None

    def __enter__(self) -> "TensorStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    # -- description ------------------------------------------------------

    @property
    def converter(self) -> "Optional[_converter.GpuConverter]":
        """The converter, or ``None`` before the first frame — it is built from
        a live frame, which picks the adapter."""
        return self._converter

    @property
    def frame(self):
        """The released :class:`Frame` behind the latest tensor, for metadata
        (``timestamp``, ``dirty_rects``, ``cursor``…). Its texture is gone."""
        return self._frame

    @property
    def frames(self) -> int:
        """Tensors yielded so far."""
        return self._frames

    @property
    def rebuilds(self) -> int:
        """Times the converter was rebuilt for a new duplicator."""
        return self._rebuilds

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self._frames} frames"
        return f"<TensorStream {self._size[0]}x{self._size[1]} {state}>"
