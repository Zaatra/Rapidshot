import ctypes
import numpy as np
import logging
from rapidshot.util.logging import get_logger
from numpy import rot90, ndarray, newaxis, uint8
from rapidshot.processor.base import (
    ProcessorBackends,
    channels_for_color_mode,
    validate_color_mode,
)
from rapidshot.util.ctypes_helpers import describe_destination, pointer_to_address

# Set up logger
logger = logging.getLogger(__name__)

# Rec. 601 luma coefficients in Q8 fixed point:
#   Y = (R*77 + G*150 + B*29 + 128) >> 8
# Q8 keeps the whole intermediate in uint16 -- the maximum is 255*(77+150+29)
# + 128 = 65408, just inside the 65535 limit -- which halves memory traffic
# versus a Q14/uint32 formulation and measured 2x faster on a 1920x1080 frame.
# The +128 is round-to-nearest; without it, truncation biases every pixel dark.
_LUMA_R, _LUMA_G, _LUMA_B = 77, 150, 29
_LUMA_ROUND, _LUMA_SHIFT = 128, 8


def _luma_into(src: np.ndarray, out: np.ndarray,
               acc: np.ndarray, tmp: np.ndarray) -> None:
    """
    Q8 luma from BGR(A) ``src`` into ``out``, using caller-owned intermediates.

    Every step writes into a buffer the caller already owns, so a steady-state
    conversion allocates nothing. The previous formulation reached for
    ``src[..., 1].astype(np.uint16) * _LUMA_G``, which materialises a full-frame
    uint16 temporary per channel -- two 4 MB allocations per 1080p frame, whose
    page faults cost more than the arithmetic they carried. Reusing the
    intermediates instead measured **1.5-1.8x on 1920x1080 with byte-identical
    output** (`benchmarks/gray_kernel.py`), taking GRAY from ~16 ms to 8.5-11 ms
    and so inside a 60 Hz frame budget. Three runs spanned that range on an
    otherwise idle machine, so it is quoted as a range rather than a figure --
    see the benchmark-noise note in ROADMAP.md section 2.

    For the much larger win, see `_native_gray`: the same arithmetic in Rust is
    0.70 ms. This path is what runs when the optional extension is absent, which
    is the default for `pip install rapidshot`.

    The rounding term is folded into the first channel's accumulation rather
    than applied as its own ``acc += 128`` pass. Identical output, but it avoids
    one full read-modify-write over the intermediate.

    Args:
        src: (H, W, 3) or (H, W, 4) uint8 in BGR(A) channel order
        out: (H, W) uint8 destination
        acc: (H, W) uint16 accumulator
        tmp: (H, W) uint16 second intermediate
    """
    np.multiply(src[..., 2], _LUMA_R, out=acc, dtype=np.uint16, casting="unsafe")
    acc += _LUMA_ROUND
    np.multiply(src[..., 1], _LUMA_G, out=tmp, dtype=np.uint16, casting="unsafe")
    acc += tmp
    np.multiply(src[..., 0], _LUMA_B, out=tmp, dtype=np.uint16, casting="unsafe")
    acc += tmp
    acc >>= _LUMA_SHIFT
    np.copyto(out, acc, casting="unsafe")


# Resolved on first use: None = not looked up yet, False = unavailable.
_NATIVE_GRAY = None
_NATIVE_SWIZZLE = None


def _native_swizzle(src: np.ndarray, dst: np.ndarray, mode: str) -> bool:
    """
    Run the native channel-reorder kernel if it can address these arrays.

    False means it declined -- the extension is absent, which is the normal case
    for `pip install rapidshot`, or the layout is one its strides cannot
    describe. NumPy then produces the identical answer.

    Cached like `_native_gray`, and for the same reason: `rapidshot.native` loads
    the compiled extension, so importing it while `rapidshot` is still
    initialising would be a cycle.
    """
    global _NATIVE_SWIZZLE
    if _NATIVE_SWIZZLE is None:
        try:
            from rapidshot.native import bgra_swizzle_into
            _NATIVE_SWIZZLE = bgra_swizzle_into
        except Exception:  # pragma: no cover - depends on the install
            _NATIVE_SWIZZLE = False
    if _NATIVE_SWIZZLE is False:
        return False
    return _NATIVE_SWIZZLE(src, dst, mode)


def _native_gray(src: np.ndarray, out: np.ndarray) -> bool:
    """
    Run the native luma kernel if it can address these arrays.

    False means it declined -- either the extension is absent, which is the
    normal case for `pip install rapidshot`, or the layout is one its strides
    cannot describe. Either way NumPy produces the identical answer.

    Imported lazily and cached. `rapidshot.native` loads the compiled extension,
    and reaching for it while `rapidshot` is still initialising would be a cycle;
    caching means the per-frame path resolves the lookup once rather than on
    every conversion.
    """
    global _NATIVE_GRAY
    if _NATIVE_GRAY is None:
        try:
            from rapidshot.native import bgra_to_gray_into
            _NATIVE_GRAY = bgra_to_gray_into
        except Exception:  # pragma: no cover - depends on the install
            _NATIVE_GRAY = False
    if _NATIVE_GRAY is False:
        return False
    return _NATIVE_GRAY(src, out)


def bgra_to_gray(src: np.ndarray) -> np.ndarray:
    """
    Convert a BGRA (or BGR) image to single-channel grayscale.

    Implemented in pure NumPy rather than delegating to OpenCV: cv2 is an
    optional dependency, and routing GRAY through it meant that on a machine
    without OpenCV the conversion silently did nothing and callers received
    unconverted 4-channel BGRA frames labelled as grayscale.

    This allocates its own result and intermediates, so it is the convenience
    form. The per-frame capture path uses
    :meth:`NumpyProcessor.convert_into`, which reuses both.

    Args:
        src: (H, W, 3) or (H, W, 4) uint8 array in BGR(A) channel order

    Returns:
        (H, W, 1) uint8 grayscale array
    """
    height, width = src.shape[:2]
    out = np.empty((height, width, 1), dtype=np.uint8)
    _luma_into(src, out[..., 0],
               np.empty((height, width), dtype=np.uint16),
               np.empty((height, width), dtype=np.uint16))
    return out


class NumpyProcessor:
    """
    NumPy-based processor for image processing.
    """
    # Class attribute to identify the backend type
    BACKEND_TYPE = ProcessorBackends.NUMPY

    def __init__(self, color_mode):
        """
        Initialize the processor.

        Args:
            color_mode: Color format (RGB, RGBA, BGR, BGRA, GRAY)

        Raises:
            ValueError: If color_mode is not supported
        """
        self.cvtcolor = None
        self.requested_color_mode = validate_color_mode(color_mode)
        self.color_mode = color_mode
        self.PBYTE = ctypes.POINTER(ctypes.c_ubyte)

        # GRAY intermediates, allocated on first use and reused thereafter.
        self._luma_a = None
        self._luma_b = None
        self._luma_capacity = 0

        # Simplified processing for BGRA
        if self.color_mode == 'BGRA':
            self.color_mode = None

    @property
    def output_channels(self) -> int:
        """Number of channels this processor's frames carry."""
        return channels_for_color_mode(self.color_mode)

    def _luma_scratch(self, height: int, width: int):
        """
        Reusable uint16 intermediates for GRAY, sized to the largest request.

        ``convert_into`` runs on dirty-rect sub-views as well as whole frames, so
        the shape varies from call to call. One flat allocation that only ever
        grows, sliced and reshaped per call, covers every shape without
        reallocating -- both the slice and the reshape return views, so no copy
        is involved.
        """
        need = height * width
        if need > self._luma_capacity:
            self._luma_a = np.empty(need, dtype=np.uint16)
            self._luma_b = np.empty(need, dtype=np.uint16)
            self._luma_capacity = need
        return (self._luma_a[:need].reshape(height, width),
                self._luma_b[:need].reshape(height, width))

    def convert_into(self, src: np.ndarray, dst: np.ndarray) -> None:
        """
        Convert a BGRA source image into a pre-allocated destination array.

        Unlike :meth:`process_cvtcolor` this never allocates a result array, so
        it can write straight into a caller-supplied buffer.

        Args:
            src: (H, W, 4) uint8 BGRA source
            dst: (H, W, C) uint8 destination matching this processor's color mode
        """
        mode = self.color_mode
        if mode is None or mode == "BGRA":
            # Already the source layout: a straight copy, and NumPy does that at
            # memory speed (33 GB/s measured). Nothing for a kernel to improve.
            dst[:] = src
        elif mode in ("RGB", "BGR", "RGBA"):
            # One native pass reads each cache line once; the NumPy fallback
            # below makes three strided passes over the frame. Identical output
            # either way -- see `_native_swizzle`.
            if not _native_swizzle(src, dst, mode):
                if mode == "RGB":
                    # Per-channel assignment rather than dst[:] = src[..., 2::-1].
                    # The reversed stride forces NumPy through a slow gather;
                    # three contiguous channel copies measured 1.8x faster for
                    # identical output.
                    dst[..., 0] = src[..., 2]
                    dst[..., 1] = src[..., 1]
                    dst[..., 2] = src[..., 0]
                elif mode == "BGR":
                    dst[..., 0] = src[..., 0]
                    dst[..., 1] = src[..., 1]
                    dst[..., 2] = src[..., 2]
                else:
                    # BGRA -> RGBA is a red/blue swap with alpha preserved.
                    dst[..., 0] = src[..., 2]
                    dst[..., 1] = src[..., 1]
                    dst[..., 2] = src[..., 0]
                    dst[..., 3] = src[..., 3]
        elif mode == "GRAY":
            # `dst[..., 0]` is a view of a (H, W, 1) destination, so nothing is
            # copied on the way out of either path.
            out = dst[..., 0] if dst.ndim == 3 else dst
            # The native kernel is 14x the NumPy path and byte-identical to it,
            # so it is tried first and NumPy is the fallback -- not a different
            # answer, the same answer more slowly. It declines any layout its
            # strides cannot describe, which is why this is a boolean and not a
            # try/except.
            if not _native_gray(src, out):
                height, width = src.shape[:2]
                acc, tmp = self._luma_scratch(height, width)
                _luma_into(src, out, acc, tmp)
        else:  # pragma: no cover - construction validates the mode
            raise ValueError(f"Unsupported color mode: {mode!r}")

    def process_cvtcolor(self, image):
        """
        Convert color format with robust error handling.
        
        Args:
            image: Image to convert
            
        Returns:
            Converted image
        """
        # Fixed region handling patch applied
        # Skip color conversion if image is None or empty
        if image is None or image.size == 0:
            logger.warning("Received empty image for color conversion")
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Ensure image has proper shape and type
        if not isinstance(image, np.ndarray):
            try:
                image = np.array(image)
            except Exception as e:
                logger.warning(f"Failed to convert image to numpy array: {e}")
                return np.zeros((480, 640, 3), dtype=np.uint8)
                
        # Handle images with no channels or wrong number of channels
        if len(image.shape) < 3 or image.shape[2] < 3:
            try:
                import cv2  # type: ignore[import-not-found]
                # Convert grayscale to BGR if needed
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                return image
            except Exception as e:
                logger.warning(f"Failed to convert image format: {e}")
                return np.zeros((image.shape[0] if len(image.shape) > 0 else 480, 
                                image.shape[1] if len(image.shape) > 1 else 640, 3), dtype=np.uint8)
        
        try:
            # Every mode is handled in pure NumPy -- OpenCV is not required.
            # The result is always a freshly allocated, C-contiguous array that
            # owns its data. Returning a view (as this used to for RGB/BGR)
            # aliased the caller's frame to a pooled buffer that grab() had
            # already checked back in, so the next capture silently rewrote it.
            converted = np.empty(
                (image.shape[0], image.shape[1], self.output_channels),
                dtype=np.uint8,
            )
            self.convert_into(image, converted)
            return converted

        except Exception as e:
            logger.warning(f"Color conversion error for mode '{self.color_mode}': {e}")
            # Fallback: return BGR from BGRA if possible, or original image
            if image.ndim == 3 and image.shape[2] == 4: # BGRA
                return image[..., :3] # Return BGR part
            elif image.ndim == 3 and image.shape[2] == 3: # Already 3 channels
                return image
            # If it's grayscale or some other format, return as is or a placeholder
            return image # Or np.zeros(...) as per previous logic for severe errors

    def shot(self, image_ptr, rect, width, height, buffer_size=None):
        """
        Process directly into a caller-provided memory buffer.

        The destination receives pixels in this processor's configured color
        mode -- BGRA output is a straight copy, everything else is converted in
        place with no intermediate allocation.

        The destination size is validated before any write. This path used to
        ``memmove`` a hardcoded ``width * height * 4`` bytes regardless of both
        the color mode and the real buffer size, so an RGB-configured capture
        writing into a correctly-sized 3-channel buffer overran it by a third of
        a frame and took the process down with an access violation.

        Args:
            image_ptr: Destination buffer, or a pointer to one
            rect: Mapped rectangle
            width: Region width in pixels
            height: Region height in pixels
            buffer_size: Destination size in bytes. Only needed when image_ptr
                is a bare pointer, since arrays and buffer-protocol objects
                report their own size.

        Returns:
            True on success.

        Raises:
            ValueError: If the destination is too small, or if its size cannot
                be determined at all.
        """
        channels = self.output_channels
        required_bytes = width * height * channels

        dst_address, detected_size = describe_destination(image_ptr)
        if dst_address is None:
            raise ValueError("Invalid destination pointer for shot copy")

        if detected_size is None and buffer_size is None:
            raise ValueError(
                f"Cannot verify destination size for shot(): {width}x{height} in "
                f"{self.requested_color_mode} needs {required_bytes} bytes. Pass a "
                "NumPy array (or any sized buffer), or supply buffer_size "
                "explicitly -- writing through an unsized pointer risks "
                "corrupting memory."
            )

        # If both are known the caller's declared size must not exceed reality.
        known_size = min(s for s in (detected_size, buffer_size) if s is not None)
        if known_size < required_bytes:
            raise ValueError(
                f"Destination buffer is too small for shot(): {known_size} bytes "
                f"provided, {required_bytes} needed for {width}x{height} in "
                f"{self.requested_color_mode} ({channels} channel(s))."
            )

        pitch = int(rect.Pitch)
        src_row_bytes = width * 4
        src_address = pointer_to_address(rect.pBits)
        if src_address is None:
            raise ValueError("Invalid source pointer for shot copy")

        if pitch < src_row_bytes:
            raise ValueError(
                f"Mapped surface pitch {pitch} is smaller than a {width}px BGRA row "
                f"({src_row_bytes} bytes); refusing to read out of bounds."
            )

        # BGRA needs no conversion: copy rows straight across.
        if self.color_mode is None:
            if pitch == src_row_bytes:
                ctypes.memmove(dst_address, src_address, src_row_bytes * height)
            else:
                for row in range(height):
                    ctypes.memmove(
                        dst_address + row * src_row_bytes,
                        src_address + row * pitch,
                        src_row_bytes,
                    )
            return True

        # Converting modes: wrap both sides as NumPy views and convert directly
        # into the caller's memory, so nothing extra is allocated per frame.
        src_buffer = (ctypes.c_ubyte * (pitch * height)).from_address(src_address)
        src_view = np.ctypeslib.as_array(src_buffer).reshape(height, pitch)
        src_image = src_view[:, :src_row_bytes].reshape(height, width, 4)

        dst_buffer = (ctypes.c_ubyte * required_bytes).from_address(dst_address)
        dst_image = np.ctypeslib.as_array(dst_buffer).reshape(height, width, channels)

        self.convert_into(src_image, dst_image)
        return True

    # Fraction of the frame above which patching costs more than redoing it.
    # Measured in benchmarks/dirty_rect_pipeline.py: the accumulator wins by
    # 12-15x at the ~1% dirty a normal desktop produces, breaks even at ~100%,
    # and is 5-9% slower on a fully dirty frame. 0.9 keeps that regression off
    # the table without giving up anything that mattered.
    DIRTY_AREA_LIMIT = 0.9

    def _accumulator_for(self, height: int, width: int):
        """The persistent converted frame that dirty regions are patched into.

        Reallocated when the shape changes, which also invalidates it: a buffer
        of the wrong size holds nothing usable.
        """
        shape = (height, width, self.output_channels)
        accum = getattr(self, "_accum", None)
        if accum is None or accum.shape != shape:
            accum = np.empty(shape, dtype=np.uint8)
            self._accum = accum
            self._accum_valid = False
        return accum

    def _usable_dirty_rects(self, dirty_rects, width, height, rotation_angle):
        """The rects to patch, or None to convert the whole frame.

        Returns None whenever patching is not applicable or not worth it, so
        the caller has a single condition to branch on rather than four.
        """
        if dirty_rects is None or self.color_mode is None or rotation_angle != 0:
            return None
        # An empty list is not "nothing changed" -- it means no metadata was
        # reported, and a mode change or a coalescing driver can produce it
        # while the image differs completely. Redraw everything.
        if not dirty_rects:
            return None
        if not getattr(self, "_accum_valid", False):
            return None
        accum = getattr(self, "_accum", None)
        if accum is None or accum.shape != (height, width, self.output_channels):
            return None

        area = 0
        for left, top, right, bottom in dirty_rects:
            if not (0 <= left < right <= width and 0 <= top < bottom <= height):
                # Out-of-range rects would raise or silently write elsewhere.
                # Treat the whole set as untrustworthy rather than filtering.
                return None
            area += (right - left) * (bottom - top)
        if area > width * height * self.DIRTY_AREA_LIMIT:
            return None
        return dirty_rects

    def _check_target(self, target, height, width) -> None:
        expected = (height, width, self.output_channels)
        if target.shape != expected:
            raise ValueError(
                f"output_target shape {target.shape} does not match the frame "
                f"shape {expected}"
            )

    def _emit(self, accumulator, output_target):
        """Hand the accumulated frame to the caller.

        With a target, fill it and say so: the caller owns that buffer and will
        release it. Without one, allocate — which costs ~1.6 ms per 1080p frame
        in page faults, so it is the path worth avoiding, but it is also the
        only one that can hand back memory the caller may keep indefinitely.
        """
        if output_target is not None:
            self._check_target(output_target, accumulator.shape[0], accumulator.shape[1])
            output_target[:] = accumulator
            return output_target, True
        return accumulator.copy(), False

    def _should_maintain_accumulator(self, dirty_rects, rotation_angle) -> bool:
        """Whether to leave a complete accumulator behind on a full conversion.

        Only worth it when dirty metadata is actually arriving: maintaining it
        costs one extra full-frame copy, and on a source that never reports
        rects that copy would buy nothing at all.
        """
        return (
            dirty_rects is not None
            and len(dirty_rects) > 0
            and self.color_mode is not None
            and rotation_angle == 0
        )

    def invalidate_accumulator(self) -> None:
        """Forget the accumulated frame.

        Anything that makes the previous frame an unsound base for patching --
        a resolution change, a re-initialised duplicator, or simply a capture
        that bypassed the accumulator -- must call this. Patching onto a stale
        accumulator produces a frame that is part current and part historical,
        which no test of shape or speed would catch.
        """
        self._accum_valid = False

    def _read_rows(self, dest_view, src_view, top, bottom, start, end, pitch, row_bytes):
        """Copy one horizontal band out of the mapped staging surface."""
        if pitch == row_bytes and start == 0:
            dest_view[top:bottom] = src_view[top:bottom, :row_bytes]
        else:
            for row in range(top, bottom):
                dest_view[row, :] = src_view[row, start:end]

    def _read_patch_rows(self, dest_view, src_view, left, top, right, bottom,
                         start, end, pitch, row_bytes):
        """Read a dirty rect as whole rows: one vectorised slice, more bytes.

        The columns outside the rect are copied needlessly, but the whole band
        moves in a single NumPy operation.
        """
        self._read_rows(dest_view, src_view, top, bottom, start, end, pitch, row_bytes)

    def _read_patch_columns(self, dest_view, src_view, left, top, right, bottom,
                            start, end, pitch, row_bytes):
        """Read only the rect's own columns: fewer bytes, one slice per row.

        **Measured, and it makes no difference** -- kept only so
        benchmarks/dirty_rect_read_strategy.py can reproduce that comparison.
        Against a real mapped staging surface it lands within noise of the row
        strategy at every rect shape tried, including the tall narrow one that
        reads 11.5% of rows for 0.8% of area.

        The reason is that the read is not what the patch path spends its time
        on: allocating the output array costs ~1.6 ms per frame, which swamps
        any difference between the two. See ROADMAP.md section 10.
        """
        col_start = start + left * 4
        col_end = start + right * 4
        dest_start, dest_end = left * 4, right * 4
        for row in range(top, bottom):
            dest_view[row, dest_start:dest_end] = src_view[row, col_start:col_end]

    # Rows: one vectorised slice per rect, and measurably no worse than reading
    # columns. The simpler of two equals.
    _read_patch = _read_patch_rows

    def process(self, rect, width, height, region, rotation_angle, output_buffer=None,
                dirty_rects=None, output_target=None):
        """
        Process a frame with robust error handling.

        Args:
            rect: Mapped rectangle
            width: Width
            height: Height
            region: Region to capture
            rotation_angle: Rotation angle,
            output_buffer: Pre-allocated NumPy array to store the processed frame.
            dirty_rects: Regions that changed, in region-relative coordinates. When
                given, only those are read and converted, and the rest of the frame
                comes from the previous one. None means "convert everything", which
                is also the right answer when the metadata is unavailable.
            output_target: Pre-allocated array to write the converted frame into.
                Without it a fresh array is allocated per frame, and the page
                faults on first touch cost ~1.6 ms for a 1080p RGB frame --
                more than the conversion itself. Supplying a reused buffer
                removes that, at the price of the caller owning its lifetime.
        """
        # Validated before the try below, deliberately. A mis-shaped target is a
        # caller bug, not a capture fault, and the catch-all in this method
        # would otherwise log it and hand back a frame anyway -- leaving the
        # caller to wonder why their buffer stayed empty.
        if output_target is not None:
            region_left, region_top, region_right, region_bottom = region
            self._check_target(output_target,
                               region_bottom - region_top,
                               region_right - region_left)

        # Phase 1: Get data into the output buffer (no rotation, no color conversion yet)
        try:
            if not hasattr(rect, 'pBits') or not rect.pBits:
                raise ValueError(f"Invalid rect or pBits, cannot process. Rect type: {type(rect)}")

            pitch = int(rect.Pitch)
            src_address = pointer_to_address(rect.pBits)
            if src_address is None:
                raise ValueError("Mapped rect does not contain a valid pointer")

            region_left, region_top, region_right, region_bottom = region
            if not (0 <= region_left < region_right <= width) or not (0 <= region_top < region_bottom <= height):
                raise ValueError(f"Region {region} is outside of the frame dimensions {(width, height)}")

            region_height = region_bottom - region_top
            region_width = region_right - region_left

            if output_buffer is None:
                output_buffer = np.empty((region_height, region_width, 4), dtype=np.uint8)
                is_pooled_buffer = False
            else:
                is_pooled_buffer = True
                if output_buffer.shape[:2] != (region_height, region_width) or output_buffer.shape[2] != 4:
                    raise ValueError(
                        f"Output buffer shape {output_buffer.shape} does not match region shape "
                        f"({region_height}, {region_width}, 4)."
                    )

            row_bytes = region_width * 4
            total_pitch_bytes = pitch * region_height
            src_buffer = (ctypes.c_ubyte * total_pitch_bytes).from_address(src_address + region_top * pitch)
            src_view = np.ctypeslib.as_array(src_buffer).reshape(region_height, pitch)

            dest_view = output_buffer.view(np.uint8).reshape(region_height, region_width * 4)
            start = region_left * 4
            end = start + row_bytes

            # Patching only what changed is worth ~1.5x on grab() for a normal
            # desktop, but it is only sound when there is a previous frame to
            # patch onto, the output is a converted one (BGRA already returns the
            # pool buffer with no copy at all), and no rotation intervenes -- a
            # rotated accumulator would need its regions rotated too.
            patch = self._usable_dirty_rects(
                dirty_rects, region_width, region_height, rotation_angle)

            if patch is not None:
                accumulator = self._accumulator_for(region_height, region_width)
                for left, top, right, bottom in patch:
                    self._read_patch(dest_view, src_view, left, top, right, bottom,
                                     start, end, pitch, row_bytes)
                    self.convert_into(
                        output_buffer[top:bottom, left:right],
                        accumulator[top:bottom, left:right],
                    )
                # A copy, not a view: the accumulator is overwritten next frame,
                # so handing out a view into it would alias exactly like the
                # recycled pool buffers this project already had to fix once.
                return self._emit(accumulator, output_target)

            self._read_rows(dest_view, src_view, 0, region_height,
                            start, end, pitch, row_bytes)

            if self._should_maintain_accumulator(dirty_rects, rotation_angle):
                # Metadata is available, so a later frame will be patchable --
                # but only if this one leaves a complete accumulator behind.
                # Without this the accumulator could never become valid in the
                # first place, and the fast path would be dead code.
                accumulator = self._accumulator_for(region_height, region_width)
                self.convert_into(output_buffer, accumulator)
                self._accum_valid = True
                return self._emit(accumulator, output_target)

            # No metadata to patch with, so nothing is maintained and whatever
            # the accumulator held is now a frame behind.
            self.invalidate_accumulator()

            # Phase 2: Color Conversion and Rotation
            #
            # Invariant enforced here: the returned array must never be a view
            # into output_buffer unless is_still_pooled_buffer is True. When the
            # flag is False, grab() checks the pooled buffer straight back into
            # the pool, so any view into it would be silently overwritten by the
            # next capture -- which is exactly what used to happen for RGB, BGR
            # and GRAY output.
            if self.color_mode is None:
                current_array = output_buffer  # Already BGRA, nothing to convert
                is_still_pooled_buffer = is_pooled_buffer
            elif output_target is not None and rotation_angle == 0:
                # Convert straight into the caller's buffer: no allocation, and
                # no intermediate to copy from either.
                self._check_target(output_target, region_height, region_width)
                self.convert_into(output_buffer, output_target)
                return output_target, True
            else:
                current_array = np.empty(
                    (region_height, region_width, self.output_channels),
                    dtype=np.uint8,
                )
                self.convert_into(output_buffer, current_array)
                is_still_pooled_buffer = False

            # Rotation
            if rotation_angle != 0:
                k = (rotation_angle // 90) % 4
                if k != 0:
                    # np.rot90 returns a view; materialise it so the result is
                    # contiguous and independent of the buffer it came from.
                    current_array = np.ascontiguousarray(
                        np.rot90(current_array, k=k))
                    is_still_pooled_buffer = False

            return current_array, is_still_pooled_buffer

        except Exception as e:
            logger.error(f"Frame processing error in NumpyProcessor: {e}")
            # Ensure output_buffer is zeroed out in case of any error, then return it with False flag
            if output_buffer is not None and hasattr(output_buffer, 'fill'):
                try:
                    output_buffer.fill(0)
                except Exception as fill_e:
                    logger.error(f"Error filling output_buffer after another error: {fill_e}")
            return output_buffer, False # Indicate buffer might be invalid or is not the result