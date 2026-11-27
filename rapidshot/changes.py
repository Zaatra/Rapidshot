"""Exact CPU change maps and the frame leases that carry them."""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

from rapidshot.memory_pool import BufferReleasedError, PooledBuffer


TILE_SIZE = 128
Rectangle = Tuple[int, int, int, int]


@dataclass(frozen=True)
class ChangeSet:
    """An immutable map of changed 128-pixel tiles in frame coordinates."""

    indices: Tuple[int, ...]
    rectangles: Tuple[Rectangle, ...]
    size: Tuple[int, int]
    reason: str
    present_time_qpc: Optional[int]
    accumulated_frames: int
    fallback_reason: Optional[str]
    tight_rectangles: Optional[Tuple[Rectangle, ...]] = None


@dataclass(frozen=True)
class ChangeAuditStats:
    """Lifetime counts from full comparisons of frames with usable dirty hints."""

    audited_frames: int = 0
    miss_frames: int = 0
    missed_tiles: int = 0


class ChangedFrame:
    """CPU BGRA pixels leased until :meth:`release` and their exact tile map."""

    __slots__ = ("_pixels", "_owner", "_changes")

    def __init__(self, frame: Union[np.ndarray, PooledBuffer], changes: ChangeSet):
        self._owner = frame if isinstance(frame, PooledBuffer) else None
        self._pixels = frame.array if self._owner is not None else frame
        self._changes = changes

    @property
    def pixels(self) -> np.ndarray:
        """Writable pixels valid while this lease is live."""
        if self._pixels is None:
            raise BufferReleasedError("This frame has been released. Copy pixels before releasing the frame.")
        return self._pixels

    @property
    def changes(self) -> ChangeSet:
        """The tile map remains readable after the frame is released."""
        return self._changes

    @property
    def released(self) -> bool:
        return self._pixels is None

    def release(self) -> None:
        """End this lease. Releasing it twice has no effect."""
        owner = self._owner
        self._owner = None
        self._pixels = None
        if owner is not None:
            owner.release()

    def __enter__(self) -> "ChangedFrame":
        if self.released:
            raise BufferReleasedError("This frame has been released.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


class TiledFrame:
    """A snapshot of BGRA tile pages and its exact change map."""

    __slots__ = ("_pages", "_changes", "_release_token")

    def __init__(self, pages, changes, *, release_token=None):
        self._pages = pages
        self._changes = changes
        self._release_token = release_token

    @property
    def changes(self) -> ChangeSet:
        """The change map remains readable after the frame is released."""
        return self._changes

    @property
    def tiles(self) -> Tuple[np.ndarray, ...]:
        """Read-only BGRA views in ``changes.indices`` order."""
        pages = self._live_pages()
        return tuple(_page_array(pages[index], rect)
                     for index, rect in zip(self._changes.indices, self._changes.rectangles))

    @property
    def released(self) -> bool:
        return self._pages is None

    def _live_pages(self):
        if self._pages is None:
            raise BufferReleasedError("This frame has been released.")
        return self._pages

    def materialize(self, rect: Optional[Rectangle] = None) -> np.ndarray:
        """Copy a frame-local half-open rectangle to an independently owned BGRA array."""
        pages = self._live_pages()
        width, height = self._changes.size
        if rect is None:
            rect = 0, 0, width, height
        try:
            left, top, right, bottom = rect
        except (TypeError, ValueError) as exc:
            raise ValueError("A materialization rectangle must have four integer coordinates") from exc
        coordinates = left, top, right, bottom
        if not all(isinstance(value, (int, np.integer)) and not isinstance(value, bool) for value in coordinates):
            raise ValueError("A materialization rectangle must have four integer coordinates")
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError("A materialization rectangle must have positive area within the frame")

        image = np.empty((bottom - top, right - left, 4), dtype=np.uint8)
        columns = (width + TILE_SIZE - 1) // TILE_SIZE
        for row in range(top // TILE_SIZE, (bottom - 1) // TILE_SIZE + 1):
            tile_top = row * TILE_SIZE
            tile_bottom = min(tile_top + TILE_SIZE, height)
            y0, y1 = max(top, tile_top), min(bottom, tile_bottom)
            for column in range(left // TILE_SIZE, (right - 1) // TILE_SIZE + 1):
                tile_left = column * TILE_SIZE
                tile_right = min(tile_left + TILE_SIZE, width)
                x0, x1 = max(left, tile_left), min(right, tile_right)
                tile = _page_array(pages[row * columns + column], (tile_left, tile_top, tile_right, tile_bottom))
                image[y0 - top:y1 - top, x0 - left:x1 - left] = tile[
                    y0 - tile_top:y1 - tile_top, x0 - tile_left:x1 - tile_left
                ]
        return image

    def release(self) -> None:
        """End the lease and return its slot. Releasing it twice has no effect."""
        token = self._release_token
        self._release_token = None
        self._pages = None
        if token is not None:
            token()

    def __enter__(self) -> "TiledFrame":
        self._live_pages()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def _page_array(page, rect):
    left, top, right, bottom = rect
    return np.frombuffer(page, dtype=np.uint8).reshape(bottom - top, right - left, 4)


def _tight_rectangle(previous, current, rect):
    """Bound the unequal positions of a tile already confirmed to differ."""
    changed = np.any(previous != current, axis=2)
    rows = np.flatnonzero(changed.any(axis=1))
    columns = np.flatnonzero(changed.any(axis=0))
    left, top = rect[:2]
    return (left + int(columns[0]), top + int(rows[0]),
            left + int(columns[-1]) + 1, top + int(rows[-1]) + 1)


def _rectangles(width: int, height: int) -> Tuple[Rectangle, ...]:
    return tuple(
        (left, top, min(left + TILE_SIZE, width), min(top + TILE_SIZE, height))
        for top in range(0, height, TILE_SIZE)
        for left in range(0, width, TILE_SIZE)
    )


def _candidate_tiles(dirty_rects, region, output_size, columns):
    """Validate output-space hints, then clip and rebase them into the crop."""
    crop_left, crop_top, crop_right, crop_bottom = region
    out_width, out_height = output_size
    candidates = set()
    for rect in dirty_rects:
        if len(rect) != 4 or not all(isinstance(value, (int, np.integer)) for value in rect):
            return None
        left, top, right, bottom = map(int, rect)
        if not (0 <= left < right <= out_width and 0 <= top < bottom <= out_height):
            return None
        left, right = max(left, crop_left), min(right, crop_right)
        top, bottom = max(top, crop_top), min(bottom, crop_bottom)
        if left >= right or top >= bottom:
            continue
        x0, x1 = (left - crop_left) // TILE_SIZE, (right - 1 - crop_left) // TILE_SIZE
        y0, y1 = (top - crop_top) // TILE_SIZE, (bottom - 1 - crop_top) // TILE_SIZE
        for y in range(y0, y1 + 1):
            candidates.update(range(y * columns + x0, y * columns + x1 + 1))
    return sorted(candidates)


def _metadata_fallback(previous_serial, source, serial, dirty, moves, accumulated, rotation):
    if rotation:
        return "rotation"
    if source is None or serial is None or previous_serial is None or serial != previous_serial + 1:
        return "frame_gap"
    if accumulated != 1:
        return "accumulated_frames"
    if dirty is None or moves is None:
        return "metadata_unavailable"
    if moves:
        return "move_rects"
    if not dirty:
        return "empty_dirty_rects"
    return None


def _valid_candidates(dirty_rects, region, output_size, width):
    try:
        columns = (width + TILE_SIZE - 1) // TILE_SIZE
        return _candidate_tiles(dirty_rects, region, output_size, columns)
    except TypeError:
        # A malformed rectangle has no trustworthy candidate area.
        return None


class _ChangeHistory:
    """A private BGRA image which never shares storage with a returned lease."""

    def __init__(self):
        self._pixels = None
        self._identity = None
        self._serial = None
        self._rectangles = ()
        self._seeded = False
        self.bytes_written = 0
        self.last_update_bytes = 0

    @property
    def size_bytes(self) -> int:
        return 0 if self._pixels is None else self._pixels.nbytes

    def clear(self) -> None:
        """Forget a failed update so the next image establishes a sound baseline."""
        self._pixels = None
        self._identity = None
        self._serial = None
        self._rectangles = ()
        self.last_update_bytes = 0

    def update(
        self, pixels, *, source, serial, region, output_size, dirty_rects,
        move_rects, accumulated_frames, present_time_qpc=None, rotation=0, tight_rectangles=False,
    ) -> ChangeSet:
        """Compare and retain an image before the caller gains write access."""
        if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("Exact tiles require a CPU BGRA8 array")
        height, width, _ = pixels.shape
        identity = (source, region, output_size, rotation, width, height)
        self.last_update_bytes = 0
        bounds = None
        if self._pixels is None or self._identity != identity:
            reason = "reset" if self._seeded else "initial"
            if self._pixels is None or self._pixels.shape != pixels.shape:
                self._pixels = np.empty((height, width, 4), dtype=np.uint8)
            np.copyto(self._pixels, pixels)
            self._rectangles = _rectangles(width, height)
            self.last_update_bytes = pixels.nbytes
            changed = tuple(range(len(self._rectangles)))
            if tight_rectangles:
                bounds = self._rectangles
            fallback = "baseline"
        else:
            fallback = self._fallback(source, serial, dirty_rects, move_rects, accumulated_frames, rotation)
            candidates = None
            if fallback is None:
                candidates = _valid_candidates(dirty_rects, region, output_size, width)
                if candidates is None:
                    fallback = "invalid_dirty_rects"
            if candidates is None:
                candidates = range(len(self._rectangles))
            if tight_rectangles:
                changed, bounds = self._patch_changed_with_bounds(pixels, candidates)
            else:
                changed = tuple(self._patch_changed(pixels, candidates))
            reason = "changed" if changed else "identical"

        self._identity = identity
        self._serial = serial
        self._seeded = True
        self.bytes_written += self.last_update_bytes
        return ChangeSet(
            indices=changed,
            rectangles=tuple(self._rectangles[index] for index in changed),
            size=(width, height),
            reason=reason,
            present_time_qpc=present_time_qpc,
            accumulated_frames=accumulated_frames,
            fallback_reason=fallback,
            tight_rectangles=bounds,
        )

    def _fallback(self, source, serial, dirty, moves, accumulated, rotation):
        return _metadata_fallback(self._serial, source, serial, dirty, moves, accumulated, rotation)

    def _patch_changed(self, pixels, candidates):
        for index in candidates:
            left, top, right, bottom = self._rectangles[index]
            previous = self._pixels[top:bottom, left:right]
            current = pixels[top:bottom, left:right]
            if not np.array_equal(previous, current):
                np.copyto(previous, current)
                self.last_update_bytes += current.nbytes
                yield index

    def _patch_changed_with_bounds(self, pixels, candidates):
        changed, bounds = [], []
        for index in candidates:
            rect = self._rectangles[index]
            left, top, right, bottom = rect
            previous = self._pixels[top:bottom, left:right]
            current = pixels[top:bottom, left:right]
            if not np.array_equal(previous, current):
                changed.append(index)
                bounds.append(_tight_rectangle(previous, current, rect))
                np.copyto(previous, current)
                self.last_update_bytes += current.nbytes
        return tuple(changed), tuple(bounds)


@dataclass(frozen=True)
class _TilePlan:
    revision: object
    identity: tuple
    serial: Optional[int]
    size: Tuple[int, int]
    rectangles: Tuple[Rectangle, ...]
    read_indices: Tuple[int, ...]
    hinted_indices: Optional[Tuple[int, ...]]
    initial_reason: Optional[str]
    fallback_reason: Optional[str]
    audited: bool
    audit_interval: int
    present_time_qpc: Optional[int]
    accumulated_frames: int


class _TiledHistory:
    """A complete snapshot whose immutable tile pages can be shared with leases."""

    def __init__(self, audit_interval=0):
        self.audit_interval = audit_interval
        self.audit_stats = ChangeAuditStats()
        self.bytes_written = 0
        self._seeded = False
        self._untrusted_source = None
        self.clear()

    @property
    def size_bytes(self) -> int:
        return 0 if self._pages is None else sum(map(len, self._pages))

    def clear(self) -> None:
        self._pages = None
        self._identity = None
        self._serial = None
        self._rectangles = ()
        self._revision = object()
        self._guided_since_audit = 0
        self.last_update_bytes = 0

    def plan(
        self, *, source, serial, region, output_size, dirty_rects, move_rects,
        accumulated_frames, present_time_qpc=None, rotation=0, force_full=False, audit_every=None,
    ) -> _TilePlan:
        audit_interval = self.audit_interval if audit_every is None else audit_every
        if not isinstance(audit_interval, int) or isinstance(audit_interval, bool) or audit_interval < 0:
            raise ValueError("The audit interval must be a nonnegative integer")

        left, top, right, bottom = region
        size = width, height = (right - left, bottom - top)
        identity = source, region, output_size, rotation
        rectangles = self._rectangles
        initial_reason = None
        hinted = None
        audited = False
        if self._pages is None or self._identity != identity:
            rectangles = _rectangles(width, height)
            initial_reason = "reset" if self._seeded else "initial"
            fallback = "baseline"
        else:
            fallback = _metadata_fallback(self._serial, source, serial, dirty_rects, move_rects,
                                          accumulated_frames, rotation)
            if force_full and fallback is None:
                fallback = force_full if isinstance(force_full, str) else "forced_full"
            if self._untrusted_source is not None and source == self._untrusted_source and fallback is None:
                fallback = "dirty_rects_audit_disabled"
            if fallback is None:
                candidates = _valid_candidates(dirty_rects, region, output_size, width)
                if candidates is None:
                    fallback = "invalid_dirty_rects"
                else:
                    hinted = tuple(candidates)
                    audited = bool(audit_interval and self._guided_since_audit + 1 >= audit_interval)
                    if audited:
                        fallback = "periodic_audit"

        indices = tuple(range(len(rectangles))) if fallback else hinted
        return _TilePlan(self._revision, identity, serial, size, rectangles, indices, hinted,
                         initial_reason, fallback, audited, audit_interval, present_time_qpc, accumulated_frames)

    def update(self, plan: _TilePlan, read_tile, *, tight_rectangles=False, release_token=None) -> TiledFrame:
        """Compare and own the needed pixels before atomically replacing the baseline."""
        if plan.revision is not self._revision:
            raise RuntimeError("The tile plan no longer matches the frame history")
        initial = plan.initial_reason is not None
        pages = [None] * len(plan.rectangles) if initial else list(self._pages)
        changed, bounds = [], []
        bytes_written = 0
        for index in plan.read_indices:
            rect = plan.rectangles[index]
            left, top, right, bottom = rect
            current = read_tile(index, rect)
            if (not isinstance(current, np.ndarray) or current.dtype != np.uint8
                    or current.shape != (bottom - top, right - left, 4)):
                raise ValueError("Each read tile must provide BGRA8 pixels of the requested size")
            if initial:
                if tight_rectangles:
                    bounds.append(rect)
            else:
                previous = _page_array(pages[index], rect)
                if np.array_equal(previous, current):
                    continue
                if tight_rectangles:
                    bounds.append(_tight_rectangle(previous, current, rect))
            pages[index] = current.tobytes(order="C")
            bytes_written += len(pages[index])
            changed.append(index)

        fallback = plan.fallback_reason
        stats = self.audit_stats
        guided_since_audit = self._guided_since_audit
        source = plan.identity[0]
        untrusted_source = self._untrusted_source if self._untrusted_source == source else None
        if plan.audited:
            misses = len(set(changed).difference(plan.hinted_indices))
            stats = ChangeAuditStats(stats.audited_frames + 1, stats.miss_frames + bool(misses),
                                     stats.missed_tiles + misses)
            guided_since_audit = 0
            if misses:
                untrusted_source = source
                fallback = "dirty_rects_audit_miss"
        elif initial:
            guided_since_audit = 0
        elif plan.hinted_indices is not None and plan.audit_interval:
            guided_since_audit += 1

        changes = ChangeSet(
            indices=tuple(changed),
            rectangles=tuple(plan.rectangles[index] for index in changed),
            size=plan.size,
            reason=plan.initial_reason or ("changed" if changed else "identical"),
            present_time_qpc=plan.present_time_qpc,
            accumulated_frames=plan.accumulated_frames,
            fallback_reason=fallback,
            tight_rectangles=tuple(bounds) if tight_rectangles else None,
        )
        pages = tuple(pages)
        next_revision = object()
        frame = TiledFrame(pages, changes, release_token=release_token)
        self._pages = pages
        self._identity = plan.identity
        self._serial = plan.serial
        self._rectangles = plan.rectangles
        self._revision = next_revision
        self._seeded = True
        self._untrusted_source = untrusted_source
        self._guided_since_audit = guided_since_audit
        self.audit_stats = stats
        self.last_update_bytes = bytes_written
        self.bytes_written += bytes_written
        return frame
