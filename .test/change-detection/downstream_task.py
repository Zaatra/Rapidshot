"""CPU OpenCV template matching over full BGRA frames or changed origin blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import cv2
import numpy as np


Detection = tuple[str, int, int]
Origin = tuple[int, int]
Block = tuple[int, int]
Rect = tuple[int, int, int, int]
CropReader = Callable[[Rect], np.ndarray]


@dataclass(frozen=True)
class SearchResult:
    detections: tuple[Detection, ...]
    searched_origins: int
    calls: int
    full: bool


@dataclass(frozen=True)
class _Template:
    gray: np.ndarray
    origin_width: int
    origin_height: int
    tolerance: float


class TemplateSearch:
    """Detect exact grayscale icons with OpenCV proposals and byte verification.

    Each detection belongs to the origin block containing its top-left pixel.
    The caller supplies changed input tiles since the last update, or requests
    a full scan when capture continuity is uncertain. Use one instance per arm.
    """

    def __init__(self, templates: dict[str, np.ndarray], width: int, height: int, tile: int = 128):
        if not templates or min(width, height, tile) <= 0:
            raise ValueError("Template search needs positive dimensions and at least one icon")
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        self.width, self.height, self.tile = width, height, tile
        self.columns = (width + tile - 1) // tile
        self.tile_count = self.columns * ((height + tile - 1) // tile)
        self._templates = {}
        for name, contents in sorted(templates.items()):
            gray = np.asarray(contents)
            if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
                raise ValueError(f"{name} template must be a nonempty uint8 grayscale array")
            th, tw = gray.shape
            if tw > width or th > height or int(gray.min()) == int(gray.max()):
                raise ValueError(f"{name} template must fit the frame and have contrasting pixels")
            gray = gray.copy(order="C")
            gray.setflags(write=False)
            # OpenCV's float32 correlation can leave positive residuals on exact
            # hits. An area-scaled margin proposes hits; byte checks decide them.
            tolerance = max(8.0, 64 * np.finfo(np.float32).eps * 255**2 * gray.size)
            self._templates[name] = _Template(gray, width - tw + 1, height - th + 1, tolerance)
        self._full_work = sum(t.origin_width * t.origin_height for t in self._templates.values())
        self.reset()

    def reset(self):
        """Forget detections. The next update scans the whole frame."""
        self._hits: dict[str, dict[Block, list[Origin]]] | None = None
        self._detections: tuple[Detection, ...] = ()

    def _check_frame(self, frame: np.ndarray):
        if (frame.shape != (self.height, self.width, 4) or frame.dtype != np.uint8
                or not frame.flags.c_contiguous):
            raise ValueError("Template search needs a packed H×W×4 uint8 BGRA frame")

    def _search(
        self,
        frame: np.ndarray | None,
        template: _Template,
        origins: Rect,
        gray_crops: dict[Rect, np.ndarray],
        read_crop: CropReader | None = None,
    ) -> dict[Block, list[Origin]]:
        left, top, right, bottom = origins
        th, tw = template.gray.shape
        crop = (left, top, right + tw - 1, bottom + th - 1)
        gray = gray_crops.get(crop)
        if gray is None:
            pixels = frame[top:crop[3], left:crop[2]] if read_crop is None else read_crop(crop)
            gray = cv2.cvtColor(pixels, cv2.COLOR_BGRA2GRAY)
            gray_crops[crop] = gray
        scores = cv2.matchTemplate(gray, template.gray, cv2.TM_SQDIFF)
        ys, xs = np.nonzero(scores <= template.tolerance)
        hits: dict[Block, list[Origin]] = {}
        for local_y, local_x in zip(ys, xs):
            if np.array_equal(gray[local_y:local_y + th, local_x:local_x + tw], template.gray):
                x, y = left + int(local_x), top + int(local_y)
                hits.setdefault((x // self.tile, y // self.tile), []).append((x, y))
        return hits

    def _commit(
        self,
        hits: dict[str, dict[Block, list[Origin]]],
        searched: int,
        calls: int,
    ) -> SearchResult:
        self._hits = hits
        self._detections = tuple(sorted(
            (name, x, y)
            for name, blocks in hits.items()
            for origins in blocks.values()
            for x, y in origins
        ))
        return SearchResult(self._detections, searched, calls, searched == self._full_work)

    def find_all(self, frame: np.ndarray) -> SearchResult:
        """Run the genuine full-frame task and seed this instance's results."""
        self._check_frame(frame)
        return self._find_all(frame, None)

    def _find_all(self, frame: np.ndarray | None, read_crop: CropReader | None) -> SearchResult:
        gray_crops: dict[Rect, np.ndarray] = {}
        hits = {
            name: self._search(frame, template, (0, 0, template.origin_width, template.origin_height), gray_crops,
                               read_crop)
            for name, template in self._templates.items()
        }
        return self._commit(hits, self._full_work, len(self._templates))

    def _origin_blocks(self, template: _Template, changed: tuple[int, ...]) -> set[Block]:
        th, tw = template.gray.shape
        blocks = set()
        for index in changed:
            pixel_top, pixel_left = (part * self.tile for part in divmod(index, self.columns))
            # A template starting before the tile can still read its pixels.
            left, top = max(0, pixel_left - tw + 1), max(0, pixel_top - th + 1)
            right = min(pixel_left + self.tile, template.origin_width)
            bottom = min(pixel_top + self.tile, template.origin_height)
            if left < right and top < bottom:
                for by in range(top // self.tile, (bottom - 1) // self.tile + 1):
                    blocks.update((bx, by) for bx in range(left // self.tile, (right - 1) // self.tile + 1))
        return blocks

    def _search_regions(self, template: _Template, blocks: set[Block]) -> Iterable[Rect]:
        columns = (template.origin_width + self.tile - 1) // self.tile
        rows = (template.origin_height + self.tile - 1) // self.tile
        if len(blocks) == columns * rows:
            yield 0, 0, template.origin_width, template.origin_height
            return
        # Join horizontal neighbors to reduce OpenCV calls without scanning any
        # origin twice or evicting an unaffected block.
        ordered = sorted((y, x) for x, y in blocks)
        start = 0
        while start < len(ordered):
            y, first = ordered[start]
            stop = start + 1
            while stop < len(ordered) and ordered[stop] == (y, first + stop - start):
                stop += 1
            yield (first * self.tile, y * self.tile,
                   min((first + stop - start) * self.tile, template.origin_width),
                   min((y + 1) * self.tile, template.origin_height))
            start = stop

    def update(self, frame: np.ndarray, changed_tiles: Iterable[int], force_full: bool = False) -> SearchResult:
        self._check_frame(frame)
        return self._update_tiles(frame, None, changed_tiles, force_full)

    def update_from_crops(self, read_crop: CropReader, changed_tiles: Iterable[int],
                          force_full: bool = False) -> SearchResult:
        """Run the same tile-origin search using only the requested BGRA input crops."""
        return self._update_tiles(None, read_crop, changed_tiles, force_full)

    def _update_tiles(self, frame, read_crop, changed_tiles, force_full):
        if force_full or self._hits is None:
            return self._find_all(frame, read_crop)
        changed = tuple(int(tile) for tile in changed_tiles)
        if any(tile < 0 or tile >= self.tile_count for tile in changed):
            raise ValueError("Changed tile falls outside the input frame")
        if not changed:
            return SearchResult(self._detections, 0, 0, False)

        gray_crops: dict[Rect, np.ndarray] = {}
        hits = {}
        searched = calls = 0
        for name, template in self._templates.items():
            invalid = self._origin_blocks(template, changed)
            updated = {block: coords for block, coords in self._hits[name].items() if block not in invalid}
            for origins in self._search_regions(template, invalid):
                left, top, right, bottom = origins
                updated.update(self._search(frame, template, origins, gray_crops, read_crop))
                searched += (right - left) * (bottom - top)
                calls += 1
            hits[name] = updated
        return self._commit(hits, searched, calls)

    def update_tight_from_crops(self, read_crop: CropReader, changed_rectangles: Iterable[Rect],
                                force_full: bool = False) -> SearchResult:
        """Invalidate only origins whose template may read from a changed rectangle."""
        if force_full or self._hits is None:
            return self._find_all(None, read_crop)
        changed = tuple(changed_rectangles)
        if not changed:
            return SearchResult(self._detections, 0, 0, False)
        gray_crops = {}
        hits = {}
        searched = calls = 0
        for name, template in self._templates.items():
            th, tw = template.gray.shape
            expanded = ((max(0, l - tw + 1), max(0, t - th + 1),
                         min(r, template.origin_width), min(b, template.origin_height)) for l, t, r, b in changed)
            regions = _union_rectangles(expanded)
            updated = {}
            for block, origins in self._hits[name].items():
                kept = [(x, y) for x, y in origins if not any(l <= x < r and t <= y < b for l, t, r, b in regions)]
                if kept:
                    updated[block] = kept
            for left, top, right, bottom in regions:
                origin = (left, top, right, bottom)
                for block, current in self._search(None, template, origin, gray_crops, read_crop).items():
                    updated.setdefault(block, []).extend(current)
                searched += (right - left) * (bottom - top)
                calls += 1
            hits[name] = updated
        return self._commit(hits, searched, calls)


def _union_rectangles(rectangles: Iterable[Rect]) -> tuple[Rect, ...]:
    """Coalesce row bands of overlapping origin rectangles without visiting an origin twice."""
    rectangles = tuple((l, t, r, b) for l, t, r, b in rectangles if l < r and t < b)
    edges = sorted({y for _, t, _, b in rectangles for y in (t, b)})
    result = []
    continuing = {}
    for top, bottom in zip(edges, edges[1:]):
        row = []
        for left, right in sorted((l, r) for l, t, r, b in rectangles if t <= top < b):
            if row and left <= row[-1][1]:
                row[-1] = (row[-1][0], max(row[-1][1], right))
            else:
                row.append((left, right))
        previous, continuing = continuing, {}
        for left, right in row:
            interval = (left, right)
            if interval in previous:
                index = previous[interval]
                result[index] = (left, result[index][1], right, bottom)
            else:
                index = len(result)
                result.append((left, top, right, bottom))
            continuing[interval] = index
    return tuple(result)


def _self_check():
    width, height, tile = 383, 278, 128
    rng = np.random.default_rng(269)
    span = rng.integers(0, 256, (19, 23), dtype=np.uint8)
    edge = rng.integers(0, 256, (15, 20), dtype=np.uint8)
    baseline = np.full((height, width, 4), 42, dtype=np.uint8)
    baseline[:, :, 3] = 255
    baseline[-15:, -20:, :3] = edge[:, :, None]
    edge_hit = (("edge", width - 20, height - 15),)
    both = edge_hit + (("span", 117, 120),)
    full = TemplateSearch({"span": span, "edge": edge}, width, height, tile)
    incremental = TemplateSearch({"span": span, "edge": edge}, width, height, tile)
    assert incremental.update(baseline, ()).detections == full.find_all(baseline).detections == edge_hit

    both_frame = baseline.copy()
    both_frame[120:139, 117:140, :3] = span[:, :, None]
    span_tiles = (0, 1, incremental.columns, incremental.columns + 1)
    assert incremental.update(both_frame, span_tiles).detections == full.find_all(both_frame).detections == both

    removed = both_frame.copy()
    value = int(removed[128, 128, 0])
    removed[128, 128, :3] = value - 1 if value else 1
    changed = (incremental.columns + 1,)
    result = incremental.update(removed, changed)
    assert result.detections == full.find_all(removed).detections == edge_hit
    assert 0 < result.searched_origins < full._full_work and result.calls
    assert incremental.update(both_frame, changed).detections == both
    alpha_changed = both_frame.copy()
    alpha_changed[128, 128, 3] ^= 127
    assert incremental.update(alpha_changed, changed).detections == full.find_all(alpha_changed).detections == both
    idle = incremental.update(both_frame, ())
    assert idle.detections == both and idle.calls == idle.searched_origins == 0
    forced = incremental.update(baseline, (), force_full=True)
    assert forced.full and forced.detections == edge_hit
    dense = incremental.update(both_frame, range(incremental.tile_count))
    assert dense.full and dense.detections == both and dense.calls == 2

    tiled = TemplateSearch({"span": span, "edge": edge}, width, height, tile)
    tight = TemplateSearch({"span": span, "edge": edge}, width, height, tile)
    crops = []

    def read_crop(frame, rect):
        crops.append(rect)
        left, top, right, bottom = rect
        return np.ascontiguousarray(frame[top:bottom, left:right])

    assert tiled.update_from_crops(lambda rect: read_crop(baseline, rect), ()).detections == edge_hit
    assert tight.update_tight_from_crops(lambda rect: read_crop(baseline, rect), ()).detections == edge_hit
    assert crops == [(0, 0, width, height)] * 2
    crops.clear()
    tiled_result = tiled.update_from_crops(lambda rect: read_crop(both_frame, rect), span_tiles)
    assert tiled_result.detections == both and (0, 0, width, height) not in crops
    crops.clear()
    tight_result = tight.update_tight_from_crops(lambda rect: read_crop(both_frame, rect),
                                                [(117, 120, 140, 139), (118, 121, 130, 133)])
    assert tight_result.detections == both and (0, 0, width, height) not in crops
    assert tight_result.searched_origins < tiled_result.searched_origins
    assert tight.update_tight_from_crops(lambda rect: read_crop(removed, rect),
                                       [(128, 128, 129, 129)]).detections == edge_hit
    assert tight.update_tight_from_crops(lambda rect: read_crop(both_frame, rect),
                                       [(128, 128, 129, 129)]).detections == both


if __name__ == "__main__":
    _self_check()
    print("OpenCV full and regional template self-checks passed")
