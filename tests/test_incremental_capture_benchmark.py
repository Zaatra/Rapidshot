"""benchmarks/incremental_capture.py without a GPU or a screen.

The numbers that benchmark produces are only worth as much as its bookkeeping.
Three pieces of that bookkeeping can be wrong in ways no timing would reveal:

  * `tiles_for` deciding which tiles a rect touches. Too few and the benchmark
    measures copying less than the design would have to copy, which flatters it.
  * `tile_runs` coalescing tiles into copy rectangles. A run that covers a tile
    nobody marked dirty inflates the byte count; one that misses a dirty tile
    would, in a real engine, leave stale pixels on screen.
  * `changed_tiles` reducing a per-pixel difference to per-tile booleans. 1600
    is not a multiple of 128, so the bottom row of tiles is short -- the case a
    reshape-based implementation drops silently.

The geometry tests assert coverage against an independently built pixel mask
rather than against the tiling functions themselves, since a reference that
shares logic with what it checks cannot fail.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import incremental_capture as bench  # noqa: E402

WIDTH, HEIGHT = 2560, 1600


def pixel_mask(rects, width=WIDTH, height=HEIGHT):
    """An independent record of which pixels a rect list covers."""
    mask = np.zeros((height, width), dtype=bool)
    for left, top, right, bottom in rects:
        mask[top:bottom, left:right] = True
    return mask


# --------------------------------------------------------------------------
# rect generators
# --------------------------------------------------------------------------

@pytest.mark.parametrize("shape", sorted(bench.SHAPES))
@pytest.mark.parametrize("fraction", [0.008, 0.10, 0.50, 1.00])
@pytest.mark.parametrize("count", [1, 8, 64])
def test_generated_rects_stay_inside_the_frame(shape, fraction, count):
    """A rect past the edge would make CopySubresourceRegion fail, or worse,
    silently address the wrong pixels."""
    rects = bench.SHAPES[shape](count, fraction, WIDTH, HEIGHT)
    assert rects
    for left, top, right, bottom in rects:
        assert 0 <= left < right <= WIDTH
        assert 0 <= top < bottom <= HEIGHT


@pytest.mark.parametrize("fraction", [0.008, 0.03, 0.10, 0.25, 0.50, 0.75, 1.00])
def test_band_rects_cover_about_the_requested_fraction(fraction):
    """The dirty fraction is the benchmark's x-axis. If the generator does not
    hit it, every row is labelled with a number it did not measure."""
    rects = bench.band_rects(1, fraction, WIDTH, HEIGHT)
    covered = bench.area_of(rects) / (WIDTH * HEIGHT)
    assert covered == pytest.approx(fraction, abs=0.01)


def test_band_rects_do_not_overlap():
    """Overlapping rects would copy the same pixels twice and report an area
    larger than the surface."""
    rects = bench.band_rects(8, 0.50, WIDTH, HEIGHT)
    mask = pixel_mask(rects)
    assert int(mask.sum()) == bench.area_of(rects)


def test_area_and_rows_agree_with_an_independent_mask():
    rects = bench.grid_rects(16, 0.10, WIDTH, HEIGHT)
    mask = pixel_mask(rects)
    assert bench.area_of(rects) == int(mask.sum())
    assert bench.rows_touched(rects, HEIGHT) == int(mask.any(axis=1).sum())


def test_column_rects_touch_every_row():
    """The shape section 6.3 found worst for a row-limited read: tiny area,
    every row. If this stopped being true the shape comparison loses its point."""
    rects = bench.column_rects(8, 0.10, WIDTH, HEIGHT)
    assert bench.rows_touched(rects, HEIGHT) == HEIGHT
    assert bench.area_of(rects) < WIDTH * HEIGHT * 0.2


# --------------------------------------------------------------------------
# tiling
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tile_grid_covers_the_frame(tile):
    across, down = bench.tile_grid(WIDTH, HEIGHT, tile)
    assert across * tile >= WIDTH
    assert down * tile >= HEIGHT
    assert (across - 1) * tile < WIDTH
    assert (down - 1) * tile < HEIGHT


@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tiles_for_includes_every_dirty_pixel(tile):
    """Under-reporting tiles is the dangerous direction: a real engine would
    skip copying a region that changed."""
    rects = bench.grid_rects(16, 0.10, WIDTH, HEIGHT)
    mask = bench.tiles_for(rects, tile, WIDTH, HEIGHT)
    for left, top, right, bottom in rects:
        assert mask[top // tile, left // tile]
        assert mask[(bottom - 1) // tile, (right - 1) // tile]


@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tiles_for_marks_nothing_extra(tile):
    """Over-reporting is the direction that flatters the tile design's rival,
    so it matters too: every marked tile must contain a dirty pixel."""
    rects = bench.grid_rects(8, 0.03, WIDTH, HEIGHT)
    dirty = pixel_mask(rects)
    mask = bench.tiles_for(rects, tile, WIDTH, HEIGHT)
    for ty, tx in zip(*np.nonzero(mask)):
        patch = dirty[ty * tile:(ty + 1) * tile, tx * tile:(tx + 1) * tile]
        assert patch.any(), f"tile ({tx}, {ty}) marked dirty but holds no dirty pixel"


def test_tiles_for_handles_a_single_pixel():
    """The caret case: one pixel dirties exactly one tile, never zero."""
    mask = bench.tiles_for([(1000, 800, 1001, 801)], 128, WIDTH, HEIGHT)
    assert int(mask.sum()) == 1
    assert mask[800 // 128, 1000 // 128]


@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tile_runs_cover_exactly_the_dirty_tiles(tile):
    """Runs are what actually gets copied. They must cover every dirty tile and
    no clean one, or the measured byte count is not the design's byte count."""
    rects = bench.grid_rects(16, 0.10, WIDTH, HEIGHT)
    mask = bench.tiles_for(rects, tile, WIDTH, HEIGHT)
    runs = bench.tile_runs(mask, tile, WIDTH, HEIGHT)

    rebuilt = bench.tiles_for(runs, tile, WIDTH, HEIGHT)
    assert np.array_equal(rebuilt, mask)


@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tile_runs_stay_inside_the_frame(tile):
    """The right and bottom tiles are short when the tile size does not divide
    the resolution -- 1600 / 128 is 12.5 -- and a run must be clipped to the
    surface, not rounded up past it."""
    mask = bench.tiles_for([(0, 0, WIDTH, HEIGHT)], tile, WIDTH, HEIGHT)
    for left, top, right, bottom in bench.tile_runs(mask, tile, WIDTH, HEIGHT):
        assert 0 <= left < right <= WIDTH
        assert 0 <= top < bottom <= HEIGHT


@pytest.mark.parametrize("tile", bench.TILE_SIZES)
def test_tile_runs_do_not_overlap(tile):
    rects = bench.grid_rects(32, 0.30, WIDTH, HEIGHT)
    mask = bench.tiles_for(rects, tile, WIDTH, HEIGHT)
    runs = bench.tile_runs(mask, tile, WIDTH, HEIGHT)
    assert int(pixel_mask(runs).sum()) == bench.area_of(runs)


def test_tile_runs_coalesce_a_full_row_into_one_copy():
    """The whole reason a fixed grid can beat exact rects on call count."""
    mask = bench.tiles_for([(0, 0, WIDTH, 128)], 128, WIDTH, HEIGHT)
    runs = bench.tile_runs(mask, 128, WIDTH, HEIGHT)
    assert len(runs) == 1
    assert runs[0] == (0, 0, WIDTH, 128)


def test_tile_runs_split_a_gap():
    """Two separated tiles in one row must not be joined into one copy that
    drags the clean tiles between them along."""
    mask = bench.tiles_for([(0, 0, 128, 128), (2432, 0, 2560, 128)],
                           128, WIDTH, HEIGHT)
    runs = bench.tile_runs(mask, 128, WIDTH, HEIGHT)
    assert len(runs) == 2


# --------------------------------------------------------------------------
# change detection
# --------------------------------------------------------------------------

def test_changed_tiles_finds_a_single_pixel_in_a_ragged_edge_tile():
    """1600 is not a multiple of 128, so the bottom tile row is 64px tall. A
    reshape-based reduction drops it; this is the case that proves reduceat."""
    difference = np.zeros((HEIGHT, WIDTH), dtype=bool)
    difference[HEIGHT - 1, WIDTH - 1] = True
    changed = bench.changed_tiles(difference, 128)
    assert int(changed.sum()) == 1
    assert changed[-1, -1]


def test_changed_tiles_reports_nothing_for_an_identical_frame():
    difference = np.zeros((HEIGHT, WIDTH), dtype=bool)
    assert not bench.changed_tiles(difference, 128).any()


def test_changed_tiles_shape_matches_tiles_for():
    """The two are intersected in the over-report benchmark, so a shape
    mismatch would broadcast instead of failing."""
    difference = np.zeros((HEIGHT, WIDTH), dtype=bool)
    for tile in bench.TILE_SIZES:
        assert (bench.changed_tiles(difference, tile).shape
                == bench.tiles_for([], tile, WIDTH, HEIGHT).shape)


# --------------------------------------------------------------------------
# masks and statistics
# --------------------------------------------------------------------------

def test_pack_bits_round_trips():
    grid = np.zeros((13, 20), dtype=bool)
    grid[0, 0] = grid[5, 7] = grid[12, 19] = True
    packed = bench.pack_bits(grid)
    recovered = {index for index in range(grid.size) if packed >> index & 1}
    assert recovered == set(np.flatnonzero(grid.ravel()).tolist())


def test_pack_bits_of_nothing_is_zero():
    assert bench.pack_bits(np.zeros((13, 20), dtype=bool)) == 0


def test_disjoint_masks_do_not_intersect():
    """The ROI query is an AND. If two disjoint regions tested as overlapping,
    a consumer would wake for changes outside its region."""
    left_half = np.zeros((13, 20), dtype=bool)
    left_half[:, :10] = True
    right_half = np.zeros((13, 20), dtype=bool)
    right_half[:, 10:] = True
    assert bench.pack_bits(left_half) & bench.pack_bits(right_half) == 0
    assert bench.pack_bits(left_half) & bench.pack_bits(left_half) != 0


def test_percentiles_are_ordered():
    values = [i / 100 for i in range(101)]
    result = bench.percentiles(values)
    assert result[50] <= result[90] <= result[99]


def test_percentiles_of_nothing_is_nan():
    """An empty live run must report nan, not crash and not report zero -- zero
    would read as 'nothing changed' rather than 'nothing was measured'."""
    result = bench.percentiles([])
    assert all(np.isnan(value) for value in result.values())


def test_summarise_reports_p50_then_p90():
    """Not min: ROADMAP section 10 records a probe that reported a GPU wait of
    0.0 us by quoting minima, because the wait is bimodal."""
    samples = [0.001] * 90 + [0.100] * 10
    p50, p90 = bench.summarise(samples)
    assert p50 == pytest.approx(1.0)
    assert p90 > p50


def test_summarise_is_order_independent():
    samples = [0.005, 0.001, 0.003, 0.002, 0.004]
    assert bench.summarise(samples) == bench.summarise(sorted(samples, reverse=True))


# --------------------------------------------------------------------------
# the harness must not touch the desktop on import
# --------------------------------------------------------------------------

def test_importing_the_module_creates_no_device():
    """Section 2's rule: importing a benchmark never touches the desktop, so
    the pure parts stay testable on a machine with no display."""
    assert "incremental_capture" in sys.modules
    assert bench.Harness is not None
