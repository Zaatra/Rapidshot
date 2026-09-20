"""NumpyProcessor paths the rest of the suite never reached.

Line coverage showed the padded-pitch read in process() ran zero times: every
live capture here is 2560 px wide, a width no driver pads, and every synthetic
surface in the other tests was built with pitch == width * 4. A GPU that aligns
rows to 16 or 32 bytes produces a pitch wider than the image, and reading a
padded surface as if it were packed shears the picture diagonally -- plausible
pixels, wrong places, no exception.
"""
import ctypes

import numpy as np
import pytest

from rapidshot.processor.numpy_processor import NumpyProcessor


class FakeMappedRect:
    """DXGI_MAPPED_RECT over a BGRA image, with optional row padding.

    Padding bytes are filled with a sentinel so a read that strays into them
    shows up as a wrong value rather than as a lucky zero.
    """

    PAD = 0xEE

    def __init__(self, bgra, pitch=None):
        height, width, _ = bgra.shape
        self.Pitch = width * 4 if pitch is None else pitch
        self._backing = (ctypes.c_ubyte * (self.Pitch * height))()
        view = np.ctypeslib.as_array(self._backing).reshape(height, self.Pitch)
        view[:] = self.PAD
        view[:, :width * 4] = bgra.reshape(height, width * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def image(height=5, width=7, seed=7):
    rng = np.random.default_rng(seed)
    bgra = rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
    # Keep real pixels away from the padding sentinel so a stray read is
    # unambiguous.
    bgra[bgra == FakeMappedRect.PAD] = 0
    return bgra


def expected(bgra, mode):
    return {
        "BGRA": bgra,
        "RGB": bgra[..., [2, 1, 0]],
        "BGR": bgra[..., :3],
        "RGBA": bgra[..., [2, 1, 0, 3]],
    }[mode]


# --------------------------------------------------------------------------
# process(): padded rows
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["BGRA", "RGB", "RGBA", "BGR"])
def test_process_reads_a_padded_surface_row_by_row(mode):
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra, pitch=width * 4 + 12)

    result, _ = NumpyProcessor(mode).process(
        rect, width, height, (0, 0, width, height), 0)

    np.testing.assert_array_equal(result, expected(bgra, mode))


def test_process_padded_read_into_a_pooled_buffer():
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra, pitch=width * 4 + 4)
    pooled = np.zeros((height, width, 4), dtype=np.uint8)

    result, still_pooled = NumpyProcessor("BGRA").process(
        rect, width, height, (0, 0, width, height), 0, pooled)

    assert result is pooled and still_pooled is True
    np.testing.assert_array_equal(pooled, bgra)


def test_process_gray_on_a_padded_surface_matches_a_packed_one():
    bgra = image(height=9, width=13)
    height, width = bgra.shape[:2]
    region = (0, 0, width, height)

    packed, _ = NumpyProcessor("GRAY").process(FakeMappedRect(bgra), width, height, region, 0)
    padded, _ = NumpyProcessor("GRAY").process(
        FakeMappedRect(bgra, pitch=width * 4 + 20), width, height, region, 0)

    np.testing.assert_array_equal(padded, packed)


def test_process_rejects_a_mis_shaped_output_target_before_touching_the_frame():
    """A caller bug, so it raises instead of disappearing into the catch-all."""
    bgra = image()
    height, width = bgra.shape[:2]
    target = np.zeros((height, width + 1, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="output_target shape"):
        NumpyProcessor("RGB").process(
            FakeMappedRect(bgra), width, height, (0, 0, width, height), 0,
            output_target=target)


# --------------------------------------------------------------------------
# the column read strategy kept for benchmarks/dirty_rect_read_strategy.py
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pitch_extra", [0, 8])
def test_column_and_row_patch_reads_agree_inside_the_rect(pitch_extra):
    """The benchmark compares the two strategies' speed; that comparison means
    nothing unless they read the same pixels."""
    bgra = image(height=6, width=10)
    height, width = bgra.shape[:2]
    pitch = width * 4 + pitch_extra
    rect = FakeMappedRect(bgra, pitch=pitch)
    src = np.ctypeslib.as_array(rect._backing).reshape(height, pitch)
    proc = NumpyProcessor("RGB")
    left, top, right, bottom = 2, 1, 7, 5

    by_rows = np.zeros((height, width * 4), dtype=np.uint8)
    by_cols = np.zeros((height, width * 4), dtype=np.uint8)
    proc._read_patch_rows(by_rows, src, left, top, right, bottom,
                          0, width * 4, pitch, width * 4)
    proc._read_patch_columns(by_cols, src, left, top, right, bottom,
                             0, width * 4, pitch, width * 4)

    inside = np.s_[top:bottom, left * 4:right * 4]
    np.testing.assert_array_equal(by_cols[inside], by_rows[inside])
    np.testing.assert_array_equal(
        by_cols.reshape(height, width, 4)[top:bottom, left:right],
        bgra[top:bottom, left:right])
    # Columns means columns: nothing outside the rect is touched.
    outside = by_cols.reshape(height, width, 4).copy()
    outside[top:bottom, left:right] = 0
    assert not outside.any()


# --------------------------------------------------------------------------
# shot()
# --------------------------------------------------------------------------

def test_shot_bgra_copies_a_padded_surface_without_the_padding():
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra, pitch=width * 4 + 12)
    dst = np.zeros((height, width, 4), dtype=np.uint8)

    assert NumpyProcessor("BGRA").shot(dst, rect, width, height) is True

    np.testing.assert_array_equal(dst, bgra)


def test_shot_refuses_a_pitch_narrower_than_a_row():
    """Trusting it would read past the end of the mapped surface."""
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra)
    rect.Pitch = width * 4 - 4
    dst = np.zeros((height, width, 4), dtype=np.uint8)

    with pytest.raises(ValueError, match="pitch"):
        NumpyProcessor("BGRA").shot(dst, rect, width, height)
    assert not dst.any()


def test_shot_refuses_a_null_source():
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra)
    rect.pBits = ctypes.c_void_p(None)
    dst = np.zeros((height, width, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="source pointer"):
        NumpyProcessor("RGB").shot(dst, rect, width, height)


def test_shot_refuses_a_destination_with_no_address():
    bgra = image()
    height, width = bgra.shape[:2]

    with pytest.raises(ValueError, match="destination pointer"):
        NumpyProcessor("RGB").shot(None, FakeMappedRect(bgra), width, height, buffer_size=10**6)


def test_process_refuses_a_pitch_narrower_than_a_row():
    """`shot()` has refused this since it was written; `process()` did not, and
    `process()` is the path every `grab()` takes.

    A pitch smaller than a row means the strided view spans further than the
    mapped surface, so the last rows are read from whatever follows it. Nothing
    about the result's shape, dtype or range would show that.
    """
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra)
    rect.Pitch = width * 4 - 4

    with pytest.raises(ValueError, match="pitch"):
        NumpyProcessor("BGRA").process(rect, width, height, (0, 0, width, height), 0)


def test_process_reads_an_offset_region_from_a_padded_surface():
    """The two conditions that send `_read_rows` down its slow branch, together:
    a padded pitch and a region whose left edge is not zero. Correctness first,
    because the branch is about to be replaced."""
    bgra = image()
    height, width = bgra.shape[:2]
    left, top = 2, 1
    rect = FakeMappedRect(bgra, pitch=width * 4 + 12)

    out, _pooled = NumpyProcessor("BGRA").process(
        rect, width, height, (left, top, width, height), 0)

    np.testing.assert_array_equal(np.asarray(out), bgra[top:, left:])


def test_a_rotated_one_pixel_region_does_not_alias_the_pool_buffer():
    """`ascontiguousarray` hands back its argument unchanged when the view is
    already contiguous, and every rotation of a 1x1 region is a no-op view --
    so the caller received the pooled buffer itself while
    `is_still_pooled_buffer` said False, which is the aliasing the pool exists
    to prevent.

    `CupyProcessor` documents this exact case and uses `.copy()`; the NumPy
    path did not. Sizes either side included: the bug is that the *view* is
    contiguous, which a 1-pixel frame guarantees.
    """
    for width, height in ((1, 1), (1, 2), (2, 1)):
        bgra = image()[:height, :width]
        rect = FakeMappedRect(bgra)
        pooled = np.zeros((height, width, 4), dtype=np.uint8)

        out, still_pooled = NumpyProcessor("BGRA").process(
            rect, width, height, (0, 0, width, height), 90, pooled)

        assert still_pooled is False, (width, height)
        assert not np.shares_memory(np.asarray(out), pooled), (width, height)


def test_process_refuses_an_impossible_pitch():
    """The lower bound has a twin. `MAX_METADATA_BUFFER_BYTES` and
    `MAX_POINTER_SHAPE_BUFFER_BYTES` already cap the other two driver-reported
    sizes at 16 MB apiece; the pitch was the one left unbounded, and it sizes
    `(c_ubyte * (pitch * height))` directly.

    A corrupt value does not fail cleanly there -- it describes a region
    gigabytes long over an address that maps a few megabytes.
    """
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra)
    rect.Pitch = 64 * 1024 * 1024

    with pytest.raises(ValueError, match="pitch"):
        NumpyProcessor("BGRA").process(rect, width, height, (0, 0, width, height), 0)


def test_process_accepts_a_generously_padded_pitch():
    """The bound must clear real padding by a wide margin. A 4K BGRA row is
    30,720 bytes; drivers pad by tens of bytes, not megabytes."""
    bgra = image()
    height, width = bgra.shape[:2]
    rect = FakeMappedRect(bgra, pitch=width * 4 + 4096)

    out, _pooled = NumpyProcessor("BGRA").process(
        rect, width, height, (0, 0, width, height), 0)

    np.testing.assert_array_equal(np.asarray(out), bgra)
