"""Tests for dirty-rect accumulated conversion (ROADMAP.md 6.3).

The optimisation converts only the regions DXGI marked dirty and takes the rest
of the frame from the previous one. That is fast — 12-15x on a normal desktop,
per benchmarks/dirty_rect_pipeline.py — and every way it can go wrong produces a
frame that is the right shape, the right dtype, and quietly part historical.
Speed is not what these tests are about.

The failure modes being pinned:

  * patching onto an accumulator that was never fully populated;
  * patching after the captured region moved, blending two places on screen;
  * handing back a view into the accumulator, which the next frame overwrites;
  * treating an empty rect list as "nothing changed" rather than "no metadata".
"""

import ctypes

import numpy as np
import pytest

from rapidshot.processor.numpy_processor import NumpyProcessor

WIDTH, HEIGHT = 64, 48


class FakeMappedRect:
    """A DXGI_MAPPED_RECT-alike over a ctypes buffer."""

    def __init__(self, bgra):
        h, w, _ = bgra.shape
        self.Pitch = w * 4
        self._backing = (ctypes.c_ubyte * (self.Pitch * h))()
        view = np.ctypeslib.as_array(self._backing).reshape(h, self.Pitch)
        view[:, : w * 4] = bgra.reshape(h, w * 4)
        self.pBits = ctypes.cast(self._backing, ctypes.c_void_p)


def frame(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (HEIGHT, WIDTH, 4), dtype=np.uint8)


def run(proc, bgra, dirty_rects=None, rotation=0):
    """One capture through the processor, returning the converted frame."""
    buffer = np.empty((HEIGHT, WIDTH, 4), np.uint8)
    out, pooled = proc.process(
        FakeMappedRect(bgra), WIDTH, HEIGHT, (0, 0, WIDTH, HEIGHT),
        rotation, buffer, dirty_rects=dirty_rects,
    )
    return out, pooled


def reference(proc, bgra):
    """What a full conversion of this frame produces."""
    out = np.empty((HEIGHT, WIDTH, proc.output_channels), np.uint8)
    proc.convert_into(bgra, out)
    return out


class TestCorrectness:
    def test_patched_frame_matches_a_full_conversion(self):
        """The whole point: same answer, less work."""
        proc = NumpyProcessor("RGB")
        first, second = frame(1), frame(2)

        run(proc, first, dirty_rects=[(0, 0, WIDTH, HEIGHT)])   # populate
        patched, _ = run(proc, second, dirty_rects=[(8, 8, 24, 24)])

        expected = reference(proc, first)
        expected[8:24, 8:24] = reference(proc, second)[8:24, 8:24]
        assert np.array_equal(patched, expected)

    def test_untouched_regions_carry_the_previous_frame(self):
        proc = NumpyProcessor("RGB")
        first, second = frame(1), frame(2)

        run(proc, first, dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        patched, _ = run(proc, second, dirty_rects=[(0, 0, 8, 8)])

        outside = reference(proc, first)
        assert np.array_equal(patched[8:, 8:], outside[8:, 8:])

    def test_the_first_frame_is_converted_in_full(self):
        """Nothing to patch onto yet, so the fast path must not engage."""
        proc = NumpyProcessor("RGB")
        only = frame(3)

        out, _ = run(proc, only, dirty_rects=[(0, 0, 4, 4)])

        assert np.array_equal(out, reference(proc, only)), \
            "patched onto an accumulator that was never populated"

    def test_result_is_not_a_view_into_the_accumulator(self):
        """A view would be rewritten by the next capture, silently."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        held, _ = run(proc, frame(2), dirty_rects=[(0, 0, 8, 8)])
        snapshot = held.copy()

        run(proc, frame(3), dirty_rects=[(0, 0, WIDTH // 2, HEIGHT // 2)])

        assert np.array_equal(held, snapshot), \
            "a previously returned frame changed when the next one was captured"

    def test_never_returns_the_pooled_buffer_flag_when_patching(self):
        """grab() would check the pool buffer back in and recycle it."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        _, pooled = run(proc, frame(2), dirty_rects=[(0, 0, 8, 8)])
        assert pooled is False


class TestFallbacks:
    def test_no_metadata_converts_everything(self):
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        current = frame(2)

        out, _ = run(proc, current, dirty_rects=None)

        assert np.array_equal(out, reference(proc, current))

    def test_an_empty_rect_list_converts_everything(self):
        """Empty means no rects were reported, not that nothing changed.

        A mode change or a coalescing driver can report none while the image
        differs completely, so treating empty as "reuse the last frame" would
        freeze the capture.
        """
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        current = frame(2)

        out, _ = run(proc, current, dirty_rects=[])

        assert np.array_equal(out, reference(proc, current))

    def test_a_mostly_dirty_frame_converts_everything(self):
        """Above the limit, patching costs more than it saves."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        current = frame(2)
        almost_all = [(0, 0, WIDTH, int(HEIGHT * 0.95))]

        out, _ = run(proc, current, dirty_rects=almost_all)

        assert np.array_equal(out, reference(proc, current))

    def test_out_of_range_rects_are_not_trusted(self):
        """Better a full conversion than a write outside the frame."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        current = frame(2)

        out, _ = run(proc, current, dirty_rects=[(0, 0, WIDTH + 10, HEIGHT)])

        assert np.array_equal(out, reference(proc, current))

    def test_rotation_bypasses_the_accumulator(self):
        """A rotated accumulator would need its regions rotated too."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        current = frame(2)

        out, _ = run(proc, current, dirty_rects=[(0, 0, 8, 8)], rotation=90)

        expected = np.ascontiguousarray(np.rot90(reference(proc, current), k=1))
        assert np.array_equal(out, expected)

    def test_bgra_keeps_the_zero_copy_path(self):
        """BGRA returns the pool buffer directly; an accumulator would add a copy."""
        proc = NumpyProcessor("BGRA")
        current = frame(2)

        out, pooled = run(proc, current, dirty_rects=[(0, 0, 8, 8)])

        assert pooled is True
        assert np.array_equal(out, current)


class TestProcessorDispatch:
    """The wrapper `grab()` actually calls, not the backend directly.

    `grab()` goes through `Processor`, whose `process()` had a fixed signature.
    Adding `dirty_rects` to the NumPy backend alone broke capture completely —
    the TypeError became a silent None inside `_grab()`'s catch-all — and the
    entire suite still passed, because nothing exercised this seam.
    """

    def test_the_wrapper_accepts_dirty_rects(self):
        from rapidshot.processor.base import Processor

        proc = Processor(backend=None, output_color="RGB")
        buffer = np.empty((HEIGHT, WIDTH, 4), np.uint8)

        out, _ = proc.process(
            FakeMappedRect(frame(1)), WIDTH, HEIGHT, (0, 0, WIDTH, HEIGHT),
            0, buffer, dirty_rects=[(0, 0, 8, 8)],
        )

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_the_wrapper_works_without_dirty_rects(self):
        """The old call must keep working; every other caller uses it."""
        from rapidshot.processor.base import Processor

        proc = Processor(backend=None, output_color="RGB")
        buffer = np.empty((HEIGHT, WIDTH, 4), np.uint8)

        out, _ = proc.process(
            FakeMappedRect(frame(1)), WIDTH, HEIGHT, (0, 0, WIDTH, HEIGHT),
            0, buffer,
        )

        assert out.shape == (HEIGHT, WIDTH, 3)

    def test_backends_without_an_accumulator_are_not_passed_rects(self):
        """Forwarding to a backend that cannot take the argument is a TypeError.

        Which `_grab()` would swallow into a None return and a re-init loop, so
        the capture would simply stop with nothing useful logged.
        """
        from rapidshot.processor.base import Processor

        class LegacyBackend:
            color_mode = "RGB"

            def __init__(self):
                self.seen = None

            def process(self, rect, width, height, region, rotation_angle,
                        output_buffer=None):
                self.seen = (width, height)
                return np.zeros((height, width, 3), np.uint8), False

        proc = Processor.__new__(Processor)
        proc.backend = LegacyBackend()
        proc.color_mode = "RGB"

        assert proc.backend_supports_dirty_rects is False
        out, _ = proc.process(None, WIDTH, HEIGHT, (0, 0, WIDTH, HEIGHT), 0,
                              None, dirty_rects=[(0, 0, 8, 8)])
        assert proc.backend.seen == (WIDTH, HEIGHT)
        assert out.shape == (HEIGHT, WIDTH, 3)


class TestInvalidation:
    def test_a_full_conversion_leaves_a_patchable_accumulator(self):
        """Otherwise the fast path could never engage at all."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        assert proc._accum_valid is True

    def test_a_frame_without_metadata_invalidates_it(self):
        """That frame did not update the accumulator, so it is now stale."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        run(proc, frame(2), dirty_rects=None)
        assert proc._accum_valid is False

    def test_stale_accumulator_is_not_patched_onto(self):
        """The observable consequence of the invalidation above."""
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        run(proc, frame(2), dirty_rects=None)          # bypasses, invalidates
        current = frame(3)

        out, _ = run(proc, current, dirty_rects=[(0, 0, 8, 8)])

        assert np.array_equal(out, reference(proc, current))

    def test_explicit_invalidation_forces_a_full_conversion(self):
        proc = NumpyProcessor("RGB")
        run(proc, frame(1), dirty_rects=[(0, 0, WIDTH, HEIGHT)])
        proc.invalidate_accumulator()
        current = frame(2)

        out, _ = run(proc, current, dirty_rects=[(0, 0, 8, 8)])

        assert np.array_equal(out, reference(proc, current))
