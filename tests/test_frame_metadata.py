"""Section 7.1's additions to Frame, and the recovery counters behind them.

These are the pieces ROADMAP section 7.1 calls "finish Frame, do not redesign
it": everything here is derived from data the capture path already had.

Headless by construction -- Frame takes plain values, so none of this needs a
desktop, a GPU or a duplicator. That matters: the capture paths these describe
are exactly the ones that cannot be exercised in CI.
"""

from __future__ import annotations

import pytest

from rapidshot.frame import CursorInfo, Frame


def make_frame(**overrides):
    params = dict(
        texture=object(),
        on_release=lambda: None,
        region=(0, 0, 100, 50),
    )
    params.update(overrides)
    return Frame(**params)


class TestSequenceAndGeneration:
    def test_default_to_zero(self):
        """A frame built without them is still valid; they are additive."""
        frame = make_frame()
        assert frame.sequence == 0
        assert frame.generation == 0

    def test_carry_what_capture_stamped(self):
        frame = make_frame(sequence=41, generation=3)
        assert frame.sequence == 41
        assert frame.generation == 3

    def test_survive_release(self):
        """Metadata outliving the surface is the existing contract.

        A consumer logging why a cached resource was rebuilt reads these after
        the frame is gone, which is the whole point of stamping them.
        """
        frame = make_frame(sequence=7, generation=2)
        frame.release()
        assert frame.sequence == 7
        assert frame.generation == 2


class TestChangedFraction:
    def test_none_when_metadata_unreadable(self):
        """None and empty mean different things and must not collapse."""
        assert make_frame(dirty_rects=None).changed_fraction is None

    def test_empty_list_means_assume_everything(self):
        """No rects is no information, so the safe reading is 1.0.

        Returning 0.0 here would tell a consumer it can skip the frame
        entirely, which is exactly backwards: an empty list is also what a mode
        change produces, and then the image differs completely.
        """
        assert make_frame(dirty_rects=[]).changed_fraction == 1.0

    def test_single_rect_is_its_area(self):
        frame = make_frame(region=(0, 0, 100, 100), dirty_rects=[(0, 0, 10, 10)])
        assert frame.changed_fraction == pytest.approx(0.01)

    def test_full_cover_is_one(self):
        frame = make_frame(region=(0, 0, 100, 100), dirty_rects=[(0, 0, 100, 100)])
        assert frame.changed_fraction == pytest.approx(1.0)

    def test_overlapping_rects_are_counted_once(self):
        """Two rects overlapping by half must not report 1.5x their area.

        Drivers do report overlapping regions. Summing areas naively can exceed
        the frame, and a consumer thresholding on "more than 90% changed" would
        then take the full-frame path for a frame that barely moved.
        """
        frame = make_frame(region=(0, 0, 100, 100),
                           dirty_rects=[(0, 0, 20, 10), (10, 0, 30, 10)])
        # Union spans x 0..30 over 10 rows = 300 of 10000.
        assert frame.changed_fraction == pytest.approx(0.03)

    def test_identical_rects_do_not_double_count(self):
        frame = make_frame(region=(0, 0, 100, 100),
                           dirty_rects=[(0, 0, 50, 50), (0, 0, 50, 50)])
        assert frame.changed_fraction == pytest.approx(0.25)

    def test_disjoint_rects_add(self):
        frame = make_frame(region=(0, 0, 100, 100),
                           dirty_rects=[(0, 0, 10, 10), (50, 50, 60, 60)])
        assert frame.changed_fraction == pytest.approx(0.02)

    def test_never_exceeds_one(self):
        frame = make_frame(region=(0, 0, 10, 10),
                           dirty_rects=[(0, 0, 10, 10), (0, 0, 10, 10), (2, 2, 8, 8)])
        assert frame.changed_fraction <= 1.0

    def test_degenerate_rects_are_dropped_before_they_reach_here(self):
        """Zero-width and inverted rects never survive construction.

        `_clip_to_region` discards them, so a frame whose rects were *all*
        degenerate arrives holding an empty list -- indistinguishable from one
        that reported no metadata at all, and therefore 1.0 rather than 0.0.

        That collapse is the safe direction: both cases mean "nothing here can
        tell you what changed", and answering 0.0 would invite a consumer to
        skip a frame that may have changed completely.
        """
        frame = make_frame(region=(0, 0, 100, 100),
                           dirty_rects=[(5, 5, 5, 50), (10, 10, 4, 4)])
        assert frame.dirty_rects == []
        assert frame.changed_fraction == 1.0

    def test_a_degenerate_rect_alongside_a_real_one_is_ignored(self):
        """The real rect still measures correctly; the degenerate one adds 0."""
        frame = make_frame(region=(0, 0, 100, 100),
                           dirty_rects=[(0, 0, 10, 10), (5, 5, 5, 50)])
        assert frame.changed_fraction == pytest.approx(0.01)


class TestAgeMs:
    def test_zero_without_a_present_time(self):
        """0.0 means "not reported", which is not the same as a fresh frame."""
        assert make_frame(present_time_qpc=0).age_ms == 0.0

    def test_positive_and_growing_for_a_real_timestamp(self):
        from rapidshot.frame import _qpc_now

        frame = make_frame(present_time_qpc=_qpc_now())
        first = frame.age_ms
        assert first >= 0.0
        for _ in range(200000):
            pass
        assert frame.age_ms >= first

    def test_never_negative_for_a_future_timestamp(self):
        """QPC read on another core can land slightly ahead; clamp not crash."""
        from rapidshot.frame import _qpc_now

        frame = make_frame(present_time_qpc=_qpc_now() + 10_000_000)
        assert frame.age_ms == 0.0


class TestCursor:
    def test_always_present_even_with_nothing_reported(self):
        cursor = make_frame().cursor
        assert isinstance(cursor, CursorInfo)
        assert cursor.visible is False
        assert cursor.position is None
        assert cursor.shape is None

    def test_falls_back_to_the_visible_flag(self):
        """`cursor_visible` predates this and must stay consistent with it."""
        frame = make_frame(cursor_visible=True)
        assert frame.cursor.visible is True
        assert frame.cursor_visible is True

    def test_carries_position_hotspot_and_shape(self):
        info = CursorInfo(visible=True, position=(120, 340), hotspot=(4, 4),
                          shape=b"\x00\x01", shape_type=2, shape_size=(32, 32),
                          shape_pitch=128)
        frame = make_frame(cursor=info, cursor_visible=True)
        assert frame.cursor.position == (120, 340)
        assert frame.cursor.hotspot == (4, 4)
        assert frame.cursor.shape == b"\x00\x01"
        assert frame.cursor.shape_size == (32, 32)
        assert frame.cursor.shape_pitch == 128

    def test_repr_names_the_shape_encoding(self):
        """Three encodings exist and compositing differs per encoding, so the
        repr says which one rather than only that a shape is present."""
        assert "monochrome" in repr(CursorInfo(shape_type=1))
        assert "color" in repr(CursorInfo(shape_type=2))
        assert "masked" in repr(CursorInfo(shape_type=4))
        assert "none" in repr(CursorInfo(shape_type=0))


class TestFrameStillUsesSlots:
    def test_no_instance_dict(self):
        """Frame is allocated per capture, so it stays __slots__-only.

        This test exists because adding the fields above without extending
        __slots__ broke every Frame construction in the suite -- the failure is
        an AttributeError at __init__, far from the omission that caused it.
        """
        frame = make_frame()
        assert not hasattr(frame, "__dict__")
        with pytest.raises(AttributeError):
            frame.something_new = 1

    def test_new_fields_are_declared(self):
        for name in ("_sequence", "_generation", "_cursor"):
            assert name in Frame.__slots__
