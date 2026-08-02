"""Tests for dirty-rect metadata (ROADMAP.md 6.3).

Two things are being pinned here.

The coordinate mapping: DXGI reports dirty rects relative to the whole
duplicated output, while a Frame may cover only a region of it. Handing back
raw desktop coordinates would make `frame.dirty_rects` index outside the frame
whenever a region is in use — wrong only when the region is off-origin, which
is exactly the case nobody tests by hand.

And the retrieval: `GetFrameDirtyRects` takes a caller-allocated buffer and can
demand a bigger one, so the buffer-growth path is driven with the HRESULT
Windows actually returns.
"""

import ctypes

import pytest

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

from rapidshot._libs.dxgi import (  # noqa: E402
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_MORE_DATA,
    RECT,
)
from rapidshot.core.duplicator import Duplicator  # noqa: E402
from rapidshot.frame import Frame  # noqa: E402


def com_error(hresult, message="injected failure"):
    return comtypes.COMError(hresult, message, (None, None, None, 0, None))


class FakeFrameInfo:
    def __init__(self, total_metadata_bytes):
        self.TotalMetadataBufferSize = total_metadata_bytes


class FakeDuplication:
    """Stands in for IDXGIOutputDuplication's dirty-rect call."""

    def __init__(self, rects=(), errors=(), report_required=None):
        self.rects = list(rects)
        self.errors = list(errors)
        self.report_required = report_required
        self.calls = []

    def GetFrameDirtyRects(self, buffer_bytes, buffer, used_ref):
        self.calls.append(buffer_bytes)
        if self.errors:
            error = self.errors.pop(0)
            if error == DXGI_ERROR_MORE_DATA and self.report_required is not None:
                used_ref._obj.value = self.report_required
            raise com_error(error)
        for i, (left, top, right, bottom) in enumerate(self.rects):
            buffer[i].left = left
            buffer[i].top = top
            buffer[i].right = right
            buffer[i].bottom = bottom
        used_ref._obj.value = len(self.rects) * ctypes.sizeof(RECT)


def make_duplicator(duplication):
    dup = Duplicator.__new__(Duplicator)
    dup.duplicator = duplication
    dup._frame_acquired = True
    dup.last_error = ""
    return dup


def frame_with(rects, region=(0, 0, 100, 100), coalesced=False):
    return Frame(
        texture=object(),
        on_release=lambda: None,
        region=region,
        dirty_rects=rects,
        rects_coalesced=coalesced,
    )


class TestCoordinateMapping:
    def test_full_screen_frame_passes_rects_through(self):
        frame = frame_with([(10, 20, 30, 40)], region=(0, 0, 100, 100))
        assert frame.dirty_rects == [(10, 20, 30, 40)]

    def test_rects_are_relative_to_an_offset_region(self):
        """The bug this exists to prevent: desktop coordinates leaking out."""
        frame = frame_with([(60, 70, 80, 90)], region=(50, 50, 150, 150))
        assert frame.dirty_rects == [(10, 20, 30, 40)]

    def test_rects_outside_the_region_are_dropped(self):
        frame = frame_with([(0, 0, 10, 10)], region=(50, 50, 150, 150))
        assert frame.dirty_rects == []

    def test_rects_straddling_the_edge_are_clipped(self):
        frame = frame_with([(40, 40, 60, 60)], region=(50, 50, 150, 150))
        # Only the part inside the region survives, in frame coordinates.
        assert frame.dirty_rects == [(0, 0, 10, 10)]

    def test_a_rect_touching_only_the_border_is_dropped(self):
        """Zero-area after clipping is not a region anyone can redraw."""
        frame = frame_with([(30, 30, 50, 50)], region=(50, 50, 150, 150))
        assert frame.dirty_rects == []

    def test_every_rect_lands_inside_the_frame(self):
        region = (50, 50, 150, 150)
        rects = [(40, 40, 60, 60), (100, 100, 200, 200), (60, 60, 70, 70)]
        frame = frame_with(rects, region=region)
        width, height = region[2] - region[0], region[3] - region[1]
        for left, top, right, bottom in frame.dirty_rects:
            assert 0 <= left < right <= width
            assert 0 <= top < bottom <= height

    def test_unknown_metadata_stays_unknown(self):
        """None must not be flattened into an empty list.

        Empty means "no rects reported"; None means "could not be read". A
        caller that skips unchanged regions has to tell those apart, or it
        silently skips everything on a frame whose metadata failed.
        """
        assert frame_with(None).dirty_rects is None

    def test_coalesced_flag_is_carried(self):
        assert frame_with([], coalesced=True).rects_coalesced is True
        assert frame_with([]).rects_coalesced is False

    def test_metadata_survives_release(self):
        frame = frame_with([(10, 20, 30, 40)])
        frame.release()
        assert frame.dirty_rects == [(10, 20, 30, 40)]


class TestRetrieval:
    def test_rects_are_read_from_the_buffer(self):
        rects = [(0, 0, 10, 10), (20, 20, 40, 50)]
        dup = make_duplicator(FakeDuplication(rects))

        got = dup.get_frame_dirty_rects(FakeFrameInfo(64 * ctypes.sizeof(RECT)))

        assert got == rects

    def test_no_metadata_means_no_rects(self):
        """Zero-sized metadata is a definite answer, not a failure."""
        dup = make_duplicator(FakeDuplication([(0, 0, 1, 1)]))
        assert dup.get_frame_dirty_rects(FakeFrameInfo(0)) == []

    def test_a_bigger_buffer_is_requested_once_and_reused(self):
        """DXGI writes nothing and reports the size it needs; honour it."""
        needed = 4 * ctypes.sizeof(RECT)
        dup = make_duplicator(FakeDuplication(
            rects=[(1, 2, 3, 4)],
            errors=[DXGI_ERROR_MORE_DATA],
            report_required=needed,
        ))

        got = dup.get_frame_dirty_rects(FakeFrameInfo(ctypes.sizeof(RECT)))

        assert got == [(1, 2, 3, 4)]
        assert len(dup.duplicator.calls) == 2, "should retry exactly once"
        assert dup.duplicator.calls[1] == needed

    def test_retrying_forever_is_not_possible(self):
        """A driver that always asks for more must not hang the capture loop."""
        dup = make_duplicator(FakeDuplication(
            errors=[DXGI_ERROR_MORE_DATA, DXGI_ERROR_MORE_DATA],
            report_required=ctypes.sizeof(RECT),
        ))

        assert dup.get_frame_dirty_rects(FakeFrameInfo(ctypes.sizeof(RECT))) is None
        assert len(dup.duplicator.calls) == 2

    def test_access_lost_propagates(self):
        """The duplicator is dead; update_frame owns that recovery."""
        dup = make_duplicator(FakeDuplication(errors=[DXGI_ERROR_ACCESS_LOST]))

        with pytest.raises(comtypes.COMError):
            dup.get_frame_dirty_rects(FakeFrameInfo(ctypes.sizeof(RECT)))

    def test_no_acquired_frame_means_unknown(self):
        """The metadata belongs to an acquired frame and is gone after release."""
        dup = make_duplicator(FakeDuplication([(0, 0, 1, 1)]))
        dup._frame_acquired = False

        assert dup.get_frame_dirty_rects(FakeFrameInfo(64)) is None
