"""The cursor position has to index the frame it came with.

DXGI reports the pointer against the whole duplicated output. A Frame may cover
only a region of it, so passing the raw value through points somewhere else
entirely -- and only when the region is off-origin, which is exactly the case
nobody checks by hand. Same rule, and the same reason, as `dirty_rects`.

A point is not a rect, though: one that lands outside the frame is kept and
shifted rather than dropped. A cursor whose hotspot sits just past the edge
still draws pixels inside the region, so a consumer compositing it needs the
true offset.
"""

import pytest

pytest.importorskip("comtypes")

from rapidshot.frame import CursorInfo, Frame  # noqa: E402


def _frame(region, cursor=None):
    return Frame(texture=object(), on_release=None, region=region, cursor=cursor)


def _cursor(position=(0, 0), **kwargs):
    return CursorInfo(visible=True, position=position, **kwargs)


# --------------------------------------------------------------------------
# Translation
# --------------------------------------------------------------------------

def test_an_off_origin_region_rebases_the_position():
    frame = _frame((100, 50, 300, 250), _cursor((150, 90)))

    assert frame.cursor.position == (50, 40)


def test_a_full_output_capture_leaves_the_position_alone():
    frame = _frame((0, 0, 1920, 1080), _cursor((640, 360)))

    assert frame.cursor.position == (640, 360)


def test_a_cursor_at_the_region_origin_lands_at_zero_zero():
    frame = _frame((100, 50, 300, 250), _cursor((100, 50)))

    assert frame.cursor.position == (0, 0)


def test_a_cursor_left_of_the_region_goes_negative():
    """Not clamped: a pointer just past the edge still draws pixels inside it,
    and clamping would claim it is somewhere it is not."""
    frame = _frame((100, 50, 300, 250), _cursor((90, 40)))

    assert frame.cursor.position == (-10, -10)


def test_a_cursor_beyond_the_region_stays_beyond_it():
    frame = _frame((100, 50, 300, 250), _cursor((1000, 800)))

    assert frame.cursor.position == (900, 750)


def test_the_hotspot_is_not_translated():
    """It is an offset inside the cursor's own shape, not a desktop position."""
    frame = _frame((100, 50, 300, 250), _cursor((150, 90), hotspot=(4, 7)))

    assert frame.cursor.hotspot == (4, 7)


# --------------------------------------------------------------------------
# Empty versus unknown
# --------------------------------------------------------------------------

def test_an_unreported_position_stays_none():
    """None means DXGI reported no position. It is not a coordinate, and it is
    not the same answer as a hidden cursor."""
    frame = _frame((100, 50, 300, 250), CursorInfo(visible=True, position=None))

    assert frame.cursor.position is None
    assert frame.cursor.visible is True


def test_a_hidden_cursor_still_reports_where_it_is():
    frame = _frame((100, 50, 300, 250),
                   CursorInfo(visible=False, position=(150, 90)))

    assert frame.cursor.visible is False
    assert frame.cursor.position == (50, 40)


def test_no_cursor_at_all_still_yields_a_cursor_info():
    frame = _frame((100, 50, 300, 250), None)

    assert frame.cursor.visible is False
    assert frame.cursor.position is None


# --------------------------------------------------------------------------
# The caller's object must not be moved under them
# --------------------------------------------------------------------------

def test_the_source_cursor_is_not_mutated():
    """Nothing stops one CursorInfo reaching two Frames; a frame shifting
    somebody else's data would compound on every use."""
    shared = _cursor((150, 90))

    _frame((100, 50, 300, 250), shared)

    assert shared.position == (150, 90)


def test_two_frames_translate_the_same_cursor_independently():
    shared = _cursor((150, 90))

    first = _frame((100, 50, 300, 250), shared)
    second = _frame((0, 0, 300, 250), shared)

    assert first.cursor.position == (50, 40)
    assert second.cursor.position == (150, 90)


def test_the_shape_buffer_is_shared_not_copied():
    """It can be megabytes per frame and is never rewritten."""
    shape = bytes(4096)
    frame = _frame((100, 50, 300, 250), _cursor((150, 90), shape=shape))

    assert frame.cursor.shape is shape


def test_every_other_field_survives_the_rebase():
    frame = _frame(
        (100, 50, 300, 250),
        _cursor((150, 90), hotspot=(2, 3), shape=b"xy",
                shape_type=2, shape_size=(32, 32), shape_pitch=128),
    )

    cursor = frame.cursor
    assert (cursor.visible, cursor.shape_type) == (True, 2)
    assert (cursor.shape_size, cursor.shape_pitch) == ((32, 32), 128)
    assert cursor.shape == b"xy"
