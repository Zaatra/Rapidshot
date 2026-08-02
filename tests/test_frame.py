"""Tests for the Frame object's GPU texture lifetime contract.

The contract these enforce is not stylistic: DXGI refuses the next
AcquireNextFrame while any reference to the previous desktop surface is
outstanding, so a Frame that outlives its window stalls capture entirely.
"""

import gc

import pytest

from rapidshot.frame import Frame, FrameReleasedError


def make_frame(on_release=None, **kw):
    released = {"count": 0}

    def _release():
        released["count"] += 1

    params = dict(
        texture=object(),
        on_release=on_release or _release,
        region=(0, 0, 640, 480),
        rotation_angle=0,
        present_time_qpc=123456789,
        accumulated_frames=1,
        protected_content=False,
        cursor_visible=True,
    )
    params.update(kw)
    return Frame(**params), released


# --------------------------------------------------------------------------
# lifetime
# --------------------------------------------------------------------------

def test_texture_accessible_while_live():
    frame, _ = make_frame()
    assert frame.d3d11_texture is not None
    assert frame.released is False


def test_texture_rejected_after_release():
    frame, _ = make_frame()
    frame.release()
    assert frame.released is True
    with pytest.raises(FrameReleasedError, match="after the frame was released"):
        _ = frame.d3d11_texture


def test_release_invokes_the_callback_exactly_once():
    frame, released = make_frame()
    frame.release()
    frame.release()  # idempotent
    frame.release()
    assert released["count"] == 1


def test_context_manager_releases_on_exit():
    frame, released = make_frame()
    with frame as f:
        assert f is frame
        assert f.d3d11_texture is not None
    assert frame.released is True
    assert released["count"] == 1


def test_context_manager_releases_on_exception():
    """A failure inside the block must not leave capture stalled."""
    frame, released = make_frame()
    with pytest.raises(ValueError):
        with frame:
            raise ValueError("boom")
    assert frame.released is True
    assert released["count"] == 1


def test_exception_propagates_out_of_context_manager():
    frame, _ = make_frame()
    with pytest.raises(RuntimeError, match="inner"):
        with frame:
            raise RuntimeError("inner")


def test_garbage_collection_releases_as_a_safety_net(caplog):
    released = {"count": 0}

    def _release():
        released["count"] += 1

    frame = Frame(texture=object(), on_release=_release, region=(0, 0, 4, 4))
    del frame
    gc.collect()
    assert released["count"] == 1, "GC must still hand the texture back"


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------

def test_metadata_survives_release():
    """Only GPU resources die with the frame; metadata stays readable."""
    frame, _ = make_frame(accumulated_frames=4, protected_content=True)
    frame.release()
    assert frame.region == (0, 0, 640, 480)
    assert frame.width == 640
    assert frame.height == 480
    assert frame.accumulated_frames == 4
    assert frame.protected_content is True
    assert frame.cursor_visible is True
    assert frame.timestamp_qpc == 123456789


def test_timestamp_converts_qpc_to_seconds():
    frame, _ = make_frame(present_time_qpc=0)
    assert frame.timestamp == 0.0

    frame2, _ = make_frame(present_time_qpc=10_000_000)
    # Whatever the QPC frequency is, seconds must be positive and proportional.
    assert frame2.timestamp > 0
    assert frame2.timestamp == pytest.approx(
        frame2.timestamp_qpc / (frame2.timestamp_qpc / frame2.timestamp), rel=1e-9)


def test_dimensions_follow_the_region():
    frame, _ = make_frame(region=(100, 50, 400, 250))
    assert frame.width == 300
    assert frame.height == 200


def test_repr_shows_lifetime_state():
    frame, _ = make_frame()
    assert "live" in repr(frame)
    frame.release()
    assert "released" in repr(frame)


# --------------------------------------------------------------------------
# ScreenCapture guard
# --------------------------------------------------------------------------

def test_capture_refuses_to_reacquire_while_a_frame_is_live():
    """
    The whole point of the type: an unreleased Frame must produce a clear
    error, not the opaque DXGI_ERROR_INVALID_CALL that stalled capture before.
    """
    from rapidshot.capture import ScreenCapture

    cam = ScreenCapture.__new__(ScreenCapture)
    frame, _ = make_frame()
    cam._live_frame = frame

    for caller in ("grab()", "shot()", "grab_frame()"):
        with pytest.raises(RuntimeError, match="has not been released"):
            cam._ensure_no_live_frame(caller)

    # Once released, the guard lets capture proceed and forgets the frame.
    frame.release()
    cam._ensure_no_live_frame("grab()")
    assert cam._live_frame is None


def test_guard_passes_when_no_frame_outstanding():
    from rapidshot.capture import ScreenCapture

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._live_frame = None
    cam._ensure_no_live_frame("grab()")  # must not raise
