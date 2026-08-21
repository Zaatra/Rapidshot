"""Rebuilding duplication after access loss — against real DXGI, not a mock.

`_on_output_change()` is the exclusive-fullscreen and mode-change path: DXGI
invalidates the duplication object, and this tears down the duplicator and the
stage surface and rebuilds them. The bug it was written for was a *surviving*
stale stage surface, still sized for the previous resolution, which is what
produced the "black screen in fullscreen" symptom.

**What these cover and what they do not.** They invoke the recovery directly,
so the teardown and rebuild run for real against real hardware — releasing the
duplicator, rebuilding the stage surface, re-reading the output description,
and capturing again afterwards. They do **not** cover *detection*: making DXGI
genuinely return `DXGI_ERROR_ACCESS_LOST` needs a true exclusive-fullscreen
swapchain or a display mode change, neither of which a test suite should
inflict on the machine it is running on. That half remains fault-injection
tested, and the two halves together are what the path is made of.

Needs a desktop session, so these skip in CI.
"""
import numpy as np
import pytest

import rapidshot


def grab_one(camera, tries=500):
    for _ in range(tries):
        buf = camera.grab()
        if buf is not None:
            return buf
    return None


@pytest.fixture
def camera():
    cam = rapidshot.create(output_color="RGB")
    try:
        yield cam
    finally:
        cam.release()
        rapidshot.reset()


def test_capture_works_after_a_rebuild(camera):
    before = grab_one(camera)
    if before is None:
        pytest.skip("no frames captured — the screen must be changing")
    shape_before = np.asarray(before).shape
    release = getattr(before, "release", None)
    if release:
        release()

    assert camera._on_output_change() is True, "rebuild reported failure"

    after = grab_one(camera)
    assert after is not None, "capture did not recover after a rebuild"
    assert np.asarray(after).shape == shape_before
    release = getattr(after, "release", None)
    if release:
        release()


def test_rebuild_replaces_the_duplicator_and_stage_surface(camera):
    """The stale-surface bug, pinned by identity rather than by symptom.

    A surviving stage surface still sized for the old mode is what produced
    black frames. Comparing pixels would not catch it — the old surface
    returns perfectly plausible content until the resolution actually changes.
    """
    first = grab_one(camera)
    if first is None:
        pytest.skip("no frames captured — the screen must be changing")
    release = getattr(first, "release", None)
    if release:
        release()

    old_duplicator = camera._duplicator
    old_stagesurf_texture = camera._stagesurf.texture

    camera._on_output_change()

    assert camera._duplicator is not old_duplicator, (
        "the duplicator survived the rebuild")
    assert camera._stagesurf.texture is not old_stagesurf_texture, (
        "the stage surface survived the rebuild; this is the exact defect "
        "that produced black frames on a mode change")


def test_repeated_rebuilds_do_not_degrade(camera):
    """Recovery has to be repeatable: a mode switch in and out of a game is
    at least two of these back to back, and the retry budget must reset."""
    for round_number in range(4):
        assert camera._on_output_change() is True, (
            f"rebuild {round_number} reported failure")
        buf = grab_one(camera)
        assert buf is not None, f"capture did not recover on round {round_number}"
        release = getattr(buf, "release", None)
        if release:
            release()


def test_region_and_resolution_survive_a_rebuild(camera):
    """`_on_output_change` re-reads the output and recomputes the region.

    A region the user set must not be silently reset to full-screen by a
    rebuild the user never asked for.
    """
    width, height = camera.width, camera.height
    region = (10, 20, min(410, width), min(320, height))
    camera.region = region
    camera._region_set_by_user = True

    camera._on_output_change()

    assert (camera.width, camera.height) == (width, height)
    assert camera.region == region, (
        "a user-set region was lost across the rebuild")
