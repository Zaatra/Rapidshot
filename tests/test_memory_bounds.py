"""Memory must be bounded, and the two capture paths are bounded differently.

`benchmarks/memory_profile.py` measured this on Machine B on 2026-09-14 and the
result is specific enough to pin: over a 60-second static run, `grab_frame()`
and release costs **1.2 MB** above the same process before its first frame,
while `grab()` costs **138.6 MB**. The surface pool is not the problem; the
NumPy output path is. These tests exist so that stays true while ROADMAP 7.2's
memory work changes the machinery underneath it.

What is pinned here is **growth**, not absolute size. An absolute threshold
would encode one machine's resolution, and 2560x1600 BGRA is 16.4 MB a frame
where 1080p is 8.3 -- a test that passes on a laptop and fails on a workstation
teaches people to ignore it. Unbounded growth is wrong everywhere.

The thresholds are deliberately far above the measured figures. A regression
that matters here is a buffer per frame at capture rate, which is tens of MB a
second; anything that squeaks past these numbers is a change worth a
measurement, not a red suite.
"""
import gc
import time

import pytest

import rapidshot

psutil = pytest.importorskip("psutil", reason="psutil not installed")

#: Long enough that a per-frame leak is unmissable at capture rate, short
#: enough to belong in a test suite.
MEASURE_SECONDS = 4.0
WARMUP_SECONDS = 1.5
#: ~9 MB/s would be one leaked 1080p frame a second; the measured paths sit
#: three orders of magnitude below this.
MAX_GROWTH_MB = 40.0


def working_set_mb():
    return psutil.Process().memory_info().rss / 1e6


def _drive(camera, take, seconds):
    """Capture for `seconds`, returning (frames, working set at start, at end).

    The clock starts after a warm-up, because the first allocations of a
    session are real and are not what this measures.
    """
    deadline = time.perf_counter() + WARMUP_SECONDS
    frames = 0
    while time.perf_counter() < deadline:
        frames += bool(take(camera))
    if frames == 0:
        pytest.skip("no frames captured - the screen must be changing")

    gc.collect()
    start_mb = working_set_mb()
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        frames += bool(take(camera))
    gc.collect()
    return frames, start_mb, working_set_mb()


def _take_frame(camera):
    frame = camera.grab_frame()
    if frame is None:
        return False
    frame.release()
    return True


def _take_array(camera):
    return camera.grab() is not None


@pytest.fixture
def camera():
    cam = rapidshot.create()
    yield cam
    cam.release()


def test_grab_frame_does_not_grow(camera):
    """The surface pool is a pool: holding no frame, it must reuse rather than
    allocate. Measured at 1.2 MB over 60 seconds."""
    frames, start_mb, end_mb = _drive(camera, _take_frame, MEASURE_SECONDS)
    growth = end_mb - start_mb
    assert growth < MAX_GROWTH_MB, (
        f"working set grew {growth:.1f} MB over {frames} frames "
        f"({start_mb:.1f} -> {end_mb:.1f} MB); the surface pool should reuse")


def test_grab_does_not_grow(camera):
    """`grab()` costs far more than `grab_frame()` in steady state, but the
    cost must still be a level rather than a slope -- the output pool is
    reused, not grown."""
    frames, start_mb, end_mb = _drive(camera, _take_array, MEASURE_SECONDS)
    growth = end_mb - start_mb
    assert growth < MAX_GROWTH_MB, (
        f"working set grew {growth:.1f} MB over {frames} frames "
        f"({start_mb:.1f} -> {end_mb:.1f} MB); the output pool should be reused")


def test_released_frames_do_not_accumulate(camera):
    """Releasing is what returns a buffer. A frame held and released many times
    must not leave anything behind -- this is the ownership property 2.5's
    recovery work established, checked as memory rather than as state."""
    for _ in range(30):
        frame = camera.grab_frame()
        if frame is not None:
            frame.release()
    gc.collect()
    start_mb = working_set_mb()

    held = 0
    for _ in range(200):
        frame = camera.grab_frame()
        if frame is not None:
            held += 1
            frame.release()
    gc.collect()
    if held == 0:
        pytest.skip("no frames captured - the screen must be changing")
    growth = working_set_mb() - start_mb
    assert growth < MAX_GROWTH_MB, (
        f"{held} acquire/release cycles grew the working set {growth:.1f} MB")
