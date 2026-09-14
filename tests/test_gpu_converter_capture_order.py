"""A converter's first read of a new frame must be that frame, not the last one.

**The bug this pins** (found 2026-09-14 by ``test_tensor_stream.py``).
``AcquireNextFrame`` returns once the copy into the duplication surface is
*submitted* on the capture device's D3D11 queue, not once it has run. A D3D12
dispatch reading the surface through a shared handle could overtake it and
convert the previous frame's pixels: 7–11 of 100 first reads against a moving
source. No error, correct shape, plausible content — the frame before.

``converter12.rs`` now signals a fence shared with D3D11 behind the capture
work and has the D3D12 queue wait on it (``order_after_capture``). Measured
after: 0 of 400.

**Why the other converter tests never saw it.** They hold one frame for a
whole module and read it long after acquisition, when the copy has long
finished. The bug needs a *first* read *immediately* after a *new* frame of
*changing* content — exactly what a stream does, and what this test does.

**How staleness is detected without a ground truth.** Read the frame at once,
then again 20 ms later with the frame still held. A held frame does not change
(verified separately), so any difference means the first read saw content that
was not yet in the surface.
"""

import time

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

# The unfixed rate was ~4-10%, so a short run can miss it by luck: at 4%, 60
# frames see no stale read ~9% of the time — observed once while checking this
# test fails without the fix. 150 frames puts that near 0.2%.
FRAMES = 150
# Long enough for the capture copy to finish (5 ms sufficed when measured),
# short enough to keep the run to a few seconds.
SETTLE_S = 0.02


@pytest.fixture(scope="module")
def camera(motion):
    cam = rapidshot.create(output_color="BGRA")
    yield cam
    cam.release()


def stale_first_reads(camera, crop, frames=FRAMES, **options):
    converter = None
    stale = checked = 0
    deadline = time.monotonic() + 60
    while checked < frames and time.monotonic() < deadline:
        frame = camera.grab_frame()
        if frame is None:
            continue
        with frame:
            if converter is None:
                # Built on a frame that is not checked: construction takes long
                # enough to let that frame's copy finish, which would hide it.
                converter = rapidshot.GpuConverter(frame, (96, 64), crop=crop, **options)
                continue
            first = converter.process(frame).numpy().copy()
            time.sleep(SETTLE_S)
            second = converter.process(frame).numpy().copy()
        stale += not np.array_equal(first, second)
        checked += 1
    if checked < frames:
        pytest.skip(f"only {checked} changed frames arrived in 60 s")
    return stale


@pytest.mark.parametrize(
    "options",
    [
        dict(dtype="uint8", layout="nhwc", sampling="nearest"),
        dict(dtype="float16", sampling="bilinear"),
    ],
    ids=["bgra8", "fp16"],
)
def test_first_read_of_a_new_frame_is_not_the_previous_frame(camera, motion, options):
    assert stale_first_reads(camera, motion, **options) == 0


def test_the_motion_source_actually_moves(camera, motion):
    """Guards the test above against passing vacuously on a still screen."""
    converter = None
    images = []
    while len(images) < 6:
        frame = camera.grab_frame()
        if frame is None:
            continue
        with frame:
            if converter is None:
                converter = rapidshot.GpuConverter(
                    frame, (96, 64), dtype="uint8", layout="nhwc", crop=motion
                )
            images.append(converter.process(frame).numpy().copy())
    changed = sum(not np.array_equal(a, b) for a, b in zip(images, images[1:]))
    assert changed >= 3
