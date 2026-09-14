"""The first GPU read of a new frame must be that frame — in the paths that
shipped before 2.6.

**The bug** (found 2026-09-14 while testing ``TensorStream``).
``AcquireNextFrame`` returns once the capture device's copy into the
duplication surface is *submitted*, not once it has *run*. A D3D12 queue
reading the surface could overtake it and read the previous frame. Measured
against a moving source, first reads stale out of 150:

* ``GpuPreprocessor12``: 7–15
* ``CrossAdapterTransfer`` (Intel iGPU -> WARP): 65

Both now order their reads behind the capture work with a fence shared with
D3D11 (``native/src/capture_order.rs``); measured after, 0 / 150 each.
``test_gpu_converter_capture_order.py`` covers ``GpuConverter`` the same way.

**Why the existing verification missed it.** ``transfer_with_reference``
compares the destination with a source-side copy made from *the same
snapshot*, so a stale read is stale on both sides and the comparison passes.
The earlier ~2000 differing bytes that motivated that snapshot were this race,
misread as a surface DXGI keeps writing to. So these tests do not compare two
copies of one read; they compare a first read with a **settled re-read** of the
same held frame, which is stable (checked by the control assertion).

**Detection is statistical.** Unfixed rates were 5–10% for the preprocessor
(150 frames: miss probability under 0.1%) and 43% for the transfer (12 frames:
~0.1%; see ``TRANSFER_FRAMES`` for why fewer).
"""

import time

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

FRAMES = 150
SETTLE_S = 0.02


@pytest.fixture(scope="module")
def camera(motion):
    cam = rapidshot.create(output_color="BGRA")
    yield cam
    cam.release()


def first_vs_settled(camera, build, read, frames=FRAMES):
    """(stale first reads, settled reads that still differed) over `frames`."""
    obj = None
    stale = unstable = checked = 0
    deadline = time.monotonic() + 90
    while checked < frames and time.monotonic() < deadline:
        frame = camera.grab_frame()
        if frame is None:
            continue
        with frame:
            if obj is None:
                # Not checked: construction is slow enough to let this frame's
                # capture copy finish, which would hide the race.
                obj = build(frame)
                continue
            first = read(obj, frame)
            time.sleep(SETTLE_S)
            second = read(obj, frame)
            time.sleep(SETTLE_S)
            third = read(obj, frame)
        stale += not np.array_equal(first, second)
        unstable += not np.array_equal(second, third)
        checked += 1
    if checked < frames:
        pytest.skip(f"only {checked} changed frames arrived in 90 s")
    return stale, unstable


def test_preprocessor12_first_read_is_current(camera):
    def read(pre, frame):
        pre.process(frame, 1.0, 0.0, False)
        return np.asarray(pre.read_back()).copy()

    stale, unstable = first_vs_settled(
        camera, lambda f: native.GpuPreprocessor12(f, 96, 64), read
    )
    assert unstable == 0, "a held frame changed between settled reads; the method is unsound"
    assert stale == 0


# Fewer than FRAMES because each destination readback through WARP costs ~0.9 s
# (an 8 MB software copy; fast on real hardware). At the unfixed 43% rate, 12
# frames miss the bug with probability ~0.1%. The rate on a hardware
# destination has not been measured and may be lower.
TRANSFER_FRAMES = 12


def _pixels(raw, transfer, frame):
    """Pixel bytes only: rows are row_pitch apart, and the padding is not data."""
    raw = np.frombuffer(raw, dtype=np.uint8)
    pitch, w, h = transfer.row_pitch, frame.width, frame.height
    return raw[: pitch * h].reshape(h, pitch)[:, : w * 4].copy()


def stale_transfers(camera, copy):
    """First production copy, read on the destination, against a *settled*
    source-side reference of the same held frame.

    The reference is `transfer_with_reference` after a delay: ordered, settled,
    and read on the fast source adapter, so only the copy under test pays the
    WARP readback.
    """
    transfer = None
    stale = checked = 0
    deadline = time.monotonic() + 90
    while checked < TRANSFER_FRAMES and time.monotonic() < deadline:
        frame = camera.grab_frame()
        if frame is None:
            continue
        with frame:
            if transfer is None:
                try:
                    transfer = native.CrossAdapterTransfer(frame)
                except Exception as exc:
                    pytest.skip(
                        f"cross-adapter transfer unavailable: {str(exc).splitlines()[0]}"
                    )
                continue
            copy(transfer, frame)
            first = _pixels(transfer.read_back_destination(), transfer, frame)
            time.sleep(SETTLE_S)
            settled = _pixels(transfer.transfer_with_reference(frame), transfer, frame)
        stale += not np.array_equal(first, settled)
        checked += 1
    if checked < TRANSFER_FRAMES:
        pytest.skip(f"only {checked} changed frames arrived in 90 s")
    return stale


def test_cross_adapter_transfer_first_copy_is_current(camera):
    assert stale_transfers(camera, lambda t, f: t.transfer(f)) == 0


def test_cross_adapter_async_transfer_first_copy_is_current(camera):
    """The async path orders on the GPU too, so it stays non-blocking and must
    still copy the current frame."""
    def copy(transfer, frame):
        transfer.wait_shared_fence(transfer.transfer_async(frame))

    assert stale_transfers(camera, copy) == 0
