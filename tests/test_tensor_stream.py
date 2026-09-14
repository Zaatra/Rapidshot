"""`TensorStream` against live capture, with a controlled moving source.

**Why a motion source.** A stream needs a run of *changed* frames, and Desktop
Duplication reports only changes, so an idle desktop would make these tests a
coin flip. ``benchmarks/motion_source.py`` animates a window at a known place,
which is what the benchmark harness already relies on.

**How correctness is checked when the frame is gone.** The stream releases
each frame before yielding, so nothing can be recomputed from it afterwards.
Instead the camera is wrapped: the wrapper converts every frame with an
independent ``GpuConverter`` *while it is still live*, then hands the same
frame on to the stream. The stream's tensor must equal that reference exactly.

**This test found a converter bug, not a stream bug** (2026-09-14). The
reference — the first reader of each new frame — was sometimes the previous
frame: D3D12 overtook the capture device's unfinished copy into the surface.
Fixed in ``converter12.rs`` (``order_after_capture``) and pinned by
``test_gpu_converter_capture_order.py``.

Control flow — release ordering, sync, failure, rebuild — is covered without a
GPU in ``test_tensor_stream_logic.py``.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

from conftest import MOTION_INSIDE as INSIDE

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


@pytest.fixture(scope="module")
def camera(motion):
    cam = rapidshot.create(output_color="BGRA")
    yield cam
    cam.release()


class ReferenceCamera:
    """Delegates to a real camera, converting each live frame on the side."""

    def __init__(self, camera, reference_factory):
        self._camera = camera
        self._factory = reference_factory
        self._reference = None
        self.expected = []

    def __getattr__(self, name):
        return getattr(self._camera, name)

    def grab_frame(self):
        frame = self._camera.grab_frame()
        if frame is not None:
            if self._reference is None:
                self._reference = self._factory(frame)
            self.expected.append(self._reference(frame))
        return frame


def take(stream, n):
    """n tensors, copied out, with the frame behind each.

    ``range`` first: zip stops at its first exhausted argument, so with the
    stream first it would capture one frame more than it returns.
    """
    out = []
    for _, tensor in zip(range(n), stream):
        out.append((tensor.numpy().copy(), stream.frame))
    return out


# --------------------------------------------------------------------------
# the stream's tensors are the conversion of the frames it captured
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        dict(dtype="float16", layout="NCHW", sampling="bilinear"),
        dict(dtype="float32", layout="nhwc", sampling="nearest", bgr=True),
        dict(dtype="uint8", layout="nhwc", sampling="nearest"),
    ],
    ids=["fp16-nchw", "fp32-nhwc-bgr", "bgra8"],
)
def test_stream_tensors_equal_an_independent_conversion(camera, options):
    size = (96, 64)

    def factory(frame):
        converter = rapidshot.GpuConverter(frame, size, crop=INSIDE, **options)
        return lambda f: converter.process(f).numpy().copy()

    wrapped = ReferenceCamera(camera, factory)
    with rapidshot.TensorStream(wrapped, size, timeout=5, crop=INSIDE, **options) as stream:
        got = take(stream, 5)

    assert len(got) == len(wrapped.expected) == 5
    for (tensor, _), expected in zip(got, wrapped.expected):
        np.testing.assert_array_equal(tensor, expected)


def test_regions_stream_as_a_batch(camera):
    size = (64, 64)
    regions = [(240, 160, 560, 480), (700, 300, 1000, 700)]

    def factory(frame):
        converter = rapidshot.GpuConverter(frame, size, dtype="float16", batch=2)
        return lambda f: converter.process(f, regions=regions).numpy().copy()

    wrapped = ReferenceCamera(camera, factory)
    with rapidshot.TensorStream(
        wrapped, size, timeout=5, dtype="float16", regions=regions
    ) as stream:
        got = take(stream, 4)
        assert stream.converter.batch == 2

    for (tensor, _), expected in zip(got, wrapped.expected):
        assert tensor.shape == (2, 3, 64, 64)
        np.testing.assert_array_equal(tensor, expected)


# --------------------------------------------------------------------------
# it is a stream: new frames, one buffer, capture never stalls
# --------------------------------------------------------------------------


def test_consecutive_tensors_come_from_new_frames(camera):
    with rapidshot.TensorStream(
        camera, (128, 96), timeout=5, dtype="uint8", layout="nhwc", crop=INSIDE
    ) as stream:
        got = take(stream, 6)

    stamps = [frame.timestamp_qpc for _, frame in got]
    assert all(b > a for a, b in zip(stamps, stamps[1:])), stamps
    # The motion source changes every frame it draws; at least most
    # consecutive captures must differ, or the stream is repeating a buffer.
    changed = sum(not np.array_equal(a, b) for (a, _), (b, _) in zip(got, got[1:]))
    assert changed >= len(got) // 2


def test_yielded_frames_are_released_and_keep_their_metadata(camera):
    with rapidshot.TensorStream(camera, (64, 64), timeout=5) as stream:
        next(stream)
        frame = stream.frame
        assert frame.released
        assert frame.width > 0 and frame.timestamp_qpc > 0
        # Holding the tensor did not stall capture: grabbing continues.
        next(stream)
        assert stream.frame is not frame


def test_the_tensor_is_one_reused_handle(camera):
    with rapidshot.TensorStream(camera, (64, 64), timeout=5) as stream:
        first = next(stream)
        second = next(stream)
    assert first is second
    assert stream.frames == 2
    assert stream.rebuilds == 0


def test_closing_the_stream_leaves_the_camera_usable(camera):
    with rapidshot.TensorStream(camera, (32, 32), timeout=5) as stream:
        next(stream)
    assert not camera.released
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            frame.release()
            break
    else:
        pytest.fail("camera produced no frame after the stream closed")


def test_continuous_capture_is_refused_clearly(camera):
    """camera.start() owns the duplicator; the stream must say so, not hang."""
    camera.start(target_fps=30)
    try:
        with pytest.raises(RuntimeError, match="continuous capture"):
            next(rapidshot.TensorStream(camera, (32, 32), timeout=2))
    finally:
        camera.stop()
