"""`TensorStream`'s control flow, against a fake camera and a fake converter.

No GPU, no desktop, so these run in CI. They cover the four hazards the stream
exists to close (see ``rapidshot/tensor_stream.py``): a frame held across
iterations, a buffer overwritten under unfinished CUDA work, a permanent
capture failure that looks like an idle screen, and a capture rebuilt onto a
new duplicator. What they cannot cover — that the tensors are right — is in
``test_tensor_stream.py``, against live capture.
"""

import pytest

import rapidshot.tensor_stream as ts_module
from rapidshot.tensor_stream import TensorStream


class FakeFrame:
    def __init__(self, source_id=1, sequence=0):
        self.source_id = source_id
        self.sequence = sequence
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.released = True
        return False


class FakeCamera:
    """Hands out scripted frames; ``None`` entries are 'nothing changed'."""

    def __init__(self, script):
        self.script = list(script)
        self.released = False
        self._capture_permanently_failed = False
        self._last_capture_error_message = ""
        self.grabs = 0

    def grab_frame(self):
        self.grabs += 1
        if not self.script:
            return None
        item = self.script.pop(0)
        if callable(item):
            return item(self)
        return item


class FakeTensor:
    def __init__(self, log):
        self.log = log

    def sync(self):
        self.log.append("sync")


class FakeConverter:
    instances = []
    fail_next = []          # exceptions to raise from upcoming process() calls

    def __init__(self, frame, size, **options):
        self.frame_source = frame.source_id
        self.size = size
        self.options = options
        self.log = []
        self._tensor = FakeTensor(self.log)
        FakeConverter.instances.append(self)

    def process(self, frame, regions=None):
        assert not frame.released, "converted a frame that was already released"
        self.log.append(("process", frame.sequence, regions))
        if FakeConverter.fail_next:
            raise FakeConverter.fail_next.pop(0)
        return self._tensor


@pytest.fixture(autouse=True)
def fake_converter(monkeypatch):
    FakeConverter.instances = []
    FakeConverter.fail_next = []
    monkeypatch.setattr(ts_module._converter, "GpuConverter", FakeConverter)
    yield FakeConverter


def frames(n, source_id=1):
    return [FakeFrame(source_id, i) for i in range(n)]


# --------------------------------------------------------------------------
# hazard 1: frames go back before the caller sees the tensor
# --------------------------------------------------------------------------


def test_each_frame_is_released_before_its_tensor_is_yielded():
    script = frames(3)
    stream = TensorStream(FakeCamera(script), (64, 64))
    for index, _ in zip(range(3), stream):
        assert script[index].released
        assert stream.frame is script[index]
    assert stream.frames == 3


def test_a_frame_is_released_even_when_conversion_raises():
    frame = FakeFrame()
    FakeConverter.fail_next = [ValueError("bad region")]
    stream = TensorStream(FakeCamera([frame]), (64, 64))
    with pytest.raises(ValueError):
        next(stream)
    assert frame.released


# --------------------------------------------------------------------------
# hazard 2: sync before overwriting, not after filling
# --------------------------------------------------------------------------


def test_sync_runs_before_each_conversion_after_the_first():
    stream = TensorStream(FakeCamera(frames(3)), (64, 64))
    for _ in zip(range(3), stream):
        pass
    log = FakeConverter.instances[0].log
    assert [entry if entry == "sync" else entry[0] for entry in log] == [
        "process", "sync", "process", "sync", "process",
    ]


def test_sync_can_be_turned_off():
    stream = TensorStream(FakeCamera(frames(3)), (64, 64), sync=False)
    for _ in zip(range(3), stream):
        pass
    assert "sync" not in FakeConverter.instances[0].log


# --------------------------------------------------------------------------
# hazard 3: None means two different things
# --------------------------------------------------------------------------


def test_idle_grabs_are_skipped_not_yielded():
    frame = FakeFrame()
    camera = FakeCamera([None, None, None, frame])
    stream = TensorStream(camera, (64, 64))
    next(stream)
    assert stream.frame is frame
    assert camera.grabs == 4


def test_a_permanent_failure_raises_instead_of_waiting_forever():
    def fail(camera):
        camera._capture_permanently_failed = True
        camera._last_capture_error_message = "Max re-initialization attempts (5) reached."
        return None

    stream = TensorStream(FakeCamera([None, fail]), (64, 64))
    with pytest.raises(RuntimeError, match="permanently failed.*Max re-initialization"):
        next(stream)


def test_a_released_camera_raises():
    camera = FakeCamera(frames(1))
    camera.released = True
    with pytest.raises(RuntimeError, match="released"):
        next(TensorStream(camera, (64, 64)))


def test_timeout_raises_on_an_idle_screen():
    stream = TensorStream(FakeCamera([]), (64, 64), timeout=0.05)
    with pytest.raises(TimeoutError, match="changed frame"):
        next(stream)


@pytest.mark.parametrize("timeout", [0, -1])
def test_non_positive_timeout_is_refused(timeout):
    with pytest.raises(ValueError, match="timeout"):
        TensorStream(FakeCamera([]), (64, 64), timeout=timeout)


# --------------------------------------------------------------------------
# hazard 4: rebuild for a new duplicator, and only then
# --------------------------------------------------------------------------


def test_converter_is_built_lazily_from_the_first_frame():
    stream = TensorStream(FakeCamera(frames(1, source_id=7)), (64, 48), dtype="float16")
    assert stream.converter is None
    next(stream)
    assert stream.converter.frame_source == 7
    assert stream.converter.size == (64, 48)
    assert stream.converter.options == {"dtype": "float16"}


def test_a_failure_on_a_new_duplicator_rebuilds_once_and_retries():
    camera = FakeCamera([FakeFrame(1, 0), FakeFrame(2, 1)])
    stream = TensorStream(camera, (64, 64))
    next(stream)
    FakeConverter.fail_next = [RuntimeError("OpenSharedHandle failed")]
    next(stream)
    assert len(FakeConverter.instances) == 2
    assert FakeConverter.instances[1].frame_source == 2
    assert stream.rebuilds == 1


def test_a_failure_on_the_same_duplicator_is_raised_not_hidden():
    stream = TensorStream(FakeCamera(frames(2)), (64, 64))
    next(stream)
    FakeConverter.fail_next = [RuntimeError("D3D12 dispatch failed")]
    with pytest.raises(RuntimeError, match="dispatch failed"):
        next(stream)
    assert len(FakeConverter.instances) == 1
    assert stream.rebuilds == 0


def test_a_rebuild_that_also_fails_is_raised():
    stream = TensorStream(FakeCamera([FakeFrame(1, 0), FakeFrame(2, 1)]), (64, 64))
    next(stream)
    FakeConverter.fail_next = [RuntimeError("first"), RuntimeError("second")]
    with pytest.raises(RuntimeError, match="second"):
        next(stream)


# --------------------------------------------------------------------------
# regions, lifetime
# --------------------------------------------------------------------------


def test_regions_set_batch_and_can_change_between_iterations():
    a, b, c = (0, 0, 10, 10), (5, 5, 20, 20), (1, 1, 2, 2)
    stream = TensorStream(FakeCamera(frames(2)), (32, 32), regions=[a, b])
    next(stream)
    assert stream.converter.options["batch"] == 2
    stream.regions = [c]
    next(stream)
    processed = [entry for entry in stream.converter.log if entry != "sync"]
    assert processed[0][2] == [a, b]
    assert processed[1][2] == [c]


def test_explicit_batch_wins_over_region_count():
    stream = TensorStream(FakeCamera(frames(1)), (32, 32), regions=[(0, 0, 4, 4)], batch=8)
    next(stream)
    assert stream.converter.options["batch"] == 8


def test_empty_regions_is_refused():
    with pytest.raises(ValueError, match="empty"):
        TensorStream(FakeCamera([]), (32, 32), regions=[])


def test_close_stops_iteration_and_leaves_the_camera_open():
    camera = FakeCamera(frames(5))
    with TensorStream(camera, (32, 32)) as stream:
        next(stream)
    assert stream.closed
    assert not camera.released
    with pytest.raises(StopIteration):
        next(stream)


def test_exported_from_the_package():
    import rapidshot

    assert rapidshot.TensorStream is TensorStream
