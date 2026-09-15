"""ScreenCapture end to end, over a fake D3D11/DXGI pipeline.

Line coverage showed 335 lines of capture.py that no test ran: every failure
branch of grab() and grab_frame(), all of shot(), the re-initialization backoff,
and the capture thread's timer, video-mode and failure handling. The live tests
cannot reach them -- a healthy desktop never loses access on demand -- and the
unit tests exercise the pieces (region mapping, processors, the duplicator) one
at a time.

Pieces tested one at a time is how grab() came to fail on every display rotated
90 or 270 degrees. Region mapping was right, and the processors were right when
handed a buffer of the right shape; nothing checked that capture handed them
one. So the fake here is a pipeline rather than a set of stubs: the "desktop" is
a real image in the panel's orientation, CopySubresourceRegion really copies the
requested box out of it, and the real ScreenCapture, Processor and memory pools
run on top.
"""
import ctypes
import threading
import time as real_time

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.capture as capture_module  # noqa: E402
from rapidshot.capture import ScreenCapture  # noqa: E402
from rapidshot.memory_pool import PoolExhaustedError  # noqa: E402
from rapidshot.processor.numpy_processor import NumpyProcessor  # noqa: E402
from rapidshot.util.errors import (  # noqa: E402
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotError,
    RapidShotProtectedContentError,
    RapidShotReinitError,
)

_real_sleep = real_time.sleep


def wait_until(predicate, timeout=5.0):
    """Poll without time.sleep, which the fixtures replace with a no-op."""
    deadline = real_time.monotonic() + timeout
    tick = threading.Event()
    while real_time.monotonic() < deadline:
        if predicate():
            return True
        tick.wait(0.002)
    return predicate()


# --------------------------------------------------------------------------
# the fake pipeline
# --------------------------------------------------------------------------

class FakeOutput:
    def __init__(self, width, height, rotation):
        self.devicename = r"\\.\FAKE1"
        self._native = (width, height)       # the panel's own orientation
        self.rotation_angle = rotation
        self.update_desc_error = None

    @property
    def surface_size(self):
        return self._native

    @property
    def resolution(self):
        width, height = self._native
        return (height, width) if self.rotation_angle in (90, 270) else (width, height)

    def update_desc(self):
        if self.update_desc_error is not None:
            raise self.update_desc_error


class FakeDesktop:
    """The duplicated texture: BGRA, panel orientation.

    `as_desktop` is what a correct capture returns: turned clockwise, which is
    what DXGI_MODE_ROTATION_ROTATE90 means (Microsoft's Desktop Duplication
    sample; test_cupy_rotation.desktop_from_texture builds it pixel by pixel).
    """

    def __init__(self, output, seed=0):
        self.output = output
        self.refresh(seed)

    def refresh(self, seed):
        width, height = self.output.surface_size
        rng = np.random.default_rng(seed)
        self.image = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)

    def as_desktop(self, color="RGB"):
        """What a correct capture of the whole desktop returns."""
        k = self.output.rotation_angle // 90
        image = np.ascontiguousarray(np.rot90(self.image, k=-k))
        return {"RGB": image[..., [2, 1, 0]], "BGRA": image}[color]


class FakeContext:
    def __init__(self, desktop):
        self.desktop = desktop
        self.log = []

    def CopySubresourceRegion(self, dst, dst_sub, x, y, z, src, src_sub, box_ref):
        box = box_ref._obj
        self.log.append(("copy", (box.left, box.top, box.right, box.bottom)))
        region = self.desktop.image[box.top:box.bottom, box.left:box.right]
        dst.buffer[:region.shape[0], :region.shape[1]] = region


class FakeDevice:
    def __init__(self, desktop):
        self.im_context = FakeContext(desktop)


class FakeMappedRect(ctypes.Structure):
    _fields_ = [("Pitch", ctypes.c_int), ("pBits", ctypes.c_void_p)]


class FakeStageSurface:
    null_bits = False
    rebuild_error = None
    release_error = None

    def __init__(self, output=None, device=None):
        self.log = getattr(device, "im_context", None)
        self.rebuild(output, device)

    def rebuild(self, output, device, dim=None):
        if FakeStageSurface.rebuild_error is not None:
            raise FakeStageSurface.rebuild_error
        self.width, self.height = dim if dim is not None else output.surface_size
        self.buffer = np.zeros((self.height, self.width, 4), dtype=np.uint8)
        self.texture = self
        self.mapped = False

    def release(self):
        if FakeStageSurface.release_error is not None:
            raise FakeStageSurface.release_error
        self.width = self.height = 0

    def map(self):
        if self.log is not None:
            self.log.log.append(("map", None))
        self.mapped = True
        rect = FakeMappedRect()
        rect.Pitch = self.width * 4
        rect.pBits = None if FakeStageSurface.null_bits else self.buffer.ctypes.data
        return rect

    def unmap(self):
        self.mapped = False


class FakeDuplicator:
    """update_frame() follows `script`, one step per call, repeating the last.

    The position is shared by every duplicator built, so a script reads as the
    camera's history: an access loss followed by frames means the *rebuilt*
    duplicator delivers the frames, rather than replaying the loss.

    Steps: "frame" (new content), "idle" (acquired, nothing new), "timeout"
    (not acquired), or an exception instance to raise.
    """

    script = ["frame"]
    position = 0
    built = []
    construct_errors = []

    def __init__(self, output=None, device=None, timeout_ms=10):
        if FakeDuplicator.construct_errors:
            raise FakeDuplicator.construct_errors.pop(0)
        self.device = device
        self.timeout_ms = timeout_ms
        self.updated = False
        self._frame_acquired = False
        self.texture = None
        self.dirty_rects = None
        # `[]` -- "this frame carried no move metadata" -- is what a real
        # duplicator reports for an ordinary frame; it reassigns this per
        # acquire. `None` means "could not be read", which forces a full
        # convert, and leaving it here made the fake model a display whose
        # move metadata failed on every single frame.
        self.move_rects = []
        self.rects_coalesced = False
        self.last_present_time = 0
        self.accumulated_frames = 0
        self.instance_id = len(FakeDuplicator.built) + 1
        self.protected_content_detected = False
        self.cursor_visible = False
        self.cursor = None
        self.released = False
        self.calls = 0
        self.frame_serial = 0         # frames with new content, as Duplicator counts them
        self.numbers_frames = True
        FakeDuplicator.built.append(self)

    def update_frame(self):
        script = FakeDuplicator.script
        step = script[min(FakeDuplicator.position, len(script) - 1)]
        FakeDuplicator.position += 1
        self.calls += 1
        self.updated = False
        if isinstance(step, BaseException):
            raise step
        if step == "unhealthy":
            return False
        if step == "timeout":
            return True
        self._frame_acquired = True
        if step == "frame":
            self.updated = True
            self.texture = "desktop"
            if self.numbers_frames:
                self.frame_serial += 1
        return True

    def release_frame(self):
        log = getattr(self.device, "im_context", None)
        if log is not None:
            log.log.append(("release_frame", None))
        self._frame_acquired = False
        self.texture = None

    def release(self):
        self.released = True


class FakeTimers:
    def __init__(self):
        self.created = self.cancelled = self.closed = 0
        self.wait_result = 0
        self.cancel_error = None
        self.close_error = None

    def install(self, monkeypatch):
        monkeypatch.setattr(capture_module, "create_high_resolution_timer", self.create)
        monkeypatch.setattr(capture_module, "set_periodic_timer", lambda h, p: None)
        monkeypatch.setattr(capture_module, "wait_for_timer", self.wait)
        monkeypatch.setattr(capture_module, "cancel_timer", self.cancel)
        monkeypatch.setattr(capture_module, "close_timer", self.close)

    def create(self):
        self.created += 1
        return 0xABC

    def wait(self, handle, timeout):
        _real_sleep(0.001)
        return self.wait_result

    def cancel(self, handle):
        self.cancelled += 1
        if self.cancel_error is not None:
            raise self.cancel_error

    def close(self, handle):
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def pipeline(monkeypatch):
    """Returns make(**kwargs) -> (camera, output, desktop, device)."""
    FakeDuplicator.script = ["frame"]
    FakeDuplicator.position = 0
    FakeDuplicator.built = []
    FakeDuplicator.construct_errors = []
    FakeStageSurface.null_bits = False
    FakeStageSurface.rebuild_error = None
    FakeStageSurface.release_error = None
    monkeypatch.setattr(capture_module, "Duplicator", FakeDuplicator)
    monkeypatch.setattr(capture_module, "StageSurface", FakeStageSurface)
    sleeps = []
    monkeypatch.setattr(capture_module.time, "sleep", sleeps.append)
    timers = FakeTimers()
    timers.install(monkeypatch)
    cameras = []

    def make(*, width=64, height=48, rotation=0, script=("frame",), **kwargs):
        FakeDuplicator.script = list(script)
        output = FakeOutput(width, height, rotation)
        desktop = FakeDesktop(output)
        device = FakeDevice(desktop)
        kwargs.setdefault("output_color", "RGB")
        cam = ScreenCapture(output=output, device=device, **kwargs)
        cameras.append(cam)
        return cam, output, desktop, device

    make.sleeps = sleeps
    make.timers = timers
    yield make
    for cam in cameras:
        cam.release()


def pool_idle(pool):
    stats = pool.get_stats()
    return stats["in_use"] == 0


# --------------------------------------------------------------------------
# rotated displays
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pool_output", [False, True])
@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_grab_on_a_rotated_display_returns_a_frame(pipeline, rotation, pool_output):
    """Every 90/270 grab used to fail: the staging buffer was sized from the
    desktop region, the staging surface holds the panel's orientation, and the
    processor refuses a buffer whose shape does not match."""
    cam, _, desktop, _ = pipeline(width=64, height=48, rotation=rotation,
                                  pool_output=pool_output, output_color="BGRA")

    frame = cam.grab()

    assert frame is not None
    assert np.asarray(frame).shape == (cam.height, cam.width, 4)
    np.testing.assert_array_equal(np.asarray(frame), desktop.as_desktop("BGRA"))
    if hasattr(frame, "release"):
        frame.release()


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_a_rotated_region_matches_the_same_region_of_a_full_grab(pipeline, rotation):
    """A property that holds whichever way rotation goes.

    The region is mapped into the texture by one convention and the pixels
    are turned by the processor's. They used to disagree -- region mapping
    clockwise, the processors counter-clockwise -- so a region grab read the
    wrong part of the screen while a full-screen grab merely looked upside
    down.
    """
    cam, _, _, _ = pipeline(width=64, height=48, rotation=rotation, pool_output=False)
    full = cam.grab()
    left, top, right, bottom = region = (4, 6, 30, 40)

    part = cam.grab(region=region)

    np.testing.assert_array_equal(part, full[top:bottom, left:right])


@pytest.mark.parametrize("rotation", [90, 270])
def test_the_staging_pool_is_sized_for_the_panel(pipeline, rotation):
    cam, _, _, _ = pipeline(width=64, height=48, rotation=rotation)

    assert tuple(cam.memory_pool.buffer_shape) == (48, 64, 4)
    assert (cam.width, cam.height) == (48, 64)


# --------------------------------------------------------------------------
# a processor failure inside grab()
# --------------------------------------------------------------------------

def test_process_raises_instead_of_returning_a_blank_buffer():
    """Returning (zeroed staging buffer, False) told grab() it was a fresh array:
    an RGB caller got a black 4-channel frame aliasing a recycled buffer."""
    class NullRect:
        Pitch = 16
        pBits = ctypes.c_void_p(None)

    proc = NumpyProcessor("RGB")
    proc._accum_valid = True
    staging = np.full((4, 4, 4), 200, dtype=np.uint8)

    with pytest.raises(ValueError, match="pBits"):
        proc.process(NullRect(), 4, 4, (0, 0, 4, 4), 0, staging)

    assert (staging == 200).all(), "the buffer must be left alone, not zeroed"
    assert proc._accum_valid is False, "a failed frame is not a base to patch onto"


def test_grab_returns_both_buffers_when_the_processor_fails(pipeline):
    cam, _, _, _ = pipeline(pool_output=True)
    FakeStageSurface.null_bits = True

    assert cam.grab() is None

    assert pool_idle(cam.memory_pool), "staging buffer leaked"
    assert cam._output_pool is not None and pool_idle(cam._output_pool), (
        "the converted-output buffer was checked out before process() raised "
        "and never returned")
    assert cam._needs_reinit is True
    assert cam.last_recovery_reason == "unhandled error during grab"
    assert "Unexpected error in _grab" in cam._last_capture_error_message


def test_repeated_processor_failures_do_not_exhaust_the_output_pool(pipeline):
    cam, _, _, _ = pipeline(pool_output=True, pool_size_frames=2)
    FakeStageSurface.null_bits = True

    for _ in range(5):
        cam._needs_reinit = False
        assert cam.grab() is None

    FakeStageSurface.null_bits = False
    cam._needs_reinit = False
    frame = cam.grab()
    assert frame is not None and hasattr(frame, "release"), (
        "the output pool should still hand out buffers")
    frame.release()


# --------------------------------------------------------------------------
# grab() failure branches
# --------------------------------------------------------------------------

@pytest.mark.parametrize("error,reason", [
    (RapidShotReinitError("access lost"), "DXGI re-init error during update_frame"),
    (RapidShotDeviceError("removed"), "DXGI device error during update_frame"),
])
def test_grab_schedules_recovery_for_lost_duplication(pipeline, error, reason):
    cam, _, _, _ = pipeline(script=[error])

    assert cam.grab() is None

    assert cam._needs_reinit is True
    assert cam.last_recovery_reason == reason
    assert pool_idle(cam.memory_pool)


@pytest.mark.parametrize("error", [
    RapidShotProtectedContentError("HDCP"),
    RapidShotDXGIError("odd"),
    RapidShotError("other"),
])
def test_grab_does_not_rebuild_for_errors_a_rebuild_cannot_fix(pipeline, error):
    cam, _, _, _ = pipeline(script=[error])

    assert cam.grab() is None

    assert cam._needs_reinit is False
    assert pool_idle(cam.memory_pool)


def test_protected_content_is_recorded_for_the_caller(pipeline):
    cam, _, _, _ = pipeline(script=[RapidShotProtectedContentError("HDCP on screen")])

    cam.grab()

    assert cam._last_capture_error_message == "HDCP on screen"


def test_an_error_with_a_frame_held_still_releases_it(pipeline):
    class AcquiredThenFails(RapidShotReinitError):
        pass

    cam, _, _, _ = pipeline()
    duplicator = cam._duplicator

    def update_frame():
        duplicator._frame_acquired = True
        raise AcquiredThenFails("lost mid-frame")

    duplicator.update_frame = update_frame
    cam.grab()

    assert duplicator._frame_acquired is False


def test_an_idle_grab_releases_the_acquired_frame(pipeline):
    cam, _, _, _ = pipeline(script=["idle"])

    assert cam.grab() is None

    assert cam._duplicator._frame_acquired is False
    assert pool_idle(cam.memory_pool)


def test_a_permanently_failed_camera_does_not_touch_the_duplicator(pipeline):
    cam, _, _, _ = pipeline()
    cam._capture_permanently_failed = True

    assert cam.grab() is None
    assert cam._duplicator.calls == 0


def test_an_uninitialized_camera_asks_for_recovery(pipeline):
    cam, _, _, _ = pipeline()
    cam._is_initialized = False

    assert cam.grab() is None
    assert cam.last_recovery_reason == "capture resources not initialized"


def test_grab_does_not_proceed_when_recovery_fails(pipeline):
    cam, _, _, _ = pipeline()
    cam._note_recovery_needed("test")
    FakeDuplicator.construct_errors = [RapidShotDXGIError("still gone")]
    calls_before = cam._duplicator.calls

    assert cam.grab() is None
    assert cam._reinit_attempts == 1
    assert cam._duplicator is None or cam._duplicator.calls == calls_before


def test_a_region_off_the_pool_shape_resizes_the_stage_surface(pipeline):
    cam, _, desktop, device = pipeline(pool_output=False)

    frame = cam.grab(region=(10, 5, 30, 25))

    assert (cam._stagesurf.width, cam._stagesurf.height) == (20, 20)
    assert device.im_context.log[0] == ("copy", (10, 5, 30, 25))
    np.testing.assert_array_equal(frame, desktop.as_desktop()[5:25, 10:30])


def test_the_frame_is_released_before_the_staging_surface_is_mapped(pipeline):
    """Mapping can stall; holding the desktop frame through it blocks DWM."""
    cam, _, _, device = pipeline(pool_output=False)

    cam.grab()

    events = [e for e, _ in device.im_context.log]
    assert events == ["copy", "release_frame", "map"]


def test_an_exhausted_staging_pool_returns_none(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=1)
    held = cam.grab()
    assert held is not None

    assert cam.grab() is None

    held.release()
    assert cam.grab() is not None


# --------------------------------------------------------------------------
# re-initialization
# --------------------------------------------------------------------------

def test_reinitialization_backs_off_then_gives_up(pipeline):
    cam, _, _, _ = pipeline()
    FakeDuplicator.construct_errors = [RapidShotDXGIError("gone")] * 10
    pipeline.sleeps.clear()

    results = [cam._attempt_reinitialization() for _ in range(6)]

    assert results == [False] * 6
    assert pipeline.sleeps == [0.5, 1.0, 2.0, 3.0, 5.0], (
        "one backoff per real attempt, and none once capture has given up")
    assert cam._capture_permanently_failed is True
    assert "5 attempts" in cam._last_capture_error_message


def test_a_successful_recovery_is_counted_and_resets_the_budget(pipeline):
    cam, _, _, _ = pipeline()
    cam._note_recovery_needed("access lost")
    FakeDuplicator.construct_errors = [RapidShotDXGIError("not yet")]

    assert cam._attempt_reinitialization() is False
    assert cam._attempt_reinitialization() is True

    assert cam._reinit_attempts == 0
    assert (cam.generation, cam.recovery_count) == (1, 1)
    assert cam.last_recovery_reason == "access lost"
    assert cam._needs_reinit is False


def test_recovery_is_skipped_once_capture_has_given_up(pipeline):
    cam, _, _, _ = pipeline()
    cam._capture_permanently_failed = True
    pipeline.sleeps.clear()

    assert cam._attempt_reinitialization() is False
    assert pipeline.sleeps == []


def test_recovery_fails_cleanly_when_the_output_is_gone(pipeline):
    cam, output, _, _ = pipeline()
    output.update_desc_error = OSError("monitor unplugged")

    assert cam._initialize_resources(is_reinit=True) is False
    assert cam._is_initialized is False


def test_grab_recovers_and_resumes(pipeline):
    cam, _, desktop, _ = pipeline(pool_output=False,
                                  script=[RapidShotReinitError("lost"), "frame"])

    assert cam.grab() is None
    frame = cam.grab()

    np.testing.assert_array_equal(frame, desktop.as_desktop())
    assert cam.recovery_count == 1


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_construction_raises_the_real_cause(pipeline):
    FakeDuplicator.construct_errors = [RapidShotDeviceError("adapter removed")]

    with pytest.raises(RapidShotDeviceError, match="adapter removed"):
        pipeline()


def test_gpu_request_without_cupy_falls_back_to_the_cpu(pipeline, monkeypatch):
    monkeypatch.setattr(capture_module, "cupy_available", lambda: False)

    cam, _, desktop, _ = pipeline(nvidia_gpu=True, pool_output=False)

    assert cam.nvidia_gpu is False
    np.testing.assert_array_equal(cam.grab(), desktop.as_desktop())


@pytest.mark.parametrize("kwargs", [
    {"timeout_ms": -1}, {"timeout_ms": 1.5}, {"timeout_ms": True},
    {"pool_size_frames": 0}, {"pool_size_frames": True},
])
def test_construction_rejects_bad_numbers_before_touching_dxgi(pipeline, kwargs):
    with pytest.raises(ValueError):
        pipeline(**kwargs)
    assert FakeDuplicator.built == []


# --------------------------------------------------------------------------
# grab_frame()
# --------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["failed", "uninitialized"])
def test_grab_frame_refuses_a_camera_that_cannot_capture(pipeline, state):
    cam, _, _, _ = pipeline()
    if state == "failed":
        cam._capture_permanently_failed = True
    else:
        cam._is_initialized = False

    assert cam.grab_frame() is None
    assert cam._needs_reinit is (state == "uninitialized")


@pytest.mark.parametrize("error,rebuild", [
    (RapidShotProtectedContentError("HDCP"), False),
    (RapidShotReinitError("lost"), True),
    (RapidShotDeviceError("removed"), True),
    (RapidShotDXGIError("odd"), False),
])
def test_grab_frame_failures(pipeline, error, rebuild):
    cam, _, _, _ = pipeline(script=[error])

    assert cam.grab_frame() is None
    assert cam._needs_reinit is rebuild


def test_grab_frame_stops_when_recovery_fails(pipeline):
    cam, _, _, _ = pipeline()
    cam._note_recovery_needed("test")
    FakeDuplicator.construct_errors = [RapidShotDXGIError("gone")]

    assert cam.grab_frame() is None


def test_grab_frame_with_an_explicit_region_records_it(pipeline):
    cam, _, _, _ = pipeline()

    with cam.grab_frame(region=(1, 2, 30, 40)) as frame:
        assert frame.region == (1, 2, 30, 40)


# --------------------------------------------------------------------------
# shot()
# --------------------------------------------------------------------------

def test_shot_writes_the_frame(pipeline):
    cam, _, desktop, _ = pipeline()
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is True

    np.testing.assert_array_equal(dst, desktop.as_desktop())
    assert cam._duplicator._frame_acquired is False


def test_shot_of_a_region_resizes_the_stage_surface_without_adopting_it(pipeline):
    cam, _, desktop, _ = pipeline()
    dst = np.zeros((10, 20, 3), dtype=np.uint8)

    assert cam.shot(dst, region=(5, 5, 25, 15)) is True

    np.testing.assert_array_equal(dst, desktop.as_desktop()[5:15, 5:25])
    assert cam.region == (0, 0, 64, 48)


def test_an_idle_shot_is_false_and_releases_the_frame(pipeline):
    cam, _, _, _ = pipeline(script=["idle"])
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is False

    assert cam._duplicator._frame_acquired is False
    assert len(FakeDuplicator.built) == 1, "an idle desktop is not an output change"


@pytest.mark.parametrize("step", [RapidShotReinitError("lost"), "unhealthy"])
def test_shot_rebuilds_a_dead_duplication(pipeline, step):
    cam, _, _, _ = pipeline(script=[step])
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is False

    assert len(FakeDuplicator.built) == 2
    assert FakeDuplicator.built[0].released is True
    assert cam._duplicator is FakeDuplicator.built[1]


def test_shot_releases_the_frame_when_the_copy_fails(pipeline):
    cam, _, _, device = pipeline()

    def broken_copy(*args):
        raise OSError("device hung")

    device.im_context.CopySubresourceRegion = broken_copy
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is False
    assert cam._duplicator._frame_acquired is False
    assert "device hung" in cam.last_capture_error
    assert cam.last_recovery_reason == "unhandled error during shot"


def test_shot_unmaps_when_conversion_fails(pipeline):
    cam, _, _, _ = pipeline()
    FakeStageSurface.null_bits = True
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is False
    assert cam._stagesurf.mapped is False


@pytest.mark.parametrize("error,rebuilds", [
    (RapidShotProtectedContentError("HDCP on screen"), False),
    (RapidShotDXGIError("odd"), False),
    (RapidShotError("wrapped"), False),
    (RapidShotReinitError("access lost"), True),
])
def test_shot_reports_capture_failures_the_way_grab_does(pipeline, error, rebuilds):
    """grab() returned None for these while shot() raised all but access loss,
    though shot()'s own contract is False on a failed capture."""
    cam, _, _, _ = pipeline(script=[error, "frame"])
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is False
    assert str(error) in cam.last_capture_error
    assert (len(FakeDuplicator.built) == 2) is rebuilds


def test_shot_on_a_camera_that_has_given_up_captures_nothing(pipeline):
    cam, _, _, _ = pipeline()
    cam._capture_permanently_failed = True

    assert cam.shot(np.zeros((48, 64, 3), dtype=np.uint8)) is False
    assert cam._duplicator.calls == 0


def test_shot_recovers_first_when_recovery_is_pending(pipeline):
    """It used to call straight into the duplicator a failed grab left behind."""
    cam, _, desktop, _ = pipeline()
    cam._note_recovery_needed("access lost")
    dst = np.zeros((48, 64, 3), dtype=np.uint8)

    assert cam.shot(dst) is True
    assert cam.recovery_count == 1
    np.testing.assert_array_equal(dst, desktop.as_desktop())


def test_shot_without_capture_resources_asks_for_recovery(pipeline):
    cam, _, _, _ = pipeline()
    cam._is_initialized = False

    assert cam.shot(np.zeros((48, 64, 3), dtype=np.uint8)) is False
    assert cam._needs_reinit is True


def test_shot_on_the_gpu_path_says_it_is_unsupported_before_capturing(pipeline):
    """The CuPy backend cannot write into caller memory. shot() found that out
    only after acquiring a frame, and then returned False and scheduled a
    rebuild on every call -- a rebuild that could never help."""
    cam, _, _, _ = pipeline()

    class GpuBackend:
        def process(self, *args, **kwargs):
            raise AssertionError("not reached")

    cam._processor.backend = GpuBackend()

    with pytest.raises(NotImplementedError, match="nvidia_gpu"):
        cam.shot(np.zeros((48, 64, 3), dtype=np.uint8))
    assert cam._duplicator.calls == 0
    assert cam._needs_reinit is False


def test_last_capture_error_starts_empty(pipeline):
    cam, _, _, _ = pipeline()
    assert cam.last_capture_error == ""


@pytest.mark.parametrize("bad", [0, -1, True, 2.5, "64"])
def test_max_buffer_len_must_be_a_positive_int(pipeline, bad):
    """0 used to build a zero-length queue whose first eviction raised
    IndexError in the capture thread, failing capture for good."""
    with pytest.raises(ValueError, match="max_buffer_len"):
        pipeline(max_buffer_len=bad)


def test_max_buffer_len_is_checked_again_at_start(pipeline):
    cam, _, _, _ = pipeline()
    cam.max_buffer_len = 0

    with pytest.raises(ValueError, match="max_buffer_len"):
        cam.start(target_fps=0)
    assert cam.is_capturing is False


def test_a_backend_that_cannot_take_a_target_gets_no_output_pool(pipeline, monkeypatch):
    """The CuPy backend allocates its own result. A host output pool was built
    and a buffer checked out and returned every frame for nothing."""
    cam, _, _, _ = pipeline(output_color="RGB", pool_output=True)
    monkeypatch.setattr(type(cam._processor.backend), "ACCEPTS_OUTPUT_TARGET", False)

    frame = cam.grab()

    assert isinstance(frame, np.ndarray) and not hasattr(frame, "release")
    assert cam._output_pool is None


def test_the_output_target_is_forwarded_only_to_a_backend_that_takes_one():
    from rapidshot.processor.base import Processor

    received = {}

    class Backend:
        def process(self, rect, width, height, region, rotation, buffer, **extra):
            received.update(extra)
            return "frame", False

        def invalidate_accumulator(self):
            pass

    proc = Processor(output_color="RGB")
    proc.backend = Backend()
    proc.process(None, 1, 1, (0, 0, 1, 1), 0, None, dirty_rects=[(0, 0, 1, 1)],
                 output_target="target")

    assert received == {"dirty_rects": [(0, 0, 1, 1)]}
    assert proc.accepts_output_target is False


def test_shot_needs_a_destination(pipeline):
    cam, _, _, _ = pipeline()
    with pytest.raises(ValueError, match="cannot be None"):
        cam.shot(None)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("color", ["RGB", "BGRA"])
def test_shot_matches_grab_on_a_rotated_display(pipeline, rotation, color):
    """shot() promises what grab() returns. It used to write the staging
    surface unturned: every pixel misplaced at 180, a transposed image at 90
    and 270."""
    cam, _, desktop, _ = pipeline(width=64, height=48, rotation=rotation,
                                  output_color=color, pool_output=False)
    dst = np.zeros((cam.height, cam.width, cam.channels), dtype=np.uint8)

    assert cam.shot(dst) is True

    np.testing.assert_array_equal(dst, desktop.as_desktop(color))


@pytest.mark.parametrize("rotation", [90, 270])
def test_rotated_shot_of_a_region_into_a_raw_pointer(pipeline, rotation):
    cam, _, desktop, _ = pipeline(width=64, height=48, rotation=rotation)
    left, top, right, bottom = region = (3, 5, 40, 29)
    dst = np.zeros((bottom - top, right - left, 3), dtype=np.uint8)

    assert cam.shot(dst.ctypes.data, region=region, buffer_size=dst.nbytes) is True

    np.testing.assert_array_equal(dst, desktop.as_desktop()[top:bottom, left:right])


# --------------------------------------------------------------------------
# the requested region survives rebuilds
# --------------------------------------------------------------------------

def test_a_start_region_survives_recovery(pipeline):
    """Continuous capture of a region used to come back from a device loss
    capturing the whole screen: the rebuild reset to the constructor's region."""
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    cam.start(region=(8, 8, 40, 30), target_fps=0)
    try:
        cam._note_recovery_needed("test")
        wait_until(lambda: cam.recovery_count >= 1)
    finally:
        cam.stop()

    assert cam.recovery_count >= 1
    assert cam.region == (8, 8, 40, 30)
    assert tuple(cam.memory_pool.buffer_shape) == (22, 32, 4)


def test_a_start_region_survives_an_output_change(pipeline):
    cam, _, _, _ = pipeline()
    cam.start(region=(8, 8, 40, 30), target_fps=0)
    cam.stop()

    assert cam._on_output_change() is True
    assert cam.region == (8, 8, 40, 30)


def test_a_create_region_survives_recovery(pipeline):
    cam, _, _, _ = pipeline(region=(4, 4, 20, 20))
    cam._note_recovery_needed("test")

    assert cam._attempt_reinitialization() is True
    assert cam.region == (4, 4, 20, 20)


def test_no_requested_region_follows_the_new_resolution(pipeline):
    cam, output, _, _ = pipeline()
    output._native = (80, 60)

    assert cam._on_output_change() is True
    assert cam.region == (0, 0, 80, 60)


def test_a_region_that_no_longer_fits_falls_back_and_comes_back(pipeline):
    """Used to raise ValueError out of the rebuild; and falling back must not
    forget the request, or a resolution that returns would not restore it."""
    cam, output, _, _ = pipeline(width=64, height=48)
    cam.start(region=(30, 20, 60, 44), target_fps=0)
    cam.stop()

    output._native = (40, 30)
    assert cam._on_output_change() is True
    assert cam.region == (0, 0, 40, 30)

    output._native = (64, 48)
    assert cam._on_output_change() is True
    assert cam.region == (30, 20, 60, 44)


def test_a_recovery_during_the_fallback_keeps_the_request(pipeline):
    """While a region does not fit, the camera shows the full screen. A recovery
    in that window must rebuild from the request, not from what is shown."""
    cam, output, _, _ = pipeline(width=64, height=48)
    cam.start(region=(30, 20, 60, 44), target_fps=0)
    cam.stop()
    output._native = (40, 30)
    cam._on_output_change()

    cam._note_recovery_needed("test")
    assert cam._attempt_reinitialization() is True
    output._native = (64, 48)
    cam._on_output_change()

    assert cam.region == (30, 20, 60, 44)


# --------------------------------------------------------------------------
# the dirty-rect accumulator follows the frame sequence
# --------------------------------------------------------------------------

def set_dirty(cam, rects):
    cam._duplicator.dirty_rects = rects


def count_patches(cam, monkeypatch):
    backend = cam._processor.backend
    calls = {"n": 0}
    real = backend._read_patch

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(backend, "_read_patch", counting)
    return calls


def small_change(desktop):
    """The next frame: the current image with only its top-left 4x4 changed."""
    image = desktop.image.copy()
    image[0:4, 0:4] = 7
    desktop.image = image


def test_consecutive_grabs_still_patch_only_the_dirty_region(pipeline, monkeypatch):
    """The fast path the fix must not lose: a frame that directly follows the
    accumulator is patched, not converted in full."""
    cam, _, desktop, _ = pipeline(pool_output=False)
    set_dirty(cam, [(0, 0, 64, 48)])
    cam.grab()
    patches = count_patches(cam, monkeypatch)

    small_change(desktop)
    set_dirty(cam, [(0, 0, 4, 4)])
    frame = cam.grab()

    assert patches["n"] == 1
    np.testing.assert_array_equal(frame, desktop.as_desktop())


@pytest.mark.parametrize("consumer", ["shot", "grab_frame"])
def test_a_frame_taken_by_another_call_is_not_patched_over(pipeline, monkeypatch, consumer):
    """The frame shot() or grab_frame() consumed carried the change since the
    accumulator's frame. Its dirty rects are gone, so the next grab's rects
    describe only the change since *that* frame: patching them onto the
    accumulator left 99.5% of the image a frame out of date."""
    cam, _, desktop, _ = pipeline(pool_output=False)
    set_dirty(cam, [(0, 0, 64, 48)])
    cam.grab()

    desktop.refresh(seed=1)
    if consumer == "shot":
        assert cam.shot(np.zeros((48, 64, 3), np.uint8))
    else:
        with cam.grab_frame():
            pass

    patches = count_patches(cam, monkeypatch)
    small_change(desktop)
    set_dirty(cam, [(0, 0, 4, 4)])
    frame = cam.grab()

    assert patches["n"] == 0, "converted in full, not patched"
    np.testing.assert_array_equal(frame, desktop.as_desktop())


def test_a_rebuilt_duplicator_is_not_a_continuation(pipeline, monkeypatch):
    """After an output change at the same size the accumulator still has the
    right shape, but the new duplicator's rects owe nothing to it.

    Frame numbers restart with the new duplicator, so they can line up with the
    old one: here the accumulator holds old frame 1 and the grab sees new
    frame 2. Only the duplicator's identity tells those apart.
    """
    cam, _, desktop, _ = pipeline(pool_output=False)
    set_dirty(cam, [(0, 0, 64, 48)])
    cam.grab()

    assert cam._on_output_change() is True
    desktop.refresh(seed=2)
    with cam.grab_frame():                 # new duplicator's frame 1
        pass
    assert cam._duplicator.frame_serial == 1
    patches = count_patches(cam, monkeypatch)
    small_change(desktop)
    set_dirty(cam, [(0, 0, 4, 4)])
    frame = cam.grab()                     # new duplicator's frame 2

    assert patches["n"] == 0
    np.testing.assert_array_equal(frame, desktop.as_desktop())


def test_a_grab_that_fails_after_acquiring_does_not_vouch_for_the_accumulator(pipeline, monkeypatch):
    """The frame was acquired, so its rects are spent, but the processor never
    saw it. The accumulator must not be recorded as holding it."""
    cam, _, desktop, _ = pipeline(pool_output=False)
    set_dirty(cam, [(0, 0, 64, 48)])
    cam.grab()

    desktop.refresh(seed=3)
    real_map = FakeStageSurface.map

    def failing_map(self):
        raise OSError("device hung")

    monkeypatch.setattr(FakeStageSurface, "map", failing_map)
    assert cam.grab() is None
    monkeypatch.setattr(FakeStageSurface, "map", real_map)
    cam._needs_reinit = False              # keep the same duplicator for the next grab

    patches = count_patches(cam, monkeypatch)
    small_change(desktop)
    set_dirty(cam, [(0, 0, 4, 4)])
    frame = cam.grab()

    assert patches["n"] == 0
    np.testing.assert_array_equal(frame, desktop.as_desktop())


def test_a_duplicator_that_does_not_number_frames_is_never_patched_onto(pipeline, monkeypatch):
    cam, _, desktop, _ = pipeline(pool_output=False)
    cam._duplicator.numbers_frames = False
    del cam._duplicator.frame_serial
    set_dirty(cam, [(0, 0, 64, 48)])
    assert cam.grab() is not None

    patches = count_patches(cam, monkeypatch)
    small_change(desktop)
    set_dirty(cam, [(0, 0, 4, 4)])
    frame = cam.grab()

    assert frame is not None, "otherwise 'no patch' would hold trivially"
    assert patches["n"] == 0
    np.testing.assert_array_equal(frame, desktop.as_desktop())


# --------------------------------------------------------------------------
# output change
# --------------------------------------------------------------------------

def test_an_output_change_while_capturing_resizes_the_frame_pool(pipeline):
    cam, output, _, _ = pipeline(output_color="BGRA")
    cam.is_capturing = True
    output._native = (80, 60)

    assert cam._on_output_change() is True

    assert (cam.width, cam.height) == (80, 60)
    assert tuple(cam.memory_pool.buffer_shape) == (60, 80, 4)
    cam.is_capturing = False


def test_cleanup_failures_after_a_bad_stage_surface_still_surface_the_cause(
        pipeline, monkeypatch):
    """A stage surface that cannot be built leaves a live duplicator behind.
    Releasing that partial pair must be attempted even when it fails, and the
    error that reaches the caller must be the build failure, not the cleanup's.
    """
    cam, _, _, _ = pipeline()
    original = cam._duplicator
    attempts = {"duplicator": 0, "stage": 0}

    def duplicator_release(self):
        if self is original:
            self.released = True
            return
        attempts["duplicator"] += 1
        raise OSError("duplicator will not release")

    def stage_release(self):
        attempts["stage"] += 1
        if attempts["stage"] > 1:           # the first is the ordinary teardown
            raise OSError("stage surface will not release")
        self.width = self.height = 0

    monkeypatch.setattr(FakeDuplicator, "release", duplicator_release)
    monkeypatch.setattr(FakeStageSurface, "release", stage_release)
    FakeStageSurface.rebuild_error = OSError("no staging texture")

    # OSError is not one of the retried types (COMError, RapidShotError), so
    # it propagates after the cleanup.
    with pytest.raises(OSError, match="no staging texture"):
        cam._on_output_change()

    assert attempts == {"duplicator": 1, "stage": 2}
    assert cam._duplicator is None, "the partial duplicator must not be published"


def test_frame_buffer_rebuild_keeps_a_pool_of_the_right_shape(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA")
    pool = cam.memory_pool

    cam._rebuild_frame_buffer(None)
    assert cam.memory_pool is pool

    cam._rebuild_frame_buffer((0, 0, 32, 24))
    assert cam.memory_pool is not pool
    assert tuple(cam.memory_pool.buffer_shape) == (24, 32, 4)


# --------------------------------------------------------------------------
# the capture thread
# --------------------------------------------------------------------------

def test_video_mode_repeats_the_last_frame_while_the_screen_is_idle(pipeline):
    cam, _, desktop, _ = pipeline(script=["frame", "idle"], pool_output=False,
                                  max_buffer_len=4)
    cam.start(target_fps=0, video_mode=True)
    try:
        assert wait_until(lambda: cam._frame_count >= 4)
    finally:
        cam.stop()

    assert FakeDuplicator.built[0].calls >= 4


def test_video_mode_duplicates_are_copies_not_aliases(pipeline):
    cam, _, desktop, _ = pipeline(script=["frame", "idle"], output_color="BGRA",
                                  pool_size_frames=4, max_buffer_len=2)
    cam.start(target_fps=0, video_mode=True)
    try:
        assert wait_until(lambda: cam._frame_count >= 3)
        first = cam.get_latest_frame_buffer()
        second = cam.get_latest_frame_buffer() or cam.get_latest_frame_buffer()
    finally:
        cam.stop()

    assert first is not None and second is not None
    assert np.asarray(first).ctypes.data != np.asarray(second).ctypes.data
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    for frame in (first, second):
        if hasattr(frame, "release"):
            frame.release()


def idle_video_tick(cam, source):
    """One capture-thread iteration with no new frame and `source` queued."""
    import collections

    cam._pooled_frames_deque = collections.deque([source], maxlen=4)
    cam._last_dup_source = source
    cam._frame_available_event.set()
    cam._stop_capture_event.clear()

    def idle(region):
        cam._stop_capture_event.set()
        return None

    cam._grab = idle
    cam._capture_thread_func(cam.region, target_fps=0, video_mode=True)


class OverlayConsumer:
    """Takes the newest frame mid-copy and draws on it -- its right once taken."""

    def __init__(self, cam):
        self.cam = cam
        self.taken = None

    def take_and_draw(self):
        self.taken = self.cam.get_latest_frame_buffer()
        np.asarray(self.taken)[:] = 222


def test_a_frame_taken_during_duplication_is_not_copied_into_the_queue(pipeline, monkeypatch):
    """video_mode copies the last frame outside the lock. If a consumer takes
    that frame meanwhile it is the consumer's, and whatever it draws on it must
    not come back out of the queue as a captured frame."""
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    source = cam.memory_pool.checkout()
    source.array[:] = 111
    consumer = OverlayConsumer(cam)
    real_copyto = np.copyto

    def copy_while_the_consumer_draws(dst, src, *args, **kwargs):
        consumer.take_and_draw()
        return real_copyto(dst, src, *args, **kwargs)

    monkeypatch.setattr(capture_module.np, "copyto", copy_while_the_consumer_draws)

    idle_video_tick(cam, source)

    assert consumer.taken is source
    assert len(cam._pooled_frames_deque) == 0, "the tainted duplicate was queued"
    assert cam.memory_pool.get_stats()["in_use"] == 1, "only the consumer's frame is out"
    assert cam._last_dup_source is None


def test_a_plain_array_taken_during_duplication_is_not_queued(pipeline):
    cam, _, _, _ = pipeline(output_color="RGB", pool_output=False)
    consumer = OverlayConsumer(cam)

    class Frame(np.ndarray):
        def copy(self, *args, **kwargs):
            consumer.take_and_draw()
            return np.ndarray.copy(self, *args, **kwargs)

    source = np.full((48, 64, 3), 111, np.uint8).view(Frame)

    idle_video_tick(cam, source)

    assert consumer.taken is source
    assert len(cam._pooled_frames_deque) == 0


def test_an_untouched_frame_is_still_duplicated(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    source = cam.memory_pool.checkout()
    source.array[:] = 111

    idle_video_tick(cam, source)

    duplicate = cam._pooled_frames_deque[-1]
    assert duplicate is not source and (duplicate.array == 111).all()
    assert cam._last_dup_source is duplicate


def test_the_gpu_copy_is_checked_only_after_it_has_run(pipeline, monkeypatch):
    """A CuPy assignment is queued, not done. A consumer taking the frame before
    the device reads it is as much a leak as one taking it mid-copy on the CPU."""
    import types
    from test_cupy_processor_paths import NumpyAsCupy

    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    source = cam.memory_pool.checkout()
    source.array[:] = 111
    consumer = OverlayConsumer(cam)
    cam.nvidia_gpu = True
    events = []

    class Stream:
        def synchronize(self):
            events.append("synchronize")
            consumer.take_and_draw()        # the queued copy lands after this

    fake = NumpyAsCupy()
    fake.cuda = types.SimpleNamespace(get_current_stream=lambda: Stream())
    monkeypatch.setattr(capture_module, "_require_cupy", lambda: fake)

    idle_video_tick(cam, source)

    assert events == ["synchronize"]
    assert consumer.taken is source
    assert len(cam._pooled_frames_deque) == 0


def test_the_thread_recovers_from_access_loss_and_keeps_delivering(pipeline):
    cam, _, desktop, _ = pipeline(script=[RapidShotReinitError("lost"), "frame"],
                                  pool_output=False)
    cam.start(target_fps=0)
    try:
        frame = cam.get_latest_frame()
    finally:
        cam.stop()

    np.testing.assert_array_equal(frame, desktop.as_desktop())
    assert cam.recovery_count == 1


def test_the_thread_stops_once_capture_has_given_up(pipeline):
    cam, _, _, _ = pipeline(script=["timeout"])
    cam.start(target_fps=0)
    cam._capture_permanently_failed = True

    assert wait_until(lambda: not cam._capture_thread.is_alive())
    assert cam.stop() is True


def test_an_unexpected_error_in_the_loop_fails_capture_loudly(pipeline, monkeypatch):
    cam, _, _, _ = pipeline()

    def exploding_grab(region):
        raise KeyError("bug in the loop")

    monkeypatch.setattr(cam, "_grab", exploding_grab)
    cam.start(target_fps=0)

    assert wait_until(lambda: not cam._capture_thread.is_alive())
    assert cam._capture_permanently_failed is True
    assert "bug in the loop" in cam._last_capture_error_message
    cam.stop()


def test_a_failed_timer_wait_stops_the_thread_and_closes_the_timer(pipeline):
    from rapidshot.util.timer import WAIT_FAILED

    cam, _, _, _ = pipeline(script=["timeout"])
    pipeline.timers.wait_result = WAIT_FAILED
    cam.start(target_fps=60)

    assert wait_until(lambda: not cam._capture_thread.is_alive())
    assert (pipeline.timers.created, pipeline.timers.cancelled,
            pipeline.timers.closed) == (1, 1, 1)
    assert cam._timer_handle is None
    cam.stop()


def test_timer_cleanup_errors_do_not_leak_the_handle(pipeline):
    cam, _, _, _ = pipeline(script=["timeout"])
    pipeline.timers.cancel_error = OSError("cancel failed")
    pipeline.timers.close_error = OSError("close failed")
    cam.start(target_fps=60)
    assert wait_until(lambda: pipeline.timers.created == 1)

    assert cam.stop() is True
    assert pipeline.timers.closed == 1, "close is attempted even when cancel fails"
    assert cam._timer_handle is None


def test_stop_closes_a_timer_the_thread_left_behind(pipeline):
    cam, _, _, _ = pipeline()
    cam.is_capturing = True
    cam._capture_thread = None
    cam._timer_handle = 0xDEF
    pipeline.timers.cancel_error = OSError("cancel failed")

    assert cam.stop() is True
    assert (pipeline.timers.cancelled, pipeline.timers.closed) == (1, 1)
    assert cam._timer_handle is None


def test_get_latest_frame_refuses_an_unknown_queued_type(pipeline):
    cam, _, _, _ = pipeline()
    cam._pooled_frames_deque = __import__("collections").deque([["not", "an", "array"]])
    cam._frame_available_event.set()

    assert cam.get_latest_frame() is None


# --------------------------------------------------------------------------
# release and small accessors
# --------------------------------------------------------------------------

def test_release_proceeds_when_a_capture_call_holds_the_lock(pipeline, monkeypatch):
    cam, _, _, _ = pipeline()

    class StuckLock:
        def acquire(self, timeout=None):
            return False

        def release(self):
            raise AssertionError("release() of a lock it never acquired")

    monkeypatch.setattr(cam, "_dup_lock", lambda: StuckLock())
    duplicator = cam._duplicator

    cam.release()

    assert duplicator.released is True
    assert cam.memory_pool is None


def test_release_frees_a_frame_still_held(pipeline):
    cam, _, _, _ = pipeline()
    frame = cam.grab_frame()

    cam.release()

    assert frame.released is True


def test_release_warns_but_finishes_when_the_thread_will_not_stop(pipeline, monkeypatch):
    cam, _, _, _ = pipeline()
    cam.is_capturing = True
    monkeypatch.setattr(cam, "stop", lambda: False)
    duplicator = cam._duplicator

    cam.release()

    assert duplicator.released is True


@pytest.mark.parametrize("region,message", [
    (None, "cannot be None"),
    (("a", 0, 1, 1), "four integers"),
    ((0, 0, 1), "four integers"),
])
def test_normalize_region_rejects_malformed_input(pipeline, region, message):
    cam, _, _, _ = pipeline()
    with pytest.raises(ValueError, match=message):
        cam._normalize_region(region)


def test_normalize_region_needs_dimensions():
    cam = ScreenCapture.__new__(ScreenCapture)
    with pytest.raises(ValueError, match="not initialized"):
        cam._normalize_region((0, 0, 1, 1))


def test_bytes_per_frame_for_an_explicit_region(pipeline):
    cam, _, _, _ = pipeline()
    assert cam.bytes_per_frame((0, 0, 10, 5)) == 150


def test_accessors_and_repr(pipeline):
    cam, _, _, _ = pipeline()
    assert cam.grab_cursor() is None
    assert "ScreenCapture" in repr(cam)

    broken = ScreenCapture.__new__(ScreenCapture)

    class Unprintable:
        def __str__(self):
            raise RuntimeError("no")

        __repr__ = __str__

    broken._device = Unprintable()
    assert repr(broken) == "<ScreenCapture: initialization incomplete>"


def test_destructor_swallows_release_errors():
    cam = ScreenCapture.__new__(ScreenCapture)

    def failing_release():
        raise RuntimeError("teardown failed")

    cam.release = failing_release
    cam.__del__()   # must not raise
    del cam.release


def test_module_keeps_the_old_cupy_names(monkeypatch):
    monkeypatch.setattr(capture_module, "cupy_available", lambda: False)
    assert capture_module.CUPY_AVAILABLE is False
    with pytest.raises(AttributeError):
        capture_module.no_such_name
