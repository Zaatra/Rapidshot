"""The last testable branches in the group 1 modules.

What line coverage still showed after the per-module suites: defensive
branches and edge cases, each small, several of which turned out to matter --
a null shot() destination went straight to a memmove into address 0.

Not here, deliberately: import fallbacks for platforms and installs this suite
does not run on, and the one check (a null immediate context from a device that
was just created) that needs a real driver misbehaving to reach.
"""
import collections
import ctypes
import sys
import types

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot  # noqa: E402
import rapidshot.capture as capture_module  # noqa: E402
import rapidshot.dxcam_compat as dxcam  # noqa: E402
import rapidshot.native as native  # noqa: E402
from rapidshot.frame import Frame  # noqa: E402
from rapidshot.memory_pool import NumpyMemoryPool  # noqa: E402
from rapidshot.processor.numpy_processor import NumpyProcessor  # noqa: E402
from rapidshot.util.errors import (  # noqa: E402
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotError,
    RapidShotProtectedContentError,
)

from test_capture_paths import (  # noqa: E402,F401
    FakeDuplicator,
    FakeOutput,
    FakeStageSurface,
    pipeline,
)
from test_duplicator_paths import make_duplicator  # noqa: E402


class Mapped:
    def __init__(self, height=4, width=4):
        self._buf = (ctypes.c_ubyte * (width * 4 * height))()
        self.Pitch = width * 4
        self.pBits = ctypes.cast(self._buf, ctypes.c_void_p)


# --------------------------------------------------------------------------
# shot(): a null destination
# --------------------------------------------------------------------------

@pytest.mark.parametrize("rotation", [0, 90])
def test_shot_refuses_a_null_pointer_before_capturing(pipeline, rotation):
    """pointer_to_address(0) is 0, not None, and every check was `is None`, so
    this reached ctypes.memmove(0, ...) -- an access violation, not an
    exception. Refused now before any capture work."""
    cam, _, _, _ = pipeline(rotation=rotation)
    calls_before = FakeDuplicator.position

    with pytest.raises(ValueError, match="destination pointer is null"):
        cam.shot(0, buffer_size=10**6)

    assert FakeDuplicator.position == calls_before, "nothing was acquired"


def test_the_lower_copy_paths_refuse_null_too(pipeline):
    """Defence in depth: both copies below shot() check for 0 themselves."""
    with pytest.raises(ValueError, match="destination pointer"):
        NumpyProcessor("RGB").shot(0, object(), 4, 4, buffer_size=10**6)

    cam, _, _, _ = pipeline(rotation=90)
    with pytest.raises(ValueError, match="destination pointer"):
        cam._shot_rotated(0, Mapped(), 4, 4)


def test_process_refuses_a_rect_whose_pointer_cannot_be_read():
    class Rect:
        Pitch = 16
        pBits = object()          # truthy, but no address can be derived

    with pytest.raises(ValueError, match="valid pointer"):
        NumpyProcessor("RGB").process(Rect(), 4, 4, (0, 0, 4, 4), 0)


# --------------------------------------------------------------------------
# NumpyProcessor validation
# --------------------------------------------------------------------------

def test_process_refuses_a_region_outside_the_frame():
    with pytest.raises(ValueError, match="outside of the frame"):
        NumpyProcessor("RGB").process(Mapped(), 4, 4, (0, 0, 5, 4), 0)


def test_process_refuses_a_staging_buffer_of_the_wrong_shape():
    with pytest.raises(ValueError, match="does not match region shape"):
        NumpyProcessor("RGB").process(Mapped(), 4, 4, (0, 0, 4, 4), 0,
                                      np.zeros((4, 4, 3), np.uint8))


def test_an_accumulator_of_another_shape_is_not_patched_onto():
    proc = NumpyProcessor("RGB")
    proc._accum = np.zeros((2, 2, 3), np.uint8)
    proc._accum_valid = True

    assert proc._usable_dirty_rects([(0, 0, 1, 1)], 4, 4, 0) is None


def test_convert_into_refuses_a_mode_that_bypassed_validation():
    proc = NumpyProcessor("RGB")
    proc.color_mode = "YUV"

    with pytest.raises(ValueError, match="Unsupported color mode"):
        proc.convert_into(np.zeros((2, 2, 4), np.uint8), np.zeros((2, 2, 3), np.uint8))


# --------------------------------------------------------------------------
# grab(): every error releases a frame it acquired
# --------------------------------------------------------------------------

@pytest.mark.parametrize("error", [
    RapidShotDeviceError("removed"),
    RapidShotProtectedContentError("HDCP"),
    RapidShotDXGIError("odd"),
    RapidShotError("other"),
])
def test_every_acquire_error_releases_a_held_frame(pipeline, error):
    """DXGI refuses the next acquire while a frame is held, so an error branch
    that forgot the release would stall capture after one failure."""
    cam, _, _, _ = pipeline()
    duplicator = cam._duplicator

    def update_frame():
        duplicator._frame_acquired = True
        raise error

    duplicator.update_frame = update_frame

    assert cam.grab() is None
    assert duplicator._frame_acquired is False


def test_an_idle_grab_frame_releases_what_it_acquired(pipeline):
    cam, _, _, _ = pipeline(script=["idle"])

    assert cam.grab_frame() is None
    assert cam._duplicator._frame_acquired is False


def test_grab_releases_the_frame_when_the_copy_fails(pipeline):
    cam, _, _, device = pipeline()

    def broken_copy(*args):
        raise OSError("device hung")

    device.im_context.CopySubresourceRegion = broken_copy

    assert cam.grab() is None
    assert cam._duplicator._frame_acquired is False
    assert cam.last_recovery_reason == "unhandled error during grab"


def test_a_buffer_that_will_not_release_does_not_mask_the_grab_error(pipeline, monkeypatch):
    cam, _, _, device = pipeline()

    def broken_copy(*args):
        raise OSError("device hung")

    device.im_context.CopySubresourceRegion = broken_copy
    original_checkout = cam.memory_pool.checkout

    def checkout():
        wrapper = original_checkout()

        def refuse():
            raise ValueError("pool destroyed")

        wrapper.release = refuse
        return wrapper

    monkeypatch.setattr(cam.memory_pool, "checkout", checkout)

    assert cam.grab() is None
    assert "device hung" in cam._last_capture_error_message


# --------------------------------------------------------------------------
# construction, region mapping, small accessors
# --------------------------------------------------------------------------

def test_an_unexpected_construction_error_is_raised_as_itself(pipeline):
    FakeStageSurface.rebuild_error = OSError("CreateTexture2D failed")

    with pytest.raises(OSError, match="CreateTexture2D failed"):
        pipeline()


def test_region_mapping_refuses_an_impossible_rotation(pipeline):
    cam, _, _, _ = pipeline()
    odd = FakeOutput(64, 48, 45)

    with pytest.raises(ValueError, match="Invalid rotation angle: 45"):
        cam.region_to_memory_region((0, 0, 1, 1), 45, odd)


def test_bytes_per_frame_defaults_to_the_camera_region(pipeline):
    cam, _, _, _ = pipeline(region=(0, 0, 10, 20), output_color="RGBA")
    assert cam.bytes_per_frame() == 10 * 20 * 4


def test_start_while_capturing_is_ignored(pipeline):
    cam, _, _, _ = pipeline()
    cam.is_capturing = True
    cam._capture_thread = None

    assert cam.start() is None
    assert cam._capture_thread is None, "no second thread was started"
    cam.is_capturing = False


def test_bgra_queue_limit_without_a_pool_uses_the_requested_size(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3, max_buffer_len=64)
    cam.memory_pool = None

    assert cam._queue_limit() == 2


def test_stop_survives_a_timer_that_will_not_close(pipeline):
    cam, _, _, _ = pipeline()
    cam.is_capturing = True
    cam._capture_thread = None
    cam._timer_handle = 0xDEF
    pipeline.timers.close_error = OSError("close failed")

    assert cam.stop() is True
    assert cam._timer_handle is None


# --------------------------------------------------------------------------
# CuPy-shaped branches, with NumPy standing in
# --------------------------------------------------------------------------

@pytest.fixture
def numpy_as_cupy(monkeypatch):
    """CuPy's array API is NumPy's for everything these branches touch."""
    fake = types.SimpleNamespace(
        empty=np.empty, uint8=np.uint8, ndarray=np.ndarray,
        asnumpy=lambda a: np.array(a, copy=True))
    monkeypatch.setattr(capture_module, "_require_cupy", lambda: fake)
    return fake


@pytest.mark.parametrize("color", ["BGRA", "RGB"])
def test_gpu_scratch_staging_uses_the_device_array_module(pipeline, numpy_as_cupy, color):
    cam, _, _, _ = pipeline(output_color=color)
    cam.nvidia_gpu = True

    buffer = cam._scratch_staging_buffer(3, 5)

    assert buffer.shape == (3, 5, 4)


@pytest.mark.parametrize("as_numpy", [True, False])
def test_get_latest_frame_copies_a_device_array(pipeline, numpy_as_cupy, as_numpy):
    cam, _, _, _ = pipeline()
    cam.nvidia_gpu = True
    queued = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    cam._pooled_frames_deque = collections.deque([queued])
    cam._frame_available_event.set()

    frame = cam.get_latest_frame(as_numpy=as_numpy)

    np.testing.assert_array_equal(frame, queued)
    assert frame is not queued


def test_a_gpu_frame_buffer_rebuild_uses_the_device_pool(pipeline, monkeypatch):
    cam, _, _, _ = pipeline(output_color="BGRA")
    cam.nvidia_gpu = True
    built = []
    monkeypatch.setattr(capture_module, "cupy_available", lambda: True)
    monkeypatch.setattr(capture_module, "CupyMemoryPool",
                        lambda *a: built.append(a) or NumpyMemoryPool(*a))

    cam._rebuild_frame_buffer((0, 0, 32, 24))

    assert built and built[0][0] == (24, 32, 4)


def test_module_cp_name_resolves_lazily(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(capture_module, "_require_cupy", lambda: sentinel)
    assert capture_module.cp is sentinel


def test_require_cupy_reports_absence_once(monkeypatch):
    monkeypatch.setattr(capture_module, "_cupy", None)
    monkeypatch.setattr(capture_module, "_cupy_import_attempted", False)
    monkeypatch.setitem(sys.modules, "cupy", None)     # import raises ImportError

    assert capture_module._require_cupy() is None
    assert capture_module._cupy_import_attempted is True


# --------------------------------------------------------------------------
# video mode, one loop iteration at a time
# --------------------------------------------------------------------------

def run_one_idle_tick(cam, dup_source):
    """Drive _capture_thread_func for exactly one tick with no new frame."""
    cam._pooled_frames_deque = collections.deque(maxlen=4)
    cam._last_dup_source = dup_source
    cam._stop_capture_event.clear()

    def idle_grab(region):
        cam._stop_capture_event.set()      # this is the last tick
        return None

    cam._grab = idle_grab
    cam._capture_thread_func(cam.region, target_fps=0, video_mode=True)
    return list(cam._pooled_frames_deque)


def test_video_mode_waits_while_there_is_nothing_to_repeat(pipeline):
    cam, _, _, _ = pipeline()
    assert run_one_idle_tick(cam, None) == []


def test_video_mode_without_a_pool_skips_the_repeat(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=2)
    source = cam.memory_pool.checkout()
    cam.memory_pool = None

    assert run_one_idle_tick(cam, source) == []
    assert cam._capture_permanently_failed is False


def test_video_mode_repeat_on_the_gpu_path_copies_through_the_pool(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    source = cam.memory_pool.checkout()
    source.array[:] = 7
    cam.nvidia_gpu = True

    queued = run_one_idle_tick(cam, source)

    assert len(queued) == 1 and queued[0] is not source
    assert (queued[0].array == 7).all()
    queued[0].release()
    source.release()


def test_a_failed_repeat_is_dropped_not_fatal(pipeline):
    cam, _, _, _ = pipeline()

    class Uncopyable:
        array = None               # _frame_array returns None; .copy() fails

    assert run_one_idle_tick(cam, Uncopyable()) == []
    assert cam._capture_permanently_failed is False


# --------------------------------------------------------------------------
# Frame.changed_fraction edge cases
# --------------------------------------------------------------------------

def frame(region=(0, 0, 10, 10), dirty=None, move=None):
    return Frame(texture=object(), on_release=None, region=region,
                 dirty_rects=dirty, move_rects=move)


def test_changed_fraction_merges_separate_spans_on_one_row():
    """Two rects on the same rows with a gap between them: the union is the
    sum of both spans, not the span from the first left to the last right."""
    f = frame(dirty=[(0, 0, 2, 10), (5, 0, 7, 10)])
    assert f.changed_fraction == pytest.approx(0.4)


def test_only_degenerate_rects_mean_redraw_everything():
    """Zero-area rects are dropped when the frame is built, leaving no rects --
    which means "unknown", so the whole frame counts as changed."""
    f = frame(dirty=[(3, 3, 3, 8)])
    assert f.dirty_rects == []
    assert f.changed_fraction == 1.0


def test_changed_fraction_of_an_empty_frame_is_zero():
    f = frame(dirty=[(0, 0, 1, 1)])
    f._width = f._height = 0
    assert f.changed_fraction == 0.0


# --------------------------------------------------------------------------
# duplicator, DXcam shim, native, diagnostics
# --------------------------------------------------------------------------

def test_get_frame_on_a_degraded_duplicator_is_none():
    dup = make_duplicator()
    dup.duplicator = None             # update_frame() now returns False
    assert dup.get_frame() is None


def test_grab_view_with_nothing_new_is_none():
    class Idle:
        def grab(self, region=None):
            return None

    assert dxcam.DXCamera(Idle()).grab_view() is None


def test_shim_attribute_guard_does_not_recurse():
    half_built = dxcam.DXCamera.__new__(dxcam.DXCamera)
    with pytest.raises(AttributeError):
        half_built._outstanding


def test_build_info_names_the_wheel_version(monkeypatch):
    class Ext:
        def build_info(self):
            return {"version": "0.1.0"}

    monkeypatch.setattr(native, "_ext", Ext())
    monkeypatch.setattr(native, "_ext_source", "rapidshot-native wheel")
    monkeypatch.setitem(sys.modules, "rapidshot_native",
                        types.SimpleNamespace(__version__="0.1.2"))

    info = native.build_info()

    assert info == {"version": "0.1.0", "source": "rapidshot-native wheel",
                    "wheel_version": "0.1.2"}


def test_kernels_refuse_rows_packed_tighter_than_their_width():
    src = np.zeros((4, 6, 4), np.uint8)
    base = np.zeros(64, np.uint8)
    dst = np.lib.stride_tricks.as_strided(base, shape=(4, 6, 3), strides=(3, 3, 1))

    assert native._addressable(src, dst, 3) is False


def test_an_unreadable_quarantine_flag_lets_the_frame_release():
    class Inner:
        def transfer_async(self, texture, source_id):
            raise RuntimeError("submit failed")

        @property
        def submission_quarantined(self):
            raise OSError("device removed")

    transfer = native.CrossAdapterTransfer.__new__(native.CrossAdapterTransfer)
    transfer._inner = Inner()
    released = []
    f = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))

    with pytest.raises(RuntimeError, match="submit failed"):
        transfer.transfer_async(f)
    f.release()

    assert released == [True]


def test_diagnose_says_the_cross_adapter_probe_was_not_run(monkeypatch):
    class Native:
        def is_available(self):
            return True

        def build_info(self):
            return {"source": "wheel"}

        def probe_shareable_buffers(self):
            return {}

        def probe_onnxruntime(self):
            return {}

    monkeypatch.setattr(rapidshot, "native", Native(), raising=False)

    assert "cross-adapter    : not probed (pass probe_gpu=True)" in rapidshot.diagnose()


def test_version_info_reports_missing_dependencies(monkeypatch):
    for name in ("numpy", "cupy", "PIL", "cv2", "comtypes"):
        monkeypatch.setitem(sys.modules, name, None)

    deps = rapidshot.get_version_info()["dependencies"]

    assert deps == {"numpy": "not installed", "cupy": "not installed",
                    "pillow": "not installed", "opencv": "not installed",
                    "comtypes": "version unknown"}


def test_shot_rotated_refuses_a_destination_too_small(pipeline):
    """`shot()` validates the caller's buffer before capturing; the rotated
    path then wrote `frame.nbytes` into it on the strength of a docstring
    saying the two must match.

    They match while nothing else is wrong. A rotated frame whose shape is not
    what the destination was sized for overruns the caller's memory -- and
    `describe_destination` was already returning the size, which this discarded
    into `_`.
    """
    cam, _, _, _ = pipeline(rotation=90)
    too_small = np.zeros(8, dtype=np.uint8)

    with pytest.raises(ValueError, match="too small"):
        cam._shot_rotated(too_small, Mapped(), 4, 4)


def test_shot_rotated_writes_a_destination_that_fits(pipeline):
    """The check must not refuse the ordinary case it is guarding."""
    cam, _, _, _ = pipeline(rotation=90)
    channels = cam._processor.output_channels
    destination = np.zeros(4 * 4 * channels, dtype=np.uint8)

    cam._shot_rotated(destination, Mapped(), 4, 4)
