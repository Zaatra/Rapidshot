"""dxcam_compat.py and native.py: the branches the other suites never reach.

Both are thin layers, which is why their gaps are easy to miss. The DXcam shim's
`latest_frame_time` was reported as covered by a unit test and was 0 on every
real camera: the test's fake camera had an attribute the real one did not.
So the shim is exercised here against a real ScreenCapture over the fake
pipeline from test_capture_paths, and the native wrappers against fake
extension objects.
"""
import ctypes
import sys
import types

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot  # noqa: E402
import rapidshot.dxcam_compat as dxcam  # noqa: E402
import rapidshot.native as native  # noqa: E402
from rapidshot.frame import Frame, FrameQuarantinedError  # noqa: E402

from test_capture_paths import FakeDuplicator, pipeline  # noqa: E402,F401


# --------------------------------------------------------------------------
# DXcam shim over a real ScreenCapture
# --------------------------------------------------------------------------

def test_latest_frame_time_comes_from_the_real_camera(pipeline, monkeypatch):
    cam, _, _, _ = pipeline(pool_output=False)
    camera = dxcam.DXCamera(cam)
    assert camera.latest_frame_ticks == 0 and camera.latest_frame_time == 0.0

    cam._duplicator.last_present_time = 30_000_000
    monkeypatch.setattr("rapidshot.frame._qpc_freq", lambda: 10_000_000)

    assert camera.latest_frame_ticks == 30_000_000
    assert camera.latest_frame_time == 3.0


def test_camera_last_present_time_survives_a_missing_duplicator(pipeline):
    cam, _, _, _ = pipeline()
    cam._duplicator = None
    assert cam.last_present_time == 0


def test_shim_attributes_mirror_the_camera(pipeline):
    cam, _, _, _ = pipeline(region=(2, 3, 30, 40), output_color="BGR")
    camera = dxcam.DXCamera(cam)

    assert (camera.width, camera.height, camera.channel_size) == (64, 48, 3)
    assert camera.region == (2, 3, 30, 40)
    assert camera.is_capturing is False
    assert camera.timeout_ms == cam.timeout_ms          # forwarded
    assert "rapidshot compat" in repr(camera)


def test_shim_shot_with_and_without_a_region(pipeline):
    cam, _, desktop, _ = pipeline()
    camera = dxcam.DXCamera(cam)
    full = np.zeros((48, 64, 3), np.uint8)
    part = np.zeros((10, 10, 3), np.uint8)

    assert camera.shot(full) is True
    assert camera.shot(part, region=(0, 0, 10, 10)) is True
    np.testing.assert_array_equal(part, full[:10, :10])


def test_grab_view_of_a_region_is_retired_by_the_next_grab(pipeline):
    cam, _, desktop, _ = pipeline(output_color="BGRA", pool_size_frames=2)
    camera = dxcam.DXCamera(cam)

    view = camera.grab_view(region=(0, 0, 64, 48))
    held = camera._outstanding
    assert view.shape == (48, 64, 4)

    camera.grab()
    assert held.state == "AVAILABLE", "the previous view's buffer went back to the pool"


def test_retiring_a_view_the_pool_already_took_back_is_quiet():
    camera = dxcam.DXCamera(object())

    class AlreadyReleased:
        def release(self):
            raise ValueError("Buffer is not 'IN_USE'")

    camera._outstanding = AlreadyReleased()
    camera._retire_view()
    assert camera._outstanding is None


def test_start_forwards_region_and_delay_and_views_the_latest_frame(pipeline):
    cam, _, _, _ = pipeline(output_color="BGRA", pool_size_frames=3)
    camera = dxcam.DXCamera(cam)
    calls = {}
    real_start = cam.start

    def recording_start(**kwargs):
        calls.update(kwargs)
        kwargs.pop("delay", None)
        real_start(**kwargs)

    cam.start = recording_start
    camera.start(region=(0, 0, 32, 24), target_fps=0, delay=1)
    try:
        view = camera.get_latest_frame_view()
        plain = camera.get_latest_frame()
    finally:
        camera.stop()

    assert calls == {"region": (0, 0, 32, 24), "target_fps": 0,
                     "video_mode": False, "delay": 1}
    assert view is not None and view.shape == (24, 32, 4)
    assert plain is not None


def test_latest_frame_view_without_a_frame_is_none(monkeypatch):
    class Idle:
        def get_latest_frame_buffer(self):
            return None

    assert dxcam.DXCamera(Idle()).get_latest_frame_view() is None


def test_shim_module_functions_delegate(monkeypatch):
    calls = []
    for name in ("device_info", "output_info", "reset", "clean_up"):
        monkeypatch.setattr(rapidshot, name, lambda name=name: calls.append(name) or name)

    captured = {}
    monkeypatch.setattr(rapidshot, "create", lambda **kw: captured.update(kw) or object())

    camera = dxcam.create(output_idx=1, region=(0, 0, 5, 5), nvidia_gpu=True)

    assert isinstance(camera, dxcam.DXCamera)
    assert captured == {"device_idx": 0, "output_color": "RGB", "max_buffer_len": 64,
                        "output_idx": 1, "region": (0, 0, 5, 5), "nvidia_gpu": True}
    assert [dxcam.device_info(), dxcam.output_info()] == ["device_info", "output_info"]
    dxcam.reset()
    dxcam.clean_up()
    assert calls == ["device_info", "output_info", "reset", "clean_up"]


# --------------------------------------------------------------------------
# native: pure-Python helpers
# --------------------------------------------------------------------------

class FrameLike:
    def __init__(self, width=100, height=50, region=(10, 20, 110, 70), rotation=0):
        self.width, self.height = width, height
        self.region = region
        self.rotation_angle = rotation


@pytest.mark.parametrize("crop,message", [
    ("nope", "must be \\(left, top, right, bottom\\)"),
    ((0, 0, 101, 10), "inside the 100x50 frame"),
    ((5, 5, 5, 10), "non-empty"),
])
def test_crop_validation(crop, message):
    with pytest.raises(ValueError, match=message):
        native._validate_crop(FrameLike(), crop)


def test_crop_is_refused_on_a_rotated_display():
    with pytest.raises(ValueError, match="rotated display"):
        native._validate_crop(FrameLike(rotation=90), (0, 0, 10, 10))


def test_texture_crop_offsets_by_the_region():
    assert native._texture_crop(FrameLike(), (5, 5, 25, 15)) == (15, 25, 20, 10)
    assert native._texture_crop(FrameLike(), None) == (10, 20, 100, 50)
    assert native._texture_crop(FrameLike(region=None), None) is None
    assert native._texture_crop(FrameLike(rotation=180), None) is None


@pytest.mark.parametrize("dst", [
    np.zeros((4, 6, 3), np.uint16),               # wrong dtype
    np.zeros((4, 5, 3), np.uint8),                # wrong shape
    np.zeros((4, 6, 4), np.uint8),                # wrong channel count
    np.zeros((4, 6, 3), np.uint8)[:, ::-1],       # reversed columns
    np.zeros((4, 12, 3), np.uint8)[:, ::2],       # strided columns
])
def test_kernels_refuse_destinations_they_cannot_address(dst):
    src = np.zeros((4, 6, 4), np.uint8)
    assert native._addressable(src, dst, 3) is False


@pytest.mark.parametrize("dst", [
    np.zeros((4, 6, 1), np.uint8),                # gray must be 2-D
    np.zeros((4, 12), np.uint8)[:, ::2],
])
def test_gray_kernel_refuses_destinations_it_cannot_address(dst):
    assert native._addressable(np.zeros((4, 6, 4), np.uint8), dst, 1) is False


def test_kernels_decline_without_the_extension(monkeypatch):
    monkeypatch.setattr(native, "_ext", None)
    src = np.zeros((4, 6, 4), np.uint8)

    assert native.bgra_swizzle_into(src, np.zeros((4, 6, 3), np.uint8), "RGB") is False
    assert native.bgra_to_gray_into(src, np.zeros((4, 6), np.uint8)) is False
    assert native.build_info() is None


def test_swizzle_declines_an_unknown_mode():
    src = np.zeros((4, 6, 4), np.uint8)
    assert native.bgra_swizzle_into(src, np.zeros((4, 6, 3), np.uint8), "GRAY") is False


def test_onnxruntime_dll_lookup(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    assert native.onnxruntime_dll_path() is None

    package = tmp_path / "onnxruntime"
    package.mkdir()
    fake = types.ModuleType("onnxruntime")
    fake.__file__ = str(package / "__init__.py")
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)
    assert native.onnxruntime_dll_path() is None

    nested = package / "somewhere" / "onnxruntime.dll"
    nested.parent.mkdir()
    nested.write_bytes(b"")
    assert native.onnxruntime_dll_path() == str(nested)

    preferred = package / "capi" / "onnxruntime.dll"
    preferred.parent.mkdir()
    preferred.write_bytes(b"")
    assert native.onnxruntime_dll_path() == str(preferred)


# --------------------------------------------------------------------------
# native: wrappers over a fake extension object
# --------------------------------------------------------------------------

def live_frame():
    released = []
    frame = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))
    return frame, released


def transfer_over(inner):
    transfer = native.CrossAdapterTransfer.__new__(native.CrossAdapterTransfer)
    transfer._inner = inner
    return transfer


def test_async_reference_transfer_defers_release_until_the_copy_lands():
    order = []

    class Inner:
        submission_quarantined = False

        def transfer_async_with_reference(self, texture, source_id):
            return 42

        def wait_shared_fence(self, value):
            order.append(("wait", value))

    frame, released = live_frame()
    assert transfer_over(Inner()).transfer_async_with_reference(frame) == 42

    frame.release()
    assert order == [("wait", 42)]
    assert released == [True]


@pytest.mark.parametrize("quarantined", [True, False])
def test_async_reference_transfer_failure_quarantines_only_when_untrackable(quarantined):
    class Inner:
        submission_quarantined = quarantined

        def transfer_async_with_reference(self, texture, source_id):
            raise RuntimeError("Signal failed")

    frame, released = live_frame()
    with pytest.raises(RuntimeError, match="Signal failed"):
        transfer_over(Inner()).transfer_async_with_reference(frame)

    if quarantined:
        with pytest.raises(FrameQuarantinedError):
            frame.release()
        assert released == []
    else:
        frame.release()
        assert released == [True]


def test_an_unreadable_quarantine_flag_is_treated_as_clear():
    class Inner:
        @property
        def submission_quarantined(self):
            raise OSError("device removed")

    assert transfer_over(Inner()).submission_quarantined is False


def test_transfer_diagnostics_and_properties_pass_through():
    class Inner:
        source, destination = "Intel", "NVIDIA"
        destination_resource_address = 0x10
        destination_device_address = 0x20
        width, height = 8, 4

        def read_back_source(self):
            return bytearray(b"\x01\x02")

        def probe_transfer_phases(self, texture, iterations, use_cache, source_id):
            return {"args": (iterations, use_cache, source_id)}

    frame, _ = live_frame()
    transfer = transfer_over(Inner())

    assert transfer.read_back_source() == b"\x01\x02"
    assert transfer.probe_transfer_phases(frame, iterations="7", use_cache=0) == {
        "args": (7, False, 0)}
    assert (transfer.destination_resource_address, transfer.destination_device_address) == (16, 32)
    assert repr(transfer) == "<CrossAdapterTransfer 8x4 'Intel' -> 'NVIDIA'>"
    frame.release()


def test_preprocessor_wrappers_pass_through():
    class Impl:
        shape = [1, 3, 2, 2]
        output_buffer_address = 0x30
        output_resource_address = 0x40
        output_gpu_address = 0x50

    pre = native.GpuPreprocessor.__new__(native.GpuPreprocessor)
    pre._impl = Impl()
    pre12 = native.GpuPreprocessor12.__new__(native.GpuPreprocessor12)
    pre12._impl = Impl()

    assert pre.shape == (1, 3, 2, 2) and pre.output_buffer_address == 48
    assert repr(pre) == "<GpuPreprocessor -> (1, 3, 2, 2)>"
    assert pre12.shape == (1, 3, 2, 2)
    assert (pre12.output_resource_address, pre12.output_gpu_address) == (64, 80)
    assert "DirectML" in repr(pre12)


def test_sharing_and_device_address_helpers_use_the_extension(monkeypatch):
    class Ext:
        def texture_sharing_info(self, address):
            return [("address", address)]

        def get_device_pointer(self, address):
            return address + 1

    monkeypatch.setattr(native, "require", lambda: Ext())
    frame, _ = live_frame()

    assert native.texture_sharing_info(frame) == {"address": 1}
    assert native.device_address(frame) == 2
    frame.release()
