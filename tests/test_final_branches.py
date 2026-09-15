"""The last headless-reachable lines of the library.

After the per-module suites, line coverage with GPU access blocked left 232
lines. These tests take every one that can be reached without a device:
the native wrappers over a fake extension, cursor snapshots, the CuPy pool,
display metadata, and the module-level fallbacks a missing DLL or dependency
selects -- the last by executing a private copy of the module with that
dependency blocked, so the real, already-imported modules are never disturbed.

What is deliberately not here: ``Device``'s success path, which needs a real
D3D11 device (a fake would hand comtypes pointers it later tries to release).
"""
import ctypes
import importlib.util
import logging
import sys
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot  # noqa: E402
import rapidshot.capture as capture_module  # noqa: E402
import rapidshot.converter as converter_module  # noqa: E402
import rapidshot.core.output as output_module  # noqa: E402
import rapidshot.memory_pool as memory_pool_module  # noqa: E402
import rapidshot.native as native  # noqa: E402
import rapidshot.processor.numpy_processor as numpy_processor_module  # noqa: E402
import rapidshot.util.desktop as desktop_module  # noqa: E402
import rapidshot.util.io as io_module  # noqa: E402
import rapidshot.util.logging as logging_module  # noqa: E402
from rapidshot.core.duplicator import Cursor  # noqa: E402
from rapidshot.frame import Frame  # noqa: E402
from rapidshot.memory_pool import BaseMemoryPool, NumpyMemoryPool  # noqa: E402
from rapidshot.processor.base import Processor  # noqa: E402

from test_capture_paths import FakeDuplicator, pipeline  # noqa: E402,F401
from test_cupy_processor_paths import NumpyAsCupy  # noqa: E402
from test_duplicator_paths import FakeDuplication, make_duplicator  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def load_copy(relative_path, name, monkeypatch, blocked=(), patch=None):
    """Execute a private copy of a module with some imports made to fail."""
    for module in blocked:
        monkeypatch.setitem(sys.modules, module, None)
    if patch is not None:
        patch()
    spec = importlib.util.spec_from_file_location(name, REPO / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame(region=(10, 20, 110, 70), source_id=9):
    return Frame(ctypes.c_void_p(0x1234), lambda: None, region, source_id=source_id)


# --------------------------------------------------------------------------
# module-level fallbacks
# --------------------------------------------------------------------------

def test_the_package_imports_without_capture_or_the_gpu_path(monkeypatch):
    probe = load_copy("rapidshot/__init__.py", "_rapidshot_probe", monkeypatch,
                      blocked=("rapidshot.capture", "rapidshot.core", "rapidshot.util.io",
                               "rapidshot.converter"))

    assert probe.ScreenCapture is None and isinstance(probe._capture_import_error, ImportError)
    assert probe.Output is None and probe.Device is None
    assert probe.enum_dxgi_adapters is None and probe.get_output_metadata is None
    assert probe.GpuConverter is None and probe.TensorStream is None
    assert callable(probe.to_nchw), "the pure-NumPy parts still import"


def _no_dlls(monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("DLL not found")
    return lambda: monkeypatch.setattr(ctypes, "WinDLL", refuse)


def test_io_imports_where_dxgi_is_missing(monkeypatch):
    probe = load_copy("rapidshot/util/io.py", "_io_probe", monkeypatch, patch=_no_dlls(monkeypatch))
    assert probe._dxgi is None and probe._user32 is None


def test_desktop_diagnostic_abstains_where_user32_is_missing(monkeypatch):
    probe = load_copy("rapidshot/util/desktop.py", "_desktop_probe", monkeypatch,
                      patch=_no_dlls(monkeypatch))
    state = probe.describe_desktop_access()
    assert (state.thread_desktop, state.is_input_desktop, state.blocked_reason) == (None, None, None)


def test_the_numpy_kernels_fall_back_when_the_extension_cannot_import(monkeypatch):
    monkeypatch.setattr(numpy_processor_module, "_NATIVE_SWIZZLE", None)
    monkeypatch.setattr(numpy_processor_module, "_NATIVE_GRAY", None)
    monkeypatch.setitem(sys.modules, "rapidshot.native", None)
    src = np.zeros((2, 2, 4), np.uint8)

    assert numpy_processor_module._native_swizzle(src, np.zeros((2, 2, 3), np.uint8), "RGB") is False
    assert numpy_processor_module._native_gray(src, np.zeros((2, 2), np.uint8)) is False
    assert numpy_processor_module._NATIVE_SWIZZLE is False, "the failed lookup is cached"


# --------------------------------------------------------------------------
# native wrappers over a fake extension
# --------------------------------------------------------------------------

class Recorder:
    """Any attribute is a callable that records its arguments and returns a canned value."""

    def __init__(self, returns=None, **attributes):
        self.calls = []
        self.returns = returns or {}
        for key, value in attributes.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            value = self.returns.get(name)
            return value(*args, **kwargs) if callable(value) else value
        return call


@pytest.fixture
def ext(monkeypatch):
    fake = Recorder(returns={
        "describe_texture": lambda address: [("width", 100), ("address", address)],
        "probe_d3d12_sharing": lambda address: [("shareable", True)],
        "probe_shareable_buffers": lambda: [("d3d12_available", True)],
        "probe_cross_adapter": lambda **kwargs: [("kwargs", kwargs)],
    })
    monkeypatch.setattr(native, "_ext", fake)
    return fake


def test_require_returns_the_loaded_extension(ext):
    assert native.require() is ext


def test_swizzle_passes_both_pitches_and_exact_lengths(ext):
    """The kernel walks raw memory; the lengths bound what it may touch."""
    src = np.zeros((3, 10, 4), np.uint8)[:, :8]            # strided rows: pitch 40
    dst = np.zeros((3, 8, 3), np.uint8)

    assert native.bgra_swizzle_into(src, dst, "RGB") is True

    name, args, _ = ext.calls[-1]
    assert name == "bgra_swizzle_into"
    assert args[1] == 40 * 2 + 8 * 4, "last row counts only its own pixels"
    assert args[3] == 24 * 2 + 8 * 3
    assert args[4:] == (8, 3, "RGB", 40, 24)


def test_gray_passes_both_pitches_and_exact_lengths(ext):
    src = np.zeros((2, 5, 4), np.uint8)
    dst = np.zeros((2, 5), np.uint8)

    assert native.bgra_to_gray_into(src, dst) is True
    _, args, _ = ext.calls[-1]
    assert args[1] == 20 + 20 and args[3] == 5 + 5 and args[4:] == (5, 2, 20, 5)

    assert native.bgra_to_gray_into(src, np.zeros((2, 5, 1), np.uint8)) is False


@pytest.mark.parametrize("src", [
    np.zeros((2, 5, 4), np.uint8)[:, ::-1],      # reversed columns
    np.zeros((2, 5, 3), np.uint8),               # no alpha
])
def test_sources_the_kernels_cannot_address_are_declined(src):
    assert native._addressable(src, np.zeros((2, 5, 3), np.uint8), 3) is False


def test_the_kernels_decline_what_they_cannot_do_even_when_loaded(ext):
    src = np.zeros((2, 5, 4), np.uint8)
    assert native.bgra_swizzle_into(src, np.zeros((2, 5, 3), np.uint8), "GRAY") is False
    assert native.bgra_swizzle_into(src, np.zeros((2, 4, 3), np.uint8), "RGB") is False
    assert ext.calls == [], "nothing reached the extension"


def test_rgb_destination_of_the_wrong_dtype_is_declined():
    assert native._addressable(np.zeros((2, 5, 4), np.uint8), np.zeros((2, 5, 3), np.float32), 3) is False


def test_probes_and_descriptions_come_back_as_dicts(ext):
    f = frame()
    assert native.describe_texture(f) == {"width": 100, "address": 0x1234}
    assert native.probe_d3d12_sharing(f) == {"shareable": True}
    assert native.probe_shareable_buffers() == {"d3d12_available": True}
    assert native.probe_cross_adapter(64, 32, 3) == {"kwargs": {"width": 64, "height": 32, "iterations": 3}}


def test_d3d11_preprocessor_wrapper(ext, monkeypatch):
    impl = Recorder(returns={"read_back": lambda: np.arange(12, dtype=np.float32).tobytes()},
                    shape=[1, 3, 2, 2])
    monkeypatch.setattr(ext, "GpuPreprocessor", lambda address, w, h: impl, raising=False)

    pre = native.GpuPreprocessor(frame(), 2.0, 2)
    pre.process(frame(), scale=2.0, bias=-1.0, bgr=True)

    assert (pre.out_width, pre.out_height) == (2, 2)
    assert impl.calls[0] == ("process", (0x1234, 2.0, -1.0, True), {"crop": (10, 20, 100, 50)})
    assert pre.read_back().shape == (1, 3, 2, 2)


def test_d3d12_preprocessor_wrapper(ext, monkeypatch):
    impl = Recorder(returns={"read_back": lambda: np.zeros(12, np.float32).tobytes()},
                    shape=[1, 3, 2, 2], shared_output_handle=0x40, output_byte_size=48,
                    adapter_luid=[1, 2, 3, 4, 5, 6, 7, 8])
    monkeypatch.setattr(ext, "GpuPreprocessor12", lambda address, w, h: impl, raising=False)

    pre = native.GpuPreprocessor12(frame(), 2, 2)
    pre.process(frame())

    assert impl.calls[0][2] == {"source_id": 9, "crop": (10, 20, 100, 50)}
    assert pre.read_back().shape == (1, 3, 2, 2)
    assert (pre.shared_output_handle, pre.output_byte_size) == (0x40, 48)
    assert pre.adapter_luid == bytes(range(1, 9))


def test_cross_adapter_transfer_wrapper(ext, monkeypatch):
    inner = Recorder(
        returns={"read_back_destination": lambda: [1, 2],
                 "transfer_with_reference": lambda address, source: [3, 4],
                 "probe_shared_handles": lambda: [[("label", "heap"), ("ok", True)]]},
        shared_fence_completed=5, shared_fence_submitted=6, shared_fence_handle=0x70,
        cached_texture_address=0x1234, cached_source_id=9, shared_destination_handle=0x80,
        destination_is_software=1, total_bytes=1024, dxgi_format=87, bytes_per_pixel=4,
        row_pitch=256)
    monkeypatch.setattr(ext, "CrossAdapterTransfer", lambda address: inner, raising=False)

    transfer = native.cross_adapter_transfer(frame())
    transfer.transfer(frame())
    transfer.set_consumer_fence("12")
    transfer.wait_for_consumer(3.0)

    assert inner.calls[:3] == [("transfer", (0x1234, 9), {}),
                               ("set_consumer_fence", (12,), {}),
                               ("wait_for_consumer", (3,), {})]
    # 3.0 == 3 in Python, so also check the types reaching the extension.
    assert [type(call[1][0]) for call in inner.calls[1:3]] == [int, int]
    assert transfer.read_back_destination() == b"\x01\x02"
    assert transfer.transfer_with_reference(frame()) == b"\x03\x04"
    assert transfer.probe_shared_handles() == [{"label": "heap", "ok": True}]
    assert (transfer.shared_fence_completed, transfer.shared_fence_submitted,
            transfer.shared_fence_handle) == (5, 6, 0x70)
    assert (transfer.cached_texture_address, transfer.cached_source_id,
            transfer.shared_destination_handle) == (0x1234, 9, 0x80)
    assert transfer.destination_is_software is True
    assert (transfer.total_bytes, transfer.dxgi_format, transfer.bytes_per_pixel,
            transfer.row_pitch) == (1024, 87, 4, 256)


# --------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------

def test_the_cursor_snapshot_copies_every_field(pipeline):
    cam, _, _, _ = pipeline()
    cursor = Cursor()
    cursor.PointerPositionInfo.Position.x, cursor.PointerPositionInfo.Position.y = 30, 40
    shape = cursor.PointerShapeInfo
    shape.Type, shape.Width, shape.Height, shape.Pitch = 2, 32, 32, 128
    shape.HotSpot.x, shape.HotSpot.y = 3, 4
    cursor.Shape = b"pixels"
    cam._duplicator.cursor = cursor
    cam._duplicator.cursor_visible = 1

    info = cam._cursor_info()
    cursor.PointerPositionInfo.Position.x = 999        # a later acquire mutates it

    assert info.position == (30, 40)
    assert (info.hotspot, info.shape, info.shape_type) == ((3, 4), b"pixels", 2)
    assert (info.shape_size, info.shape_pitch, info.visible) == ((32, 32), 128, True)


def test_a_declined_output_target_goes_back_to_its_pool(pipeline):
    """Rotation makes the processor allocate; the output buffer it was offered
    must be returned rather than leaked."""
    cam, _, _, _ = pipeline(rotation=90, output_color="RGB", pool_output=True)

    frame_out = cam.grab()

    assert isinstance(frame_out, np.ndarray)
    assert cam._output_pool.get_stats()["in_use"] == 0


def test_discarding_a_frame_that_will_not_release_is_quiet():
    class Stubborn:
        def release(self):
            raise ValueError("already released")

    capture_module.ScreenCapture._discard_frame(Stubborn())


def test_released_reports_the_camera_state(pipeline):
    cam, _, _, _ = pipeline()
    assert cam.released is False
    cam.release()
    assert cam.released is True


def test_a_gpu_camera_builds_a_device_staging_pool(pipeline, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", NumpyAsCupy())
    monkeypatch.setattr(capture_module, "cupy_available", lambda: True)
    built = []
    monkeypatch.setattr(capture_module, "CupyMemoryPool",
                        lambda *args: built.append(args) or NumpyMemoryPool(*args))

    cam, _, _, _ = pipeline(nvidia_gpu=True, output_color="BGRA")

    assert cam.nvidia_gpu is True and built[0][0] == (48, 64, 4)


def test_cupy_is_found_when_installed(monkeypatch):
    fake = NumpyAsCupy()
    monkeypatch.setitem(sys.modules, "cupy", fake)
    monkeypatch.setattr(capture_module, "_cupy", None)
    monkeypatch.setattr(capture_module, "_cupy_import_attempted", False)

    assert capture_module.cupy_available() is True
    assert capture_module._require_cupy() is fake


def test_a_flagged_region_without_a_recorded_request_is_kept(pipeline):
    cam, _, _, _ = pipeline()
    cam.__dict__.pop("_requested_region", None)
    cam._region_set_by_user = True
    cam.region = (1, 2, 30, 40)

    assert cam._fit_requested_region() == (1, 2, 30, 40)


# --------------------------------------------------------------------------
# factory
# --------------------------------------------------------------------------

def test_create_with_cupy_installed_keeps_the_gpu_request(monkeypatch):
    from test_factory_paths import FakeCapture, one_display

    monkeypatch.setitem(sys.modules, "cupy", NumpyAsCupy())
    factory, _ = one_display(monkeypatch)

    camera = factory.create(nvidia_gpu=True)

    assert isinstance(camera, FakeCapture) and camera.kwargs["nvidia_gpu"] is True


def test_version_info_reports_an_installed_cupy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", NumpyAsCupy(version="14.1.1"))
    assert rapidshot.get_version_info()["dependencies"]["cupy"] == "14.1.1"


# --------------------------------------------------------------------------
# processors, pools, frames, duplicator
# --------------------------------------------------------------------------

def test_bgra_convert_into_is_a_copy():
    src = np.arange(16, dtype=np.uint8).reshape(2, 2, 4)
    dst = np.zeros_like(src)
    numpy_processor_module.NumpyProcessor("BGRA").convert_into(src, dst)
    np.testing.assert_array_equal(dst, src)


def test_cupy_bgra_conversion_is_the_identity(monkeypatch):
    from rapidshot.processor.cupy_processor import CupyProcessor

    monkeypatch.setitem(sys.modules, "cupy", NumpyAsCupy())
    image = np.zeros((2, 2, 4), np.uint8)
    assert CupyProcessor("BGRA").process_cvtcolor(image) is image


def test_dependency_checks_tolerate_missing_packages(monkeypatch):
    proc = Processor(output_color="RGB")
    real_numpy = sys.modules["numpy"]
    monkeypatch.setitem(sys.modules, "cv2", None)
    sys.modules["numpy"] = None
    try:
        proc._check_dependencies()
    finally:
        sys.modules["numpy"] = real_numpy


def test_cupy_pool_allocates_on_the_device(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", NumpyAsCupy())
    pool = memory_pool_module.CupyMemoryPool((2, 2, 4), np.uint8, 2)
    assert pool.checkout().array.shape == (2, 2, 4)


def test_cupy_pool_without_cupy_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ImportError, match="requires CuPy"):
        memory_pool_module.CupyMemoryPool((2, 2, 4), np.uint8, 2)


def test_checkin_to_a_pool_that_is_not_initialized_is_refused():
    pool = NumpyMemoryPool((1,), np.uint8, 1)
    buf = pool.checkout()
    pool._initialized = False
    with pytest.raises(RuntimeError, match="not initialized"):
        pool.checkin(buf)


def test_initialization_that_lost_a_race_does_nothing():
    class Racing(BaseMemoryPool):
        checks = 0

        @property
        def _initialized(self):
            Racing.checks += 1
            return Racing.checks > 1          # another thread won between checks

        @_initialized.setter
        def _initialized(self, value):
            pass

        def _create_buffer(self):
            raise AssertionError("must not allocate")

    Racing((1,), np.uint8, 1).initialize_pool()


def test_release_all_leaves_the_pool_unchanged_when_a_replacement_cannot_be_allocated(monkeypatch):
    pool = NumpyMemoryPool((1,), np.uint8, 2)
    held = pool.checkout()

    def out_of_memory():
        raise MemoryError("no replacement")

    monkeypatch.setattr(pool, "_create_buffer", out_of_memory)
    with pytest.raises(MemoryError):
        pool.release_all_buffers()

    assert held.state == "IN_USE"
    assert pool.get_stats() == {"total": 2, "available": 1, "in_use": 1, "initialized": True}


def test_a_failing_non_quarantine_drain_still_releases(caplog):
    released = []
    f = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))

    def broken():
        raise OSError("event handle closed")

    f.defer_release_until(broken)
    # On the package logger, not root: benchmarks/perf_suite.py sets "rapidshot"
    # to ERROR when imported, which filters this warning for the rest of a run.
    with caplog.at_level(logging.WARNING, logger="rapidshot"):
        f.release()

    assert released == [True] and "A release drain failed" in caplog.text


def test_a_new_cursor_shape_replaces_the_held_one():
    def acquire(info):
        info.LastMouseUpdateTime = 5
        info.PointerShapeBufferSize = 16

    def shape(size, buf, required, info):
        info._obj.Width = 16
        return 0

    duplication = FakeDuplication(acquire)
    duplication.pointer_shape = shape
    dup = make_duplicator(duplication)

    dup.update_frame()

    assert dup.cursor.Shape is not None and len(dup.cursor.Shape) == 16
    assert dup.cursor.PointerShapeInfo.Width == 16


def test_cuda_view_destructor_swallows_close_errors():
    view = converter_module._CudaView.__new__(converter_module._CudaView)

    def fail():
        raise RuntimeError("driver gone")

    view.close = fail
    view.__del__()


# --------------------------------------------------------------------------
# display metadata, DPI, desktop, logging
# --------------------------------------------------------------------------

class FakeEnumDisplay:
    """EnumDisplayDevicesW over a scripted adapter -> monitors table."""

    ACTIVE, PRIMARY = 0x1, 0x4

    def __init__(self, adapters):
        self.adapters = adapters

    def EnumDisplayDevicesW(self, device, index, info_ref, flags):
        info = info_ref._obj
        if device is None:
            if index >= len(self.adapters):
                return 0
            name, string, state, _ = self.adapters[index]
            info.DeviceName, info.DeviceString, info.StateFlags = name, string, state
            return 1
        monitors = next(a[3] for a in self.adapters if a[0] == device)
        if index >= len(monitors):
            return 0
        info.DeviceName, info.DeviceString = monitors[index]
        return 1


def test_output_metadata_lists_active_adapters_and_their_monitors(monkeypatch):
    table = [
        ("\\\\.\\DISPLAY1", "Intel UHD", 0x1 | 0x4, [("\\\\.\\DISPLAY1\\Monitor0", "Built-in")]),
        ("\\\\.\\DISPLAY2", "Intel UHD", 0x0, []),
        ("\\\\.\\DISPLAY3", "NVIDIA", 0x1, [("\\\\.\\DISPLAY3\\Monitor0", "Dell"),
                                           ("\\\\.\\DISPLAY3\\Monitor1", "LG")]),
    ]
    monkeypatch.setattr(io_module, "_user32", FakeEnumDisplay(table))

    metadata = io_module.get_output_metadata()

    assert set(metadata) == {"\\\\.\\DISPLAY1", "\\\\.\\DISPLAY3"}
    assert metadata["\\\\.\\DISPLAY1"][:2] == ["Intel UHD", True]
    assert metadata["\\\\.\\DISPLAY3"][1] is False
    assert [m[1] for m in metadata["\\\\.\\DISPLAY3"][2]] == ["Dell", "LG"]


def test_dpi_setup_without_shcore_is_skipped(monkeypatch, caplog):
    monkeypatch.setattr(output_module, "_dpi_awareness_attempted", False)

    def refuse(name):
        raise OSError("shcore.dll missing")

    monkeypatch.setattr(output_module.ctypes, "WinDLL", refuse)
    with caplog.at_level(logging.DEBUG, logger=output_module.logger.name):
        output_module._ensure_process_dpi_awareness()
    assert "Could not set process DPI awareness" in caplog.text


def test_an_unreadable_current_awareness_is_reported_as_unknown(monkeypatch, caplog):
    from test_support_modules import FakeShcore, _Function

    monkeypatch.setattr(output_module, "_dpi_awareness_attempted", False)
    shcore = FakeShcore(-2147024891)

    def broken(process, value_ref):
        raise OSError("GetProcessDpiAwareness failed")

    shcore.GetProcessDpiAwareness = _Function(broken)
    monkeypatch.setattr(output_module.ctypes, "WinDLL", lambda name: shcore)
    with caplog.at_level(logging.WARNING, logger=output_module.logger.name):
        output_module._ensure_process_dpi_awareness()
    assert "already set to -1" in caplog.text


def test_the_desktop_diagnostic_never_raises(monkeypatch):
    from test_support_modules import FakeUser32

    class Broken(FakeUser32):
        def GetThreadDesktop(self, thread_id):
            raise OSError("access denied")

    class CloseFails(FakeUser32):
        def CloseDesktop(self, handle):
            raise OSError("invalid handle")

    kernel32 = types.SimpleNamespace(GetCurrentThreadId=lambda: 1)
    monkeypatch.setattr(desktop_module, "_kernel32", kernel32)

    monkeypatch.setattr(desktop_module, "_user32", Broken({}))
    assert desktop_module.describe_desktop_access().thread_desktop is None

    monkeypatch.setattr(desktop_module, "_user32", CloseFails({1: "Default", 2: "Default"}))
    assert desktop_module.describe_desktop_access().is_input_desktop is True


def test_the_log_file_can_come_from_the_environment(tmp_path, monkeypatch):
    logger = logging.getLogger("rapidshot")
    saved = logger.handlers[:], logger.level
    logger.handlers = []
    target = tmp_path / "from-env.log"
    monkeypatch.setenv(logging_module.LOG_FILE_ENV_VAR, str(target))
    monkeypatch.delenv(logging_module.LOG_LEVEL_ENV_VAR, raising=False)
    try:
        configured = logging_module.setup_logging()
        assert configured.handlers[1].baseFilename == str(target)
    finally:
        for handler in logger.handlers:
            handler.close()
        logger.handlers, logger.level = saved
