"""The support modules: pools, errors, logging, surfaces, outputs, diagnostics.

Headless line coverage -- measured with GPU access blocked, so only what these
fakes reach counts -- left most of these modules between 40% and 80%. Several
of the gaps were bugs rather than missing tests, each found while writing the
test that now pins it:

- ``import rapidshot`` attached a console handler and a DEBUG-level rotating
  file under ``~/.rapidshot/logs``: 55 MB in two days on the dev machine.
- A frame held while its pool was destroyed raised ``AttributeError`` when read
  and ``RuntimeError`` when released.
- Every ``StageSurface`` shared one ``D3D11_TEXTURE2D_DESC``.
- ``RapidShotDXGIError`` printed HRESULTs as ``-0x7785ffda``.
- ``_create_dxgi_factory1`` ignored the HRESULT it was given.
"""
import ctypes
import logging
import sys
import types

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.core.output as output_module  # noqa: E402
import rapidshot.core.stagesurf as stagesurf_module  # noqa: E402
import rapidshot.processor.base as base_module  # noqa: E402
import rapidshot.util.desktop as desktop_module  # noqa: E402
import rapidshot.util.io as io_module  # noqa: E402
import rapidshot.util.logging as logging_module  # noqa: E402
import rapidshot.util.timer as timer_module  # noqa: E402
import rapidshot.util.topology as topology_module  # noqa: E402
from rapidshot.memory_pool import (  # noqa: E402
    BaseMemoryPool,
    BufferReleasedError,
    NumpyMemoryPool,
    PoolExhaustedError,
)
from rapidshot.processor.base import Processor, ProcessorBackends  # noqa: E402
from rapidshot.profiling import Profiler, RELIABLE_SAMPLES  # noqa: E402
from rapidshot.util.ctypes_helpers import (  # noqa: E402
    describe_destination,
    pointer_to_address,
)
from rapidshot.util.errors import RapidShotDXGIError, RapidShotReinitError  # noqa: E402

DXGI_ERROR_ACCESS_LOST = -2005270490      # 0x887A0026, as comtypes reports it


# --------------------------------------------------------------------------
# logging: a library leaves configuration to the application
# --------------------------------------------------------------------------

def test_importing_rapidshot_configures_nothing():
    """Only a NullHandler: no console output, and no file in the user's home."""
    handlers = logging.getLogger("rapidshot").handlers
    assert handlers and all(isinstance(h, logging.NullHandler) for h in handlers)


@pytest.fixture
def fresh_rapidshot_logger(monkeypatch):
    logger = logging.getLogger("rapidshot")
    saved = logger.handlers[:], logger.level
    logger.handlers = []
    yield logger
    for handler in logger.handlers:
        handler.close()
    logger.handlers, logger.level = saved


def test_setup_logging_is_an_explicit_opt_in(fresh_rapidshot_logger, tmp_path, monkeypatch):
    monkeypatch.setenv(logging_module.LOG_LEVEL_ENV_VAR, "error")
    log_file = tmp_path / "nested" / "rapidshot.log"

    logger = logging_module.setup_logging(log_file=str(log_file), file_level=logging.INFO)

    console, file_handler = logger.handlers
    assert console.level == logging.ERROR, "the environment variable sets the console level"
    assert file_handler.level == logging.INFO
    assert log_file.parent.is_dir()
    assert logging_module.setup_logging() is logger
    assert len(logger.handlers) == 2, "a second call does not add handlers"


@pytest.mark.parametrize("env,expected", [("15", 15), ("nonsense", logging.WARNING)])
def test_numeric_and_unknown_log_levels(fresh_rapidshot_logger, tmp_path, monkeypatch, env, expected):
    monkeypatch.setenv(logging_module.LOG_LEVEL_ENV_VAR, env)
    logger = logging_module.setup_logging(log_file=str(tmp_path / "r.log"))
    assert logger.handlers[0].level == expected


def test_an_unwritable_log_location_keeps_the_console(fresh_rapidshot_logger, tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise OSError("read-only volume")

    monkeypatch.delenv(logging_module.LOG_LEVEL_ENV_VAR, raising=False)
    monkeypatch.setattr(logging_module.os, "makedirs", refuse)
    logger = logging_module.setup_logging(log_file=str(tmp_path / "missing" / "r.log"))
    assert len(logger.handlers) == 1

    fresh_rapidshot_logger.handlers = []
    monkeypatch.setattr(logging_module, "RotatingFileHandler", refuse)
    logger = logging_module.setup_logging(log_file=str(tmp_path / "r.log"))
    assert len(logger.handlers) == 1


def test_get_logger_namespaces_and_caches():
    assert logging_module.get_logger("widget").name == "rapidshot.widget"
    assert logging_module.get_logger("rapidshot.core").name == "rapidshot.core"
    assert logging_module.get_logger("widget") is logging_module.get_logger("widget")


# --------------------------------------------------------------------------
# memory pool
# --------------------------------------------------------------------------

def pool(n=2):
    return NumpyMemoryPool((2, 3, 4), np.uint8, n)


def test_a_held_buffer_survives_its_pool_being_destroyed():
    """A capture rebuild destroys the pool. A frame the caller still holds must
    stay readable and releasable -- it used to raise AttributeError on read and
    RuntimeError on release."""
    p = pool()
    held = p.checkout()
    held.array[:] = 7

    p.destroy_pool()

    assert (np.asarray(held) == 7).all()
    held.release()
    with pytest.raises(BufferReleasedError):
        np.asarray(held)


def test_idle_buffers_are_invalidated_by_a_destroy():
    p = pool(1)
    idle = p._buffers[0]

    p.destroy_pool()

    assert idle.state == "DESTROYED"
    with pytest.raises(BufferReleasedError):
        idle[0]


def test_pool_argument_and_state_errors():
    with pytest.raises(ValueError, match="positive"):
        NumpyMemoryPool((1,), np.uint8, 0)

    raw = BaseMemoryPool((1,), np.uint8, 1)
    with pytest.raises(NotImplementedError):
        raw._create_buffer()
    with pytest.raises(RuntimeError, match="not initialized"):
        raw.checkout()
    assert raw.get_stats() == {"total": 1, "available": 0, "in_use": 0, "initialized": False}
    raw.release_all_buffers()        # warns, does not raise


def test_initialize_twice_is_a_no_op():
    p = pool()
    buffers = list(p._buffers)
    p.initialize_pool()
    assert p._buffers == buffers


def test_a_failed_allocation_leaves_an_empty_pool():
    class Failing(BaseMemoryPool):
        made = 0

        def _create_buffer(self):
            Failing.made += 1
            if Failing.made == 2:
                raise MemoryError("out of memory")
            return np.empty(1)

    p = Failing((1,), np.uint8, 3)
    with pytest.raises(MemoryError):
        p.initialize_pool()
    assert p._buffers == [] and not p._initialized


def test_a_buffer_from_another_pool_is_refused():
    a, b = pool(), pool()
    stranger = b.checkout()
    stranger._pool = a

    with pytest.raises(ValueError, match="not created by this pool"):
        a.checkin(stranger)


def test_release_all_buffers_resets_availability():
    p = pool(2)
    p.checkout()
    p.checkout()
    with pytest.raises(PoolExhaustedError):
        p.checkout()

    p.release_all_buffers()
    assert p.get_stats()["available"] == 2


def test_release_all_never_hands_a_held_buffer_to_a_second_owner():
    """It used to mark held buffers AVAILABLE, so the next checkout returned
    memory a caller was still reading."""
    p = pool(1)
    held = p.checkout()
    held.array[:] = 7

    p.release_all_buffers()
    other = p.checkout()
    other.array[:] = 99

    assert other is not held
    assert (np.asarray(held) == 7).all(), "the holder's frame is untouched"
    held.release()                       # ends the holder's use; the pool is unaffected
    assert p.get_stats()["in_use"] == 1
    other.release()
    assert p.get_stats() == {"total": 1, "available": 1, "in_use": 0, "initialized": True}


def test_a_pooled_gpu_buffer_has_a_repr():
    """CuPy arrays have no .ctypes; repr() raised AttributeError for every
    buffer from CupyMemoryPool, including inside a debugger or a log line."""
    from rapidshot.memory_pool import PooledBuffer

    class DeviceArray:
        data = types.SimpleNamespace(ptr=0xD000)

    assert "data_ptr=0xD000" in repr(PooledBuffer(DeviceArray(), pool()))
    assert "IN_USE" not in repr(PooledBuffer(object(), pool()))


def test_releasing_a_buffer_twice_is_harmless():
    p = pool(1)
    buf = p.checkout()

    buf.release()
    buf.release()

    assert buf.state == "AVAILABLE"
    assert p.get_stats() == {"total": 1, "available": 1, "in_use": 0, "initialized": True}
    assert p.checkout() is buf, "returned to the free list once, not twice"
    with pytest.raises(PoolExhaustedError):
        p.checkout()


def test_releasing_a_detached_buffer_twice_is_harmless():
    p = pool(1)
    buf = p.checkout()
    p.destroy_pool()

    buf.release()
    buf.release()

    assert buf.state == "RELEASED"


def test_pooled_buffer_array_surface():
    p = pool()
    buf = p.checkout()
    buf[0, 0, 0] = 9

    assert buf.array[0, 0, 0] == 9
    assert buf.nbytes == 24
    assert "IN_USE" in repr(buf)


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------

def test_hresults_print_unsigned():
    """comtypes reports them signed; this read -0x7785ffda."""
    error = RapidShotReinitError("access lost", hresult=DXGI_ERROR_ACCESS_LOST)
    assert str(error) == "access lost (HRESULT: 0x887a0026)"


def test_hresult_that_is_not_a_number_is_printed_as_is():
    assert str(RapidShotDXGIError("odd", hresult="E_WHATEVER")) == "odd (HRESULT: E_WHATEVER)"
    assert str(RapidShotDXGIError("plain")) == "plain"


# --------------------------------------------------------------------------
# DXGI factory creation
# --------------------------------------------------------------------------

def test_a_failed_factory_creation_raises_its_hresult(monkeypatch):
    """The return value used to be ignored, so a failure surfaced later as an
    unexplained NULL COM pointer access."""
    import comtypes

    class Dxgi:
        def CreateDXGIFactory1(self, iid, out):
            return -2147467259          # E_FAIL, nothing written

    monkeypatch.setattr(io_module, "_dxgi", Dxgi())

    with pytest.raises(comtypes.COMError, match="0x80004005"):
        io_module._create_dxgi_factory1()


def test_the_dead_factory_wrappers_are_gone():
    import rapidshot._libs.dxgi as dxgi
    for name in ("CreateDXGIFactory1", "CreateDXGIFactory6", "CreateLatestDXGIFactory"):
        assert not hasattr(dxgi, name)


# --------------------------------------------------------------------------
# stage surface
# --------------------------------------------------------------------------

class FakeSurface:
    def __init__(self, log):
        self.log = log

    def Map(self, rect_ref, flags):
        self.log.append("Map")
        rect_ref._obj.Pitch = 16

    def Unmap(self):
        self.log.append("Unmap")


def fake_texture_type(log):
    """An ID3D11Texture2D whose pointer can be queried without a device."""
    from rapidshot._libs.d3d11 import ID3D11Texture2D
    import comtypes

    class Texture(ID3D11Texture2D):
        _iid_ = comtypes.GUID("{7f5b2a4e-0000-4000-8000-0000000000aa}")

        def QueryInterface(self, interface):
            log.append("QueryInterface")
            return FakeSurface(log)

    return Texture


class FakeD3DDevice:
    def __init__(self, log):
        self.log = log
        self.device = self

    def CreateTexture2D(self, desc_ref, initial, texture_ref):
        desc = desc_ref._obj
        self.log.append(("CreateTexture2D", desc.Width, desc.Height, desc.Usage))


class FakeOutputSize:
    surface_size = (64, 48)


@pytest.fixture
def surface_log(monkeypatch):
    log = []
    monkeypatch.setattr(stagesurf_module, "ID3D11Texture2D", fake_texture_type(log))
    return log


def test_each_stage_surface_has_its_own_description(surface_log):
    """A struct default is created once and shared by every instance."""
    a = stagesurf_module.StageSurface(output=FakeOutputSize(), device=FakeD3DDevice(surface_log))
    b = stagesurf_module.StageSurface(output=FakeOutputSize(), device=FakeD3DDevice(surface_log))
    assert a.desc is not b.desc


def test_a_stage_surface_is_a_cpu_readable_staging_texture(surface_log):
    from rapidshot._libs.d3d11 import D3D11_USAGE_STAGING

    surface = stagesurf_module.StageSurface(output=FakeOutputSize(), device=FakeD3DDevice(surface_log))

    assert surface_log[0] == ("CreateTexture2D", 64, 48, D3D11_USAGE_STAGING)
    assert surface.desc.CPUAccessFlags and surface.desc.BindFlags == 0
    assert repr(surface) == "<StageSurface Initialized:True Size:(64, 48) Format:DXGI_FORMAT_B8G8R8A8_UNORM>"


def test_rebuild_with_a_texture_only_resizes_the_record(surface_log):
    device = FakeD3DDevice(surface_log)
    surface = stagesurf_module.StageSurface(output=FakeOutputSize(), device=device)
    surface_log.clear()

    surface.rebuild(FakeOutputSize(), device, dim=(10, 20))

    assert (surface.width, surface.height) == (10, 20)
    assert surface_log == [], "callers release first when the size really changes"


@pytest.mark.parametrize("cached", [True, False])
def test_map_and_unmap_use_the_cached_interface_or_query_one(surface_log, cached):
    surface = stagesurf_module.StageSurface(output=FakeOutputSize(), device=FakeD3DDevice(surface_log))
    if not cached:
        surface.interface = None
    surface_log.clear()

    rect = surface.map()
    surface.unmap()

    assert rect.Pitch == 16
    expected = ["Map", "Unmap"] if cached else ["QueryInterface", "Map", "QueryInterface", "Unmap"]
    assert surface_log == expected


def test_release_only_clears_a_surface_that_exists(surface_log):
    surface = stagesurf_module.StageSurface(output=FakeOutputSize(), device=FakeD3DDevice(surface_log))
    surface.release()
    assert (surface.texture, surface.interface, surface.width) == (None, None, 0)
    surface.release()


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------

class FakeDxgiOutput:
    def __init__(self, width=2560, height=1600, rotation=1, name="\\\\.\\DISPLAY1"):
        self.width, self.height, self.rotation, self.name = width, height, rotation, name
        self.calls = 0

    def GetDesc(self, desc_ref):
        self.calls += 1
        desc = desc_ref._obj
        desc.DeviceName = self.name
        desc.DesktopCoordinates.left, desc.DesktopCoordinates.top = 100, 0
        desc.DesktopCoordinates.right = 100 + self.width
        desc.DesktopCoordinates.bottom = self.height
        desc.AttachedToDesktop = True
        desc.Rotation = self.rotation


@pytest.fixture
def dpi_done(monkeypatch):
    monkeypatch.setattr(output_module, "_dpi_awareness_attempted", True)


@pytest.mark.parametrize("dxgi_rotation,angle,surface", [
    (0, 0, (2560, 1600)),      # DXGI_MODE_ROTATION_UNSPECIFIED
    (1, 0, (2560, 1600)),      # IDENTITY
    (2, 90, (1600, 2560)),
    (3, 180, (2560, 1600)),
    (4, 270, (1600, 2560)),
])
def test_output_geometry(dpi_done, dxgi_rotation, angle, surface):
    out = output_module.Output(FakeDxgiOutput(rotation=dxgi_rotation))

    assert out.resolution == (2560, 1600)
    assert out.rotation_angle == angle
    assert out.surface_size == surface
    assert out.attached_to_desktop is True
    assert out.devicename == "\\\\.\\DISPLAY1"
    assert out.hmonitor is None
    assert repr(out) == f"<Output Name:\\\\.\\DISPLAY1 Resolution:(2560, 1600) Rotation:{angle}>"


def test_update_desc_rereads_the_output(dpi_done):
    dxgi = FakeDxgiOutput()
    out = output_module.Output(dxgi)
    out.desc = None
    dxgi.width = 1920

    out.update_desc()

    assert out.resolution == (1920, 1600)
    assert dxgi.calls == 2


class _Function:
    """A foreign function stand-in: callable, and accepts argtypes/restype.

    A bound method does not, so a fake built from methods sent every call down
    the "shcore is missing" branch and tested nothing it claimed to.
    """

    def __init__(self, body):
        self.body = body
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.body(*args)


class FakeShcore:
    def __init__(self, set_result, current=2):
        self.SetProcessDpiAwareness = _Function(lambda value: set_result)

        def get(process, value_ref):
            value_ref._obj.value = current

        self.GetProcessDpiAwareness = _Function(get)


@pytest.mark.parametrize("set_result,current,warns", [
    (0, 2, False),                 # set successfully
    (-2147024891, 2, False),       # already per-monitor
    (-2147024891, 1, True),        # already set to something else by the host
    (-2147467259, 2, False),       # some other failure: logged, not raised
])
def test_dpi_awareness_is_requested_once_and_reported(monkeypatch, caplog, set_result, current, warns):
    monkeypatch.setattr(output_module, "_dpi_awareness_attempted", False)
    shcore = FakeShcore(set_result, current)
    calls = []
    monkeypatch.setattr(output_module.ctypes, "WinDLL",
                        lambda name: calls.append(name) or shcore)

    with caplog.at_level(logging.DEBUG, logger=output_module.logger.name):
        output_module._ensure_process_dpi_awareness()
        output_module._ensure_process_dpi_awareness()

    assert calls == ["shcore"], "attempted once per process"
    assert shcore.SetProcessDpiAwareness.calls == [(2,)], "asked for per-monitor"
    assert shcore.SetProcessDpiAwareness.argtypes == [ctypes.c_int]
    assert not any("Could not set" in r.getMessage() for r in caplog.records), (
        "the fake reached the shcore-missing branch instead of the one under test")
    assert bool(shcore.GetProcessDpiAwareness.calls) is (set_result == -2147024891)
    warnings_logged = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert bool(warnings_logged) is warns


# --------------------------------------------------------------------------
# desktop access
# --------------------------------------------------------------------------

class FakeUser32:
    def __init__(self, names, input_handle=2):
        self.names = names             # handle -> desktop name (None: unreadable)
        self.input_handle = input_handle
        self.closed = []

    def GetThreadDesktop(self, thread_id):
        return 1

    def OpenInputDesktop(self, flags, inherit, access):
        return self.input_handle

    def CloseDesktop(self, handle):
        self.closed.append(handle.value)

    def GetUserObjectInformationW(self, handle, index, buf, size, needed_ref):
        name = self.names.get(handle)
        if name is None:
            needed_ref._obj.value = 0
            return 0
        needed_ref._obj.value = (len(name) + 1) * 2
        if buf is None:
            return 0
        buf.value = name
        return 1


@pytest.fixture
def desktops(monkeypatch):
    def install(user32):
        monkeypatch.setattr(desktop_module, "_user32", user32)
        monkeypatch.setattr(desktop_module, "_kernel32",
                            types.SimpleNamespace(GetCurrentThreadId=lambda: 42))
        return user32
    return install


def test_the_normal_case_is_the_input_desktop(desktops):
    user32 = desktops(FakeUser32({1: "Default", 2: "Default"}))

    state = desktop_module.describe_desktop_access()

    assert (state.thread_desktop, state.input_desktop, state.is_input_desktop) == ("Default", "Default", True)
    assert state.blocked_reason is None
    assert user32.closed == [2], "the input desktop handle is closed"
    assert repr(state) == "<DesktopState thread='Default' input='Default' is_input=True>"


def test_a_locked_workstation_is_named(desktops):
    desktops(FakeUser32({1: "Default"}, input_handle=0))

    state = desktop_module.describe_desktop_access()

    assert state.is_input_desktop is False
    assert "secure desktop is active" in state.blocked_reason


def test_a_different_desktop_is_named(desktops):
    desktops(FakeUser32({1: "Service-0x0", 2: "Default"}))

    state = desktop_module.describe_desktop_access()

    assert "'Service-0x0'" in state.blocked_reason and "'Default'" in state.blocked_reason


def test_an_unreadable_name_is_unknown_not_yes(desktops):
    """Guessing 'this is the input desktop' is the failure the module exists to avoid."""
    desktops(FakeUser32({2: "Default"}))

    state = desktop_module.describe_desktop_access()

    assert state.is_input_desktop is None and state.blocked_reason is None


def test_a_name_whose_second_read_fails_is_unknown(desktops):
    class Flaky(FakeUser32):
        def GetUserObjectInformationW(self, handle, index, buf, size, needed_ref):
            if buf is None:
                needed_ref._obj.value = 16
                return 0
            return 0

    desktops(Flaky({}))
    assert desktop_module._desktop_name(1) is None
    assert desktop_module._desktop_name(0) is None


# --------------------------------------------------------------------------
# processor wrapper
# --------------------------------------------------------------------------

def test_the_wrapper_reports_its_backend_and_sizes():
    proc = Processor(output_color="RGBA")

    assert proc.active_backend is ProcessorBackends.NUMPY
    assert proc.bytes_required(10, 5) == 200


def test_a_gpu_request_without_cupy_falls_back_to_numpy(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)

    proc = Processor(output_color="RGB", nvidia_gpu=True)

    assert proc.active_backend is ProcessorBackends.NUMPY
    assert type(proc.backend).__name__ == "NumpyProcessor"


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="Unknown backend"):
        Processor(backend="PIL")


def test_process2_needs_a_backend_that_can_write_directly():
    proc = Processor(output_color="RGB")
    proc.backend = object()
    with pytest.raises(NotImplementedError):
        proc.process2(np.zeros(12, np.uint8), None, 2, 2)


def test_old_dependency_versions_are_warned_about(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(__version__="4.1.0"))
    real_numpy = sys.modules["numpy"]
    proc = Processor(output_color="RGB")
    monkeypatch.setitem(sys.modules, "numpy", types.SimpleNamespace(__version__="1.19.5"))
    try:
        with caplog.at_level(logging.WARNING, logger=base_module.logger.name):
            proc._check_dependencies()
    finally:
        sys.modules["numpy"] = real_numpy

    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "NumPy 1.19.5" in messages and "OpenCV 4.1.0" in messages


def test_the_cupy_install_hint_names_this_packages_extras():
    from rapidshot.processor.cupy_processor import CupyProcessor

    hint = CupyProcessor.__new__(CupyProcessor)._get_platform_specific_cupy_install()

    assert "rapidshot[gpu_cuda13]" in hint and "nvidia-smi" in hint
    assert "cuda10x" not in hint


# --------------------------------------------------------------------------
# topology probe
# --------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, name="GPU", desc_error=None, outputs_error=None, software=False):
        self.name, self.desc_error, self.outputs_error = name, desc_error, outputs_error
        self.software = software

    def GetDesc1(self, desc_ref):
        if self.desc_error:
            raise self.desc_error
        desc = desc_ref._obj
        desc.Description = self.name
        desc.VendorId = 0x10DE
        desc.Flags = 2 if self.software else 0


def test_probe_reports_unreachable_dxgi_as_no_adapters(monkeypatch):
    def fail():
        raise OSError("dxgi.dll missing")

    monkeypatch.setattr(io_module, "enum_dxgi_adapters", fail)
    assert topology_module.probe_topology().adapters == ()


def test_probe_keeps_adapters_it_cannot_fully_read(monkeypatch):
    adapters = [FakeAdapter(desc_error=OSError("GetDesc1 failed")),
                FakeAdapter("NVIDIA", outputs_error=OSError("EnumOutputs failed")),
                FakeAdapter("Microsoft Basic Render Driver", software=True)]
    monkeypatch.setattr(io_module, "enum_dxgi_adapters", lambda: adapters)

    def outputs(adapter):
        if adapter.outputs_error:
            raise adapter.outputs_error
        return []

    monkeypatch.setattr(io_module, "enum_dxgi_outputs", outputs)

    topology = topology_module.probe_topology()

    unreadable, nvidia, warp = topology.adapters
    assert unreadable.description == "<unreadable>" and "GetDesc1" in unreadable.error
    assert nvidia.output_count == 0 and "EnumOutputs" in nvidia.error
    assert warp.is_software
    assert "unusable: EnumOutputs failed" in str(nvidia)


# --------------------------------------------------------------------------
# profiler
# --------------------------------------------------------------------------

class Observed:
    def __init__(self, accumulated=1, changed=0.25, generation=0):
        self.accumulated_frames = accumulated
        self.changed_fraction = changed
        self.generation = generation


def test_report_covers_every_section():
    profiler = Profiler("demo")
    profiler.observe(Observed(accumulated=3, changed=0.5, generation=0))
    profiler.observe(Observed(generation=1))
    profiler.observe(None)
    profiler.record("grab", 2.0)
    profiler.record("empty", 1.0)
    profiler._stages["empty"].samples_ms.clear()
    profiler.stop()

    text = profiler.report()

    assert "updates missed    : 2" in text
    assert "changed fraction  : median" in text
    assert "RECOVERIES        : 1 during this run (generations [0, 1])" in text
    assert "LOW CONFIDENCE: grab" in text
    assert "empty" not in text.split("-" * 10)[1], "an empty stage is left out of the table"
    assert profiler._stages["empty"].stats() == {"count": 0}
    assert repr(profiler) == "<Profiler 'demo': 2 frames, 2 stages>"


def test_report_without_stages_says_so():
    assert Profiler("idle").report().endswith("no stages timed")


def test_enough_samples_are_not_low_confidence():
    profiler = Profiler()
    for i in range(RELIABLE_SAMPLES):
        profiler.record("grab", float(i))
    assert "LOW CONFIDENCE" not in profiler.report()


# --------------------------------------------------------------------------
# ctypes helpers
# --------------------------------------------------------------------------

def test_destination_description_for_each_buffer_kind():
    array = np.zeros((2, 3), np.uint8)
    assert describe_destination(array) == (array.ctypes.data, 6)

    c_array = (ctypes.c_ubyte * 8)()
    assert describe_destination(c_array) == (ctypes.addressof(c_array), 8)

    address, size = describe_destination(bytearray(5))
    assert address and size == 5

    assert describe_destination(None) == (None, None)
    assert describe_destination(1234) == (1234, None)


@pytest.mark.parametrize("make,message", [
    (lambda: np.zeros((4, 4), np.uint8)[:, ::2], "C-contiguous"),
    (lambda: np.frombuffer(bytes(4), np.uint8), "read-only"),
    (lambda: memoryview(bytes(4)), "read-only"),
])
def test_destinations_that_cannot_be_written_are_refused(make, message):
    with pytest.raises(ValueError, match=message):
        describe_destination(make())


def test_pointer_to_address_for_each_pointer_kind():
    value = ctypes.c_int(5)
    assert pointer_to_address(ctypes.c_void_p(99)) == 99
    assert pointer_to_address(ctypes.pointer(value)) == ctypes.addressof(value)
    assert pointer_to_address(types.SimpleNamespace(value=77)) == 77
    assert pointer_to_address(object()) is None
    assert pointer_to_address(ctypes.POINTER(ctypes.c_int)()) is None


# --------------------------------------------------------------------------
# timer
# --------------------------------------------------------------------------

@pytest.fixture
def kernel32(monkeypatch):
    dll = timer_module.__dict__["__kernel32"]

    def patch(name, value):
        monkeypatch.setattr(dll, name, value)

    monkeypatch.setattr(timer_module.ctypes, "get_last_error", lambda: 5)
    return patch


def test_timer_creation_failure_raises_the_windows_error(kernel32):
    kernel32("CreateWaitableTimerExW", lambda *a: None)
    with pytest.raises(OSError):
        timer_module.create_high_resolution_timer()


def test_timer_arming_failure_raises(kernel32):
    kernel32("SetWaitableTimer", lambda *a: 0)
    with pytest.raises(OSError):
        timer_module.set_periodic_timer(1, 16)


def test_closing_a_timer(kernel32):
    kernel32("CloseHandle", lambda handle: 0)
    with pytest.raises(OSError):
        timer_module.close_timer(1)
    assert timer_module.close_timer(0) is True, "no handle, nothing to close"


# --------------------------------------------------------------------------
# tensor stream
# --------------------------------------------------------------------------

def test_tensor_stream_repr_names_its_state():
    from rapidshot.tensor_stream import TensorStream

    stream = TensorStream(object(), (640, 360))
    assert repr(stream) == "<TensorStream 640x360 0 frames>"
    stream._closed = True
    assert repr(stream) == "<TensorStream 640x360 closed>"
