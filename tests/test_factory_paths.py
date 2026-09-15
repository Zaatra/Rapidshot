"""rapidshot/__init__.py: the factory, the module-level API, and diagnostics.

Line coverage put this at 54%. The live tests construct one real factory on a
healthy single-display machine, so every branch that decides what a user sees
when something is wrong -- no adapters, adapters that will not open, an index
out of range, a metadata gap, a diagnostic section that fails -- had never run.
Those are the messages people paste into issues.
"""
import weakref

import pytest

import rapidshot
import rapidshot.util.topology as topology_module


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeTopology:
    def __init__(self, kind="single", help_text="", hybrid=False, adapters=()):
        self.kind = kind
        self._help = help_text
        self.is_hybrid = hybrid
        self.adapters = list(adapters)

    def help_text(self):
        return self._help

    def describe(self):
        return f"topology: {self.kind}"


class FakeDevice:
    def __init__(self, description="GPU", outputs=("DISPLAY1",), fail=None):
        if fail is not None:
            raise fail
        self.desc = type("Desc", (), {"Description": description})()
        self._outputs = list(outputs)

    def enum_outputs(self):
        return self._outputs

    def __repr__(self):
        return f"<FakeDevice {self.desc.Description}>"


class FakeOutput:
    def __init__(self, name="DISPLAY1", resolution=(1920, 1080), rotation=0):
        self.devicename = name
        self.resolution = resolution
        self.rotation_angle = rotation
        self.update_desc_calls = 0

    def update_desc(self):
        self.update_desc_calls += 1


class FakeCapture:
    error = None

    def __init__(self, **kwargs):
        if FakeCapture.error is not None:
            raise FakeCapture.error
        self.kwargs = kwargs
        self.released = False
        self.release_error = None

    def release(self):
        self.released = True
        if self.release_error is not None:
            raise self.release_error


def factory_over(monkeypatch, devices, outputs, metadata=None, topology=None):
    """A factory with its state set directly, as __init__ would leave it."""
    factory = object.__new__(rapidshot.RapidshotFactory)
    factory.devices = list(devices)
    factory.outputs = [list(o) for o in outputs]
    factory.all_devices = list(devices)
    factory.device_failures = []
    factory.topology = topology or FakeTopology()
    factory.output_metadata = (
        metadata if metadata is not None
        else {o.devicename: (None, i == 0)
              for group in outputs for i, o in enumerate(group)})
    monkeypatch.setattr(rapidshot.RapidshotFactory, "_screencapture_instances",
                        weakref.WeakValueDictionary())
    # reset() clears the singleton registry; never let it clear the real one.
    monkeypatch.setattr(rapidshot.Singleton, "_instances", {})
    FakeCapture.error = None
    monkeypatch.setattr(rapidshot, "ScreenCapture", FakeCapture)
    monkeypatch.setattr(rapidshot.time, "sleep", lambda _s: None)
    return factory


def one_display(monkeypatch, **kwargs):
    device = FakeDevice()
    return factory_over(monkeypatch, [device], [[FakeOutput()]], **kwargs), device


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

@pytest.fixture
def enumeration(monkeypatch):
    """Patches what RapidshotFactory.__init__ enumerates. Returns a config dict."""
    config = {
        "topology": FakeTopology(),
        "adapters": ["adapter0"],
        "devices": {"adapter0": {"description": "GPU", "outputs": ["DISPLAY1"]}},
        "metadata": {"DISPLAY1": (None, True)},
    }

    def make_device(adapter):
        spec = config["devices"][adapter]
        return FakeDevice(spec.get("description", "GPU"), spec.get("outputs", ()),
                          spec.get("fail"))

    monkeypatch.setattr(rapidshot, "probe_topology", lambda: config["topology"])
    monkeypatch.setattr(rapidshot, "enum_dxgi_adapters", lambda: list(config["adapters"]))
    monkeypatch.setattr(rapidshot, "Device", make_device)
    monkeypatch.setattr(rapidshot, "Output", lambda p: FakeOutput(p))
    monkeypatch.setattr(rapidshot, "get_output_metadata", lambda: config["metadata"])
    monkeypatch.setattr(rapidshot, "_core_import_error", None)
    monkeypatch.setattr(rapidshot, "_io_import_error", None)
    return config


def construct():
    """Run __init__ directly, bypassing the process-wide singleton."""
    factory = object.__new__(rapidshot.RapidshotFactory)
    rapidshot.RapidshotFactory.__init__(factory)
    return factory


def test_construction_keeps_adapters_without_outputs_as_candidates(enumeration):
    enumeration["adapters"] = ["dgpu", "igpu"]
    enumeration["devices"] = {
        "dgpu": {"description": "NVIDIA", "outputs": ["DISPLAY1"]},
        "igpu": {"description": "Intel", "outputs": []},
    }

    factory = construct()

    assert [d.desc.Description for d in factory.devices] == ["NVIDIA"]
    assert [d.desc.Description for d in factory.all_devices] == ["NVIDIA", "Intel"]
    assert [o.devicename for o in factory.outputs[0]] == ["DISPLAY1"]


def test_no_adapters_is_a_headless_error_carrying_the_topology(enumeration):
    enumeration["adapters"] = []
    enumeration["topology"] = FakeTopology(kind="headless", help_text="No display attached.")

    with pytest.raises(rapidshot.HeadlessError, match="No display attached") as excinfo:
        construct()

    assert excinfo.value.topology is enumeration["topology"]


def test_adapters_that_will_not_open_are_named(enumeration):
    """Outputs exist, so this is not headless; saying so would send the user
    looking for a cable problem."""
    enumeration["devices"] = {"adapter0": {"fail": OSError("E_OUTOFMEMORY")}}

    with pytest.raises(rapidshot.HeadlessError) as excinfo:
        construct()

    message = str(excinfo.value)
    assert "none could be opened as a Direct3D 11 device" in message
    assert "E_OUTOFMEMORY" in message


def test_one_failing_adapter_does_not_stop_the_others(enumeration):
    enumeration["adapters"] = ["bad", "good"]
    enumeration["devices"] = {
        "bad": {"fail": OSError("driver crashed")},
        "good": {"description": "Working", "outputs": ["DISPLAY1"]},
    }

    factory = construct()

    assert [d.desc.Description for d in factory.devices] == ["Working"]
    assert factory.device_failures == ["driver crashed"]


def test_headless_help_is_kept_and_failures_appended(enumeration):
    enumeration["topology"] = FakeTopology(kind="headless", help_text="Headless machine.")
    enumeration["devices"] = {"adapter0": {"fail": OSError("refused")}}

    with pytest.raises(rapidshot.HeadlessError) as excinfo:
        construct()

    assert str(excinfo.value).startswith("Headless machine.")
    assert "Device creation errors:\n  refused" in str(excinfo.value)


def test_an_unexpected_failure_is_wrapped_with_its_cause(enumeration, monkeypatch):
    def broken():
        raise KeyError("registry")

    monkeypatch.setattr(rapidshot, "get_output_metadata", broken)

    with pytest.raises(rapidshot.RapidshotError, match="Failed to initialize") as excinfo:
        construct()
    assert isinstance(excinfo.value.__cause__, KeyError)


def test_construction_refuses_where_dxgi_cannot_be_imported(enumeration, monkeypatch):
    monkeypatch.setattr(rapidshot, "_core_import_error", ImportError("no comtypes"))

    with pytest.raises(rapidshot.RapidshotError, match="not available on this platform"):
        construct()


def test_a_hybrid_system_still_constructs(enumeration):
    enumeration["topology"] = FakeTopology(kind="hybrid", hybrid=True)
    assert construct().topology.is_hybrid


# --------------------------------------------------------------------------
# create()
# --------------------------------------------------------------------------

def test_create_refuses_where_capture_is_unavailable(monkeypatch):
    factory, _ = one_display(monkeypatch)
    monkeypatch.setattr(rapidshot, "ScreenCapture", None)

    with pytest.raises(rapidshot.RapidshotError, match="not available on this platform"):
        factory.create()


@pytest.mark.parametrize("kwargs,error,message", [
    ({"device_idx": 1}, rapidshot.DeviceError, "Invalid device index: 1"),
    ({"device_idx": -1}, rapidshot.DeviceError, "Invalid device index: -1"),
    ({"output_idx": 1}, rapidshot.OutputError, "Invalid output index: 1"),
    ({"output_idx": -1}, rapidshot.OutputError, "Invalid output index: -1"),
    ({"output_color": "YUV"}, rapidshot.ConfigurationError, "Invalid color format"),
])
def test_create_rejects_bad_arguments(monkeypatch, kwargs, error, message):
    """Negative indices used to be accepted: Python indexing made -1 the last
    device, under a cache key different from its positive index -- so one
    output could end up with two cameras competing for its duplication."""
    factory, _ = one_display(monkeypatch)

    with pytest.raises(error, match=message):
        factory.create(**kwargs)


def test_create_picks_the_primary_output(monkeypatch):
    device = FakeDevice()
    outputs = [FakeOutput("DISPLAY1"), FakeOutput("DISPLAY2")]
    factory = factory_over(monkeypatch, [device], [outputs],
                           metadata={"DISPLAY1": (None, False), "DISPLAY2": (None, True)})

    camera = factory.create()

    assert camera.kwargs["output"] is outputs[1]
    assert outputs[1].update_desc_calls == 1


def test_create_without_a_primary_uses_the_first_output(monkeypatch):
    device = FakeDevice()
    outputs = [FakeOutput("DISPLAY1"), FakeOutput("DISPLAY2")]
    factory = factory_over(monkeypatch, [device], [outputs], metadata={})

    assert factory.create().kwargs["output"] is outputs[0]


def test_prefer_integrated_selects_an_output_owning_igpu(monkeypatch):
    dgpu, igpu = FakeDevice("NVIDIA"), FakeDevice("Intel(R) UHD")
    factory = factory_over(monkeypatch, [dgpu, igpu],
                           [[FakeOutput("DISPLAY1")], [FakeOutput("DISPLAY2")]],
                           metadata={"DISPLAY1": (None, True), "DISPLAY2": (None, True)})

    camera = factory.create(prefer_integrated=True)

    assert camera.kwargs["device"] is igpu
    assert camera.kwargs["candidate_devices"][0] is igpu


def test_candidate_order_is_stable_and_complete(monkeypatch):
    a, b, c = FakeDevice("AMD"), FakeDevice("Intel Arc"), FakeDevice("NVIDIA")
    factory = factory_over(monkeypatch, [a], [[FakeOutput()]])
    factory.all_devices = [a, b, c]

    assert factory._duplication_candidates(c, False) == [c, a, b]
    assert factory._duplication_candidates(c, True) == [b, c, a]
    assert rapidshot.RapidshotFactory._is_integrated(object()) is False


def test_create_names_every_setting_that_differs(monkeypatch):
    factory, _ = one_display(monkeypatch)
    camera = factory.create(output_color="RGB", timeout_ms=10)

    with pytest.raises(rapidshot.ConfigurationError) as excinfo:
        factory.create(output_color="BGRA", timeout_ms=0)

    message = str(excinfo.value)
    assert "output_color='RGB' (asked for 'BGRA')" in message
    assert "timeout_ms=10 (asked for 0)" in message
    assert camera.released is False


def test_a_failing_capture_is_wrapped_with_its_cause(monkeypatch):
    factory, _ = one_display(monkeypatch)
    FakeCapture.error = ValueError("region out of bounds")

    with pytest.raises(rapidshot.RapidshotError, match="region out of bounds") as excinfo:
        factory.create()
    assert isinstance(excinfo.value.__cause__, ValueError)


# --------------------------------------------------------------------------
# info, clean-up, reset
# --------------------------------------------------------------------------

def test_device_and_topology_info(monkeypatch):
    factory, _ = one_display(monkeypatch, topology=FakeTopology(kind="hybrid"))

    assert factory.device_info() == "Device[0]:<FakeDevice GPU>\ntopology: hybrid\n"
    assert factory.topology_info() == "topology: hybrid"


def test_output_info_lists_every_output(monkeypatch):
    device = FakeDevice()
    outputs = [FakeOutput("DISPLAY1", (2560, 1600)), FakeOutput("DISPLAY2", (1080, 1920), 90)]
    factory = factory_over(monkeypatch, [device], [outputs])

    assert factory.output_info() == (
        "Device[0] Output[0]: Resolution:(2560, 1600) Rotation:0 Primary:True\n"
        "Device[0] Output[1]: Resolution:(1080, 1920) Rotation:90 Primary:False\n")


def test_output_info_survives_an_output_missing_from_the_metadata(monkeypatch):
    """create() already tolerates this gap -- a display attached after the
    metadata was read. output_info() raised TypeError on None[1] instead."""
    device = FakeDevice()
    factory = factory_over(monkeypatch, [device], [[FakeOutput("DISPLAY9")]], metadata={})

    assert factory.output_info() == (
        "Device[0] Output[0]: Resolution:(1920, 1080) Rotation:0 Primary:unknown\n")


def test_clean_up_releases_every_camera_and_tolerates_failures(monkeypatch):
    factory, _ = one_display(monkeypatch)
    factory.outputs[0].append(FakeOutput("DISPLAY2"))
    first = factory.create(output_idx=0)
    second = factory.create(output_idx=1)
    first.release_error = RuntimeError("already gone")

    factory.clean_up()

    assert first.released and second.released


def test_reset_forgets_the_singleton(monkeypatch):
    factory, _ = one_display(monkeypatch)
    rapidshot.Singleton._instances[rapidshot.RapidshotFactory] = factory
    camera = factory.create()

    factory.reset()

    assert camera.released
    assert rapidshot.RapidshotFactory not in rapidshot.Singleton._instances
    assert len(factory._screencapture_instances) == 0


# --------------------------------------------------------------------------
# module-level API
# --------------------------------------------------------------------------

@pytest.fixture
def no_global_factory():
    """The module-level factory, emptied for the test and put back after."""
    name = "__factory"
    original = rapidshot.__dict__[name]
    rapidshot.__dict__[name] = None
    yield name
    rapidshot.__dict__[name] = original


def test_a_failed_factory_is_not_cached(monkeypatch, no_global_factory):
    attempts = []

    def failing():
        attempts.append(1)
        raise rapidshot.HeadlessError("no display")

    monkeypatch.setattr(rapidshot, "RapidshotFactory", failing)

    for _ in range(2):
        with pytest.raises(rapidshot.HeadlessError):
            rapidshot.get_factory()
    assert len(attempts) == 2, "a machine that gains a display must be able to retry"


def test_module_functions_delegate_to_the_factory(monkeypatch, no_global_factory):
    factory, _ = one_display(monkeypatch)
    monkeypatch.setattr(rapidshot, "RapidshotFactory", lambda: factory)

    camera = rapidshot.create(output_color="BGR")
    assert camera.kwargs["output_color"] == "BGR"
    assert rapidshot.device_info().startswith("Device[0]")
    assert rapidshot.output_info().startswith("Device[0] Output[0]")
    assert rapidshot.topology_info() == "topology: single"

    rapidshot.clean_up()
    assert camera.released

    rapidshot.reset()
    assert rapidshot.__dict__[no_global_factory] is None


def test_topology_info_without_a_factory_probes_directly(monkeypatch, no_global_factory):
    monkeypatch.setattr(rapidshot, "probe_topology", lambda: FakeTopology(kind="headless"))
    assert rapidshot.topology_info() == "topology: headless"


def test_clean_up_and_reset_without_a_factory_are_no_ops(no_global_factory):
    rapidshot.clean_up()
    rapidshot.reset()


# --------------------------------------------------------------------------
# capabilities / diagnose / get_version_info
# --------------------------------------------------------------------------

class FakeAdapter:
    def __init__(self, description, vendor, outputs, vram_mb=0, software=False):
        self.description = description
        self.vendor = vendor
        self.output_count = outputs
        self.dedicated_video_memory = vram_mb * 1024 * 1024
        self.is_software = software


class FakeNative:
    def __init__(self, available=True, fail=()):
        self.available = available
        self.fail = set(fail)
        self.cross_adapter_calls = 0

    def is_available(self):
        return self.available

    def build_info(self):
        return {"source": "wheel", "version": "0.1.0", "stage": "release"}

    def _maybe_fail(self, name, value):
        if name in self.fail:
            raise RuntimeError(f"{name} blew up")
        return value

    def probe_shareable_buffers(self):
        return self._maybe_fail("shareable", {"ok": True})

    def probe_onnxruntime(self):
        return self._maybe_fail("onnxruntime", {"ok": True})

    def probe_cross_adapter(self):
        self.cross_adapter_calls += 1
        return self._maybe_fail("cross", {"source": "Intel", "destination": "NVIDIA",
                                          "representative": True})


@pytest.fixture
def report_env(monkeypatch):
    native = FakeNative()
    topology = FakeTopology(kind="hybrid", adapters=[
        FakeAdapter("Intel UHD", "Intel", 1, vram_mb=128),
        FakeAdapter("NVIDIA RTX", "NVIDIA", 0, vram_mb=8192),
    ])
    monkeypatch.setattr(rapidshot, "native", native, raising=False)
    monkeypatch.setattr(topology_module, "probe_topology", lambda: topology)
    return native, topology


def test_capabilities_reports_every_section(report_env):
    native, _ = report_env

    report = rapidshot.capabilities()

    assert report["rapidshot"]["native_extension"] is True
    assert report["capture"]["kind"] == "hybrid"
    assert report["capture"]["display_adapters"] == 1
    assert report["capture"]["adapters"][1]["dedicated_vram_mb"] == 8192
    assert report["capture"]["cross_adapter_required_for_gpu_tensor"] is True
    assert report["gpu"]["shareable_buffers"] == {"ok": True}
    assert report["gpu"]["cross_adapter"] == "not probed (pass probe_gpu=True)"
    assert native.cross_adapter_calls == 0, "GPU work only when asked for"
    assert set(report["dependencies"]) == {"numpy", "comtypes", "cupy", "cv2", "PIL", "onnxruntime"}


def test_capabilities_never_raises(report_env, monkeypatch):
    """The machine where this is run is the one where something is broken."""
    native, _ = report_env
    native.fail = {"shareable", "cross"}

    def no_topology():
        raise OSError("DXGI unavailable")

    monkeypatch.setattr(topology_module, "probe_topology", no_topology)

    report = rapidshot.capabilities(probe_gpu=True)

    assert report["capture"] == {"error": "OSError: DXGI unavailable"}
    assert report["gpu"]["shareable_buffers"] == {"error": "RuntimeError: shareable blew up"}
    assert report["gpu"]["onnxruntime"] == {"ok": True}
    assert report["gpu"]["cross_adapter"] == {"error": "RuntimeError: cross blew up"}


def test_capabilities_survives_a_native_module_that_explodes(report_env, monkeypatch):
    class Exploding:
        def is_available(self):
            raise ImportError("DLL load failed")

    monkeypatch.setattr(rapidshot, "native", Exploding(), raising=False)

    report = rapidshot.capabilities()

    assert report["rapidshot"]["native_extension"] is False
    assert "DLL load failed" in report["rapidshot"]["error"]
    assert "DLL load failed" in report["gpu"]["error"]


def test_diagnose_renders_a_hybrid_machine_with_a_probe(report_env):
    text = rapidshot.diagnose(probe_gpu=True)

    assert "native extension : yes (wheel)" in text
    assert "topology  : hybrid (2 adapters, 1 driving a display)" in text
    assert "- Intel UHD (Intel, 1 output)" in text
    assert "- NVIDIA RTX (NVIDIA, 0 outputs)" in text
    assert "NOTE: hybrid system" in text
    assert "cross-adapter    : Intel -> NVIDIA (representative=True)" in text


def test_diagnose_without_native_or_topology(report_env, monkeypatch):
    native, _ = report_env
    native.available = False

    def no_topology():
        raise OSError("no DXGI")

    monkeypatch.setattr(topology_module, "probe_topology", no_topology)

    text = rapidshot.diagnose()

    assert "native extension : NO" in text
    assert "pip install rapidshot-native" in text
    assert "topology unavailable: OSError: no DXGI" in text
    assert "cross-adapter    :" not in text


def test_version_info_shape():
    info = rapidshot.get_version_info()

    assert info["rapidshot"]["version"] == rapidshot.__version__
    assert set(info["dependencies"]) >= {"numpy", "cupy", "opencv", "comtypes"}
    assert info["dependencies"]["numpy"] != "not installed"
