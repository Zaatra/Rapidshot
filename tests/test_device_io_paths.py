"""core/device.py and util/io.py: device creation and DXGI enumeration.

The live suite creates devices on adapters that work, so every failure branch of
Device creation, and every enumeration error that is not "no more items", had
never run. These drive both through fakes of D3D11CreateDevice and the DXGI
interfaces; no device is created.
"""
import ctypes
import logging

import pytest

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.core.device as device_module  # noqa: E402
import rapidshot.util.io as io_module  # noqa: E402
from rapidshot._libs.d3d11 import (  # noqa: E402
    D3D11_CREATE_DEVICE_BGRA_SUPPORT,
    D3D11_CREATE_DEVICE_DEBUG,
    D3D_DRIVER_TYPE_UNKNOWN,
)
from rapidshot._libs.dxgi import DXGI_ERROR_NOT_FOUND, E_INVALIDARG  # noqa: E402
from rapidshot.core.device import Device  # noqa: E402

E_FAIL = -2147467259
S_OK = 0


def com_error(hresult):
    return comtypes.COMError(hresult, "injected", (None, None, None, 0, None))


class FakeAdapter:
    def __init__(self, description="Fake Adapter", outputs=2, enum_error=None,
                 null=False):
        self.description = description
        self.outputs = outputs
        self.enum_error = enum_error
        self.null = null

    def __bool__(self):
        return not self.null

    def GetDesc1(self, desc_ref):
        desc = desc_ref._obj
        desc.Description = self.description
        desc.VendorId = 0x8086
        desc.DedicatedVideoMemory = 256 * 1048576

    def EnumOutputs(self, index, out_ref):
        if index < self.outputs:
            return
        raise com_error(self.enum_error if self.enum_error is not None
                        else DXGI_ERROR_NOT_FOUND)


class CreateDeviceRecorder:
    """Stands in for D3D11CreateDevice; fails every attempt by default."""

    def __init__(self, result=E_FAIL):
        self.calls = []
        self.levels = []
        self.result = result

    def __call__(self, adapter, driver_type, software, flags, levels, n_levels,
                 sdk, device_ref, level_ref, context_ref):
        self.calls.append((adapter, driver_type, flags))
        self.levels.append([levels[i] for i in range(n_levels)])
        return self.result(flags, self.levels[-1]) if callable(self.result) else self.result


@pytest.fixture
def create_device(monkeypatch):
    recorder = CreateDeviceRecorder()
    monkeypatch.setattr(device_module, "_D3D11CreateDevice", recorder)
    return recorder


# --------------------------------------------------------------------------
# device creation
# --------------------------------------------------------------------------

def test_the_first_attempt_is_this_adapter_with_bgra_support(create_device):
    with pytest.raises(RuntimeError):
        Device(FakeAdapter())

    adapter, driver_type, flags = create_device.calls[0]
    assert isinstance(adapter, FakeAdapter)
    assert driver_type == D3D_DRIVER_TYPE_UNKNOWN
    assert flags & D3D11_CREATE_DEVICE_BGRA_SUPPORT, (
        "duplicated surfaces are BGRA; a device without BGRA support cannot "
        "share them")


def test_every_failure_is_reported_with_the_last_error(create_device):
    with pytest.raises(RuntimeError, match="on Fake Adapter") as excinfo:
        Device(FakeAdapter())

    # Unsigned, the form HRESULTs are looked up by. This read -0x7fffbffb.
    assert "failed with code 0x80004005" in str(excinfo.value)


def test_a_success_code_with_null_pointers_is_not_a_device(create_device):
    """S_OK with nothing written is not a device: each null result moves on to
    the next attempt rather than being wrapped as an interface."""
    create_device.result = S_OK

    with pytest.raises(RuntimeError, match="returned no device"):
        Device(FakeAdapter())
    assert len(create_device.calls) > 1, "each null result moves on to the next attempt"


def test_every_attempt_is_made_on_this_adapter(create_device):
    """A Device describes one adapter. Creation used to fall back to the default
    adapter and then to WARP, REFERENCE and SOFTWARE, so a device living
    somewhere else carried this adapter's name."""
    with pytest.raises(RuntimeError):
        Device(FakeAdapter())

    assert all(isinstance(adapter, FakeAdapter) and driver == D3D_DRIVER_TYPE_UNKNOWN
               for adapter, driver, _ in create_device.calls), create_device.calls


def test_a_runtime_without_11_1_is_retried_without_it(create_device):
    """A runtime predating feature level 11.1 rejects the whole list with
    E_INVALIDARG instead of skipping the level it does not know."""
    def refuse_11_1(flags, levels):
        return E_INVALIDARG if 0xB100 in levels else S_OK

    create_device.result = refuse_11_1

    with pytest.raises(RuntimeError):     # S_OK here writes no device
        Device(FakeAdapter())

    assert 0xB100 in create_device.levels[0]
    assert any(0xB100 not in levels and 0xB000 in levels
               for levels in create_device.levels)


def test_an_adapter_that_will_not_open_is_named_by_the_factory(create_device, monkeypatch):
    """The replacement for the fallbacks: the factory records the failure and
    its error says which adapter and why."""
    import rapidshot

    class Topology:
        kind, is_hybrid = "single", False

        def help_text(self):
            return ""

    monkeypatch.setattr(rapidshot, "probe_topology", Topology)
    monkeypatch.setattr(rapidshot, "enum_dxgi_adapters", lambda: [FakeAdapter("Broken GPU")])
    monkeypatch.setattr(rapidshot, "Device", Device)
    factory = rapidshot.RapidshotFactory.__new__(rapidshot.RapidshotFactory)

    with pytest.raises(rapidshot.HeadlessError) as excinfo:
        rapidshot.RapidshotFactory.__init__(factory)

    message = str(excinfo.value)
    assert "Failed to create a D3D11 device on Broken GPU" in message
    assert "0x80004005" in message


def test_the_debug_layer_is_tried_only_at_debug_verbosity(create_device, monkeypatch):
    logger = logging.getLogger("rapidshot.core.device")
    monkeypatch.setattr(logger, "level", logging.DEBUG)
    with pytest.raises(RuntimeError):
        Device(FakeAdapter())
    assert any(flags & D3D11_CREATE_DEVICE_DEBUG for _, _, flags in create_device.calls)

    create_device.calls.clear()
    monkeypatch.setattr(logger, "level", logging.WARNING)
    with pytest.raises(RuntimeError):
        Device(FakeAdapter())
    assert not any(flags & D3D11_CREATE_DEVICE_DEBUG for _, _, flags in create_device.calls)


# --------------------------------------------------------------------------
# adapter description and outputs
# --------------------------------------------------------------------------

def bare_device(adapter=None, desc=True):
    device = Device.__new__(Device)
    device.adapter = adapter if adapter is not None else FakeAdapter()
    device.desc = None
    if desc:
        device.desc = device_module.DXGI_ADAPTER_DESC1()
        device.adapter.GetDesc1(ctypes.byref(device.desc))
    return device


def test_description_properties_and_repr():
    device = bare_device()

    assert device.description == "Fake Adapter"
    assert (device.vendor_id, device.vram_size) == (0x8086, 256 * 1048576)
    assert repr(device) == "<Device Name:Fake Adapter Dedicated VRAM:256Mb VendorId:32902>"


def test_properties_without_a_description():
    device = bare_device(desc=False)

    assert (device.description, device.vendor_id, device.vram_size) == ("Unknown", 0, 0)
    assert repr(device) == "<Device Name:Unknown Dedicated VRAM:0Mb VendorId:0>"


@pytest.mark.parametrize("level,text", [(0xB100, "11.1"), (0xB000, "11.0"), (0x9300, "9.3")])
def test_feature_level_names(level, text):
    assert bare_device().feature_level_to_str(level) == text


def test_outputs_are_enumerated_until_dxgi_says_there_are_no_more():
    assert len(bare_device(FakeAdapter(outputs=3)).enum_outputs()) == 3


def test_an_enumeration_error_is_not_mistaken_for_the_end():
    """Stopping quietly would report a display that exists as absent."""
    device = bare_device(FakeAdapter(outputs=1, enum_error=E_FAIL))

    with pytest.raises(comtypes.COMError):
        device.enum_outputs()


def test_a_null_adapter_has_no_outputs():
    device = bare_device(FakeAdapter(), desc=True)
    device.adapter = FakeAdapter(null=True)

    assert device.enum_outputs() == []


# --------------------------------------------------------------------------
# util/io enumeration
# --------------------------------------------------------------------------

class FakeFactory:
    def __init__(self, adapters=2, error=None):
        self.adapters = adapters
        self.error = error

    def EnumAdapters1(self, index, out_ref):
        if index < self.adapters:
            return
        raise com_error(self.error if self.error is not None else DXGI_ERROR_NOT_FOUND)


def test_adapter_enumeration_stops_at_not_found(monkeypatch):
    monkeypatch.setattr(io_module, "_create_dxgi_factory1", lambda: FakeFactory(adapters=3))
    assert len(io_module.enum_dxgi_adapters()) == 3


def test_adapter_enumeration_errors_propagate(monkeypatch):
    monkeypatch.setattr(io_module, "_create_dxgi_factory1",
                        lambda: FakeFactory(adapters=1, error=E_FAIL))
    with pytest.raises(comtypes.COMError):
        io_module.enum_dxgi_adapters()


def test_output_enumeration_stops_at_not_found_and_raises_otherwise():
    assert len(io_module.enum_dxgi_outputs(FakeAdapter(outputs=2))) == 2
    with pytest.raises(comtypes.COMError):
        io_module.enum_dxgi_outputs(FakeAdapter(outputs=0, enum_error=E_FAIL))
