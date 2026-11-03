"""Sources and backends (ROADMAP § 7.3): what to capture, kept apart from how.

Headless throughout: the factory is assembled over fakes the way
tests/test_factory_paths.py does it, and the camera-level checks run on
tests/test_capture_paths.py's fake pipeline.
"""
import dataclasses
import types

import pytest

import rapidshot
import rapidshot.util.windows as windows_module
from rapidshot import Monitor, Region, Window
from rapidshot.frame import Frame

from test_factory_paths import FakeDevice, FakeOutput, factory_over

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.capture as capture_module  # noqa: E402
from rapidshot.core.backend import BACKENDS, CaptureBackend, SourceGeometry  # noqa: E402
from rapidshot.core.duplicator import Duplicator  # noqa: E402
from rapidshot.core.output import Output  # noqa: E402
from rapidshot.util.errors import RapidShotConfigError  # noqa: E402

from test_capture_paths import pipeline  # noqa: E402,F401


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class DesktopOutput(FakeOutput):
    """A FakeOutput placed on the virtual desktop."""

    def __init__(self, name, left, top, width=1920, height=1080):
        super().__init__(name, (width, height))
        rect = types.SimpleNamespace(left=left, top=top,
                                     right=left + width, bottom=top + height)
        self.desc = types.SimpleNamespace(DesktopCoordinates=rect)


def two_displays(monkeypatch, primary="DISPLAY2"):
    """Two adapters, one display each, side by side; DISPLAY2 is primary."""
    left = DesktopOutput("DISPLAY1", 0, 0)
    right = DesktopOutput("DISPLAY2", 1920, 0)
    devices = [FakeDevice("NVIDIA GPU"), FakeDevice("Intel GPU")]
    metadata = {"DISPLAY1": (None, primary == "DISPLAY1"),
                "DISPLAY2": (None, primary == "DISPLAY2")}
    factory = factory_over(monkeypatch, devices, [[left], [right]], metadata=metadata)
    return factory, left, right, devices


# --------------------------------------------------------------------------
# the values themselves
# --------------------------------------------------------------------------

def test_sources_are_plain_hashable_values():
    """They key caches and are built before any camera exists."""
    assert Monitor(1) == Monitor(1) and hash(Monitor(1)) == hash(Monitor(1))
    assert Region(0, 0, 10, 10) == Region(0, 0, 10, 10)
    assert {Window(5): "a"}[Window(5)] == "a"
    assert Monitor() != Monitor(0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        Monitor(1).index = 2


@pytest.mark.parametrize("build", [
    lambda: Monitor(-1), lambda: Monitor(True), lambda: Monitor("0"),
    lambda: Window(0), lambda: Window(-3), lambda: Window(True),
    lambda: Region(0, 0, 0, 10), lambda: Region(5, 0, 1, 10),
    lambda: Region(0, 10, 10, 10), lambda: Region(0.0, 0, 10, 10),
])
def test_invalid_sources_are_refused_at_construction(build):
    with pytest.raises(ValueError):
        build()


def test_region_reports_its_size():
    region = Region(100, 50, 740, 530)
    assert (region.width, region.height) == (640, 480)


def test_window_find_matches_titles(monkeypatch):
    monkeypatch.setattr(windows_module, "visible_windows",
                        lambda: [(0x10, "Untitled - Notepad"), (0x20, "Notepad")])

    assert Window.find("notepad") == Window(0x10)          # contains, topmost first
    assert Window.find("NOTEPAD", exact=True) == Window(0x20)
    with pytest.raises(LookupError, match="Paint"):
        Window.find("Paint")


# --------------------------------------------------------------------------
# resolving a source in the factory
# --------------------------------------------------------------------------

def test_no_source_keeps_the_2_6_behaviour(monkeypatch):
    factory, left, _, devices = two_displays(monkeypatch, primary="DISPLAY1")

    camera = factory.create()

    assert camera.kwargs["output"] is left
    assert camera.kwargs["device"] is devices[0]
    assert camera.kwargs["region"] is None
    assert camera.kwargs["backend"] == "dxgi"
    assert camera.kwargs["source"] is None


def test_primary_monitor_is_found_across_adapters(monkeypatch):
    """output_idx=None only ever looked at device 0; Monitor() looks everywhere."""
    factory, _, right, devices = two_displays(monkeypatch)

    camera = factory.create(source=Monitor())

    assert camera.kwargs["output"] is right
    assert camera.kwargs["device"] is devices[1]
    assert camera.kwargs["source"] == Monitor()


def test_monitor_index_counts_displays_across_adapters(monkeypatch):
    factory, left, right, _ = two_displays(monkeypatch)

    assert factory.create(source=Monitor(0)).kwargs["output"] is left
    assert factory.create(source=Monitor(1)).kwargs["output"] is right
    with pytest.raises(rapidshot.OutputError, match="2 display"):
        factory.create(source=Monitor(2))


def test_a_source_and_display_indices_together_are_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="names the display"):
        factory.create(output_idx=0, source=Monitor(1))
    with pytest.raises(rapidshot.ConfigurationError, match="names the display"):
        factory.create(device_idx=1, source=Monitor(1))


def test_monitor_source_still_takes_a_crop(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    camera = factory.create(source=Monitor(1), region=(0, 0, 100, 100))
    assert camera.kwargs["region"] == (0, 0, 100, 100)


def test_prefer_integrated_does_not_move_an_explicit_monitor(monkeypatch):
    """It reorders the adapters tried for duplication, but must not swap the
    display the caller named for the one the Intel adapter owns."""
    factory, left, _, devices = two_displays(monkeypatch)

    camera = factory.create(source=Monitor(0), prefer_integrated=True)

    assert camera.kwargs["output"] is left
    assert camera.kwargs["candidate_devices"][0] is devices[1]   # Intel tried first


def test_region_resolves_to_its_display_in_local_coordinates(monkeypatch):
    factory, _, right, _ = two_displays(monkeypatch)

    camera = factory.create(source=Region(2020, 100, 2660, 580))

    assert camera.kwargs["output"] is right
    assert camera.kwargs["region"] == (100, 100, 740, 580)
    assert camera.kwargs["source"] == Region(2020, 100, 2660, 580)


def test_region_spanning_two_displays_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="spans more than one"):
        factory.create(source=Region(1800, 0, 2000, 100))


def test_region_off_every_display_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.OutputError, match="not on any display"):
        factory.create(source=Region(5000, 0, 5100, 100))


def test_region_past_the_desktop_edge_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="past the edge"):
        factory.create(source=Region(3800, 1000, 3900, 1200))


def test_region_source_with_region_argument_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="already a region"):
        factory.create(source=Region(0, 0, 10, 10), region=(0, 0, 5, 5))


def test_window_source_names_what_it_needs(monkeypatch):
    """Without the extension's WGC, a window cannot be captured at all -- and
    the message says which component provides it."""
    monkeypatch.setattr("rapidshot.core.wgc.is_available", lambda: False)
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError,
                       match="Windows.Graphics.Capture.*rapidshot-native >= 0.3.0"):
        factory.create(source=Window(0x1234))


def test_unknown_source_type_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="Monitor, Window or Region"):
        factory.create(source=(0, 0, 10, 10))


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------

def test_backend_names():
    assert BACKENDS == ("auto", "dxgi", "wgc")


@pytest.mark.parametrize("backend", ["auto", "dxgi"])
def test_auto_and_dxgi_resolve_to_desktop_duplication(monkeypatch, backend):
    factory, *_ = two_displays(monkeypatch)
    assert factory.create(backend=backend).kwargs["backend"] == "dxgi"


def test_wgc_is_refused_without_the_extension(monkeypatch):
    monkeypatch.setattr("rapidshot.core.wgc.is_available", lambda: False)
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="rapidshot-native >= 0.3.0"):
        factory.create(backend="wgc")


def test_unknown_backend_is_refused(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    with pytest.raises(rapidshot.ConfigurationError, match="backend must be one of"):
        factory.create(backend="winrt")


def test_same_source_returns_the_same_camera(monkeypatch):
    factory, *_ = two_displays(monkeypatch)
    first = factory.create(source=Monitor(1))
    assert factory.create(source=Monitor(1)) is first
    assert factory.create(source=Monitor(1), backend="dxgi") is first


def test_module_create_forwards_source_and_backend(monkeypatch):
    captured = {}

    class Factory:
        def create(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(rapidshot, "get_factory", lambda: Factory())
    rapidshot.create(source=Monitor(1), backend="dxgi")
    assert captured["source"] == Monitor(1)
    assert captured["backend"] == "dxgi"


def test_screen_capture_refuses_an_unresolved_backend():
    """"auto" is the factory's to resolve; the camera takes a concrete backend.
    Checked before anything is touched, so nothing needs releasing."""
    with pytest.raises(RapidShotConfigError, match="'auto' is resolved by"):
        capture_module.ScreenCapture(output=None, device=None, backend="auto")


def test_camera_and_frames_report_backend_and_source(pipeline):  # noqa: F811
    cam, *_ = pipeline(output_color="BGRA", source=Monitor(1))

    assert cam.backend == "dxgi"
    assert cam.source == Monitor(1)
    with cam.grab_frame() as frame:
        assert frame.backend == "dxgi"
        assert frame.source == Monitor(1)


def test_frame_defaults_describe_desktop_duplication():
    frame = Frame(texture=object(), on_release=lambda: None, region=(0, 0, 4, 4))
    assert frame.backend == "dxgi"
    assert frame.source is None
    frame.release()


# --------------------------------------------------------------------------
# the backend contract
# --------------------------------------------------------------------------

def _members(protocol):
    """The protocol's declared attributes, methods and properties -- not the
    ABC machinery typing.Protocol adds to every protocol class."""
    names = set(getattr(protocol, "__annotations__", {}))
    names |= {name for name, value in vars(protocol).items()
              if not name.startswith("__")
              and (callable(value) or isinstance(value, property))}
    return names


def test_duplicator_meets_the_capture_backend_contract():
    """CaptureBackend writes down what ScreenCapture uses; Desktop Duplication
    must keep providing all of it."""
    provided = {f.name for f in dataclasses.fields(Duplicator)} | set(dir(Duplicator))
    assert _members(CaptureBackend) - provided == set()


def test_output_meets_the_source_geometry_contract():
    provided = {f.name for f in dataclasses.fields(Output)} | set(dir(Output))
    assert _members(SourceGeometry) - provided == set()


def test_an_output_change_rebuild_counts_as_a_generation(pipeline):  # noqa: F811
    """`generation` counts rebuilds. The output-change path rebuilt without
    counting, so a consumer comparing Frame.generation missed mode changes."""
    cam, *_ = pipeline()
    before = cam.generation

    assert cam._on_output_change() is True

    assert cam.generation == before + 1
