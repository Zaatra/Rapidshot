"""The capture planner and backend="auto" (ROADMAP § 7.3), headless.

Calibration itself -- the test source, real cameras, pixel age -- is replaced
here by recorded Measurements; tests/test_planner_live.py runs it for real.
"""
import json
import sys
import types

import pytest

import rapidshot
from rapidshot import Monitor, Region, Window

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.__main__ as cli  # noqa: E402
import rapidshot.core.wgc as wgc  # noqa: E402
import rapidshot.planner as planner  # noqa: E402
from rapidshot.planner import Measurement  # noqa: E402
from rapidshot.util.errors import RapidShotConfigError  # noqa: E402

from test_sources import two_displays  # noqa: E402


def measured(backend, age, fps=160.0, cpu=1.0, capture=(1920, 1080)):
    return Measurement(backend=backend, capture=capture, frames=int(fps * 2), seconds=2.0, fps=fps,
                       age_p50_ms=age, age_p95_ms=age + 0.5, age_p99_ms=age + 4,
                       cpu_ms_per_frame=cpu, measured_at="2026-10-29T20:00:00+00:00")


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    path = tmp_path / "plans.json"
    monkeypatch.setenv("RAPIDSHOT_PLAN_CACHE", str(path))
    return path


@pytest.fixture
def machine(monkeypatch):
    """Two displays, WGC available, the factory reachable as rapidshot's own."""
    factory, left, right, devices = two_displays(monkeypatch)
    monkeypatch.setattr(rapidshot, "get_factory", lambda: factory)
    monkeypatch.setattr(wgc, "is_available", lambda: True)
    return factory


# --------------------------------------------------------------------------
# the decision
# --------------------------------------------------------------------------

def test_the_freshest_backend_wins():
    choice, why = planner._choose({"dxgi": measured("dxgi", 12.2),
                                   "wgc": measured("wgc", 14.3)})
    assert choice == "dxgi"
    assert "2.1 ms fresher" in why[0]


def test_a_tie_goes_to_frames_then_cpu():
    """Within the ~1 ms run-to-run spread, pixel age cannot separate them."""
    choice, why = planner._choose({"dxgi": measured("dxgi", 12.2, fps=120),
                                   "wgc": measured("wgc", 12.9, fps=164)})
    assert choice == "wgc" and "more frames (164 fps against 120)" in why[0]

    choice, why = planner._choose({"dxgi": measured("dxgi", 12.2, cpu=3.0),
                                   "wgc": measured("wgc", 12.4, cpu=1.0)})
    assert choice == "wgc" and "CPU cost decided" in why[0]


def test_a_fraction_of_a_frame_per_second_is_not_a_reason():
    """Measured on Machine B, 2026-10-29: both backends paced by the same
    165 Hz display read 164 fps, and the first version of this rule chose
    dxgi on the decimals -- then said it was cheaper, which it was not."""
    choice, why = planner._choose({"dxgi": measured("dxgi", 16.05, fps=164.4, cpu=1.14),
                                   "wgc": measured("wgc", 15.60, fps=164.2, cpu=0.70)})
    assert choice == "wgc"
    assert "wgc 0.70 ms/frame against dxgi's 1.14" in why[0]
    assert "more frames" not in why[0]


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def test_the_cache_round_trips(isolated_cache):
    planner._store("k1", {"choice": "wgc", "evidence": {}})
    planner._store("k2", {"choice": "dxgi", "evidence": {}})

    assert json.loads(isolated_cache.read_text())["version"] == 1
    assert planner._load_cache()["k1"]["choice"] == "wgc"
    assert planner._load_cache()["k2"]["choice"] == "dxgi"


def test_a_corrupt_cache_reads_as_empty(isolated_cache):
    isolated_cache.write_text("{not json")
    assert planner._load_cache() == {}


def test_the_key_changes_with_what_a_measurement_depends_on(machine, monkeypatch):
    output = machine.outputs[0][0]
    full = (1920, 1080)
    base = planner._key("display", machine, output, full)
    assert planner._key("display", machine, output, full) == base

    by_size = planner._key("display", machine, output, (640, 640))
    monkeypatch.setattr(planner, "_display_mode", lambda o: (2560, 1600, 60))
    by_mode = planner._key("display", machine, output, full)
    monkeypatch.setattr(planner, "_topology", lambda f: ["GPU [10de:28e0] 32.0.16.1714"])
    by_driver = planner._key("display", machine, output, full)
    monkeypatch.setattr("rapidshot.native.extension_version", lambda: "0.3.0")
    by_native = planner._key("display", machine, output, full)
    monkeypatch.setattr(planner, "_power_source", lambda: "battery")
    by_power = planner._key("display", machine, output, full)

    assert len({base, by_size, by_mode, by_driver, by_native, by_power}) == 6
    assert planner._key("window", machine, output, full) != by_native


# --------------------------------------------------------------------------
# plan()
# --------------------------------------------------------------------------

def test_a_mux_switch_changes_the_key(machine, monkeypatch):
    """Same adapters, same drivers, same display mode -- but in discrete mode
    the dGPU drives the panel instead of the iGPU, which is what decides
    between the backends. A measurement from one mode must not answer for
    the other."""
    output = machine.outputs[0][0]
    hybrid = planner._key("display", machine, output, (1920, 1080))
    machine.outputs[0], machine.outputs[1] = machine.outputs[1], machine.outputs[0]
    discrete = planner._key("display", machine, output, (1920, 1080))
    assert hybrid != discrete


def test_the_power_source_is_read():
    assert planner._power_source() in ("ac", "battery", "unknown")


def test_a_window_plan_needs_no_measurement(machine):
    result = rapidshot.plan(Window(0x1234))

    assert result.backend == "wgc" and result.candidates == ("wgc",)
    assert "only Windows.Graphics.Capture" in result.reasons[0]
    assert "wgc" in str(result)


def test_an_unmeasured_display_says_so(machine):
    result = rapidshot.plan(Monitor(1))

    assert result.backend == "dxgi" and not result.measured
    assert result.candidates == ("dxgi", "wgc")
    assert "calibrate=True" in result.reasons[0]
    assert "(unmeasured)" in str(result)


def test_without_wgc_desktop_duplication_is_the_only_choice(machine, monkeypatch):
    monkeypatch.setattr(wgc, "is_available", lambda: False)
    result = rapidshot.plan(Region(0, 0, 100, 100))
    assert result.backend == "dxgi" and result.candidates == ("dxgi",)
    assert "only backend available" in result.reasons[0]


def test_calibration_measures_caches_and_decides(machine, monkeypatch):
    sources = []

    class FakeSource:
        def __init__(self, *mode):
            sources.append(self)
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(planner, "_Source", FakeSource)
    monkeypatch.setattr(planner, "_display_mode", lambda output: (1920, 1080, 60))
    ages = {"dxgi": 12.2, "wgc": 14.1}
    monkeypatch.setattr(planner, "_measure",
                        lambda backend, source, seconds, capture: measured(backend, ages[backend]))

    result = rapidshot.plan(Monitor(1), calibrate=True)

    assert result.measured and result.backend == "dxgi"
    assert set(result.evidence) == {"dxgi", "wgc"}
    assert sources[0].closed, "the test source must always be stopped"
    # ...and the next plan, and backend="auto", find it without measuring.
    monkeypatch.setattr(planner, "_measure", lambda *a: pytest.fail("measured again"))
    again = rapidshot.plan(Monitor(1))
    assert again.measured and again.backend == "dxgi"
    assert "measured dxgi at 1920x1080: pixel age p50 12.20 ms" in str(again)
    assert "grab() to NumPy (BGRA) of a 1920x1080 area" in str(again)
    # ...and a GPU consumer is told that its own path was not what was timed.
    gpu = rapidshot.plan(Monitor(1), consumer="cupy")
    assert any("GPU consumer takes grab_frame() and GpuConverter" in r for r in gpu.reasons)
    assert again.evidence["dxgi"].capture == (1920, 1080)   # a tuple again after JSON


def test_a_backend_that_cannot_be_measured_is_reported(machine, monkeypatch):
    class FakeSource:
        def __init__(self, *mode):
            pass

        def close(self):
            pass

    def measure(backend, source, seconds, capture):
        if backend == "wgc":
            raise RuntimeError("wgc delivered 3 timed frames in 2.0 s")
        return measured(backend, 12.0)

    monkeypatch.setattr(planner, "_Source", FakeSource)
    monkeypatch.setattr(planner, "_display_mode", lambda output: (1920, 1080, 60))
    monkeypatch.setattr(planner, "_measure", measure)

    result = rapidshot.plan(Monitor(1), calibrate=True)

    assert not result.measured and result.backend == "dxgi"
    assert any("wgc could not be measured" in r for r in result.reasons)
    assert planner._load_cache() == {}, "half a comparison is not cached as a decision"


def test_a_cpu_consumer_is_described(machine):
    cpu = rapidshot.plan(Monitor(1), consumer="cpu")
    assert cpu.consumer_adapter == "the CPU" and cpu.crosses_adapters is False
    assert "to_nchw" in cpu.preprocessing

    with pytest.raises(ValueError, match="consumer must be one of"):
        rapidshot.plan(Monitor(1), consumer="tensorflow")


def test_without_cupy_the_cuda_side_is_unknown(machine, monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    torch = rapidshot.plan(Monitor(1), consumer="torch", size=(640, 640))
    assert torch.consumer_adapter is None and torch.crosses_adapters is None
    assert any("CuPy is not installed" in r for r in torch.reasons)
    assert "GpuConverter to 640x640" in torch.preprocessing


@pytest.mark.parametrize("lookup, adapter, crosses, reason", [
    (lambda cp, luid: 0, "CUDA device 0, the capture adapter", False, None),
    ("cross", "a CUDA device on another adapter", True, None),
    ("broken", None, None, "could not tell which adapter CUDA runs on"),
])
def test_a_cuda_consumer_is_located(machine, monkeypatch, lookup, adapter, crosses, reason):
    """Same adapter, the hybrid-laptop case, and "could not tell" -- which
    must not be reported as "on another adapter"."""
    from rapidshot import converter

    def cross(cp, luid):
        raise converter.CrossAdapterRequired("no CUDA device is on the adapter")

    def broken(cp, luid):
        raise RuntimeError("cuInit failed with CUDA error 100")

    monkeypatch.setitem(sys.modules, "cupy", types.ModuleType("cupy"))
    monkeypatch.setattr(converter, "_device_for_adapter",
                        {"cross": cross, "broken": broken}.get(lookup, lookup))

    result = rapidshot.plan(Monitor(1), consumer="torch")

    assert result.consumer_adapter == adapter
    assert result.crosses_adapters is crosses
    if reason:
        assert any(reason in r for r in result.reasons)


# --------------------------------------------------------------------------
# backend="auto"
# --------------------------------------------------------------------------

def cache_choice(factory, source, choice):
    kind, output, _ = planner._resolve(factory, source)
    planner._store(planner._key(kind, factory, output, planner._capture_size(source, output)),
                   {"choice": choice, "evidence": {}})


def test_a_source_is_calibrated_at_the_size_it_captures(machine, monkeypatch):
    """Machine B, 2026-10-29: cropped to the marker WGC was 0.4 ms fresher;
    reading the whole display back, Desktop Duplication was 2.3 ms fresher.
    So a Monitor is measured at full size and a Region at its own size -- and
    each has its own cached answer."""
    sizes = []

    class FakeSource:
        def __init__(self, *mode):
            pass

        def close(self):
            pass

    monkeypatch.setattr(planner, "_Source", FakeSource)
    monkeypatch.setattr(planner, "_display_mode", lambda output: (1920, 1080, 60))
    monkeypatch.setattr(planner, "_measure", lambda backend, source, seconds, capture:
                        sizes.append(capture) or measured(backend, 12 if backend == "dxgi" else 14))

    rapidshot.plan(Monitor(0), calibrate=True)
    rapidshot.plan(Region(0, 0, 640, 640), calibrate=True)

    assert sizes == [(1920, 1080), (1920, 1080), (640, 640), (640, 640)]
    assert len(planner._load_cache()) == 2, "one cached answer per capture size"


@pytest.mark.parametrize("region", [Region(0, 0, 200, 10), Region(0, 0, 100, 100)])
def test_a_capture_too_small_for_the_marker_cannot_be_calibrated(machine, monkeypatch, region):
    monkeypatch.setattr(planner, "_display_mode", lambda output: (1920, 1080, 60))
    with pytest.raises(rapidshot.ConfigurationError, match="frame-ID marker"):
        rapidshot.plan(region, calibrate=True)


@pytest.fixture
def wgc_geometry(monkeypatch):
    monkeypatch.setattr(wgc, "WgcMonitorGeometry", lambda output: ("wgc-geometry", output))


def test_auto_follows_the_measured_choice_for_a_source(machine, wgc_geometry):
    cache_choice(machine, Monitor(1), "wgc")
    camera = machine.create(source=Monitor(1))
    assert camera.kwargs["backend"] == "wgc"


def test_auto_without_a_source_keeps_the_2_6_behaviour(machine, wgc_geometry):
    """Whatever has been measured, a call that names no source is a 2.6 call."""
    cache_choice(machine, Monitor(0), "wgc")
    assert machine.create().kwargs["backend"] == "dxgi"


def test_a_cached_choice_that_cannot_run_is_ignored(machine, monkeypatch):
    cache_choice(machine, Monitor(1), "wgc")
    monkeypatch.setattr(wgc, "is_available", lambda: False)
    assert planner.cached_backend(machine, Monitor(1)) is None
    assert machine.create(source=Monitor(1)).kwargs["backend"] == "dxgi"


def test_a_broken_cache_never_breaks_create(machine, isolated_cache):
    isolated_cache.write_text("[1, 2, 3]")
    assert planner.cached_backend(machine, Monitor(1)) is None
    assert machine.create(source=Monitor(1)).kwargs["backend"] == "dxgi"


def _refusing_capture(monkeypatch, marked=True):
    built = []

    class Capture:
        def __init__(self, **kwargs):
            built.append(kwargs)
            if kwargs["backend"] == "dxgi":
                error = RapidShotConfigError("every adapter refused duplication")
                if marked:
                    error.duplication_refused = True
                raise error
            self.kwargs = kwargs

    monkeypatch.setattr(rapidshot, "ScreenCapture", Capture)
    return built


def test_auto_falls_back_to_wgc_when_duplication_is_refused_everywhere(
        machine, monkeypatch, wgc_geometry):
    built = _refusing_capture(monkeypatch)
    warned = []
    # The factory's own logger, intercepted: other tests reconfigure the
    # "rapidshot" logger tree, which makes caplog's view order-dependent.
    monkeypatch.setattr(rapidshot.logger, "warning", lambda message, *a, **k: warned.append(message))

    camera = machine.create(source=Monitor(1))

    assert [b["backend"] for b in built] == ["dxgi", "wgc"]
    assert camera.kwargs["output"][0] == "wgc-geometry"
    assert any("through Windows.Graphics.Capture instead" in m for m in warned)


def test_no_fallback_for_other_refusals_or_an_explicit_backend(machine, monkeypatch):
    """A desktop or protected-content refusal would stop WGC too; and a
    caller who asked for "dxgi" gets "dxgi" or an error."""
    _refusing_capture(monkeypatch, marked=False)
    with pytest.raises(rapidshot.RapidshotError, match="every adapter refused"):
        machine.create(source=Monitor(1))

    _refusing_capture(monkeypatch, marked=True)
    with pytest.raises(rapidshot.RapidshotError, match="every adapter refused"):
        machine.create(source=Monitor(1), backend="dxgi")


# --------------------------------------------------------------------------
# the command line
# --------------------------------------------------------------------------

def test_the_plan_command_prints_a_plan(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(rapidshot, "plan",
                        lambda source, **kw: calls.append((source, kw)) or "THE PLAN")

    assert cli.main(["plan", "--monitor", "1", "--consumer", "torch", "--calibrate"]) == 0

    assert "THE PLAN" in capsys.readouterr().out
    assert calls[0][0] == Monitor(1)
    assert calls[0][1]["consumer"] == "torch" and calls[0][1]["calibrate"] is True


def test_the_plan_command_finds_a_window_by_title(monkeypatch, capsys):
    monkeypatch.setattr("rapidshot.util.windows.visible_windows", lambda: [(0x42, "Notepad")])
    monkeypatch.setattr(rapidshot, "plan", lambda source, **kw: f"plan for {source}")
    assert cli.main(["plan", "--window", "notepad"]) == 0
    assert "plan for Window(0x42)" in capsys.readouterr().out
