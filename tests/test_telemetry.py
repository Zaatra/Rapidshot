"""Whether the conditions sampler knows what it did and did not measure.

The interesting case is a provider that answers every time and measures
nothing. ``CallNtPowerInformation`` returns the nominal clock unchanged on
plenty of machines, and a perfectly flat frequency series looks exactly like a
machine holding its boost clock rock-steady -- which is the opposite
conclusion. One query could never tell those apart; repeated queries can at
least establish when the answer is not a measurement, and that is what is
tested here.

Nothing here samples for long enough to describe anything. The providers are
exercised for shape; the judgement is exercised with synthetic series.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import telemetry  # noqa: E402


def record(samples, **overrides):
    rec = telemetry.TelemetryRecord(interval=1.0, started_at="t0", samples=samples)
    for key, value in overrides.items():
        setattr(rec, key, value)
    return rec


def frequency_samples(series, maximum=2200):
    return [{"cpu_frequency": {"current_mhz": list(values),
                               "max_mhz": [maximum] * len(values)}}
            for values in series]


# ---------------------------------------------------------------------------
# The interval floor
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_an_interval_below_the_floor_is_refused_with_a_reason():
    """These counters are documented for low-frequency collection."""
    with pytest.raises(ValueError) as caught:
        telemetry.TelemetrySampler(interval=0.01)
    assert "low-frequency collection" in str(caught.value)


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_the_floor_itself_is_accepted():
    telemetry.TelemetrySampler(interval=telemetry.MINIMUM_INTERVAL)


# ---------------------------------------------------------------------------
# Frequency: present but not trustworthy
# ---------------------------------------------------------------------------

def test_a_frequency_pinned_to_the_maximum_is_called_out_as_nominal():
    """A flat series at max is the nominal figure echoed back, not a held clock."""
    flat = record(frequency_samples([[2200, 2200]] * 5, maximum=2200))
    notes = telemetry.TelemetrySampler._frequency_limitations(flat)
    assert len(notes) == 1
    assert "NOT as evidence" in notes[0]
    assert "nominal figure being echoed back" in notes[0]


def test_a_flat_frequency_below_the_maximum_is_still_suspect():
    flat = record(frequency_samples([[1200, 1200]] * 5, maximum=2200))
    notes = telemetry.TelemetrySampler._frequency_limitations(flat)
    assert notes and "may not be measuring" in notes[0]
    assert "NOT as evidence" not in notes[0]


def test_a_frequency_that_actually_varies_raises_no_limitation():
    varying = record(frequency_samples([[1466, 2200], [1800, 2100], [1500, 2000]]))
    assert telemetry.TelemetrySampler._frequency_limitations(varying) == []


def test_missing_frequency_is_unavailable_rather_than_assumed():
    notes = telemetry.TelemetrySampler._frequency_limitations(
        record([{"load": {"system_cpu_percent": 3.0}}]))
    assert notes == ["CPU frequency was not sampled; sustained clock is unavailable, "
                     "not assumed"]


# ---------------------------------------------------------------------------
# What the record admits about itself
# ---------------------------------------------------------------------------

def test_no_samples_is_stated_plainly():
    sampler_notes = telemetry.TelemetrySampler._limitations.__wrapped__ \
        if hasattr(telemetry.TelemetrySampler._limitations, "__wrapped__") else None
    assert sampler_notes is None          # plain method; exercised via a fake below


class _Judge(telemetry.TelemetrySampler):
    """Only the judgement, with no providers and no Windows calls."""

    def __init__(self):
        self.limitations = []


def test_an_empty_window_describes_nothing():
    notes = _Judge()._limitations(record([]))
    assert notes == ["no samples were collected; nothing here describes the run"]


def test_too_few_samples_to_describe_a_window_is_a_limitation():
    notes = _Judge()._limitations(record(frequency_samples([[1466], [1800]])))
    assert any("only 2 samples" in note for note in notes)


def test_missed_deadlines_are_reported_as_gaps():
    notes = _Judge()._limitations(
        record(frequency_samples([[1466], [1800], [1500]]), missed=4))
    assert any("4 sampling deadline(s) missed" in note for note in notes)


def test_the_summary_flags_a_series_that_never_moved():
    summary = record([{"load": {"cpu": 5.0}}, {"load": {"cpu": 5.0}}]).summary()
    assert summary["load.cpu"]["varied"] is False
    summary = record([{"load": {"cpu": 5.0}}, {"load": {"cpu": 9.0}}]).summary()
    assert summary["load.cpu"]["varied"] is True


def test_the_record_refuses_to_call_its_spread_a_confidence_interval():
    """These are serial, autocorrelated readings of one machine."""
    payload = record([{"load": {"cpu": 1.0}}, {"load": {"cpu": 3.0}}]).as_dict()
    assert "not a confidence interval" in payload["note"]
    assert "stdev" in payload["summary"]["load.cpu"]
    assert not any(key.startswith("ci") or "confidence" in key
                   for key in payload["summary"]["load.cpu"])


def test_per_processor_arrays_reduce_to_extremes_rather_than_flooding_the_series():
    summary = record(frequency_samples([[1000, 2000, 3000]])).summary()
    assert summary["cpu_frequency.current_mhz.min"]["median"] == 1000
    assert summary["cpu_frequency.current_mhz.max"]["median"] == 3000
    assert summary["cpu_frequency.current_mhz.mean"]["median"] == 2000
    # Not 32 separate series per sample.
    assert "cpu_frequency.current_mhz[0]" not in summary


# ---------------------------------------------------------------------------
# Background load: a warning, never a verdict
# ---------------------------------------------------------------------------

def load_record(other_values):
    return record([{"load": {"other_cpu_percent": value,
                             "system_cpu_percent": value + 10,
                             "benchmark_tree_cpu_percent": 10.0}}
                   for value in other_values])


def test_a_quiet_machine_produces_no_warning():
    assert telemetry.background_load_warning(load_record([1.0, 2.0, 1.5])) == []


def test_a_loaded_machine_warns_and_says_it_is_not_proof():
    reasons = telemetry.background_load_warning(load_record([30.0, 35.0, 40.0]))
    assert len(reasons) == 1
    assert "absolute timings are inflated" in reasons[0]
    # The residual includes the compositor and driver threads, so it indicts
    # nobody on its own.
    assert "not proof of contamination" in reasons[0]


def test_the_warning_threshold_is_the_median_not_the_peak():
    # One spike in an otherwise quiet window is not a contaminated run.
    assert telemetry.background_load_warning(load_record([1.0, 60.0, 1.0])) == []


def test_no_telemetry_at_all_cannot_warn_about_anything():
    assert telemetry.background_load_warning(None) == []
    assert telemetry.background_load_warning(record([])) == []


# ---------------------------------------------------------------------------
# Providers and overhead
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_an_unavailable_provider_is_recorded_not_dropped(monkeypatch):
    """"No GPU telemetry" and "a GPU nobody asked about" must not look alike."""
    class Broken:
        def __init__(self, **kwargs):
            raise OSError("nvml.dll not found")

    monkeypatch.setattr(telemetry, "NvmlProvider", Broken)
    sampler = telemetry.TelemetrySampler(interval=1.0)
    assert sampler.providers["gpu"]["available"] is False
    assert "nvml.dll not found" in sampler.providers["gpu"]["reason"]
    # The rest still works.
    assert sampler.providers["load"]["available"] is True


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_a_provider_that_throws_mid_run_is_recorded_per_sample(monkeypatch):
    sampler = telemetry.TelemetrySampler(interval=1.0, want_gpu=False,
                                         want_frequency=False)

    class Exploding:
        name = "boom"

        def read(self):
            raise RuntimeError("counter went away")

    sampler._readers.append(("flaky", Exploding()))
    sample = sampler.sample_telemetry()
    assert "counter went away" in sample["flaky"]["error"]


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_one_shot_sampling_states_that_an_instant_is_not_a_window():
    sample = telemetry.sample_telemetry(want_gpu=False)
    assert "not a sustained frequency" in sample["limitation"]


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_overhead_is_measured_rather_than_assumed():
    """Turn nothing on by default without knowing what it costs."""
    overhead = telemetry.measure_overhead(interval=1.0, duration=0.4)
    if not overhead.get("available"):
        pytest.skip(f"no providers here: {overhead['reason']}")
    assert overhead["samples"] > 0
    assert overhead["wall_ms_per_sample"]["median"] >= 0
    assert 0 <= overhead["duty_cycle_at_interval"] < 1
    assert "re-measure rather than quoting this" in overhead["note"]


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_a_short_sampled_window_records_its_providers_and_interval():
    sampler = telemetry.TelemetrySampler(interval=telemetry.MINIMUM_INTERVAL,
                                         want_gpu=False)
    with sampler:
        pass
    payload = sampler.stopped.as_dict()
    assert payload["interval_seconds"] == telemetry.MINIMUM_INTERVAL
    assert payload["providers"]["load"]["available"] is True
    assert payload["sample_count"] >= 1


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_the_process_tree_is_counted_as_this_benchmark_not_as_background():
    """Attributing the motion source to strangers would indict every run."""
    provider = telemetry.SystemLoadProvider()
    reading = provider.read()
    assert reading["processes_counted"] >= 1
    assert reading["other_cpu_percent"] >= 0
    assert reading["other_cpu_percent"] <= reading["system_cpu_percent"] + 0.01


# ---------------------------------------------------------------------------
# Process attribution
# ---------------------------------------------------------------------------

class _FakeProcess:
    """`cpu_percent(interval=None)` is stateful per object, as psutil's is."""

    def __init__(self, pid, busy, registry):
        self.pid = pid
        self._busy = busy
        self._seen = False
        registry.setdefault(pid, []).append(self)

    def cpu_percent(self, interval=None):
        if not self._seen:
            self._seen = True
            return 0.0            # psutil's first call always reports zero
        return self._busy

    def memory_info(self):
        return SimpleNamespace(rss=100 * 1024 * 1024)

    def children(self, recursive=False):
        # A fresh object every call, which is what psutil does and what broke it.
        return [_FakeProcess(99, 800.0, self._registry)]


def _load_provider(monkeypatch):
    registry = {}

    class Error(Exception):
        pass

    root = None

    def make(pid):
        nonlocal root
        root = _FakeProcess(pid, 50.0, registry)
        root._registry = registry
        return root

    fake = SimpleNamespace(
        Process=make, Error=Error,
        cpu_percent=lambda interval=None: 30.0,
        virtual_memory=lambda: SimpleNamespace(percent=40.0,
                                               available=8 * 1024 ** 3))
    monkeypatch.setitem(sys.modules, "psutil", fake)
    provider = telemetry.SystemLoadProvider(tree_root_pid=1)
    return provider, registry


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_a_child_is_primed_before_it_is_counted(monkeypatch):
    """A newly seen process contributes its priming zero to nothing."""
    provider, _ = _load_provider(monkeypatch)
    first = provider.read()
    assert first["processes_newly_seen"] == 1
    assert first["processes_counted"] == 1          # the root only


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_process_objects_persist_so_a_busy_child_is_not_read_as_idle(monkeypatch):
    """The live-run bug: a worker burning 113% of a core read as 0.05%.

    `children()` hands back new objects every call, so rebuilding the list each
    sample meant every child's only reading was its priming zero -- and the
    whole of the benchmark's own CPU was attributed to background load.
    """
    provider, registry = _load_provider(monkeypatch)
    provider.read()
    second = provider.read()
    assert second["processes_counted"] == 2
    # The child's 800% over 32 cores, plus the root's 50%.
    assert second["benchmark_tree_cpu_percent"] > 0
    # And the cached object was reused rather than replaced.
    assert sum(1 for process in registry[99] if process._seen) == 1


@pytest.mark.skipif(not telemetry.IS_WINDOWS, reason="Windows-only")
def test_the_residual_shrinks_once_the_tree_is_attributed(monkeypatch):
    provider, _ = _load_provider(monkeypatch)
    first = provider.read()
    second = provider.read()
    assert second["other_cpu_percent"] < first["other_cpu_percent"]
