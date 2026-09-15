"""The arithmetic behind published benchmark numbers, with no benchmark run.

`perf_suite.py` gates releases and `compare_libraries.py` produces the
cross-library tables in README, so how they reduce samples decides what gets
quoted. Headless coverage had `compare_libraries.py` at 0%.

Writing these found that `compare_libraries.py`'s per-frame timings were per-
*call* timings: the interval clock advanced on every grab, hit or miss, so a
polling library reported ms_p50 1.59 ms at ~100 fps while RapidShot's blocking
default reported 9.96 ms at the same frame rate. README quotes only fps, CPU
and memory from those runs, which were computed correctly; the recorded JSON's
ms_p50/p99/jitter columns predate the fix.

Everything is driven by scripted clocks and synthetic rows.
"""
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import compare_libraries  # noqa: E402
import perf_suite  # noqa: E402


# --------------------------------------------------------------------------
# compare_libraries: measuring one cell
# --------------------------------------------------------------------------

class ScriptedCapture:
    """A library whose grab() takes 1 ms and yields a frame on a pattern."""

    def __init__(self, pattern):
        self.pattern = pattern
        self.now = 0.0
        self.calls = 0

    def clock(self):
        return self.now

    def grab(self):
        hit = self.pattern[self.calls % len(self.pattern)]
        self.calls += 1
        self.now += 0.001
        return True if hit else None


class NoMonitor:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def summary(self):
        return {"cpu_percent_mean": 12.5}


@pytest.fixture
def worker(monkeypatch):
    def run(pattern, seconds=0.06, warmup=0):
        capture = ScriptedCapture(pattern)
        monkeypatch.setitem(compare_libraries.ADAPTERS, "scripted",
                            lambda scenario, colour: (capture.grab, None, {"extra": 1}))
        monkeypatch.setattr(compare_libraries.time, "perf_counter", capture.clock)
        monkeypatch.setattr(compare_libraries, "ResourceMonitor", NoMonitor)
        monkeypatch.setattr(compare_libraries, "control_ms", lambda: 0.25)
        return compare_libraries.run_worker("scripted", "fullscreen", "BGRA", seconds, warmup)
    return run


def test_timings_are_frame_intervals_not_call_durations(worker):
    """Three 1 ms calls per frame is a 3 ms interval, however cheap the misses."""
    result = worker([False, False, True])

    assert result["ms_p50"] == pytest.approx(3.0)
    assert result["ms_jitter_stdev"] == pytest.approx(0.0, abs=1e-6)
    assert result["ms_basis"] == "frame_interval"
    assert result["misses"] == 2 * result["frames"]
    assert result["fps_mean"] == pytest.approx(1000 / 3, rel=0.05)


def test_a_library_that_never_misses_has_call_length_intervals(worker):
    result = worker([True])
    assert result["ms_p50"] == pytest.approx(1.0)
    assert (result["cpu_percent_mean"], result["control_ms"], result["extra"]) == (12.5, 0.25, 1)


def test_warmup_calls_are_not_measured(worker):
    result = worker([True], warmup=10)
    assert result["frames"] == pytest.approx(60, abs=1)


def test_a_cell_with_no_frames_says_so(worker):
    result = worker([False])
    assert result["frames"] == 0 and result["error"] == "no frames returned"


# --------------------------------------------------------------------------
# compare_libraries: aggregation and presentation
# --------------------------------------------------------------------------

def cell(fps, cpu, rss, **extra):
    base = {"library": "rapidshot", "scenario": "fullscreen", "colour": "BGRA",
            "frames": 100, "fps_mean": fps, "cpu_percent_mean": cpu, "rss_mb_end": rss,
            "ms_p50": 10.0, "ms_p99": 12.0, "ms_jitter_stdev": 0.5}
    base.update(extra)
    return base


def test_aggregate_reports_the_median_and_each_spread():
    merged = compare_libraries.aggregate([
        cell(100.0, 10.0, 120.0), cell(90.0, 20.0, 120.0), cell(110.0, 15.0, 120.0),
        {"library": "rapidshot", "frames": 0, "error": "crashed"},
    ])

    assert merged["repeats"] == 3
    assert merged["fps_mean"] == 100.0
    assert merged["fps_spread_pct"] == 20.0          # (110 - 90) / 100
    assert merged["cpu_spread_pct"] == 66.7          # (20 - 10) / 15
    assert merged["rss_spread_pct"] == 0.0
    assert merged["worst_spread_pct"] == 66.7


def test_aggregate_of_only_failures_returns_the_failure():
    failed = {"library": "mss", "frames": 0, "error": "no output"}
    assert compare_libraries.aggregate([failed]) is failed


def test_one_repeat_has_no_spread():
    merged = compare_libraries.aggregate([cell(100.0, 10.0, 120.0)])
    assert merged["worst_spread_pct"] == 0.0


def test_the_table_flags_cells_too_noisy_to_argue_from():
    quiet = dict(cell(100.0, 10.0, 120.0), worst_spread_pct=2.0, fps_spread_pct=2.0)
    noisy = dict(cell(90.0, 30.0, 130.0), library="dxcam", worst_spread_pct=25.0)
    broken = {"library": "bettercam", "scenario": "region", "colour": "RGB",
              "frames": 0, "error": "ImportError: bettercam"}

    lines = compare_libraries.table([quiet, noisy, broken]).splitlines()

    assert "100.0 ± 2.0%" in lines[2] and not lines[2].rstrip().endswith("<<")
    assert lines[3].rstrip().endswith("<<")
    assert "ImportError: bettercam" in lines[4]


def test_resource_summary_is_cpu_seconds_over_wall_time():
    monitor = compare_libraries.ResourceMonitor.__new__(compare_libraries.ResourceMonitor)
    times = types.SimpleNamespace
    monitor.rss = [100_000_000, 150_000_000]
    monitor._cpu_start = times(user=1.0, system=0.5)
    monitor._cpu_end = times(user=2.0, system=1.0)
    monitor._elapsed = 3.0

    assert monitor.summary() == {
        "cpu_percent_mean": 50.0, "cpu_seconds": 1.5,
        "rss_mb_start": 100.0, "rss_mb_end": 150.0, "rss_mb_growth": 50.0}

    monitor._cpu_end = None
    assert monitor.summary()["cpu_percent_mean"] is None


def test_environment_distinguishes_unversioned_from_absent(monkeypatch):
    """DXcam ships no __version__; it was recorded as 'unavailable'."""
    monkeypatch.setitem(sys.modules, "dxcam", types.ModuleType("dxcam"))
    monkeypatch.setitem(sys.modules, "bettercam", None)
    import importlib.metadata

    def no_metadata(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", no_metadata)

    info = compare_libraries.environment(motion=True)

    assert info["dxcam"] == "installed, version unknown"
    assert info["bettercam"].startswith("unavailable")
    assert info["motion_on_screen"] is True


def test_spawn_reads_the_worker_result_or_its_last_error(monkeypatch):
    outputs = iter([
        types.SimpleNamespace(stdout='noise\n{"library": "mss", "frames": 3}\n', stderr=""),
        types.SimpleNamespace(stdout="", stderr="Traceback...\nImportError: no mss\n"),
    ])
    monkeypatch.setattr(compare_libraries.subprocess, "run", lambda *a, **k: next(outputs))

    assert compare_libraries.spawn("mss", "region", "BGRA", 1, 0) == {"library": "mss", "frames": 3}
    failed = compare_libraries.spawn("mss", "region", "BGRA", 1, 0)
    assert failed["frames"] == 0 and failed["error"] == "ImportError: no mss"


# --------------------------------------------------------------------------
# perf_suite: reducing samples
# --------------------------------------------------------------------------

def test_results_pool_rounds_and_recompute():
    first = perf_suite.Result("conv.rgb", "synthetic", [0.004, 0.002], bytes_moved=8_000_000)
    second = perf_suite.Result("conv.rgb", "synthetic", [0.001, 0.003])
    other = perf_suite.Result("conv.gray", "synthetic", [0.010])

    merged = perf_suite.merge_rounds([[first, other], [second]])

    assert [r.name for r in merged] == ["conv.rgb", "conv.gray"]
    pooled = merged[0]
    assert (pooled.n, pooled.min_ms, pooled.median_ms) == (4, 1.0, 2.5)
    assert pooled.gb_per_s == pytest.approx(8_000_000 / 1e9 / 0.0025)
    assert pooled.to_dict()["gb_per_s"] == round(pooled.gb_per_s, 3)
    assert merged[1].stdev_ms == 0.0


def test_to_dict_keeps_sub_microsecond_precision_and_notes():
    r = perf_suite.Result("com.call", "synthetic", [0.0000004], note="per call")
    d = r.to_dict()
    assert d["min_ms"] == 0.0004 and d["note"] == "per call" and "gb_per_s" not in d


@pytest.mark.parametrize("value,text", [(1.23456, "1.235m"), (0.0004, "0.40u")])
def test_durations_are_never_collapsed_to_zero(value, text):
    assert perf_suite._fmt_ms(value) == text


def test_the_table_and_the_cpu_versus_gpu_headline(capsys):
    def result(name, ms):
        return perf_suite.Result(name, "live", [ms / 1000.0])

    perf_suite.print_table([
        result("live.grab_with_frame", 8.0), result("live.grab_frame_gpu", 2.0),
        result("pipeline.cpu_to_nchw", 12.0), result("pipeline.gpu_dispatch", 0.5),
        result("pipeline.gpu_plus_readback", 20.0),
    ])
    out = capsys.readouterr().out

    assert "4.0x faster, 6.00 ms/frame saved" in out
    assert "GPU  dispatch + forced readback" in out and "SLOWER than the CPU arm" in out


def test_no_headline_without_both_capture_rows(capsys):
    perf_suite.print_cpu_vs_gpu([perf_suite.Result("live.grab_with_frame", "live", [0.008])])
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------
# perf_suite: the comparison branches the existing tests do not reach
# --------------------------------------------------------------------------

MACHINE = {
    "rapidshot": "2.6.0", "processor": "CPU", "platform": "OS", "gpu": "GPU",
    "python": "3.13.0", "numpy": "2.0.0", "cpu_topology": "uniform",
    "pinned_to_performance_cores": True,
}


def compare(tmp_path, capsys, monkeypatch, baseline_rows, current_rows,
            baseline_machine=None, now_machine=None):
    base = dict(MACHINE, **(baseline_machine or {}))
    now = dict(MACHINE, **(now_machine or {}))
    monkeypatch.setattr(perf_suite, "machine_info", lambda: now)
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"machine": base, "results": baseline_rows}))
    current = []
    for name, ms in current_rows:
        r = perf_suite.Result(name, "synthetic", [ms / 1000.0])
        current.append(r)
    regressions = perf_suite.print_comparison(current, path)
    return capsys.readouterr().out, regressions


def row(name, ms):
    return {"name": name, "kind": "synthetic", "min_ms": ms, "median_ms": ms}


def test_other_hardware_prints_the_difference_and_gates_nothing(tmp_path, capsys, monkeypatch):
    out, regressions = compare(tmp_path, capsys, monkeypatch,
                               [row("conv.rgb", 2.0)], [("conv.rgb", 8.0)],
                               baseline_machine={"gpu": "Other GPU"})

    assert "gpu: baseline 'Other GPU' vs now 'GPU'" in out
    assert "CROSS-MACHINE COMPARISON" in out
    assert "SLOWER 4.00x (cross-machine: indicative)" in out
    assert "No verdict gated" in out and regressions == 0


def test_a_hybrid_cpu_against_a_baseline_without_pinning_is_not_gated(tmp_path, capsys, monkeypatch):
    base = {k: v for k, v in MACHINE.items() if k != "pinned_to_performance_cores"}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"machine": base, "results": [row("conv.rgb", 2.0)]}))
    monkeypatch.setattr(perf_suite, "machine_info", lambda: dict(MACHINE, cpu_topology="hybrid"))

    regressions = perf_suite.print_comparison([perf_suite.Result("conv.rgb", "synthetic", [0.008])], path)
    out = capsys.readouterr().out

    assert "UNKNOWN BASELINE SCHEDULING COMPARISON" in out
    assert "how it was pinned cannot be recovered" in out
    assert regressions == 0


def test_different_pinning_on_the_same_hardware_is_not_gated(tmp_path, capsys, monkeypatch):
    out, regressions = compare(tmp_path, capsys, monkeypatch,
                               [row("conv.rgb", 2.0)], [("conv.rgb", 8.0)],
                               now_machine={"pinned_to_performance_cores": False})

    assert "DIFFERENT CPU SCHEDULING COMPARISON" in out and regressions == 0


def test_new_rows_zero_timings_drift_and_unchanged_rows(tmp_path, capsys, monkeypatch):
    out, regressions = compare(
        tmp_path, capsys, monkeypatch,
        [row(perf_suite.CONTROL_BENCHMARK, 1.0), row("conv.rgb", 2.0), row("conv.zero", 0.0)],
        [(perf_suite.CONTROL_BENCHMARK, 2.0), ("conv.rgb", 4.1), ("conv.zero", 1.0),
         ("conv.brand_new", 3.0)])

    assert "Control benchmark moved 2.00x" in out and "machine is SLOWER now" in out
    assert "'adjusted' divides out that drift" in out
    assert "conv.brand_new" in out and "new" in out
    assert "conv.zero" not in out.split("verdict")[1], "a zero timing is skipped"
    assert "~ same" in out, "2x slower at 2x machine drift is unchanged code"
    assert regressions == 0
