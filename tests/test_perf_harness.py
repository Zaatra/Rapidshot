"""Tests for the benchmark harness's pacing and duty-cycle detection.

This logic decides whether a reported regression is believed, so it gets tested
like production code.

It exists because GRAY reported 8.75, 13.48, 15.08 and 15.70 ms across four runs
of identical code. The cause is *duty cycle*: sustained heavy vector work holds
the CPU in a lower power state, and GRAY has two modes on this machine — about
9.2 ms and about 15.5 ms.

The fix is to pace each rep to a frame period rather than to a fixed idle gap,
because a benchmark's duty cycle in production follows from its own cost:

    RGB    1.8 ms of a 16.7 ms frame   ~11% duty cycle, mostly idle
    GRAY  15.9 ms of a 16.7 ms frame   ~95% duty cycle, effectively sustained

A fixed gap gets GRAY wrong in the flattering direction: a 16 ms gap gives it a
50%% duty cycle and reports 9.16 ms, which a capture loop never achieves.

The memcpy control benchmark cannot catch any of this, because memcpy is not
heavy enough to trigger it.

Timings are driven by a fake clock rather than real sleeps, so the tests are
fast and not flaky.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import perf_suite  # noqa: E402


class FakeClock:
    """A perf_counter stand-in that advances by a scripted amount per rep.

    The harness calls the clock twice per rep — once before the function and
    once after — so alternating calls drive the measured duration exactly.
    """

    def __init__(self, duration_for):
        self.t = 0.0
        self.duration_for = duration_for
        self.reps = 0
        self._mid_rep = False

    def __call__(self):
        if not self._mid_rep:
            self._mid_rep = True
            return self.t
        self._mid_rep = False
        self.t += self.duration_for(self.reps)
        self.reps += 1
        return self.t


@pytest.fixture
def harness(monkeypatch):
    """Install a fake clock and record every sleep the harness asks for."""
    slept = []

    def install(duration_for):
        fake = FakeClock(duration_for)
        monkeypatch.setattr(perf_suite.time, "perf_counter", fake)
        monkeypatch.setattr(perf_suite.time, "sleep", slept.append)
        return fake

    install.slept = slept
    return install


@pytest.fixture(autouse=True)
def clean_state():
    perf_suite._DUTY_SENSITIVE.clear()
    yield
    perf_suite._DUTY_SENSITIVE.clear()


def noop():
    pass


class TestPacing:
    def test_cheap_work_is_measured_mostly_idle(self, harness):
        """RGB costs ~1.8 ms of a 16.7 ms frame: ~11% duty cycle in production."""
        harness(lambda rep: 0.0018)

        perf_suite.time_it(noop, reps=5, period_ms=None)

        # Each rep sleeps out the rest of its frame.
        per_rep = [s for s in harness.slept
                   if abs(s - (perf_suite.FRAME_PERIOD_MS / 1000.0 - 0.0018)) < 1e-9]
        assert len(per_rep) == 5

    def test_work_that_fills_a_frame_gets_no_idle_time(self, harness):
        """GRAY costs ~15.9 ms of a 16.7 ms frame, so it runs near-sustained.

        This is the case a fixed idle gap models wrongly: it would hand GRAY a
        50%% duty cycle and measure 9.16 ms, a number a capture loop never sees.
        """
        harness(lambda rep: 0.0159)

        perf_suite.time_it(noop, reps=5, period_ms=None)

        settle = perf_suite.FRAME_PERIOD_MS * 4 / 1000.0
        per_rep = [s for s in harness.slept if s != settle]
        assert all(s < 0.001 for s in per_rep), "a full frame leaves no idle time"

    def test_work_that_overruns_a_frame_never_sleeps(self, harness):
        """A conversion too slow to keep up must be measured under full load."""
        harness(lambda rep: 0.030)  # 30 ms: nearly two frames

        perf_suite.time_it(noop, reps=5, period_ms=None)

        settle = perf_suite.FRAME_PERIOD_MS * 4 / 1000.0
        assert [s for s in harness.slept if s != settle] == []

    def test_sub_millisecond_benchmarks_are_not_paced(self, harness):
        """Pacing a 0.2 ms memcpy would cost far more than it measures."""
        harness(lambda rep: 0.0002)  # like the control benchmark

        perf_suite.time_it(noop, reps=5, period_ms=None)

        assert harness.slept == [], "cheap benchmarks must sample back-to-back"

    def test_pacing_can_be_forced_off(self, harness):
        """A benchmark that genuinely models sustained throughput may opt out."""
        harness(lambda rep: 0.0018)

        perf_suite.time_it(noop, reps=5, period_ms=0)

        assert harness.slept == []

    def test_the_cpu_is_allowed_to_settle_before_sampling(self, harness):
        """Warm-up is itself a burn loop; sampling must not start inside it."""
        harness(lambda rep: 0.0018)

        perf_suite.time_it(noop, reps=3, period_ms=None)

        assert harness.slept[0] == perf_suite.FRAME_PERIOD_MS * 4 / 1000.0

    def test_returns_one_sample_per_rep(self, harness):
        harness(lambda rep: 0.0018)
        assert len(perf_suite.time_it(noop, reps=7, period_ms=0)) == 7


class TestDutyCycleDetection:
    def test_a_benchmark_that_slows_under_load_is_flagged(self, harness):
        """The GRAY shape: fine when paced, much slower back-to-back."""
        # check_duty_cycle warms 3 reps then samples; make every rep slow.
        harness(lambda rep: 0.016)

        perf_suite.check_duty_cycle(noop, "convert.GRAY", paced_min=0.0095)

        paced, sustained = perf_suite._DUTY_SENSITIVE["convert.GRAY"]
        assert paced == pytest.approx(9.5)
        assert sustained == pytest.approx(16.0)

    def test_a_stable_benchmark_is_not_flagged(self, harness):
        harness(lambda rep: 0.0096)

        perf_suite.check_duty_cycle(noop, "convert.RGB", paced_min=0.0095)

        assert "convert.RGB" not in perf_suite._DUTY_SENSITIVE

    def test_the_check_lets_the_cpu_recover_afterwards(self, harness):
        """Otherwise the check spreads the artefact it exists to detect."""
        harness(lambda rep: 0.016)

        perf_suite.check_duty_cycle(noop, "convert.GRAY", paced_min=0.0095)

        assert harness.slept, "no recovery gap after the burn loop"
        assert harness.slept[-1] == perf_suite.FRAME_PERIOD_MS * 8 / 1000.0


class TestGating:
    """A duty-cycle-sensitive benchmark must never fail the gate."""

    class FakeBaseline:
        """Stands in for the baseline Path: print_comparison only reads it."""

        name = "baseline.json"

        def __init__(self, results):
            import json
            self._text = json.dumps(
                {"machine": {"timestamp": "2026-07-30T00:00:00"},
                 "results": results})

        def read_text(self):
            return self._text

    def _compare(self, baseline_results, current):
        return perf_suite.print_comparison(
            current, self.FakeBaseline(baseline_results), threshold=1.30)

    def _entry(self, name, min_ms, note=""):
        return {"name": name, "kind": "synthetic", "samples": 100,
                "median_ms": min_ms, "min_ms": min_ms, "p95_ms": min_ms,
                "stdev_ms": 0.0, "note": note}

    def test_a_baseline_flag_survives_a_run_that_did_not_flag(self):
        """The detector only fires when a paced sample reached the fast mode.

        A run that stays in the slow mode throughout looks self-consistent and
        goes unflagged — so without honouring the baseline's own flag, it would
        be gated against a baseline that got lucky. This is the exact case that
        failed a build: baseline 9.02 ms, run 14.92 ms, "SLOWER 1.65x".
        """
        baseline = [
            self._entry("control.memcopy", 0.2),
            self._entry("convert.GRAY", 9.02, note="duty-cycle sensitive: ..."),
        ]
        current = [
            perf_suite.Result("control.memcopy", "control", [0.0002]),
            perf_suite.Result("convert.GRAY", "synthetic", [0.01492]),
        ]
        assert perf_suite._DUTY_SENSITIVE == {}, "this run flagged nothing"

        assert self._compare(baseline, current) == 0

    def test_an_unflagged_benchmark_still_gates(self):
        """The exemption must stay narrow, or the suite stops catching anything."""
        baseline = [
            self._entry("control.memcopy", 0.2),
            self._entry("convert.RGB", 1.78),
        ]
        current = [
            perf_suite.Result("control.memcopy", "control", [0.0002]),
            perf_suite.Result("convert.RGB", "synthetic", [0.0350]),
        ]

        assert self._compare(baseline, current) == 1


class TestReporting:
    def test_sensitive_benchmarks_are_annotated_and_warned_about(self):
        """The flag has to reach the operator, not just the dict."""
        result = perf_suite.Result("convert.GRAY", "synthetic", [0.0095, 0.010])
        perf_suite._DUTY_SENSITIVE["convert.GRAY"] = (9.5, 16.3)

        warnings = perf_suite.annotate_duty_cycle([result])

        assert len(warnings) == 1
        assert "convert.GRAY" in warnings[0]
        assert "1.72x" in warnings[0]
        assert "duty-cycle sensitive" in result.note
        # The note travels with the JSON, so a stored baseline records that its
        # own number depends on how it was driven.
        assert "duty-cycle sensitive" in result.to_dict()["note"]

    def test_stable_benchmarks_are_left_alone(self):
        result = perf_suite.Result("convert.RGB", "synthetic", [0.001, 0.001],
                                   note="original")

        assert perf_suite.annotate_duty_cycle([result]) == []
        assert result.note == "original"


# ---------------------------------------------------------------------------
# Rows whose definition changed must not be compared across that change
# ---------------------------------------------------------------------------
#
# `pipeline.gpu_plus_readback` was substantially a CPython allocator benchmark
# until 2.3.0 made `read_back` return bytes instead of a `Vec<f32>` that PyO3
# expanded into 1.2M Python floats per call. Comparing a 2.3.0 run against the
# 2.1.0 `baseline.json` therefore reports **FASTER 6.08x**, which reads as a
# hardware result and is nothing of the kind -- it is a code change that had
# already landed, on one of the two rows Stage 6 was promoted on.
#
# A spurious improvement is the dangerous direction: nobody investigates good
# news. See ROADMAP.md section 10.

class TestRedefinedRows:
    def test_row_redefined_after_the_baseline_is_flagged(self):
        assert perf_suite._redefined_since(
            "pipeline.gpu_plus_readback", "2.1.0") == "2.3.0"
        assert perf_suite._redefined_since(
            "pipeline.cpu_to_nchw", "2.1.0") == "2.3.0"
        assert perf_suite._redefined_since(
            "pipeline.cpu_to_nchw", "2.2.0") == "2.3.0"
        assert perf_suite._redefined_since(
            "pipeline.cpu_to_nchw", "2.2.99") == "2.3.0"

    def test_baseline_at_or_after_the_change_compares_normally(self):
        # The whole point is to suppress only what is genuinely incomparable.
        # Over-suppressing would quietly stop these rows ever gating again.
        assert perf_suite._redefined_since(
            "pipeline.gpu_plus_readback", "2.3.0") is None
        assert perf_suite._redefined_since(
            "pipeline.gpu_plus_readback", "2.4.0") is None
        assert perf_suite._redefined_since(
            "pipeline.cpu_to_nchw", "2.3.0") is None
        assert perf_suite._redefined_since(
            "pipeline.cpu_to_nchw", "2.4.0") is None

    def test_untouched_rows_are_never_flagged(self):
        for name in ("convert.RGB", "shot.GRAY", "control.memcopy"):
            assert perf_suite._redefined_since(name, "1.0.0") is None

    def test_missing_version_counts_as_older(self):
        """A recording from before the field existed cannot be trusted here.

        Absent metadata must fail closed: treating it as "new enough" would
        silently compare exactly the recordings least likely to be comparable.
        """
        for absent in ("", None, "not-a-version"):
            assert perf_suite._redefined_since(
                "pipeline.gpu_plus_readback", absent) == "2.3.0"

    def test_version_ordering(self):
        assert perf_suite._version_tuple("2.10.0") > perf_suite._version_tuple("2.9.0")
        assert perf_suite._version_tuple("") < perf_suite._version_tuple("0.0.1")


# ---------------------------------------------------------------------------
# A caveat must apply in both directions
# ---------------------------------------------------------------------------
#
# The FASTER branch used to carry a comment saying qualifiers were applied
# "in both directions on purpose" while only implementing cross-machine, so a
# live row could print a bare "FASTER 8.70x". Found 2026-08-22 by running the
# verification pass section 2 requires after re-recording a baseline: the same
# row read 8.70x faster on one run and 2.26x slower on the next, on identical
# code, and only the second was labelled.
#
# Nobody investigates good news, which is exactly why the improvement
# direction needs the label more, not less.

# A fully specified machine record. Deriving one from the live machine_info()
# makes these tests depend on the host: where the CPU topology query fails --
# a non-Windows runner, or a Windows box whose GetSystemCpuSetInformation call
# errors -- the record carries `cpu_topology="unknown (...)"` and no
# `pinned_to_performance_cores`, which print_comparison deliberately treats as
# an unknown-scheduling comparison that gates nothing. The gating tests would
# then pass or fail for reasons having nothing to do with the code under test.
SYNTHETIC_MACHINE = {
    "rapidshot": "2.3.0",
    "processor": "SyntheticCPU",
    "platform": "SyntheticOS",
    "gpu": "SyntheticGPU",
    "python": "3.13.0",
    "numpy": "2.0.0",
    "frame": "1920x1080",
    "cpu_topology": "uniform",
    "pinned_to_performance_cores": False,
    "affinity_mask": "0xff",
}


def _compare(tmp_path, capsys, baseline_rows, current_rows, machine=None,
             monkeypatch=None, return_regressions=False):
    """Run print_comparison over hand-built rows and return its output.

    Both sides use SYNTHETIC_MACHINE so the comparison is unambiguously
    same-machine with known scheduling, whatever the host reports.
    """
    import json
    from pathlib import Path

    base_machine = dict(SYNTHETIC_MACHINE)
    base_machine.update(machine or {})
    if monkeypatch is not None:
        monkeypatch.setattr(perf_suite, "machine_info",
                            lambda: dict(SYNTHETIC_MACHINE))

    path = Path(tmp_path) / "baseline.json"
    path.write_text(json.dumps({"machine": base_machine, "results": baseline_rows}))

    current = []
    for row in current_rows:
        result = perf_suite.Result(
            row["name"], row.get("kind", "synthetic"), [row["min_ms"] / 1000.0])
        result.median_ms = row["min_ms"]
        current.append(result)
    regressions = perf_suite.print_comparison(current, path)
    output = capsys.readouterr().out
    if return_regressions:
        return output, regressions
    return output


class TestCaveatsApplyBothWays:
    def test_a_live_row_is_labelled_when_it_reports_faster(self, tmp_path, capsys, monkeypatch):
        out = _compare(
            tmp_path, capsys,
            [{"name": "live.x", "kind": "live", "min_ms": 8.0, "median_ms": 8.0}],
            [{"name": "live.x", "kind": "live", "min_ms": 1.0}],
            monkeypatch=monkeypatch,
        )
        assert "FASTER" in out
        assert "live: informational" in out, (
            "a live row reported an unqualified speed-up; the improvement "
            "direction needs the caveat more, not less")

    def test_a_live_row_is_labelled_when_it_reports_slower(self, tmp_path, capsys, monkeypatch):
        out = _compare(
            tmp_path, capsys,
            [{"name": "live.x", "kind": "live", "min_ms": 1.0, "median_ms": 1.0}],
            [{"name": "live.x", "kind": "live", "min_ms": 8.0}],
            monkeypatch=monkeypatch,
        )
        assert "SLOWER" in out and "live: informational" in out

    def test_a_redefined_row_is_labelled_when_it_reports_faster(self, tmp_path, capsys, monkeypatch):
        out = _compare(
            tmp_path, capsys,
            [{"name": "pipeline.gpu_plus_readback", "min_ms": 14.8, "median_ms": 14.8}],
            [{"name": "pipeline.gpu_plus_readback", "min_ms": 2.8}],
            machine={"rapidshot": "2.1.0"},
            monkeypatch=monkeypatch,
        )
        assert "NOT COMPARABLE" in out

    def test_cpu_row_redefined_in_230_does_not_gate_against_220(
            self, tmp_path, capsys, monkeypatch):
        out, regressions = _compare(
            tmp_path, capsys,
            [{"name": "pipeline.cpu_to_nchw", "min_ms": 1.0,
              "median_ms": 1.0}],
            [{"name": "pipeline.cpu_to_nchw", "min_ms": 8.0}],
            machine={"rapidshot": "2.2.0"},
            monkeypatch=monkeypatch,
            return_regressions=True,
        )
        assert "SLOWER" in out
        assert "redefined in 2.3.0: NOT COMPARABLE" in out
        assert regressions == 0

    def test_a_caveated_row_never_counts_as_a_regression(self, tmp_path, capsys, monkeypatch):
        """The caveat and the gate must agree, or the suite gates on noise."""
        import json
        from pathlib import Path
        monkeypatch.setattr(perf_suite, "machine_info",
                            lambda: dict(SYNTHETIC_MACHINE))
        base_machine = dict(SYNTHETIC_MACHINE)
        path = Path(tmp_path) / "b.json"
        path.write_text(json.dumps({
            "machine": base_machine,
            "results": [{"name": "live.x", "kind": "live",
                         "min_ms": 1.0, "median_ms": 1.0}],
        }))
        result = perf_suite.Result("live.x", "live", [0.008])
        result.median_ms = 8.0
        assert perf_suite.print_comparison([result], path) == 0
        capsys.readouterr()

    def test_an_ordinary_row_still_gates(self, tmp_path, capsys, monkeypatch):
        """The fix must not have made everything informational."""
        import json
        from pathlib import Path
        monkeypatch.setattr(perf_suite, "machine_info",
                            lambda: dict(SYNTHETIC_MACHINE))
        base_machine = dict(SYNTHETIC_MACHINE)
        path = Path(tmp_path) / "b.json"
        path.write_text(json.dumps({
            "machine": base_machine,
            "results": [{"name": "convert.RGB", "kind": "synthetic",
                         "min_ms": 1.0, "median_ms": 1.0}],
        }))
        result = perf_suite.Result("convert.RGB", "synthetic", [0.008])
        result.median_ms = 8.0
        assert perf_suite.print_comparison([result], path) == 1
        capsys.readouterr()


class TestAutoBaselineSelection:
    """`--compare auto` picks the baseline recorded on the current host.

    The release gate named one file, which is one machine. Anywhere else the
    comparison correctly declined to gate, so the step printed a table where
    every verdict was indicative and nothing could fail -- and a gate that
    always passes still gets quoted as evidence that it passed.

    Selection is therefore load-bearing, and its refusals more so than its
    matches: picking the wrong baseline is worse than picking none.
    """

    HOST = {
        "processor": "TestCPU",
        "platform": "Windows-11",
        "gpu": "TestGPU",
        "native_extension": True,
    }

    def _write(self, directory, name, **overrides):
        machine = dict(self.HOST, rapidshot="2.4.0", timestamp="2026-01-01T00:00:00")
        machine.update(overrides)
        (directory / name).write_text(json.dumps({"machine": machine, "results": []}))

    def test_matching_host_is_selected(self, tmp_path):
        self._write(tmp_path, "baseline-mine.json")
        self._write(tmp_path, "baseline-other.json", processor="SomeoneElse")
        assert perf_suite.resolve_auto_baseline(
            self.HOST, tmp_path).name == "baseline-mine.json"

    def test_no_match_refuses_rather_than_falling_back(self, tmp_path):
        # The whole failure being fixed: reaching for another machine's file
        # produces a table of indicative verdicts that gates nothing.
        self._write(tmp_path, "baseline-other.json", gpu="SomeoneElsesGPU")
        with pytest.raises(SystemExit) as excinfo:
            perf_suite.resolve_auto_baseline(self.HOST, tmp_path)
        assert "no baseline recorded on this machine" in str(excinfo.value)
        assert "differs on gpu" in str(excinfo.value)

    def test_native_mismatch_disqualifies(self, tmp_path):
        # baseline.json is recorded with the extension and
        # baseline-nonative.json without; pairing them reports every
        # conversion row 6-20x slower on every run, and every row moves
        # together so it reads as hardware rather than as a mistake.
        self._write(tmp_path, "baseline-nonative.json", native_extension=False)
        with pytest.raises(SystemExit) as excinfo:
            perf_suite.resolve_auto_baseline(self.HOST, tmp_path)
        assert "differs on native_extension" in str(excinfo.value)

    def test_absent_native_flag_is_unknown_not_mismatch(self, tmp_path):
        """Older recordings predate the field and must stay selectable.

        Treating absence as a mismatch would disqualify every baseline
        recorded before it existed -- which is all of them but one.
        """
        machine = dict(self.HOST, rapidshot="2.3.0", timestamp="2026-01-01T00:00:00")
        del machine["native_extension"]
        (tmp_path / "baseline-old.json").write_text(
            json.dumps({"machine": machine, "results": []}))
        assert perf_suite.resolve_auto_baseline(
            self.HOST, tmp_path).name == "baseline-old.json"

    def test_newer_recording_wins(self, tmp_path):
        # An older one is likelier to predate a redefinition and suppress rows.
        self._write(tmp_path, "baseline-old.json", rapidshot="2.3.0")
        self._write(tmp_path, "baseline-new.json", rapidshot="2.4.0")
        assert perf_suite.resolve_auto_baseline(
            self.HOST, tmp_path).name == "baseline-new.json"

    def test_exact_tie_refuses_rather_than_guessing(self, tmp_path):
        """Two equally valid baselines must not be resolved by glob() order.

        Silently taking the first would make the verdict depend on a filename,
        which is the invisible coupling this suite exists to remove.
        """
        self._write(tmp_path, "baseline-a.json")
        self._write(tmp_path, "baseline-b.json")
        with pytest.raises(SystemExit) as excinfo:
            perf_suite.resolve_auto_baseline(self.HOST, tmp_path)
        assert "cannot choose between them" in str(excinfo.value)

    def test_a_different_core_set_is_a_different_apparatus(self, tmp_path):
        """A machine that withholds core 0 (RAPIDSHOT_BENCH_EXCLUDE_CPUS=0,1).

        Its 0xffff baselines must stop gating 0xfffc runs: same hardware, one
        physical core fewer, and nothing in the rows would show it.
        """
        self._write(tmp_path, "baseline-all-cores.json", affinity_mask="0xffff")
        with pytest.raises(SystemExit) as excinfo:
            perf_suite.resolve_auto_baseline(
                dict(self.HOST, affinity_mask="0xfffc"), tmp_path)
        assert "differs on affinity_mask" in str(excinfo.value)
        self._write(tmp_path, "baseline-core0-out.json", affinity_mask="0xfffc",
                    rapidshot="2.3.0")
        assert perf_suite.resolve_auto_baseline(
            dict(self.HOST, affinity_mask="0xfffc"), tmp_path).name == "baseline-core0-out.json"

    def test_unreadable_baseline_is_reported_not_skipped(self, tmp_path):
        # Dropping it silently would hand the comparison to another machine's
        # file and look like a clean match.
        (tmp_path / "baseline-corrupt.json").write_text("{ not json")
        with pytest.raises(SystemExit) as excinfo:
            perf_suite.resolve_auto_baseline(self.HOST, tmp_path)
        assert "unreadable" in str(excinfo.value)
