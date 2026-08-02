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
