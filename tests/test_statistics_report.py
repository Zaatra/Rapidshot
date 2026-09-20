"""The rules that stop a difference being declared before it is one.

Every test here is a way of getting a finding you have not earned: too few
runs, no threshold chosen in advance, a threshold below what the machine can
resolve, a tail metric judged from five passes, an unpaired set, or frames
passed off as runs. All of them have to come back `inconclusive`.

The arithmetic is checked against values computed by hand, so a change to the
interval maths cannot pass by agreeing with itself.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import statistics_report as stats  # noqa: E402


def runs(metric, values, **extra):
    return [dict({metric: value}, **extra) for value in values]


def calibration_for(metric, values, configuration="cfg"):
    return stats.calibrate_noise(runs(metric, values), configuration=configuration)


# ---------------------------------------------------------------------------
# The unit of observation
# ---------------------------------------------------------------------------

def test_frames_offered_as_runs_are_refused():
    """A thousand frames is a within-run sample array, not a thousand experiments."""
    with pytest.raises(stats.NotIndependent) as caught:
        stats.calibrate_noise(runs("age_p50", [33.0 + i * 0.001 for i in range(1200)]),
                              configuration="cfg")
    assert "far too narrow" in str(caught.value)


def test_a_plausible_number_of_runs_is_accepted():
    stats.calibrate_noise(runs("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2]),
                          configuration="cfg")


def test_the_paired_comparison_refuses_frame_level_input_too():
    many = [{"age_p50": 33.0 + i * 0.001} for i in range(400)]
    with pytest.raises(stats.NotIndependent):
        stats.compare_paired_runs(many, many, thresholds={"age_p50": 0.1},
                                  planned_pairs=400)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def test_a_calibration_below_five_runs_is_not_a_noise_floor():
    calibration = calibration_for("age_p50", [33.0, 33.2, 33.1])
    assert calibration.sufficient is False
    assert "at least 5" in calibration.warnings[0]


def test_five_runs_is_sufficient_to_proceed_and_says_so_for_tails():
    calibration = calibration_for("age_p99", [38.0, 38.4, 38.1, 38.9, 38.2])
    assert calibration.sufficient is True
    # Sufficient to proceed is not sufficient as evidence about a tail.
    assert "estimated from the few" in calibration.metrics["age_p99"].note


def test_calibration_reports_spread_and_a_resolvable_difference():
    calibration = calibration_for("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2])
    entry = calibration.metrics["age_p50"]
    assert entry.runs == 5
    assert entry.spread_abs == pytest.approx(0.6)
    # t(4) at 95% is 2.776; stdev of these five is 0.24083.
    assert entry.resolution_abs == pytest.approx(2.776 * 0.24083 / math.sqrt(5),
                                                 rel=0.001)


def test_a_noisier_machine_resolves_less():
    quiet = calibration_for("age_p50", [33.0, 33.1, 33.0, 33.1, 33.0])
    noisy = calibration_for("age_p50", [30.0, 36.0, 31.0, 38.0, 33.0])
    assert noisy.resolution("age_p50") > quiet.resolution("age_p50") * 10


def test_a_calibration_never_calls_its_spread_a_confidence_interval():
    payload = calibration_for("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2]).as_dict()
    assert "not an uncertainty" in payload["note"]


# ---------------------------------------------------------------------------
# Interval arithmetic
# ---------------------------------------------------------------------------

def test_the_interval_uses_t_not_the_normal_approximation():
    """At five pairs, 1.96 would understate the half-width by about 30%."""
    assert stats._t_critical(0.95, 4) == 2.776
    assert stats._t_critical(0.95, 29) == 2.045
    assert stats._t_critical(0.95, 5000) == 1.960


def test_an_unlisted_degrees_of_freedom_rounds_down_not_up():
    # The conservative direction: a wider interval never turns a non-finding
    # into a finding.
    assert stats._t_critical(0.95, 35) == stats._t_critical(0.95, 30)


def test_the_paired_difference_is_computed_by_hand_and_matches():
    base = runs("age_p50", [36.0, 36.2, 36.1, 36.3, 36.4])
    cand = runs("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2])
    report = stats.compare_paired_runs(
        base, cand, calibration=calibration_for("age_p50", [33.0, 33.4, 33.1,
                                                           33.6, 33.2]),
        thresholds={"age_p50": 0.5}, planned_pairs=5)
    entry = report.metrics["age_p50"]
    # Lower is better for a latency, so improvement = baseline - candidate.
    assert entry.differences == pytest.approx([3.0, 2.8, 3.0, 2.7, 3.2])
    assert entry.mean_difference == pytest.approx(2.94)
    assert entry.improvement_percent == pytest.approx(2.94 / 36.2 * 100)


def test_higher_is_better_metrics_flip_the_sign_of_an_improvement():
    base = runs("unique_fps", [108.0, 108.5, 107.8, 108.2, 108.1])
    cand = runs("unique_fps", [140.0, 140.5, 139.8, 140.2, 140.1])
    report = stats.compare_paired_runs(
        base, cand, calibration=calibration_for("unique_fps", [140.0, 140.5, 139.8,
                                                              140.2, 140.1]),
        thresholds={"unique_fps": 1.0}, planned_pairs=5)
    entry = report.metrics["unique_fps"]
    assert entry.improvement_abs > 0                 # more frames is better
    assert entry.verdict == "improvement"


def test_a_frame_rate_that_fell_is_reported_as_a_regression():
    base = runs("unique_fps", [140.0, 140.5, 139.8, 140.2, 140.1])
    cand = runs("unique_fps", [108.0, 108.5, 107.8, 108.2, 108.1])
    report = stats.compare_paired_runs(
        base, cand, calibration=calibration_for("unique_fps", [108.0, 108.5, 107.8,
                                                              108.2, 108.1]),
        thresholds={"unique_fps": 1.0}, planned_pairs=5)
    assert report.metrics["unique_fps"].verdict == "regression"


# ---------------------------------------------------------------------------
# The decision rule
# ---------------------------------------------------------------------------

def solid_comparison(**overrides):
    kwargs = dict(
        baseline_runs=runs("age_p50", [36.0, 36.2, 36.1, 36.3, 36.4]),
        candidate_runs=runs("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2]),
        calibration=calibration_for("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2]),
        thresholds={"age_p50": 0.5}, planned_pairs=5)
    kwargs.update(overrides)
    base = kwargs.pop("baseline_runs")
    cand = kwargs.pop("candidate_runs")
    return stats.compare_paired_runs(base, cand, **kwargs)


def test_a_large_well_calibrated_difference_is_an_improvement():
    entry = solid_comparison().metrics["age_p50"]
    assert entry.verdict == "improvement"
    assert "lies beyond the" in entry.reasons[-1]


def test_no_threshold_means_no_finding():
    """A threshold picked after seeing the runs is not a threshold."""
    entry = solid_comparison(thresholds={}).metrics["age_p50"]
    assert entry.verdict == "inconclusive"
    assert "chosen afterwards" in entry.reasons[0]


def test_a_threshold_below_the_machine_resolution_cannot_be_claimed():
    """A box whose identical runs move by 3 ms cannot report 0.01 ms."""
    entry = solid_comparison(
        calibration=calibration_for("age_p50", [30.0, 36.0, 31.0, 38.0, 33.0]),
        thresholds={"age_p50": 0.01}).metrics["age_p50"]
    assert entry.verdict == "inconclusive"
    assert "calibration does not support" in entry.reasons[0]


def test_an_interval_spanning_zero_is_inconclusive():
    entry = solid_comparison(
        baseline_runs=runs("age_p50", [33.0, 34.5, 32.0, 35.0, 33.5]),
        candidate_runs=runs("age_p50", [34.0, 33.0, 34.5, 32.5, 34.0]),
        thresholds={"age_p50": 0.5}).metrics["age_p50"]
    assert entry.verdict == "inconclusive"
    assert "includes zero" in entry.reasons[-1]


def test_a_real_but_trivial_difference_does_not_clear_the_threshold():
    """Detectable is not the same as worth reporting."""
    entry = solid_comparison(
        baseline_runs=runs("age_p50", [33.20, 33.21, 33.19, 33.20, 33.21]),
        candidate_runs=runs("age_p50", [33.10, 33.11, 33.09, 33.10, 33.11]),
        calibration=calibration_for("age_p50", [33.10, 33.11, 33.09, 33.10, 33.11]),
        thresholds={"age_p50": 1.0}).metrics["age_p50"]
    assert entry.verdict == "inconclusive"
    assert "does not clear the" in entry.reasons[-1]


def test_no_calibration_at_all_is_inconclusive_not_optimistic():
    entry = solid_comparison(calibration=None).metrics["age_p50"]
    assert entry.verdict == "inconclusive"
    assert "repeatability is unknown" in entry.reasons[0]


# ---------------------------------------------------------------------------
# Tail metrics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("metric", ["age_p95", "age_p99", "jitter_ms", "call_max"])
def test_a_tail_metric_needs_more_than_five_pairs(metric):
    report = stats.compare_paired_runs(
        runs(metric, [40.0, 40.2, 40.1, 40.3, 40.4]),
        runs(metric, [37.0, 37.4, 37.1, 37.6, 37.2]),
        calibration=calibration_for(metric, [37.0, 37.4, 37.1, 37.6, 37.2]),
        thresholds={metric: 0.5}, planned_pairs=5)
    entry = report.metrics[metric]
    assert entry.verdict == "inconclusive"
    assert "slowest frames" in entry.reasons[0]


def test_a_tail_metric_with_enough_pairs_can_be_called():
    base = [40.0, 40.2, 40.1, 40.3, 40.4, 40.1, 40.2, 40.0, 40.3, 40.2]
    cand = [37.0, 37.4, 37.1, 37.6, 37.2, 37.3, 37.1, 37.5, 37.2, 37.4]
    report = stats.compare_paired_runs(
        runs("age_p99", base), runs("age_p99", cand),
        calibration=calibration_for("age_p99", cand),
        thresholds={"age_p99": 0.5}, planned_pairs=10)
    assert report.metrics["age_p99"].verdict == "improvement"


def test_a_median_is_not_treated_as_a_tail_metric():
    assert stats.is_tail_metric("age_p50") is False
    assert stats.is_tail_metric("present_to_ready_ms.p99") is True


# ---------------------------------------------------------------------------
# The stopping rule
# ---------------------------------------------------------------------------

def test_more_pairs_than_planned_blocks_every_verdict():
    """Adding runs until one looks good is invisible in the final numbers."""
    report = solid_comparison(
        baseline_runs=runs("age_p50", [36.0, 36.2, 36.1, 36.3, 36.4, 36.2]),
        candidate_runs=runs("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2, 33.3]),
        planned_pairs=5)
    assert report.valid is False
    assert "favourable stopping point" in report.blocked[0]
    assert report.metrics["age_p50"].verdict == "inconclusive"


def test_fewer_pairs_than_planned_blocks_too():
    report = solid_comparison(
        baseline_runs=runs("age_p50", [36.0, 36.2, 36.1, 36.3]),
        candidate_runs=runs("age_p50", [33.0, 33.4, 33.1, 33.6]),
        planned_pairs=5)
    assert report.valid is False


def test_an_unpaired_set_cannot_be_compared():
    report = stats.compare_paired_runs(
        runs("age_p50", [36.0, 36.2, 36.1, 36.3, 36.4]),
        runs("age_p50", [33.0, 33.4, 33.1]),
        thresholds={"age_p50": 0.5})
    assert report.valid is False
    assert "cannot be compared" in report.blocked[0]


def test_fewer_than_five_pairs_is_never_enough():
    report = stats.compare_paired_runs(
        runs("age_p50", [36.0, 36.2, 36.1]), runs("age_p50", [33.0, 33.4, 33.1]),
        thresholds={"age_p50": 0.5}, planned_pairs=3)
    assert report.valid is False
    assert any("at least 5" in reason for reason in report.blocked)


def test_an_insufficient_calibration_blocks_the_whole_comparison():
    report = solid_comparison(calibration=calibration_for("age_p50", [33.0, 33.4]))
    assert report.valid is False
    assert any("noise calibration" in reason for reason in report.blocked)


def test_the_recorded_run_order_survives_into_the_report():
    """Which configuration went first is part of the evidence."""
    report = solid_comparison(order=["A", "B", "B", "A", "A", "B", "B", "A",
                                     "A", "B"])
    assert report.as_dict()["order"][:2] == ["A", "B"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def test_the_table_carries_the_interval_beside_every_number():
    text = stats.render_markdown(solid_comparison(), units={"age_p50": " ms"},
                                 title="Pixel age")
    assert "| age_p50 |" in text
    assert "95% interval" in text
    assert "not across frames within a run" in text


def test_a_blocked_comparison_says_so_before_it_shows_any_numbers():
    text = stats.render_markdown(solid_comparison(planned_pairs=99))
    assert text.index("No verdict is available") < text.index("| metric |")


def test_inconclusive_rows_explain_themselves_in_the_report():
    text = stats.render_markdown(solid_comparison(thresholds={}))
    assert "Why the inconclusive rows are inconclusive" in text
    assert "chosen afterwards" in text


def test_the_calibration_table_refuses_to_be_read_as_uncertainty():
    text = stats.render_calibration_markdown(
        calibration_for("age_p50", [33.0, 33.4, 33.1, 33.6, 33.2]))
    assert "It is not a confidence interval" in text
    assert "anything below it is not a finding" in text


def test_the_report_serialises_without_nan_for_a_missing_threshold():
    """The store writes with allow_nan=False, so NaN here would fail the commit."""
    report = solid_comparison(thresholds={})
    payload = report.as_dict()
    assert math.isnan(payload["metrics"]["age_p50"]["threshold_abs"])
    with pytest.raises(ValueError):
        import json
        json.dumps(payload, allow_nan=False)
