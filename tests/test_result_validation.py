"""What the validator will and will not call a measurement.

The interesting tests are the ones where a worker exits zero and reports
numbers that cannot be true. A benchmark that crashes gets noticed; one that
confidently reports a frame rate its own frame count contradicts gets quoted.

Nothing here is statistical. The validator sees one row and says whether that
row is coherent -- never whether a difference is real, which needs repeated runs
and belongs elsewhere.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

from result_validation import (percentile_families, is_hardware_failure,  # noqa: E402
                               should_stop, validate_result)

PIXEL_AGE_REQUIRED = ("unique_frames", "unique_fps", "elapsed_seconds")


def row(**overrides):
    base = {"path": "rapidshot-cpu", "unique_frames": 1124, "unique_fps": 140.5,
            "elapsed_seconds": 8.0, "returncode": 0,
            "present_to_ready_ms": {"p50": 33.57, "p95": 36.13, "p99": 38.05}}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# The four outcomes
# ---------------------------------------------------------------------------

def test_a_coherent_row_passes():
    verdict = validate_result(row(), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "passed" and verdict.ok and not verdict.reasons


def test_unavailable_is_not_a_failure():
    """A CUDA export on an Optimus laptop is refused by design, not broken."""
    verdict = validate_result(row(unavailable="CrossAdapterRequired"),
                              required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "unavailable"
    assert verdict.reasons == ["CrossAdapterRequired"]


def test_an_error_is_a_failure():
    verdict = validate_result(row(error="RuntimeError: device removed"))
    assert verdict.status == "failed"


def test_a_nonzero_exit_is_a_failure_even_with_plausible_numbers():
    verdict = validate_result(row(returncode=3), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "failed"


def test_contamination_downgrades_but_does_not_discard():
    verdict = validate_result(row(), required=PIXEL_AGE_REQUIRED,
                              contamination=["source achieved 128.7/s against 165 requested"])
    assert verdict.status == "contaminated"
    assert verdict.reasons == ["source achieved 128.7/s against 165 requested"]
    assert verdict.ok is False


def test_an_incoherent_row_is_invalid_before_it_is_contaminated():
    """Contamination is about conditions; invalidity is about the numbers."""
    verdict = validate_result(row(unique_fps=-1.0), required=PIXEL_AGE_REQUIRED,
                              contamination=["background load"])
    assert verdict.status == "invalid"


# ---------------------------------------------------------------------------
# Numbers that cannot be true
# ---------------------------------------------------------------------------

def test_non_finite_metrics_are_located_by_name():
    verdict = validate_result(row(jitter_ms=float("inf")), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"
    assert "jitter_ms" in verdict.reasons[0]


def test_non_finite_metrics_are_found_when_nested():
    bad = row()
    bad["present_to_ready_ms"] = {"p50": float("nan"), "p95": 36.1, "p99": 38.0}
    verdict = validate_result(bad, required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"
    assert "present_to_ready_ms.p50" in verdict.reasons[0]


@pytest.mark.parametrize("missing", PIXEL_AGE_REQUIRED)
def test_a_missing_required_metric_is_invalid(missing):
    incomplete = row()
    del incomplete[missing]
    verdict = validate_result(incomplete, required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"
    assert any(missing in reason for reason in verdict.reasons)


def test_zero_frames_in_a_positive_interval_measured_nothing():
    verdict = validate_result(row(unique_frames=0, unique_fps=0.0),
                              required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"


def test_percentiles_out_of_order_are_invalid():
    verdict = validate_result(row(present_to_ready_ms={"p50": 40.0, "p95": 36.0, "p99": 38.0}),
                              required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"
    assert "out of order" in verdict.reasons[0]


def test_flat_percentile_spelling_is_checked_too():
    """`ms_p50`/`ms_p95` and the nested form are both in use across the runners."""
    flat = {"path": "x", "frames": 1000, "fps": 125.0, "elapsed_seconds": 8.0,
            "ms_p50": 9.9, "ms_p95": 7.0, "ms_p99": 13.4}
    assert set(percentile_families(flat)) == {"ms"}
    assert validate_result(flat, required=("frames", "fps")).status == "invalid"


def test_a_rate_that_contradicts_its_own_frame_count_is_invalid():
    # 1124 frames in 8 s is 140.5/s, not 400.
    verdict = validate_result(row(unique_fps=400.0), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "invalid"
    assert "disagrees" in verdict.reasons[0]


def test_small_disagreement_between_rate_and_count_is_tolerated():
    verdict = validate_result(row(unique_fps=142.0), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "passed"


# ---------------------------------------------------------------------------
# Stage decomposition
# ---------------------------------------------------------------------------

def test_stages_that_do_not_fit_inside_the_total_are_invalid():
    verdict = validate_result(
        row(stages={"capture": 20.0, "convert": 18.0, "transfer": 6.0}),
        required=PIXEL_AGE_REQUIRED, end_to_end="present_to_ready_ms")
    assert verdict.status == "invalid"
    assert "did not measure the same interval" in verdict.reasons[0]


def test_stages_that_fit_are_fine():
    verdict = validate_result(
        row(stages={"capture": 2.07, "convert": 1.88, "transfer": 0.36, "decode": 1.23}),
        required=PIXEL_AGE_REQUIRED, end_to_end="present_to_ready_ms")
    assert verdict.status == "passed"


def test_the_stage_check_is_skipped_when_no_end_to_end_family_is_named():
    # Absent an end-to-end figure there is nothing to check against, and the
    # validator does not invent one by adding the stages together.
    verdict = validate_result(row(stages={"capture": 999.0}), required=PIXEL_AGE_REQUIRED)
    assert verdict.status == "passed"


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------

def test_unavailable_and_contaminated_never_stop_the_suite():
    assert should_stop("unavailable") is False
    assert should_stop("contaminated", ["background load"]) is False
    assert should_stop("passed") is False


def test_an_ordinary_failure_stops_unless_the_caller_opted_in():
    assert should_stop("failed", ["worker exit 1"]) is True
    assert should_stop("failed", ["worker exit 1"], continue_on_failure=True) is False


@pytest.mark.parametrize("reason", [
    "WHEA-Logger recorded a new machine check",
    "DXGI_ERROR_DEVICE_REMOVED",
    "CUDA error: out of memory",
])
def test_a_hardware_failure_stops_even_under_a_continue_policy(reason):
    """The next case measures the same broken machine, so continuing adds bad data."""
    assert is_hardware_failure([reason]) is True
    assert should_stop("failed", [reason], continue_on_failure=True) is True


def test_an_ordinary_message_is_not_a_hardware_failure():
    assert is_hardware_failure(["worker result is missing valid measurements"]) is False
