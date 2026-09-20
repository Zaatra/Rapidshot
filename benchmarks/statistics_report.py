"""Whether a difference between two configurations is real, and by how much.

The unit of observation here is **one run**, never one frame. Consecutive frame
timings are heavily autocorrelated -- they share a thermal state, a scheduler
decision, a compositor cadence -- so treating a few thousand of them as
independent draws produces an interval narrow enough to make any difference
look certain. ROADMAP § 7.0 already shows what that costs: a single 5-second
pass had the semaphore path as the lowest-latency path, and three passes did
not support it. The frames were never the experiment; the pass was.

So: each run contributes one number per metric, runs are paired and
interleaved, and the interval is computed across the paired differences with a
t distribution because five pairs is not a large sample.

**Spread is not uncertainty.** An IQR, a standard deviation, or the max-min
bracket § 7.0 reports across passes all describe how much the numbers moved.
None of them is a confidence interval, and this module does not print one
beside the other in a way that invites the confusion.

**Three things must line up before an improvement is declared:**

1. The interval on the paired difference clears a **predefined practical
   threshold** -- not merely zero. A statistically detectable 0.05 ms is not a
   finding about a capture pipeline.
2. The **unchanged-code calibration** for that metric, on that configuration,
   supports resolving a difference that size. A machine whose repeat runs of
   identical code move by 3 ms cannot report a 1 ms improvement.
3. There are **enough pairs for the metric in question**. Five is a starting
   estimate for a median; it is not enough for a p99, and a tail metric with
   too few pairs is reported inconclusive rather than reported quietly.

Anything else is ``inconclusive``. That is a result, not a failure to get one.

**The stopping rule is enforced, not requested.** :func:`compare_paired_runs`
takes the number of pairs that were *planned* and refuses to return a verdict
if a different number arrived. Running until the answer looks good is the
easiest way to manufacture a finding, and an honour system does not prevent it.
"""

from __future__ import annotations

import dataclasses
import json
import math
import statistics

SCHEMA_VERSION = 1

#: Two-sided critical values, by degrees of freedom. A table rather than a
#: dependency: scipy is not installed on the benchmark machines, and the
#: alternative -- using 1.96 at n=5 -- understates the interval by about 30%.
_T_TABLE = {
    0.95: {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
           8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
           14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
           20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
           26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021,
           60: 2.000, 120: 1.980},
    0.99: {1: 63.657, 2: 9.925, 3: 5.841, 4: 4.604, 5: 4.032, 6: 3.707, 7: 3.499,
           8: 3.355, 9: 3.250, 10: 3.169, 11: 3.106, 12: 3.055, 13: 3.012,
           14: 2.977, 15: 2.947, 16: 2.921, 17: 2.898, 18: 2.878, 19: 2.861,
           20: 2.845, 25: 2.787, 30: 2.750, 40: 2.704, 60: 2.660, 120: 2.617},
}
_T_INFINITY = {0.95: 1.960, 0.99: 2.576}

#: Metrics whose value comes from the tail of a within-run distribution. Five
#: runs can place a median; they cannot place a p99, because each run's p99 is
#: itself estimated from the handful of slowest frames in that run.
TAIL_MARKERS = ("p95", "p99", "p999", "max", "jitter")

#: Minimum pairs before a tail metric may be called at all. Still a floor, not
#: a sufficiency claim: a p99 comparison at ten pairs is a weak instrument and
#: is labelled as one.
TAIL_MINIMUM_PAIRS = 10

#: Minimum pairs for any verdict.
MINIMUM_PAIRS = 5

#: Minimum unchanged-code runs before a calibration means anything.
MINIMUM_CALIBRATION_RUNS = 5

#: Lower is better for latency and cost; higher is better for throughput. Got
#: wrong, a report announces a regression as an improvement, so the direction
#: is declared rather than guessed from the name.
HIGHER_IS_BETTER = ("fps", "unique_fps", "frames", "unique_frames",
                    "detected_fps", "unique_detected_fps", "throughput")


def _t_critical(confidence: float, degrees_of_freedom: int) -> float:
    table = _T_TABLE.get(confidence)
    if table is None:
        raise ValueError(f"no critical values tabulated for {confidence}")
    if degrees_of_freedom < 1:
        return float("inf")
    # Nearest tabulated df at or below, which is the conservative direction:
    # a slightly wider interval never turns a non-finding into a finding.
    candidates = [df for df in table if df <= degrees_of_freedom]
    if not candidates:
        return table[min(table)]
    if degrees_of_freedom > max(table):
        return _T_INFINITY[confidence]
    return table[max(candidates)]


def higher_is_better(metric: str) -> bool:
    name = metric.rsplit(".", 1)[-1].lower()
    return any(marker in name for marker in HIGHER_IS_BETTER)


def is_tail_metric(metric: str) -> bool:
    name = metric.lower()
    return any(marker in name for marker in TAIL_MARKERS)


# ---------------------------------------------------------------------------
# Guarding the unit of observation
# ---------------------------------------------------------------------------

class NotIndependent(ValueError):
    """Raised when per-frame samples are offered as independent observations."""


def reject_frame_level(values, *, label: str = "observations",
                       suspicious_count: int = 200) -> None:
    """Refuse a series that is obviously frames rather than runs.

    A crude check on purpose. Nobody records two hundred independent
    eight-second runs of one configuration, so a series that long is a
    within-run sample array that has been handed to the wrong function -- and
    the resulting interval would be perhaps an order of magnitude too narrow
    while looking entirely respectable.
    """
    if len(values) >= suspicious_count:
        raise NotIndependent(
            f"{len(values)} {label} were offered as independent runs. Consecutive "
            "frame timings share thermal state, scheduling and compositor cadence; "
            "treating them as independent draws produces an interval that is far "
            "too narrow. Pass one value per run instead.")


# ---------------------------------------------------------------------------
# Noise calibration
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetricNoise:
    """How much one metric moves when nothing changed."""

    metric: str
    runs: int
    values: list
    median: float
    spread_abs: float
    spread_relative: float
    #: The smallest difference this configuration can resolve on this metric,
    #: taken as the half-width of a paired interval the same size as this
    #: calibration would have produced. Not a promise; a floor.
    resolution_abs: float
    resolution_relative: float
    note: str = ""

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Calibration:
    """Per-metric repeatability for one configuration, from unchanged code."""

    configuration: str
    runs: int
    metrics: dict = dataclasses.field(default_factory=dict)
    warnings: list = dataclasses.field(default_factory=list)

    @property
    def sufficient(self) -> bool:
        return self.runs >= MINIMUM_CALIBRATION_RUNS

    def resolution(self, metric: str):
        entry = self.metrics.get(metric)
        return None if entry is None else entry.resolution_abs

    def as_dict(self) -> dict:
        return {"configuration": self.configuration, "runs": self.runs,
                "sufficient": self.sufficient,
                "metrics": {name: entry.as_dict()
                            for name, entry in sorted(self.metrics.items())},
                "warnings": list(self.warnings),
                "note": ("repeatability of unchanged code on this configuration; "
                         "spread here is how much the machine moves, not an "
                         "uncertainty about any particular difference")}


def calibrate_noise(runs, *, configuration: str, metrics=None,
                    confidence: float = 0.95) -> Calibration:
    """Establish what unchanged code does on this configuration, per metric.

    ``runs`` is a sequence of dicts, one per **run** of identical code -- at
    least :data:`MINIMUM_CALIBRATION_RUNS` of them. Five is an initial estimate
    and is treated as one: a calibration at five runs is marked sufficient to
    proceed and is not evidence about any tail metric, which is recorded per
    metric rather than left for a reader to infer.

    This is the floor every later comparison is judged against. A difference
    smaller than what identical code produces on this machine is not a finding
    no matter how tidy its interval looks.
    """
    runs = list(runs)
    calibration = Calibration(configuration=configuration, runs=len(runs))
    if len(runs) < MINIMUM_CALIBRATION_RUNS:
        calibration.warnings.append(
            f"{len(runs)} unchanged-code run(s); at least {MINIMUM_CALIBRATION_RUNS} "
            "are needed before this configuration has a noise floor at all")
    names = sorted(metrics) if metrics else sorted(
        {key for run in runs for key, value in run.items() if _numeric(value)})

    for metric in names:
        values = [run[metric] for run in runs if _numeric(run.get(metric))]
        if len(values) < 2:
            calibration.warnings.append(f"{metric}: fewer than two runs reported it")
            continue
        reject_frame_level(values, label=f"values for {metric}")
        median = statistics.median(values)
        deviation = statistics.stdev(values)
        # What a paired comparison of this size could resolve if the difference
        # it was measuring had this much run-to-run movement in it.
        half_width = (_t_critical(confidence, len(values) - 1) * deviation
                      / math.sqrt(len(values)))
        note = ""
        if is_tail_metric(metric):
            note = ("tail metric: each run's own value is estimated from the few "
                    "slowest frames in that run, so this floor is itself noisy")
        calibration.metrics[metric] = MetricNoise(
            metric=metric, runs=len(values), values=list(values),
            median=median, spread_abs=max(values) - min(values),
            spread_relative=(max(values) - min(values)) / median if median else 0.0,
            resolution_abs=half_width,
            resolution_relative=half_width / median if median else 0.0,
            note=note)
    return calibration


def _numeric(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class MetricComparison:
    """One metric, compared across paired runs, with the verdict's reasoning."""

    metric: str
    pairs: int
    baseline_median: float
    candidate_median: float
    differences: list
    mean_difference: float
    confidence: float
    interval: tuple
    #: Positive means the candidate is better, whichever direction that is.
    improvement_abs: float
    improvement_percent: float
    threshold_abs: float
    resolution_abs: float
    verdict: str
    reasons: list = dataclasses.field(default_factory=list)

    def as_dict(self) -> dict:
        data = dataclasses.asdict(self)
        data["interval"] = list(self.interval)
        return data


@dataclasses.dataclass
class PairedReport:
    comparison: str
    baseline: str
    candidate: str
    pairs: int
    planned_pairs: int
    order: list
    metrics: dict = dataclasses.field(default_factory=dict)
    blocked: list = dataclasses.field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.blocked

    def as_dict(self) -> dict:
        return {"schema_version": SCHEMA_VERSION, "comparison": self.comparison,
                "baseline": self.baseline, "candidate": self.candidate,
                "pairs": self.pairs, "planned_pairs": self.planned_pairs,
                "order": list(self.order), "valid": self.valid,
                "blocked": list(self.blocked),
                "metrics": {name: entry.as_dict()
                            for name, entry in sorted(self.metrics.items())},
                "note": ("intervals are computed across independent paired runs, "
                         "not across frames within a run")}


def compare_paired_runs(baseline_runs, candidate_runs, *, calibration=None,
                        thresholds=None, comparison: str = "", baseline: str = "",
                        candidate: str = "", planned_pairs: int = None,
                        order=None, confidence: float = 0.95,
                        metrics=None) -> PairedReport:
    """Compare two configurations across interleaved, paired runs.

    ``baseline_runs[i]`` and ``candidate_runs[i]`` are the two halves of pair
    *i*, measured next to each other in the recorded ``order``. Pairing is what
    lets the comparison survive drift: the machine warms up and clocks fall
    over a long session, and that movement lands on both halves of a pair
    instead of on whichever configuration went last.

    ``thresholds`` maps a metric to the smallest difference worth reporting, in
    that metric's own units, and must be chosen **before** the runs are looked
    at. Absent one, the metric is compared and reported ``inconclusive`` with
    that stated as the reason -- never silently promoted to a finding against a
    threshold of zero.
    """
    report = PairedReport(comparison=comparison, baseline=baseline,
                          candidate=candidate, pairs=min(len(baseline_runs),
                                                         len(candidate_runs)),
                          planned_pairs=planned_pairs if planned_pairs is not None
                          else min(len(baseline_runs), len(candidate_runs)),
                          order=list(order or []))

    if len(baseline_runs) != len(candidate_runs):
        report.blocked.append(
            f"{len(baseline_runs)} baseline run(s) against {len(candidate_runs)} "
            "candidate run(s); an unpaired run cannot be compared")
    if planned_pairs is not None and report.pairs != planned_pairs:
        # The stopping rule, enforced. Stopping early on a good-looking result,
        # or adding runs until one appears, are the same mistake from opposite
        # directions, and both are invisible in the final numbers.
        report.blocked.append(
            f"{report.pairs} pair(s) arrived against {planned_pairs} planned; the "
            "number of runs must be fixed before they are taken, or the comparison "
            "is a search for a favourable stopping point")
    if report.pairs < MINIMUM_PAIRS:
        report.blocked.append(
            f"{report.pairs} pair(s); at least {MINIMUM_PAIRS} are needed for any "
            "verdict")
    if calibration is not None and not calibration.sufficient:
        report.blocked.append(
            f"the noise calibration for this configuration has {calibration.runs} "
            f"run(s); at least {MINIMUM_CALIBRATION_RUNS} are needed before a "
            "difference can be judged against it")

    names = sorted(metrics) if metrics else sorted(
        {key for run in list(baseline_runs) + list(candidate_runs)
         for key, value in run.items() if _numeric(value)})
    thresholds = thresholds or {}

    for metric in names:
        entry = _compare_metric(metric, baseline_runs[:report.pairs],
                                candidate_runs[:report.pairs], calibration,
                                thresholds, confidence)
        if entry is not None:
            if report.blocked:
                entry.verdict = "inconclusive"
                entry.reasons = list(report.blocked) + entry.reasons
            report.metrics[metric] = entry
    return report


def _compare_metric(metric, baseline_runs, candidate_runs, calibration, thresholds,
                    confidence):
    pairs = [(base.get(metric), cand.get(metric))
             for base, cand in zip(baseline_runs, candidate_runs)]
    pairs = [(a, b) for a, b in pairs if _numeric(a) and _numeric(b)]
    if len(pairs) < 2:
        return None
    reject_frame_level(pairs, label=f"pairs for {metric}")

    base_values = [a for a, _ in pairs]
    cand_values = [b for _, b in pairs]
    better_is_higher = higher_is_better(metric)
    # Signed so that positive always means "the candidate is better", whichever
    # direction better is for this metric.
    differences = [(b - a) if better_is_higher else (a - b) for a, b in pairs]

    mean = statistics.fmean(differences)
    if len(differences) > 1:
        deviation = statistics.stdev(differences)
        half = _t_critical(confidence, len(differences) - 1) * deviation \
            / math.sqrt(len(differences))
    else:
        half = float("inf")
    interval = (mean - half, mean + half)

    base_median = statistics.median(base_values)
    percent = (mean / base_median * 100.0) if base_median else 0.0
    threshold = thresholds.get(metric)
    resolution = calibration.resolution(metric) if calibration else None

    entry = MetricComparison(
        metric=metric, pairs=len(pairs), baseline_median=base_median,
        candidate_median=statistics.median(cand_values),
        differences=differences, mean_difference=mean, confidence=confidence,
        interval=interval, improvement_abs=mean, improvement_percent=percent,
        threshold_abs=threshold if threshold is not None else float("nan"),
        resolution_abs=resolution if resolution is not None else float("nan"),
        verdict="inconclusive")
    entry.verdict, entry.reasons = _decide(entry, metric, threshold, resolution)
    return entry


def _decide(entry, metric, threshold, resolution):
    """The rule, in one place, applied the same way to every metric."""
    reasons = []

    if threshold is None:
        return "inconclusive", [
            f"no practical threshold was set for {metric} before the runs were "
            "taken; a difference cannot be called meaningful against a threshold "
            "chosen afterwards"]

    if is_tail_metric(metric) and entry.pairs < TAIL_MINIMUM_PAIRS:
        return "inconclusive", [
            f"{entry.pairs} pair(s) for a tail metric; each run's own {metric} is "
            f"estimated from its slowest frames, so at least {TAIL_MINIMUM_PAIRS} "
            "pairs are needed before the comparison means anything"]

    low, high = entry.interval
    if resolution is not None and math.isfinite(resolution):
        if threshold < resolution:
            reasons.append(
                f"the practical threshold ({threshold:.4g}) is below what unchanged "
                f"code on this configuration can resolve ({resolution:.4g}); the "
                "calibration does not support a difference this small")
            return "inconclusive", reasons
    else:
        reasons.append("no unchanged-code calibration for this metric, so the "
                       "machine's own repeatability is unknown")
        return "inconclusive", reasons

    if low > threshold:
        return "improvement", reasons + [
            f"the whole {int(entry.confidence * 100)}% interval "
            f"[{low:.4g}, {high:.4g}] lies beyond the {threshold:.4g} threshold"]
    if high < -threshold:
        return "regression", reasons + [
            f"the whole {int(entry.confidence * 100)}% interval "
            f"[{low:.4g}, {high:.4g}] lies beyond the threshold on the worse side"]
    if low <= 0 <= high:
        return "inconclusive", reasons + [
            f"the interval [{low:.4g}, {high:.4g}] includes zero"]
    return "inconclusive", reasons + [
        f"the interval [{low:.4g}, {high:.4g}] does not clear the "
        f"{threshold:.4g} threshold"]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_VERDICT_MARK = {"improvement": "better", "regression": "WORSE",
                 "inconclusive": "inconclusive"}


def render_markdown(report: PairedReport, *, units=None, title=None) -> str:
    """A table a reader can check, with the uncertainty beside every number."""
    units = units or {}
    lines = [f"### {title or report.comparison or 'Paired comparison'}", ""]
    lines.append(f"`{report.candidate}` against `{report.baseline}`, "
                 f"{report.pairs} interleaved pairs "
                 f"(planned {report.planned_pairs}).")
    lines.append("")
    if report.order:
        lines.append(f"Run order: `{' '.join(str(item) for item in report.order)}`")
        lines.append("")
    if report.blocked:
        lines.append("**No verdict is available from this comparison.**")
        lines.append("")
        for reason in report.blocked:
            lines.append(f"- {reason}")
        lines.append("")

    lines.append("| metric | baseline | candidate | difference | 95% interval | "
                 "threshold | verdict |")
    lines.append("| --- | ---: | ---: | ---: | :---: | ---: | --- |")
    for name, entry in sorted(report.metrics.items()):
        unit = units.get(name, "")
        low, high = entry.interval
        lines.append(
            f"| {name} | {entry.baseline_median:.4g}{unit} | "
            f"{entry.candidate_median:.4g}{unit} | "
            f"{entry.improvement_abs:+.4g}{unit} ({entry.improvement_percent:+.1f}%) | "
            f"[{low:+.4g}, {high:+.4g}] | "
            f"{'--' if math.isnan(entry.threshold_abs) else f'{entry.threshold_abs:.4g}'}"
            f"{unit} | {_VERDICT_MARK[entry.verdict]} |")

    lines.append("")
    lines.append("Positive difference means the candidate is better. Intervals are "
                 "computed across independent paired runs, not across frames within "
                 "a run; frame timings are autocorrelated and would give an interval "
                 "far too narrow.")
    unresolved = [(name, entry) for name, entry in sorted(report.metrics.items())
                  if entry.verdict == "inconclusive" and entry.reasons]
    if unresolved:
        lines.append("")
        lines.append("Why the inconclusive rows are inconclusive:")
        lines.append("")
        for name, entry in unresolved:
            lines.append(f"- **{name}** -- {entry.reasons[-1]}")
    return "\n".join(lines) + "\n"


def render_calibration_markdown(calibration: Calibration) -> str:
    lines = [f"### Repeatability: `{calibration.configuration}`", "",
             f"{calibration.runs} runs of unchanged code."]
    if not calibration.sufficient:
        lines.append("")
        lines.append(f"**Not enough runs.** At least {MINIMUM_CALIBRATION_RUNS} are "
                     "needed before this configuration has a noise floor.")
    lines += ["", "| metric | median | spread | resolvable difference |",
              "| --- | ---: | ---: | ---: |"]
    for name, entry in sorted(calibration.metrics.items()):
        lines.append(f"| {name} | {entry.median:.4g} | "
                     f"{entry.spread_abs:.4g} ({entry.spread_relative * 100:.1f}%) | "
                     f"{entry.resolution_abs:.4g} |")
    lines += ["", "Spread is how far apart identical runs landed. It is not a "
                  "confidence interval. The last column is the smallest difference a "
                  "paired comparison of this size could resolve on this machine; "
                  "anything below it is not a finding."]
    for warning in calibration.warnings:
        lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def write_report(path, payload) -> None:
    from pathlib import Path
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        target.write_text(payload, encoding="utf-8")
    else:
        target.write_text(json.dumps(payload, indent=2, allow_nan=False),
                          encoding="utf-8")
