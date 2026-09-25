"""Decide what a finished measurement is, before it is committed.

This is a **structural and physical** check, not a statistical one. It answers
"is this row internally coherent and physically possible?" and deliberately
answers nothing about confidence, significance or regression. Those need
repeated runs and belong in the statistics work; a validator that implied them
from a single row would be manufacturing certainty out of arithmetic.

Four outcomes, and the distinctions between them are the point:

``passed``
    A coherent measurement. The only status that may be quoted as evidence.
``unavailable``
    The path cannot run on this machine and that is not a fault -- a CUDA
    export on an Optimus laptop is refused *by design*. Kept distinct from
    ``failed`` because treating a designed refusal as a bug is how a real
    failure gets lost in the noise of expected ones.
``failed``
    The worker errored, crashed or exited non-zero.
``invalid``
    The worker claimed success and produced numbers that cannot be true --
    non-finite values, percentiles out of order, a frame rate its own frame
    count contradicts. This is the status that exists because a benchmark
    reporting a confidently wrong number is worse than one that crashes.
``contaminated``
    The measurement is coherent but the conditions it was taken under are
    suspect: background load, a source that could not keep up, a topology that
    could not be read. **Retained and reported, never promoted and never
    silently dropped.** Whether a contaminated row may be used is a policy
    decision made elsewhere, with the reasons in hand.

Every check records what it looked at, so a rejected result carries its own
explanation into the committed record instead of a bare status.
"""

from __future__ import annotations

import dataclasses
import math
import re

#: Percentile keys, in the order they must not decrease.
PERCENTILES = ("p50", "p95", "p99")

_SUFFIXED = re.compile(r"^(?P<prefix>.+)_(?P<percentile>p50|p95|p99)$")

#: How far a derived frame rate may disagree with frames/elapsed before the row
#: is called incoherent. Generous on purpose: the two are computed over slightly
#: different windows in several runners, and this check exists to catch a rate
#: that was never divided by anything, not to police rounding.
THROUGHPUT_TOLERANCE = 0.05

#: Stage timings may exceed the end-to-end figure by this fraction before the
#: row is rejected. Stages are timed with their own clocks inside the same
#: interval, so small overshoot is measurement, not contradiction.
STAGE_TOLERANCE = 0.05


@dataclasses.dataclass
class Validation:
    """The verdict, and everything that was looked at to reach it."""

    status: str
    reasons: list = dataclasses.field(default_factory=list)
    checks: list = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict:
        return {"status": self.status, "reasons": list(self.reasons),
                "checks": list(self.checks)}


def _finite_number(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _non_finite_paths(node, trail="") -> list:
    """Every location holding NaN or Infinity, named so it can be fixed.

    Worth finding explicitly rather than letting the JSON writer refuse: the
    writer says the payload is unserialisable, which sends whoever reads the
    error looking at the store instead of at the metric that is wrong.
    """
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            found += _non_finite_paths(value, f"{trail}.{key}" if trail else str(key))
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            found += _non_finite_paths(value, f"{trail}[{index}]")
    elif isinstance(node, float) and not math.isfinite(node):
        found.append(trail or "<root>")
    return found


def percentile_families(result: dict) -> dict:
    """Find every percentile family in a row, however it is spelled.

    Two spellings are in use across the runners -- flat ``ms_p50``/``ms_p95``
    and nested ``present_to_ready_ms: {p50, p95, ...}`` -- and both are found
    here so neither escapes the ordering check by virtue of its shape.
    """
    families = {}
    for key, value in result.items():
        match = _SUFFIXED.match(key)
        if match and _finite_number(value):
            families.setdefault(match.group("prefix"), {})[match.group("percentile")] = value
        elif isinstance(value, dict):
            nested = {name: value[name] for name in PERCENTILES
                      if name in value and _finite_number(value[name])}
            if nested:
                families.setdefault(key, {}).update(nested)
    return families


def _check_percentile_order(result: dict, checks: list) -> list:
    problems = []
    for prefix, family in sorted(percentile_families(result).items()):
        present = [name for name in PERCENTILES if name in family]
        values = [family[name] for name in present]
        ordered = all(earlier <= later for earlier, later in zip(values, values[1:]))
        checks.append({"check": "percentile-order", "family": prefix,
                       "values": dict(zip(present, values)), "ok": ordered})
        if not ordered:
            problems.append(f"{prefix} percentiles are out of order: "
                            + ", ".join(f"{n}={v}" for n, v in zip(present, values)))
    return problems


def _check_throughput(result: dict, checks: list) -> list:
    """A rate must agree with the count and the window it was derived from."""
    problems = []
    pairs = (("frames", "fps"), ("unique_frames", "unique_fps"))
    elapsed = result.get("elapsed_seconds")
    if not _finite_number(elapsed) or elapsed <= 0:
        return problems
    for count_key, rate_key in pairs:
        count, rate = result.get(count_key), result.get(rate_key)
        if not (_finite_number(count) and _finite_number(rate)) or rate <= 0:
            continue
        derived = count / elapsed
        agrees = abs(derived - rate) <= THROUGHPUT_TOLERANCE * max(derived, rate)
        checks.append({"check": "throughput-consistency", "rate": rate_key,
                       "reported": rate, "derived": derived, "ok": agrees})
        if not agrees:
            problems.append(f"{rate_key}={rate} disagrees with {count_key}/elapsed_seconds"
                            f"={derived:.4g}")
    return problems


def _check_stages(result: dict, end_to_end: str, checks: list) -> list:
    """Stage timings must fit inside the end-to-end figure they decompose.

    The direction matters. This does **not** reconstruct an end-to-end number by
    adding stage percentiles -- percentiles are not additive, and a p99 built
    that way describes a frame that never happened. It only catches the
    contradiction: a decomposition whose parts do not fit inside the whole means
    the stages and the total were not measuring the same interval.
    """
    problems = []
    stages = result.get("stages")
    family = percentile_families(result).get(end_to_end, {})
    total = family.get("p50")
    if not isinstance(stages, dict) or not _finite_number(total) or total <= 0:
        return problems
    parts = {}
    for name, value in stages.items():
        if _finite_number(value):
            parts[name] = value
        elif isinstance(value, dict) and _finite_number(value.get("p50")):
            parts[name] = value["p50"]
    if not parts:
        return problems
    combined = sum(parts.values())
    fits = combined <= total * (1.0 + STAGE_TOLERANCE)
    checks.append({"check": "stages-within-end-to-end", "end_to_end": end_to_end,
                   "end_to_end_p50": total, "stage_p50_sum": combined,
                   "stages": parts, "ok": fits})
    if not fits:
        problems.append(f"stage p50 total {combined:.4g} ms exceeds {end_to_end} p50 "
                        f"{total:.4g} ms; the stages and the total did not measure the "
                        "same interval")
    return problems


def validate_result(result, *, required=(), contamination=(), end_to_end=None) -> Validation:
    """Classify one finished measurement.

    ``required`` names the metrics that must be present, numeric, finite and
    strictly positive for this category -- a benchmark that reports zero frames
    in a positive interval measured nothing. ``contamination`` carries reasons
    the *conditions* were suspect, gathered by the caller; they downgrade an
    otherwise coherent result rather than discarding it.

    ``end_to_end`` names the percentile family that stage timings decompose, if
    the category has one.
    """
    checks: list = []

    if not isinstance(result, dict):
        return Validation("invalid", [f"result is {type(result).__name__}, not an object"],
                          checks)

    # Two spellings are in use: section7.py sets `unavailable`, ai_ingestion.py
    # sets `skipped`. Both mean the same thing -- this path cannot run here --
    # and both must stay distinct from a failure.
    for key in ("unavailable", "skipped"):
        if result.get(key):
            reason = result[key]
            return Validation("unavailable",
                              [str(reason) if reason is not True else "path unavailable"],
                              checks)

    if result.get("error"):
        return Validation("failed", [str(result["error"])], checks)

    returncode = result.get("returncode")
    if isinstance(returncode, int) and returncode != 0:
        return Validation("failed", [f"worker exited with status {returncode}"], checks)

    problems = []

    non_finite = _non_finite_paths(result)
    checks.append({"check": "finite-metrics", "offenders": non_finite, "ok": not non_finite})
    if non_finite:
        problems.append("non-finite values at " + ", ".join(sorted(non_finite)))

    missing = []
    for key in required:
        value = result.get(key)
        if not _finite_number(value):
            missing.append(f"{key} is missing or not a finite number")
        elif value <= 0:
            missing.append(f"{key}={value} is not positive")
    checks.append({"check": "required-metrics", "required": list(required),
                   "problems": missing, "ok": not missing})
    problems += missing

    problems += _check_percentile_order(result, checks)
    problems += _check_throughput(result, checks)
    if end_to_end:
        problems += _check_stages(result, end_to_end, checks)

    if problems:
        return Validation("invalid", problems, checks)

    reasons = [str(reason) for reason in contamination if reason]
    if reasons:
        return Validation("contaminated", reasons, checks)
    return Validation("passed", [], checks)


# ---------------------------------------------------------------------------
# Workload coverage
# ---------------------------------------------------------------------------

#: How much of the captured area must actually be changing before a throughput
#: figure describes a capture path rather than a still desktop. Desktop
#: Duplication reports only what changed, so a source covering a corner of the
#: screen makes every path look fast for a reason that has nothing to do with
#: the path.
MINIMUM_ANIMATED_FRACTION = 0.9


def animated_fraction(animated_rect, captured_rect) -> float:
    """What proportion of the captured area the source was actually animating.

    Both rectangles are ``(left, top, right, bottom)`` in physical pixels.
    Returns the overlap as a fraction of the *captured* area, so a source
    larger than the capture still reports 1.0 -- what matters is whether the
    thing being measured was moving, not whether the source wasted effort
    outside it.
    """
    left = max(animated_rect[0], captured_rect[0])
    top = max(animated_rect[1], captured_rect[1])
    right = min(animated_rect[2], captured_rect[2])
    bottom = min(animated_rect[3], captured_rect[3])
    captured = ((captured_rect[2] - captured_rect[0])
                * (captured_rect[3] - captured_rect[1]))
    if captured <= 0:
        return 0.0
    overlap = max(0, right - left) * max(0, bottom - top)
    return overlap / captured


def coverage_reasons(animated_rect, captured_rect, *,
                     threshold: float = MINIMUM_ANIMATED_FRACTION) -> list:
    """Flag a run whose source was not covering what was being captured.

    This exists because it had happened and nothing noticed. `motion_source.py`
    animated a hardcoded 900x700 window while `memory_profile.py` captured the
    whole screen: on a 2560x1600 display **15% of the captured area was
    moving**, so the compositor reported a small dirty rectangle and every
    library did a fraction of the work a real workload would ask of it. The
    discount also varied with the monitor, so two machines running the same
    command were not running the same benchmark.

    A warning with a number, not a refusal: a deliberately small animated
    region is a legitimate thing to measure. Measuring one by accident is not.
    """
    if animated_rect is None or captured_rect is None:
        return ["the animated and captured areas were not both recorded, so it "
                "cannot be shown that the source covered what was measured"]
    fraction = animated_fraction(animated_rect, captured_rect)
    if fraction >= threshold:
        return []
    return [f"only {fraction * 100:.1f}% of the captured area was being animated "
            f"({list(animated_rect)} against {list(captured_rect)}); Desktop "
            "Duplication reports only what changed, so throughput here reflects a "
            "mostly-still screen rather than the capture path"]


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------

#: Statuses that end the suite unless a policy says otherwise. A device or
#: hardware fault is not an isolated test failure: the next case measures the
#: same broken machine, so continuing produces more bad data, not more coverage.
HARDWARE_MARKERS = ("whea", "device removed", "dxgi_error_device", "device lost",
                    "hung", "cuda error", "out of memory")


def is_hardware_failure(reasons) -> bool:
    """Whether a failure looks like the machine rather than the case.

    Matched on text because that is what the workers actually report. Being
    wrong in the cautious direction stops a suite that could have continued;
    being wrong the other way records a run's worth of measurements taken on
    hardware that had already faulted.
    """
    joined = " ".join(str(reason) for reason in reasons).lower()
    return any(marker in joined for marker in HARDWARE_MARKERS)


def should_stop(status: str, reasons=(), *, continue_on_failure: bool = False) -> bool:
    """Whether the suite must stop after this case.

    ``unavailable`` never stops anything -- it is an expected outcome. A
    hardware failure always stops. An ordinary failure stops unless the caller
    has explicitly opted into continuing, because "keep going and see" is how a
    suite ends up reporting thirty cases measured after the first one broke the
    device.
    """
    if status in ("passed", "contaminated", "unavailable"):
        return False
    if is_hardware_failure(reasons):
        return True
    return not continue_on_failure
