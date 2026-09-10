"""A structured profiler for capture loops.

Section 7.1 asks for ``report()`` / ``json()`` / ``summary()`` rather than
printed text, and the reason is in this repository's own history: every
performance claim here is argued from numbers, and numbers that only ever
reached a terminal cannot be diffed, stored beside a baseline, or attached to an
issue.

    from rapidshot.profiling import Profiler

    profiler = Profiler()
    with camera.grab_frame() as frame:
        with profiler.time("grab"):
            ...
        profiler.observe(frame)

    print(profiler.report())        # for a human
    profiler.summary()              # dict, for asserting on
    profiler.json()                 # string, for storing

Three deliberate choices, each taken from a mistake recorded in ROADMAP § 2:

**Percentiles and minimum, never a mean.** Background load can only make a
sample slower, so the minimum is the least contaminated estimate and the tail is
what a real-time consumer actually feels. A mean hides both.

**Small samples are labelled, not silently reported.** A p99 over 12 samples is
the largest of 12 numbers wearing a statistical name. :meth:`summary` marks any
stage below :data:`RELIABLE_SAMPLES` as ``low_confidence``.

**Frame metadata is recorded, not just wall clock.** ``accumulated_frames``
counts display updates the OS coalesced -- frames the consumer never saw. A loop
can look fast while dropping most of what it was supposed to capture, and only
this distinguishes the two.
"""

from __future__ import annotations

import json as _json
import platform
import statistics
import time
from typing import Any, Dict, Iterator, List, Optional

__all__ = ["Profiler", "RELIABLE_SAMPLES"]

#: Below this many samples a percentile describes the sample, not the process.
RELIABLE_SAMPLES = 30


class _Stage:
    """Timings for one named stage."""

    __slots__ = ("name", "samples_ms")

    def __init__(self, name: str) -> None:
        self.name = name
        self.samples_ms: List[float] = []

    def add(self, milliseconds: float) -> None:
        self.samples_ms.append(milliseconds)

    def stats(self) -> Dict[str, Any]:
        if not self.samples_ms:
            return {"count": 0}
        ordered = sorted(self.samples_ms)

        def pick(fraction: float) -> float:
            index = min(int(len(ordered) * fraction), len(ordered) - 1)
            return ordered[index]

        return {
            "count": len(ordered),
            "min_ms": round(ordered[0], 4),
            "p50_ms": round(pick(0.50), 4),
            "p95_ms": round(pick(0.95), 4),
            "p99_ms": round(pick(0.99), 4),
            "max_ms": round(ordered[-1], 4),
            "stdev_ms": round(statistics.pstdev(ordered), 4) if len(ordered) > 1 else 0.0,
            "total_ms": round(sum(ordered), 4),
            "low_confidence": len(ordered) < RELIABLE_SAMPLES,
        }


class _Timer:
    """Context manager returned by :meth:`Profiler.time`."""

    __slots__ = ("_stage", "_started")

    def __init__(self, stage: _Stage) -> None:
        self._stage = stage
        self._started = 0.0

    def __enter__(self) -> "_Timer":
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Recorded even when the block raised. A stage that fails slowly is
        # exactly what someone profiling a flaky capture loop needs to see, and
        # dropping the sample would make the failure invisible in the numbers.
        self._stage.add((time.perf_counter() - self._started) * 1000.0)
        return False


class Profiler:
    """Collects stage timings and frame metadata from a capture loop.

    Not thread-safe by design: a capture loop is one thread, and the locking to
    make it otherwise would cost more than the measurement. Use one profiler per
    thread and merge the summaries.
    """

    def __init__(self, name: str = "capture") -> None:
        self.name = name
        self._stages: Dict[str, _Stage] = {}
        self._frames = 0
        self._empty_grabs = 0
        self._coalesced_updates = 0
        self._changed_fractions: List[float] = []
        self._generations: set = set()
        self._first_generation: Optional[int] = None
        self._started = time.perf_counter()
        self._elapsed: Optional[float] = None

    # -- collection --------------------------------------------------------
    def time(self, stage: str) -> _Timer:
        """Time a block, recording it under ``stage``.

        Nesting is allowed and the stages are independent, so an outer stage
        includes its inner ones -- which is usually what you want when asking
        "how much of grab() is the conversion".
        """
        return _Timer(self._stages.setdefault(stage, _Stage(stage)))

    def record(self, stage: str, milliseconds: float) -> None:
        """Add a timing measured elsewhere, in milliseconds."""
        self._stages.setdefault(stage, _Stage(stage)).add(float(milliseconds))

    def observe(self, frame: Any) -> None:
        """Record what a captured frame says about the loop's health.

        Accepts ``None`` and counts it, because a grab that returned nothing is
        a fact about the loop -- on a still desktop most grabs return None, and
        a profile that silently ignored them would report a frame rate the
        consumer never saw.
        """
        if frame is None:
            self._empty_grabs += 1
            return
        self._frames += 1
        accumulated = getattr(frame, "accumulated_frames", 0) or 0
        if accumulated > 1:
            # accumulated_frames counts display updates folded into this one, so
            # everything beyond the first is an update the consumer never saw.
            self._coalesced_updates += accumulated - 1
        changed = getattr(frame, "changed_fraction", None)
        if changed is not None:
            self._changed_fractions.append(float(changed))
        generation = getattr(frame, "generation", None)
        if generation is not None:
            if self._first_generation is None:
                self._first_generation = int(generation)
            self._generations.add(int(generation))

    def stop(self) -> "Profiler":
        """Freeze the elapsed window. Idempotent; returns self."""
        if self._elapsed is None:
            self._elapsed = time.perf_counter() - self._started
        return self

    def __enter__(self) -> "Profiler":
        self._started = time.perf_counter()
        self._elapsed = None
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.stop()
        return False

    # -- output ------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        """Everything collected, as plain data.

        The shape is stable: `stages` maps a name to its statistics, `frames`
        describes the loop, `environment` records where it ran. Anything absent
        is absent rather than zero, so a caller can tell "not measured" from
        "measured as none".
        """
        elapsed = self._elapsed if self._elapsed is not None else (
            time.perf_counter() - self._started)
        frames: Dict[str, Any] = {
            "captured": self._frames,
            "empty_grabs": self._empty_grabs,
            "coalesced_updates_missed": self._coalesced_updates,
            # Six places, not four. A tight loop over a few frames finishes in
            # well under a tenth of a millisecond, and rounding that to 0.0
            # while still reporting a frame rate derived from it is incoherent
            # -- and divides by zero in anything recomputing the rate.
            "elapsed_seconds": round(elapsed, 6),
        }
        if elapsed > 0:
            frames["fps"] = round(self._frames / elapsed, 2)
        if self._changed_fractions:
            ordered = sorted(self._changed_fractions)
            frames["changed_fraction_median"] = round(
                statistics.median(ordered), 4)
            frames["changed_fraction_max"] = round(ordered[-1], 4)
        if self._generations:
            # More than one generation in a profile means capture rebuilt
            # mid-run, so timings before and after describe different
            # duplicators and should not be pooled.
            frames["generations_seen"] = sorted(self._generations)
            frames["recoveries_during_run"] = len(self._generations) - 1
        return {
            "name": self.name,
            "frames": frames,
            "stages": {name: stage.stats()
                       for name, stage in sorted(self._stages.items())},
            "environment": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "processor": platform.processor(),
            },
        }

    def json(self, **kwargs: Any) -> str:
        """:meth:`summary` as a JSON string.

        ``allow_nan=False`` by default: a NaN in a profile is a bug in the
        measurement, and it should fail here rather than produce JSON that
        other tools reject later.
        """
        kwargs.setdefault("indent", 2)
        kwargs.setdefault("allow_nan", False)
        return _json.dumps(self.summary(), **kwargs)

    def report(self) -> str:
        """A human-readable table. Returned, never printed."""
        data = self.summary()
        frames = data["frames"]
        lines = [f"profile: {data['name']}", "=" * (9 + len(data["name"])), ""]

        lines.append(f"frames captured   : {frames['captured']}")
        if "fps" in frames:
            lines.append(f"frames per second : {frames['fps']}")
        lines.append(f"empty grabs       : {frames['empty_grabs']}"
                     "   (no new content; normal on a still desktop)")
        if frames["coalesced_updates_missed"]:
            lines.append(
                f"updates missed    : {frames['coalesced_updates_missed']}"
                "   (OS coalesced these; the consumer never saw them)")
        if "changed_fraction_median" in frames:
            lines.append(f"changed fraction  : median "
                         f"{frames['changed_fraction_median']}, max "
                         f"{frames['changed_fraction_max']}")
        if frames.get("recoveries_during_run"):
            lines.append(
                f"RECOVERIES        : {frames['recoveries_during_run']} during this run "
                f"(generations {frames['generations_seen']})")
            lines.append("                    timings either side of a recovery "
                         "describe different duplicators")
        lines.append("")

        if not data["stages"]:
            lines.append("no stages timed")
            return "\n".join(lines)

        header = (f"{'stage':22}{'n':>6}{'min':>9}{'p50':>9}{'p95':>9}"
                  f"{'p99':>9}{'max':>9}")
        lines += [header, "-" * len(header)]
        low = []
        for name, stats in data["stages"].items():
            if not stats["count"]:
                continue
            lines.append(f"{name:22}{stats['count']:6d}{stats['min_ms']:9.3f}"
                         f"{stats['p50_ms']:9.3f}{stats['p95_ms']:9.3f}"
                         f"{stats['p99_ms']:9.3f}{stats['max_ms']:9.3f}")
            if stats["low_confidence"]:
                low.append(name)
        lines.append("-" * len(header))
        lines.append("milliseconds. Minimum is the least contaminated estimate; "
                     "the tail is what a")
        lines.append("real-time consumer feels. No means are reported -- they "
                     "hide both.")
        if low:
            lines += ["", f"LOW CONFIDENCE: {', '.join(low)} — fewer than "
                          f"{RELIABLE_SAMPLES} samples.",
                      "A p99 over a handful of samples is the largest of them "
                      "wearing a statistical name."]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<Profiler {self.name!r}: {self._frames} frames, "
                f"{len(self._stages)} stages>")
