"""The structured profiler from ROADMAP section 7.1.

Headless: the profiler takes timings and duck-typed frames, so none of this
needs a desktop or a GPU. That is deliberate -- a profiler that could only be
tested on capture hardware could not be tested in CI at all.
"""

from __future__ import annotations

import json

import pytest

from rapidshot.profiling import RELIABLE_SAMPLES, Profiler


class FakeFrame:
    def __init__(self, accumulated_frames=1, changed_fraction=None, generation=0):
        self.accumulated_frames = accumulated_frames
        self.changed_fraction = changed_fraction
        self.generation = generation


class TestTiming:
    def test_records_a_sample_per_block(self):
        profiler = Profiler()
        for _ in range(3):
            with profiler.time("grab"):
                pass
        assert profiler.summary()["stages"]["grab"]["count"] == 3

    def test_records_even_when_the_block_raises(self):
        """A stage that fails slowly is the interesting one.

        Dropping the sample would make a failing path invisible in exactly the
        profile someone is reading to find it.
        """
        profiler = Profiler()
        with pytest.raises(ValueError):
            with profiler.time("convert"):
                raise ValueError("boom")
        assert profiler.summary()["stages"]["convert"]["count"] == 1

    def test_record_accepts_timings_measured_elsewhere(self):
        profiler = Profiler()
        profiler.record("upload", 1.5)
        profiler.record("upload", 2.5)
        stats = profiler.summary()["stages"]["upload"]
        assert stats["count"] == 2
        assert stats["min_ms"] == 1.5
        assert stats["max_ms"] == 2.5

    def test_percentiles_are_ordered(self):
        profiler = Profiler()
        for value in range(1, 101):
            profiler.record("stage", float(value))
        stats = profiler.summary()["stages"]["stage"]
        assert stats["min_ms"] <= stats["p50_ms"] <= stats["p95_ms"]
        assert stats["p95_ms"] <= stats["p99_ms"] <= stats["max_ms"]

    def test_no_mean_is_reported(self):
        """Means hide both contamination and the tail; percentiles do not."""
        profiler = Profiler()
        profiler.record("stage", 1.0)
        stats = profiler.summary()["stages"]["stage"]
        assert not any("mean" in key or "avg" in key for key in stats)

    def test_nested_stages_are_independent(self):
        profiler = Profiler()
        with profiler.time("outer"):
            with profiler.time("inner"):
                pass
        stages = profiler.summary()["stages"]
        assert stages["outer"]["count"] == 1
        assert stages["inner"]["count"] == 1
        assert stages["outer"]["total_ms"] >= stages["inner"]["total_ms"]


class TestLowConfidence:
    def test_small_samples_are_flagged(self):
        profiler = Profiler()
        for _ in range(5):
            profiler.record("stage", 1.0)
        assert profiler.summary()["stages"]["stage"]["low_confidence"] is True

    def test_enough_samples_are_not_flagged(self):
        profiler = Profiler()
        for _ in range(RELIABLE_SAMPLES):
            profiler.record("stage", 1.0)
        assert profiler.summary()["stages"]["stage"]["low_confidence"] is False

    def test_report_names_the_low_confidence_stages(self):
        profiler = Profiler()
        profiler.record("thin", 1.0)
        for _ in range(RELIABLE_SAMPLES):
            profiler.record("thick", 1.0)
        report = profiler.report()
        assert "LOW CONFIDENCE" in report
        assert "thin" in report.split("LOW CONFIDENCE")[1]


class TestFrameObservation:
    def test_counts_captured_frames(self):
        profiler = Profiler()
        for _ in range(4):
            profiler.observe(FakeFrame())
        assert profiler.summary()["frames"]["captured"] == 4

    def test_none_is_counted_not_ignored(self):
        """Empty grabs are a fact about the loop, not noise.

        On a still desktop most grabs return None; a profile that dropped them
        would report a frame rate the consumer never experienced.
        """
        profiler = Profiler()
        profiler.observe(None)
        profiler.observe(FakeFrame())
        frames = profiler.summary()["frames"]
        assert frames["captured"] == 1
        assert frames["empty_grabs"] == 1

    def test_counts_updates_the_os_coalesced(self):
        """accumulated_frames > 1 means updates the consumer never saw."""
        profiler = Profiler()
        profiler.observe(FakeFrame(accumulated_frames=3))
        profiler.observe(FakeFrame(accumulated_frames=1))
        assert profiler.summary()["frames"]["coalesced_updates_missed"] == 2

    def test_tracks_changed_fraction(self):
        profiler = Profiler()
        for value in (0.1, 0.3, 0.5):
            profiler.observe(FakeFrame(changed_fraction=value))
        frames = profiler.summary()["frames"]
        assert frames["changed_fraction_median"] == pytest.approx(0.3)
        assert frames["changed_fraction_max"] == pytest.approx(0.5)

    def test_absent_changed_fraction_is_absent_not_zero(self):
        """None means unknown; reporting 0.0 would assert something false."""
        profiler = Profiler()
        profiler.observe(FakeFrame(changed_fraction=None))
        assert "changed_fraction_median" not in profiler.summary()["frames"]

    def test_detects_a_recovery_mid_run(self):
        """Timings either side of a rebuild describe different duplicators."""
        profiler = Profiler()
        profiler.observe(FakeFrame(generation=0))
        profiler.observe(FakeFrame(generation=1))
        frames = profiler.summary()["frames"]
        assert frames["recoveries_during_run"] == 1
        assert frames["generations_seen"] == [0, 1]
        assert "RECOVERIES" in profiler.report()

    def test_no_recovery_is_not_reported_as_one(self):
        profiler = Profiler()
        profiler.observe(FakeFrame(generation=2))
        profiler.observe(FakeFrame(generation=2))
        assert profiler.summary()["frames"]["recoveries_during_run"] == 0

    def test_tolerates_objects_without_the_metadata(self):
        """Anything duck-typed should work, including a bare ndarray."""
        profiler = Profiler()
        profiler.observe(object())
        assert profiler.summary()["frames"]["captured"] == 1


class TestOutputs:
    def test_json_round_trips(self):
        profiler = Profiler("demo")
        profiler.record("grab", 1.0)
        profiler.observe(FakeFrame())
        parsed = json.loads(profiler.json())
        assert parsed["name"] == "demo"
        assert parsed["stages"]["grab"]["count"] == 1

    def test_json_rejects_nan(self):
        """A NaN is a measurement bug and must fail here, not downstream."""
        profiler = Profiler()
        profiler.record("stage", float("nan"))
        with pytest.raises(ValueError):
            profiler.json()

    def test_report_is_returned_not_printed(self, capsys):
        profiler = Profiler()
        profiler.record("grab", 1.0)
        text = profiler.report()
        assert isinstance(text, str) and text
        assert capsys.readouterr().out == ""

    def test_report_survives_an_empty_profile(self):
        assert "no stages timed" in Profiler().report()

    def test_summary_records_where_it_ran(self):
        """Provenance, for the same reason perf_suite records a machine block:
        a timing without a machine cannot be compared to anything."""
        environment = Profiler().summary()["environment"]
        assert environment["platform"] and environment["python"]


class TestElapsedWindow:
    def test_context_manager_bounds_the_window(self):
        with Profiler() as profiler:
            profiler.observe(FakeFrame())
        first = profiler.summary()["frames"]["elapsed_seconds"]
        assert profiler.summary()["frames"]["elapsed_seconds"] == first

    def test_stop_is_idempotent(self):
        profiler = Profiler()
        profiler.stop()
        first = profiler.summary()["frames"]["elapsed_seconds"]
        profiler.stop()
        assert profiler.summary()["frames"]["elapsed_seconds"] == first

    def test_fps_uses_the_measured_window(self):
        import time

        with Profiler() as profiler:
            for _ in range(10):
                profiler.observe(FakeFrame())
            time.sleep(0.02)          # a window long enough to divide by
        frames = profiler.summary()["frames"]
        assert frames["elapsed_seconds"] > 0
        assert frames["fps"] == pytest.approx(
            10 / frames["elapsed_seconds"], rel=0.05)

    def test_elapsed_is_not_rounded_away_on_a_fast_loop(self):
        """A sub-millisecond window must still report a nonzero duration.

        Reporting 0.0 seconds beside a frame rate computed from it is
        incoherent, and divides by zero in anything recomputing the rate.
        """
        with Profiler() as profiler:
            profiler.observe(FakeFrame())
        assert profiler.summary()["frames"]["elapsed_seconds"] > 0
