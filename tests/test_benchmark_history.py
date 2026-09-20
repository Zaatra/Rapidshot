"""section7.py against the durable store, with no desktop, GPU or source.

The store's own guarantees are covered in `test_result_store.py`. What is tested
here is the wiring: that a real runner opens a run, commits one case per path
before starting the next, records what the worker actually produced, resumes
without re-measuring, and refuses to call a source-limited run a clean one.

Everything live is replaced: the D3D source, the WHEA guard, the display query
and the worker launch. Nothing here captures a pixel.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import result_store as store  # noqa: E402
import section7  # noqa: E402

MODE = {"width": 2560, "height": 1600, "refresh_hz": 165}


def worker_row(path, fps=140.5, frames=1124, **extra):
    row = {"path": path, "category": "ingestion", "unique_frames": frames, "unique_fps": fps,
           "elapsed_seconds": 8.0, "frames": 1300, "returncode": 0,
           "present_to_ready_ms": {"p50": 33.5, "p95": 36.1, "p99": 38.0},
           "call_ms": {"p50": 2.1, "p95": 3.0, "p99": 3.4},
           "stages": {"capture_call_ms": {"p50": 2.07}}}
    row.update(extra)
    return row


class FakeSource:
    """Stands in for the D3D latency source, including its achieved rate."""

    def __init__(self, logs, args, guard, scene=None, rates=(400.0, 410.0)):
        self.scene = scene
        self.present_log = Path(logs.directory) / "presents.jsonl"
        self.present_log.write_text("")
        self.rates = []
        self._feed = list(rates)

    def start(self):
        self.rates.append(self._feed[0])

    def check(self):
        # The real source reports its achieved rate periodically while the run
        # polls it, so the fake does too.
        self.rates.append(self._feed[min(len(self.rates), len(self._feed) - 1)])

    def close(self):
        pass

    def summary(self):
        return {"rate_samples": list(self.rates),
                "minimum_updates_per_second": min(self.rates) if self.rates else None}


@pytest.fixture
def harness(monkeypatch):
    """Everything section7.main() touches that needs hardware, replaced."""
    monkeypatch.setattr(section7, "display_mode", lambda: dict(MODE))
    monkeypatch.setattr(section7, "HealthGuard",
                        lambda logs: SimpleNamespace(check=lambda force=False: None,
                                                     after_case=lambda row: row))
    state = {"rows": {}, "source_rates": (400.0, 410.0), "spawned": []}

    def make_source(logs, args, guard, scene=None):
        state["source"] = FakeSource(logs, args, guard, scene,
                                     rates=state["source_rates"])
        return state["source"]

    monkeypatch.setattr(section7, "VisualSource", make_source)

    def fake_spawn(path, seconds, warmup, verify, *, logs, motion, index, command,
                   result_parser):
        state["spawned"].append(path)
        motion.check()
        samples_out = Path(command[command.index("--samples-at") + 1]
                           if "--samples-at" in command
                           else command[command.index("--samples-out") + 1])
        section7.write_samples(samples_out,
                               samples=[(7, 0.0, 0.0), (8, 0.0, 0.0)],
                               ages=[33.4, 33.6], tensor_ages=[32.0, 32.2],
                               calls=[2.0, 2.2, 2.4],
                               stages={"capture_call_ms": [2.0, 2.1, 2.2]})
        return dict(state["rows"].get(path, worker_row(path)))

    monkeypatch.setattr(section7, "spawn", fake_spawn)
    return state


def run(tmp_path, paths=("rapidshot-cpu", "dxcam"), extra=()):
    # --no-telemetry by default: conditions sampling is load-sensitive on
    # purpose, and running the whole test suite loads this machine enough that
    # the background-load detector contaminates these cases. That detector
    # working is tested with synthetic series in `test_telemetry.py`; what is
    # tested here is the store wiring, which must not depend on how busy the
    # machine happens to be.
    argv = ["--paths", *paths, "--seconds", "1", "--warmup", "0", "--no-telemetry",
            "--out", str(tmp_path / "out.json"),
            "--history-root", str(tmp_path / "history"), *extra]
    return section7.main(argv=argv)


def only_run_dir(tmp_path):
    runs = sorted((tmp_path / "history" / "runs").iterdir())
    assert len(runs) == 1
    return runs[0]


# ---------------------------------------------------------------------------

def test_every_path_is_committed_as_its_own_case(tmp_path, harness):
    assert run(tmp_path) == 0

    recovery = store.recover_run(only_run_dir(tmp_path))
    assert len(recovery.completed) == 2
    by_path = {case.identity.path: case for case in recovery.completed.values()}
    assert set(by_path) == {"rapidshot-cpu", "dxcam"}
    for case in by_path.values():
        assert case.status == "passed" and case.verified
        assert case.identity.configuration == "2560x1600@165-ingestion-float16-pinned"
        assert case.identity.workload == "motion" and case.identity.repeat == 1
    assert not recovery.interrupted and not recovery.corrupt


def test_the_committed_result_is_what_the_worker_produced(tmp_path, harness):
    run(tmp_path, paths=("rapidshot-cpu",))
    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    committed = json.loads((case.directory / store.RESULT_NAME).read_text())
    assert committed["unique_fps"] == 140.5
    assert committed["validation"]["status"] == "passed"


def test_raw_samples_are_committed_alongside_the_percentiles(tmp_path, harness):
    run(tmp_path, paths=("rapidshot-cpu",))
    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    rows = store.read_samples(case.directory / store.SAMPLES_NAME)
    frames = [row for row in rows if row["kind"] == "unique_frame"]
    calls = [row for row in rows if row["kind"] == "call"]
    # Two populations of different length, which is the point of keeping them
    # apart: three calls returned two unique frames.
    assert [row["frame_id"] for row in frames] == [7, 8]
    assert [row["present_to_ready_ms"] for row in frames] == [33.4, 33.6]
    assert len(calls) == 3 and calls[0]["capture_call_ms"] == 2.0


def test_a_case_is_committed_before_the_next_one_starts(tmp_path, harness,
                                                        monkeypatch):
    """The ordering the whole design exists for, observed from outside."""
    committed_when_second_started = {}
    original = section7.spawn

    def spy(path, *args, **kwargs):
        run_dirs = sorted((tmp_path / "history" / "runs").iterdir())
        recovery = store.recover_run(run_dirs[0])
        committed_when_second_started[path] = set(
            case.identity.path for case in recovery.completed.values())
        return original(path, *args, **kwargs)

    # monkeypatch, not assign-and-restore: a try/finally leaks the patch if the
    # run is killed between them, and this module's other fakes all go through
    # the fixture already.
    monkeypatch.setattr(section7, "spawn", spy)
    run(tmp_path)
    assert committed_when_second_started["rapidshot-cpu"] == set()
    assert committed_when_second_started["dxcam"] == {"rapidshot-cpu"}


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def test_a_source_slower_than_the_path_contaminates_the_case(tmp_path, harness):
    # 120/s of source against a path claiming 140.5 unique fps: the throughput
    # figure describes the apparatus.
    harness["source_rates"] = (120.0, 130.0)
    run(tmp_path, paths=("rapidshot-cpu",))

    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    assert case.status == "contaminated"
    assert "source-limited" in case.reasons[0]
    assert case.usable is False
    # Contaminated, not discarded: the numbers are still there to be weighed.
    assert json.loads((case.directory / store.RESULT_NAME).read_text())["unique_fps"] == 140.5


def test_a_worker_that_exits_zero_with_impossible_numbers_is_invalid(tmp_path, harness):
    harness["rows"]["rapidshot-cpu"] = worker_row("rapidshot-cpu", fps=400.0)
    # The run stops and reports failure; the case is still committed, with the
    # arithmetic that condemned it.
    assert run(tmp_path, paths=("rapidshot-cpu",)) == 1

    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    assert case.status == "invalid"
    assert "disagrees" in case.reasons[0]


def test_an_unavailable_path_does_not_stop_the_suite(tmp_path, harness):
    harness["rows"]["rapidshot-cpu"] = {"path": "rapidshot-cpu", "returncode": 0,
                                        "error": "CrossAdapterRequired", "unavailable": True}
    assert run(tmp_path) == 0
    assert harness["spawned"] == ["rapidshot-cpu", "dxcam"]

    recovery = store.recover_run(only_run_dir(tmp_path))
    statuses = {case.identity.path: case.status for case in recovery.completed.values()}
    assert statuses == {"rapidshot-cpu": "unavailable", "dxcam": "passed"}


def test_an_ordinary_failure_stops_the_suite_and_is_still_recorded(tmp_path, harness):
    harness["rows"]["rapidshot-cpu"] = {"path": "rapidshot-cpu", "returncode": 1,
                                        "error": "worker exit 1"}
    assert run(tmp_path) == 1
    assert harness["spawned"] == ["rapidshot-cpu"]      # dxcam was never started

    recovery = store.recover_run(only_run_dir(tmp_path))
    case = recovery.completed[next(iter(recovery.completed))]
    assert case.status == "failed" and case.verified
    assert "worker exit 1" in case.reasons[0]


def test_continue_on_failure_keeps_going_but_still_records_the_failure(tmp_path, harness):
    harness["rows"]["rapidshot-cpu"] = {"path": "rapidshot-cpu", "returncode": 1,
                                        "error": "worker exit 1"}
    run(tmp_path, extra=["--continue-on-failure"])
    assert harness["spawned"] == ["rapidshot-cpu", "dxcam"]

    recovery = store.recover_run(only_run_dir(tmp_path))
    statuses = {case.identity.path: case.status for case in recovery.completed.values()}
    assert statuses == {"rapidshot-cpu": "failed", "dxcam": "passed"}


def test_a_hardware_failure_stops_even_under_continue_on_failure(tmp_path, harness):
    harness["rows"]["rapidshot-cpu"] = {
        "path": "rapidshot-cpu", "returncode": 1,
        "error": "MotionError: WHEA log changed; stopping live benchmarks"}
    run(tmp_path, extra=["--continue-on-failure"])
    assert harness["spawned"] == ["rapidshot-cpu"]


def test_background_load_reaches_the_case_as_contamination(tmp_path, harness,
                                                           monkeypatch):
    """The detector is load-sensitive, so it is driven rather than waited for."""
    monkeypatch.setattr(section7.telemetry_module, "background_load_warning",
                        lambda record, **kw: ["something else was using the machine"])
    run(tmp_path, paths=("rapidshot-cpu",), extra=["--telemetry-interval", "0.25"])

    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    assert case.status == "contaminated"
    assert case.reasons == ["something else was using the machine"]


def test_a_machine_that_changed_mid_run_contaminates_the_case(tmp_path, harness,
                                                              monkeypatch):
    """A changed display or power state makes the cases either side different."""
    drift = section7.machine_inventory.EnvironmentCheck(
        changed=[{"field": "display_fingerprint", "before": "a", "after": "b"}],
        blocking=[{"field": "display_fingerprint", "before": "a", "after": "b"}])
    monkeypatch.setattr(section7.machine_inventory, "verify_environment",
                        lambda *a, **kw: drift)
    run(tmp_path, paths=("rapidshot-cpu",))

    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    assert case.status == "contaminated"
    assert "display_fingerprint changed mid-run" in case.reasons[0]
    assert "not comparable" in case.reasons[0]
    committed = json.loads((case.directory / store.RESULT_NAME).read_text())
    assert committed["environment_check"]["comparable"] is False


def test_a_worker_on_the_wrong_cores_contaminates_the_case(tmp_path, harness):
    """Affinity is inherited, but "inherited" is an assumption until checked."""
    harness["rows"]["rapidshot-cpu"] = worker_row(
        "rapidshot-cpu", affinity={"available": True, "process_mask": 0xFFFFFFFF,
                                   "process_mask_hex": "0xffffffff"})
    run(tmp_path, paths=("rapidshot-cpu",))

    case = next(iter(store.recover_run(only_run_dir(tmp_path)).completed.values()))
    if case.status == "passed":
        # Skip, not return: a `return` here passes with no assertion at all, and
        # a test that quietly asserts nothing on some machines is worse than one
        # that says why it did not run.
        pytest.skip("this machine applied no affinity, so there is nothing to "
                    "mismatch against")
    assert case.status == "contaminated"
    assert "does not match" in " ".join(case.reasons)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_resuming_skips_completed_cases_and_measures_the_rest(tmp_path, harness):
    harness["rows"]["dxcam"] = {"path": "dxcam", "returncode": 1, "error": "worker exit 1"}
    run(tmp_path)
    assert harness["spawned"] == ["rapidshot-cpu", "dxcam"]

    run_dir = only_run_dir(tmp_path)
    harness["rows"].clear()
    harness["spawned"].clear()
    run(tmp_path, extra=["--resume", str(run_dir)])

    # A failed case is not retried just because a resume ran: that is how a
    # suite ends up looping on a broken machine. It has to be asked for.
    assert harness["spawned"] == []

    run(tmp_path, extra=["--resume", str(run_dir), "--retry-failed"])
    assert harness["spawned"] == ["dxcam"]
    recovery = store.recover_run(run_dir)
    dxcam = [case for case in recovery.completed.values() if case.identity.path == "dxcam"][0]
    assert dxcam.attempt_id == "attempt-002" and dxcam.status == "passed"
    attempts = recovery.attempts[dxcam.case_id]
    assert [case.status for case in attempts] == ["failed", "passed"]


def test_a_repeat_is_a_different_case_not_a_second_attempt(tmp_path, harness):
    run(tmp_path, paths=("rapidshot-cpu",))
    run_dir = only_run_dir(tmp_path)
    run(tmp_path, paths=("rapidshot-cpu",), extra=["--resume", str(run_dir), "--repeat", "2"])

    recovery = store.recover_run(run_dir)
    assert len(recovery.completed) == 2
    assert {case.identity.repeat for case in recovery.completed.values()} == {1, 2}


def test_no_history_still_runs_and_writes_nothing(tmp_path, harness):
    assert run(tmp_path, extra=["--no-history"]) == 0
    assert not (tmp_path / "history").exists()
    assert json.loads((tmp_path / "out.json").read_text())["status"] == "measured"


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def configuration(**overrides):
    args = SimpleNamespace(width=2560, height=1600, motion_fps=165.0, category="ingestion",
                           allow_cpu_nodes=False, codec="png", quality=90, verify=False,
                           no_pin=False, geometry="letterbox", conf=0.25, iou=0.45)
    for key, value in overrides.items():
        setattr(args, key, value)
    payload = {"contract": {"dtype": "float16"}, "model": {"sha256": "a" * 64}}
    return section7.case_configuration(args, payload)


def test_configuration_separates_everything_that_changes_the_measurement():
    base = configuration()
    assert base == "2560x1600@165-ingestion-float16-pinned"
    assert configuration(width=1920, height=1080) != base
    assert configuration(motion_fps=60.0) != base
    assert configuration(category="inference") != base
    assert configuration(verify=True) != base
    # A pinned and an unpinned run on a hybrid CPU are not the same experiment.
    assert configuration(no_pin=True) != base


def test_a_model_hash_is_part_of_the_configuration():
    with_model = configuration(category="inference")
    assert "model-" + "a" * 12 in with_model
    assert configuration(category="inference", allow_cpu_nodes=True) != with_model


@pytest.mark.parametrize("change", [
    {"geometry": "stretch"}, {"conf": 0.5}, {"iou": 0.7},
])
def test_the_detection_contract_is_part_of_the_configuration(change):
    """Two runs that disagree on what is being timed are not repeats."""
    base = configuration(category="inference")
    assert configuration(category="inference", **change) != base
    assert "letterbox" in base and "conf0.25" in base


def test_a_run_with_no_observed_source_rate_cannot_claim_to_be_path_limited():
    motion = SimpleNamespace(rates=[])
    reasons = section7.source_limited(motion, 0, worker_row("rapidshot-cpu"))
    assert reasons and "never reported an achieved rate" in reasons[0]


def test_a_source_comfortably_faster_than_the_path_is_not_flagged():
    motion = SimpleNamespace(rates=[400.0, 410.0])
    assert section7.source_limited(motion, 0, worker_row("rapidshot-cpu")) == []


def test_only_rates_observed_during_this_case_are_considered():
    """A slow window before this case started says nothing about this case."""
    motion = SimpleNamespace(rates=[10.0, 400.0, 410.0])
    assert section7.source_limited(motion, 1, worker_row("rapidshot-cpu")) == []
    assert section7.source_limited(motion, 0, worker_row("rapidshot-cpu"))
