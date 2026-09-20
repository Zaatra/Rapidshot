"""What survives a crash, and what a crash is allowed to do to the record.

Every test here is about the same question: after something goes wrong, can a
previously committed measurement be lost, altered, or silently reclassified? The
answer has to be no for all of them, and "something goes wrong" has to include
the cases this machine actually produces -- a killed parent, a full disk, a
half-written journal -- without any of them requiring a real Windows crash to
reproduce.

Nothing here captures, measures or touches a GPU. The store is deliberately
ignorant of what a benchmark is, so its guarantees can be tested at full speed
on a machine with no desktop session.
"""
import errno
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import result_store as store  # noqa: E402
from result_store import CaseIdentity  # noqa: E402


def identity(path="rapidshot-cpu", repeat=1, benchmark="section7",
             configuration="2560x1600@165-fp16", workload="motion"):
    return CaseIdentity(benchmark=benchmark, path=path, configuration=configuration,
                        workload=workload, repeat=repeat)


def sample_result(fps=140.5):
    return {"path": "rapidshot-cpu", "unique_fps": fps, "unique_frames": 1124,
            "elapsed_seconds": 8.0, "present_to_ready_ms": {"p50": 33.5, "p95": 36.1,
                                                            "p99": 38.0}}


# ---------------------------------------------------------------------------
# The ordinary path
# ---------------------------------------------------------------------------

def test_commit_then_recover_round_trips_result_and_samples(tmp_path):
    run = store.open_run(tmp_path, metadata={"note": "unit"})
    handle = run.begin_case(identity())
    samples = [{"frame": index, "age_ms": 33.0 + index * 0.01} for index in range(50)]
    run.commit_case(handle, sample_result(), status="passed", samples=samples)
    run.close()

    recovery = store.recover_run(run.run_dir)
    assert recovery.truncated_tail is False
    assert not recovery.interrupted and not recovery.corrupt
    case = recovery.completed[identity().case_id]
    assert case.status == "passed" and case.verified and case.usable
    assert json.loads((case.directory / store.RESULT_NAME).read_text()) == sample_result()
    assert store.read_samples(case.directory / store.SAMPLES_NAME) == samples


def test_case_id_is_stable_and_separates_every_identity_axis():
    base = identity()
    assert base.case_id == identity().case_id
    for changed in (identity(path="dxcam"), identity(repeat=2), identity(workload="static"),
                    identity(configuration="1920x1080@60-fp16"),
                    identity(benchmark="ai_ingestion")):
        assert changed.case_id != base.case_id


def test_samples_bytes_are_reproducible():
    """gzip stores an mtime in its header; if it leaked in, the manifest hash
    would change on every write and could not prove the file is unaltered.

    Asserts the header field directly rather than sleeping past a second
    boundary -- the `sleep(1.1)` this replaces cost more than every other
    test in this file put together.
    """
    rows = [{"frame": 1, "age_ms": 33.0}]
    first = store._gzip_samples(rows)
    assert store._gzip_samples(rows) == first
    # Bytes 4..8 of a gzip member are MTIME. Zero means "no timestamp".
    assert first[4:8] == bytes(4)


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------

def test_resume_skips_completed_work_and_refuses_to_rerun_it(tmp_path):
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity()), sample_result(), status="passed")
    run.close()

    resumed = store.resume_run(run.run_dir)
    assert resumed.completed_status(identity()) == "passed"
    with pytest.raises(store.CaseAlreadyCommitted):
        resumed.begin_case(identity())


def test_explicit_retry_creates_another_attempt_and_keeps_the_first(tmp_path):
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity()), sample_result(fps=140.0), status="passed")
    second = run.begin_case(identity(), retry=True)
    assert second.attempt_id == "attempt-002"
    run.commit_case(second, sample_result(fps=141.0), status="passed")

    recovery = store.recover_run(run.run_dir)
    attempts = recovery.attempts[identity().case_id]
    assert [case.attempt_id for case in attempts] == ["attempt-001", "attempt-002"]
    first = json.loads((attempts[0].directory / store.RESULT_NAME).read_text())
    assert first["unique_fps"] == 140.0          # the earlier attempt is untouched
    assert recovery.completed[identity().case_id].attempt_id == "attempt-002"


def test_an_attempt_directory_left_by_a_crash_is_never_reused(tmp_path):
    run = store.open_run(tmp_path)
    abandoned = run.begin_case(identity())
    (abandoned.directory / "partial.tmp").write_bytes(b"debris")

    resumed = store.resume_run(run.run_dir)
    assert [case.case_id for case in resumed.recovery.interrupted] == [identity().case_id]
    retried = resumed.begin_case(identity(), retry=True)
    assert retried.attempt_id == "attempt-002"
    assert (abandoned.directory / "partial.tmp").exists()


def test_interrupted_case_requires_an_explicit_retry(tmp_path):
    run = store.open_run(tmp_path)
    run.begin_case(identity())
    resumed = store.resume_run(run.run_dir)
    # It never committed, so it is not "completed" and begin_case does not
    # refuse -- but it is reported as interrupted so a caller cannot mistake it
    # for a case that was never attempted.
    assert resumed.recovery.interrupted[0].status == "interrupted"
    assert resumed.completed_status(identity()) is None


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------

CHILD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, sys.argv[1])
    import result_store as store
    from result_store import CaseIdentity

    def ident(repeat):
        return CaseIdentity(benchmark="section7", path="rapidshot-cpu",
                            configuration="2560x1600@165-fp16", workload="motion",
                            repeat=repeat)

    run = store.open_run(sys.argv[2])
    for repeat in (1, 2):
        handle = run.begin_case(ident(repeat))
        run.commit_case(handle, {"path": "rapidshot-cpu", "unique_fps": 140.0,
                                 "unique_frames": 1120, "elapsed_seconds": 8.0},
                        status="passed", samples=[{"frame": 0}])
    run.begin_case(ident(3))
    print(run.run_dir, flush=True)
    time.sleep(120)
""")


def test_killing_the_parent_mid_case_loses_only_that_case(tmp_path):
    """The acceptance case: a real termination, with no Windows crash involved."""
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    benchmarks = str(Path(__file__).resolve().parent.parent / "benchmarks")
    proc = subprocess.Popen([sys.executable, "-u", str(script), benchmarks, str(tmp_path)],
                            stdout=subprocess.PIPE, text=True)
    try:
        run_dir = proc.stdout.readline().strip()
        assert run_dir, "child did not reach its third case"
    finally:
        proc.kill()
        proc.wait(timeout=30)

    recovery = store.recover_run(run_dir)
    assert len(recovery.completed) == 2
    assert all(case.verified and case.status == "passed"
               for case in recovery.completed.values())
    assert [case.identity.repeat for case in recovery.interrupted] == [3]
    assert not recovery.corrupt


# ---------------------------------------------------------------------------
# Persistence failure
# ---------------------------------------------------------------------------

def test_disk_full_during_commit_stops_and_preserves_earlier_cases(tmp_path, monkeypatch):
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity(repeat=1)), sample_result(), status="passed")

    handle = run.begin_case(identity(repeat=2))

    def no_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store.os, "fsync", no_space)
    with pytest.raises(store.PersistenceError):
        run.commit_case(handle, sample_result(), status="passed")
    monkeypatch.undo()

    recovery = store.recover_run(run.run_dir)
    assert len(recovery.completed) == 1
    assert recovery.completed[identity(repeat=1).case_id].verified
    # The case that could not be written is interrupted, never "passed".
    assert [case.identity.repeat for case in recovery.interrupted] == [2]


def test_a_commit_that_fails_at_the_manifest_leaves_no_commit_record(tmp_path, monkeypatch):
    """The ordering guarantee, tested at the seam it exists to protect."""
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    real = store._write_atomic

    def fail_on_manifest(path, payload):
        if Path(path).name == store.MANIFEST_NAME:
            raise store.PersistenceError("simulated failure writing the manifest")
        return real(path, payload)

    monkeypatch.setattr(store, "_write_atomic", fail_on_manifest)
    with pytest.raises(store.PersistenceError):
        run.commit_case(handle, sample_result(), status="passed", samples=[{"frame": 0}])
    monkeypatch.undo()

    # result.json and samples are on disk, but nothing claims they are a result.
    assert (handle.directory / store.RESULT_NAME).exists()
    recovery = store.recover_run(run.run_dir)
    assert not recovery.completed
    assert recovery.interrupted[0].case_id == identity().case_id


def test_committing_the_same_handle_twice_is_refused(tmp_path):
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    run.commit_case(handle, sample_result(), status="passed")
    with pytest.raises(store.StoreError):
        run.commit_case(handle, sample_result(), status="passed")


def test_non_finite_metrics_are_refused_at_commit(tmp_path):
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    with pytest.raises(ValueError):
        run.commit_case(handle, {"path": "x", "fps": float("nan")}, status="passed")


def test_unknown_status_is_refused(tmp_path):
    run = store.open_run(tmp_path)
    with pytest.raises(ValueError):
        run.commit_case(run.begin_case(identity()), sample_result(), status="probably-fine")


# ---------------------------------------------------------------------------
# Journal damage
# ---------------------------------------------------------------------------

def test_a_truncated_final_record_is_accepted_and_the_run_resumes(tmp_path):
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity(repeat=1)), sample_result(), status="passed")
    run.begin_case(identity(repeat=2))

    journal = run.journal_path
    data = journal.read_bytes()
    journal.write_bytes(data[:-20])          # a torn write of the final append

    recovery = store.recover_run(run.run_dir)
    assert recovery.truncated_tail is True
    assert len(recovery.completed) == 1
    resumed = store.resume_run(run.run_dir)
    # Sequence continues from the records that survived, so the journal stays
    # readable rather than acquiring a duplicate seq at the join.
    resumed.commit_case(resumed.begin_case(identity(repeat=3)), sample_result(),
                        status="passed")
    records, truncated = store.read_journal(journal)
    assert truncated is False
    assert [record["seq"] for record in records] == list(range(len(records)))

    # The torn bytes are the only evidence of what the lost append was saying,
    # so they are moved aside rather than deleted to make the file parse.
    debris = list(run.run_dir.glob(store.JOURNAL_NAME + ".truncated-*"))
    assert len(debris) == 1 and debris[0].read_bytes()


def test_a_record_that_lost_only_its_newline_is_kept(tmp_path):
    """The checksum is proof the record is whole; a missing byte is not damage."""
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity()), sample_result(), status="passed")
    journal = run.journal_path
    journal.write_bytes(journal.read_bytes().rstrip(b"\n"))

    scan = store.scan_journal(journal)
    assert scan.needs_newline is True and scan.truncated is True
    recovery = store.recover_run(run.run_dir)
    assert recovery.completed[identity().case_id].status == "passed"

    resumed = store.resume_run(run.run_dir)
    resumed.commit_case(resumed.begin_case(identity(repeat=2)), sample_result(),
                        status="passed")
    records, truncated = store.read_journal(journal)
    assert truncated is False
    assert [record["seq"] for record in records] == list(range(len(records)))


def test_corruption_in_the_middle_refuses_and_leaves_the_file_untouched(tmp_path):
    run = store.open_run(tmp_path)
    for repeat in (1, 2, 3):
        run.commit_case(run.begin_case(identity(repeat=repeat)), sample_result(),
                        status="passed")

    journal = run.journal_path
    lines = journal.read_bytes().split(b"\n")
    victim = json.loads(lines[2])
    victim["status"] = "failed"               # plausible edit, wrong checksum
    lines[2] = json.dumps(victim).encode()
    damaged = b"\n".join(lines)
    journal.write_bytes(damaged)

    with pytest.raises(store.JournalCorruption):
        store.recover_run(run.run_dir)
    with pytest.raises(store.JournalCorruption):
        store.resume_run(run.run_dir)
    assert journal.read_bytes() == damaged    # preserved for investigation


def test_a_missing_sequence_number_in_the_middle_is_corruption(tmp_path):
    run = store.open_run(tmp_path)
    for repeat in (1, 2):
        run.commit_case(run.begin_case(identity(repeat=repeat)), sample_result(),
                        status="passed")
    journal = run.journal_path
    lines = journal.read_bytes().split(b"\n")
    del lines[2]
    journal.write_bytes(b"\n".join(lines))
    with pytest.raises(store.JournalCorruption):
        store.recover_run(run.run_dir)


# ---------------------------------------------------------------------------
# Artifact damage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("victim", [store.RESULT_NAME, store.SAMPLES_NAME])
def test_an_altered_artifact_is_reported_corrupt_not_passed(tmp_path, victim):
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    run.commit_case(handle, sample_result(), status="passed", samples=[{"frame": 0}])

    (handle.directory / victim).write_bytes(b"tampered")

    recovery = store.recover_run(run.run_dir)
    assert not recovery.completed
    assert recovery.corrupt[0].case_id == identity().case_id
    assert any(victim in problem for problem in recovery.corrupt[0].problems)


def test_a_rewritten_manifest_cannot_pass_as_the_original(tmp_path):
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    run.commit_case(handle, sample_result(), status="passed")

    manifest = json.loads((handle.directory / store.MANIFEST_NAME).read_text())
    manifest["status"] = "passed"
    manifest["reasons"] = ["looks fine to me"]
    # Rehash the result so the manifest is internally consistent; only the
    # journal's copy of the manifest hash can catch this.
    (handle.directory / store.MANIFEST_NAME).write_bytes(store.canonical_bytes(manifest))

    recovery = store.recover_run(run.run_dir)
    assert not recovery.completed
    assert "does not match the hash in the journal" in " ".join(recovery.corrupt[0].problems)


def test_a_later_good_attempt_supersedes_an_earlier_damaged_one(tmp_path):
    run = store.open_run(tmp_path)
    first = run.begin_case(identity())
    run.commit_case(first, sample_result(fps=1.0), status="passed")
    (first.directory / store.RESULT_NAME).write_bytes(b"tampered")
    run.commit_case(run.begin_case(identity(), retry=True), sample_result(fps=2.0),
                    status="passed")

    recovery = store.recover_run(run.run_dir)
    assert recovery.completed[identity().case_id].attempt_id == "attempt-002"
    assert not recovery.corrupt


# ---------------------------------------------------------------------------
# Retention and derived views
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["failed", "invalid", "contaminated", "unavailable"])
def test_non_passing_results_are_retained_with_their_reasons(tmp_path, status):
    run = store.open_run(tmp_path)
    handle = run.begin_case(identity())
    run.commit_case(handle, {"path": "rapidshot-cpu", "note": "kept"}, status=status,
                    reasons=["why this happened"])

    recovery = store.recover_run(run.run_dir)
    case = recovery.completed[identity().case_id]
    assert case.status == status and case.reasons == ["why this happened"]
    assert case.verified is True and case.usable is False
    assert json.loads((case.directory / store.RESULT_NAME).read_text())["note"] == "kept"


def test_the_summary_is_derived_and_rebuildable(tmp_path):
    run = store.open_run(tmp_path)
    run.commit_case(run.begin_case(identity(repeat=1)), sample_result(), status="passed")
    run.commit_case(run.begin_case(identity(repeat=2)), sample_result(), status="failed",
                    reasons=["worker exit 1"])
    run.begin_case(identity(repeat=3))

    summary = run.rebuild_summary()
    assert summary["derived"] is True
    assert {case["status"] for case in summary["cases"]} == {"passed", "failed"}
    assert summary["interrupted"][0]["identity"]["repeat"] == 3

    # Deleting it costs nothing, because it was never the record.
    (run.run_dir / store.SUMMARY_NAME).unlink()
    assert run.rebuild_summary()["cases"] == summary["cases"]


# ---------------------------------------------------------------------------
# The runner-facing context manager
# ---------------------------------------------------------------------------

PIXEL_AGE_REQUIRED = ("unique_frames", "unique_fps", "elapsed_seconds")


def test_the_context_manager_derives_status_from_validation(tmp_path):
    run = store.open_run(tmp_path)
    with run.case(identity(), required=PIXEL_AGE_REQUIRED) as case:
        case.result = sample_result()
        case.samples = [{"frame": 0}]
    assert case.record.status == "passed"
    assert case.validation.ok
    # The verdict travels with the result, so a committed row explains itself.
    committed = json.loads((case.directory / store.RESULT_NAME).read_text())
    assert committed["validation"]["status"] == "passed"


def test_the_context_manager_will_not_let_a_caller_declare_success(tmp_path):
    """A worker that exits zero with impossible numbers is still invalid."""
    run = store.open_run(tmp_path)
    with run.case(identity(), required=PIXEL_AGE_REQUIRED) as case:
        case.result = sample_result(fps=400.0)          # 1124 frames in 8 s is not 400/s
    assert case.record.status == "invalid"
    assert store.recover_run(run.run_dir).completed[identity().case_id].usable is False


def test_contamination_is_recorded_on_the_case(tmp_path):
    run = store.open_run(tmp_path)
    with run.case(identity(), required=PIXEL_AGE_REQUIRED) as case:
        case.result = sample_result()
        case.contamination.append("source achieved 128.7/s against 165 requested")
    assert case.record.status == "contaminated"
    assert case.record.reasons == ["source achieved 128.7/s against 165 requested"]


def test_an_exception_inside_the_body_is_committed_as_failed_and_re_raised(tmp_path):
    run = store.open_run(tmp_path)
    with pytest.raises(TimeoutError):
        with run.case(identity(), required=PIXEL_AGE_REQUIRED) as case:
            case.result = {"path": "rapidshot-cpu", "partial": True}
            raise TimeoutError("worker exceeded its time limit")

    recovery = store.recover_run(run.run_dir)
    committed = recovery.completed[identity().case_id]
    assert committed.status == "failed" and committed.verified
    assert "TimeoutError" in committed.reasons[0]
    # The partial result is kept: what the worker managed to report before it
    # died is usually the only clue about why it died.
    assert json.loads((committed.directory / store.RESULT_NAME).read_text())["partial"] is True
    assert not recovery.interrupted


def test_a_store_overtaken_by_another_process_refuses_to_write(tmp_path):
    """A run directory is written by several processes over its life.

    `section7_suite.py` resumes the same run once per cell, so a handle held
    across those invocations is stale. Appending with the sequence it remembers
    would put a duplicate seq in the journal, which recovery reads as mid-file
    corruption -- a stale handle would silently destroy every record after it.
    """
    first = store.open_run(tmp_path)
    first.commit_case(first.begin_case(identity(repeat=1)), sample_result(), status="passed")

    other = store.resume_run(first.run_dir)         # stands in for a child process
    other.commit_case(other.begin_case(identity(repeat=2)), sample_result(), status="passed")

    with pytest.raises(store.StaleStore):
        first.begin_case(identity(repeat=3))

    # Re-opening is the fix, and the journal is still intact.
    resumed = store.resume_run(first.run_dir)
    resumed.commit_case(resumed.begin_case(identity(repeat=3)), sample_result(), status="passed")
    records, truncated = store.read_journal(first.journal_path)
    assert truncated is False
    assert [record["seq"] for record in records] == list(range(len(records)))
    assert len(store.recover_run(first.run_dir).completed) == 3


def test_latest_run_finds_the_newest_run(tmp_path):
    assert store.latest_run(tmp_path) is None
    first = store.open_run(tmp_path, run_id="20260101T000000Z-aaaaaaaa")
    second = store.open_run(tmp_path, run_id="20260102T000000Z-bbbbbbbb")
    assert store.latest_run(tmp_path) == second.run_dir
    assert first.run_dir.exists()
