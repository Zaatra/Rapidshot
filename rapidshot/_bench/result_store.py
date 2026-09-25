"""One durable store for benchmark results, shared by every supported runner.

A benchmark that loses its results to a crash has not been run. This machine
crashes, and until now each runner persisted differently: ``section7.py``
rewrites one whole JSON payload after every path, ``section7_suite.py`` rewrites
a matrix of cell statuses, ``perf_suite.py`` writes once at the end and nothing
before it. Every one of those writes is atomic on its own -- ``save_results``
does tmp + fsync + ``os.replace`` -- and none of them can answer the question
that matters after a crash: *which measurements are trustworthy?*

The unit here is a **case**: benchmark x path x configuration x workload x
repeat. A case is committed once, immutably, and its commit record is appended
to a journal **after** the artifacts that record points at are on disk. That
ordering is the whole design. A crash can lose the case that was in flight; it
cannot corrupt, alter or silently drop one already committed.

Layout, chosen to match the results-repository layout so a later uploader copies
directories rather than reshaping them::

    <root>/runs/<run-id>/
        run.json                     run metadata, written once at open
        journal.jsonl                append-only, checksummed, sequenced
        summary.json                 derived, rebuildable, never authoritative
        cases/<case-id>/<attempt-id>/
            result.json              the measurement
            samples.jsonl.gz         raw per-sample rows, optional
            manifest.json            identity, status, and hashes of the above

**Summaries are derived.** Nothing here treats an aggregate as the record of
what happened: :meth:`ResultStore.rebuild_summary` reads the committed case
artifacts back and verifies each against its manifest. A summary is a
convenience, never the only copy of a result.

**A committed case is never rewritten.** Re-running a case that already
committed produces a *new attempt* under the same case id. Attempts are numbered
and never reused, so "retried after a crash" and "measured twice" stay
distinguishable in the history rather than collapsing into one overwritten file.

**What fsync buys here, and what it does not.** Each file is written to a
temporary, flushed, ``os.fsync``-ed, then ``os.replace``-d into place, which is
atomic on NTFS. POSIX code would also fsync the containing *directory* so the
rename itself is durable; Windows exposes no directory handle that ``os.fsync``
accepts, so a power loss between the rename and NTFS flushing its metadata can
lose a whole attempt directory. It cannot half-commit one -- the journal record
is the last write of the sequence -- so such a case reads back as interrupted
and is retried. That residual risk is bounded, and recorded here rather than
papered over.

Timing note: every write in this module happens between cases, never inside a
measured interval. Flushing to disk costs milliseconds and would land in the
timings if it were done while a path was being measured.
"""

from __future__ import annotations

import contextlib
import dataclasses
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import uuid

SCHEMA_VERSION = 1

RUN_NAME = "run.json"
JOURNAL_NAME = "journal.jsonl"
SUMMARY_NAME = "summary.json"
RESULT_NAME = "result.json"
SAMPLES_NAME = "samples.jsonl.gz"
MANIFEST_NAME = "manifest.json"

#: History lives under ``build/`` -- git-ignored, and deliberately *not* beside
#: the committed baselines in ``benchmarks/``. Those are release gates read by
#: ``make_badges.py --check`` and ``perf_suite.py --compare auto``; this is
#: growing per-run history. Conflating the two is how a baseline gets silently
#: re-recorded by a routine run.
from ._paths import WORK  # noqa: E402

DEFAULT_ROOT = WORK / "performance-history"

#: Statuses a *committed* case may carry. ``interrupted`` is deliberately absent:
#: it is not something a runner claims, it is what recovery infers from a case
#: that began and never committed.
COMMITTED_STATUSES = ("passed", "failed", "invalid", "contaminated", "unavailable")

#: The only status that means "this measurement may be quoted as evidence".
#: ``contaminated`` is retained and reported, never promoted.
USABLE_STATUSES = ("passed",)


class StoreError(RuntimeError):
    """Base class for every refusal this module makes."""


class JournalCorruption(StoreError):
    """Damage before the final journal record. The run cannot be resumed.

    Deliberately fatal rather than best-effort. A journal with a hole in the
    middle has lost the ordering that makes the rest of it meaningful, and
    guessing which side of the hole is real would be exactly the silent
    misclassification this store exists to prevent. The file is left untouched
    for investigation; nothing here rewrites or truncates it.
    """


class CaseAlreadyCommitted(StoreError):
    """This case has a committed attempt and ``retry=True`` was not given."""


class PersistenceError(StoreError):
    """A write failed. Callers must stop the suite rather than measure on."""


class StaleStore(StoreError):
    """Another process has appended to this run since this store was opened."""


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(text: object, limit: int) -> str:
    cleaned = _UNSAFE.sub("-", str(text)).strip("-").lower()
    return cleaned[:limit] or "x"


@dataclasses.dataclass(frozen=True)
class CaseIdentity:
    """What makes one measurement a distinct thing to measure.

    ``configuration`` carries everything that changes what is being measured but
    does not deserve a directory name of its own: display mode, dtype, model
    hash, pinning choice. It is hashed into the case id and stored in full in
    the manifest, so two cases that differ only there are still distinct cases.
    """

    benchmark: str
    path: str
    configuration: str
    workload: str
    repeat: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def case_id(self) -> str:
        # The digest is what actually makes this unique; the readable prefix is
        # for a human reading a directory listing. Both halves are kept short
        # because the full path runs <root>/runs/<run>/cases/<case>/<attempt>/
        # and plenty of Windows tooling still refuses paths past MAX_PATH.
        digest = hashlib.sha256(canonical_bytes(self.as_dict())).hexdigest()[:12]
        return (f"{_slug(self.benchmark, 16)}-{_slug(self.path, 24)}"
                f"-{_slug(self.workload, 10)}-r{int(self.repeat):03d}-{digest}")


def identity_from_dict(data: dict) -> CaseIdentity:
    return CaseIdentity(benchmark=data["benchmark"], path=data["path"],
                        configuration=data["configuration"], workload=data["workload"],
                        repeat=int(data["repeat"]))


# ---------------------------------------------------------------------------
# Encoding primitives
# ---------------------------------------------------------------------------

def canonical_bytes(obj) -> bytes:
    """Sorted, separator-fixed JSON, so the same content always hashes the same.

    ``allow_nan=False`` on purpose: JSON has no NaN or Infinity, and a metric
    that is one of those is a broken measurement rather than a serialisation
    inconvenience. Refusing here surfaces it at commit time instead of writing a
    file that no conforming parser can read back.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _encode_record(record: dict) -> bytes:
    body = {key: value for key, value in record.items() if key != "checksum"}
    body["checksum"] = hashlib.sha256(canonical_bytes(body)).hexdigest()
    return canonical_bytes(body) + b"\n"


def _verify_record(line: bytes) -> dict:
    record = json.loads(line.decode("utf-8"))
    if not isinstance(record, dict):
        raise ValueError("journal record is not an object")
    stated = record.get("checksum")
    body = {key: value for key, value in record.items() if key != "checksum"}
    if stated != hashlib.sha256(canonical_bytes(body)).hexdigest():
        raise ValueError("journal record checksum mismatch")
    return record


def _write_atomic(path: Path, payload: bytes) -> str:
    """Write bytes durably and return their sha256.

    Failures propagate as :class:`PersistenceError`. A store that swallows a
    write error and lets the suite keep measuring is worse than one that stops,
    because the results it goes on to report are the ones nobody can find
    afterwards.
    """
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                             suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    except OSError as exc:
        raise PersistenceError(f"cannot write {path}: {exc}") from exc
    return hashlib.sha256(payload).hexdigest()


def _append_journal(path: Path, record: dict) -> dict:
    line = _encode_record(record)
    try:
        with open(path, "ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PersistenceError(f"cannot append to {path}: {exc}") from exc
    return json.loads(line.decode("utf-8"))


def _gzip_samples(rows) -> bytes:
    buffer = io.BytesIO()
    # mtime=0 because gzip stores a timestamp in its header: without this the
    # same samples produce different bytes on every write, and the manifest hash
    # could no longer prove the file is the one that was committed.
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as stream:
        for row in rows:
            stream.write(canonical_bytes(row) + b"\n")
    return buffer.getvalue()


def read_samples(path) -> list:
    """Read back a ``samples.jsonl.gz`` written by :meth:`ResultStore.commit_case`."""
    with gzip.open(Path(path), "rb") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Journal reading and recovery
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class JournalScan:
    """Every verified record, plus where the intact prefix of the file ends."""

    records: list
    truncated: bool
    #: Bytes from the start of the file that are known-good records. Anything
    #: past this is the debris of an interrupted append.
    valid_bytes: int
    #: The last good record verified, but its terminating newline never landed.
    needs_newline: bool


def scan_journal(path) -> JournalScan:
    """Verify a journal and locate the end of its intact prefix.

    An incomplete **final** record is expected and accepted: a crash between the
    ``write`` and the ``fsync`` of the last append leaves exactly that, and the
    case it describes is retried anyway. Damage anywhere earlier raises
    :class:`JournalCorruption`, because a journal with a hole in it can no
    longer be read as an ordered account of what happened.

    A record whose checksum verifies is complete even if its newline is missing
    -- the checksum is the proof -- so it is kept, and the missing byte is
    recorded for :func:`resume_run` to supply rather than costing a measurement.
    """
    path = Path(path)
    if not path.exists():
        return JournalScan([], False, 0, False)
    data = path.read_bytes()
    ends_clean = data.endswith(b"\n")
    lines = data.split(b"\n")
    # A clean file ends with the newline of its last record, so the split leaves
    # a trailing empty element. Its absence *is* the torn-write signal.
    if lines and lines[-1] == b"":
        lines.pop()
    records, truncated, offset, needs_newline = [], False, 0, False
    for index, line in enumerate(lines):
        last = index == len(lines) - 1
        try:
            record = _verify_record(line)
            if record.get("seq") != index:
                raise ValueError(f"expected seq {index}, found {record.get('seq')!r}")
        except (ValueError, UnicodeDecodeError) as exc:
            if last:
                truncated = True
                break
            raise JournalCorruption(
                f"{path} is damaged at record {index} of {len(lines)}: {exc}. "
                "The file has been left unmodified for investigation; this run "
                "cannot be resumed.") from exc
        records.append(record)
        if last and not ends_clean:
            offset, needs_newline, truncated = len(data), True, True
        else:
            offset += len(line) + 1
    return JournalScan(records, truncated, offset, needs_newline)


def read_journal(path) -> tuple:
    """``(records, truncated_tail)`` -- the read-only view of :func:`scan_journal`."""
    scan = scan_journal(path)
    return scan.records, scan.truncated


def _heal_truncated_tail(path: Path, scan: JournalScan) -> None:
    """Make a torn journal appendable again without discarding the debris.

    Appending straight onto a partial line would splice the new record into it
    and turn an expected, recoverable truncation into mid-file corruption --
    which is exactly the state this module refuses to read. The bytes past the
    intact prefix are copied to a sidecar first: they are the only evidence of
    what the interrupted append was trying to say, and nothing here is willing
    to delete evidence to make a file parse.
    """
    data = path.read_bytes()
    debris = data[scan.valid_bytes:]
    if debris:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        _write_atomic(path.with_name(f"{path.name}.truncated-{stamp}"), debris)
    try:
        with open(path, "r+b") as stream:
            stream.truncate(scan.valid_bytes)
            if scan.needs_newline:
                stream.seek(scan.valid_bytes)
                stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise PersistenceError(f"cannot repair the tail of {path}: {exc}") from exc


@dataclasses.dataclass
class CaseRecord:
    """One committed attempt, as recovery understands it."""

    identity: CaseIdentity
    case_id: str
    attempt_id: str
    status: str
    reasons: list
    directory: Path
    committed_at: str
    manifest: dict = dataclasses.field(default_factory=dict)
    verified: bool = False
    problems: list = dataclasses.field(default_factory=list)

    @property
    def usable(self) -> bool:
        return self.verified and self.status in USABLE_STATUSES


@dataclasses.dataclass
class RunRecovery:
    """What a run directory contains, after every artifact has been re-verified."""

    run_dir: Path
    run_id: str
    metadata: dict
    records: list
    #: case_id -> the latest *verified* committed attempt.
    completed: dict
    #: case_id -> every committed attempt, in order, verified or not.
    attempts: dict
    #: Cases that began and never committed. These need an explicit retry.
    interrupted: list
    #: Cases whose commit record exists but whose artifacts do not match it.
    corrupt: list
    truncated_tail: bool
    next_sequence: int

    def status_of(self, identity: CaseIdentity):
        record = self.completed.get(identity.case_id)
        return record.status if record is not None else None


def _verify_case_artifacts(run_dir: Path, record: dict) -> CaseRecord:
    """Re-derive a committed case from disk and check it against its manifest."""
    case_id, attempt_id = record["case_id"], record["attempt_id"]
    directory = run_dir / "cases" / case_id / attempt_id
    case = CaseRecord(identity=identity_from_dict(record["identity"]), case_id=case_id,
                      attempt_id=attempt_id, status=record["status"],
                      reasons=list(record.get("reasons", [])), directory=directory,
                      committed_at=record.get("time", ""))
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        case.problems.append(f"{MANIFEST_NAME} is missing")
        return case
    if _sha256_file(manifest_path) != record.get("manifest_sha256"):
        # The journal records the manifest's hash precisely so that rebuilding
        # or editing a manifest after the fact cannot pass as the original.
        case.problems.append(f"{MANIFEST_NAME} does not match the hash in the journal")
        return case
    case.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in sorted(case.manifest.get("files", {}).items()):
        target = directory / name
        if not target.exists():
            case.problems.append(f"{name} is missing")
        elif _sha256_file(target) != expected.get("sha256"):
            case.problems.append(f"{name} does not match its manifest hash")
    case.verified = not case.problems
    return case


def recover_run(run_dir) -> RunRecovery:
    """Read a run back and classify every case it contains.

    This is the read-only half of recovery. It never repairs, deletes or
    rewrites anything: a case is completed, interrupted or corrupt, and each of
    those is reported rather than resolved. Deciding to retry is the caller's,
    because a retry costs a measurement and may not be safe to take
    unattended.
    """
    run_dir = Path(run_dir)
    journal = run_dir / JOURNAL_NAME
    records, truncated = read_journal(journal)
    metadata = {}
    run_path = run_dir / RUN_NAME
    if run_path.exists():
        metadata = json.loads(run_path.read_text(encoding="utf-8"))

    began, attempts, completed, corrupt = {}, {}, {}, {}
    for record in records:
        event = record.get("event")
        if event == "case-begin":
            began[(record["case_id"], record["attempt_id"])] = record
        elif event == "case-commit":
            began.pop((record["case_id"], record["attempt_id"]), None)
            case = _verify_case_artifacts(run_dir, record)
            attempts.setdefault(case.case_id, []).append(case)
            if case.verified:
                completed[case.case_id] = case
                corrupt.pop(case.case_id, None)
            elif case.case_id not in completed:
                # A later good attempt supersedes an earlier damaged one; a
                # damaged attempt never displaces a verified one.
                corrupt[case.case_id] = case

    interrupted = [CaseRecord(identity=identity_from_dict(record["identity"]),
                              case_id=record["case_id"], attempt_id=record["attempt_id"],
                              status="interrupted", reasons=["began and never committed"],
                              directory=run_dir / "cases" / record["case_id"]
                              / record["attempt_id"],
                              committed_at="")
                   for record in began.values()]

    return RunRecovery(run_dir=run_dir, run_id=metadata.get("run_id", run_dir.name),
                       metadata=metadata, records=records, completed=completed,
                       attempts=attempts, interrupted=interrupted,
                       corrupt=list(corrupt.values()), truncated_tail=truncated,
                       next_sequence=len(records))


def latest_run(root=None):
    """The most recently created run directory, or ``None``."""
    runs = Path(root or DEFAULT_ROOT) / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(entry for entry in runs.iterdir() if (entry / JOURNAL_NAME).exists())
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CaseHandle:
    """A case that has begun. Hand it back to :meth:`ResultStore.commit_case`."""

    identity: CaseIdentity
    case_id: str
    attempt_id: str
    directory: Path
    began_at: str
    committed: bool = False


@dataclasses.dataclass
class CaseContext:
    """What a runner fills in inside :meth:`ResultStore.case`."""

    handle: CaseHandle
    result: dict = dataclasses.field(default_factory=dict)
    samples: list = None
    #: Reasons the *conditions* were suspect. These downgrade a coherent result
    #: to ``contaminated``; they never discard it.
    contamination: list = dataclasses.field(default_factory=list)
    validation: object = None
    record: CaseRecord = None

    @property
    def directory(self) -> Path:
        return self.handle.directory


class ResultStore:
    """Append-only, crash-tolerant storage for one run's cases."""

    def __init__(self, run_dir: Path, run_id: str, sequence: int, recovery=None):
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self._sequence = sequence
        self.recovery = recovery
        self._attempts_seen = {}
        journal = self.run_dir / JOURNAL_NAME
        self._journal_bytes = journal.stat().st_size if journal.exists() else 0
        if recovery is not None:
            for case_id, cases in recovery.attempts.items():
                self._attempts_seen[case_id] = max(_attempt_number(case.attempt_id)
                                                   for case in cases)
            for case in recovery.interrupted:
                number = _attempt_number(case.attempt_id)
                self._attempts_seen[case.case_id] = max(
                    self._attempts_seen.get(case.case_id, 0), number)

    # -- properties ---------------------------------------------------------

    @property
    def journal_path(self) -> Path:
        return self.run_dir / JOURNAL_NAME

    @property
    def cases_dir(self) -> Path:
        return self.run_dir / "cases"

    # -- journal ------------------------------------------------------------

    def _emit(self, event: str, **fields) -> dict:
        # A run directory can legitimately be written by more than one process
        # over its life -- section7_suite.py resumes the same run once per cell
        # -- so an instance that has been overtaken must not append with the
        # sequence number it remembers. That would put a duplicate seq in the
        # journal, which recovery reads as mid-file corruption: a stale handle
        # would quietly destroy the readability of every record after it.
        # Comparing sizes is O(1) where rescanning the journal per append would
        # be quadratic over a long run.
        size = self.journal_path.stat().st_size if self.journal_path.exists() else 0
        if size != self._journal_bytes:
            raise StaleStore(
                f"{self.journal_path} has grown from {self._journal_bytes} to {size} bytes "
                "since this store was opened; another process has written to this run. "
                "Call resume_run() to pick up its records before writing.")
        record = {"schema_version": SCHEMA_VERSION, "seq": self._sequence, "event": event,
                  "run_id": self.run_id, "time": _utc_now(), **fields}
        written = _append_journal(self.journal_path, record)
        self._sequence += 1
        self._journal_bytes = self.journal_path.stat().st_size
        return written

    # -- case lifecycle -----------------------------------------------------

    def completed_status(self, identity: CaseIdentity):
        """The status of this case's latest verified attempt, or ``None``.

        Callers use this to resume: a case that already completed is skipped
        rather than re-measured, which is what makes a resumed run cheap and
        what stops a crash loop from quietly re-recording the same case.
        """
        if self.recovery is None:
            return None
        return self.recovery.status_of(identity)

    def committed_result(self, identity: CaseIdentity):
        """The result of this case's latest verified attempt, or ``None``.

        Read from the case artifact rather than from anything held in memory,
        so a resume reuses the bytes that were actually committed -- the same
        ones recovery hashed -- instead of a summary's paraphrase of them.
        """
        if self.recovery is None:
            return None
        case = self.recovery.completed.get(identity.case_id)
        if case is None:
            return None
        target = case.directory / RESULT_NAME
        if not target.is_file():
            return None
        return json.loads(target.read_text(encoding="utf-8"))

    def begin_case(self, identity: CaseIdentity, *, retry: bool = False) -> CaseHandle:
        """Record that a case is about to run, before the worker is launched.

        The ``running`` record goes down first so that a crash during the
        measurement is distinguishable from a case that was never started. A
        case with a verified committed attempt refuses to begin again unless
        ``retry=True``, and a retry always creates a *new* attempt rather than
        reusing the old directory.
        """
        existing = self.completed_status(identity)
        if existing is not None and not retry:
            raise CaseAlreadyCommitted(
                f"{identity.case_id} already committed with status {existing!r}; "
                "pass retry=True to record another attempt")
        number = self._attempts_seen.get(identity.case_id, 0) + 1
        # Trust the directory listing over the journal for the attempt number:
        # a crash after mkdir and before the begin record would otherwise let
        # the next attempt reuse a directory that already has files in it.
        case_root = self.cases_dir / identity.case_id
        if case_root.is_dir():
            on_disk = max((_attempt_number(entry.name) for entry in case_root.iterdir()
                           if entry.is_dir()), default=0)
            number = max(number, on_disk + 1)
        attempt_id = f"attempt-{number:03d}"
        directory = case_root / attempt_id
        try:
            directory.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            raise PersistenceError(f"cannot create {directory}: {exc}") from exc
        self._attempts_seen[identity.case_id] = number
        handle = CaseHandle(identity=identity, case_id=identity.case_id, attempt_id=attempt_id,
                            directory=directory, began_at=_utc_now())
        self._emit("case-begin", case_id=handle.case_id, attempt_id=attempt_id,
                   identity=identity.as_dict(), state="running", retry=bool(retry))
        return handle

    def commit_case(self, handle: CaseHandle, result: dict, *, status: str,
                    samples=None, reasons=(), extra_files=None) -> CaseRecord:
        """Persist a finished case, then record its completion.

        The write order is the contract: samples, then the result, then the
        manifest that hashes both, and only then the journal record that points
        at the manifest. A crash at any point either leaves no commit record --
        the case reads back as interrupted and is retried -- or leaves one whose
        artifacts are all present and hash-verified. There is no ordering in
        which a commit record can refer to a partial result.

        Failed, invalid and contaminated results are committed exactly like
        passing ones, with their reasons. Discarding them would hide the two
        things a benchmark history is most often needed to answer: what broke,
        and what was thrown away.
        """
        if handle.committed:
            raise StoreError(f"{handle.case_id}/{handle.attempt_id} is already committed")
        if status not in COMMITTED_STATUSES:
            raise ValueError(f"status must be one of {COMMITTED_STATUSES}, not {status!r}")

        files = {}
        if samples is not None:
            payload = _gzip_samples(samples)
            files[SAMPLES_NAME] = {"sha256": _write_atomic(handle.directory / SAMPLES_NAME,
                                                          payload),
                                   "bytes": len(payload),
                                   "rows": len(samples)}
        for name, payload in sorted((extra_files or {}).items()):
            blob = payload if isinstance(payload, bytes) else canonical_bytes(payload)
            files[name] = {"sha256": _write_atomic(handle.directory / name, blob),
                           "bytes": len(blob)}
        result_bytes = canonical_bytes(result)
        files[RESULT_NAME] = {"sha256": _write_atomic(handle.directory / RESULT_NAME,
                                                     result_bytes),
                              "bytes": len(result_bytes)}

        manifest = {"schema_version": SCHEMA_VERSION, "run_id": self.run_id,
                    "case_id": handle.case_id, "attempt_id": handle.attempt_id,
                    "identity": handle.identity.as_dict(), "status": status,
                    "reasons": list(reasons), "began_at": handle.began_at,
                    "committed_at": _utc_now(), "files": files}
        manifest_sha = _write_atomic(handle.directory / MANIFEST_NAME,
                                     canonical_bytes(manifest))

        record = self._emit("case-commit", case_id=handle.case_id,
                            attempt_id=handle.attempt_id, identity=handle.identity.as_dict(),
                            status=status, reasons=list(reasons),
                            manifest_sha256=manifest_sha)
        handle.committed = True
        case = CaseRecord(identity=handle.identity, case_id=handle.case_id,
                          attempt_id=handle.attempt_id, status=status, reasons=list(reasons),
                          directory=handle.directory, committed_at=record["time"],
                          manifest=manifest, verified=True)
        if self.recovery is not None:
            self.recovery.attempts.setdefault(case.case_id, []).append(case)
            self.recovery.completed[case.case_id] = case
            self.recovery.corrupt = [other for other in self.recovery.corrupt
                                     if other.case_id != case.case_id]
        return case

    @contextlib.contextmanager
    def case(self, identity: CaseIdentity, *, required=(), end_to_end=None,
             retry: bool = False, samples=None):
        """Run one case, and commit it whatever happens to it.

        The handle is opened before the body runs and committed on the way out,
        so the only way to leave a case uncommitted is for the process to die --
        which is precisely the state recovery is built to recognise. An
        exception inside the body is committed as ``failed`` with its text and
        then re-raised, because a suite that keeps measuring after an unhandled
        error is a suite whose later results nobody can interpret.

        The status comes from :func:`result_validation.validate_result`, not
        from the caller, so "the worker did not crash" and "the worker produced
        a usable measurement" cannot be conflated at the call site.
        """
        from .result_validation import validate_result

        handle = self.begin_case(identity, retry=retry)
        context = CaseContext(handle=handle, samples=samples)
        try:
            yield context
        except BaseException as exc:
            # commit_case may itself fail; let that surface rather than masking
            # it with the original error, since a store that cannot write is a
            # bigger problem than whatever the case was doing.
            self.commit_case(handle, context.result or {"error": f"{type(exc).__name__}: {exc}"},
                             status="failed", samples=context.samples,
                             reasons=[f"{type(exc).__name__}: {exc}"])
            raise
        verdict = validate_result(context.result, required=required,
                                  contamination=context.contamination, end_to_end=end_to_end)
        context.validation = verdict
        context.record = self.commit_case(
            handle, dict(context.result, validation=verdict.as_dict()),
            status=verdict.status, samples=context.samples, reasons=verdict.reasons)

    def close(self, status: str = "complete", **fields) -> dict:
        """Mark the run finished. Absence of this record is not an error.

        A run whose journal has no ``run-end`` simply crashed or is still going,
        and recovery reads it the same way either way -- which is why nothing
        downstream is allowed to depend on this record existing.
        """
        return self._emit("run-end", status=status, **fields)

    # -- derived views ------------------------------------------------------

    def rebuild_summary(self, write: bool = True) -> dict:
        """Rebuild the run summary from verified case artifacts.

        Always recomputed from the manifests, never accumulated in memory, so a
        summary cannot drift from the cases it claims to summarise and cannot
        become the only surviving copy of a result.
        """
        recovery = recover_run(self.run_dir)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "derived": True,
            "source_of_truth": "cases/<case-id>/<attempt-id>/manifest.json",
            "run_id": recovery.run_id,
            "generated_at": _utc_now(),
            "metadata": recovery.metadata,
            "truncated_tail": recovery.truncated_tail,
            "cases": [{"case_id": case.case_id, "attempt_id": case.attempt_id,
                       "identity": case.identity.as_dict(), "status": case.status,
                       "reasons": case.reasons, "committed_at": case.committed_at,
                       "files": case.manifest.get("files", {})}
                      for case in sorted(recovery.completed.values(),
                                         key=lambda item: item.case_id)],
            "interrupted": [{"case_id": case.case_id, "attempt_id": case.attempt_id,
                             "identity": case.identity.as_dict()}
                            for case in recovery.interrupted],
            "corrupt": [{"case_id": case.case_id, "attempt_id": case.attempt_id,
                         "problems": case.problems} for case in recovery.corrupt],
        }
        if write:
            _write_atomic(self.run_dir / SUMMARY_NAME, canonical_bytes(summary))
        return summary


def _attempt_number(attempt_id: str) -> int:
    try:
        return int(str(attempt_id).rsplit("-", 1)[-1])
    except ValueError:
        return 0


def open_run(root=None, *, metadata=None, run_id=None) -> ResultStore:
    """Start a new run and write its metadata before any case begins.

    ``metadata`` is where the environment snapshot goes. It is deliberately
    opaque here: this module's job is to make records durable, not to decide
    what a complete description of a machine is.
    """
    root = Path(root or DEFAULT_ROOT)
    # Sortable timestamp plus randomness: two runs started in the same second
    # -- which happens the moment anything scripts this -- must not collide.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_id or f"{stamp}-{uuid.uuid4().hex[:8]}"
    run_dir = root / "runs" / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PersistenceError(f"cannot create {run_dir}: {exc}") from exc
    _write_atomic(run_dir / RUN_NAME, canonical_bytes(
        {"schema_version": SCHEMA_VERSION, "run_id": run_id, "created_at": _utc_now(),
         "metadata": metadata or {}}))
    store = ResultStore(run_dir, run_id, sequence=0, recovery=None)
    store.recovery = RunRecovery(run_dir=run_dir, run_id=run_id, metadata=metadata or {},
                                 records=[], completed={}, attempts={}, interrupted=[],
                                 corrupt=[], truncated_tail=False, next_sequence=0)
    store._emit("run-begin", metadata=metadata or {})
    return store


# ---------------------------------------------------------------------------
# Runner-facing glue
# ---------------------------------------------------------------------------

#: Statuses a resume treats as settled. A measurement happened and was
#: committed; whether it was *clean* is a separate question, answered by its
#: reasons. Re-running one automatically would overwrite nothing -- attempts are
#: immutable -- but it would spend a measurement to learn what is already known.
RESUME_SETTLED = ("passed", "unavailable", "contaminated")


def open_for_runner(benchmark, *, root=None, resume=None, disabled=False, metadata=None):
    """Open or resume the history a runner writes to, or ``None`` if disabled.

    Called before anything is measured, and deliberately outside whatever
    try/except the runner wraps its measurements in: a store that cannot be
    opened is a reason to stop, not a reason to run the benchmark and find out
    afterwards that nothing was recorded.
    """
    if disabled:
        return None
    if resume is not None:
        return resume_run(resume)
    return open_run(root, metadata={"benchmark": benchmark, **(metadata or {})})


def resume_decision(store, identity, *, retry_failed: bool = False):
    """``(action, status)`` -- what to do about a case that may already exist.

    ``action`` is one of ``measure``, ``skip`` or ``retry``. A failed or invalid
    case is never retried automatically, even on an explicit resume: a retry
    costs a measurement, and a suite that silently re-runs whatever failed is one
    that can loop on a broken machine until it produces a number someone likes.
    """
    if store is None:
        return "measure", None
    status = store.completed_status(identity)
    if status is None:
        return "measure", None
    if status in RESUME_SETTLED:
        return "skip", status
    return ("retry" if retry_failed else "skip"), status


@contextlib.contextmanager
def case_context(store, identity, *, required=(), end_to_end=None, retry=False):
    """A case to fill in, whether or not durable history is enabled.

    ``--no-history`` yields a context with no handle so a runner has one code
    path rather than two, and so the flag cannot accidentally change what is
    measured -- only whether it is kept.
    """
    if store is None:
        yield CaseContext(handle=None)
        return
    with store.case(identity, required=required, end_to_end=end_to_end, retry=retry) as case:
        yield case


def add_history_arguments(parser, *, default_root=None):
    """The history flags every supported runner takes, spelled the same way."""
    parser.add_argument("--history-root", type=Path, default=default_root,
                        help=f"durable result history (default {DEFAULT_ROOT})")
    parser.add_argument("--resume", type=Path, default=None,
                        help="resume an existing run directory; settled cases are not "
                             "re-measured")
    parser.add_argument("--retry-failed", action="store_true",
                        help="when resuming, measure cases that previously failed or were "
                             "invalid; each retry is recorded as a new attempt")
    parser.add_argument("--no-history", action="store_true",
                        help="skip the durable store. For debugging the harness only.")
    return parser


def resume_run(run_dir) -> ResultStore:
    """Reopen an existing run so completed cases are not measured again.

    Refuses outright if the journal is damaged anywhere but its final record --
    see :class:`JournalCorruption`. A run with a truncated tail resumes
    normally: the lost record is a case that will be retried, and the torn bytes
    are moved aside by :func:`_heal_truncated_tail` first.

    This is the only entry point that writes to an existing journal.
    :func:`recover_run` stays read-only so that inspecting a damaged run -- the
    thing you do when you do not yet know what happened -- cannot change it.
    """
    run_dir = Path(run_dir)
    scan = scan_journal(run_dir / JOURNAL_NAME)
    if scan.truncated:
        _heal_truncated_tail(run_dir / JOURNAL_NAME, scan)
    recovery = recover_run(run_dir)
    return ResultStore(run_dir, recovery.run_id, sequence=recovery.next_sequence,
                       recovery=recovery)
