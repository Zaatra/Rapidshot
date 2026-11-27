"""Run two rounds of the approved research probes with one wall-clock deadline."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

from idea_live import restore_display, save

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MAX_SECONDS = 570
CLEANUP_SECONDS = 12


@dataclass(frozen=True)
class Case:
    name: str
    seconds: int
    cli: tuple[str, ...]
    output_flag: str = "--output"


CASES = (
    Case("gpu", 130, ("idea_gpu.py", "--telemetry"), "--out"),
    Case("live", 100, ("idea_live.py",)),
    Case("search", 105, ("idea_search.py",)),
    Case("trace", 95, ("idea_trace.py", "--telemetry")),
    Case("surfaces", 70, ("idea_surfaces.py",)),
    Case("guardrails", 32, ("-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_exact_tiles.py",
                             "tests/test_duplicate_output1_dpi.py", "tests/test_protected_content.py",
                             "tests/test_hdr_capture.py"), ""),
)


def stop_owned_processes(process):
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
        for child in reversed(descendants):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
    except psutil.NoSuchProcess:
        pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=3)


def run_case(case, rounds, deadline_ns, output):
    before_ns = time.perf_counter_ns()
    seconds = min(case.seconds, int((deadline_ns - before_ns) / 1e9) - CLEANUP_SECONDS)
    if seconds <= 3:
        return {"phase": case.name, "status": "deadline", "elapsed_seconds": 0}

    case_dir = output / case.name
    case_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "run.log"
    args = [sys.executable, "-E", "-X", "utf8", "-B"]
    if case.output_flag:
        args += [str(HERE / case.cli[0]), *case.cli[1:], case.output_flag, str(case_dir), "--rounds", str(rounds),
                 "--seconds", str(seconds - 4)]
    else:
        args += list(case.cli)
    env = {key: value for key, value in os.environ.items() if key.upper() not in {"PYTHONPATH", "PYTHONHOME"}}
    print(f"Starting {case.name} (budget {seconds}s)", flush=True)
    with log_path.open("w", encoding="utf8") as stream:
        process = subprocess.Popen(args, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   stdin=subprocess.DEVNULL)
        try:
            status = process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            stop_owned_processes(process)
            status = "timeout"
        finally:
            if process.poll() is None:
                stop_owned_processes(process)
            if case.name == "live":
                restore_display(case_dir / "display-restore.json")
    elapsed = (time.perf_counter_ns() - before_ns) / 1e9
    print(f"Finished {case.name} in {elapsed:.2f}s, exit {status}", flush=True)
    return {"phase": case.name, "status": "ok" if status == 0 else "incomplete",
            "exit": status, "elapsed_seconds": elapsed, "budget_seconds": seconds, "log": str(log_path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=MAX_SECONDS)
    parser.add_argument("--only", choices=[case.name for case in CASES], nargs="+")
    args = parser.parse_args()
    if not 15 <= args.seconds <= MAX_SECONDS:
        parser.error(f"--seconds must be between 15 and {MAX_SECONDS}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "campaign.json"
    if marker.exists():
        parser.error("Choose a fresh output directory. This directory contains a campaign.")
    started_ns = time.perf_counter_ns()
    deadline_ns = started_ns + args.seconds * 1_000_000_000
    phases = []
    campaign = {"rounds": 2, "wall_time_cap_seconds": args.seconds, "phases": phases, "elapsed_seconds": 0}
    save(marker, campaign)
    for case in CASES:
        if args.only and case.name not in args.only:
            continue
        try:
            phases.append(run_case(case, 2, deadline_ns, output))
        except Exception as exc:
            phases.append({"phase": case.name, "status": "error", "error": repr(exc)})
            print(f"{case.name} failed: {exc}", flush=True)
        campaign["elapsed_seconds"] = (time.perf_counter_ns() - started_ns) / 1e9
        save(marker, campaign)
    print(f"Total {campaign['elapsed_seconds']:.2f}s. Artifacts: {output}", flush=True)
    return int(any(phase["status"] != "ok" for phase in phases))


if __name__ == "__main__":
    raise SystemExit(main())
