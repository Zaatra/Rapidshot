"""Run only the two outlier workloads and the matched 30/s comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from idea_live import save
from run_idea_sweep import stop_owned_processes


HERE = Path(__file__).resolve().parent
CONTROL_ARMS = ("rapidshot_plain", "rapidshot_exact", "rapidshot_tiled", "rapidshot_tight")
CASES = (
    ("idle-30", "30", "idle", "capture", (*CONTROL_ARMS, "rapidshot_adaptive_wait", "windows_capture_wgc")),
    ("sparse-100", "100", "sparse", "capture", (*CONTROL_ARMS, "rapidshot_adaptive_wait")),
    ("matched-30", "30", "sparse", "shared_tiles", (*CONTROL_ARMS, "dxcam_view")),
)


def run(output, seconds):
    output.mkdir(parents=True, exist_ok=True)
    timer = output / "campaign.json"
    if timer.exists():
        raise ValueError("Select a fresh output directory")
    started = time.perf_counter()
    deadline = started + seconds
    report = {"rounds": 2, "cap_seconds": seconds, "phases": [], "aggregation": {
        "memory_steady_mib": 12, "memory_peak_mib": 20, "idle_points": 0.5,
        "rule": "Report median, worst and count within limits. Both fresh pairs must pass each declared gate.",
    }}
    save(timer, report)
    env = {k: v for k, v in os.environ.items() if k.upper() not in {"PYTHONHOME", "PYTHONPATH"}}
    for name, rate, workload, pipeline, arms in CASES:
        directory = output / name
        directory.mkdir(exist_ok=True)
        log = directory / "run.log"
        begin = time.perf_counter()
        remaining = deadline - begin - 8
        if remaining <= 0:
            report["phases"].append({"phase": name, "status": "deadline"})
            break
        command = [sys.executable, "-E", "-X", "utf8", "-B", str(HERE / "cost_benchmark.py"),
                   "--rates", rate, "--workloads", workload, "--pipelines", pipeline, "--arms", *arms,
                   "--rounds", "2", "--duration", "10.2", "--output", str(directory)]
        if pipeline == "shared_tiles":
            command.append("--save-wrong-frames")
        print(f"Starting {name}", flush=True)
        with log.open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, env=env)
            try:
                status = child.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                stop_owned_processes(child)
                status = "timeout"
            finally:
                if child.poll() is None:
                    stop_owned_processes(child)
        count = sum(1 for _ in (directory / "trials").glob("*.json"))
        result = {"phase": name, "elapsed_seconds": time.perf_counter() - begin,
                  "exit": status, "trials": count, "expected_trials": 2 * len(arms)}
        result["status"] = "ok" if status == 0 and count == 2 * len(arms) else "incomplete"
        report["phases"].append(result)
        report["elapsed_seconds"] = time.perf_counter() - started
        save(timer, report)
        print(f"Finished {name}: {result}", flush=True)
    save(timer, report)
    return int(any(row["status"] != "ok" for row in report["phases"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=570)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 570:
        parser.error("The release comparisons must fit within 9 minutes 30 seconds")
    raise SystemExit(run(args.output.resolve(), args.seconds))
