"""Compare one-shot retries and stock DXcam streaming on the shared tight-search fixture."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

from cost_adapters import DXCAM_MATCHED_ARMS, DXCAM_TIGHT_RETRY_SECONDS
from idea_live import save
from run_idea_sweep import stop_owned_processes
from run_tight_controls import STARTUP_UPDATES, state_summary, trial_strata


HERE = Path(__file__).resolve().parent
ARMS = ("rapidshot_tight", *DXCAM_MATCHED_ARMS)
ROUNDS = 5
RATE = 30
DURATION = 10.2
MIB = 2**20


def paired_stats(trials, states):
    observations = [row for trial in states for row in trial]
    steady = [row for trial in states for row in trial[STARTUP_UPDATES:]]
    cpu = [trial["process"]["cpu"] for trial in trials]
    copy = [trial["worker"]["path_accounting"] for trial in trials]
    empty = sum(trial["worker"]["empty_grab_calls"] for trial in trials)
    pause = sum(trial["worker"]["explicit_empty_poll_sleep_ns"] for trial in trials)
    requested = trials[0]["worker"]["requested_empty_poll_sleep_seconds"] if trials else None

    def median_metric(rows, key):
        values = [row[key] for row in rows if row.get(key) is not None]
        return median(values) if values else None

    payload = [trial["summary"]["payload_per_delivered_frame"] for trial in trials]
    return {
        "rounds_completed": len(trials), "all_updates": state_summary(observations),
        "steady_updates": state_summary(steady), "empty_calls": empty, "empty_retry_sleep_ns": pause,
        "requested_empty_retry_seconds": requested,
        "mean_observed_empty_retry_ms": pause / empty / 1e6 if empty and requested is not None else None,
        "cpu_percent_one_core": median_metric(cpu, "percent_of_one_core"),
        "process_cycles_per_second": median_metric(cpu, "cycles_per_second"),
        "process_cycles_per_delivered_frame": median_metric(cpu, "cycles_per_delivered_frame"),
        "gpu_payload_bytes_per_delivered_frame": median_metric(payload, "gpu_staging_payload_bytes"),
        "history_writes_bytes_per_delivered_frame": median_metric(payload, "private_history_bytes_written"),
        "stream_produced_frames": sum(row.get("stream_produced_frames", 0) for row in copy),
        "stream_consumer_reads": sum(row.get("stream_consumer_frame_allocations", 0) for row in copy),
        "stream_producer_payload_bytes": sum(row.get("gpu_staging_payload_bytes", 0) for row in copy)
        if any("stream_produced_frames" in row for row in copy) else None,
        "stream_consumer_copy_bytes": sum(row.get("stream_consumer_copy_bytes", 0) for row in copy),
    }


def summarize(output):
    trials, states, misses = {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, []
    for path in sorted((output / "trials").glob("*.json")):
        if path.name.endswith(".failure.json"):
            continue
        trial = json.loads(path.read_text(encoding="utf8"))
        strata, rows = trial_strata(trial)
        trials[trial["arm"]].append(trial)
        states[trial["arm"]].append(rows)
        misses.append(strata)
    result = {arm: paired_stats(trials[arm], states[arm]) for arm in ARMS}
    save(output / "streaming-controls.json", {"arms": result, "trials": misses})

    def display(value, scale=1):
        return f"{value / scale:.2f}" if value is not None else "—"

    lines = ["# Matched tight-search streaming controls", "",
             "All arms run the same tight OpenCV search. The two DXcam one-shot arms differ only in the "
             "requested pause after an empty `grab_view()`. Stock DXcam streaming uses `start(target_fps=100, "
             "video_mode=False)` and `get_latest_frame(copy=True, with_timestamp=True)`. Producer pacing, "
             "threading, latest-slot selection, and copying all change in that arm.", "",
             "One-shot DXcam copies changed deliveries into a private history. The streaming history retains "
             "the consumer's owned array. The stream counts full GPU staging and CPU ring writes per successful "
             "production and an additional full copy per consumer return. The source fixture retains metrics.", "",
             f"The deadline is {1000 / RATE:.2f} ms from source submission. Pixel age starts at the desktop present. "
             f"Steady-state excludes the first {STARTUP_UPDATES} offered updates each round. Frame counts and "
             "latencies pool rounds. CPU and per-delivery copy figures are medians of round values.", "",
             "| Arm | Rounds | Correct on time | Steady correct on time | Wrong | Undelivered | "
             "Pixel age at result p50/p95 ms | Submission to result p50/p95 ms | "
             "Process Mcycles/s | Process % of one core | Empty calls | Mean measured retry ms | GPU MiB per delivery |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for arm, row in result.items():
        all_updates, steady = row["all_updates"], row["steady_updates"]

        def pair(key):
            metric = all_updates["latency"][key]
            return "/".join(display(metric[tag]) for tag in ("p50", "p95"))

        lines.append(f"| {arm} | {row['rounds_completed']} | {all_updates['on_time']}/{all_updates['offered']} | "
                     f"{steady['on_time']}/{steady['offered']} | {all_updates['wrong']} | {all_updates['undelivered']} | "
                     f"{pair('present_to_result_ms')} | {pair('submit_to_result_ms')} | "
                     f"{display(row['process_cycles_per_second'], 1e6)} | {display(row['cpu_percent_one_core'])} | "
                     f"{row['empty_calls']} | {display(row['mean_observed_empty_retry_ms'])} | "
                     f"{display(row['gpu_payload_bytes_per_delivered_frame'], MIB)} |")
    lines.extend(["", "[Resource and timing tables](report.md) · [Individual misses and copy counters](streaming-controls.json) "
                  "· [Raw trials](trials)", ""])
    (output / "streaming-controls.md").write_text("\n".join(lines), encoding="utf8")
    return sum(map(len, trials.values()))


def run(output, seconds):
    if version("dxcam") != "0.3.0":
        raise RuntimeError("These controls require the pinned DXcam 0.3.0")
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    timer = output / "campaign.json"
    if timer.exists() or (output / "manifest.json").exists():
        raise ValueError("Select a fresh output directory")
    expected = ROUNDS * len(ARMS)
    status = {"rounds": ROUNDS, "window_seconds": DURATION, "rate": RATE, "arms": ARMS,
              "expected_trials": expected, "cap_seconds": seconds, "status": "running",
              "dxcam_version": "0.3.0", "dxcam_empty_retries_seconds": DXCAM_TIGHT_RETRY_SECONDS,
              "dxcam_stream_target_fps": 100, "dxcam_stream_video_mode": False, "retain_frame_evidence": False}
    save(timer, status)
    command = [sys.executable, "-E", "-X", "utf8", "-B", str(HERE / "cost_benchmark.py"), "--rates", str(RATE),
               "--workloads", "sparse", "--pipelines", "shared_tiles", "--arms", *ARMS, "--rounds", str(ROUNDS),
               "--duration", str(DURATION), "--output", str(output)]
    env = {name: value for name, value in os.environ.items() if name.upper() not in {"PYTHONHOME", "PYTHONPATH"}}
    print(f"Starting {expected} matched trials with a {seconds:g} second cap", flush=True)
    with (output / "run.log").open("w", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        try:
            exit_code = child.wait(timeout=max(0, seconds - 8 - (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            stop_owned_processes(child)
            exit_code = "timeout"
        finally:
            if child.poll() is None:
                stop_owned_processes(child)
    complete = summarize(output)
    done = exit_code == 0 and complete == expected
    status.update(elapsed_seconds=time.perf_counter() - started, exit=exit_code, trials=complete,
                  status="ok" if done else "incomplete")
    save(timer, status)
    print(f"{complete}/{expected} trials completed in {status['elapsed_seconds']:.1f}s", flush=True)
    return int(not done)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results-streaming-controls")
    parser.add_argument("--seconds", type=float, default=570)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 570:
        parser.error("The matched controls cap must be between 30 and 570 seconds")
    raise SystemExit(run(args.output.resolve(), args.seconds))
