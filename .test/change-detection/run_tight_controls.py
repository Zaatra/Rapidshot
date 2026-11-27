"""Five shuffled 30/s trials of RapidShot search paths and the full-frame DXcam baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from capture_timing_followup import quantiles
from idea_live import save
from run_idea_sweep import stop_owned_processes


HERE = Path(__file__).resolve().parent
ARMS = (
    "rapidshot_plain", "rapidshot_exact", "rapidshot_tiled", "rapidshot_tight", "dxcam_view",
    "rapidshot_exact_tight",
)
ROUNDS = 5
RATE = 30
DURATION = 10.2
STARTUP_UPDATES = 3


def observed_state(trial, submission, row):
    if row is None:
        return {"outcome": "undelivered"}
    returned = row["capture_returned_ns"]
    finished = row.get("result_ns")
    if finished is None:
        return {"outcome": "no_result"}
    begin, present = submission["send_start_ns"], row["present_ns"]
    end_to_end_ms = (finished - begin) / 1e6
    late = end_to_end_ms > 1000 / trial["rate"] or finished > trial["requested_stop_ns"]
    outcome = "wrong" if row.get("correct_fixture") is not True else "late" if late else "on_time"
    has_present = present is not None and begin <= present <= returned
    return {
        "outcome": outcome,
        "submit_to_result_ms": end_to_end_ms,
        "submit_to_present_ms": (present - begin) / 1e6 if has_present else None,
        "present_to_return_ms": (returned - present) / 1e6 if has_present else None,
        "present_to_result_ms": (finished - present) / 1e6
        if has_present and finished <= trial["requested_stop_ns"] else None,
        "return_to_result_ms": (finished - returned) / 1e6,
        "search_ms": row["search_ns"] / 1e6,
        "crop_materialization_ms": row["task_crop_materialization_ns"] / 1e6
        if row.get("task_crop_materialization_ns") is not None else None,
    }


def state_summary(states):
    counts = {key: sum(item["outcome"] == key for item in states)
              for key in ("on_time", "late", "wrong", "undelivered", "no_result")}
    fields = ("submit_to_result_ms", "submit_to_present_ms", "present_to_return_ms", "present_to_result_ms",
              "return_to_result_ms", "search_ms", "crop_materialization_ms")
    return {"offered": len(states), **counts, "latency": {
        key: quantiles(row[key] for row in states if row.get(key) is not None) for key in fields
    }}


def trial_strata(trial):
    source = {item["stamp"]: item for item in trial["submitted"]}
    observed = {}
    for row in trial["worker"]["observations"]:
        stamp = row["stamp"]
        if (stamp in source and row["outcome"] == "delivered"
                and source[stamp]["send_start_ns"] <= row["capture_returned_ns"] <= trial["requested_stop_ns"]):
            observed[stamp] = row
    states = [dict(stamp=stamp, offered_order=order, **observed_state(trial, item, observed.get(stamp)))
              for order, (stamp, item) in enumerate(source.items(), 1)]
    all_updates = state_summary(states)
    if (all_updates["offered"] != trial["summary"]["eligible_markers"]
            or all_updates["on_time"] != trial["summary"]["correct_on_time"]):
        raise ValueError(f"Stratum counting disagrees with the benchmark for {trial['arm']} round {trial['round']}")
    return {
        "arm": trial["arm"], "round": trial["round"], "all_updates": all_updates,
        "cold_updates": state_summary(states[:STARTUP_UPDATES]),
        "steady_updates": state_summary(states[STARTUP_UPDATES:]),
        "misses": [item for item in states if item["outcome"] != "on_time"],
    }, states


def write_strata(output):
    results, groups = [], {arm: [] for arm in ARMS}
    for path in sorted((output / "trials").glob("*.json")):
        if path.name.endswith(".failure.json"):
            continue
        trial = json.loads(path.read_text(encoding="utf8"))
        item, states = trial_strata(trial)
        results.append(item)
        groups.setdefault(trial["arm"], []).append(states)
    aggregate = {}
    for arm, trials in groups.items():
        aggregate[arm] = {"rounds_completed": len(trials),
                          "all_updates": state_summary([row for trial in trials for row in trial]),
                          "cold_updates": state_summary([row for trial in trials for row in trial[:STARTUP_UPDATES]]),
                          "steady_updates": state_summary([row for trial in trials for row in trial[STARTUP_UPDATES:]])}
    save(output / "strata.json", {"first_offered_updates_per_round_excluded_from_steady": STARTUP_UPDATES,
                                  "aggregate": aggregate, "trials": results})
    search_scope = (
        "This archive contains earlier matched-search controls, including DXcam with benchmark-added change detection."
        if "dxcam_view_tight" in groups else
        "RapidShot supplies tight rectangles to its tight-search arms. Stock DXcam supplies full frames to the "
        "original OpenCV template search. The tiled RapidShot arm materializes crops and charges those copies."
    )
    lines = ["# Live 30/s search comparison", "", search_scope, "",
             f"Headlines include every offered update at {RATE}/s. Steady-state excludes the first {STARTUP_UPDATES} "
             "offered updates of each round for every arm. A missed result is classified as wrong before late, so "
             "the categories are mutually exclusive. Deadlines start at submission. Pixel age starts at DXGI present. "
             "Latency quantiles pool eligible timestamps across rounds and exclude undelivered results."]
    for key, title in (("all_updates", "All updates"), ("cold_updates", "First three offered per round"),
                       ("steady_updates", "Steady-state")):
        lines.extend(["", f"## {title}", "",
                      "| Arm | Rounds | Correct on time | Late | Wrong | Undelivered | No result | "
                      "Submission to result p50/p95 ms | Present to result p50/p95 ms | Search p50/p95 ms |",
                      "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
        for arm, item in aggregate.items():
            values = item[key]

            def pair(metric):
                detail = values["latency"][metric]
                return "/".join(f"{detail[q]:.2f}" if detail[q] is not None else "—" for q in ("p50", "p95"))

            lines.append(f"| {arm} | {item['rounds_completed']} | {values['on_time']}/{values['offered']} | "
                         f"{values['late']} | {values['wrong']} | {values['undelivered']} | {values['no_result']} | "
                         f"{pair('submit_to_result_ms')} | {pair('present_to_result_ms')} | {pair('search_ms')} |")
    lines.extend(["", "[CPU, memory and copy measurements](report.md) · "
                  "[Per-round strata and individual misses](strata.json) · [Raw trials](trials)", ""])
    (output / "tight-controls.md").write_text("\n".join(lines), encoding="utf8")
    return len(results)


def run(output, seconds):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    timer = output / "campaign.json"
    if timer.exists() or (output / "manifest.json").exists():
        raise ValueError("Select a fresh output directory")
    expected = ROUNDS * len(ARMS)
    status = {"rounds": ROUNDS, "window_seconds": DURATION, "rate": RATE, "arms": ARMS, "expected_trials": expected,
              "cap_seconds": seconds, "startup_excluded_updates_per_round": STARTUP_UPDATES, "status": "running"}
    save(timer, status)
    command = [sys.executable, "-E", "-X", "utf8", "-B", str(HERE / "cost_benchmark.py"), "--rates", str(RATE),
               "--workloads", "sparse", "--pipelines", "shared_tiles", "--arms", *ARMS, "--rounds", str(ROUNDS),
               "--duration", str(DURATION), "--output", str(output), "--save-wrong-frames"]
    env = {name: value for name, value in os.environ.items() if name.upper() not in {"PYTHONHOME", "PYTHONPATH"}}
    print(f"Starting {expected} trials with a {seconds:g} second cap", flush=True)
    with (output / "run.log").open("w", encoding="utf8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        try:
            exit_code = child.wait(timeout=max(0, seconds - 8 - (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            stop_owned_processes(child)
            exit_code = "timeout"
        finally:
            if child.poll() is None:
                stop_owned_processes(child)
    complete = write_strata(output)
    done = exit_code == 0 and complete == expected
    status.update(elapsed_seconds=time.perf_counter() - started, exit=exit_code, trials=complete,
                  status="ok" if done else "incomplete")
    save(timer, status)
    print(f"{complete}/{expected} trials completed in {status['elapsed_seconds']:.1f}s", flush=True)
    return int(not done)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results-tiled-final" / "matched")
    parser.add_argument("--seconds", type=float, default=570)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 570:
        parser.error("The matched trial cap must be between 30 and 570 seconds")
    return run(args.output.resolve(), args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
