"""Run the frozen quiet-cycle batches and adaptive-wait probe in sequence."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from cost_benchmark import display_awake
from downstream_benchmark import dpi_aware
from idea_live import make_camera, save
from monitor_target import select_monitor
from quiet_cycle_gate import ARMS, BLOCKS_PER_BATCH
from run_idea_sweep import stop_owned_processes


def prepare_resume(original, output):
    preserved = set()
    unmeasured = []
    for path in sorted(original.glob("quiet-batch-*.json")):
        record = json.loads(path.read_text(encoding="utf8"))
        blocks = []
        for block in record["blocks"]:
            arms = block["arms"]
            complete = set(arms) == set(ARMS)
            measured = any(row.get("telemetry") is not None for row in arms.values())
            if complete and measured:
                blocks.append(block)
                preserved.add(block["block"])
            elif not measured:
                unmeasured.append(block["block"])
            else:
                raise ValueError(f"Block {block['block']} ran only some measurements and requires review")
        record["blocks"] = blocks
        save(output / path.name, record)
    save(output / "resumption.json", {"original": str(original), "preserved": sorted(preserved),
                                      "unmeasured_in_original": unmeasured})
    return preserved


def check_capture():
    dpi_aware()
    target = select_monitor()
    camera = make_camera(target, 5)
    camera.release()
    print(f"Opened DXGI on {target['monitor']}", flush=True)


def run(output, resume_from=None):
    output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name("quiet_cycle_gate.py")
    manifest = output / "campaign.json"
    if manifest.exists():
        raise ValueError("Use a new study directory to preserve the original trials")
    env = {k: v for k, v in os.environ.items() if k.upper() not in ("PYTHONHOME", "PYTHONPATH")}
    check_capture()
    preserved = prepare_resume(resume_from, output) if resume_from else set()
    cases = [(f"quiet-{n}", ["--batch", str(n), *( ["--resume"] if resume_from else [] )]) for n in range(1, 5)
             if not all(block in preserved for block in range((n - 1) * BLOCKS_PER_BATCH + 1,
                                                              n * BLOCKS_PER_BATCH + 1))]
    cases.append(("adaptive", ["--adaptive"]))
    started = time.perf_counter()
    result = {"phases": [], "max_seconds_per_phase": 570}
    save(manifest, result)
    for name, options in cases:
        phase_start = time.perf_counter()
        command = [sys.executable, "-E", "-X", "utf8", "-B", str(script), *options,
                   "--output", str(output), "--seconds", "556"]
        print(f"Starting {name}", flush=True)
        with (output / f"{name}.log").open("w", encoding="utf8") as log:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            try:
                status = process.wait(timeout=565)
            except subprocess.TimeoutExpired:
                stop_owned_processes(process)
                status = "timeout"
            finally:
                if process.poll() is None:
                    stop_owned_processes(process)
        phase = {"phase": name, "exit": status, "seconds": time.perf_counter() - phase_start}
        result["phases"].append(phase)
        result["elapsed_seconds"] = time.perf_counter() - started
        save(manifest, result)
        print(f"Finished {name} in {phase['seconds']:.1f}s with status {status}", flush=True)
    return int(any(row["exit"] != 0 for row in result["phases"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.probe:
        check_capture()
    elif args.output is not None:
        with display_awake():
            raise SystemExit(run(args.output.resolve(), args.resume_from.resolve() if args.resume_from else None))
    else:
        parser.error("--output is required when running the study")
