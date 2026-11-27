"""Run a single fixed continuation of the confirmed-quiet cycle gate."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from random import Random
import subprocess
import sys
import time

from cost_benchmark import display_awake
from downstream_benchmark import dpi_aware
from downstream_scene import ScenePlayer, scenes
from idea_live import make_camera, save
from monitor_target import select_monitor
from quiet_cycle_gate import (
    ARMS as ORIGINAL_ARMS,
    LIMIT_PERCENT,
    MIN_PAIRS,
    PROTOCOL as ORIGINAL_PROTOCOL,
    TRIAL_RESERVE_SECONDS,
    adjudicate,
    adjudicate_pairs,
    quiet_trial,
)
from run_idea_sweep import stop_owned_processes


SCRIPT = Path(__file__).resolve()
DEFAULT_ARCHIVE = SCRIPT.parent / "results-tiled-final" / "quiet-resumed"
PROTOCOL = "seeded-quiet-cycles-v1-fixed-extension"
ARMS = ("plain", "exact", "tiled")
FIRST_BLOCK = 25
BLOCKS_PER_BATCH = 6
BATCHES = (1, 2, 3, 4)
BLOCKS = tuple(range(FIRST_BLOCK, FIRST_BLOCK + len(BATCHES) * BLOCKS_PER_BATCH))
BATCH_SOFT_SECONDS = 550
BATCH_KILL_SECONDS = 560
BATCH_MAX_SECONDS = 570


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def archived_records(archive):
    records = []
    checksums = {}
    for batch in BATCHES:
        name = f"quiet-batch-{batch}.json"
        path = archive / name
        raw = path.read_bytes()
        checksums[name] = sha256(raw).hexdigest()
        records.append(json.loads(raw))
    original = adjudicate(records)
    blocks = [block for record in records for block in record["blocks"]]
    if not original["complete"] or sorted(b["block"] for b in blocks) != list(range(1, FIRST_BLOCK)):
        raise ValueError("Archive must contain exactly the completed original blocks 1 through 24")
    return records, original, checksums


def planned_blocks(batch):
    start = FIRST_BLOCK + (batch - 1) * BLOCKS_PER_BATCH
    return tuple(range(start, start + BLOCKS_PER_BATCH))


def arm_order(number):
    order = list(ARMS)
    Random(6159849 + number).shuffle(order)
    return order


def batch_path(output, batch):
    return output / f"extension-batch-{batch}.json"


def load_batch(output, batch, target):
    path = batch_path(output, batch)
    data = (read(path) if path.exists() else
            dict(protocol=PROTOCOL, source_protocol=ORIGINAL_PROTOCOL, batch=batch, target=target, blocks=[]))
    if (data["protocol"], data["source_protocol"], data["batch"], data["target"]) != (
            PROTOCOL, ORIGINAL_PROTOCOL, batch, target):
        raise ValueError(f"Batch {batch} does not match the fixed protocol or monitor")
    block_numbers = [block["block"] for block in data["blocks"]]
    if block_numbers != list(planned_blocks(batch))[:len(block_numbers)]:
        raise ValueError(f"Batch {batch} does not contain the planned block prefix")
    for block in data["blocks"]:
        order = arm_order(block["block"])
        if block["order"] != order or list(block["arms"]) != order[:len(block["arms"])]:
            raise ValueError(f"Block {block['block']} diverges from the fixed shuffle")
    return data


def batch_complete(data):
    return (len(data["blocks"]) == BLOCKS_PER_BATCH and
            all(set(block["arms"]) == set(ARMS) and
                not any(row.get("attempt_state") == "started" for row in block["arms"].values())
                for block in data["blocks"]))


def resolve_interrupted_attempts(data):
    for block in data["blocks"]:
        for row in block["arms"].values():
            if row.get("attempt_state") == "started":
                row.update(attempt_state="interrupted", qualifying=False, rejected_reasons=[
                    "Capture attempt was interrupted before a durable result. Measurement status is unknown."
                ])


def probe_target(expected):
    dpi_aware()
    target = select_monitor()
    if target != expected:
        raise ValueError(f"Display changed from the archived target. Expected {expected!r}, got {target!r}")
    camera = make_camera(target, 5)
    camera.release()
    return target


def measure_batch(output, batch, expected_target):
    started = time.perf_counter()
    target = probe_target(expected_target)
    data = load_batch(output, batch, target)
    resolve_interrupted_attempts(data)
    path = batch_path(output, batch)
    save(path, data)
    previous = {block["block"]: block for block in data["blocks"]}
    fixture = scenes(visible_steps=1, passive_steps=1)["idle"][0]
    with ScenePlayer(tuple(target["bounds"])) as player:
        for number in planned_blocks(batch):
            block = previous.get(number)
            if block is None:
                block = dict(block=number, order=arm_order(number), arms={})
                data["blocks"].append(block)
            for arm in block["order"]:
                if arm in block["arms"]:
                    continue
                if time.perf_counter() - started + TRIAL_RESERVE_SECONDS > BATCH_SOFT_SECONDS:
                    data["elapsed_last_invocation_seconds"] = time.perf_counter() - started
                    save(path, data)
                    return 2
                print(f"Quiet extension block {number:02d} {arm}", flush=True)
                block["arms"][arm] = dict(attempt_state="started", qualifying=False)
                save(path, data)
                try:
                    row = quiet_trial(player, fixture, target, arm, 0x5800 + number * 4 + ORIGINAL_ARMS.index(arm))
                except (RuntimeError, TimeoutError, OSError) as error:
                    row = dict(qualifying=False, rejected_reasons=[repr(error)])
                block["arms"][arm] = dict(attempt_state="recorded", **row)
                data["elapsed_last_invocation_seconds"] = time.perf_counter() - started
                save(path, data)
    return 0


def final_results(output, records, original):
    target = records[0]["target"]
    fresh_batches = [load_batch(output, batch, target) for batch in BATCHES]
    if not all(batch_complete(batch) for batch in fresh_batches):
        raise ValueError("Final inference is available after all 24 continuation blocks have been attempted")
    old_blocks = [block for record in records for block in record["blocks"]]
    new_blocks = [block for record in fresh_batches for block in record["blocks"]]
    fresh = adjudicate_pairs(new_blocks, ARMS[1:], complete=True)
    pooled = adjudicate_pairs(old_blocks + new_blocks, ARMS[1:], complete=True)
    tight = original["arms"]["tight"]
    final = dict(protocol=PROTOCOL, target=target, complete=True, threshold_percent=LIMIT_PERCENT,
                 minimum_pairs=MIN_PAIRS, original_blocks=len(old_blocks), fresh_blocks=len(new_blocks),
                 fixed_new_block_ids=list(BLOCKS), original=original["arms"], fresh=fresh, pooled=pooled,
                 release_gate={"exact": pooled["exact"]["status"], "tiled": pooled["tiled"]["status"],
                               "tight": tight["status"]},
                 fresh_interpretation="Fresh-only result is a diagnostic; the fixed decision uses pooled observations.",
                 interval_interpretation="Central exact 95% median interval at each fixed look. Under the common "
                 "sampling assumptions, the upper-end directional false-pass error across two looks is at most 5% "
                 "per path. The display of both two-sided intervals does not imply joint 95% coverage across looks.")
    save(output / "quiet-cycle-extension.json", final)
    return final


def run(output, archive, resume=False):
    records, original, checksums = archived_records(archive)
    expected_target = records[0]["target"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    provenance = {"protocol": PROTOCOL, "archive": str(archive), "archive_sha256": checksums,
                  "max_seconds_per_phase": BATCH_MAX_SECONDS, "planned_blocks": list(BLOCKS)}
    manifest = dict(provenance, phases=[])
    if path.exists():
        if not resume:
            raise ValueError("Use --resume for an interrupted campaign or choose a fresh output directory")
        manifest = read(path)
        if any(manifest.get(key) != value for key, value in provenance.items()):
            raise ValueError("Cannot resume with different source data or protocol")
    elif any(output.glob("extension-batch-*.json")):
        raise ValueError("Cannot resume raw batches without their provenance manifest")
    save(path, manifest)
    env = {key: value for key, value in os.environ.items() if key.upper() not in ("PYTHONHOME", "PYTHONPATH")}
    with display_awake():
        probe_target(expected_target)
        for batch in BATCHES:
            if batch_complete(load_batch(output, batch, expected_target)):
                continue
            attempt = 1 + sum(row["batch"] == batch for row in manifest["phases"])
            log_path = output / f"extension-{batch}-attempt-{attempt}.log"
            command = [sys.executable, "-E", "-X", "utf8", "-B", str(SCRIPT), "--batch", str(batch),
                       "--archive", str(archive), "--output", str(output)]
            started = time.perf_counter()
            phase = {"batch": batch, "attempt": attempt, "log": log_path.name, "status": "started"}
            manifest["phases"].append(phase)
            save(path, manifest)
            print(f"Starting quiet extension batch {batch}/4", flush=True)
            with log_path.open("w", encoding="utf8") as log:
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                try:
                    status = process.wait(timeout=BATCH_KILL_SECONDS)
                except subprocess.TimeoutExpired:
                    stop_owned_processes(process)
                    status = "timeout"
                finally:
                    if process.poll() is None:
                        stop_owned_processes(process)
            phase.update(status=status, elapsed_seconds=time.perf_counter() - started)
            save(path, manifest)
            print(f"Finished quiet extension batch {batch} in {phase['elapsed_seconds']:.1f}s with status {status}", flush=True)
            if status != 0:
                return 1
    final = final_results(output, records, original)
    manifest["complete"] = True
    save(path, manifest)
    print(f"Fixed pooled quiet-cycle gate: {final['release_gate']}", flush=True)
    return 0


def self_test(archive):
    records, original, _ = archived_records(archive)
    assert original["arms"]["exact"]["eligible_pairs"] == 19
    assert original["arms"]["tiled"]["eligible_pairs"] == 20
    assert original["arms"]["tight"]["status"] == "pass"
    assert tuple(number for batch in BATCHES for number in planned_blocks(batch)) == BLOCKS
    for block in BLOCKS:
        assert set(arm_order(block)) == set(ARMS)
    rows = {arm: {"attempt_state": "recorded", "qualifying": True,
                  "telemetry": {"cpu": {"cycles_per_second": cycles}}}
            for arm, cycles in zip(ARMS, (100, 100, 100))}
    new = [dict(block=number, arms=rows.copy()) for number in BLOCKS]
    new[-1]["arms"]["plain"] = {"attempt_state": "started", "qualifying": False}
    resolve_interrupted_attempts({"blocks": new})
    assert new[-1]["arms"]["plain"]["attempt_state"] == "interrupted"
    assert adjudicate_pairs(new, ARMS[1:], complete=True)["tiled"]["eligible_pairs"] == 23
    pooled = adjudicate_pairs([block for batch in records for block in batch["blocks"]] + new,
                              ARMS[1:], complete=True)
    assert [row["eligible_pairs"] for row in pooled.values()] == [42, 43]
    print("Verified archived eligibility, fixed shuffle, interruption preservation and pooled exact inference.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--batch", type=int, choices=BATCHES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    archive = args.archive.resolve()
    if args.self_test:
        self_test(archive)
        return 0
    if args.output is None:
        parser.error("--output is required")
    output = args.output.resolve()
    if args.batch or args.report_only:
        records, original, checksums = archived_records(archive)
        manifest = read(output / "campaign.json")
        if (manifest.get("protocol"), manifest.get("archive"), manifest.get("archive_sha256")) != (
                PROTOCOL, str(archive), checksums):
            raise ValueError("Archive and campaign provenance differ")
        if args.batch:
            return measure_batch(output, args.batch, records[0]["target"])
        report = final_results(output, records, original)
        print(json.dumps(report["release_gate"], indent=2))
        return 0
    return run(output, archive, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
