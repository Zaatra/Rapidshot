"""Bounded quiet-idle cycle gate and adaptive timeout/shutdown comparison."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
from random import Random
from statistics import median
import sys
import threading
import time

from cost_adapters import AdaptiveWait
from cost_telemetry import ProcessSampler
from downstream_scene import STAMP_ORIGIN, ScenePlayer, _read_stamp, _stamp_rect, read_stamp, scenes
from idea_live import NS_MS, Worker, distribution, make_camera, save
from capture_timing_followup import frequency
from downstream_benchmark import dpi_aware
from monitor_target import select_monitor


SCRIPT = Path(__file__).resolve()
ARMS = ("plain", "exact", "tiled", "tight")
BLOCKS_PER_BATCH = 6
BATCHES = 4
TARGET_PAIRS = 20
MIN_PAIRS = 10
LIMIT_PERCENT = 5.0
WINDOW_SECONDS = 10.05
TRIAL_RESERVE_SECONDS = 40
MAX_BATCH_SECONDS = 570
ADAPTIVE_REPEATS = 10
PROTOCOL = "seeded-quiet-cycles-v1"


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def camera_state(camera):
    from rapidshot._libs.dxgi import DXGI_ERROR_WAIT_TIMEOUT
    from rapidshot.core.duplicator import _format_hresult

    duplicate = camera._duplicator
    if duplicate is None:
        raise RuntimeError("The DXGI duplication is unavailable")
    last_error = duplicate.last_error or ""
    is_timeout = last_error.endswith(f"(HRESULT: {_format_hresult(DXGI_ERROR_WAIT_TIMEOUT)})")
    history = camera._tile_history or camera._change_history
    return {"serial": duplicate.frame_serial, "generation": camera.generation,
            "recovery_count": camera.recovery_count, "needs_reinit": bool(camera._needs_reinit),
            "last_capture_error": camera.last_capture_error, "last_recovery_reason": camera.last_recovery_reason,
            "duplicate_error": "" if is_timeout else last_error,
            "history_bytes": history.size_bytes if history is not None else 0,
            "region": camera.region, "width": camera.width, "height": camera.height,
            "timeout_ms": camera.timeout_ms}


def capture_loop(config, ready, begin, stop, result):
    camera = None
    adaptive = None
    try:
        camera = make_camera(config["target"], config["timeout_ms"])
        arm = config["arm"]
        if arm == "adaptive":
            adaptive = AdaptiveWait(camera)
        if arm == "exact":
            acquire = camera.grab_with_changes
        elif arm == "tiled":
            acquire = camera.grab_tiles
        elif arm == "tight":
            def acquire():
                return camera.grab_tiles(tight_rectangles=True)
        else:
            acquire = camera.grab

        from rapidshot._libs.dxgi import DXGI_ERROR_WAIT_TIMEOUT
        from rapidshot.core.duplicator import _format_hresult

        timeout_suffix = f"(HRESULT: {_format_hresult(DXGI_ERROR_WAIT_TIMEOUT)})"
        stamp_rect = _stamp_rect(STAMP_ORIGIN)
        qpc_hz = frequency()
        tracked = config["seed_stamp"]
        seeded = False
        ready.set()
        if not begin.wait(8):
            raise TimeoutError("Parent handshake did not start acquisition")
        deliveries = []
        failures = []
        recoveries = []
        serial_changes = []
        generation = camera.generation
        serial = camera._duplicator.frame_serial
        calls = 0
        unchanged = 0
        while not stop.is_set():
            lease = acquire()
            returned_ns = time.perf_counter_ns()
            calls += 1
            duplicate = camera._duplicator
            if generation != camera.generation:
                generation = camera.generation
                recoveries.append({"at_ns": returned_ns, "generation": generation})
            if serial != duplicate.frame_serial:
                serial = duplicate.frame_serial
                serial_changes.append(returned_ns)
            error = duplicate.last_error or ""
            if error and not error.endswith(timeout_suffix):
                failures.append({"at_ns": returned_ns, "error": error})
            if lease is None:
                if camera._last_grab_unchanged:
                    unchanged += 1
                elif not error:
                    failures.append({"at_ns": returned_ns, "error": camera.last_capture_error or "capture unavailable"})
                continue
            deliveries.append(returned_ns)
            try:
                if tracked is None:
                    continue
                if arm == "exact":
                    stamp = read_stamp(lease.pixels)
                elif arm in ("tiled", "tight"):
                    stamp = _read_stamp(lease.materialize(stamp_rect), (0, 0))
                else:
                    stamp = read_stamp(lease.array)
                if stamp == tracked:
                    event = "wake" if seeded else "seeded"
                    qpc_ticks = camera.last_present_time
                    present = qpc_ticks * 1_000_000_000 // qpc_hz if qpc_ticks else None
                    emit(event, stamp=stamp, return_ns=returned_ns, present_ns=present, state=camera_state(camera))
                    tracked = config.get("wake_stamp") if not seeded else None
                    seeded = True
            finally:
                lease.release()

        result.update(delivery_times_ns=deliveries, error_events=failures, recovery_events=recoveries,
                      serial_change_times_ns=serial_changes, grab_calls=calls, unchanged_grabs=unchanged,
                      final_state=camera_state(camera), adaptive_wait=adaptive.snapshot() if adaptive else None)
    except Exception as exc:
        result["error"] = repr(exc)
        if begin.is_set() and not stop.is_set():
            emit("failed", error=result["error"])
    finally:
        try:
            if adaptive is not None:
                adaptive.close()
            if camera is not None:
                camera.release()
        except Exception as exc:
            result["error"] = f"Capture teardown failed: {exc!r}"
        finally:
            ready.set()


def worker(config):
    dpi_aware()
    ready, begin, stop = (threading.Event() for _ in range(3))
    result = {}
    thread = threading.Thread(target=capture_loop, args=(config, ready, begin, stop, result), daemon=True)
    thread.start()
    if not ready.wait(8):
        raise TimeoutError("DXGI worker did not initialize")
    if "error" in result:
        emit("failed", **result)
        return 1
    emit("ready", pid=os.getpid())
    begin.set()
    sys.stdin.readline()
    stopped_ns = time.perf_counter_ns()
    stop.set()
    thread.join(4)
    if thread.is_alive():
        raise TimeoutError("DXGI worker did not release capture")
    finished_ns = time.perf_counter_ns()
    emit("stopped", stop_received_ns=stopped_ns, release_finished_ns=finished_ns,
         stop_to_release_ms=(finished_ns - stopped_ns) / NS_MS, **result)
    return 0


def event(process, kind):
    row = process.get(4)
    if row.get("event") != kind:
        raise RuntimeError(f"Expected {kind} from capture worker, got {row.get('event')}")
    return row


def validate_health(seed, worker_result, target, start_ns, end_ns):
    before = seed["state"]
    after = worker_result["final_state"]
    expected_region = [0, 0, int(target["bounds"][2] - target["bounds"][0]),
                       int(target["bounds"][3] - target["bounds"][1])]
    reasons = []
    if list(before["region"]) != expected_region or list(after["region"]) != expected_region:
        reasons.append("source geometry different from expected full monitor")
    if after["generation"] != before["generation"] or after["recovery_count"] != before["recovery_count"]:
        reasons.append("camera recovered after seeding")
    if any(row[key] for row in (before, after) for key in (
            "needs_reinit", "last_capture_error", "last_recovery_reason", "duplicate_error")):
        reasons.append("camera or duplicator reported a failure")
    if any(start_ns <= e["at_ns"] <= end_ns for e in worker_result["error_events"]):
        reasons.append("capture failure during measured window")
    if any(start_ns <= e["at_ns"] <= end_ns for e in worker_result["recovery_events"]):
        reasons.append("capture recovered during measured window")
    return reasons


def quiet_trial(player, fixture, target, arm, stamp):
    config = {"target": target, "timeout_ms": 5, "arm": arm, "seed_stamp": stamp}
    with Worker(config, script=SCRIPT) as process:
        player.present(replace(fixture, stamp=stamp), restart=True, wait_for_desktop=False)
        seeded = event(process, "seeded")
        time.sleep(0.4)
        sampler = ProcessSampler(process.pid).start()
        try:
            time.sleep(WINDOW_SECONDS)
            telemetry = sampler.finish()
        finally:
            sampler.close()
        stopped = process.stop()
    cpu = telemetry["cpu"]
    start_ns, end_ns = cpu["start_ns"], cpu["end_ns"]
    delivered = sum(start_ns <= when <= end_ns for when in stopped["delivery_times_ns"])
    presents = sum(start_ns <= when <= end_ns for when in stopped["serial_change_times_ns"])
    rejected = validate_health(seeded, stopped, target, start_ns, end_ns)
    expected_bytes = seeded["state"]["width"] * seeded["state"]["height"] * 4
    if arm != "plain" and seeded["state"]["history_bytes"] != expected_bytes:
        rejected.append("exact history was not seeded before measurement")
    if seeded["return_ns"] >= start_ns:
        rejected.append("worker seeded after measurement started")
    if delivered or presents:
        rejected.append("measured window had a delivery or DXGI desktop present")
    if cpu["cycles_per_second"] <= 0 or not stopped["unchanged_grabs"]:
        rejected.append("worker did not complete quiet polling")
    return dict(qualifying=not rejected, rejected_reasons=rejected, deliveries_in_window=delivered,
                desktop_presents_in_window=presents, telemetry=telemetry, seeded=seeded, worker=stopped)


def median_interval(values):
    ordered = sorted(values)
    n = len(ordered)
    # P(no more than k-1 samples on either side of a population median).
    tails = 0
    for k in range(1, (n + 1) // 2 + 1):
        tails += math.comb(n, k - 1)
        if 40 * tails > 2 ** n:
            k -= 1
            break
    else:
        k = (n + 1) // 2
    if k == 0:
        return None
    return {"low": ordered[k - 1], "high": ordered[n - k], "rank_one_based": [k, n - k + 1],
            "coverage_at_least": 1 - 2 * sum(math.comb(n, j) for j in range(k)) / 2 ** n}


def adjudicate_pairs(blocks, variants, *, complete):
    arms = {}
    for arm in variants:
        paired = []
        exclusions = []
        for block in blocks:
            pair = [block["arms"].get(kind) for kind in ("plain", arm)]
            if not all(row and row["qualifying"] for row in pair):
                exclusions.append(block["block"])
                continue
            plain, variant = [row["telemetry"]["cpu"]["cycles_per_second"] for row in pair]
            paired.append({"block": block["block"], "plain_cycles_per_s": plain, "cycles_per_s": variant,
                           "increase_percent": 100 * (variant - plain) / plain})
        values = [row["increase_percent"] for row in paired]
        interval = median_interval(values)
        if complete and len(values) >= MIN_PAIRS and interval is not None:
            decision = ("pass" if interval["high"] <= LIMIT_PERCENT else
                        "fail" if interval["low"] > LIMIT_PERCENT else "inconclusive")
        else:
            decision = "inconclusive"
        arms[arm] = dict(eligible_pairs=len(paired), median_increase_percent=median(values) if values else None,
                         ci95_percent=interval, status=decision, pairs=paired, excluded_blocks=exclusions)
    return arms


def adjudicate(batch_records):
    targets = {json.dumps(batch["target"], sort_keys=True) for batch in batch_records}
    batch_ids = {batch["batch"] for batch in batch_records}
    if len(targets) > 1 or len(batch_ids) != len(batch_records):
        raise ValueError("Mixed display targets or duplicate batches cannot be paired")
    if any(batch["protocol"] != PROTOCOL for batch in batch_records):
        raise ValueError("Mixed quiet-window protocols cannot be paired")
    blocks = [block for batch in batch_records for block in batch["blocks"]]
    ids = {block["block"] for block in blocks}
    if len(ids) != len(blocks):
        raise ValueError("Duplicate paired blocks")
    complete = len(blocks) == BATCHES * BLOCKS_PER_BATCH and len(batch_ids) == BATCHES and all(
        set(block["arms"]) == set(ARMS) for block in blocks)
    arms = adjudicate_pairs(blocks, ARMS[1:], complete=complete)
    return dict(protocol=PROTOCOL, limit_percent=LIMIT_PERCENT, minimum_eligible_pairs=MIN_PAIRS,
                targeted_eligible_pairs=TARGET_PAIRS, attempted_block_cap=BATCHES * BLOCKS_PER_BATCH,
                blocks_started=len(blocks), batches_present=sorted(batch_ids), complete=complete,
                metric="Paired percent change in process cycles per second with QPC wall-clock denominator",
                interval="Two-sided central binomial order-statistic interval with at least 95% coverage",
                arms=arms)


def report(output):
    batches = [json.loads(path.read_text(encoding="utf8")) for path in sorted(output.glob("quiet-batch-*.json"))]
    summary = adjudicate(batches)
    save(output / "quiet-cycle-gate.json", summary)
    return summary


def quiet_batch(output, batch, seconds, resume=False):
    dpi_aware()
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"quiet-batch-{batch}.json"
    if path.exists() and not resume:
        raise ValueError(f"Batch already executed: {path}; preserve its raw observations")
    started = time.perf_counter()
    target = select_monitor()
    data = dict(protocol=PROTOCOL, batch=batch, target=target, blocks=[], budget_seconds=seconds)
    preserved = []
    if resume and path.exists():
        previous = json.loads(path.read_text(encoding="utf8"))
        if previous["protocol"] != PROTOCOL or previous["target"] != target or previous["batch"] != batch:
            raise ValueError("The resumed batch does not match the original protocol or monitor")
        data["blocks"] = previous["blocks"]
        preserved = [row["block"] for row in data["blocks"]]
        data["preserved_blocks"] = preserved
    base = scenes(visible_steps=1, passive_steps=1)["idle"][0]
    with ScenePlayer(tuple(target["bounds"])) as player:
        for number in range((batch - 1) * BLOCKS_PER_BATCH + 1, batch * BLOCKS_PER_BATCH + 1):
            if number in preserved:
                continue
            order = list(ARMS)
            Random(6159849 + number).shuffle(order)
            block = dict(block=number, order=order, arms={})
            data["blocks"].append(block)
            for arm in order:
                if time.perf_counter() + TRIAL_RESERVE_SECONDS > started + seconds:
                    data["ended_on_budget"] = True
                    break
                print(f"Idle block {number:02d} {arm}", flush=True)
                try:
                    result = quiet_trial(player, base, target, arm, 0x5800 + number * 4 + ARMS.index(arm))
                except (RuntimeError, TimeoutError, OSError) as exc:
                    result = dict(qualifying=False, rejected_reasons=[repr(exc)])
                block["arms"][arm] = result
                data["elapsed_seconds"] = time.perf_counter() - started
                save(path, data)
            if data.get("ended_on_budget"):
                break
    data["elapsed_seconds"] = time.perf_counter() - started
    save(path, data)
    report(output)
    print(f"Quiet batch {batch} completed in {data['elapsed_seconds']:.1f}s: {path}", flush=True)
    return 0


def adaptive_trial(player, before, after, target, arm, timeout, pause):
    config = {"target": target, "timeout_ms": timeout, "arm": arm, "seed_stamp": before.stamp,
              "wake_stamp": after.stamp}
    with Worker(config, script=SCRIPT) as process:
        player.present(before, restart=True, wait_for_desktop=False)
        seeded = event(process, "seeded")
        time.sleep(pause)
        submitted_ns = time.perf_counter_ns()
        player.present(after, wait_for_desktop=False)
        wake = event(process, "wake")
        time.sleep(pause * 0.73 + 0.061)
        stopped = process.stop()
    policy = stopped["adaptive_wait"]
    transitions = [] if policy is None else policy["transitions"]
    transitioned = (any(item["to_ms"] == 50 and item["qpc_ns"] < submitted_ns for item in transitions)
                    and any(item["to_ms"] == 5 and submitted_ns <= item["qpc_ns"] <= wake["return_ns"]
                            for item in transitions))
    reset_confirmed = arm != "adaptive" or (transitioned and wake["state"]["timeout_ms"] == 5
                                            and policy["final_wait_ms"] == 50)
    rejected = validate_health(seeded, stopped, target, seeded["return_ns"], stopped["stop_received_ns"])
    if not reset_confirmed:
        rejected.append("adaptive downshift, resume reset, or idle stop was not confirmed")
    present = wake["present_ns"]
    if not present or not submitted_ns <= present <= wake["return_ns"]:
        rejected.append("resumed compositor timestamp could not be bound to submission")
    return dict(qualifying=not rejected, rejected_reasons=rejected, seeded=seeded, wake=wake,
                submitted_ns=submitted_ns, submit_to_delivery_ms=(wake["return_ns"] - submitted_ns) / NS_MS,
                desktop_to_delivery_ms=(wake["return_ns"] - present) / NS_MS if present else None,
                stop_to_release_ms=stopped["stop_to_release_ms"], worker=stopped)


def adaptive_probe(output, seconds):
    dpi_aware()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "adaptive-stop-resume.json"
    if path.exists():
        raise ValueError(f"Adaptive probe already executed: {path}; preserve its raw observations")
    started = time.perf_counter()
    target = select_monitor()
    source = scenes(visible_steps=2, passive_steps=1)["sparse"]
    data = dict(target=target, rounds=[], budget_seconds=seconds, planned_rounds=ADAPTIVE_REPEATS)
    choices = ("fixed_5ms", "adaptive", "fixed_50ms")
    with ScenePlayer(tuple(target["bounds"])) as player:
        for number in range(1, ADAPTIVE_REPEATS + 1):
            order = list(choices)
            random = Random(3537529 + number)
            random.shuffle(order)
            pause = random.uniform(0.45, 0.95)
            row = dict(round=number, order=order, arms={})
            data["rounds"].append(row)
            for name in order:
                if time.perf_counter() + TRIAL_RESERVE_SECONDS > started + seconds:
                    data["ended_on_budget"] = True
                    break
                stamp = 0x5900 + (number * len(choices) + choices.index(name)) * 2
                try:
                    result = adaptive_trial(player, replace(source[0], stamp=stamp), replace(source[1], stamp=stamp + 1),
                                            target, name if name == "adaptive" else "plain",
                                            50 if name == "fixed_50ms" else 5, pause)
                except (RuntimeError, TimeoutError, OSError) as exc:
                    result = dict(qualifying=False, rejected_reasons=[repr(exc)])
                row["arms"][name] = result
                data["elapsed_seconds"] = time.perf_counter() - started
                save(path, data)
            if data.get("ended_on_budget"):
                break
    data["summary"] = {}
    for name in choices:
        good = [row["arms"][name] for row in data["rounds"]
                if name in row["arms"] and row["arms"][name]["qualifying"]]
        data["summary"][name] = {key: {**distribution([row[key] for row in good]), "max": max(row[key] for row in good)}
                                  if good else None for key in ("submit_to_delivery_ms", "desktop_to_delivery_ms", "stop_to_release_ms")}
    data["elapsed_seconds"] = time.perf_counter() - started
    save(path, data)
    print(f"Adaptive stop/resume completed in {data['elapsed_seconds']:.1f}s: {path}", flush=True)
    return 0


def self_test():
    assert median_interval([0]) is None
    assert median_interval(range(20))["rank_one_based"] == [6, 15]
    assert median_interval(range(10))["rank_one_based"] == [2, 9]
    assert median_interval(range(24))["coverage_at_least"] >= .95
    batches = []
    for idx in range(1, BATCHES + 1):
        blocks = []
        for n in range((idx - 1) * BLOCKS_PER_BATCH + 1, idx * BLOCKS_PER_BATCH + 1):
            values = (100, 103, 108, 104 if n % 2 else 108)
            arms = {arm: dict(qualifying=True, telemetry={"cpu": {"cycles_per_second": value}})
                    for arm, value in zip(ARMS, values)}
            blocks.append(dict(block=n, arms=arms))
        batches.append(dict(batch=idx, target={}, protocol=PROTOCOL, blocks=blocks))
    assert [row["status"] for row in adjudicate(batches)["arms"].values()] == ["pass", "fail", "inconclusive"]
    for block in batches[0]["blocks"][:3]:
        block["arms"]["plain"]["qualifying"] = False
    checked = adjudicate(batches)
    assert all(row["eligible_pairs"] == 21 and row["excluded_blocks"] == [1, 2, 3]
               for row in checked["arms"].values())
    assert {row["status"] for row in adjudicate(batches[:2])["arms"].values()} == {"inconclusive"}
    print("Verified conservative median interval ranks and pass/fail/inconclusive classifications.")


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch", type=int, choices=range(1, BATCHES + 1))
    mode.add_argument("--report-only", action="store_true")
    mode.add_argument("--adaptive", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--worker", type=json.loads, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds", type=float, default=MAX_BATCH_SECONDS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args.worker)
    if args.self_test:
        self_test()
        return 0
    if args.output is None or not TRIAL_RESERVE_SECONDS <= args.seconds <= MAX_BATCH_SECONDS:
        parser.error(f"--output and a {TRIAL_RESERVE_SECONDS}–{MAX_BATCH_SECONDS}s budget are required")
    if args.report_only:
        print(json.dumps(report(args.output), indent=2))
        return 0
    if args.adaptive:
        return adaptive_probe(args.output, args.seconds)
    return quiet_batch(args.output, args.batch, args.seconds, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
