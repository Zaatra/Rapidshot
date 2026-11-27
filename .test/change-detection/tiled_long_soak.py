"""Continuous public-API soak with memory samples kept outside the capture process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import numpy as np

import rapidshot
from cost_telemetry import _ProcessCounters
from downstream_benchmark import dpi_aware, wait_until
from downstream_scene import ScenePlayer, read_stamp, render, scenes
from idea_live import save
from monitor_target import select_monitor
from run_idea_sweep import stop_owned_processes
from tiled_soak import pixel_oracle


NS = 1_000_000_000
MIB = 1 << 20
MODE_NAMES = ("grab_with_changes", "grab_tiles")


def phase_at(seconds, warmup, phase_seconds):
    if seconds < warmup:
        return -2 if seconds < warmup / 2 else -1
    return int((seconds - warmup) // phase_seconds)


def phase_mode(phase):
    return MODE_NAMES[phase % len(MODE_NAMES)]


def trend(rows, metric, warmup, seconds, phase_seconds):
    """Estimate slopes from the medians of complete post-warmup blocks."""
    windows = []
    for start in range(warmup, seconds - phase_seconds + 1, phase_seconds):
        values = [row[metric] for row in rows if start <= row["elapsed_seconds"] < start + phase_seconds]
        if len(values) >= phase_seconds * 0.8:
            windows.append({"start_seconds": start, "mode": phase_mode((start - warmup) // phase_seconds),
                            "median_mib": median(values) / MIB, "samples": len(values)})

    def slope(selected):
        pairs = [(b["median_mib"] - a["median_mib"]) * 3600 / (b["start_seconds"] - a["start_seconds"])
                 for i, a in enumerate(selected) for b in selected[i + 1:]]
        return median(pairs) if pairs else None

    return {"windows": windows, "mib_per_hour": slope(windows),
            "by_mode_mib_per_hour": {mode: slope([row for row in windows if row["mode"] == mode])
                                     for mode in MODE_NAMES}}


def allocation_snapshot(camera, held):
    """Attribute backing bytes to the API without retaining the backing objects."""
    tiled = camera._tile_history
    exact = camera._change_history
    history = {} if tiled is None or tiled._pages is None else {id(page): len(page) for page in tiled._pages}
    snapshot = {} if held is None or held[0] != MODE_NAMES[1] else {id(page): len(page) for page in held[1]._pages}
    all_pages = history | snapshot
    return {"exact_history_bytes": 0 if exact is None else exact.size_bytes,
            "tiled_history_unique_pages": len(history), "tiled_history_bytes": sum(history.values()),
            "tiled_unique_pages_with_lease": len(all_pages), "tiled_bytes_with_lease": sum(all_pages.values()),
            "held_tile_extra_pages": len(snapshot.keys() - history.keys()),
            "held_tile_extra_bytes": sum(value for key, value in snapshot.items() if key not in history),
            "held_exact_pixel_bytes": 0 if held is None or held[0] != MODE_NAMES[0] else held[1].pixels.nbytes,
            "outstanding_deliberate_leases": int(held is not None)}


def run_worker(args):
    from cost_benchmark import display_awake

    dpi_aware()
    output = args.output
    target = select_monitor()
    dimensions = target["bounds"][2] - target["bounds"][0], target["bounds"][3] - target["bounds"][1]
    steps = scenes(visible_steps=8, passive_steps=1)["sparse"]
    expected = {step.stamp: render(step) for step in steps}
    data = {"status": "starting", "target": target, "capture_pid": os.getpid(), "elapsed_seconds": 0,
            "modes": {mode: dict.fromkeys(("offered", "delivered", "verified", "fixture_matches",
                                           "idle_polls", "resets", "retained_checks"), 0) for mode in MODE_NAMES},
            "phase_seconds": args.phase_seconds, "warmup_seconds": args.warmup, "phases": {},
            "oracle_scope": "BGRA maps over consecutive deliveries of each API. Fixture and audits counted separately."}
    checkpoint = output / "soak.json"
    previous = {}
    held = None
    elapsed_by_mode = dict.fromkeys(MODE_NAMES, 0)
    peak_lease = dict.fromkeys(("held_tile_extra_pages", "held_tile_extra_bytes", "held_exact_pixel_bytes"), 0)
    crop = (64, 89, 267, 221)
    region = np.s_[crop[1]:crop[3], crop[0]:crop[2]]

    def verify_held(camera, source, lease, reference):
        actual = lease.materialize(crop) if source == MODE_NAMES[1] else lease.pixels[region]
        if not np.array_equal(actual, reference):
            raise AssertionError(f"{source}: pixels changed under a held lease")
        footprint = allocation_snapshot(camera, (source, lease, reference))
        for key in peak_lease:
            peak_lease[key] = max(peak_lease[key], footprint[key])
        data["modes"][source]["retained_checks"] += 1

    with display_awake(), ScenePlayer(tuple(target["bounds"])) as player:
        player.prepare({"sparse": steps})
        player.present(steps[0], restart=True)
        camera = rapidshot.create(device_idx=target["device_index"], output_idx=target["output_index"],
                                  backend="dxgi", output_color="BGRA", pool_output=True,
                                  pool_size_frames=2, nvidia_gpu=False, timeout_ms=5)
        started = time.perf_counter_ns()
        cutoff = started + args.seconds * NS
        initial_generation = camera.generation
        data.update(status="running", started_ns=started, dimensions=dimensions, process_restarts=0,
                    maximum_loop_gap_seconds=0.0, loop_gaps_over_two_seconds=0, capture_generation_start=initial_generation)
        next_checkpoint = 0
        step_index = 0
        next_offer = started
        last_at = started
        last_mode = MODE_NAMES[0]

        def update_timing(at):
            nonlocal last_at
            elapsed_by_mode[last_mode] += min(at, cutoff) - last_at
            gap = (at - last_at) / NS
            last_at = at
            data["maximum_loop_gap_seconds"] = max(data["maximum_loop_gap_seconds"], gap)
            data["loop_gaps_over_two_seconds"] += gap > 2

        def add_run_state():
            data["mode_elapsed_seconds"] = {mode: value / NS for mode, value in elapsed_by_mode.items()}
            data["capture_generation_end"] = camera.generation
            data["capture_resource_rebuilds"] = camera.generation - initial_generation

        try:
            save(output / "worker-ready.json", {"pid": os.getpid(), "started_ns": started})
            while (now := time.perf_counter_ns()) < cutoff:
                update_timing(now)
                elapsed = (now - started) / NS
                phase = phase_at(elapsed, args.warmup, args.phase_seconds)
                mode = phase_mode(phase)
                phase_counts = data["phases"].setdefault(str(phase), {"api": mode, "offered": 0, "verified": 0})
                last_mode = mode
                counts = data["modes"][mode]
                if elapsed % 55 < 53:
                    wait_until(min(next_offer, cutoff))
                    if time.perf_counter_ns() >= cutoff:
                        break
                    next_offer = time.perf_counter_ns() + NS // 10
                    step_index = (step_index + 1) % len(steps)
                    step = steps[step_index]
                    player.present(step, restart=step_index == 0)
                    stamp = step.stamp
                    counts["offered"] += 1
                    phase_counts["offered"] += 1
                else:
                    time.sleep(.015)
                    stamp = None

                lease = (camera.grab_tiles(tight_rectangles=True, audit_every=60) if mode == MODE_NAMES[1]
                         else camera.grab_with_changes(tight_rectangles=True))
                if lease is None:
                    if stamp is None:
                        counts["idle_polls"] += 1
                else:
                    try:
                        pixels = lease.materialize() if mode == MODE_NAMES[1] else lease.pixels.copy()
                        change = lease.changes
                        last = previous.get(mode)
                        if change.reason in ("initial", "reset"):
                            indices, _ = pixel_oracle(None, pixels)
                            width, height = dimensions
                            bounds = tuple((x, y, min(x + 128, width), min(y + 128, height))
                                           for y in range(0, height, 128) for x in range(0, width, 128))
                            counts["resets"] += 1
                        else:
                            indices, bounds = pixel_oracle(last, pixels)
                        if change.indices != indices or change.tight_rectangles != bounds:
                            np.save(output / "mismatch-current.npy", pixels)
                            if last is not None:
                                np.save(output / "mismatch-prior.npy", last)
                            raise AssertionError(f"{mode}: change map mismatch after {counts['verified']} checks")
                        counts["delivered"] += 1
                        counts["verified"] += 1
                        phase_counts["verified"] += 1
                        observed = read_stamp(pixels)
                        if stamp is not None and observed == stamp and np.array_equal(
                            pixels[..., :3], expected[stamp][..., :3]
                        ):
                            counts["fixture_matches"] += 1
                        previous[mode] = pixels
                        if held is not None:
                            verify_held(camera, *held)
                            held[1].release()
                            held = None
                        if counts["delivered"] % 29 == 0:
                            held = (mode, lease, pixels[region].copy())
                    finally:
                        if held is None or held[1] is not lease:
                            lease.release()

                if elapsed >= next_checkpoint:
                    if held is not None:
                        verify_held(camera, *held)
                        held[1].release()
                        held = None
                    data.update(elapsed_seconds=elapsed, active_mode=mode,
                                released_footprint=allocation_snapshot(camera, None), held_peak=peak_lease)
                    add_run_state()
                    save(checkpoint, data)
                    print(f"{elapsed:.0f}s {mode}: {counts['verified']} maps verified", flush=True)
                    next_checkpoint = elapsed + 30

            finished = time.perf_counter_ns()
            update_timing(finished)
            data["elapsed_seconds"] = (finished - started) / NS
            add_run_state()
            if held is not None:
                verify_held(camera, *held)
                held[1].release()
                held = None
            data.update(released_footprint=allocation_snapshot(camera, None), held_peak=peak_lease)
            audits = camera.change_audit_stats()
            data["audits"] = {key: getattr(audits, key) for key in ("audited_frames", "miss_frames", "missed_tiles")}
            exercised_both = all(all(values[name] for name in ("verified", "fixture_matches", "retained_checks"))
                                 for values in data["modes"].values())
            if not exercised_both or audits.miss_frames or args.long_run and not audits.audited_frames:
                raise AssertionError("Missing fixture, lease or audit agreement")
            data["status"] = "pass"
        except BaseException as error:
            if last_at < cutoff:
                update_timing(time.perf_counter_ns())
            add_run_state()
            data.update(status="failed", error=repr(error), elapsed_seconds=(time.perf_counter_ns() - started) / NS)
            raise
        finally:
            if held is not None:
                held[1].release()
            camera.release()
            save(checkpoint, data)


def supervise(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launcher_start = time.perf_counter_ns()
    ready_path = output / "worker-ready.json"
    rows = []
    counters = None
    ready = None
    error = None
    result = "startup_timeout"
    with (output / "soak.log").open("w", encoding="utf-8") as log, \
            (output / "memory.ndjson").open("w", encoding="utf-8", buffering=1) as samples:
        child = subprocess.Popen(
            [sys.executable, "-E", "-X", "utf8", "-B", __file__, "--worker", "--output", str(output),
             "--seconds", str(args.seconds), "--warmup", str(args.warmup), "--phase-seconds", str(args.phase_seconds),
             *(["--long-run"] if args.long_run else [])], stdout=log, stderr=subprocess.STDOUT)
        try:
            while child.poll() is None and (time.perf_counter_ns() - launcher_start) / NS < 20:
                if ready_path.is_file():
                    ready = json.loads(ready_path.read_text(encoding="utf-8"))
                    break
                time.sleep(.1)
            if ready is not None:
                counters = _ProcessCounters(ready["pid"])
                started = int(ready["started_ns"])
                deadline = started + (args.seconds + 8) * NS
                while child.poll() is None and time.perf_counter_ns() < deadline:
                    try:
                        row = counters.memory()
                    except Exception as exc:
                        error = repr(exc)
                    else:
                        row["elapsed_seconds"] = (row["at_ns"] - started) / NS
                        rows.append(row)
                        samples.write(json.dumps(row) + "\n")
                    try:
                        child.wait(timeout=min(1, max(0.001, (deadline - time.perf_counter_ns()) / NS)))
                    except subprocess.TimeoutExpired:
                        pass
                result = child.poll() if child.poll() is not None else "timeout"
            elif child.poll() is not None:
                result = child.returncode
        finally:
            try:
                if child.poll() is None:
                    stop_owned_processes(child)
            finally:
                if counters is not None:
                    counters.close()
                run = {"exit": result, "elapsed_seconds": (time.perf_counter_ns() - launcher_start) / NS,
                       "target_capture_seconds": args.seconds, "startup_grace_seconds": 20,
                       "cleanup_grace_seconds": 8, "memory_sample_count": len(rows),
                       "memory_sample_error": error, "capture_pid": None if ready is None else ready["pid"]}
                worker_result = output / "soak.json"
                if worker_result.is_file():
                    worker = json.loads(worker_result.read_text(encoding="utf-8"))
                    run.update({key: worker.get(key) for key in ("mode_elapsed_seconds", "capture_resource_rebuilds",
                                                                 "process_restarts", "maximum_loop_gap_seconds",
                                                                 "loop_gaps_over_two_seconds")})
                    run["capture_elapsed_seconds"] = worker["elapsed_seconds"]
                    run["worker_status"] = worker["status"]
                for metric in ("private_bytes", "working_set_bytes"):
                    if rows:
                        run[metric] = trend(rows, metric, args.warmup, args.seconds, args.phase_seconds)
                        run[metric]["sampled_peak_mib"] = max(row[metric] for row in rows) / MIB
                        run[metric]["os_peak_mib"] = max(row[f"os_peak_{metric}"] for row in rows) / MIB
                save(output / "soak-run.json", run)
    print(json.dumps({key: run[key] for key in ("exit", "elapsed_seconds", "target_capture_seconds", "memory_sample_count")}),
          flush=True)
    return int(result != 0 or error is not None or not rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--long-run", action="store_true", help="Explicit approval for one continuous hour")
    parser.add_argument("--seconds", type=int, default=450)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--phase-seconds", type=int)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.warmup is None:
        args.warmup = 300 if args.long_run else 60
    if args.phase_seconds is None:
        args.phase_seconds = 300 if args.long_run else 60
    if args.long_run and (args.seconds, args.warmup, args.phase_seconds) != (3600, 300, 300):
        parser.error("the long run requires --seconds 3600 with the default phase and warm-up")
    if not args.long_run and not 20 <= args.seconds <= 450:
        parser.error("without --long-run, capture must fit the short-run watchdog and its startup/cleanup grace")
    if not 0 < args.warmup <= args.seconds - 2 * args.phase_seconds or args.phase_seconds <= 0:
        parser.error("the capture must include a positive warm-up and at least two complete active phases")
    return run_worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    raise SystemExit(main())
