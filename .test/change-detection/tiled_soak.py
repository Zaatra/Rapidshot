"""Bounded one-process soak through the public tiled capture API."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

import numpy as np

import rapidshot
from cost_telemetry import ProcessSampler
from downstream_benchmark import dpi_aware, wait_until
from downstream_scene import ScenePlayer, read_stamp, render, scenes
from idea_live import save
from monitor_target import select_monitor
from run_idea_sweep import stop_owned_processes


def pixel_oracle(previous, current):
    """Compare BGRA words, then independently group positions into tiles."""
    columns = (current.shape[1] + 127) // 128
    if previous is None:
        return tuple(range(columns * ((current.shape[0] + 127) // 128))), None
    y, x = np.nonzero(previous.view(np.uint32)[..., 0] != current.view(np.uint32)[..., 0])
    ids = y // 128 * columns + x // 128
    indices = tuple(map(int, np.unique(ids)))
    bounds = []
    for index in indices:
        members = ids == index
        bounds.append((int(x[members].min()), int(y[members].min()),
                       int(x[members].max()) + 1, int(y[members].max()) + 1))
    return indices, tuple(bounds)


def run_worker(output, seconds):
    dpi_aware()
    target = select_monitor()
    workloads = scenes(visible_steps=8, passive_steps=1)
    steps = workloads["sparse"]
    expected = {step.stamp: render(step) for step in steps}
    path = output / "soak.json"
    data = {"status": "running", "target": target, "offered": 0, "delivered": 0, "verified": 0,
            "matched_presentations": 0, "idle_polls": 0, "resets": 0, "retained_checks": 0,
            "gpu_readback_bytes": 0, "history_bytes_written": 0, "oracle_materialized_bytes": 0,
            "measurement_scope": "Public capture, fixture presentation, pixel oracle and evidence in one process."}
    camera = None
    held = None
    previous = None
    started = time.perf_counter()
    deadline = started + seconds
    ages_ms = []
    from capture_timing_followup import frequency
    qpc_rate = frequency()

    with ScenePlayer(tuple(target["bounds"])) as player:
        player.prepare({"sparse": steps})
        player.present(steps[0], restart=True)
        camera = rapidshot.create(device_idx=target["device_index"], output_idx=target["output_index"],
                                  backend="dxgi", output_color="BGRA", pool_output=True,
                                  pool_size_frames=2, nvidia_gpu=False, timeout_ms=5)
        sampler = ProcessSampler(os.getpid()).start()
        try:
            next_checkpoint = started
            next_offer = time.perf_counter_ns()
            next_idle = started + 55
            idle_until = started
            phase = 0
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if now >= next_idle:
                    idle_until = now + 2
                    next_idle = now + 55
                if now < idle_until:
                    expected_stamp = None
                    time.sleep(.015)
                else:
                    wait_until(next_offer)
                    next_offer = max(next_offer + 100_000_000, time.perf_counter_ns())
                    phase = (phase + 1) % len(steps)
                    step = steps[phase]
                    player.present(step, restart=phase == 0)
                    expected_stamp = step.stamp
                    data["offered"] += 1

                frame = camera.grab_tiles(tight_rectangles=True, audit_every=60)
                returned = time.perf_counter_ns()
                if frame is None:
                    if expected_stamp is None:
                        data["idle_polls"] += 1
                    continue
                try:
                    pixels = frame.materialize()
                    changes = frame.changes
                    indices, bounds = pixel_oracle(previous, pixels)
                    if changes.reason in ("initial", "reset"):
                        indices, bounds = pixel_oracle(None, pixels)
                        bounds = changes.rectangles
                        data["resets"] += 1
                    if changes.indices != indices or changes.tight_rectangles != bounds:
                        np.save(output / "mismatch-current.npy", pixels)
                        if previous is not None:
                            np.save(output / "mismatch-prior.npy", previous)
                        raise AssertionError(f"Independent map mismatch at delivery {data['delivered']}")
                    for key in ("gpu_readback_bytes", "history_bytes_written"):
                        data[key] += camera._last_tile_capture_cost[key]
                    data["oracle_materialized_bytes"] += pixels.nbytes
                    data["verified"] += 1
                    data["delivered"] += 1
                    stamp = read_stamp(pixels)
                    if expected_stamp == stamp and stamp in expected and np.array_equal(
                        pixels[..., :3], expected[stamp][..., :3]
                    ):
                        data["matched_presentations"] += 1
                    previous = pixels
                    if changes.present_time_qpc:
                        age = (returned - changes.present_time_qpc * 1_000_000_000 // qpc_rate) / 1e6
                        if age >= 0:
                            ages_ms.append(age)
                    if held is not None:
                        retained, reference = held
                        retained_pixels = retained.materialize((64, 89, 267, 221))
                        if not np.array_equal(retained_pixels, reference):
                            raise AssertionError("A later capture changed a held snapshot")
                        retained.release()
                        held = None
                        data["retained_checks"] += 1
                    if data["delivered"] % 29 == 0:
                        held = (frame, frame.materialize((64, 89, 267, 221)))
                finally:
                    if held is None or held[0] is not frame:
                        frame.release()

                if now >= next_checkpoint:
                    data["elapsed_seconds"] = time.perf_counter() - started
                    save(path, data)
                    print(f"{data['elapsed_seconds']:.0f}s: {data['verified']} pixel-oracle matches", flush=True)
                    next_checkpoint = now + 30

            data["elapsed_seconds"] = time.perf_counter() - started
            data["telemetry"] = sampler.finish(data["delivered"])
            audits = camera.change_audit_stats()
            data["audits"] = {name: getattr(audits, name) for name in ("audited_frames", "miss_frames", "missed_tiles")}
            ages_ms.sort()
            data["pixel_age_ms"] = {"p50": median(ages_ms), "p95": ages_ms[(95 * len(ages_ms) - 1) // 100]}
            if not data["verified"] or not data["matched_presentations"] or not data["retained_checks"] or audits.miss_frames:
                raise AssertionError("Soak did not pass capture, retention and audit checks")
            data["status"] = "pass"
        except BaseException as error:
            data["status"] = "failed"
            data["error"] = repr(error)
            raise
        finally:
            sampler.close()
            if held is not None:
                held[0].release()
            camera.release()
            save(path, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=480)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if not 20 <= args.seconds <= 480:
        parser.error("the soak is capped at 480 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return run_worker(args.output, args.seconds - 12)
    log = args.output / "soak.log"
    started = time.perf_counter()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [sys.executable, "-E", "-X", "utf8", "-B", __file__, "--worker", "--seconds", str(args.seconds),
             "--output", str(args.output)], stdout=stream, stderr=subprocess.STDOUT)
        try:
            exit_code = process.wait(timeout=max(0, args.seconds - 4 - (time.perf_counter() - started)))
        except subprocess.TimeoutExpired:
            stop_owned_processes(process)
            exit_code = "timeout"
        finally:
            if process.poll() is None:
                stop_owned_processes(process)
    result = {"exit": exit_code, "elapsed_seconds": time.perf_counter() - started, "cap_seconds": args.seconds}
    save(args.output / "soak-run.json", result)
    print(result, flush=True)
    return int(exit_code != 0)


if __name__ == "__main__":
    raise SystemExit(main())
