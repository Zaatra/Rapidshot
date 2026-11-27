"""Bounded native D3D11 staging and live capture prototype."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
from random import Random
from statistics import median
import subprocess
import threading
import time

from cost_benchmark import Fixture
from downstream_benchmark import dpi_aware, wait_until, write_json
from downstream_scene import ScenePlayer
from monitor_target import select_monitor


ROOT = Path(__file__).resolve().parent
EXE = ROOT / "target" / "release" / "idea_gpu.exe"
NS_SECOND = 1_000_000_000
ACTIVE_SECONDS = 2.25
NATIVE_SECONDS = 2.55
LIVE_ARMS = ("bare", "marker", "sync", "two_slot")


def remaining(deadline):
    left = deadline - time.monotonic()
    if left <= 0:
        raise TimeoutError("GPU experiment stage reached its deadline")
    return left


def environment(telemetry=False):
    env = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    env["RAPIDSHOT_IDEA_GPU_TELEMETRY"] = "1" if telemetry else "0"
    return env


def run_replay(target, deadline, rounds, telemetry):
    budget = min(30, remaining(deadline) - 0.5)
    if budget < 0.1:
        raise TimeoutError("No time remains for the GPU texture replay")
    command = [str(EXE), "replay", str(target["device_index"]), str(rounds), "24", str(budget)]
    if not telemetry:
        p = subprocess.run(command, capture_output=True, check=False, encoding="utf8", errors="replace",
                           creationflags=subprocess.CREATE_NO_WINDOW, env=environment(), timeout=remaining(deadline))
        if p.returncode:
            raise RuntimeError(p.stderr.strip() or f"Native replay exited {p.returncode}")
        return json.loads(p.stdout)
    p = launch(command, telemetry)
    sampler = None
    try:
        if wait_ready(p, deadline) != "replay":
            raise RuntimeError("Native replay returned an unexpected handshake")
        sampler = start_telemetry(p, target)
        send(p, "GO")
        data = read_result(p, deadline, sampler)
        send(p, "DONE")
        expect_exit(p, deadline)
        return data
    finally:
        if sampler is not None:
            sampler.close()
        terminate_owned(p)


def launch(command, telemetry):
    return subprocess.Popen(command, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW,
                            env=environment(telemetry))


def send(p, message):
    p.stdin.write(message + "\n")
    p.stdin.flush()


def read_line(p, deadline, timeout):
    output = queue.Queue(maxsize=1)

    def read():
        output.put(p.stdout.readline())

    threading.Thread(target=read, daemon=True).start()
    try:
        line = output.get(timeout=min(timeout, remaining(deadline))).strip()
    except queue.Empty as error:
        raise TimeoutError("Native DXGI worker did not respond in time") from error
    if not line:
        status = p.poll()
        detail = p.stderr.read().strip() if status is not None else ""
        raise RuntimeError(f"Native GPU worker returned no output ({status}): {detail}")
    return line


def wait_ready(p, deadline):
    ready = read_line(p, deadline, 5)
    if not ready.startswith("READY "):
        raise RuntimeError(f"Native worker rejected its setup: {ready}")
    return ready[6:]


def start_telemetry(p, target):
    from cost_telemetry import ProcessSampler, VideoMemory

    luid = None
    try:
        with VideoMemory(adapter_index=target["device_index"]) as adapter:
            luid = adapter.adapter_luid
    except (RuntimeError, OSError):
        pass
    return ProcessSampler(p.pid, sample_seconds=.1, pdh=True, gpu_adapter_luid=luid).prepare().start()


def read_result(p, deadline, sampler):
    data = json.loads(read_line(p, deadline, remaining(deadline)))
    memory = read_line(p, deadline, 3)
    if not memory.startswith("VIDEO_MEMORY "):
        raise RuntimeError(f"Native video-memory result is missing: {memory[:160]}")
    data["telemetry"] = {"process": sampler.finish(), "video_memory": json.loads(memory[13:])}
    return data


def expect_exit(p, deadline):
    if p.wait(timeout=remaining(deadline)):
        raise RuntimeError(p.stderr.read().strip() or f"Native GPU process exited {p.returncode}")


def terminate_owned(p):
    if p.poll() is None:
        p.kill()
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    for stream in (p.stdin, p.stdout, p.stderr):
        if stream is not None:
            stream.close()


def run_live_trial(arm, round_number, player, fixture, target, deadline, telemetry):
    if remaining(deadline) < NATIVE_SECONDS + 1:
        raise TimeoutError("Insufficient stage time to start another complete live trial")
    fixture.seed(player, "sparse")
    command = [str(EXE), "live", arm, str(target["device_index"]), str(target["output_index"]),
               str(NATIVE_SECONDS), "5"]
    p = launch(command, telemetry)
    sampler = None
    try:
        output = wait_ready(p, deadline)
        if output.casefold() != target["output"].casefold():
            raise RuntimeError(f"Native capture opened {output}, expected {target['output']}")
        if telemetry:
            sampler = start_telemetry(p, target)
        send(p, "GO")
        active_start = time.perf_counter_ns() + 100_000_000
        period = 10_000_000
        active_end = active_start + round(ACTIVE_SECONDS * NS_SECOND)
        submitted = []
        skipped = 0
        for sample_index, due in enumerate(range(active_start, active_end, period)):
            if time.monotonic() >= deadline:
                raise TimeoutError("GPU experiment stage reached its deadline")
            wait_until(due)
            before = time.perf_counter_ns()
            if before - due >= period:
                skipped += 1
                continue
            stamp = fixture.show(player, "sparse", sample_index)
            submitted.append({"stamp": stamp, "due_ns": due, "submission_start_ns": before,
                              "submission_end_ns": time.perf_counter_ns()})
        if sampler is not None:
            received = read_result(p, deadline, sampler)
            send(p, "DONE")
            expect_exit(p, deadline)
        else:
            stdout, stderr = p.communicate(timeout=remaining(deadline))
            if p.returncode:
                raise RuntimeError(stderr.strip() or f"Native capture exited {p.returncode}")
            received = json.loads(stdout)
        received.update(round=round_number, submitted=submitted, scheduling_skips=skipped,
                        active_start_ns=active_start, scheduled_active_end_ns=active_end)
        received["summary"] = live_summary(received)
        return received
    finally:
        if sampler is not None:
            sampler.close()
        terminate_owned(p)


def live_summary(trial):
    rate = trial["qpc_hz"]
    sent = {r["stamp"]: r["submission_start_ns"] for r in trial["submitted"]}
    begin = min(sent.values(), default=trial["active_start_ns"])
    end = trial["scheduled_active_end_ns"]
    selected = {}
    native = set()
    for event in trial["native_acquisitions"]:
        update = event["present_qpc"] * NS_SECOND // rate
        if begin <= update < end:
            native.add(event["present_qpc"])
    for row in trial["frames"]:
        if row["stamp"] not in sent or row["stamp"] in selected:
            continue
        if row["delivered_qpc"] * NS_SECOND // rate >= sent[row["stamp"]]:
            selected[row["stamp"]] = row
    age = [(f["delivered_qpc"] - f["present_qpc"]) / rate * 1000 for f in selected.values()
           if f["delivered_qpc"] >= f["present_qpc"]]
    age.sort()
    queue_age = [(f["delivered_qpc"] - f["producer_qpc"]) / rate * 1000 for f in selected.values()
                 if f["producer_qpc"] is not None and f["delivered_qpc"] >= f["producer_qpc"]]
    queue_age.sort()
    count = len(selected)
    total = len(sent)
    return {
        "unique_native_present_times_in_active_window": len(native),
        "native_present_times_per_active_second": len(native) / ACTIVE_SECONDS,
        "marker_coverage": count / total if total and trial["arm"] != "bare" else None,
        "unique_delivered_markers": count if trial["arm"] != "bare" else None,
        "distinct_marker_states_per_source_active_second": count / ACTIVE_SECONDS if trial["arm"] != "bare" else None,
        "desktop_to_consumer_age_ms_p50": median(age) if age else None,
        "desktop_to_consumer_age_ms_p95": age[min(len(age) - 1, int(len(age) * .95))] if age else None,
        "producer_to_consumer_age_ms_p50": median(queue_age) if queue_age else None,
        "process_cycles_per_delivered_marker": trial["process_cycles"] / count if trial["process_cycles"] and count else None,
    }


def report(data):
    def fmt(value):
        return "—" if value is None else f"{value:.2f}"

    lines = [
        "# GPU staging and live throughput", "",
        "Replay uses real D3D11 textures uploaded from a paired byte fixture outside the timed window. The measured adapter is named in the raw data. Copy bytes describe requested payload and CPU writes. The full-size staging allocation is common to all replay arms. The full-frame arm copies every pixel then compares the same candidate list. The GPU-exact total includes the flag readback wait.", "",
        "Candidate ROI requests one changed tile with a 35-pixel margin. Full materialization requests every pixel. Tile-only arms return owned immutable changed pages. These have different output contracts. A prior page lease was held and checked across every update. Full-frame and materialized output bytes and tile indices were independently verified. Every COW arm retains a fully populated initial history.", "",
        "| Fixture | Arm | Samples | Total p50 µs | Map p50 µs | GPU pixel bytes/frame | CPU pixel writes/frame | Returned pixel bytes/frame |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    replay = data.get("replay")
    if replay:
        groups = {}
        for row in replay["samples"]:
            groups.setdefault((row["layout"], row["arm"]), []).append(row)
        for (fixture, arm), rows in sorted(groups.items()):
            copied = [r["cpu_frame_write_bytes"] + r["cpu_history_patch_bytes"] + r["cpu_materialize_write_bytes"]
                      for r in rows]
            lines.append(f"| {fixture} | {arm} | {len(rows)} | {fmt(median(r['total_us'] for r in rows))} | "
                         f"{fmt(median(r['map_us'] for r in rows))} | "
                         f"{median(r['gpu_readback_payload_bytes'] for r in rows):,.0f} | "
                         f"{median(copied):,.0f} | {median(r['returned_cpu_payload_bytes'] for r in rows):,.0f} |")
    live = data["live"]
    if live:
        lines.extend([
            "", "## Controlled source at 100 updates/s", "",
            "The bare DXGI arm counts distinct nonzero desktop timestamps and cannot verify markers. The marker arm reads only the 9 KiB marker crop. Sync and two-slot arms copy all pixels. The two-slot producer enqueues the current GPU copy before mapping the prior slot when possible. A separate consumer reads owned CPU pixels through a bounded queue; raw data records mapping, input consumption, and drops.", "",
            "| Round | Arm | Actual submitted | Native desktop updates/s | Delivered source states/s | Desktop-to-consumer p50/p95 ms | Queue drops |",
            "|---:|---|---:|---:|---:|---|---:|",
        ])
        for row in sorted(live, key=lambda v: (v["round"], v["arm"])):
            s = row["summary"]
            lines.append(f"| {row['round']} | {row['arm']} | {len(row['submitted'])} | "
                         f"{fmt(s['native_present_times_per_active_second'])} | "
                         f"{fmt(s['distinct_marker_states_per_source_active_second'])} | "
                         f"{fmt(s['desktop_to_consumer_age_ms_p50'])}/{fmt(s['desktop_to_consumer_age_ms_p95'])} | "
                         f"{len(row['queue_drops'])} |")
    telemetry_rows = []
    if replay and "telemetry" in replay:
        telemetry_rows.append(("D3D11 replay, all arms", replay["telemetry"]))
    telemetry_rows.extend((f"Live {r['arm']} round {r['round']}", r["telemetry"]) for r in live if "telemetry" in r)
    if telemetry_rows:
        lines.extend([
            "", "## Measured process and GPU memory", "",
            "Each row is one entire worker process. Replay memory includes all resident input textures, staging, and compute resources shared across its arms. GPU memory is the maximum sampled process usage for each DXGI segment. PDH counts describe sampled-engine activity during the brief process window.", "",
            "| Worker | Steady private/Windows peak MiB | Steady RSS/Windows peak MiB | Sampled GPU local/non-local peaks MiB | PDH engines observed |",
            "|---|---|---|---|---:|",
        ])
        for name, telemetry in telemetry_rows:
            process = telemetry["process"]
            memory = process["memory"]
            video = telemetry["video_memory"]["samples"]
            field = lambda key: fmt(memory[key] / (1024 * 1024)) if key in memory else "—"

            def gpu(segment):
                usage = [s[segment]["current_usage_bytes"] for s in video if "current_usage_bytes" in s[segment]]
                return fmt(max(usage) / (1024 * 1024)) if usage else "—"

            engines = process["gpu_engines"]
            engine_count = len(engines["engines"]) if engines["available"] else "—"
            lines.append(f"| {name} | {field('steady_private_bytes')}/{field('os_peak_private_bytes')} | "
                         f"{field('steady_working_set_bytes')}/{field('os_peak_working_set_bytes')} | "
                         f"{gpu('local')}/{gpu('non_local')} | {engine_count} |")
    if data["unavailable"]:
        lines.extend(["", "## Unavailable or partial", ""])
        lines.extend(f"- {line}" for line in data["unavailable"])
    lines.extend(["", "[Raw results](gpu.json)"])
    return "\n".join(lines) + "\n"


def run(out_dir, deadline, rounds=2, *, live=True, telemetry=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"replay": None, "live": [], "unavailable": [], "target": None, "rounds_requested": rounds,
            "telemetry_requested": telemetry}

    def save():
        write_json(out_dir / "gpu.json", data)
        (out_dir / "gpu.md").write_text(report(data), encoding="utf8")

    if not EXE.is_file():
        data["unavailable"].append(f"Native executable is missing: {EXE}")
        save()
        return data
    try:
        dpi_aware()
        target = select_monitor()
        data["target"] = target
        data["replay"] = run_replay(target, deadline, rounds, telemetry)
        if not data["replay"]["complete"]:
            data["unavailable"].append("The deadline ended the GPU texture replay before all planned repetitions.")
        save()
    except (RuntimeError, OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
        data["unavailable"].append("GPU texture replay: " + str(error))
    if not live:
        data["unavailable"].append("Live throughput was disabled.")
    elif data["target"] is not None:
        try:
            if target["physical_refresh_hz"] < 100:
                raise RuntimeError("The selected HP monitor does not refresh at 100 Hz")
            fixture = Fixture(target)
            fixture.prewarm(["sparse"])
            rng = Random(85183)
            with ScenePlayer(target["bounds"]) as player:
                for round_number in range(1, rounds + 1):
                    arms = list(LIVE_ARMS)
                    rng.shuffle(arms)
                    for arm in arms:
                        try:
                            data["live"].append(run_live_trial(arm, round_number, player, fixture, target, deadline, telemetry))
                        except (RuntimeError, OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
                            data["unavailable"].append(f"Live {arm} round {round_number}: {error}")
                            if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
                                save()
                                return data
                        save()
        except (RuntimeError, OSError, ValueError, TimeoutError, subprocess.SubprocessError) as error:
            data["unavailable"].append("Live throughput: " + str(error))
    else:
        data["unavailable"].append("Live throughput requires the HP 324pe capture target.")
    save()
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=130)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--telemetry", action="store_true", help="Sample actual native worker process and adapter memory")
    args = parser.parse_args()
    if not 1 <= args.seconds <= 135 or not 1 <= args.rounds <= 2:
        parser.error("Choose 1–135 seconds and one or two rounds")
    result = run(args.out, time.monotonic() + args.seconds, args.rounds, live=not args.no_live,
                 telemetry=args.telemetry)
    return 0 if result["replay"] or result["live"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
