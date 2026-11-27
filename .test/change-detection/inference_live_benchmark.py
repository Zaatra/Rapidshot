"""Live capture libraries feeding one real CPU detector, measured per process.

The parent paints a paced dashboard on the target monitor and spawns one worker
per arm. Each worker opens its capture library, pulls frames for the measured
window, decides whether the detector must run again, and runs the verified
OpenCV Zoo YOLOX-S FP32 model on the CPU when it must. CPU cycles and memory
come from the worker process alone, so the cost of painting the fixture never
lands in a capture library's column.

Every arm feeds the detector the same thing: a whole BGRA frame. The arms differ
only in how the frame arrives and how the skip decision is reached.

``rapidshot_exact``
    ``grab_with_changes()`` — whole frame plus a byte-exact change verdict that
    capture already paid for, so the gate is free and needs no second copy.
``rapidshot_plain``
    ``grab()`` and a full-frame NumPy compare against a retained copy.
``dxcam_copy`` / ``dxcam_view``
    DXcam's copying and view grabs, each with the same retained-copy compare.
``mss_gdi``
    MSS through GDI, same compare. GDI exposes no present timestamp, so this arm
    has no pixel age.

``--no-gate`` runs an arm that infers on every delivered frame, which is the
control for what gating is worth.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes as ct
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from capture_timing_followup import frequency
from cost_adapters import CostAdapter
from monitor_target import select_monitor


ROOT = Path(__file__).resolve().parent
GATED_ARMS = ("rapidshot_exact", "rapidshot_plain", "dxcam_copy", "dxcam_view", "mss_gdi")
NS_SECOND = 1_000_000_000
NS_MS = 1_000_000
LABELS = {
    "rapidshot_exact": "RapidShot `grab_with_changes()`",
    "rapidshot_plain": "RapidShot `grab()`",
    "dxcam_copy": "DXcam `grab()`",
    "dxcam_view": "DXcam `grab_view()`",
    "mss_gdi": "MSS (GDI)",
}
NO_PRESENT_CLOCK = ("mss_gdi",)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

@dataclass
class Observation:
    present_ns: int | None
    delivered_ns: int
    decided_ns: int
    done_ns: int
    inferred: bool
    detections: int
    gate_source: str


def wait_until(deadline_ns):
    while True:
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2 * NS_MS:
            time.sleep(remaining / NS_SECOND / 2)


def run_worker(arm, target, config):
    from whole_frame_task import YoloX

    gate = config["gate"]
    audit = config["audit"]
    detector = YoloX()
    width = target["bounds"][2] - target["bounds"][0]
    height = target["bounds"][3] - target["bounds"][1]
    detector.infer(np.zeros((height, width, 4), dtype=np.uint8))  # Warm the graph.

    adapter = CostAdapter(arm, target, config["hz"], frequency())
    payload = {"pid": ct.windll.kernel32.GetCurrentProcessId(), "adapter_luid": None}
    print("READY " + json.dumps(payload), flush=True)

    command = json.loads(sys.stdin.readline())
    start_ns, stop_ns = command["start_ns"], command["stop_ns"]

    observations: list[Observation] = []
    previous = None          # retained copy for the compare-based arms
    empty_reads = 0
    audit_failures = 0
    audit_checked = 0
    retained_copies = 0
    detections = 0

    wait_until(start_ns)
    try:
        while time.perf_counter_ns() < stop_ns:
            sample = adapter.read(0.004)
            if sample is None:
                empty_reads += 1
                continue
            delivered_ns = time.perf_counter_ns()
            try:
                pixels = sample.pixels
                if pixels is None:
                    raise RuntimeError(f"{arm} delivered no whole-frame pixels")
                identical = False
                source = "none"
                if gate:
                    if sample.changes is not None:
                        identical = not sample.changes.indices
                        source = f"capture change metadata ({sample.changes.reason})"
                    elif previous is not None:
                        identical = np.array_equal(pixels, previous)
                        source = "retained full-frame compare"
                decided_ns = time.perf_counter_ns()
                # Verifying the free verdict is a check on the library, not part
                # of the pipeline, so it sits outside the decision timer.
                if audit and sample.changes is not None and previous is not None:
                    audit_checked += 1
                    if identical != np.array_equal(pixels, previous):
                        audit_failures += 1
                if not identical:
                    found = detector.infer(pixels)
                    detections = len(found)
                done_ns = time.perf_counter_ns()
                observations.append(Observation(
                    sample.present_ns, delivered_ns, decided_ns, done_ns,
                    not identical, detections, source))
                # The compare-based arms must own a copy to compare against; the
                # exact path already keeps private history inside capture, and
                # only retains one here when auditing its verdict.
                needs_copy = gate and (sample.changes is None or audit)
                if needs_copy:
                    if previous is None or previous.shape != pixels.shape:
                        previous = pixels.copy()
                    else:
                        np.copyto(previous, pixels)
                    retained_copies += 1
            finally:
                sample.release()
    finally:
        snapshot = adapter.snapshot()
        adapter.close()

    result = {
        "arm": arm, "gate": gate, "audit": audit,
        "empty_reads": empty_reads, "retained_copy_writes": retained_copies,
        "audit_checked": audit_checked, "audit_failures": audit_failures,
        "copy_events": snapshot,
        "observations": [vars(o) for o in observations],
    }
    print("RESULT " + json.dumps(result), flush=True)
    json.loads(sys.stdin.readline())


# --------------------------------------------------------------------------- #
# Parent
# --------------------------------------------------------------------------- #

class Worker:
    def __init__(self, arm, target, config):
        self.arm = arm
        self.ready = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.ready_payload = None
        self.messages = []
        self.process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
             "--worker", arm, "--target-json", json.dumps(target),
             "--config-json", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf8", cwd=str(ROOT),
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith("READY "):
                    self.ready_payload = json.loads(line[6:])
                    self.ready.set()
                elif line.startswith("RESULT "):
                    self.result = json.loads(line[7:])
                    self.done.set()
                elif len(self.messages) < 80:
                    self.messages.append(line.strip())
        finally:
            self.ready.set()
            self.done.set()

    def __enter__(self):
        if not self.ready.wait(180) or self.ready_payload is None:
            self.close()
            raise RuntimeError(f"{self.arm} failed to open: {' '.join(self.messages[-12:])}")
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def collect(self, timeout=90):
        if not self.done.wait(timeout) or self.result is None:
            raise RuntimeError(f"{self.arm} produced no result: {' '.join(self.messages[-12:])}")
        with contextlib.suppress(Exception):
            self.send({"finalized": True})
        return self.result

    def close(self):
        with contextlib.suppress(Exception):
            self.process.stdin.close()
        with contextlib.suppress(Exception):
            self.process.wait(10)
        if self.process.poll() is None:
            self.process.kill()


@contextlib.contextmanager
def display_awake():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes = [ct.c_uint32]
    set_state.restype = ct.c_uint32
    continuous, system, display = 0x80000000, 0x00000001, 0x00000002
    if not set_state(continuous | system | display):
        raise ct.WinError(ct.get_last_error())
    try:
        yield
    finally:
        set_state(continuous)


def quantiles(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {"n": len(values), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)), "mean": statistics.fmean(values)}


def summarize(run, elapsed_seconds, cpu, memory):
    observations = run["observations"]
    inferred = [o for o in observations if o["inferred"]]
    ages = [(o["done_ns"] - o["present_ns"]) / NS_MS
            for o in inferred if o["present_ns"] is not None]
    capture_ages = [(o["delivered_ns"] - o["present_ns"]) / NS_MS
                    for o in observations if o["present_ns"] is not None]
    decision = [(o["decided_ns"] - o["delivered_ns"]) / NS_MS for o in observations[1:]]
    infer_ms = [(o["done_ns"] - o["decided_ns"]) / NS_MS for o in inferred]
    return {
        "delivered_frames": len(observations),
        "detector_calls": len(inferred),
        "skipped_frames": len(observations) - len(inferred),
        "empty_reads": run["empty_reads"],
        "delivered_per_second": len(observations) / elapsed_seconds,
        "detector_calls_per_second": len(inferred) / elapsed_seconds,
        "detections_ready_ms": quantiles(ages),
        "capture_age_ms": quantiles(capture_ages),
        "decision_ms": quantiles(decision),
        "inference_ms": quantiles(infer_ms),
        "gate_source": observations[-1]["gate_source"] if observations else "none",
        "retained_copy_writes": run["retained_copy_writes"],
        "audit_checked": run["audit_checked"],
        "audit_failures": run["audit_failures"],
        "cpu": cpu, "memory": memory,
    }


def run_trial(player, fixture, target, arm, settings):
    from cost_telemetry import ProcessSampler

    workload = settings["workload"]
    fixture.seed(player, workload)
    config = {"hz": settings["hz"], "gate": settings["gate"], "audit": settings["audit"]}
    with Worker(arm, target, config) as child:
        sampler = ProcessSampler(child.ready_payload["pid"], sample_seconds=0.1,
                                 pdh=False, gpu_adapter_luid=None)
        sampler.prepare().start()
        start = sampler._initial_cpu.at_ns
        stop = start + round(settings["duration"] * NS_SECOND)
        submitted = 0
        try:
            child.send({"start_ns": start, "stop_ns": stop})
            period = round(NS_SECOND / settings["hz"])
            for index, due in enumerate(range(start, stop, period)):
                wait_until(due)
                if time.perf_counter_ns() >= stop:
                    break
                fixture.show(player, workload, index)
                submitted += 1
            wait_until(stop)
        finally:
            process = sampler.finish()
        run = child.collect()
    cpu = process["cpu"]
    memory = process.get("memory")
    summary = summarize(run, cpu["elapsed_seconds"], cpu, memory)
    summary["submitted_updates"] = submitted
    return {"arm": arm, "settings": settings, "summary": summary,
            "copy_events": run["copy_events"]}


def report(data):
    lines = ["# Live capture libraries through one CPU detector", "",
             f"One worker process per arm on {data['target']['monitor']} "
             f"({data['target']['bounds'][2]}×{data['target']['bounds'][3]}, "
             f"{data['target']['physical_refresh_hz']} Hz, {data['target']['adapter']}). "
             f"The dashboard was repainted at "
             f"{', '.join(str(r) for r in data.get('rates', [data['hz']]))} updates a second, "
             f"{data['duration']:.0f} s per arm per rate, {data['rounds']} round(s). Every arm "
             "fed the same OpenCV Zoo YOLOX-S FP32 model on the CPU, one thread, OpenCL off.", "",
             "CPU and memory are the worker process only, so painting the fixture is not "
             "charged to any library. `detections ready` is the age of the captured pixels at "
             "the moment the detector's output exists, measured from the DXGI present that "
             "produced them.", ""]
    def millis(block, places=2):
        return "—" if not block else f"{block['p50']:.{places}f} ms"

    for round_data in data["trials"]:
        lines += [f"## {round_data.get('rate', data['hz'])} updates a second "
                  f"— round {round_data['round']}", "",
                  "| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | "
                  "Decision p50 | Mcycles/s | CPU % of one core | Peak working set |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for trial in round_data["arms"]:
            summary = trial["summary"]
            memory = summary["memory"] or {}
            peak = (memory.get("os_peak_working_set_bytes")
                    or memory.get("sampled_peak_working_set_bytes"))
            cpu = summary["cpu"]["percent_of_one_core"]
            lines.append(
                f"| {LABELS.get(trial['arm'], trial['arm'])} "
                f"| {summary['detector_calls_per_second']:.2f} "
                f"| {millis(summary['detections_ready_ms'])} "
                f"| {millis(summary['capture_age_ms'])} "
                f"| {millis(summary['decision_ms'], 3)} "
                f"| {summary['cpu']['cycles_per_second'] / 1e6:.1f} "
                f"| {'—' if cpu is None else f'{cpu:.2f}'} "
                f"| {'—' if peak is None else f'{peak / 2**20:.1f} MiB'} |")
        lines += ["", "| Arm | Delivered | Detector calls | Skipped | Empty reads | "
                  "Retained copies | Gate |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"]
        for trial in round_data["arms"]:
            summary = trial["summary"]
            lines.append(
                f"| {LABELS.get(trial['arm'], trial['arm'])} "
                f"| {summary['delivered_frames']} | {summary['detector_calls']} "
                f"| {summary['skipped_frames']} | {summary['empty_reads']} "
                f"| {summary['retained_copy_writes']} | {summary['gate_source']} |")
        audited = [t for t in round_data["arms"] if t["summary"]["audit_checked"]]
        if audited:
            lines += [""]
            for trial in audited:
                summary = trial["summary"]
                lines.append(
                    f"Audit of {LABELS.get(trial['arm'], trial['arm'])}: "
                    f"{summary['audit_checked']} verdicts checked against an independent "
                    f"full-frame compare, **{summary['audit_failures']} disagreed**.")
        lines.append("")
    if data["target"]["monitor"]:
        lines += ["MSS reaches the desktop through GDI, which exposes no present timestamp, so "
                  "that arm has no pixel age and its latency columns are blank. Every other arm "
                  "times from the DXGI present that produced the pixels.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--target-json")
    parser.add_argument("--config-json")
    parser.add_argument("--arms", nargs="+", default=list(GATED_ARMS))
    parser.add_argument("--hz", type=int, default=30)
    parser.add_argument("--rates", type=int, nargs="+",
                        help="Sweep these update rates instead of a single --hz")
    parser.add_argument("--duration", type=float, default=40)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workload", default="sparse")
    parser.add_argument("--no-gate", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--monitor", default="HP 324pe")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()

    if arguments.worker:
        run_worker(arguments.worker, json.loads(arguments.target_json),
                   json.loads(arguments.config_json))
        return

    from cost_benchmark import Fixture
    from downstream_scene import ScenePlayer

    target = select_monitor(arguments.monitor)
    rates = arguments.rates or [arguments.hz]
    data = {"target": target, "rates": rates, "hz": rates[0],
            "duration": arguments.duration,
            "rounds": arguments.rounds, "workload": arguments.workload,
            "gate": not arguments.no_gate, "audit": arguments.audit, "trials": []}

    fixture = Fixture(target)
    fixture.prewarm([arguments.workload])
    with display_awake(), ScenePlayer(target["bounds"]) as player:
        player.prepare(fixture.steps)
        for round_index in range(arguments.rounds):
            for rate in rates:
                settings = {"hz": rate, "duration": arguments.duration,
                            "workload": arguments.workload, "gate": not arguments.no_gate,
                            "audit": arguments.audit}
                arms = []
                for arm in arguments.arms:
                    print(f"round {round_index + 1}, {rate} Hz: {arm}", flush=True)
                    trial = run_trial(player, fixture, target, arm, settings)
                    summary = trial["summary"]
                    percent = summary["cpu"]["percent_of_one_core"]
                    print(f"    {summary['detector_calls']} calls, "
                          f"{summary['delivered_frames']} frames, "
                          f"{summary['skipped_frames']} skipped, "
                          f"{summary['empty_reads']} empty, "
                          f"{summary['cpu']['cycles_per_second'] / 1e6:.1f} Mcycles/s"
                          + (f", {percent:.2f}% CPU" if percent is not None else ""), flush=True)
                    arms.append(trial)
                data["trials"].append({"round": round_index + 1, "rate": rate, "arms": arms})

    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        arguments.out.with_suffix(".md").write_text(report(data), encoding="utf-8")
        print(f"Wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
