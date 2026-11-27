"""Measure timestamped capture coverage with an independently paced desktop scene."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import ctypes as ct
from importlib.metadata import version
import json
from pathlib import Path
import random
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from downstream_benchmark import Camera, Capture, dpi_aware, wait_until, write_json
from downstream_scene import ScenePlayer, STAMP_ORIGIN, _read_stamp, _stamp_rect, read_stamp, scenes
from monitor_target import select_monitor


ARMS = ("rapidshot_poll", "dxcam_poll", "mss_gdi", "rapidshot_stream", "dxcam_stream", "dxcam_unpaced")
POLL_CAMERA = {"rapidshot_poll": "rapidshot_full", "dxcam_poll": "dxcam", "mss_gdi": "mss"}
LABELS = {
    "rapidshot_poll": "RapidShot one-shot, 5 ms DXGI wait",
    "dxcam_poll": "DXcam one-shot, 0 ms DXGI wait",
    "mss_gdi": "MSS GDI",
    "rapidshot_stream": "RapidShot streaming at 100 fps",
    "dxcam_stream": "DXcam streaming at 100 fps",
    "dxcam_unpaced": "DXcam streaming unpaced",
}
NS_MS = 1_000_000
DEFAULT_OUT = Path(__file__).resolve().parent / "results-timing-followup"


def frequency():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    value = ct.c_longlong()
    if not kernel.QueryPerformanceFrequency(ct.byref(value)) or not value.value:
        raise ct.WinError(ct.get_last_error())
    before = time.perf_counter_ns()
    tick = ct.c_longlong()
    if not kernel.QueryPerformanceCounter(ct.byref(tick)):
        raise ct.WinError(ct.get_last_error())
    after = time.perf_counter_ns()
    converted = tick.value * 1_000_000_000 // value.value
    if not before - 1000 <= converted <= after + 1000:
        raise RuntimeError("QPC and perf_counter_ns do not share a verified timing origin")
    return value.value


class Probe:
    def __init__(self, arm, target):
        self.arm = arm
        self.owner = None
        self.stream_records = {}
        self.reader_thread = threading.get_ident()
        self.selected = None
        self.producer = {}
        if arm in POLL_CAMERA:
            self.owner = Camera(POLL_CAMERA[arm], target)
            self.camera = self.owner.camera
            return
        common = dict(device_idx=target["device_index"], output_idx=target["output_index"],
                      backend="dxgi", output_color="BGRA")
        if arm.startswith("rapidshot") or arm == "witness":
            import rapidshot

            opts = dict(timeout_ms=2 if arm == "witness" else 5, nvidia_gpu=False, pool_output=True)
            if arm == "witness":
                opts["region"] = _stamp_rect(STAMP_ORIGIN)
            self.camera = rapidshot.create(**common, **opts)
            if arm == "rapidshot_stream":
                self._tap_rapidshot_queue()
                self.camera.start(target_fps=100, video_mode=False)
        else:
            import dxcam

            self.camera = dxcam.create(**common)
            self._tap_dxcam_producer()
            self.camera.start(target_fps=0 if arm == "dxcam_unpaced" else 100, video_mode=False)

    def _tap_rapidshot_queue(self):
        queue_frame = self.camera._queue_frame
        forget_queued = self.camera._forget_queued

        def enqueue(frame, pixels_qpc, confirmed_qpc=None):
            duplicate = self.camera._duplicator
            self.stream_records[id(frame)] = (int(duplicate.last_present_time),
                                             int(duplicate.accumulated_frames),
                                             confirmed_qpc)
            return queue_frame(frame, pixels_qpc, confirmed_qpc)

        def forget(frame, now_qpc=None):
            # Both callbacks run under the camera's existing queue lock.
            measured = self.stream_records.pop(id(frame), None)
            if threading.get_ident() == self.reader_thread:
                self.selected = measured
            return forget_queued(frame, now_qpc)

        self.camera._queue_frame = enqueue
        self.camera._forget_queued = forget

    def _tap_dxcam_producer(self):
        grab_into = self.camera._grab_into

        def instrumented(region, destination):
            result = grab_into(region, destination)
            if result[0]:
                ticks = result[1]
                self.producer[ticks] = (int(self.camera._duplicator.accumulated_frames),
                                        time.perf_counter_ns())
            return result

        self.camera._grab_into = instrumented

    def read(self, qpc_rate):
        if self.arm == "rapidshot_stream":
            self.selected = None
            buffer = self.camera.get_latest_frame_buffer()
            if buffer is None:
                return None, None, None, None
            details = self.selected
            try:
                pixels = np.array(getattr(buffer, "array", buffer), copy=True)
            finally:
                release = getattr(buffer, "release", None)
                if release is not None:
                    release()
            if details is None:
                raise RuntimeError("RapidShot stream delivered a buffer without its queued timestamp")
            ticks, accumulated, producer_qpc = details
            return Capture(pixels, pixels), ticks, accumulated, (producer_qpc, "qpc")

        if self.arm in ("dxcam_stream", "dxcam_unpaced"):
            result = self.camera.get_latest_frame(copy=True, with_timestamp=True)
            if result is None:
                return None, None, None, None
            pixels, seconds = result
            ticks = round(seconds * qpc_rate)
            if ticks not in self.producer:
                raise RuntimeError("DXcam stream delivered a buffer without its producer timestamp")
            accumulated, ready_ns = self.producer[ticks]
            return Capture(pixels, pixels), ticks, accumulated, (ready_ns, "ns")

        if self.owner is not None:
            shot = self.owner.read()
        else:
            handle = self.camera.grab()
            shot = None if handle is None else Capture(handle, getattr(handle, "array", handle))
        if shot is None or self.arm == "mss_gdi":
            return shot, None, None, None
        if self.arm == "dxcam_poll":
            duplicate = self.camera._duplicator
            return shot, int(duplicate.latest_frame_ticks), int(duplicate.accumulated_frames), None
        duplicate = self.camera._duplicator
        return shot, int(duplicate.last_present_time), int(duplicate.accumulated_frames), None

    def close(self):
        if self.owner is not None:
            self.owner.close()
        else:
            self.camera.release()


def observe(probe, rate, config):
    start_ns, stop_ns = config["start_ns"], config["stop_ns"]
    wait_until(start_ns)
    cpu_start = time.process_time_ns()
    observations, empty = [], []
    while (called_ns := time.perf_counter_ns()) < stop_ns:
        frame, tick, accumulated, producer = probe.read(rate)
        returned_ns = time.perf_counter_ns()
        try:
            if frame is None:
                empty.append([called_ns, returned_ns])
                if probe.arm in ("rapidshot_poll", "dxcam_poll", "witness"):
                    time.sleep(0.0005)
                continue
            stamp = (_read_stamp(frame.pixels, (0, 0)) if probe.arm == "witness" else read_stamp(frame.pixels))
            observations.append({
                "stamp": stamp, "called_ns": called_ns, "returned_ns": returned_ns,
                "present_qpc": tick if tick and accumulated else None,
                "accumulated_frames": accumulated,
                "producer_ready_ns": (producer[0] * 1_000_000_000 // rate
                                      if producer[1] == "qpc" else producer[0]) if producer else None,
            })
        finally:
            if frame is not None:
                frame.close()
    return {"qpc_frequency": rate, "observations": observations, "empty_calls": empty,
            "process_cpu_ms": (time.process_time_ns() - cpu_start) / NS_MS}


def worker(arm, target):
    dpi_aware()
    qpc_rate = frequency()
    probe = Probe(arm, target)
    try:
        print("READY", flush=True)
        config = json.loads(sys.stdin.readline())
        result = observe(probe, qpc_rate, config)
        print("OBS " + json.dumps(result, allow_nan=False, separators=(",", ":")), flush=True)
    finally:
        probe.close()


class Observer:
    def __init__(self, arm, target):
        self.arm = arm
        self.process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", __file__, "--worker", arm,
             "--target", json.dumps(target, separators=(",", ":"))],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf8", creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.ready = threading.Event()
        self.result = None
        self.log = []
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.strip() == "READY":
                    self.ready.set()
                elif line.startswith("OBS "):
                    self.result = json.loads(line[4:])
                elif len(self.log) < 60:
                    self.log.append(line.strip())
        finally:
            self.ready.set()

    def __enter__(self):
        if not self.ready.wait(12) or not self.reader.is_alive():
            self.__exit__()
            raise RuntimeError(f"{self.arm} observer could not open: {' '.join(self.log[-10:])}")
        return self

    def start(self, start_ns, stop_ns):
        self.process.stdin.write(json.dumps(dict(start_ns=start_ns, stop_ns=stop_ns)) + "\n")
        self.process.stdin.flush()

    def read(self):
        self.reader.join(12)
        if self.reader.is_alive() or self.process.wait(timeout=2) or self.result is None:
            raise RuntimeError(f"{self.arm} observer failed: {' '.join(self.log[-14:])}")
        return self.result

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=5)
        self.reader.join(2)
        self.process.stdin.close()
        self.process.stdout.close()


def trial(player, steps, sentinel, target, arm, rate, witness):
    period_ns = round(1_000_000_000 / rate)
    player.present(steps[0], restart=True)
    with ExitStack() as stack:
        camera = stack.enter_context(Observer(arm, target))
        reference = stack.enter_context(Observer("witness", target)) if witness else None
        first = time.perf_counter_ns() + 100 * NS_MS
        initial = first - 50 * NS_MS
        stop = first + len(steps[1:]) * period_ns + 150 * NS_MS
        for observer in (camera, reference):
            if observer is not None:
                observer.start(initial, stop)
        submitted = []
        for index, step in enumerate(steps[1:]):
            due = first + period_ns * index
            wait_until(due)
            begin = time.perf_counter_ns()
            player.present(step, wait_for_desktop=False)
            submitted.append({"stamp": step.stamp, "due_ns": due, "send_start_ns": begin,
                              "send_end_ns": time.perf_counter_ns()})
        wait_until(stop + 5 * NS_MS)
        # Wake a blocking reader whose last scheduled appearance has already ended.
        player.present(sentinel, restart=True)
        return {"submitted": submitted, "camera": camera.read(),
                "witness": reference.read() if reference else None}


def first_per_stamp(run, key):
    data = run[key]
    submitted = {s["stamp"]: s for s in run["submitted"]}
    appearances = {}
    if data is not None:
        for frame in data["observations"]:
            stamp = frame["stamp"]
            if stamp not in submitted or frame["returned_ns"] < submitted[stamp]["send_start_ns"]:
                continue
            appearances.setdefault(stamp, frame)
    return appearances


def measurements(run):
    camera = first_per_stamp(run, "camera")
    witness = first_per_stamp(run, "witness")
    values = {}
    qpc_rate = run["camera"]["qpc_frequency"]
    for step in run["submitted"]:
        stamp = step["stamp"]
        if stamp not in camera:
            continue
        frame = camera[stamp]
        observed = witness.get(stamp) if run["arm"] == "mss_gdi" else frame
        tick = observed["present_qpc"] if observed else None
        desktop_ns = tick * 1_000_000_000 // qpc_rate if tick else None
        if desktop_ns is not None and not step["send_start_ns"] <= desktop_ns <= frame["returned_ns"]:
            desktop_ns = None
        values[stamp] = {
            "submit_to_return_ms": (frame["returned_ns"] - step["send_start_ns"]) / NS_MS,
            "flush_to_return_ms": (frame["returned_ns"] - step["send_end_ns"]) / NS_MS,
            "submit_to_desktop_ms": ((desktop_ns - step["send_start_ns"]) / NS_MS
                                     if desktop_ns is not None else None),
            "desktop_to_return_ms": ((frame["returned_ns"] - desktop_ns) / NS_MS
                                     if desktop_ns is not None else None),
            "desktop_to_producer_ms": ((frame["producer_ready_ns"] - desktop_ns) / NS_MS
                                       if desktop_ns is not None and frame["producer_ready_ns"] else None),
        }
    return values, witness


def quantiles(values):
    v = [n for n in values if n is not None]
    return {"n": len(v), "p50": statistics.median(v) if v else None,
            "p95": float(np.percentile(v, 95)) if v else None}


def summarize(runs):
    result = {}
    reference_arm = "rapidshot_poll"
    for has_witness in (False, True):
        group = [r for r in runs if bool(r["witness"]) == has_witness]
        stage = result["witness" if has_witness else "unassisted"] = {}
        for hz in sorted({r["rate"] for r in group}):
            rate_rows = [r for r in group if r["rate"] == hz]
            output = stage[str(hz)] = {}
            for arm in ARMS:
                chosen = [r for r in rate_rows if r["arm"] == arm]
                if not chosen:
                    continue
                reference_by_round = {r["round"]: measurements(r)[0] for r in rate_rows
                                      if r["arm"] == reference_arm}
                stages, differences, intervals, cpu = [], [], [], []
                observed = offered = witness_seen = observed_by_both = on_time = 0
                for run in chosen:
                    timing, witness = measurements(run)
                    reference = reference_by_round.get(run["round"], {})
                    offered += len(run["submitted"])
                    observed += len(timing)
                    witness_seen += len(witness)
                    observed_by_both += len(witness.keys() & timing.keys())
                    on_time += sum(x["submit_to_return_ms"] <= 1000 / hz for x in timing.values())
                    stages.extend(timing.values())
                    differences.extend(timing[k]["submit_to_return_ms"] - reference[k]["submit_to_return_ms"]
                                       for k in timing.keys() & reference.keys())
                    starts = [step["send_start_ns"] for step in run["submitted"]]
                    intervals.extend((b - a) / NS_MS for a, b in zip(starts, starts[1:]))
                    cpu.append(run["camera"]["process_cpu_ms"])
                fields = ("submit_to_return_ms", "flush_to_return_ms", "submit_to_desktop_ms",
                          "desktop_to_return_ms", "desktop_to_producer_ms")
                output[arm] = {
                    "offered": offered, "observed": observed, "on_time": on_time,
                    "witness_seen": witness_seen, "observed_by_both": observed_by_both,
                    "empty_calls": sum(len(r["camera"]["empty_calls"]) for r in chosen),
                    "accumulated_gt_one": sum(f["accumulated_frames"] > 1
                                              for r in chosen for f in r["camera"]["observations"]
                                              if f["accumulated_frames"] is not None),
                    "source_interval_ms": quantiles(intervals), "process_cpu_ms_per_run": quantiles(cpu),
                    "stages": {key: quantiles([row[key] for row in stages]) for key in fields},
                    "matched_delta_vs_rapidshot_poll_ms": quantiles(differences),
                }
    return result


def report(data):
    def fmt(row):
        return "—" if row["p50"] is None else f'{row["p50"]:.2f} / {row["p95"]:.2f} ({row["n"]})'

    target = data["target"]
    rows = ["# Timestamped capture-only follow-up", "",
            f'{target["monitor"]}, 1920×1080 at {target["physical_refresh_hz"]} Hz, {target["adapter"]}.',
            "", "A separate producer submits the same sparse fixture states in every mode. Capture starts before those submissions and drains after the last one. Coverage includes frames returned after their deadlines. Streaming waits on each library’s next-frame API. The unpaced DXcam control shows the effect of removing its producer timer. Both stream consumers own their returned bytes.", "",
            "All times are QPC-based and shown as median / 95th percentile (sample count). DXGI timestamps describe desktop-image updates. A small DXGI marker crop runs as an independent witness only in the witness rounds. MSS timings against DXGI come from that marker reference. A marker absent from both streams may have been coalesced or missed by both. Comparisons on common states match fixture indices within each round across separately timed runs.", "",
            "RapidShot poll retains a pooled lease. DXcam poll returns an owned array. Streaming returns owned copies for both. The prior request-per-marker sweep used different scheduling and discarded some captures once their expected update had expired.", ""]
    for block, rates in data["summary"].items():
        rows.extend([f'## {"Without witness" if block == "unassisted" else "With independent marker witness"}', "",
                     "| Updates/s | Capture mode | Returned / submitted | On time | Witness only | App submit → return ms | Desktop update → return ms | Matched submit-to-return delta vs RapidShot one-shot ms |",
                     "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |"])
        for hz, arms in rates.items():
            for arm, row in arms.items():
                stages = row["stages"]
                extra = row["witness_seen"] - row["observed_by_both"] if block == "witness" else "—"
                rows.append(f'| {hz} | {LABELS[arm]} | {row["observed"]}/{row["offered"]} | {row["on_time"]} | {extra} | '
                            f'{fmt(stages["submit_to_return_ms"])} | {fmt(stages["desktop_to_return_ms"])} | '
                            f'{fmt(row["matched_delta_vs_rapidshot_poll_ms"])} |')
        rows.append("")
    rows.append("[Raw observations](capture-raw.json)")
    return "\n".join(rows) + "\n"


def run(args):
    dpi_aware()
    frequency()
    target = select_monitor(args.monitor)
    if target["physical_refresh_hz"] < max(args.rates):
        raise RuntimeError("The selected display refresh is below the requested producer rate")
    args.out.mkdir(parents=True, exist_ok=True)
    raw_path = args.out / "capture-raw.json"
    if raw_path.exists():
        raise FileExistsError(raw_path)
    source = scenes(visible_steps=args.steps, passive_steps=1)
    rng = random.Random(args.seed)
    data = {"target": target, "rates": args.rates, "steps": args.steps, "seed": args.seed,
            "versions": {k: version(k) for k in ("rapidshot", "dxcam", "mss")}, "runs": []}
    with ScenePlayer(target["bounds"]) as player:
        player.prepare({name: source[name] for name in ("sparse", "dense")})
        for has_witness, rounds in ((False, args.rounds), (True, args.witness_rounds)):
            for repetition in range(1, rounds + 1):
                schedule = [(rate, arm) for rate in args.rates for arm in args.arms]
                rng.shuffle(schedule)
                for rate, arm in schedule:
                    print(f'{"Witness" if has_witness else "Primary"} round {repetition}/{rounds}, {rate}/s, {arm}',
                          flush=True)
                    sample = trial(player, source["sparse"], source["dense"][0], target, arm, rate, has_witness)
                    data["runs"].append({**sample, "round": repetition, "rate": rate, "arm": arm})
                write_json(raw_path, data)
    data["summary"] = summarize(data["runs"])
    write_json(raw_path, data)
    (args.out / "capture-report.md").write_text(report(data), encoding="utf8")
    print(f"Saved {len(data['runs'])} capture trials to {args.out}", flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--monitor", default="HP 324pe")
    cli.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    cli.add_argument("--rates", type=int, nargs="+", default=[10, 30, 100])
    cli.add_argument("--steps", type=int, default=24)
    cli.add_argument("--rounds", type=int, default=5)
    cli.add_argument("--witness-rounds", type=int, default=2)
    cli.add_argument("--seed", type=int, default=3242026)
    cli.add_argument("--out", type=Path, default=DEFAULT_OUT)
    cli.add_argument("--worker", help=argparse.SUPPRESS)
    cli.add_argument("--target", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.worker is not None:
        worker(args.worker, json.loads(args.target))
    else:
        args.rates = list(dict.fromkeys(args.rates))
        args.arms = list(dict.fromkeys(args.arms))
        if (not 1 <= args.steps <= 64 or args.rounds < 1 or args.witness_rounds < 0
                or any(not 1 <= rate <= 100 for rate in args.rates)):
            cli.error("require 1–64 steps, positive rounds, and rates from 1 to 100")
        run(args)


if __name__ == "__main__":
    main()
