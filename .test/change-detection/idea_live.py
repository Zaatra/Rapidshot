"""Short live DXGI tests on the named monitor. No persistent display changes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes as ct
from ctypes import wintypes as wt
import json
import os
from pathlib import Path
from queue import Empty, Queue
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from capture_timing_followup import frequency
from cost_telemetry import ProcessSampler
from downstream_scene import ICON_SIZE, ScenePlayer, read_stamp, render, scenes, templates
from monitor_target import DisplayMode, select_monitor, windows_displays
from downstream_benchmark import dpi_aware, wait_until

NS_MS = 1_000_000


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "p50": statistics.median(values),
            "p95": values[max(0, (95 * len(values) + 99) // 100 - 1)]}


def save(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf8")
    temporary.replace(path)


def display_api():
    api = ct.WinDLL("user32", use_last_error=True)
    api.EnumDisplaySettingsW.argtypes = [wt.LPCWSTR, wt.DWORD, ct.POINTER(DisplayMode)]
    api.ChangeDisplaySettingsExW.argtypes = [wt.LPCWSTR, ct.POINTER(DisplayMode), wt.HWND,
                                            wt.DWORD, ct.c_void_p]
    api.ChangeDisplaySettingsExW.restype = wt.LONG
    return api


def current_mode(api, output):
    mode = DisplayMode(size=ct.sizeof(DisplayMode))
    if not api.EnumDisplaySettingsW(output, 0xffffffff, ct.byref(mode)):
        raise RuntimeError(f"Could not query display mode of {output}")
    return mode


def mode_key(mode):
    return [mode.position.x, mode.position.y, mode.width, mode.height, mode.bpp,
            mode.refresh_hz, mode.rotation, mode.fixed, mode.flags]


def restore_display(journal):
    if not journal.exists():
        return
    data = json.loads(journal.read_text(encoding="utf8"))
    displays = windows_displays()
    if not any(row["output"] == data["output"] and row["monitor"] == data["monitor"]
               and row["bounds"] == data["bounds"] for row in displays):
        raise RuntimeError("Original display identity or bounds changed. Mode journal retained.")
    api = display_api()
    original = DisplayMode.from_buffer_copy(bytes.fromhex(data["mode"]))
    current = current_mode(api, data["output"])
    if mode_key(current) != mode_key(original):
        status = api.ChangeDisplaySettingsExW(data["output"], ct.byref(original), None, 0, None)
        if status != 0 or mode_key(current_mode(api, data["output"])) != mode_key(original):
            raise RuntimeError(f"Original display mode could not be restored: status={status}")
    journal.unlink()


@contextmanager
def temporary_refresh(target, hz, journal):
    output = target["output"]
    api = display_api()
    before = current_mode(api, output)
    if before.refresh_hz == hz:
        yield
        return
    request = DisplayMode.from_buffer_copy(bytes(before))
    request.refresh_hz = hz
    request.fields |= 0x400000  # DM_DISPLAYFREQUENCY
    status = api.ChangeDisplaySettingsExW(output, ct.byref(request), None, 2, None)  # CDS_TEST
    if status != 0:
        raise RuntimeError(f"Driver rejected {hz} Hz in mode test: {status}")
    save(journal, {"output": output, "monitor": target["monitor"], "bounds": target["bounds"],
                   "mode": bytes(before).hex()})
    try:
        status = api.ChangeDisplaySettingsExW(output, ct.byref(request), None, 0, None)
        applied = current_mode(api, output)
        if status != 0 or mode_key(applied) != mode_key(request):
            raise RuntimeError(f"Driver did not apply the requested mode: {status}, {mode_key(applied)}")
        yield
    finally:
        restore_display(journal)


def make_camera(target, timeout):
    import rapidshot

    return rapidshot.create(device_idx=target["device_index"], output_idx=target["output_index"],
                            backend="dxgi", output_color="BGRA", pool_output=True,
                            nvidia_gpu=False, timeout_ms=timeout)


def worker(config):
    dpi_aware()
    target = config["target"]
    stop = threading.Event()
    ready = threading.Event()
    result = {}

    def capture():
        camera = None
        try:
            camera = make_camera(target, config["timeout_ms"])
            if config["case"] == "idle":
                arm = config.get("idle_arm", "plain")
                if arm == "exact":
                    acquire = camera.grab_with_changes
                elif arm == "tiled":
                    acquire = camera.grab_tiles
                elif arm == "tight":
                    def acquire():
                        return camera.grab_tiles(tight_rectangles=True)
                else:
                    acquire = camera.grab
                ready.set()
                calls = 0
                delivery_times_ns = []
                while not stop.is_set():
                    frame = acquire()
                    calls += 1
                    if frame is not None:
                        delivery_times_ns.append(time.perf_counter_ns())
                        frame.release()
                history = camera._tile_history or camera._change_history
                result.update(grab_calls=calls, delivered=len(delivery_times_ns), delivery_times_ns=delivery_times_ns,
                              history_bytes=history.size_bytes if history else 0)
            else:
                collect_scene(camera, config, ready, stop, result)
        except Exception as exc:
            result["error"] = repr(exc)
        finally:
            ready.set()
            if camera is not None:
                camera.release()

    thread = threading.Thread(target=capture)
    thread.start()
    if not ready.wait(8):
        raise RuntimeError("DXGI worker did not initialize")
    if "error" in result:
        print(json.dumps(result), flush=True)
        return
    print(json.dumps({"pid": os.getpid()}), flush=True)
    sys.stdin.readline()
    stopped_ns = time.perf_counter_ns()
    stop.set()
    thread.join(4)
    if thread.is_alive():
        raise RuntimeError("DXGI worker failed to stop")
    result["stop_to_release_ms"] = (time.perf_counter_ns() - stopped_ns) / NS_MS
    print(json.dumps(result, allow_nan=False), flush=True)


def fixture(steps):
    source = scenes(visible_steps=steps, passive_steps=4)
    return source, source["sparse"]


def collect_scene(camera, config, ready, stop, result):
    import cv2
    from downstream_scene import render

    _, steps = fixture(config["steps"])
    regions = []
    for name in templates():
        positions = [step.objects[name] for step in steps if name in step.objects]
        if positions:
            x, y = zip(*positions)
            regions.append((min(x), min(y), max(x) + ICON_SIZE, max(y) + ICON_SIZE))
    expected = {}
    for step in steps:
        raster = render(step, 1920, 1080)
        expected[step.stamp] = [(r, raster[r[1]:r[3], r[0]:r[2], :3].copy()) for r in regions]
    last = None
    last_ticks = None
    rows = []
    evidence = []
    frequency_hz = frequency()
    ready.set()
    while not stop.is_set():
        frame = camera.grab()
        returned = time.perf_counter_ns()
        if frame is None:
            continue
        try:
            duplicate = camera._duplicator
            ticks = int(duplicate.last_present_time)
            present = ticks * 1_000_000_000 // frequency_hz if ticks else None
            pixels = frame.array
            stamp = read_stamp(pixels)
            if stamp not in expected or ticks == last_ticks or not present:
                continue
            correct = all(np.array_equal(pixels[t:b, l:r, :3], wanted)
                          for (l, t, r, b), wanted in expected[stamp])
            same = (last is not None and np.array_equal(last, pixels)) if config["verify"] else False
            if config["verify"]:
                if last is None:
                    last = np.empty_like(pixels)
                np.copyto(last, pixels)
            checked = time.perf_counter_ns()
            last_ticks = ticks
            row = {"stamp": stamp, "present_ns": present, "return_ns": returned,
                   "checked_ns": checked, "correct": correct, "same_as_previous": same,
                   "accumulated_frames": int(duplicate.accumulated_frames)}
            rows.append(row)
            if not correct and config["verify"] and len(evidence) < 12:
                evidence.append((len(rows), pixels.copy()))
        finally:
            frame.release()
    for index, pixels in evidence:
        filename = Path(config["evidence"]) / f"mixed-{index:03}.png"
        if not cv2.imwrite(str(filename), pixels):
            raise RuntimeError(f"could not save {filename}")
    result.update(rows=rows, delivered=len(rows), verification_history_bytes=(last.nbytes if last is not None else 0),
                  saved_wrong_frames=len(evidence), verification_history_writes=sum(
                      last.nbytes for _ in rows) if last is not None else 0)


class Worker:
    def __init__(self, config, script=None):
        self.config = config
        self.script = __file__ if script is None else str(script)
        self.records = Queue()

    def __enter__(self):
        self.process = subprocess.Popen(
            [sys.executable, "-E", "-X", "utf8", "-B", self.script, "--worker", json.dumps(self.config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf8", creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.pid = self.get(10)["pid"]
        except BaseException:
            self.__exit__()
            raise
        return self

    def _read(self):
        for line in self.process.stdout:
            self.records.put(line.strip())
        self.records.put(None)

    def get(self, seconds):
        end = time.perf_counter() + seconds
        diagnostics = []
        while True:
            try:
                line = self.records.get(timeout=max(0, end - time.perf_counter()))
            except Empty:
                line = None
            if line is None:
                raise RuntimeError(f"Worker gave no result: {diagnostics[-6:]}")
            try:
                result = json.loads(line)
            except ValueError:
                diagnostics.append(line)
                continue
            if isinstance(result, dict):
                if "error" in result:
                    raise RuntimeError(result["error"])
                return result

    def stop(self, timeout=7):
        self.process.stdin.write("stop\n")
        self.process.stdin.flush()
        return self.get(timeout)

    def __exit__(self, *_):
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.reader.join(2)
        self.process.stdin.close()
        self.process.stdout.close()


def idle_case(player, idle_step, target, timeout, arm="plain"):
    player.present(idle_step, restart=True)
    with Worker(dict(target=target, case="idle", timeout_ms=timeout, idle_arm=arm)) as process:
        time.sleep(1.0)
        sampler = ProcessSampler(process.pid).start()
        try:
            time.sleep(10.05)
            measured = sampler.finish()
        finally:
            sampler.close()
        result = process.stop()
    cpu = measured["cpu"]
    delivered = sum(cpu["start_ns"] <= at <= cpu["end_ns"] for at in result["delivery_times_ns"])
    cpu.update(delivered_frames=delivered, cycles_per_delivered_frame=(cpu["cycles"] / delivered if delivered else None))
    return {"timeout_ms": timeout, "telemetry": measured, "quiet_window": not delivered,
            "deliveries_in_window": delivered, **result}


def quiet_tiled(output, rounds, seconds):
    from random import Random

    dpi_aware()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "quiet-tiled.json"
    target = select_monitor()
    deadline = time.perf_counter() + seconds
    data = {"target": target, "rounds": [], "limit_cpu_points": 0.5, "qualification":
            "A worker must deliver no frames during its measured idle window. Exact histories must be seeded."}
    random = Random(61598)
    with ScenePlayer(tuple(target["bounds"])) as player:
        idle = scenes(visible_steps=1, passive_steps=1)["idle"][0]
        for number in range(1, rounds + 1):
            order = ["plain", "exact", "tiled", "tight"]
            random.shuffle(order)
            row = {"round": number, "order": order, "arms": {}}
            data["rounds"].append(row)
            for arm in order:
                if time.perf_counter() + 14 > deadline:
                    raise TimeoutError("The quiet follow-up reached its time limit")
                result = idle_case(player, idle, target, 5, arm)
                result["qualifying"] = result["quiet_window"] and (arm == "plain" or result["history_bytes"] > 0)
                row["arms"][arm] = result
                save(path, data)
    return 0


def classify_observations(rows, submissions):
    useful = [row for row in rows if row["stamp"] in submissions and
              submissions[row["stamp"]] <= row["present_ns"] <= row["return_ns"]]
    first, confirmed = {}, {}
    previous = None
    for row in useful:
        stamp = row["stamp"]
        first.setdefault(stamp, row)
        if row["same_as_previous"] and previous is not None and previous["stamp"] == stamp:
            confirmed.setdefault(stamp, {**row, "first_present_ns": previous["present_ns"]})
        previous = row
    output = {"observed_acquisitions": len(useful), "offered": len(submissions), "rows": useful}
    for label, group in (("immediate", first), ("two_distinct_equal_acquisitions", confirmed)):
        output[label] = {"delivered_states": len(group), "wrong_states": sum(not r["correct"] for r in group.values()),
                         "missed_states": len(submissions) - len(group),
                         "pixel_age_at_capture_ms": distribution((r["return_ns"] - r["present_ns"]) / NS_MS
                                                                   for r in group.values()),
                         "pixel_age_at_result_ms": distribution((r["checked_ns"] - r.get("first_present_ns", r["present_ns"]))
                                                                  / NS_MS for r in group.values())}
    for hold_ms in (10, 20):
        selected, superseded = {}, 0
        for pos, row in enumerate(useful):
            stamp = row["stamp"]
            expiry = row["checked_ns"] + hold_ms * NS_MS
            following = useful[pos + 1] if pos + 1 < len(useful) else None
            if following is not None and following["return_ns"] < expiry:
                superseded += 1
                continue
            selected.setdefault(stamp, (row, expiry))
        output[f"hold_{hold_ms}ms_replaced_by_newer_acquisition"] = {
            "delivered_states": len(selected), "wrong_states": sum(not r["correct"] for r, _ in selected.values()),
            "missed_states": len(submissions) - len(selected), "superseded_acquisitions": superseded,
            "pixel_age_at_result_ms": distribution((t - r["present_ns"]) / NS_MS for r, t in selected.values()),
            "simulated_from_live_arrivals": True}
    return output


def active_case(target, rate, verify, evidence):
    source, steps = fixture(64 if verify else 48)
    player = ScenePlayer(tuple(target["bounds"]))
    player.prepare({"sparse": steps})
    evidence.mkdir(parents=True, exist_ok=True)
    period = round(1e9 / rate)
    with player:
        player.present(steps[0], restart=True)
        with Worker(dict(case="active", target=target, timeout_ms=5, steps=len(steps) - 1,
                         verify=verify, evidence=str(evidence))) as process:
            sampler = ProcessSampler(process.pid).start()
            try:
                first = time.perf_counter_ns() + 250 * NS_MS
                submissions = {}
                for i, step in enumerate(steps[1:]):
                    wait_until(first + period * i)
                    submissions[step.stamp] = time.perf_counter_ns()
                    player.present(step, wait_for_desktop=False)
                wait_until(first + period * len(steps) + 100 * NS_MS)
                telemetry = sampler.finish()
            finally:
                sampler.close()
            outcome = process.stop()
    judged = classify_observations(outcome.pop("rows"), submissions)
    count = sum(telemetry["cpu"]["start_ns"] <= r["checked_ns"] <= telemetry["cpu"]["end_ns"]
                for r in judged["rows"])
    telemetry["cpu"].update(delivered_frames=count, cycles_per_delivered_frame=(
        telemetry["cpu"]["cycles"] / count if count else None))
    return {"physical_hz": target["physical_refresh_hz"], "offered_hz": rate, "verification": verify,
            "telemetry": telemetry, **outcome, **judged}


def occlusion_case(target):
    from rapidshot.changes import TILE_SIZE

    left, top, _, _ = target["bounds"]
    step = scenes(visible_steps=1, passive_steps=1)["sparse"][0]
    x, y = 683, 587
    width, height = 185, 111
    color = (6, 124, 245, 255)
    cover = np.full((height, width, 4), color, dtype=np.uint8)
    background = render(step)[y + 9, x + 9, :3]
    qpc_rate = frequency()

    def get_after(camera, expected):
        until = time.perf_counter() + 0.6
        while time.perf_counter() < until:
            delivery = camera.grab_with_changes()
            ready = time.perf_counter_ns()
            if delivery is not None:
                if read_stamp(delivery.pixels) == step.stamp and np.array_equal(
                    delivery.pixels[y + 9, x + 9, :3], expected
                ):
                    return delivery, ready
                delivery.release()
        raise RuntimeError("Could not capture the requested visible overlay state on the test window")

    with ScenePlayer(tuple(target["bounds"])) as player:
        camera = make_camera(target, 5)
        try:
            player.present(step, restart=True)
            with get_after(camera, background)[0] as seed:
                previous = seed.pixels.copy()
            sampler = ProcessSampler(os.getpid()).start()
            samples = []

            def overlay_state(expected):
                nonlocal previous
                delivery, received = get_after(camera, expected)
                with delivery:
                    change = delivery.changes
                    image = delivery.pixels
                    yy, xx = np.where(np.any(previous != image, axis=2))
                    tiles = tuple(map(int, np.unique(yy // TILE_SIZE * (image.shape[1] // TILE_SIZE) + xx // TILE_SIZE)))
                    samples.append({"tiles": len(tiles), "oracle_match": tiles == change.indices,
                                    "pixel_age_at_capture_ms": (received - change.present_time_qpc * 1e9 / qpc_rate) / NS_MS
                                    if change.present_time_qpc else None})
                    previous = image.copy()

            try:
                with player.windows.scene((left + x, top + y, left + x + width, top + y + height),
                                          0xF57C06, excluded=False) as overlay:
                    player.windows.repaint(overlay.handle, cover, None, True)
                    overlay_state(cover[0, 0, :3])
                    player.windows.images.pop(overlay.handle, None)
                overlay_state(background)
                telemetry = sampler.finish(len(samples))
            finally:
                sampler.close()
        finally:
            camera.release()
    return {"samples": samples, "matched": all(row["oracle_match"] and row["tiles"] for row in samples),
            "telemetry": telemetry, "telemetry_includes_presenter_and_oracle": True}


def run(output, rounds, seconds, idle_only=False):
    dpi_aware()
    output.mkdir(parents=True, exist_ok=True)
    path = output / "live-ideas.json"
    journal = output / "display-restore.json"
    restore_display(journal)
    target = select_monitor()
    deadline = time.perf_counter() + seconds
    probes = {"target": target, "rounds": [], "errors": [], "restored_displays": None}

    def record(row, name, minimum, action):
        if time.perf_counter() + minimum > deadline:
            probes["errors"].append({"round": row["round"], "case": name, "error": "remaining deadline"})
        else:
            try:
                row[name] = action()
            except Exception as exc:
                probes["errors"].append({"round": row["round"], "case": name, "error": repr(exc)})
        save(path, probes)
        if journal.exists():
            raise RuntimeError("Display restore is still pending. Stopping display tests.")

    def at_60():
        with temporary_refresh(target, 60, journal):
            current = select_monitor()
            if current["physical_refresh_hz"] != 60:
                raise RuntimeError("60 Hz mode could not be verified")
            return active_case(current, 60, False, output / "60hz")

    try:
        if target["physical_refresh_hz"] != 100:
            raise RuntimeError("Expected the HP 324pe to start at 100 Hz")
        for number in range(1, rounds + 1):
            row = {"round": number}
            probes["rounds"].append(row)
            # Reverse idle order to reduce bias from warming or background drift.
            with ScenePlayer(tuple(target["bounds"])) as player:
                idle = scenes(visible_steps=1, passive_steps=1)["idle"][0]
                for timeout in ((5, 50) if number == 1 else (50, 5)):
                    record(row, f"idle_{timeout}ms", 12, lambda: idle_case(player, idle, target, timeout))
            if idle_only:
                continue
            record(row, "settled_100_at_100", 5,
                   lambda: active_case(target, 100, True, output / f"mixed-round-{number}"))
            record(row, "visible_occlusion", 4, lambda: occlusion_case(target))
            if number == 1:
                record(row, "control_60_at_100", 5, lambda: active_case(target, 60, False, output / "100hz"))
                record(row, "mode_60_at_60", 6, at_60)
            else:
                record(row, "mode_60_at_60", 6, at_60)
                record(row, "control_60_at_100", 5, lambda: active_case(target, 60, False, output / "100hz"))
    finally:
        restore_display(journal)
        probes["restored_displays"] = windows_displays()
        save(path, probes)
    return bool(probes["errors"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=92)
    parser.add_argument("--worker", type=json.loads)
    parser.add_argument("--restore", type=Path)
    parser.add_argument("--idle-only", action="store_true")
    parser.add_argument("--quiet-tiled", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
        return 0
    if args.restore:
        restore_display(args.restore)
        return 0
    if args.quiet_tiled:
        return quiet_tiled(args.output, args.rounds, args.seconds)
    return run(args.output, args.rounds, args.seconds, args.idle_only)


if __name__ == "__main__":
    raise SystemExit(main())
