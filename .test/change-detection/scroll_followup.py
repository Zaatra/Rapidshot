"""Record longer controlled app clips and count DXGI move rectangles."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import csv
import json
import os
from pathlib import Path
from queue import Full, Queue
import random
import subprocess
import sys
import time
import uuid
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from monitor_target import select_monitor
from real_trace import FORMAT, WindowGuard, browser_marker, metadata
from trace_launchers import (
    ROOT, Output, browser_motion, controlled_chrome, controlled_excel, dpi_context,
    place_fixture, process_id,
)


WORKLOADS = {
    "browser": "Chrome wheel scrolling a controlled dashboard",
    "spreadsheet": "Excel scrolling a controlled worksheet with two cell edits",
    "word": "Word scrolling a controlled document by line",
    "game": "Chrome Dino gameplay",
}


def write_member(archive, name, value):
    with archive.open(f"{name}.npy", "w") as entry:
        np.lib.format.write_array(entry, value, allow_pickle=False)


class ArchiveWriter:
    def __init__(self, pool, temp_path, max_bytes):
        self.queue = Queue(maxsize=12)
        self.task = pool.submit(self._write, temp_path, max_bytes)

    def _write(self, path, max_bytes):
        with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=1) as archive:
            while True:
                key, value = self.queue.get()
                if key is None:
                    if value is not None:
                        contract = np.frombuffer(json.dumps(value, allow_nan=False).encode(), dtype=np.uint8)
                        write_member(archive, "contract", contract)
                    return
                write_member(archive, key, value)
                if path.stat().st_size > max_bytes:
                    raise RuntimeError("Compressed trace exceeded its per-clip disk budget")

    def frame(self, index, pixels):
        if self.task.done():
            self.task.result()
        try:
            self.queue.put_nowait((f"frame_{index:04d}", pixels))
        except Full:
            return False
        return True

    def finish(self, contract):
        if not self.task.done():
            try:
                self.queue.put((None, contract), timeout=10)
            except Full as error:
                if self.task.done():
                    self.task.result()
                raise RuntimeError("Trace archive writer did not drain") from error
        self.task.result(timeout=20)


def intersects(rect, crop):
    return (rect[0] < crop[2] and rect[2] > crop[0] and
            rect[1] < crop[3] and rect[3] > crop[1])


def count_moves(entries, region):
    moves = Counter()
    for row in entries:
        rectangles = row["move_rects"]
        if rectangles is None:
            moves["metadata_unavailable"] += 1
            continue
        moves["frames_with_move_rects"] += bool(rectangles)
        moves["move_rects"] += len(rectangles)
        relevant = sum(intersects(rect[2:], region) for rect in rectangles)
        moves["frames_with_crop_moves"] += bool(relevant)
        moves["move_rects_intersecting_crop"] += relevant
    return dict(moves)


def record_clip(config):
    import rapidshot

    app, bounds = config["app"], config["bounds"]
    guard = WindowGuard(config["title"], config["pid"], bounds, config["region"], config["hwnd"])
    print("Waiting for the controlled fixture to cover the selected monitor region.", flush=True)
    wait_deadline = time.perf_counter() + 10
    while not guard.check(wait=True):
        if time.perf_counter() >= wait_deadline:
            raise RuntimeError(f"Could not verify the controlled fixture: {guard.reason}")
        time.sleep(0.08)
    left, top, right, bottom = guard.region
    crop = (left - bounds[0], top - bounds[1], right - bounds[0], bottom - bounds[1])
    width, height = right - left, bottom - top
    monitor_size = (bounds[2] - bounds[0], bounds[3] - bounds[1])
    out = Path(config["out"])
    temp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    camera = rapidshot.create(device_idx=config["device"], output_idx=config["output_index"],
                              backend="dxgi", output_color="BGRA", timeout_ms=1,
                              nvidia_gpu=False, pool_output=True, region=crop)
    contract = None
    rows, stored_rows = [], []
    counts = Counter()
    previous = None
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            writer = ArchiveWriter(pool, temp, config["max_file_mib"] * 1024**2)
            try:
                if tuple(camera.region) != crop or camera.rotation_angle:
                    raise RuntimeError(f"Unexpected crop {camera.region} or rotation {camera.rotation_angle}")
                guard.check()
                print("Recording the verified fixture.", flush=True)
                start = time.perf_counter_ns()
                deadline = start + round(config["seconds"] * 1e9)
                interval = round(1e9 / config["record_hz"])
                for due in range(start, deadline, interval):
                    remaining = due - time.perf_counter_ns()
                    if remaining > 0:
                        time.sleep(remaining / 1e9)
                    elif remaining < -interval:
                        counts["recorder_ticks_missed"] += 1
                        continue
                    guard.check()
                    owned = camera.grab()
                    captured = time.perf_counter_ns()
                    counts["capture_polls"] += 1
                    if owned is None:
                        counts["no_frame_polls"] += 1
                        continue
                    try:
                        guard.check()
                        raw = getattr(owned, "array", owned)
                        if raw.shape != (height, width, 4) or raw.dtype != np.uint8:
                            raise RuntimeError(f"Unexpected crop pixels {raw.shape} {raw.dtype}")
                        if app == "browser" and not browser_marker(raw):
                            raise RuntimeError("The controlled Chrome fixture marker was not visible")
                        pixels = np.array(raw, order="C", copy=True)
                        entry = metadata(camera, captured, start)
                        rows.append(entry)
                        if writer.frame(len(stored_rows), pixels):
                            if previous is not None:
                                key = "pixel_identical_pairs" if np.array_equal(previous, pixels) else "pixel_changed_pairs"
                                counts[key] += 1
                            previous = pixels
                            stored_rows.append(entry)
                        else:
                            counts["archive_queue_full"] += 1
                    finally:
                        release = getattr(owned, "release", None)
                        if release:
                            release()
                guard.check()
                if not counts["pixel_changed_pairs"]:
                    raise RuntimeError("The controlled fixture produced no changed captured pixels")
                contract = {
                    "format": FORMAT, "app": app, "monitor_bounds": bounds,
                    "monitor_size": monitor_size, "capture_region": crop,
                    "ownership_guard": "verified fixture title, PID, HWND, crop and occlusion",
                    "browser_fixture_marker": app == "browser", "output_index": config["output_index"],
                    "device": config["device"], "size": [width, height],
                    "record_target_hz": config["record_hz"], "requested_seconds": config["seconds"],
                    "memory_cap_reached": False, "poll_counts": dict(counts), "frames": stored_rows,
                    "dxgi_acquisitions": len(rows), "dxgi_moves": count_moves(rows, crop),
                }
            finally:
                writer.finish(contract)
        temp.rename(out)
    finally:
        camera.release()
        temp.unlink(missing_ok=True)
    print(f"Recorded {len(stored_rows)} images from {len(rows)} acquisitions to {out}", flush=True)


def word_pids():
    system = Path(os.environ["SystemRoot"]) / "System32" / "tasklist.exe"
    result = subprocess.run([str(system), "/FI", "IMAGENAME eq WINWORD.EXE", "/FO", "CSV", "/NH"],
                            check=True, stdout=subprocess.PIPE, text=True, timeout=5,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    return {int(row[1]) for row in csv.reader(result.stdout.splitlines())
            if len(row) > 1 and row[0].casefold() == "winword.exe"}


@contextmanager
def controlled_word():
    from comtypes.client import CreateObject

    existing = word_pids()
    app = CreateObject("Word.Application", dynamic=True)
    created = word_pids() - existing
    if len(created) != 1 or app.Documents.Count:
        raise RuntimeError("Could not identify the new empty Word process")
    doc = None
    owned = True
    try:
        app.DisplayAlerts = 0
        doc = app.Documents.Add()
        win = doc.ActiveWindow
        pid = process_id(int(win.Hwnd) & 0xFFFFFFFF)
        owned = pid in created
        if not owned:
            raise RuntimeError("Word fixture window did not belong to the new process")
        title = f"RapidShot controlled word {uuid.uuid4().hex[:8]}"
        win.Caption = title
        data = random.Random(9324)
        doc.Content.Text = "\r".join(
            f"Entry {i:03d}  |  File {data.randrange(1000, 9999)}  |  "
            f"{('Reviewed', 'Queued', 'Complete', 'Draft')[i % 4]}  |  Section {i // 20 + 1}"
            for i in range(1, 401)
        )
        doc.Content.Font.Name = "Consolas"
        doc.Content.Font.Size = 13
        app.Visible = True
        win.Activate()
        win.VerticalPercentScrolled = 0
        yield win, title, pid
    finally:
        if owned:
            try:
                if doc is not None:
                    doc.Close(0)
            finally:
                app.Quit(0)


def scroll_duration(seconds):
    return time.monotonic() + seconds - 1.1


def chrome_scroll(pipe, seconds):
    x, y = pipe.evaluate("[Math.floor(innerWidth / 2), Math.floor(innerHeight / 2)]")
    stop = scroll_duration(seconds)
    step = 0
    positions = set()
    while time.monotonic() < stop:
        delta = 98 if step % 32 < 18 else -82
        pipe.send("Input.dispatchMouseEvent", type="mouseWheel", x=x, y=y, deltaX=0, deltaY=delta)
        positions.add(pipe.evaluate("scrollY"))
        step += 1
        time.sleep(0.16 if step % 12 else 0.65)
    if len(positions) < 2:
        raise RuntimeError("The controlled Chrome page did not scroll")


def office_scroll(seconds, advance):
    stop = scroll_duration(seconds)
    step = 0
    while time.monotonic() < stop:
        advance(step)
        step += 1
        time.sleep(0.12 if step % 12 else 0.65)


def excel_scroll(sheet, win, seconds):
    def advance(step):
        row = 1 + 2 * (step % 38)
        win.ScrollRow = row
        if step in (8, 24):
            sheet.Range[f"D{row + 6}"].Value2 = 300 + step

    office_scroll(seconds, advance)


def word_scroll(win, seconds):
    positions = {int(win.VerticalPercentScrolled)}

    def advance(step):
        if step % 32 < 22:
            win.SmallScroll(3)
        else:
            win.SmallScroll(0, 3)
        positions.add(int(win.VerticalPercentScrolled))

    office_scroll(seconds, advance)
    if len(positions) < 2:
        raise RuntimeError("The controlled document did not scroll")


def run_clip(args, target, app, title, pid, region, motion):
    path = args.out / f"{app}.npz"
    if path.exists():
        raise FileExistsError(path)
    staged_path = path.with_name(f".{path.stem}.{os.getpid()}.staged.npz")
    hwnd = place_fixture(title, pid, target["bounds"])
    config = {
        "app": app, "title": title, "pid": pid, "hwnd": hwnd, "region": region,
        "bounds": target["bounds"], "device": target["device_index"], "output_index": target["output_index"],
        "record_hz": args.record_hz, "seconds": args.seconds, "max_file_mib": args.max_file_mib,
        "out": str(staged_path),
    }
    process = subprocess.Popen([sys.executable, "-X", "utf8", "-B", __file__, "_record", json.dumps(config)],
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", creationflags=subprocess.CREATE_NO_WINDOW)
    output = Output(process, mirror=True)
    try:
        if not output.read(16).startswith("Waiting for the controlled fixture"):
            raise RuntimeError("The guarded trace recorder did not arm")
        if not output.read(16).startswith("Recording the verified fixture"):
            raise RuntimeError("The guarded trace recorder did not verify the fixture")
        time.sleep(0.45)
        motion()
        if process.wait(timeout=28) or not staged_path.is_file():
            raise RuntimeError(f"{app} trace recording failed")
        verified = trace_summary(staged_path)
        staged_path.rename(path)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=4)
        output.reader.join(timeout=4)
        staged_path.unlink(missing_ok=True)
        staged_path.with_name(f".{staged_path.name}.{process.pid}.tmp").unlink(missing_ok=True)
    return {**verified, "file": path.name}


def trace_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        data = json.loads(archive["contract"].tobytes())
    acquisitions = data.get("dxgi_acquisitions", len(data["frames"]))
    counts = data.get("poll_counts", {})
    return {
        "app": data["app"], "workload": WORKLOADS[data["app"]], "file": path.name,
        "captured_span_seconds": (data["frames"][-1]["captured_ns"] - data["frames"][0]["captured_ns"]) / 1e9,
        "record_target_hz": data["record_target_hz"], "stored_images": len(data["frames"]),
        "dxgi_acquisitions": acquisitions, "archive_queue_full": counts.get("archive_queue_full", 0),
        "pixel_identical_pairs": counts.get("pixel_identical_pairs"),
        "pixel_changed_pairs": counts.get("pixel_changed_pairs"),
        **data.get("dxgi_moves", count_moves(data["frames"], data["capture_region"])),
    }


def launch(args, target, app):
    l, t, r, b = target["bounds"]
    sheet_body = (l + 70, t + 230, r - 175, b - 135)
    if app == "spreadsheet":
        with controlled_excel(target) as (sheet, win, title, pid):
            return run_clip(args, target, app, title, pid, sheet_body, lambda: excel_scroll(sheet, win, args.seconds))
    if app == "word":
        with controlled_word() as (win, title, pid):
            return run_clip(args, target, app, title, pid, sheet_body, lambda: word_scroll(win, args.seconds))
    with controlled_chrome(app, target) as (pipe, title, pid):
        action = chrome_scroll if app == "browser" else lambda p, s: browser_motion(p, "game", s)
        return run_clip(args, target, app, title, pid, None, lambda: action(pipe, args.seconds))


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "_record":
        record_clip(json.loads(sys.argv[2]))
        return
    sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", choices=(*WORKLOADS, "all"), default="all")
    parser.add_argument("--monitor", default="HP 324pe")
    parser.add_argument("--out", type=Path, default=ROOT / "results-timing-followup" / "traces")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--record-hz", type=int, default=30)
    parser.add_argument("--max-file-mib", type=int, default=384)
    args = parser.parse_args()
    if not 4 <= args.seconds <= 30 or not 1 <= args.record_hz <= 100 or not 1 <= args.max_file_mib <= 1024:
        parser.error("choose 4–30 seconds, 1–100 Hz and a per-clip disk budget of 1–1,024 MiB")
    dpi_context()
    target = select_monitor(args.monitor)
    if args.record_hz > target["physical_refresh_hz"]:
        raise RuntimeError("The requested recorder rate exceeds the selected monitor refresh")
    print(f"Recording on {target['monitor']} at {target['physical_refresh_hz']} Hz physical refresh", flush=True)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    results = {}
    failures = {}
    for app in WORKLOADS if args.app == "all" else (args.app,):
        try:
            if (args.out / f"{app}.npz").is_file():
                results[app] = trace_summary(args.out / f"{app}.npz")
            else:
                results[app] = launch(args, target, app)
            print(json.dumps(results[app], ensure_ascii=True), flush=True)
        except Exception as error:
            failures[app] = str(error)
            print(f"{app} failed: {error}", file=sys.stderr, flush=True)
    summary = args.out / ("scroll-summary.json" if args.app == "all" else f"{args.app}-scroll-summary.json")
    summary.write_text(json.dumps({"monitor": target, "results": results, "failures": failures}, indent=2), encoding="utf-8")
    if failures:
        raise SystemExit("Scroll capture failed for " + ", ".join(failures))


if __name__ == "__main__":
    main()
