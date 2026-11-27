"""Record controlled Windows application clips and replay exact tile search."""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import statistics
import time
from types import SimpleNamespace

import numpy as np

from cpu_default import Grid, Native, TILE
from downstream_task import TemplateSearch


RECT = tuple[int, int, int, int]
FORMAT = "rapidshot-controlled-app-trace-v1"
APP_NAMES = ("browser", "spreadsheet", "game")
MARKER_BGR = np.array(((212, 59, 113), (56, 145, 10), (22, 108, 229), (223, 168, 22)), dtype=np.uint8)


def bind(library, name, result, *args):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = args
    return function


def inside(outer, inner):
    l, t, r, b = inner
    return outer[0] <= l < r <= outer[2] and outer[1] <= t < b <= outer[3]


class WindowGuard:
    def __init__(self, title, pid, monitor, region, hwnd=None):
        self.expected_title = title.casefold()
        self.expected_pid = pid
        self.monitor = monitor
        self.requested_region = region
        self.expected_handle = hwnd
        self.handle = None
        self.region = None
        self.reason = None
        user = ct.WinDLL("user32", use_last_error=True)
        self.foreground = bind(user, "GetForegroundWindow", wt.HWND)
        self.get_text = bind(user, "GetWindowTextW", ct.c_int, wt.HWND, wt.LPWSTR, ct.c_int)
        self.get_pid = bind(user, "GetWindowThreadProcessId", wt.DWORD, wt.HWND, ct.POINTER(wt.DWORD))
        self.get_rect = bind(user, "GetClientRect", wt.BOOL, wt.HWND, ct.POINTER(wt.RECT))
        self.to_screen = bind(user, "ClientToScreen", wt.BOOL, wt.HWND, ct.POINTER(wt.POINT))
        self.window_at = bind(user, "WindowFromPoint", wt.HWND, wt.POINT)
        self.get_root = bind(user, "GetAncestor", wt.HWND, wt.HWND, wt.UINT)
        self.is_visible = bind(user, "IsWindowVisible", wt.BOOL, wt.HWND)
        self.is_minimized = bind(user, "IsIconic", wt.BOOL, wt.HWND)
        self.get_window = bind(user, "GetWindow", wt.HWND, wt.HWND, wt.UINT)
        self.get_window_rect = bind(user, "GetWindowRect", wt.BOOL, wt.HWND, ct.POINTER(wt.RECT))
        dpi = bind(user, "SetThreadDpiAwarenessContext", ct.c_void_p, ct.c_void_p)
        if not dpi(ct.c_void_p(-4)):
            raise ct.WinError(ct.get_last_error())

    def _client(self, handle):
        rect = wt.RECT()
        if not self.get_rect(handle, ct.byref(rect)):
            raise ct.WinError(ct.get_last_error())
        start, end = wt.POINT(rect.left, rect.top), wt.POINT(rect.right, rect.bottom)
        if not self.to_screen(handle, ct.byref(start)) or not self.to_screen(handle, ct.byref(end)):
            raise ct.WinError(ct.get_last_error())
        return start.x, start.y, end.x, end.y

    def _unavailable(self, reason, wait):
        self.reason = reason
        if wait:
            return False
        raise RuntimeError(f"{reason}. No recording was written.")

    def check(self, *, wait=False):
        current = self.expected_handle or self.foreground()
        title = ct.create_unicode_buffer(256)
        process = wt.DWORD()
        if current:
            self.get_text(current, title, len(title))
            self.get_pid(current, ct.byref(process))
        matches = (current and self.is_visible(current) and not self.is_minimized(current)
                   and self.expected_title in title.value.casefold()
                   and (self.expected_pid is None or self.expected_pid == process.value))
        if not matches:
            return self._unavailable("The visible window did not match the controlled fixture identity", wait)
        if self.handle is not None and self.handle != current:
            raise RuntimeError("The foreground window changed during capture. No recording was written.")
        client = self._client(current)
        if self.requested_region:
            region = self.requested_region
        else:
            inset = 12
            region = (max(client[0] + inset, self.monitor[0]), max(client[1] + inset, self.monitor[1]),
                      min(client[2] - inset, self.monitor[2]), min(client[3] - inset, self.monitor[3]))
        if not inside(client, region) or not inside(self.monitor, region):
            return self._unavailable(f"Fixture client {client} does not contain region {region} on the selected monitor", wait)
        if self.region is not None and region != self.region:
            raise RuntimeError("The fixture client area moved during capture. No recording was written.")
        l, t, r, b = region
        # Reject any visible top-level overlap, including a small overlay between sample points.
        above = self.get_window(current, 3)
        while above:
            rect = wt.RECT()
            if self.is_visible(above) and not self.is_minimized(above) and self.get_window_rect(above, ct.byref(rect)):
                if rect.left < r and rect.right > l and rect.top < b and rect.bottom > t:
                    return self._unavailable("A higher window intersects the fixture crop", wait)
            above = self.get_window(above, 3)
        for x in np.linspace(l + 3, r - 4, 5, dtype=int):
            for y in np.linspace(t + 3, b - 4, 4, dtype=int):
                owner = self.window_at(wt.POINT(int(x), int(y)))
                if owner != current and self.get_root(owner, 2) != current:
                    return self._unavailable("A point inside the crop is owned by another window", wait)
        self.handle = current
        self.region = region
        return True


def browser_marker(frame):
    # Search only a small header region. Match all four colors in order in the
    # same raster row so browser chrome cannot stand in for the local fixture.
    sample = frame[: min(180, frame.shape[0]), : min(180, frame.shape[1]), :3]
    positions = [np.all(sample == color, axis=2) for color in MARKER_BGR]
    for row in np.flatnonzero(np.logical_and.reduce([mask.any(axis=1) for mask in positions])):
        starts = [np.flatnonzero(mask[row])[0] for mask in positions]
        if all(2 <= b - a <= 40 for a, b in zip(starts, starts[1:])):
            return True
    return False


def metadata(camera, captured_ns, start):
    duplicator = camera._duplicator
    rects = duplicator.dirty_rects
    moves = duplicator.move_rects
    return {
        "captured_ns": captured_ns - start,
        "serial": int(duplicator.frame_serial),
        "source_id": str(duplicator.instance_id),
        "generation": int(camera._generation),
        "accumulated_frames": int(duplicator.accumulated_frames),
        "last_present_time_qpc": getattr(duplicator, "last_present_time", None),
        "dirty_rects": None if rects is None else [list(r) for r in rects],
        "move_rects": None if moves is None else [list(r) for r in moves],
    }


def keep_trace(path, info, frames):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    data = {f"frame_{i:04d}": frame for i, frame in enumerate(frames)}
    data["contract"] = np.frombuffer(json.dumps(info, allow_nan=False).encode("utf-8"), dtype=np.uint8)
    try:
        with tmp.open("xb") as output:
            np.savez_compressed(output, **data)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def record(args):
    if not (0 < args.seconds <= 3 and 0 < args.hz <= 100 and args.wait >= 0):
        raise ValueError("Choose a capture duration of at most 3 seconds and a rate of 1–100 Hz.")
    guard = WindowGuard(args.title_contains, args.pid, tuple(args.bounds),
                        tuple(args.region) if args.region else None, args.hwnd)
    print("Waiting for the controlled fixture to cover the selected monitor region.", flush=True)
    wait_deadline = time.perf_counter() + args.wait
    while not guard.check(wait=True):
        if time.perf_counter() >= wait_deadline:
            raise RuntimeError(f"Could not verify the controlled fixture: {guard.reason}")
        time.sleep(0.08)
    import rapidshot

    target = guard.region
    local = (target[0] - args.bounds[0], target[1] - args.bounds[1],
             target[2] - args.bounds[0], target[3] - args.bounds[1])
    width, height = local[2] - local[0], local[3] - local[1]
    monitor_size = (args.bounds[2] - args.bounds[0], args.bounds[3] - args.bounds[1])
    camera = rapidshot.create(device_idx=args.device, output_idx=args.output_index, backend="dxgi",
                              output_color="BGRA", timeout_ms=1, nvidia_gpu=False,
                              pool_output=True, region=local)
    frames, entries = [], []
    interval = round(1e9 / args.hz)
    counts = Counter()
    memory_cap = args.max_mib * 1024**2
    truncated = False
    try:
        if tuple(camera.region) != local or camera.rotation_angle:
            raise RuntimeError(f"Unexpected BGRA camera crop {camera.region}, rotation {camera.rotation_angle}")
        guard.check()
        print("Recording the verified fixture.", flush=True)
        start = time.perf_counter_ns()
        deadline = start + round(args.seconds * 1e9)
        for due in range(start, deadline, interval):
            remaining = due - time.perf_counter_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)
            elif remaining < -interval:
                counts["recorder_ticks_missed"] += 1
                continue
            guard.check()
            owner = camera.grab()
            captured = time.perf_counter_ns()
            counts["capture_polls"] += 1
            if owner is None:
                counts["no_frame_polls"] += 1
                continue
            try:
                guard.check()
                raw = getattr(owner, "array", owner)
                if raw.shape != (height, width, 4) or raw.dtype != np.uint8:
                    raise RuntimeError(f"Unexpected recorder pixels {raw.shape} {raw.dtype}")
                if args.app == "browser" and not browser_marker(raw):
                    raise RuntimeError("The controlled browser fixture marker is missing. No recording was written.")
                if (len(frames) + 1) * raw.nbytes > memory_cap:
                    truncated = True
                    break
                entries.append(metadata(camera, captured, start))
                frames.append(np.array(raw, order="C", copy=True))
            finally:
                release = getattr(owner, "release", None)
                if release:
                    release()
        guard.check()
    finally:
        camera.release()
    if len(frames) < 2:
        raise RuntimeError(f"Only {len(frames)} images arrived. Keep the fixture moving during the recording.")
    if not any(not np.array_equal(frames[0], frame) for frame in frames[1:]):
        raise RuntimeError("The fixture produced no changed pixels. Start its motion during recording.")
    info = {
        "format": FORMAT, "app": args.app, "monitor_bounds": list(args.bounds),
        "monitor_size": monitor_size, "capture_region": local,
        "ownership_guard": "fixture title, foreground HWND, client bounds and sampled window ownership",
        "browser_fixture_marker": args.app == "browser", "output_index": args.output_index,
        "device": args.device, "size": [width, height], "record_target_hz": args.hz,
        "requested_seconds": args.seconds, "memory_cap_mib": args.max_mib,
        "memory_cap_reached": truncated, "poll_counts": dict(counts), "frames": entries,
    }
    keep_trace(args.out, info, frames)
    print(f"Recorded {len(frames)} frames of {width}×{height} to {args.out}", flush=True)


def load_trace(path):
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(archive["contract"].tobytes())
        if info.get("format") != FORMAT:
            raise ValueError("Unknown application trace format")
        width, height = info["size"]
        entries = info["frames"]
        if len(entries) < 2 or width <= 0 or height <= 0:
            raise ValueError("Incomplete application trace")
        times = [entry["captured_ns"] for entry in entries]
        if any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Trace timestamps are not strictly increasing")
        frames = [archive[f"frame_{i:04d}"] for i in range(len(entries))]
    if any(frame.shape != (height, width, 4) or frame.dtype != np.uint8 or not frame.flags.c_contiguous
           for frame in frames):
        raise ValueError("Unexpected trace frame layout")
    return info, frames


def oracle_tiles(before, after):
    changed = np.any(before != after, axis=2)
    rows = np.logical_or.reduceat(changed, np.arange(0, changed.shape[0], TILE), axis=0)
    tiles = np.logical_or.reduceat(rows, np.arange(0, changed.shape[1], TILE), axis=1)
    return np.flatnonzero(tiles).astype(np.uint32)


def candidates_for(grid, previous, current, clip, monitor_size):
    prev_id = (previous["source_id"], previous["generation"])
    cur_id = (current["source_id"], current["generation"])
    if prev_id != cur_id or current["serial"] != previous["serial"] + 1:
        return grid.all, "source or frame gap"
    if current["accumulated_frames"] != 1:
        return grid.all, "accumulated frames"
    if not current["dirty_rects"] or current["move_rects"] is None:
        return grid.all, "missing or empty metadata"
    if current["move_rects"]:
        return grid.all, "move rectangles"
    wl, wt, wr, wb = clip
    width, height = monitor_size
    clipped = []
    for l, t, r, b in current["dirty_rects"]:
        if not (0 <= l < r <= width and 0 <= t < b <= height):
            return grid.all, "invalid rectangle"
        l, t, r, b = max(l, wl), max(t, wt), min(r, wr), min(b, wb)
        if l < r and t < b:
            clipped.append((l - wl, t - wt, r - wl, b - wt))
    return grid.masks(clipped)


def template_anchors(first, last):
    """Choose several non-flat regions in different parts of the first frame."""
    import cv2

    gray = cv2.cvtColor(first, cv2.COLOR_BGRA2GRAY)
    end = cv2.cvtColor(last, cv2.COLOR_BGRA2GRAY)
    height, width = gray.shape
    size = min(26, height // 3, width // 3)
    if size < 8:
        raise ValueError("Application crop is too small for template replay")
    options = []
    for y in range(8, height - size, size * 2):
        for x in range(8, width - size, size * 2):
            patch = gray[y : y + size, x : x + size]
            variance = float(patch.std())
            if variance > 12:
                moving = float(np.count_nonzero(patch != end[y : y + size, x : x + size])) / patch.size
                options.append((variance, moving, x, y))
    chosen = []
    for expression in (lambda a: a[0] * a[1], lambda a: a[0] * (1 - a[1]), lambda a: a[0]):
        for _, _, x, y in sorted(options, key=expression, reverse=True):
            if all(abs(cx - x) + abs(cy - y) >= size * 2 for cx, cy in chosen):
                chosen.append((x, y))
                break
    if not chosen:
        raise RuntimeError("Could not find a distinctive application template in the initial frame")
    templates = {f"anchor_{i + 1}": gray[y : y + size, x : x + size].copy()
                 for i, (x, y) in enumerate(chosen)}
    return templates, [dict(x=x, y=y, width=size, height=size) for x, y in chosen]


def rate_indices(timestamps, hz):
    count = round(1e9 / hz)
    ticks = np.arange(timestamps[0], timestamps[-1] + 1, count, dtype=np.int64)
    return np.searchsorted(timestamps, ticks, side="right") - 1


def percentile(values, p):
    return float(np.percentile(values, p)) if values else None


def replay_trace(info, frames, rates, seed):
    width, height = info["size"]
    grid = Grid(width, height)
    output = np.empty_like(grid.all)
    native = Native()
    templates, anchors = template_anchors(frames[0], frames[-1])
    source = info["frames"]
    timestamps = [entry["captured_ns"] for entry in source]
    timings = {}
    schedule = list(rates)
    random.Random(seed).shuffle(schedule)
    # Each full result is independent of cadence. Cache that oracle and its
    # measured stage cost against the same captured pixels across the replays.
    full = TemplateSearch(templates, width, height, TILE)
    full.find_all(frames[0])
    reference = {}

    def full_reference(index):
        if index not in reference:
            started = time.perf_counter_ns()
            result = full.find_all(frames[index])
            reference[index] = (result, (time.perf_counter_ns() - started) / 1e6)
        return reference[index]

    for hz in schedule:
        print(f"Replaying {info['app']} at {hz} Hz", flush=True)
        indices = rate_indices(timestamps, hz).tolist()
        if len(indices) < 2:
            raise RuntimeError(f"The clip is shorter than the replay interval at {hz} Hz")
        last_seen = last_delivered = indices[0]
        local = TemplateSearch(templates, width, height, TILE)
        first = local.find_all(frames[last_seen])
        if first.detections != full_reference(last_seen)[0].detections:
            raise RuntimeError("Initial trace template detections differ")
        rows = []
        repeated_ticks = 0
        for index in indices[1:]:
            if index == last_seen:
                repeated_ticks += 1
                continue
            old, current = frames[last_delivered], frames[index]
            started = time.perf_counter_ns()
            candidates, fallback = candidates_for(grid, source[last_seen], source[index],
                                                  info["capture_region"], info["monitor_size"])
            length = native.cpu_compare(old, current, candidates, output)
            changed = output[:length]
            detector_ms = (time.perf_counter_ns() - started) / 1e6
            started = time.perf_counter_ns()
            result = local.update(current, changed)
            search_ms = (time.perf_counter_ns() - started) / 1e6
            baseline, full_ms = full_reference(index)
            if not np.array_equal(oracle_tiles(old, current), changed):
                raise RuntimeError(f"{info['app']} at {hz} Hz frame {index}: exact comparison missed CPU truth")
            if result.detections != baseline.detections:
                raise RuntimeError(f"{info['app']} at {hz} Hz frame {index}: regional matches differ from full search")
            rows.append({"index": index, "byte_changed_tiles": int(length), "candidates": int(candidates.size),
                         "fallback": fallback, "detector_ms": detector_ms, "tile_search_ms": search_ms,
                         "full_search_ms": full_ms, "tile_origins": int(result.searched_origins),
                         "full_origins": int(baseline.searched_origins), "detections": result.detections})
            last_seen = index
            if length:
                last_delivered = index
        full_time = sum(row["full_search_ms"] for row in rows)
        elapsed = sum(row["detector_ms"] + row["tile_search_ms"] for row in rows)
        total_tiles = int(grid.all.size)
        timings[hz] = {"ticks": len(indices) - 1, "repeated_previous_snapshot": repeated_ticks,
                       "observed_updates": len(rows), "total_tiles": total_tiles,
                       "changed_fraction_p50": percentile([row["byte_changed_tiles"] / total_tiles for row in rows], 50),
                       "changed_fraction_p95": percentile([row["byte_changed_tiles"] / total_tiles for row in rows], 95),
                       "full_search_ms": full_time, "tile_search_plus_detector_ms": elapsed,
                       "time_saved_pct": 100 * (1 - elapsed / full_time) if full_time else None,
                       "skipped_searches": sum(row["tile_origins"] == 0 for row in rows),
                       "fallbacks": dict(Counter(row["fallback"] for row in rows if row["fallback"])),
                       "rows": rows}
    return {"app": info["app"], "size": info["size"], "monitor_bounds": info["monitor_bounds"],
            "capture_region": info["capture_region"], "record_target_hz": info["record_target_hz"],
            "recorded_frames": len(frames), "captured_span_seconds": (timestamps[-1] - timestamps[0]) / 1e9,
            "poll_counts": info["poll_counts"], "memory_cap_reached": info["memory_cap_reached"],
            "anchors": anchors, "seed": seed, "rate_order": schedule, "rates": timings}


def replay(args):
    if any(rate <= 0 or rate > 100 for rate in args.rates) or len(set(args.rates)) != len(args.rates):
        raise ValueError("Replay rates must be unique and between 1 and 100 Hz")
    info, frames = load_trace(args.trace)
    result = replay_trace(info, frames, args.rates, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    recovered = args.out / f"{result['app']}-replay.json"
    recovered.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    report = args.out / f"{result['app']}-replay.md"
    width, height = result["size"]
    lines = [f"# {result['app'].capitalize()} trace replay", "",
             f"Recorded {result['recorded_frames']} BGRA captures of a {width} × {height} application region across {result['captured_span_seconds']:.2f} seconds.", "",
             "Replay samples the original recorded timeline. Each scheduled tick uses the latest source image at that time. Repeated snapshots reuse the previous result. The full OpenCV search was measured once for each source image and reused as the reference across rates.", "",
             "| Requested replay rate | Distinct samples | Reused previous snapshot | Changed tile fraction, median / p95 | OpenCV searches avoided | Search and detector time saved |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for hz, row in sorted(result["rates"].items()):
        def fmt(value, scale=1):
            return f"{value * scale:.1f}%" if value is not None else "n/a"
        lines.append(f"| {hz} Hz | {row['observed_updates']} | {row['repeated_previous_snapshot']} | "
                     f"{fmt(row['changed_fraction_p50'], 100)} / {fmt(row['changed_fraction_p95'], 100)} | "
                     f"{row['skipped_searches']} | {fmt(row['time_saved_pct'])} |")
    lines += ["", "Every regional result matched the full OpenCV output. Every changed-tile list matched an independent CPU pixel scan.", ""]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {recovered} and {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("record")
    capture.add_argument("--app", choices=APP_NAMES, required=True)
    capture.add_argument("--title-contains", required=True)
    capture.add_argument("--pid", type=int)
    capture.add_argument("--hwnd", type=int, help="Verified top-level fixture handle, with its process ID")
    capture.add_argument("--device", type=int, default=0)
    capture.add_argument("--output-index", type=int, required=True)
    capture.add_argument("--bounds", type=int, nargs=4, metavar=("L", "T", "R", "B"), required=True)
    capture.add_argument("--region", type=int, nargs=4, metavar=("L", "T", "R", "B"))
    capture.add_argument("--seconds", type=float, default=2.4)
    capture.add_argument("--hz", type=int, default=100)
    capture.add_argument("--wait", type=float, default=15)
    capture.add_argument("--max-mib", type=int, default=1024)
    capture.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("replay")
    run.add_argument("--trace", type=Path, required=True)
    run.add_argument("--rates", type=int, nargs="+", default=(10, 20, 30, 60, 100))
    run.add_argument("--seed", type=int, default=32_400)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "record":
        if not args.title_contains.strip() or args.max_mib <= 0 or (args.hwnd and not args.pid):
            parser.error("recording requires a fixture title, a positive memory cap and the PID of any explicit window")
        record(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
