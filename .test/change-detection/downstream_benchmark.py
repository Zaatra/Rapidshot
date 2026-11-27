"""Benchmark captured dashboard pixels through a real OpenCV icon search."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import ctypes as ct
from importlib.metadata import version
import json
import math
from pathlib import Path
import queue
import random
import statistics
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from cpu_default import Grid, Native
from downstream_scene import (ScenePlayer, capture_cover_state, expected_detections, read_cover_stamp,
                              read_stamp, render, scenes, templates)
from downstream_task import TemplateSearch
from monitor_target import select_monitor


ROOT = Path(__file__).resolve().parent
COMMIT = "4b20e5c2230322dc68ffcec3934bd4a33286075d"
WIDTH, HEIGHT, TILE = 1920, 1080, 128
ARMS = ("rapidshot_tiles", "dxcam_tiles", "rapidshot_full", "dxcam", "mss")
LABELS = {"rapidshot_tiles": "RapidShot dirty candidates + tile search",
          "dxcam_tiles": "DXcam full byte scan + tile search", "rapidshot_full": "RapidShot + full search",
          "dxcam": "DXcam + full search", "mss": "MSS + full search"}
TILE_ARMS = ("rapidshot_tiles", "dxcam_tiles", "paired")
CAPTURE_ARMS = ("rapidshot_full", "dxcam", "mss")
DEFAULT_RATES = tuple(range(10, 101, 10))
CONTROL_RATES = (10, 30, 100)
WIRE = "DOWNSTREAM "
MS = 1_000_000


def ns():
    return time.perf_counter_ns()


def delta_ms(start):
    return (ns() - start) / MS


def qpc_calibration():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    freq, ticks = ct.c_int64(), ct.c_int64()
    before = ns()
    if not kernel.QueryPerformanceCounter(ct.byref(ticks)):
        raise ct.WinError(ct.get_last_error())
    after = ns()
    if not kernel.QueryPerformanceFrequency(ct.byref(freq)):
        raise ct.WinError(ct.get_last_error())
    return {"frequency_hz": freq.value, "anchor_ticks": ticks.value,
            "anchor_perf_counter_ns": (before + after) // 2, "anchor_interval_ns": after - before}


def oracle_detections(step, cover_state):
    return tuple(sorted((name, int(x), int(y))
                        for name, (x, y) in expected_detections(step, cover_state).items()))


def wait_until(deadline):
    remaining = deadline - ns()
    if remaining > 0:
        time.sleep(remaining / 1e9)


def dpi_aware():
    user = ct.WinDLL("user32", use_last_error=True)
    user.SetThreadDpiAwarenessContext.argtypes = [ct.c_void_p]
    user.SetThreadDpiAwarenessContext.restype = ct.c_void_p
    if not user.SetThreadDpiAwarenessContext(ct.c_void_p(-4)):
        raise ct.WinError(ct.get_last_error())


class ThreadCycles:
    def __init__(self):
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentThread.restype = ct.c_void_p
        kernel.QueryThreadCycleTime.argtypes = [ct.c_void_p, ct.POINTER(ct.c_uint64)]
        kernel.QueryThreadCycleTime.restype = ct.c_int
        self.handle = kernel.GetCurrentThread()
        self.query = kernel.QueryThreadCycleTime
        self.value = ct.c_uint64()

    def read(self):
        if not self.query(self.handle, ct.byref(self.value)):
            raise ct.WinError(ct.get_last_error())
        return self.value.value


def source_metadata(camera):
    duplicator = camera._duplicator
    source = duplicator.instance_id
    return SimpleNamespace(
        source_id=(camera._generation, source), serial=int(duplicator.frame_serial),
        dirty_rects=duplicator.dirty_rects, move_rects=duplicator.move_rects,
        accumulated_frames=int(duplicator.accumulated_frames),
    )


def native_present(camera, arm):
    duplicator = getattr(getattr(camera, "camera", None), "_duplicator", None)
    accumulated = getattr(duplicator, "accumulated_frames", None)
    ticks = (getattr(duplicator, "latest_frame_ticks", None) if arm in ("dxcam", "dxcam_tiles")
             else getattr(duplicator, "last_present_time", None))
    return (int(ticks) if ticks and accumulated else None,
            int(accumulated) if accumulated is not None else None)


@dataclass
class Capture:
    handle: object
    pixels: np.ndarray
    metadata: object = None

    def close(self):
        if self.handle is not None:
            release = getattr(self.handle, "release", None)
            if release is not None:
                release()
            self.handle = None


class Camera:
    def __init__(self, arm, target):
        self.arm = arm
        left, top, right, bottom = target["bounds"]
        if arm in ("dxcam", "dxcam_tiles"):
            import dxcam
            self.camera = dxcam.create(device_idx=target["device_index"],
                                       output_idx=target["output_index"],
                                       backend="dxgi", output_color="BGRA")
            if self.camera is None:
                raise RuntimeError("DXcam could not open the benchmark monitor")
        elif arm == "mss":
            import mss
            self.camera = mss.MSS(backend="gdi", with_cursor=False)
            self.region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
            if not any(all(monitor[key] == value for key, value in self.region.items())
                       for monitor in self.camera.monitors[1:]):
                self.camera.close()
                raise RuntimeError(f"MSS could not resolve benchmark monitor {self.region}")
        else:
            import rapidshot
            self.camera = rapidshot.create(device_idx=target["device_index"],
                                           output_idx=target["output_index"],
                                           backend="dxgi", output_color="BGRA", timeout_ms=5,
                                           nvidia_gpu=False, pool_output=True)
            if self.camera.region != (0, 0, WIDTH, HEIGHT) or self.camera.rotation_angle:
                self.camera.release()
                raise RuntimeError("RapidShot did not open the expected unrotated 1080p monitor")

    def read(self):
        if self.arm == "mss":
            shot = self.camera.grab(self.region)
            return Capture(shot, np.asarray(shot))
        value = (self.camera.grab(copy=True, new_frame_only=True)
                 if self.arm in ("dxcam", "dxcam_tiles") else self.camera.grab())
        if value is None:
            return None
        metadata = source_metadata(self.camera) if self.arm in ("rapidshot_tiles", "paired") else None
        return Capture(value, getattr(value, "array", value), metadata)

    def close(self):
        if self.arm == "mss":
            self.camera.close()
        else:
            self.camera.release()


class SearchPipeline:
    def __init__(self, arm, target=None):
        self.arm = arm
        target = target or {}
        self.scenes = scenes(WIDTH, HEIGHT, target.get("visible_steps", 12), target.get("passive_steps", 12))
        self.templates = templates()
        self.cycles = ThreadCycles()
        self.grid = Grid(WIDTH, HEIGHT)
        self.output = np.empty_like(self.grid.all)
        self.oracle_output = np.empty_like(self.grid.all) if arm == "paired" else None
        self.native = Native() if arm in TILE_ARMS else None
        self.mismatches = []
        self.reset()

    def reset(self, capture_only=False):
        previous = getattr(self, "previous", None)
        if previous is not None:
            previous.close()
        self.previous = None
        self.last_seen = None
        self.last_detections = ()
        self.last_cover_state = None
        self.last_stamp = None
        self.last_content_marker = None
        self.gap = False
        self.capture_only = capture_only
        self.search = None if capture_only else TemplateSearch(self.templates, WIDTH, HEIGHT, TILE)
        self.reference = TemplateSearch(self.templates, WIDTH, HEIGHT, TILE) if self.arm == "paired" else None

    def close(self):
        if self.previous is not None:
            self.previous.close()
            self.previous = None

    def _exact_candidates(self, frame):
        if self.arm == "dxcam_tiles":
            candidates, fallback = self.grid.all, None
        else:
            metadata = frame.metadata
            last_seen = None if self.gap else self.last_seen
            candidates, fallback = self.grid.candidate_mask(last_seen, metadata)
            if self.gap and self.previous is not None:
                fallback = "previous update missed"
            self.last_seen = (metadata.source_id, metadata.serial)
        if self.previous is None:
            return candidates, fallback, True, int(candidates.size)
        count = self.native.cpu_compare(self.previous.pixels, frame.pixels, candidates, self.output)
        selected = self.output[:count]
        return selected, fallback, bool(selected.size), int(candidates.size)

    def _result(self, frame, step, index):
        result = {"detector_ms": 0.0, "search_ms": 0.0, "search_cycles": 0,
                  "searched_origins": 0, "search_calls": 0, "fallback": None}
        frame_view = frame.pixels
        cover_state = capture_cover_state(step, frame_view)
        marker = (step.stamp, cover_state, read_cover_stamp(step, frame_view))
        result.update(cover_state=cover_state, repeated_scene_snapshot=marker == self.last_content_marker)
        if self.capture_only:
            self.last_stamp = step.stamp
            self.last_content_marker = marker
            self.last_cover_state = cover_state
            self.gap = False
            return {**result, "correct": True, "full_search": False}
        if self.native is not None:
            started = ns()
            changed_tiles, fallback, changed, candidate_count = self._exact_candidates(frame)
            result.update(detector_ms=delta_ms(started), changed_tiles=int(changed_tiles.size),
                          comparison_candidate_tiles=candidate_count, fallback=fallback)
            if self.oracle_output is not None and self.previous is not None:
                started = ns()
                count = self.native.cpu_compare(self.previous.pixels, frame_view, self.grid.all, self.oracle_output)
                result["paired_full_byte_scan_ms"] = delta_ms(started)
                if not np.array_equal(changed_tiles, self.oracle_output[:count]):
                    raise RuntimeError("metadata-guided comparison missed changed tiles on the paired frame")
        else:
            changed_tiles = None
            fallback = None
            changed = True
        if self.reference is not None and index % 2 == 0:
            self._check_full(frame_view, result)
        cycles = self.cycles.read()
        started = ns()
        if self.native is None:
            search = self.search.find_all(frame_view)
        else:
            # A gap broadens the byte comparison to the full screen. Its exact
            # result still proves which prior search positions remain valid.
            search = self.search.update(frame_view, changed_tiles, force_full=self.previous is None)
        result.update(search_ms=delta_ms(started), search_cycles=self.cycles.read() - cycles,
                      searched_origins=int(search.searched_origins), search_calls=int(search.calls),
                      full_search=bool(search.full))
        if self.reference is not None:
            if index % 2 != 0:
                self._check_full(frame_view, result)
            reference = result.pop("paired_reference_detections")
            if search.detections != reference:
                raise RuntimeError(f"paired result differs from full frame in {step.workload} at {step.stamp}: "
                                   f"{search.detections} versus {reference}")
            if self.previous is not None and changed == np.array_equal(self.previous.pixels, frame_view):
                raise RuntimeError("exact screen decision differs from whole-frame byte comparison")
        expected = oracle_detections(step, cover_state)
        result.update(detections=search.detections, expected=expected, correct=search.detections == expected)
        self.last_detections = search.detections
        self.last_cover_state = cover_state
        self.last_stamp = step.stamp
        self.last_content_marker = marker
        self.gap = False
        if self.native is not None and changed:
            if self.previous is not None:
                self.previous.close()
            self.previous = frame
        return result

    def _check_full(self, pixels, result):
        cycles = self.cycles.read()
        started = ns()
        full = self.reference.find_all(pixels)
        result.update(paired_full_ms=delta_ms(started), paired_full_cycles=self.cycles.read() - cycles,
                      paired_full_origins=int(full.searched_origins), paired_reference_detections=full.detections)

    def save_mismatches(self):
        if not self.mismatches:
            return
        reference = TemplateSearch(self.templates, WIDTH, HEIGHT, TILE)
        for path, pixels, evidence in self.mismatches:
            steps = self.scenes[evidence["workload"]]
            index = evidence["index"]
            intended_step, prior_step = steps[index], steps[max(index - 1, 0)]
            include_cover = evidence["cover_state"] == "visible"
            intended = render(intended_step, WIDTH, HEIGHT, include_cover=include_cover)
            prior = render(prior_step, WIDTH, HEIGHT, include_cover=include_cover)
            # The GDI raster's alpha is not a meaningful fixture reference.
            current_match = np.all(pixels[:, :, :3] == intended[:, :, :3], axis=2)
            prior_match = np.all(pixels[:, :, :3] == prior[:, :, :3], axis=2)
            changed = np.any(intended[:, :, :3] != prior[:, :, :3], axis=2)
            full = reference.find_all(pixels).detections
            evidence.update(
                full_frame_detections=full,
                intended_fixture_detections=reference.find_all(intended).detections,
                prior_fixture_detections=reference.find_all(prior).detections,
                prior_fixture_stamp=prior_step.stamp,
                bgr_pixels_different_from_intended=int(np.count_nonzero(~current_match)),
                bgr_pixels_different_from_prior=int(np.count_nonzero(~prior_match)),
                changed_bgr_pixels_from_prior=int(np.count_nonzero(changed)),
                changed_bgr_pixels_matching_intended=int(np.count_nonzero(changed & current_match)),
                changed_bgr_pixels_matching_prior=int(np.count_nonzero(changed & prior_match)),
                changed_bgr_pixels_matching_neither=int(np.count_nonzero(changed & ~current_match & ~prior_match)),
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.with_suffix(".npy.tmp").open("wb") as handle:
                np.save(handle, pixels, allow_pickle=False)
            path.with_suffix(".npy.tmp").replace(path)
            write_json(path.with_suffix(".json"), {**evidence, "pixels": path.name})
        self.mismatches.clear()

    def sample(self, camera, request):
        index = request["index"]
        step = self.scenes[request["workload"]][index]
        deadline = request["deadline"]
        presented = request["presented"]
        seed = request["seed"]
        elapsed_capture = 0.0
        empty = stale = 0
        call_ms = []
        grab_trace = [] if request.get("trace_grabs") else None
        started = ns()
        mismatch_complete = None
        captured = None
        result = {}
        previous_capture_stamp = self.last_stamp
        pending_new = previous_capture_stamp != step.stamp
        while ns() < deadline:
            capture_started = ns()
            captured = camera.read()
            capture_returned = ns()
            duration = (capture_returned - capture_started) / MS
            call_ms.append(duration)
            elapsed_capture += duration
            if captured is None:
                if grab_trace is not None:
                    grab_trace.append(dict(start_ns=capture_started, return_ns=capture_returned, outcome="empty",
                                           observed_stamp=None, dxgi_last_present_ticks=None,
                                           dxgi_accumulated_frames=None))
                empty += 1
                if not pending_new and not seed:
                    break
                time.sleep(.0005)
                continue
            if captured.pixels.shape != (HEIGHT, WIDTH, 4) or captured.pixels.dtype != np.uint8:
                captured.close()
                raise RuntimeError(f"unexpected pixels {captured.pixels.shape} {captured.pixels.dtype}")
            observed = read_stamp(captured.pixels)
            if grab_trace is not None:
                ticks, accumulated = native_present(camera, self.arm)
                outcome = "accepted" if observed == step.stamp else "rejected"
                grab_trace.append(dict(start_ns=capture_started, return_ns=capture_returned, outcome=outcome,
                                       observed_stamp=observed, dxgi_last_present_ticks=ticks,
                                       dxgi_accumulated_frames=accumulated))
            if observed != step.stamp:
                stale += 1
                captured.close()
                captured = None
                if not pending_new and not seed:
                    break
                time.sleep(.0005)
                continue
            try:
                result = self._result(captured, step, index)
                if result.get("correct") is False and request.get("evidence_dir") is not None:
                    mismatch_complete = ns()
                    suffix = f"-seed-{request['seed_attempt']}" if seed else ""
                    filename = f"{index:03d}-stamp-{observed:04x}{suffix}.npy"
                    path = Path(request["evidence_dir"]) / filename
                    ticks, accumulated = native_present(camera, self.arm)
                    evidence = dict(arm=self.arm, workload=step.workload, index=index, observed_stamp=observed,
                                    expected_stamp=step.stamp, cover_state=result["cover_state"],
                                    previous_capture_stamp=previous_capture_stamp, seed=seed,
                                    dxgi_last_present_ticks=ticks, dxgi_accumulated_frames=accumulated,
                                    observed_detections=result["detections"], expected_detections=result["expected"],
                                    submission_ns=presented, result_ns=mismatch_complete)
                    self.mismatches.append((path, captured.pixels.copy(order="C"), evidence))
                    result["mismatch_evidence"] = str(path.with_suffix(".json"))
            finally:
                if captured is not self.previous:
                    captured.close()
            break

        if result:
            status = "frame_processed"
        elif not call_ms:
            status = "expired_before_capture"
            self.gap |= pending_new or seed
        elif pending_new or seed:
            status = "missed_update"
            self.gap = True
        elif self.last_stamp is not None:
            status = "reused_no_frame" if empty else "reused_after_stale"
            if self.capture_only:
                result.update(cover_state=self.last_cover_state, correct=True)
            else:
                expected = oracle_detections(step, self.last_cover_state)
                result.update(detections=self.last_detections, expected=expected,
                              cover_state=self.last_cover_state, correct=self.last_detections == expected)
        else:
            status = "missed_update"
        completed = mismatch_complete if mismatch_complete is not None else ns()
        result.update(status=status, workload=step.workload, index=index, stamp=step.stamp,
                      seed=seed, empty_polls=empty, stale_polls=stale, capture_ms=elapsed_capture,
                      grab_calls_ms=call_ms, grab_attempts=len(call_ms),
                      successful_grab_ms=call_ms[-1] if status == "frame_processed" else None,
                      queue_ms=(started - presented) / MS,
                      work_ms=(completed - started) / MS, from_present_ms=(completed - presented) / MS,
                      late=completed > deadline, visible_update=bool(step.visible_update))
        if grab_trace is not None:
            result.update(grab_trace=grab_trace, deadline_ns=deadline)
        return result


def send_reply(value):
    print(WIRE + json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def worker_main(arm, target):
    dpi_aware()
    camera = Camera(arm, target)
    pipe = SearchPipeline(arm, target)
    try:
        send_reply({"status": "ready"})
        for line in sys.stdin:
            request = json.loads(line)
            command = request["command"]
            if command == "close":
                break
            if command == "reset":
                pipe.reset(request.get("capture_only", False))
                send_reply({"status": "reset"})
            elif command == "sample":
                send_reply(pipe.sample(camera, request))
            elif command == "save_mismatches":
                pipe.save_mismatches()
                send_reply({"status": "evidence_saved"})
            else:
                raise ValueError(f"unexpected worker command {command}")
    finally:
        try:
            try:
                pipe.save_mismatches()
            finally:
                pipe.close()
        finally:
            camera.close()


class Worker:
    def __init__(self, arm, target):
        args = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", arm,
                "--target", json.dumps(target, separators=(",", ":"))]
        self.arm = arm
        self.input = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                      creationflags=subprocess.CREATE_NO_WINDOW)
        self.events = queue.Queue()
        self.diagnostics = []
        threading.Thread(target=self._lines, daemon=True).start()
        threading.Thread(target=self._errors, daemon=True).start()

    def _lines(self):
        try:
            for line in self.input.stdout:
                if line.startswith(WIRE):
                    self.events.put(json.loads(line[len(WIRE):]))
                elif line.strip():
                    self.diagnostics.append(line.strip())
        finally:
            self.events.put(None)

    def _errors(self):
        for line in self.input.stderr:
            self.diagnostics.append(line.rstrip())

    def send(self, **value):
        try:
            self.input.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self.input.stdin.flush()
        except (BrokenPipeError, OSError) as err:
            raise RuntimeError(f"{self.arm} worker closed: {' '.join(self.diagnostics[-14:])}") from err

    def receive(self, seconds=15):
        try:
            value = self.events.get(timeout=seconds)
        except queue.Empty:
            raise RuntimeError(f"{self.arm} worker did not reply") from None
        if value is None:
            self.input.wait(timeout=5)
            raise RuntimeError(f"{self.arm} worker exited: {' '.join(self.diagnostics[-14:])}")
        return value

    def __enter__(self):
        try:
            first = self.receive()
            if first["status"] != "ready":
                raise RuntimeError(f"unexpected worker response {first}")
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *_):
        if self.input.poll() is None:
            try:
                self.send(command="close")
                self.input.wait(timeout=5)
            except (RuntimeError, subprocess.TimeoutExpired):
                self.input.terminate()
                self.input.wait(timeout=5)
        for file in (self.input.stdin, self.input.stdout, self.input.stderr):
            file.close()


def prove_initial(worker, player, case, steps, capture_only=False, evidence_dir=None):
    for attempt in range(2):
        worker.send(command="reset", capture_only=capture_only)
        if worker.receive()["status"] != "reset":
            raise RuntimeError("worker could not reset")
        player.present(steps[0], restart=True)
        present_time = ns()
        evidence = {} if evidence_dir is None else {"evidence_dir": evidence_dir, "seed_attempt": attempt + 1}
        worker.send(command="sample", workload=case, index=0, presented=present_time,
                    deadline=present_time + 1_000 * MS, seed=True, **evidence)
        sample = worker.receive()
        if sample["status"] == "frame_processed" and sample["correct"]:
            return
        if attempt == 0:
            print(f"Repainting the untimed {case} seed for {worker.arm}", flush=True)
    raise RuntimeError(f"{worker.arm} failed to seed {case}: {sample}")


def measure_scene(worker, player, case, steps, hz, capture_only=False, evidence_dir=None, trace_grabs=False):
    prove_initial(worker, player, case, steps, capture_only, evidence_dir)
    evidence = {} if evidence_dir is None else {"evidence_dir": evidence_dir}
    if trace_grabs:
        evidence["trace_grabs"] = True
    period = round(1e9 / hz)
    next_tick = ns() + period
    epochs = []
    for index, step in enumerate(steps[1:], start=1):
        wait_until(next_tick)
        player.present(step, wait_for_desktop=worker.arm == "paired")
        presented = ns()
        # Bound each attempt from GDI submission; DWM and capture can coalesce.
        worker.send(command="sample", workload=case, index=index, presented=presented,
                    deadline=presented + period, seed=False, **evidence)
        epochs.append(presented)
        next_tick += period
        if worker.arm == "paired":
            result = worker.receive()
            if result["index"] != index:
                raise RuntimeError("paired audit responses arrived out of order")
            yield result
    if worker.arm != "paired":
        for i in range(1, len(steps)):
            result = worker.receive()
            if result["index"] != i:
                raise RuntimeError("capture results arrived out of order")
            result["presented_ns"] = epochs[i - 1]
            yield result
    if evidence_dir is not None:
        worker.send(command="save_mismatches")
        if worker.receive(seconds=60)["status"] != "evidence_saved":
            raise RuntimeError("worker could not save capture evidence")


def check_audit(audit, source):
    for case, rows in audit.items():
        captured = [row for row in rows if row["status"] == "frame_processed"]
        if any(not row.get("correct", True) for row in rows):
            raise RuntimeError(f"paired validation disagreed with script for {case}")
        if case in ("sparse", "dense"):
            if len(captured) != len(source[case]) - 1:
                raise RuntimeError(f"paired validation retained only {len(captured)} / {len(source[case]) - 1} {case} updates")
        if case == "excluded_cover" and not captured:
            print("DXGI supplied no new excluded-cover images in the paired verification", flush=True)


def stats(values):
    if not values:
        return None
    values = sorted(values)
    point = (len(values) - 1) * .95
    lo = math.floor(point)
    return {"p50": statistics.median(values),
            "p95": values[lo] + (values[math.ceil(point)] - values[lo]) * (point - lo)}


def summarize(runs):
    rows = [row for run in runs for row in run]
    captured = [row for row in rows if row["status"] == "frame_processed"]
    decisions = [row for row in rows if row.get("correct") is True]
    counts = Counter(row["status"] for row in rows)
    summary = dict(samples=len(rows), matched_captures=len(captured),
                   incorrect=sum(row.get("correct") is False for row in rows),
                   missed_updates=sum(row["visible_update"] and row["status"] != "frame_processed" for row in rows),
                   reused_no_frame=counts["reused_no_frame"],
                   expired_before_capture=counts["expired_before_capture"],
                   grab_attempts=sum(row["grab_attempts"] for row in rows),
                   late=sum(row["late"] for row in decisions),
                   empty_polls=sum(row["empty_polls"] for row in rows),
                   stale_polls=sum(row["stale_polls"] for row in rows),
                   repeated_scene_snapshots=sum(row["repeated_scene_snapshot"] for row in captured),
                   search_calls=sum(row["search_calls"] for row in captured),
                   searches_skipped=sum(not row["search_calls"] for row in captured),
                   searched_origins=sum(row["searched_origins"] for row in captured),
                   search_ms=sum(row["search_ms"] for row in captured),
                   search_cycles=sum(row["search_cycles"] for row in captured),
                   cover_states=dict(Counter(row.get("cover_state") for row in captured)),
                   fallbacks=dict(Counter(row["fallback"] for row in captured if row["fallback"])))
    summary["successful_grab_ms"] = stats([row["successful_grab_ms"] for row in captured])
    summary["all_grab_calls_ms"] = stats([value for row in rows for value in row["grab_calls_ms"]])
    summary["visible_opportunities"] = sum(row["visible_update"] for row in rows)
    summary["changed_tiles"] = stats([row["changed_tiles"] for row in captured if "changed_tiles" in row])
    summary["candidate_tiles"] = stats([row["comparison_candidate_tiles"] for row in captured
                                       if "comparison_candidate_tiles" in row])
    for field in ("capture_ms", "search_ms", "detector_ms", "work_ms", "from_present_ms"):
        loop = [[row[field] for row in run if row["status"] == "frame_processed"] for run in runs]
        summary[field + "_on_capture"] = stats([value for run in loop for value in run])
        medians = [statistics.median(run) for run in loop if run]
        summary[field + "_round_p50s"] = medians
    active = [row for row in decisions if row["visible_update"]]
    summary["visible_result_count"] = len(active)
    summary["visible_results_on_time"] = sum(not row["late"] for row in active)
    summary["visible_update_result_ms"] = stats([row["from_present_ms"] for row in active])
    summary["source_interval_ms"] = stats([(b["presented_ns"] - a["presented_ns"]) / MS
                                           for run in runs for a, b in zip(run, run[1:])])
    if any("grab_trace" in row for row in rows):
        coverage = Counter()
        for run in runs:
            offered = {row["stamp"] for row in run if row["visible_update"]}
            seen = {attempt["observed_stamp"] for row in run for attempt in row["grab_trace"]} & offered
            accepted = {row["stamp"] for row in run if row["status"] == "frame_processed"} & offered
            coverage.update(offered=len(offered), accepted=len(accepted), seen=len(seen),
                            seen_only_on_rejected_reads=len(seen - accepted), never_seen=len(offered - seen))
        summary["grab_trace_coverage"] = dict(coverage)
    return summary


def paired_results(audit):
    result = {}
    for case, samples in audit.items():
        rows = [row for row in samples if "paired_full_ms" in row]
        if not rows:
            continue
        full_time = sum(row["paired_full_ms"] for row in rows)
        tile_time = sum(row["search_ms"] for row in rows)
        full_work = sum(row["paired_full_origins"] for row in rows)
        tile_work = sum(row["searched_origins"] for row in rows)
        detector = sum(row["detector_ms"] for row in rows)
        result[case] = dict(frames=len(rows), full_search_ms=full_time, tile_search_ms=tile_time,
                            detector_ms=detector,
                            guided_detector_ms=stats([row["detector_ms"] for row in rows]),
                            full_byte_scan_ms=stats([row["paired_full_byte_scan_ms"] for row in rows]),
                            full_search_origins=full_work, tile_search_origins=tile_work,
                            full_search_cycles=sum(row["paired_full_cycles"] for row in rows),
                            tile_search_cycles=sum(row["search_cycles"] for row in rows),
                            search_time_saved_pct=100 * (1 - tile_time / full_time),
                            net_ms_saved_pct=100 * (1 - (tile_time + detector) / full_time),
                            origins_saved_pct=100 * (1 - tile_work / full_work))
    return result


def same_live_updates(trials, workloads, arms=ARMS):
    synced = {}
    for workload in workloads:
        rows = {arm: [] for arm in arms}
        for trial in trials:
            indexes = {arm: {row["index"]: row for row in trial["trials"][arm][workload]
                             if row.get("correct") and row["status"] == "frame_processed"
                             and row["visible_update"]} for arm in arms}
            common = sorted(set.intersection(*(set(indexes[arm]) for arm in arms)))
            for arm in arms:
                rows[arm].extend(indexes[arm][index] for index in common)
        synced[workload] = {
            "frames": len(rows[arms[0]]),
            "per_arm": {arm: {metric: stats([row[metric] for row in rows[arm]])
                             for metric in ("search_ms", "capture_ms", "from_present_ms")} for arm in arms},
        }
    return synced


def render_report(data):
    target = data["target"]
    versions = data["versions"]
    sparse = data["paired"]["sparse"]
    dense = data["paired"]["dense"]
    tile_sparse = data["summary"]["sparse"]["rapidshot_tiles"]
    tile_cover = data["summary"]["excluded_cover"]["rapidshot_tiles"]
    lines = ["# RapidShot downstream icon-search benchmark", "",
             f"Windows at {WIDTH} × {HEIGHT}. {target['adapter']} on {target['output']}. "
             f"RapidShot `{COMMIT[:7]}`, DXcam {versions['dxcam']}, MSS {versions['mss']} and OpenCV {versions['opencv-python']}.", "",
             "The task searches for dashboard icons anywhere in the capture, then returns their identities and positions. "
             "The scripted scenes provide expected results. A paired run also compares the tile result to a full search on the same received frame.", "",
             "## Findings", "",
             f"- With sparse changes, the tile search plus detector took {sparse['net_ms_saved_pct']:.1f}% less time. "
             f"The live tile pipeline returned {tile_sparse['visible_result_count']} of {tile_sparse['samples']} results, with "
             f"{tile_sparse['visible_results_on_time']} within one frame interval.",
             f"- Dense changes saved {dense['origins_saved_pct']:.1f}% of search positions. The tile pipeline took "
             f"{100 - dense['net_ms_saved_pct']:.1f}% of the full-search time on the same captured frames.",
             f"- For the excluded cover, the tile pipeline skipped OpenCV on {tile_cover['searches_skipped']} of "
             f"{tile_cover['matched_captures']} returned images.", "",
             "## Same pixels, two search methods", "",
             "| Workload | Verified frames | Full search total | Tile search total | Change detector total | Net time saved | Search positions saved |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for case, row in data["paired"].items():
        lines.append(f"| {case.replace('_', ' ')} | {row['frames']} | {row['full_search_ms']:.2f} ms | "
                     f"{row['tile_search_ms']:.2f} ms | {row['detector_ms']:.2f} ms | "
                     f"{row['net_ms_saved_pct']:.1f}% | {row['origins_saved_pct']:.1f}% |")
    lines += ["", "## Live capture and search", "",
              f"{data['rounds']} shuffled rounds at a {data['hz']} Hz target. Each capture library ran in its own worker. "
              "Timing starts after the renderer issues a scene update. Time inside grab and search p50 use marker-verified frames. "
              "Result p50/p95 uses verified responses to visible scene changes. Empty polls, stale grabs and missed updates are tracked in the raw data.", "",
              "| Workload | Pipeline | Frames | Visible results on time | Missed updates | Wrong results | Time inside grab p50 | Search p50 | Result p50 / p95 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(value):
        return f"{value:.2f} ms"
    for case, arms in data["summary"].items():
        for name in ARMS:
            row = arms[name]
            result = row["visible_update_result_ms"]
            capture = row["capture_ms_on_capture"]
            search = row["search_ms_on_capture"]
            lines.append(f"| {case.replace('_', ' ')} | {LABELS[name]} | {row['matched_captures']} | "
                         f"{row['visible_results_on_time']} | {row['missed_updates']} | {row['incorrect']} | "
                         f"{fmt(capture['p50']) if capture else '—'} | {fmt(search['p50']) if search else '—'} | "
                         f"{fmt(result['p50']) + ' / ' + fmt(result['p95']) if result else '—'} |")
    lines += ["", "The privacy cover is evaluated against each backend’s captured view. "
              "Cover visibility, search calls, CPU cycles, duplicate opportunities, and all five run orders are recorded in [the raw results](results.json).", "",
              "## Same live updates", "",
              "These rows use only updates processed by all four pipelines at the same step and in the same round. Values are median time to a correct result.", "",
              "| Workload | Common updates | RapidShot exact tiles | Plain RapidShot | DXcam | MSS |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for case, synced in data["same_live_updates"].items():
        if synced["frames"]:
            values = [fmt(synced["per_arm"][name]["from_present_ms"]["p50"]) for name in ARMS]
            lines.append("| " + " | ".join([case.replace('_', ' '), str(synced['frames']), *values]) + " |")
    lines += ["", "## Rerun", "", "Run from the RapidShot workspace in a normal Windows desktop session. "
              "The build command creates the scratch native comparator and the Python command replays the recorded order.", "",
              "```powershell", r".\.test\change-detection\build.cmd",
              rf".\.venv\Scripts\python.exe -B .\.test\change-detection\downstream_benchmark.py --rounds {data['rounds']} --hz {data['hz']} --seed {data['seed']}",
              "```", ""]
    return "\n".join(lines)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def ordered_trials(rates, rounds, seed, selected=ARMS):
    rng = random.Random(seed)
    result = []
    for index in range(1, rounds + 1):
        arms, cadence = list(ARMS), list(rates)
        rng.shuffle(arms)
        rng.shuffle(cadence)
        result.append({"round": index, "order": [arm for arm in arms if arm in selected],
                       "rate_order": cadence, "by_rate": {}, "controls": {}})
    return result


def summarize_rate(trials, rate, source, arms=ARMS):
    measured = [{"round": trial["round"], "order": trial["order"],
                 "trials": trial["by_rate"][str(rate)]} for trial in trials]
    summary = {case: {arm: summarize([trial["trials"][arm][case] for trial in measured])
                      for arm in arms} for case in source}
    controls = {}
    if str(rate) in trials[0]["controls"]:
        controls = {case: {arm: summarize([trial["controls"][str(rate)][arm][case] for trial in trials])
                          for arm in CAPTURE_ARMS if arm in arms} for case in ("sparse", "dense")}
    return {"summary": summary, "controls": controls, "same_live_updates": same_live_updates(measured, source, arms)}


def sweep_report(data):
    target = data["target"]
    capture_only = data.get("capture_only", False)
    labels = {arm: label.split(" + ")[0] if capture_only else label for arm, label in LABELS.items()}
    selected = ", ".join(labels[arm] for arm in data.get("arms", ARMS))
    lines = [
        "# RapidShot downstream benchmark on HP 324pe", "",
        f"{target['monitor']} on `{target['output']}` at 1920 × 1080 and {target['physical_refresh_hz']} Hz physical refresh.", "",
        f"Measured pipelines are {selected}.", "",
        "These are capture-only passes on the sparse fixture."
        if capture_only else "The live comparisons time application pipelines. A full-frame OpenCV search and a changed-tile search do different amounts of work. "
        "The two tile pipelines run the identical OpenCV search. RapidShot uses DXGI rectangles to narrow its byte comparison and DXcam compares all tiles.", "",
        "Time inside grab sums the calls made for an update. It can include compositor waiting and retries. A worker that starts late can find an image already queued. "
        "The capture-only controls run at a common offered rate with the downstream search removed.", "",
        "The rate labels are requested updates per second on a 100 Hz panel. QPC timestamps measure the source's paint submissions. "
        "Verified markers show how many distinct visible states each pipeline processed. A deadline that elapsed before a grab began is counted separately from an empty grab.", "",
    ]

    def milliseconds(value):
        return f"{value['p50']:.2f} ms" if value else "—"

    if data["paired"]:
        lines += ["## Same pixels", "",
                  "| Workload | Paired frames | Net search and detector time saved | Guided byte check p50 | Full byte check p50 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
    for case, row in data["paired"].items():
        lines.append(f"| {case.replace('_', ' ')} | {row['frames']} | {row['net_ms_saved_pct']:.1f}% | "
                     f"{milliseconds(row['guided_detector_ms'])} | {milliseconds(row['full_byte_scan_ms'])} |")
    for rate, record in data["by_rate"].items():
        lines += ["", f"## {rate} updates/s target", "",
                  "| Workload | Pipeline | Median submitted updates/s | Marker-verified frames | On-time visible results | Missed visible updates | Expired before grab | Wrong results | Time inside grabs p50 | Search p50 | Time to visible result p50 / p95 |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for case, group in record["summary"].items():
            for arm, row in group.items():
                timing = row["visible_update_result_ms"]
                interval = row["source_interval_ms"]
                achieved = f"{1000 / interval['p50']:.1f}" if interval and interval["p50"] else "—"
                result = f"{timing['p50']:.2f} / {timing['p95']:.2f} ms" if timing else "—"
                lines.append(f"| {case.replace('_', ' ')} | {labels[arm]} | {achieved} | {row['matched_captures']} | "
                             f"{row['visible_results_on_time']} | {row['missed_updates']} | "
                             f"{row['expired_before_capture']} | {row['incorrect']} | "
                             f"{milliseconds(row['capture_ms_on_capture'])} | "
                             f"{milliseconds(row['search_ms_on_capture'])} | {result} |")
        matched = record["same_live_updates"]
        common = ", ".join(f"{row['frames']} {case.replace('_', ' ')}" for case, row in matched.items()
                           if case in ("sparse", "dense"))
        lines += ["", f"Every pipeline processed {common} common visible updates in the same round and step. "
                  "Their timing distributions are in the raw results."]
        if data.get("trace_grabs"):
            lines += ["", "| Workload | Pipeline | Offered states | Seen in any returned frame | Accepted for its deadline request | Seen only on rejected reads | Never seen in a returned frame |",
                      "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
            for case, group in record["summary"].items():
                for arm, row in group.items():
                    coverage = row["grab_trace_coverage"]
                    lines.append(f"| {case.replace('_', ' ')} | {labels[arm]} | {coverage['offered']} | "
                                 f"{coverage['seen']} | {coverage['accepted']} | "
                                 f"{coverage['seen_only_on_rejected_reads']} | {coverage['never_seen']} |")
        if record["controls"]:
            lines += ["", "### Capture-only controls", "",
                      "| Workload | Camera | Verified updates | Missed updates | Successful grab p50 |",
                      "| --- | --- | ---: | ---: | ---: |"]
            for case, group in record["controls"].items():
                for arm, row in group.items():
                    lines.append(f"| {case} | {LABELS[arm].split(' + ')[0]} | {row['matched_captures']} | "
                                 f"{row['missed_updates']} | {milliseconds(row['successful_grab_ms'])} |")
    if data.get("save_mismatches"):
        lines += ["", "Each incorrect marker-verified frame was copied after its result timestamp was recorded. "
                  "Lossless BGRA evidence and independent whole-frame checks were saved after its workload ended. "
                  "Rows with a retained image link to its evidence JSON. A rare copy may affect the next capture."]
    if data.get("trace_grabs"):
        clocks = data["qpc_calibration"][-1]
        lines += ["", f"The QPC frequency was {clocks['frequency_hz']:,} Hz. The raw results record the calibration and "
                  "capture call start and return times in perf-counter nanoseconds. MSS has no DXGI present ticks. "
                  "Convert native ticks to that clock using the recorded anchor plus the tick delta divided by the frequency. "
                  "A state accepted for its request may still have returned after the deadline. "
                  "The rejected-read coverage includes frames with a valid stamp for a different request."]
    profile = ("The benchmark uses one device and one sparse capture fixture." if capture_only else
               "The benchmark uses one device, four controlled workloads, and one local template-matching task.")
    lines += ["", profile + " Read [per-attempt records, versions, and trial order](results.json).", ""]
    return "\n".join(lines)


def run(args):
    dpi_aware()
    target = select_monitor()
    if target["physical_refresh_hz"] < max(args.rates):
        raise RuntimeError(f"The HP 324pe is running at {target['physical_refresh_hz']} Hz. "
                           f"The requested sweep reaches {max(args.rates)} updates/s.")
    target.update(visible_steps=args.visible_steps, passive_steps=args.passive_steps)
    source = scenes(WIDTH, HEIGHT, args.visible_steps, args.passive_steps)
    if args.capture_only:
        source = {"sparse": source["sparse"]}
    versions = {key: version(key) for key in ("rapidshot", "dxcam", "mss", "opencv-python", "numpy")}
    config = {"commit": COMMIT, "rounds": args.rounds, "rates": args.rates, "target": target,
              "versions": versions}
    if args.arms != ARMS:
        config["arms"] = list(args.arms)
    if args.save_mismatches:
        config["save_mismatches"] = True
    if args.capture_only:
        config["capture_only"] = True
    if args.trace_grabs:
        config["trace_grabs"] = True
    progress_path = args.output / "progress.json"
    if args.resume:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["config"] != config or args.seed is not None and progress["seed"] != args.seed:
            raise RuntimeError("The requested sweep differs from the saved monitor, versions, or settings")
    else:
        if progress_path.exists():
            raise RuntimeError("This output already contains a sweep. Use --resume or a new --output directory.")
        seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(64)
        progress = {"config": config, "seed": seed, "audit": None,
                    "trials": ordered_trials(args.rates, args.rounds, seed, args.arms)}
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(progress_path, progress)
    if args.trace_grabs:
        progress.setdefault("qpc_calibration", []).append(qpc_calibration())
        write_json(progress_path, progress)

    with ScenePlayer(target["bounds"]) as player:
        player.prepare(source)
        if progress["audit"] is None:
            audit = {}
            if not args.capture_only:
                print("Checking paired searches and byte comparisons on the HP 324pe", flush=True)
                with Worker("paired", target) as worker:
                    audit = {name: list(measure_scene(worker, player, name, steps, 10)) for name, steps in source.items()}
                check_audit(audit, source)
            progress["audit"] = audit
            write_json(progress_path, progress)
        for trial in progress["trials"]:
            for arm in trial["order"]:
                rates = trial["rate_order"]
                unfinished = [rate for rate in rates if arm not in trial["by_rate"].get(str(rate), {})
                              or not args.capture_only and arm in CAPTURE_ARMS and rate in CONTROL_RATES
                              and arm not in trial["controls"].get(str(rate), {})]
                if not unfinished:
                    continue
                with Worker(arm, target) as worker:
                    for rate in unfinished:
                        label = LABELS[arm].split(" + ")[0] if args.capture_only else LABELS[arm]
                        print(f"Round {trial['round']}/{args.rounds} | {rate} updates/s | {label}", flush=True)
                        group = trial["by_rate"].setdefault(str(rate), {})
                        if arm not in group:
                            evidence_root = (args.output.resolve() / "mismatch-frames" / arm /
                                             f"round-{trial['round']:02d}" / f"{rate}-hz"
                                             if args.save_mismatches else None)
                            group[arm] = {}
                            for name, steps in source.items():
                                evidence_dir = str(evidence_root / name) if evidence_root is not None else None
                                group[arm][name] = list(measure_scene(worker, player, name, steps, rate,
                                                                    capture_only=args.capture_only,
                                                                    evidence_dir=evidence_dir, trace_grabs=args.trace_grabs))
                            write_json(progress_path, progress)
                        if not args.capture_only and arm in CAPTURE_ARMS and rate in CONTROL_RATES:
                            controls = trial["controls"].setdefault(str(rate), {})
                            if arm not in controls:
                                controls[arm] = {name: list(measure_scene(worker, player, name, source[name], rate, True,
                                                                         trace_grabs=args.trace_grabs))
                                                 for name in ("sparse", "dense")}
                                write_json(progress_path, progress)
    data = {**config, "seed": progress["seed"], "paired": paired_results(progress["audit"]),
            "audit": progress["audit"], "trials": progress["trials"],
             "by_rate": {str(rate): summarize_rate(progress["trials"], rate, source, args.arms) for rate in args.rates}}
    if args.trace_grabs:
        data["qpc_calibration"] = progress["qpc_calibration"]
    write_json(args.output / "results.json", data)
    (args.output / "report.md").write_text(sweep_report(data), encoding="utf-8")
    incorrect = sum(row["incorrect"] for rate in data["by_rate"].values()
                    for arms in rate["summary"].values() for row in arms.values())
    print(f"Saved results in {args.output} | {incorrect} incorrect verified results", flush=True)
    if incorrect:
        raise RuntimeError(f"{incorrect} live detections differed from the scene's expected result")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--rounds", type=int, default=5)
    rate = cli.add_mutually_exclusive_group()
    rate.add_argument("--hz", type=int)
    rate.add_argument("--rates", nargs="+", type=int, default=DEFAULT_RATES)
    cli.add_argument("--seed", type=int)
    cli.add_argument("--visible-steps", type=int, default=32)
    cli.add_argument("--passive-steps", type=int, default=12)
    cli.add_argument("--arms", nargs="+", choices=ARMS, help="Run only these live pipelines")
    cli.add_argument("--save-mismatches", action="store_true",
                     help="Save lossless pixels and full-frame checks for wrong synthetic-fixture captures")
    cli.add_argument("--trace-grabs", action="store_true", help="Record every timed grab call and observed scene stamp")
    cli.add_argument("--capture-only", action="store_true", help="Run legacy deadline loops on the sparse fixture without search")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--output", type=Path, default=ROOT / "results-downstream-hp324pe")
    cli.add_argument("--worker", help=argparse.SUPPRESS)
    cli.add_argument("--target", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.worker:
        worker_main(args.worker, json.loads(args.target))
    else:
        args.rates = [args.hz] if args.hz is not None else list(dict.fromkeys(args.rates))
        args.arms = tuple(dict.fromkeys(args.arms or (CAPTURE_ARMS if args.capture_only else ARMS)))
        if args.capture_only and any(arm not in CAPTURE_ARMS for arm in args.arms):
            cli.error("capture-only supports rapidshot_full, dxcam, and mss")
        if args.rounds < 1 or any(rate < 1 or rate > 100 for rate in args.rates):
            cli.error("rounds must be positive and rates must be between 1 and 100")
        run(args)


if __name__ == "__main__":
    main()
