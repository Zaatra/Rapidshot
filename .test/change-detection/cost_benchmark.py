"""Measure capture and exact-change costs in isolated Windows worker processes."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from collections import defaultdict
import ctypes as ct
from dataclasses import replace
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
from random import Random
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from capture_timing_followup import frequency, quantiles
from cost_adapters import (ADAPTIVE_WAIT_ARM, ARM_DEPENDENCIES, CAPTURE_ARMS, DXCAM_MATCHED_ARMS,
                           DXCAM_TIGHT_RETRY_SECONDS, DXCAM_TIGHT_STREAM, DXCAM_VIEW_ARMS, FULL_EXACT_ARMS,
                           ROOT, STREAM_ARMS, TIGHT_ARMS, TIGHT_CONTROLS, TILED_ARMS, CostAdapter)
from downstream_benchmark import dpi_aware, wait_until, write_json
from downstream_scene import (ScenePlayer, STAMP_ORIGIN, _read_stamp, _stamp, _stamp_rect, capture_cover_state,
                              expected_detections, read_cover_stamp, read_stamp, render, scenes, templates)
from downstream_task import TemplateSearch
from monitor_target import select_monitor


DEFAULT_ARMS = ("baseline",) + CAPTURE_ARMS + STREAM_ARMS
ARMS = DEFAULT_ARMS + TILED_ARMS + TIGHT_CONTROLS + DXCAM_MATCHED_ARMS + (ADAPTIVE_WAIT_ARM,)
WORKLOADS = ("sparse", "dense", "excluded_cover", "idle")
PHASES = ("capture", "template", "shared_tiles")
NS_SECOND = 1_000_000_000
NS_MS = 1_000_000
STAMP_FIRST = 0x1000


class Fixture:
    """Reusable scene rasters, with a unique visible stamp per scheduled state."""

    def __init__(self, target):
        left, top, right, bottom = target["bounds"]
        self.width, self.height = right - left, bottom - top
        self.steps = scenes(self.width, self.height, visible_steps=12, passive_steps=12)
        self.images = {}

    def prewarm(self, workloads):
        for workload in workloads:
            for step in self.steps[workload]:
                if step.visible_update:
                    self._raster(step)

    def _raster(self, step):
        if step.stamp not in self.images:
            self.images[step.stamp] = render(step, self.width, self.height)
        return self.images[step.stamp]

    def seed(self, player, workload):
        base = self.steps[workload][0]
        # The first paint clears the cover and previous arm, then seeds stable
        # content before the child creates its camera.
        image = self._raster(base)
        _stamp(image, STAMP_ORIGIN, base.stamp)
        player.rasters[id(base)] = image
        try:
            player.present(base, restart=True)
        finally:
            player.rasters.pop(id(base), None)
        return base.stamp

    def show(self, player, workload, index):
        variants = self.steps[workload]
        variant = 1 + index % (len(variants) - 1)
        base = variants[variant]
        if workload == "excluded_cover":
            player.present(base, wait_for_desktop=False)
            return base.stamp
        stamped = replace(base, stamp=STAMP_FIRST + index)
        image = self._raster(base)
        _stamp(image, STAMP_ORIGIN, stamped.stamp)
        player.rasters[id(stamped)] = image
        try:
            player.present(stamped, wait_for_desktop=False)
        finally:
            player.rasters.pop(id(stamped), None)
        return stamped.stamp


class SnapshotHistory:
    """Owned full-frame comparison baseline independent of the capture API."""

    def __init__(self, width, height, *, tight=False, retain_owned=False):
        from cpu_default import TILE, Grid, Native

        self.grid = Grid(width, height)
        self.native = Native()
        self.indices = np.empty_like(self.grid.all)
        self.pixels = None
        self.bytes_written = 0
        self.allocations = 0
        self.retain_owned = retain_owned
        self.bounds = ()
        self.tile_rectangles = tuple(
            (left, top, min(left + TILE, width), min(top + TILE, height))
            for top in range(0, height, TILE) for left in range(0, width, TILE)
        ) if tight else ()

    def compare_and_retain(self, pixels):
        if self.pixels is None:
            if self.retain_owned:
                self.pixels = pixels
            else:
                self.allocations += 1
                self.pixels = pixels.copy(order="C")
                self.bytes_written += pixels.nbytes
            self.bounds = self.tile_rectangles
            return self.grid.all
        count = self.native.cpu_compare(self.pixels, pixels, self.grid.all, self.indices)
        indices = self.indices[:count]
        if self.tile_rectangles:
            # Tight byte bounds precede snapshot replacement. Keep this worker
            # independent of RapidShot and its COM registrations.
            bounds = []
            for index in indices:
                left, top, right, bottom = self.tile_rectangles[index]
                region = np.s_[top:bottom, left:right]
                changed = np.any(self.pixels[region] != pixels[region], axis=2)
                rows = np.flatnonzero(changed.any(axis=1))
                columns = np.flatnonzero(changed.any(axis=0))
                bounds.append((left + int(columns[0]), top + int(rows[0]),
                               left + int(columns[-1]) + 1, top + int(rows[-1]) + 1))
            self.bounds = tuple(bounds)
        if count:
            if self.retain_owned:
                self.pixels = pixels
            else:
                np.copyto(self.pixels, pixels)
                self.bytes_written += pixels.nbytes
        return indices


class VideoSamples:
    def __init__(self, target, enabled):
        self.rows = []
        self.interval_ns = 500 * NS_MS
        self.due_ns = 0
        self.query = None
        self.unavailable = None if enabled else "disabled"
        if enabled:
            from cost_telemetry import VideoMemory

            try:
                self.query = VideoMemory(adapter_index=target["device_index"])
            except (RuntimeError, OSError) as error:
                self.unavailable = str(error)

    def sample(self, now, *, force=False):
        if self.query is None or (not force and now < self.due_ns):
            return
        self.due_ns = now + self.interval_ns
        try:
            value = self.query.sample()
        except (RuntimeError, OSError) as error:
            self.unavailable = str(error)
            self.close()
            return
        self.rows.append({"qpc_ns": now, "segments": value})

    def close(self):
        if self.query is not None:
            close = getattr(self.query, "close", None)
            if close is not None:
                close()
            self.query = None


class PresentClock:
    """Use exactly one QPC-origin timestamp domain per native frame."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.seen_native = set()

    def describe(self, sample, returned_ns):
        present = sample.present_ns
        if present is None:
            return None, None, None
        if not (0 < present <= returned_ns):
            return None, False, "native timestamp outside verified QPC ordering"
        first = present not in self.seen_native
        self.seen_native.add(present)
        return present, first, None


class Pipeline:
    def __init__(self, mode, arm, workload, width, height):
        self.arm, self.workload = arm, workload
        self.packed_copy_bytes = 0
        self.packed_copy_allocations = 0
        self.materialized_bytes = self.materialized_allocations = self.materialized_ns = 0
        self.probe_bytes = self.probe_allocations = 0
        self.search = TemplateSearch(templates(), width, height) if mode != "capture" else None
        self.snapshot = None
        if mode == "shared_tiles" and arm not in ("baseline", "dxcam_copy", *DXCAM_VIEW_ARMS, *FULL_EXACT_ARMS, *TILED_ARMS):
            self.snapshot = SnapshotHistory(width, height, tight=arm in DXCAM_MATCHED_ARMS,
                                            retain_owned=arm == DXCAM_TIGHT_STREAM)
        self.fixture = scenes(width, height, visible_steps=12, passive_steps=12)[workload]
        self.unseen_tile_union = set()
        self.unseen_rectangles = set()
        self.last_passive_key = None
        self.seen_visible = set()

    def _crop(self, sample, rect=None, *, fixture_probe=False):
        started = time.perf_counter_ns()
        pixels = sample.owner.materialize(rect)
        elapsed = time.perf_counter_ns() - started
        if rect is None:
            width, height = sample.changes.size
        else:
            left, top, right, bottom = rect
            width, height = right - left, bottom - top
        if (pixels.shape != (height, width, 4) or pixels.dtype != np.uint8
                or not pixels.flags.c_contiguous):
            raise RuntimeError(f"{self.arm} returned invalid materialized BGRA pixels for {rect!r}")
        if fixture_probe:
            self.probe_bytes += pixels.nbytes
            self.probe_allocations += 1
        else:
            self.materialized_bytes += pixels.nbytes
            self.materialized_allocations += 1
            self.materialized_ns += elapsed
        return pixels

    def _run_task(self, sample, exact_tiles, tight_rectangles=()):
        pixels = sample.pixels
        if pixels is None:
            started = time.perf_counter_ns()

            def read_crop(rect):
                return self._crop(sample, rect)

            if self.arm in TIGHT_ARMS:
                result = self.search.update_tight_from_crops(read_crop, tight_rectangles, force_full=exact_tiles is None)
            else:
                result = self.search.update_from_crops(read_crop, exact_tiles or (), force_full=exact_tiles is None)
            return result, sample.gate_ns, 0, time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        if not pixels.flags.c_contiguous:
            pixels = np.ascontiguousarray(pixels)
            self.packed_copy_bytes += pixels.nbytes
            self.packed_copy_allocations += 1
        packed_ns = time.perf_counter_ns() - started
        gate_ns = sample.gate_ns
        if self.snapshot is not None:
            started = time.perf_counter_ns()
            exact_tiles = self.snapshot.compare_and_retain(pixels)
            if self.arm in DXCAM_MATCHED_ARMS:
                tight_rectangles = self.snapshot.bounds
            gate_ns = time.perf_counter_ns() - started
        started = time.perf_counter_ns()
        if self.arm in TIGHT_ARMS:
            def read_crop(rect):
                left, top, right, bottom = rect
                return pixels[top:bottom, left:right]

            result = self.search.update_tight_from_crops(read_crop, tight_rectangles, force_full=exact_tiles is None)
        else:
            result = self.search.find_all(pixels) if exact_tiles is None else self.search.update(pixels, exact_tiles)
        return result, gate_ns, packed_ns, time.perf_counter_ns() - started

    def _interpret_stamp(self, sample):
        pixels = sample.pixels
        if pixels is None:
            marker = self._crop(sample, _stamp_rect(STAMP_ORIGIN), fixture_probe=True)
            stamp = _read_stamp(marker, (0, 0))
        else:
            stamp = read_stamp(pixels)
        if self.workload in ("sparse", "dense"):
            index = stamp - STAMP_FIRST if stamp is not None else -1
            if index < 0:
                return stamp, None, None
            expected = self.fixture[1 + index % (len(self.fixture) - 1)]
            return stamp, None, expected
        seed = self.fixture[0]
        if stamp != seed.stamp:
            return stamp, None, None
        probe_step = seed
        if pixels is None and seed.covers:
            left, top, right, bottom = seed.cover_rect
            pixels = self._crop(sample, seed.cover_rect, fixture_probe=True)
            probe_step = replace(seed, cover_rect=(0, 0, right - left, bottom - top),
                                 objects={name: (x - left, y - top) for name, (x, y) in seed.objects.items()})
        cover = capture_cover_state(probe_step, pixels)
        cover_stamp = read_cover_stamp(probe_step, pixels) if cover == "visible" else None
        return stamp, (cover, cover_stamp), seed

    def process(self, sample, called, returned, clock):
        stamp, cover_key, expected = self._interpret_stamp(sample)
        present, native_new, timestamp_error = clock.describe(sample, returned)
        changes = sample.changes
        tiles = changes.indices if changes is not None else None
        if tiles is not None and self.search is not None:
            # The public gate advances on every API acquisition. The downstream
            # task receives only valid fixture frames, so retain skipped changes.
            self.unseen_tile_union.update(tiles)
            if self.arm in TIGHT_ARMS:
                bounds = changes.tight_rectangles
                self.unseen_rectangles.update(changes.rectangles if bounds is None else bounds)
        row = {
            "stamp": stamp, "cover": cover_key, "capture_started_ns": called, "capture_returned_ns": returned,
            "present_ns": present, "producer_ready_ns": sample.ready_ns, "timestamp_error": timestamp_error,
            "raw_native_present_ns": sample.present_ns,
            "accumulated_frames": sample.accumulated_frames, "new_native_timestamp": native_new,
            "history_bytes_written": sample.history_bytes, "gate_ns": sample.gate_ns,
            "exact_tile_count": len(tiles) if tiles is not None else None,
            "change_reason": changes.reason if changes is not None else None,
            "fallback_reason": changes.fallback_reason if changes is not None else None,
        }
        if expected is None:
            row["outcome"] = "outside_measured_fixture"
            return row
        if self.workload in ("sparse", "dense"):
            deliver = stamp not in self.seen_visible
            if deliver:
                self.seen_visible.add(stamp)
        else:
            accepted_cover = cover_key is not None and cover_key[1] is not None
            new_cover = accepted_cover and cover_key != self.last_passive_key
            self.last_passive_key = cover_key
            deliver = native_new is True or new_cover
        row["outcome"] = "delivered" if deliver else "repeated"
        if self.search is None or not deliver:
            return row
        if tiles is not None:
            tiles = tuple(sorted(self.unseen_tile_union))
        materialized_before = self.materialized_ns
        bytes_before = self.materialized_bytes
        snapshot_before = self.snapshot.bytes_written if self.snapshot is not None else 0
        result, gate_ns, packing_ns, search_ns = self._run_task(sample, tiles, tuple(sorted(self.unseen_rectangles)))
        if self.snapshot is not None:
            row["history_bytes_written"] = self.snapshot.bytes_written - snapshot_before
        self.unseen_tile_union.clear()
        self.unseen_rectangles.clear()
        actual = tuple(result.detections)
        cover_state = cover_key[0] if cover_key is not None else None
        oracle = expected_detections(expected, cover_state or "none")
        expected_pairs = tuple(sorted((name, *position) for name, position in oracle.items()))
        row.update(gate_ns=gate_ns, packing_ns=packing_ns, search_ns=search_ns,
                   result_ns=time.perf_counter_ns(), detections=actual, expected_detections=expected_pairs,
                   correct_fixture=actual == expected_pairs, searched_origins=result.searched_origins,
                   search_calls=result.calls,
                   task_crop_materialization_ns=self.materialized_ns - materialized_before,
                   task_crop_materialization_bytes=self.materialized_bytes - bytes_before)
        return row

    def accounting(self):
        result = {"task_input_pack_bytes": self.packed_copy_bytes,
                  "task_input_pack_allocations": self.packed_copy_allocations}
        if self.arm in TILED_ARMS:
            result.update(task_crop_materialization_bytes=self.materialized_bytes,
                          task_crop_materialization_allocations=self.materialized_allocations,
                          fixture_probe_materialization_bytes=self.probe_bytes,
                          fixture_probe_materialization_allocations=self.probe_allocations)
        if self.snapshot is not None:
            history = self.snapshot
            result.update(private_history_bytes_written=history.bytes_written,
                          private_history_payload_bytes_retained=(history.pixels.nbytes
                                                                  if history.pixels is not None else 0),
                          private_history_payload_allocations=history.allocations)
        return result


def record_wrong_frames(pipeline, folder):
    """Keep exact bytes for diagnostic trials whose timings are set aside."""
    folder = Path(folder)
    process = pipeline.process

    def save_if_wrong(sample, called, returned, clock):
        row = process(sample, called, returned, clock)
        if row.get("correct_fixture") is False:
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{row['stamp']:05d}-{returned}"
            pixels = folder / f"{name}.npy"
            wrong = pipeline._crop(sample) if sample.pixels is None else sample.pixels
            np.save(pixels, wrong, allow_pickle=False)
            write_json(folder / f"{name}.json", {**row, "pixels": pixels.name})
        return row

    pipeline.process = save_if_wrong


def baseline_work(config, pipeline, pixels):
    """Compare worker overhead while importing Python, NumPy and OpenCV only."""
    import cv2
    del cv2
    rows = []
    stop = config["stop_ns"]
    period = round(NS_SECOND / config["rate"])
    if config["workload"] in ("idle", "excluded_cover"):
        wait_until(stop)
    else:
        for due in range(config["start_ns"], stop, period):
            wait_until(due)
            now = time.perf_counter_ns()
            if now >= stop:
                break
            # Exercise the marker decode even with no native capture library.
            read_stamp(pixels)
            if pipeline.search is not None:
                result = pipeline.search.find_all(pixels)
                rows.append((now, time.perf_counter_ns(), len(result.detections)))
    imports = sorted(name for name in ("rapidshot", "dxcam", "mss", "windows_capture") if name in sys.modules)
    if imports:
        raise RuntimeError(f"Baseline unexpectedly imported capture packages: {imports}")
    return rows


def measure_window(adapter, pipeline, config, video):
    rows = []
    empty_calls = 0
    empty_call_ns = 0
    wait_ns = 0
    retry_seconds = DXCAM_TIGHT_RETRY_SECONDS.get(adapter.arm)
    if retry_seconds is None and adapter.arm in ("dxcam_copy", *DXCAM_VIEW_ARMS, "rapidshot_plain", *FULL_EXACT_ARMS,
                                                  ADAPTIVE_WAIT_ARM, *TILED_ARMS):
        retry_seconds = 0.0005
    clock = PresentClock(adapter)
    stop = config["stop_ns"]
    while (called := time.perf_counter_ns()) < stop:
        sample = adapter.read(budget=max(0, min(0.005, (stop - called) / NS_SECOND)))
        returned = time.perf_counter_ns()
        if sample is None:
            empty_calls += 1
            empty_call_ns += returned - called
            if retry_seconds is not None:
                paused = time.perf_counter_ns()
                time.sleep(retry_seconds)
                wait_ns += time.perf_counter_ns() - paused
        else:
            try:
                if sample.pixels is None:
                    if adapter.arm not in TILED_ARMS or sample.changes.size != (config["width"], config["height"]):
                        raise RuntimeError(f"{adapter.arm} returned unexpected tiled frame dimensions")
                elif sample.pixels.shape != (config["height"], config["width"], 4) or sample.pixels.dtype != np.uint8:
                    raise RuntimeError(f"{adapter.arm} returned unexpected shape/dtype {sample.pixels.shape}, {sample.pixels.dtype}")
                row = pipeline.process(sample, called, returned, clock)
            finally:
                sample.release()
            rows.append(row)
        video.sample(time.perf_counter_ns())
    return {"observations": rows, "empty_grab_calls": empty_calls, "empty_grab_call_ns": empty_call_ns,
            "explicit_empty_poll_sleep_ns": wait_ns, "requested_empty_poll_sleep_seconds": retry_seconds}


def emit(kind, data):
    print(kind + " " + json.dumps(data, separators=(",", ":"), allow_nan=False), flush=True)


def worker(arm, target, config):
    dpi_aware()
    clock_frequency = frequency()
    left, top, right, bottom = target["bounds"]
    setup = dict(config, width=right - left, height=bottom - top)
    adapter = None
    allocation = None
    video = None
    if arm != "baseline":
        adapter = CostAdapter(arm, target, config["rate"], clock_frequency)
    try:
        pipeline = Pipeline(config["pipeline"], arm, config["workload"], setup["width"], setup["height"])
        baseline = None
        if adapter is None:
            # Build the static reference before the fixed measurement window.
            import cv2

            baseline = render(pipeline.fixture[0], setup["width"], setup["height"])
        # Seed lazy capture allocation before process-window telemetry.
        if adapter is not None:
            for _ in range(3):
                warm = adapter.read(budget=0.02)
                if warm is not None:
                    try:
                        if pipeline.search is not None:
                            pipeline._run_task(warm, None)
                    finally:
                        warm.release()
                    break
            pipeline = Pipeline(config["pipeline"], arm, config["workload"], setup["width"], setup["height"])
        if config.get("wrong_frame_folder"):
            record_wrong_frames(pipeline, config["wrong_frame_folder"])
        video = VideoSamples(target, config["gpu_memory"])
        if config["allocation_profile"]:
            from cost_telemetry import AllocationProfiler

            allocation = AllocationProfiler()
        emit("READY", {"pid": os.getpid(),
                       "adapter_luid": getattr(video.query, "adapter_luid", None)})
        timing = json.loads(sys.stdin.readline())
        setup.update(timing)
        wait_until(timing["start_ns"])
        if adapter is not None:
            adapter.reset_window(timing)
        if allocation is not None:
            allocation.start()
        video.sample(time.perf_counter_ns(), force=True)
        begin = time.perf_counter_ns()
        if adapter is None:
            replay = baseline_work(setup, pipeline, baseline)
            result = {"observations": [], "baseline_opencv_replay": replay, "empty_grab_calls": 0}
        else:
            result = measure_window(adapter, pipeline, setup, video)
        finished = time.perf_counter_ns()
        video.sample(finished, force=True)
        result.update(
            measurement_start_ns=begin, measurement_end_ns=finished, qpc_frequency_hz=clock_frequency,
            stamp_clock=adapter.stamp_clock if adapter is not None else "no capture",
            path_accounting=(adapter.snapshot() if adapter is not None else {}) | pipeline.accounting(),
            video_memory={"samples": video.rows, "unavailable_reason": video.unavailable},
            allocations=allocation.finish() if allocation is not None else None,
        )
        # The parent takes its last kernel counter snapshot at the fixed stop
        # time. Keep the process and camera alive until it acknowledges.
        sys.stdin.readline()
        emit("RESULT", result)
        sys.stdin.readline()
    finally:
        if video is not None:
            video.close()
        if adapter is not None:
            adapter.close()


class Worker:
    def __init__(self, arm, target, config):
        self.arm = arm
        self.ready = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.ready_payload = None
        self.messages = []
        self.process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", __file__, "--worker", arm,
             "--target-json", json.dumps(target), "--config-json", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf8",
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
        if not self.ready.wait(15) or self.ready_payload is None or not self.reader.is_alive():
            self.close()
            raise RuntimeError(f"{self.arm} failed to open: {' '.join(self.messages[-12:])}")
        return self

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def collect(self):
        if not self.done.wait(12) or self.result is None:
            raise RuntimeError(f"{self.arm} failed to produce observations: {' '.join(self.messages[-12:])}")
        self.send({"finalized": True})
        return self.result

    def close(self):
        if self.process.poll() is None:
            try:
                self.process.stdin.close()
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)
        self.reader.join(2)
        self.process.stdout.close()

    def __exit__(self, *_):
        self.close()


def run_trial(player, fixture, target, settings):
    from cost_telemetry import ProcessSampler

    arm, hz, workload = (settings[key] for key in ("arm", "rate", "workload"))
    fixture.seed(player, workload)
    with Worker(arm, target, settings) as child:
        sampler = ProcessSampler(child.ready_payload["pid"], sample_seconds=0.1, pdh=settings["pdh"],
                                 gpu_adapter_luid=child.ready_payload["adapter_luid"])
        sampler.prepare().start()
        # Source producer, capture worker, and final counter snapshot use the
        # same agreed QPC window, starting at the first kernel CPU sample.
        start = sampler._initial_cpu.at_ns
        stop = start + round(settings["duration"] * NS_SECOND)
        submitted = []
        try:
            child.send(dict(start_ns=start, stop_ns=stop))
            if workload != "idle":
                period = round(NS_SECOND / hz)
                for i, due in enumerate(range(start, stop, period)):
                    wait_until(due)
                    begin = time.perf_counter_ns()
                    if begin >= stop:
                        break
                    stamp = fixture.show(player, workload, i)
                    submitted.append({"index": i, "due_ns": due, "send_start_ns": begin,
                                      "send_end_ns": time.perf_counter_ns(), "stamp": stamp})
            wait_until(stop)
        finally:
            process = sampler.finish()
        child.send({"export": True})
        observed = child.collect()
    result = dict(settings, submitted=submitted, worker=observed, process=process,
                  launcher_pid=child.process.pid, worker_pid=child.ready_payload["pid"],
                  requested_start_ns=start, requested_stop_ns=stop, timing_frequency_hz=frequency())
    result["summary"] = summarize_trial(result)
    delivered = result["summary"]["new_delivered_frames"]
    cpu = process["cpu"]
    cpu["delivered_frames"] = delivered
    cpu["cycles_per_delivered_frame"] = safe_ratio(cpu["cycles"], delivered)
    for engine in process["gpu_engines"]["engines"]:
        if engine["sampled_seconds"] >= 0.9 * cpu["elapsed_seconds"]:
            engine["estimated_busy_ms_per_delivered_frame"] = safe_ratio(
                engine["busy_ms_per_second"] * cpu["elapsed_seconds"], delivered)
    return result


def safe_ratio(value, denominator):
    return value / denominator if value is not None and denominator else None


def summarize_trial(run):
    worker = run["worker"]
    observed = worker["observations"]
    expected = {frame["stamp"]: frame for frame in run["submitted"]} if run["workload"] in ("sparse", "dense") else {}
    accepted = []
    for row in observed:
        if row["outcome"] != "delivered":
            continue
        submission = expected.get(row["stamp"])
        if expected and (submission is None or row["capture_returned_ns"] < submission["send_start_ns"]):
            continue
        if not expected and row["present_ns"] is not None and row["present_ns"] < run["requested_start_ns"]:
            continue
        if row["capture_returned_ns"] <= run["requested_stop_ns"]:
            accepted.append((row, submission))
    returned_after = sum(r["capture_returned_ns"] > run["requested_stop_ns"] for r in observed)
    native = sum(r["new_native_timestamp"] is True for r in observed)
    duration = (run["requested_stop_ns"] - run["requested_start_ns"]) / NS_SECOND
    accounted = worker["path_accounting"]
    result_ages, return_ages, queue_ages, end_to_end = [], [], [], []
    on_time = 0
    correct_on_time = (0 if run["pipeline"] != "capture" and run["arm"] != "baseline"
                       and run["workload"] in ("sparse", "dense") else None)
    for row, submission in accepted:
        present = row["present_ns"]
        returned = row["capture_returned_ns"]
        finished = row.get("result_ns")
        if present is not None and (submission is None or present >= submission["send_start_ns"]):
            return_ages.append((returned - present) / NS_MS)
            if finished is not None and finished <= run["requested_stop_ns"]:
                result_ages.append((finished - present) / NS_MS)
        ready = row["producer_ready_ns"]
        if ready is not None and present is not None and present <= ready <= returned:
            queue_ages.append((returned - ready) / NS_MS)
        if submission is not None:
            done = returned if finished is None else finished
            delay = (done - submission["send_start_ns"]) / NS_MS
            end_to_end.append(delay)
            within_deadline = delay <= 1000 / run["rate"] and done <= run["requested_stop_ns"]
            on_time += within_deadline
            if correct_on_time is not None:
                correct_on_time += within_deadline and row.get("correct_fixture") is True
    payload_fields = {name: value for name, value in accounted.items()
                      if name.endswith(("_bytes", "_written", "_allocations", "_operations")) and value is not None}
    source_starts = [row["send_start_ns"] for row in run["submitted"]]
    return {
        "grab_returns": len(observed), "observed_native_instances": native,
        "new_delivered_frames": len(accepted), "returns_after_stop": returned_after, "eligible_markers": len(expected),
        "delivered_per_second": len(accepted) / duration, "on_time": on_time, "correct_on_time": correct_on_time,
        "new_state_recall": safe_ratio(len(accepted), len(expected)),
        "schedule_interval_ms": quantiles((b-a)/NS_MS for a, b in zip(source_starts, source_starts[1:])),
        "desktop_to_delivery_ms": quantiles(return_ages), "desktop_to_downstream_ms": quantiles(result_ages),
        "stream_producer_ready_to_read_ms": quantiles(queue_ages), "submission_to_done_ms": quantiles(end_to_end),
        "gate_ms": quantiles(row["gate_ns"]/NS_MS for row, _ in accepted if row["gate_ns"] is not None),
        "search_ms": quantiles(row["search_ns"]/NS_MS for row, _ in accepted if row.get("search_ns") is not None),
        "task_crop_materialization_ms": quantiles(row["task_crop_materialization_ns"] / NS_MS
                                                   for row, _ in accepted if row.get("task_crop_materialization_ns")
                                                   is not None and run["arm"] in TILED_ARMS),
        "fixture_result_mismatches": sum(row.get("correct_fixture") is False for row, _ in accepted),
        "payload_per_second": {name: value/duration for name, value in payload_fields.items()},
        "payload_per_delivered_frame": {name: safe_ratio(value, len(accepted))
                                         for name, value in payload_fields.items()},
    }


@contextmanager
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


def workload_matrix(args):
    chosen = []
    for arm in args.arms:
        if arm in ("windows_capture_dxgi", "windows_capture_wgc") and (
            importlib.util.find_spec("windows_capture") is None
            and not (ROOT / "vendor" / "windows_capture").is_dir()
        ):
            print(f"{arm} requires Windows Capture 2.0.1 in an isolated vendor directory.", flush=True)
        if arm.startswith("windows_capture") or arm in STREAM_ARMS or arm == ADAPTIVE_WAIT_ARM:
            phases = ("capture",) if "capture" in args.pipelines else ()
        elif arm in DXCAM_MATCHED_ARMS:
            phases = ("shared_tiles",) if "shared_tiles" in args.pipelines else ()
        elif arm in ("dxcam_copy", *DXCAM_VIEW_ARMS):
            # Both downstream selections use the original full-frame OpenCV task for DXcam.
            phases = tuple(dict.fromkeys("template" if phase == "shared_tiles" else phase for phase in args.pipelines))
        else:
            phases = args.pipelines
        for phase in phases:
            for workload in args.workloads:
                # Idle must be a separate fixed 10-second run. Its rates are
                # duplicates across one-shot arms but matter for WGC pacing.
                for rate in args.rates:
                    chosen.append((arm, phase, workload, rate))
    return chosen


def run_matrix(args):
    dpi_aware()
    target = select_monitor(args.monitor)
    if target["physical_refresh_hz"] < max(args.rates):
        raise RuntimeError("Requested source rate exceeds the monitor refresh")
    base = args.output.resolve()
    manifest = base / "manifest.json"
    trial_dir = base / "trials"
    configuration = {"schema": 1, "monitor": target, "duration": args.duration, "seed": args.seed,
                     "gpu_memory": args.gpu_memory, "pdh": args.pdh, "allocation_profile": args.allocation_profile,
                     "versions": {name: version(name) for name in ("rapidshot", "dxcam", "mss", "numpy", "opencv-python")},
                     "optional_dependency_versions": ARM_DEPENDENCIES}
    if args.save_wrong_frames:
        configuration["save_wrong_frames"] = True
    if manifest.exists():
        if not args.resume:
            raise FileExistsError(f"{manifest} already exists. Specify --resume to add missing trials.")
        previous = json.loads(manifest.read_text(encoding="utf8"))
        if configuration != previous:
            raise ValueError("Resume settings, capture target, or dependency versions changed")
    else:
        trial_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest, configuration)

    fixture = Fixture(target)
    fixture.prewarm(args.workloads)
    trial_dir.mkdir(parents=True, exist_ok=True)
    matrix = workload_matrix(args)
    if not matrix:
        raise ValueError("Selected arm and pipeline filters leave no trials")
    failures = 0
    with display_awake(), ScenePlayer(target["bounds"]) as player:
        for round_number in range(1, args.rounds + 1):
            ordered = list(matrix)
            Random(args.seed + round_number).shuffle(ordered)
            for arm, phase, workload, rate in ordered:
                key = f"r{round_number:03d}-{phase}-{workload}-{rate:03d}-{arm}"
                path = trial_dir / f"{key}.json"
                error_path = trial_dir / f"{key}.failure.json"
                if path.exists():
                    continue
                print(f"Benchmarking {key}", flush=True)
                settings = dict(arm=arm, pipeline=phase, workload=workload, rate=rate, round=round_number,
                                duration=args.duration, gpu_memory=args.gpu_memory, pdh=args.pdh,
                                allocation_profile=args.allocation_profile)
                if args.save_wrong_frames:
                    settings["wrong_frame_folder"] = str(base / "wrong-frames" / key)
                try:
                    result = run_trial(player, fixture, target, settings)
                except (RuntimeError, OSError) as error:
                    print(f"{key} unavailable: {error}", flush=True)
                    write_json(error_path, {"settings": settings, "error": str(error)})
                    failures += 1
                    if args.fail_fast:
                        raise
                else:
                    write_json(path, result)
                    error_path.unlink(missing_ok=True)
    write_report(base)
    print(f"Results saved under {base}; {failures} failed trial(s)", flush=True)
    return int(failures != 0)


def median(values):
    usable = [v for v in values if v is not None]
    return statistics.median(usable) if usable else None


def memory_usage(video, segment):
    rows = video["samples"]
    values = []
    for row in rows:
        value = row["segments"].get(segment)
        if value is not None and value.get("current_usage_bytes") is not None:
            values.append(value["current_usage_bytes"])
    return max(values, default=None)


def format_value(value, divisor=1):
    return "—" if value is None else f"{value/divisor:.2f}"


def cost_row(key, rows):
    phase, workload, rate, arm = key
    metrics = [r["summary"] for r in rows]
    cpu = [r["process"]["cpu"] for r in rows]
    memory = [r["process"]["memory"] for r in rows]
    video = [r["worker"]["video_memory"] for r in rows]
    native_lookup = [r["process"]["gpu_engines"] for r in rows]
    stage_types = set()
    for r in native_lookup:
        stage_types.update(g["engine_type"] for g in r["engines"])
    gpu = ", ".join(str(engine) + " " + format_value(median(
        max((e["mean_busy_percent"] for e in data["engines"] if e["engine_type"] == engine), default=None)
        for data in native_lookup))
        for engine in sorted(stage_types)) or "—"
    copied_fields = (
        "mapped_staging_to_cpu_bytes", "dib_to_cpu_bytes", "stream_consumer_copy_bytes",
        "private_history_bytes_written", "task_input_pack_bytes", "task_crop_materialization_bytes",
        "fixture_probe_materialization_bytes",
    )

    def copied(row, denominator):
        values = [row[denominator].get(field) for field in copied_fields]
        if denominator == "payload_per_delivered_frame" and not row["new_delivered_frames"]:
            return None
        return sum(v for v in values if v is not None)

    per_sec = median(copied(x, "payload_per_second") for x in metrics)
    per_frame = median(copied(x, "payload_per_delivered_frame") for x in metrics)
    gpu_payload = "/".join(format_value(median(m[field].get("gpu_staging_payload_bytes") for m in metrics), 1048576)
                           for field in ("payload_per_second", "payload_per_delivered_frame"))
    checked = [m for m in metrics if m.get("correct_on_time") is not None]
    pool = [x["worker"]["path_accounting"].get("pool") for x in rows]
    result = [
        phase, workload, str(rate), arm, str(len(rows)),
        format_value(median(m["delivered_per_second"] for m in metrics)),
        format_value(median(m["grab_returns"] for m in metrics)),
        format_value(median(c["percent_of_one_core"] for c in cpu)),
        "/".join(format_value(median(c[field] for c in cpu), 1e6) for field in (
            "cycles_per_second", "cycles_per_delivered_frame")),
        "/".join(format_value(median(m.get(field) for m in memory), 1048576) for field in (
            "steady_private_bytes", "os_peak_private_bytes", "steady_working_set_bytes", "os_peak_working_set_bytes")),
        "/".join(format_value(median(memory_usage(item, segment) for item in video), 1048576)
                 for segment in ("local", "non_local")),
        gpu,
        f"{format_value(per_sec, 1048576)}/{format_value(per_frame, 1048576)}",
        gpu_payload,
        "/".join(format_value(median(m["desktop_to_delivery_ms"].get(q) for m in metrics)) for q in ("p50", "p95")),
        "/".join(format_value(median(m["desktop_to_downstream_ms"].get(q) for m in metrics)) for q in ("p50", "p95")),
        format_value(median(m["stream_producer_ready_to_read_ms"]["p50"] for m in metrics)),
        (f"{sum(m['correct_on_time'] for m in checked)}/{sum(m['eligible_markers'] for m in checked)}"
         if checked else "—"),
        f"{max(p['peak_outstanding'] for p in pool if p)}/{sum(p['exhaustions'] for p in pool if p)}"
        if any(pool) else "—",
    ]
    return "| " + " | ".join(result) + " |"


def write_report(output):
    groups = defaultdict(list)
    failed = []
    for path in sorted((output / "trials").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf8"))
        if path.name.endswith(".failure.json"):
            failed.append((path, row["error"]))
            continue
        key = tuple(row[field] for field in ("pipeline", "workload", "rate", "arm"))
        groups[key].append(row)
    lines = [
        "# Capture cost and frame age", "",
        "Values are medians of matched-format windows. The CPU numerator covers the measured worker and its background threads. Active coverage counts each offered marker once. Passive counts use new native timestamps or visible cover updates when available. Repeated grab returns stay in the raw totals. Missing measurements appear as —.", "",
        "CPU uses whole-process cycles plus Windows process time over the fixed window. GPU-engine values show the busiest observed instance of each engine type. They are process busy counters, not timing for specific copy commands. GPU memory shows peak process-reported local and non-local usage. Logical output copy bytes exclude unobserved driver traffic. CPU copies include private history and materialized task and fixture-probe crops. GPU staging and all copy stages are recorded separately in each trial.", "",
        "Desktop age uses each backend’s validated presentation timestamp. GDI and the Windows Capture DXGI binding report no native timestamp. The baseline imports Python, NumPy and OpenCV without capture libraries and runs at the listed offered rate.", "",
        "| Pipeline | Workload | Updates/s | Arm | Waves | New/second | Grab returns | CPU % of one core | Mcycles/s/frame | Private steady/peak and RSS steady/peak MiB | GPU local/non-local peak MiB | GPU engine busy % | CPU-copied MiB/s/frame | GPU-staged MiB/s/frame | Desktop→delivery p50/p95 ms | Desktop→result p50/p95 ms | Queue age p50 ms | Correct on time / offered | Peak pool outstanding/exhaustions |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|---:|---|",
    ]
    lines.extend(cost_row(key, rows) for key, rows in sorted(groups.items()))
    deltas = []
    for (phase, workload, rate, arm), rows in sorted(groups.items()):
        if arm not in ("rapidshot_exact", *TILED_ARMS):
            continue
        plain = {row["round"]: row for row in groups.get((phase, workload, rate, "rapidshot_plain"), ())}
        pairs = [(plain[row["round"]], row) for row in rows if row["round"] in plain]
        if not pairs:
            continue
        values = []
        for key in ("steady_private_bytes", "os_peak_private_bytes"):
            values.append(median(b["process"]["memory"][key] - a["process"]["memory"][key]
                                 for a, b in pairs))
        idle_cpu = median(b["process"]["cpu"]["percent_of_one_core"] - a["process"]["cpu"]["percent_of_one_core"]
                          for a, b in pairs if a["process"]["cpu"]["percent_of_one_core"] is not None
                          and b["process"]["cpu"]["percent_of_one_core"] is not None)
        deltas.append(f"| {arm} | {phase} | {workload} | {rate} | {len(pairs)} | {format_value(values[0], 1048576)} | "
                      f"{format_value(values[1], 1048576)} | {format_value(idle_cpu)} |")
    if deltas:
        lines.extend([
            "", "## Exact and tiled arms minus plain RapidShot", "",
            "| Arm | Pipeline | Workload | Updates/s | Paired rounds | Steady private MiB | Windows peak private MiB | CPU percentage points of one core |",
            "|---|---|---|---:|---:|---:|---:|---:|", *deltas,
        ])
    if failed:
        lines.extend(["", "## Incomplete trials", ""])
        for path, error in failed:
            lines.append(f"- {error.replace(chr(10), ' ')} [Details](trials/{path.name})")
    lines.extend(["", "[Configuration](manifest.json) · [Raw trials](trials)"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--monitor", default="HP 324pe")
    cli.add_argument("--arms", nargs="+", choices=ARMS, default=list(DEFAULT_ARMS))
    cli.add_argument("--rates", nargs="+", type=int, default=[10, 30, 100])
    cli.add_argument("--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    cli.add_argument("--pipelines", nargs="+", choices=PHASES, default=["capture"])
    cli.add_argument("--duration", type=float, default=10)
    cli.add_argument("--rounds", type=int, default=5)
    cli.add_argument("--seed", type=int, default=412430)
    cli.add_argument("--output", type=Path, default=ROOT / "results-cost-intel")
    cli.add_argument("--resume", action="store_true")
    cli.add_argument("--allocation-profile", action="store_true")
    cli.add_argument("--save-wrong-frames", action="store_true")
    cli.add_argument("--pdh", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--gpu-memory", action=argparse.BooleanOptionalAction, default=True)
    cli.add_argument("--fail-fast", action="store_true")
    cli.add_argument("--report-only", action="store_true")
    cli.add_argument("--worker", choices=ARMS, help=argparse.SUPPRESS)
    cli.add_argument("--target-json", help=argparse.SUPPRESS)
    cli.add_argument("--config-json", help=argparse.SUPPRESS)
    args = cli.parse_args()
    if args.worker is not None:
        worker(args.worker, json.loads(args.target_json), json.loads(args.config_json))
        return 0
    if args.report_only:
        write_report(args.output.resolve())
        return 0
    args.arms, args.workloads, args.pipelines, args.rates = map(
        lambda values: list(dict.fromkeys(values)), (args.arms,args.workloads,args.pipelines,args.rates)
    )
    if not args.rounds >= 1 or not 0.1 <= args.duration <= 40 or any(not 1 <= hz <= 100 for hz in args.rates):
        cli.error("Choose 1+ rounds, 0.1–40 seconds, and source update rates from 1–100")
    if args.save_wrong_frames and any(phase == "capture" for phase in args.pipelines):
        cli.error("Wrong-frame evidence requires a downstream task")
    if any(arm in DXCAM_MATCHED_ARMS for arm in args.arms):
        if "shared_tiles" not in args.pipelines:
            cli.error("The opt-in DXcam matched arms require --pipelines shared_tiles")
        if args.save_wrong_frames:
            cli.error("DXcam matched controls retain fixture metrics only. Remove --save-wrong-frames")
    if "idle" in args.workloads and args.duration < 10:
        print("Short idle windows can exercise plumbing. CPU utilization requires a 10-second window.", flush=True)
    return run_matrix(args)


if __name__ == "__main__":
    raise SystemExit(main())
