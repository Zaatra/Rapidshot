"""Paired offline 4K, positive YOLOX, and geometry/lease probes."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

import numpy as np

from cost_telemetry import ProcessSampler
from downstream_scene import render, scenes, templates
from downstream_task import TemplateSearch
from idea_live import distribution, save
from rapidshot.changes import ChangedFrame, _ChangeHistory
from rapidshot.memory_pool import NumpyMemoryPool
from whole_frame_task import YoloX, _canvas, load_fixtures, same_detections


def tile_oracle(before, after):
    width = before.shape[1]
    columns = (width + 127) // 128
    y, x = np.where(np.any(before != after, axis=2))
    return tuple(map(int, np.unique((y // 128) * columns + x // 128)))


def capture_telemetry(count, action):
    sampler = ProcessSampler(os.getpid(), pdh=False).start()
    try:
        result = action()
        measured = sampler.finish(count)
    finally:
        sampler.close()
    # Observed process-wide CPU includes the sampler. The same setup is used in both arms.
    return {**result, "cpu": measured["cpu"], "memory": measured["memory"],
            "memory_samples": measured["memory_samples"], "physical_pixel_age_ms": None,
            "gpu_engines": measured["gpu_engines"]}


def new_history(history, pixels, serial, **overrides):
    height, width = pixels.shape[:2]
    region = (0, 0, width, height)
    metadata = {"source": 0, "serial": serial, "region": region, "output_size": (width, height),
                "dirty_rects": [region], "move_rects": [], "accumulated_frames": 1, **overrides}
    return history.update(pixels, **metadata)


def search_4k(number):
    from downstream_scene import _changed_regions

    width, height = 3840, 2160
    steps = scenes(width, height, visible_steps=5, passive_steps=1)["sparse"]
    frames = [render(step, width, height) for step in steps]
    expected = [tuple(sorted((n, *p) for n, p in step.objects.items())) for step in steps[1:]]
    dirty = [_changed_regions(before, after) for before, after in zip(steps, steps[1:])]
    independent = [tile_oracle(before, after) for before, after in zip(frames, frames[1:])]

    def arm(use_tiles):
        search = TemplateSearch(templates(), width, height)
        history = _ChangeHistory()
        search.find_all(frames[0])
        if use_tiles:
            new_history(history, frames[0], 1)

        def measure():
            latency = []
            records = []
            for i, (frame, hints) in enumerate(zip(frames[1:], dirty), 2):
                begin = time.perf_counter_ns()
                changes = new_history(history, frame, i, dirty_rects=hints) if use_tiles else None
                after_gate = time.perf_counter_ns()
                found = search.update(frame, changes.indices) if use_tiles else search.find_all(frame)
                finish = time.perf_counter_ns()
                elapsed = (finish - begin) / 1e6
                latency.append(elapsed)
                records.append({"result_ms": elapsed, "byte_check_ms": (after_gate - begin) / 1e6 if use_tiles else 0,
                                "correct_detections": found.detections == expected[i - 2],
                                "oracle_tiles_match": changes.indices == independent[i - 2] if use_tiles else None,
                                "changed_tiles": len(independent[i - 2]), "search_calls": found.calls,
                                "origins": found.searched_origins, "history_bytes_written": history.last_update_bytes})
            return {"records": records, "result_ms": distribution(latency), "retained_history_bytes": history.size_bytes,
                    "initial_history_bytes": frames[0].nbytes if use_tiles else 0, "gpu_to_cpu_bytes": None}

        return capture_telemetry(len(frames) - 1, measure)

    order = ("full", "tiles") if number == 1 else ("tiles", "full")
    return {"size": [width, height], "pixels_per_frame": width * height, "bytes_per_frame": frames[0].nbytes,
            "live": False, "total_tiles": (width + 127) // 128 * ((height + 127) // 128),
            "arms": {label: arm(label == "tiles") for label in order}}


def positive_model(number, model, frames):
    indices = [0, 0, 1, 1, 1, 2, 2, 0, 0, 2]
    if number == 2:
        indices = [2, 2, 0, 0, 0, 1, 1, 2, 2, 1]

    def arm(gate):
        def measure():
            previous = None
            detections = None
            rows = []
            for index in indices:
                frame = frames[index]
                begin = time.perf_counter_ns()
                skip = gate and previous is not None and np.array_equal(previous, frame)
                after_gate = time.perf_counter_ns()
                if not skip:
                    detections = model.infer(frame)
                finish = time.perf_counter_ns()
                # Replay inputs are owned and immutable. No extra previous-frame copy is needed.
                previous = frame
                rows.append({"fixture": index, "skipped": skip, "detections": detections,
                             "gate_ms": (after_gate - begin) / 1e6 if gate else 0, "result_ms": (finish - begin) / 1e6})
            return {"records": rows, "positive_frames": sum(bool(r["detections"]) for r in rows),
                    "model_calls": sum(not r["skipped"] for r in rows),
                    "result_ms": distribution(r["result_ms"] for r in rows),
                    "model_call_ms": distribution(r["result_ms"] for r in rows if not r["skipped"]),
                    "gate_ms": distribution(r["gate_ms"] for r in rows) if gate else None,
                    "consumer_snapshot_bytes": 0, "retained_previous_frame_reference_bytes": frame.nbytes if gate else 0}

        return capture_telemetry(len(indices), measure)

    order = ("every_frame", "exact_skip") if number == 1 else ("exact_skip", "every_frame")
    results = {label: arm(label == "exact_skip") for label in order}
    comparisons = [same_detections(a["detections"], b["detections"]) for a, b in zip(
        results["every_frame"]["records"], results["exact_skip"]["records"])]
    return {"model": "OpenCV Zoo YOLOX-S FP32 COCO", "images": ["dog", "dog shifted 72 px", "graf1"],
            "prearranged_repeat_frames": 5, "match_frames": sum(comparisons), "arms": results}


def robustness():
    def run_probes():
        result = {}
        bgra = np.full((157, 191, 4), (12, 34, 56, 255), dtype=np.uint8)
        for dpi in (125, 150):
            # Convert odd logical crop origins/ends into desktop pixel coordinates before clipping hints.
            region = tuple(round(value * dpi / 100) for value in (31, 41, 185, 167))
            l, t, r, b = region
            original = np.tile(bgra[0, 0], (b - t, r - l, 1))
            changed = original.copy()
            changed[-1, -1, 3] = 10
            history = _ChangeHistory()
            options = {"region": region, "output_size": (400, 300)}
            new_history(history, original, 1, **options)
            change = new_history(history, changed, 2, dirty_rects=[(r - 1, b - 1, r + 3, b + 3)], **options)
            result[f"dpi_{dpi}_coordinate_fixture"] = {"matches": change.indices == tile_oracle(original, changed),
                                                       "region": region, "rectangles": change.rectangles}

        for degrees in (90, 270):
            logical = np.ascontiguousarray(np.rot90(bgra, degrees // 90))
            changed = logical.copy()
            changed[-1, -1, 0] ^= 1
            history = _ChangeHistory()
            new_history(history, bgra, 1)
            first = new_history(history, logical, 2, rotation=degrees)
            after = new_history(history, changed, 3, rotation=degrees, dirty_rects=[(1, 1, 2, 2)])
            result[f"rotation_{degrees}_metadata_fixture"] = {"matches": first.reason == "reset" and
                after.fallback_reason == "rotation" and after.indices == tile_oracle(logical, changed),
                "first_reason": first.reason, "next_fallback": after.fallback_reason}

        history = _ChangeHistory()
        pool = NumpyMemoryPool(bgra.shape, dtype=np.uint8, num_buffers=2)
        pool.initialize_pool()
        previous = pool.checkout()
        np.copyto(previous.array, bgra)
        initial = new_history(history, previous.array, 1)
        with ChangedFrame(previous, initial) as held:
            different = bgra.copy()
            different[-1, -1, 3] = 0
            reset = new_history(history, different, 2, source=1)
            result["monitor_switch_source_fixture"] = {"matches": reset.reason == "reset" and
                                                       np.array_equal(held.pixels, bgra),
                                                       "held_pixels_bytes": bgra.nbytes, "pool_buffers": 2}

        # A caught acquisition failure clears the baseline; the next valid frame is fully marked.
        latest = bgra.copy()
        latest[-1, -1, 3] = 0
        history.clear()
        resumed = new_history(history, latest, 3, source=2)
        result["generation_loss_history_fixture"] = {"matches": resumed.reason == "reset" and len(resumed.indices) == 4}
        return {"cases": result, "correct": all(row["matches"] for row in result.values())}

    return capture_telemetry(6, run_probes)


def run(output, rounds, seconds):
    output.mkdir(parents=True, exist_ok=True)
    deadline = time.perf_counter() + seconds
    path = output / "surface-ideas.json"
    results = {"rounds": [], "errors": [], "limits": ["Offline 4K pixels, without a live capture surface.",
        "Geometry and generation are injected fixtures, without changes to desktop DPI, rotation, or secure desktop.",
        "YOLOX uses positive COCO photos, without a trained UI-specific model."]}
    model = frames = None
    for number in range(1, rounds + 1):
        row = {"round": number}
        results["rounds"].append(row)
        for name, action in (("geometry", robustness), ("replay_4k", lambda: search_4k(number)),
                             ("yolox", lambda: positive_model(number, model, frames))):
            if time.perf_counter() > deadline - 5:
                results["errors"].append({"round": number, "case": name, "error": "remaining deadline"})
                continue
            try:
                if name == "yolox" and model is None:
                    dog, negative = load_fixtures()
                    frames = (_canvas(dog), _canvas(dog, 72), _canvas(negative))
                    model = YoloX()
                    model.infer(frames[0])
                row[name] = action()
                if name == "geometry" and not row[name]["correct"]:
                    raise AssertionError("A robustness fixture disagreed with its oracle")
                if name == "replay_4k" and not all(r["correct_detections"] and r["oracle_tiles_match"] is not False
                    for a in row[name]["arms"].values() for r in a["records"]):
                    raise AssertionError("4K detections or tile indices disagreed with the oracle")
                if name == "yolox" and (row[name]["match_frames"] != 10 or
                                       not row[name]["arms"]["every_frame"]["positive_frames"]):
                    raise AssertionError("Positive YOLOX replay did not pass the paired checks")
            except Exception as exc:
                results["errors"].append({"round": number, "case": name, "error": repr(exc)})
            save(path, results)
    return bool(results["errors"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=70)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.rounds, args.seconds))
