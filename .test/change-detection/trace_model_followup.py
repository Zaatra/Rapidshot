"""Compare whole-image YOLOX with exact unchanged gating on recorded app images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import cv2
import numpy as np

from real_trace import FORMAT, rate_indices
from whole_frame_task import MODEL_SHA256, MODEL_URL, YoloX, same_detections


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACES = ROOT / "results-downstream-hp324pe" / "traces"
DEFAULT_OUTPUT = ROOT / "results-timing-followup" / "model"
ARMS = ("always_infer", "skip_exact_unchanged")


def read_frame(archive, shape, index):
    image = archive[f"frame_{index:04d}"]
    if image.shape != shape or image.dtype != np.uint8 or not image.flags.c_contiguous:
        raise ValueError(f"Unexpected BGRA image layout at recorded frame {index}")
    return image


def inspect_trace(archive, info, hz):
    if info.get("format") != FORMAT:
        raise ValueError("Unknown application trace format")
    width, height = info["size"]
    entries = info["frames"]
    times = [entry["captured_ns"] for entry in entries]
    if len(times) < 2 or min(width, height) <= 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("A trace needs at least two BGRA frames with strictly increasing capture timestamps")

    ticks = rate_indices(times, hz).tolist()
    sampled = list(dict.fromkeys(ticks))
    spot = set(sampled)
    shape = (height, width, 4)
    previous_acquired = previous_sample = None
    acquired_duplicates = 0
    sampled_changed = []
    for index in range(len(entries)):
        current = read_frame(archive, shape, index)
        acquired_duplicates += previous_acquired is not None and np.array_equal(previous_acquired, current)
        previous_acquired = current
        if index in spot:
            sampled_changed.append(previous_sample is None or not np.array_equal(previous_sample, current))
            previous_sample = current

    moves = [entry["move_rects"] for entry in entries if entry.get("move_rects") is not None]
    return {
        "app": info["app"],
        "size": [width, height],
        "bounds": info["monitor_bounds"],
        "capture_region": info["capture_region"],
        "captured_images": len(entries),
        "captured_span_seconds": (times[-1] - times[0]) / 1e9,
        "requested_capture_hz": info["record_target_hz"],
        "capture_poll_counts": info["poll_counts"],
        "capture_memory_cap_reached": info["memory_cap_reached"],
        "recorder_dxgi_acquisitions": info.get("dxgi_acquisitions"),
        "recorder_dxgi_moves": info.get("dxgi_moves"),
        "acquired_image_comparisons": len(entries) - 1,
        "acquired_exact_duplicates": acquired_duplicates,
        "move_metadata_images": len(moves),
        "images_with_move_rects": sum(bool(rects) for rects in moves),
        "move_rects": sum(len(rects) for rects in moves),
        "samples_hz": hz,
        "sampling_ticks": len(ticks),
        "repeated_source_ticks_omitted": len(ticks) - len(sampled),
        "sampled_frame_indices": sampled,
        "sampled_times_ns": [times[index] for index in sampled],
        "sampled_changed": sampled_changed,
    }, shape


def run_arm(archive, shape, indices, detector, *, gate):
    rows = []
    previous = None
    detections = ()
    for index in indices:
        current = read_frame(archive, shape, index)
        start = time.perf_counter_ns()
        changed = not gate or previous is None or not np.array_equal(previous, current)
        after_check = time.perf_counter_ns()
        if changed:
            detections = detector.infer(current)
        end = time.perf_counter_ns()
        rows.append({
            "source_frame": index,
            "inferred": changed,
            "comparison_ms": (after_check - start) / 1e6 if gate and previous is not None else 0.0,
            "elapsed_ms": (end - start) / 1e6,
            "detections": detections,
        })
        previous = current
    return rows


def evaluate_trace(path, detector, hz, rounds, rng):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(archive["contract"].tobytes())
        result, shape = inspect_trace(archive, info, hz)
        result.update(source=str(path.resolve()), source_sha256=digest, trials=[])
        indices = result["sampled_frame_indices"]
        if detector is None:
            return result
        if len(indices) < 2:
            raise ValueError(f"Only {len(indices)} images sampled from {path.name} at {hz} Hz")
        detector.infer(read_frame(archive, shape, indices[0]))
        for trial in range(1, rounds + 1):
            order = list(ARMS)
            rng.shuffle(order)
            arms = {}
            for arm in order:
                print(f"{path.stem} | round {trial}/{rounds} | {arm} | {len(indices)} images", flush=True)
                arms[arm] = run_arm(archive, shape, indices, detector, gate=arm == "skip_exact_unchanged")
            always, gated = (arms[arm] for arm in ARMS)
            mismatch = [idx for idx, expected, baseline, measured in
                        zip(indices, result["sampled_changed"], always, gated)
                        if measured["inferred"] != expected
                        or not same_detections(measured["detections"], baseline["detections"])]
            result["trials"].append({"round": trial, "order": order, "arms": arms, "mismatched_source_frames": mismatch})
    return result


def summarize(results):
    lines = [
        "# Whole-image YOLOX on application traces",
        "",
        "Image identity was checked directly for every consecutive pair of recorded BGRA images. "
        "An exact match includes all bytes, including alpha.",
        "",
        "| Trace | Capture span | Recorded images | Byte-identical pairs | Images with DXGI move rectangles | Total move rectangles |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for trace in results["traces"]:
        same = trace["acquired_exact_duplicates"]
        compared = trace["acquired_image_comparisons"]
        name = Path(trace["source"]).stem.replace("|", "\\|")
        lines.append(f"| {name} | {trace['captured_span_seconds']:.2f} s | {trace['captured_images']} | "
                     f"{same}/{compared} ({same / compared:.1%}) | "
                     f"{trace['images_with_move_rects']}/{trace['move_metadata_images']} | {trace['move_rects']} |")

    recorder_counts = [trace for trace in results["traces"] if trace["recorder_dxgi_moves"] is not None]
    if recorder_counts:
        lines.extend([
            "",
            "Recorder-level metadata includes DXGI acquisitions the archive writer did not retain.",
            "",
            "| Trace | All DXGI acquisitions | Archive queue drops | Acquisitions with moves | Acquisitions with moves intersecting crop | Metadata unavailable |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ])
        for trace in recorder_counts:
            name = Path(trace["source"]).stem.replace("|", "\\|")
            moves = trace["recorder_dxgi_moves"]
            lines.append(f"| {name} | {trace['recorder_dxgi_acquisitions']} | "
                         f"{trace['capture_poll_counts'].get('archive_queue_full', 0)} | "
                         f"{moves.get('frames_with_move_rects', 0)} | {moves.get('frames_with_crop_moves', 0)} | "
                         f"{moves.get('metadata_unavailable', 0)} |")

    if results["rounds"]:
        lines.extend([
            "",
            f"At each {results['samples_hz']} Hz replay tick the benchmark selects the most recent acquired image. "
            "Repeated ticks that select the same capture index are omitted before both arms. "
            "Separately acquired byte-identical images remain in the input. Times include the OpenCV CPU model's "
            "preprocessing, inference and decode, and the exact comparison where enabled. "
            "Capture, archive reads and model warm-up are outside timing.",
            "",
            "| Trace | Model inputs across rounds | YOLOX calls avoided | Inputs with detections | Mean baseline / gated time | Mean time saved | Exact compare p50 | Mismatches |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for trace in results["traces"]:
            name = Path(trace["source"]).stem.replace("|", "\\|")
            baseline = [row for trial in trace["trials"] for row in trial["arms"][ARMS[0]]]
            gated = [row for trial in trace["trials"] for row in trial["arms"][ARMS[1]]]
            mean_base = statistics.mean(row["elapsed_ms"] for row in baseline)
            mean_gate = statistics.mean(row["elapsed_ms"] for row in gated)
            comparisons = [row["comparison_ms"] for row in gated if row["comparison_ms"] > 0]
            skipped = sum(not row["inferred"] for row in gated)
            nonempty = sum(bool(row["detections"]) for row in baseline)
            mismatches = sum(len(trial["mismatched_source_frames"]) for trial in trace["trials"])
            lines.append(f"| {name} | {len(baseline)} | {skipped}/{len(baseline)} ({skipped / len(baseline):.1%}) | "
                         f"{nonempty} | {mean_base:.2f} / {mean_gate:.2f} ms | {1 - mean_gate / mean_base:.1%} | "
                         f"{statistics.median(comparisons):.2f} ms | {mismatches} |")
        lines.extend([
            "",
            "The call counts describe the exact captured input sequences at the stated sampling rate. "
            "The recorded application crops and the model's COCO classes limit what these clips test. "
            "See the detection counts for coverage. A live exact change gate can suppress duplicate images before they reach the model.",
            "",
            f"[Raw timings, detections and capture counts](results.json) · [Pinned YOLOX model]({MODEL_URL})",
        ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", nargs="*", type=Path, help="NPZ archives or directories containing them")
    parser.add_argument("--hz", type=int, default=10, help="Replay sampling rate")
    parser.add_argument("--rounds", type=int, default=1, help="Paired, shuffled model benchmark rounds per trace")
    parser.add_argument("--seed", type=int, default=32410)
    parser.add_argument("--metadata-only", action="store_true", help="Count exact pairs and move rectangles without inference")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.hz <= 100 or args.rounds < 1:
        parser.error("choose a replay rate of 1–100 Hz and at least one round")
    paths = list(dict.fromkeys(path.resolve() for source in (args.trace or [DEFAULT_TRACES])
                              for path in (sorted(source.glob("*.npz")) if source.is_dir() else [source])))
    if not paths or any(not path.is_file() or path.suffix.lower() != ".npz" for path in paths):
        parser.error("supply trace archives or directories containing .npz files")
    detector = None if args.metadata_only else YoloX()
    rounds = 0 if args.metadata_only else args.rounds
    rng = random.Random(args.seed)
    results = {
        "model": None if args.metadata_only else "OpenCV Zoo YOLOX-S FP32",
        "model_sha256": None if args.metadata_only else MODEL_SHA256,
        "opencv_version": cv2.__version__,
        "opencv_threads": cv2.getNumThreads() if detector else None,
        "samples_hz": args.hz,
        "rounds": rounds,
        "seed": args.seed,
        "traces": [evaluate_trace(path, detector, args.hz, rounds, rng) for path in paths],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for name, content in (("results.json", json.dumps(results, indent=2) + "\n"), ("summary.md", summarize(results))):
        output = args.output / name
        pending = output.with_suffix(output.suffix + ".tmp")
        pending.write_text(content, encoding="utf-8")
        pending.replace(output)
    print(f"Saved {args.output / 'summary.md'}")
    if any(trial["mismatched_source_frames"] for trace in results["traces"] for trial in trace["trials"]):
        raise SystemExit("Some gated YOLOX results differed from the always-infer reference")


if __name__ == "__main__":
    main()
