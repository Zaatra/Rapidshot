"""Replay recorded application traces through the verified YOLOX CPU detector.

Three arms decide, per captured frame, whether the detector must run again:

``always``
    Infer on every delivered frame. The reference for both correctness and cost.
``full_compare``
    Byte-compare the whole frame against the previous one and reuse the previous
    detections when they are identical. This is what a caller had to do before
    change metadata was available.
``change_metadata``
    Narrow to candidate tiles with the recorded DXGI dirty rectangles, then
    byte-compare only those tiles. This is the decision ``grab_tiles`` makes.

Frame decompression is outside every timer. Only the decision, the inference
when one is needed, and the detection decode are measured.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np

from cpu_default import Grid, TILE
from real_trace import candidates_for, load_trace
from whole_frame_task import YoloX, same_detections


ARMS = ("always", "full_compare", "change_metadata")
LABELS = {"always": "Always infer",
          "full_compare": "Skip identical, full-frame compare",
          "change_metadata": "Skip identical, candidate tiles only"}
MS = 1_000_000


@dataclass
class FrameOutcome:
    index: int
    inferred: bool
    elapsed_ms: float
    decision_ms: float
    candidates: int
    fallback: str | None
    detections: tuple


def tile_rects(grid: Grid, indices, width: int, height: int):
    """Pixel rectangles for candidate tile indices, clipped to the frame."""
    rects = []
    for index in indices:
        row, column = divmod(int(index), grid.columns)
        left, top = column * TILE, row * TILE
        rects.append((left, top, min(left + TILE, width), min(top + TILE, height)))
    return rects


def candidate_tiles_differ(current, previous, rects) -> bool:
    """Compare only the candidate tiles, stopping at the first real difference."""
    for left, top, right, bottom in rects:
        if not np.array_equal(current[top:bottom, left:right], previous[top:bottom, left:right]):
            return True
    return False


def run_arm(arm: str, detector: YoloX, frames, entries, grid: Grid, clip, monitor_size):
    """Walk the trace once under one arm's decision rule."""
    outcomes = []
    previous_frame = None
    previous_entry = None
    detections = ()
    for index, (frame, entry) in enumerate(zip(frames, entries)):
        started = time.perf_counter_ns()
        identical = False
        fallback = None
        candidates = grid.all.size
        if previous_frame is not None:
            if arm == "full_compare":
                identical = np.array_equal(frame, previous_frame)
            elif arm == "change_metadata":
                indices, fallback = candidates_for(grid, previous_entry, entry, clip, monitor_size)
                candidates = int(np.size(indices))
                rects = tile_rects(grid, indices, grid.width, grid.height)
                identical = not candidate_tiles_differ(frame, previous_frame, rects)
        decision_ms = (time.perf_counter_ns() - started) / MS
        if not identical:
            detections = detector.infer(frame)
        elapsed_ms = (time.perf_counter_ns() - started) / MS
        outcomes.append(FrameOutcome(index, not identical, elapsed_ms, decision_ms,
                                     candidates, fallback, detections))
        previous_frame, previous_entry = frame, entry
    return outcomes


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize(outcomes):
    every = [o.elapsed_ms for o in outcomes]
    skipped = [o.elapsed_ms for o in outcomes if not o.inferred]
    inferred = [o.elapsed_ms for o in outcomes if o.inferred]
    decisions = [o.decision_ms for o in outcomes[1:]]
    return {
        "frames": len(outcomes),
        "detector_calls": sum(o.inferred for o in outcomes),
        "mean_ms": statistics.fmean(every),
        "p50_ms": percentile(every, 50),
        "p95_ms": percentile(every, 95),
        "total_s": sum(every) / 1000,
        "skipped_p50_ms": percentile(skipped, 50),
        "skipped_p95_ms": percentile(skipped, 95),
        "inferred_p50_ms": percentile(inferred, 50),
        "decision_p50_ms": percentile(decisions, 50),
        "decision_p95_ms": percentile(decisions, 95),
        "fallbacks": sum(1 for o in outcomes if o.fallback),
    }


def replay(trace: Path, info, frames, detector: YoloX, rounds: int, seed: int) -> dict:
    entries = info["frames"][:len(frames)]
    width, height = info["size"]
    grid = Grid(width, height)
    clip = tuple(info["capture_region"])
    monitor_size = tuple(info["monitor_size"])
    identical_pairs = sum(1 for a, b in zip(frames, frames[1:]) if np.array_equal(a, b))

    rng = random.Random(seed)
    rounds_out = []
    for round_index in range(rounds):
        order = list(ARMS)
        rng.shuffle(order)
        arms = {}
        for arm in order:
            print(f"    {trace.stem}: round {round_index + 1}, {arm}", flush=True)
            arms[arm] = run_arm(arm, detector, frames, entries, grid, clip, monitor_size)
        reference = arms["always"]
        for arm in ("full_compare", "change_metadata"):
            for gated, base in zip(arms[arm], reference):
                if not same_detections(gated.detections, base.detections):
                    raise RuntimeError(f"{arm} detections differ from always-infer at frame {base.index}")
        truth = [True] + [not np.array_equal(frames[i], frames[i - 1]) for i in range(1, len(frames))]
        if [o.inferred for o in arms["full_compare"]] != truth:
            raise RuntimeError("full_compare did not rerun exactly on changed frames")
        conservative = [o.index for o, want in zip(arms["change_metadata"], truth)
                        if o.inferred != want]
        if any(not arms["change_metadata"][i].inferred for i in conservative):
            raise RuntimeError("change_metadata skipped a frame whose bytes changed")
        rounds_out.append({
            "round": round_index + 1, "order": order,
            "conservative_frames": conservative,
            "arms": {arm: summarize(outcomes) for arm, outcomes in arms.items()},
            "candidate_tiles_p50": percentile(
                [o.candidates for o in arms["change_metadata"][1:]], 50),
            "candidate_tiles_p95": percentile(
                [o.candidates for o in arms["change_metadata"][1:]], 95),
            "fallback_reasons": sorted({o.fallback for o in arms["change_metadata"] if o.fallback}),
        })
    return {
        "trace": trace.name, "app": info["app"], "size": [width, height],
        "frames": len(frames), "tiles": int(grid.all.size),
        "identical_consecutive": identical_pairs,
        "identical_fraction": identical_pairs / (len(frames) - 1),
        "record_target_hz": info["record_target_hz"],
        "rounds": rounds_out,
    }


def report(data: dict) -> str:
    lines = ["# YOLOX inference over recorded application traces", "",
             "Each recorded capture is replayed through the OpenCV Zoo YOLOX-S FP32 CPU",
             "detector three times: inferring on every delivered frame, skipping frames",
             "proven identical by a full-frame byte compare, and skipping frames proven",
             "identical by comparing only the candidate tiles that DXGI dirty rectangles",
             "point at. Frame decompression sits outside every timer.", "",
             f"OpenCV {data['opencv']}, {data['opencv_threads']} thread, OpenCL "
             f"{'on' if data['opencl'] else 'off'}. Model "
             f"{data['model_bytes']:,} bytes, SHA-256 `{data['sha256']}`.", ""]
    for trace in data["traces"]:
        first = trace["rounds"][0]
        lines += [f"## {trace['app'].title()} — {trace['size'][0]} × {trace['size'][1]}, "
                  f"{trace['frames']} frames, {trace['tiles']} tiles", "",
                  f"{trace['identical_consecutive']} of {trace['frames'] - 1} consecutive frames were "
                  f"byte-identical ({100 * trace['identical_fraction']:.1f}%).", "",
                  "| Arm | Detector calls | Mean per frame | p95 | Total | Decision p50 |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for arm in ARMS:
            row = first["arms"][arm]
            decision = "—" if arm == "always" else f"{row['decision_p50_ms']:.3f} ms"
            lines.append(f"| {LABELS[arm]} | {row['detector_calls']} | {row['mean_ms']:.2f} ms | "
                         f"{row['p95_ms']:.2f} ms | {row['total_s']:.3f} s | {decision} |")
        base = first["arms"]["always"]
        saved = {arm: 100 * (base["mean_ms"] - first["arms"][arm]["mean_ms"]) / base["mean_ms"]
                 for arm in ("full_compare", "change_metadata")}
        lines += ["", f"Detector time saved: **{saved['full_compare']:.2f}%** by full-frame compare, "
                  f"**{saved['change_metadata']:.2f}%** by candidate tiles. Candidate tiles per "
                  f"frame: {first['candidate_tiles_p50']:.0f} median, "
                  f"{first['candidate_tiles_p95']:.0f} p95, of {trace['tiles']}.", ""]
        if first["fallback_reasons"]:
            lines += [f"Full-grid fallbacks: {first['arms']['change_metadata']['fallbacks']} frames "
                      f"({', '.join(first['fallback_reasons'])}).", ""]
        if first["conservative_frames"]:
            lines += [f"Frames the metadata arm inferred although the bytes were unchanged: "
                      f"{len(first['conservative_frames'])}. Never the reverse.", ""]
        else:
            lines += ["The metadata arm reached the same decision as the full compare on every "
                      "frame.", ""]
    lines += ["Every gated arm produced detections identical to the always-infer reference: same "
              "classes, boxes within 0.01 px, scores within 0.00001.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Smoke-test cap on frames per trace")
    parser.add_argument("--seed", type=int, default=412_430)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    import cv2

    detector = YoloX()
    data = {"model": "OpenCV Zoo YOLOX-S FP32", "model_bytes": 35_858_002,
            "sha256": "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063",
            "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
            "opencl": cv2.ocl.useOpenCL(), "tile": TILE, "seed": arguments.seed,
            "rounds": arguments.rounds,
            "measured_scope": "decision, inference when needed, and detection decode",
            "traces": []}
    for trace in arguments.traces:
        print(f"  loading {trace}", flush=True)
        info, frames = load_trace(trace)
        if arguments.limit:
            frames = frames[:arguments.limit]
        detector.infer(frames[0])  # Warm the graph at this trace's resolution.
        data["traces"].append(
            replay(trace, info, frames, detector, arguments.rounds, arguments.seed))
        del frames

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    arguments.out.with_suffix(".md").write_text(report(data), encoding="utf-8")
    print(f"Wrote {arguments.out} and {arguments.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
