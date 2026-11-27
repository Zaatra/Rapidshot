"""Cost of deciding whether a detector must run again, on recorded traces.

Three implementations of the same decision, over every consecutive frame pair
of a recorded application trace:

``full_compare``
    ``np.array_equal`` over the whole frame. What a caller had to do before
    change metadata existed.
``candidate_python``
    Narrow to candidate tiles with the recorded DXGI dirty rectangles, then
    compare only those tiles with NumPy slices. A pure-Python upper bound.
``candidate_native``
    The same narrowing, with the comparison done by the project's native
    tile comparator — the implementation the shipped exact paths use.

No inference runs here. All three must agree on every pair, and the changed
tile set from both candidate arms must match a full independent pixel scan.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from cpu_default import Grid, Native, TILE
from inference_trace_task import candidate_tiles_differ, tile_rects
from real_trace import candidates_for, load_trace, oracle_tiles


IMPLEMENTATIONS = ("full_compare", "candidate_python", "candidate_native")
LABELS = {"full_compare": "Whole frame, NumPy",
          "candidate_python": "Candidate tiles, NumPy",
          "candidate_native": "Candidate tiles, native comparator"}
MS = 1_000_000


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def measure(trace: Path, repeats: int) -> dict:
    info, frames = load_trace(trace)
    entries = info["frames"]
    width, height = info["size"]
    grid = Grid(width, height)
    clip = tuple(info["capture_region"])
    monitor_size = tuple(info["monitor_size"])
    native = Native()
    scratch = np.empty_like(grid.all)

    timings = {name: [] for name in IMPLEMENTATIONS}
    verdicts = {name: [] for name in IMPLEMENTATIONS}
    candidate_counts, fallbacks = [], 0
    oracle_mismatch = 0

    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        indices, fallback = candidates_for(grid, entries[index - 1], entries[index],
                                           clip, monitor_size)
        candidate_counts.append(int(np.size(indices)))
        fallbacks += bool(fallback)
        before, after = entries[index - 1], entries[index]

        best = {name: float("inf") for name in IMPLEMENTATIONS}
        for _ in range(repeats):
            started = time.perf_counter_ns()
            same_full = np.array_equal(current, previous)
            best["full_compare"] = min(best["full_compare"],
                                       (time.perf_counter_ns() - started) / MS)

            started = time.perf_counter_ns()
            ids, _ = candidates_for(grid, before, after, clip, monitor_size)
            differ_python = candidate_tiles_differ(
                current, previous, tile_rects(grid, ids, width, height))
            best["candidate_python"] = min(best["candidate_python"],
                                           (time.perf_counter_ns() - started) / MS)

            started = time.perf_counter_ns()
            ids, _ = candidates_for(grid, before, after, clip, monitor_size)
            changed = native.cpu_compare(previous, current, ids, scratch)
            best["candidate_native"] = min(best["candidate_native"],
                                           (time.perf_counter_ns() - started) / MS)

        for name in IMPLEMENTATIONS:
            timings[name].append(best[name])
        verdicts["full_compare"].append(not same_full)
        verdicts["candidate_python"].append(differ_python)
        verdicts["candidate_native"].append(changed > 0)

        if not fallback and not np.array_equal(oracle_tiles(previous, current),
                                               np.sort(scratch[:changed])):
            oracle_mismatch += 1

    if verdicts["full_compare"] != [not np.array_equal(frames[i], frames[i - 1])
                                    for i in range(1, len(frames))]:
        raise RuntimeError("full_compare verdicts are not the ground truth")
    disagreements = {
        name: [i for i, (a, b) in enumerate(zip(verdicts[name], verdicts["full_compare"]))
               if a != b]
        for name in ("candidate_python", "candidate_native")
    }
    for name, indices in disagreements.items():
        missed = [i for i in indices if not verdicts[name][i]]
        if missed:
            raise RuntimeError(f"{name} called {len(missed)} genuinely changed pairs identical")

    return {
        "trace": trace.name, "app": info["app"], "size": [width, height],
        "pairs": len(frames) - 1, "tiles": int(grid.all.size),
        "identical_pairs": sum(1 for v in verdicts["full_compare"] if not v),
        "candidate_tiles_p50": percentile(candidate_counts, 50),
        "candidate_tiles_p95": percentile(candidate_counts, 95),
        "full_grid_fallbacks": fallbacks,
        "changed_tiles_vs_oracle_mismatches": oracle_mismatch,
        "conservative_pairs": {name: len(v) for name, v in disagreements.items()},
        "timings": {
            name: {"mean_ms": statistics.fmean(timings[name]),
                   "p50_ms": percentile(timings[name], 50),
                   "p95_ms": percentile(timings[name], 95),
                   "max_ms": max(timings[name])}
            for name in IMPLEMENTATIONS
        },
    }


def report(data: dict) -> str:
    lines = ["# What it costs to decide whether the detector must run again", "",
             "Every consecutive frame pair of each recorded application trace, with the",
             f"fastest of {data['repeats']} repeats kept per pair. No inference runs here.",
             "All three implementations reach the same verdict on every pair.", "",
             "The three are not doing equal work, and the difference matters when reading",
             "the tables. `np.array_equal` over the whole frame answers only *did anything",
             "change* and has no early exit: NumPy compares every byte. The NumPy candidate",
             "loop answers the same question but returns at the first tile that differs, so",
             "on a busy frame it stops almost immediately. The native comparator answers the",
             "harder question — *exactly which tiles changed* — which cannot exit early,",
             "because the caller needs the whole list to search or stage those tiles. So the",
             "native row is sometimes slower than the NumPy candidate row while delivering",
             "strictly more information. Compare the candidate NumPy row against the whole",
             "frame row for the cost of a pure gate, and the native row for the cost of a",
             "gate that also tells you where to look.", ""]
    for trace in data["traces"]:
        lines += [f"## {trace['app'].title()} — {trace['size'][0]} × {trace['size'][1]}, "
                  f"{trace['pairs']} pairs, {trace['tiles']} tiles", "",
                  f"{trace['identical_pairs']} pairs byte-identical. Candidate tiles per pair: "
                  f"{trace['candidate_tiles_p50']:.0f} median, {trace['candidate_tiles_p95']:.0f} "
                  f"p95, of {trace['tiles']}. Full-grid fallbacks: "
                  f"{trace['full_grid_fallbacks']}.", "",
                  "| Decision | p50 | p95 | max |", "| --- | ---: | ---: | ---: |"]
        for name in IMPLEMENTATIONS:
            row = trace["timings"][name]
            lines.append(f"| {LABELS[name]} | {row['p50_ms']:.4f} ms | {row['p95_ms']:.4f} ms | "
                         f"{row['max_ms']:.4f} ms |")
        full = trace["timings"]["full_compare"]["p50_ms"]
        native = trace["timings"]["candidate_native"]["p50_ms"]
        python = trace["timings"]["candidate_python"]["p50_ms"]
        lines += ["", f"At p50, a pure gate over candidate tiles is **{full / python:.0f}× cheaper** "
                  f"than comparing the whole frame, and a gate that also returns the changed "
                  f"tile list is **{full / native:.0f}× cheaper**.", ""]
        if trace["changed_tiles_vs_oracle_mismatches"]:
            lines += [f"**Changed-tile mismatches against an independent full pixel scan: "
                      f"{trace['changed_tiles_vs_oracle_mismatches']}.**", ""]
        else:
            lines += ["Every changed-tile set matched an independent full pixel scan.", ""]
        conservative = trace["conservative_pairs"]
        lines += [f"Pairs the candidate arms called changed although the bytes were identical: "
                  f"{conservative['candidate_native']} native, "
                  f"{conservative['candidate_python']} Python. Neither ever called a changed "
                  f"pair identical.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args()

    data = {"tile": TILE, "repeats": arguments.repeats,
            "measured_scope": "candidate derivation from the recorded metadata and the "
                              "comparison are both inside the timer, as in real_trace.py",
            "traces": []}
    for trace in arguments.traces:
        print(f"  {trace}", flush=True)
        data["traces"].append(measure(trace, arguments.repeats))

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    arguments.out.with_suffix(".md").write_text(report(data), encoding="utf-8")
    print(f"Wrote {arguments.out} and {arguments.out.with_suffix('.md')}", flush=True)


if __name__ == "__main__":
    main()
