"""Trace the production history allocation separately from live timing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cost_telemetry import AllocationProfiler
from downstream_benchmark import write_json
from rapidshot.changes import _ChangeHistory


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results-cost-matrix" / "history-allocation.json"


def profile(before, after, enabled):
    height, width, _ = before.shape
    context = dict(source=1, region=(0, 0, width, height), output_size=(width, height),
                   move_rects=(), accumulated_frames=1)
    profiler = AllocationProfiler()
    profiler.start()
    if enabled:
        tracker = _ChangeHistory()
        tracker.update(before, serial=1, dirty_rects=None, **context)
        seed_bytes = tracker.last_update_bytes
        tiles = tracker.update(after, serial=2, dirty_rects=((0, 0, 1, 1),), **context)
        if tiles.indices != (0,):
            raise AssertionError("One alpha change must select exactly the first tile")
        result = dict(history_bytes=tracker.size_bytes, seed_bytes=seed_bytes,
                      active_history_write_bytes=tracker.last_update_bytes)
    else:
        result = dict(history_bytes=0, seed_bytes=0, active_history_write_bytes=0)
    return result | {"allocations": profiler.finish()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    before = np.zeros((1080, 1920, 4), dtype=np.uint8)
    after = before.copy()
    after[0, 0, 3] = 255
    data = {"size": [1920, 1080], "control": profile(before, after, False),
            "private_history": profile(before, after, True)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, data)
    print(args.output)


if __name__ == "__main__":
    main()
