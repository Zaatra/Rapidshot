"""Measure same-input template-search savings over controlled changed-tile fractions."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import median
import time

import numpy as np

from downstream_benchmark import write_json
from downstream_scene import render, scenes, templates
from downstream_task import TemplateSearch
from rapidshot.changes import _ChangeHistory


TILE = 128
WIDTH, HEIGHT = 1920, 1080
FRACTIONS = (0, .04, .08, .12, .20, .30, .40, .50, .59, .65, .75, .90, 1)
PATTERNS = ("clustered", "scattered")
DEFAULT_OUT = Path(__file__).resolve().parent / "results-cost-matrix"


def tile_rectangles(width, height):
    return [(x, y, min(width, x + TILE), min(height, y + TILE))
            for y in range(0, height, TILE) for x in range(0, width, TILE)]


def tile_order(pattern, rects, rng):
    order = list(range(len(rects)))
    rng.shuffle(order)
    if pattern == "clustered":
        # Growing the neighborhood from a deterministic random center makes
        # these changes spatially local and varies their relation to the icons.
        x0, y0, _, _ = rects[order[0]]
        order.sort(key=lambda i: abs(rects[i][0] - x0) + abs(rects[i][1] - y0))
    return order


def changed_fixture(before, rects, selected):
    after = before.copy()
    for index in selected:
        left, top, right, bottom = rects[index]
        after[top:bottom, left:right, :3] ^= np.uint8(0x77)
    independently_changed = tuple(i for i, (l, t, r, b) in enumerate(rects)
                                  if not np.array_equal(before[t:b, l:r], after[t:b, l:r]))
    if independently_changed != selected:
        raise AssertionError("Fixture did not produce the exact requested tile count")
    return after


def paired(before, after, selected, rects, arm_order, *, use_hints):
    height, width = before.shape[:2]
    icon_data = templates()
    tracker = _ChangeHistory()
    common = dict(source=1, region=(0, 0, width, height), output_size=(width, height),
                  move_rects=(), accumulated_frames=1)
    tracker.update(before, serial=1, dirty_rects=None, **common)
    task = TemplateSearch(icon_data, width, height, TILE)
    task.find_all(before)
    full = TemplateSearch(icon_data, width, height, TILE)
    pixels = tuple(rects[i] for i in selected)
    dirty_hints = (pixels or tuple(rects[:1])) if use_hints else None
    for arm in arm_order:
        if arm == "full":
            started = time.perf_counter_ns()
            oracle = full.find_all(after)
            full_ns = time.perf_counter_ns() - started
        else:
            started = time.perf_counter_ns()
            changes = tracker.update(after, serial=2, dirty_rects=dirty_hints, **common)
            gate_ns = time.perf_counter_ns() - started
            resumed = time.perf_counter_ns()
            partial = task.update(after, changes.indices)
            tile_ns = time.perf_counter_ns() - resumed
    if tuple(changes.indices) != selected or partial.detections != oracle.detections:
        raise AssertionError("Exact tile or OpenCV output differs from the independent reference")
    history_bytes = sum((r - l) * (b - t) * 4 for l, t, r, b in pixels)
    if tracker.last_update_bytes != history_bytes:
        raise AssertionError("Private history bytes differ from the verified tile bounds")
    return dict(full_search_ns=full_ns, tile_search_ns=tile_ns, gate_ns=gate_ns,
                history_bytes=history_bytes, template_origins=partial.searched_origins,
                detections=list(oracle.detections), fallback_reason=changes.fallback_reason)


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["pattern"], row["requested_fraction"])].append(row)
    result = {}
    for (pattern, fraction), members in sorted(groups.items()):
        elapsed = [r["tile_search_ns"] + r["gate_ns"] for r in members]
        gains = [100 * (1 - net / r["full_search_ns"]) for net, r in zip(elapsed, members)]
        result.setdefault(pattern, []).append(dict(
            requested_fraction=fraction, actual_fraction=members[0]["actual_fraction"],
            samples=len(members), full_ms=median(r["full_search_ns"] for r in members) / 1e6,
            tile_and_gate_ms=median(elapsed) / 1e6, gate_ms=median(r["gate_ns"] for r in members) / 1e6,
            history_bytes=median(r["history_bytes"] for r in members), paired_saving_pct=median(gains),
        ))
    return result


def report(data):
    rows = ["# Changed-tile fraction and template-search crossover", "",
            f"Same-input replay at {WIDTH}×{HEIGHT}. "
            f"Each point has {data['rounds']} paired shuffled trials per spatial layout.",
            ("Dirty hints match the scripted tile bounds. A single unchanged hint models an acquired identical frame at 0%."
             if data["use_hints"] else "Dirty hints are withheld to exercise full comparison."),
            "Each timed gate includes exact byte comparison and private history writes. "
            "The full and regional searches ran on the same final BGRA bytes. "
            "Frame preparation, first-frame history seeding, and initial template-search seeding are outside timing.",
            "", "| Layout | Changed tiles | Full search p50 | Gate + tile search p50 | Gate p50 | History written | Paired median saving |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for pattern, cases in data["summary"].items():
        for case in cases:
            rows.append(f'| {pattern} | {100 * case["actual_fraction"]:.1f}% | {case["full_ms"]:.2f} ms | '
                        f'{case["tile_and_gate_ms"]:.2f} ms | {case["gate_ms"]:.3f} ms | '
                        f'{case["history_bytes"] / 2**20:.2f} MiB | {case["paired_saving_pct"]:.1f}% |')
        positive = []
        for case in cases:
            if case["paired_saving_pct"] <= 0:
                break
            positive.append(case["actual_fraction"])
        rows.extend(["", f"The {pattern} sweep saved time at every tested point through "
                     f"{100 * positive[-1]:.1f}% changed tiles." if positive else
                     f"The first {pattern} point did not save time.", ""])
    rows.extend(["The crossover is bounded by the tested points. Different template footprints and spatial "
                 "patterns can move it. The 30% threshold is in the API spec. [Raw paired samples](crossover.json)", ""])
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--fractions", type=float, nargs="+", default=FRACTIONS)
    parser.add_argument("--seed", type=int, default=240128)
    parser.add_argument("--no-hints", action="store_true")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.rounds <= 0 or any(not math.isfinite(f) or not 0 <= f <= 1 for f in args.fractions):
        parser.error("rounds must be positive and fractions must be within 0..1")
    rng = random.Random(args.seed)
    baseline = render(scenes(visible_steps=1, passive_steps=1)["sparse"][0])
    rects = tile_rectangles(WIDTH, HEIGHT)
    combinations = [(pattern, fraction) for pattern in PATTERNS for fraction in sorted(set(args.fractions))]
    rows = []
    for iteration in range(1, args.rounds + 1):
        order = list(combinations)
        rng.shuffle(order)
        for pattern, fraction in order:
            selected = tuple(sorted(tile_order(pattern, rects, rng)[:math.ceil(fraction * len(rects))]))
            after = changed_fixture(baseline, rects, selected)
            arms = ["full", "tiles"]
            rng.shuffle(arms)
            sample = paired(baseline, after, selected, rects, arms, use_hints=not args.no_hints)
            rows.append(dict(round=iteration, pattern=pattern, requested_fraction=fraction,
                             actual_fraction=len(selected) / len(rects), changed_indices=selected,
                             arm_order=arms, **sample))
        print(f"Finished crossover round {iteration}/{args.rounds}", flush=True)
    data = dict(seed=args.seed, rounds=args.rounds, use_hints=not args.no_hints, rows=rows, summary=summary(rows))
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "crossover.json", data)
    (args.out / "crossover.md").write_text(report(data), encoding="utf8")
    print(f"Saved {len(rows)} independently checked pairs to {args.out}")


if __name__ == "__main__":
    main()
