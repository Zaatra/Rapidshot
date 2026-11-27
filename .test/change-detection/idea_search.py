"""Two-round offline comparisons for exact-tile downstream search ideas."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
import statistics
import time
import zlib

import numpy as np

from cost_telemetry import _ProcessCounters
from downstream_benchmark import write_json
from downstream_scene import render, scenes, templates
from downstream_task import TemplateSearch
from rapidshot.changes import _ChangeHistory
from tile_crossover import FRACTIONS, PATTERNS, changed_fixture, tile_order, tile_rectangles


WIDTH, HEIGHT, TILE = 1920, 1080, 128
DEFAULT_OUT = Path(__file__).resolve().parent / "results-idea-search"
MIB = 2**20
COUNTERS = None


def area(rect):
    left, top, right, bottom = rect
    return (right - left) * (bottom - top)


def union_rectangles(rectangles):
    """Merge overlaps and neighbors into disjoint half-open rectangles."""
    rectangles = tuple(rectangles)
    boundaries = sorted({y for _, top, _, bottom in rectangles for y in (top, bottom)})
    result = []
    continuing = {}
    for top, bottom in zip(boundaries, boundaries[1:]):
        spans = []
        for left, right in sorted((l, r) for l, t, r, b in rectangles if t <= top < b):
            if spans and left <= spans[-1][1]:
                spans[-1] = (spans[-1][0], max(right, spans[-1][1]))
            else:
                spans.append((left, right))
        prior = continuing
        continuing = {}
        for left, right in spans:
            key = (left, right)
            if key in prior:
                index = prior[key]
                result[index] = (left, result[index][1], right, bottom)
            else:
                index = len(result)
                result.append((left, top, right, bottom))
            continuing[key] = index
    return tuple(result)


def search_regions(rectangles, size, margin=0):
    """Prototype helper for a radius or individual left/top/right/bottom margins."""
    width, height = size
    if isinstance(margin, int):
        margin = (margin,) * 4
    ml, mt, mr, mb = margin
    if min(margin) < 0:
        raise ValueError("Search margins must be nonnegative")
    expanded = []
    for left, top, right, bottom in rectangles:
        left, right = max(0, left - ml), min(width, right + mr)
        top, bottom = max(0, top - mt), min(height, bottom + mb)
        if left < right and top < bottom:
            expanded.append((left, top, right, bottom))
    return union_rectangles(expanded)


@dataclass(frozen=True)
class Cost:
    per_origin: float
    per_call: float

    def estimate(self, rectangles):
        return sum(self.per_origin * area(rect) + self.per_call for rect in rectangles)


def elapsed(call):
    start = time.perf_counter_ns()
    result = call()
    return result, (time.perf_counter_ns() - start) / 1e6


def arm_start():
    memory = COUNTERS.memory()
    cycles = COUNTERS.cpu().cycles
    return time.perf_counter_ns(), cycles, memory


def arm_end(start):
    begin, cycles, previous = start
    ms = (time.perf_counter_ns() - begin) / 1e6
    elapsed_cycles = COUNTERS.cpu().cycles - cycles
    current = COUNTERS.memory()
    return ms, {
        "process_cycles": elapsed_cycles,
        "private_bytes_before": previous["private_bytes"],
        "private_bytes_after": current["private_bytes"],
        "working_set_bytes_before": previous["working_set_bytes"],
        "working_set_bytes_after": current["working_set_bytes"],
    }


def measure_arm(call):
    start = arm_start()
    result = call()
    ms, counters = arm_end(start)
    return result, ms, counters


def calibrate(task, frame):
    """Fit a small fixed range of OpenCV areas before timing any fixture pair."""
    template = next(iter(task._templates.values()))
    full = (0, 0, template.origin_width, template.origin_height)
    middle = (0, 0, 640, 360)
    small = (0, 0, 128, 128)
    samples = {}
    for label, rect in (("full", full), ("middle", middle), ("small", small)):
        values = [elapsed(lambda: task._search(frame, template, rect, {}))[1] for _ in range(3)]
        samples[label] = statistics.median(values)
    slope = (samples["full"] - samples["middle"]) / (area(full) - area(middle))
    origin = max(slope, samples["full"] / (2 * area(full)))
    overhead = max(samples["small"] - origin * area(small), 0.0)
    return Cost(origin, overhead), samples


def _origin_rectangles(pixel_rects, template):
    th, tw = template.gray.shape
    return search_regions(
        pixel_rects,
        (template.origin_width, template.origin_height),
        margin=(tw - 1, th - 1, 0, 0),
    )


class RegionalMatcher:
    """Incremental OpenCV wrapper with precise invalidated origins."""

    def __init__(self, task, cost, guard=0.90):
        self.task = task
        self.cost = cost
        self.guard = guard

    def update(self, frame, pixel_rects, force_full=False):
        task = self.task
        if force_full or task._hits is None:
            return task.find_all(frame)
        task._check_frame(frame)
        if not pixel_rects:
            return task.update(frame, ())
        plans = {name: _origin_rectangles(pixel_rects, template)
                 for name, template in task._templates.items()}
        projected = sum(self.cost.estimate(rects) for rects in plans.values())
        full = self.cost.per_origin * task._full_work + self.cost.per_call * len(plans)
        if projected >= full * self.guard:
            return task.find_all(frame)
        gray_crops, hits = {}, {}
        searched = calls = 0
        for name, template in task._templates.items():
            rects = plans[name]
            retained = {}
            for block, origins in task._hits[name].items():
                kept = [(x, y) for x, y in origins
                        if not any(l <= x < r and t <= y < b for l, t, r, b in rects)]
                if kept:
                    retained[block] = kept
            for rect in rects:
                for block, witnesses in task._search(frame, template, rect, gray_crops).items():
                    retained.setdefault(block, []).extend(witnesses)
            hits[name] = retained
            searched += sum(map(area, rects))
            calls += len(rects)
        return task._commit(hits, searched, calls)


def changed_boxes(previous, current, indices, rectangles):
    """Bounds of all changed BGRA positions within each identified tile."""
    result = []
    for index in indices:
        left, top, right, bottom = rectangles[index]
        a, b = previous[top:bottom, left:right], current[top:bottom, left:right]
        y, x = np.where(np.any(a != b, axis=2))
        if not len(x):
            raise AssertionError("An indicated tile had no unequal pixels")
        result.append((left + int(x.min()), top + int(y.min()),
                       left + int(x.max()) + 1, top + int(y.max()) + 1))
    return tuple(result)


def diff_indices(previous, current, rects):
    """Independent full BGRA oracle using reductions over the whole image."""
    changed = np.any(previous != current, axis=2)
    rows = np.logical_or.reduceat(changed, np.arange(0, changed.shape[0], TILE), axis=0)
    tiled = np.logical_or.reduceat(rows, np.arange(0, changed.shape[1], TILE), axis=1)
    result = tuple(map(int, np.flatnonzero(tiled)))
    if tiled.size != len(rects):
        raise AssertionError("Fixture and full-pixel oracle disagree on dimensions")
    return result


def track(history, pixels, serial, hints):
    return history.update(
        pixels, source=1, serial=serial, region=(0, 0, WIDTH, HEIGHT),
        output_size=(WIDTH, HEIGHT), move_rects=(), accumulated_frames=1, dirty_rects=hints,
    )


class Fixture:
    def __init__(self, baseline, cost=None, icons=None):
        self.before = baseline
        self.rects = tile_rectangles(WIDTH, HEIGHT)
        self.icons = templates() if icons is None else icons
        self.reference = TemplateSearch(self.icons, WIDTH, HEIGHT, TILE)
        self.reference.find_all(baseline)
        # Search updates construct a new map and leave the seed maps/lists alone.
        self.hits, self.detections = self.reference._hits, self.reference._detections
        self.cost = cost

    def seeded_search(self):
        task = TemplateSearch(self.icons, WIDTH, HEIGHT, TILE)
        task._hits, task._detections = self.hits, self.detections
        return task

    def seeded_history(self, history_type=_ChangeHistory):
        history = history_type()
        track(history, self.before, 1, None)
        return history

    def compare(self, after, expected, order, include_tight=False):
        if diff_indices(self.before, after, self.rects) != expected:
            raise AssertionError("Fixture tile labels differ from the independent BGRA oracle")
        hints = tuple(self.rects[index] for index in expected) or (self.rects[0],)
        timings, detections = {}, {}
        for mode in order:
            if mode == "full":
                outcome, duration, counters = measure_arm(lambda: self.reference.find_all(after))
                timings[mode] = dict(ms=duration, search_ms=duration, gate_ms=0.0,
                                     calls=outcome.calls, origins=outcome.searched_origins,
                                     full=outcome.full, **counters)
                detections[mode] = outcome.detections
                continue
            history = self.seeded_history()
            task = self.seeded_search()
            matcher = RegionalMatcher(task, self.cost)
            start = arm_start()
            change, gate_ms = elapsed(lambda: track(history, after, 2, hints))
            box_ms = 0.0
            if mode == "tight":
                boxes, box_ms = elapsed(lambda: changed_boxes(self.before, after, change.indices, self.rects))
            else:
                boxes = change.rectangles
            if mode == "legacy":
                outcome, search_ms = elapsed(lambda: task.update(after, change.indices))
            else:
                outcome, search_ms = elapsed(lambda: matcher.update(after, boxes))
            total_ms, counters = arm_end(start)
            if change.indices != expected:
                raise AssertionError("Production exact tracker disagrees with full BGRA")
            timings[mode] = dict(ms=total_ms, gate_ms=gate_ms, search_ms=search_ms, box_ms=box_ms,
                                 calls=outcome.calls, origins=outcome.searched_origins, full=outcome.full,
                                 pixel_area=sum(map(area, boxes)), history_bytes=history.last_update_bytes,
                                 **counters)
            detections[mode] = outcome.detections
        if any(result != detections["full"] for result in detections.values()):
            raise AssertionError(f"An incremental search differs from full OpenCV: {detections}")
        if not include_tight and "tight" in timings:
            raise AssertionError("A tight-box test is out of scope for this fixture")
        return dict(arms=timings, detections=len(detections["full"]), changed_count=len(expected))


class DenseHistory(_ChangeHistory):
    """Opt-in consumer experiment that drops its history after a dense frame."""

    def __init__(self):
        super().__init__()
        self.dense = False

    def _patch_changed(self, pixels, candidates):
        changed = []
        for index in candidates:
            left, top, right, bottom = self._rectangles[index]
            if not np.array_equal(self._pixels[top:bottom, left:right], pixels[top:bottom, left:right]):
                changed.append(index)
        self.dense = len(changed) > 0.25 * len(self._rectangles)
        if not self.dense:
            for index in changed:
                left, top, right, bottom = self._rectangles[index]
                current = pixels[top:bottom, left:right]
                np.copyto(self._pixels[top:bottom, left:right], current)
                self.last_update_bytes += current.nbytes
        yield from changed


def dense_sequence(fixture, indices, sequence, modes):
    frames = []
    previous = fixture.before
    for label in sequence:
        candidates = indices if label == "dense" else (0, len(fixture.rects) - 1)
        current = changed_fixture(previous, fixture.rects, candidates)
        if diff_indices(previous, current, fixture.rects) != candidates:
            raise AssertionError("Dense fixture differs from independent BGRA")
        frames.append((label, current, candidates))
        previous = current
    results = {}
    references = {}
    for mode in modes:
        history = fixture.seeded_history(DenseHistory if mode == "skip" else _ChangeHistory)
        matcher = RegionalMatcher(fixture.seeded_search(), fixture.cost)
        result = []
        for serial, (label, frame, real) in enumerate(frames, 2):
            reseed = history.size_bytes == 0
            marks = tuple(fixture.rects[i] for i in real)
            if mode == "skip":
                history.dense = False
            start = arm_start()
            if mode == "full":
                outcome = fixture.reference.find_all(frame)
                chosen = "full"
                used = 0
            else:
                change = track(history, frame, serial, marks)
                if not reseed and change.indices != real:
                    raise AssertionError("Dense policy changed map differs from the independent oracle")
                if reseed and change.reason != "reset":
                    raise AssertionError("Dense history must be reseeded before trusting metadata")
                if mode == "skip" and history.dense:
                    chosen = "dense"
                    history.clear()
                elif reseed:
                    chosen = "reseed"
                else:
                    chosen = "adaptive"
                used = history.last_update_bytes
                outcome = matcher.update(frame, change.rectangles, force_full=chosen != "adaptive")
            ms, counters = arm_end(start)
            result.append(dict(label=label, chosen=chosen, history_bytes=used, ms=ms,
                               calls=outcome.calls, origins=outcome.searched_origins, **counters))
            references.setdefault(serial, {})[mode] = outcome.detections
        results[mode] = result
    for outputs in references.values():
        if any(value != outputs["full"] for value in outputs.values()):
            raise AssertionError("Dense fallback/reseed changed template detections")
    return results


def priority_fixture(icon, scene):
    before = np.full((HEIGHT, WIDTH, 4), 72, dtype=np.uint8)
    before[..., 3] = 255
    after = before.copy()
    # Changed background is in left tiles, with one arrangement forcing the
    # actual icon behind large unrelated changes.
    for x, y in ((8, 8), (142, 16), (263, 29), (18, 148), (264, 140)):
        side = 92 if scene == "distracted" else 14
        after[y:y + side, x:x + side, :3] = 143
    positions = {
        "absent": (),
        "single": ((803, 488),),
        "boundary": ((757, 501),),
        "multiple": ((803, 488), (1418, 704)),
        "distracted": ((1418, 704),),
    }[scene]
    th, tw = icon.shape
    for x, y in positions:
        after[y:y + th, x:x + tw, :3] = icon[..., None]
    return before, after, positions


def priority_search(before, after, task, changed, rectangles, *, rank):
    """Find any byte-verified hit over identical ranked or row-major blocks."""
    template = task._templates["network"]
    scores = defaultdict(int)
    for index in changed:
        l, t, r, b = rectangles[index]
        magnitude = int(np.count_nonzero(before[t:b, l:r] != after[t:b, l:r])) if rank else 0
        for block in task._origin_blocks(template, (index,)):
            scores[block] += magnitude
    ranked = sorted(scores, key=lambda block: (-scores[block], block[1], block[0]))
    hit, calls = None, 0
    for bx, by in ranked:
        origins = (bx * TILE, by * TILE, min((bx + 1) * TILE, template.origin_width),
                   min((by + 1) * TILE, template.origin_height))
        proposal = task._search(after, template, origins, {})
        calls += 1
        if proposal:
            hit = min(loc for matches in proposal.values() for loc in matches)
            break
    return hit, tuple(ranked), calls


def check_priority(scene, order):
    icon = templates()["network"]
    before, after, expected = priority_fixture(icon, scene)
    task = TemplateSearch({"network": icon}, WIDTH, HEIGHT, TILE)
    rects = tile_rectangles(WIDTH, HEIGHT)
    changed = diff_indices(before, after, rects)
    outcomes, measures = {}, {}
    for mode in order:
        if mode == "full":
            result, ms, counters = measure_arm(lambda: task.find_all(after))
        else:
            result, ms, counters = measure_arm(
                lambda: priority_search(before, after, task, changed, rects, rank=mode == "priority"))
        outcomes[mode] = result
        measures[mode] = dict(ms=ms, **counters)
    hits = {(x, y) for _, x, y in outcomes["full"].detections}
    if hits != set(expected):
        raise AssertionError("Priority fixture differs from whole-image template detections")
    if set(outcomes["priority"][1]) != set(outcomes["row_major"][1]):
        raise AssertionError("Any-hit controls must have identical candidate sets")
    for mode in ("priority", "row_major"):
        hit, ranks, calls = outcomes[mode]
        if (hit is None) != (not hits) or (hit is not None and hit not in hits):
            raise AssertionError("An any-hit result differs from whole-image presence")
        measures[mode].update(
            calls=calls, candidate_blocks=len(ranks),
            hit_ranks=sorted(1 + ranks.index((x // TILE, y // TILE)) for x, y in hits),
            matched_origin=hit,
        )
    return dict(case=scene, arms=measures, detections=len(hits), order=order)


def tight_cases():
    steps = scenes(visible_steps=5, passive_steps=1)["sparse"]
    pairs = [(f"sparse_{left.stamp:#x}_{right.stamp:#x}", render(left), render(right))
             for left, right in ((steps[0], steps[1]), (steps[2], steps[3]), (steps[4], steps[5]))]
    baseline = render(steps[0])
    alpha = baseline.copy()
    alpha[-1, -1, 3] ^= np.uint8(0xFF)
    pairs.append(("alpha_only_edge", baseline, alpha))
    return pairs


def _task(pixels, config):
    # A deterministic tile-local proxy with a different output for the test
    # states even when their lookup fingerprints are forced to collide.
    luminance = pixels[:, :, 1]
    return (int(luminance.sum()) + config, int((luminance >= 128).sum()),
            int(np.count_nonzero(pixels[:, 1:] != pixels[:, :-1])))


class TileCache:
    """Prototype bounded cache whose hash never substitutes for byte equality."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.identity = None
        self.entries = defaultdict(deque)
        self.evicted_hashes = deque(maxlen=64)
        self.counts = Counter()

    def evaluate(self, pixels, source, tile, config, *, forced_hash=None):
        if not self.capacity:
            self.counts["task_calls"] += 1
            return _task(pixels, config)
        identity = (source, config, pixels.shape, pixels.dtype)
        if self.identity != identity:
            if self.identity is not None:
                self.counts["invalidations"] += 1
            self.entries.clear()
            self.evicted_hashes.clear()
            self.identity = identity
        # The immutable byte snapshot is also the verifier and cache payload.
        payload = pixels.tobytes()
        digest = zlib.crc32(payload) if forced_hash is None else forced_hash
        local = self.entries[tile]
        for index, (stored_hash, stored_bytes, outcome) in enumerate(local):
            if digest == stored_hash:
                if payload == stored_bytes:
                    local.rotate(-index)
                    hit = local.popleft()
                    local.rotate(index)
                    local.appendleft(hit)
                    self.counts["hits"] += 1
                    return outcome
                self.counts["rejected_collisions"] += 1
        if len(local) == self.capacity:
            evicted = local.pop()[0]
            self.evicted_hashes.append((tile, evicted))
            self.counts["evictions"] += 1
        if (tile, digest) in self.evicted_hashes:
            self.counts["recently_evicted_hash_on_miss"] += 1
        outcome = _task(pixels, config)
        self.counts["task_calls"] += 1
        local.appendleft((digest, payload, outcome))
        return outcome

    @property
    def retained_bytes(self):
        return sum(len(payload) for entries in self.entries.values() for _, payload, _ in entries)


def cache_case(scene, order):
    rng = np.random.default_rng(811)
    background = rng.integers(10, 110, size=(128, 128, 4), dtype=np.uint8)
    states = []
    for state in range(5):
        variant = background.copy()
        variant[19:54, 29:29 + state * 3, :3] = 215
        states.append(variant)
    schedule = []
    for i in range(40):
        source = int(scene == "reset" and i >= 20)
        period = 5 if scene == "churn" else 2
        forced = 97 if scene == "collision" else None
        schedule.append((states[i % period], source, 0, source, forced))
    results, outputs = {}, {}
    for capacity in order:
        cache = TileCache(capacity)
        run, total_ms, counters = measure_arm(
            lambda: [cache.evaluate(pixels, source, tile, config, forced_hash=forced)
                     for pixels, source, tile, config, forced in schedule])
        outputs[capacity] = run
        results[str(capacity)] = dict(ms=total_ms, retained_pixel_bytes=cache.retained_bytes,
                                      retained_evicted_hashes=len(cache.evicted_hashes),
                                      **counters, **cache.counts)
    if outputs[0] != outputs[4]:
        raise AssertionError("A cached result differed from uncached analysis")
    if scene == "collision" and not results["4"].get("rejected_collisions", 0):
        raise AssertionError("Forced same-hash frames did not exercise byte verification")
    if scene == "churn" and not results["4"].get("recently_evicted_hash_on_miss", 0):
        raise AssertionError("Cache churn did not exercise an evicted-hash miss")
    if scene == "reset" and results["4"].get("invalidations") != 1:
        raise AssertionError("Cache failed to invalidate source/task configuration")
    return dict(case=scene, order=order, results=results)


def helper_demo():
    """Check disjoint output, clipping, and complete radius coverage."""
    size = (12, 11)
    edits = ((1, 2, 4, 6), (4, 3, 6, 6), (8, 10, 12, 11))
    regions = search_regions(edits, size, margin=2)
    expected = np.zeros((size[1], size[0]), dtype=np.uint8)
    for l, t, r, b in edits:
        expected[max(t - 2, 0):min(b + 2, size[1]), max(l - 2, 0):min(r + 2, size[0])] = 1
    actual = np.zeros_like(expected)
    for l, t, r, b in regions:
        actual[t:b, l:r] += 1
    if not np.array_equal(expected, actual):
        raise AssertionError("Prototype search regions lose pixels or overlap")
    snippets = {
        "full": "matcher = TemplateSearch(templates, width, height)\nhits = matcher.find_all(pixels).detections",
        "incremental": "matcher = RegionalMatcher(TemplateSearch(templates, width, height), cost)\nhits = matcher.update(pixels, changes.rectangles).detections",
    }
    return dict(clipped_disjoint_regions=regions, expanded_pixels=int(actual.sum()),
                excerpts=snippets, example_lines={key: len(code.splitlines()) for key, code in snippets.items()})


def shuffled(rng, modes):
    result = list(modes)
    rng.shuffle(result)
    return result


def run(args, data):
    start = time.perf_counter()
    rng = random.Random(args.seed)
    base = render(scenes(visible_steps=1, passive_steps=1)["sparse"][0])
    fixture = Fixture(base)
    fixture.cost, measurements = calibrate(fixture.reference, base)
    data["calibration"] = dict(per_origin=fixture.cost.per_origin, per_call=fixture.cost.per_call,
                               samples_ms=measurements, full_fallback_guard=0.90)
    data["helpers"] = helper_demo()
    fractions = (0.08, 0.59, 1.0) if args.smoke else FRACTIONS
    sparse_pairs = tight_cases()
    for iteration in range(1, args.rounds + 1):
        jobs = [("adaptive", (pattern, fraction)) for pattern in PATTERNS for fraction in fractions]
        jobs += [("tight", pair) for pair in sparse_pairs]
        jobs += [("dense", pattern) for pattern in PATTERNS]
        jobs += [("priority", case) for case in ("single", "multiple", "boundary", "absent", "distracted")]
        jobs += [("cache", case) for case in ("blink", "churn", "collision", "reset")]
        rng.shuffle(jobs)
        for section, detail in jobs:
            if time.perf_counter() - start >= args.seconds:
                data["status"] = "time_budget_reached"
                return
            if section == "adaptive":
                pattern, fraction = detail
                chosen = tuple(sorted(tile_order(pattern, fixture.rects, rng)[:math.ceil(fraction * len(fixture.rects))]))
                after = changed_fixture(fixture.before, fixture.rects, chosen)
                order = shuffled(rng, ("adaptive", "legacy", "full"))
                row = fixture.compare(after, chosen, order)
                row.update(pattern=pattern, requested_fraction=fraction,
                           actual_fraction=len(chosen) / len(fixture.rects), order=order)
            elif section == "tight":
                name, before, after = detail
                tight = Fixture(before, fixture.cost)
                exact = diff_indices(before, after, tight.rects)
                order = shuffled(rng, ("full", "legacy", "adaptive", "tight"))
                row = tight.compare(after, exact, order, include_tight=True)
                row.update(case=name, order=order)
            elif section == "dense":
                chosen = tuple(sorted(tile_order(detail, fixture.rects, rng)[:math.ceil(0.59 * len(fixture.rects))]))
                order = shuffled(rng, ("skip", "adaptive", "full"))
                sequence = ("dense", "sparse", "dense", "sparse")
                row = dict(pattern=detail, order=order, frames=dense_sequence(fixture, chosen, sequence, order))
            elif section == "priority":
                row = check_priority(detail, shuffled(rng, ("full", "priority", "row_major")))
            else:
                row = cache_case(detail, shuffled(rng, (0, 4)))
            data[section].append(dict(round=iteration, **row))
        data["completed_rounds"] = iteration
        print(f"Search ideas round {iteration}/{args.rounds} completed", flush=True)
    data["status"] = "pass"


def grouped(rows, *keys):
    groups = defaultdict(list)
    for sample in rows:
        groups[tuple(sample[key] for key in keys)].append(sample)
    return sorted(groups.items())


def saving(candidate_ms, full_ms):
    return 100 * (1 - candidate_ms / full_ms)


def fmt(value):
    return f"{value:.2f}"


def report(data):
    med = statistics.median
    lines = ["# Offline search and history experiments", "",
             f"{data['completed_rounds']}/{data['requested_rounds']} shuffled rounds. "
             f"Status: {data['status']}. Same 1920×1080 BGRA inputs and full OpenCV detection oracle.",
             "", "## Adaptive search", "",
             "The region planner unions directional, template-size margins in origin coordinates and estimates "
             "area and per-call cost from separate calibration. The gate and history writes are timed.",
             "",
             "| Layout | Changed tiles | Legacy gate + search saving | Adaptive gate + search saving | Full fallback count |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for (pattern, fraction), samples in grouped(data["adaptive"], "pattern", "actual_fraction"):
        timings = [row["arms"] for row in samples]
        legacy = med(saving(r["legacy"]["ms"], r["full"]["ms"]) for r in timings)
        adaptive = med(saving(r["adaptive"]["ms"], r["full"]["ms"]) for r in timings)
        used_full = sum(r["adaptive"]["full"] for r in timings)
        lines.append(f"| {pattern} | {fraction:.1%} | {legacy:.1f}% | {adaptive:.1f}% | {used_full}/{len(samples)} |")
    if data["adaptive"]:
        bad = [row for row in data["adaptive"] if saving(
            row["arms"]["adaptive"]["ms"], row["arms"]["full"]["ms"]) < -2]
        worst = min(saving(row["arms"]["adaptive"]["ms"], row["arms"]["full"]["ms"])
                    for row in data["adaptive"])
        lines += ["", f"The 2% maximum-loss criterion failed in {len(bad)}/{len(data['adaptive'])} individual "
                       f"pairs. The worst paired saving was {worst:.1f}%."]
    lines += ["", "## Dense history skip", "",
              "A confirmed change to more than 25% of tiles drops the baseline and uses full search. The next "
              "frame reseeds history and also uses full search, even if only two tiles changed.",
              "", "| Layout | Keep history p50 per 4-frame sequence | Skip and reseed p50 | Full search p50 | "
              "History kept / skip writes per sequence |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for (pattern,), samples in grouped(data["dense"], "pattern"):
        totals = {arm: med(sum(f["ms"] for f in s["frames"][arm]) for s in samples)
                  for arm in ("adaptive", "skip", "full")}
        writes = {arm: med(sum(f["history_bytes"] for f in s["frames"][arm]) for s in samples) / MIB
                  for arm in ("adaptive", "skip")}
        lines.append(f"| {pattern} | {fmt(totals['adaptive'])} ms | {fmt(totals['skip'])} ms | "
                     f"{fmt(totals['full'])} ms | {fmt(writes['adaptive'])} / {fmt(writes['skip'])} MiB |")
    lines += ["", "## Priority any-hit", "",
              "Inputs begin without the target. Priority may return any exact hit, so correctness compares "
              "presence and verifies the returned origin belongs to the whole-image result. Row-major any-hit "
              "uses the same candidate blocks. Full search enumerates all hits. Ranking is timed.",
              "", "| Fixture | Targets | Ranked first hit | Ranked / row-major calls | Ranked / row-major / full p50 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for (name,), samples in grouped(data["priority"], "case"):
        a = [row["arms"] for row in samples]
        ranks = [r["priority"]["hit_ranks"][0] for r in a if r["priority"]["hit_ranks"]]
        calls = " / ".join(f"{med(r[m]['calls'] for r in a):g}" for m in ("priority", "row_major"))
        times = " / ".join(fmt(med(r[m]["ms"] for r in a)) for m in ("priority", "row_major", "full"))
        lines.append(f"| {name} | {samples[0]['detections']} | {med(ranks) if ranks else 'absent'} | "
                     f"{calls} | {times} ms |")
    lines += ["", "## Tight all-byte boxes and helpers", "",
              "Tight rectangle time includes per-tile BGRA bounds calculation and template-size expansion.",
              "", "| Fixture | Changed tile-box area / tight area | Legacy origins | Tight origins | "
              "Tight gate + search saving |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for (name,), samples in grouped(data["tight"], "case"):
        a = [s["arms"] for s in samples]
        ratio = med(r["legacy"]["pixel_area"] / r["tight"]["pixel_area"] for r in a)
        origins = [med(r[m]["origins"] for r in a) for m in ("legacy", "tight")]
        gain = med(saving(r["tight"]["ms"], r["full"]["ms"]) for r in a)
        lines.append(f"| {name} | {ratio:.1f}× | {origins[0]:,.0f} | {origins[1]:,.0f} | {gain:.1f}% |")
    if "helpers" in data:
        counts = data["helpers"]["example_lines"]
        lines += ["", f"With inputs prepared, the full matcher and helper examples each use "
                       f"{counts['full']} and {counts['incremental']} lines respectively. "
                       "The examples exclude capture, calibration, imports and fixture setup."]
    lines += ["", "## Byte-verified cache", "",
              "Every result matched an uncached tile-local proxy. Resident payloads verify the hash before "
              "reuse. Recently evicted hashes record churn and confer no cache hit.",
              "", "| Fixture | Task calls kept out of 40 | Collision rejections | Evictions | Retained payload |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for (name,), samples in grouped(data["cache"], "case"):
        a = [r["results"]["4"] for r in samples]
        kept = med(r["hits"] if "hits" in r else 0 for r in a)
        collisions = med(r.get("rejected_collisions", 0) for r in a)
        evictions = med(r.get("evictions", 0) for r in a)
        retained = med(r["retained_pixel_bytes"] for r in a) / 1024
        lines.append(f"| {name} | {kept:g} | {collisions:g} | {evictions:g} | {retained:g} KiB |")
    lines += ["", "Each arm also records process cycles and Windows private bytes and working set "
                  "before and after. These snapshots share a process and cannot isolate peaks or retained memory "
                  "to an arm. Two offline rounds are descriptive. The helper, dense policy and cache are benchmark "
                  "prototypes. [Raw timings and examples](results.json)", ""]
    return "\n".join(lines)


def main():
    global COUNTERS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=105)
    parser.add_argument("--seed", type=int, default=251126)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.rounds <= 2 or not 0 < args.seconds <= 600:
        parser.error("require 1–2 rounds and a positive budget up to 600 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    data = dict(requested_rounds=args.rounds, completed_rounds=0, status="incomplete",
                adaptive=[], dense=[], priority=[], tight=[], cache=[], seed=args.seed)
    start = time.perf_counter()
    COUNTERS = _ProcessCounters(os.getpid())
    try:
        run(args, data)
    except BaseException as error:
        data["status"] = "failed"
        data["failure"] = str(error)
        raise
    finally:
        COUNTERS.close()
        data["elapsed_seconds"] = time.perf_counter() - start
        write_json(args.output / "results.json", data)
        (args.output / "report.md").write_text(report(data), encoding="utf8")
    if data["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
