"""Benchmark CPU dirty-candidate comparison on RapidShot's normal grab path."""

import argparse
import ctypes as ct
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parent
REVISION = "4b20e5c2230322dc68ffcec3934bd4a33286075d"
TILE = 128
WIDTH, HEIGHT = 1920, 1080
WORKLOADS = ("sparse", "full_motion", "excluded_cover", "under_cover")
LABELS = {
    "sparse": "Sparse visible update",
    "full_motion": "Full-screen motion",
    "excluded_cover": "Excluded cover animation",
    "under_cover": "Visible change under excluded cover",
}
ARMS = ("Dirty rectangles", "CPU exact candidates", "CPU exact full-screen",
        "GPU exact hybrid", "GPU exact full-screen")
US = 1e-3
SIZE_MAX = ct.c_size_t(-1).value


class Timing(ct.Structure):
    _fields_ = [("delivery_us", ct.c_double), ("decision_us", ct.c_double),
                ("wall_us", ct.c_double)]


def address(array):
    return array.ctypes.data


class Native:
    def __init__(self):
        dll = ct.CDLL(str(ROOT / "target/release/rapidshot_cpu_default_benchmark.dll"))
        pointer, size, word = ct.c_void_p, ct.c_size_t, ct.c_uint32
        exports = {
            "rs_compare_tiles": ([pointer, pointer, size, size, size, size, size, pointer,
                                  size, pointer, size], size),
            "rs_replay_init": ([word, word, word, pointer, size, word], pointer),
            "rs_replay_reset": ([pointer, size], ct.c_int32),
            "rs_replay_decide": ([pointer, word, size, pointer, size, pointer,
                                  ct.POINTER(size), ct.POINTER(Timing)], ct.c_int32),
            "rs_replay_pixels": ([pointer, pointer, size], ct.c_int32),
            "rs_replay_close": ([pointer], None),
        }
        for name, (argtypes, restype) in exports.items():
            function = getattr(dll, name)
            function.argtypes = argtypes
            function.restype = restype
            setattr(self, name, function)

    def cpu_compare(self, previous, current, candidates, output):
        height, width, channels = current.shape
        for pixels in (previous, current):
            if (pixels.shape != current.shape or channels != 4 or pixels.dtype != np.uint8
                    or pixels.strides[1:] != (4, 1) or pixels.strides[0] < 4 * width):
                raise ValueError("CPU comparison requires equally sized BGRA frames")
        count = self.rs_compare_tiles(address(previous), address(current), width, height,
                                      previous.strides[0], current.strides[0], TILE,
                                      address(candidates), candidates.size, address(output), output.size)
        if count == SIZE_MAX:
            raise RuntimeError("native comparator rejected the frame or candidate IDs")
        return count


class Grid:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.columns = (width + TILE - 1) // TILE
        rows = (height + TILE - 1) // TILE
        self.all = np.arange(self.columns * rows, dtype=np.uint32)

    def masks(self, rects):
        candidates = set()
        for rect in rects:
            if len(rect) != 4:
                return self.all, "invalid rectangle"
            left, top, right, bottom = rect
            if not (0 <= left < right <= self.width and 0 <= top < bottom <= self.height):
                return self.all, "invalid rectangle"
            for y in range(top // TILE, (bottom - 1) // TILE + 1):
                first = y * self.columns + left // TILE
                last = y * self.columns + (right - 1) // TILE
                candidates.update(range(first, last + 1))
        return np.array(sorted(candidates), dtype=np.uint32), None

    def candidate_mask(self, last_seen, frame):
        if last_seen is None:
            return self.all, "initial frame"
        current_stamp = stamp(frame)
        if current_stamp[0] != last_seen[0] or current_stamp[1] != last_seen[1] + 1:
            return self.all, "source or frame gap"
        if frame.accumulated_frames != 1:
            return self.all, "accumulated frames"
        if frame.dirty_rects is None or frame.move_rects is None:
            return self.all, "metadata unavailable"
        if frame.move_rects:
            return self.all, "move rectangles"
        if not frame.dirty_rects:
            return self.all, "present without rectangles"
        return self.masks(frame.dirty_rects)

    def metadata_mask(self, frame):
        if frame.dirty_rects is None or frame.move_rects is None:
            return self.all
        rects = list(frame.dirty_rects)
        for rect in frame.move_rects:
            if len(rect) != 6:
                return self.all
            x, y, left, top, right, bottom = rect
            rects.extend(((left, top, right, bottom), (x, y, x + right - left, y + bottom - top)))
        return self.masks(rects)[0] if rects else self.all

    def oracle(self, previous, current):
        # Independent NumPy reference. Compare every byte, including alpha, and
        # form a tile mask from changed coordinates without using metadata.
        ys, xs = np.nonzero(np.any(previous != current, axis=2))
        return np.unique(ys // TILE * self.columns + xs // TILE).astype(np.uint32)


def stamp(frame):
    return frame.source_id, frame.serial


def elapsed_us(start):
    return (time.perf_counter_ns() - start) * US


class LiveRound:
    def __init__(self, native, grid):
        self.native, self.grid = native, grid
        self.previous_seen = {}
        self.data = defaultdict(list)
        self.output = np.empty_like(grid.all)

    def __call__(self, workload, frame, previous_delivered):
        previous = None if previous_delivered is None else previous_delivered.pixels
        start = time.perf_counter_ns()
        candidates, fallback = self.grid.candidate_mask(self.previous_seen.get(workload), frame)
        native_start = time.perf_counter_ns()
        count = (self.grid.all.size if previous is None else
                 self.native.cpu_compare(previous, frame.pixels, candidates, self.output))
        native_us = elapsed_us(native_start)
        decision_us = elapsed_us(start)
        self.previous_seen[workload] = stamp(frame)
        changed = bool(count)
        expected = previous is None or not np.array_equal(previous, frame.pixels)
        if changed != expected:
            raise RuntimeError(f"{workload} live delivery decision differs from full CPU scan")
        self.data[workload].append({
            "seed_frame": previous is None, "grab_us": frame.grab_us,
            "comparison_call_us": native_us, "decision_us": decision_us,
            "grab_and_decision_us": frame.grab_us + decision_us, "changed": changed,
            "selected_tiles": int(count), "candidates": int(candidates.size),
            "fallback": fallback, "serial": int(frame.serial),
        })
        return changed


def check_fallbacks(native):
    grid = Grid(260, 135)
    blank = np.zeros((135, 260, 4), dtype=np.uint8)
    changed = blank.copy()
    changed[-1, -1, 3] = 1
    near = ((0, 0, 20, 20),)
    variations = {
        "source or frame gap": dict(serial=9),
        "source replacement": dict(source_id=2),
        "metadata unavailable": dict(dirty_rects=None),
        "unknown moves": dict(move_rects=None),
        "move rectangles": dict(move_rects=((0, 0, 10, 10, 20, 20),)),
        "present without rectangles": dict(dirty_rects=()),
        "accumulated frames": dict(accumulated_frames=2),
        "invalid rectangle": dict(dirty_rects=((0, 0, 261, 20),)),
    }
    output = np.empty_like(grid.all)
    for name, updates in variations.items():
        base = dict(source_id=1, serial=8, accumulated_frames=1, dirty_rects=near, move_rects=())
        frame = SimpleNamespace(**(base | updates))
        candidates, reason = grid.candidate_mask((1, 7), frame)
        n = native.cpu_compare(blank, changed, candidates, output)
        expected = grid.oracle(blank, changed)
        if reason is None or not np.array_equal(output[:n], expected) or not n:
            raise RuntimeError(f"full comparison failed after {name}")
    return list(variations)


class Replay:
    def __init__(self, native, grid, frames):
        self.native, self.grid = native, grid
        uploads = np.stack([frame.pixels for frame in frames])
        self.handle = native.rs_replay_init(grid.width, grid.height, TILE, address(uploads),
                                           len(frames), 0)
        if not self.handle:
            raise RuntimeError("native replay initialization failed")
        self.output = np.empty_like(grid.all)
        self.count = ct.c_size_t()
        self.timing = Timing()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.native.rs_replay_close(self.handle)

    def reset(self):
        if self.native.rs_replay_reset(self.handle, 0):
            raise RuntimeError("native replay could not seed the previous frame")

    def run(self, mode, index, candidates):
        status = self.native.rs_replay_decide(self.handle, mode, index, address(candidates),
                                             candidates.size, address(self.output), ct.byref(self.count),
                                             ct.byref(self.timing))
        if status:
            raise RuntimeError(f"native replay arm {ARMS[mode]} failed on frame {index}")
        return self.output[:self.count.value]

    def verify_pixels(self, frame):
        out = np.empty_like(frame.pixels)
        if self.native.rs_replay_pixels(self.handle, address(out), out.nbytes):
            raise RuntimeError("native replay did not return CPU pixels")
        if not np.array_equal(out, frame.pixels):
            raise RuntimeError("native replay produced different CPU pixels")


def pace(start, hz):
    remaining = 1 / hz - (time.perf_counter() - start)
    if remaining > 0:
        time.sleep(remaining)


def replay_workload(native, grid, frames, rounds, reps, rng):
    if len(frames) < 2:
        raise RuntimeError("workload needs a baseline and at least one live frame")
    expected = [grid.oracle(a.pixels, b.pixels) for a, b in zip(frames, frames[1:])]
    metadata = [grid.metadata_mask(frame) for frame in frames[1:]]
    candidates = [grid.candidate_mask(stamp(a), b)[0] for a, b in zip(frames, frames[1:])]
    choices = [metadata, candidates, [grid.all] * len(expected), candidates, [grid.all] * len(expected)]
    correctness = {}
    records = []
    with Replay(native, grid, frames) as engine:
        for mode, name in enumerate(ARMS):
            engine.reset()
            counts = Counter()
            for i, oracle in enumerate(expected):
                output = engine.run(mode, i + 1, choices[mode][i])
                if mode and not np.array_equal(output, oracle):
                    raise RuntimeError(f"{name} differs from CPU oracle on frame {i + 1}")
                engine.verify_pixels(frames[i + 1])
                counts["missed_changes"] += int(not output.size and bool(oracle.size))
                counts["redundant_deliveries"] += int(bool(output.size) and not oracle.size)
                counts["extra_tiles"] += len(np.setdiff1d(output, oracle, assume_unique=True))
                counts["missed_tiles"] += len(np.setdiff1d(oracle, output, assume_unique=True))
                counts["delivered_frames"] += int(bool(output.size))
            correctness[name] = dict(counts)
        for round_index in range(rounds):
            order = list(range(len(ARMS)))
            rng.shuffle(order)
            trials = []
            for mode in order:
                for run_reps, measured in ((len(expected), False), (reps, True)):
                    samples = []
                    for j in range(run_reps):
                        i = j % len(expected)
                        if i == 0:
                            engine.reset()
                        pace_start = time.perf_counter()
                        output = engine.run(mode, i + 1, choices[mode][i])
                        if mode and not np.array_equal(output, expected[i]):
                            raise RuntimeError(f"{ARMS[mode]} timed decision differs from CPU oracle")
                        if measured:
                            samples.append({key: getattr(engine.timing, key) for key, _ in Timing._fields_})
                        pace(pace_start, 60)
                    if measured:
                        trials.append({"arm": ARMS[mode], "samples": samples})
            records.append({"round": round_index + 1, "trials": trials})
    return {"verified_pairs": len(expected), "oracle_changed_frames": sum(bool(ids.size) for ids in expected),
            "correctness": correctness, "rounds": records}


def percentile(samples, fraction):
    values = sorted(samples)
    if not values:
        return None
    index = (len(values) - 1) * fraction
    first = math.floor(index)
    return values[first] + (values[math.ceil(index)] - values[first]) * (index - first)


def summarize(runs, metric):
    p50 = [statistics.median(sample[metric] for sample in samples) for samples in runs if samples]
    p95 = [percentile([sample[metric] for sample in samples], .95) for samples in runs if samples]
    return {"p50_us": statistics.median(p50), "p95_us": statistics.median(p95),
            "round_p50_min_us": min(p50), "round_p50_max_us": max(p50),
            "samples": sum(len(samples) for samples in runs)} if p50 else None


def prior_measurement():
    previous = json.loads((ROOT / "results-followup/results.json").read_text(encoding="utf-8"))
    out = defaultdict(dict)
    prior_keys = {"sparse_live": "sparse", "full_motion_live": "full_motion",
                  "excluded_cover_live": "excluded_cover", "under_excluded_cover_live": "under_cover"}
    for group in previous["groups"]:
        case, endpoint = group["name"].split(" / ")
        if case not in prior_keys:
            continue
        arms = {arm["name"]: arm for arm in group["arms"]}
        entry = out[prior_keys[case]]
        if endpoint == "CPU delivery":
            entry["mapped_cpu_plus_delivery_p50_us"] = arms["mapped CPU exact"]["latency_us"]["p50"]
            entry["mapped_cpu_delivery_increment_p50_us"] = arms["mapped CPU exact"]["incremental_us"]["p50"]
        elif endpoint == "detector":
            entry["gpu_hybrid_detector_only_p50_us"] = arms["exact hybrid with history copy"]["latency_us"]["p50"]
            entry["gpu_full_detector_only_p50_us"] = arms["exact full screen with history copy"]["latency_us"]["p50"]
    return dict(out)


def format_ms(pair):
    return f"{pair['p50_us'] * US:.3f} / {pair['p95_us'] * US:.3f}"


def report(data):
    lines = ["# RapidShot CPU exact default benchmark", "",
             f"GitHub baseline `{data['commit']}`. The live test uses synchronous `grab()` on the Intel DXGI output with 1080p BGRA pixels and 128×128 tiles.", "",
             "A live sample starts when `grab()` is called after issuing a scene update and ends with a change decision on the returned CPU pixels. The prior delivered buffer remains leased until the next changed frame replaces it. No-image calls are recorded separately. Full-frame verification is outside the timers.", "",
             "The replay compares identical input pairs. All arms produce CPU BGRA pixels plus a tile decision. Both GPU arms queue comparison, history maintenance, and their small result readback before the blocking frame readback. Capture acquisition and initial replay seeding are outside the replay timers.", "",
             f"Each method ran in {data['rounds']} shuffled rounds with {data['reps']} measured decisions per replay round. Reported times are medians of the per-round p50 and p95 values in milliseconds.", "",
             "## Live RapidShot path", "",
             "| Workload | Grab + decision p50 / p95 | Decision p50 / p95 | Native compare call p50 / p95 | Suppressed / presentations |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for key in WORKLOADS:
        row = data["live_summary"][key]
        lines.append(f"| {LABELS[key]} | {format_ms(row['grab_and_decision_us'])} | {format_ms(row['decision_us'])} | "
                     f"{format_ms(row['comparison_call_us'])} | {row['suppressed']} / {row['presentations']} |")
    lines += ["", "## Matched replay", "",
              "| Workload | Dirty rectangles | CPU candidates | CPU full | GPU hybrid | GPU full |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in WORKLOADS:
        row = data["replay_summary"][key]
        lines.append("| " + " | ".join([LABELS[key], *(format_ms(row[name]["wall_us"]) for name in ARMS)]) + " |")
    lines += ["", "## Earlier results", "",
              "The previous CPU row includes delivery. The previous GPU rows end at the detector decision. Use the matched replay table above to compare CPU and GPU delivery costs at a common endpoint.", "",
              "| Workload | Earlier mapped CPU + delivery p50 (ms) | Earlier GPU hybrid detector p50 (ms) | Earlier GPU full detector p50 (ms) |",
              "| --- | ---: | ---: | ---: |"]
    for key in WORKLOADS:
        row = data["previous"][key]
        fields = ["mapped_cpu_plus_delivery_p50_us", "gpu_hybrid_detector_only_p50_us", "gpu_full_detector_only_p50_us"]
        lines.append("| " + " | ".join([LABELS[key], *(f"{row[name] * US:.3f}" for name in fields)]) + " |")
    lines += ["", "[Raw results](results.json) include frame counts, correctness checks, trial order, fallback reasons and individual timings.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--reps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "results-cpu-default")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.rounds < 1 or args.reps < 1:
        parser.error("rounds and reps must both be positive")
    native = Native()
    fallback_checks = check_fallbacks(native)
    if args.self_check:
        print(f"CPU exact comparator passed {len(fallback_checks)} fallback scenarios")
        return

    from live_cpu_default import capture_trace

    seed = args.seed if args.seed is not None else random.SystemRandom().getrandbits(64)
    rng = random.Random(seed)
    grid = Grid(WIDTH, HEIGHT)
    traces = None
    live_rounds = []
    for i in range(args.rounds):
        order = list(WORKLOADS)
        rng.shuffle(order)
        print(f"Live round {i + 1}/{args.rounds}: {', '.join(order)}", flush=True)
        live = LiveRound(native, grid)
        captured, observations = capture_trace(on_frame=live, order=order, retain_trace=i == 0)
        if i == 0:
            traces = captured
        live_rounds.append({"round": i + 1, "order": order, "samples": dict(live.data),
                            "observations": observations})
    live_summary = {}
    for key in WORKLOADS:
        runs = [[row for row in trial["samples"].get(key, []) if not row["seed_frame"]]
                for trial in live_rounds]
        if not all(runs):
            raise RuntimeError(f"{key} produced no comparative frames in at least one live round")
        samples = [row for rows in runs for row in rows]
        live_summary[key] = {metric: summarize(runs, metric) for metric in
                             ("grab_us", "decision_us", "grab_and_decision_us", "comparison_call_us")}
        live_summary[key].update(presentations=len(samples), suppressed=sum(not row["changed"] for row in samples),
                                 fallbacks=dict(Counter(row["fallback"] for row in samples if row["fallback"])))

    replay = {}
    replay_summary = {}
    for key in WORKLOADS:
        print(f"Matched replay: {LABELS[key]}", flush=True)
        replay[key] = replay_workload(native, grid, traces.pop(key), args.rounds, args.reps, rng)
        by_arm = replay_summary[key] = {}
        for name in ARMS:
            runs = [next(trial["samples"] for trial in round_['trials'] if trial["arm"] == name)
                    for round_ in replay[key]["rounds"]]
            by_arm[name] = {metric: summarize(runs, metric) for metric, _ in Timing._fields_}

    data = {"commit": REVISION, "seed": seed, "rounds": args.rounds, "reps": args.reps, "tile_size": TILE,
            "width": WIDTH, "height": HEIGHT, "fallback_checks": fallback_checks, "live_summary": live_summary,
            "live_rounds": live_rounds, "replay_summary": replay_summary, "replay": replay,
            "previous": prior_measurement()}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output / "report.md").write_text(report(data), encoding="utf-8")
    print(f"Saved benchmarks in {args.output}", flush=True)


if __name__ == "__main__":
    main()
