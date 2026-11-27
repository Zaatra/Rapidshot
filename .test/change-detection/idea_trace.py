"""Bounded offline checks for captured change metadata and scroll reuse."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
from statistics import median
import time
from zlib import crc32

import numpy as np

from cpu_default import Grid, Native, TILE
from cost_telemetry import _ProcessCounters, _cpu_delta, _memory_summary
from downstream_scene import render, scenes
from downstream_task import TemplateSearch
from real_trace import FORMAT, candidates_for, template_anchors


ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "results-timing-followup" / "traces"
RISK_DIR = ROOT / "results-cost-matrix" / "mixed-frame-100"
APPS = ("browser", "spreadsheet", "word")


def milliseconds(start):
    return (time.perf_counter_ns() - start) / 1e6


class TraceTelemetry:
    """Sample coarse memory boundaries and CPU cycles at selected stages."""

    def __init__(self):
        self.counters = _ProcessCounters(os.getpid())
        self.windows = {}

    def cycles(self):
        return self.counters.cpu().cycles

    def window(self, name, action, *args):
        memory = [self.counters.memory()]
        start = self.counters.cpu()
        value = action(*args)
        finish = self.counters.cpu()
        memory.append(self.counters.memory())
        summary = {key: value for key, value in _memory_summary(memory).items() if not key.startswith("steady_")}
        self.windows[name] = {"cpu": _cpu_delta(start, finish, None),
                              "memory": summary, "memory_samples": memory}
        return value


def write_checkpoint(output, value):
    pending = output / "trace-ideas.json.tmp"
    pending.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(output / "trace-ideas.json")


def risk_audit():
    proof = json.loads((RISK_DIR / "evidence-audit.json").read_text(encoding="utf-8"))
    manifest = json.loads((RISK_DIR / "manifest.json").read_text(encoding="utf-8"))
    refresh_period_ns = 1e9 / manifest["monitor"]["physical_refresh_hz"]
    truth = set()
    for row in proof["records"]:
        assert row["exact_previous_and_current_mix"] and row["full_search_agrees"]
        match = Path(row["metadata"])
        truth.add((match.parent.name, *map(int, match.stem.split("-"))))
    groups = defaultdict(list)
    for path in sorted((RISK_DIR / "trials").glob("*.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        submitted = {row["stamp"]: row["send_end_ns"] for row in run["submitted"]}
        for row in run["worker"]["observations"]:
            correct = row.get("correct_fixture")
            if not row["new_native_timestamp"] or row["outcome"] != "delivered" or correct is None:
                continue
            stamp = row["stamp"]
            key = (path.stem, stamp, row["capture_returned_ns"])
            if not correct:
                truth.remove(key)
            known = stamp in submitted and stamp - 1 in submitted
            interval = submitted[stamp] - submitted[stamp - 1] >= refresh_period_ns if known else None
            accumulated = row["accumulated_frames"] > 1 if row["accumulated_frames"] is not None else None
            decisions = {"accumulated": accumulated,
                         "fallback_reported": row["fallback_reason"] is not None,
                         "fixture_interval_at_least_period": interval,
                         "accumulated_or_fixture_interval": accumulated or interval}
            for group in ("all", run["arm"], f"archive_round_{run['round']}"):
                groups[group].append((not correct, decisions))
    assert not truth, "Some proven mixed frames were not in the comparable delivered dataset"
    result = {}
    for group, values in groups.items():
        counts = Counter(mixed for mixed, _ in values)
        rules = {}
        for rule in values[0][1]:
            hits = Counter((mixed, decisions[rule]) for mixed, decisions in values)
            rules[rule] = {"true_positive": hits[True, True], "false_positive": hits[False, True],
                           "false_negative": hits[True, False], "true_negative": hits[False, False],
                           "missing_mixed": hits[True, None], "missing_correct": hits[False, None]}
        result[group] = {"verified_mixed": counts[True], "correct_detections": counts[False], "rules": rules}
    return result


def changes_to_tiles(mask):
    rows = np.logical_or.reduceat(mask, np.arange(0, mask.shape[0], TILE), axis=0)
    tiles = np.logical_or.reduceat(rows, np.arange(0, mask.shape[1], TILE), axis=1)
    return np.flatnonzero(tiles).astype(np.uint32)


def packed(pixels):
    return pixels.view(np.uint32).squeeze(axis=2)


def phase_sample(index, phase):
    return (index + phase) % 8 == 0


def dirty_audit(native, grid, before, after, info, index, oracle, phase, telemetry=None):
    frames = info["frames"]
    candidates, fallback = candidates_for(grid, frames[index - 1], frames[index],
                                          info["capture_region"], info["monitor_size"])
    output = np.empty_like(grid.all)
    n = native.cpu_compare(before, after, candidates, output)
    missed = np.setdiff1d(oracle, output[:n], assume_unique=True).tolist()
    if np.setdiff1d(output[:n], oracle, assume_unique=True).size:
        raise AssertionError("The native check reported change on identical BGRA tile bytes")
    full_ms = full_cycles = None
    if fallback is None and phase_sample(index, phase):
        cycles = telemetry.cycles() if telemetry is not None else None
        start = time.perf_counter_ns()
        count = native.cpu_compare(before, after, grid.all, output)
        full_ms = milliseconds(start)
        if cycles is not None:
            full_cycles = telemetry.cycles() - cycles
        if not np.array_equal(output[:count], oracle):
            raise AssertionError("Independent and native full tile checks disagree")
    return {"frame": index, "present_qpc": frames[index]["last_present_time_qpc"],
            "fallback": fallback, "candidates": int(candidates.size), "changed": int(oracle.size),
            "missed_tiles": missed, "sampled_full_native_ms": full_ms,
            "sampled_full_native_process_cycles": full_cycles}


def omission_proof(native):
    size = TILE * 2
    grid = Grid(size, size)
    old = np.zeros((size, size, 4), np.uint8)
    new = old.copy()
    new[TILE, TILE, 3] = 7
    def entry(serial):
        return {"source_id": 1, "generation": 0, "serial": serial, "accumulated_frames": 1,
                "dirty_rects": [(1, 1, 2, 2)], "move_rects": [], "last_present_time_qpc": serial}
    info = {"frames": [entry(1), entry(2)], "capture_region": (0, 0, size, size),
            "monitor_size": (size, size)}
    oracle = changes_to_tiles(packed(old) != packed(new))
    result = dirty_audit(native, grid, old, new, info, 1, oracle, 7)
    if result["missed_tiles"] != [3] or result["fallback"] is not None:
        raise AssertionError("Dirty audit failed to catch an alpha-only change outside the supplied rectangle")
    return result


def row_votes(before, after):
    height, width = before.shape[:2]
    left, right = width // 4, width * 3 // 4
    previous = defaultdict(list)
    for y, row in enumerate(before[:, left:right]):
        previous[crc32(row)].append(y)
    votes = Counter()
    for y, row in enumerate(after[:, left:right]):
        matching = previous.get(crc32(row), ())
        if len(matching) == 1 and 0 < abs(y - matching[0]) <= min(300, height // 2):
            votes[y - matching[0]] += 1
    return votes.most_common(3)


def scroll_shift(before, after, changed):
    start = time.perf_counter_ns()
    votes = row_votes(before, after)
    proposal_time = milliseconds(start)
    old, new = packed(before), packed(after)
    height = old.shape[0]
    best = None
    best_explained = 0
    changed_count = np.count_nonzero(changed)
    for offset, votes_for_offset in votes:
        if votes_for_offset < 6:
            continue
        top, bottom = max(0, offset), min(height, height + offset)
        equal = np.equal(old[top - offset:bottom - offset], new[top:bottom])
        explained = int(np.count_nonzero(equal & changed[top:bottom]))
        if explained > best_explained:
            best = offset, votes_for_offset, equal, top, bottom
            best_explained = explained
    row = {"row_fingerprint_ms": proposal_time, "offset": None,
           "explained_changed_fraction": float(best_explained / changed_count)}
    if best is None or row["explained_changed_fraction"] < .80:
        row["shift_analysis_ms"] = milliseconds(start)
        return row, None
    offset, votes_for_offset, equal, top, bottom = best
    residual = np.ones_like(changed)
    residual[top:bottom] = ~equal
    residual_tiles = changes_to_tiles(residual)
    row.update(offset=offset, row_votes=votes_for_offset, residual_tiles=int(residual_tiles.size),
               exact_changed_overlap=bool(np.all(equal[changed[top:bottom]])), shift_analysis_ms=milliseconds(start))
    return row, residual_tiles


def compare_searches(templates, before, after, changed, offset, residual_tiles, telemetry=None):
    height, width = before.shape[:2]
    baseline = TemplateSearch(templates, width, height, TILE)
    baseline_result = baseline.find_all(before)
    cycles = telemetry.cycles() if telemetry is not None else None
    start = time.perf_counter_ns()
    plain = baseline.update(after, changed)
    tiled_ms = milliseconds(start)
    tile_cycles = telemetry.cycles() - cycles if cycles is not None else None
    reference = TemplateSearch(templates, width, height, TILE)
    cycles = telemetry.cycles() if telemetry is not None else None
    start = time.perf_counter_ns()
    whole = reference.find_all(after)
    full_ms = milliseconds(start)
    full_cycles = telemetry.cycles() - cycles if cycles is not None else None
    if plain.detections != whole.detections:
        raise AssertionError("Standard tile search differs from full reference on a scroll clip")
    row = {"full_ms": full_ms, "tile_ms": tiled_ms, "full_origins": whole.searched_origins,
           "tile_origins": plain.searched_origins, "full_detections": len(whole.detections),
           "full_process_cycles": full_cycles, "tile_process_cycles": tile_cycles}
    if offset is None:
        return row | {"shifted_ms": tiled_ms, "shifted_origins": plain.searched_origins,
                      "shifted_process_cycles": tile_cycles,
                      "verified_reused_detections": 0}
    proposal = TemplateSearch(templates, width, height, TILE)
    cycles = telemetry.cycles() if telemetry is not None else None
    start = time.perf_counter_ns()
    translated = {key: {} for key in templates}
    residual_indices = tuple(int(index) for index in residual_tiles)
    invalid = {key: proposal._origin_blocks(spec, residual_indices)
               for key, spec in proposal._templates.items()}
    retained = 0
    for key, x, y in baseline_result.detections:
        new_y = y + offset
        th, tw = templates[key].shape
        if not (0 <= new_y <= height - th):
            continue
        block = x // TILE, new_y // TILE
        if block in invalid[key]:
            continue
        if not np.array_equal(before[y:y + th, x:x + tw], after[new_y:new_y + th, x:x + tw]):
            raise AssertionError("An unverified template footprint entered translated reuse")
        translated[key].setdefault(block, []).append((x, new_y))
        retained += 1
    proposal._commit(translated, 0, 0)
    shifted = proposal.update(after, residual_indices)
    shifted_ms = milliseconds(start)
    shifted_cycles = telemetry.cycles() - cycles if cycles is not None else None
    if shifted.detections != whole.detections:
        raise AssertionError("Translated reuse differs from full reference on a scroll clip")
    return row | {"shifted_ms": shifted_ms, "shifted_origins": shifted.searched_origins,
                  "shifted_process_cycles": shifted_cycles,
                  "verified_reused_detections": retained}


def tile_storage(width, height, touched):
    columns = (width + TILE - 1) // TILE
    total = width * height * 4
    tiles = ((height + TILE - 1) // TILE) * columns
    bytes_touched = 0
    rounded = 0
    for index in touched:
        y, x = divmod(index, columns)
        logical = min(TILE, height - y * TILE) * min(TILE, width - x * TILE) * 4
        bytes_touched += logical
        rounded += (logical + 4095) // 4096 * 4096
    return {"tiles": tiles, "ever_changed_tiles": len(touched), "baseline_bgra_bytes": total,
            "history_without_backing_seed_bytes": total, "optimistic_overlay_logical_bytes": bytes_touched,
            "optimistic_overlay_4k_page_bytes": rounded, "baseline_plus_overlay_bytes": total + rounded}


def inspect_trace(path, native, phase, deadline, telemetry=None):
    with np.load(path, allow_pickle=False) as archive:
        info = json.loads(archive["contract"].tobytes())
        if info["format"] != FORMAT or info["memory_cap_reached"]:
            raise ValueError(f"Unexpected or limited trace in {path}")
        width, height = info["size"]
        count = len(info["frames"])
        grid = Grid(width, height)
        first = archive["frame_0000"]
        templates, _ = template_anchors(first, archive[f"frame_{count - 1:04d}"])
        old = first
        touched = set()
        audit = []
        scroll = []
        sampled_segments = set()
        verified_shifts = 0
        for index in range(1, count):
            if time.monotonic() >= deadline:
                break
            current = archive[f"frame_{index:04d}"]
            changed = packed(old) != packed(current)
            oracle = changes_to_tiles(changed)
            touched.update(oracle.tolist())
            audit.append(dirty_audit(native, grid, old, current, info, index, oracle, phase, telemetry))
            if oracle.size:
                segment = min(index * 6 // count, 5)
                needs_segment = segment not in sampled_segments and (index + phase) % 2 == 0
                collect_cycles = telemetry is not None and (needs_segment or phase_sample(index, phase))
                cycles = telemetry.cycles() if collect_cycles else None
                row, residual = scroll_shift(old, current, changed)
                row["shift_analysis_process_cycles"] = telemetry.cycles() - cycles if cycles is not None else None
                needs_shift = row["offset"] is not None and verified_shifts < 3
                row.update(frame=index, changed_tiles=int(oracle.size))
                if needs_segment or needs_shift:
                    sampled_segments.add(segment)
                    verified_shifts += row["offset"] is not None
                    row.update(compare_searches(templates, old, current, oracle, row["offset"], residual, telemetry))
                scroll.append(row)
            old = current
    return {"archive": str(path.relative_to(ROOT)), "record_target_hz": info["record_target_hz"],
            "archived_frames": count, "audited_pairs": len(audit), "all_adjacent_pairs": len(audit) == count - 1,
            "move_rect_frames": sum(bool(row["move_rects"]) for row in info["frames"]),
            "dirty_audit": audit, "scroll": scroll, "storage": tile_storage(width, height, touched)}


def inspect_fixture():
    width, height = 1920, 1080
    sequence = scenes(width, height, 24, 12)["sparse"]
    touched = set()
    old = render(sequence[0])
    for step in sequence[1:]:
        new = render(step)
        touched.update(changes_to_tiles(packed(old) != packed(new)).tolist())
        old = new
    return {"frames": len(sequence), "storage": tile_storage(width, height, touched)}


def report(output, results):
    lines = ["# Offline metadata and scroll checks", "",
             "The retrospective risk dataset contains the saved mixed frames and correct-detection controls. The scroll tests use consecutive acquired BGRA frames. No replay tick is inserted.", "",
             "| Risk rule | Mixed frames flagged | Correct-detection controls flagged |", "| --- | ---: | ---: |"]
    all_risk = results["risk"]["all"]
    for name, score in all_risk["rules"].items():
        lines.append(f"| {name} | {score['true_positive']}/{all_risk['verified_mixed'] - score['missing_mixed']} | "
                     f"{score['false_positive']}/{all_risk['correct_detections'] - score['missing_correct']} |")
    lines += ["", "Fixture submission timing is instrumentation from the source, outside ordinary capture metadata. The raw tables give per-arm and archived-round counts.", "",
              "| Sweep round | App | Audited pairs | Guided pairs with missed tiles | Sampled full scans | Full scan median | Verified shift proposals / changed pairs | Row fingerprint median | Full OpenCV equality checks for accepted shifts | Median shifted path vs ordinary tile search | Ever-changed tiles |", "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def fmt(values):
        return f"{median(values):.3f} ms" if values else "—"
    for block in results["rounds"]:
        for app, data in block["traces"].items():
            guided = [row for row in data["dirty_audit"] if row["fallback"] is None]
            full = [row["sampled_full_native_ms"] for row in guided if row["sampled_full_native_ms"] is not None]
            scroll = data["scroll"]
            proposals = [row for row in scroll if row["offset"] is not None]
            verified = [row for row in proposals if "shifted_ms" in row]
            shifted = [r["shift_analysis_ms"] + r["shifted_ms"] for r in verified]
            ordinary = [r["tile_ms"] for r in verified]
            storage = data["storage"]
            lines.append(f"| {block['round']} | {app} | {data['audited_pairs']} | "
                         f"{sum(bool(row['missed_tiles']) for row in guided)}/{len(guided)} | "
                         f"{len(full)} | {fmt(full)} | {len(proposals)}/{len(scroll)} | "
                         f"{fmt([row['row_fingerprint_ms'] for row in scroll])} | {len(verified)} | "
                         f"{fmt(shifted)} / {fmt(ordinary)} | {storage['ever_changed_tiles']}/{storage['tiles']} |")
    fixture = results["sparse_fixture"]["storage"]
    lines += ["", "A proposed shift must explain at least 80% of changed pixels after byte verification. "
              "All exposed and unequal pixels are searched with template margins. Full OpenCV equivalence is checked on a stratified sample and at least the first three accepted proposals per app and round when available.", "",
              "The audit found the injected alpha-only change outside the reported rectangle. Real hint misses and sampled scan times are in the raw file. Every archived comparison also has the independent full NumPy tile oracle.", "",
              f"The 1080p sparse fixture touched {fixture['ever_changed_tiles']}/{fixture['tiles']} tiles. "
              "An exact byte baseline still needs a source for every tile, which costs the full crop size if held independently. "
              "The raw page estimates show optimistic overlay-only bytes and the baseline plus overlay bytes separately.", "",
              f"Completed all requested rounds and trace pairs: {results['complete']}.", "", "[Raw timing, frame counts, and audit rows](trace-ideas.json)", ""]
    (output / "trace-ideas.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--rounds", type=int, default=2)
    cli.add_argument("--seconds", type=float, default=95)
    cli.add_argument("--telemetry", action="store_true", help="Export process cycles and coarse memory samples")
    args = cli.parse_args()
    if args.rounds < 1 or args.seconds <= 0:
        cli.error("rounds and seconds must be positive")
    deadline = time.monotonic() + args.seconds
    args.output.mkdir(parents=True, exist_ok=True)
    native = Native()
    telemetry = TraceTelemetry() if args.telemetry else None

    def measured(name, action, *params):
        return telemetry.window(name, action, *params) if telemetry is not None else action(*params)

    result = {"risk": measured("risk_retrospective", risk_audit),
              "injected_omission": omission_proof(native),
              "sparse_fixture": measured("sparse_page_estimator", inspect_fixture),
              "rounds": [], "complete": False}
    if telemetry is not None:
        result["telemetry"] = {"windows": telemetry.windows,
                               "scope": "Whole worker CPU and memory. Trace windows include archive I/O and the "
                                        "oracle. Memory is sampled at phase boundaries and OS peaks are process-lifetime "
                                        "high-water marks. Stage cycles cover sampled execution plus counter overhead."}
    write_checkpoint(args.output, result)
    for number in range(1, args.rounds + 1):
        order = list(APPS)
        random.Random(60314 + number).shuffle(order)
        run = {"round": number, "order": order, "traces": {}}
        result["rounds"].append(run)
        for app in order:
            if time.monotonic() >= deadline:
                break
            print(f"Offline trace check, round {number}: {app}", flush=True)
            run["traces"][app] = measured(f"round_{number}_{app}", inspect_trace,
                                          TRACE_DIR / f"{app}.npz", native, number, deadline, telemetry)
            write_checkpoint(args.output, result)
        if time.monotonic() >= deadline:
            break
    result["complete"] = len(result["rounds"]) == args.rounds and all(
        len(run["traces"]) == len(APPS) and all(trace["all_adjacent_pairs"] for trace in run["traces"].values())
        for run in result["rounds"]
    )
    write_checkpoint(args.output, result)
    report(args.output, result)
    if telemetry is not None:
        telemetry.counters.close()
    print(args.output / "trace-ideas.md", flush=True)
    if not result["complete"]:
        raise SystemExit("Offline deadline reached. Partial trace results were retained.")


if __name__ == "__main__":
    main()
