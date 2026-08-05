"""Diff stored perf_suite recordings against each other.

`perf_suite.py --compare` needs a live run on one side, so it answers "is this
change a regression right now?". This answers a different question -- "what did
this version buy?" -- from committed JSON files, with no measuring involved.

Every recording is normalised by its own `control.memcopy` row before any ratio
is quoted. The control's code is identical in every recording, so its movement is
the machine and not the library; this is the same calibration `print_comparison`
applies, and without it a recording taken on a cold machine silently flatters
itself. See ROADMAP.md section 2 on why raw numbers across recordings mislead.

    python benchmarks/compare_recordings.py
    python benchmarks/compare_recordings.py --base old.json new.json other.json

With no arguments it compares every `baseline*.json` in this directory, oldest
first, using the second-oldest as the reference (the oldest predates the harness
pacing correction and is not comparable -- again, section 2).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
CONTROL = "control.memcopy"

# Presentation order: conversions, then the paths that wrap them, then live.
ORDER = [
    "convert.BGRA", "convert.RGB", "convert.BGR", "convert.RGBA", "convert.GRAY",
    "shot.BGRA", "shot.RGB", "shot.BGR", "shot.RGBA", "shot.GRAY",
    "process.BGRA", "process.RGB",
    "pipeline.cpu_to_nchw", "pipeline.gpu_dispatch", "pipeline.gpu_plus_readback",
    CONTROL,
    "com.get_desc_call", "live.grab_with_frame", "live.grab_frame_gpu",
]


def load(path: Path) -> Tuple[Dict[str, Optional[float]], dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {r["name"]: r.get("min_ms") for r in data["results"]}
    return rows, data.get("machine", {})


def usable(value) -> bool:
    return isinstance(value, (int, float)) and value > 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path,
                    help="recordings to compare, oldest first")
    ap.add_argument("--base", type=Path,
                    help="recording to quote gains against "
                         "(default: the second file given)")
    args = ap.parse_args()

    files: List[Path] = args.files or sorted(HERE.glob("baseline*.json"))
    if len(files) < 2:
        print("need at least two recordings to compare")
        return 1

    loaded = []
    for path in files:
        if not path.exists():
            print(f"missing: {path}")
            return 1
        rows, machine = load(path)
        loaded.append((path.stem, rows, machine))
    loaded.sort(key=lambda e: e[2].get("timestamp", ""))

    print("=" * 104)
    print("RECORDINGS (oldest first)")
    print("=" * 104)
    for name, rows, m in loaded:
        print(f"  {name:<30} {m.get('timestamp', '?')[:19]}  "
              f"rapidshot={str(m.get('rapidshot', '?')):<6} "
              f"control={rows.get(CONTROL)}")

    labels = [name for name, _, _ in loaded]
    width = max(13, max(len(l) for l in labels) + 2)

    print()
    print("=" * 104)
    print("RAW MINIMUMS (ms/frame) -- do not read ratios off this table; see below")
    print("=" * 104)
    hdr = f"{'benchmark':<24}" + "".join(f"{l[-width + 2:]:>{width}}" for l in labels)
    print(hdr)
    print("-" * len(hdr))
    names = ORDER + [n for _, rows, _ in loaded for n in rows if n not in ORDER]
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        cells = ""
        for _, rows, _ in loaded:
            v = rows.get(name)
            cells += f"{v:>{width}.3f}" if usable(v) else f"{'-':>{width}}"
        print(f"{name:<24}{cells}")

    # Reference recording: explicit, or the second-oldest.
    if args.base:
        base_name = args.base.stem
        match = [e for e in loaded if e[0] == base_name]
        if not match:
            print(f"\n--base {base_name} is not among the files given")
            return 1
        base = match[0]
    else:
        base = loaded[1]
    base_name, base_rows, _ = base
    base_ctrl = base_rows.get(CONTROL)

    others = [e for e in loaded if e[0] != base_name]

    print()
    print("=" * 104)
    print(f"DRIFT-ADJUSTED GAINS vs {base_name}")
    print("=" * 104)
    drifts = {}
    for name, rows, _ in others:
        c = rows.get(CONTROL)
        drifts[name] = (c / base_ctrl) if (usable(c) and usable(base_ctrl)) else 1.0
        state = ("machine slower then" if drifts[name] > 1.05
                 else "machine faster then" if drifts[name] < 0.95
                 else "machine comparable")
        print(f"  {name:<30} control drift {drifts[name]:.2f}x  ({state})")
    print("  A gain of 1.00x means unchanged. Ratios are multiplied by the drift,")
    print("  so a recording taken on a faster machine is not credited for it.")
    print()

    hdr2 = f"{'benchmark':<24}{base_name[-11:]:>12}"
    for name, _, _ in others:
        hdr2 += f"{name[-11:]:>12}{'gain':>9}"
    print(hdr2)
    print("-" * len(hdr2))
    for name in ORDER:
        if name == CONTROL:
            continue
        b = base_rows.get(name)
        if not usable(b):
            continue
        line = f"{name:<24}{b:>12.3f}"
        for other_name, rows, _ in others:
            v = rows.get(name)
            if usable(v):
                line += f"{v:>12.3f}{(b / v) * drifts[other_name]:>8.2f}x"
            else:
                line += f"{'-':>12}{'-':>9}"
        note = "  <- live: informational, tracks the screen" \
            if name.startswith("live.") else ""
        print(line + note)

    print()
    print("Live rows are not evidence. They depend on what was on screen during")
    print("each recording, so they move by more than most code changes do --")
    print("ROADMAP.md section 3 records a 1.65-4.08 ms spread on unchanged code.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
