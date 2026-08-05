"""Generate shields.io endpoint badges from the recorded baseline.

Why not measure these in CI? GitHub runners have no desktop session, so Desktop
Duplication does not run there and the numbers worth showing -- `grab()` and
`grab_frame()` -- do not exist. What CI *can* measure is synthetic conversion
work on a shared VM of unknown generation, which fluctuates enough that this
project has recorded 1.9x swings on identical code even on dedicated hardware.
A badge fed from that would report the runner, not the library.

So the source of truth is `benchmarks/baseline.json`: a measurement taken on
stated hardware, committed, and re-recorded deliberately. The badges are
regenerated from it, and CI fails if they drift apart -- which is the failure
mode that let the README claim "240Hz+" and "Python 3.7+" long after neither
was true.

**`baseline.json` is recorded with the optional native extension built**, so the
badges describe the library at full capability. `baseline-nonative.json` is the
same suite without it -- what a plain `pip install rapidshot` gets, and what CI
compares against, since CI has no toolchain. The README says which is which next
to the badges; if that ever stops being true, the badges are lying by omission
even while matching the file they came from.

A note on the two kinds of row here. `live.*` come from real capture and depend
on what was on screen at the time: ROADMAP.md section 3 records a 1.65-4.08 ms
spread on unchanged code, so those badges move for reasons that have nothing to
do with the library. `convert.*` are synthetic and deterministic, which is why a
conversion badge was added -- it is the one that actually tracks work done.

    python benchmarks/make_badges.py           # write badge JSON
    python benchmarks/make_badges.py --check    # fail if out of date (CI)
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "benchmarks" / "baseline.json"
BADGE_DIR = ROOT / ".github" / "badges"

# (badge filename, label, benchmark name in baseline.json)
#
# Synthetic rows only. The `live.*` rows were badged until 2026-08-05 and had to
# go: over six recordings on code that only ever got faster,
# `live.grab_frame_gpu` spanned 0.17-0.77 ms -- a 4.5x swing on a path that does
# no conversion at all, so the badge was reporting the desktop rather than the
# library. A badge that moves for reasons the reader cannot see is worse than no
# badge. The live figures are still measured and still in ROADMAP.md section 3,
# with their range stated; they are just not advertised as if they were stable.
BADGES = [
    # RGB is the mode most consumers hand to a model.
    ("convert-rgb.json", "BGRA->RGB", "convert.RGB"),
    # GRAY was the slowest mode by an order of magnitude; worth showing it is not.
    ("convert-gray.json", "BGRA->GRAY", "convert.GRAY"),
    # The direct-to-buffer capture path: staging read plus conversion, which is
    # the closest deterministic analogue of what `grab()` costs.
    ("shot.json", "shot()->buffer", "shot.RGB"),
]


def colour(ms: float) -> str:
    """Green when a frame fits comfortably inside a 60 Hz budget."""
    if ms < 1.0:
        return "brightgreen"
    if ms < 8.0:
        return "green"
    if ms < 16.7:
        return "yellow"
    return "orange"


def build() -> dict:
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    results = {r["name"]: r for r in data["results"]}
    machine = data.get("machine", {})

    badges = {}
    for filename, label, name in BADGES:
        entry = results.get(name)
        if entry is None:
            raise SystemExit(f"{name} missing from baseline.json")
        ms = entry["min_ms"]
        badges[filename] = {
            "schemaVersion": 1,
            "label": label,
            "message": f"{ms:.2f} ms/frame",
            "color": colour(ms),
        }

    # One badge naming the hardware, so no figure above is quoted without it.
    badges["measured-on.json"] = {
        "schemaVersion": 1,
        "label": "measured on",
        "message": f"{machine.get('gpu', 'unknown')} @ {machine.get('frame', '?')}",
        "color": "blue",
    }
    return badges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed badges are out of date")
    args = parser.parse_args()

    badges = build()
    BADGE_DIR.mkdir(parents=True, exist_ok=True)

    stale = []
    for filename, payload in badges.items():
        path = BADGE_DIR / filename
        rendered = json.dumps(payload, indent=2) + "\n"
        if args.check:
            current = path.read_text(encoding="utf-8") if path.exists() else ""
            if current != rendered:
                stale.append(filename)
        else:
            path.write_text(rendered, encoding="utf-8")
            print(f"  {filename}: {payload['label']} = {payload['message']}")

    if args.check:
        if stale:
            print("These badges no longer match benchmarks/baseline.json:")
            for name in stale:
                print(f"  {name}")
            print("\nRun: python benchmarks/make_badges.py")
            return 1
        print("badges match the baseline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
