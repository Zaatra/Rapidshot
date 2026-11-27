"""Run the resumable Intel HP 324pe cost benchmark in ordered groups."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results-cost-matrix"
CORE = ("baseline", "rapidshot_plain", "rapidshot_exact", "dxcam_copy", "dxcam_view", "mss_gdi")
CAPTURE = (*CORE, "windows_capture_dxgi", "windows_capture_wgc",
           "rapidshot_stream", "dxcam_stream", "dxcam_unpaced")
ACTIVE = ("sparse", "dense", "excluded_cover")
ANCHORS = (10, 30, 100)
INTERMEDIATE = (20, 40, 50, 60, 70, 80, 90)
GROUPS = (
    ("release-gates", (30,), ("sparse", "idle"), ("rapidshot_plain", "rapidshot_exact"), ("capture",), False),
    ("comparison-preview", (30,), ("sparse",), CAPTURE, ("capture", "template"), False),
    ("allocation-snapshots", (30,), ("sparse", "idle"), CAPTURE, ("capture",), True),
    ("capture-anchors", ANCHORS, ACTIVE, CAPTURE, ("capture",), False),
    ("idle", (30,), ("idle",), CAPTURE, ("capture",), False),
    ("template-anchors", ANCHORS, ACTIVE, CORE, ("template",), False),
    ("matched-tiles", ANCHORS, ("sparse",), CORE, ("shared_tiles",), False),
    ("capture-intermediate", INTERMEDIATE, ACTIVE, CORE, ("capture",), False),
    ("template-intermediate", INTERMEDIATE, ACTIVE, CORE, ("template",), False),
)


def run_group(group):
    label, rates, workloads, arms, phases, allocations = group
    output = OUT / ("allocation-profile" if allocations else "measurements")
    command = [
        sys.executable, "-X", "utf8", "-B", str(ROOT / "cost_benchmark.py"),
        "--rates", *(str(rate) for rate in rates), "--workloads", *workloads,
        "--arms", *arms, "--pipelines", *phases, "--duration", "10.2",
        "--rounds", "1" if allocations or label == "comparison-preview" else "5", "--output", str(output),
    ]
    if (output / "manifest.json").exists():
        command.append("--resume")
    if allocations:
        command.append("--allocation-profile")
    print(f"Starting {label}", flush=True)
    status = subprocess.run(command, cwd=ROOT.parents[1], check=False)
    print(f"Finished {label} with exit {status.returncode}", flush=True)
    return status.returncode


def main():
    names = tuple(group[0] for group in GROUPS)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups", nargs="+", choices=names)
    args = parser.parse_args()
    failures = [group[0] for group in GROUPS
                if (not args.groups or group[0] in args.groups) and run_group(group)]
    if failures:
        print("Groups with failed trials: " + ", ".join(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
