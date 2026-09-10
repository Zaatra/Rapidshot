"""Resumable matrix. Never substitutes source pacing for physical display refresh.

Run again after manually selecting another physical mode. In-progress/failed
cells are retained and are NOT automatically retried after a crash.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ai_ingestion import save_results, _child_options
from section7 import display_mode, ROOT


def matrix(mode):
    # Include this machine's native mode in addition to the specified targets.
    modes = {(1920,1080,hz) for hz in (60,120,144,240)}
    modes |= {(2560,1440,hz) for hz in (60,120,144,240)}
    modes |= {(3840,2160,hz) for hz in (60,120,144,240)}
    modes.add((mode["width"],mode["height"],mode["refresh_hz"]))
    return [{"width":w,"height":h,"refresh_hz":hz,"workload":workload,"category":category}
            for w,h,hz in sorted(modes) for workload in ("static","scroll","motion")
            for category in ("ingestion","inference","agent")]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT/"build/section7/matrix.json")
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    if args.passes < 1:
        ap.error("passes must be positive")
    mode = display_mode()
    payload = json.loads(args.out.read_text()) if args.out.exists() else {"schema_version":1,"cells":{}}
    current = (mode["width"], mode["height"], mode["refresh_hz"])
    for cell in matrix(mode):
        for repeat in range(args.passes):
            key = f"{cell['width']}x{cell['height']}@{cell['refresh_hz']}-{cell['workload']}-{cell['category']}-{repeat+1}"
            old = payload["cells"].get(key, {})
            if old.get("status") in ("measured", "in_progress", "failed"):
                continue
            matching = (cell["width"],cell["height"],cell["refresh_hz"]) == current
            row = {**cell,"pass":repeat+1,"status":"pending" if matching else "unmeasured",
                   "reason":None if matching else "requires matching physical display mode"}
            payload["cells"][key] = row
            save_results(args.out, payload)
            if not args.execute or not matching:
                continue
            output = args.out.parent / "runs" / (key+".json")
            cmd = [sys.executable,"-u",str(ROOT/"benchmarks/section7.py"),
                   "--category",cell["category"],"--workload",cell["workload"],
                   "--motion-fps",str(cell["refresh_hz"]),"--seconds",str(args.seconds),
                   "--out",str(output)]
            if cell["category"] == "inference":
                cmd += ["--model",str(args.model.resolve()),"--model-sha256",args.model_sha256]
            row.update(status="in_progress", output=str(output))
            save_results(args.out,payload)
            # Parent crash leaves the saved cell in_progress. No automatic rerun.
            result = subprocess.run(cmd, **_child_options())
            row["status"] = "measured" if result.returncode == 0 else "failed"
            save_results(args.out,payload)
            if result.returncode:
                return result.returncode
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
