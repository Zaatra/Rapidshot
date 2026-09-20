"""Resumable matrix. Never substitutes source pacing for physical display refresh.

Run again after manually selecting another physical mode.

**The durable store is the record; `matrix.json` is an index.** Every cell runs
as a `section7.py` invocation against one shared run directory, so each path
inside each cell is committed as its own case before the next one starts. This
file's cell statuses are a convenience for deciding what to run next -- they are
rewritten whole on every update, and a whole-file rewrite is exactly the thing
that must not be the only copy of a result.

Interrupted and failed cells are retained and are NOT retried automatically. A
retry is asked for (`--retry-failed`), and is recorded as another attempt rather
than replacing what it retries.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from ai_ingestion import save_results, _child_options
import result_store
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
    ap.add_argument("--history-root", type=Path, default=None)
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue an existing run directory instead of starting one")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-measure cells and cases that previously failed")
    ap.add_argument("--continue-on-failure", action="store_true")
    args = ap.parse_args()
    if args.passes < 1:
        ap.error("passes must be positive")
    mode = display_mode()

    # One run for the whole matrix, opened before the first measurement. Each
    # child resumes it, so a crash anywhere leaves every case committed so far
    # intact and the next invocation picks up from there rather than starting a
    # second, disconnected history.
    store = (result_store.resume_run(args.resume) if args.resume
             else result_store.open_run(args.history_root, metadata={
                 "benchmark": "section7_suite", "seconds": args.seconds,
                 "passes": args.passes, "native_mode": mode, "argv": sys.argv,
                 "environment_snapshot": "incomplete (phase 2 not yet integrated)"}))
    print(f"History: {store.run_dir}", flush=True)

    payload = json.loads(args.out.read_text()) if args.out.exists() else {"schema_version":2,"cells":{}}
    payload["history"] = {"run_id": store.run_id, "run_dir": str(store.run_dir)}
    payload["derived"] = "cell statuses index the run directory; the run directory is the record"
    current = (mode["width"], mode["height"], mode["refresh_hz"])
    for cell in matrix(mode):
        for repeat in range(args.passes):
            key = f"{cell['width']}x{cell['height']}@{cell['refresh_hz']}-{cell['workload']}-{cell['category']}-{repeat+1}"
            old = payload["cells"].get(key, {})
            if old.get("status") == "measured":
                continue
            if old.get("status") in ("in_progress", "failed") and not args.retry_failed:
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
                   "--repeat",str(repeat+1),"--resume",str(store.run_dir),
                   "--out",str(output)]
            if cell["category"] == "inference":
                cmd += ["--model",str(args.model.resolve()),"--model-sha256",args.model_sha256]
            if args.retry_failed:
                cmd.append("--retry-failed")
            if args.continue_on_failure:
                cmd.append("--continue-on-failure")
            row.update(status="in_progress", output=str(output))
            save_results(args.out,payload)
            # A parent crash leaves the cell in_progress here and the case in
            # flight recorded as interrupted in the journal. Neither is retried
            # without being asked for.
            result = subprocess.run(cmd, **_child_options())
            row["status"] = "measured" if result.returncode == 0 else "failed"
            save_results(args.out,payload)
            if result.returncode:
                store.rebuild_summary()
                return result.returncode
    # Re-opened because the children have appended to this journal since the
    # handle above was created; writing with the sequence it remembers would
    # corrupt the run it is trying to close.
    store = result_store.resume_run(store.run_dir)
    store.rebuild_summary()
    store.close()
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
