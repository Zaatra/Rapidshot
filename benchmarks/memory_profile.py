"""What a capture library costs in memory, per workload. (ROADMAP.md § 7.2)

There is a target — *within 1.25x DXcam at equivalent capture* — and until now
no instrument that could say whether it is met. `memory_pool_stress_test.py`
exercises the pool but records no process memory at all, and the AI-ingestion
harness records `rss_mb` for a single 8-second run of one path, which is a
by-product rather than a measurement.

**The number that matters is capture-attributable memory, not process memory.**
A bare interpreter with NumPy imported is already ~60 MB, and RapidShot pulls in
`comtypes` where DXcam does not, so comparing working sets directly charges
RapidShot for its imports and flatters nothing. Every library is therefore
measured twice: once after importing and creating its camera but *before* the
first frame, and once in steady state. The difference is what capture costs.

**Three workloads, because memory behaviour is not one number.** Desktop
Duplication only reports *changed* content, so an idle screen is the case where
a correct implementation allocates nothing and a leaky one still grows — it is
the most diagnostic of the three, and the easiest to accidentally not test. The
same controlled D3D source as § 7.0 drives all three, so the screen is doing a
known thing rather than whatever happened to be on it.

    static   unchanging window     — allocation on a quiet desktop
    scroll   moderate change       — the ordinary case
    motion   full-frame at source  — sustained worst case

**Growth is reported as a slope, not as end-minus-start.** A single pair of
samples cannot distinguish a leak from the allocator's warm-up; a least-squares
fit over the steady-state window can, and it is the only figure here that
answers "does this grow without bound".

Each library runs in its own process so nothing else is on its books, and the
WHEA guard from § 7.0 is in force: a machine-check event part way through
invalidates every number after it, and a crash loses the recording outright.

Usage::

    python benchmarks/memory_profile.py --seconds 10
    python benchmarks/memory_profile.py --libraries rapidshot dxcam --workloads static
"""

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import machine_inventory
import result_store
from result_store import CaseIdentity
import result_validation
from result_validation import should_stop
from ai_ingestion import RunLogs, _child_options, save_results, stage  # noqa: E402
from section7 import SOURCE, HealthGuard, display_mode  # noqa: E402

LIBRARIES = ("mss", "dxcam", "rapidshot", "rapidshot-frame")
WORKLOADS = ("static", "scroll", "motion")

#: Seconds of each run discarded before steady state. The first allocations of
#: a capture session are real but they are not the thing being measured; what
#: is being measured is whether anything keeps allocating after them.
WARMUP_SECONDS = 2.0
SAMPLE_INTERVAL = 0.1


def _mb(value):
    return round(value / 1e6, 2)


def _memory(proc):
    """Working set, private bytes and commit, as Windows accounts for them.

    psutil's Windows `pmem` carries all three plus the OS's own peak, which is
    worth more than a peak computed from samples: it cannot miss a spike
    between two of them.
    """
    info = proc.memory_info()
    return {
        "working_set": info.rss,
        "private": getattr(info, "private", info.rss),
        "commit": getattr(info, "pagefile", info.vms),
        "peak_working_set": getattr(info, "peak_wset", info.rss),
    }


def _slope_mb_per_s(times, values):
    """Least-squares slope. Two samples cannot tell a leak from warm-up."""
    if len(times) < 3:
        return None
    mean_t, mean_v = statistics.fmean(times), statistics.fmean(values)
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator == 0:
        return None
    numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    return round(numerator / denominator / 1e6, 4)


def _capture_loop(library):
    """Return (grab, close, baseline_note) for a library.

    `grab` returns True when a new frame arrived. Desktop Duplication reports
    only changed content, so on the static workload it mostly returns False —
    which is the point of that workload, not a failure of it.
    """
    if library == "mss":
        import mss

        session = mss.mss()
        monitor = session.monitors[1]

        def grab():
            session.grab(monitor)
            return True

        return grab, session.close, "mss re-grabs unconditionally; it has no change detection"

    if library == "dxcam":
        import dxcam

        camera = dxcam.create()

        def grab():
            return camera.grab() is not None

        return grab, camera.release, "dxcam grab() -> numpy, None when unchanged"

    import rapidshot

    camera = rapidshot.create()
    if library == "rapidshot-frame":
        def grab():
            frame = camera.grab_frame()
            if frame is None:
                return False
            frame.release()
            return True

        return (grab, camera.release,
                "grab_frame() + release; exercises the surface pool with no CPU copy")

    def grab():
        return camera.grab() is not None

    return grab, camera.release, "grab() -> numpy, None when unchanged"


def run_worker(library, seconds):
    import psutil

    result = {"library": library}
    close = None
    try:
        proc = psutil.Process()
        stage("memory-imports", library=library)
        grab, close, note = _capture_loop(library)
        result["note"] = note

        # Taken after the camera exists but before any frame: everything the
        # library costs merely by being set up, which is not capture's bill.
        idle = _memory(proc)
        result["baseline"] = {k: _mb(v) for k, v in idle.items()}

        stage("memory-measuring", library=library)
        cpu0 = proc.cpu_times()
        start = time.perf_counter()
        end = start + seconds
        next_sample = start
        samples, frames, misses = [], 0, 0
        while True:
            now = time.perf_counter()
            if now >= end:
                break
            if grab():
                frames += 1
            else:
                misses += 1
            if now >= next_sample:
                samples.append((now - start, _memory(proc)))
                next_sample = now + SAMPLE_INTERVAL
        wall = time.perf_counter() - start
        cpu1 = proc.cpu_times()

        steady = [(t, m) for t, m in samples if t >= WARMUP_SECONDS]
        if len(steady) < 3:
            steady = samples
        times = [t for t, _ in steady]
        result["samples"] = len(samples)
        result["steady_samples"] = len(steady)
        for key in ("working_set", "private", "commit"):
            values = [m[key] for _, m in steady]
            result[f"{key}_mb"] = _mb(statistics.median(values))
            result[f"{key}_peak_mb"] = _mb(max(values))
            result[f"{key}_growth_mb_per_s"] = _slope_mb_per_s(times, values)
            # Capture's own bill: steady state minus what setup already cost.
            result[f"{key}_over_baseline_mb"] = round(
                result[f"{key}_mb"] - _mb(idle[key]), 2)
        result["os_peak_working_set_mb"] = _mb(
            max(m["peak_working_set"] for _, m in samples)) if samples else None

        used = cpu1.user - cpu0.user + cpu1.system - cpu0.system
        result.update(
            frames=frames, misses=misses, elapsed_seconds=round(wall, 3),
            fps=round(frames / wall, 1) if wall else 0.0,
            cpu_percent=round(100 * used / wall, 1) if wall else 0.0,
            cpu_ms_per_frame=round(used / frames * 1000, 3) if frames else None)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if close is not None:
            try:
                close()
            except Exception as exc:
                result.setdefault("cleanup_error", str(exc))
    return result


class WorkloadSource:
    """The § 7.0 D3D source, driving a known screen rather than a guessed one."""

    def __init__(self, logs, workload, width, height, fps):
        self.logs, self.workload = logs, workload
        self.width, self.height, self.fps = width, height, fps
        self.proc = self.stdout = self.stderr = None

    def start(self):
        if not SOURCE.is_file():
            raise RuntimeError(
                "build the source first: cargo build --release --bin latency_source "
                "--manifest-path native/Cargo.toml")
        directory = self.logs.directory
        self.stdout = (directory / f"source-{self.workload}.stdout.log").open("wb")
        self.stderr = (directory / f"source-{self.workload}.stderr.log").open("wb")
        command = [str(SOURCE), str(self.width), str(self.height), str(self.fps),
                   self.workload, str(directory / f"presents-{self.workload}.jsonl")]
        self.logs.event("source-starting", command=command)
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.stdout,
                                     stderr=self.stderr, **_child_options())
        reader = (directory / f"source-{self.workload}.stdout.log").open(encoding="utf-8")
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                line = reader.readline()
                if line.strip().startswith("{") and "ready" in line:
                    self.logs.event("source-ready", workload=self.workload)
                    return
                if self.proc.poll() is not None:
                    raise RuntimeError(f"source exited with {self.proc.returncode}")
                time.sleep(0.05)
        finally:
            reader.close()
        raise RuntimeError(f"source readiness timeout for workload {self.workload}")

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for handle in (self.stdout, self.stderr):
            if handle is not None:
                handle.close()


def spawn_worker(library, workload, seconds, logs):
    stem = f"{workload}-{library}"
    out = logs.directory / (stem + ".stdout.log")
    err = logs.directory / (stem + ".stderr.log")
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--worker", library, "--seconds", str(seconds)]
    logs.event("worker-starting", library=library, workload=workload)
    with out.open("wb") as stdout, err.open("wb") as stderr:
        proc = subprocess.Popen(command, stdout=stdout, stderr=stderr, **_child_options())
        try:
            proc.wait(timeout=max(120.0, seconds * 8))
        except subprocess.TimeoutExpired:
            proc.kill()
            return {"library": library, "error": "worker timeout"}
    text = out.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    for line in reversed(text):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {"library": library,
            "error": f"no worker result (exit {proc.returncode}); see {err}"}


def print_table(rows):
    header = (f"{'workload':<9}{'library':<17}{'fps':>7}{'WS MB':>8}{'private':>9}"
              f"{'commit':>8}{'capture':>9}{'growth/s':>10}{'peak':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        if "error" in row:
            print(f"{row['workload']:<9}{row['library']:<17}  ERROR: {row['error'][:50]}")
            continue
        growth = row.get("working_set_growth_mb_per_s")
        print(f"{row['workload']:<9}{row['library']:<17}{row['fps']:>7.1f}"
              f"{row['working_set_mb']:>8.1f}{row['private_mb']:>9.1f}"
              f"{row['commit_mb']:>8.1f}{row['working_set_over_baseline_mb']:>9.1f}"
              f"{'n/a' if growth is None else f'{growth:+.3f}':>10}"
              f"{row['working_set_peak_mb']:>8.1f}")
    print("\n'capture' is steady state minus the same process before its first "
          "frame:\nthe memory capture itself is responsible for.")


#: What a memory row must contain to be a measurement. Growth may legitimately
#: be zero or negative, so it is not required to be positive -- only present and
#: finite, which the non-finite check already covers.
MEMORY_REQUIRED = ("fps", "working_set_mb", "elapsed_seconds")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=LIBRARIES)
    parser.add_argument("--libraries", nargs="+", choices=LIBRARIES,
                        default=list(LIBRARIES))
    parser.add_argument("--workloads", nargs="+", choices=WORKLOADS,
                        default=list(WORKLOADS))
    parser.add_argument("--seconds", type=float, default=10.0)
    # The display, not a window. This captured the whole screen while animating
    # 900x700 of it, so on a 2560x1600 panel 15% of the captured area was moving
    # and every library was measured against a mostly-still desktop. The
    # recorded memory table in ROADMAP section 7.0 was taken that way.
    parser.add_argument("--width", type=int, default=0,
                        help="source width; 0 follows the display")
    parser.add_argument("--height", type=int, default=0,
                        help="source height; 0 follows the display")
    parser.add_argument("--source-fps", type=float, default=0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=1,
                        help="which repeat of this configuration this run is")
    parser.add_argument("--continue-on-failure", action="store_true")
    machine_inventory.add_policy_arguments(parser)
    result_store.add_history_arguments(parser)
    args = parser.parse_args(argv)

    if args.worker:
        print(json.dumps(run_worker(args.worker, args.seconds), allow_nan=False),
              flush=True)
        return 0
    if args.seconds <= WARMUP_SECONDS:
        parser.error(f"--seconds must exceed the {WARMUP_SECONDS}s warm-up")
    mode = display_mode()
    args.width = args.width or mode["width"]
    args.height = args.height or mode["height"]
    if args.source_fps == 0:
        # Follow the panel, for section 7.0's reason: a fixed default describes
        # the default rather than the machine, and vsync caps the source at the
        # refresh rate anyway. The source refuses anything outside 1..240.
        args.source_fps = min(240.0, float(display_mode()["refresh_hz"]))

    policy, machine = machine_inventory.prepare_run(args)
    configuration = (f"{args.width}x{args.height}@{args.source_fps:g}-"
                     f"{args.seconds:g}s-warmup{WARMUP_SECONDS:g}s"
                     + ("-unpinned" if args.no_pin else "-pinned"))
    captured = (0, 0, mode["width"], mode["height"])
    animated = (0, 0, args.width, args.height)
    coverage = result_validation.coverage_reasons(animated, captured)
    store = result_store.open_for_runner(
        "memory_profile", root=args.history_root, resume=args.resume,
        disabled=args.no_history,
        metadata={"configuration": configuration, "seconds": args.seconds,
                  "warmup_seconds": WARMUP_SECONDS, "sample_interval": SAMPLE_INTERVAL,
                  "argv": sys.argv, "machine_id": machine["machine_id"],
                  "display_fingerprint": machine["display_fingerprint"],
                  "cpu_policy": policy.as_dict(), "environment": machine})
    logs = RunLogs(args.log_dir, args.out)
    print(f"Diagnostics: {logs.directory}")
    if store is not None:
        print(f"History: {store.run_dir}")
    guard = HealthGuard(logs)
    rows = []
    payload = {"schema_version": 1, "warmup_seconds": WARMUP_SECONDS,
               "sample_interval": SAMPLE_INTERVAL, "seconds": args.seconds,
               "results": rows, "logs": str(logs.directory),
               "environment": machine, "cpu_policy": policy.as_dict(),
               "animated_rect": list(animated), "captured_rect": list(captured),
               "coverage_warnings": list(coverage)}
    try:
        for workload in args.workloads:
            source = WorkloadSource(logs, workload, args.width, args.height,
                                    args.source_fps)
            source.start()
            try:
                for library in args.libraries:
                    identity = CaseIdentity(benchmark="memory_profile", path=library,
                                            configuration=configuration, workload=workload,
                                            repeat=args.repeat)
                    action, done = result_store.resume_decision(
                        store, identity, retry_failed=args.retry_failed)
                    if action == "skip":
                        hint = ("" if done in result_store.RESUME_SETTLED
                                else "; pass --retry-failed to measure it again")
                        print(f"  [{workload}/{library}] already {done}{hint}", flush=True)
                        continue
                    guard.check(force=True)
                    print(f"  [{workload}/{library}] ...", flush=True)
                    with result_store.case_context(
                            store, identity, retry=action == "retry",
                            required=MEMORY_REQUIRED) as case:
                        row = spawn_worker(library, workload, args.seconds, logs)
                        row["workload"] = workload
                        row["animated_rect"] = list(animated)
                        row["captured_rect"] = list(captured)
                        case.result = row
                        case.contamination = list(coverage)
                    if case.record is not None:
                        row = dict(row, case_status=case.record.status,
                                   case={"run_id": store.run_id,
                                         "case_id": case.record.case_id,
                                         "attempt_id": case.record.attempt_id})
                    rows.append(row)
                    if "error" in row:
                        print(f"    ERROR: {row['error']}", flush=True)
                    else:
                        print(f"    {row['fps']:.1f} fps; working set "
                              f"{row['working_set_mb']:.1f} MB "
                              f"({row['working_set_over_baseline_mb']:+.1f} over "
                              f"baseline)", flush=True)
                    if args.out:
                        args.out.write_text(json.dumps(payload, indent=2) + "\n",
                                            encoding="utf-8")
            finally:
                source.close()
    except KeyboardInterrupt:
        payload["error"] = "interrupted"
    finally:
        if args.out:
            args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print()
    print_table(rows)
    if args.out:
        print(f"\nWrote {args.out}")
    return int(any("error" in row for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
