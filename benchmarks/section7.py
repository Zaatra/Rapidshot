"""Supervised section 7 benchmarks. Desktop/GPU work happens only in workers."""
import argparse
import ctypes
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from ai_ingestion import RunLogs, MotionSource, MotionError, _child_options, spawn, save_results, stage
from benchmark_contract import SHAPE, PresentLog, canonical_rgb, normalized_tensor, percentiles, qpc_clock, sha256
from section7_adapters import Adapter, PATHS, CPU_PATHS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "native" / "target" / "release" / "latency_source.exe"


def display_mode():
    """Read primary DEVMODEW without creating a graphics device or changing mode."""
    class Mode(ctypes.Structure):
        _fields_ = [("name", ctypes.c_wchar * 32), ("spec", ctypes.c_ushort),
            ("driver", ctypes.c_ushort), ("size", ctypes.c_ushort), ("extra", ctypes.c_ushort),
            ("fields", ctypes.c_ulong), ("display", ctypes.c_byte * 16),
            ("color", ctypes.c_short), ("duplex", ctypes.c_short), ("yres", ctypes.c_short),
            ("tt", ctypes.c_short), ("collate", ctypes.c_short), ("form", ctypes.c_wchar * 32),
            ("logpixels", ctypes.c_ushort), ("bits", ctypes.c_ulong),
            ("width", ctypes.c_ulong), ("height", ctypes.c_ulong), ("flags", ctypes.c_ulong),
            ("hz", ctypes.c_ulong), ("tail", ctypes.c_ulong * 8)]
    user = ctypes.WinDLL("user32", use_last_error=True)
    fn = user.EnumDisplaySettingsW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.POINTER(Mode)]
    mode = Mode()
    mode.size = ctypes.sizeof(Mode)
    if not fn(None, 0xFFFFFFFF, ctypes.byref(mode)):
        raise OSError("EnumDisplaySettingsW failed")
    return {"width": mode.width, "height": mode.height, "refresh_hz": mode.hz}


class HealthGuard:
    """A read-only WHEA check. A failed check blocks live work; it is not 'healthy'."""
    def __init__(self, logs):
        self.logs, self.last = logs, 0.0
        self.baseline = self.query()
        logs.event("health-baseline", **self.baseline)

    @staticmethod
    def query():
        script = "$ErrorActionPreference='Stop'; $e=@(Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WHEA-Logger'} -ErrorAction SilentlyContinue -ErrorVariable ev); if($ev -and $ev[0].FullyQualifiedErrorId -notmatch 'NoMatchingEventsFound'){throw $ev[0]}; @{latest=if($e.Count){$e[0].RecordId}else{0}; count=$e.Count} | ConvertTo-Json -Compress"
        result = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                                capture_output=True, text=True, timeout=15, **_child_options())
        if result.returncode:
            raise RuntimeError("cannot inspect WHEA log: " + result.stderr[-500:])
        return json.loads(result.stdout)

    def check(self, force=False):
        if not force and time.monotonic() - self.last < 5:
            return
        self.last = time.monotonic()
        current = self.query()
        if current != self.baseline:
            self.logs.event("hardware-error-or-log-change", current=current)
            raise MotionError("WHEA log changed; stopping live benchmarks")


class VisualSource(MotionSource):
    def __init__(self, logs, args, guard):
        super().__init__(logs, args.motion_fps)
        self.args, self.guard = args, guard
        self.present_log = logs.directory / "presents.jsonl"

    def start(self):
        if not SOURCE.is_file():
            raise RuntimeError("build source first: cargo build --release --bin latency_source --manifest-path native/Cargo.toml")
        self.stdout = (self.logs.directory / "motion.stdout.log").open("wb")
        self.stderr = (self.logs.directory / "motion.stderr.log").open("wb")
        self.reader = (self.logs.directory / "motion.stdout.log").open(encoding="utf-8")
        command = [str(SOURCE), str(self.args.width), str(self.args.height),
                   str(self.fps), self.args.workload, str(self.present_log)]
        self.logs.event("d3d-source-starting", command=command)
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=self.stdout,
                                     stderr=self.stderr, **_child_options())
        self.logs.event("d3d-source-launched", child_pid=self.proc.pid)
        deadline = time.monotonic() + 20
        while not self.ready:
            self.check()
            if time.monotonic() > deadline:
                raise MotionError("D3D source readiness timeout")
            time.sleep(.05)

    def check(self):
        self.guard.check()
        super().check()


def worker(args):
    result = {"path": args.worker, "category": args.category}
    adapter = tracker = cp = None
    try:
        stage("imports", path=args.worker)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import psutil
        if args.category != "agent":
            import cupy as cp
        now, frequency = qpc_clock()
        tracker = PresentLog(args.present_log, frequency)
        infer = None
        if args.category == "inference":
            from ai_pipeline import Inference
            if sha256(args.model) != args.model_sha256:
                raise ValueError("model hash mismatch")
            stage("inference-starting")
            infer = Inference(args.model, cp, np)
        stage("adapter-starting", path=args.worker)
        adapter = Adapter(args.worker, cp, np, verify=args.verify, agent=args.category == "agent")

        def one():
            sample = adapter.capture()
            if sample is None:
                return None
            tensor_ready = now()
            t0 = time.perf_counter()
            if infer is not None:
                binding = infer.run(sample.tensor)
                if args.verify:
                    outputs = binding.copy_outputs_to_cpu()
                    if not outputs or not all(x.size and np.isfinite(x).all() for x in outputs):
                        raise ValueError("model output is empty or nonfinite")
                sample.stages["inference_ms"] = (time.perf_counter() - t0) * 1000
            elif args.category == "agent":
                from agent_pipeline import encode
                data, size = encode(sample.rgb, args.codec, args.quality)
                sample.stages["encode_ms"] = (time.perf_counter() - t0) * 1000
                result["compressed_bytes"] = size
                result["api_payload_bytes"] = len(data.encode("ascii"))
                if args.verify:
                    import base64, io
                    from PIL import Image
                    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(data.split(",", 1)[1]))))
                    if decoded.shape != sample.rgb.shape:
                        raise ValueError("encoded screenshot dimensions changed")
                    if args.codec == "png" and not np.array_equal(decoded, sample.rgb):
                        raise ValueError("PNG round-trip mismatch")
            return sample, tensor_ready, now()

        stage("verification" if args.verify else "warmup")
        if args.verify:
            deadline = time.perf_counter() + 20
            item = None
            while item is None and time.perf_counter() < deadline:
                item = one()
            if item is None:
                raise RuntimeError("no verification frame")
            sample = item[0]
            if sample.frame_id is None:
                raise ValueError("source marker missing or obscured")
            if args.category != "agent":
                reference = normalized_tensor(canonical_rgb(sample.reference, np), np)
                actual = cp.asnumpy(sample.tensor)
                if actual.shape != SHAPE or actual.dtype != np.float16 or not np.array_equal(actual, reference):
                    raise ValueError("tensor is not bit-identical to canonical FP16 reference")
            result.update(verified=True, frame_id=sample.frame_id)
        else:
            got, deadline = 0, time.perf_counter() + 30
            while got < args.warmup and time.perf_counter() < deadline:
                if one() is not None:
                    got += 1
            if got != args.warmup:
                raise RuntimeError(f"warmup incomplete: {got}/{args.warmup}")
            proc = psutil.Process()
            cpu0, rss0 = proc.cpu_times(), proc.memory_info().rss
            samples, unique, invalid, misses, stages, calls = [], set(), 0, 0, {}, []
            stage("measurement")
            start_qpc, start = now(), time.perf_counter()
            while time.perf_counter() - start < args.seconds:
                call = time.perf_counter()
                item = one()
                if item is None:
                    misses += 1
                    continue
                sample, tensor_ready, ready = item
                calls.append((time.perf_counter() - call) * 1000)
                for name, value in sample.stages.items():
                    stages.setdefault(name, []).append(value)
                result["h2d_bytes_per_frame"] = sample.h2d_bytes
                if sample.frame_id is None:
                    invalid += 1
                    continue
                if sample.frame_id not in unique:
                    unique.add(sample.frame_id)
                    samples.append((sample.frame_id, tensor_ready, ready))
            end_qpc, elapsed = now(), time.perf_counter() - start
            cpu1 = proc.cpu_times()
            used = cpu1.user + cpu1.system - cpu0.user - cpu0.system
            tracker.refresh()
            ages = [tracker.age(i, ready) for i, _, ready in samples]
            tensor_ages = [tracker.age(i, ready) for i, ready, _ in samples]
            if not samples or invalid or any(x is None for x in ages):
                raise RuntimeError(f"invalid pixel-age run: {len(samples)} unique, {invalid} undecodable; missing timestamps={sum(x is None for x in ages)}")
            presented = {i for i, row in tracker.frames.items() if start_qpc <= row["qpc_before"] <= end_qpc}
            result.update(frames=len(calls), unique_frames=len(unique), unique_fps=len(unique)/elapsed,
                elapsed_seconds=elapsed, requested_seconds=args.seconds, no_frame_polls=misses,
                duplicate_frames=len(calls)-len(unique), dropped_source_frames=len(presented-unique),
                source_frames_in_window=len(presented), qpc_frequency=frequency,
                present_to_ready_ms=percentiles(ages), present_to_tensor_ms=percentiles(tensor_ages),
                call_ms=percentiles(calls), stages={k: percentiles(v) for k,v in stages.items()},
                cpu_percent=100*used/elapsed, cpu_ms_per_unique_frame=1000*used/len(unique),
                rss_mb=proc.memory_info().rss/1e6, rss_growth_mb=(proc.memory_info().rss-rss0)/1e6)
            if cp is not None:
                free, total = cp.cuda.runtime.memGetInfo()
                result["device_vram_used_mb"] = (total-free)/1e6
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        if type(exc).__name__ == "CrossAdapterRequired":
            result["unavailable"] = True
    finally:
        for resource in (adapter, tracker):
            if resource is not None:
                try:
                    resource.close()
                except Exception as exc:
                    result.setdefault("error", "cleanup failed")
                    result.setdefault("cleanup_errors", []).append(str(exc))
    return result


def parse_result(path, verify, returncode, stdout, stderr):
    try:
        row = json.loads(stdout.strip().splitlines()[-1])
        if not isinstance(row, dict) or row.get("path") != path:
            raise ValueError("wrong result path or shape")
        if returncode:
            row.setdefault("error", f"worker exit {returncode}")
        if "error" not in row:
            if verify:
                valid = row.get("verified") is True
            else:
                valid = all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                            and math.isfinite(row[k]) and row[k] > 0
                            for k in ("unique_frames", "unique_fps", "elapsed_seconds"))
                valid = valid and isinstance(row.get("present_to_ready_ms"), dict)
            if not valid:
                raise ValueError("missing valid measurements")
        row["returncode"] = returncode
        return row
    except (ValueError, IndexError) as exc:
        return {"path": path, "error": f"invalid worker result: {exc}", "returncode": returncode}


def main(category="ingestion", argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=PATHS)
    parser.add_argument("--category", choices=("ingestion", "inference", "agent"), default=category)
    parser.add_argument("--paths", nargs="+", choices=PATHS)
    parser.add_argument("--seconds", type=float, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--with-motion", action="store_true", help="compatibility flag; controlled D3D source is always used")
    # 240, not 60. The source is the measurement apparatus: at 60 it presents
    # 60 frames a second, every capture path returns 60 unique frames a second,
    # and the run reports a tie that describes this argument rather than any
    # library. That is what happened at 60 (all paths ~60 fps) and at a 120 cap
    # that achieved 64 (all paths 82-86 fps).
    #
    # The swap chain is vsync-locked, so 240 becomes ~165 on a 165 Hz panel --
    # comfortably above what any path here sustains, which is the condition the
    # measurement needs. Always read the achieved rate the source reports before
    # trusting a throughput figure; if it is not clearly above the fastest path,
    # the numbers describe the source.
    parser.add_argument("--motion-fps", type=float, default=0,
                        help="0 follows the display's physical refresh")
    parser.add_argument("--workload", choices=("static", "scroll", "motion"), default="motion")
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-sha256")
    parser.add_argument("--codec", choices=("png", "jpeg"), default="png")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--present-log", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "section7" / "last-run.json")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0 or args.warmup < 0:
        parser.error("seconds must be positive and finite; warmup must be nonnegative")
    if args.motion_fps == 0:
        # Follow the panel. A fixed default is wrong on any display it does not
        # match: at 60 on this 165 Hz panel every path returned exactly 60
        # unique frames a second and the run reported a tie that described the
        # default rather than any library. Vsync caps the source at the refresh
        # rate anyway, so the physical mode is both the highest useful value and
        # the only one that cannot exceed what the display can present.
        args.motion_fps = float(display_mode()["refresh_hz"])
    if not math.isfinite(args.motion_fps) or not 1 <= args.motion_fps <= 240:
        parser.error("motion-fps must be in 1..240")
    if not 1 <= args.quality <= 100:
        parser.error("quality must be in 1..100")
    if args.category == "inference" and (not args.model or not args.model_sha256):
        parser.error("inference requires --model and --model-sha256")
    if args.worker:
        row = worker(args)
        print(json.dumps(row, allow_nan=False), flush=True)
        return int("error" in row)
    args.paths = args.paths or list(CPU_PATHS if args.category == "agent" else PATHS)
    logs, motion = RunLogs(out=args.out), None
    print(f"Logs: {logs.directory}", flush=True)
    payload = {"schema_version": 3, "category": args.category, "results": [],
        "status": "in_progress", "started_at": datetime.now(timezone.utc).isoformat(),
        "logs": str(logs.directory), "python": sys.version, "platform": platform.platform(),
        "contract": {"shape": SHAPE, "dtype": "float16", "resize": "rational-bilinear-round-half-up-rgb8"},
        "latency_origin": "QPC immediately before successful Present(1); submission age, not photon age",
        "instrumentation": "marker decode and completion barriers included; VRAM is device-wide",
        "workload": args.workload, "requested_source_fps": args.motion_fps,
        "versions": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    save_results(args.out, payload)
    try:
        mode = display_mode()
        args.width, args.height = args.width or mode["width"], args.height or mode["height"]
        payload["display"] = mode
        if (args.width, args.height) != (mode["width"], mode["height"]):
            raise ValueError("requested resolution is not the current physical desktop mode")
        if args.motion_fps > mode["refresh_hz"] + 1:
            raise ValueError("requested cadence exceeds current physical refresh")
        if args.model:
            payload["model"] = {"path": str(args.model.resolve()), "sha256": sha256(args.model)}
            if payload["model"]["sha256"] != args.model_sha256:
                raise ValueError("model hash mismatch")
        guard = HealthGuard(logs)
        motion = VisualSource(logs, args, guard)
        motion.start()
        for index, path in enumerate(args.paths):
            print(f"[{path}] {'verify' if args.verify else 'measure'}", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", path,
                "--category", args.category, "--seconds", str(args.seconds), "--warmup", str(args.warmup),
                "--present-log", str(motion.present_log), "--codec", args.codec, "--quality", str(args.quality)]
            if args.model:
                command += ["--model", str(args.model.resolve()), "--model-sha256", args.model_sha256]
            if args.verify:
                command.append("--verify")
            row = spawn(path, args.seconds, args.warmup, args.verify, logs=logs, motion=motion,
                        index=index, command=command, result_parser=parse_result)
            payload["results"].append(row)
            save_results(args.out, payload)
            print(json.dumps(row), flush=True)
            if row.get("error") and not row.get("unavailable"):
                raise RuntimeError(f"{path} failed; see worker logs")
        guard.check(force=True)
        payload["status"] = "verified" if args.verify else "measured"
        payload["health"] = "no new WHEA records"
    except (Exception, KeyboardInterrupt) as exc:
        payload.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        print(payload["error"], file=sys.stderr, flush=True)
    finally:
        if motion is not None:
            try:
                motion.close()
            except Exception as exc:
                payload.update(status="failed", cleanup_error=str(exc))
            payload["source"] = motion.summary()
        save_results(args.out, payload)
    return int(payload["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
