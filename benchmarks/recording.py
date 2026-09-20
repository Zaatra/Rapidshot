"""Full-resolution conversion for **recording**, not for a model.

`section7.py` and `ai_ingestion.py` both measure one job: screen pixels to a
(1, 3, 640, 640) tensor. The resize is part of that job, and a large part of
why the converter paths are cheap there -- 2.46 MB crosses instead of a
16.38 MB frame. Nothing in either harness says what a *recording* pipeline
costs, where the output is the whole frame at capture resolution and the
consumer is a video encoder rather than a model.

That gap is why this exists. `GpuConverter(pixel_format="nv12")` has been
correct since 2.6 and has never been measured.

**What an encoder actually wants.** 8-bit 4:2:0. The GPU path produces NV12
(interleaved chroma); OpenCV produces I420 (planar). Both are 4:2:0 at
`w*h*3/2` bytes and both are accepted by every encoder here; the plane layout
differs and neither pays for the other's. Both are configured for **BT.601
limited range**, which is what `cv2.cvtColor` implements, so the two paths
compute the same picture rather than two different standards.

**Where the output lands is the whole point.** A GPU encoder (NVENC, AMF,
Quick Sync, Media Foundation) takes a surface that is already in VRAM: the
`-resident` rows stop there, because a copy to system memory would be work
the encoder does not need. A CPU encoder (x264, the OpenCV writer, ffmpeg on
a pipe) needs the bytes in system memory: the `-readback` row pays for that
copy and is the honest comparison against the CPU route. They are different
destinations, so they are separate rows rather than one number.

**The floors matter more than the totals.** `capture-*-only` does the capture
and nothing else, so conversion cost is the difference between a row and its
floor, rather than a number that quietly includes capture.

    python benchmarks/recording.py --verify --with-motion
    python benchmarks/recording.py --with-motion --seconds 8 --repeat 1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
# A worker is launched as `python benchmarks/recording.py`, so the repository
# root is not on its path and `import rapidshot` would find nothing.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import machine_inventory
import result_store
from result_store import CaseIdentity
from result_validation import should_stop
from ai_ingestion import (HealthGuard, MotionSource, PathUnavailable, RunLogs,
                          save_results, spawn)

#: Every row this benchmark can produce. Order is the order they are measured.
PATHS = ("capture-bgra-only", "capture-bgr-cpu", "cpu-cv2-i420",
         "gpu-nv12-resident", "gpu-nv12-readback")

#: 4:2:0 is half a byte of chroma per pixel on top of a byte of luma.
YUV420_BYTES_PER_PIXEL = 1.5

#: The conversion both sides are configured for. cv2.cvtColor implements
#: BT.601 limited range and offers no choice, so the GPU is told to match it;
#: comparing BT.709 output against BT.601 output would be comparing standards,
#: not implementations.
MATRIX = "bt601"

REQUIRED = ("frames", "fps", "elapsed_seconds")


def _capture_bgra():
    """Frames as captured, released immediately. The floor for the GPU paths."""
    import rapidshot

    camera = rapidshot.create(output_color="BGRA")

    def produce():
        frame = camera.grab_frame()
        if frame is None:
            return None
        frame.release()
        return 0

    return produce, camera.release, {"bytes_to_cpu_per_frame": 0,
                                     "note": "capture only; no conversion"}


def _capture_bgr_cpu():
    """A BGR frame in system memory. The floor for the OpenCV route."""
    import numpy as np
    import rapidshot

    camera = rapidshot.create(output_color="BGR")

    def produce():
        image = camera.grab()
        if image is None:
            return None
        nbytes = np.asarray(image).nbytes
        _return_to_pool(image)
        return nbytes

    return produce, camera.release, {"note": "capture only; BGR in system memory"}


def _cpu_cv2_i420():
    """What `examples/capture_to_video.py` pays for today, made explicit.

    `cv2.VideoWriter.write` takes BGR and converts to 4:2:0 on the CPU for
    every frame; doing it here with `cvtColor` measures that conversion
    instead of hiding it inside the writer along with the encode.
    """
    import rapidshot

    try:
        import cv2
    except ImportError as exc:
        raise PathUnavailable(f"OpenCV is not installed: {exc}") from None

    import numpy as np

    camera = rapidshot.create(output_color="BGR")
    state = {"last": None}

    def produce():
        image = camera.grab()
        if image is None:
            return None
        # `grab()` hands back a PooledBuffer, which OpenCV 5 refuses outright
        # ("src is not a numpy array"). asarray() is a view, not a copy, so
        # this is the wrapping a caller has to do rather than a cost added to
        # the measurement.
        yuv = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2YUV_I420)
        _return_to_pool(image)
        state["last"] = yuv
        return yuv.nbytes

    return produce, camera.release, {
        "layout": "i420", "matrix": MATRIX, "state": state,
        "note": "cv2.cvtColor on the CPU, the conversion cv2.VideoWriter does per frame"}


def _return_to_pool(buffer):
    """Hand a pooled buffer back, as any real caller must.

    Part of the library's per-frame cost, so it belongs inside the timed work
    -- `compare_libraries.py` counts it the same way. A plain array has no
    `release`, so this is a no-op for anything else.
    """
    release = getattr(buffer, "release", None)
    if release is not None:
        release()


def _gpu_nv12(readback: bool, verify: bool = False):
    """`GpuConverter(pixel_format="nv12")` at capture resolution.

    ``process()`` waits on its D3D12 fence before returning, so the time
    recorded is a converted frame *finished* in VRAM, not one submitted.
    """
    import rapidshot
    from rapidshot import native

    if not native.is_available() or rapidshot.GpuConverter is None:
        raise PathUnavailable("the native extension is not available")

    camera = rapidshot.create(output_color="BGRA")
    state = {"converter": None, "rgb_converter": None, "last": None,
             "reference_rgb": None, "size": None}

    def produce():
        frame = camera.grab_frame()
        if frame is None:
            return None
        with frame:
            if state["converter"] is None:
                if frame.width % 2 or frame.height % 2:
                    raise PathUnavailable(
                        f"4:2:0 needs even dimensions; this display is "
                        f"{frame.width}x{frame.height}")
                state["size"] = (frame.width, frame.height)
                state["converter"] = rapidshot.GpuConverter(
                    frame, state["size"], pixel_format="nv12", matrix=MATRIX)
            tensor = state["converter"].process(frame)
            if verify:
                # The reference comes off the *same held frame*: a second
                # converter at the same size with no resampling, so a
                # mismatch is the colour conversion and nothing else. Both
                # readbacks are outside any timed loop.
                if state["rgb_converter"] is None:
                    state["rgb_converter"] = rapidshot.GpuConverter(
                        frame, state["size"], dtype="float32", layout="nhwc",
                        normalize=True)
                state["reference_rgb"] = state["rgb_converter"].process(frame).numpy()[0]
                state["last"] = tensor.numpy()
                return int(state["last"].nbytes)
            if not readback:
                return 0
            plane = tensor.numpy()
            state["last"] = plane
            return int(plane.nbytes)

    def teardown():
        state.update(converter=None, rgb_converter=None, last=None,
                     reference_rgb=None)
        camera.release()

    where = ("copied to system memory for a CPU encoder" if readback
             else "left in VRAM for a GPU encoder")
    return produce, teardown, {"layout": "nv12", "matrix": MATRIX, "state": state,
                               "note": f"GPU conversion, result {where}"}


ADAPTERS = {
    "capture-bgra-only": lambda verify=False: _capture_bgra(),
    "capture-bgr-cpu": lambda verify=False: _capture_bgr_cpu(),
    "cpu-cv2-i420": lambda verify=False: _cpu_cv2_i420(),
    "gpu-nv12-resident": lambda verify=False: _gpu_nv12(False, verify),
    "gpu-nv12-readback": lambda verify=False: _gpu_nv12(True, verify),
}


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------

def nv12_reference(rgb, np):
    """NV12 codes for an (H, W, 3) float32 RGB image in 0..1, BT.601 limited.

    float64 throughout and written from the standard, so it shares no
    arithmetic with the shader it checks. Chroma is the mean of each 2x2
    block, which is what the kernel averages.
    """
    kr, kb = 0.299, 0.114
    kg = 1.0 - kr - kb
    rgb = rgb.astype(np.float64)
    y = kr * rgb[..., 0] + kg * rgb[..., 1] + kb * rgb[..., 2]
    height, width = y.shape
    block = rgb.reshape(height // 2, 2, width // 2, 2, 3).mean(axis=(1, 3))
    yb = kr * block[..., 0] + kg * block[..., 1] + kb * block[..., 2]
    cb = (block[..., 2] - yb) / (2.0 * (1.0 - kb))
    cr = (block[..., 0] - yb) / (2.0 * (1.0 - kr))
    luma = np.rint(16.0 + 219.0 * y)
    chroma = np.rint(128.0 + 224.0 * np.stack([cb, cr], axis=-1))
    plane = np.empty((height * 3 // 2, width), dtype=np.float64)
    plane[:height] = luma
    plane[height:] = chroma.reshape(height // 2, width)
    return np.clip(plane, 0, 255)


def compare_codes(got, expected, np):
    """Max code deviation and the fraction that match exactly.

    Both, not either: the GPU rounds in float32 and a tie can land one code
    either side, but a *systematic* off-by-one would sit inside a max-1 bound
    while failing the exact-match fraction.
    """
    diff = np.abs(got.astype(np.float64) - expected)
    return {"max_code_deviation": float(diff.max()),
            "fraction_exact": float((diff == 0).mean())}


def verify_path(path: str) -> dict:
    """One frame through the path, checked against an independent reference.

    The GPU rows are checked properly. The CPU row is not: `cv2.cvtColor` is
    the reference implementation for its own output, so checking it against a
    re-implementation of the same standard would only restate it. Its shape
    and size are checked, and that is said rather than implied.
    """
    import numpy as np

    result = {"path": path, "matrix": MATRIX}
    produce = teardown = None
    try:
        produce, teardown, meta = ADAPTERS[path](verify=True)
        result.update({k: v for k, v in meta.items() if k != "state"})
        deadline = time.perf_counter() + 20.0
        produced = None
        while produced is None and time.perf_counter() < deadline:
            produced = produce()
        if produced is None:
            raise RuntimeError("no frame to verify")

        state = meta.get("state") or {}
        if path.startswith("gpu-nv12"):
            got, rgb = state["last"], state["reference_rgb"]
            width, height = state["size"]
            comparison = compare_codes(got, nv12_reference(rgb, np), np)
            result.update(comparison)
            result["verified"] = bool(comparison["max_code_deviation"] <= 1
                                      and comparison["fraction_exact"] >= 0.99)
            result["shape"] = list(got.shape)
            result["expected_bytes"] = int(width * height * YUV420_BYTES_PER_PIXEL)
            if got.nbytes != result["expected_bytes"]:
                result["verified"] = False
                result["error"] = (f"NV12 is {got.nbytes} bytes; "
                                   f"{width}x{height} 4:2:0 is "
                                   f"{result['expected_bytes']}")
        elif path == "cpu-cv2-i420":
            last = state.get("last")
            result["checked"] = ("shape and size only; cv2.cvtColor is the "
                                 "reference implementation for its own output")
            result["shape"] = list(last.shape)
            result["expected_bytes"] = int(_capture_size(last) * YUV420_BYTES_PER_PIXEL)
            result["verified"] = bool(last.nbytes == result["expected_bytes"])
        else:
            result["verified"] = True
            result["checked"] = "capture only; there is no conversion to check"
    except PathUnavailable as exc:
        result["skipped"] = str(exc)
    except Exception as exc:                     # noqa: BLE001 - recorded, not handled
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if teardown is not None:
            try:
                teardown()
            except Exception as exc:             # noqa: BLE001
                result.setdefault("cleanup_error", f"{type(exc).__name__}: {exc}")
    return result


def run_path(path: str, seconds: float, warmup: int) -> dict:
    """Measure one path for ``seconds``, after ``warmup`` produced frames."""
    import numpy as np                            # noqa: F401 - adapters need it

    result = {"path": path, "matrix": MATRIX, "requested_seconds": seconds}
    produce = teardown = None
    try:
        produce, teardown, meta = ADAPTERS[path](verify=False)
        result.update({k: v for k, v in meta.items() if k != "state"})
        deadline = time.perf_counter() + 30.0
        done = 0
        while done < warmup and time.perf_counter() < deadline:
            if produce() is not None:
                done += 1
        if done < warmup:
            raise RuntimeError("the screen is not changing; no frames to warm up on")

        process = _process_probe()
        cpu_before = process.cpu_times() if process else None
        durations, bytes_seen, misses = [], 0, 0
        started = time.perf_counter()
        end = started + seconds
        while time.perf_counter() < end:
            call = time.perf_counter()
            produced = produce()
            elapsed = (time.perf_counter() - call) * 1000.0
            if produced is None:
                misses += 1
                continue
            durations.append(elapsed)
            bytes_seen = produced
        elapsed_seconds = time.perf_counter() - started
        if not durations:
            raise RuntimeError("no frames during the measurement window")

        durations.sort()
        result.update(
            frames=len(durations), misses=misses,
            elapsed_seconds=round(elapsed_seconds, 6),
            fps=round(len(durations) / elapsed_seconds, 1),
            bytes_to_cpu_per_frame=int(bytes_seen),
            ms_p50=round(_pct(durations, 0.50), 3),
            ms_p95=round(_pct(durations, 0.95), 3),
            ms_p99=round(_pct(durations, 0.99), 3),
            ms_min=round(durations[0], 3), ms_max=round(durations[-1], 3),
            ms_jitter_stdev=round(_stdev(durations), 3))
        if process is not None:
            after = process.cpu_times()
            cpu_seconds = ((after.user - cpu_before.user)
                           + (after.system - cpu_before.system))
            result["cpu_seconds"] = round(cpu_seconds, 3)
            result["cpu_percent"] = round(100.0 * cpu_seconds / elapsed_seconds, 1)
            result["cpu_ms_per_frame"] = round(cpu_seconds * 1000.0 / len(durations), 3)
            result["rss_mb"] = round(process.memory_info().rss / 1e6, 1)
    except PathUnavailable as exc:
        result["skipped"] = str(exc)
    except Exception as exc:                     # noqa: BLE001 - recorded, not handled
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if teardown is not None:
            try:
                teardown()
            except Exception as exc:             # noqa: BLE001
                result.setdefault("cleanup_error", f"{type(exc).__name__}: {exc}")
    return result


def _process_probe():
    try:
        import psutil
    except Exception:                            # noqa: BLE001 - optional
        return None
    return psutil.Process() if psutil is not None else None


def _pct(ordered, fraction):
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _stdev(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))


def parse_result(path, verify, returncode, stdout, stderr):
    """This benchmark's rows, not `ai_ingestion`'s tensor rows.

    The shared parser requires the tensor fields (`dtype_ok`, `content_ok`
    and so on) that a 4:2:0 frame has no equivalent of, so it would call every
    row here invalid.
    """
    try:
        row = json.loads(stdout.strip().splitlines()[-1])
        if not isinstance(row, dict) or row.get("path") != path:
            raise ValueError("wrong worker result shape or path")
    except (ValueError, IndexError) as exc:
        return {"path": path, "error": f"invalid worker result: {exc}",
                "returncode": returncode}
    if returncode:
        row.setdefault("error", f"worker exited with status {returncode}")
    if "error" not in row and "skipped" not in row:
        if verify:
            if row.get("verified") is not True:
                row["error"] = "worker did not report a verified conversion"
        else:
            numbers = all(isinstance(row.get(k), (int, float))
                          and not isinstance(row[k], bool)
                          and math.isfinite(row[k]) and row[k] > 0
                          for k in REQUIRED)
            if not numbers:
                row["error"] = "worker result is missing valid measurements"
    row["returncode"] = returncode
    return row


def _capture_size(i420):
    """Pixels in the frame an I420 buffer came from: it is 1.5 bytes each."""
    return int(i420.shape[0] * i420.shape[1] / YUV420_BYTES_PER_PIXEL)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=PATHS)
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--with-motion", action="store_true")
    parser.add_argument("--motion-fps", type=float, default=0.0)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--continue-on-failure", action="store_true")
    machine_inventory.add_policy_arguments(parser)
    result_store.add_history_arguments(parser)
    args = parser.parse_args(argv)

    if args.seconds <= 0 or args.warmup < 0:
        parser.error("seconds must be positive and warmup nonnegative")

    if args.worker:
        row = (verify_path(args.worker) if args.verify
               else run_path(args.worker, args.seconds, args.warmup))
        print(json.dumps(row, allow_nan=False), flush=True)
        return int("error" in row)

    policy, machine = machine_inventory.prepare_run(args)
    workload = "motion" if args.with_motion else "uncontrolled-desktop"
    configuration = ("recording-full-resolution-" + MATRIX
                     + ("-unpinned" if args.no_pin else "-pinned")
                     + ("-verify" if args.verify else ""))
    store = result_store.open_for_runner(
        "recording", root=args.history_root, resume=args.resume,
        disabled=args.no_history,
        metadata={"configuration": configuration, "workload": workload,
                  "seconds": args.seconds, "argv": sys.argv,
                  "machine_id": machine["machine_id"],
                  "display_fingerprint": machine["display_fingerprint"],
                  "cpu_policy": policy.as_dict(), "environment": machine})
    logs = RunLogs(args.log_dir, args.out)
    print(f"Diagnostics: {logs.directory}", flush=True)
    rows = []
    payload = {"schema_version": 1, "target": "encoder-ready 4:2:0 at capture resolution",
               "matrix": MATRIX, "results": rows, "logs": str(logs.directory),
               "environment": machine, "cpu_policy": policy.as_dict(),
               "platform": platform.platform()}
    motion = MotionSource(logs, args.motion_fps) if args.with_motion else None
    interrupted = False
    try:
        guard = HealthGuard(logs)
        if motion:
            motion.start()
        for index, path in enumerate(args.paths):
            identity = CaseIdentity(benchmark="recording", path=path,
                                    configuration=configuration, workload=workload,
                                    repeat=args.repeat)
            action, done = result_store.resume_decision(store, identity,
                                                        retry_failed=args.retry_failed)
            if action == "skip":
                print(f"  [{path}] already {done}", flush=True)
                continue
            print(f"  [{path}] ...", flush=True)
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--worker", path, "--seconds", str(args.seconds),
                       "--warmup", str(args.warmup)]
            if args.verify:
                command.append("--verify")
            with result_store.case_context(
                    store, identity, retry=action == "retry",
                    required=() if args.verify else REQUIRED) as case:
                row = spawn(path, args.seconds, args.warmup, args.verify, logs=logs,
                            motion=motion, index=index, command=command,
                            result_parser=parse_result)
                guard.after_case(row)
                case.result = row
            if case.record is not None:
                row = dict(row, case_status=case.record.status)
            rows.append(row)
            save_results(args.out, payload)
            if case.record is not None and should_stop(
                    case.record.status, case.record.reasons,
                    continue_on_failure=args.continue_on_failure):
                raise RuntimeError(f"{path} recorded {case.record.status}: "
                                   f"{'; '.join(case.record.reasons)}")
            print("    " + _one_line(row), flush=True)
    except KeyboardInterrupt:
        payload["error"] = "interrupted"
        interrupted = True
    except Exception as exc:                     # noqa: BLE001
        payload["error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["health"] = "no new WHEA records"
    finally:
        if motion is not None:
            try:
                motion.close()
            except Exception as exc:             # noqa: BLE001
                payload.setdefault("error", f"motion cleanup failed: {exc}")
            payload["motion"] = motion.summary()
    save_results(args.out, payload)
    if "error" in payload:
        print(f"ERROR: {payload['error']}", file=sys.stderr, flush=True)
    failed = "error" in payload or any("error" in row for row in rows)
    logs.event("parent-finished", failed=failed, completed_paths=len(rows))
    if args.out:
        print(f"Wrote {args.out}", flush=True)
    return 130 if interrupted else int(failed)


def _one_line(row):
    if "skipped" in row:
        return f"SKIPPED: {row['skipped']}"
    if "error" in row:
        return f"ERROR: {row['error']}"
    if "verified" in row:
        detail = row.get("max_code_deviation")
        return ("verified" if row["verified"] else "NOT verified") + (
            f"; max code deviation {detail}" if detail is not None else "")
    return (f"{row['fps']:.1f} fps; p50 {row['ms_p50']:.3f} ms; "
            f"CPU {row.get('cpu_ms_per_frame', float('nan')):.2f} ms/frame; "
            f"{row['bytes_to_cpu_per_frame'] / 1e6:.2f} MB to CPU")


if __name__ == "__main__":
    raise SystemExit(main())
