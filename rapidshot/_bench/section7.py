"""Supervised section 7 benchmarks. Desktop/GPU work happens only in workers."""
import argparse
import contextlib
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

from .ai_ingestion import (RunLogs, MotionSource, MotionError, HealthGuard, _child_options, spawn,
                          save_results, stage)
from .benchmark_contract import (PIPELINE_TOLERANCE_RGB8, SHAPE, TENSOR_GEOMETRY, PresentLog,
                                canonical_rgb, percentiles, qpc_clock, sha256)
from .section7_adapters import Adapter, PATHS, CPU_PATHS
from . import detection as detection_module
from . import machine_inventory
from . import scenes as scenes_module
from . import result_store
from . import telemetry as telemetry_module
from .result_store import CaseIdentity
from .result_validation import should_stop

from ._paths import REPO, WORK, worker_command

MODULE = "rapidshot._bench.section7"
BUILT_SOURCE = (REPO / "native" / "target" / "release" / "latency_source.exe"
                if REPO is not None else None)


def find_source(built=BUILT_SOURCE):
    """The test source to run: a local build first, then the rapidshot-native wheel's.

    A local build wins because whoever rebuilt it is testing *that* build, the
    same precedence `rapidshot.native` gives the development extension. Without
    either, the local path is returned so the error names what to build.
    """
    if built is not None and built.is_file():
        return built
    try:
        import rapidshot_native
        return Path(rapidshot_native.latency_source_path())
    except (ImportError, AttributeError, FileNotFoundError):
        return built


SOURCE = find_source()


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


class VisualSource(MotionSource):
    def __init__(self, logs, args, guard, scene=None):
        super().__init__(logs, args.motion_fps)
        self.args, self.guard = args, guard
        self.scene = scene
        self.present_log = logs.directory / "presents.jsonl"

    def start(self):
        if not SOURCE.is_file():
            raise RuntimeError("no test source: pip install \"rapidshot-native>=0.2.1\", or build one with "
                               "cargo build --release --bin latency_source --manifest-path native/Cargo.toml")
        self.stdout = (self.logs.directory / "motion.stdout.log").open("wb")
        self.stderr = (self.logs.directory / "motion.stderr.log").open("wb")
        self.reader = (self.logs.directory / "motion.stdout.log").open(encoding="utf-8")
        command = [str(SOURCE), str(self.args.width), str(self.args.height),
                   str(self.fps), self.args.workload, str(self.present_log)]
        if self.scene is not None:
            # The manifest is read here and passed as numbers rather than as a
            # path the source would have to parse. There is no JSON reader in
            # that crate and four integers do not justify adding one.
            step_x, step_y = self.scene["workload_steps"][self.args.workload]
            command += [str(Path(self.args.scene) / self.scene["background"]),
                        str(self.scene["canvas_width"]),
                        str(self.scene["canvas_height"]), str(step_x), str(step_y)]
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
    # Read before the imports below, not after. NumPy's BLAS, ONNX Runtime and
    # CuPy all size their thread pools from the affinity they see when they
    # initialise; a pool built for the whole machine and then run on eight
    # cores is slower than either choice made consistently, and the recording
    # would claim the pinned configuration either way.
    result["affinity"] = machine_inventory.verify_affinity(args.expect_affinity)
    adapter = tracker = cp = None
    try:
        stage("imports", path=args.worker)
        import numpy as np
        import psutil
        if args.category != "agent":
            import cupy as cp
        now, frequency = qpc_clock()
        tracker = PresentLog(args.present_log, frequency)
        infer = None
        if args.category == "inference":
            from .ai_pipeline import Inference
            if sha256(args.model) != args.model_sha256:
                raise ValueError("model hash mismatch")
            stage("inference-starting")
            infer = Inference(args.model, cp, np, allow_cpu_nodes=args.allow_cpu_nodes)
            result["model_input_overrides"] = infer.overrides
            result["cpu_nodes_allowed"] = infer.allow_cpu_nodes
        geometry = contract = None
        if infer is not None:
            geometry = detection_module.Geometry(
                args.width or SHAPE[-1], args.height or SHAPE[-2], SHAPE[-1],
                mode=args.geometry)
            contract = detection_module.DetectionContract(
                model_sha256=args.model_sha256, model_size=SHAPE[-1],
                dtype="float16", geometry_mode=args.geometry,
                confidence_threshold=args.conf, iou_threshold=args.iou)
            result["detection_contract"] = contract.as_dict()
            result["geometry"] = geometry.as_dict()
        detected = {"frames": 0, "total": 0, "min": None, "max": None}
        stage("adapter-starting", path=args.worker)
        adapter = Adapter(args.worker, cp, np, verify=args.verify, agent=args.category == "agent")

        def one():
            sample = adapter.capture()
            if sample is None:
                return None
            tensor_ready = now()
            t0 = time.perf_counter()
            if infer is not None:
                # Ends at boxes/classes/scores on the host, not at the forward
                # pass. That is the point an application can branch on them,
                # and the NMS and transfer in between are real work that scales
                # with how much is on screen.
                detections, binding = infer.detect(sample.tensor, geometry, contract)
                if args.verify:
                    outputs = binding.copy_outputs_to_cpu()
                    if not outputs or not all(x.size and np.isfinite(x).all() for x in outputs):
                        raise ValueError("model output is empty or nonfinite")
                    _verify_detections(result, outputs[0], detections, geometry,
                                       contract, np)
                sample.stages["inference_and_postprocess_ms"] = (
                    time.perf_counter() - t0) * 1000
                result["postprocess_location"] = infer.postprocess_location
                count = len(detections)
                detected["frames"] += 1
                detected["total"] += count
                detected["min"] = count if detected["min"] is None else min(
                    detected["min"], count)
                detected["max"] = count if detected["max"] is None else max(
                    detected["max"], count)
            elif args.category == "agent":
                from .agent_pipeline import encode
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
                # The measured loop runs `pipeline_rgb` -- cv2's fixed-point
                # bilinear on CPU paths, the exact contract on GPU ones -- so a
                # bit-identical check can only ever pass the GPU paths. Compare
                # in RGB8, which FP16 represents exactly for k/255, against the
                # documented tolerance, and record the deviation actually seen.
                reference = canonical_rgb(sample.reference, np).astype(np.int16)
                actual = cp.asnumpy(sample.tensor)
                if actual.shape != SHAPE or actual.dtype != np.float16:
                    raise ValueError(f"tensor is {actual.shape} {actual.dtype}, expected {SHAPE} float16")
                rgb8 = np.rint(actual[0].astype(np.float32) * 255).transpose(1, 2, 0).astype(np.int16)
                deviation = np.abs(rgb8 - reference)
                result.update(verify_max_rgb8_deviation=int(deviation.max()),
                              verify_fraction_off_by_one=float((deviation > 0).mean()))
                if deviation.max() > PIPELINE_TOLERANCE_RGB8:
                    raise ValueError(f"tensor deviates from the canonical reference by "
                                     f"{int(deviation.max())} RGB8 levels; tolerance is "
                                     f"{PIPELINE_TOLERANCE_RGB8}")
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
            if infer is not None and detected["frames"]:
                result["detections"] = {
                    "frames_with_output": detected["frames"],
                    "total": detected["total"],
                    "mean_per_frame": detected["total"] / detected["frames"],
                    "min_per_frame": detected["min"], "max_per_frame": detected["max"]}
                if detected["total"] == 0:
                    # Timing an empty scene measures the cheapest possible
                    # postprocess and none of the work a real one does.
                    result.setdefault("warnings", []).append(
                        "no detections in any frame; the scene contains nothing the "
                        "model recognises, so postprocessing cost is not "
                        "representative")
            if cp is not None:
                free, total = cp.cuda.runtime.memGetInfo()
                result["device_vram_used_mb"] = (total-free)/1e6
            if args.samples_out:
                # After the measured window on purpose. Writing a few hundred
                # kilobytes is milliseconds of I/O, and doing it inside the loop
                # would put that cost into the timings it is recording.
                write_samples(args.samples_out, samples, ages, tensor_ages, calls, stages)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        # Both mean "cannot run on this machine", not "broke": CrossAdapterRequired
        # from the library, PathUnavailable from the harness's own refusals.
        if type(exc).__name__ in ("CrossAdapterRequired", "PathUnavailable"):
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


def _verify_detections(result, raw, detections, geometry, contract, np):
    """Check the timed path against a host reference on the same raw output.

    The device postprocess and the host one must find the same objects: if they
    do not, the fast path is fast because it is doing something else. Run once,
    during verification, outside the measured loop.
    """
    reference = detection_module.postprocess(raw, geometry, contract, np)
    parity = detection_module.detection_parity(reference, detections)
    result["detection_parity"] = parity.as_dict()
    result["scene_fingerprint"] = detection_module.scene_fingerprint(reference)
    if not parity.ok:
        raise ValueError("timed detections disagree with the host reference on the "
                         f"same model output: {parity.as_dict()}")


def write_samples(path, samples, ages, tensor_ages, calls, stages):
    """Emit the per-frame rows the percentiles were reduced from.

    Percentiles are a summary, and a summary cannot be re-examined: it cannot
    answer whether a tail came from three bad frames or three hundred, and it
    cannot be recomputed with a different definition later. The raw rows can.

    **Two populations, kept apart.** Ages are per *unique frame*; call durations
    and stage timings are per *call*, and a call that returns a duplicate frame
    still has both. Zipping them would silently pair a frame's age with some
    other frame's stages, so calls go in their own array with its own length.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for (frame_id, _, _), age, tensor_age in zip(samples, ages, tensor_ages):
            stream.write(json.dumps({"kind": "unique_frame", "frame_id": frame_id,
                                     "present_to_ready_ms": age,
                                     "present_to_tensor_ms": tensor_age}) + "\n")
        for index, duration in enumerate(calls):
            row = {"kind": "call", "index": index, "call_ms": duration}
            for name, values in stages.items():
                if index < len(values):
                    row[name] = values[index]
            stream.write(json.dumps(row) + "\n")


#: What a pixel-age row must contain to be a measurement at all. Mirrors the
#: checks `parse_result` already made, so validation is a superset of them
#: rather than a second, competing opinion about what a valid row is.
PIXEL_AGE_REQUIRED = ("unique_frames", "unique_fps", "elapsed_seconds")


def load_scene(args, payload):
    """Read a scene pack, and refuse one that cannot support what is being run.

    An inference run against the procedural pattern finds nothing in any frame
    and times the cheapest possible postprocess. That is not a slow result, it
    is a meaningless one, so it is refused rather than recorded -- with the
    command that fixes it.
    """
    if args.scene is None:
        if args.category == "inference":
            raise ValueError(
                "an inference run needs --scene: without one the source draws a "
                "frame-ID pattern containing no object any detector recognises, so "
                "every frame would return zero detections and the timings would "
                "describe an empty screen. Build one with "
                "`python benchmarks/scenes.py --out build/scenes/default "
                "--model build/section7/model/yolo11n.onnx`.")
        return None

    manifest = scenes_module.load_manifest(args.scene)
    if (manifest["capture_width"], manifest["capture_height"]) != (args.width,
                                                                   args.height):
        raise ValueError(
            f"scene was built for {manifest['capture_width']}x"
            f"{manifest['capture_height']} and this run is "
            f"{args.width}x{args.height}; object sizes are chosen for a capture "
            "resolution and a mismatched scene is not the one that was verified")

    payload["scene"] = {key: manifest[key] for key in
                        ("scene_id", "canvas_width", "canvas_height", "seed",
                         "workload_steps", "verified_classes")}
    payload["scene"]["objects"] = len(manifest["objects"])

    detections = Path(args.scene) / scenes_module.DETECTIONS_NAME
    if detections.is_file():
        report = json.loads(detections.read_text(encoding="utf-8"))
        payload["scene"]["verification"] = {
            "usable": report.get("usable"), "problems": report.get("problems", []),
            "model_sha256": report.get("model_sha256"),
            "detections_per_frame": {name: summary["mean_detections"]
                                     for name, summary in
                                     report.get("workloads", {}).items()}}
        if not report.get("usable"):
            raise ValueError(f"scene {manifest['scene_id']} did not pass verification: "
                             f"{'; '.join(report.get('problems', []))}")
    else:
        # Usable-looking and unchecked are not the same thing. An unverified
        # scene may still be run -- it is a valid capture workload -- but it
        # cannot support a detection claim, and the record has to say so.
        payload["scene"]["verification"] = {
            "usable": None,
            "problems": ["this scene has never been put through a detector; "
                         "run scenes.py --model to find out whether anything in "
                         "it is detected at all"]}
        if args.category == "inference":
            raise ValueError(
                f"scene {manifest['scene_id']} is unverified and this is an "
                "inference run; re-run scenes.py with --model first")
    return manifest


def start_telemetry(args):
    """Begin conditions sampling for one case, or return ``None`` with a reason.

    Started outside the measured window and sampled at a rate meant for
    conditions rather than profiling. The sampler is rooted at this process, so
    the motion source and every worker count as *this benchmark's* CPU rather
    than as background load.
    """
    if args.no_telemetry:
        return None
    try:
        return telemetry_module.TelemetrySampler(
            interval=args.telemetry_interval, tree_root_pid=os.getpid()).start()
    except Exception as exc:  # noqa: BLE001
        print(f"  telemetry unavailable: {type(exc).__name__}: {exc}", flush=True)
        return None


def stop_telemetry(sampler):
    if sampler is None:
        return None
    try:
        return sampler.stop()
    except Exception as exc:  # noqa: BLE001
        print(f"  telemetry stop failed: {type(exc).__name__}: {exc}", flush=True)
        return None


def affinity_reasons(row, policy):
    """Flag a worker that did not run on the cores the parent asked for.

    Affinity is inherited, so this normally reports nothing -- but "normally"
    is an assumption, and a worker that quietly ran on the whole machine while
    the recording claims a pinned configuration produces numbers that cannot be
    compared with anything, and nothing else would notice.
    """
    affinity = row.get("affinity")
    if not isinstance(affinity, dict):
        return []
    if not affinity.get("available", False):
        return [f"the worker could not read its own affinity "
                f"({affinity.get('reason')}), so the CPU policy is unverified"]
    if policy.effective_mask is not None and \
            affinity.get("process_mask") != policy.effective_mask:
        return [f"worker affinity {affinity.get('process_mask_hex')} does not match "
                f"the {hex(policy.effective_mask)} the parent applied"]
    return []


def drift_reasons(check):
    """Turn a changed machine into reasons, not into a silent verdict.

    A display mode that changed, a laptop that came off mains or a rebuilt
    extension makes the cases either side of it measurements of different
    things. That is recorded against the case so no automatic comparison can
    step over it.
    """
    return [f"{entry['field']} changed mid-run: {entry['before']!r} -> "
            f"{entry['after']!r}; cases on either side are not comparable"
            for entry in check.blocking]


def display_target(environment, mode):
    """Which physical output everything in this run is pointed at.

    The motion source, every capture backend and the refresh check all assume
    the primary display. Recording its identity is what lets a later reader
    tell whether two runs used the same panel -- and the exact refresh is worth
    having beside the rounded one, because source pacing is compared against
    the rounded figure and a 165 Hz panel does not present at 165.000 Hz.
    """
    displays = environment.get("displays", {})
    if not displays.get("available", False):
        return {"available": False,
                "reason": displays.get("reason", "displays were not discovered")}
    primary = next((output for output in displays.get("outputs", [])
                    if output.get("primary")), None)
    if primary is None:
        return {"available": False, "reason": "no primary output was reported"}
    exact = (primary.get("refresh_rate") or {}).get("hz")
    target = {
        "available": True,
        "monitor_device_path": primary.get("monitor_device_path"),
        "gdi_device_name": primary.get("gdi_device_name"),
        "adapter_luid": primary.get("adapter_luid"),
        "adapter_device_path": primary.get("adapter_device_path"),
        "rotation_degrees": primary.get("rotation_degrees"),
        "scale_percent": primary.get("scale_percent"),
        "reported_refresh_hz": mode.get("refresh_hz"),
        "exact_refresh_hz": exact,
    }
    if exact is not None and abs(exact - mode.get("refresh_hz", 0)) > 1.0:
        target["refresh_disagreement"] = (
            f"EnumDisplaySettingsW reports {mode.get('refresh_hz')} Hz and the "
            f"driver programmed {exact:.4f} Hz; source pacing uses the former")
    if displays.get("physical_pixel_warning"):
        target["physical_pixel_warning"] = displays["physical_pixel_warning"]
    return target


def case_configuration(args, payload):
    """Everything that changes what is being measured, as one identity string.

    Two runs that differ here are different cases, not repeats of one. Getting
    this wrong is how a 1080p number ends up averaged with a 1440p one.
    """
    parts = [f"{args.width}x{args.height}@{args.motion_fps:g}", args.category,
             payload["contract"]["dtype"]]
    if args.category == "inference":
        parts.append("model-" + payload["model"]["sha256"][:12])
        # The detection contract decides what work is being timed, so two runs
        # that disagree on it are different cases rather than repeats.
        parts.append(detection_module.DetectionContract(
            model_sha256=payload["model"]["sha256"], model_size=SHAPE[-1],
            dtype=payload["contract"]["dtype"], geometry_mode=args.geometry,
            confidence_threshold=args.conf,
            iou_threshold=args.iou).configuration_suffix())
        if args.allow_cpu_nodes:
            parts.append("cpu-nodes-allowed")
    if args.category == "agent":
        parts.append(f"{args.codec}{args.quality}")
    # --no-pin is a configuration, not the absence of one. Pooling a pinned and
    # an unpinned run on a hybrid CPU is how this suite reported false
    # regressions up to 2.57x against its own output.
    parts.append("unpinned" if args.no_pin else "pinned")
    if payload.get("scene"):
        # A scene with more objects in it does more postprocessing work per
        # frame, so two runs against different scenes are not repeats.
        parts.append(payload["scene"]["scene_id"])
    if args.verify:
        # A verification run produces correctness evidence and no timings. It is
        # kept as its own configuration so it can never be mistaken for, or
        # pooled with, a measurement of the same path.
        parts.append("verify")
    return "-".join(parts)


def open_history(args, payload, environment, policy):
    return result_store.open_for_runner("section7", root=args.history_root,
                                        resume=args.resume, disabled=args.no_history,
                                        metadata={
        "category": args.category, "workload": args.workload,
        "requested_source_fps": args.motion_fps, "seconds": args.seconds,
        "python": sys.version, "platform": platform.platform(),
        "contract": payload["contract"], "argv": sys.argv,
        "machine_id": environment["machine_id"],
        "display_fingerprint": environment["display_fingerprint"],
        "cpu_policy": policy.as_dict(),
        "environment": environment})


@contextlib.contextmanager
def case_context(store, identity, *, verify, retry=False):
    """A case to fill in, with this benchmark's validation contract applied."""
    with result_store.case_context(
            store, identity, retry=retry,
            required=() if verify else PIXEL_AGE_REQUIRED,
            end_to_end=None if verify else "present_to_ready_ms") as case:
        yield case


def read_sample_rows(path):
    """The worker's per-frame rows, or ``None`` if it never got far enough."""
    path = Path(path)
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def source_limited(motion, rates_before, row):
    """Flag a case whose throughput describes the source rather than the path.

    This is the failure ROADMAP § 7.0 records hitting three separate times: when
    the motion source cannot present faster than the fastest path reads, every
    path returns the source's rate and the run looks like a tie. The source
    reports its achieved rate continuously, so the check is simply whether its
    slowest observed window during this case was above what the path returned.

    Contamination, not invalidity. The measurement happened; what it measured is
    partly the apparatus, and that is for a reader to weigh with the reason in
    hand rather than for this function to decide by discarding it.
    """
    fps = row.get("unique_fps")
    if not isinstance(fps, (int, float)) or isinstance(fps, bool):
        return []
    observed = motion.rates[rates_before:]
    if observed:
        slowest, provenance = min(observed), "during this case"
    elif motion.rates:
        # The source reports periodically, so a short case can end without a
        # fresh window. The last rate before it started is weaker evidence and
        # is labelled as such rather than being passed off as a measurement of
        # these seconds -- or, worse, than treating the case as unknowable and
        # contaminating every short run.
        slowest, provenance = motion.rates[rates_before - 1 if rates_before else 0], \
            "last observed before this case"
    else:
        return ["the source never reported an achieved rate, so throughput cannot be "
                "shown to be path-limited rather than source-limited"]
    if slowest <= fps:
        return [f"source presented as few as {slowest:.1f}/s ({provenance}) while this "
                f"path returned {fps:.1f} unique fps; throughput is source-limited"]
    return []


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
    parser.add_argument("--allow-cpu-nodes", action="store_true",
                        help="let ONNX Runtime place nodes it cannot run on CUDA on the CPU; "
                             "recorded in every result")
    parser.add_argument("--scene", type=Path,
                        help="a scene pack from benchmarks/scenes.py. Without one the "
                             "source draws the procedural pattern, which contains "
                             "nothing a detector recognises -- fine for pixel age, "
                             "useless for measuring detection.")
    parser.add_argument("--geometry", choices=("letterbox", "stretch"),
                        default=TENSOR_GEOMETRY,
                        help="how the source is mapped into the model's square input. "
                             f"This harness's tensor contract is {TENSOR_GEOMETRY!r} "
                             "and the postprocessing must invert the same transform, "
                             "so anything else is refused here.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="confidence threshold; part of the detection contract")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold; part of the detection contract")
    parser.add_argument("--codec", choices=("png", "jpeg"), default="png")
    parser.add_argument("--quality", type=int, default=90)
    parser.add_argument("--present-log", type=Path)
    parser.add_argument("--out", type=Path, default=WORK / "section7" / "last-run.json")
    parser.add_argument("--samples-out", type=Path,
                        help="worker-side: where to write the per-frame rows")
    parser.add_argument("--expect-affinity", type=lambda value: int(value, 16),
                        help="worker-side: the affinity mask the parent applied")
    parser.add_argument("--no-pin", action="store_true",
                        help="do not restrict to performance cores. Recorded as its "
                             "own configuration and never pooled with pinned runs.")
    parser.add_argument("--telemetry-interval", type=float,
                        default=telemetry_module.DEFAULT_INTERVAL,
                        help="seconds between conditions samples")
    parser.add_argument("--no-telemetry", action="store_true",
                        help="skip conditions sampling entirely")
    parser.add_argument("--repeat", type=int, default=1,
                        help="which repeat of this configuration this run is; part of case identity")
    result_store.add_history_arguments(parser)
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="keep measuring after an ordinary path failure. Hardware failures "
                             "still stop the suite.")
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
    if args.geometry != TENSOR_GEOMETRY:
        # Found live: the default was `letterbox` while `canonical_rgb`
        # stretches, so the detector was fed a vertically squashed frame and
        # the boxes were restored with the wrong inverse. Detections per frame
        # went from the scene's verified ~8 to 16.7, peaking at 37.
        parser.error(
            f"--geometry {args.geometry} does not match this harness's tensor "
            f"contract, which is {TENSOR_GEOMETRY}. The postprocessing inverts the "
            "transform the tensor was built with; mismatching them feeds the model "
            "one picture and restores boxes for another. Letterboxing would mean "
            "changing benchmark_contract.canonical_rgb, which is a different "
            "measurement and belongs in its own configuration.")
    if args.worker:
        row = worker(args)
        print(json.dumps(row, allow_nan=False), flush=True)
        return int("error" in row)
    args.paths = args.paths or list(CPU_PATHS if args.category == "agent" else PATHS)
    # Pinned here, in the parent, before any worker is spawned: affinity is
    # inherited, so this is the only place it can be applied early enough for
    # every child. Each worker verifies it independently anyway, because
    # "inherited" is an assumption until something checks.
    policy = machine_inventory.apply_cpu_policy("none" if args.no_pin else "performance")
    print(f"CPU policy: {policy.policy} on a {policy.topology} CPU"
          + (f" -> {hex(policy.effective_mask)}" if policy.effective_mask else "")
          + ("" if policy.verified or policy.policy == "none"
             else f" [{'; '.join(policy.reasons) or 'not applied'}]"), flush=True)
    environment = machine_inventory.discover_machine()
    print(f"Machine: {environment['machine_id']} "
          f"displays {environment['display_fingerprint']}", flush=True)
    logs, motion = RunLogs(out=args.out), None
    print(f"Logs: {logs.directory}", flush=True)
    payload = {"schema_version": 3, "category": args.category, "results": [],
        "status": "in_progress", "started_at": datetime.now(timezone.utc).isoformat(),
        "logs": str(logs.directory), "python": sys.version, "platform": platform.platform(),
        "contract": {"shape": SHAPE, "dtype": "float16", "resize": "rational-bilinear-round-half-up-rgb8"},
        "latency_origin": "QPC immediately before successful Present(1); submission age, not photon age",
        "instrumentation": "marker decode and completion barriers included; VRAM is device-wide",
        "workload": args.workload, "requested_source_fps": args.motion_fps,
        "environment": environment, "cpu_policy": policy.as_dict(),
        "versions": {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
        "native_loaded": machine_inventory.native_loaded(), "test_source": str(SOURCE)}
    # Opened before anything is measured, and deliberately not inside the try
    # below: a store that cannot be opened is a reason to stop, not a reason to
    # run the benchmark and discover afterwards that nothing was recorded.
    store = open_history(args, payload, environment, policy)
    if store is not None:
        payload["history"] = {"run_id": store.run_id, "run_dir": str(store.run_dir)}
        print(f"History: {store.run_dir}", flush=True)
    save_results(args.out, payload)
    try:
        mode = display_mode()
        args.width, args.height = args.width or mode["width"], args.height or mode["height"]
        payload["display"] = mode
        payload["display_target"] = display_target(environment, mode)
        if (args.width, args.height) != (mode["width"], mode["height"]):
            raise ValueError("requested resolution is not the current physical desktop mode")
        if args.motion_fps > mode["refresh_hz"] + 1:
            raise ValueError("requested cadence exceeds current physical refresh")
        if args.model:
            payload["model"] = {"path": str(args.model.resolve()), "sha256": sha256(args.model),
                                "cpu_nodes_allowed": args.allow_cpu_nodes}
            if payload["model"]["sha256"] != args.model_sha256:
                raise ValueError("model hash mismatch")
        scene = load_scene(args, payload)
        configuration = case_configuration(args, payload)
        payload["configuration"] = configuration
        guard = HealthGuard(logs)
        motion = VisualSource(logs, args, guard, scene)
        motion.start()
        for index, path in enumerate(args.paths):
            identity = CaseIdentity(benchmark="section7", path=path,
                                    configuration=configuration, workload=args.workload,
                                    repeat=args.repeat)
            action, done = result_store.resume_decision(store, identity,
                                                        retry_failed=args.retry_failed)
            retry = action == "retry"
            if action == "skip":
                hint = ("" if done in result_store.RESUME_SETTLED
                        else "; pass --retry-failed to measure it again")
                print(f"[{path}] already {done} in {store.run_id}{hint}", flush=True)
                continue
            print(f"[{path}] {'verify' if args.verify else 'measure'}", flush=True)
            samples_path = logs.directory / f"{index:02d}-{path}.samples.jsonl"
            command = worker_command(MODULE, "--worker", path,
                "--category", args.category, "--seconds", str(args.seconds), "--warmup", str(args.warmup),
                "--present-log", str(motion.present_log), "--codec", args.codec, "--quality", str(args.quality),
                "--samples-out", str(samples_path))
            if policy.effective_mask is not None:
                command += ["--expect-affinity", hex(policy.effective_mask)]
            if args.model:
                command += ["--model", str(args.model.resolve()), "--model-sha256", args.model_sha256,
                            "--geometry", args.geometry, "--conf", str(args.conf),
                            "--iou", str(args.iou), "--width", str(args.width),
                            "--height", str(args.height)]
            if args.allow_cpu_nodes:
                command.append("--allow-cpu-nodes")
            if args.verify:
                command.append("--verify")

            rates_before = len(motion.rates)
            sampler = start_telemetry(args)
            with case_context(store, identity, verify=args.verify, retry=retry) as case:
                # Assigned before anything else can fail, and mutated in place
                # afterwards: whatever the worker managed to report survives an
                # exception, which is usually the only clue about why it died.
                row = spawn(path, args.seconds, args.warmup, args.verify, logs=logs, motion=motion,
                            index=index, command=command, result_parser=parse_result)
                # The source checks the guard at most every 5 s, so the tail of
                # a case could end unchecked; this closes that before commit.
                guard.after_case(row)
                case.result = row
                case.samples = read_sample_rows(samples_path)
                conditions = stop_telemetry(sampler)
                if conditions is not None:
                    row["telemetry"] = conditions.as_dict()
                drift = machine_inventory.verify_environment(environment)
                row["environment_check"] = drift.as_dict()
                case.contamination = (source_limited(motion, rates_before, row)
                                      + telemetry_module.background_load_warning(conditions)
                                      + affinity_reasons(row, policy)
                                      + drift_reasons(drift))

            status = case.record.status if case.record is not None else (
                "failed" if row.get("error") and not row.get("unavailable") else "passed")
            reasons = case.record.reasons if case.record is not None else [row.get("error") or ""]
            row = dict(row, case_status=status)
            if case.record is not None:
                row["case"] = {"run_id": store.run_id, "case_id": case.record.case_id,
                               "attempt_id": case.record.attempt_id}
            payload["results"].append(row)
            save_results(args.out, payload)
            print(json.dumps(row), flush=True)
            if should_stop(status, reasons, continue_on_failure=args.continue_on_failure):
                raise RuntimeError(f"{path} recorded {status}: {'; '.join(reasons)}; "
                                   "see worker logs")
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
