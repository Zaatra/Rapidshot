"""How much inference a change-aware pipeline avoids, with duplicates controlled.

The earlier live benchmark could not answer this. Desktop Duplication delivers
on present, so it rarely hands a consumer a byte-identical frame, and the number
it did hand over swung from 214 of 255 to 0 of 61 across runs of one
configuration. With nothing duplicated there is nothing to avoid, and the
no-filter baseline ran the detector exactly as often as the gated arm.

So this fixture **re-presents the same state deliberately**. Each scheduled tick
either advances to a new dashboard state or repaints the current one, which puts
a present on the wire whose pixels are byte-identical to the last. The ratio is
a dial, and the marker in each state says which state a captured frame holds, so
a consumer that skips work can still be scored on whether it saw every state.

Three groups, the comparison the earlier benchmark was missing:

``none``
    Infer on every delivered frame. No filtering of any kind.
``full_compare``
    Compare the whole frame against a retained copy; infer when it differs.
``metadata``
    Take the byte-exact verdict capture already computed; infer when it says so.

The headline is not frames per second. It is how many inference calls were
avoided, how much CPU time that returned, whether any state went unseen, and how
long a new state took to reach a detection.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes as ct
from dataclasses import dataclass
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

from capture_timing_followup import frequency
from cost_adapters import CostAdapter
from monitor_target import select_monitor


ROOT = Path(__file__).resolve().parent
NS_SECOND = 1_000_000_000
NS_MS = 1_000_000
GROUPS = ("none", "full_compare", "metadata")
GROUP_LABELS = {"none": "No filtering", "full_compare": "Full-frame compare",
                "metadata": "Change metadata"}
METADATA_ARMS = ("rapidshot_exact", "rapidshot_tiled", "rapidshot_tight")


# --------------------------------------------------------------------------- #
# Detectors
# --------------------------------------------------------------------------- #

class SyntheticDetector:
    """Calibrated real CPU arithmetic, sized to a requested cost per frame.

    Not a sleep. Sleeping would model a slow consumer without consuming the
    cycles an inference actually costs, and cycles are the thing being saved.
    """

    def __init__(self, target_ms, width, height):
        self.target_ms = target_ms
        rng = np.random.default_rng(412_430)
        self.left = rng.random((192, 192), dtype=np.float32)
        self.right = rng.random((192, 192), dtype=np.float32)
        self.rounds = max(1, self._calibrate(target_ms))

    def _one(self, rounds):
        total = np.float32(0)
        for _ in range(rounds):
            total += (self.left @ self.right).sum()
        return total

    def _calibrate(self, target_ms):
        self._one(4)  # warm
        started = time.perf_counter_ns()
        self._one(16)
        per_round_ms = (time.perf_counter_ns() - started) / NS_MS / 16
        return round(target_ms / per_round_ms) if per_round_ms > 0 else 1

    def infer(self, pixels):
        # Touch the pixels so the frame is genuinely read, then do the work.
        checksum = int(pixels[::97, ::97, 0].sum())
        self._one(self.rounds)
        return (checksum,)

    def describe(self):
        return {"detector": "synthetic calibrated matmul", "target_ms": self.target_ms,
                "rounds_per_inference": self.rounds, "matrix": [192, 192],
                "thread_env": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


class YoloDetector:
    """The verified OpenCV Zoo YOLOX-S FP32 model, with a thread dial."""

    def __init__(self, threads, width, height):
        import cv2

        from whole_frame_task import MODEL_BYTES, MODEL_SHA256, YoloX

        self.model = YoloX()
        cv2.setNumThreads(threads)
        self.threads = cv2.getNumThreads()
        self.opencv = cv2.__version__
        self.bytes, self.sha256 = MODEL_BYTES, MODEL_SHA256
        self.model.infer(np.zeros((height, width, 4), dtype=np.uint8))

    def infer(self, pixels):
        return self.model.infer(pixels)

    def describe(self):
        return {"detector": "OpenCV Zoo YOLOX-S FP32", "threads": self.threads,
                "opencv": self.opencv, "model_bytes": self.bytes, "sha256": self.sha256}


class OrtYoloDetector:
    """The same verified YOLOX-S model through ONNX Runtime's CPU provider.

    Preprocessing and decode are the verified ones from ``whole_frame_task``;
    only the forward pass changes, so detections stay comparable with the
    OpenCV arm. ``intra_op_num_threads`` is the dial that makes a real model
    fast enough to stop being the only thing the benchmark measures.
    """

    def __init__(self, threads, width, height):
        import cv2
        import onnxruntime as ort

        from whole_frame_task import (ASSETS, INPUT_SIZE, MODEL_BYTES, MODEL_NAME, MODEL_SHA256,
                                      YoloX, verify_model)

        path = ASSETS / MODEL_NAME
        verify_model(path)
        self.helper = YoloX(path)      # verified preprocessing tables and decode
        self.input_size = INPUT_SIZE
        self.bytes, self.sha256 = MODEL_BYTES, MODEL_SHA256
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # ONNX Runtime's intra-op pool spin-waits between runs by default, so it
        # burns cycles on every thread whether or not an inference is happening.
        # Left on, the CPU-seconds column reflects the idle pool rather than the
        # work, and gating cannot show any saving at all.
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(path), options,
                                            providers=["CPUExecutionProvider"])
        if self.session.get_providers()[:1] != ["CPUExecutionProvider"]:
            raise RuntimeError(f"Unexpected provider: {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.requested_threads = threads
        self.ort_version = ort.__version__
        self.cv2_version = cv2.__version__
        self.blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self.measured_ms = None
        warm = np.zeros((height, width, 4), dtype=np.uint8)
        for _ in range(3):
            self.infer(warm)
        samples = []
        for _ in range(5):
            started = time.perf_counter_ns()
            self.infer(warm)
            samples.append((time.perf_counter_ns() - started) / NS_MS)
        self.measured_ms = statistics.median(samples)

    def infer(self, bgra):
        import cv2

        height, width = bgra.shape[:2]
        ratio = self.input_size / max(height, width)
        resized = cv2.resize(bgra, (max(1, int(width * ratio)), max(1, int(height * ratio))),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        self.blob.fill(114)
        self.blob[0, :, :rgb.shape[0], :rgb.shape[1]] = rgb.transpose(2, 0, 1)
        output = self.session.run(None, {self.input_name: self.blob})[0]
        return self.helper._decode(output[0], width, height, ratio)

    def describe(self):
        return {"detector": "OpenCV Zoo YOLOX-S FP32 via ONNX Runtime",
                "intra_op_num_threads": self.requested_threads,
                "measured_ms": self.measured_ms, "onnxruntime": self.ort_version,
                "opencv_preprocess": self.cv2_version, "spinning": "disabled",
                "model_bytes": self.bytes, "sha256": self.sha256,
                "thread_env": {name: os.environ.get(name) for name in
                               ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}}


class CudaYoloDetector:
    """The same verified YOLOX-S model through ONNX Runtime's CUDA provider.

    Two things this is careful about, because both are easy to get wrong and
    both would flatter the result:

    * **CPU fallback is disabled.** Without
      ``session.disable_cpu_ep_fallback`` a node that CUDA cannot run drops to
      the CPU silently and still gets counted as GPU inference.
    * **Timing covers completed inference, not submission.** CUDA work is
      asynchronous, so the output binding is synchronized before the clock
      stops. A benchmark that skips this measures how fast ORT can queue work.

    The input is bound from host memory. Keeping the *input* GPU-resident is
    RapidShot's `GpuConverter`/`TensorTransfer` path and a deeper integration
    than this measurement wires up; see the roadmap entry on GPU-resident
    preprocessing. What this measures is inference avoided, not transfer
    avoided.
    """

    def __init__(self, width, height, device_id=0):
        import cv2
        import onnxruntime as ort

        from whole_frame_task import (ASSETS, INPUT_SIZE, MODEL_BYTES, MODEL_NAME, MODEL_SHA256,
                                      YoloX, verify_model)

        path = ASSETS / MODEL_NAME
        verify_model(path)
        self.helper = YoloX(path)
        self.input_size = INPUT_SIZE
        self.bytes, self.sha256 = MODEL_BYTES, MODEL_SHA256
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(
            str(path), options,
            providers=[("CUDAExecutionProvider", {"device_id": device_id})])
        providers = self.session.get_providers()
        if providers[:1] != ["CUDAExecutionProvider"]:
            raise RuntimeError(f"CUDA provider not loaded: {providers}")
        self.providers = providers
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.ort_version = ort.__version__
        self.cv2_version = cv2.__version__
        self.device_id = device_id
        self.blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        self.measured_ms = None
        warm = np.zeros((height, width, 4), dtype=np.uint8)
        for _ in range(5):
            self.infer(warm)
        samples = []
        for _ in range(10):
            started = time.perf_counter_ns()
            self.infer(warm)
            samples.append((time.perf_counter_ns() - started) / NS_MS)
        self.measured_ms = statistics.median(samples)

    def infer(self, bgra):
        import cv2

        height, width = bgra.shape[:2]
        ratio = self.input_size / max(height, width)
        resized = cv2.resize(bgra, (max(1, int(width * ratio)), max(1, int(height * ratio))),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        self.blob.fill(114)
        self.blob[0, :, :rgb.shape[0], :rgb.shape[1]] = rgb.transpose(2, 0, 1)
        binding = self.session.io_binding()
        binding.bind_cpu_input(self.input_name, self.blob)
        binding.bind_output(self.output_name, "cuda", self.device_id)
        self.session.run_with_iobinding(binding)
        # Wait for the device before reading the clock or the result.
        binding.synchronize_outputs()
        output = binding.copy_outputs_to_cpu()[0]
        return self.helper._decode(output[0], width, height, ratio)

    def describe(self):
        return {"detector": "OpenCV Zoo YOLOX-S FP32 via ONNX Runtime CUDA",
                "providers": self.providers, "device_id": self.device_id,
                "cpu_ep_fallback": "disabled", "io_binding": "output GPU-resident, input from host",
                "synchronized": True, "measured_ms": self.measured_ms,
                "onnxruntime": self.ort_version, "opencv_preprocess": self.cv2_version,
                "model_bytes": self.bytes, "sha256": self.sha256}


def build_detector(config, width, height):
    if config["detector"] == "synthetic":
        return SyntheticDetector(config["infer_ms"], width, height)
    if config["detector"] == "ort":
        return OrtYoloDetector(config["threads"], width, height)
    if config["detector"] == "cuda":
        return CudaYoloDetector(width, height)
    return YoloDetector(config["threads"], width, height)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

def wait_until(deadline_ns):
    while True:
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2 * NS_MS:
            time.sleep(remaining / NS_SECOND / 2)


def run_worker(arm, target, config):
    from downstream_scene import read_stamp

    group = config["group"]
    if group == "metadata" and arm not in METADATA_ARMS:
        raise RuntimeError(f"{arm} exposes no change metadata")

    width = target["bounds"][2] - target["bounds"][0]
    height = target["bounds"][3] - target["bounds"][1]
    detector = build_detector(config, width, height)

    adapter = CostAdapter(arm, target, config["hz"], frequency())
    print("READY " + json.dumps({"pid": ct.windll.kernel32.GetCurrentProcessId(),
                                 "adapter_luid": None,
                                 "detector": detector.describe()}), flush=True)
    command = json.loads(sys.stdin.readline())
    start_ns, stop_ns = command["start_ns"], command["stop_ns"]

    records = []
    previous = None
    empty_reads = 0
    inference_ns = 0
    decision_ns = 0

    wait_until(start_ns)
    try:
        while time.perf_counter_ns() < stop_ns:
            sample = adapter.read(0.004)
            if sample is None:
                empty_reads += 1
                continue
            delivered_ns = time.perf_counter_ns()
            try:
                pixels = sample.pixels
                if pixels is None:
                    raise RuntimeError(f"{arm} delivered no whole-frame pixels")
                identical = False
                decision_started = time.perf_counter_ns()
                if group == "metadata":
                    identical = not sample.changes.indices
                elif group == "full_compare" and previous is not None:
                    identical = np.array_equal(pixels, previous)
                decided_ns = time.perf_counter_ns()
                decision_ns += decided_ns - decision_started
                stamp = read_stamp(pixels)
                if not identical:
                    detector.infer(pixels)
                done_ns = time.perf_counter_ns()
                if not identical:
                    inference_ns += done_ns - decided_ns
                records.append({"present_ns": sample.present_ns, "delivered_ns": delivered_ns,
                                "done_ns": done_ns, "inferred": not identical, "stamp": stamp})
                if group == "full_compare":
                    if previous is None or previous.shape != pixels.shape:
                        previous = pixels.copy()
                    else:
                        np.copyto(previous, pixels)
            finally:
                sample.release()
    finally:
        adapter.close()

    print("RESULT " + json.dumps({
        "arm": arm, "group": group, "empty_reads": empty_reads,
        "inference_ns": inference_ns, "decision_ns": decision_ns,
        "records": records,
    }), flush=True)
    json.loads(sys.stdin.readline())


# --------------------------------------------------------------------------- #
# Parent
# --------------------------------------------------------------------------- #

class Worker:
    def __init__(self, arm, target, config):
        self.arm = arm
        self.ready = threading.Event()
        self.done = threading.Event()
        self.result = None
        self.ready_payload = None
        self.messages = []
        # Pin the numeric runtimes for the synthetic detector only. Without it
        # NumPy's BLAS parallelizes that workload across cores, so its cost
        # becomes load-dependent and its calibration drifts. The ONNX Runtime
        # detector manages its own thread pool and sets the count explicitly,
        # so pinning there would silently contradict --threads.
        environment = dict(os.environ)
        if config["detector"] == "synthetic":
            environment.update({name: "1" for name in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})
        self.process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", str(Path(__file__).resolve()),
             "--worker", arm, "--target-json", json.dumps(target),
             "--config-json", json.dumps(config)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW, text=True, encoding="utf8",
            cwd=str(ROOT), env=environment)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                if line.startswith("READY "):
                    self.ready_payload = json.loads(line[6:])
                    self.ready.set()
                elif line.startswith("RESULT "):
                    self.result = json.loads(line[7:])
                    self.done.set()
                elif len(self.messages) < 80:
                    self.messages.append(line.strip())
        finally:
            self.ready.set()
            self.done.set()

    def __enter__(self):
        if not self.ready.wait(240) or self.ready_payload is None:
            self.close()
            raise RuntimeError(f"{self.arm} failed to open: {' '.join(self.messages[-12:])}")
        return self

    def __exit__(self, *exc):
        self.close()

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def collect(self, timeout=180):
        if not self.done.wait(timeout) or self.result is None:
            raise RuntimeError(f"{self.arm} produced no result: {' '.join(self.messages[-12:])}")
        with contextlib.suppress(Exception):
            self.send({"finalized": True})
        return self.result

    def close(self):
        with contextlib.suppress(Exception):
            self.process.stdin.close()
        with contextlib.suppress(Exception):
            self.process.wait(10)
        if self.process.poll() is None:
            self.process.kill()


@contextlib.contextmanager
def display_awake():
    kernel = ct.WinDLL("kernel32", use_last_error=True)
    set_state = kernel.SetThreadExecutionState
    set_state.argtypes = [ct.c_uint32]
    set_state.restype = ct.c_uint32
    continuous, system, display = 0x80000000, 0x00000001, 0x00000002
    if not set_state(continuous | system | display):
        raise ct.WinError(ct.get_last_error())
    try:
        yield
    finally:
        set_state(continuous)


def schedule(ticks, duplicates_per_change):
    """One new state, then N repaints of that same state, repeating."""
    plan = []
    index = 0
    while len(plan) < ticks:
        plan.append(("new", index))
        for _ in range(duplicates_per_change):
            if len(plan) >= ticks:
                break
            plan.append(("repeat", index))
        index += 1
    return plan


def repaint_identical(player, fixture, workload, index):
    """Present the current state's pixels again, byte for byte.

    ``ScenePlayer.present`` computes its own dirty regions and returns without
    painting when nothing moved, so repeating a state through it puts nothing on
    the wire. This presents the already-stamped cached raster with no region
    hint, which forces a full present of identical pixels.
    """
    variants = fixture.steps[workload]
    base = variants[1 + index % (len(variants) - 1)]
    image = fixture._raster(base)
    player.windows.repaint(player.base.handle, image, None, False)


def percentile(values, q):
    return float(np.percentile(values, q)) if values else None


def summarize(run, submitted, cpu, memory, elapsed_seconds):
    records = run["records"]
    inferred = [r for r in records if r["inferred"]]
    presented = {entry["stamp"] for entry in submitted}
    # A state counts as processed when the detector ran on a frame carrying it.
    processed = {r["stamp"] for r in inferred if r["stamp"] is not None}
    seen = {r["stamp"] for r in records if r["stamp"] is not None}
    redundant = 0
    already = set()
    for record in inferred:
        if record["stamp"] is not None and record["stamp"] in already:
            redundant += 1
        already.add(record["stamp"])

    first_touch = {}
    for record in inferred:
        stamp = record["stamp"]
        if stamp is None or stamp in first_touch or record["present_ns"] is None:
            continue
        first_touch[stamp] = (record["done_ns"] - record["present_ns"]) / NS_MS
    freshness = list(first_touch.values())

    inference_s = run["inference_ns"] / NS_SECOND
    return {
        "delivered_frames": len(records),
        "detector_calls": len(inferred),
        "skipped_frames": len(records) - len(inferred),
        "redundant_inferences": redundant,
        "empty_reads": run["empty_reads"],
        "states_presented": len(presented),
        "states_delivered": len(seen & presented),
        "states_processed": len(processed & presented),
        "states_missed": len(presented - processed),
        "inference_seconds": inference_s,
        "decision_seconds": run["decision_ns"] / NS_SECOND,
        "inference_duty_cycle": inference_s / elapsed_seconds if elapsed_seconds else None,
        "cpu_seconds": (cpu["user_time_ns"] + cpu["kernel_time_ns"]) / NS_SECOND,
        "cycles": cpu["cycles"],
        "cpu_seconds_per_state_processed": (
            ((cpu["user_time_ns"] + cpu["kernel_time_ns"]) / NS_SECOND) / len(processed & presented)
            if processed & presented else None),
        "freshness_ms": {"n": len(freshness), "p50": percentile(freshness, 50),
                         "p95": percentile(freshness, 95), "p99": percentile(freshness, 99)},
        "cpu": cpu, "memory": memory,
    }


def run_trial(player, fixture, target, arm, group, settings):
    from cost_telemetry import ProcessSampler

    workload = settings["workload"]
    fixture.seed(player, workload)
    config = {"hz": settings["hz"], "group": group, "detector": settings["detector"],
              "infer_ms": settings["infer_ms"], "threads": settings["threads"]}
    with Worker(arm, target, config) as child:
        sampler = ProcessSampler(child.ready_payload["pid"], sample_seconds=0.1,
                                 pdh=False, gpu_adapter_luid=None)
        sampler.prepare().start()
        start = sampler._initial_cpu.at_ns
        stop = start + round(settings["duration"] * NS_SECOND)
        period = round(NS_SECOND / settings["hz"])
        ticks = max(1, (stop - start) // period)
        plan = schedule(int(ticks), settings["duplicates"])
        submitted = []
        try:
            child.send({"start_ns": start, "stop_ns": stop})
            last_stamp = None
            for position, (kind, index) in enumerate(plan):
                due = start + position * period
                wait_until(due)
                if time.perf_counter_ns() >= stop:
                    break
                if kind == "new":
                    last_stamp = fixture.show(player, workload, index)
                else:
                    repaint_identical(player, fixture, workload, index)
                submitted.append({"position": position, "index": index, "kind": kind,
                                  "stamp": last_stamp, "due_ns": due})
            wait_until(stop)
        finally:
            process = sampler.finish()
        run = child.collect()
    summary = summarize(run, submitted, process["cpu"], process.get("memory"),
                        process["cpu"]["elapsed_seconds"])
    summary["submitted_presents"] = len(submitted)
    summary["submitted_new_states"] = sum(1 for e in submitted if e["kind"] == "new")
    summary["submitted_repeats"] = sum(1 for e in submitted if e["kind"] == "repeat")
    summary["detector"] = child.ready_payload["detector"]
    return {"arm": arm, "group": group, "summary": summary}


def report(data):
    lines = ["# Inference avoided, with duplicate frames controlled", "",
             f"{data['target']['monitor']}, "
             f"{data['target']['bounds'][2]}×{data['target']['bounds'][3]} at "
             f"{data['target']['physical_refresh_hz']} Hz, {data['target']['adapter']}. "
             f"The dashboard was repainted {data['hz']} times a second for "
             f"{data['duration']:.0f} s per arm, **{data['duplicates']} identical repaints after "
             f"each new state**, {data['rounds']} round(s).", "",
             "Each arm ran in its own worker process, so CPU and memory are the consumer alone. "
             "A state counts as *processed* when the detector ran on a frame carrying its marker; "
             "*missed* means no frame carrying it ever reached the detector.", ""]
    for round_data in data["trials"]:
        detector = round_data["arms"][0]["summary"]["detector"]
        label = detector.get("detector")
        if "target_ms" in detector:
            extra = f"{detector['target_ms']} ms target, {detector['rounds_per_inference']} rounds"
        elif "providers" in detector:
            extra = (f"CUDA device {detector['device_id']}, "
                     f"{detector['measured_ms']:.1f} ms measured idle, "
                     f"CPU fallback {detector['cpu_ep_fallback']}")
        elif "intra_op_num_threads" in detector:
            extra = (f"{detector['intra_op_num_threads']} intra-op thread(s), "
                     f"{detector['measured_ms']:.1f} ms measured idle")
        else:
            extra = f"{detector.get('threads')} thread(s)"
        lines += [f"## Round {round_data['round']} — {label} ({extra})", "",
                  "| Arm | Group | Delivered | Detector calls | Redundant | Avoided | "
                  "Inference s | CPU s | States missed |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        baseline = next((t for t in round_data["arms"] if t["group"] == "none"), None)
        base_calls = baseline["summary"]["detector_calls"] if baseline else None
        for trial in round_data["arms"]:
            s = trial["summary"]
            avoided = ("baseline" if trial["group"] == "none" or not base_calls
                       else f"{100 * (base_calls - s['detector_calls']) / base_calls:.1f}%")
            lines.append(
                f"| {trial['arm']} | {GROUP_LABELS[trial['group']]} | {s['delivered_frames']} "
                f"| {s['detector_calls']} | {s['redundant_inferences']} | {avoided} "
                f"| {s['inference_seconds']:.2f} | {s['cpu_seconds']:.2f} "
                f"| {s['states_missed']} of {s['states_presented']} |")
        lines += ["", "| Arm | Group | States processed | CPU s per state | "
                  "Freshness p50 | p95 | p99 | Decision s | Peak RSS |",
                  "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for trial in round_data["arms"]:
            s = trial["summary"]
            fresh = s["freshness_ms"]
            memory = s["memory"] or {}
            peak = (memory.get("os_peak_working_set_bytes")
                    or memory.get("sampled_peak_working_set_bytes"))
            per_state = s["cpu_seconds_per_state_processed"]

            def ms(value):
                return "—" if value is None else f"{value:.1f} ms"

            lines.append(
                f"| {trial['arm']} | {GROUP_LABELS[trial['group']]} "
                f"| {s['states_processed']} of {s['states_presented']} "
                f"| {'—' if per_state is None else f'{per_state:.4f}'} "
                f"| {ms(fresh['p50'])} | {ms(fresh['p95'])} | {ms(fresh['p99'])} "
                f"| {s['decision_seconds']:.4f} "
                f"| {'—' if peak is None else f'{peak / 2**20:.1f} MiB'} |")
        lines.append("")
    lines += ["No energy figures appear here. This host exposes processor clock through "
              "`CallNtPowerInformation` but no power or energy counter, so CPU seconds and "
              "cycles are the resource measure.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker")
    parser.add_argument("--target-json")
    parser.add_argument("--config-json")
    parser.add_argument("--arms", nargs="+", default=["rapidshot_exact", "dxcam_copy", "mss_gdi"])
    parser.add_argument("--groups", nargs="+", default=list(GROUPS))
    parser.add_argument("--hz", type=int, default=20)
    parser.add_argument("--duplicates", type=int, default=3,
                        help="Identical repaints after each new state")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workload", default="sparse")
    parser.add_argument("--detector", choices=("synthetic", "yolox", "ort", "cuda"),
                        default="synthetic")
    parser.add_argument("--infer-ms", type=float, default=15)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--monitor", default="HP 324pe")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()

    if arguments.worker:
        run_worker(arguments.worker, json.loads(arguments.target_json),
                   json.loads(arguments.config_json))
        return

    from cost_benchmark import Fixture
    from downstream_scene import ScenePlayer

    target = select_monitor(arguments.monitor)
    settings = {"hz": arguments.hz, "duration": arguments.duration,
                "workload": arguments.workload, "duplicates": arguments.duplicates,
                "detector": arguments.detector, "infer_ms": arguments.infer_ms,
                "threads": arguments.threads}
    data = {"target": target, "hz": arguments.hz, "duration": arguments.duration,
            "duplicates": arguments.duplicates, "rounds": arguments.rounds,
            "workload": arguments.workload, "detector": arguments.detector,
            "infer_ms": arguments.infer_ms, "threads": arguments.threads, "trials": []}

    fixture = Fixture(target)
    fixture.prewarm([arguments.workload])
    with display_awake(), ScenePlayer(target["bounds"]) as player:
        player.prepare(fixture.steps)
        for round_index in range(arguments.rounds):
            arms = []
            for arm in arguments.arms:
                for group in arguments.groups:
                    if group == "metadata" and arm not in METADATA_ARMS:
                        continue
                    print(f"round {round_index + 1}: {arm} / {group}", flush=True)
                    trial = run_trial(player, fixture, target, arm, group, settings)
                    s = trial["summary"]
                    print(f"    {s['detector_calls']} calls of {s['delivered_frames']} frames, "
                          f"{s['redundant_inferences']} redundant, "
                          f"{s['states_processed']}/{s['states_presented']} states, "
                          f"{s['cpu_seconds']:.2f} CPU s", flush=True)
                    arms.append(trial)
            data["trials"].append({"round": round_index + 1, "arms": arms})

    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        arguments.out.with_suffix(".md").write_text(report(data), encoding="utf-8")
        print(f"Wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
