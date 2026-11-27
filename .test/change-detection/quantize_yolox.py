"""Quantize the verified YOLOX-S model to INT8 and check what it costs.

Quantization is only usable here if it is both faster *and* still finds the same
objects. Dynamic quantization targets MatMul and Attention rather than
convolution, and YOLOX-S is convolution-heavy, so a speed-up is not a given --
this script measures it instead of assuming, and reports detection agreement
against the FP32 reference on the same public fixtures the frame-gate benchmark
uses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import numpy as np

from whole_frame_task import (ASSETS, INPUT_SIZE, MODEL_NAME, YoloX, _canvas, load_fixtures,
                              same_detections, verify_model)


INT8_NAME = "yolox_s_int8_dynamic.onnx"
PREPARED_NAME = "yolox_s_shape_inferred.onnx"
INT8_PREPARED_NAME = "yolox_s_int8_dynamic_prepared.onnx"
NS_MS = 1_000_000


class OrtRunner:
    """FP32 or INT8 session with the verified preprocessing and decode."""

    def __init__(self, model_path: Path, threads: int, helper: YoloX):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        self.session = ort.InferenceSession(str(model_path), options,
                                            providers=["CPUExecutionProvider"])
        self.name = self.session.get_inputs()[0].name
        self.helper = helper
        self.blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

    def infer(self, bgra):
        import cv2

        height, width = bgra.shape[:2]
        ratio = INPUT_SIZE / max(height, width)
        resized = cv2.resize(bgra, (max(1, int(width * ratio)), max(1, int(height * ratio))),
                             interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        self.blob.fill(114)
        self.blob[0, :, :rgb.shape[0], :rgb.shape[1]] = rgb.transpose(2, 0, 1)
        output = self.session.run(None, {self.name: self.blob})[0]
        return self.helper._decode(output[0], width, height, ratio)


def timed(runner, frame, repeats):
    runner.infer(frame)
    runner.infer(frame)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        runner.infer(frame)
        samples.append((time.perf_counter_ns() - started) / NS_MS)
    return {"p50_ms": statistics.median(samples), "min_ms": min(samples), "max_ms": max(samples)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=(1, 4))
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args()

    from onnxruntime.quantization import QuantType, quantize_dynamic

    source = ASSETS / MODEL_NAME
    verify_model(source)
    target = ASSETS / INT8_NAME
    if not target.is_file():
        print(f"quantizing {source.name} -> {target.name}", flush=True)
        quantize_dynamic(source, target, weight_type=QuantType.QInt8)

    # ORT warns that quantizing without shape inference and graph cleanup costs
    # accuracy, so do it the recommended way too and compare both.
    prepared = ASSETS / PREPARED_NAME
    prepared_int8 = ASSETS / INT8_PREPARED_NAME
    if not prepared_int8.is_file():
        from onnxruntime.quantization.shape_inference import quant_pre_process

        print(f"pre-processing {source.name} -> {prepared.name}", flush=True)
        # Symbolic shape inference needs sympy, which is not installed. This
        # model's input is a fixed [1, 3, 640, 640] and its output a fixed
        # [1, 8400, 85], so there are no dynamic dimensions for symbolic
        # inference to resolve; ONNX shape inference plus the graph cleanup that
        # quant_pre_process still performs is what the warning is really about.
        quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
        print(f"quantizing {prepared.name} -> {prepared_int8.name}", flush=True)
        quantize_dynamic(prepared, prepared_int8, weight_type=QuantType.QInt8)

    print(f"FP32 {source.stat().st_size:,} bytes | INT8 {target.stat().st_size:,} bytes "
          f"| INT8 pre-processed {prepared_int8.stat().st_size:,} bytes", flush=True)

    helper = YoloX(source)                      # verified decode tables
    dog, wall = load_fixtures()
    frames = {"dog": _canvas(dog), "wall": _canvas(wall), "moved_dog": _canvas(dog, 120)}

    variants = {"int8_plain": target, "int8_prepared": prepared_int8}
    result = {"fp32_bytes": source.stat().st_size,
              "int8_bytes": {name: path.stat().st_size for name, path in variants.items()},
              "repeats": arguments.repeats, "threads": {}}

    for threads in arguments.threads:
        fp32 = OrtRunner(source, threads, helper)
        row = {"fp32": timed(fp32, frames["dog"], arguments.repeats), "variants": {}}
        print(f"threads={threads}: FP32 {row['fp32']['p50_ms']:.1f} ms", flush=True)
        for label, path in variants.items():
            candidate_runner = OrtRunner(path, threads, helper)
            entry = {"timing": timed(candidate_runner, frames["dog"], arguments.repeats),
                     "agreement": {}}
            entry["speedup"] = row["fp32"]["p50_ms"] / entry["timing"]["p50_ms"]
            for name, frame in frames.items():
                reference = fp32.infer(frame)
                candidate = candidate_runner.infer(frame)
                entry["agreement"][name] = {
                    "fp32_detections": len(reference), "int8_detections": len(candidate),
                    "identical_by_benchmark_tolerance": same_detections(candidate, reference),
                    "fp32_classes": sorted({d[0] for d in reference}),
                    "int8_classes": sorted({d[0] for d in candidate}),
                }
            row["variants"][label] = entry
            print(f"  {label:14s} {entry['timing']['p50_ms']:6.1f} ms "
                  f"({entry['speedup']:.2f}x)", flush=True)
            for name, agree in entry["agreement"].items():
                print(f"      {name:10s} fp32={agree['fp32_detections']} "
                      f"int8={agree['int8_detections']} "
                      f"identical={agree['identical_by_benchmark_tolerance']} "
                      f"classes {agree['fp32_classes']} vs {agree['int8_classes']}", flush=True)
        result["threads"][str(threads)] = row

    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {arguments.out}", flush=True)


if __name__ == "__main__":
    main()
