"""Full-image OpenCV Zoo YOLOX-S CPU task and exact-unchanged smoke replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.request import urlopen

import cv2
import numpy as np


ASSETS = Path(__file__).resolve().parent / "assets" / "yolox"
MODEL_NAME = "object_detection_yolox_2022nov.onnx"
MODEL_REVISION = "c5c37f2482d90a975d2dc6a2d491f307cf41bd2d"
MODEL_BYTES = 35_858_002
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
MODEL_URL = (
    "https://huggingface.co/opencv/object_detection_yolox/resolve/"
    f"{MODEL_REVISION}/{MODEL_NAME}?download=true"
)
POSITIVE_URL = "https://raw.githubusercontent.com/Megvii-BaseDetection/YOLOX/0.1.1rc0/assets/dog.jpg"
NEGATIVE_URL = "https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/graf1.png"
ASSET_SPECS = (
    (MODEL_NAME, MODEL_URL, MODEL_BYTES, MODEL_SHA256),
    ("dog.jpg", POSITIVE_URL, 163_759, "5a9522051c3cec2bbd2f6323fccba32e8fbf3ddcc2b3e2fd46b04c720bc6f866"),
    ("graf1.png", NEGATIVE_URL, 951_440, "1504b769303c7bde00fa578eeaad3c68e02aceabeb1242e556f1f8d19e4bdea5"),
)
INPUT_SIZE = 640
CONFIDENCE = 0.50
NMS_IOU = 0.50
DOG_CLASS = 16  # Zero-based COCO class index used by the OpenCV Zoo model.
Detection = tuple[int, float, float, float, float, float]


def _sha256(path: Path) -> str:
    with path.open("rb") as opened:
        return hashlib.file_digest(opened, "sha256").hexdigest()


def verify_model(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing verified YOLOX model: {path}")
    if path.stat().st_size != MODEL_BYTES or _sha256(path) != MODEL_SHA256:
        raise ValueError(f"YOLOX model size or SHA-256 differs from the OpenCV-published LFS pin: {path}")


def fetch_assets() -> dict:
    """Fetch only the pinned public model and sample files, validating before writing."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, url, expected_size, expected_hash in ASSET_SPECS:
        path = ASSETS / name
        if path.is_file():
            digest, size = _sha256(path), path.stat().st_size
            if size != expected_size or digest != expected_hash:
                raise ValueError(f"Existing asset differs from the pin: {path}")
        else:
            with urlopen(url, timeout=60) as response:
                data = response.read(expected_size + 1)
            digest, size = hashlib.sha256(data).hexdigest(), len(data)
            if size != expected_size or digest != expected_hash:
                raise ValueError(f"Downloaded {name} differs from the pin: {size} bytes, SHA-256 {digest}")
            partial = path.with_suffix(path.suffix + ".part")
            partial.write_bytes(data)
            partial.replace(path)
        provenance[name] = {"url": url, "bytes": size, "sha256": digest}
    (ASSETS / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return provenance


def load_fixtures() -> tuple[np.ndarray, np.ndarray]:
    decoded = []
    for name, _, expected_size, expected_hash in ASSET_SPECS[1:]:
        path = ASSETS / name
        contents = path.read_bytes()
        if len(contents) != expected_size or hashlib.sha256(contents).hexdigest() != expected_hash:
            raise ValueError(f"Fixture differs from the recorded publisher asset: {path}")
        pixels = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if pixels is None:
            raise ValueError(f"Could not decode fixture {path}")
        bgra = cv2.cvtColor(pixels, cv2.COLOR_BGR2BGRA)
        bgra.setflags(write=False)
        decoded.append(bgra)
    return decoded[0], decoded[1]


class YoloX:
    """The official OpenCV Zoo FP32 YOLOX-S model on the OpenCV CPU backend."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = Path(model_path) if model_path is not None else ASSETS / MODEL_NAME
        verify_model(self.model_path)
        cv2.setNumThreads(1)
        cv2.ocl.setUseOpenCL(False)
        # OpenCV 5's graph engine uses only CPU. Its target setter is unsupported.
        self.net = cv2.dnn.readNetFromONNX(str(self.model_path), cv2.dnn.ENGINE_OPENCV)
        self.blob = np.empty((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

        grids, strides = [], []
        for stride in (8, 16, 32):
            side = INPUT_SIZE // stride
            yy, xx = np.indices((side, side), dtype=np.float32)
            grids.append(np.stack((xx, yy), axis=-1).reshape(-1, 2))
            strides.append(np.full((side * side, 1), stride, dtype=np.float32))
        self.grid = np.concatenate(grids)
        self.stride = np.concatenate(strides)

    def infer(self, bgra: np.ndarray) -> tuple[Detection, ...]:
        if bgra.ndim != 3 or bgra.shape[2] != 4 or bgra.dtype != np.uint8 or not bgra.size:
            raise ValueError("YOLOX input must be a nonempty H×W×4 uint8 BGRA frame")
        height, width = bgra.shape[:2]
        ratio = INPUT_SIZE / max(height, width)
        resized_width, resized_height = max(1, int(width * ratio)), max(1, int(height * ratio))
        resized = cv2.resize(bgra, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2RGB)
        self.blob.fill(114)
        self.blob[0, :, :resized_height, :resized_width] = rgb.transpose(2, 0, 1)
        self.net.setInput(self.blob)
        output = self.net.forward()
        if output.shape != (1, len(self.grid), 85):
            raise RuntimeError(f"Unexpected output from official YOLOX model: {output.shape}")
        return self._decode(output[0], width, height, ratio)

    def _decode(self, raw, width: int, height: int, ratio: float) -> tuple[Detection, ...]:
        classes = np.argmax(raw[:, 5:], axis=1)
        scores = raw[:, 4] * raw[np.arange(len(raw)), 5 + classes]
        candidate = np.flatnonzero(scores >= CONFIDENCE)
        if not candidate.size:
            return ()
        size = np.exp(raw[candidate, 2:4]) * self.stride[candidate]
        center = (raw[candidate, :2] + self.grid[candidate]) * self.stride[candidate]
        origin = center - size / 2
        boxes = np.concatenate((origin, size), axis=1)
        indices = cv2.dnn.NMSBoxesBatched(
            boxes.tolist(), scores[candidate].tolist(), classes[candidate].tolist(), CONFIDENCE, NMS_IOU,
        )
        detections = []
        for index in indices:
            x, y = origin[index] / ratio
            right, bottom = (origin[index] + size[index]) / ratio
            x, right = np.clip([x, right], 0, width)
            y, bottom = np.clip([y, bottom], 0, height)
            if right > x and bottom > y:
                source = candidate[index]
                detections.append((
                    int(classes[source]), round(float(x), 4), round(float(y), 4),
                    round(float(right), 4), round(float(bottom), 4), round(float(scores[source]), 6),
                ))
        return tuple(sorted(detections))


def same_detections(actual: tuple[Detection, ...], expected: tuple[Detection, ...]) -> bool:
    return len(actual) == len(expected) and all(
        left[0] == right[0]
        and all(abs(a - b) <= 0.01 for a, b in zip(left[1:5], right[1:5]))
        and abs(left[5] - right[5]) <= 0.00001
        for left, right in zip(actual, expected)
    )


def _canvas(image: np.ndarray, shift: int = 0) -> np.ndarray:
    """Fit a public image on a common 1080p desktop-sized BGRA canvas."""
    height, width = 1080, 1920
    source_height, source_width = image.shape[:2]
    scale = min((width - 2 * abs(shift)) / source_width, height / source_height)
    resized_width = int(source_width * scale)
    resized_height = int(source_height * scale)
    left = (width - resized_width) // 2 + shift
    top = (height - resized_height) // 2
    frame = np.full((height, width, 4), (33, 33, 33, 255), dtype=np.uint8)
    frame[top:top + resized_height, left:left + resized_width] = cv2.resize(image, (resized_width, resized_height))
    frame.setflags(write=False)
    return frame


def run(replay_inputs=None, output: Path | None = None, *, rounds: int = 1, seed: int = 269) -> dict:
    """Run both arms on the same immutable image sequence and verify every result."""
    fixture_sequence = replay_inputs is None
    if fixture_sequence:
        photographed_dog, photographed_wall = load_fixtures()
        positive = _canvas(photographed_dog)
        negative = _canvas(photographed_wall)
        moved = _canvas(photographed_dog, 120)
        replay_inputs = (positive, positive.copy(), negative, negative.copy(), moved, moved.copy(), positive, positive.copy())
        phases = ("dog", "same dog", "wall", "same wall", "moved dog", "same moved dog", "dog returned", "same returned dog")
    images = tuple(replay_inputs)
    if not images or rounds < 1:
        raise ValueError("YOLOX replay needs images and at least one round")
    if not fixture_sequence:
        phases = tuple(str(index) for index in range(len(images)))
    detector = YoloX()
    detector.infer(images[0])  # Warm the fixed graph before measuring either arm.
    rng = random.Random(seed)
    trials = []
    for round_index in range(rounds):
        order = ["always_infer", "skip_exact_unchanged"]
        rng.shuffle(order)
        results = {}
        for mode in order:
            previous_image = None
            detections = ()
            frame_results = []
            for phase, image in zip(phases, images):
                start = time.perf_counter_ns()
                equal = mode == "skip_exact_unchanged" and previous_image is not None and np.array_equal(image, previous_image)
                if not equal:
                    detections = detector.infer(image)
                elapsed_us = (time.perf_counter_ns() - start) / 1_000
                frame_results.append({"phase": phase, "inferred": not equal, "elapsed_us": elapsed_us,
                                      "detections": detections})
                previous_image = image
            results[mode] = frame_results
        always = results["always_infer"]
        gated = results["skip_exact_unchanged"]
        if not all(same_detections(gate["detections"], base["detections"]) for gate, base in zip(gated, always)):
            raise RuntimeError("Gated YOLOX results differ from the always-infer reference")
        expected_reruns = [i == 0 or not np.array_equal(images[i], images[i - 1]) for i in range(len(images))]
        if [frame["inferred"] for frame in gated] != expected_reruns:
            raise RuntimeError("Unchanged gate did not rerun exactly on changed images")
        if fixture_sequence:
            for index in (0, 4, 6):
                if DOG_CLASS not in {hit[0] for hit in always[index]["detections"]}:
                    raise RuntimeError(f"YOLOX did not detect the known dog at {phases[index]}: {always[index]['detections']}")
            if always[2]["detections"]:
                raise RuntimeError(f"YOLOX negative fixture produced detections: {always[2]['detections']}")
            if same_detections(always[0]["detections"], always[4]["detections"]):
                raise RuntimeError("YOLOX did not update coordinates when the positive image moved")
        trials.append({"round": round_index + 1, "order": order, "arms": results})
    result = {
        "model": "OpenCV Zoo YOLOX-S FP32", "model_bytes": MODEL_BYTES, "sha256": MODEL_SHA256,
        "model_url": MODEL_URL, "opencv": cv2.__version__, "opencv_threads": cv2.getNumThreads(),
        "opencl": cv2.ocl.useOpenCL(), "input_size": INPUT_SIZE, "confidence": CONFIDENCE, "nms_iou": NMS_IOU,
        "box_tolerance_px": 0.01, "score_tolerance": 0.00001, "fixtures": fixture_sequence,
        "input_resolution": [list(image.shape[:2][::-1]) for image in images],
        "measured_scope": "in-memory CPU preprocessing, exact gating where enabled, inference, and decode",
        "seed": seed, "trials": trials,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=1)
    arguments = parser.parse_args()
    if arguments.fetch_assets:
        print(json.dumps(fetch_assets(), indent=2))
    else:
        result = run(output=arguments.output, rounds=arguments.rounds)
        for trial in result["trials"]:
            print({mode: sum(frame["inferred"] for frame in frames) for mode, frames in trial["arms"].items()})
