"""Reproducible scenes containing objects YOLO11n actually detects.

ROADMAP § 7.0c records why this exists: `latency_source.rs` draws a frame-ID
marker pattern, which is exactly what the pixel-age clock needs and contains
nothing a detector recognises. A detection benchmark against it finds zero
objects in every frame, times the cheapest possible postprocess, and reports a
number that describes an empty screen.

**Which objects work was measured, not guessed.** Drawn procedurally at
2026-09-19 and put through the exported `yolo11n.onnx`:

===============  ==========================================
drawn            detected
===============  ==========================================
stop sign        ``stop_sign`` 0.93
clock            ``clock`` 0.90
traffic light    ``traffic_light`` 0.82
keyboard         ``keyboard`` 0.66
cup              ``cup`` 0.64 in company, nothing alone
person           nothing
laptop           nothing
bottle           nothing
sports ball      nothing
===============  ==========================================

So the scene is built from the four that hold up on their own. A crude
rectangle is not a laptop to a detector trained on photographs, and pretending
otherwise would give a scene that silently degrades to empty.

**The marker was checked against the scene, not assumed compatible.** With the
48-cell frame-ID pattern composited over a full scene, detection counts are
identical and confidences move by at most 0.02 -- so the pixel-age clock and
the detector can share a frame.

**The canvas tiles.** It is drawn at twice the capture size in each direction,
with every object repeated across the seams, so a source can pan over it
without a discontinuity. That is what makes the three workloads different
*dirty fractions* of the same scene rather than three different scenes:

``static``
    A fixed offset. Only the marker changes, so the compositor reports a tiny
    dirty rectangle -- the case where capture has least to do.
``scroll``
    The offset advances vertically. The whole frame changes every present.
``motion``
    The offset advances on both axes at different rates, so objects travel
    diagonally across the screen and their boxes move between frames.

**A limitation, stated rather than discovered:** panning moves everything
together. Objects do not move *relative to each other*, so this exercises a
detector's throughput and a capture path's dirty-rectangle behaviour, and does
not exercise tracking. Independent per-object motion needs sprite compositing
in the source and is not built.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

from ._paths import WORK  # noqa: E402

#: The COCO classes this scene is built from, because these are the ones that
#: survive being drawn rather than photographed. Anything else added here must
#: be re-verified with `--verify` before it is trusted.
VERIFIED_CLASSES = ("stop_sign", "clock", "traffic_light", "keyboard")

#: Below this, a detection run is timing an empty scene rather than a pipeline.
MINIMUM_DETECTIONS_PER_FRAME = 4

#: How large each object should be **in the model's 640px input**, not on the
#: screen. This is the fix for a failure the first verified scene walked into:
#: sizes fixed in source pixels shrink with the letterbox scale, so a 110px
#: clock is 55px to the model at 1280x800 and 27px at 2560x1600 -- detected in
#: one case and missed in the other, for no reason a reader would guess from
#: the configuration. Fixing the model-space size instead keeps a scene
#: comparably detectable at every capture resolution.
#: Taken directly from the drawings that verified, converted to model space:
#: the clock that scored 0.90 was radius 90 at 1280x800, which is a letterbox
#: scale of 0.5 and so 45 model pixels. Guessing larger is not safer -- an
#: object filling a third of the frame is as unlike the training data as one
#: filling twelve pixels.
MODEL_SPACE_SIZES = {"clock": 45, "stop_sign": 40, "traffic_light": 55,
                     "keyboard": 95}

#: Objects across the canvas. Raised from 7x5 after a live run: the harness's
#: tensor contract *stretches* 2560x1600 into a 640 square, which squashes
#: every object by 1.6x vertically and costs confidence on each one. Verifying
#: against a letterbox had hidden that -- the scene promised 6-9 detections and
#: the stretched harness saw 2-10, with frames below the floor. More objects in
#: view is the honest compensation; drawing them pre-distorted so they come out
#: round would be the other, and belongs with the letterbox configuration.
LAYOUT_COLUMNS, LAYOUT_ROWS = 9, 6

COCO_NAMES = (
    "person bicycle car motorcycle airplane bus train truck boat traffic_light "
    "fire_hydrant stop_sign parking_meter bench bird cat dog horse sheep cow "
    "elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee "
    "skis snowboard sports_ball kite baseball_bat baseball_glove skateboard "
    "surfboard tennis_racket bottle wine_glass cup fork knife spoon bowl banana "
    "apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch "
    "potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
    "cell_phone microwave oven toaster sink refrigerator book clock vase "
    "scissors teddy_bear hair_drier toothbrush").split()


def _cv2():
    import cv2
    return cv2


# ---------------------------------------------------------------------------
# The objects that survived verification
# ---------------------------------------------------------------------------

def draw_clock(img, cx, cy, size=110):
    cv2 = _cv2()
    r = int(size)
    cv2.circle(img, (cx, cy), r, (245, 245, 245), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (30, 30, 30), max(4, r // 15), cv2.LINE_AA)
    for tick in range(12):
        angle = tick * np.pi / 6
        inner = (int(cx + np.sin(angle) * r * 0.82), int(cy - np.cos(angle) * r * 0.82))
        outer = (int(cx + np.sin(angle) * r * 0.93), int(cy - np.cos(angle) * r * 0.93))
        cv2.line(img, inner, outer, (30, 30, 30), max(2, r // 30), cv2.LINE_AA)
    cv2.line(img, (cx, cy), (int(cx + r * 0.5), cy), (20, 20, 20), max(4, r // 18),
             cv2.LINE_AA)
    cv2.line(img, (cx, cy), (cx, int(cy - r * 0.72)), (20, 20, 20), max(3, r // 26),
             cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(4, r // 18), (20, 20, 20), -1, cv2.LINE_AA)


def draw_stop_sign(img, cx, cy, size=95):
    cv2 = _cv2()
    r = int(size)
    points = np.array([[int(cx + r * np.cos(np.pi / 8 + index * np.pi / 4)),
                        int(cy + r * np.sin(np.pi / 8 + index * np.pi / 4))]
                       for index in range(8)], np.int32)
    cv2.fillPoly(img, [points], (30, 30, 200), cv2.LINE_AA)
    cv2.polylines(img, [points], True, (255, 255, 255), max(4, r // 14), cv2.LINE_AA)
    scale = r / 62.0
    (text_w, text_h), _ = cv2.getTextSize("STOP", cv2.FONT_HERSHEY_SIMPLEX, scale,
                                          max(2, int(4 * scale)))
    cv2.putText(img, "STOP", (cx - text_w // 2, cy + text_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255),
                max(2, int(4 * scale)), cv2.LINE_AA)


def draw_traffic_light(img, cx, cy, size=110):
    cv2 = _cv2()
    half_w, half_h = int(size * 0.36), int(size)
    cv2.rectangle(img, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
                  (35, 35, 35), -1)
    cv2.rectangle(img, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
                  (90, 90, 90), max(3, size // 28))
    radius = int(size * 0.24)
    for index, colour in enumerate(((40, 40, 220), (40, 200, 220), (60, 200, 60))):
        cv2.circle(img, (cx, cy - int(size * 0.59) + index * int(size * 0.59)),
                   radius, colour, -1, cv2.LINE_AA)


def draw_keyboard(img, cx, cy, size=190):
    cv2 = _cv2()
    half_w, half_h = int(size), int(size * 0.32)
    cv2.rectangle(img, (cx - half_w, cy - half_h), (cx + half_w, cy + half_h),
                  (55, 55, 60), -1)
    key = max(8, int(size * 0.135))
    gap = max(2, key // 5)
    for row in range(4):
        for col in range(14):
            x = cx - half_w + 12 + col * (key + gap)
            y = cy - half_h + 12 + row * (key + gap)
            if x + key < cx + half_w and y + key < cy + half_h:
                cv2.rectangle(img, (x, y), (x + key, y + key), (120, 120, 128), -1)


DRAWINGS = {"clock": draw_clock, "stop_sign": draw_stop_sign,
            "traffic_light": draw_traffic_light, "keyboard": draw_keyboard}


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Placement:
    """One object on the canvas, in canvas pixels."""

    kind: str
    x: int
    y: int
    size: int

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def letterbox_scale(width: int, height: int, model_size: int = 640) -> float:
    """The single factor a letterbox applies. Objects shrink by exactly this."""
    return min(model_size / width, model_size / height)


def layout(canvas_width: int, canvas_height: int, capture_width: int,
           capture_height: int, seed: int = 20260919) -> list:
    """A deterministic spread of verified objects across the canvas.

    Seeded and jittered rather than on a plain grid: a regular lattice of
    identical objects is an unrealistic input, and at some pan offsets it puts
    every object on a row where they merge into one another under NMS.

    Sizes come from :data:`MODEL_SPACE_SIZES` divided by the letterbox scale,
    so what the detector sees is the same at any capture resolution.
    """
    rng = np.random.default_rng(seed)
    scale = letterbox_scale(capture_width, capture_height)
    placements = []
    # Shuffled rather than cycled. An arithmetic phase looks well mixed and is
    # not: with 7 columns and 4 classes, `(row * 7 + column + row) % 4` reduces
    # to `column % 4`, so every column held one class and the first rendered
    # scene was three vertical stripes of identical objects.
    kinds = list(VERIFIED_CLASSES)
    assignment = [kinds[index % len(kinds)]
                  for index in range(LAYOUT_ROWS * LAYOUT_COLUMNS)]
    rng.shuffle(assignment)
    for row in range(LAYOUT_ROWS):
        for column in range(LAYOUT_COLUMNS):
            kind = assignment[row * LAYOUT_COLUMNS + column]
            cell_w = canvas_width / LAYOUT_COLUMNS
            cell_h = canvas_height / LAYOUT_ROWS
            jitter_x = int(rng.integers(-int(cell_w * 0.18), int(cell_w * 0.18) + 1))
            jitter_y = int(rng.integers(-int(cell_h * 0.18), int(cell_h * 0.18) + 1))
            size_jitter = float(rng.uniform(0.9, 1.2))
            placements.append(Placement(
                kind=kind,
                x=int(cell_w * (column + 0.5) + jitter_x) % canvas_width,
                y=int(cell_h * (row + 0.5) + jitter_y) % canvas_height,
                size=max(24, int(MODEL_SPACE_SIZES[kind] / scale * size_jitter))))
    return placements


def render_canvas(width: int, height: int, seed: int = 20260919) -> np.ndarray:
    """A tileable BGR canvas, twice the capture size on each axis.

    Every object is drawn again at each wrapped position, so panning across the
    seam shows an object entering rather than a hard edge. Without that, the
    scroll and motion workloads would present a discontinuity once per lap that
    no real desktop produces.
    """
    cv2 = _cv2()
    canvas_width, canvas_height = width * 2, height * 2
    image = np.full((canvas_height, canvas_width, 3), (44, 48, 56), np.uint8)
    # Some background structure, so the scene is not a flat colour field that
    # compresses to nothing and gives an unrealistically cheap encode.
    for x in range(0, canvas_width, 160):
        cv2.rectangle(image, (x, 0), (x + 80, canvas_height), (38, 42, 50), -1)
    for y in range(0, canvas_height, 240):
        cv2.line(image, (0, y), (canvas_width, y), (52, 56, 66), 3)

    for placement in layout(canvas_width, canvas_height, width, height, seed):
        draw = DRAWINGS[placement.kind]
        for offset_x in (-canvas_width, 0, canvas_width):
            for offset_y in (-canvas_height, 0, canvas_height):
                x, y = placement.x + offset_x, placement.y + offset_y
                if (-placement.size * 2 < x < canvas_width + placement.size * 2
                        and -placement.size * 2 < y < canvas_height + placement.size * 2):
                    draw(image, x, y, placement.size)
    return image


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

#: Pixels of pan per presented frame, per workload. Chosen so scroll changes
#: most of the frame and motion moves objects visibly without turning the
#: screen into a blur no detector could work with.
WORKLOAD_STEPS = {"static": (0, 0), "scroll": (0, 6), "motion": (5, 3)}


def offset_for(workload: str, frame: int, width: int, height: int):
    """Where the visible window sits on the canvas for this presented frame."""
    if workload not in WORKLOAD_STEPS:
        raise ValueError(f"unknown workload {workload!r}")
    step_x, step_y = WORKLOAD_STEPS[workload]
    return ((step_x * frame) % (width * 2), (step_y * frame) % (height * 2))


def view(canvas: np.ndarray, width: int, height: int, offset_x: int, offset_y: int):
    """The window a viewer sees, wrapping across the canvas seam."""
    rows = (np.arange(height) + offset_y) % canvas.shape[0]
    columns = (np.arange(width) + offset_x) % canvas.shape[1]
    return canvas[rows][:, columns]


def apply_marker(image: np.ndarray, frame: int) -> np.ndarray:
    """Composite the frame-ID pattern `latency_source.rs` draws, for verification.

    Kept byte-identical to the shader's layout -- 48 cells of 8 pixels across
    the top 16 rows, magic 167, then the frame number, then a checksum -- so
    that verifying a scene verifies the thing the source will actually present.
    """
    check = (frame ^ (frame >> 8) ^ (frame >> 16) ^ (frame >> 24) ^ 167) & 255
    for cell in range(48):
        if cell < 8:
            bit = (167 >> cell) & 1
        elif cell < 40:
            bit = (frame >> (cell - 8)) & 1
        else:
            bit = (check >> (cell - 40)) & 1
        image[0:16, cell * 8:(cell + 1) * 8] = 255 if bit else 0
    return image


# ---------------------------------------------------------------------------
# Writing a scene pack
# ---------------------------------------------------------------------------

BACKGROUND_NAME = "background.bgra"
MANIFEST_NAME = "scene.json"
DETECTIONS_NAME = "scene.detections.json"


def write_scene(directory, width: int, height: int, seed: int = 20260919) -> dict:
    """Render the canvas and write it with a manifest the source can read.

    Raw BGRA rather than PNG: the D3D source uploads it straight to a texture,
    and adding an image decoder to the Rust binary would be a dependency and a
    failure mode for no benefit. The manifest carries the dimensions, so the
    raw file is never interpreted without them.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    canvas = render_canvas(width, height, seed)
    bgra = np.dstack([canvas, np.full(canvas.shape[:2], 255, np.uint8)])
    payload = np.ascontiguousarray(bgra).tobytes()
    (directory / BACKGROUND_NAME).write_bytes(payload)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "capture_width": width, "capture_height": height,
        "canvas_width": canvas.shape[1], "canvas_height": canvas.shape[0],
        "format": "BGRA8", "background": BACKGROUND_NAME,
        "background_sha256": hashlib.sha256(payload).hexdigest(),
        "seed": seed,
        "workload_steps": {name: list(step) for name, step in WORKLOAD_STEPS.items()},
        "objects": [placement.as_dict() for placement in
                    layout(canvas.shape[1], canvas.shape[0], width, height, seed)],
        "verified_classes": list(VERIFIED_CLASSES),
        "note": ("tileable canvas at twice the capture size; workloads pan over it. "
                 "Objects do not move relative to each other -- see the module "
                 "docstring."),
    }
    manifest["scene_id"] = scene_id(manifest)
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    return manifest


def scene_id(manifest: dict) -> str:
    """A content identity for the scene, for the case configuration.

    Two runs against different scenes are not repeats of one measurement: a
    scene with more objects in it does more postprocessing work per frame.
    """
    stable = {key: manifest[key] for key in
              ("capture_width", "capture_height", "canvas_width", "canvas_height",
               "background_sha256", "seed", "workload_steps", "objects")}
    return "s-" + hashlib.sha256(
        json.dumps(stable, sort_keys=True).encode()).hexdigest()[:16]


def load_manifest(directory) -> dict:
    return json.loads((Path(directory) / MANIFEST_NAME).read_text(encoding="utf-8"))


def load_canvas(directory) -> np.ndarray:
    """Read a written scene back as BGR, for verification or inspection."""
    manifest = load_manifest(directory)
    raw = (Path(directory) / manifest["background"]).read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["background_sha256"]:
        raise ValueError(f"{directory}: background does not match its manifest hash")
    bgra = np.frombuffer(raw, np.uint8).reshape(
        manifest["canvas_height"], manifest["canvas_width"], 4)
    return bgra[:, :, :3]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def model_tensor(image: np.ndarray, size: int = 640, mode: str = "stretch"):
    """Prepare a frame the way the harness that will time it does.

    **Must match `benchmark_contract.TENSOR_GEOMETRY`.** Verifying a scene with
    a letterbox while the harness stretches measures a different picture: the
    counts this function reports would be the counts for a frame nobody
    presents. Found live -- the verified scene promised 6-9 detections and the
    stretched harness saw 2-10.
    """
    cv2 = _cv2()
    if mode == "stretch":
        canvas = cv2.resize(image, (size, size))
    else:
        scale = min(size / image.shape[1], size / image.shape[0])
        resized = cv2.resize(image, (int(round(image.shape[1] * scale)),
                                     int(round(image.shape[0] * scale))))
        canvas = np.full((size, size, 3), 114, np.uint8)
        top = (size - resized.shape[0]) // 2
        left = (size - resized.shape[1]) // 2
        canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])


def verify_scene(directory, model, *, frames=8, confidence=0.25,
                 minimum=MINIMUM_DETECTIONS_PER_FRAME, geometry_mode=None) -> dict:
    """Put the scene through the real model and record what it finds.

    This is what makes the scene trustworthy. A scene is only usable for a
    detection benchmark if the detector *actually detects things in it*, at
    every workload, at the offsets the source will present -- and the only way
    to know that is to ask the model.

    Writes the detections beside the scene as a reference: a later run whose
    detections disagree with it is looking at a different scene, not a faster
    pipeline.
    """
    import onnxruntime as ort
    from . import detection as detection_module

    directory = Path(directory)
    manifest = load_manifest(directory)
    canvas = load_canvas(directory)
    width, height = manifest["capture_width"], manifest["capture_height"]

    # The geometry the harness will actually use, not a plausible default.
    from .benchmark_contract import TENSOR_GEOMETRY
    geometry_mode = geometry_mode or TENSOR_GEOMETRY

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    geometry = detection_module.Geometry(width, height, 640, mode=geometry_mode)
    contract = detection_module.DetectionContract(
        model_sha256=hashlib.sha256(Path(model).read_bytes()).hexdigest(),
        geometry_mode=geometry_mode, confidence_threshold=confidence)

    report = {"schema_version": SCHEMA_VERSION, "scene_id": manifest["scene_id"],
              "model_sha256": contract.model_sha256, "frames_per_workload": frames,
              "confidence_threshold": confidence, "geometry_mode": geometry_mode,
              "workloads": {}, "problems": []}

    for workload in WORKLOAD_STEPS:
        rows = []
        for index in range(frames):
            # Spread the sampled frames across a full lap of the canvas, so a
            # workload is judged on where it actually goes rather than on its
            # first few frames.
            frame = index * max(1, (height * 2) // max(frames, 1))
            offset_x, offset_y = offset_for(workload, frame, width, height)
            window = np.ascontiguousarray(view(canvas, width, height,
                                               offset_x, offset_y))
            apply_marker(window, frame)
            raw = session.run(None,
                              {input_name: model_tensor(window, mode=geometry_mode)})[0]
            found = detection_module.postprocess(raw, geometry, contract, np)
            rows.append({
                "frame": frame, "offset": [offset_x, offset_y],
                "count": len(found),
                "classes": sorted(COCO_NAMES[int(found.classes[i])]
                                  for i in range(len(found))),
                "scene_fingerprint": detection_module.scene_fingerprint(found),
                "detections": found.as_records(),
            })
        counts = [row["count"] for row in rows]
        report["workloads"][workload] = {
            "frames": rows,
            "min_detections": min(counts), "max_detections": max(counts),
            "mean_detections": sum(counts) / len(counts),
        }
        if min(counts) < minimum:
            report["problems"].append(
                f"{workload}: as few as {min(counts)} detection(s) in a frame, below "
                f"the {minimum} this scene needs to be timing real postprocessing "
                "rather than an empty screen")

    report["usable"] = not report["problems"]
    (directory / DETECTIONS_NAME).write_text(json.dumps(report, indent=2),
                                             encoding="utf-8")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=WORK / "scenes" / "default")
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--model", type=Path,
                        help="yolo11n.onnx; with it, the scene is verified against "
                             "the real detector and refuses to be called usable "
                             "until it detects things")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--preview", type=Path,
                        help="also write a PNG of one window, for looking at")
    args = parser.parse_args(argv)

    manifest = write_scene(args.out, args.width, args.height, args.seed)
    print(f"scene {manifest['scene_id']}: "
          f"{manifest['canvas_width']}x{manifest['canvas_height']} canvas, "
          f"{len(manifest['objects'])} objects -> {args.out}")

    if args.preview:
        cv2 = _cv2()
        canvas = load_canvas(args.out)
        window = np.ascontiguousarray(view(canvas, args.width, args.height, 0, 0))
        apply_marker(window, 4242)
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), window)
        print(f"preview -> {args.preview}")

    if not args.model:
        print("no --model given, so this scene is UNVERIFIED: nothing here has "
              "checked that a detector finds anything in it.")
        return 0

    report = verify_scene(args.out, args.model, frames=args.frames)
    for workload, summary in report["workloads"].items():
        print(f"  {workload:>7}: {summary['min_detections']}-"
              f"{summary['max_detections']} detections per frame "
              f"(mean {summary['mean_detections']:.1f})")
    for problem in report["problems"]:
        print(f"  PROBLEM: {problem}")
    print("usable" if report["usable"] else "NOT USABLE for a detection benchmark")
    return 0 if report["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
