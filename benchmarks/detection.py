"""From a model's raw output to detections an application can actually use.

ROADMAP § 7.0's inference table stops the clock when the forward pass
completes, and says so: *"Raw detector output, no NMS or postprocessing
timed."* That is the wrong boundary for the claim the benchmark is trying to
make. An application does not receive a `(1, 84, 8400)` tensor sitting on the
GPU; it receives boxes, class ids and confidence scores it can branch on. The
work between those two points -- confidence filtering, non-maximum suppression,
coordinate restoration, and **the transfer back to the host** -- is real, it
scales with how much is on screen, and leaving it out flatters every path
equally right up until one of them moves it to the GPU.

So the timer here ends at :class:`Detections`, on the CPU, readable.

**Matched settings or no comparison.** Two pipelines that differ in
letterboxing, confidence threshold, NMS IoU or class filtering are not
measuring the same work, and the difference can be larger than anything being
claimed -- a lower threshold means more boxes survive into NMS, which is
quadratic in the worst case. :class:`DetectionContract` is the declaration of
those settings, hashed into the case configuration so two runs that disagree
cannot be pooled by accident.

**Geometry is a correctness problem, not a formatting one.** Ultralytics
letterboxes -- scales by a single factor and pads to square -- while the
existing § 7.0 contract stretches the full frame to 640x640. Both are
defensible; running one against the other and comparing detections is not,
because the boxes come back in different coordinate systems and the model sees
differently distorted objects. :class:`Geometry` makes the choice explicit and
invertible, and :func:`restore_boxes` is checked against a hand-computed
inverse in the tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

SCHEMA_VERSION = 1

#: YOLO11n's exported layout: 4 box coordinates then one score per class,
#: across every anchor. Declared rather than inferred, so a model with a
#: different head fails loudly instead of being silently misread as boxes.
YOLO_BOX_ROWS = 4


class DetectionError(ValueError):
    """The model output is not the shape this postprocessing understands."""


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Geometry:
    """How a source frame was mapped into the model's square input.

    ``mode`` is ``"letterbox"`` (one scale factor, padded to square -- what
    Ultralytics does) or ``"stretch"`` (independent x and y scales, filling the
    square -- what the existing section 7 contract does). Both are invertible;
    they are not interchangeable, and a comparison that mixes them is comparing
    two different pictures.
    """

    source_width: int
    source_height: int
    model_size: int
    mode: str = "letterbox"

    def __post_init__(self):
        if self.mode not in ("letterbox", "stretch"):
            raise ValueError(f"unknown geometry mode {self.mode!r}")
        if min(self.source_width, self.source_height, self.model_size) <= 0:
            raise ValueError("source and model dimensions must be positive")

    @property
    def scale(self):
        """``(scale_x, scale_y)`` applied to the source to reach model space."""
        if self.mode == "stretch":
            return (self.model_size / self.source_width,
                    self.model_size / self.source_height)
        factor = min(self.model_size / self.source_width,
                     self.model_size / self.source_height)
        return (factor, factor)

    @property
    def padding(self):
        """``(pad_x, pad_y)`` -- half the leftover, as Ultralytics centres it."""
        if self.mode == "stretch":
            return (0.0, 0.0)
        scale_x, scale_y = self.scale
        return ((self.model_size - self.source_width * scale_x) / 2.0,
                (self.model_size - self.source_height * scale_y) / 2.0)

    def as_dict(self) -> dict:
        scale_x, scale_y = self.scale
        pad_x, pad_y = self.padding
        return {"mode": self.mode, "source_width": self.source_width,
                "source_height": self.source_height, "model_size": self.model_size,
                "scale_x": scale_x, "scale_y": scale_y,
                "pad_x": pad_x, "pad_y": pad_y}


def restore_boxes(boxes, geometry: Geometry, xp):
    """Map boxes from model space back to source pixels, then clamp.

    Padding is removed **before** dividing by the scale, not after. Doing it in
    the other order is a plausible-looking mistake that offsets every box by
    the pad in source units, which on a 2560x1600 frame at 640 is about 100
    pixels -- large enough to move a detection onto a different object and
    small enough to look like a tracking wobble.
    """
    if boxes.shape[0] == 0:
        return boxes
    scale_x, scale_y = geometry.scale
    pad_x, pad_y = geometry.padding
    restored = xp.empty_like(boxes)
    restored[:, 0] = (boxes[:, 0] - pad_x) / scale_x
    restored[:, 1] = (boxes[:, 1] - pad_y) / scale_y
    restored[:, 2] = (boxes[:, 2] - pad_x) / scale_x
    restored[:, 3] = (boxes[:, 3] - pad_y) / scale_y
    restored[:, 0] = xp.clip(restored[:, 0], 0, geometry.source_width)
    restored[:, 2] = xp.clip(restored[:, 2], 0, geometry.source_width)
    restored[:, 1] = xp.clip(restored[:, 1], 0, geometry.source_height)
    restored[:, 3] = xp.clip(restored[:, 3], 0, geometry.source_height)
    return restored


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class DetectionContract:
    """Every setting two pipelines must share to be measuring the same work."""

    model_sha256: str
    model_size: int = 640
    dtype: str = "float16"
    geometry_mode: str = "letterbox"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 300
    class_agnostic_nms: bool = False
    #: Where the detections must be readable from when the timer stops. "host"
    #: is the only honest default: an application branching on a box cannot do
    #: it from device memory without paying for the transfer itself.
    output_location: str = "host"

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()[:16]

    def configuration_suffix(self) -> str:
        """The part of a case identity that this contract decides."""
        return (f"{self.geometry_mode}-conf{self.confidence_threshold:g}"
                f"-iou{self.iou_threshold:g}-{self.dtype}-{self.output_location}")

    def differences(self, other: "DetectionContract") -> list:
        """Every setting on which two pipelines disagree, named.

        Returned rather than raised so a caller can record an unmatched
        comparison as unmatched, instead of a run failing with no evidence of
        what it would have measured.
        """
        mine, theirs = self.as_dict(), other.as_dict()
        return [f"{key}: {mine[key]!r} vs {theirs[key]!r}"
                for key in sorted(mine) if mine[key] != theirs[key]]


# ---------------------------------------------------------------------------
# Detections
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Detections:
    """What an application receives: boxes, classes and scores, on the host."""

    boxes: object                 # (n, 4) xyxy in source pixels
    classes: object               # (n,) int
    scores: object                # (n,) float
    geometry: Geometry = None
    contract: DetectionContract = None

    def __len__(self) -> int:
        return int(self.boxes.shape[0]) if self.boxes is not None else 0

    def as_records(self) -> list:
        """Plain rows, for committing to the result store."""
        return [{"box": [float(value) for value in self.boxes[index]],
                 "class": int(self.classes[index]),
                 "score": float(self.scores[index])}
                for index in range(len(self))]

    def summary(self) -> dict:
        return {"count": len(self),
                "classes": sorted({int(value) for value in self.classes}),
                "max_score": float(self.scores.max()) if len(self) else None,
                "min_score": float(self.scores.min()) if len(self) else None}


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------

def xywh_to_xyxy(boxes, xp):
    """Centre/size to corners. YOLO emits the former; everything else wants the latter."""
    half_w = boxes[:, 2] / 2
    half_h = boxes[:, 3] / 2
    out = xp.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - half_w
    out[:, 1] = boxes[:, 1] - half_h
    out[:, 2] = boxes[:, 0] + half_w
    out[:, 3] = boxes[:, 1] + half_h
    return out


def nms(boxes, scores, iou_threshold: float, xp, max_detections: int = 300):
    """Greedy non-maximum suppression, highest score first.

    Deliberately the plain algorithm rather than a fused kernel: it has to give
    the same answer on NumPy and CuPy for the parity check to mean anything,
    and at the box counts that survive a 0.25 confidence threshold it is not
    where the time goes.
    """
    order = xp.argsort(-scores)
    keep = []
    areas = ((boxes[:, 2] - boxes[:, 0]).clip(0)
             * (boxes[:, 3] - boxes[:, 1]).clip(0))
    while order.size > 0 and len(keep) < max_detections:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        left = xp.maximum(boxes[current, 0], boxes[rest, 0])
        top = xp.maximum(boxes[current, 1], boxes[rest, 1])
        right = xp.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = xp.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = (right - left).clip(0) * (bottom - top).clip(0)
        union = areas[current] + areas[rest] - overlap
        # A zero-area box would otherwise divide by zero and suppress or keep
        # arbitrarily; it is simply never overlapping anything.
        iou = xp.where(union > 0, overlap / xp.where(union > 0, union, 1), 0.0)
        order = rest[iou <= iou_threshold]
    return keep


def postprocess(raw, geometry: Geometry, contract: DetectionContract, xp,
                to_host=None) -> Detections:
    """Raw model output to host-readable detections, timed as one unit.

    ``raw`` is ``(1, 4 + classes, anchors)`` as YOLO11n exports it, on whatever
    device ``xp`` addresses. The transfer to the host happens **inside** this
    function because the contract says detections are usable when the
    application can read them, and a GPU pipeline that skipped the copy would
    be credited with work it never finished.
    """
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise DetectionError(f"expected (1, 4+classes, anchors), got {raw.shape}")
    channels = raw.shape[1]
    if channels <= YOLO_BOX_ROWS:
        raise DetectionError(f"{channels} channels leaves no room for class scores")

    predictions = raw[0].T                       # (anchors, 4 + classes)
    boxes = predictions[:, :YOLO_BOX_ROWS].astype("float32")
    class_scores = predictions[:, YOLO_BOX_ROWS:].astype("float32")
    best = class_scores.max(axis=1)
    classes = class_scores.argmax(axis=1)

    survivors = best >= contract.confidence_threshold
    boxes = xywh_to_xyxy(boxes[survivors], xp)
    scores = best[survivors]
    classes = classes[survivors]

    if boxes.shape[0]:
        boxes = restore_boxes(boxes, geometry, xp)
        keep = _suppress(boxes, scores, classes, contract, xp)
        index = xp.asarray(keep, dtype="int64") if keep else xp.zeros(0, dtype="int64")
        boxes, scores, classes = boxes[index], scores[index], classes[index]

    # The transfer that makes them usable. Counted, not skipped.
    move = to_host or (lambda value: value)
    return Detections(boxes=move(boxes), classes=move(classes), scores=move(scores),
                      geometry=geometry, contract=contract)


def _suppress(boxes, scores, classes, contract, xp):
    if contract.class_agnostic_nms:
        return nms(boxes, scores, contract.iou_threshold, xp,
                   contract.max_detections)
    keep = []
    for label in [int(value) for value in xp.unique(classes)]:
        mask = classes == label
        indices = xp.nonzero(mask)[0]
        chosen = nms(boxes[mask], scores[mask], contract.iou_threshold, xp,
                     contract.max_detections)
        keep += [int(indices[position]) for position in chosen]
    # Highest confidence first, so a truncated list is the most confident one
    # rather than whichever class happened to sort first.
    keep.sort(key=lambda position: float(scores[position]), reverse=True)
    return keep[:contract.max_detections]


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Parity:
    """Whether two pipelines found the same things in the same frame."""

    matched: int = 0
    only_in_reference: list = dataclasses.field(default_factory=list)
    only_in_candidate: list = dataclasses.field(default_factory=list)
    max_box_deviation_px: float = 0.0
    max_score_deviation: float = 0.0
    reasons: list = dataclasses.field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (not self.only_in_reference and not self.only_in_candidate
                and not self.reasons)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "matched": self.matched,
                "only_in_reference": list(self.only_in_reference),
                "only_in_candidate": list(self.only_in_candidate),
                "max_box_deviation_px": self.max_box_deviation_px,
                "max_score_deviation": self.max_score_deviation,
                "reasons": list(self.reasons)}


def detection_parity(reference: Detections, candidate: Detections, *,
                     box_tolerance_px: float = 2.0,
                     score_tolerance: float = 0.02) -> Parity:
    """Compare two detection sets from the *same source frame*.

    Tolerances rather than equality, and declared rather than implied: FP16
    arithmetic, a different resize kernel and a different NMS tie-break all
    move a box by a fraction of a pixel without changing what was detected. A
    tolerance wide enough to hide a *missing* object would defeat the check, so
    an unmatched detection is always a failure regardless of tolerance -- only
    the positions of matched ones are allowed to drift.
    """
    parity = Parity()
    if reference.contract and candidate.contract:
        differences = reference.contract.differences(candidate.contract)
        if differences:
            parity.reasons.append(
                "the two pipelines did not agree on the detection contract: "
                + "; ".join(differences))

    remaining = list(range(len(candidate)))
    for index in range(len(reference)):
        label = int(reference.classes[index])
        box = [float(value) for value in reference.boxes[index]]
        best, best_distance = None, None
        for position in remaining:
            if int(candidate.classes[position]) != label:
                continue
            other = [float(value) for value in candidate.boxes[position]]
            distance = max(abs(a - b) for a, b in zip(box, other))
            if best_distance is None or distance < best_distance:
                best, best_distance = position, distance
        if best is None or best_distance > box_tolerance_px:
            parity.only_in_reference.append(
                {"class": label, "box": box,
                 "score": float(reference.scores[index]),
                 "nearest_deviation_px": best_distance})
            continue
        remaining.remove(best)
        parity.matched += 1
        parity.max_box_deviation_px = max(parity.max_box_deviation_px, best_distance)
        score_gap = abs(float(reference.scores[index]) - float(candidate.scores[best]))
        parity.max_score_deviation = max(parity.max_score_deviation, score_gap)
        if score_gap > score_tolerance:
            parity.reasons.append(
                f"class {label} matched but confidence differs by {score_gap:.4f}, "
                f"beyond the {score_tolerance} tolerance")

    parity.only_in_candidate = [
        {"class": int(candidate.classes[position]),
         "box": [float(value) for value in candidate.boxes[position]],
         "score": float(candidate.scores[position])}
        for position in remaining]
    return parity


def scene_fingerprint(detections: Detections) -> str:
    """A digest of what was detected, for spotting a scene that changed.

    Rounded to whole pixels and two decimal places of confidence on purpose:
    the question this answers is "is this the same scene with the same objects
    in it", not "are these bit-identical", and an exact digest would report a
    difference on every frame.
    """
    rows = sorted((int(detections.classes[index]),
                   *[round(float(value)) for value in detections.boxes[index]],
                   round(float(detections.scores[index]), 2))
                  for index in range(len(detections)))
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def detection_metrics(records) -> dict:
    """Throughput and delivery quality, from per-frame detection records.

    ``records`` is one entry per *call*, each with ``frame_id`` (``None`` when
    the capture returned nothing new) and ``age_ms``. Duplicates and misses are
    counted rather than filtered away: a pipeline that returns the same frame
    twice as fast as another returns new ones has not delivered more
    detections, and a throughput figure that cannot tell those apart is the one
    that makes it look like it has.
    """
    ids = [row.get("frame_id") for row in records]
    unique = {value for value in ids if value is not None}
    ages = [row["age_ms"] for row in records
            if row.get("frame_id") is not None and _finite(row.get("age_ms"))]
    deltas = [abs(later - earlier) for earlier, later in zip(ages, ages[1:])]
    return {
        "calls": len(records),
        "unique_detected_frames": len(unique),
        "duplicate_frames": sum(1 for value in ids if value is not None) - len(unique),
        "empty_calls": sum(1 for value in ids if value is None),
        # Median absolute successive difference: a jitter figure one stalled
        # frame cannot dominate the way a standard deviation can. Note a stall
        # costs *two* differences -- into it and out of it -- so this is a
        # robust statistic over a run's worth of frames and not over a handful.
        "delivery_jitter_ms": _median(deltas),
    }


def _finite(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _median(values):
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2
