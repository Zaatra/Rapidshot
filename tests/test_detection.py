"""Detections an application could actually branch on, and whether two
pipelines found the same ones.

The geometry tests matter most. Removing the letterbox pad after dividing by
the scale instead of before is a plausible-looking mistake that offsets every
box by roughly a hundred source pixels on this machine's panel -- far enough to
move a detection onto a different object, close enough to read as tracking
wobble. The inverse is checked against values computed by hand here.

No model is loaded and no GPU is touched: raw output is synthesised, which is
also the only way to test a detection that should be suppressed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import detection as det  # noqa: E402


CONTRACT = det.DetectionContract(model_sha256="a" * 64, confidence_threshold=0.25,
                                 iou_threshold=0.45)


def raw_output(boxes_xywh, class_scores, classes=80, anchors=None):
    """Synthesise a YOLO-shaped `(1, 4 + classes, anchors)` tensor."""
    anchors = anchors or len(boxes_xywh)
    out = np.zeros((1, 4 + classes, anchors), dtype=np.float32)
    for index, (box, (label, score)) in enumerate(zip(boxes_xywh, class_scores)):
        out[0, :4, index] = box
        out[0, 4 + label, index] = score
    return out


def detections_from(boxes, classes, scores):
    return det.Detections(boxes=np.array(boxes, dtype=np.float32),
                          classes=np.array(classes, dtype=np.int64),
                          scores=np.array(scores, dtype=np.float32),
                          contract=CONTRACT)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_letterbox_uses_one_scale_and_centres_the_padding():
    geometry = det.Geometry(2560, 1600, 640, mode="letterbox")
    # 640/2560 = 0.25 is the binding factor; 1600*0.25 = 400, leaving 240 to
    # split evenly top and bottom.
    assert geometry.scale == (0.25, 0.25)
    assert geometry.padding == (0.0, 120.0)


def test_stretch_uses_independent_scales_and_no_padding():
    geometry = det.Geometry(2560, 1600, 640, mode="stretch")
    assert geometry.scale == (0.25, 0.4)
    assert geometry.padding == (0.0, 0.0)


def test_restoring_a_letterboxed_box_removes_the_pad_before_dividing():
    """The mistake this catches offsets every box by ~120/0.25 = 480 px."""
    geometry = det.Geometry(2560, 1600, 640, mode="letterbox")
    # A box at model-space (160, 220) to (240, 300). Undo pad: y 100..180.
    # Undo scale: x 640..960, y 400..720.
    boxes = np.array([[160.0, 220.0, 240.0, 300.0]], dtype=np.float32)
    restored = det.restore_boxes(boxes, geometry, np)
    np.testing.assert_allclose(restored[0], [640.0, 400.0, 960.0, 720.0], atol=1e-4)


def test_restoring_a_stretched_box_uses_both_scales():
    geometry = det.Geometry(2560, 1600, 640, mode="stretch")
    boxes = np.array([[160.0, 160.0, 320.0, 320.0]], dtype=np.float32)
    restored = det.restore_boxes(boxes, geometry, np)
    np.testing.assert_allclose(restored[0], [640.0, 400.0, 1280.0, 800.0], atol=1e-4)


def test_restored_boxes_are_clamped_to_the_source_frame():
    geometry = det.Geometry(1920, 1080, 640, mode="stretch")
    boxes = np.array([[-50.0, -50.0, 700.0, 700.0]], dtype=np.float32)
    restored = det.restore_boxes(boxes, geometry, np)
    assert restored[0][0] == 0.0 and restored[0][1] == 0.0
    assert restored[0][2] == 1920.0 and restored[0][3] == 1080.0


def test_a_round_trip_through_the_geometry_returns_the_original():
    geometry = det.Geometry(2560, 1600, 640, mode="letterbox")
    scale_x, scale_y = geometry.scale
    pad_x, pad_y = geometry.padding
    source = np.array([[100.0, 200.0, 900.0, 1400.0]], dtype=np.float32)
    forward = np.array([[source[0][0] * scale_x + pad_x,
                         source[0][1] * scale_y + pad_y,
                         source[0][2] * scale_x + pad_x,
                         source[0][3] * scale_y + pad_y]], dtype=np.float32)
    np.testing.assert_allclose(det.restore_boxes(forward, geometry, np), source,
                               atol=1e-3)


@pytest.mark.parametrize("mode", ["letterbox", "stretch"])
def test_a_square_source_makes_the_two_modes_agree(mode):
    geometry = det.Geometry(800, 800, 640, mode=mode)
    assert geometry.scale == (0.8, 0.8) and geometry.padding == (0.0, 0.0)


def test_an_unknown_geometry_mode_is_refused():
    with pytest.raises(ValueError):
        det.Geometry(1920, 1080, 640, mode="squash")


# ---------------------------------------------------------------------------
# Boxes and suppression
# ---------------------------------------------------------------------------

def test_centre_size_becomes_corners():
    boxes = np.array([[100.0, 200.0, 40.0, 60.0]], dtype=np.float32)
    np.testing.assert_allclose(det.xywh_to_xyxy(boxes, np)[0],
                               [80.0, 170.0, 120.0, 230.0])


def test_nms_keeps_the_highest_score_of_an_overlapping_pair():
    boxes = np.array([[0, 0, 100, 100], [5, 5, 105, 105]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    assert det.nms(boxes, scores, 0.45, np) == [1]


def test_nms_keeps_boxes_that_do_not_overlap_enough():
    boxes = np.array([[0, 0, 100, 100], [200, 200, 300, 300]], dtype=np.float32)
    scores = np.array([0.6, 0.9], dtype=np.float32)
    assert sorted(det.nms(boxes, scores, 0.45, np)) == [0, 1]


def test_nms_respects_the_detection_cap():
    boxes = np.array([[i * 500, 0, i * 500 + 10, 10] for i in range(10)],
                     dtype=np.float32)
    scores = np.linspace(0.9, 0.3, 10).astype(np.float32)
    assert len(det.nms(boxes, scores, 0.45, np, max_detections=3)) == 3


def test_a_zero_area_box_does_not_divide_by_zero():
    boxes = np.array([[10, 10, 10, 10], [0, 0, 100, 100]], dtype=np.float32)
    scores = np.array([0.9, 0.8], dtype=np.float32)
    assert sorted(det.nms(boxes, scores, 0.45, np)) == [0, 1]


def test_suppression_is_per_class_by_default():
    """Two different objects in the same place are two detections, not one."""
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    raw = raw_output([[100, 100, 50, 50], [102, 102, 50, 50]],
                     [(0, 0.9), (1, 0.8)])
    result = det.postprocess(raw, geometry, CONTRACT, np)
    assert len(result) == 2
    assert sorted(int(value) for value in result.classes) == [0, 1]


def test_class_agnostic_suppression_collapses_them():
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    contract = det.DetectionContract(model_sha256="a" * 64,
                                     class_agnostic_nms=True)
    raw = raw_output([[100, 100, 50, 50], [102, 102, 50, 50]],
                     [(0, 0.9), (1, 0.8)])
    assert len(det.postprocess(raw, geometry, contract, np)) == 1


# ---------------------------------------------------------------------------
# Postprocessing end to end
# ---------------------------------------------------------------------------

def test_detections_come_back_in_source_coordinates():
    geometry = det.Geometry(2560, 1600, 640, mode="letterbox")
    raw = raw_output([[200.0, 220.0, 80.0, 80.0]], [(5, 0.9)])
    result = det.postprocess(raw, geometry, CONTRACT, np)
    # Model-space corners 160..240 x, 180..260 y. Undo pad 120 on y, then /0.25.
    np.testing.assert_allclose(result.boxes[0], [640.0, 240.0, 960.0, 560.0],
                               atol=1e-3)
    assert int(result.classes[0]) == 5


def test_low_confidence_anchors_never_reach_suppression():
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    raw = raw_output([[100, 100, 50, 50], [300, 300, 50, 50]],
                     [(0, 0.9), (1, 0.10)])
    result = det.postprocess(raw, geometry, CONTRACT, np)
    assert len(result) == 1 and int(result.classes[0]) == 0


def test_a_frame_with_nothing_in_it_yields_no_detections():
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    raw = raw_output([[100, 100, 50, 50]], [(0, 0.05)])
    result = det.postprocess(raw, geometry, CONTRACT, np)
    assert len(result) == 0 and result.summary()["count"] == 0


def test_the_transfer_to_the_host_happens_inside_the_timed_function():
    """A GPU pipeline that skipped the copy would be credited with work it
    never finished, so the move is part of postprocess, not after it."""
    moved = []

    def to_host(value):
        moved.append(value.shape)
        return np.asarray(value)

    geometry = det.Geometry(640, 640, 640, mode="stretch")
    raw = raw_output([[100, 100, 50, 50]], [(0, 0.9)])
    det.postprocess(raw, geometry, CONTRACT, np, to_host=to_host)
    assert len(moved) == 3            # boxes, classes, scores


def test_detections_are_readable_as_plain_rows():
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    raw = raw_output([[100, 100, 50, 50]], [(3, 0.77)])
    records = det.postprocess(raw, geometry, CONTRACT, np).as_records()
    assert records[0]["class"] == 3
    assert records[0]["score"] == pytest.approx(0.77, abs=1e-5)
    assert len(records[0]["box"]) == 4


@pytest.mark.parametrize("shape", [(84, 8400), (1, 8400, 84, 2), (1, 3, 8400)])
def test_an_output_this_postprocessing_cannot_read_is_refused(shape):
    geometry = det.Geometry(640, 640, 640, mode="stretch")
    with pytest.raises(det.DetectionError):
        det.postprocess(np.zeros(shape, dtype=np.float32), geometry, CONTRACT, np)


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------

def test_a_contract_fingerprint_moves_with_every_setting_that_matters():
    base = det.DetectionContract(model_sha256="a" * 64)
    for change in ({"confidence_threshold": 0.3}, {"iou_threshold": 0.5},
                   {"geometry_mode": "stretch"}, {"dtype": "float32"},
                   {"model_sha256": "b" * 64}, {"output_location": "device"}):
        assert det.DetectionContract(
            **{**dataclass_fields(base), **change}).fingerprint != base.fingerprint


def dataclass_fields(contract):
    return contract.as_dict()


def test_mismatched_contracts_name_what_differs():
    a = det.DetectionContract(model_sha256="a" * 64, confidence_threshold=0.25)
    b = det.DetectionContract(model_sha256="a" * 64, confidence_threshold=0.30,
                              geometry_mode="stretch")
    differences = a.differences(b)
    assert any("confidence_threshold" in item for item in differences)
    assert any("geometry_mode" in item for item in differences)


def test_the_contract_reaches_the_case_configuration():
    suffix = det.DetectionContract(model_sha256="a" * 64).configuration_suffix()
    assert "letterbox" in suffix and "conf0.25" in suffix and "host" in suffix


def test_the_default_output_location_is_the_host():
    """An application branching on a box cannot read it from device memory."""
    assert det.DetectionContract(model_sha256="a" * 64).output_location == "host"


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def test_identical_detections_are_at_parity():
    one = detections_from([[10, 10, 50, 50]], [0], [0.9])
    parity = det.detection_parity(one, detections_from([[10, 10, 50, 50]], [0], [0.9]))
    assert parity.ok and parity.matched == 1


def test_a_sub_pixel_drift_is_tolerated_and_reported():
    reference = detections_from([[10, 10, 50, 50]], [0], [0.90])
    candidate = detections_from([[10.4, 10.2, 50.1, 50.3]], [0], [0.901])
    parity = det.detection_parity(reference, candidate)
    assert parity.ok
    assert parity.max_box_deviation_px == pytest.approx(0.4, abs=1e-5)


def test_a_missing_detection_is_never_tolerated():
    """A tolerance wide enough to hide a missing object would defeat the check."""
    reference = detections_from([[10, 10, 50, 50], [100, 100, 150, 150]],
                                [0, 1], [0.9, 0.8])
    candidate = detections_from([[10, 10, 50, 50]], [0], [0.9])
    parity = det.detection_parity(reference, candidate, box_tolerance_px=10000)
    assert parity.ok is False
    assert parity.only_in_reference[0]["class"] == 1


def test_an_extra_detection_is_reported_too():
    reference = detections_from([[10, 10, 50, 50]], [0], [0.9])
    candidate = detections_from([[10, 10, 50, 50], [300, 300, 350, 350]],
                                [0, 2], [0.9, 0.7])
    parity = det.detection_parity(reference, candidate)
    assert parity.ok is False and parity.only_in_candidate[0]["class"] == 2


def test_the_same_box_with_a_different_class_is_not_a_match():
    reference = detections_from([[10, 10, 50, 50]], [0], [0.9])
    candidate = detections_from([[10, 10, 50, 50]], [7], [0.9])
    parity = det.detection_parity(reference, candidate)
    assert parity.ok is False
    assert parity.only_in_reference and parity.only_in_candidate


def test_a_confidence_gap_beyond_tolerance_fails_even_when_boxes_match():
    reference = detections_from([[10, 10, 50, 50]], [0], [0.90])
    candidate = detections_from([[10, 10, 50, 50]], [0], [0.60])
    parity = det.detection_parity(reference, candidate)
    assert parity.ok is False
    assert "confidence differs" in parity.reasons[0]


def test_pipelines_that_disagreed_on_the_contract_cannot_be_at_parity():
    reference = detections_from([[10, 10, 50, 50]], [0], [0.9])
    candidate = detections_from([[10, 10, 50, 50]], [0], [0.9])
    candidate.contract = det.DetectionContract(model_sha256="a" * 64,
                                               confidence_threshold=0.5)
    parity = det.detection_parity(reference, candidate)
    assert parity.ok is False
    assert "did not agree on the detection contract" in parity.reasons[0]


def test_a_scene_fingerprint_is_stable_under_sub_pixel_noise():
    one = detections_from([[10.0, 10.0, 50.0, 50.0]], [0], [0.900])
    two = detections_from([[10.2, 9.8, 50.1, 50.2]], [0], [0.902])
    assert det.scene_fingerprint(one) == det.scene_fingerprint(two)


def test_a_scene_fingerprint_moves_when_the_scene_does():
    one = detections_from([[10, 10, 50, 50]], [0], [0.9])
    two = detections_from([[400, 400, 450, 450]], [0], [0.9])
    assert det.scene_fingerprint(one) != det.scene_fingerprint(two)


# ---------------------------------------------------------------------------
# Delivery metrics
# ---------------------------------------------------------------------------

def records(frame_ids, ages=None):
    ages = ages or [33.0] * len(frame_ids)
    return [{"frame_id": value, "age_ms": age}
            for value, age in zip(frame_ids, ages)]


def test_duplicates_are_counted_not_credited():
    """Returning the same frame twice is not delivering two detections."""
    metrics = det.detection_metrics(records([1, 1, 2, 2, 3]))
    assert metrics["unique_detected_frames"] == 3
    assert metrics["duplicate_frames"] == 2
    assert metrics["calls"] == 5


def test_calls_that_returned_nothing_are_counted_separately():
    metrics = det.detection_metrics(records([1, None, None, 2]))
    assert metrics["empty_calls"] == 2
    assert metrics["unique_detected_frames"] == 2


def test_jitter_is_robust_to_one_stalled_frame():
    """At a realistic frame count, one stall moves the median barely at all.

    A stall costs *two* successive differences -- into it and out of it -- so
    at five frames it is half the sample and dominates. At the ~1000 frames an
    eight-second run returns it is noise, which is the regime this metric is
    for.
    """
    ages = [33.0 + (index % 3) * 0.1 for index in range(400)]
    steady = det.detection_metrics(records(list(range(400)), ages))
    ages[200] = 90.0
    stalled = det.detection_metrics(records(list(range(400)), ages))
    assert stalled["delivery_jitter_ms"] == pytest.approx(
        steady["delivery_jitter_ms"], abs=0.05)


def test_an_empty_run_reports_no_jitter_rather_than_zero():
    assert det.detection_metrics([])["delivery_jitter_ms"] is None
