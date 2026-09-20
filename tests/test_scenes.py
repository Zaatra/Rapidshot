"""Whether the scene is the one that was verified, and whether it tiles.

The scene exists because a detection benchmark against the frame-ID pattern
measures an empty screen. That makes two properties load-bearing: the objects
have to be the sizes that were checked against the real model, and the canvas
has to wrap without a seam, because every pan offset is a frame someone will
time.

No model is loaded here -- that is `scenes.py --model`, which needs
onnxruntime. These tests cover the geometry, the tiling, the identity and the
refusals.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import scenes  # noqa: E402

pytest.importorskip("cv2", reason="scene rendering needs OpenCV")

SMALL = (320, 200)


def build(tmp_path, width=SMALL[0], height=SMALL[1], seed=7):
    return scenes.write_scene(tmp_path / "scene", width, height, seed)


# ---------------------------------------------------------------------------
# Object sizing
# ---------------------------------------------------------------------------

def test_objects_are_sized_in_model_space_not_screen_space():
    """The bug this encodes: a fixed pixel size is a different object to the
    model at every capture resolution, detected at one and missed at another."""
    big = scenes.layout(5120, 3200, 2560, 1600)
    small = scenes.layout(2560, 1600, 1280, 800)
    for placement in big:
        matching = [p for p in small if p.kind == placement.kind]
        assert matching
    # 2560x1600 letterboxes by 0.25, 1280x800 by 0.5, so source sizes differ by
    # exactly two while model-space sizes match.
    assert scenes.letterbox_scale(2560, 1600) == 0.25
    assert scenes.letterbox_scale(1280, 800) == 0.5
    biggest = max(p.size for p in big if p.kind == "clock")
    smallest = max(p.size for p in small if p.kind == "clock")
    assert biggest == pytest.approx(smallest * 2, rel=0.01)


def test_every_object_is_one_of_the_verified_classes():
    """Adding a class that was never checked gives a scene that quietly empties."""
    for placement in scenes.layout(5120, 3200, 2560, 1600):
        assert placement.kind in scenes.VERIFIED_CLASSES


def test_the_layout_mixes_classes_across_columns():
    """An arithmetic phase looked mixed and was not: with 7 columns and 4
    classes it reduced to `column % 4`, and the first scene rendered as three
    vertical stripes of identical objects."""
    placements = scenes.layout(5120, 3200, 2560, 1600)
    by_column = {}
    for index, placement in enumerate(placements):
        by_column.setdefault(index % scenes.LAYOUT_COLUMNS, set()).add(placement.kind)
    assert sum(1 for kinds in by_column.values() if len(kinds) > 1) >= 4


def test_the_layout_is_deterministic_for_a_seed():
    first = scenes.layout(5120, 3200, 2560, 1600, seed=99)
    second = scenes.layout(5120, 3200, 2560, 1600, seed=99)
    assert [p.as_dict() for p in first] == [p.as_dict() for p in second]
    other = scenes.layout(5120, 3200, 2560, 1600, seed=100)
    assert [p.as_dict() for p in first] != [p.as_dict() for p in other]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------

def test_the_canvas_is_twice_the_capture_size_on_each_axis():
    canvas = scenes.render_canvas(*SMALL)
    assert canvas.shape[:2] == (SMALL[1] * 2, SMALL[0] * 2)


def test_the_canvas_wraps_without_a_seam():
    """Every pan offset is a frame someone will time, including the ones that
    straddle the edge."""
    canvas = scenes.render_canvas(*SMALL)
    # A window taken at the wrap point is the same pixels as the two halves
    # taken separately, which is what "tileable" has to mean here.
    width, height = SMALL
    offset = canvas.shape[1] - width // 2
    wrapped = scenes.view(canvas, width, height, offset, 0)
    left = canvas[:height, offset:]
    right = canvas[:height, :width - left.shape[1]]
    np.testing.assert_array_equal(wrapped, np.hstack([left, right]))


def test_objects_are_drawn_across_the_seam():
    """Without the wrapped copies an object is sliced in half at the edge and
    the scroll workload shows a hard discontinuity once per lap."""
    canvas = scenes.render_canvas(*SMALL)
    background = np.array([44, 48, 56], np.uint8)
    # Some column within an object's width of the right edge must be non-empty,
    # or nothing was drawn across the seam.
    strip = canvas[:, -40:]
    assert not np.all(np.abs(strip.astype(int) - background) < 12)


def test_a_window_is_exactly_the_capture_size():
    canvas = scenes.render_canvas(*SMALL)
    window = scenes.view(canvas, SMALL[0], SMALL[1], 137, 91)
    assert window.shape == (SMALL[1], SMALL[0], 3)


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------

def test_static_does_not_move():
    assert scenes.offset_for("static", 0, 320, 200) == (0, 0)
    assert scenes.offset_for("static", 9999, 320, 200) == (0, 0)


def test_scroll_moves_on_one_axis_and_motion_on_both():
    assert scenes.offset_for("scroll", 10, 320, 200)[0] == 0
    assert scenes.offset_for("scroll", 10, 320, 200)[1] > 0
    x, y = scenes.offset_for("motion", 10, 320, 200)
    assert x > 0 and y > 0 and x != y


def test_the_offset_wraps_within_the_canvas():
    for frame in (0, 1, 1000, 10_000_000):
        x, y = scenes.offset_for("motion", frame, 320, 200)
        assert 0 <= x < 640 and 0 <= y < 400


def test_an_unknown_workload_is_refused():
    with pytest.raises(ValueError):
        scenes.offset_for("wobble", 0, 320, 200)


# ---------------------------------------------------------------------------
# The marker
# ---------------------------------------------------------------------------

def test_the_marker_matches_the_shader_layout():
    """Verifying a scene has to verify the thing the source will present, so
    this must stay byte-identical to the pattern latency_source.rs draws."""
    image = np.zeros((32, 512, 3), np.uint8)
    scenes.apply_marker(image, frame=0)
    # The first eight cells are the magic 167 = 0b10100111, low bit first.
    bits = [int(image[0, cell * 8, 0] > 0) for cell in range(8)]
    assert bits == [(167 >> index) & 1 for index in range(8)]
    # Rows 16 and below are untouched.
    assert np.all(image[16:] == 0)


def test_the_marker_changes_with_the_frame():
    a = np.zeros((16, 384, 3), np.uint8)
    b = np.zeros((16, 384, 3), np.uint8)
    scenes.apply_marker(a, 1)
    scenes.apply_marker(b, 2)
    assert not np.array_equal(a, b)


def test_the_marker_covers_only_the_top_left():
    image = np.full((200, 800, 3), 99, np.uint8)
    scenes.apply_marker(image, 4242)
    assert np.all(image[:16, 384:] == 99)
    assert np.all(image[16:, :] == 99)


# ---------------------------------------------------------------------------
# Writing and reading back
# ---------------------------------------------------------------------------

def test_a_written_scene_round_trips(tmp_path):
    manifest = build(tmp_path)
    canvas = scenes.load_canvas(tmp_path / "scene")
    np.testing.assert_array_equal(canvas, scenes.render_canvas(*SMALL, seed=7))
    assert manifest["canvas_width"] == SMALL[0] * 2
    assert manifest["format"] == "BGRA8"


def test_a_tampered_background_is_refused_on_read(tmp_path):
    """The manifest hash is what says the bytes are the ones that were verified."""
    build(tmp_path)
    target = tmp_path / "scene" / scenes.BACKGROUND_NAME
    data = bytearray(target.read_bytes())
    data[1000] ^= 0xFF
    target.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="does not match its manifest hash"):
        scenes.load_canvas(tmp_path / "scene")


def test_the_background_is_exactly_the_bytes_the_source_expects(tmp_path):
    """The source is handed dimensions on the command line and validates the
    file length against them, so a mismatch here is a startup failure there."""
    manifest = build(tmp_path)
    size = (tmp_path / "scene" / scenes.BACKGROUND_NAME).stat().st_size
    assert size == manifest["canvas_width"] * manifest["canvas_height"] * 4


def test_the_scene_id_is_content_derived(tmp_path):
    one = build(tmp_path / "a", seed=7)
    same = build(tmp_path / "b", seed=7)
    other = build(tmp_path / "c", seed=8)
    assert one["scene_id"] == same["scene_id"]
    assert one["scene_id"] != other["scene_id"]


def test_the_scene_id_moves_with_the_capture_resolution(tmp_path):
    one = build(tmp_path / "a", 320, 200)
    other = build(tmp_path / "b", 400, 240)
    assert one["scene_id"] != other["scene_id"]


def test_the_manifest_records_the_steps_the_source_will_use(tmp_path):
    manifest = build(tmp_path)
    assert set(manifest["workload_steps"]) == set(scenes.WORKLOAD_STEPS)
    for name, step in scenes.WORKLOAD_STEPS.items():
        assert manifest["workload_steps"][name] == list(step)


def test_a_freshly_written_scene_is_not_claimed_to_be_verified(tmp_path):
    """Writing a scene proves nothing about whether a detector sees anything."""
    build(tmp_path)
    assert not (tmp_path / "scene" / scenes.DETECTIONS_NAME).exists()


# ---------------------------------------------------------------------------
# What section7 does with a scene
# ---------------------------------------------------------------------------

def section7_args(tmp_path, **overrides):
    from types import SimpleNamespace
    base = dict(scene=tmp_path / "scene", category="ingestion",
                width=SMALL[0], height=SMALL[1])
    base.update(overrides)
    return SimpleNamespace(**base)


def test_an_inference_run_without_a_scene_is_refused(tmp_path):
    import section7
    with pytest.raises(ValueError, match="needs --scene"):
        section7.load_scene(section7_args(tmp_path, scene=None,
                                          category="inference"), {})


def test_an_ingestion_run_without_a_scene_is_fine(tmp_path):
    """Pixel age does not need objects; the procedural pattern is the right
    workload for it and always has been."""
    import section7
    assert section7.load_scene(section7_args(tmp_path, scene=None), {}) is None


def test_a_scene_built_for_another_resolution_is_refused(tmp_path):
    import section7
    build(tmp_path)
    with pytest.raises(ValueError, match="scene was built for"):
        section7.load_scene(section7_args(tmp_path, width=1920, height=1080), {})


def test_an_unverified_scene_blocks_an_inference_run(tmp_path):
    import section7
    build(tmp_path)
    with pytest.raises(ValueError, match="unverified"):
        section7.load_scene(section7_args(tmp_path, category="inference"), {})


def test_an_unverified_scene_is_recorded_as_unverified_for_other_runs(tmp_path):
    import section7
    build(tmp_path)
    payload = {}
    section7.load_scene(section7_args(tmp_path), payload)
    assert payload["scene"]["verification"]["usable"] is None
    assert "never been put through a detector" in \
        payload["scene"]["verification"]["problems"][0]


def test_a_scene_that_failed_verification_is_refused(tmp_path):
    import section7
    build(tmp_path)
    (tmp_path / "scene" / scenes.DETECTIONS_NAME).write_text(json.dumps(
        {"usable": False, "problems": ["static: as few as 0 detection(s)"],
         "workloads": {}}))
    with pytest.raises(ValueError, match="did not pass verification"):
        section7.load_scene(section7_args(tmp_path), {})


def test_a_verified_scene_carries_its_evidence_into_the_payload(tmp_path):
    import section7
    build(tmp_path)
    (tmp_path / "scene" / scenes.DETECTIONS_NAME).write_text(json.dumps(
        {"usable": True, "problems": [], "model_sha256": "b" * 64,
         "workloads": {"static": {"mean_detections": 8.0},
                       "scroll": {"mean_detections": 7.5},
                       "motion": {"mean_detections": 7.0}}}))
    payload = {}
    section7.load_scene(section7_args(tmp_path), payload)
    verification = payload["scene"]["verification"]
    assert verification["usable"] is True
    assert verification["model_sha256"] == "b" * 64
    assert verification["detections_per_frame"]["static"] == 8.0
