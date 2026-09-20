"""benchmarks/recording.py without a GPU, an encoder, or a screen.

The live numbers this harness produces are only as good as what it refuses to
report: a conversion that is quietly wrong, a row whose bytes never left the
GPU being compared against one that copied a frame, or a path that is missing
its dependency being recorded as a failure. Those are the parts checked here.

The NV12 reference is checked against hand-computed BT.601 codes rather than
against the shader, because a reference that shares arithmetic with the thing
it verifies cannot fail.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import recording as bench  # noqa: E402


@pytest.fixture(autouse=True)
def quiet_whea_log(monkeypatch):
    monkeypatch.setattr(bench.HealthGuard, "query",
                        staticmethod(lambda: {"latest": 1, "count": 1}))


# -- the reference the GPU rows are judged against -------------------------


def test_the_nv12_reference_matches_the_standard_by_hand():
    """Mid grey and pure red, worked out from BT.601 limited range.

    Y = 16 + 219*luma, chroma = 128 + 224*(difference). If the reference drifts,
    every GPU row it verifies drifts with it and nothing else would notice.
    """
    grey = np.full((2, 2, 3), 0.5, dtype=np.float32)
    plane = bench.nv12_reference(grey, np)
    assert plane.shape == (3, 2)
    assert np.all(plane[:2] == round(16 + 219 * 0.5))
    assert np.all(plane[2] == 128), "grey carries no colour"

    red = np.zeros((2, 2, 3), dtype=np.float32)
    red[..., 0] = 1.0
    plane = bench.nv12_reference(red, np)
    assert np.all(plane[:2] == round(16 + 219 * 0.299))
    cb, cr = plane[2, 0], plane[2, 1]
    assert cb == round(128 + 224 * (0.0 - 0.299) / (2 * (1 - 0.114)))
    assert cr == round(128 + 224 * (1.0 - 0.299) / (2 * (1 - 0.299)))


def test_the_reference_averages_chroma_over_each_block():
    """4:2:0 carries one chroma pair per 2x2 block. Sampling one pixel of the
    block instead would pass a flat image and fail only on real content."""
    rgb = np.zeros((2, 2, 3), dtype=np.float32)
    rgb[0, 0, 0] = 1.0                       # one red pixel in the block
    plane = bench.nv12_reference(rgb, np)
    quarter = bench.nv12_reference(np.full((2, 2, 3), 0.25, dtype=np.float32), np)
    assert plane[2, 1] != quarter[2, 1], "the red quarter must reach chroma"
    assert plane.shape == (3, 2)


@pytest.mark.parametrize("deviation,off_by,verified", [
    (0, 0, True),                            # identical
    (1, 5, True),                            # float32 rounding at a few ties
    (1, 400, False),                         # systematic off-by-one
    (2, 1, False),                           # one sample too far, however rare
])
def test_the_verdict_needs_both_a_bound_and_an_exact_fraction(deviation, off_by, verified):
    """Neither condition alone is enough. A bound of one code passes a whole
    image shifted by one; an exact-match fraction passes an image with a few
    wildly wrong samples."""
    got = np.zeros(1000, dtype=np.uint8)
    expected = np.zeros(1000, dtype=np.float64)
    expected[:off_by] = deviation
    comparison = bench.compare_codes(got, expected, np)
    decision = (comparison["max_code_deviation"] <= 1
                and comparison["fraction_exact"] >= 0.99)
    assert decision is verified


# -- what the parent accepts from a worker ---------------------------------


def measurement(**overrides):
    row = {"path": "gpu-nv12-resident", "frames": 100, "fps": 120.0,
           "elapsed_seconds": 0.83, "ms_p50": 1.0, "bytes_to_cpu_per_frame": 0}
    row.update(overrides)
    return row


def test_a_measurement_row_is_accepted():
    row = bench.parse_result("gpu-nv12-resident", False, 0,
                             json.dumps(measurement()), "")
    assert "error" not in row


@pytest.mark.parametrize("bad", [{"frames": 0}, {"fps": 0}, {"elapsed_seconds": 0},
                                 {"fps": "fast"}, {"frames": True}])
def test_a_row_without_real_measurements_is_an_error(bad):
    row = bench.parse_result("gpu-nv12-resident", False, 0,
                             json.dumps(measurement(**bad)), "")
    assert row["error"] == "worker result is missing valid measurements"


def test_an_unverified_conversion_is_an_error_not_a_quiet_pass():
    """The whole point of --verify: a path that converts wrongly must not be
    allowed to report a fast number."""
    row = bench.parse_result("gpu-nv12-readback", True, 0, json.dumps(
        {"path": "gpu-nv12-readback", "verified": False,
         "max_code_deviation": 9.0}), "")
    assert row["error"] == "worker did not report a verified conversion"


def test_a_skipped_path_is_not_an_error():
    """OpenCV missing is a machine without a dependency, not a broken path."""
    row = bench.parse_result("cpu-cv2-i420", True, 0, json.dumps(
        {"path": "cpu-cv2-i420", "skipped": "OpenCV is not installed"}), "")
    assert "error" not in row and row["skipped"]


def test_a_worker_that_died_is_reported_with_its_status():
    row = bench.parse_result("cpu-cv2-i420", False, 1, "", "boom")
    assert "invalid worker result" in row["error"]
    assert row["returncode"] == 1


# -- the adapters, over a fake camera --------------------------------------


class FakeFrame:
    def __init__(self, width=8, height=4):
        self.width, self.height = width, height
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.released = True
        return False

    def release(self):
        self.released = True


def fake_rapidshot(monkeypatch, *, frame=None, image=None, converter=None):
    frames = [frame if frame is not None else FakeFrame()]
    camera = SimpleNamespace(
        grab_frame=lambda: frames[0],
        grab=lambda: image,
        release=lambda: None)
    module = SimpleNamespace(
        create=lambda **kw: camera,
        GpuConverter=converter or (lambda *a, **kw: None),
        native=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "rapidshot", module)
    monkeypatch.setitem(sys.modules, "rapidshot.native", module.native)
    return camera


def test_the_capture_floor_releases_every_frame(monkeypatch):
    """The floor exists to be subtracted from the conversion rows, so it must
    do the capture and nothing else -- and leak nothing while doing it."""
    frame = FakeFrame()
    fake_rapidshot(monkeypatch, frame=frame)
    produce, teardown, meta = bench.ADAPTERS["capture-bgra-only"]()
    assert produce() == 0
    assert frame.released
    assert meta["bytes_to_cpu_per_frame"] == 0
    teardown()


def test_an_odd_sized_display_is_refused_rather_than_mangled(monkeypatch):
    """4:2:0 has no half chroma sample. Converting anyway would silently drop
    a row or column."""
    fake_rapidshot(monkeypatch, frame=FakeFrame(width=7, height=5))
    produce, teardown, _ = bench.ADAPTERS["gpu-nv12-resident"]()
    with pytest.raises(bench.PathUnavailable, match="even dimensions"):
        produce()
    teardown()


def test_the_resident_row_moves_nothing_to_the_cpu(monkeypatch):
    """A readback here would make the GPU-encoder row measure a copy the
    encoder never asks for, which is the mistake this benchmark exists to
    avoid making."""
    readbacks = []

    class Converter:
        def __init__(self, frame, size, **kwargs):
            self.size = size

        def process(self, frame):
            return SimpleNamespace(numpy=lambda: readbacks.append(1) or np.zeros(6, np.uint8))

    fake_rapidshot(monkeypatch, converter=Converter)
    produce, teardown, meta = bench.ADAPTERS["gpu-nv12-resident"]()
    assert produce() == 0
    assert readbacks == [], "the resident path must not read back"
    assert "VRAM" in meta["note"]

    produce, teardown2, meta = bench.ADAPTERS["gpu-nv12-readback"]()
    assert produce() == 6
    assert readbacks == [1], "the CPU-encoder path pays for exactly one copy"
    assert "system memory" in meta["note"]
    teardown()
    teardown2()


def test_missing_opencv_is_a_skip_not_a_failure(monkeypatch):
    fake_rapidshot(monkeypatch)
    monkeypatch.setitem(sys.modules, "cv2", None)
    monkeypatch.setattr(bench, "ADAPTERS", dict(bench.ADAPTERS))
    with pytest.raises(bench.PathUnavailable, match="OpenCV"):
        bench._cpu_cv2_i420()


def test_both_sides_are_told_to_use_the_same_matrix():
    """cv2.cvtColor implements BT.601 and offers no choice. A GPU converter
    left on its BT.709 default would make the two rows different pictures,
    and the timing comparison would be between standards."""
    assert bench.MATRIX == "bt601"


def test_the_paths_measured_include_a_floor_for_each_route():
    """Conversion cost is a difference from a floor. Without both floors the
    rows would silently include capture."""
    assert "capture-bgra-only" in bench.PATHS      # GPU route floor
    assert "capture-bgr-cpu" in bench.PATHS        # CPU route floor
