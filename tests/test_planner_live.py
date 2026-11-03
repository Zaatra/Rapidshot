"""The planner's calibration, live: both backends timed against the test source.

The test source covers the primary display for a few seconds while each
backend is measured. Skipped unless WGC can capture here (so there is a choice
to measure) and the test source exists. The headless half is
tests/test_planner.py.
"""
import pytest

import rapidshot
from rapidshot import Monitor

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.planner as planner  # noqa: E402

from test_wgc_live import _UNAVAILABLE  # noqa: E402

pytestmark = [
    pytest.mark.skipif(_UNAVAILABLE is not None, reason=_UNAVAILABLE or ""),
    pytest.mark.skipif(planner._find_source() is None,
                       reason="the test source latency_source.exe is not built or installed"),
]


@pytest.fixture
def private_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("RAPIDSHOT_PLAN_CACHE", str(tmp_path / "plans.json"))


def test_calibration_measures_both_backends_and_auto_follows_it(private_cache):
    result = rapidshot.plan(Monitor(), consumer="cpu", calibrate=True, seconds=1.5)
    print(result)

    assert result.measured, result.reasons
    assert set(result.evidence) == {"dxgi", "wgc"}
    for measurement in result.evidence.values():
        # Plausible, not a benchmark: a frame younger than a millisecond or
        # older than a tenth of a second means the instrument is wrong.
        assert 1 < measurement.age_p50_ms < 100, measurement
        assert measurement.fps > 30, measurement
        assert measurement.frames >= planner.MIN_FRAMES

    assert planner.cached_backend(rapidshot.get_factory(), Monitor()) == result.backend
    camera = rapidshot.create(source=Monitor())
    try:
        assert camera.backend == result.backend
    finally:
        camera.release()
