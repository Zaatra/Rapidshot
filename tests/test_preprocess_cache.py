"""The `process()` texture cache, including the miss path.

`GpuPreprocessor12.process()` caches the captured texture it opened on the D3D12
device, keyed on the raw pointer. A 12-minute soak saw **exactly one** texture
across 211,726 frames, so ordinary use never misses -- which is what makes this
worth testing deliberately. Silently reusing a stale resource would build a
tensor from a surface the caller is no longer capturing, and the output would
look entirely reasonable while being wrong.

Tearing a camera down and rebuilding it produces a genuinely new
`IDXGIOutputDuplication` and therefore a new texture. That is also the realistic
trigger: a device reset or display-mode change ends up in the same place.

These need live capture and so skip without screen activity (ROADMAP section 2).
"""
import numpy as np
import pytest

import rapidshot
from rapidshot import native

if not native.is_available():
    pytest.skip("native extension not built", allow_module_level=True)

OUT = 64


def first_frame(camera, tries=600):
    for _ in range(tries):
        frame = camera.grab_frame()
        if frame is not None:
            return frame
    return None


@pytest.fixture
def rebuilt_cameras():
    """Yield (frame_a, preprocessor, frame_b) spanning a full camera rebuild.

    `create()` returns a cached instance per output, so asking twice gives the
    same camera and the same texture. Only a teardown produces a new one.
    """
    camera_a = rapidshot.create(output_color="BGRA")
    frame_a = first_frame(camera_a)
    if frame_a is None:
        camera_a.release()
        pytest.skip("no frame captured — the screen must be changing")

    pre = native.GpuPreprocessor12(frame_a, OUT, OUT)
    pre.process(frame_a)
    address_a = native._texture_address(frame_a)

    frame_a.release()
    camera_a.release()
    rapidshot.reset()

    camera_b = rapidshot.create(output_color="BGRA")
    frame_b = first_frame(camera_b)
    if frame_b is None:
        camera_b.release()
        pytest.skip("no frame from the rebuilt camera")

    address_b = native._texture_address(frame_b)
    # Deliberately NOT skipped when the addresses match. An earlier version of
    # this fixture did, which meant the one case the cache key has to survive --
    # Windows recycling a released COM address for an unrelated texture -- was
    # the case the test stepped around. The key is (address, source_id) now, so
    # a recycled address still misses because the duplicator differs.
    yield pre, address_a, frame_b, address_b

    frame_b.release()
    camera_b.release()
    rapidshot.reset()


def test_cache_holds_the_texture_it_opened(rebuilt_cameras):
    pre, address_a, _, _ = rebuilt_cameras
    # Still keyed to camera A's texture, before anything else is processed.
    assert pre._impl.cached_texture_address == address_a


def test_cache_rekeys_on_a_different_texture(rebuilt_cameras):
    """The miss path, with a real second DXGI surface rather than a mock.

    Asserted on the *key*, not on the pixels. A stale resource would keep
    producing plausible output — the same reason the CUDA lifetime test checks
    reachability instead of comparing arrays.
    """
    pre, _, frame_b, address_b = rebuilt_cameras
    pre.process(frame_b)
    assert pre._impl.cached_texture_address == address_b


def test_frames_from_different_duplicators_have_different_source_ids():
    """What makes the cache key sound when an address is recycled.

    The address alone cannot distinguish a live surface from a released one
    whose pointer has been handed to something else, so the key pairs it with
    the duplicator that produced the frame.
    """
    camera_a = rapidshot.create(output_color="BGRA")
    frame_a = first_frame(camera_a)
    if frame_a is None:
        camera_a.release()
        pytest.skip("no frame captured — the screen must be changing")
    id_a = frame_a.source_id
    frame_a.release()
    camera_a.release()
    rapidshot.reset()

    camera_b = rapidshot.create(output_color="BGRA")
    frame_b = first_frame(camera_b)
    if frame_b is None:
        camera_b.release()
        pytest.skip("no frame from the rebuilt camera")
    try:
        assert frame_b.source_id != id_a, (
            "a rebuilt duplicator reused a source id, so a recycled texture "
            "address would produce a false cache hit")
    finally:
        frame_b.release()
        camera_b.release()
        rapidshot.reset()


def test_tensor_is_valid_after_a_miss(rebuilt_cameras):
    pre, _, frame_b, _ = rebuilt_cameras
    pre.process(frame_b)
    tensor = pre.read_back()
    assert tensor.shape == (1, 3, OUT, OUT)
    assert np.isfinite(tensor).all()
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_repeat_call_does_not_rekey(rebuilt_cameras):
    """It has to remain a cache, not merely be correct.

    Reopening every call cost ~98 us of fixed work and made every pixel 2.5x
    more expensive (ROADMAP section 10), so a change that quietly disabled the
    cache would be a large regression with no visible symptom.
    """
    pre, _, frame_b, address_b = rebuilt_cameras
    pre.process(frame_b)
    for _ in range(5):
        pre.process(frame_b)
        assert pre._impl.cached_texture_address == address_b
