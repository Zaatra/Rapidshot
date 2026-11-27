"""Byte-exact CPU comparison and its one-shot capture lifetime contract."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import numpy as np
import pytest

from rapidshot import ChangedFrame, ChangeSet
from rapidshot.changes import _ChangeHistory
from rapidshot.memory_pool import BufferReleasedError, NumpyMemoryPool, PoolExhaustedError


SHAPE = (131, 129, 4)
REGION = (0, 0, SHAPE[1], SHAPE[0])


def _update(history, pixels, **updates):
    height, width, _ = pixels.shape
    region = (0, 0, width, height)
    metadata = dict(source=1, serial=1, region=region, output_size=(width, height),
                    dirty_rects=[region], move_rects=[], accumulated_frames=1)
    return history.update(pixels, **(metadata | updates))


def _oracle(previous, current):
    """Independent pixel scan groups any unequal BGRA position into its tile."""
    columns = (current.shape[1] + 127) // 128
    y, x = np.where(np.any(previous != current, axis=2))
    return tuple(int(n) for n in np.unique((y // 128) * columns + x // 128))


def test_initial_frame_and_later_alpha_and_boundary_changes_match_pixel_oracle():
    history = _ChangeHistory()
    original = np.zeros(SHAPE, dtype=np.uint8)
    first = _update(history, original, present_time_qpc=19)
    assert isinstance(first, ChangeSet)
    assert (first.reason, first.indices, first.size) == ("initial", (0, 1, 2, 3), (129, 131))
    assert first.rectangles[-1] == (128, 128, 129, 131)
    assert first.present_time_qpc == 19
    assert history.last_update_bytes == original.nbytes == history.size_bytes
    with pytest.raises(FrozenInstanceError):
        first.reason = "other"
    modified = original.copy()
    modified[127, 127, 0] = 1
    modified[130, 128, 3] = 7
    second = _update(history, modified, serial=2, dirty_rects=[(127, 127, 129, 131)])
    assert second.indices == _oracle(original, modified) == (0, 3)
    assert second.rectangles == ((0, 0, 128, 128), (128, 128, 129, 131))
    assert second.fallback_reason is None
    assert history.last_update_bytes == (128 * 128 + 3) * 4
    assert history.bytes_written == original.nbytes + history.last_update_bytes


def test_history_is_private_and_equal_presented_pixels_have_empty_tiles():
    frame = np.zeros(SHAPE, dtype=np.uint8)
    history = _ChangeHistory()
    _update(history, frame)
    frame[:] = 255
    unchanged = _update(history, np.zeros_like(frame), serial=2)
    assert (unchanged.indices, unchanged.rectangles, unchanged.reason) == ((), (), "identical")
    assert history.last_update_bytes == 0
    assert history.bytes_written == frame.nbytes


@pytest.mark.parametrize("change,fallback", [
    ({"serial": 3}, "frame_gap"),
    ({"accumulated_frames": 2}, "accumulated_frames"),
    ({"move_rects": [(7, 9, 130, 128, 131, 129)]}, "move_rects"),
    ({"move_rects": None}, "metadata_unavailable"),
    ({"dirty_rects": None}, "metadata_unavailable"),
    ({"dirty_rects": []}, "empty_dirty_rects"),
    ({"dirty_rects": [(-1, 0, 5, 5)]}, "invalid_dirty_rects"),
    ({"dirty_rects": [7]}, "invalid_dirty_rects"),
    ({"dirty_rects": [(1, 2, 3)]}, "invalid_dirty_rects"),
])
def test_uncertain_metadata_uses_a_full_exact_comparison(change, fallback):
    history = _ChangeHistory()
    before = np.zeros(SHAPE, dtype=np.uint8)
    after = before.copy()
    after[130, 128, 3] = 1
    _update(history, before)
    options = dict(serial=2, dirty_rects=[(0, 0, 2, 2)]) | change
    result = _update(history, after, **options)
    assert result.fallback_reason == fallback
    assert result.indices == _oracle(before, after)


def test_rotation_metadata_is_bypassed_and_rotation_changes_reset_the_baseline():
    history = _ChangeHistory()
    before = np.zeros(SHAPE, dtype=np.uint8)
    after = before.copy()
    after[130, 128, 1] = 3
    _update(history, before)
    rotated = _update(history, before, serial=2, rotation=90)
    assert rotated.reason == "reset"
    result = _update(history, after, serial=3, rotation=90, dirty_rects=[(1, 1, 2, 2)])
    assert result.fallback_reason == "rotation"
    assert result.indices == _oracle(before, after)


def test_output_coordinates_clip_before_selecting_crop_tiles():
    history = _ChangeHistory()
    before = np.zeros(SHAPE, dtype=np.uint8)
    after = before.copy()
    after[-1, -1, 3] = 1
    region = (20, 30, 149, 161)
    display = (220, 240)
    _update(history, before, region=region, output_size=display)
    result = _update(history, after, serial=2, region=region, output_size=display,
                     dirty_rects=[(148, 160, 170, 190), (0, 0, 10, 10)])
    assert result.fallback_reason is None
    assert result.indices == _oracle(before, after)
    outside = _update(history, after, serial=3, region=region, output_size=display,
                      dirty_rects=[(0, 0, 10, 10)])
    assert outside.reason == "identical"
    shifted = _update(history, after, serial=4, region=(30, 30, 159, 161), output_size=display)
    assert shifted.reason == "reset"
    assert shifted.indices == (0, 1, 2, 3)


def test_new_source_and_geometry_reset_even_when_pixels_are_unchanged():
    history = _ChangeHistory()
    frame = np.zeros(SHAPE, dtype=np.uint8)
    _update(history, frame)
    replaced = _update(history, frame, source=2, serial=2)
    assert replaced.reason == "reset"
    resized = _update(history, frame[:12, :12], source=2, serial=3)
    assert resized.reason == "reset"
    assert resized.rectangles == ((0, 0, 12, 12),)
    assert history.size_bytes == 12 * 12 * 4


def test_array_lease_copies_and_invalidates_without_mutating_saved_changes():
    pixels = np.zeros(SHAPE, dtype=np.uint8)
    changes = _update(_ChangeHistory(), pixels)
    with ChangedFrame(pixels, changes) as result:
        assert result.pixels is pixels
        owned = result.pixels.copy()
    assert result.released and result.changes is changes
    assert owned.shape == SHAPE
    with pytest.raises(BufferReleasedError):
        _ = result.pixels
    result.release()


def _camera(frames, slots=2):
    pytest.importorskip("comtypes")
    from rapidshot.capture import ScreenCapture

    camera = ScreenCapture.__new__(ScreenCapture)
    camera.nvidia_gpu, camera.output_color, camera.is_capturing = False, "BGRA", False
    camera.width, camera.height, camera.region, camera.rotation_angle = 129, 131, REGION, 0
    camera._generation, camera._change_history, camera._live_frame = 0, None, None
    camera.memory_pool = NumpyMemoryPool(SHAPE, np.uint8, slots)
    camera._stagesurf = SimpleNamespace(dxgi_format=87, release=lambda: None)
    camera._duplicator = SimpleNamespace(instance_id=4, frame_serial=0, accumulated_frames=1,
                                         last_present_time=29, dirty_rects=[REGION], move_rects=[],
                                         release=lambda: None)
    frames = iter(frames)

    def grab(region):
        try:
            owner = camera.memory_pool.checkout()
        except PoolExhaustedError:
            return None
        try:
            frame = next(frames)
        except StopIteration:
            owner.release()
            return None
        np.copyto(owner.array, frame)
        camera._duplicator.frame_serial += 1
        return owner

    camera._grab_locked = grab
    return camera


def test_one_shot_copies_history_before_exposing_writable_leases():
    blank = np.zeros(SHAPE, dtype=np.uint8)
    camera = _camera([blank, blank])
    first = camera.grab_with_changes()
    assert first.changes.present_time_qpc == 29
    first.pixels[:] = 31
    with camera.grab_with_changes() as second:
        assert second.changes.reason == "identical"
        assert first.pixels[0, 0, 0] == 31
        assert second.pixels[0, 0, 0] == 0
    first.release()
    assert camera.grab_with_changes() is None
    assert camera.memory_pool.get_stats()["available"] == 2
    assert camera._change_history.size_bytes == blank.nbytes
    camera.release()
    assert camera._change_history is None


def test_pool_exhaustion_and_duplicate_release_cannot_recycle_a_new_lease():
    blank = np.zeros(SHAPE, dtype=np.uint8)
    camera = _camera([blank, blank], slots=1)
    first = camera.grab_with_changes()
    assert camera.grab_with_changes() is None
    first.release()
    with camera.grab_with_changes() as second:
        assert second.changes.reason == "identical"
        first.release()
        assert camera.memory_pool.get_stats()["in_use"] == 1
        assert second.pixels[0, 0, 0] == 0
    assert camera.memory_pool.get_stats()["in_use"] == 0


@pytest.mark.parametrize("updates,problem", [
    ({"_backend_name": "wgc"}, "DXGI"),
    ({"nvidia_gpu": True}, "CPU"),
    ({"output_color": "RGB"}, "BGRA8"),
    ({"is_capturing": True}, "one-shot"),
])
def test_unsupported_camera_settings_refuse_before_grabbing(updates, problem):
    camera = _camera([])
    for name, value in updates.items():
        setattr(camera, name, value)
    camera._grab_locked = lambda region: pytest.fail("unexpected capture")
    try:
        with pytest.raises(ValueError, match=problem):
            camera.grab_with_changes()
    finally:
        camera.is_capturing = False


def test_hdr_surface_is_rejected_and_input_pool_lease_returned():
    blank = np.zeros(SHAPE, dtype=np.uint8)
    camera = _camera([blank], slots=1)
    camera._stagesurf.dxgi_format = 10
    with pytest.raises(ValueError, match="SDR"):
        camera.grab_with_changes()
    assert camera._change_history is None
    assert camera.memory_pool.get_stats()["available"] == 1


def test_history_failure_invalidates_baseline_and_returns_the_lease(monkeypatch):
    blank = np.zeros(SHAPE, dtype=np.uint8)
    camera = _camera([blank, blank, blank], slots=1)
    with camera.grab_with_changes() as seed:
        assert seed.changes.reason == "initial"
    history = camera._change_history
    with monkeypatch.context() as patch:
        def fail(*args, **kwargs):
            raise MemoryError("interrupted history upkeep")

        patch.setattr(history, "update", fail)
        with pytest.raises(MemoryError, match="upkeep"):
            camera.grab_with_changes()
    assert history.size_bytes == 0
    assert camera.memory_pool.get_stats()["available"] == 1
    with camera.grab_with_changes() as recovered:
        assert recovered.changes.reason == "reset"


def test_released_camera_refuses_capture():
    camera = _camera([])
    camera.release()
    with pytest.raises(RuntimeError, match="released"):
        camera.grab_with_changes()
