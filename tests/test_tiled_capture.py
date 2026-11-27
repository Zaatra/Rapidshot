"""The public tiled path over a pitched D3D11 fake and independent BGRA oracle."""

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

from rapidshot.core.stagesurf import StageSurface
from rapidshot.memory_pool import BufferReleasedError, PoolExhaustedError
from test_capture_paths import FakeContext, FakeStageSurface, pipeline  # noqa: F401


@pytest.fixture
def pitched_pipeline(pipeline, monkeypatch):  # noqa: F811
    original_rebuild = FakeStageSurface.rebuild
    original_map = FakeStageSurface.map

    def rebuild(stage, output, device, dim=None):
        original_rebuild(stage, output, device, dim)
        stage.pitch = stage.width * 4 + 32
        stage.storage = np.full((stage.height, stage.pitch), 0xAD, np.uint8)
        stage.buffer = stage.storage[:, : stage.width * 4].reshape(stage.height, stage.width, 4)

    def map_with_padding(stage):
        result = original_map(stage)
        result.Pitch = stage.pitch
        return result

    def copy_with_offsets(context, dst, dst_sub, x, y, z, src, src_sub, box_ref):
        box = box_ref._obj
        left, top, right, bottom = box.left, box.top, box.right, box.bottom
        context.log.append(("copy", (left, top, right, bottom, x, y)))
        assert not dst.mapped and dst_sub == src_sub == z == 0
        assert 0 <= left < right <= context.desktop.image.shape[1]
        assert 0 <= top < bottom <= context.desktop.image.shape[0]
        assert x + right - left <= dst.width and y + bottom - top <= dst.height
        dst.buffer[y : y + bottom - top, x : x + right - left] = context.desktop.image[top:bottom, left:right]

    monkeypatch.setattr(FakeStageSurface, "rebuild", rebuild)
    monkeypatch.setattr(FakeStageSurface, "map", map_with_padding)
    monkeypatch.setattr(FakeStageSurface, "copy_bgra_regions", StageSurface.copy_bgra_regions, raising=False)
    monkeypatch.setattr(FakeStageSurface, "bgra_view", StageSurface.bgra_view, raising=False)
    monkeypatch.setattr(FakeContext, "CopySubresourceRegion", copy_with_offsets)

    def make(**settings):
        options = {"width": 320, "height": 312, "output_color": "BGRA", **settings}
        camera, _, desktop, device = pipeline(**options)
        camera._duplicator.accumulated_frames = 1
        camera._duplicator.dirty_rects = [(0, 0, *desktop.output.surface_size)]
        return camera, desktop, device.im_context.log

    return make


def events(log, name):
    return [value for event, value in log if event == name]


def oracle_tiles(previous, current):
    ys, xs = np.nonzero(np.any(previous != current, axis=2))
    columns = (current.shape[1] + 127) // 128
    return tuple(np.unique(ys // 128 * columns + xs // 128))


def test_explicit_tiled_region_is_validated_after_recovery_changes_size(pitched_pipeline, monkeypatch):
    camera, _, log = pitched_pipeline()
    camera._needs_reinit = True

    def recover_to_smaller_output():
        camera.width = 160
        camera._needs_reinit = False
        return True

    monkeypatch.setattr(camera, "_attempt_reinitialization", recover_to_smaller_output)
    serial = camera._duplicator.frame_serial
    with pytest.raises(ValueError, match="exceeds capture bounds"):
        camera.grab_tiles((10, 10, 290, 180))
    assert camera._duplicator.frame_serial == serial
    assert not events(log, "copy")


def test_cropped_readback_uses_candidate_boxes_and_holds_immutable_page_snapshots(pitched_pipeline):
    camera, desktop, log = pitched_pipeline()
    crop = (17, 11, 283, 277)
    baseline = desktop.image[11:277, 17:283].copy()
    first = camera.grab_tiles(crop, tight_rectangles=True)
    assert first is not None
    with first:
        assert first.changes.reason == "initial"
        assert first.changes.indices == tuple(range(9))
        assert first.changes.tight_rectangles == first.changes.rectangles
        assert events(log, "copy") == [(*crop, 0, 0)]
        np.testing.assert_array_equal(first.materialize(), baseline)

        log.clear()
        desktop.image[140, 147, 3] ^= 255
        desktop.image[146, 150, 0] ^= 255
        desktop.image[276, 282, 3] ^= 255
        camera._duplicator.dirty_rects = [(147, 140, 151, 147), (282, 276, 283, 277)]
        current = desktop.image[11:277, 17:283]
        with camera.grab_tiles(crop, tight_rectangles=True) as second:
            assert second.changes.indices == oracle_tiles(baseline, current) == (4, 8)
            assert second.changes.rectangles == ((128, 128, 256, 256), (256, 256, 266, 266))
            assert second.changes.tight_rectangles == ((130, 129, 134, 136), (265, 265, 266, 266))
            assert events(log, "copy") == [(145, 139, 273, 267, 128, 128), (273, 267, 283, 277, 256, 256)]
            assert len(events(log, "map")) == 1
            assert [name for name, _ in log] == ["copy", "copy", "release_frame", "map"]
            assert not camera._stagesurf.mapped
            np.testing.assert_array_equal(second.tiles[0], current[128:256, 128:256])
            np.testing.assert_array_equal(second.tiles[1], current[256:, 256:])
            with pytest.raises(ValueError):
                second.tiles[0].setflags(write=True)
            roi = second.materialize((123, 125, 262, 262))
            assert roi.flags.c_contiguous and roi.flags.writeable
            np.testing.assert_array_equal(roi, current[125:262, 123:262])
            roi[:] = 0
            np.testing.assert_array_equal(second.materialize(), current)
            np.testing.assert_array_equal(first.materialize(), baseline)
            for invalid in ((-1, 0, 1, 1), (0, 0, 267, 1), (1, 1, 1, 2)):
                with pytest.raises(ValueError):
                    second.materialize(invalid)
        second.release()
        with pytest.raises(BufferReleasedError):
            second.materialize()
        with pytest.raises(BufferReleasedError):
            _ = second.tiles
        assert second.changes.indices == (4, 8)


def test_idle_differs_from_acquired_identical_without_a_copy(pitched_pipeline):
    camera, desktop, log = pitched_pipeline(script=("frame", "timeout", "frame"))
    crop = (0, 0, 200, 180)
    with camera.grab_tiles(crop) as first:
        assert first.changes.tight_rectangles is None
        assert camera.grab_tiles(crop) is None
        assert [name for name, _ in log].count("map") == 1
        log.clear()
        camera._duplicator.dirty_rects = [(250, 260, 310, 309)]
        with camera.grab_tiles(crop, tight_rectangles=True) as same:
            assert same.changes.indices == same.tiles == same.changes.tight_rectangles == ()
            assert same.changes.reason == "identical"
            np.testing.assert_array_equal(same.materialize(), desktop.image[:180, :200])
        assert log == [("release_frame", None)]


@pytest.mark.parametrize(
    ("metadata", "fallback"),
    [({"accumulated_frames": 2}, "accumulated_frames"), ({"move_rects": [(10, 10, 20, 20, 0, 0)]}, "move_rects"),
     ({"dirty_rects": None}, "metadata_unavailable"), ({"dirty_rects": []}, "empty_dirty_rects"),
     ({"dirty_rects": [(0, 0, 321, 1)]}, "invalid_dirty_rects"), ({"frame_serial": 9}, "frame_gap")],
)
def test_uncertain_metadata_copies_same_acquired_texture_in_full(pitched_pipeline, metadata, fallback):
    camera, desktop, log = pitched_pipeline()
    with camera.grab_tiles() as first:
        baseline = first.materialize()
    log.clear()
    desktop.image[-1, -1, 3] ^= 255
    camera._duplicator.dirty_rects = [(0, 0, 1, 1)]
    for key, value in metadata.items():
        setattr(camera._duplicator, key, value)
    with camera.grab_tiles(tight_rectangles=True) as frame:
        assert frame.changes.fallback_reason == fallback
        assert frame.changes.indices == oracle_tiles(baseline, desktop.image) == (8,)
        assert frame.changes.tight_rectangles == ((319, 311, 320, 312),)
        np.testing.assert_array_equal(frame.materialize(), desktop.image)
    assert events(log, "copy") == [(0, 0, 320, 312, 0, 0)]
    assert camera._duplicator.calls == 2


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_rotated_tiled_readback_uses_logical_coordinates(pitched_pipeline, rotation):
    camera, desktop, _ = pitched_pipeline(rotation=rotation)
    crop = (17, 11, 281, 277)
    with camera.grab_tiles(crop) as first:
        before = desktop.as_desktop("BGRA")[11:277, 17:281]
        np.testing.assert_array_equal(first.materialize(), before)
    desktop.image[173, 161, 3] ^= 255
    camera._duplicator.dirty_rects = [(0, 0, 1, 1)]
    with camera.grab_tiles(crop) as second:
        after = desktop.as_desktop("BGRA")[11:277, 17:281]
        assert second.changes.fallback_reason == "rotation"
        assert second.changes.indices == oracle_tiles(before, after)
        np.testing.assert_array_equal(second.materialize(), after)


def test_protected_content_transition_cannot_reuse_old_pixels(pitched_pipeline):
    camera, desktop, log = pitched_pipeline()
    with camera.grab_tiles() as first:
        baseline = first.materialize()
    camera._duplicator.dirty_rects = [(0, 0, 1, 1)]
    for protected in (True, False):
        log.clear()
        camera._duplicator.protected_content_detected = protected
        desktop.image[150:170, 160:180] = 0 if protected else baseline[150:170, 160:180]
        with camera.grab_tiles() as frame:
            assert frame.changes.fallback_reason == "protected_content"
            np.testing.assert_array_equal(frame.materialize(), desktop.image)
        assert events(log, "copy") == [(0, 0, 320, 312, 0, 0)]


def test_audit_repairs_an_omitted_tile_and_disables_selective_copy(pitched_pipeline):
    camera, desktop, log = pitched_pipeline()
    with camera.grab_tiles(audit_every=1) as baseline:
        assert baseline.changes.reason == "initial"
    desktop.image[-1, -1, 3] ^= 255
    camera._duplicator.dirty_rects = [(0, 0, 1, 1)]
    with camera.grab_tiles(audit_every=1) as audited:
        assert audited.changes.fallback_reason == "dirty_rects_audit_miss"
        assert audited.changes.indices == (8,)
        np.testing.assert_array_equal(audited.materialize(), desktop.image)
    stats = camera.change_audit_stats()
    assert (stats.audited_frames, stats.miss_frames, stats.missed_tiles) == (1, 1, 1)

    log.clear()
    desktop.image[127, 130, 0] ^= 255
    with camera.grab_tiles() as after_miss:
        assert after_miss.changes.fallback_reason == "dirty_rects_audit_disabled"
        assert after_miss.changes.indices == (1,)
        np.testing.assert_array_equal(after_miss.materialize(), desktop.image)
    assert events(log, "copy") == [(0, 0, 320, 312, 0, 0)]


def test_tiled_pool_exhaustion_does_not_consume_a_frame(pitched_pipeline):
    camera, desktop, _ = pitched_pipeline(pool_size_frames=1)
    with camera.grab_tiles() as held:
        first = held.materialize()
        desktop.image[-1, -1, 3] ^= 255
        camera._duplicator.dirty_rects = [(319, 311, 320, 312)]
        with pytest.raises(PoolExhaustedError):
            camera.grab_tiles()
        assert camera._duplicator.calls == 1
        np.testing.assert_array_equal(held.materialize(), first)
    with camera.grab_tiles() as recovered:
        held.release()
        assert recovered.changes.indices == (8,)
        assert camera._duplicator.calls == 2
        np.testing.assert_array_equal(recovered.materialize(), desktop.image)


def test_tiled_snapshot_survives_camera_release(pitched_pipeline):
    camera, desktop, _ = pitched_pipeline(pool_size_frames=1)
    with camera.grab_tiles() as held:
        camera.release()
        np.testing.assert_array_equal(held.materialize(), desktop.image)
        with pytest.raises(RuntimeError, match="released"):
            camera.grab_tiles()


def test_failed_map_does_not_leak_slot_or_publish_stale_tiles(pitched_pipeline, monkeypatch):
    camera, desktop, _ = pitched_pipeline(pool_size_frames=1)
    with camera.grab_tiles() as first:
        original = first.materialize()
    desktop.image[-1, -1, 3] ^= 255
    camera._duplicator.dirty_rects = [(319, 311, 320, 312)]
    with monkeypatch.context() as patch:
        patch.setattr(FakeStageSurface, "null_bits", True)
        assert camera.grab_tiles() is None
        assert not camera._duplicator._frame_acquired
        assert not camera._stagesurf.mapped
    with camera.grab_tiles() as recovered:
        assert recovered.changes.reason == "reset"
        assert oracle_tiles(original, recovered.materialize()) == (8,)
        np.testing.assert_array_equal(recovered.materialize(), desktop.image)


def test_full_frame_api_exposes_optional_tight_bounds_without_changing_its_lease(pitched_pipeline):
    camera, desktop, _ = pitched_pipeline()
    with camera.grab_with_changes() as first:
        assert first.changes.tight_rectangles is None
        assert first.pixels.flags.writeable
    desktop.image[129, 127, 3] ^= 255
    camera._duplicator.dirty_rects = [(127, 129, 128, 130)]
    with camera.grab_with_changes(tight_rectangles=True) as second:
        assert second.changes.tight_rectangles == ((127, 129, 128, 130),)
        assert second.changes.rectangles == ((0, 128, 128, 256),)
        np.testing.assert_array_equal(second.pixels, desktop.image)
