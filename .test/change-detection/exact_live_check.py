"""Verify every delivered public exact-tiles result against full BGRA pixels."""

import argparse
from collections import Counter, defaultdict
import ctypes
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

from downstream_benchmark import dpi_aware, write_json
from downstream_scene import STAMP_ORIGIN, ScenePlayer, _read_stamp, capture_cover_state, scenes
from monitor_target import select_monitor
from rapidshot.changes import _ChangeHistory
from rapidshot.memory_pool import PoolExhaustedError, PooledBuffer


TILE = 128
ROOT = Path(__file__).resolve().parent


def changed_indices(previous, current):
    """Scan all channels and derive tile IDs from changed pixel coordinates."""
    rows, columns = np.where(np.any(previous != current, axis=2))
    stride = (current.shape[1] + TILE - 1) // TILE
    return tuple(map(int, np.unique(rows // TILE * stride + columns // TILE)))


def changed_rectangles(indices, size):
    width, height = size
    stride = (width + TILE - 1) // TILE
    result = []
    for index in indices:
        row, column = divmod(index, stride)
        x, y = column * TILE, row * TILE
        result.append((x, y, min(x + TILE, width), min(y + TILE, height)))
    return tuple(result)


def offline_alpha():
    original = np.zeros((129, 130, 4), dtype=np.uint8)
    pixels = original.copy()
    identity = dict(source=1, region=(0, 0, 130, 129), output_size=(130, 129), move_rects=[],
                    accumulated_frames=1)
    history = _ChangeHistory()
    initial = history.update(pixels, serial=1, dirty_rects=[(0, 0, 130, 129)], **identity)
    assert initial.reason == "initial"
    pixels[:, :, 3] = 0xFF  # Mutation of the prior delivered array cannot affect private history.
    current = original.copy()
    current[-1, -1, 3] = 1
    change = history.update(current, serial=2, dirty_rects=[(129, 128, 130, 129)], **identity)
    assert change.indices == changed_indices(original, current) == (3,)
    assert change.rectangles == ((128, 128, 130, 129),)
    assert change.fallback_reason is None
    assert history.last_update_bytes == 8
    return {"alpha_only_indices": change.indices, "edge_rectangles": change.rectangles,
            "history_update_bytes": history.last_update_bytes, "prior_array_mutation_safe": True}


class Verifier:
    def __init__(self, camera):
        self.camera = camera
        self.previous = self.identity = None
        self.counts = defaultdict(Counter)
        self.frames = []

    @staticmethod
    def stamp(pixels, region):
        return _read_stamp(pixels, (STAMP_ORIGIN[0] - region[0], STAMP_ORIGIN[1] - region[1]))

    def sample(self, workload, step, region=None, timeout=0.2, single=False):
        counts = self.counts[workload]
        deadline = time.perf_counter() + timeout
        effective = self.camera.region if region is None else region
        while True:
            frame = self.camera.grab_with_changes(region)
            counts["capture_attempts"] += 1
            if frame is not None:
                try:
                    self.check(workload, step, frame, effective)
                except BaseException:
                    frame.release()
                    raise
                if self.stamp(frame.pixels, effective) == step.stamp:
                    counts["marker_matched"] += 1
                    return frame
                counts["other_marker"] += 1
                frame.release()
            else:
                counts["no_image"] += 1
            if single or time.perf_counter() >= deadline:
                return None
            time.sleep(0.001)

    def check(self, workload, step, frame, region):
        pixels = frame.pixels
        value = frame.changes
        width, height = region[2] - region[0], region[3] - region[1]
        assert pixels.shape == (height, width, 4) and pixels.dtype == np.uint8
        assert value.size == (width, height)
        producer = self.camera._duplicator
        assert value.present_time_qpc == (int(producer.last_present_time) or None)
        assert value.accumulated_frames == int(producer.accumulated_frames)
        identity = (producer.instance_id, self.camera.generation, self.camera.rotation_angle, region)
        if identity != self.identity:
            expected = tuple(range(((width + TILE - 1) // TILE) * ((height + TILE - 1) // TILE)))
            reason = "initial" if self.identity is None else "reset"
        else:
            expected = changed_indices(self.previous, pixels)
            reason = "changed" if expected else "identical"
        row = {
            "workload": workload,
            "requested_stamp": step.stamp,
            "observed_stamp": self.stamp(pixels, region),
            "frame_serial": int(producer.frame_serial),
            "present_time_qpc": value.present_time_qpc,
            "accumulated_frames": value.accumulated_frames,
            "region": region,
            "reason": value.reason,
            "fallback": value.fallback_reason,
            "changed_tiles": len(value.indices),
            "expected_tiles": len(expected),
            "history_update_bytes": self.camera._change_history.last_update_bytes,
        }
        if step.covers:
            row["cover_state"] = capture_cover_state(step, pixels)
        if value.indices != expected or value.reason != reason or value.rectangles != changed_rectangles(expected, value.size):
            row.update({"expected_reason": reason, "actual_indices": value.indices, "expected_indices": expected})
            self.frames.append(row)
            raise AssertionError(f"Public exact tiles differ from the full BGRA oracle: {row}")
        self.frames.append(row)
        counts = self.counts[workload]
        counts["oracle_matches"] += 1
        counts[reason] += 1
        counts[f"fallback_{value.fallback_reason or 'none'}"] += 1
        self.previous, self.identity = pixels.copy(), identity


def _require_frame(verifier, player, workload, step, region=None, restart=False):
    player.present(step, restart=restart)
    result = verifier.sample(workload, step, region)
    if result is None:
        raise RuntimeError(f"{workload} did not deliver requested marker {step.stamp:#x}")
    return result


def _verify_gap(verifier, player, steps):
    with _require_frame(verifier, player, "missed_frame", steps[0], restart=True):
        seed_serial = int(verifier.camera._duplicator.frame_serial)
    player.present(steps[1])
    deadline = time.perf_counter() + 0.2
    while time.perf_counter() < deadline:
        plain = verifier.camera.grab()
        if plain is None:
            continue
        try:
            array = plain.array if isinstance(plain, PooledBuffer) else plain
            if verifier.stamp(array, verifier.camera.region) == steps[1].stamp:
                break
        finally:
            if isinstance(plain, PooledBuffer):
                plain.release()
    else:
        raise RuntimeError("Could not verify the intermediate ordinary grab")
    with _require_frame(verifier, player, "missed_frame", steps[2]) as result:
        assert verifier.camera._duplicator.frame_serial >= seed_serial + 2
        # Verify the first exact delivery after the gap, even if stale or an
        # unrelated compositor update landed before the requested marker.
        frames = [row for row in verifier.frames if row["workload"] == "missed_frame"]
        after_gap = next(row for row in frames if row["frame_serial"] >= seed_serial + 2)
        assert after_gap["fallback"] == "frame_gap", after_gap
        assert result.changes.reason == "changed"


def verify_live(target, steps, record):
    import rapidshot

    left, top, right, bottom = target["bounds"]
    width, height = right - left, bottom - top
    fixture = scenes(width, height, visible_steps=steps, passive_steps=steps)
    with ScenePlayer(target["bounds"]) as player:
        player.prepare(fixture)
        camera = rapidshot.create(device_idx=target["device_index"], output_idx=target["output_index"],
                                  backend="dxgi", output_color="BGRA", timeout_ms=3,
                                  nvidia_gpu=False, pool_output=True, pool_size_frames=2)
        verifier = Verifier(camera)
        record["frames"] = verifier.frames
        record["counts"] = verifier.counts
        try:
            assert camera.region == (0, 0, width, height) and camera.rotation_angle == 0
            for workload, updates in fixture.items():
                with _require_frame(verifier, player, workload, updates[0], restart=True):
                    pass
                for update in updates[1:]:
                    if update.visible_update:
                        with _require_frame(verifier, player, workload, update):
                            pass
                    else:
                        player.present(update)
                        result = verifier.sample(workload, update, single=True)
                        if result is not None:
                            result.release()

            roi = (256, 12, 1280, 900)
            shifted = (272, 20, 1296, 908)
            sparse = fixture["sparse"]
            first_crop = len(verifier.frames)
            with _require_frame(verifier, player, "retained_crop", sparse[0], roi, restart=True) as held:
                assert verifier.frames[first_crop]["reason"] == "reset"
                held.pixels[-1, -1, 3] ^= 0xFF
                mutated = int(held.pixels[-1, -1, 3])
                with _require_frame(verifier, player, "retained_crop", sparse[1], roi) as second:
                    assert not np.shares_memory(held.pixels, second.pixels)
                    assert int(held.pixels[-1, -1, 3]) == mutated
            first_shift = len(verifier.frames)
            with _require_frame(verifier, player, "retained_crop", sparse[2], shifted):
                assert verifier.frames[first_shift]["reason"] == "reset"
            with _require_frame(verifier, player, "retained_pool", sparse[0], restart=True) as held:
                held.pixels[-1, -1, 3] ^= 0xFF
                mutated = int(held.pixels[-1, -1, 3])
                assert camera.memory_pool.get_stats()["in_use"] == 1
                with _require_frame(verifier, player, "retained_pool", sparse[1]) as next_frame:
                    assert camera.memory_pool.get_stats()["in_use"] == 2
                    assert not np.shares_memory(held.pixels, next_frame.pixels)
                    assert int(held.pixels[-1, -1, 3]) == mutated
            assert camera.memory_pool.get_stats()["in_use"] == 0
            _verify_gap(verifier, player, sparse[-3:])
            _verify_live_pool_pressure(camera, player, verifier, sparse[:3], record)
        finally:
            camera.release()


def _verify_live_pool_pressure(camera, player, verifier, steps, record):
    with _require_frame(verifier, player, "live_pool_exhaustion", steps[0], restart=True) as first:
        with _require_frame(verifier, player, "live_pool_exhaustion", steps[1]) as second:
            first_pixels, second_pixels = first.pixels.copy(), second.pixels.copy()
            player.present(steps[2])
            serial = camera._duplicator.frame_serial
            assert camera.memory_pool.get_stats()["available"] == 0
            assert camera.grab_with_changes() is None
            assert camera._duplicator.frame_serial == serial
            np.testing.assert_array_equal(first.pixels, first_pixels)
            np.testing.assert_array_equal(second.pixels, second_pixels)
        recovered = verifier.sample("live_pool_exhaustion", steps[2], timeout=0.5)
        if recovered is None:
            raise RuntimeError("Public full-pixel grab did not resume after pool exhaustion")
        with recovered:
            assert camera._duplicator.frame_serial > serial
            np.testing.assert_array_equal(first.pixels, first_pixels)
    record["full_frame_pool_exhaustion"] = {"slots": 2, "acquired_while_full": False, "recovered": True}


class FullDXGIOracle:
    """Read the same acquired texture through an independent full copy.

    This adds an extra map per acquisition and is used only for correctness.
    Neither the candidate list nor the paged history affects this readback.
    """

    def __init__(self, camera):
        from rapidshot.core.stagesurf import StageSurface

        self.camera = camera
        self.duplicator = camera._duplicator
        self.original_update = self.duplicator.update_frame
        self.stage = StageSurface(output=camera._output, device=camera._device)
        self.serial = self.pixels = None

    def __enter__(self):
        self.duplicator.update_frame = self.update
        return self

    def __exit__(self, *_):
        self.duplicator.update_frame = self.original_update
        self.stage.release()

    def update(self):
        from rapidshot._libs.d3d11 import D3D11_BOX

        result = self.original_update()
        if not self.duplicator.updated:
            return result
        try:
            camera = self.camera
            width, height = camera.width, camera.height
            source = self.duplicator.texture
            source_format = self.stage.format_of(source)
            assert camera.rotation_angle == 0 and source_format == 87
            self.stage.ensure(camera._output, camera._device, (width, height), source_format)
            bounds = D3D11_BOX(left=0, top=0, front=0, right=width, bottom=height, back=1)
            camera._device.im_context.CopySubresourceRegion(
                self.stage.texture, 0, 0, 0, 0, source, 0, ctypes.byref(bounds)
            )
            mapped = self.stage.map()
            try:
                assert mapped.pBits and mapped.Pitch >= width * 4
                pitched = np.ctypeslib.as_array(
                    ctypes.cast(mapped.pBits, ctypes.POINTER(ctypes.c_uint8)), shape=(height, mapped.Pitch)
                )
                self.pixels = pitched[:, : width * 4].reshape(height, width, 4).copy()
                self.serial = self.duplicator.frame_serial
            finally:
                self.stage.unmap()
        except BaseException:
            self.duplicator.release_frame()
            raise
        return result


class TiledVerifier(Verifier):
    def __init__(self, camera, oracle):
        super().__init__(camera)
        self.oracle = oracle

    def sample(self, workload, step, region=None, timeout=0.5, single=False):
        counts = self.counts[workload]
        deadline = time.perf_counter() + timeout
        effective = self.camera.region if region is None else region
        while True:
            frame = self.camera.grab_tiles(region, tight_rectangles=True, audit_every=3)
            counts["capture_attempts"] += 1
            if frame is not None:
                try:
                    stamp = self.check(workload, step, frame, effective)
                except BaseException:
                    frame.release()
                    raise
                if stamp == step.stamp:
                    counts["marker_matched"] += 1
                    return frame
                counts["other_marker"] += 1
                frame.release()
            else:
                counts["no_image"] += 1
            if single or time.perf_counter() >= deadline:
                return None

    def check(self, workload, step, frame, region):
        producer = self.camera._duplicator
        assert producer is self.oracle.duplicator and producer.frame_serial == self.oracle.serial
        left, top, right, bottom = region
        width, height = right - left, bottom - top
        independent = self.oracle.pixels[top:bottom, left:right]
        actual = frame.materialize()
        np.testing.assert_array_equal(actual, independent)
        value = frame.changes
        identity = (producer.instance_id, self.camera.generation, self.camera.rotation_angle, region)
        if identity != self.identity:
            expected = tuple(range(((width + TILE - 1) // TILE) * ((height + TILE - 1) // TILE)))
            reason = "initial" if self.identity is None else "reset"
        else:
            expected = changed_indices(self.previous, independent)
            reason = "changed" if expected else "identical"
        rectangles = changed_rectangles(expected, (width, height))
        stamp = self.stamp(independent, region)
        row = {
            "workload": workload, "requested_stamp": step.stamp, "observed_stamp": stamp,
            "region": region, "frame_serial": int(producer.frame_serial), "reason": value.reason,
            "fallback": value.fallback_reason, "changed_tiles": len(value.indices),
            "expected_tiles": len(expected), "present_time_qpc": value.present_time_qpc,
        }
        self.frames.append(row)
        assert value.size == (width, height) and value.reason == reason, row
        assert value.present_time_qpc == (int(producer.last_present_time) or None), row
        assert value.accumulated_frames == int(producer.accumulated_frames), row
        assert value.indices == expected and value.rectangles == rectangles, row
        assert len(frame.tiles) == len(rectangles)
        tight = []
        for tile, (x1, y1, x2, y2) in zip(frame.tiles, rectangles):
            pixels = independent[y1:y2, x1:x2]
            assert not tile.flags.writeable
            np.testing.assert_array_equal(tile, pixels)
            if reason in {"initial", "reset"}:
                tight.append((x1, y1, x2, y2))
                continue
            old = self.previous[y1:y2, x1:x2]
            ys, xs = np.nonzero(np.any(pixels != old, axis=2))
            tight.append((x1 + int(xs.min()), y1 + int(ys.min()), x1 + int(xs.max()) + 1, y1 + int(ys.max()) + 1))
        assert value.tight_rectangles == tuple(tight), row
        window = (119, 123, 261, 263)
        np.testing.assert_array_equal(frame.materialize(window), independent[123:263, 119:261])
        self.previous, self.identity = independent.copy(), identity
        self.counts[workload]["oracle_matches"] += 1
        self.counts[workload][reason] += 1
        self.counts[workload][f"fallback_{value.fallback_reason or 'none'}"] += 1
        row["oracle_match"] = True
        return stamp


def verify_tiled_live(target, steps, record):
    import rapidshot

    left, top, right, bottom = target["bounds"]
    fixture = scenes(right - left, bottom - top, visible_steps=steps, passive_steps=steps)
    with ScenePlayer(target["bounds"]) as player:
        player.prepare(fixture)
        camera = rapidshot.create(device_idx=target["device_index"], output_idx=target["output_index"],
                                  backend="dxgi", output_color="BGRA", timeout_ms=3,
                                  nvidia_gpu=False, pool_output=True, pool_size_frames=2)
        try:
            with FullDXGIOracle(camera) as oracle:
                verifier = TiledVerifier(camera, oracle)
                record.update(frames=verifier.frames, counts=verifier.counts, oracle="same_acquisition_full_bgra")
                for workload, updates in fixture.items():
                    with _require_frame(verifier, player, workload, updates[0], restart=True):
                        pass
                    for update in updates[1:]:
                        if update.visible_update:
                            with _require_frame(verifier, player, workload, update):
                                pass
                        else:
                            player.present(update)
                            delivered = verifier.sample(workload, update, single=True)
                            if delivered is not None:
                                delivered.release()
                sparse = fixture["sparse"]
                _verify_gap(verifier, player, sparse[-3:])
                crop, shifted = (256, 12, 1280, 900), (272, 20, 1296, 908)
                with _require_frame(verifier, player, "retained_tile_pages", sparse[0], crop, restart=True) as first:
                    first_pixels = first.materialize()
                    with _require_frame(verifier, player, "retained_tile_pages", sparse[1], shifted) as second:
                        assert second.changes.reason == "reset"
                        player.present(sparse[2])
                        serial = camera._duplicator.frame_serial
                        try:
                            camera.grab_tiles(shifted)
                        except PoolExhaustedError:
                            pass
                        else:
                            raise AssertionError("The two held tiled leases did not exhaust the lease limit")
                        assert camera._duplicator.frame_serial == serial
                    recovered = verifier.sample("retained_tile_pages", sparse[2], shifted)
                    if recovered is None:
                        raise RuntimeError("Public tiled grab did not resume after pool exhaustion")
                    with recovered:
                        assert camera._duplicator.frame_serial > serial
                        np.testing.assert_array_equal(first.materialize(), first_pixels)
                stats = camera.change_audit_stats()
                assert stats.audited_frames > 0 and stats.miss_frames == 0
                record["audit"] = {
                    "audited_frames": stats.audited_frames, "miss_frames": stats.miss_frames,
                    "missed_tiles": stats.missed_tiles,
                }
                record["tiled_pool_exhaustion"] = {"slots": 2, "acquired_while_full": False, "recovered": True}
        finally:
            camera.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", default="HP 324pe")
    parser.add_argument("--steps", type=int, choices=range(3, 25), default=12, metavar="3..24")
    parser.add_argument("--offline-only", action="store_true")
    parser.add_argument("--with-tiles", action="store_true", help="also verify candidate readback with a shadow full copy")
    parser.add_argument("--tiled-only", action="store_true", help="verify only candidate readback with a shadow full copy")
    parser.add_argument("--out", type=Path, default=ROOT / "results-ownership" / "exact-live-check.json")
    args = parser.parse_args()
    record = {"rapidshot": version("rapidshot"), "offline": offline_alpha(), "status": "incomplete"}
    if args.offline_only:
        print(record["offline"])
        return

    dpi_aware()
    target = select_monitor(args.monitor)
    if target["physical_refresh_hz"] < 1:
        raise RuntimeError("The selected fixture monitor has no verified refresh rate")
    record["target"], record["steps"] = target, args.steps
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if not args.tiled_only:
            verify_live(target, args.steps, record)
        if args.with_tiles or args.tiled_only:
            record["tiled"] = {}
            verify_tiled_live(target, args.steps, record["tiled"])
        record["status"] = "pass"
    except BaseException as error:
        record["failure"] = str(error)
        raise
    finally:
        write_json(args.out, record)
    count = len(record.get("frames", [])) + len(record.get("tiled", {}).get("frames", []))
    print(f"Verified {count} delivered frames against the BGRA oracle. {args.out}")


if __name__ == "__main__":
    main()
