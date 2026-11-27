"""Count operations on the fixed-wait DXGI idle path in separate workers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
from random import Random
import sys
import threading
import time
from unittest.mock import patch

from cost_telemetry import ProcessSampler
from downstream_benchmark import dpi_aware
from downstream_scene import ScenePlayer, scenes
from idea_live import Worker, distribution, make_camera, save
from monitor_target import select_monitor


ARMS = ("plain", "exact", "tiled")
WAIT_MS = 5
WINDOW_SECONDS = 10.05
WINDOW_RESERVE_SECONDS = 29


class InterfaceProbe:
    """Forward a child-local handle, observing only the named calls."""

    def __init__(self, target, **observers):
        self.target = target
        self.observers = observers

    def __bool__(self):
        return bool(self.target)

    def __getattr__(self, name):
        method = getattr(self.target, name)
        observe = self.observers.get(name)
        if observe is None:
            return method

        def call(*args, **kwargs):
            return observe(method, *args, **kwargs)

        return call


class IdlePathProbe:
    """Attach after seeding. Restores the inspected objects on scope exit."""

    def __init__(self, camera):
        self.camera = camera
        self.rows = []
        self.current = None
        self.stack = ExitStack()

    def __enter__(self):
        from comtypes import COMError
        from rapidshot._libs.dxgi import DXGI_ERROR_WAIT_TIMEOUT
        from rapidshot.changes import _ChangeHistory, _TiledHistory
        from rapidshot.core.stagesurf import StageSurface
        from rapidshot.memory_pool import BaseMemoryPool

        self.com_error = COMError
        self.wait_timeout_hresult = int(DXGI_ERROR_WAIT_TIMEOUT) & 0xFFFFFFFF
        duplication = self.camera._duplicator
        original_handle = duplication.duplicator
        device = self.camera._device
        original_context = device.im_context
        try:
            observed = InterfaceProbe(original_handle, AcquireNextFrame=self._acquire,
                                      ReleaseFrame=self._call_counter("dxgi_release_calls"))
            duplication.duplicator = observed

            def restore_duplication():
                # A loss can set the production handle to None. Keep that result.
                if duplication.duplicator is observed:
                    duplication.duplicator = original_handle

            self.stack.callback(restore_duplication)
            self.stack.enter_context(patch.object(device, "im_context", InterfaceProbe(
                original_context, CopySubresourceRegion=self._copy_box,
            )))
            slots = self.camera._tile_lease_slots
            self.stack.enter_context(patch.object(self.camera, "_tile_lease_slots", InterfaceProbe(
                slots, acquire=self._tile_slot_acquire, release=self._call_counter("tile_slot_releases"),
            )))
            for kind, name, operation in (
                (BaseMemoryPool, "checkout", "pool_checkout_calls"),
                (BaseMemoryPool, "checkin", "pool_checkin_calls"),
                (StageSurface, "map", "stage_maps"),
                (StageSurface, "unmap", "stage_unmaps"),
                (type(self.camera._processor), "process", "processor_calls"),
            ):
                self._patch_method(kind, name, self._call_counter(operation))
            for kind in (_ChangeHistory, _TiledHistory):
                self._patch_method(kind, "update", self._update_history)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc_info):
        return self.stack.__exit__(*exc_info)

    def _patch_method(self, kind, name, observe):
        original = getattr(kind, name)

        def scoped(*args, **kwargs):
            return observe(original, *args, **kwargs)

        self.stack.enter_context(patch.object(kind, name, scoped))

    def _count(self, name, value=1):
        if self.current is not None:
            self.current["operations"][name] += value

    def _call_counter(self, name):
        def observe(method, *args, **kwargs):
            self._count(name)
            return method(*args, **kwargs)

        return observe

    def _acquire(self, method, wait_ms, info_ptr, resource_ptr):
        started = time.perf_counter_ns()
        status, hresult = "exception", None
        try:
            value = method(wait_ms, info_ptr, resource_ptr)
            status = "present" if info_ptr._obj.LastPresentTime else "no_present"
            return value
        except self.com_error as error:
            if error.args and isinstance(error.args[0], int):
                hresult = error.args[0] & 0xFFFFFFFF
            status = "timeout" if hresult == self.wait_timeout_hresult else "dxgi_error"
            raise
        finally:
            if self.current is not None:
                self.current["acquires"].append({
                    "wait_ms": int(wait_ms), "status": status,
                    "hresult": None if hresult is None else f"0x{hresult:08x}",
                    "start_ns": started, "end_ns": time.perf_counter_ns(),
                })

    def _copy_box(self, method, *args):
        self._count("d3d_copy_box_calls")
        source_box = args[-1]
        if source_box is not None:
            box = source_box._obj
            self._count("d3d_copy_requested_payload_bytes", (box.right - box.left) * (box.bottom - box.top) * 4)
        else:
            self._count("d3d_copy_payload_unknown_calls")
        return method(*args)

    def _tile_slot_acquire(self, method, *args, **kwargs):
        self._count("tile_slot_acquire_calls")
        result = method(*args, **kwargs)
        if not result:
            self._count("tile_slot_rejections")
        return result

    def _update_history(self, method, history, *args, **kwargs):
        self._count("history_update_calls")
        result = method(history, *args, **kwargs)
        self._count("history_written_bytes", history.last_update_bytes)
        return result

    def grab(self, acquire):
        row = {"start_ns": time.perf_counter_ns(), "delivered": False,
               "acquires": [], "operations": Counter()}
        self.current = row
        try:
            frame = acquire()
            if frame is not None:
                row["delivered"] = True
                frame.release()
        finally:
            row["end_ns"] = time.perf_counter_ns()
            self.current = None
            self.rows.append(row)


def worker(config):
    ready = threading.Event()
    stop = threading.Event()
    result = {}

    def capture():
        camera = None
        try:
            dpi_aware()
            arm = config["arm"]
            camera = make_camera(config["target"], WAIT_MS)
            acquire = {"plain": camera.grab, "exact": camera.grab_with_changes, "tiled": camera.grab_tiles}[arm]
            ready.set()
            deadline = time.perf_counter() + 3
            while time.perf_counter() < deadline and not stop.is_set():
                frame = acquire()
                if frame is not None:
                    frame.release()
                    break
            else:
                raise RuntimeError("The worker could not seed capture on the controlled monitor")

            history = camera._tile_history or camera._change_history
            seed = history.size_bytes if history is not None else 0
            with IdlePathProbe(camera) as probe:
                print(json.dumps({"seeded_bytes": seed, "seeded": True}), flush=True)
                while not stop.is_set():
                    probe.grab(acquire)
                result["rows"] = probe.rows
            result["camera_timeout_ms"] = camera.timeout_ms
        except Exception as error:
            result["error"] = repr(error)
        finally:
            ready.set()
            if camera is not None:
                camera.release()

    # Announce PID before the capture thread can announce its seed.
    print(json.dumps({"pid": os.getpid()}), flush=True)
    thread = threading.Thread(target=capture)
    thread.start()
    if not ready.wait(5):
        stop.set()
        raise RuntimeError("The probe worker could not open the camera")
    if "error" not in result:
        sys.stdin.readline()
    stop.set()
    thread.join(4)
    if thread.is_alive():
        raise RuntimeError("The probe worker did not exit after stop")
    print(json.dumps(result, allow_nan=False), flush=True)


def summarize(rows, start_ns, end_ns, seeded_bytes, arm):
    complete = [row for row in rows if start_ns <= row["start_ns"] and row["end_ns"] <= end_ns]
    overlapping = [row for row in rows if row["end_ns"] > start_ns and row["start_ns"] < end_ns]
    operations = Counter({name: 0 for name in (
        "pool_checkout_calls", "pool_checkin_calls", "tile_slot_acquire_calls", "tile_slot_releases",
        "tile_slot_rejections", "dxgi_release_calls", "d3d_copy_box_calls", "d3d_copy_requested_payload_bytes",
        "d3d_copy_payload_unknown_calls", "stage_maps", "stage_unmaps", "processor_calls",
        "history_update_calls", "history_written_bytes",
    )})
    outcomes = Counter({name: 0 for name in ("timeout", "no_present", "present", "dxgi_error", "exception")})
    wait_ms = Counter()
    wait_durations = defaultdict(list)
    for row in complete:
        operations.update(row["operations"])
        for acquire in row["acquires"]:
            status = acquire["status"]
            outcomes[status] += 1
            wait_ms[acquire["wait_ms"]] += 1
            wait_durations[status].append((acquire["end_ns"] - acquire["start_ns"]) / 1e6)
    diagnostics = []
    if arm != "plain" and not seeded_bytes:
        diagnostics.append("The exact history was empty before measurement.")
    if any(row["delivered"] for row in overlapping):
        diagnostics.append("A grab delivered a desktop frame in an overlapping call.")
    if any(acq["status"] not in ("timeout", "no_present") for row in overlapping for acq in row["acquires"]):
        diagnostics.append("DXGI delivered content or returned a capture error in an overlapping call.")
    if not complete or any(len(row["acquires"]) != 1 for row in complete):
        diagnostics.append("A fully observed grab did not make exactly one instrumented DXGI acquire.")
    if not outcomes["timeout"] or set(wait_ms) != {WAIT_MS}:
        diagnostics.append("The window lacked confirmed DXGI timeouts at the fixed wait.")
    return {"qualified": not diagnostics, "qualification_issues": diagnostics, "seeded_history_bytes": seeded_bytes,
            "start_ns": start_ns, "end_ns": end_ns, "complete_grabs": len(complete),
            "boundary_grabs": len(overlapping) - len(complete), "calls_with_delivery": sum(r["delivered"] for r in complete),
            "operations": dict(operations), "acquisition_outcomes": dict(outcomes), "acquisitions_by_wait_ms": dict(wait_ms),
            "acquire_to_return_ms": {name: distribution(values) for name, values in wait_durations.items()},
            "raw_grabs": overlapping}


def measure_arm(player, idle, alternate, target, arm, deadline):
    player.present(alternate, restart=True)
    with Worker({"target": target, "arm": arm}, script=__file__) as process:
        player.present(idle, restart=True)
        seeded = process.get(5)
        time.sleep(0.45)
        if time.perf_counter() + WINDOW_SECONDS + 5 > deadline:
            raise TimeoutError("The mechanism probe has no time left for a complete measurement")
        sampler = ProcessSampler(process.pid, pdh=False).start()
        try:
            time.sleep(WINDOW_SECONDS)
            telemetry = sampler.finish()
        finally:
            sampler.close()
        result = process.stop(timeout=5)
    cpu = telemetry["cpu"]
    summary = summarize(result["rows"], cpu["start_ns"], cpu["end_ns"], seeded["seeded_bytes"], arm)
    return {"arm": arm, "camera_timeout_ms": result["camera_timeout_ms"],
            "instrumented_cpu_context": cpu, "memory_context": telemetry["memory"], **summary}


def run(output, rounds, seconds):
    started = time.perf_counter()
    deadline = started + seconds
    output.mkdir(parents=True, exist_ok=True)
    path = output / "idle-path-probe.json"
    dpi_aware()
    target = select_monitor()
    if target["physical_refresh_hz"] != 100:
        raise RuntimeError("Expected the HP 324pe to be at 100 Hz")
    fixture = scenes(visible_steps=1, passive_steps=1)
    data = {"target": target, "fixed_wait_ms": WAIT_MS, "minimum_window_seconds": WINDOW_SECONDS,
            "elapsed_seconds": None, "complete": False, "arms": [], "errors": [],
            "scope": "Counts operations during a separate instrumented run. Payload sizes are the requested BGRA copy sizes. "
                     "Process CPU includes these hooks and is supplied for context."}
    random = Random(44268)
    with ScenePlayer(tuple(target["bounds"])) as player:
        for number in range(1, rounds + 1):
            order = list(ARMS)
            random.shuffle(order)
            if number > 1 and order == previous_order:
                order.reverse()
            previous_order = order
            for arm in order:
                if time.perf_counter() + WINDOW_RESERVE_SECONDS > deadline:
                    data["errors"].append({"round": number, "arm": arm, "error": "remaining deadline"})
                    break
                try:
                    row = measure_arm(player, fixture["idle"][0], fixture["sparse"][0], target, arm, deadline)
                    data["arms"].append({"round": number, **row})
                except (RuntimeError, OSError, TimeoutError) as error:
                    data["errors"].append({"round": number, "arm": arm, "error": repr(error)})
                save(path, data)
    data["elapsed_seconds"] = time.perf_counter() - started
    data["complete"] = len(data["arms"]) == rounds * len(ARMS) and not data["errors"]
    save(path, data)
    return 0 if data["complete"] and all(row["qualified"] for row in data["arms"]) else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=115)
    parser.add_argument("--worker", type=json.loads, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.worker)
        return 0
    if args.output is None or args.rounds < 1 or not WINDOW_RESERVE_SECONDS <= args.seconds <= 570:
        parser.error("Provide --output, positive --rounds, and --seconds between 29 and 570")
    return run(args.output, args.rounds, args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
