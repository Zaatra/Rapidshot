"""Isolated capture adapters for the cross-library cost benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import threading
import time

import numpy as np

from capture_timing_followup import Probe
from downstream_benchmark import Camera


ROOT = Path(__file__).resolve().parent
CAPTURE_ARMS = ("rapidshot_plain", "rapidshot_exact", "dxcam_copy", "dxcam_view", "mss_gdi",
                "windows_capture_dxgi", "windows_capture_wgc")
TILED_ARMS = ("rapidshot_tiled", "rapidshot_tight")
TIGHT_CONTROLS = ("rapidshot_exact_tight",)
FULL_EXACT_ARMS = ("rapidshot_exact", "rapidshot_exact_tight")
DXCAM_VIEW_ARMS = ("dxcam_view",)
DXCAM_TIGHT_RETRY_SECONDS = {"dxcam_view_tight": 0.0005, "dxcam_view_tight_2ms": 0.002}
DXCAM_TIGHT_STREAM = "dxcam_stream_tight"
DXCAM_MATCHED_ARMS = (*DXCAM_TIGHT_RETRY_SECONDS, DXCAM_TIGHT_STREAM)
TIGHT_ARMS = ("rapidshot_tight", *TIGHT_CONTROLS, *DXCAM_MATCHED_ARMS)
ADAPTIVE_WAIT_ARM = "rapidshot_adaptive_wait"
STREAM_ARMS = ("rapidshot_stream", "dxcam_stream", "dxcam_unpaced")
ARM_DEPENDENCIES = {
    "dxcam_copy": "dxcam==0.3.0", "dxcam_view": "dxcam==0.3.0",
    "mss_gdi": "mss==10.2.0",
    "windows_capture_dxgi": "windows-capture==2.0.1", "windows_capture_wgc": "windows-capture==2.0.1",
    "dxcam_stream": "dxcam==0.3.0", "dxcam_unpaced": "dxcam==0.3.0",
    **{arm: "dxcam==0.3.0" for arm in DXCAM_MATCHED_ARMS},
}


@dataclass
class Sample:
    pixels: np.ndarray | None
    owner: object
    present_ns: int | None
    accumulated_frames: int | None
    ready_ns: int | None = None
    changes: object = None
    history_bytes: int = 0
    gate_ns: int | None = None
    on_release: object = None

    def release(self):
        release = getattr(self.owner, "release", None)
        if release is not None:
            release()
        if self.on_release is not None:
            self.on_release(self.owner)
            self.on_release = None
        self.owner = None
        self.pixels = None


def qpc_to_ns(ticks, frequency):
    return ticks * 1_000_000_000 // frequency if ticks else None


class PoolAccounting:
    """Record actual pooled checkouts and checkins in this worker process."""

    def __init__(self):
        from rapidshot.memory_pool import BaseMemoryPool, PoolExhaustedError

        self.kind = BaseMemoryPool
        self.checkout = BaseMemoryPool.checkout
        self.checkin = BaseMemoryPool.checkin
        self.lock = threading.Lock()
        self.outstanding = {}
        self.durations = []
        self.high_water = 0
        self.failed = 0
        self.issued = 0
        self.returned = 0
        self.tile_delivered_ns = {}
        self.completed_tile_leases = []
        self.exhaustion_error = PoolExhaustedError

        def checkout(pool, *args, **kwargs):
            try:
                output = self.checkout(pool, *args, **kwargs)
            except PoolExhaustedError:
                with self.lock:
                    self.failed += 1
                raise
            now = time.perf_counter_ns()
            with self.lock:
                self.outstanding[id(output)] = now
                self.high_water = max(self.high_water, len(self.outstanding))
                self.issued += 1
            return output

        def checkin(pool, frame):
            # Keep the accounting lock through checkin so a racing checkout of a
            # recycled wrapper cannot replace its old timestamp prematurely.
            with self.lock:
                result = self.checkin(pool, frame)
                self.returned += 1
                started = self.outstanding.pop(id(frame), None)
                if started is not None:
                    self.durations.append(time.perf_counter_ns() - started)
                return result

        BaseMemoryPool.checkout = checkout
        BaseMemoryPool.checkin = checkin

    def reset_window(self):
        with self.lock:
            self.high_water = len(self.outstanding)
            self.failed = self.issued = self.returned = 0
            self.durations.clear()
            self.completed_tile_leases.clear()

    def tile_checkout(self, frame, started_ns, delivered_ns):
        with self.lock:
            key = id(frame)
            self.outstanding[key] = started_ns
            self.tile_delivered_ns[key] = delivered_ns
            self.high_water = max(self.high_water, len(self.outstanding))
            self.issued += 1

    def tile_checkin(self, frame):
        closed_ns = time.perf_counter_ns()
        with self.lock:
            key = id(frame)
            start = self.outstanding.pop(key)
            delivered_ns = self.tile_delivered_ns.pop(key)
            self.returned += 1
            self.durations.append(closed_ns - start)
            self.completed_tile_leases.append({"grab_started_ns": start, "delivered_ns": delivered_ns,
                                               "closed_ns": closed_ns})

    def tile_exhaustion(self):
        with self.lock:
            self.failed += 1

    def snapshot(self, camera):
        with self.lock:
            result = {
                "checkouts": self.issued, "checkins": self.returned, "exhaustions": self.failed,
                "peak_outstanding": self.high_water, "outstanding_at_end": len(self.outstanding),
                "completed_pool_lease_ns": list(self.durations),
                "completed_tile_leases": list(self.completed_tile_leases),
            }
        result["pools_at_end"] = {
            name: {"slots": pool.get_stats(), "shape": list(pool.buffer_shape)}
            for name in ("memory_pool", "_output_pool")
            if (pool := getattr(camera, name, None)) is not None
        }
        return result

    def close(self):
        self.kind.checkout = self.checkout
        self.kind.checkin = self.checkin


class AdaptiveWait:
    """Benchmark only: count real timeout HRESULTs from the Python DXGI duplicator."""

    active_ms = 5
    idle_ms = 50
    quiet_threshold = 4

    def __init__(self, camera):
        from rapidshot._libs.dxgi import DXGI_ERROR_WAIT_TIMEOUT
        from rapidshot.core.duplicator import Duplicator, _format_hresult

        if type(camera._duplicator) is not Duplicator:
            raise RuntimeError("Adaptive wait requires the Python DXGI duplicator")
        self.camera = camera
        self.kind = Duplicator
        self.original_update = Duplicator.update_frame
        self.timeout_suffix = f"(HRESULT: {_format_hresult(DXGI_ERROR_WAIT_TIMEOUT)})"
        self.reset_window()

        def observe(duplicator):
            if duplicator is not camera._duplicator:
                return self.original_update(duplicator)
            attempted_ms = duplicator.timeout_ms
            self.attempts[attempted_ms] += 1
            serial = duplicator.frame_serial
            try:
                healthy = self.original_update(duplicator)
            except Exception:
                self._on_failure()
                raise
            # The Python implementation clears last_error at each update and
            # appends this precise HRESULT only for the failure it just handled.
            # Pool exhaustion and other grab-level None results cannot count.
            if not healthy:
                self._on_failure()
            elif duplicator.frame_serial != serial:
                self.desktop_updates += 1
                self.consecutive_timeouts = 0
                self._set_wait(self.active_ms, "desktop_update")
            elif duplicator.last_error.endswith(self.timeout_suffix):
                self.confirmed_timeouts += 1
                self.consecutive_timeouts += 1
                if self.consecutive_timeouts == self.quiet_threshold:
                    self._set_wait(self.idle_ms, "confirmed_dxgi_timeouts")
            else:
                self.other_healthy_acquires += 1
                self.consecutive_timeouts = 0
            return healthy

        Duplicator.update_frame = observe

    def _on_failure(self):
        self.failures += 1
        self.consecutive_timeouts = 0
        self._set_wait(self.active_ms, "duplication_failure")

    def _set_wait(self, milliseconds, reason):
        if milliseconds != self.camera.timeout_ms:
            self.transitions.append({"qpc_ns": time.perf_counter_ns(), "from_ms": self.camera.timeout_ms,
                                     "to_ms": milliseconds, "reason": reason})
            self.camera.timeout_ms = milliseconds

    def reset_window(self):
        self.camera.timeout_ms = self.active_ms
        self.consecutive_timeouts = self.confirmed_timeouts = self.desktop_updates = 0
        self.other_healthy_acquires = self.failures = 0
        self.attempts = {self.active_ms: 0, self.idle_ms: 0}
        self.transitions = []

    def snapshot(self):
        return {"active_wait_ms": self.active_ms, "idle_wait_ms": self.idle_ms,
                "quiet_threshold": self.quiet_threshold, "attempts_by_wait_ms": self.attempts,
                "confirmed_timeouts": self.confirmed_timeouts, "desktop_updates": self.desktop_updates,
                "other_healthy_acquires": self.other_healthy_acquires, "duplication_failures": self.failures,
                "final_wait_ms": self.camera.timeout_ms, "transitions": self.transitions}

    def close(self):
        self.kind.update_frame = self.original_update


class WgcLatest:
    def __init__(self, target, hz):
        from windows_capture import WindowsCapture

        self.ready = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.frames = 0
        self.replaced = 0
        self.native_bytes = 0
        self.closed = False
        # Match the package's one-based EnumDisplayMonitors order by the exact
        # display device name, independently of any DXGI output index.
        monitor = wgc_monitor_index(target["output"])
        self.capture = WindowsCapture(monitor_index=monitor, cursor_capture=False,
                                      minimum_update_interval=1000 // hz)

        @self.capture.event
        def on_frame_arrived(frame, _):
            now = time.perf_counter_ns()
            with self.lock:
                self.frames += 1
                self.native_bytes += frame.frame_buffer.nbytes
                self.replaced += self.latest is not None
                self.latest = (frame, now)
                self.ready.set()

        @self.capture.event
        def on_closed():
            self.closed = True
            self.ready.set()

        self.control = self.capture.start_free_threaded()

    def read(self):
        if not self.ready.wait(0.010):
            return None
        with self.lock:
            latest, self.latest = self.latest, None
            self.ready.clear()
        if latest is None:
            if self.closed:
                raise RuntimeError("Windows Graphics Capture closed during the run")
            return None
        frame, ready_ns = latest
        return Sample(frame.frame_buffer, frame, frame.timespan * 100 if frame.timespan else None,
                      None, ready_ns)

    def close(self):
        self.control.stop()
        self.latest = None


def wgc_monitor_index(device):
    import ctypes as ct
    from ctypes import wintypes as wt

    class MonitorInfo(ct.Structure):
        _fields_ = [("size", wt.DWORD), ("monitor", wt.RECT), ("work", wt.RECT),
                    ("flags", wt.DWORD), ("device", wt.WCHAR * 32)]

    user = ct.WinDLL("user32", use_last_error=True)
    callback_type = ct.WINFUNCTYPE(wt.BOOL, wt.HANDLE, wt.HDC, ct.POINTER(wt.RECT), wt.LPARAM)
    user.GetMonitorInfoW.argtypes = [wt.HANDLE, ct.POINTER(MonitorInfo)]
    user.EnumDisplayMonitors.argtypes = [wt.HDC, ct.POINTER(wt.RECT), callback_type, wt.LPARAM]
    displays = []

    @callback_type
    def enumerate_one(handle, _dc, _rect, _param):
        info = MonitorInfo(size=ct.sizeof(MonitorInfo))
        displays.append(info.device if user.GetMonitorInfoW(handle, ct.byref(info)) else None)
        return True

    if not user.EnumDisplayMonitors(None, None, enumerate_one, 0):
        raise ct.WinError(ct.get_last_error())
    if displays.count(device) != 1:
        raise RuntimeError(f"Cannot resolve {device!r} in WGC monitor enumeration: {displays}")
    return displays.index(device) + 1


class DxcamStockStream:
    """Observe the stock DXcam 0.3.0 producer without changing its pacing."""

    def __init__(self, camera, frequency):
        self.camera = camera
        self.frequency = frequency
        self.events = []
        self.metadata = {}
        self.reshaped_ns = None
        self.original_grab_into = camera._grab_into
        camera._grab_into = self._observe_producer

    def _observe_producer(self, region, destination):
        result = self.original_grab_into(region, destination)
        captured, ticks, width, height = result
        if captured:
            ready_ns = time.perf_counter_ns()
            accumulated = int(self.camera._duplicator.accumulated_frames)
            # The stock producer publishes the returned tick with this slot
            # immediately after _grab_into. A reused tick has ambiguous age.
            self.metadata[ticks] = None if ticks in self.metadata else (accumulated, ready_ns)
            self.events.append((ready_ns, int(destination.nbytes)))
        elif width and height:
            self.reshaped_ns = time.perf_counter_ns()
        return result

    def read(self, budget):
        # The stock reader has no timeout and waits indefinitely between
        # updates. Observe its own event with the trial's remaining budget.
        if not self.camera._DXCamera__frame_available.wait(budget):
            return None
        read = self.camera.get_latest_frame(copy=True, with_timestamp=True)
        if read is None:
            return None
        pixels, seconds = read
        ticks = round(seconds * self.frequency)
        if ticks not in self.metadata:
            raise RuntimeError("DXcam returned a frame without matching producer metadata")
        details = self.metadata[ticks]
        accumulated, ready_ns = details if details is not None else (None, None)
        return Sample(pixels, pixels, qpc_to_ns(ticks, self.frequency) if accumulated else None,
                      accumulated, ready_ns)

    def accounting(self, start_ns, stop_ns, ring_payload_bytes):
        if self.reshaped_ns is not None:
            raise RuntimeError("DXcam stream resized during the fixed-geometry trial")
        frames = [size for ready_ns, size in self.events if start_ns <= ready_ns < stop_ns]
        produced_bytes = sum(frames)
        return {"stream_produced_frames": len(frames), "gpu_staging_payload_bytes": produced_bytes,
                "mapped_staging_to_cpu_bytes": produced_bytes,
                "stream_ring_payload_bytes_retained": ring_payload_bytes}

    def restore(self):
        self.camera._grab_into = self.original_grab_into


class CostAdapter:
    def __init__(self, arm, target, hz, qpc_frequency):
        self.arm = arm
        self.frequency = qpc_frequency
        self.target = target
        self.copy_events = {}
        self.pool = None
        self.adaptive_wait = None
        self.camera = None
        self.probe = None
        self.tight_stream = None
        self.wgc = None
        self.exact_gate_ns = None
        self.history_start = 0
        self.tile_history_retained = 0
        self.payload_bytes = target["bounds"][2] - target["bounds"][0]
        self.payload_bytes *= 4 * (target["bounds"][3] - target["bounds"][1])
        if arm.startswith("rapidshot"):
            self.pool = PoolAccounting()
        try:
            if arm in STREAM_ARMS:
                self.probe = Probe(arm, target)
                self.camera = self.probe.camera
            elif arm.startswith("windows_capture"):
                path = ROOT / "vendor"
                if path.is_dir():
                    sys.path.insert(0, str(path))
                if arm == "windows_capture_dxgi":
                    from windows_capture import DxgiDuplicationSession

                    self.camera = DxgiDuplicationSession(wgc_monitor_index(target["output"]))
                else:
                    self.wgc = WgcLatest(target, hz)
                    self.camera = self.wgc
            elif arm.startswith("rapidshot"):
                self.camera = Camera("rapidshot_full", target).camera
                if arm in (*FULL_EXACT_ARMS, *TILED_ARMS):
                    from rapidshot.changes import _ChangeHistory, _TiledHistory

                    self._hook_exact_gate(_TiledHistory if arm in TILED_ARMS else _ChangeHistory)
                if arm == ADAPTIVE_WAIT_ARM:
                    self.adaptive_wait = AdaptiveWait(self.camera)
            elif arm.startswith("dxcam"):
                self.camera = Camera("dxcam", target).camera
                if arm == DXCAM_TIGHT_STREAM:
                    self.tight_stream = DxcamStockStream(self.camera, self.frequency)
                    self.camera.start(target_fps=100, video_mode=False)
            elif arm == "mss_gdi":
                self.mss = Camera("mss", target)
                self.camera = self.mss.camera
            else:
                raise ValueError(f"Unknown capture arm {arm!r}")
        except BaseException:
            self.close()
            raise

    @property
    def stamp_clock(self):
        return {"mss_gdi": "unavailable: GDI has no timestamp",
                "windows_capture_dxgi": "unavailable: binding omits DXGI LastPresentTime",
                "windows_capture_wgc": "WGC SystemRelativeTime (100 ns units)"}.get(self.arm, "DXGI LastPresentTime")

    def _hook_exact_gate(self, history_type):
        self._history_type = history_type
        self._history_update = history_type.update

        def timed(history, *args, **kwargs):
            started = time.perf_counter_ns()
            result = self._history_update(history, *args, **kwargs)
            self.exact_gate_ns = time.perf_counter_ns() - started
            return result

        history_type.update = timed

    def _count(self, name, total):
        self.copy_events[name] = self.copy_events.get(name, 0) + total

    def reset_window(self, timing=None):
        self.copy_events.clear()
        self.measurement_window = timing
        history = getattr(self.camera, "_change_history", None)
        self.history_start = history.bytes_written if history is not None else 0
        if self.pool is not None:
            self.pool.reset_window()
        if self.adaptive_wait is not None:
            self.adaptive_wait.reset_window()
        if self.wgc is not None:
            with self.wgc.lock:
                self.wgc_before = (self.wgc.frames, self.wgc.native_bytes, self.wgc.replaced)
        if self.probe is not None:
            self.stream_before = len(self.probe.producer) if not self.pool else self.camera.stream_stats["frames_captured"]

    def read(self, budget=0.005):
        arm = self.arm
        if self.tight_stream is not None:
            shot = self.tight_stream.read(budget)
            if shot is not None:
                self._count("stream_consumer_copy_bytes", shot.pixels.nbytes)
                self._count("stream_consumer_frame_allocations", 1)
            return shot
        if arm in STREAM_ARMS:
            event = (self.camera._frame_available_event if self.pool is not None
                     else self.camera._DXCamera__frame_available)
            if not event.wait(budget):
                return None
            shot, qpc, accumulated, producer = self.probe.read(self.frequency)
            if shot is None:
                return None
            ready_ns = qpc_to_ns(producer[0], self.frequency) if producer and producer[1] == "qpc" else (
                producer[0] if producer else None)
            self._count("stream_consumer_copy_bytes", shot.pixels.nbytes)
            self._count("stream_consumer_frame_allocations", 1)
            return Sample(shot.pixels, shot, qpc_to_ns(qpc, self.frequency) if accumulated else None,
                          accumulated, ready_ns)
        if self.wgc is not None:
            return self.wgc.read()
        if arm == "windows_capture_dxgi":
            frame = self.camera.acquire_frame(timeout_ms=5)
            if frame is None:
                return None
            if frame.color_format != "bgra8":
                raise RuntimeError(f"Unsupported Windows Capture DXGI format {frame.color_format!r}")
            pixels = frame.to_numpy()
            self._count("gpu_staging_payload_bytes", pixels.nbytes)
            self._count("gpu_staging_allocations", 1)
            return Sample(pixels, frame, None, None)
        if arm == "mss_gdi":
            frame = self.camera.grab(self.mss.region)
            pixels = np.asarray(frame)
            self._count("gdi_source_to_dib_payload_bytes", pixels.nbytes)
            self._count("dib_to_cpu_bytes", pixels.nbytes)
            self._count("cpu_frame_payload_allocations", 1)
            return Sample(pixels, frame, None, None)
        if arm in TILED_ARMS:
            started_ns = time.perf_counter_ns()
            try:
                frame = self.camera.grab_tiles(tight_rectangles=arm in TIGHT_ARMS)
            except self.pool.exhaustion_error:
                self.pool.tile_exhaustion()
                raise
            if frame is None:
                return None
            self.pool.tile_checkout(frame, started_ns, time.perf_counter_ns())
            try:
                cost = self.camera._last_tile_capture_cost
                changes = frame.changes
                written = cost["history_bytes_written"]
                self._count("gpu_staging_payload_bytes", cost["gpu_readback_bytes"])
                self._count("gpu_copy_operations", cost["gpu_copy_operations"])
                # Committing immutable history pages is the only host pixel
                # write in this path until a consumer materializes a crop.
                self._count("private_history_bytes_written", written)
                self.tile_history_retained = cost["history_bytes_retained"]
                present_ns = qpc_to_ns(changes.present_time_qpc, self.frequency)
                return Sample(None, frame, present_ns, changes.accumulated_frames,
                              changes=changes, history_bytes=written, gate_ns=self.exact_gate_ns,
                              on_release=self.pool.tile_checkin)
            except BaseException:
                frame.release()
                self.pool.tile_checkin(frame)
                raise
        if arm in FULL_EXACT_ARMS:
            frame = self.camera.grab_with_changes(tight_rectangles=arm in TIGHT_ARMS)
            if frame is None:
                return None
            changes = frame.changes
            history = self.camera._change_history
            pixels = frame.pixels
            present_ns = qpc_to_ns(changes.present_time_qpc, self.frequency)
            accumulated = changes.accumulated_frames
            written = history.last_update_bytes
            gate = self.exact_gate_ns
        else:
            frame = self.camera.grab_view() if arm in (*DXCAM_VIEW_ARMS, *DXCAM_TIGHT_RETRY_SECONDS) else self.camera.grab()
            if frame is None:
                return None
            duplicate = self.camera._duplicator
            accumulated = int(duplicate.accumulated_frames)
            qpc = duplicate.last_present_time if arm.startswith("rapidshot") else duplicate.latest_frame_ticks
            present_ns = qpc_to_ns(qpc, self.frequency) if accumulated else None
            pixels = getattr(frame, "array", frame)
            changes = gate = None
            written = 0
        self._count("gpu_staging_payload_bytes", pixels.nbytes)
        self._count("mapped_staging_to_cpu_bytes", pixels.nbytes)
        if arm == "dxcam_copy":
            self._count("cpu_frame_payload_allocations", 1)
        return Sample(pixels, frame, present_ns, accumulated, changes=changes, history_bytes=written, gate_ns=gate)

    def snapshot(self):
        result = dict(self.copy_events)
        history = getattr(self.camera, "_change_history", None)
        if history is not None:
            result["private_history_bytes_written"] = history.bytes_written - self.history_start
            result["private_history_payload_bytes_retained"] = history.size_bytes
        if self.arm in TILED_ARMS:
            result.setdefault("private_history_bytes_written", 0)
            result.setdefault("gpu_staging_payload_bytes", 0)
            result["private_history_payload_bytes_retained"] = self.tile_history_retained
        if self.wgc is not None:
            with self.wgc.lock:
                frames = self.wgc.frames - self.wgc_before[0]
                result.update(gpu_staging_payload_bytes=self.wgc.native_bytes - self.wgc_before[1],
                              gpu_staging_allocations=frames, wgc_produced_frames=frames,
                              wgc_replaced_frames=self.wgc.replaced - self.wgc_before[2])
        if self.probe is not None:
            if self.pool is not None:
                result["stream"] = self.camera.stream_stats
                frames = result["stream"]["frames_captured"] - self.stream_before
            else:
                frames = len(self.probe.producer) - self.stream_before
            result["stream_produced_frames"] = frames
            result["gpu_staging_payload_bytes"] = frames * self.payload_bytes
            result["mapped_staging_to_cpu_bytes"] = frames * self.payload_bytes
        if self.tight_stream is not None:
            result.update(self.tight_stream.accounting(self.measurement_window["start_ns"],
                                                       self.measurement_window["stop_ns"],
                                                       self.camera.max_buffer_len * self.payload_bytes))
        if self.pool is not None:
            result["pool"] = self.pool.snapshot(self.camera)
            if self.arm in TILED_ARMS:
                result["pool"]["tile_snapshot_slots"] = self.camera._init_args["pool_size_frames"]
        if self.adaptive_wait is not None:
            result["adaptive_wait"] = self.adaptive_wait.snapshot()
        return result

    def close(self):
        try:
            if self.probe is not None:
                self.probe.close()
            elif self.wgc is not None:
                self.wgc.close()
            elif self.camera is not None:
                if self.arm == "mss_gdi":
                    self.mss.close()
                elif self.arm != "windows_capture_dxgi":
                    self.camera.release()
        finally:
            if self.tight_stream is not None:
                self.tight_stream.restore()
            if self.adaptive_wait is not None:
                self.adaptive_wait.close()
            if hasattr(self, "_history_update"):
                self._history_type.update = self._history_update
            if self.pool is not None:
                self.pool.close()
