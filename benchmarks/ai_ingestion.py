"""Measure screen pixels -> (1, 3, 640, 640) FP16 (or FP32) RGB on CUDA.

    python benchmarks/ai_ingestion.py --seconds 8 --with-motion --out results.json
    python benchmarks/ai_ingestion.py --verify --with-motion
    python benchmarks/ai_ingestion.py --with-motion --motion-fps 120

Each path runs in a fresh interpreter. All paths resize the complete source with
half-pixel bilinear sampling. CPU OpenCV and GPU floating-point rounding may
differ by up to 2/255. Verification compares against a NumPy reference from the
SAME captured frame, outside timing. Cross-adapter input must be BGRA8.

With --with-motion, startup stages, PIDs, errors and animation rates are retained
in a unique log directory printed before startup. --motion-fps is an optional
diagnostic cap (0 preserves uncapped operation); achieved animation rate must be
considered when interpreting throughput. The old provisional dataset used different resizing and must not be pooled with
these results.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
OUT = 640
TARGET_SHAPE = (1, 3, OUT, OUT)
PIXEL_TOLERANCE = 2.0 / 255.0

#: Element type every path must produce. ROADMAP § 7.0 specifies **FP16** — it
#: is half the bytes and what production inference consumes — and this script
#: measured FP32 until 2026-09-14, which is the representation § 6.1 found on
#: the *losing* side of the convert-first question at 1080p. FP32 stays
#: selectable so the 2026-09-10 recording remains reproducible. The dtype is
#: recorded in the results: **rows taken at different dtypes are not
#: comparable**, because the payload that crosses the bus is a different size.
DTYPES = ("float16", "float32")
TARGET_DTYPE = "float16"

PATHS = ("mss", "dxcam", "rapidshot-cpu", "rapidshot-cupy", "rapidshot-xadapter",
         "rapidshot-xadapter-async", "rapidshot-converter-xadapter",
         "rapidshot-converter")


class PathUnavailable(RuntimeError):
    """This path cannot run on this machine, and that is not a failure.

    Kept distinct from an error because the two deserve different reactions. A
    CUDA export on an Optimus laptop is refused *by design* — capture is on the
    iGPU and CUDA is on the discrete GPU — whereas a path that breaks is a bug.
    Recorded as a skip so it neither fails the run nor silently disappears.
    """


@dataclass
class Sample:
    tensor: object
    reference: object = None


def stage(event, **fields):
    print(json.dumps({"event": event, "pid": os.getpid(), **fields}), file=sys.stderr,
          flush=True)
    try:
        os.fsync(sys.stderr.fileno())
    except (OSError, ValueError):
        pass


def _check_bgra(image):
    if len(image.shape) != 3 or image.shape[2] != 4 or min(image.shape[:2]) <= 0:
        raise ValueError("expected a nonempty H x W x 4 BGRA image")
    if str(image.dtype) != "uint8":
        raise ValueError("expected uint8 BGRA pixels")


def _gpu_to_tensor(bgra, cp):
    """Bilinear resize over the full extent, including upscaling and edge clamp."""
    _check_bgra(bgra)
    height, width = bgra.shape[:2]
    y = cp.maximum((cp.arange(OUT, dtype=cp.float32) + 0.5) * (height / OUT) - 0.5, 0)
    x = cp.maximum((cp.arange(OUT, dtype=cp.float32) + 0.5) * (width / OUT) - 0.5, 0)
    y0, x0 = y.astype(cp.int32), x.astype(cp.int32)
    y1, x1 = cp.minimum(y0 + 1, height - 1), cp.minimum(x0 + 1, width - 1)
    wy, wx = (y - y0)[:, None, None], (x - x0)[None, :, None]
    top_left = bgra[y0[:, None], x0[None, :]].astype(cp.float32)
    top_right = bgra[y0[:, None], x1[None, :]].astype(cp.float32)
    bottom_left = bgra[y1[:, None], x0[None, :]].astype(cp.float32)
    bottom_right = bgra[y1[:, None], x1[None, :]].astype(cp.float32)
    top = top_left + (top_right - top_left) * wx
    bottom = bottom_left + (bottom_right - bottom_left) * wx
    small = cp.rint(top + (bottom - top) * wy).clip(0, 255).astype(cp.uint8)
    chw = cp.ascontiguousarray(small[:, :, 2::-1].transpose(2, 0, 1))
    # Normalise in FP32 and narrow once: FP16's precision then applies to
    # the result rather than to the arithmetic that produced it.
    scaled = chw.astype(cp.float32) / cp.float32(255)
    return scaled.astype(TARGET_DTYPE).reshape(TARGET_SHAPE)


def _cpu_to_tensor(bgra, cp, np):
    import cv2

    _check_bgra(bgra)
    small = cv2.resize(bgra, (OUT, OUT), interpolation=cv2.INTER_LINEAR)
    chw = np.ascontiguousarray(small[:, :, 2::-1].transpose(2, 0, 1))
    scaled = cp.asarray(chw).astype(cp.float32) / cp.float32(255)
    return scaled.astype(TARGET_DTYPE).reshape(TARGET_SHAPE)


def reference_tensor(bgra, np):
    """Independent separable, float64 NumPy interpolation; no GPU/OpenCV calls."""
    _check_bgra(bgra)
    height, width = bgra.shape[:2]
    xs = (np.arange(OUT) + 0.5) * width / OUT - 0.5
    ys = (np.arange(OUT) + 0.5) * height / OUT - 0.5
    # np.interp clamps outside the source interval to its edge pixel.
    horizontal = np.empty((height, OUT, 3), dtype=np.float64)
    for row in range(height):
        for channel in range(3):
            horizontal[row, :, channel] = np.interp(
                xs, np.arange(width), bgra[row, :, 2 - channel])
    resized = np.empty((OUT, OUT, 3), dtype=np.float64)
    for column in range(OUT):
        for channel in range(3):
            resized[:, column, channel] = np.interp(
                ys, np.arange(height), horizontal[:, column, channel])
    return (resized.transpose(2, 0, 1)[None] / 255.0).astype(np.float32)


def validate_tensor(array, source, np):
    expected = reference_tensor(source, np)
    shape_ok = tuple(array.shape) == TARGET_SHAPE
    dtype_ok = array.dtype == np.dtype(TARGET_DTYPE)
    finite = bool(np.isfinite(array).all())
    range_ok = bool(array.size and finite and array.min() >= -1e-6
                    and array.max() <= 1 + 1e-6)
    out = {"shape": list(array.shape), "dtype": str(array.dtype),
           "target_dtype": TARGET_DTYPE,
           "shape_ok": shape_ok, "dtype_ok": dtype_ok, "range_ok": range_ok,
           "reference_shape": list(source.shape), "pixel_tolerance": PIXEL_TOLERANCE}
    if array.size and finite:
        out.update(min=float(array.min()), max=float(array.max()), mean=float(array.mean()))
    if shape_ok and finite:
        diff = np.abs(array.astype(np.float64) - expected)
        out.update(max_abs_error=float(diff.max()), mean_abs_error=float(diff.mean()))
    out["content_ok"] = bool(shape_ok and finite
                             and out["max_abs_error"] <= PIXEL_TOLERANCE)
    out["verified"] = all((shape_ok, dtype_ok, range_ok, out["content_ok"]))
    if not out["verified"]:
        out["error"] = (f"tensor failed shape, {TARGET_DTYPE}, range, or "
                        "same-frame content validation")
    return out


def _cpu_sample(bgra, cp, np, verify):
    source = np.array(bgra, copy=True) if verify else None
    return Sample(_cpu_to_tensor(bgra, cp, np), source)


def _close_camera(cam):
    close = getattr(cam, "release", None) or getattr(cam, "stop", None)
    if close is None:
        raise RuntimeError("camera has no supported cleanup method")
    close()


def _adapter_mss(cp, np, verify=False):
    import mss

    sct = mss.mss()
    try:
        monitor = sct.monitors[1]
    except BaseException:
        sct.close()
        raise

    def produce():
        shot = sct.grab(monitor)
        bgra = np.frombuffer(shot.raw, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return _cpu_sample(bgra, cp, np, verify)

    return produce, sct.close, {"h2d_bytes_per_frame": 3 * OUT * OUT}


def _adapter_dxcam(cp, np, verify=False):
    import dxcam

    cam = dxcam.create(output_color="BGRA")

    def produce():
        bgra = cam.grab()
        return None if bgra is None else _cpu_sample(bgra, cp, np, verify)

    return produce, lambda: _close_camera(cam), {"h2d_bytes_per_frame": 3 * OUT * OUT}


def _adapter_rapidshot_cpu(cp, np, verify=False):
    import rapidshot

    cam = rapidshot.create(output_color="BGRA")

    def produce():
        frame = cam.grab()
        if frame is None:
            return None
        try:
            return _cpu_sample(np.asarray(frame), cp, np, verify)
        finally:
            release = getattr(frame, "release", None)
            if release:
                release()

    return produce, cam.release, {"h2d_bytes_per_frame": 3 * OUT * OUT}


def _adapter_rapidshot_cupy(cp, np, verify=False):
    import rapidshot

    cam = rapidshot.create(output_color="BGRA", nvidia_gpu=True)

    def produce():
        frame = cam.grab()
        if frame is None:
            return None
        inner = getattr(frame, "array", frame)
        try:
            source = cp.asnumpy(inner) if verify else None
            return Sample(_gpu_to_tensor(inner, cp), source)
        finally:
            release = getattr(frame, "release", None)
            if release:
                # Never return pooled storage while CUDA is still reading it.
                cp.cuda.runtime.deviceSynchronize()
                release()

    return produce, cam.release, {"h2d_bytes_per_frame": None,
                                  "note": "full BGRA upload; bilinear resize on CUDA"}


def _validate_transfer(transfer):
    if transfer.dxgi_format != 87 or transfer.bytes_per_pixel != 4:
        raise ValueError(f"cross-adapter benchmark requires BGRA8 (DXGI 87), "
                         f"got DXGI {transfer.dxgi_format}")
    if (transfer.width <= 0 or transfer.height <= 0
            or transfer.row_pitch < transfer.width * 4
            or transfer.row_pitch % 4 or transfer.total_bytes % 4
            or transfer.total_bytes < transfer.height * transfer.row_pitch):
        raise ValueError("invalid cross-adapter image dimensions, pitch, or allocation")


def _pitched_bgra(raw, transfer):
    _validate_transfer(transfer)
    image_bytes = transfer.height * transfer.row_pitch
    if raw.size < image_bytes:
        raise ValueError("cross-adapter buffer is shorter than the image footprint")
    rows = raw[:image_bytes].reshape(transfer.height, transfer.row_pitch)
    return rows[:, :transfer.width * 4].reshape(transfer.height, transfer.width, 4)


def _adapter_rapidshot_xadapter(cp, np, verify=False):
    sys.path.insert(0, str(REPO / "examples"))
    import rapidshot
    from rapidshot import native
    from gpu_tensor_to_cupy import CudaTensor

    cam = rapidshot.create()
    state = {"transfer": None, "view": None}

    class TensorSource:
        cuda_handle_type = 4  # CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
        cuda_dedicated = False
        def __init__(self, transfer):
            self._transfer = transfer  # Own the resource for the CUDA import's lifetime.
            self.shared_output_handle = transfer.shared_destination_handle
            self.output_byte_size = transfer.total_bytes

    def produce():
        # Warmup, measurement and verification all synchronize before reusing
        # this destination. D3D12's copy fence alone cannot order CUDA readers.
        frame = cam.grab_frame()
        if frame is None:
            return None
        with frame:
            if state["transfer"] is None:
                transfer = native.cross_adapter_transfer(frame)
                _validate_transfer(transfer)
                state["transfer"] = transfer
                state["view"] = CudaTensor(
                    TensorSource(transfer), (transfer.total_bytes // 4,), device=0)
            transfer = state["transfer"]
            if verify:
                reference_bytes = transfer.transfer_with_reference(frame)
            else:
                transfer.transfer(frame)
        bgra = _pitched_bgra(state["view"].array.view(cp.uint8), transfer)
        source = None
        if verify:
            source = _pitched_bgra(np.frombuffer(reference_bytes, dtype=np.uint8), transfer).copy()
        return Sample(_gpu_to_tensor(bgra, cp), source)

    def teardown():
        try:
            if state["view"] is not None:
                state["view"].close()
        finally:
            state["view"] = None
            state["transfer"] = None
            cam.release()

    return produce, teardown, {"h2d_bytes_per_frame": 0,
                               "note": "system-memory shared heap; no CPU pixel round trip"}


def _adapter_rapidshot_xadapter_async(cp, np, verify=False):
    """The same cross-adapter path, submitted without blocking.

    The blocking variant measured ~2 ms slower per frame than the CPU path,
    which is about what the copy itself costs. `transfer_async()` exists to hide
    that: it submits and returns the calling thread, leaving synchronisation to
    `wait_shared_fence(value)` or to a GPU-side wait on `shared_fence_handle`.

    **This uses the CPU-side wait, which is the weaker of the two.** The README
    reports the GPU-side semaphore wait as the one worth 7-14%, and the CPU wait
    as buying nothing against a GPU consumer -- the calling thread was never the
    constraint. Doing it the better way needs `cuImportExternalSemaphore` glue
    that does not exist here: `examples/gpu_tensor_to_cupy.py` imports external
    *memory*, not a semaphore.

    So read a null result as "the CPU-side wait did not help", which is what the
    README already predicts, rather than as "async does not help".
    """
    sys.path.insert(0, str(REPO / "examples"))
    import rapidshot
    from rapidshot import native
    from gpu_tensor_to_cupy import CudaTensor

    cam = rapidshot.create()
    state = {"transfer": None, "view": None}

    class TensorSource:
        cuda_handle_type = 4
        cuda_dedicated = False
        def __init__(self, transfer):
            self._transfer = transfer  # Own the resource for the CUDA import's lifetime.
            self.shared_output_handle = transfer.shared_destination_handle
            self.output_byte_size = transfer.total_bytes

    def produce():
        frame = cam.grab_frame()
        if frame is None:
            return None
        reference_bytes = None
        with frame:
            if state["transfer"] is None:
                transfer = native.cross_adapter_transfer(frame)
                _validate_transfer(transfer)
                state["transfer"] = transfer
                state["view"] = CudaTensor(
                    TensorSource(transfer), (transfer.total_bytes // 4,), device=0)
            transfer = state["transfer"]
            # The wait stays inside the `with`: the frame has to outlive the
            # copy that reads it, and async submission returns before that copy
            # has run. Releasing the frame first is the whole hazard async
            # introduces over the blocking call.
            if verify:
                value = transfer.transfer_async_with_reference(frame)
                transfer.wait_shared_fence(value)
                reference_bytes = transfer.read_back_source()
            else:
                transfer.wait_shared_fence(transfer.transfer_async(frame))
        bgra = _pitched_bgra(state["view"].array.view(cp.uint8), transfer)
        source = None
        if verify and reference_bytes is not None:
            source = _pitched_bgra(
                np.frombuffer(reference_bytes, dtype=np.uint8), transfer).copy()
        return Sample(_gpu_to_tensor(bgra, cp), source)

    def teardown():
        try:
            if state["view"] is not None:
                state["view"].close()
        finally:
            state["view"] = None
            state["transfer"] = None
            cam.release()

    return produce, teardown, {"h2d_bytes_per_frame": 0,
                               "note": "transfer_async + CPU-side fence wait"}


def _validate_tensor_transfer(transfer, np):
    """The payload must be the tensor's own size, on a real second GPU."""
    expected = 3 * OUT * OUT * np.dtype(TARGET_DTYPE).itemsize
    # Checked against `expected`, not against what arrived: CudaTensor maps
    # 4-byte words and the result is viewed back to its own dtype, so a target
    # whose size is not a whole number of words cannot be read at all. Testing
    # the arrival instead made this unreachable -- the size check below would
    # already have rejected anything that failed it.
    if expected % 4:
        raise ValueError(
            f"{TARGET_SHAPE} {TARGET_DTYPE} is {expected} bytes, which is not a "
            "whole number of 4-byte words and cannot be viewed after transfer")
    if transfer.total_bytes != expected:
        raise ValueError(
            f"TensorTransfer carries {transfer.total_bytes} bytes; "
            f"{TARGET_SHAPE} {TARGET_DTYPE} is {expected}")
    if transfer.destination_is_software:
        raise PathUnavailable(
            "the only second adapter is WARP, so 'no host-to-device transfer' "
            "would be measuring a system-memory copy rather than a GPU bus")


def _adapter_rapidshot_converter_xadapter(cp, np, verify=False):
    """2.6 ordering B: convert on the capture adapter, move the small result.

    `rapidshot-xadapter` moves the whole frame - 16.38 MB at 2560x1600 - and
    resizes it on the destination with CuPy. This moves the *finished* tensor,
    2.46 MB at FP16, with the resize done by a shader on the capture adapter.
    Both rows end with a CUDA tensor imported from a shared D3D12 heap, so what
    differs between them is the ordering and the kernel, not the interop.
    """
    sys.path.insert(0, str(REPO / "examples"))
    import rapidshot
    from rapidshot import native
    from gpu_tensor_to_cupy import CudaTensor

    if rapidshot.GpuConverter is None:
        raise PathUnavailable("the native extension is not available")

    cam = rapidshot.create()
    state = {"converter": None, "transfer": None, "view": None, "reference": None}

    class TensorSource:
        cuda_handle_type = 4  # CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP
        cuda_dedicated = False

        def __init__(self, transfer):
            self._transfer = transfer  # Own the heap for the import's lifetime.
            self.shared_output_handle = transfer.shared_destination_handle
            self.output_byte_size = transfer.total_bytes

    def produce():
        frame = cam.grab_frame()
        if frame is None:
            return None
        source = None
        with frame:
            if state["converter"] is None:
                converter = rapidshot.GpuConverter(
                    frame, (OUT, OUT), dtype=TARGET_DTYPE, layout="nchw",
                    normalize=True)
                transfer = rapidshot.TensorTransfer(converter)
                _validate_tensor_transfer(transfer, np)
                state.update(converter=converter, transfer=transfer)
                # CudaTensor maps float32 words; the result is viewed back to
                # its own dtype below, as the whole-frame path views uint8.
                state["view"] = CudaTensor(
                    TensorSource(transfer), (transfer.total_bytes // 4,), device=0)
            state["converter"].process(frame)
            state["transfer"].transfer()
            if verify:
                # The converted tensor cannot yield the source pixels the
                # reference needs, so the same frame is read a second way.
                # Untimed: verification runs in its own pass.
                if state["reference"] is None:
                    reference = native.cross_adapter_transfer(frame)
                    _validate_transfer(reference)
                    state["reference"] = reference
                raw = state["reference"].transfer_with_reference(frame)
                source = _pitched_bgra(
                    np.frombuffer(raw, dtype=np.uint8), state["reference"]).copy()
        tensor = state["view"].array.view(cp.dtype(TARGET_DTYPE)).reshape(TARGET_SHAPE)
        return Sample(tensor, source)

    def teardown():
        try:
            if state["view"] is not None:
                state["view"].close()
        finally:
            state.update(view=None, transfer=None, converter=None, reference=None)
            cam.release()

    return produce, teardown, {
        "h2d_bytes_per_frame": 0,
        "note": "GpuConverter then TensorTransfer; the converted payload crosses, not the frame"}


def _adapter_rapidshot_converter(cp, np, verify=False):
    """2.6's one-call export: convert, and hand CUDA the tensor where it is.

    No transfer at all, which is only possible when capture and CUDA are the
    same adapter. On a hybrid laptop they are not - capture is on the iGPU -
    and `to_cupy()` refuses with `CrossAdapterRequired`. That is the design,
    not a fault, so it is reported as a skip.
    """
    sys.path.insert(0, str(REPO))
    import rapidshot
    from rapidshot import native

    if rapidshot.GpuConverter is None:
        raise PathUnavailable("the native extension is not available")

    cam = rapidshot.create()
    state = {"converter": None, "reference": None}

    def produce():
        frame = cam.grab_frame()
        if frame is None:
            return None
        source = None
        with frame:
            if state["converter"] is None:
                state["converter"] = rapidshot.GpuConverter(
                    frame, (OUT, OUT), dtype=TARGET_DTYPE, layout="nchw",
                    normalize=True)
            tensor = state["converter"].process(frame)
            try:
                array = tensor.to_cupy()
            except rapidshot.CrossAdapterRequired as exc:
                raise PathUnavailable(
                    f"capture adapter has no CUDA device: {exc}") from None
            if verify:
                if state["reference"] is None:
                    reference = native.cross_adapter_transfer(frame)
                    _validate_transfer(reference)
                    state["reference"] = reference
                raw = state["reference"].transfer_with_reference(frame)
                source = _pitched_bgra(
                    np.frombuffer(raw, dtype=np.uint8), state["reference"]).copy()
        return Sample(array.reshape(TARGET_SHAPE), source)

    def teardown():
        state.update(converter=None, reference=None)
        cam.release()

    return produce, teardown, {
        "h2d_bytes_per_frame": 0,
        "note": "GpuConverter only; requires capture and CUDA on one adapter"}


ADAPTERS = dict(zip(PATHS, (_adapter_mss, _adapter_dxcam, _adapter_rapidshot_cpu,
                          _adapter_rapidshot_cupy, _adapter_rapidshot_xadapter,
                          _adapter_rapidshot_xadapter_async,
                          _adapter_rapidshot_converter_xadapter,
                          _adapter_rapidshot_converter)))


def _cleanup(result, teardown, sync):
    if teardown is None:
        return
    errors = []
    for label, action in (("synchronize", sync), ("teardown", teardown)):
        try:
            action()
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    if errors:
        result["cleanup_errors"] = errors
        result.setdefault("error", "adapter cleanup failed")


def _validate_options(seconds, warmup):
    if not math.isfinite(seconds) or seconds <= 0 or warmup < 0:
        raise ValueError("seconds must be finite and positive; warmup must be nonnegative")


def run_path(path: str, seconds: float, warmup: int) -> dict:
    result = {"path": path}
    teardown = None
    sync = lambda: None
    try:
        _validate_options(seconds, warmup)
        sys.path.insert(0, str(REPO))
        stage("imports", path=path)
        import cupy as cp
        import numpy as np

        sync = cp.cuda.runtime.deviceSynchronize
        stage("adapter-starting", path=path)
        produce, teardown, meta = ADAPTERS[path](cp, np)
        result.update(meta)
        proc = None
        try:
            import psutil
            proc = psutil.Process()
        except ImportError:
            pass
        stage("warmup", path=path)
        got = 0
        deadline = time.perf_counter() + 20.0
        while got < warmup and time.perf_counter() < deadline:
            sample = produce()
            if sample is not None:
                sync()
                got += 1
        if warmup and got < warmup:
            raise RuntimeError(f"warmup incomplete: {got}/{warmup} frames")
        sync()
        cpu0 = proc.cpu_times() if proc else None
        rss0 = proc.memory_info().rss if proc else 0
        samples, misses = [], 0
        stage("measurement", path=path)
        start = time.perf_counter()
        end = start + seconds
        while time.perf_counter() < end:
            t0 = time.perf_counter()
            sample = produce()
            if sample is None:
                misses += 1
                continue
            sync()
            samples.append((time.perf_counter() - t0) * 1000.0)
        wall = time.perf_counter() - start
        cpu1 = proc.cpu_times() if proc else None
        if not samples:
            raise RuntimeError("no frames during the measurement window")
        samples.sort()

        def pct(p):
            return samples[min(int(len(samples) * p), len(samples) - 1)]

        result.update(frames=len(samples), misses=misses,
                      elapsed_seconds=wall, requested_seconds=seconds,
                      fps=round(len(samples) / wall, 1), ms_min=round(samples[0], 3),
                      ms_p50=round(pct(0.5), 3), ms_p95=round(pct(0.95), 3),
                      ms_p99=round(pct(0.99), 3), ms_max=round(samples[-1], 3),
                      ms_jitter_stdev=round(statistics.pstdev(samples), 3))
        if cpu0 and cpu1:
            used = cpu1.user - cpu0.user + cpu1.system - cpu0.system
            result.update(cpu_seconds=round(used, 3), cpu_percent=round(100 * used / wall, 1))
        if proc:
            rss = proc.memory_info().rss
            result.update(rss_mb=round(rss / 1e6, 1), rss_growth_mb=round((rss - rss0) / 1e6, 1))
        try:
            free, total = cp.cuda.runtime.memGetInfo()
            result["vram_used_mb"] = round((total - free) / 1e6, 1)
        except Exception as exc:
            result["memory_stats_error"] = str(exc)
    except PathUnavailable as exc:
        result["skipped"] = str(exc)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup(result, teardown, sync)
    return result


def verify_path(path: str) -> dict:
    result = {"path": path}
    teardown = None
    sync = lambda: None
    try:
        sys.path.insert(0, str(REPO))
        stage("verification-imports", path=path)
        import cupy as cp
        import numpy as np

        sync = cp.cuda.runtime.deviceSynchronize
        stage("verification-adapter", path=path)
        produce, teardown, meta = ADAPTERS[path](cp, np, verify=True)
        result.update(meta)
        sample = None
        deadline = time.perf_counter() + 20.0
        while sample is None and time.perf_counter() < deadline:
            sample = produce()
        sync()
        if sample is None:
            raise RuntimeError("no frame for verification")
        if sample.reference is None:
            raise RuntimeError("adapter did not provide a same-frame reference")
        result.update(validate_tensor(cp.asnumpy(sample.tensor), sample.reference, np))
    except PathUnavailable as exc:
        result["skipped"] = str(exc)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _cleanup(result, teardown, sync)
    return result


class MotionError(RuntimeError):
    pass


def stop_process(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


class RunLogs:
    def __init__(self, directory=None, out=None):
        base = Path(directory) if directory else (out.parent / (out.stem + ".logs") if out else None)
        if base is not None:
            base.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="rapidshot-ai-", dir=base))
        self.event("parent-starting", argv=sys.argv)

    def event(self, event, **fields):
        record = {"event": event, "pid": os.getpid(), "time": time.time(), **fields}
        with (self.directory / "parent.log").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _child_options():
    return {"env": {**os.environ, "PYTHONIOENCODING": "utf-8"},
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}


class MotionSource:
    def __init__(self, logs, fps=0.0, startup_timeout=20.0):
        self.logs, self.fps, self.startup_timeout = logs, fps, startup_timeout
        self.proc = self.reader = self.stdout = self.stderr = None
        self.ready = False
        self.pending = ""
        self.rates = []
        self.last_progress = time.monotonic()

    def start(self):
        self.logs.event("motion-starting", fps_limit=self.fps)
        self.stdout = (self.logs.directory / "motion.stdout.log").open("wb")
        self.stderr = (self.logs.directory / "motion.stderr.log").open("wb")
        self.reader = (self.logs.directory / "motion.stdout.log").open("r", encoding="utf-8")
        self.proc = subprocess.Popen(
            [sys.executable, "-u", str(HERE / "motion_source.py"),
             "--parent-controlled", "--fps", str(self.fps)],
            stdin=subprocess.PIPE, stdout=self.stdout, stderr=self.stderr, **_child_options())
        self.logs.event("motion-launched", child_pid=self.proc.pid)
        deadline = time.monotonic() + self.startup_timeout
        while True:
            self.check()
            if self.ready:
                return
            if time.monotonic() >= deadline:
                raise MotionError("motion source did not become ready; see motion logs")
            time.sleep(0.05)

    def _read(self):
        if self.reader is None:
            return
        self.pending += self.reader.read()
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MotionError("invalid motion telemetry") from exc
            if not isinstance(event, dict):
                raise MotionError("invalid motion telemetry record")
            if event.get("event") == "error":
                raise MotionError(f"motion source: {event.get('error')}")
            if event.get("event") in ("ready", "rate"):
                self.last_progress = time.monotonic()
            if event.get("event") == "ready":
                self.ready = True
            if event.get("event") == "rate":
                rate = event.get("updates_per_second")
                if not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0:
                    raise MotionError("motion source reported an invalid rate")
                self.rates.append(rate)
                self.logs.event("motion-rate", updates_per_second=rate)

    def check(self):
        self._read()
        if self.proc is None or self.proc.poll() is not None:
            status = self.proc.returncode if self.proc is not None else "not started"
            raise MotionError(f"motion source exited ({status}); see motion logs")
        if self.ready and time.monotonic() - self.last_progress > 30:
            raise MotionError("motion source stopped reporting progress")

    def summary(self):
        return {"fps_limit": self.fps, "rate_samples": list(self.rates),
                "minimum_updates_per_second": min(self.rates) if self.rates else None,
                "rate_observed": bool(self.rates)}

    def close(self):
        try:
            if self.proc is not None:
                if self.proc.stdin is not None:
                    try:
                        self.proc.stdin.close()
                    except OSError:
                        pass
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    stop_process(self.proc)
        finally:
            for stream in (self.reader, self.stdout, self.stderr):
                if stream is not None:
                    stream.close()


def _worker_result(path, verify, returncode, stdout, stderr):
    try:
        row = json.loads(stdout.strip().splitlines()[-1])
        if not isinstance(row, dict) or row.get("path") != path:
            raise ValueError("wrong worker result shape or path")
    except (ValueError, IndexError) as exc:
        return {"path": path, "error": f"invalid worker result: {exc}", "returncode": returncode}
    if returncode:
        row.setdefault("error", f"worker exited with status {returncode}")
    if "error" not in row:
        if verify:
            valid = (row.get("verified") is True and row.get("dtype_ok") is True
                     and row.get("content_ok") is True and row.get("shape_ok") is True
                     and row.get("range_ok") is True)
        else:
            positive = ("frames", "fps", "elapsed_seconds")
            timings = ("ms_p50", "ms_p95", "ms_p99")
            valid = all(isinstance(row.get(k), (int, float)) and not isinstance(row[k], bool)
                        and math.isfinite(row[k]) and row[k] >= 0 for k in positive + timings)
            valid = valid and all(row[k] > 0 for k in positive)
        if not valid:
            row["error"] = "worker result is missing valid measurements or verification"
    row["returncode"] = returncode
    return row


def spawn(path: str, seconds: float, warmup: int, verify: bool, *, logs=None,
          motion=None, index=0, command=None, result_parser=None,
          dtype=None) -> dict:
    logs = logs if logs is not None else RunLogs()
    stem = f"{index:02d}-{path}"
    stdout_path = logs.directory / (stem + ".stdout.log")
    stderr_path = logs.directory / (stem + ".stderr.log")
    cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--call-duration",
           "--worker", path, "--seconds", str(seconds), "--warmup", str(warmup)]
    # A worker is a fresh process, so the dtype has to travel with it or the
    # child silently measures the module default while the parent reports the
    # requested one.
    if dtype:
        cmd += ["--dtype", dtype]
    if verify:
        cmd.append("--verify")
    if command is not None:
        cmd = command
    proc, failure = None, None
    logs.event("worker-starting", path=path, command=cmd)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            if motion:
                motion.check()
            proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, **_child_options())
            logs.event("worker-launched", path=path, child_pid=proc.pid)
            deadline = time.monotonic() + max(180.0, seconds * 8)
            while proc.poll() is None:
                if motion:
                    motion.check()
                if time.monotonic() >= deadline:
                    raise TimeoutError("worker exceeded its time limit")
                time.sleep(0.05)
            if motion:
                motion.check()
        except Exception as exc:
            failure = {"path": path, "error": f"{type(exc).__name__}: {exc}"}
            if isinstance(exc, MotionError):
                failure["motion_failed"] = True
        finally:
            stop_process(proc)
    row = failure if failure is not None else (result_parser or _worker_result)(
        path, verify, proc.returncode, stdout_path.read_text(encoding="utf-8", errors="replace"),
        stderr_path.read_text(encoding="utf-8", errors="replace"))
    row.update(stdout_log=str(stdout_path), stderr_log=str(stderr_path))
    logs.event("worker-finished", path=path, error=row.get("error"))
    return row


def save_results(path, payload):
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main(argv=None) -> int:
    global TARGET_DTYPE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worker", choices=PATHS)
    parser.add_argument("--dtype", choices=DTYPES, default=TARGET_DTYPE,
                        help="tensor element type (default: %(default)s)")
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=list(PATHS))
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--with-motion", action="store_true")
    parser.add_argument("--motion-fps", type=float, default=0.0,
                        help="optional animation cap; 0 is uncapped")
    parser.add_argument("--log-dir", type=Path, help="parent directory for unique run logs")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    TARGET_DTYPE = args.dtype
    try:
        _validate_options(args.seconds, args.warmup)
        if not math.isfinite(args.motion_fps) or args.motion_fps < 0:
            raise ValueError("motion-fps must be finite and nonnegative")
    except ValueError as exc:
        parser.error(str(exc))
    if args.worker:
        row = (verify_path(args.worker) if args.verify
               else run_path(args.worker, args.seconds, args.warmup))
        print(json.dumps(row, allow_nan=False), flush=True)
        return int("error" in row)

    logs = RunLogs(args.log_dir, args.out)
    print(f"Diagnostics: {logs.directory}", flush=True)
    print(f"AI ingestion: full-image bilinear -> {TARGET_SHAPE} {TARGET_DTYPE} RGB",
          flush=True)
    rows = []
    payload = {"schema_version": 3,
               "target": {"shape": list(TARGET_SHAPE), "dtype": TARGET_DTYPE,
                          "resize": "bilinear-half-pixel"},
               "results": rows, "logs": str(logs.directory)}
    motion = MotionSource(logs, args.motion_fps) if args.with_motion else None
    interrupted = False
    try:
        if motion:
            motion.start()
        for index, path in enumerate(args.paths):
            print(f"  [{path}] ...", flush=True)
            row = spawn(path, args.seconds, args.warmup, args.verify,
                        logs=logs, motion=motion, index=index, dtype=args.dtype)
            rows.append(row)
            if motion:
                payload["motion"] = motion.summary()
            save_results(args.out, payload)
            if "skipped" in row:
                print(f"    SKIPPED: {row['skipped']}", flush=True)
            elif "error" in row:
                print(f"    ERROR: {row['error']}", flush=True)
            elif args.verify:
                print(f"    verified; max pixel error {row['max_abs_error']:.6f}", flush=True)
            else:
                print(f"    {row['fps']:.1f} fps; p50 {row['ms_p50']:.3f} ms; "
                      f"elapsed {row['elapsed_seconds']:.3f}s", flush=True)
            if row.get("motion_failed"):
                raise MotionError(row["error"])
    except KeyboardInterrupt:
        payload["error"] = "interrupted"
        interrupted = True
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if motion:
            try:
                motion.close()
            except Exception as exc:
                payload["cleanup_error"] = str(exc)
                payload.setdefault("error", "motion cleanup failed")
            payload["motion"] = motion.summary()
    if "error" in payload:
        print(f"ERROR: {payload['error']}", file=sys.stderr, flush=True)
    save_results(args.out, payload)
    failed = "error" in payload or any("error" in row for row in rows)
    logs.event("parent-finished", failed=failed, completed_paths=len(rows))
    if motion:
        rate = payload["motion"]["minimum_updates_per_second"]
        print(f"Motion minimum reported rate: {rate if rate is not None else 'not observed'}",
              flush=True)
    if args.out:
        print(f"Wrote {args.out}", flush=True)
    return 130 if interrupted else int(failed)


if __name__ == "__main__":
    # This file is two benchmarks. `section7.py`'s pixel-age harness is the
    # default, because pixel age is what ROADMAP § 7.0 actually asks for. The
    # call-duration harness in `main()` above stays reachable behind a flag --
    # without it `main()`, `run_path()` and `ADAPTERS` are dead from the
    # command line, including for the workers `spawn()` starts.
    if "--call-duration" in sys.argv:
        sys.argv.remove("--call-duration")
        raise SystemExit(main())
    from section7 import main as section7_main
    raise SystemExit(section7_main("ingestion"))
