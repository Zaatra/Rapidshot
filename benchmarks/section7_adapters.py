"""Capture adapters with explicit per-frame resource ownership."""
from dataclasses import dataclass
from pathlib import Path
import sys
import time

from benchmark_contract import (canonical_rgb, normalized_tensor, decode_marker,
                                pipeline_rgb, CELL, MARKER_BITS, MARKER_HEIGHT, OUT)

PATHS = ("mss", "dxcam", "dxcam-wgc", "rapidshot-cpu", "rapidshot-cupy",
         "rapidshot-xadapter", "rapidshot-xadapter-async", "rapidshot-xadapter-semaphore",
         "rapidshot-direct", "rapidshot-converter-xadapter", "rapidshot-converter")
CPU_PATHS = PATHS[:4]

#: The frame-ID marker is 48 cells of 8 px read from row 8, so it only decodes
#: at 1:1. 384x16 BGRA is ~24 kB.
MARKER_CROP = (0, 0, MARKER_BITS * CELL, MARKER_HEIGHT)


@dataclass
class Captured:
    tensor: object
    frame_id: object
    stages: dict
    h2d_bytes: int
    reference: object = None
    rgb: object = None


class Adapter:
    def __init__(self, path, cp, np, verify=False, agent=False):
        self.path, self.cp, self.np, self.verify, self.agent = path, cp, np, verify, agent
        self.cam = self.view = self.transfer = self.sem = self.pre = None
        self.converter = self.marker_converter = None
        self.tensor_transfer = self.marker_transfer = self.marker_view = None
        self.reference_transfer = None
        self.closed = False
        if agent and path not in CPU_PATHS:
            raise ValueError("agent benchmark uses CPU capture paths")
        if path == "mss":
            import mss
            self.cam = mss.mss()
            try:
                self.monitor = self.cam.monitors[1]
            except BaseException:
                self.close()
                raise
        elif path in ("dxcam", "dxcam-wgc"):
            import dxcam
            # DXcam names this backend "winrt", not "wgc" -- its supported set is
            # {dxgi, winrt}. The path keeps the "wgc" label because Windows
            # Graphics Capture is what the API is called everywhere else, but
            # the argument has to use DXcam's spelling or create() raises
            # ValueError and the worker dies with no adapter.
            self.cam = dxcam.create(output_color="BGRA",
                                    **({"backend": "winrt"} if path.endswith("wgc") else {}))
        else:
            import rapidshot
            self.cam = rapidshot.create(output_color="BGRA", nvidia_gpu=path == "rapidshot-cupy")

    def sync(self):
        if self.cp is not None:
            self.cp.cuda.runtime.deviceSynchronize()

    def capture(self):
        from ai_ingestion import _pitched_bgra, _validate_transfer
        cp, np = self.cp, self.np
        stages, h2d, reference = {}, 0, None
        fence_events = None
        frame = None
        t0 = time.perf_counter()
        if self.path == "mss":
            shot = self.cam.grab(self.monitor)
            raw = np.frombuffer(shot.raw, np.uint8).reshape(shot.height, shot.width, 4)
        elif self.path in CPU_PATHS or self.path == "rapidshot-cupy":
            frame = self.cam.grab()
            if frame is None:
                return None
            raw = getattr(frame, "array", frame)
        else:
            frame = self.cam.grab_frame()
            if frame is None:
                return None
            raw = None
        stages["capture_call_ms"] = (time.perf_counter() - t0) * 1000
        try:
            if raw is None:
                sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
                from gpu_tensor_to_cupy import CudaTensor
                from rapidshot import native
                if self.path.startswith("rapidshot-converter"):
                    return self._converter_capture(frame, stages, CudaTensor)
                t0 = time.perf_counter()
                if self.path == "rapidshot-direct":
                    if self.pre is None:
                        # Preserve native shader behavior: source-size D3D conversion,
                        # followed by the shared canonical CUDA bilinear contract.
                        self.pre = native.GpuPreprocessor12(frame, frame.width, frame.height)
                        self.view = CudaTensor(self.pre, (1, 3, frame.height, frame.width))
                    self.pre.process(frame)
                    stages["d3d_preprocess_and_fence_ms"] = (time.perf_counter() - t0) * 1000
                    rgb = cp.rint(self.view.array[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(cp.uint8)
                    raw = cp.empty((*rgb.shape[:2], 4), cp.uint8)
                    raw[:, :, :3], raw[:, :, 3] = rgb[:, :, ::-1], 255
                else:
                    if self.transfer is None:
                        from types import SimpleNamespace
                        self.transfer = native.cross_adapter_transfer(frame)
                        _validate_transfer(self.transfer)
                        owner = SimpleNamespace(shared_output_handle=self.transfer.shared_destination_handle,
                            output_byte_size=self.transfer.total_bytes, transfer=self.transfer,
                            cuda_handle_type=4, cuda_dedicated=False)
                        self.view = CudaTensor(owner, (self.transfer.total_bytes // 4,), device=0)
                        if self.path.endswith("semaphore"):
                            from cuda_semaphore import CudaFence
                            self.sem = CudaFence(self.transfer, cp)
                    t0 = time.perf_counter()
                    if self.verify:
                        value = self.transfer.transfer_async_with_reference(frame)
                    elif self.path == "rapidshot-xadapter":
                        self.transfer.transfer(frame)
                        value = None
                    else:
                        value = self.transfer.transfer_async(frame)
                    stages["transfer_submit_or_block_ms"] = (time.perf_counter() - t0) * 1000
                    if value is not None:
                        t0 = time.perf_counter()
                        if self.sem is not None:
                            begin, end = cp.cuda.Event(), cp.cuda.Event()
                            begin.record()
                            self.sem.wait(value)
                            end.record()
                            fence_events = (begin, end)
                        else:
                            self.transfer.wait_shared_fence(value)
                        stages["fence_completion_wall_ms"] = (time.perf_counter() - t0) * 1000
                    raw = _pitched_bgra(self.view.array.view(cp.uint8), self.transfer)
                    if self.verify:
                        reference = _pitched_bgra(np.frombuffer(self.transfer.read_back_source(), np.uint8), self.transfer).copy()
            xp = np if self.path in CPU_PATHS else cp
            if self.verify and reference is None:
                reference = np.array(raw, copy=True) if xp is np else cp.asnumpy(raw)
            t0 = time.perf_counter()
            frame_id = decode_marker(raw, xp) if xp is np else None
            stages["marker_decode_ms"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            if self.agent:
                rgb = np.ascontiguousarray(raw[:, :, 2::-1])
                return Captured(None, frame_id, stages, 0, reference, rgb)
            gpu_begin = gpu_end = None
            if xp is cp:
                gpu_begin, gpu_end = cp.cuda.Event(), cp.cuda.Event()
                gpu_begin.record()
            # Idiomatic per backend; `canonical_rgb` stays the reference that
            # verification compares against. See pipeline_rgb's docstring.
            rgb = pipeline_rgb(raw, xp)
            if xp is np:
                h2d = rgb.nbytes
                tensor = normalized_tensor(cp.asarray(rgb), cp)
            else:
                tensor = normalized_tensor(rgb, cp)
                if self.path == "rapidshot-cupy":
                    h2d = raw.nbytes
            if gpu_end is not None:
                gpu_end.record()
            self.sync()
            stages["preprocess_completion_ms"] = (time.perf_counter() - t0) * 1000
            if gpu_end is not None:
                stages["gpu_preprocess_ms"] = cp.cuda.get_elapsed_time(gpu_begin, gpu_end)
            if fence_events is not None:
                stages["gpu_fence_wait_ms"] = cp.cuda.get_elapsed_time(*fence_events)
            if xp is cp:
                t0 = time.perf_counter()
                frame_id = decode_marker(raw, xp)
                stages["marker_decode_ms"] = (time.perf_counter() - t0) * 1000
            return Captured(tensor, frame_id, stages, h2d, reference)
        finally:
            self.sync()
            if frame is not None and hasattr(frame, "release"):
                frame.release()

    def _heap_view(self, transfer, CudaTensor):
        """A CUDA view of a TensorTransfer's destination heap."""
        from types import SimpleNamespace
        owner = SimpleNamespace(
            shared_output_handle=transfer.shared_destination_handle,
            output_byte_size=transfer.total_bytes, transfer=transfer,
            cuda_handle_type=4, cuda_dedicated=False)
        return CudaTensor(owner, (transfer.total_bytes // 4,), device=0)

    def _converter_capture(self, frame, stages, CudaTensor):
        """2.6's convert-on-the-capture-adapter path, in pixel-age terms.

        Every other path here carries the whole frame to wherever the tensor is
        built, so the frame-ID marker arrives for free. This one deliberately
        does not -- the resize happens *before* anything crosses, and a
        640-square resize destroys a marker whose cells are 8 px wide and read
        from row 8. So the marker travels as its own 1:1 uint8 crop, ~24 kB,
        converted in the same dispatch style and moved beside the tensor.

        That second conversion is a real cost of this ordering and is left in
        the timings rather than subtracted. It is also small: 24 kB against the
        2.46 MB tensor, and against the 16.38 MB whole frame the other paths
        move to get the same marker.
        """
        import rapidshot
        from rapidshot import native
        from ai_ingestion import _pitched_bgra, _validate_transfer
        cp, np = self.cp, self.np
        crossing = self.path.endswith("xadapter")

        if self.converter is None:
            self.converter = rapidshot.GpuConverter(
                frame, (OUT, OUT), dtype="float16", layout="nchw", normalize=True)
            self.marker_converter = rapidshot.GpuConverter(
                frame, (MARKER_CROP[2], MARKER_CROP[3]), dtype="uint8",
                layout="nhwc", crop=MARKER_CROP, sampling="nearest")
            if crossing:
                self.tensor_transfer = rapidshot.TensorTransfer(self.converter)
                self.marker_transfer = rapidshot.TensorTransfer(self.marker_converter)
                self.view = self._heap_view(self.tensor_transfer, CudaTensor)
                self.marker_view = self._heap_view(self.marker_transfer, CudaTensor)

        t0 = time.perf_counter()
        tensor_handle = self.converter.process(frame)
        marker_handle = self.marker_converter.process(frame)
        stages["d3d_convert_ms"] = (time.perf_counter() - t0) * 1000

        if crossing:
            t0 = time.perf_counter()
            self.tensor_transfer.transfer()
            self.marker_transfer.transfer()
            stages["transfer_submit_or_block_ms"] = (time.perf_counter() - t0) * 1000
            tensor = self.view.array.view(cp.float16).reshape(1, 3, OUT, OUT)
            marker = self.marker_view.array.view(cp.uint8).reshape(
                MARKER_CROP[3], MARKER_CROP[2], 4)
        else:
            # No transfer at all: only possible when capture and CUDA are the
            # same adapter. `to_cupy()` raises CrossAdapterRequired otherwise,
            # which section7's worker already records as unavailable.
            tensor = tensor_handle.to_cupy().reshape(1, 3, OUT, OUT)
            marker = marker_handle.to_cupy()[0]

        reference = None
        if self.verify:
            # The converted tensor cannot yield the source pixels the reference
            # needs, so the same frame is read a second way. Untimed.
            if self.reference_transfer is None:
                self.reference_transfer = native.cross_adapter_transfer(frame)
                _validate_transfer(self.reference_transfer)
            raw = self.reference_transfer.transfer_with_reference(frame)
            reference = _pitched_bgra(
                np.frombuffer(raw, dtype=np.uint8), self.reference_transfer).copy()

        t0 = time.perf_counter()
        frame_id = decode_marker(marker, cp)
        stages["marker_decode_ms"] = (time.perf_counter() - t0) * 1000
        self.sync()
        return Captured(tensor, frame_id, stages, 0, reference)

    def close(self):
        if self.closed:
            return
        self.sync()
        errors = []
        for resource in (self.sem, self.view, self.marker_view, self.cam):
            if resource is None:
                continue
            try:
                close = getattr(resource, "close", None) or getattr(resource, "release", None) or getattr(resource, "stop", None)
                if close is None:
                    raise RuntimeError("capture resource has no cleanup method")
                close()
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))
        self.closed = True
        self.sem = self.view = self.transfer = self.pre = self.cam = None
        self.marker_view = self.converter = self.marker_converter = None
        self.tensor_transfer = self.marker_transfer = self.reference_transfer = None
