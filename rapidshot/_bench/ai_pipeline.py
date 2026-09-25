"""Present submission -> detections an application can read.

Requires --model and --model-sha256; there is no synthetic-model fallback.

The boundary moved. This used to end at forward-pass completion and said so --
*no claim that raw model outputs are postprocessed detections*. It now ends
where `detection.postprocess` hands back boxes, classes and scores on the
host, because that is the point an application can act on them. Numbers taken
before that change are not comparable with numbers taken after it.
"""
import os
from pathlib import Path

from . import detection
import sys

_DLL_HANDLES = []


def enable_cuda_dlls():
    """Retain DLL-directory handles for the entire provider lifetime."""
    found = []
    for base in dict.fromkeys((sys.prefix, sys.base_prefix)):
        for dll in (Path(base) / "Lib" / "site-packages" / "nvidia").glob("**/*.dll"):
            if str(dll.parent) not in found:
                found.append(str(dll.parent))
    for directory in found:
        os.environ["PATH"] = directory + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLES.append(os.add_dll_directory(directory))
    return found


INPUT_SHAPE = (1, 3, 640, 640)


def accepts_input_shape(shape):
    """True if a model input declared as ``shape`` can take a 1x3x640x640 tensor.

    Symbolic dimensions are accepted. The official Ultralytics ``yolo11n.onnx``
    declares ``['batch', 3, 'height', 'width']``, and rewriting the file to pin
    them would mean benchmarking something other than the published model --
    which is the whole reason for using it. Any concrete dimension must still
    match, and the tensor bound at run time is always exactly 1x3x640x640.
    """
    if shape is None or len(shape) != len(INPUT_SHAPE):
        return False
    return all(not isinstance(d, int) or d == want for d, want in zip(shape, INPUT_SHAPE))


def symbolic_overrides(shape):
    """Map each named symbolic input axis to the size this benchmark binds.

    Given to ONNX Runtime as free-dimension overrides, so the *session* treats
    the axes as fixed while the file stays byte-identical to the published one.
    Without them ORT keeps the shape arithmetic for dynamic axes on the CPU,
    and this harness refuses CPU fallback so no model can run partly on the CPU
    and have that counted as GPU inference time.
    """
    return {d: want for d, want in zip(shape, INPUT_SHAPE) if isinstance(d, str) and d}


def declared_input_shape(model):
    """The first graph input's declared shape: ints, axis names, or None."""
    import onnx
    graph = onnx.load(str(model), load_external_data=False).graph
    initializers = {i.name for i in graph.initializer}
    inputs = [i for i in graph.input if i.name not in initializers]
    if len(inputs) != 1:
        return None
    return [d.dim_param or (d.dim_value if d.HasField("dim_value") else None)
            for d in inputs[0].type.tensor_type.shape.dim]


class Inference:
    #: Set by `detect` to "device" or "host". Recorded in every result: the two
    #: cost differently, and a table that mixed them without saying so would
    #: credit one path with a transfer it never made.
    postprocess_location = None

    def __init__(self, model, cp, np, allow_cpu_nodes=False):
        """ONNX Runtime on CUDA, bound to CuPy's stream.

        By default every node must be placed on CUDA: session creation fails
        otherwise, so no model can run partly on the CPU and have that counted
        as GPU inference. ``allow_cpu_nodes`` relaxes that for a model ORT's
        CUDA provider cannot fully place -- the published yolo11n.onnx is
        opset 22, which has no CUDA MaxPool kernel in ORT 1.30 -- and callers
        must record that they used it.
        """
        enable_cuda_dlls()
        import onnxruntime as ort
        self.cp, self.np = cp, np
        self.allow_cpu_nodes = bool(allow_cpu_nodes)
        options = ort.SessionOptions()
        options.log_severity_level = 2
        if not self.allow_cpu_nodes:
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.overrides = symbolic_overrides(declared_input_shape(model) or [])
        for name, size in self.overrides.items():
            options.add_free_dimension_override_by_name(name, size)
        self.session = ort.InferenceSession(str(model), options, providers=[
            ("CUDAExecutionProvider", {"device_id": 0,
                                       "user_compute_stream": str(cp.cuda.get_current_stream().ptr)})])
        # ORT always lists CPUExecutionProvider as registered, even when every
        # node is on CUDA, so the provider list cannot prove placement -- the
        # disabled fallback above is what does. This only rejects a session
        # where CUDA failed to load at all.
        if self.session.get_providers()[:1] != ["CUDAExecutionProvider"]:
            raise RuntimeError(f"CUDA provider not loaded: {self.session.get_providers()}")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or not accepts_input_shape(inputs[0].shape):
            raise ValueError("model must have one input that accepts 1x3x640x640, "
                             f"declared {[i.shape for i in inputs]}")
        self.input = inputs[0]
        if self.input.type not in ("tensor(float)", "tensor(float16)"):
            raise ValueError(f"unsupported model input {self.input.type}")
        self.dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32

    def run(self, tensor):
        value = tensor.astype(self.dtype, copy=False)
        self.cp.cuda.runtime.deviceSynchronize()
        binding = self.session.io_binding()
        binding.bind_input(self.input.name, "cuda", 0, self.dtype, value.shape, value.data.ptr)
        for output in self.session.get_outputs():
            binding.bind_output(output.name, "cuda", 0)
        self.session.run_with_iobinding(binding)
        self.cp.cuda.runtime.deviceSynchronize()
        return binding


    def detect(self, tensor, geometry, contract):
        """Forward pass, then detections an application could branch on.

        The boundary section 7.0's inference table stops short of. Confidence
        filtering, NMS, coordinate restoration and the transfer to the host are
        all inside this call, because "the forward pass completed" is not the
        moment an application can read a box.

        Postprocessing runs on the device when the output can be wrapped
        without copying, and falls back to the host otherwise. **Which one
        happened is recorded**, because they cost very differently and a table
        mixing them silently would attribute a transfer to whichever path
        failed to avoid it.
        """
        binding = self.run(tensor)
        raw, where = self._raw_output(binding)
        xp = self.cp if where == "device" else self.np
        to_host = self.cp.asnumpy if where == "device" else None
        detections = detection.postprocess(raw, geometry, contract, xp,
                                           to_host=to_host)
        self.postprocess_location = where
        return detections, binding

    def _raw_output(self, binding):
        """The model output, on the device if it can be reached without copying."""
        try:
            value = binding.get_outputs()[0]
            shape = tuple(value.shape())
            dtype = self.np.dtype(
                "float16" if "float16" in value.data_type() else "float32")
            size = int(self.np.prod(shape)) * dtype.itemsize
            memory = self.cp.cuda.UnownedMemory(value.data_ptr(), size, value)
            pointer = self.cp.cuda.MemoryPointer(memory, 0)
            return self.cp.ndarray(shape, dtype=dtype, memptr=pointer), "device"
        except Exception:
            # Not a failure worth stopping for: the host path produces the same
            # detections, it just pays for the whole output tensor instead of
            # for the handful of boxes that survive.
            return binding.copy_outputs_to_cpu()[0], "host"


if __name__ == "__main__":
    from .section7 import main
    raise SystemExit(main("inference"))
