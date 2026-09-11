"""Present submission -> trained-model forward-pass completion.

Requires --model and --model-sha256. No synthetic-model fallback and no claim
that raw model outputs are postprocessed detections. See ROADMAP.md section 7.0.
"""
import os
from pathlib import Path
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


class Inference:
    def __init__(self, model, cp, np):
        enable_cuda_dlls()
        import onnxruntime as ort
        self.cp, self.np = cp, np
        options = ort.SessionOptions()
        options.log_severity_level = 2
        options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        self.session = ort.InferenceSession(str(model), options, providers=[
            ("CUDAExecutionProvider", {"device_id": 0,
                                       "user_compute_stream": str(cp.cuda.get_current_stream().ptr)})])
        if self.session.get_providers() != ["CUDAExecutionProvider"]:
            raise RuntimeError(f"expected CUDA-only execution, got {self.session.get_providers()}")
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].shape != [1, 3, 640, 640]:
            raise ValueError("model must have one fixed 1x3x640x640 input")
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


if __name__ == "__main__":
    from section7 import main
    raise SystemExit(main("inference"))
