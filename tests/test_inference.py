"""The inference half of section 7: `ai_pipeline.Inference`.

`accepts_input_shape`, `symbolic_overrides` and `enable_cuda_dlls` are covered
in `test_section7.py`. What was not covered is everything that decides whether
a measured inference time *means* anything -- and this is a benchmark, so a
session that quietly runs half the graph on the CPU does not produce a slower
number, it produces a **wrong** one that looks fine.

Three properties carry that weight:

* **CPU fallback is banned unless the caller says otherwise.** ORT lists
  `CPUExecutionProvider` as registered even when every node is on CUDA, so the
  provider list cannot prove placement. `session.disable_cpu_ep_fallback` is
  what does, and `allow_cpu_nodes` has to both lift it *and* be recorded.
* **The symbolic axes are pinned in the session, not in the file.** Rewriting
  `yolo11n.onnx` to pin them would mean benchmarking something other than the
  published model, which is the entire reason for using it.
* **`run()` binds CuPy's device pointer**, so nothing is copied and the
  measurement is of inference rather than of a transfer.

`onnxruntime` is not installed on every machine this runs on, and requiring it
would mean these never execute where they are most needed. It is faked --
narrowly, recording what it was asked to do -- which is also the only way to
assert that a *refusal* happens, since a real session that fails to place a
node raises from inside ORT rather than from this code.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import ai_pipeline


# -- fakes ---------------------------------------------------------------


class FakeOptions:
    def __init__(self):
        self.config, self.overrides = {}, {}
        self.log_severity_level = None

    def add_session_config_entry(self, key, value):
        self.config[key] = value

    def add_free_dimension_override_by_name(self, name, size):
        self.overrides[name] = size


class FakeBinding:
    def __init__(self):
        self.inputs, self.outputs = [], []

    def bind_input(self, name, device, device_id, dtype, shape, ptr):
        self.inputs.append({"name": name, "device": device, "device_id": device_id,
                            "dtype": dtype, "shape": tuple(shape), "ptr": ptr})

    def bind_output(self, name, device, device_id):
        self.outputs.append({"name": name, "device": device, "device_id": device_id})


class FakeSession:
    #: Set per test before construction.
    inputs = [SimpleNamespace(name="images", shape=[1, 3, 640, 640],
                              type="tensor(float)")]
    outputs = [SimpleNamespace(name="output0")]
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    def __init__(self, path, options, providers):
        self.path, self.options, self.requested = path, options, providers
        self.binding = None
        FakeSession.last = self

    def get_providers(self):
        return list(self.providers)

    def get_inputs(self):
        return list(self.inputs)

    def get_outputs(self):
        return list(self.outputs)

    def io_binding(self):
        self.binding = FakeBinding()
        return self.binding

    def run_with_iobinding(self, binding):
        EVENTS.append("run")


EVENTS = []


class FakeArray:
    """Just enough CuPy array for `run()`: a dtype cast and a device pointer."""

    def __init__(self, dtype=np.float32, ptr=0xDEADBEEF, shape=(1, 3, 640, 640)):
        self.dtype, self.shape = dtype, shape
        self.data = SimpleNamespace(ptr=ptr)
        self.casts = []

    def astype(self, dtype, copy=True):
        self.casts.append((dtype, copy))
        if dtype == self.dtype:
            return self
        cast = FakeArray(dtype, self.data.ptr + 1, self.shape)
        cast.casts = self.casts
        return cast


@pytest.fixture
def ort(monkeypatch):
    """A fake `onnxruntime`, plus a CuPy stand-in and no DLL side effects."""
    EVENTS.clear()
    FakeSession.inputs = [SimpleNamespace(name="images", shape=[1, 3, 640, 640],
                                          type="tensor(float)")]
    FakeSession.outputs = [SimpleNamespace(name="output0")]
    FakeSession.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    module = SimpleNamespace(SessionOptions=FakeOptions, InferenceSession=FakeSession)
    monkeypatch.setitem(sys.modules, "onnxruntime", module)
    monkeypatch.setattr(ai_pipeline, "enable_cuda_dlls", lambda: [])
    # Tested separately below; here the declared shape is an input, not the
    # thing under test.
    monkeypatch.setattr(ai_pipeline, "declared_input_shape",
                        lambda model: ["batch", 3, "height", "width"])
    return module


@pytest.fixture
def cp():
    stream = SimpleNamespace(ptr=0x1234)
    return SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: stream,
        runtime=SimpleNamespace(deviceSynchronize=lambda: EVENTS.append("sync"))))


def build(cp, **kwargs):
    return ai_pipeline.Inference("model.onnx", cp, np, **kwargs)


# -- placement: the property the numbers depend on -----------------------


def test_cpu_fallback_is_banned_by_default(ort, cp):
    """Without this a model ORT cannot fully place runs partly on the CPU and
    reports a time that is not GPU inference."""
    infer = build(cp)
    assert infer.session.options.config["session.disable_cpu_ep_fallback"] == "1"
    assert infer.allow_cpu_nodes is False


def test_allow_cpu_nodes_lifts_the_ban_and_says_so(ort, cp):
    """The escape hatch exists for opset 22's missing CUDA MaxPool kernel. A
    run that used it is not comparable with one that did not, so the flag has
    to survive onto the object for the caller to record."""
    infer = build(cp, allow_cpu_nodes=True)
    assert "session.disable_cpu_ep_fallback" not in infer.session.options.config
    assert infer.allow_cpu_nodes is True


def test_the_provider_list_alone_is_not_treated_as_proof(ort, cp):
    """CPUExecutionProvider is always listed. The check is only that CUDA
    loaded at all -- it must not reject a perfectly good CUDA session for
    having CPU in the list."""
    FakeSession.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert build(cp).session.get_providers()[0] == "CUDAExecutionProvider"


def test_a_session_that_fell_back_to_cpu_is_refused(ort, cp):
    FakeSession.providers = ["CPUExecutionProvider"]
    with pytest.raises(RuntimeError, match="CUDA provider not loaded"):
        build(cp)


def test_the_session_runs_on_cupys_stream(ort, cp):
    """`run()` synchronises CuPy's stream around inference, which only orders
    anything if ORT was given that same stream."""
    infer = build(cp)
    (_, options), = [(p[0], p[1]) for p in infer.session.requested]
    assert options["user_compute_stream"] == str(0x1234)
    assert options["device_id"] == 0


# -- the model contract ---------------------------------------------------


def test_symbolic_axes_reach_the_session(ort, cp):
    """Pinned in the session so the file stays byte-identical to the published
    model."""
    assert build(cp).session.options.overrides == {"batch": 1, "height": 640,
                                                   "width": 640}
    assert build(cp).overrides == {"batch": 1, "height": 640, "width": 640}


def test_a_model_with_two_inputs_is_refused(ort, cp):
    FakeSession.inputs = [
        SimpleNamespace(name="images", shape=[1, 3, 640, 640], type="tensor(float)"),
        SimpleNamespace(name="extra", shape=[1], type="tensor(float)")]
    with pytest.raises(ValueError, match="one input"):
        build(cp)


def test_an_input_that_cannot_take_640_is_refused(ort, cp):
    FakeSession.inputs = [SimpleNamespace(name="images", shape=[1, 3, 320, 320],
                                          type="tensor(float)")]
    with pytest.raises(ValueError, match="one input"):
        build(cp)


def test_a_non_float_input_is_refused(ort, cp):
    """int8 would need a quantised pipeline; silently binding float bytes to it
    would produce numbers rather than an error."""
    FakeSession.inputs = [SimpleNamespace(name="images", shape=[1, 3, 640, 640],
                                          type="tensor(int8)")]
    with pytest.raises(ValueError, match="unsupported model input"):
        build(cp)


@pytest.mark.parametrize("declared, dtype", [("tensor(float)", np.float32),
                                             ("tensor(float16)", np.float16)])
def test_the_model_decides_the_dtype(ort, cp, declared, dtype):
    FakeSession.inputs = [SimpleNamespace(name="images", shape=[1, 3, 640, 640],
                                          type=declared)]
    assert build(cp).dtype is dtype


# -- run() ----------------------------------------------------------------


def test_run_binds_the_device_pointer_rather_than_copying(ort, cp):
    """The whole path exists to keep the tensor on the GPU. Binding anything
    but its own pointer would mean the measurement included a copy."""
    infer = build(cp)
    tensor = FakeArray(np.float32, ptr=0xABCD000)
    infer.run(tensor)
    bound, = infer.session.binding.inputs
    assert bound["ptr"] == 0xABCD000
    assert bound["device"] == "cuda" and bound["device_id"] == 0
    assert bound["shape"] == (1, 3, 640, 640)
    assert bound["name"] == "images"


def test_run_does_not_copy_a_tensor_already_in_the_models_dtype(ort, cp):
    infer = build(cp)
    tensor = FakeArray(np.float32)
    infer.run(tensor)
    assert tensor.casts == [(np.float32, False)]


def test_run_casts_when_the_model_wants_float16(ort, cp):
    FakeSession.inputs = [SimpleNamespace(name="images", shape=[1, 3, 640, 640],
                                          type="tensor(float16)")]
    infer = build(cp)
    tensor = FakeArray(np.float32)
    infer.run(tensor)
    assert tensor.casts == [(np.float16, False)]
    assert infer.session.binding.inputs[0]["dtype"] is np.float16


def test_run_binds_every_output(ort, cp):
    """An unbound output makes ORT allocate and copy to the host, which would
    appear in the timing as inference."""
    FakeSession.outputs = [SimpleNamespace(name="output0"),
                           SimpleNamespace(name="output1")]
    infer = build(cp)
    infer.run(FakeArray())
    assert [o["name"] for o in infer.session.binding.outputs] == ["output0", "output1"]
    assert all(o["device"] == "cuda" for o in infer.session.binding.outputs)


def test_run_synchronises_on_both_sides_of_inference(ort, cp):
    """Before, so the tensor the converter wrote is complete; after, so the
    time measured includes the work rather than just its submission."""
    infer = build(cp)
    EVENTS.clear()
    infer.run(FakeArray())
    assert EVENTS == ["sync", "run", "sync"]


# -- declared_input_shape -------------------------------------------------


def dim(param="", value=None):
    return SimpleNamespace(dim_param=param, dim_value=value or 0,
                           HasField=lambda field: value is not None)


def fake_onnx(monkeypatch, inputs, initializers=()):
    graph = SimpleNamespace(
        initializer=[SimpleNamespace(name=n) for n in initializers],
        input=[SimpleNamespace(
            name=name,
            type=SimpleNamespace(tensor_type=SimpleNamespace(
                shape=SimpleNamespace(dim=dims))))
            for name, dims in inputs])
    module = SimpleNamespace(load=lambda path, load_external_data=False:
                             SimpleNamespace(graph=graph))
    monkeypatch.setitem(sys.modules, "onnx", module)


def test_declared_shape_reads_names_values_and_gaps(monkeypatch):
    fake_onnx(monkeypatch, [("images", [dim("batch"), dim(value=3),
                                        dim("height"), dim()])])
    assert ai_pipeline.declared_input_shape("m.onnx") == ["batch", 3, "height", None]


def test_initializers_do_not_count_as_inputs(monkeypatch):
    """Older exporters list every weight as a graph input. Counting them would
    make a single-input model look like it had dozens and be refused."""
    fake_onnx(monkeypatch,
              [("images", [dim(value=1), dim(value=3), dim(value=640), dim(value=640)]),
               ("conv.weight", [dim(value=64)])],
              initializers=["conv.weight"])
    assert ai_pipeline.declared_input_shape("m.onnx") == [1, 3, 640, 640]


def test_a_model_with_two_real_inputs_declares_nothing(monkeypatch):
    """None rather than a guess: `accepts_input_shape(None)` is False, so the
    model is refused rather than bound to whichever input came first."""
    fake_onnx(monkeypatch, [("a", [dim(value=1)]), ("b", [dim(value=1)])])
    assert ai_pipeline.declared_input_shape("m.onnx") is None
