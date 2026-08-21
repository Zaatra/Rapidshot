"""The D3D12 tensor -> CUDA path, and the handle lifetime it depends on.

Two things are covered here that nothing else covers:

* **`shared_output_handle` is a raw integer**, so every lifetime mistake around
  it produces a plausible-looking number rather than an error. Nothing in the
  value distinguishes a live handle from a closed one, or from a handle Windows
  has since recycled for something unrelated.
* **`examples/gpu_tensor_to_cupy.py` is executable documentation.** ROADMAP § 5
  is explicit that anything not run before a release is not verified for that
  release, and an example verified only by someone running it by hand is in
  exactly that category. These tests import the shipped file, so the example
  itself is what gets exercised.

The D3D12 preprocessor cannot be built over a synthetic texture (ROADMAP § 2),
so everything here needs live capture. Desktop Duplication only reports changed
content, so these skip rather than fail on an idle screen -- a red suite that
means "nothing moved on screen" trains people to ignore red suites.
"""
import ctypes
import gc
import importlib.util
import weakref
from pathlib import Path

import numpy as np
import pytest

from rapidshot import native

cp = pytest.importorskip("cupy", reason="CuPy not installed")

if not native.is_available():
    pytest.skip("native extension not built", allow_module_level=True)

try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # pragma: no cover - driver present but unusable
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

import rapidshot  # noqa: E402

OUT = 64
EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "gpu_tensor_to_cupy.py"


def load_example():
    """Import the shipped example as a module, so the tests exercise *it*."""
    spec = importlib.util.spec_from_file_location("gpu_tensor_to_cupy", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def live_frame():
    """A live captured frame, or a skip. Released after the module finishes."""
    camera = rapidshot.create(output_color="BGRA")
    frame = None
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        camera.release()
        pytest.skip("no frame captured — the screen must be changing")
    yield frame
    frame.release()
    camera.release()


def handle_is_open(handle: int) -> bool:
    """Does this integer still name a kernel object we own?"""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetHandleInformation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    flags = ctypes.c_ulong()
    return bool(k32.GetHandleInformation(
        ctypes.c_void_p(handle), ctypes.byref(flags)))


# --------------------------------------------------------------------------
# the handle itself
# --------------------------------------------------------------------------

def test_handle_is_open_while_the_preprocessor_lives(live_frame):
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    assert pre.shared_output_handle != 0
    assert handle_is_open(pre.shared_output_handle)


def test_handle_is_stable_across_calls(live_frame):
    """One handle for the preprocessor's lifetime, not one per access.

    Minting per call would hand the caller something it must close at a moment
    it cannot determine, since importers reference the handle rather than
    taking ownership of it.
    """
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    first = pre.shared_output_handle
    assert [pre.shared_output_handle for _ in range(5)] == [first] * 5


def test_separate_preprocessors_get_separate_handles(live_frame):
    a = native.GpuPreprocessor12(live_frame, OUT, OUT)
    b = native.GpuPreprocessor12(live_frame, OUT, OUT)
    assert a.shared_output_handle != b.shared_output_handle


def test_byte_size_matches_the_tensor(live_frame):
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    assert pre.output_byte_size == 1 * 3 * OUT * OUT * 4
    assert pre.read_back().nbytes == pre.output_byte_size


def test_handle_closes_with_the_preprocessor(live_frame):
    """The documented contract: borrowed, not owned.

    Asserted so that anyone who later makes the handle outlive its preprocessor
    has to change this test deliberately rather than by accident.
    """
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    handle = pre.shared_output_handle
    assert handle_is_open(handle)

    del pre
    gc.collect()

    # Windows recycles handle values, so a *reopened* handle could in principle
    # make this pass for the wrong reason. Nothing else in this process opens a
    # kernel object between those two lines, which is why the check is worth
    # making here and would not be inside a busier test.
    assert not handle_is_open(handle), (
        "the shared handle outlived its preprocessor; it is documented as "
        "borrowed and closed in Drop")


# --------------------------------------------------------------------------
# the example, imported and run
# --------------------------------------------------------------------------

def test_example_imports_the_tensor_byte_exactly(live_frame):
    """The check that matters: shape and dtype would agree even if the import
    had mapped completely unrelated device memory. The pixels would not."""
    module = load_example()
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    pre.process(live_frame)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        assert isinstance(view.array, cp.ndarray)
        assert view.array.shape == (1, 3, OUT, OUT)
        assert np.array_equal(cp.asnumpy(view.array), pre.read_back())


def test_example_tensor_is_readable_by_a_cuda_kernel(live_frame):
    """Real device memory a kernel can read, not an address that merely copies
    back correctly."""
    module = load_example()
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    pre.process(live_frame)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        gpu_sum = float(view.array.sum())
        assert gpu_sum == pytest.approx(float(pre.read_back().sum()), rel=1e-5)


def test_example_tensor_sees_later_dispatches(live_frame):
    """The import is paid once. `process()` overwrites the buffer the array
    already points at, so a capture loop must not need to re-import."""
    module = load_example()
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        pre.process(live_frame, scale=1.0, bias=0.0)
        first = cp.asnumpy(view.array).copy()

        # A different normalisation must change what the same array reports,
        # through the same pointer, with no re-import.
        pre.process(live_frame, scale=0.5, bias=0.0)
        second = cp.asnumpy(view.array)

        assert not np.array_equal(first, second), (
            "the CuPy view did not observe a later dispatch")
        assert np.allclose(second, first * 0.5, atol=1e-6)


def test_example_view_keeps_its_preprocessor_alive(live_frame):
    """The lifetime bug this file exists for.

    `CudaTensor` must hold the preprocessor: it owns both the D3D12 resource
    the array addresses and the shared handle. Built from a temporary, the
    array would otherwise address freed VRAM.

    **This asserts reachability, not pixel equality, and that distinction is
    the whole point.** Measured 2026-08-06 with the ownership chain removed:
    the shared handle was closed and the D3D12 resource released, and reading
    the CuPy view *still returned byte-identical data* — the freed VRAM simply
    had not been claimed by anything else yet. A pixel comparison here passes
    with the bug present, which is precisely why this class of defect survives
    testing and reaches users as an intermittent corruption instead.
    """
    module = load_example()

    def build():
        pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
        pre.process(live_frame)
        return (module.CudaTensor(pre, (1, 3, OUT, OUT)),
                weakref.ref(pre),
                pre.read_back())

    view, pre_ref, expected = build()   # no named reference survives `build`
    gc.collect()

    assert pre_ref() is not None, (
        "nothing kept the preprocessor alive, so the CuPy array is addressing "
        "released VRAM. Note that reading it would probably still return the "
        "right pixels — do not 'fix' this by comparing arrays.")
    assert np.array_equal(cp.asnumpy(view.array), expected)
    view.close()
