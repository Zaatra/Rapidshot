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
#
# These need CUDA to be able to *see* the adapter the frame was captured on,
# which is not a given. On an Optimus laptop Desktop Duplication runs on the
# Intel iGPU while CUDA only ever sees the discrete GPU, so the import is
# refused by design and the frame has to cross adapters first (ROADMAP § 6.1).
#
# That is a supported configuration, not a failure, so it skips. Verified
# 2026-08-22 on a real Optimus system, where these four reported as failures
# until this fixture existed -- the tests were written on machines where
# capture and CUDA were always the same adapter, which every machine this
# project had run on until then happened to be.

@pytest.fixture(scope="module")
def cuda_visible_frame(live_frame):
    """`live_frame`, or a skip if CUDA cannot see the capturing adapter.

    The check runs through the shipped example rather than reimplementing the
    LUID comparison, so a change to how the example decides cannot silently
    diverge from how the tests decide.
    """
    module = load_example()
    pre = native.GpuPreprocessor12(live_frame, OUT, OUT)
    try:
        module.CudaTensor(pre, (1, 3, OUT, OUT)).close()
    except module.CrossAdapterRequired as exc:
        pytest.skip(f"capture adapter has no CUDA device: {exc}")
    return live_frame


def test_example_imports_the_tensor_byte_exactly(cuda_visible_frame):
    """The check that matters: shape and dtype would agree even if the import
    had mapped completely unrelated device memory. The pixels would not."""
    module = load_example()
    pre = native.GpuPreprocessor12(cuda_visible_frame, OUT, OUT)
    pre.process(cuda_visible_frame)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        assert isinstance(view.array, cp.ndarray)
        assert view.array.shape == (1, 3, OUT, OUT)
        assert np.array_equal(cp.asnumpy(view.array), pre.read_back())


def test_example_tensor_is_readable_by_a_cuda_kernel(cuda_visible_frame):
    """Real device memory a kernel can read, not an address that merely copies
    back correctly."""
    module = load_example()
    pre = native.GpuPreprocessor12(cuda_visible_frame, OUT, OUT)
    pre.process(cuda_visible_frame)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        gpu_sum = float(view.array.sum())
        assert gpu_sum == pytest.approx(float(pre.read_back().sum()), rel=1e-5)


def test_example_tensor_sees_later_dispatches(cuda_visible_frame):
    """The import is paid once. `process()` overwrites the buffer the array
    already points at, so a capture loop must not need to re-import."""
    module = load_example()
    pre = native.GpuPreprocessor12(cuda_visible_frame, OUT, OUT)

    with module.CudaTensor(pre, (1, 3, OUT, OUT)) as view:
        pre.process(cuda_visible_frame, scale=1.0, bias=0.0)
        first = cp.asnumpy(view.array).copy()

        # A different normalisation must change what the same array reports,
        # through the same pointer, with no re-import.
        pre.process(cuda_visible_frame, scale=0.5, bias=0.0)
        second = cp.asnumpy(view.array)

        assert not np.array_equal(first, second), (
            "the CuPy view did not observe a later dispatch")
        assert np.allclose(second, first * 0.5, atol=1e-6)


def test_example_view_keeps_its_preprocessor_alive(cuda_visible_frame):
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
        pre = native.GpuPreprocessor12(cuda_visible_frame, OUT, OUT)
        pre.process(cuda_visible_frame)
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


# --------------------------------------------------------------------------
# the zero-CPU-sync path: CUDA waits on the D3D12 fence, GPU-side
# --------------------------------------------------------------------------
#
# This is the whole cross-adapter story finishing. Capture runs on the iGPU,
# the frame crosses to the dGPU, and CUDA waits for the copy *in a stream*
# rather than on the CPU -- so nothing blocks a thread anywhere in the handoff.
#
# Measured 2026-08-22 against a real GPU consumer: the CPU-side async wait was
# worth -0.9% and +2.5% across two runs (noise), while this path was worth
# +14.2% and +7.5%. The GPU-side wait is where the gain is.
#
# Notable and undocumented by anyone: the fence is created on the **Intel**
# device and imported by CUDA on the **NVIDIA** one.

CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4


class _SemWin32(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _SemHandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _SemWin32),
                ("nvSciSyncObj", ctypes.c_void_p)]


class _SemHandleDesc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("handle", _SemHandleUnion),
                ("flags", ctypes.c_uint), ("reserved", ctypes.c_uint * 16)]


class _FenceParams(ctypes.Structure):
    _fields_ = [("value", ctypes.c_ulonglong)]


class _NvSciParams(ctypes.Union):
    _fields_ = [("fence", ctypes.c_void_p), ("reserved", ctypes.c_ulonglong)]


class _KeyedParams(ctypes.Structure):
    _fields_ = [("key", ctypes.c_ulonglong), ("timeoutMs", ctypes.c_uint)]


class _WaitInner(ctypes.Structure):
    _fields_ = [("fence", _FenceParams), ("nvSciSync", _NvSciParams),
                ("keyedMutex", _KeyedParams), ("reserved", ctypes.c_uint * 10)]


class _SemWaitParams(ctypes.Structure):
    _fields_ = [("params", _WaitInner), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


@pytest.fixture
def cross_adapter_frame(live_frame):
    """`live_frame`, or a skip when there is no second hardware adapter."""
    from rapidshot.util.topology import probe_topology
    hardware = [a for a in probe_topology().adapters if not a.is_software]
    if len(hardware) < 2:
        pytest.skip("cross-adapter transfer needs a second hardware adapter")
    return live_frame


def test_cuda_waits_on_the_d3d12_fence_gpu_side(cross_adapter_frame):
    """Capture -> transfer -> CUDA waits in a stream -> reads correct bytes.

    The wait is queued on a CUDA stream, so the correctness claim is that work
    ordered behind it observes the completed copy. A test that synchronised
    first and then read would pass even if the semaphore did nothing.
    """
    cuda = ctypes.WinDLL("nvcuda.dll")
    # CuPy's primary context must exist and be current on this thread before
    # any driver-API import, or every call returns CUDA_ERROR_INVALID_CONTEXT.
    cuda.cuInit(0)
    cp.cuda.Device(0).use()
    cp.zeros(1)
    transfer = native.cross_adapter_transfer(cross_adapter_frame)

    desc = _SemHandleDesc()
    ctypes.memset(ctypes.byref(desc), 0, ctypes.sizeof(desc))
    desc.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
    desc.handle.win32.handle = ctypes.c_void_p(transfer.shared_fence_handle)
    sem = ctypes.c_void_p()
    assert cuda.cuImportExternalSemaphore(
        ctypes.byref(sem), ctypes.byref(desc)) == 0, (
        "CUDA refused to import the D3D12 fence as an external semaphore")

    module = load_example()
    mem_desc = module.ExternalMemoryHandleDesc()
    ctypes.memset(ctypes.byref(mem_desc), 0, ctypes.sizeof(mem_desc))
    mem_desc.type = 4                       # D3D12_HEAP, not RESOURCE
    mem_desc.handle.win32.handle = ctypes.c_void_p(
        transfer.shared_destination_handle)
    mem_desc.size = transfer.total_bytes
    ext = ctypes.c_void_p()
    assert cuda.cuImportExternalMemory(
        ctypes.byref(ext), ctypes.byref(mem_desc)) == 0

    buf = module.ExternalMemoryBufferDesc()
    ctypes.memset(ctypes.byref(buf), 0, ctypes.sizeof(buf))
    buf.offset, buf.size, buf.flags = 0, transfer.total_bytes, 0
    ptr = ctypes.c_ulonglong()
    assert cuda.cuExternalMemoryGetMappedBuffer(
        ctypes.byref(ptr), ext, ctypes.byref(buf)) == 0

    try:
        memory = cp.cuda.UnownedMemory(ptr.value, transfer.total_bytes,
                                       owner=transfer)
        view = cp.ndarray((transfer.total_bytes,), dtype=cp.uint8,
                          memptr=cp.cuda.MemoryPointer(memory, 0))
        stream = cp.cuda.Stream(non_blocking=True)

        value = transfer.transfer_async(cross_adapter_frame)
        wait = _SemWaitParams()
        ctypes.memset(ctypes.byref(wait), 0, ctypes.sizeof(wait))
        wait.params.fence.value = value
        assert cuda.cuWaitExternalSemaphoresAsync(
            ctypes.byref(sem), ctypes.byref(wait), 1,
            ctypes.c_void_p(stream.ptr)) == 0

        with stream:
            total = view.sum()          # queued behind the semaphore wait
        stream.synchronize()

        expected = np.frombuffer(
            transfer.read_back_destination(), dtype=np.uint8)
        assert expected.any(), "the transfer produced an all-zero destination"
        assert int(total) == int(expected.sum())
    finally:
        cuda.cuDestroyExternalSemaphore(sem)
        cuda.cuDestroyExternalMemory(ext)
