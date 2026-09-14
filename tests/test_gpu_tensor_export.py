"""The framework exports on `GpuTensor`, and the LUID match they turn on.

`to_cupy()`, `to_torch()` and `to_dlpack()` all reduce to one decision: which
CUDA device, if any, sits on the adapter holding the tensor. Get that wrong in
the permissive direction and CUDA maps memory belonging to another GPU; get it
wrong in the strict direction and every export raises `CrossAdapterRequired`,
which on a laptop reads as ordinary Optimus behaviour rather than as a bug.

2.6 shipped the second failure. `_device_for_adapter` compared LUIDs using
CuPy's `getDeviceProperties()["luid"]`, which converts the fixed-size
`char luid[8]` field as if it were a C string and so truncates at the first
zero byte — of which a LUID almost always has several. The comparison could
never succeed on any adapter, so the exports were dead on every machine, and
the machine the feature was written on had no CUDA to notice. The worked
example in `examples/gpu_tensor_to_cupy.py` had always used `cuDeviceGetLuid`
and was correct; the library re-implementation of it was not.

`test_the_library_agrees_with_the_example` is the one that matters most here:
the two implementations diverging silently is what caused this, so they are
pinned together rather than only checked one at a time.
"""
import ctypes

import numpy as np
import pytest

import rapidshot
from rapidshot import native
from rapidshot.converter import CrossAdapterRequired, _device_for_adapter

cp = pytest.importorskip("cupy", reason="CuPy not installed")

if not native.is_available():
    pytest.skip("native extension not built", allow_module_level=True)

try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no CUDA device", allow_module_level=True)
except Exception:  # pragma: no cover - driver present but unusable
    pytest.skip("CUDA runtime unavailable", allow_module_level=True)

OUT = 64
# No adapter has this LUID, so it stands in for "a tensor somewhere CUDA
# cannot reach" without depending on the machine being hybrid.
NO_SUCH_ADAPTER = bytes([0xFF]) * 8


def driver_luid(ordinal: int) -> bytes:
    """Device `ordinal`'s adapter LUID, read the way the example reads it.

    Deliberately not routed through `_device_for_adapter`: a test for a
    comparison must not source its inputs from the thing being compared.
    """
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuInit(0)
    dev = ctypes.c_int()
    assert cuda.cuDeviceGet(ctypes.byref(dev), ordinal) == 0
    buffer = (ctypes.c_char * 8)()
    node_mask = ctypes.c_uint()
    assert cuda.cuDeviceGetLuid(buffer, ctypes.byref(node_mask), dev) == 0
    return buffer.raw[:8]


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


@pytest.fixture(scope="module")
def tensor(live_frame):
    """A converted tensor on the capture adapter."""
    converter = rapidshot.GpuConverter(live_frame, (OUT, OUT), dtype="float32")
    return converter.process(live_frame)


# -- the LUID match ------------------------------------------------------


def test_a_cuda_device_resolves_from_its_own_adapter_luid():
    """The regression test. Ask for the adapter a CUDA device is demonstrably
    on, and that device must come back — against the truncating comparison
    this raises instead, for every device on every machine."""
    for ordinal in range(cp.cuda.runtime.getDeviceCount()):
        assert _device_for_adapter(cp, driver_luid(ordinal)) == ordinal


def test_an_adapter_with_no_cuda_device_is_refused():
    """The strict direction still has to hold: silently falling back to
    device 0 would map another GPU's memory and report plausible garbage."""
    with pytest.raises(CrossAdapterRequired):
        _device_for_adapter(cp, NO_SUCH_ADAPTER)


def test_the_library_agrees_with_the_example():
    """`_CudaView` documents itself as mirroring the shipped example. Pin the
    two together: they diverging unnoticed is the whole history of this bug."""
    example = pytest.importorskip(
        "tests.test_cuda_interop", reason="example loader unavailable"
    ).load_example()

    # Only the comparison is under test, so the instance is built without
    # importing anything: __init__ would need a live preprocessor.
    view = example.CudaTensor.__new__(example.CudaTensor)
    view._cuda = ctypes.WinDLL("nvcuda.dll")

    for ordinal in range(cp.cuda.runtime.getDeviceCount()):
        luid = driver_luid(ordinal)
        assert view._device_for_adapter(luid) == _device_for_adapter(cp, luid)

    assert view._device_for_adapter(NO_SUCH_ADAPTER) is None


# -- the exports themselves ----------------------------------------------


def test_the_capture_adapter_without_cuda_is_reported_as_such(tensor):
    """On a hybrid machine capture is on the iGPU, so an export is refused —
    and must be refused with the type that tells a caller to transfer, not a
    bare RuntimeError."""
    try:
        _device_for_adapter(cp, tensor.adapter_luid)
    except CrossAdapterRequired:
        with pytest.raises(CrossAdapterRequired):
            tensor.to_cupy()
    else:
        pytest.skip("capture adapter has a CUDA device — nothing to refuse")


@pytest.fixture
def cuda_tensor(tensor):
    """`tensor`, or a skip when CUDA cannot see the capturing adapter."""
    try:
        _device_for_adapter(cp, tensor.adapter_luid)
    except CrossAdapterRequired as exc:
        pytest.skip(f"capture adapter has no CUDA device: {exc}")
    return tensor


def test_to_cupy_is_byte_equal_to_the_readback(cuda_tensor):
    """Shape and dtype would agree even if the import had mapped unrelated
    device memory. The pixels would not."""
    array = cuda_tensor.to_cupy()
    assert array.shape == cuda_tensor.shape
    assert np.array_equal(cp.asnumpy(array), cuda_tensor.numpy())


def test_to_torch_shares_the_same_memory(cuda_tensor):
    """Zero-copy: the point of the export is that no pixels move."""
    torch = pytest.importorskip("torch", reason="PyTorch not installed")

    tensor_out = cuda_tensor.to_torch()
    assert tensor_out.is_cuda
    assert tuple(tensor_out.shape) == tuple(cuda_tensor.shape)
    assert tensor_out.data_ptr() == cuda_tensor.to_cupy().data.ptr
    assert np.array_equal(tensor_out.cpu().numpy(), cuda_tensor.numpy())


def test_to_dlpack_round_trips_through_cupy(cuda_tensor):
    """DLPack is the vendor-neutral capsule, so it must carry the same bytes."""
    back = cp.from_dlpack(cuda_tensor.to_dlpack())
    assert np.array_equal(cp.asnumpy(back), cuda_tensor.numpy())
