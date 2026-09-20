"""converter.py over fake native, CuPy and CUDA objects -- no GPU.

The live converter tests need a D3D12 device, and the CUDA export needs an
NVIDIA device on the capture adapter as well, so headless coverage of this
module was 36%. Everything here is the Python layer: argument validation, what
is handed to the native converter, how results are sliced, and the CUDA
external-memory import. The pixels themselves stay with the live suite.
"""
import ctypes
import types

import numpy as np
import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.converter as converter_module  # noqa: E402
import rapidshot.native as native  # noqa: E402
from rapidshot.converter import (  # noqa: E402
    CrossAdapterRequired,
    GpuConverter,
    TensorTransfer,
)
from rapidshot.frame import Frame  # noqa: E402


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeImpl:
    """native GpuConverter12: records construction and dispatches."""

    def __init__(self, texture, width, height, **kwargs):
        self.args = dict(texture=texture, width=width, height=height, **kwargs)
        self.calls = []
        self.batch = kwargs["batch"]
        dtype = kwargs["dtype"]
        self.pixel_format = dtype if dtype in ("nv12", "p010") else None
        self.dtype = {"bgra8": "uint8", "nv12": "uint8", "p010": "uint16"}.get(dtype, dtype)
        if self.pixel_format:
            self.shape = [height * 3 // 2, width]
        elif dtype == "bgra8":
            self.shape = [self.batch, height, width, 4]
        elif kwargs["layout"] == "nhwc":
            self.shape = [self.batch, height, width, 3]
        else:
            self.shape = [self.batch, 3, height, width]
        self.sampling = kwargs["sampling"]
        self.yuv = (kwargs["matrix"], kwargs["full_range"])
        self.source_format = "B8G8R8A8_UNORM"
        itemsize = np.dtype(self.dtype).itemsize
        self.output_byte_size = int(np.prod(self.shape)) * itemsize
        self.output_resource_address = 0x10
        self.output_gpu_address = 0x20
        self.shared_output_handle = 0x30
        self.fail = None

    def process(self, texture, scale, bias, bgr, **kwargs):
        if self.fail:
            raise self.fail
        self.calls.append(dict(texture=texture, scale=scale, bias=bias, bgr=bgr, **kwargs))

    def read_back(self):
        return np.arange(self.output_byte_size // np.dtype(self.dtype).itemsize,
                         dtype=self.dtype).tobytes()

    def adapter_luid(self):
        return list(b"\x32\x34\x01\x00\x00\x00\x00\x00")


class FakeTransferImpl:
    def __init__(self, converter_impl):
        self.converter_impl = converter_impl
        self.transfers = 0
        self.source, self.destination = "Intel UHD", "NVIDIA RTX 4060"
        self.destination_is_software = 0
        self.total_bytes = 2_460_000
        self.shared_destination_handle = 0x40
        self.destination_resource_address = 0x50

    def transfer(self, converter_impl):
        assert converter_impl is self.converter_impl
        self.transfers += 1

    def read_back_destination(self):
        return self.converter_impl.read_back()

    def destination_luid(self):
        return list(range(8))


@pytest.fixture
def ext(monkeypatch):
    fake = types.SimpleNamespace(GpuConverter12=FakeImpl, TensorTransfer=FakeTransferImpl)
    monkeypatch.setattr(native, "require", lambda: fake)
    return fake


def frame(region=(0, 0, 100, 50), rotation=0, source_id=7):
    f = Frame(ctypes.c_void_p(0x1234), lambda: None, region,
              rotation_angle=rotation, source_id=source_id)
    return f


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,native_dtype,native_layout", [
    ({}, "float32", "nchw"),
    ({"dtype": "float16", "layout": "NHWC"}, "float16", "nhwc"),
    ({"dtype": "uint8", "layout": "hwc"}, "bgra8", "nhwc"),
    ({"pixel_format": "NV12"}, "nv12", None),
])
def test_construction_hands_the_native_converter_normalised_options(ext, kwargs, native_dtype, native_layout):
    conv = GpuConverter(frame(), (64, 32), **kwargs)

    args = conv._impl.args
    assert (args["texture"], args["width"], args["height"]) == (0x1234, 64, 32)
    assert (args["dtype"], args["layout"]) == (native_dtype, native_layout)
    assert conv.layout == (native_layout or "yuv420")


@pytest.mark.parametrize("kwargs,message", [
    ({"batch": 0}, "batch must be a positive integer"),
    ({"batch": True}, "batch must be a positive integer"),
    ({"batch": 1.5}, "batch must be a positive integer"),
    ({"pixel_format": "nv12", "batch": 2}, "not a batch"),
    ({"matrix": "bt2020"}, "matrix must be one of"),
    ({"pixel_format": "yuy2"}, "pixel_format must be one of"),
    ({"pixel_format": "nv12", "dtype": "float16", "bgr": True}, "dtype, bgr does not apply"),
    ({"pixel_format": "nv12", "normalize": False}, "normalize does not apply"),
    ({"dtype": "float64"}, "dtype must be one of"),
    ({"dtype": "uint8", "layout": "nchw"}, "layout must be 'nhwc'"),
    ({"layout": "chw"}, "layout must be 'nchw' or 'nhwc'"),
    ({"crop": (0, 0, 200, 10)}, "inside the 100x50 frame"),
])
def test_construction_refuses_contradictions(ext, kwargs, message):
    with pytest.raises(ValueError, match=message):
        GpuConverter(frame(), (64, 32), **kwargs)


def test_yuv_needs_even_dimensions(ext):
    with pytest.raises(ValueError, match="even width and height"):
        GpuConverter(frame(), (63, 32), pixel_format="p010")


# --------------------------------------------------------------------------
# process
# --------------------------------------------------------------------------

def test_process_translates_crop_into_texture_coordinates(ext):
    conv = GpuConverter(frame(region=(10, 20, 110, 70)), (64, 32), normalize=False, bgr=True)

    tensor = conv.process(frame(region=(10, 20, 110, 70)), crop=(5, 5, 25, 15))

    call = conv._impl.calls[-1]
    assert call["crop"] == (15, 25, 20, 10)
    assert (call["scale"], call["bias"], call["bgr"], call["source_id"]) == (255.0, 0.0, True, 7)
    assert tensor is conv.process(frame())      # one reused handle


def test_the_constructor_crop_is_the_default(ext):
    conv = GpuConverter(frame(), (64, 32), crop=(0, 0, 50, 25))
    conv.process(frame())

    assert conv.crop == (0, 0, 50, 25)
    assert conv._impl.calls[-1]["crop"] == (0, 0, 50, 25)
    assert conv._impl.calls[-1]["scale"] == 1.0


def test_regions_fill_slots_and_the_shape_follows_the_count(ext):
    conv = GpuConverter(frame(), (8, 4), batch=4)

    tensor = conv.process(frame(), regions=[(0, 0, 10, 10), (20, 0, 40, 30)])

    assert conv._impl.calls[-1]["regions"] == [(0, 0, 10, 10), (20, 0, 20, 30)]
    assert tensor.shape == (2, 3, 4, 8)
    assert tensor.numpy().shape == (2, 3, 4, 8)
    assert tensor.nbytes == 2 * 3 * 4 * 8 * 4


@pytest.mark.parametrize("kwargs,message", [
    ({"regions": [(0, 0, 1, 1)], "crop": (0, 0, 1, 1)}, "not both"),
    ({"regions": []}, "must not be empty"),
    ({"regions": [(0, 0, 1, 1)] * 3}, "batch=2"),
    ({"regions": [(0, 0, 1, 1), (0, 0, 500, 1)]}, r"regions\[1\]"),
])
def test_region_arguments_are_checked(ext, kwargs, message):
    conv = GpuConverter(frame(), (8, 4), batch=2)
    with pytest.raises(ValueError, match=message):
        conv.process(frame(), **kwargs)


def test_a_refused_dispatch_keeps_the_previous_shape(ext):
    """Shape describes what is in the buffer; a failed call did not change it."""
    conv = GpuConverter(frame(), (8, 4), batch=3)
    conv.process(frame(), regions=[(0, 0, 1, 1)] * 3)
    conv._impl.fail = RuntimeError("device removed")

    with pytest.raises(RuntimeError):
        conv.process(frame(), regions=[(0, 0, 1, 1)])

    assert conv.shape[0] == 3


def test_yuv_results_are_not_sliced_by_count(ext):
    conv = GpuConverter(frame(), (8, 4), pixel_format="p010", full_range=True, matrix="bt601")
    conv.process(frame())

    assert conv.shape == (6, 8)
    assert conv._slots(conv._impl.read_back()).shape == (6, 8)
    assert (conv.pixel_format, conv.matrix, conv.full_range) == ("p010", "bt601", True)
    assert conv.dtype == "uint16"


def test_converter_properties_and_repr(ext):
    conv = GpuConverter(frame(), (8, 4), dtype="uint8", layout="nhwc", sampling="nearest")

    assert (conv.batch, conv.sampling, conv.source_format) == (1, "nearest", "B8G8R8A8_UNORM")
    assert (conv.output_resource_address, conv.output_gpu_address) == (0x10, 0x20)
    assert conv.output_byte_size == 8 * 4 * 4
    assert repr(conv) == "<GpuConverter -> (1, 4, 8, 4) uint8 (nearest) on the capture adapter>"
    tensor = conv.process(frame())
    assert (tensor.dtype, tensor.shared_handle) == ("uint8", 0x30)
    assert tensor.adapter_luid == b"\x32\x34\x01\x00\x00\x00\x00\x00"
    assert repr(tensor) == "<GpuTensor (1, 4, 8, 4) uint8 (128 bytes) on the capture adapter>"
    tensor.sync()          # no CUDA view yet: nothing to wait for


# --------------------------------------------------------------------------
# TensorTransfer
# --------------------------------------------------------------------------

def test_tensor_transfer_passes_through(ext):
    conv = GpuConverter(frame(), (8, 4), dtype="float16")
    conv.process(frame())
    transfer = TensorTransfer(conv)

    transfer.transfer()

    assert transfer._impl.transfers == 1
    assert transfer.read_back_destination().shape == (1, 3, 4, 8)
    assert (transfer.source, transfer.destination) == ("Intel UHD", "NVIDIA RTX 4060")
    assert transfer.destination_is_software is False
    assert transfer.total_bytes == 2_460_000
    assert (transfer.shared_destination_handle, transfer.destination_resource_address) == (0x40, 0x50)
    assert transfer.destination_luid == bytes(range(8))
    assert repr(transfer) == "<TensorTransfer 2.46 MB 'Intel UHD' -> 'NVIDIA RTX 4060'>"


# --------------------------------------------------------------------------
# CUDA export
# --------------------------------------------------------------------------

class CudaFunction:
    """A foreign function stand-in that accepts argtypes, like the real thing."""

    def __init__(self, body):
        self.body = body
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.body(*args)


class FakeNvcuda:
    def __init__(self, luids, import_code=0, map_code=0, luid_code=0, get_code=0,
                 init_code=0, free_code=0, destroy_code=0):
        self.luids = luids
        self.imported = []
        self.freed = []
        self.destroyed = []

        def import_memory(ext_ref, desc_ref):
            desc = desc_ref._obj
            self.imported.append(dict(type=desc.type, handle=desc.handle.win32.handle,
                                      size=desc.size, flags=desc.flags))
            ext_ref._obj.value = 0xE0
            return import_code

        def mapped(ptr_ref, ext, buf_ref):
            ptr_ref._obj.value = 0xD000
            return map_code

        def device_get(dev_ref, ordinal):
            dev_ref._obj.value = ordinal
            # An int applies to every device; a dict to the ordinals it names.
            return get_code.get(ordinal, 0) if isinstance(get_code, dict) else get_code

        def get_luid(buf, mask_ref, dev):
            # ctypes converts a c_int argument for the real function; do the same.
            ctypes.memmove(buf, self.luids[getattr(dev, 'value', dev)], 8)
            return luid_code

        self.cuImportExternalMemory = CudaFunction(import_memory)
        self.cuExternalMemoryGetMappedBuffer = CudaFunction(mapped)
        self.cuMemFree = CudaFunction(lambda ptr: self.freed.append(ptr.value) or free_code)
        self.cuDestroyExternalMemory = CudaFunction(
            lambda ext: self.destroyed.append(ext.value) or destroy_code)
        self.cuInit = CudaFunction(lambda flags: init_code)
        self.cuDeviceGet = CudaFunction(device_get)
        self.cuDeviceGetLuid = CudaFunction(get_luid)


class FakeArray:
    def __init__(self, shape, dtype, memptr):
        self.shape, self.dtype, self.memptr = tuple(shape), dtype, memptr

    def __getitem__(self, key):
        count = len(range(*key.indices(self.shape[0])))
        return FakeArray((count,) + self.shape[1:], self.dtype, self.memptr)

    def __dlpack__(self, *, stream=None):
        """The protocol method, which is what `to_dlpack()` calls now.

        CuPy's `toDlpack()` is deprecated and warned on every call, so the
        library moved to `__dlpack__()`. `stream` is accepted and ignored: the
        DLPack spec passes it, and a fake that rejected it would fail for a
        reason the real array would not.
        """
        return ("dlpack", self.shape)

    def toDlpack(self):
        """Kept so a caller still on the deprecated name is not silently broken
        by this double, and so a regression back to it fails loudly instead."""
        raise AssertionError(
            "to_dlpack() must use __dlpack__(); toDlpack() is deprecated in CuPy"
        )


def fake_cupy(device_count=1):
    events = []

    class Device:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            events.append(("enter", self.index))

        def __exit__(self, *exc):
            return False

    cuda = types.SimpleNamespace(
        Device=Device,
        UnownedMemory=lambda ptr, size, owner, device_id: ("memory", ptr, size, device_id),
        MemoryPointer=lambda memory, offset: ("pointer", memory, offset),
        runtime=types.SimpleNamespace(
            getDeviceCount=lambda: device_count,
            free=lambda ptr: events.append(("context", ptr)),
            deviceSynchronize=lambda: events.append("synchronize")),
    )
    return types.SimpleNamespace(cuda=cuda, ndarray=FakeArray, dtype=np.dtype, events=events)


CAPTURE_LUID = b"\x32\x34\x01\x00\x00\x00\x00\x00"


@pytest.fixture
def cuda(monkeypatch, ext):
    def install(luids=(CAPTURE_LUID,), device_count=None, **codes):
        nvcuda = FakeNvcuda(list(luids), **codes)
        cp = fake_cupy(device_count if device_count is not None else len(luids))
        monkeypatch.setitem(__import__("sys").modules, "cupy", cp)
        monkeypatch.setattr(converter_module.ctypes, "WinDLL",
                            lambda name: nvcuda if name == "nvcuda.dll" else ctypes.WinDLL(name),
                            raising=False)
        return nvcuda, cp
    return install


def test_to_cupy_imports_the_whole_buffer_once(cuda):
    nvcuda, cp = cuda()
    conv = GpuConverter(frame(), (8, 4), batch=3)
    tensor = conv.process(frame(), regions=[(0, 0, 1, 1)])

    first = tensor.to_cupy()
    conv.process(frame(), regions=[(0, 0, 1, 1)] * 3)
    second = tensor.to_cupy()

    assert len(nvcuda.imported) == 1, "the import is cached across calls"
    desc = nvcuda.imported[0]
    assert desc["type"] == converter_module.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
    assert desc["handle"] == 0x30
    assert desc["size"] == conv.output_byte_size, "every slot, not this call's"
    assert desc["flags"] == converter_module.CUDA_EXTERNAL_MEMORY_DEDICATED
    assert (first.shape[0], second.shape[0]) == (1, 3)
    assert ("enter", 0) in cp.events


def test_the_cuda_device_is_matched_by_the_full_luid(cuda):
    """CuPy truncates the LUID at its first zero byte; the driver does not."""
    other = b"\x99\x00\x00\x00\x00\x00\x00\x00"
    nvcuda, cp = cuda(luids=(other, CAPTURE_LUID))
    conv = GpuConverter(frame(), (8, 4))

    conv.process(frame()).to_cupy()

    assert ("enter", 1) in cp.events
    assert nvcuda.cuDeviceGetLuid.argtypes[0] is ctypes.c_char_p


def test_no_matching_device_is_cross_adapter_required(cuda):
    cuda(luids=(b"\x99" * 8,))
    conv = GpuConverter(frame(), (8, 4))

    with pytest.raises(CrossAdapterRequired, match="3234010000000000"):
        conv.process(frame()).to_cupy()


def test_a_device_the_driver_will_not_describe_is_a_driver_failure(cuda):
    """It used to count as "not on this adapter", so a broken driver was
    reported as the hybrid-laptop case and the caller told to transfer."""
    cuda(get_code=100)
    conv = GpuConverter(frame(), (8, 4))
    with pytest.raises(RuntimeError, match=r"cuDeviceGet\(0\) failed with CUDA error 100") as info:
        conv.process(frame()).to_cupy()
    assert not isinstance(info.value, CrossAdapterRequired)

    with pytest.raises(RuntimeError, match=r"cuDeviceGetLuid\(0\) failed with CUDA error 1"):
        converter_module._cuda_device_luid(FakeNvcuda([CAPTURE_LUID], luid_code=1), 0)


def test_an_undescribed_device_does_not_hide_a_matching_one(cuda):
    nvcuda, cp = cuda(luids=(b"\x99" * 8, CAPTURE_LUID), get_code={0: 100})
    conv = GpuConverter(frame(), (8, 4))

    conv.process(frame()).to_cupy()

    assert ("enter", 1) in cp.events


def test_a_driver_that_will_not_initialise_is_not_cross_adapter(cuda):
    nvcuda, _ = cuda(init_code=100)
    conv = GpuConverter(frame(), (8, 4))
    with pytest.raises(RuntimeError, match="cuInit failed with CUDA error 100") as info:
        conv.process(frame()).to_cupy()
    assert not isinstance(info.value, CrossAdapterRequired)
    assert nvcuda.cuDeviceGet.calls == []


def test_the_context_exists_before_the_import(cuda):
    """A fresh process calling to_cupy() first failed with CUDA error 201.

    cuImportExternalMemory is a driver-API call and needs a current context.
    Entering ``cp.cuda.Device`` does not create one, so a script whose first
    CUDA work was to_cupy() had none -- the live tests passed only because
    earlier tests in the same process had already made one.
    """
    nvcuda, cp = cuda()
    real_import = nvcuda.cuImportExternalMemory.body

    def import_memory(ext_ref, desc_ref):
        cp.events.append("import")
        return real_import(ext_ref, desc_ref)

    nvcuda.cuImportExternalMemory.body = import_memory
    GpuConverter(frame(), (8, 4)).process(frame()).to_cupy()

    assert cp.events.index(("context", 0)) < cp.events.index("import")
    assert cp.events.index(("enter", 0)) < cp.events.index(("context", 0)), (
        "initialised on the device being imported into, not whichever is current")


def test_the_cuda_calls_declare_their_argument_types(cuda):
    nvcuda, _ = cuda()
    GpuConverter(frame(), (8, 4)).process(frame()).to_cupy()

    assert nvcuda.cuImportExternalMemory.argtypes[1]._type_ is converter_module._ExternalMemoryHandleDesc
    assert nvcuda.cuExternalMemoryGetMappedBuffer.argtypes[1] is ctypes.c_void_p
    assert nvcuda.cuDestroyExternalMemory.argtypes == [ctypes.c_void_p]
    assert nvcuda.cuInit.argtypes == [ctypes.c_uint]


def test_close_reports_driver_failures_and_still_drops_the_handles(cuda, caplog):
    nvcuda, _ = cuda(free_code=4, destroy_code=5)
    tensor = GpuConverter(frame(), (8, 4)).process(frame())
    tensor.to_cupy()
    view = tensor._cuda_view

    with caplog.at_level("WARNING", logger="rapidshot.converter"):
        view.close()
        view.close()

    assert "cuMemFree failed with CUDA error 4" in caplog.text
    assert "cuDestroyExternalMemory failed with CUDA error 5" in caplog.text
    assert nvcuda.freed == [0xD000] and nvcuda.destroyed == [0xE0], "once each"


def test_an_explicit_device_skips_the_search(cuda):
    nvcuda, cp = cuda(luids=(b"\x99" * 8,))
    conv = GpuConverter(frame(), (8, 4))

    conv.process(frame()).to_cupy(device=0)

    assert nvcuda.cuDeviceGetLuid.calls == []


@pytest.mark.parametrize("codes,what", [
    ({"import_code": 999}, "cuImportExternalMemory failed with CUDA error 999"),
    ({"map_code": 3}, "cuExternalMemoryGetMappedBuffer failed with CUDA error 3"),
])
def test_import_failures_name_the_cuda_call(cuda, codes, what):
    cuda(**codes)
    conv = GpuConverter(frame(), (8, 4))
    with pytest.raises(RuntimeError, match=what):
        conv.process(frame()).to_cupy()


def test_dlpack_torch_sync_and_close(cuda, monkeypatch):
    nvcuda, cp = cuda()

    def from_dlpack(obj):
        # Real `torch.from_dlpack` takes either a capsule or an object
        # implementing `__dlpack__`, and `to_torch()` now hands it the array so
        # Torch calls the protocol itself — which avoids CuPy's deprecated
        # `toDlpack()` and a single-use capsule. The fake resolves it the same
        # way, so this still asserts a working `__dlpack__` reached Torch rather
        # than merely that *something* did.
        return ("torch", obj.__dlpack__() if hasattr(obj, "__dlpack__") else obj)

    torch = types.SimpleNamespace(from_dlpack=from_dlpack)
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    conv = GpuConverter(frame(), (8, 4))
    tensor = conv.process(frame())

    assert tensor.to_dlpack() == ("dlpack", (1, 3, 4, 8))
    assert tensor.to_torch() == ("torch", ("dlpack", (1, 3, 4, 8)))

    view = tensor._cuda_view
    tensor.sync()
    assert "synchronize" in cp.events

    view.close()
    view.close()                    # idempotent
    assert nvcuda.freed == [0xD000] and nvcuda.destroyed == [0xE0]
    view.__del__()
