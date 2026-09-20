"""`GpuConverter` and `GpuTensor` — the 2.6 transform boundary (ROADMAP § 7.2).

Two objects, and the split between them is the point:

``GpuConverter``
    Owns the D3D12 pipeline and the output buffer. Built once, reused per
    frame. Configured with size, dtype, layout, sampling and normalisation.

``GpuTensor``
    A *handle* to the converter's output, and the only place framework interop
    lives. ``.to_torch()``, ``.to_cupy()`` and ``.to_dlpack()`` are here.

**Why not on `Frame`.** ROADMAP § 7.2 is explicit: a `Frame` is a captured
surface, not a tensor. Putting DLPack on it would promise that any frame can
become a model input, when what actually makes that true is a transform that
has to be configured and that allocates a buffer. Keeping the boundary at the
transform output means the type you can export from is the type that was
produced *for* export.

**What is verified where.** The conversion itself — every dtype, both sampling
modes — is verified on Machine A against live capture in
``tests/test_gpu_converter.py``. The CUDA export path is **not** verified on
Machine A, which has an Intel iGPU and no CUDA device at all; it needs Machine
B. ROADMAP § 5's rule applies: anything not run before a release is not
verified for that release.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Optional, Sequence, Tuple

from . import native
# Region and crop translation is shared with native.GpuPreprocessor12, which
# needs it for the same reason and must not drift from this path.
from .native import _texture_crop, _validate_crop

__all__ = ["GpuConverter", "GpuTensor", "TensorTransfer", "CrossAdapterRequired"]

logger = logging.getLogger(__name__)

# Mirrors examples/gpu_tensor_to_cupy.py, which stays as the worked example.
CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1

_DTYPE_BYTES = {"float32": 4, "float16": 2, "uint8": 1}
# Every dtype a converter can report, including P010's uint16, which is not a
# dtype a caller can ask for.
_ITEMSIZE = dict(_DTYPE_BYTES, uint16=2)
_PIXEL_FORMATS = ("nv12", "p010")
_MATRICES = ("bt709", "bt601")


class CrossAdapterRequired(RuntimeError):
    """The tensor is on an adapter the consumer's CUDA device cannot see.

    Distinct from a bare RuntimeError because the two need different
    responses: this one is routine on any Optimus laptop and is fixed by
    transferring the frame first, whereas an import failure is a bug.
    """


def _load_cuda_driver():
    """``nvcuda.dll``, or an error naming the situation rather than the file.

    This module already separates the two answers a caller has to tell apart:
    :class:`CrossAdapterRequired`, which is routine on a hybrid laptop, and an
    import failure, which is a bug. A machine with no NVIDIA driver at all is a
    third case, and it was getting the least useful message of the three --
    ``OSError: [WinError 126] The specified module could not be found``, which
    names a DLL and not the reason it is missing.
    """
    try:
        return ctypes.WinDLL("nvcuda.dll")
    except OSError as exc:
        raise RuntimeError(
            "the NVIDIA CUDA driver (nvcuda.dll) is not present on this "
            "machine, so a GPU tensor cannot be exported to CUDA. This is "
            "not the hybrid-laptop case CrossAdapterRequired describes: there "
            "is no CUDA driver to reach at all."
        ) from exc


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _Win32Handle),
        ("nvSciBufObject", ctypes.c_void_p),
    ]


class _ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class GpuTensor:
    """A handle to a converter's output, still resident on the GPU.

    Returned by :meth:`GpuConverter.process`. It does **not** own the memory —
    the converter does, and reuses it every frame. So a tensor is only valid
    until the next ``process()`` call, exactly like the buffer underneath it.
    Exporting is cheap and the import is cached; what is not free is assuming
    the contents stand still.

    **Ordering is the caller's, and no API here provides it.** ``process()``
    blocks on the D3D12 fence, so the tensor is complete when it returns. It
    knows nothing about CUDA work queued against the same memory. A kernel
    still reading while the next dispatch lands reads a half-written frame —
    which surfaces as wrong numbers, never as an error. Call
    :meth:`sync` before reusing the buffer.
    """

    def __init__(self, converter: "GpuConverter") -> None:
        self._converter = converter
        self._cuda_view = None

    # -- description ------------------------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._converter.shape

    @property
    def dtype(self) -> str:
        return self._converter.dtype

    @property
    def nbytes(self) -> int:
        """Bytes of this result: the regions the last call filled, not the
        whole buffer. See :attr:`GpuConverter.output_byte_size` for that."""
        count = 1
        for dim in self.shape:
            count *= dim
        return count * _ITEMSIZE[self.dtype]

    @property
    def shared_handle(self) -> int:
        """Borrowed NT handle. Owned by the converter; do not close it."""
        return self._converter._impl.shared_output_handle

    @property
    def adapter_luid(self) -> bytes:
        """LUID of the adapter holding this tensor, 8 little-endian bytes."""
        return bytes(self._converter._impl.adapter_luid())

    # -- CPU escape hatch -------------------------------------------------

    def numpy(self):
        """Copy to the CPU as a NumPy array. **Verification only.**

        This is the round-trip the whole path exists to avoid; ROADMAP § 3
        measures it as slower than doing the work on the CPU in the first
        place. It exists so tests can check the numbers.
        """
        return self._converter._slots(self._converter._impl.read_back())

    # -- framework interop ------------------------------------------------

    def to_cupy(self, device: Optional[int] = None):
        """Zero-copy ``cupy.ndarray`` view of this tensor.

        The external-memory import happens once and is cached: a capture loop
        pays nothing per frame to keep the view, because ``process()``
        overwrites the same buffer the array points at.

        Raises:
            CrossAdapterRequired: if no CUDA device matches this tensor's
                adapter — the ordinary Optimus case, where capture runs on the
                iGPU and the only CUDA device is the discrete GPU.
        """
        if self._cuda_view is None:
            self._cuda_view = _CudaView(self, device)
        # The import covers every batch slot and is cached; each call hands
        # back a view of only the slots the last process() filled.
        return self._cuda_view.array[: self.shape[0]]

    def to_dlpack(self):
        """DLPack capsule, for any framework implementing the protocol.

        Routed through CuPy, which already implements DLPack over a device
        pointer. That makes this CUDA-only, which matches what the tensor can
        actually reach: DirectML consumers bind
        :attr:`GpuConverter.output_resource_address` instead, and that is the
        vendor-neutral route ROADMAP § 8 documents.

        Uses ``__dlpack__()`` rather than CuPy's ``toDlpack()``, which is
        deprecated and emitted a ``VisibleDeprecationWarning`` on every call.
        The return type is unchanged -- both hand back a PyCapsule, and
        ``cupy.from_dlpack`` and ``torch.from_dlpack`` accept it either way
        (checked against CuPy 14.1.1 and Torch 2.11).
        """
        return self.to_cupy().__dlpack__()

    def to_torch(self, device: Optional[int] = None):
        """Zero-copy ``torch.Tensor`` sharing this tensor's memory.

        Goes through DLPack so no pixels move. The result aliases the
        converter's buffer, so the reuse warning in the class docstring
        applies to it too — clone it if you need it to outlive the next
        ``process()``.
        """
        import torch

        array = self.to_cupy(device)
        # from_dlpack is the documented zero-copy route; torch.as_tensor over
        # __cuda_array_interface__ would also work but copies on some versions,
        # which would silently undo the point of this method.
        #
        # The array is passed directly rather than as a capsule: Torch then
        # calls `__dlpack__` itself, which avoids CuPy's deprecated
        # `toDlpack()` and its per-call warning. A capsule is also single-use,
        # so handing over the object is the more forgiving of the two.
        return torch.from_dlpack(array)

    def sync(self) -> None:
        """Block until CUDA work against this memory has finished.

        A full stream synchronise — blunt, but correct. The version that
        overlaps instead of stalling needs a shared D3D12 fence imported with
        ``cuImportExternalSemaphore``; that is the same work ROADMAP § 6.1
        shipped for the cross-adapter path and has not been applied here.
        """
        if self._cuda_view is not None:
            self._cuda_view.sync()

    def __repr__(self) -> str:
        return (
            f"<GpuTensor {self.shape} {self.dtype} "
            f"({self.nbytes} bytes) on the capture adapter>"
        )


class _CudaView:
    """The ~60 lines of ctypes that ROADMAP § 7.2 asks be collapsed to one call.

    Kept private: it is an implementation detail of ``to_cupy()``, not an API.
    ``examples/gpu_tensor_to_cupy.py`` remains the worked, commented version.
    """

    def __init__(self, tensor: GpuTensor, device: Optional[int]) -> None:
        import cupy as cp

        self._cp = cp
        self._cuda = _load_cuda_driver()
        # Declared so ctypes converts each argument to the width the driver
        # reads, instead of guessing from the Python value.
        self._cuda.cuMemFree.argtypes = [ctypes.c_ulonglong]
        self._cuda.cuImportExternalMemory.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_ExternalMemoryHandleDesc)
        ]
        self._cuda.cuExternalMemoryGetMappedBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_void_p,
            ctypes.POINTER(_ExternalMemoryBufferDesc),
        ]
        self._cuda.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        self._ext = ctypes.c_void_p()
        self._device_ptr = None
        # Hold the tensor so the D3D12 resource and its shared handle outlive
        # the mapping that points into them.
        self._tensor = tensor

        self._device = (
            device if device is not None else _device_for_adapter(cp, tensor.adapter_luid)
        )

        desc = _ExternalMemoryHandleDesc()
        ctypes.memset(ctypes.byref(desc), 0, ctypes.sizeof(desc))
        desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        desc.handle.win32.handle = ctypes.c_void_p(tensor.shared_handle)
        # The whole buffer, not this call's slots: the import is cached across
        # calls whose region count differs.
        nbytes = tensor._converter.output_byte_size
        desc.size = nbytes
        # A committed D3D12 resource is a dedicated allocation; omitting this
        # flag makes the import fail on some drivers and succeed on others.
        desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED

        with cp.cuda.Device(self._device):
            # The import below is a driver-API call, and it needs a current
            # context. Entering the Device block does not create one: when this
            # is the first CUDA work in the process -- a fresh script doing
            # capture, convert, to_cupy() -- nothing has made the primary
            # context current yet, and the import fails with CUDA error 201
            # (INVALID_CONTEXT). cudaFree(0) is the runtime's documented way to
            # initialise it, and a no-op once it exists.
            cp.cuda.runtime.free(0)
            _check(
                self._cuda.cuImportExternalMemory(ctypes.byref(self._ext), ctypes.byref(desc)),
                "cuImportExternalMemory",
            )
            buf = _ExternalMemoryBufferDesc()
            ctypes.memset(ctypes.byref(buf), 0, ctypes.sizeof(buf))
            buf.offset = 0
            buf.size = nbytes
            ptr = ctypes.c_ulonglong()
            _check(
                self._cuda.cuExternalMemoryGetMappedBuffer(
                    ctypes.byref(ptr), self._ext, ctypes.byref(buf)
                ),
                "cuExternalMemoryGetMappedBuffer",
            )
            self._device_ptr = ptr.value

            # UnownedMemory because the allocation belongs to D3D12: CuPy must
            # not free it. `owner=self` keeps this view alive as long as the
            # array is, which keeps the mapping alive with it.
            memory = self._cp.cuda.UnownedMemory(
                self._device_ptr, nbytes, owner=self, device_id=self._device
            )
            self.array = self._cp.ndarray(
                tensor._converter._capacity_shape,
                dtype=self._cp.dtype(tensor.dtype),
                memptr=self._cp.cuda.MemoryPointer(memory, 0),
            )

    def sync(self) -> None:
        with self._cp.cuda.Device(self._device):
            self._cp.cuda.runtime.deviceSynchronize()

    def close(self) -> None:
        # Teardown carries on past a failure -- the handles are dropped either
        # way -- but says so, rather than leaking a mapping without a trace.
        if self._device_ptr is not None:
            self.sync()
            code = self._cuda.cuMemFree(ctypes.c_ulonglong(self._device_ptr))
            if code:
                logger.warning(f"cuMemFree failed with CUDA error {code}")
            self._device_ptr = None
        if self._ext:
            code = self._cuda.cuDestroyExternalMemory(self._ext)
            if code:
                logger.warning(f"cuDestroyExternalMemory failed with CUDA error {code}")
            self._ext = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _check(code: int, what: str) -> None:
    if code != 0:
        raise RuntimeError(f"{what} failed with CUDA error {code}")


def _cuda_device_luid(cuda, ordinal: int) -> bytes:
    """LUID of a CUDA device as the full 8 bytes.

    Raises RuntimeError naming the call and code if the driver cannot say.

    Read from the driver API rather than CuPy's device properties, which
    cannot be trusted for this. CuPy converts the fixed-size ``char luid[8]``
    field to Python bytes as though it were a C string, so the value stops at
    its first zero byte — and a LUID almost always contains one. The RTX 4060
    here reports 8 bytes as ``3234010000000000`` and CuPy hands back three.
    Comparing that against a real LUID never matches, on any adapter.
    """
    dev = ctypes.c_int()
    _check(cuda.cuDeviceGet(ctypes.byref(dev), ordinal), f"cuDeviceGet({ordinal})")
    buf = (ctypes.c_char * 8)()
    node_mask = ctypes.c_uint()
    _check(cuda.cuDeviceGetLuid(buf, ctypes.byref(node_mask), dev),
           f"cuDeviceGetLuid({ordinal})")
    return bytes(buf)


def _device_for_adapter(cp, luid: bytes) -> int:
    """Find the CUDA device sitting on a given adapter, by LUID.

    Counting devices cannot do this. On a hybrid laptop CUDA reports exactly
    one device and it is the *wrong* one — capture is on the iGPU. Matching
    LUIDs is what distinguishes "pick device 0" from "there is no device here".
    """
    count = cp.cuda.runtime.getDeviceCount()
    cuda = _load_cuda_driver()
    cuda.cuInit.argtypes = [ctypes.c_uint]
    cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    cuda.cuDeviceGetLuid.argtypes = [
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_int
    ]
    # Harmless once CuPy has already initialised the driver; required when it
    # has not, because a device query before cuInit fails. Checked, because a
    # driver that will not start is not a hybrid laptop and must not be
    # reported as one.
    _check(cuda.cuInit(0), "cuInit")
    failures = []
    for index in range(count):
        try:
            if _cuda_device_luid(cuda, index) == luid[:8]:
                return index
        except RuntimeError as e:
            failures.append(str(e))
    if failures:
        # A device that could not be described may be the one on this
        # adapter, so "no device here" has not been established.
        raise RuntimeError(
            f"could not match a CUDA device to the adapter holding this tensor "
            f"(LUID {luid.hex()}): {'; '.join(failures)}"
        )
    raise CrossAdapterRequired(
        f"no CUDA device is on the adapter holding this tensor (LUID {luid.hex()}); "
        f"{count} CUDA device(s) present, none matching. This is the ordinary "
        "hybrid-laptop case: capture is on the integrated GPU. Transfer the "
        "frame to the CUDA adapter first."
    )


class TensorTransfer:
    """Move a converted tensor to another adapter — ordering **B** of § 6.1.

    The library has always been able to move a *frame* across adapters and let
    the consumer convert on arrival (ordering A, ``native.CrossAdapterTransfer``).
    ROADMAP § 6.1 measured the other order — convert first, then move the much
    smaller result — and found it wins at every size at 2560×1600, and up to
    640² FP16 at 1080p. Selecting it was impossible until now: the converter
    writes a buffer and the frame transfer takes a texture.

    **The dtype is the decision.** What crosses is whatever the converter
    produced, so at 640² that is 1.64 MB as ``uint8``, 2.46 MB as
    ``float16``, or 4.92 MB as ``float32`` — against a 1080p frame's 8.29 MB
    or a 1600p frame's 16.38 MB. At 1080p, ``float16`` is on the winning side
    of § 6.1 and ``float32`` is not.

    Usage::

        converter = rapidshot.GpuConverter(frame, (640, 640), dtype="float16")
        transfer = rapidshot.TensorTransfer(converter)

        with camera.grab_frame() as frame:
            converter.process(frame)
            transfer.transfer()
        # the tensor is now readable on transfer.destination

    Raises:
        RuntimeError: if this machine has only one adapter — there is then
            nothing to transfer to, and ``GpuTensor.shared_handle`` already
            reaches the tensor where it is.
    """

    def __init__(self, converter: "GpuConverter") -> None:
        self._converter = converter
        self._impl = native.require_feature("TensorTransfer")(converter._impl)

    def transfer(self) -> None:
        """Copy the converter's current output across. Blocks until complete.

        Call :meth:`GpuConverter.process` first: this moves whatever is in the
        buffer, and a buffer never written is zeros rather than an error.
        """
        self._impl.transfer(self._converter._impl)

    def read_back_destination(self):
        """Destination-side bytes as a NumPy array. **Verification only.**"""
        return self._converter._slots(self._impl.read_back_destination())

    @property
    def source(self) -> str:
        """Adapter the tensor was produced on — the capture adapter."""
        return self._impl.source

    @property
    def destination(self) -> str:
        return self._impl.destination

    @property
    def destination_is_software(self) -> bool:
        """True when the destination is WARP.

        Machine A has no second hardware GPU, so its measurements land here.
        Only the *source* side of a WARP transfer is representative.
        """
        return bool(self._impl.destination_is_software)

    @property
    def total_bytes(self) -> int:
        """Bytes that cross — the whole point of choosing this ordering."""
        return int(self._impl.total_bytes)

    @property
    def shared_destination_handle(self) -> int:
        """Borrowed NT handle for the destination heap, for a consumer there.

        Owned by this transfer and closed with it; do not close it yourself.
        """
        return int(self._impl.shared_destination_handle)

    @property
    def destination_resource_address(self) -> int:
        """``ID3D12Resource`` address on the destination adapter."""
        return int(self._impl.destination_resource_address)

    @property
    def destination_luid(self) -> bytes:
        """LUID of the destination adapter, as 8 little-endian bytes.

        Pair with CUDA's ``cuDeviceGetLuid`` to find the device that can
        import the transferred tensor.
        """
        return bytes(self._impl.destination_luid())

    def __repr__(self) -> str:
        return (
            f"<TensorTransfer {self.total_bytes / 1e6:.2f} MB "
            f"{self.source!r} -> {self.destination!r}>"
        )


class GpuConverter:
    """Resize, convert and normalise a captured frame, entirely on the GPU.

    Args:
        frame: A live frame, used to pick the adapter and check shareability.
            Nothing is read from it at construction.
        size: ``(width, height)`` of the output.
        dtype: ``"float32"``, ``"float16"``, or ``"uint8"`` for a resized
            BGRA frame rather than a tensor.
        layout: ``"nchw"`` (default) or ``"nhwc"`` for the float dtypes —
            ``(N, 3, H, W)`` or ``(N, H, W, 3)``, the same values in either
            order. ``uint8`` is inherently interleaved and must be ``"nhwc"``.
        sampling: ``"bilinear"`` (default) or ``"nearest"``.
        normalize: emit 0..1 (the default). ``False`` emits 0..255 instead.
            Note this is not a division: the capture format is UNORM, so the
            GPU hands the shader 0..1 already and the byte range is what has
            to be reconstructed.
        bgr: emit BGR instead of RGB. Ignored for ``uint8``.
        pixel_format: ``"nv12"`` or ``"p010"`` for a 4:2:0 frame an encoder
            accepts, instead of a tensor. Mutually exclusive with ``dtype``,
            ``layout``, ``normalize`` and ``bgr``, which describe tensors.
        crop: ``(left, top, right, bottom)`` in **frame coordinates** — the
            same convention as :attr:`Frame.region` and ``dirty_rects`` — to
            convert only that part of the frame. ``None`` (default) is the
            whole frame. Overridable per call in :meth:`process`.
        matrix: ``"bt709"`` (default) or ``"bt601"``. YUV output only.
        full_range: ``False`` (default) for limited range — Y 16–235, what
            encoders assume — or ``True`` for 0–255. YUV output only.
        batch: most regions one :meth:`process` call can convert (default 1).
            The output buffer is allocated for this many at construction.
            Not available with ``pixel_format``.

    **Multi-ROI.** ``process(frame, regions=[...])`` converts every region to
    the same output size in **one** GPU dispatch and returns an
    ``(N, 3, H, W)`` tensor (``(N, H, W, 4)`` for ``uint8``), one slot per
    region in the order given. ``N`` is the number of regions in that call, up
    to ``batch``, so a caller tracking a varying number of windows builds one
    converter and does not reallocate::

        conv = rapidshot.GpuConverter(frame, (224, 224), batch=8)
        tensor = conv.process(frame, regions=[(0, 0, 400, 300), (800, 40, 1000, 240)])
        tensor.shape   # (2, 3, 224, 224)

    Slots past ``N`` keep whatever an earlier call wrote; :attr:`GpuTensor.shape`
    and every export stop at ``N``, but a consumer binding
    :attr:`output_resource_address` directly sees the whole buffer.

    **NV12 and P010** are laid out the standard way: the Y plane, then Cb/Cr
    interleaved at half resolution, so :meth:`GpuTensor.numpy` returns shape
    ``(H * 3 // 2, W)`` — ``uint8`` for NV12, ``uint16`` for P010 with the value
    in the high ten bits. Width and height must be even. Chroma is the average
    of each 2×2 block (centre-sited). The source is taken as gamma-encoded
    BT.709 RGB, which a UNORM desktop surface is; an HDR ``R16G16B16A16_FLOAT``
    surface is refused, because it holds linear light and no tone-mapping or
    PQ decision has been made for it.

    **Crop is applied before resizing, and the frame's region is honoured.** A
    camera created with ``region=`` hands back frames whose texture is still
    the whole monitor; the converter reads only the region, and ``crop`` is
    relative to it. (Before crop existed this path resized the entire monitor
    and ignored the region, with the output shape giving no sign of it.) With
    bilinear sampling the filter never reaches outside the crop, so upscaling a
    small crop does not blend in the pixels beyond its edge. A crop reaching
    outside the frame is refused rather than clamped.

    **Rotated displays** are not handled: the surface is unrotated, and
    translating frame coordinates onto it is not implemented, so ``crop`` is
    refused when :attr:`Frame.rotation_angle` is non-zero.

    **The default is bilinear, and that is a deliberate break with
    `GpuPreprocessor12`.** The old path samples with ``Load()``, which at
    2560×1600 → 640² discards roughly fifteen of every sixteen pixels rather
    than filtering them — aliasing exactly the small text and thin borders
    desktop capture is usually pointed at. The old path is unchanged and still
    available; pass ``sampling="nearest"`` to reproduce it here.
    """

    def __init__(
        self,
        frame,
        size: Tuple[int, int],
        *,
        dtype: Optional[str] = None,
        layout: Optional[str] = None,
        sampling: str = "bilinear",
        normalize: bool = True,
        bgr: bool = False,
        pixel_format: Optional[str] = None,
        crop: Optional[Tuple[int, int, int, int]] = None,
        matrix: str = "bt709",
        full_range: bool = False,
        batch: int = 1,
    ) -> None:
        gpu_converter = native.require_feature("GpuConverter12")
        width, height = int(size[0]), int(size[1])
        self._crop = None if crop is None else _validate_crop(frame, crop)

        if isinstance(batch, bool) or int(batch) != batch or batch < 1:
            raise ValueError(f"batch must be a positive integer, got {batch!r}")
        batch = int(batch)
        if pixel_format is not None and batch != 1:
            raise ValueError(
                f"pixel_format={pixel_format!r} is one frame for an encoder, "
                f"not a batch; got batch={batch}"
            )

        if matrix not in _MATRICES:
            raise ValueError(f"matrix must be one of {_MATRICES}, got {matrix!r}")

        if pixel_format is not None:
            pixel_format = pixel_format.lower()
            if pixel_format not in _PIXEL_FORMATS:
                raise ValueError(
                    f"pixel_format must be one of {_PIXEL_FORMATS}, got {pixel_format!r}"
                )
            # Refuse rather than ignore: a caller passing bgr=True to an NV12
            # converter believes something about the output that is not true.
            conflicting = [
                name
                for name, given in (
                    ("dtype", dtype is not None),
                    ("layout", layout is not None),
                    ("normalize", normalize is not True),
                    ("bgr", bgr),
                )
                if given
            ]
            if conflicting:
                raise ValueError(
                    f"pixel_format={pixel_format!r} produces a YUV frame, so "
                    f"{', '.join(conflicting)} does not apply; drop it"
                )
            if width % 2 or height % 2:
                raise ValueError(
                    f"{pixel_format} requires even width and height, got "
                    f"{width}x{height}: 4:2:0 subsamples in 2x2 blocks"
                )
            native_dtype, native_layout = pixel_format, None
            layout = "yuv420"
        else:
            dtype, layout, native_dtype = self._tensor_format(dtype, layout)
            native_layout = layout

        self._impl = gpu_converter(
            native._texture_address(frame),
            width,
            height,
            sampling=sampling,
            dtype=native_dtype,
            matrix=matrix,
            full_range=bool(full_range),
            batch=batch,
            layout=native_layout,
        )
        self.out_width = width
        self.out_height = height
        # Slots the last process() filled. The whole capacity until one runs,
        # since nothing narrower has been asked for.
        self._count = batch
        self._layout = layout
        self._normalize = bool(normalize)
        self._bgr = bool(bgr)
        self._tensor = GpuTensor(self)

    @staticmethod
    def _tensor_format(dtype: Optional[str], layout: Optional[str]):
        dtype = "float32" if dtype is None else dtype
        if dtype not in _DTYPE_BYTES:
            raise ValueError(
                f"dtype must be one of {sorted(_DTYPE_BYTES)}, got {dtype!r}"
            )
        layout = "nchw" if layout is None else layout.lower()
        # "hwc" names the same memory as "nhwc"; the batch dimension is there
        # either way, so it is normalised rather than kept as a second spelling.
        if layout == "hwc":
            layout = "nhwc"
        if dtype == "uint8":
            if layout != "nhwc":
                raise ValueError(
                    "uint8 output is an interleaved BGRA frame, so layout must "
                    f"be 'nhwc'; got {layout!r}"
                )
            return dtype, layout, "bgra8"
        if layout not in ("nchw", "nhwc"):
            raise ValueError(
                f"layout must be 'nchw' or 'nhwc' for {dtype}; got {layout!r}"
            )
        return dtype, layout, dtype

    @property
    def layout(self) -> str:
        """``"nchw"`` or ``"nhwc"`` for tensors and BGRA, ``"yuv420"`` for
        NV12/P010."""
        return self._layout

    def process(
        self,
        frame,
        crop: Optional[Tuple[int, int, int, int]] = None,
        *,
        regions: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    ) -> GpuTensor:
        """Convert one frame and return a handle to the result.

        Args:
            frame: A live frame.
            crop: ``(left, top, right, bottom)`` in frame coordinates for this
                call only; ``None`` uses the constructor's ``crop``. Per call
                because the region worth converting usually moves.
            regions: several crops, same coordinates, converted together in
                one dispatch into slots ``0..len(regions)``. At most ``batch``;
                not combinable with ``crop``.

        The returned :class:`GpuTensor` is the *same object* every call — it
        describes a buffer that is reused, so holding two of them would imply
        two results that do not exist.
        """
        if regions is not None:
            if crop is not None:
                raise ValueError("give crop or regions, not both")
            regions = list(regions)
            if not regions:
                raise ValueError("regions must not be empty")
            if len(regions) > self.batch:
                raise ValueError(
                    f"{len(regions)} regions given, but this converter was built "
                    f"with batch={self.batch}"
                )
            texture_regions = []
            for index, region in enumerate(regions):
                try:
                    checked = _validate_crop(frame, region)
                except ValueError as error:
                    raise ValueError(f"regions[{index}]: {error}") from None
                texture_regions.append(_texture_crop(frame, checked))
            native_args = dict(regions=texture_regions)
            count = len(regions)
        else:
            native_args = dict(
                crop=_texture_crop(
                    frame, self._crop if crop is None else _validate_crop(frame, crop)
                )
            )
            count = 1
        # The SRV is B8G8R8A8_UNORM, so the texture fetch *already* returns
        # 0..1 — the hardware normalises on read. Dividing by 255 here would
        # normalise twice and hand the model a tensor 255x too dark, which is
        # a plausible-looking array that no shape or dtype check would catch.
        # So 0..1 is scale=1.0, and the un-normalised 0..255 range is the one
        # that costs a multiply.
        scale = 1.0 if self._normalize else 255.0
        # The texture address alone is not an identity (COM addresses get
        # recycled), so the frame's source_id goes with it.
        self._impl.process(
            native._texture_address(frame),
            scale,
            0.0,
            self._bgr,
            source_id=getattr(frame, "source_id", 0),
            **native_args,
        )
        # Only after the dispatch succeeded: a refused call must not shrink
        # the shape describing the result that is still in the buffer.
        self._count = count
        return self._tensor

    @property
    def crop(self) -> Optional[Tuple[int, int, int, int]]:
        """The default crop, in frame coordinates, or ``None`` for the whole frame."""
        return self._crop

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the last result: leading dimension is its region count."""
        shape = self._capacity_shape
        if self.pixel_format is not None:
            return shape
        return (self._count,) + shape[1:]

    @property
    def batch(self) -> int:
        """Most regions one :meth:`process` call can convert."""
        return int(self._impl.batch)

    @property
    def _capacity_shape(self) -> Tuple[int, ...]:
        """Shape of the whole buffer, every batch slot."""
        return tuple(self._impl.shape)

    def _slots(self, raw):
        """Whole-buffer bytes -> the slots the last call filled, as NumPy."""
        import numpy as np

        array = np.frombuffer(raw, dtype=np.dtype(self.dtype)).reshape(self._capacity_shape)
        return array if self.pixel_format is not None else array[: self._count]

    @property
    def dtype(self) -> str:
        return self._impl.dtype

    @property
    def sampling(self) -> str:
        return self._impl.sampling

    @property
    def pixel_format(self) -> Optional[str]:
        """``"nv12"``, ``"p010"``, or ``None`` for a tensor or BGRA output."""
        return self._impl.pixel_format

    @property
    def matrix(self) -> str:
        return self._impl.yuv[0]

    @property
    def full_range(self) -> bool:
        return bool(self._impl.yuv[1])

    @property
    def source_format(self) -> str:
        """DXGI format of the captured surface, or ``"none"`` before the first
        frame.

        All four formats ``DuplicateOutput1`` requests are accepted:
        ``B8G8R8A8_UNORM``, ``R8G8B8A8_UNORM``, ``R10G10B10A2_UNORM`` and
        ``R16G16B16A16_FLOAT``. They need no separate shader — a
        ``Texture2D<float4>`` read returns RGBA order for all of them — but
        the SRV follows the surface, because declaring BGRA8 over a 10-bit one
        reinterprets the bits instead of converting them.

        **Range differs, and only for HDR.** The three UNORM formats normalise
        to 0..1 on read. ``R16G16B16A16_FLOAT`` is scRGB and is not bounded by
        1.0: a highlight legitimately reads above it, and float outputs pass
        that through rather than clamping, since clamping would discard what
        the format exists to carry. ``dtype="uint8"`` saturates, having no
        choice.
        """
        return self._impl.source_format

    @property
    def output_byte_size(self) -> int:
        return int(self._impl.output_byte_size)

    @property
    def output_resource_address(self) -> int:
        """``ID3D12Resource`` address — what DirectML and ONNX Runtime bind."""
        return int(self._impl.output_resource_address)

    @property
    def output_gpu_address(self) -> int:
        return int(self._impl.output_gpu_address)

    def __repr__(self) -> str:
        return (
            f"<GpuConverter -> {self.shape} {self.dtype} "
            f"({self.sampling}) on the capture adapter>"
        )
