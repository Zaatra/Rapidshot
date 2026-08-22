"""Optional native GPU-interop shim (Stage 6).

The native extension is **optional by design**. Rapidshot's capture path is pure
Python and stays that way — profiling put the Python/COM binding overhead at
about 0.003 ms per frame, which is not worth a build dependency. The extension
exists only for the one thing Python genuinely cannot do: hand a captured
Direct3D texture to a GPU consumer such as DirectML, whose
``CreateGPUAllocationFromD3DResource`` has no Python binding.

So `pip install rapidshot` works with no toolchain, and everything except
GPU-tensor interop behaves identically. Callers check availability with
:func:`is_available` and get a clear, actionable error from :func:`require`
rather than an ImportError from somewhere deep in the stack.

Building it (needs Rust and the MSVC C++ toolset)::

    cd native && cargo build --release
    python native/install_dev.py     # copies the artifact next to this file
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ext: Optional[Any] = None
_import_error: Optional[BaseException] = None

try:  # pragma: no cover - depends on whether the extension was built
    from rapidshot import _rapidshot_native as _ext  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover
    _import_error = exc
    logger.debug(f"Native GPU interop shim not present: {exc}")


BUILD_HINT = (
    "The native GPU interop extension is not built. Everything except "
    "GPU-tensor interop works without it.\n"
    "To build it you need Rust (https://rustup.rs) and the MSVC C++ build "
    "tools, then:\n"
    "    cd native && cargo build --release\n"
    "    python native/install_dev.py"
)


def is_available() -> bool:
    """True if the native GPU interop extension is loaded."""
    return _ext is not None


def require() -> Any:
    """
    Return the native module, or raise with instructions for building it.

    Raises:
        RuntimeError: If the extension is not available.
    """
    if _ext is None:
        raise RuntimeError(f"{BUILD_HINT}\n\nOriginal import error: {_import_error}")
    return _ext


def build_info() -> Optional[Dict[str, Any]]:
    """Version/stage of the loaded extension, or None if absent."""
    if _ext is None:
        return None
    return dict(_ext.build_info())


def _addressable(src, dst, channels) -> bool:
    """Whether the kernels can address this pair of arrays directly.

    Shared by both conversion wrappers. Everything refused here is a layout where
    the arrays' strides would not describe the memory a raw-pointer kernel is
    about to walk -- a 3-channel source, a column slice, a transposed view. All
    are legitimate inputs to the NumPy path, so refusing is routine.
    """
    if src.dtype != "uint8" or src.ndim != 3 or src.shape[2] != 4:
        return False
    if src.strides[2] != 1 or src.strides[1] != 4 or src.strides[0] < src.shape[1] * 4:
        return False

    if dst.dtype != "uint8" or dst.shape[:2] != src.shape[:2]:
        return False
    if channels == 1:
        if dst.ndim != 2 or dst.strides[1] != 1 or dst.strides[0] < dst.shape[1]:
            return False
    else:
        if dst.ndim != 3 or dst.shape[2] != channels:
            return False
        # Pixels must be packed within a row; rows themselves may be strided.
        if dst.strides[2] != 1 or dst.strides[1] != channels:
            return False
        if dst.strides[0] < dst.shape[1] * channels:
            return False
    return True


def bgra_swizzle_into(src, dst, mode: str) -> bool:
    """
    Reorder BGRA `src` into `dst` for mode RGB, BGR or RGBA.

    Returns True if the native path ran, False if the layout is not one it can
    address, in which case the caller must fall back to NumPy. Byte-identical to
    what `NumpyProcessor.convert_into` produces for the same mode.

    These are the most-used modes, and the ones furthest from the memory system's
    limit: the NumPy path assigns one channel at a time, so it makes three
    strided passes over the frame where one pass suffices. See `native/src/
    swizzle.rs` for the measured gap.

    Both row pitches come from the arrays' own strides, so a dirty-rect
    sub-rectangle of the accumulator is written in place without a copy.
    """
    if _ext is None:
        return False

    channels = 4 if mode == "RGBA" else 3
    if mode not in ("RGB", "BGR", "RGBA"):
        return False
    if not _addressable(src, dst, channels):
        return False

    height, width = dst.shape[:2]
    pitch, dst_pitch = src.strides[0], dst.strides[0]
    src_len = pitch * (height - 1) + width * 4
    dst_len = dst_pitch * (height - 1) + width * channels

    _ext.bgra_swizzle_into(
        src.ctypes.data, src_len, dst.ctypes.data, dst_len,
        width, height, mode, pitch, dst_pitch,
    )
    return True


def bgra_to_gray_into(src, dst) -> bool:
    """
    Convert BGRA `src` into single-channel `dst` using the native kernel.

    Returns True if the native path ran, False if this pair of arrays is not
    something it can address, in which case the caller must fall back to NumPy.
    Returning a flag rather than raising is deliberate: an unsupported layout is
    an ordinary occurrence, not an error, and the NumPy path is always correct.

    Byte-identical to `rapidshot.processor.numpy_processor.bgra_to_gray` -- the
    Rust side asserts that over all 2^24 BGR triples and the Python suite checks
    it again through this wrapper. Measured 0.686 ms against NumPy's 9.4 ms on a
    1920x1080 frame (`benchmarks/gray_kernel.py`).

    The geometry is passed explicitly because the kernel takes raw addresses:
    both row pitches come from the arrays' own strides, so a sub-rectangle of a
    larger buffer is addressed correctly without being copied first. Everything
    this function refuses is a layout where those strides would not describe the
    memory the kernel is about to walk.
    """
    if _ext is None:
        return False
    if not _addressable(src, dst, 1):
        return False

    height, width = dst.shape
    pitch, dst_pitch = src.strides[0], dst.strides[0]
    # The span each view actually owns, from its own first byte to its last.
    # Passing these lets the kernel cross-check the geometry it was handed
    # against the buffer it was pointed at, instead of trusting either alone.
    src_len = pitch * (height - 1) + width * 4
    dst_len = dst_pitch * (height - 1) + width

    _ext.bgra_to_gray_into(
        src.ctypes.data, src_len, dst.ctypes.data, dst_len,
        width, height, pitch, dst_pitch,
    )
    return True


def describe_texture(frame) -> Dict[str, Any]:
    """
    Read the Direct3D description of a live :class:`~rapidshot.frame.Frame`.

    Args:
        frame: A Frame that has not been released.

    Returns:
        Dimensions, format and flags of the underlying texture.

    Raises:
        RuntimeError: If the extension is unavailable.
        FrameReleasedError: If the frame was already released.
    """
    return dict(require().describe_texture(_texture_address(frame)))


def texture_sharing_info(frame) -> Dict[str, Any]:
    """
    Report whether a frame's texture can be shared with another device.

    Stage 6 needs the captured surface visible to the DirectML device. Desktop
    duplication surfaces are not created with sharing flags, so
    ``needs_intermediate_copy`` is expected to be True — meaning the interop has
    to route through a shared intermediate resource rather than binding the
    duplicated surface directly.
    """
    return dict(require().texture_sharing_info(_texture_address(frame)))


class GpuPreprocessor:
    """
    Converts captured frames into model-ready NCHW float32 tensors, on the GPU.

    Replaces the staging read, color conversion, resize and normalisation that
    the CPU path spends roughly 8 ms/frame on at 1080p. Built once for a given
    output size and reused, so the per-frame path allocates nothing.

    ::

        pre = GpuPreprocessor(frame, 640, 640)
        with camera.grab_frame() as frame:
            pre.process(frame)            # stays on the GPU

    Args:
        frame: A live Frame, used to bind to the capture device.
        out_width / out_height: Model input size; the shader resizes to it.
    """

    def __init__(self, frame, out_width: int, out_height: int) -> None:
        ext = require()
        self._impl = ext.GpuPreprocessor(
            _texture_address(frame), int(out_width), int(out_height)
        )
        self.out_width = int(out_width)
        self.out_height = int(out_height)

    def process(self, frame, scale: float = 1.0, bias: float = 0.0,
                bgr: bool = False) -> None:
        """
        Convert one frame. Nothing is copied to the CPU.

        Args:
            frame: A live Frame.
            scale / bias: Applied as ``value * scale + bias`` after the 0..1
                texture fetch. Defaults give 0..1; use scale=2, bias=-1 for -1..1.
            bgr: Emit BGR channel order instead of RGB.
        """
        self._impl.process(_texture_address(frame), scale, bias, bgr)

    def read_back(self):
        """
        Copy the tensor to the CPU as a NumPy array of shape (1, 3, H, W).

        Verification and debugging only — this reintroduces the CPU round-trip
        the GPU path exists to avoid.
        """
        import numpy as np

        # The extension hands back raw bytes, not a list of floats: a list cost
        # one Python object per element and dominated the readback benchmark it
        # was meant to price. `frombuffer` is a reinterpret, not a conversion.
        flat = np.frombuffer(self._impl.read_back(), dtype=np.float32)
        return flat.reshape(1, 3, self.out_height, self.out_width)

    @property
    def shape(self):
        """Tensor shape as (1, 3, H, W)."""
        return tuple(self._impl.shape)

    @property
    def output_buffer_address(self) -> int:
        """GPU buffer address, for DirectML binding (milestone 3b)."""
        return int(self._impl.output_buffer_address)

    def __repr__(self) -> str:
        return f"<GpuPreprocessor -> {self.shape}>"


def probe_d3d12_sharing(frame) -> Dict[str, Any]:
    """
    Test whether a captured frame can be opened on a D3D12 device.

    DirectML runs on D3D12, so this is the precondition for zero-copy inference.
    Each step of the chain reports separately, so a failure identifies which
    link broke rather than just that the chain did.

    Returns a dict containing at least ``zero_copy_possible``; on failure it also
    carries ``failed_at``, ``error`` and an ``interpretation`` describing the
    fallback.
    """
    return dict(require().probe_d3d12_sharing(_texture_address(frame)))


class GpuPreprocessor12:
    """
    Like :class:`GpuPreprocessor`, but the tensor lands on the **DirectML device**.

    The D3D11 version produces a correct tensor that DirectML can never reach:
    D3D11 shares only 2D textures, never buffers, so a D3D11-written buffer has
    no route to a D3D12 device. Running the same shader on D3D12 removes that
    problem — the captured texture *is* shareable, and the output buffer is then
    already resident where DirectML binds.

    Use this when the tensor is destined for inference; use the D3D11 version
    when you just want GPU-side preprocessing.

    Args:
        frame: A live Frame, used to pick the adapter and validate shareability.
        out_width / out_height: Model input size.

    Raises:
        RuntimeError: If the frame's texture is not shareable — checked at
            construction rather than on the first dispatch.
    """

    def __init__(self, frame, out_width: int, out_height: int) -> None:
        ext = require()
        self._impl = ext.GpuPreprocessor12(
            _texture_address(frame), int(out_width), int(out_height)
        )
        self.out_width = int(out_width)
        self.out_height = int(out_height)

    def process(self, frame, scale: float = 1.0, bias: float = 0.0,
                bgr: bool = False) -> None:
        """Convert one frame. The result stays on the DirectML device."""
        # The texture address alone is not an identity -- COM addresses get
        # recycled -- so the frame's source_id goes with it. See Frame.source_id.
        self._impl.process(_texture_address(frame), scale, bias, bgr,
                           source_id=getattr(frame, "source_id", 0))

    def read_back(self):
        """Copy the tensor to the CPU as (1, 3, H, W). Verification only."""
        import numpy as np

        # Raw bytes rather than a list of floats — see the note on the D3D11
        # preprocessor's read_back.
        flat = np.frombuffer(self._impl.read_back(), dtype=np.float32)
        return flat.reshape(1, 3, self.out_height, self.out_width)

    @property
    def shape(self):
        return tuple(self._impl.shape)

    @property
    def shared_output_handle(self) -> int:
        """
        Shared NT handle for the tensor, for a consumer on another device or API.

        This is what CUDA's ``cudaImportExternalMemory`` takes with
        ``cudaExternalMemoryHandleTypeD3D12Resource``. Pair it with
        :attr:`output_byte_size`, which is the size such an importer must map.

        **Borrowed, not owned.** It is closed when this preprocessor is dropped;
        do not close it yourself, and do not use it after that point.
        """
        return int(self._impl.shared_output_handle)

    @property
    def output_byte_size(self) -> int:
        """Size of the tensor in bytes — what an importer must map."""
        return int(self._impl.output_byte_size)

    @property
    def adapter_luid(self) -> bytes:
        """
        LUID of the adapter holding the tensor, as 8 little-endian bytes.

        The tensor is not cross-adapter, so only a consumer on *this* adapter
        can import it. CUDA exposes the same identity via ``cuDeviceGetLuid``,
        which is how a caller picks the right device — and how it finds out
        there isn't one. That is the ordinary hybrid-laptop case: capture runs
        on the iGPU, CUDA reports exactly one device, and it is the wrong one,
        so counting devices cannot detect the mismatch.
        """
        return bytes(self._impl.adapter_luid)

    @property
    def output_resource_address(self) -> int:
        """
        Address of the ``ID3D12Resource`` holding the tensor.

        This is what ``OrtDmlApi::CreateGPUAllocationFromD3DResource`` takes.
        """
        return int(self._impl.output_resource_address)

    @property
    def output_gpu_address(self) -> int:
        """GPU virtual address of the tensor buffer."""
        return int(self._impl.output_gpu_address)

    def __repr__(self) -> str:
        return f"<GpuPreprocessor12 -> {self.shape} on the DirectML device>"


def probe_onnxruntime(dll_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Check that ONNX Runtime is reachable from the native shim.

    Loads ``onnxruntime.dll`` at runtime rather than linking against it, so ORT
    stays an optional dependency.

    **Pass an explicit path.** Resolving by name uses the DLL search path, which
    frequently finds an unrelated ONNX Runtime installed by some other
    application — on this machine that resolves to 1.17.1 while the Python
    package ships 1.24.4. Since the C API's struct layout is version-dependent,
    silently binding to the wrong runtime is a real hazard. Locate the DLL via
    ``onnxruntime.__file__`` instead.

    Returns:
        ``loaded``, ``version``, ``max_api_version`` and
        ``supported_api_versions``; on failure, ``error`` and a ``hint``.
    """
    return dict(require().probe_onnxruntime(dll_path))


def onnxruntime_dll_path() -> Optional[str]:
    """
    Locate the ``onnxruntime.dll`` that ships with the installed Python package.

    Returns None if onnxruntime is not installed.
    """
    try:
        import onnxruntime
    except ImportError:
        return None
    from pathlib import Path

    package = Path(onnxruntime.__file__).parent
    candidate = package / "capi" / "onnxruntime.dll"
    if candidate.exists():
        return str(candidate)
    found = list(package.rglob("onnxruntime.dll"))
    return str(found[0]) if found else None


def probe_shareable_buffers() -> Dict[str, Any]:
    """
    Report which D3D11 buffer configurations can be shared with D3D12.

    A diagnostic kept because its answer is surprising and load-bearing: on
    Windows, **no** D3D11 buffer configuration is shareable. Only 2D
    non-mipmapped textures can be shared, so a tensor written by a D3D11 compute
    shader cannot be handed to DirectML — the conversion has to happen on the
    D3D12 device instead.

    Returns a dict with ``d3d12_available`` and a ``candidates`` mapping, each
    entry reporting how far that configuration got (created / shared_handle /
    opened_on_d3d12) and a ``usable`` verdict.
    """
    return dict(require().probe_shareable_buffers())


def probe_cross_adapter(
    width: int = 1920, height: int = 1080, iterations: int = 50
) -> Dict[str, Any]:
    """
    Report whether a frame can be moved to a second adapter, and what it costs.

    On a hybrid-GPU laptop Desktop Duplication only runs against the adapter
    driving the display, so a GPU-resident frame is produced on the iGPU while
    the model usually lives on the dGPU. Getting it across means a cross-adapter
    shared heap.

    Two corrections worth carrying: the shared heap lives in **system memory**,
    not either adapter's VRAM — this is not peer-to-peer VRAM-to-VRAM DMA — and
    the mechanism is ``D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER``, not
    ``IDXGIAdapter3`` (which is video-memory budgeting).

    The timing covers the half Rapidshot owns: the copy from a GPU-local texture
    on the capture adapter into the shared heap. Check ``representative`` before
    quoting the number — it is False when the only available second adapter is
    WARP, which proves the mechanism works but says nothing about the cost of a
    real iGPU-to-dGPU transfer.

    Returns a dict with ``adapters``, ``supported``, ``representative``,
    ``copy_ms_min`` / ``copy_ms_median``, ``throughput_mb_s``, and the
    ``CrossAdapterRowMajorTextureSupported`` capability for both devices. When
    sharing is not possible, ``supported`` is False and ``reason`` says why.
    """
    return dict(
        require().probe_cross_adapter(
            width=width, height=height, iterations=iterations
        )
    )


def _source_id(frame) -> int:
    """Which duplicator produced this frame, for cache keys.

    A texture address alone is not an identity: COM addresses are recycled, so
    a released surface and a later unrelated one can share a pointer. See
    ``Frame.source_id``.
    """
    return int(getattr(frame, "source_id", 0))


class CrossAdapterTransfer:
    """Carries captured frames to a second GPU (ROADMAP.md 6.1).

    On a hybrid laptop Desktop Duplication only runs against the adapter that
    drives the display, so a GPU-resident frame lands on the iGPU while the
    model usually lives on the dGPU. This moves it across through a shared
    cross-adapter heap, with no CPU round-trip.

    Build one per capture session and reuse it. The heap and both placed
    resources are allocated up front, because none of them depend on the frame;
    only the copy is per-frame work.

        transfer = native.cross_adapter_transfer(frame)
        with camera.grab_frame() as frame:
            transfer.transfer(frame)
            # consume transfer.destination_resource_address on the other adapter

    The frame passed to :meth:`transfer` must have the same dimensions and DXGI
    format as the one the transfer was built from; rebuild it after a
    resolution or SDR/HDR mode change.

    Pixels cross losslessly in their source format; this layer does not convert
    HDR or 10-bit content to BGRA8. Inspect :attr:`dxgi_format` and
    :attr:`bytes_per_pixel` before interpreting the destination buffer.

    Note that the shared heap lives in **system memory**, not either adapter's
    VRAM. This is not peer-to-peer VRAM-to-VRAM DMA — the win is that a GPU copy
    engine moves the bytes instead of CPU cores.
    """

    def __init__(self, frame):
        self._inner = require().CrossAdapterTransfer(_texture_address(frame))

    def transfer(self, frame) -> None:
        """Copy one frame across.

        Blocks until the source GPU has finished, so the frame is readable from
        the destination adapter when this returns.
        """
        self._inner.transfer(_texture_address(frame), _source_id(frame))

    def read_back_destination(self) -> bytes:
        """Read the frame back through the destination device.

        Verification only — in production, bind
        :attr:`destination_resource_address` on that adapter instead. Rows are
        :attr:`row_pitch` bytes apart, which is padded to D3D12's 256-byte copy
        alignment and so is not always ``width * bytes_per_pixel``. Pixels stay
        in :attr:`dxgi_format`; this method performs no colour conversion.
        """
        return bytes(self._inner.read_back_destination())

    def transfer_with_reference(self, frame) -> bytes:
        """Transfer, and return a source-side copy of the same bytes.

        Verification only. Both copies come from one snapshot of the frame taken
        in a single command list, because the duplicated surface is live: copies
        submitted separately genuinely observe different pixels.
        """
        return bytes(self._inner.transfer_with_reference(
            _texture_address(frame), _source_id(frame)))

    def transfer_async(self, frame) -> int:
        """Submit a transfer without blocking; returns the fence value to await.

        :meth:`transfer` blocks until the copy completes, and on a hybrid
        laptop that is most of its cost -- measured 2026-08-22 at 2.67 ms of a
        2.70 ms transfer (Intel iGPU to RTX 4060, 2560x1600). This hands the
        calling thread back instead and leaves synchronisation to you::

            value = transfer.transfer_async(frame)
            ...                                   # your work overlaps the copy
            transfer.wait_shared_fence(value)     # or wait GPU-side, below

        Measured against the blocking path with 2 ms of consumer work per
        frame: **wall clock 6.22 -> 3.59 ms (42% faster), and 98% of
        calling-thread time returned.** The gain is real overlap, not
        bookkeeping -- the copy runs while the caller works.

        It still waits for the *previous* submission before recording, because
        the command allocator cannot be reset while the GPU is reading it, so
        this pipelines to depth one. Frame N's copy overlaps whatever you do
        next; you pay at the start of frame N+1 only if the GPU has not
        finished.

        A consumer on the destination adapter can skip the CPU round-trip
        entirely by opening :attr:`shared_fence_handle` and waiting on it from
        its own queue.

        .. warning::

            **Every transfer reuses one destination buffer, and this fence
            says nothing about the consumer.** It reports that the *copy*
            finished. If your consumer is still reading frame N when you submit
            frame N+1, the copy overwrites the buffer underneath it and the
            consumer silently reads a mixture -- nothing raises.

            Measured 2026-08-22 over a 60-frame loop whose consumer was slower
            than the producer: **28 of 60 frames were wrong** without a
            handshake, and 0 with one. With a fast consumer the same loop shows
            no corruption at all over 100 frames, which is why this is easy to
            miss until a real workload arrives.

            If your consumer is asynchronous, use :meth:`set_consumer_fence`
            and :meth:`wait_for_consumer`. If it synchronises on the CPU
            between frames, you do not need them.
        """
        try:
            value = int(self._inner.transfer_async(
                _texture_address(frame), _source_id(frame)))
        except Exception as exc:
            # ExecuteCommandLists has no return value, so a later Signal failure
            # can leave real GPU work with no completion marker. The native
            # layer first tries a private fallback fence and checks for device
            # removal. Only the remaining, genuinely untrackable case reports
            # `submission_quarantined`.
            try:
                quarantined = bool(self._inner.submission_quarantined)
            except Exception:
                quarantined = False
            if quarantined:
                quarantine = getattr(frame, "_quarantine_release", None)
                if quarantine is not None:
                    quarantine(str(exc))
            raise
        # Keep the captured surface acquired until this copy finishes.
        #
        # The duplicated surface is only valid between AcquireNextFrame and
        # ReleaseFrame, and this call returns while the GPU is still reading
        # it. The documented idiom -- `with camera.grab_frame() as frame:` --
        # releases at scope exit, so without this the surface goes back to DXGI
        # mid-copy, DXGI recycles it, and the destination silently receives a
        # blend of two frames. Nothing raises; the pixels are just wrong.
        #
        # Releasing therefore waits for this fence. That does not undo the
        # async win: the calling thread is free between submit and release, and
        # a consumer that waits GPU-side on `shared_fence_handle` pays nothing
        # extra, because by the time it releases the copy has long finished.
        defer = getattr(frame, "defer_release_until", None)
        if defer is not None:
            defer(lambda: self.wait_shared_fence(value),
                  quarantine_on_failure=True)
        return value

    def wait_shared_fence(self, value: int) -> None:
        """Block until the shared fence reaches ``value``. 0 returns at once."""
        self._inner.wait_shared_fence(int(value))

    def set_consumer_fence(self, handle: int) -> None:
        """Adopt the consumer's fence so the producer can wait on it.

        `handle` is a shared NT handle for a D3D12 fence that the consumer
        signals once it has finished reading the destination buffer. A CUDA
        consumer produces one by importing this transfer's own fence style --
        ``cuImportExternalSemaphore`` then ``cuSignalExternalSemaphoresAsync``
        -- which is verified to work across vendors here: CUDA on an NVIDIA
        dGPU signalled a fence created by an Intel iGPU's D3D12 device and the
        producer observed it.

        Opened once on the source device; calling again replaces it.
        """
        self._inner.set_consumer_fence(int(handle))

    def wait_for_consumer(self, value: int) -> None:
        """Make the next copy wait until the consumer has reached `value`.

        **Every transfer reuses one destination buffer.** The producer fence
        only says "the copy finished"; it says nothing about whether the
        consumer is still reading. Without this, an asynchronous consumer that
        waits GPU-side on :attr:`shared_fence_handle` can still be reading
        frame N when the copy for frame N+1 overwrites the allocation under it
        -- the two frames blend, and nothing raises.

        The wait is queued on the source queue, so it orders ahead of the next
        copy without blocking the calling thread. The loop is::

            transfer.set_consumer_fence(consumer_handle)   # once
            v = transfer.transfer_async(frame)             # frame N
            # consumer waits for v GPU-side, reads, signals consumer fence = N
            transfer.wait_for_consumer(N)                  # before frame N+1
            v = transfer.transfer_async(next_frame)

        A consumer that synchronises on the CPU between frames does not need
        this; one that stays asynchronous does.
        """
        self._inner.wait_for_consumer(int(value))

    @property
    def shared_fence_completed(self) -> int:
        """Value the shared fence has actually reached on the GPU.

        Diagnostic. :attr:`shared_fence_submitted` is what was handed to the
        queue; this is what has completed.
        """
        return int(self._inner.shared_fence_completed)

    @property
    def shared_fence_submitted(self) -> int:
        """Highest value submitted to the shared fence so far."""
        return int(self._inner.shared_fence_submitted)

    @property
    def shared_fence_handle(self) -> int:
        """NT handle for the cross-adapter fence, for a GPU-side wait.

        Open it on the destination adapter with
        ``ID3D12Device::OpenSharedHandle`` and wait on it from that queue, so
        the copy and the consuming work overlap with no CPU involvement. The
        fence is created with ``SHARED | SHARED_CROSS_ADAPTER``, which is the
        only configuration both adapters can observe.

        Waiting on this tells you the copy landed. It does **not** coordinate
        the other direction: see the warning on :meth:`transfer_async` about
        the shared destination buffer, and :meth:`wait_for_consumer` for the
        return path.

        Borrowed, like :attr:`shared_destination_handle`: closed when this
        transfer is dropped, and likewise **not yours to close**.
        """
        return int(self._inner.shared_fence_handle)

    @property
    def cached_texture_address(self) -> int:
        """Raw pointer of the capture texture currently cached, or 0.

        Opening the captured texture is cached per texture rather than redone
        per frame. Measured 2026-08-22, Intel iGPU to RTX 4060: that removed
        268 us of opening and 114 us of closing from every transfer, 10.3% of
        the whole. Keyed on the pointer, so a changed surface reopens.

        Exposed because the cache produces identical output either way, so a
        change that silently disabled it would be a large regression with no
        visible symptom -- tests assert on this key, not on pixels.
        """
        return int(self._inner.cached_texture_address)

    @property
    def cached_source_id(self) -> int:
        """``source_id`` of the cached capture texture, or 0.

        The other half of the cache key. The texture address alone is not an
        identity, so tests assert on both.
        """
        return int(self._inner.cached_source_id)

    @property
    def shared_destination_handle(self) -> int:
        """NT handle for the destination heap, for a consumer on that adapter.

        This is what makes the Optimus path complete. Capture runs on the
        integrated GPU, so the Stage 6 tensor lands on an adapter CUDA cannot
        see (ROADMAP.md 6.1); the frame crosses here, and this handle is how a
        consumer on the destination adapter gets at it.

        **Import it as a heap, not a resource.** The transferred buffers are
        *placed* resources and cannot be shared at all -- ``CreateSharedHandle``
        refuses them with E_INVALIDARG on both devices, which
        ``probe_shared_handles()`` demonstrates. So the shared object is the
        heap, and CUDA imports it as
        ``CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`` (4), size
        :attr:`total_bytes`, offset 0. This differs from
        :attr:`GpuPreprocessor12.shared_output_handle`, which is a committed
        resource and imports as type 5.

        **Borrowed, not owned. Do not close it.** This transfer closes it on
        drop, and Windows recycles handle *values* -- so closing it yourself
        leaves the transfer holding a number that may by then name an unrelated
        file, event or socket, which it will duly close later. Import from it
        and leave it alone.

        A consumer holding the integer, or a device pointer mapped from it, has
        nothing that looks wrong once the transfer is gone, so keep the
        transfer alive for as long as anything reads the frame. If you need an
        independently owned handle, duplicate it with ``DuplicateHandle``.

        Verified 2026-08-22 on an Intel iGPU -> RTX 4060 pair: the imported
        buffer reads byte-identical to :meth:`read_back_destination`, and
        observes later :meth:`transfer` calls without re-importing.
        """
        return int(self._inner.shared_destination_handle)

    def probe_shared_handles(self) -> list:
        """Which objects on this transfer will yield a shareable NT handle.

        Diagnostic, for bringing up a new adapter pairing: the answer is a
        property of the driver pair, and this project has verified exactly one
        (Intel iGPU to NVIDIA dGPU). Run it before assuming
        :attr:`shared_destination_handle` behaves the same elsewhere.

        Leaks nothing -- every handle it mints is closed before returning, so
        the result reports whether each call is permitted rather than handing
        back handles to clean up.

        Returns a list of dicts with ``label``, ``cuda_handle_type``
        (4 = D3D12_HEAP, 5 = D3D12_RESOURCE), ``ok``, and ``error`` when not ok.
        """
        return [dict(row) for row in self._inner.probe_shared_handles()]

    @property
    def destination_resource_address(self) -> int:
        """Address of the ``ID3D12Resource`` on the destination adapter.

        Borrowed, not owned: valid only while this object is alive.
        """
        return int(self._inner.destination_resource_address)

    @property
    def destination_device_address(self) -> int:
        return int(self._inner.destination_device_address)

    @property
    def source(self) -> str:
        return self._inner.source

    @property
    def destination(self) -> str:
        return self._inner.destination

    @property
    def destination_is_software(self) -> bool:
        """True when the only second adapter is WARP.

        The path is exercised in full, but timings from it say nothing about a
        real iGPU-to-dGPU move.
        """
        return bool(self._inner.destination_is_software)

    @property
    def total_bytes(self) -> int:
        return int(self._inner.total_bytes)

    @property
    def dxgi_format(self) -> int:
        """Numeric DXGI_FORMAT of the raw pixels in the shared destination."""
        return int(self._inner.dxgi_format)

    @property
    def bytes_per_pixel(self) -> int:
        """Storage bytes per pixel for :attr:`dxgi_format`."""
        return int(self._inner.bytes_per_pixel)

    @property
    def row_pitch(self) -> int:
        return int(self._inner.row_pitch)

    @property
    def width(self) -> int:
        return int(self._inner.width)

    @property
    def height(self) -> int:
        return int(self._inner.height)

    def __repr__(self) -> str:
        return (f"<CrossAdapterTransfer {self.width}x{self.height} "
                f"{self.source!r} -> {self.destination!r}>")


def cross_adapter_transfer(frame) -> CrossAdapterTransfer:
    """Build a :class:`CrossAdapterTransfer` for frames shaped like this one.

    Raises RuntimeError if this system has only one adapter, since there is then
    nowhere to transfer to. Check ``rapidshot.topology_info()`` first if you
    need to branch on that rather than handle an exception.
    """
    return CrossAdapterTransfer(frame)


def device_address(frame) -> int:
    """
    Address of the ``ID3D11Device`` that owns a frame's texture.

    Borrowed, not owned: valid only while the capture session is alive.
    """
    return int(require().get_device_pointer(_texture_address(frame)))


def _texture_address(frame) -> int:
    """
    Integer address of a Frame's ID3D11Texture2D.

    Accessing ``frame.d3d11_texture`` raises if the frame was released, which is
    what we want — the pointer would otherwise be dangling by the time it
    reached Rust.
    """
    import ctypes

    texture = frame.d3d11_texture
    address = ctypes.cast(texture, ctypes.c_void_p).value
    if not address:
        raise ValueError("Frame's texture pointer is null")
    return int(address)
