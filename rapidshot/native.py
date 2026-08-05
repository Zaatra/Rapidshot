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

        flat = np.asarray(self._impl.read_back(), dtype=np.float32)
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
        self._impl.process(_texture_address(frame), scale, bias, bgr)

    def read_back(self):
        """Copy the tensor to the CPU as (1, 3, H, W). Verification only."""
        import numpy as np

        flat = np.asarray(self._impl.read_back(), dtype=np.float32)
        return flat.reshape(1, 3, self.out_height, self.out_width)

    @property
    def shape(self):
        return tuple(self._impl.shape)

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

    The frame passed to :meth:`transfer` must have the same dimensions as the
    one the transfer was built from; rebuild it after a resolution change.

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
        self._inner.transfer(_texture_address(frame))

    def read_back_destination(self) -> bytes:
        """Read the frame back through the destination device.

        Verification only — in production, bind
        :attr:`destination_resource_address` on that adapter instead. Rows are
        :attr:`row_pitch` bytes apart, which is padded to D3D12's 256-byte copy
        alignment and so is not always ``width * 4``.
        """
        return bytes(self._inner.read_back_destination())

    def transfer_with_reference(self, frame) -> bytes:
        """Transfer, and return a source-side copy of the same bytes.

        Verification only. Both copies come from one snapshot of the frame taken
        in a single command list, because the duplicated surface is live: copies
        submitted separately genuinely observe different pixels.
        """
        return bytes(self._inner.transfer_with_reference(_texture_address(frame)))

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
