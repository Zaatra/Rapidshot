"""Capture straight into a CuPy tensor, with no CPU round-trip.

    python examples/gpu_tensor_to_cupy.py

RapidShot produces an `ID3D12Resource` holding a model-ready NCHW float32
tensor and a shared NT handle for it. This example is the ~60 lines of ctypes
that turn that handle into a `cupy.ndarray`. It lives in `examples/` rather
than in the package on purpose: RapidShot produces frames and does not own its
consumers' bindings (ROADMAP § 11). Copy it, do not import it.

Why the CUDA *driver* API rather than the runtime API: `nvcuda.dll` ships with
the display driver and is always present, whereas the runtime lives in a
versioned `cudart64_*.dll` you would have to locate. CuPy drives a primary
context on the driver API anyway, so the imported pointer lands in the context
CuPy is already using.

Requires: an NVIDIA GPU, CuPy, and the RapidShot native extension.
"""
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cupy as cp
except ImportError:
    sys.exit("this example needs CuPy:  pip install cupy-cuda13x")

# Running `python examples/gpu_tensor_to_cupy.py` puts *examples/* on the path,
# not the repo root, so the import below fails on a source checkout even though
# the package is right there -- the same line the other scripts use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rapidshot  # noqa: E402
from rapidshot import native  # noqa: E402

# --------------------------------------------------------------------------
# CUDA external-memory import
# --------------------------------------------------------------------------

CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1


class _Win32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


class _HandleUnion(ctypes.Union):
    _fields_ = [("fd", ctypes.c_int), ("win32", _Win32Handle),
                ("nvSciBufObject", ctypes.c_void_p)]


class ExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _HandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class ExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class CudaTensor:
    """A `cupy.ndarray` view of a RapidShot GPU tensor.

    The import happens once. After that the device pointer stays valid for the
    preprocessor's lifetime, so a capture loop pays nothing per frame to keep
    the view — `process()` overwrites the same buffer the CuPy array points at.
    """

    def __init__(self, preprocessor, shape, device=None):
        self._cuda = ctypes.WinDLL("nvcuda.dll")
        self._cuda.cuMemFree.argtypes = [ctypes.c_ulonglong]
        self._ext = ctypes.c_void_p()
        self._device_ptr = None
        # Hold the preprocessor. It owns the D3D12 resource this array points
        # into *and* the shared handle, both released when it is collected --
        # so without this reference `CudaTensor(native.GpuPreprocessor12(...))`
        # would leave `self.array` addressing freed VRAM as soon as the
        # temporary went out of scope. The handle is an integer; nothing about
        # it would look wrong afterwards.
        self._preprocessor = preprocessor

        # The tensor is NOT cross-adapter: it can only be imported by the CUDA
        # device that owns the D3D12 adapter which captured the frame. Assuming
        # ordinal 0 is wrong on a machine with more than one CUDA GPU, and the
        # symptom is an opaque cuImportExternalMemory failure -- so refuse the
        # ambiguous case rather than guess it.
        #
        # Matching properly means comparing the D3D12 adapter's LUID against
        # cuDeviceGetLuid for each device. That needs the adapter LUID exposed
        # from the extension, and hybrid hardware to test it on; see ROADMAP
        # section 6.1.
        if device is None:
            count = cp.cuda.runtime.getDeviceCount()
            if count != 1:
                raise RuntimeError(
                    f"{count} CUDA devices are present, so which one owns the "
                    f"captured adapter is ambiguous. Pass device=N explicitly "
                    f"— the import fails opaquely if N is the wrong one.")
            device = 0

        # CuPy's primary context must exist before anything is imported into it.
        cp.cuda.Device(device).use()
        cp.zeros(1)

        desc = ExternalMemoryHandleDesc()
        desc.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        desc.handle.win32.handle = ctypes.c_void_p(
            preprocessor.shared_output_handle)
        desc.handle.win32.name = None
        desc.size = preprocessor.output_byte_size
        # A committed D3D12 resource is a dedicated allocation, and CUDA rejects
        # the import without this flag — with a bare INVALID_VALUE that names no
        # field, so it is an expensive one to omit.
        desc.flags = CUDA_EXTERNAL_MEMORY_DEDICATED

        self._check(
            self._cuda.cuImportExternalMemory(
                ctypes.byref(self._ext), ctypes.byref(desc)),
            "cuImportExternalMemory")

        buf = ExternalMemoryBufferDesc()
        buf.offset = 0
        buf.size = preprocessor.output_byte_size
        buf.flags = 0

        ptr = ctypes.c_ulonglong()
        self._check(
            self._cuda.cuExternalMemoryGetMappedBuffer(
                ctypes.byref(ptr), self._ext, ctypes.byref(buf)),
            "cuExternalMemoryGetMappedBuffer")

        # `UnownedMemory` because the allocation belongs to D3D12 and CuPy must
        # never free it. `owner=self` is what keeps the chain alive: the array
        # holds the memory, which holds this object, which holds both the CUDA
        # external-memory handle and the preprocessor. Hand back an array whose
        # owner is None and the caller can drop everything that keeps its
        # storage mapped while still holding a perfectly normal-looking array.
        # CUDA requires this pointer be released with cuMemFree *before* the
        # external-memory object is destroyed; destroying the object does not
        # release the mapping. Held so close() can do that in the right order.
        self._device_ptr = ptr.value

        memory = cp.cuda.UnownedMemory(
            ptr.value, preprocessor.output_byte_size, owner=self)
        self.array = cp.ndarray(
            shape, dtype=cp.float32, memptr=cp.cuda.MemoryPointer(memory, 0))
        self.device_ptr = ptr.value

    def _check(self, code, what):
        if code != 0:
            name = ctypes.c_char_p()
            self._cuda.cuGetErrorName(code, ctypes.byref(name))
            raise RuntimeError(
                f"{what} failed: {code} "
                f"({name.value.decode() if name.value else '?'})")

    def close(self):
        """Release the CUDA mapping.

        `self.array` must not be read afterwards: the mapping it addresses is
        gone. It will very likely keep returning the right pixels anyway, until
        something else claims that memory — so this is not a mistake testing
        will catch for you.
        """
        # Order matters: free the mapping, then destroy the object that owns it.
        if self._device_ptr is not None:
            self._cuda.cuMemFree(ctypes.c_ulonglong(self._device_ptr))
            self._device_ptr = None
        if self._ext:
            self._cuda.cuDestroyExternalMemory(self._ext)
            self._ext = ctypes.c_void_p()
        self.array = None

    def __del__(self):
        # The documented one-liner keeps only `.array`, which holds this object
        # alive through UnownedMemory(owner=self) but gives the caller nothing
        # to close. Without this, dropping the array would leak the import.
        try:
            self.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------
# capture -> tensor -> CuPy
# --------------------------------------------------------------------------

OUT = 640


def main() -> int:
    if not native.is_available():
        print("the native extension is not built; see ROADMAP § 2")
        return 1

    camera = rapidshot.create(output_color="BGRA")
    try:
        # Desktop Duplication only reports changed content, so an idle screen
        # produces nothing. Poll rather than assume the first call succeeds.
        frame = None
        for _ in range(600):
            frame = camera.grab_frame()
            if frame is not None:
                break
        if frame is None:
            print("no frame captured — the screen must be changing")
            return 1

        print(f"captured {frame.width}x{frame.height}")

        pre = native.GpuPreprocessor12(frame, OUT, OUT)
        pre.process(frame)

        with CudaTensor(pre, (1, 3, OUT, OUT)) as view:
            tensor = view.array
            print(f"tensor  {tensor.shape} {tensor.dtype} "
                  f"on {tensor.device} at 0x{view.device_ptr:x}")

            # The check that matters. Shape and dtype would agree even if the
            # import had mapped unrelated memory; the pixels would not.
            reference = pre.read_back()
            if not np.array_equal(cp.asnumpy(tensor), reference):
                print("MISMATCH: the CuPy view disagrees with read_back()")
                return 1
            print("verified: CuPy view is byte-identical to read_back()")

            # --- what a capture loop actually costs -------------------------
            def timed(fn, reps=100):
                fn()
                xs = []
                for _ in range(reps):
                    t0 = time.perf_counter()
                    fn()
                    xs.append((time.perf_counter() - t0) * 1e3)
                xs.sort()
                return xs[0], xs[len(xs) // 2]

            def to_gpu_tensor():
                pre.process(frame)

            def gpu_then_touch():
                # Include a CUDA kernel so the timing covers a consumer really
                # reading the tensor, not just the dispatch that produced it.
                pre.process(frame)
                float(tensor.sum())

            d_min, d_p50 = timed(to_gpu_tensor)
            t_min, t_p50 = timed(gpu_then_touch, 50)

            print()
            print(f"  dispatch only          min {d_min:6.3f} ms  p50 {d_p50:6.3f} ms")
            print(f"  dispatch + CUDA reduce min {t_min:6.3f} ms  p50 {t_p50:6.3f} ms")
            print()
            print("  The tensor never touches the CPU. The import is paid once;")
            print("  per frame the CuPy array simply sees the new contents.")

        frame.release()
        return 0
    finally:
        camera.release()


if __name__ == "__main__":
    raise SystemExit(main())
