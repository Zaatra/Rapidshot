"""D3D12 fence -> CUDA stream wait, adapted from test_cross_adapter.py.

ABI: NVIDIA CUDA driver external-resource interoperability. The owner retains
the borrowed NT handle. Close only after every stream using the semaphore drains.
"""
import ctypes as c


class Win32(c.Structure):
    _fields_ = [("handle", c.c_void_p), ("name", c.c_void_p)]


class Handle(c.Union):
    _fields_ = [("fd", c.c_int), ("win32", Win32), ("nvSciSyncObj", c.c_void_p)]


class Desc(c.Structure):
    _fields_ = [("type", c.c_int), ("handle", Handle), ("flags", c.c_uint),
                ("reserved", c.c_uint * 16)]


class Fence(c.Structure):
    _fields_ = [("value", c.c_ulonglong)]


class NvSci(c.Union):
    _fields_ = [("fence", c.c_void_p), ("reserved", c.c_ulonglong)]


class Keyed(c.Structure):
    _fields_ = [("key", c.c_ulonglong), ("timeoutMs", c.c_uint)]


class WaitInner(c.Structure):
    _fields_ = [("fence", Fence), ("nvSciSync", NvSci), ("keyedMutex", Keyed),
                ("reserved", c.c_uint * 10)]


class Wait(c.Structure):
    _fields_ = [("params", WaitInner), ("flags", c.c_uint), ("reserved", c.c_uint * 16)]


class CudaFence:
    def __init__(self, owner, cp):
        self.owner, self.cp = owner, cp
        self.handle = c.c_void_p()
        self.cuda = c.WinDLL("nvcuda.dll")
        self.cuda.cuImportExternalSemaphore.argtypes = [c.POINTER(c.c_void_p), c.POINTER(Desc)]
        self.cuda.cuWaitExternalSemaphoresAsync.argtypes = [c.POINTER(c.c_void_p),
                                                          c.POINTER(Wait), c.c_uint, c.c_void_p]
        self.cuda.cuDestroyExternalSemaphore.argtypes = [c.c_void_p]
        desc = Desc()
        desc.type = 4  # CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        desc.handle.win32.handle = owner.shared_fence_handle
        self.check(self.cuda.cuImportExternalSemaphore(c.byref(self.handle), c.byref(desc)))

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"CUDA external semaphore error {code}")

    def wait(self, value):
        params = Wait()
        params.params.fence.value = value
        self.check(self.cuda.cuWaitExternalSemaphoresAsync(
            c.byref(self.handle), c.byref(params), 1,
            c.c_void_p(self.cp.cuda.get_current_stream().ptr)))

    def close(self):
        if self.handle:
            self.cp.cuda.runtime.deviceSynchronize()
            self.check(self.cuda.cuDestroyExternalSemaphore(self.handle))
            self.handle = c.c_void_p()
            self.owner = None
