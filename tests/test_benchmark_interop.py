"""The CUDA interop plumbing the section 7 benchmarks bind through.

Two small pieces of ctypes stand between a D3D12 resource and a CUDA consumer,
and both fail the same expensive way: CUDA rejects a mis-described import with
a bare `INVALID_VALUE` that names no field. The example this code was adapted
from says so in a comment, having paid for it. So what is asserted here is the
*description* -- handle type, dedication flag, element count, which handle --
because those are what a driver silently refuses.

`CudaFence` additionally owns a lifetime rule that is not visible in its
signature: the NT handle is **borrowed**, the owner keeps it, and destroying
the semaphore before the streams using it have drained is a use-after-free the
driver will not diagnose. `close()` synchronises first for that reason, and a
reordering there would pass every functional test while corrupting frames under
load.

Faked rather than run against a driver: these are the paths where a *refusal*
is the thing worth asserting, and a real `nvcuda.dll` raises from inside itself
long before this code can be checked.
"""
import ctypes
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import cuda_semaphore
import section7_adapters


# -- CudaFence -----------------------------------------------------------


class FakeCuda:
    """Records what the driver was asked to do, and can be told to refuse."""

    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on
        self.cuImportExternalSemaphore = self._make("import")
        self.cuWaitExternalSemaphoresAsync = self._make("wait")
        self.cuDestroyExternalSemaphore = self._make("destroy")

    def _make(self, name):
        def call(*args):
            self.calls.append((name, args))
            if name == "import":
                args[0]._obj.value = 0xFE0000
            return 1 if name == self.fail_on else 0
        call.argtypes = None
        return call


@pytest.fixture
def fence_env(monkeypatch):
    cuda = FakeCuda()
    monkeypatch.setattr(cuda_semaphore.c, "WinDLL", lambda name: cuda, raising=False)
    events = []
    stream = SimpleNamespace(ptr=0xABC)
    cp = SimpleNamespace(cuda=SimpleNamespace(
        get_current_stream=lambda: stream,
        runtime=SimpleNamespace(deviceSynchronize=lambda: events.append("sync"))))
    owner = SimpleNamespace(shared_fence_handle=0x5150)
    return SimpleNamespace(cuda=cuda, cp=cp, owner=owner, events=events)


def test_the_fence_is_imported_as_a_d3d12_fence(fence_env):
    """Type 4 is CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE. Any other value
    describes a different object and is refused without saying which field."""
    cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    (name, args), = fence_env.cuda.calls
    assert name == "import"
    desc = args[1]._obj
    assert desc.type == 4
    assert desc.handle.win32.handle == 0x5150


def test_a_refused_import_raises_rather_than_returning_a_null_fence(fence_env,
                                                                   monkeypatch):
    """A null semaphore that waits on nothing would make every frame look
    perfectly synchronised."""
    monkeypatch.setattr(cuda_semaphore.c, "WinDLL",
                        lambda name: FakeCuda(fail_on="import"), raising=False)
    with pytest.raises(RuntimeError, match="external semaphore error 1"):
        cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)


def test_wait_targets_the_value_and_the_current_stream(fence_env):
    """The wait is only meaningful against the stream the consumer will use."""
    fence = cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    fence.wait(37)
    name, args = fence_env.cuda.calls[-1]
    assert name == "wait"
    assert args[1]._obj.params.fence.value == 37
    assert args[2] == 1
    assert args[3].value == 0xABC


def test_a_refused_wait_is_not_swallowed(fence_env, monkeypatch):
    monkeypatch.setattr(cuda_semaphore.c, "WinDLL",
                        lambda name: FakeCuda(fail_on="wait"), raising=False)
    fence = cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    with pytest.raises(RuntimeError, match="external semaphore error 1"):
        fence.wait(1)


def test_close_drains_before_destroying(fence_env):
    """The lifetime rule. Destroying a semaphore a stream is still waiting on
    is a use-after-free the driver does not report -- it corrupts frames."""
    fence = cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    fence.close()
    assert fence_env.events == ["sync"]
    assert fence_env.cuda.calls[-1][0] == "destroy"


def test_close_is_idempotent_and_releases_the_owner(fence_env):
    """Cleanup runs from `finally` blocks that can be reached twice, and the
    owner reference must go so the borrowed handle is not kept alive."""
    fence = cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    fence.close()
    fence.close()
    assert [c[0] for c in fence_env.cuda.calls].count("destroy") == 1
    assert fence.owner is None


def test_close_never_closes_the_borrowed_handle(fence_env):
    """The owner retains it. Closing it here would invalidate the handle the
    transfer still holds, for everyone."""
    fence = cuda_semaphore.CudaFence(fence_env.owner, fence_env.cp)
    handle = fence_env.owner.shared_fence_handle
    fence.close()
    assert fence_env.owner.shared_fence_handle == handle


# -- Adapter._heap_view --------------------------------------------------


def test_a_transferred_heap_is_described_as_a_heap_not_a_resource():
    """Type 4 is D3D12_HEAP; 5 is D3D12_RESOURCE. A `TensorTransfer` destination
    is a heap, and the dedication flag has to match: a committed resource is a
    dedicated allocation and a heap is not. Get either wrong and the import
    fails on some drivers and succeeds on others."""
    captured = {}

    def CudaTensor(owner, shape, device=None):
        captured.update(owner=owner, shape=shape, device=device)
        return "view"

    transfer = SimpleNamespace(shared_destination_handle=0x1234,
                               total_bytes=2_457_600)
    adapter = section7_adapters.Adapter.__new__(section7_adapters.Adapter)
    assert adapter._heap_view(transfer, CudaTensor) == "view"

    owner = captured["owner"]
    assert owner.cuda_handle_type == 4
    assert owner.cuda_dedicated is False
    assert owner.shared_output_handle == 0x1234
    assert owner.output_byte_size == 2_457_600


def test_the_view_covers_the_whole_payload_as_four_byte_words():
    """CudaTensor maps float32 words and the result is viewed back to its own
    dtype. A shape shorter than the mapping reads part of the tensor; longer
    reads past the end of GPU memory and returns a perfectly ordinary array."""
    captured = {}

    def CudaTensor(owner, shape, device=None):
        captured.update(shape=shape, device=device)

    transfer = SimpleNamespace(shared_destination_handle=1, total_bytes=2_457_600)
    section7_adapters.Adapter.__new__(section7_adapters.Adapter)._heap_view(
        transfer, CudaTensor)
    assert captured["shape"] == (2_457_600 // 4,)
    assert captured["device"] == 0


def test_the_view_owner_keeps_the_transfer_alive():
    """The mapping points into the transfer's heap. If the transfer is
    collected the mapping dangles, so the owner handed to CUDA holds it."""
    held = []

    def CudaTensor(owner, shape, device=None):
        held.append(owner)

    transfer = SimpleNamespace(shared_destination_handle=1, total_bytes=8)
    section7_adapters.Adapter.__new__(section7_adapters.Adapter)._heap_view(
        transfer, CudaTensor)
    assert held[0].transfer is transfer


# -- Captured -------------------------------------------------------------


def test_a_captured_sample_defaults_to_carrying_no_reference():
    """`reference` and `rgb` are only produced under --verify and for the agent
    category. Defaulting them to anything but None would make a measuring run
    look like it had verified something."""
    sample = section7_adapters.Captured(tensor="t", frame_id=7, stages={},
                                        h2d_bytes=0)
    assert sample.reference is None and sample.rgb is None
    assert sample.frame_id == 7 and sample.h2d_bytes == 0
