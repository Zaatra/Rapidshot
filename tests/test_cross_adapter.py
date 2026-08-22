"""Tests for the cross-adapter transfer API (ROADMAP.md 6.1).

What can be tested here is the contract around the transfer: that it refuses
what it cannot handle, and refuses it with a message that says what to do.

The transfer *itself* cannot be unit tested. It needs a genuinely duplicated
surface: D3D11 refuses SHARED_NTHANDLE without SHARED_KEYEDMUTEX, and a
keyed-mutex resource reads as zeros until acquired, so a synthetic texture
exercises none of the path.

It *can* be tested against live capture, and now is -- see the live section at
the bottom. That was previously only in ``examples/verify_cross_adapter.py``,
run by hand, which ROADMAP section 5 is explicit about: anything not run before
a release is not verified for that release. The live test skips cleanly without
a second adapter or without screen activity, so it costs nothing where it
cannot run and stops the correctness check depending on someone remembering.
"""

import time

import numpy as np
import pytest

import rapidshot
from rapidshot import native
from rapidshot.util.topology import probe_topology

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)


def test_the_api_is_exposed():
    assert hasattr(native, "cross_adapter_transfer")
    assert hasattr(native.require(), "CrossAdapterTransfer")


def test_a_released_frame_is_refused():
    """The texture pointer would be dangling by the time Rust saw it."""
    from rapidshot.frame import Frame, FrameReleasedError

    frame = Frame(texture=object(), on_release=lambda: None, region=(0, 0, 4, 4))
    frame.release()
    with pytest.raises(FrameReleasedError):
        native.cross_adapter_transfer(frame)


def test_a_null_texture_is_refused():
    with pytest.raises(ValueError, match="null"):
        native.require().CrossAdapterTransfer(0)


def test_a_non_shareable_texture_is_refused_at_construction():
    """Fail at setup, not on the first frame.

    A texture that cannot be shared with D3D12 will never work on this path, and
    finding that out while building the transfer is far easier to act on than a
    failure mid-capture.
    """
    import numpy as np

    ext = native.require()
    pattern = np.zeros((16, 16, 4), dtype=np.uint8)
    texture = ext.TestTexture(16, 16, pattern.tobytes())

    with pytest.raises(RuntimeError) as excinfo:
        ext.CrossAdapterTransfer(texture.pointer)

    message = str(excinfo.value)
    assert "cross-adapter setup failed" in message
    # Either this machine has one adapter, or the synthetic texture is not
    # shareable. Both are legitimate; both must say which.
    assert "adapter" in message.lower() or "shar" in message.lower()


def test_single_adapter_systems_get_an_actionable_error():
    """There is nothing to transfer to, and the message must say so.

    Skipped where a second adapter exists — including WARP, which every
    ordinary Windows install has.
    """
    from rapidshot.util.topology import probe_topology

    topology = probe_topology()
    if len(topology.adapters) > 1:
        pytest.skip(f"{len(topology.adapters)} adapters present")

    import numpy as np

    ext = native.require()
    texture = ext.TestTexture(16, 16, np.zeros((16, 16, 4), np.uint8).tobytes())
    with pytest.raises(RuntimeError, match="no second adapter"):
        ext.CrossAdapterTransfer(texture.pointer)


# --------------------------------------------------------------------------
# live capture -- the transfer's actual correctness
# --------------------------------------------------------------------------
#
# Verified 2026-08-22 on a real Optimus laptop: Intel iGPU capturing, RTX 4060
# receiving, 16,384,000 bytes per frame at 2560x1600, byte-exact. Before that
# the destination had only ever been WARP (ROADMAP section 10).


def _hardware_destination_available() -> bool:
    """Two non-software adapters, so the transfer has somewhere real to go."""
    topology = probe_topology()
    return len([a for a in topology.adapters if not a.is_software]) >= 2


@pytest.fixture(scope="module")
def live_capture():
    """A camera producing frames, or a skip.

    Desktop Duplication only reports *changed* content, so an idle screen
    yields nothing and this skips rather than failing -- a red suite meaning
    "nothing moved on screen" teaches people to ignore red suites.
    """
    if len(probe_topology().adapters) < 2:
        pytest.skip("cross-adapter transfer needs a second adapter")
    try:
        camera = rapidshot.create(output_color="BGRA")
    except Exception as exc:                      # no capturable adapter
        pytest.skip(f"capture unavailable: {str(exc).splitlines()[0]}")
    yield camera
    camera.release()
    rapidshot.reset()


def _grab(camera, tries=600):
    for _ in range(tries):
        frame = camera.grab_frame()
        if frame is not None:
            return frame
        time.sleep(0.005)
    return None


def test_a_captured_frame_arrives_byte_exact_on_the_other_adapter(live_capture):
    """The whole point of section 6.1, as a test rather than a script.

    ``transfer_with_reference`` records the shared-heap copy and the reference
    readback into ONE command list, which is load-bearing rather than tidy: the
    duplicated surface is live, and two copies of "the same" frame taken a
    millisecond apart genuinely differ. Comparing separate reads of it was
    measured producing ~2,000 differing bytes in one screen region, reproducibly
    at the same offset. Anything that reads a captured texture twice and expects
    agreement is wrong.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        expected = np.frombuffer(
            transfer.transfer_with_reference(frame), dtype=np.uint8)
        arrived = np.frombuffer(
            transfer.read_back_destination(), dtype=np.uint8)
    finally:
        frame.release()

    # An all-zero destination is the specific failure that looks like success
    # if you only compare lengths: it means the destination opened a heap the
    # source never wrote into.
    assert arrived.any(), (
        "every byte arrived as zero -- the destination is reading a heap the "
        "source never wrote to")
    assert arrived.shape == expected.shape
    assert np.array_equal(arrived, expected), (
        f"{int(np.count_nonzero(arrived != expected))} of {arrived.size} bytes "
        f"differ, first at offset {int(np.flatnonzero(arrived != expected)[0])}")


def test_the_destination_is_real_hardware_when_one_exists(live_capture):
    """A pass against WARP proves the mechanism, not the configuration.

    ROADMAP section 10 carried "verified against WARP only" for months. This
    asserts the destination is a hardware adapter whenever the machine has a
    second one, so that limitation cannot quietly return.
    """
    if not _hardware_destination_available():
        pytest.skip("only one hardware adapter; destination can only be WARP")
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        assert transfer.destination_is_software is False, (
            f"destination is {transfer.destination!r}, a software adapter, "
            "despite a second hardware adapter being present")
        assert transfer.total_bytes > 0
    finally:
        frame.release()


# --------------------------------------------------------------------------
# the shared destination handle -- what completes the Optimus path
# --------------------------------------------------------------------------
#
# Capture runs on the integrated GPU, so the Stage 6 tensor lands where CUDA
# cannot see it. The frame crosses adapters, and this handle is how a consumer
# on the destination adapter reaches it. Before this existed the two halves
# were each verified and nothing joined them.
#
# Measured 2026-08-22, Intel iGPU -> RTX 4060: only the *heaps* can be shared.
# CreateSharedHandle on either placed buffer fails with E_INVALIDARG, which is
# why the handle is imported as a heap (CUDA type 4) rather than as a resource
# (type 5) the way GpuPreprocessor12's committed output is.


def _handle_is_open(handle: int) -> bool:
    """Does this integer still name a kernel object this process owns?"""
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetHandleInformation.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    flags = ctypes.c_ulong()
    return bool(k32.GetHandleInformation(
        ctypes.c_void_p(handle), ctypes.byref(flags)))


def test_the_destination_handle_is_open_while_the_transfer_lives(live_capture):
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        handle = transfer.shared_destination_handle
        assert handle != 0
        assert _handle_is_open(handle)
        # Borrowed, so it must be the same handle every time rather than a
        # fresh one per access -- minting per call would hand the caller
        # something to close at a moment it cannot determine.
        assert [transfer.shared_destination_handle for _ in range(5)] == [handle] * 5
    finally:
        frame.release()


def test_the_destination_handle_closes_with_the_transfer(live_capture):
    """The handle is owned by the transfer, and dies with it.

    Asserted on the handle rather than on data: reading through a stale
    mapping returns correct bytes right up until something else claims the
    memory, so a data comparison passes with the bug fully present. ROADMAP
    section 10 records that trap for the Stage 6 handle.
    """
    import gc
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        handle = transfer.shared_destination_handle
        assert _handle_is_open(handle)
        del transfer
        gc.collect()
        assert not _handle_is_open(handle), (
            "the destination handle outlived the transfer that owns it")
    finally:
        frame.release()


def test_probe_reports_that_only_heaps_can_be_shared(live_capture):
    """The measurement the handle's design rests on.

    If a future driver starts allowing placed resources to be shared, this
    fails and the type-4-vs-type-5 decision deserves revisiting. That is the
    point: the decision was made from a measurement, so the measurement is
    what guards it.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        rows = {r["label"]: r for r in transfer.probe_shared_handles()}
    finally:
        frame.release()

    assert set(rows) == {"dst_resource", "dst_heap", "src_resource", "src_heap"}
    assert rows["dst_heap"]["ok"], rows["dst_heap"].get("error")
    assert rows["dst_heap"]["cuda_handle_type"] == 4
    assert not rows["dst_resource"]["ok"], (
        "a placed resource became shareable; the heap-vs-resource decision "
        "in shared_destination_handle was measured, not assumed, and this is "
        "the measurement")
    assert rows["dst_resource"]["cuda_handle_type"] == 5


def test_the_probe_leaks_nothing(live_capture):
    """A diagnostic that hands back live handles makes the caller clean up
    after a question it only asked.

    Checked by process handle count rather than by inspecting the returned
    rows, because the rows deliberately carry no handles -- the absence of a
    leak is the property, not the absence of a field.
    """
    import ctypes
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        count = ctypes.c_ulong()
        k32.GetProcessHandleCount.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
        proc = ctypes.c_void_p(k32.GetCurrentProcess())

        transfer.probe_shared_handles()          # warm any one-off allocation
        assert k32.GetProcessHandleCount(proc, ctypes.byref(count))
        before = count.value
        for _ in range(10):
            transfer.probe_shared_handles()
        assert k32.GetProcessHandleCount(proc, ctypes.byref(count))
        after = count.value
    finally:
        frame.release()

    # 10 calls x 2 mintable handles = 20 leaked if the probe kept them.
    assert after - before < 10, (
        f"process handle count rose {before} -> {after} across 10 probe calls; "
        "the probe is leaking the handles it mints")


def test_the_returned_rows_carry_no_handles(live_capture):
    """The diagnostic value is the outcome, not a usable handle."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        rows = transfer.probe_shared_handles()
    finally:
        frame.release()
    for row in rows:
        assert "handle" not in row, (
            "probe rows must not carry handles; use shared_destination_handle")


# --------------------------------------------------------------------------
# the shared fence -- async transfer
# --------------------------------------------------------------------------
#
# Section 6.1 deferred this on a 2026-08-05 measurement taken against WARP,
# concluding the fence would buy "latency and pipelining, not throughput".
# Re-measured 2026-08-22 on real hybrid hardware that turned out to be wrong in
# the useful direction: with 2 ms of consumer work per frame, wall clock went
# 6.22 -> 3.59 ms (42% faster) because the copy overlaps the consumer, and 98%
# of calling-thread time came back. The old conclusion was correct for the
# hardware it was taken on and did not survive the hardware it was about.


def test_async_transfer_is_byte_exact(live_capture):
    """Not blocking must not mean not correct."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        value = transfer.transfer_async(frame)
        assert value > 0
        transfer.wait_shared_fence(value)
        arrived = np.frombuffer(transfer.read_back_destination(), dtype=np.uint8)
        assert arrived.any(), "async transfer produced an all-zero destination"
        expected = np.frombuffer(
            transfer.transfer_with_reference(frame), dtype=np.uint8)
        assert np.array_equal(
            np.frombuffer(transfer.read_back_destination(), dtype=np.uint8),
            expected)
    finally:
        frame.release()


def test_the_fence_value_advances_per_transfer(live_capture):
    """Each submission gets its own value, so waits cannot be confused."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        values = [transfer.transfer_async(frame) for _ in range(4)]
        transfer.wait_shared_fence(values[-1])
    finally:
        frame.release()
    assert values == sorted(values) and len(set(values)) == len(values), values


def test_waiting_on_zero_returns_immediately(live_capture):
    """Nothing submitted yet is not an error, and must not hang."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        native.cross_adapter_transfer(frame).wait_shared_fence(0)
    finally:
        frame.release()


def test_the_shared_fence_handle_is_open_and_stable(live_capture):
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        handle = transfer.shared_fence_handle
        assert handle != 0
        assert _handle_is_open(handle)
        assert [transfer.shared_fence_handle for _ in range(3)] == [handle] * 3
    finally:
        frame.release()


def test_the_shared_fence_handle_closes_with_the_transfer(live_capture):
    import gc
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        handle = transfer.shared_fence_handle
        assert _handle_is_open(handle)
        del transfer
        gc.collect()
        assert not _handle_is_open(handle)
    finally:
        frame.release()


# --------------------------------------------------------------------------
# the per-frame handle cache
# --------------------------------------------------------------------------

def test_the_capture_texture_is_opened_once_not_per_frame(live_capture):
    """Asserted on the cache key, not on pixels.

    Reopening per frame costs 382 us here and produces byte-identical output,
    so a change that disabled the cache would be a large regression with no
    visible symptom. Section 10 records the same reasoning for the Stage 6
    preprocessor's texture cache.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        assert transfer.cached_texture_address == 0, "nothing opened yet"
        transfer.transfer(frame)
        key = transfer.cached_texture_address
        assert key != 0
        for _ in range(3):
            transfer.transfer(frame)
            assert transfer.cached_texture_address == key, (
                "the cache re-keyed on an unchanged texture")
    finally:
        frame.release()


# --------------------------------------------------------------------------
# the GIL, which is the gap between a Rust-level win and a Python-level one
# --------------------------------------------------------------------------

def test_wait_shared_fence_releases_the_gil(live_capture):
    """Handing back a thread that cannot run Python is not handing it back.

    `wait_shared_fence` originally held the interpreter for the whole wait, so
    the calling-thread time the async path "returned" was unusable by any other
    Python thread. Measured 2026-08-22 before the fix: a second thread made
    *zero* progress across a 7 ms wait.

    Asserted on another thread making progress rather than on wall-clock, which
    would be timing-sensitive. The bar is deliberately low -- any progress at
    all falsifies a held GIL -- because the failure this guards against is
    total, not marginal.
    """
    import threading

    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")

    stop = threading.Event()
    ticks = [0]

    def counter():
        while not stop.is_set():
            ticks[0] += 1

    worker = threading.Thread(target=counter, daemon=True)
    worker.start()
    try:
        transfer = native.cross_adapter_transfer(frame)
        time.sleep(0.05)                       # let the counter reach steady state
        before = ticks[0]
        for _ in range(5):
            value = transfer.transfer_async(frame)
            transfer.wait_shared_fence(value)
        progressed = ticks[0] - before
    finally:
        stop.set()
        worker.join(timeout=2)
        frame.release()

    assert progressed > 0, (
        "another Python thread made no progress across five fence waits; "
        "wait_shared_fence is holding the GIL")


def test_the_cache_key_includes_the_source_id(live_capture):
    """A texture address alone is not an identity.

    COM addresses are recycled, so after an access-loss rebuild a new
    duplicator can hand back an unrelated texture at the same pointer. Keying
    on the pointer alone would return the previously opened D3D12 resource and
    copy a stale frame -- silently, since the output still looks like a frame.
    `Frame.source_id` says which duplicator produced the texture, and the pair
    is what `GpuPreprocessor12` has always keyed on.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        transfer.transfer(frame)
        key = (transfer.cached_texture_address, transfer.cached_source_id)
        assert key[0] != 0

        # Same pointer, different producer: the pair must miss even though the
        # address matches, which is the whole point of the second component.
        transfer._inner.transfer(transfer.cached_texture_address,
                                 transfer.cached_source_id + 1)
        assert transfer.cached_source_id == key[1] + 1
        assert transfer.cached_texture_address == key[0]
    finally:
        frame.release()


def test_dropping_after_an_async_submit_does_not_crash(live_capture):
    """Drop must not free resources the GPU is still reading.

    `transfer()` blocks, so before `transfer_async()` this was unreachable. It
    is not now: a caller that submits and drops -- or whose exception skips the
    wait -- would otherwise release the cached source resource, both heaps, the
    command list and the fences mid-copy. D3D12 requires them alive until the
    GPU finishes; the drop path waits for the last signalled fence value.

    A crash here is a process-level failure, so this asserting "we got here"
    is the assertion.
    """
    import gc
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        for _ in range(3):
            transfer = native.cross_adapter_transfer(frame)
            transfer.transfer_async(frame)      # deliberately no wait
            del transfer
            gc.collect()
    finally:
        frame.release()
    assert True
