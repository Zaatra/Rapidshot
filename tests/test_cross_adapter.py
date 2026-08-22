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

import ctypes
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
    """Compare bytes produced by one asynchronous command list."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        value = transfer._inner.transfer_async_with_reference(
            native._texture_address(frame), native._source_id(frame))
        assert value > 0
        transfer.wait_shared_fence(value)
        expected = np.frombuffer(
            transfer._inner.read_back_source(), dtype=np.uint8)
        arrived = np.frombuffer(transfer.read_back_destination(), dtype=np.uint8)
    finally:
        frame.release()

    assert arrived.any(), "async transfer produced an all-zero destination"
    assert arrived.shape == expected.shape
    assert np.array_equal(arrived, expected), (
        f"{int(np.count_nonzero(arrived != expected))} of {arrived.size} bytes "
        f"differ, first at offset {int(np.flatnonzero(arrived != expected)[0])}")


def test_live_layout_reports_the_captured_dxgi_format(live_capture):
    """Consumers need the raw format and its real footprint, not a BGRA guess."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        bytes_per_pixel = {87: 4, 28: 4, 24: 4, 10: 8}
        assert transfer.dxgi_format in bytes_per_pixel
        assert transfer.bytes_per_pixel == bytes_per_pixel[transfer.dxgi_format]
        assert transfer.row_pitch >= transfer.width * transfer.bytes_per_pixel
        assert transfer.total_bytes >= transfer.row_pitch * transfer.height

        transfer.transfer(frame)
        assert len(transfer.read_back_destination()) == transfer.total_bytes
    finally:
        frame.release()


def test_destination_readback_waits_for_an_async_copy(live_capture):
    """Immediate verification readback must order behind the source queue."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        value = transfer.transfer_async(frame)
        # Deliberately no wait_shared_fence call: readback owns this ordering.
        arrived = np.frombuffer(transfer.read_back_destination(), dtype=np.uint8)
        assert transfer.shared_fence_completed >= value
        assert arrived.any(), "ordered readback produced an all-zero destination"
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


def test_phase_probe_drains_an_unwaited_async_submit(live_capture):
    """The diagnostic must not reset an allocator the GPU is still reading."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        value = transfer.transfer_async(frame)  # deliberately no explicit wait
        phases = transfer._inner.probe_transfer_phases(
            native._texture_address(frame), 2, True, native._source_id(frame))
        assert set(phases) == {"open", "record", "submit", "signal", "wait", "close"}
        assert transfer.shared_fence_completed >= value
    finally:
        frame.release()


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


def test_releasing_a_frame_waits_for_an_async_copy(live_capture):
    """`with camera.grab_frame()` must not hand the surface back mid-copy.

    The duplicated surface is only valid between AcquireNextFrame and
    ReleaseFrame, and transfer_async returns while the GPU is still reading it.
    Releasing in between lets DXGI recycle the surface, and the destination
    ends up holding a blend of two frames -- silently, which is why this
    asserts on the fence being waited rather than on pixels. Comparing pixels
    would pass whenever the race happened not to be lost.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")

    transfer = native.cross_adapter_transfer(frame)
    waited = []
    real_wait = transfer.wait_shared_fence
    transfer.wait_shared_fence = lambda v: (waited.append(v), real_wait(v))[1]

    value = transfer.transfer_async(frame)
    assert waited == [], "the submit should not have waited"
    frame.release()
    assert waited == [value], (
        "releasing the frame did not wait for the outstanding copy; the "
        "surface can go back to DXGI while the GPU is still reading it")


def test_the_release_drain_runs_before_the_frame_is_marked_released():
    """Ordering matters: the surface must still be held while draining."""
    from rapidshot.frame import Frame

    order = []
    frame = Frame(texture=object(), on_release=lambda: order.append("released"),
                  region=(0, 0, 8, 8))
    frame.defer_release_until(lambda: order.append("drained"))
    frame.release()
    assert order == ["drained", "released"], order
    assert frame.released


def test_a_failing_drain_still_releases_the_surface():
    """A stuck consumer must not strand capture.

    DXGI refuses the next acquire while a surface is outstanding, so a drain
    that raises has to log and continue rather than leave the frame unreleased
    -- that would stall capture completely rather than degrade.
    """
    from rapidshot.frame import Frame

    released = []
    frame = Frame(texture=object(), on_release=lambda: released.append(True),
                  region=(0, 0, 8, 8))

    def boom():
        raise RuntimeError("consumer exploded")

    frame.defer_release_until(boom)
    frame.release()
    assert released == [True]
    assert frame.released


# --------------------------------------------------------------------------
# the reverse handshake: telling the producer the consumer is done
# --------------------------------------------------------------------------
#
# Every transfer reuses one destination buffer, and the producer fence only
# says the copy finished -- nothing says whether the consumer is still reading.
# An asynchronous consumer can therefore be reading frame N when the copy for
# N+1 overwrites the allocation under it.
#
# NOTE: this race has NOT been reproduced. Attempts to force it on the
# development machine failed -- CuPy's allocator synchronises the calling
# thread with the consumer, so the producer never ran far enough ahead, even
# with a 527 ms consumer. The mechanism below is correct by construction
# rather than validated against an observed failure, and ROADMAP section 5's
# rule applies: that is weaker evidence than a reproduced bug.
#
# What *is* measured: CUDA on an NVIDIA dGPU can signal a D3D12 fence created
# by an Intel iGPU's device, and the producer observes it (completed value
# 0 -> 5000). Without that the handshake would not be buildable at all.


def test_wait_for_consumer_refuses_before_a_fence_is_adopted(live_capture):
    """Failing loudly beats queueing a wait on nothing."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        with pytest.raises(RuntimeError, match="consumer fence"):
            transfer.wait_for_consumer(1)
    finally:
        frame.release()


def test_a_consumer_fence_can_be_adopted_and_waited_on(live_capture):
    """Adopting a fence and queueing a wait must not disturb the transfer.

    The wait is queued on the source queue, so it orders ahead of the next copy
    without blocking the caller. Waiting for a value already reached must
    return rather than hang, which is what makes the first iteration of a
    capture loop safe before the consumer has signalled anything.
    """
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        donor = native.cross_adapter_transfer(frame)
        transfer.set_consumer_fence(donor.shared_fence_handle)
        transfer.wait_for_consumer(0)          # already reached; must not hang
        value = transfer.transfer_async(frame)
        assert value > 0
        transfer.wait_shared_fence(value)
        arrived = np.frombuffer(transfer.read_back_destination(), dtype=np.uint8)
        assert arrived.any(), "the copy did not run after a queued consumer wait"
    finally:
        frame.release()


def test_consumer_fence_can_be_replaced_after_a_queued_wait(live_capture):
    """A superseded fence must outlive the GPU-side wait that names it."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        first_owner = native.cross_adapter_transfer(frame)
        second_owner = native.cross_adapter_transfer(frame)

        # Queue an unsatisfied wait, then replace the adopted fence before its
        # owner submits the signal. The old fence must remain alive until the
        # source queue consumes that wait.
        transfer.set_consumer_fence(first_owner.shared_fence_handle)
        transfer.wait_for_consumer(1)
        transfer.set_consumer_fence(second_owner.shared_fence_handle)

        assert first_owner.transfer_async(frame) == 1
        value = transfer.transfer_async(frame)
        transfer.wait_shared_fence(value)
        assert transfer.shared_fence_completed >= value

        # The replacement is active for subsequent waits.
        transfer.wait_for_consumer(1)
        assert second_owner.transfer_async(frame) == 1
        value = transfer.transfer_async(frame)
        transfer.wait_shared_fence(value)
        assert transfer.shared_fence_completed >= value
    finally:
        frame.release()


def test_fence_progress_is_observable(live_capture):
    """submitted vs completed -- the instrument the handshake was built with."""
    frame = _grab(live_capture)
    if frame is None:
        pytest.skip("no frame captured -- the screen must be changing")
    try:
        transfer = native.cross_adapter_transfer(frame)
        assert transfer.shared_fence_submitted == 0
        value = transfer.transfer_async(frame)
        assert transfer.shared_fence_submitted == value
        transfer.wait_shared_fence(value)
        assert transfer.shared_fence_completed >= value
    finally:
        frame.release()


def _load_cuda_example():
    """Import the shipped CuPy example, so its structures are the real ones."""
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / "examples" / "gpu_tensor_to_cupy.py"
    spec = importlib.util.spec_from_file_location("gpu_tensor_to_cupy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



# --------------------------------------------------------------------------
# the shared destination buffer, and the consumer that is still reading it
# --------------------------------------------------------------------------
#
# Every transfer writes the same buffer, and the producer fence only says the
# *copy* finished -- nothing says the consumer is done reading. Reproduced
# deterministically 2026-08-22: with the consumer's read gated in its stream,
# a copy of frame B completed underneath it, and the consumer -- which had
# waited on the producer fence for frame A -- read B's bytes. 3/3 runs.
#
# Getting there took four failed probe designs, all defeated the same way:
# CuPy's allocator synchronises the calling thread, so anything that allocates
# inside the gated region either hides the race or self-deadlocks (one version
# hung for ten minutes). Nothing below allocates after the gate closes.


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


class _SigInner(ctypes.Structure):
    _fields_ = [("fence", _FenceParams), ("nvSciSync", _NvSciParams),
                ("keyedMutex", _KeyedParams), ("reserved", ctypes.c_uint * 12)]


class _SemSignalParams(ctypes.Structure):
    _fields_ = [("params", _SigInner), ("flags", ctypes.c_uint),
                ("reserved", ctypes.c_uint * 16)]


def test_the_consumer_handshake_prevents_a_stale_read(live_capture):
    """A gated consumer must still see the frame it waited for.

    Construction, which is deterministic rather than timing-dependent:

      1. frame A is transferred and its checksum recorded
      2. the consumer's read is queued behind a semaphore this test holds shut
      3. frame B is transferred *and allowed to complete* -- provably while the
         consumer's read is still pending
      4. the gate opens

    Without `wait_for_consumer` step 3 succeeds and the consumer reads B.
    With it, the producer's copy is queued behind the consumer's fence and
    cannot land until the consumer signals, so the consumer still sees A.

    Only the guarded path is asserted. A test that required the *unguarded*
    path to corrupt would fail the day something upstream fixed it, which is
    the wrong direction for a regression test to point.
    """
    cp = pytest.importorskip("cupy", reason="CuPy not installed")
    if not _hardware_destination_available():
        pytest.skip("cross-adapter transfer needs a second hardware adapter")
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuInit(0)
    cp.cuda.Device(0).use()
    cp.zeros(1)

    def _grab_one():
        for _ in range(600):
            f = live_capture.grab_frame()
            if f is not None:
                return f
            time.sleep(0.004)
        return None

    seed = _grab_one()
    if seed is None:
        pytest.skip("no frame captured -- the screen must be changing")
    transfer = native.cross_adapter_transfer(seed)
    gate_owner = native.cross_adapter_transfer(seed)
    transfer.set_consumer_fence(gate_owner.shared_fence_handle)
    seed.release()

    def import_sem(handle):
        d = _SemHandleDesc()
        ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
        d.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        d.handle.win32.handle = ctypes.c_void_p(handle)
        sem = ctypes.c_void_p()
        assert cuda.cuImportExternalSemaphore(
            ctypes.byref(sem), ctypes.byref(d)) == 0
        return sem

    gate = import_sem(gate_owner.shared_fence_handle)
    module = _load_cuda_example()
    mem_desc = module.ExternalMemoryHandleDesc()
    ctypes.memset(ctypes.byref(mem_desc), 0, ctypes.sizeof(mem_desc))
    mem_desc.type = 4
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

    # Preallocated: nothing may allocate once the gate is shut.
    snapshot = cp.empty(transfer.total_bytes, dtype=cp.uint8)
    consumer = cp.cuda.Stream(non_blocking=True)
    releaser = cp.cuda.Stream(non_blocking=True)

    def grab():
        for _ in range(600):
            f = live_capture.grab_frame()
            if f is not None:
                return f
            time.sleep(0.004)
        return None

    try:
        # 1. frame A lands.
        frame_a = grab()
        if frame_a is None:
            pytest.skip("no frame captured -- the screen must be changing")
        value = transfer.transfer_async(frame_a)
        transfer.wait_shared_fence(value)
        checksum_a = int(np.frombuffer(
            transfer.read_back_destination(), dtype=np.uint8).sum())
        frame_a.release()

        # 2. the consumer's read is gated shut.
        wait = _SemWaitParams()
        ctypes.memset(ctypes.byref(wait), 0, ctypes.sizeof(wait))
        wait.params.fence.value = 1
        assert cuda.cuWaitExternalSemaphoresAsync(
            ctypes.byref(gate), ctypes.byref(wait), 1,
            ctypes.c_void_p(consumer.ptr)) == 0
        assert cuda.cuMemcpyDtoDAsync_v2(
            ctypes.c_ulonglong(int(snapshot.data.ptr)),
            ctypes.c_ulonglong(ptr.value),
            ctypes.c_size_t(transfer.total_bytes),
            ctypes.c_void_p(consumer.ptr)) == 0

        # 3. frame B is submitted behind the consumer's fence, then 4. released.
        frame_b = grab()
        if frame_b is None:
            pytest.skip("no second frame -- the screen must be changing")
        transfer.wait_for_consumer(1)
        value_b = transfer.transfer_async(frame_b)
        signal = _SemSignalParams()
        ctypes.memset(ctypes.byref(signal), 0, ctypes.sizeof(signal))
        signal.params.fence.value = 1
        assert cuda.cuSignalExternalSemaphoresAsync(
            ctypes.byref(gate), ctypes.byref(signal), 1,
            ctypes.c_void_p(releaser.ptr)) == 0
        releaser.synchronize()
        transfer.wait_shared_fence(value_b)
        checksum_b = int(np.frombuffer(
            transfer.read_back_destination(), dtype=np.uint8).sum())
        frame_b.release()
        consumer.synchronize()
        seen = int(cp.asnumpy(snapshot).sum())
    finally:
        cuda.cuDestroyExternalSemaphore(gate)
        cuda.cuDestroyExternalMemory(ext)

    if checksum_a == checksum_b:
        pytest.skip("the two frames are identical -- nothing to distinguish")
    assert seen == checksum_a, (
        f"the consumer read frame B ({checksum_b}) after waiting for frame A "
        f"({checksum_a}); the producer overwrote the shared buffer while the "
        f"consumer was still reading it")


def test_the_handshake_holds_over_a_sustained_loop(live_capture):
    """Many frames, not one A/B cycle: no fence drift, no deadlock, no leak.

    A single cycle proves the mechanism; it does not prove the bookkeeping
    survives repetition. This runs a real loop with a consumer heavy enough to
    lag the producer, and checks every frame the consumer read against the
    frame it was told to wait for.

    Measured 2026-08-22 at 60 frames with a consumer slower than the producer:
    **28 of 60 frames wrong unguarded, 0 guarded.** With a light consumer the
    same loop is clean either way over 100 frames -- the hazard only appears
    once the consumer actually falls behind, which is why it stays invisible
    until a real workload shows up.

    Needs on-screen motion: with a static desktop every frame is identical and
    the comparison cannot distinguish anything, so it skips rather than
    passing vacuously.
    """
    cp = pytest.importorskip("cupy", reason="CuPy not installed")
    if not _hardware_destination_available():
        pytest.skip("cross-adapter transfer needs a second hardware adapter")

    FRAMES, SAMPLE, HEAVY = 12, 1 << 16, 40
    cuda = ctypes.WinDLL("nvcuda.dll")
    cuda.cuInit(0)
    cp.cuda.Device(0).use()
    cp.zeros(1)

    def grab():
        for _ in range(600):
            f = live_capture.grab_frame()
            if f is not None:
                return f
            time.sleep(0.003)
        return None

    seed = grab()
    if seed is None:
        pytest.skip("no frame captured -- the screen must be changing")
    transfer = native.cross_adapter_transfer(seed)
    consumer_owner = native.cross_adapter_transfer(seed)
    transfer.set_consumer_fence(consumer_owner.shared_fence_handle)

    module = _load_cuda_example()
    md = module.ExternalMemoryHandleDesc()
    ctypes.memset(ctypes.byref(md), 0, ctypes.sizeof(md))
    md.type = 4
    md.handle.win32.handle = ctypes.c_void_p(transfer.shared_destination_handle)
    md.size = transfer.total_bytes
    ext = ctypes.c_void_p()
    assert cuda.cuImportExternalMemory(ctypes.byref(ext), ctypes.byref(md)) == 0
    bd = module.ExternalMemoryBufferDesc()
    ctypes.memset(ctypes.byref(bd), 0, ctypes.sizeof(bd))
    bd.offset, bd.size, bd.flags = 0, transfer.total_bytes, 0
    ptr = ctypes.c_ulonglong()
    assert cuda.cuExternalMemoryGetMappedBuffer(
        ctypes.byref(ptr), ext, ctypes.byref(bd)) == 0

    def import_sem(handle):
        d = _SemHandleDesc()
        ctypes.memset(ctypes.byref(d), 0, ctypes.sizeof(d))
        d.type = CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        d.handle.win32.handle = ctypes.c_void_p(handle)
        sem = ctypes.c_void_p()
        assert cuda.cuImportExternalSemaphore(
            ctypes.byref(sem), ctypes.byref(d)) == 0
        return sem

    producer_sem = import_sem(transfer.shared_fence_handle)
    consumer_sem = import_sem(consumer_owner.shared_fence_handle)
    seed.release()

    # Preallocated: nothing may allocate on the consumer stream, or CuPy's
    # allocator synchronises the calling thread and the lag disappears.
    slots = cp.empty(FRAMES * SAMPLE, dtype=cp.uint8)
    scratch = cp.empty(transfer.total_bytes, dtype=cp.uint8)
    stream = cp.cuda.Stream(non_blocking=True)

    def offset_that_changes():
        """Sample where the screen actually moves; a static corner proves nothing."""
        f = grab()
        if f is None:
            return None
        transfer.transfer(f)
        a = np.frombuffer(transfer.read_back_destination(), np.uint8).copy()
        f.release()
        time.sleep(0.12)
        f = grab()
        if f is None:
            return None
        transfer.transfer(f)
        b = np.frombuffer(transfer.read_back_destination(), np.uint8)
        f.release()
        diff = np.flatnonzero(a != b)
        if diff.size == 0:
            return None
        return min(int(diff[diff.size // 2]), transfer.total_bytes - SAMPLE)

    try:
        offset = offset_that_changes()
        if offset is None:
            pytest.skip("nothing on screen is changing -- run motion_source.py")

        truth, captured, consumer_value = [], 0, 0
        for i in range(FRAMES):
            f = grab()
            if f is None:
                break
            if consumer_value:
                transfer.wait_for_consumer(consumer_value)
            value = transfer.transfer_async(f)

            wait = _SemWaitParams()
            ctypes.memset(ctypes.byref(wait), 0, ctypes.sizeof(wait))
            wait.params.fence.value = value
            assert cuda.cuWaitExternalSemaphoresAsync(
                ctypes.byref(producer_sem), ctypes.byref(wait), 1,
                ctypes.c_void_p(stream.ptr)) == 0
            for _ in range(HEAVY):          # make the consumer lag the producer
                assert cuda.cuMemcpyDtoDAsync_v2(
                    ctypes.c_ulonglong(int(scratch.data.ptr)),
                    ctypes.c_ulonglong(ptr.value),
                    ctypes.c_size_t(transfer.total_bytes),
                    ctypes.c_void_p(stream.ptr)) == 0
            assert cuda.cuMemcpyDtoDAsync_v2(
                ctypes.c_ulonglong(int(slots.data.ptr) + i * SAMPLE),
                ctypes.c_ulonglong(ptr.value + offset),
                ctypes.c_size_t(SAMPLE), ctypes.c_void_p(stream.ptr)) == 0

            consumer_value += 1
            signal = _SemSignalParams()
            ctypes.memset(ctypes.byref(signal), 0, ctypes.sizeof(signal))
            signal.params.fence.value = consumer_value
            assert cuda.cuSignalExternalSemaphoresAsync(
                ctypes.byref(consumer_sem), ctypes.byref(signal), 1,
                ctypes.c_void_p(stream.ptr)) == 0

            transfer.wait_shared_fence(value)
            truth.append(int(np.frombuffer(
                transfer.read_back_destination(), dtype=np.uint8,
                count=SAMPLE, offset=offset).sum()))
            f.release()
            captured += 1
        stream.synchronize()
    finally:
        cuda.cuDestroyExternalSemaphore(producer_sem)
        cuda.cuDestroyExternalSemaphore(consumer_sem)
        cuda.cuDestroyExternalMemory(ext)

    if captured < 3:
        pytest.skip("too few frames captured -- the screen must be changing")
    seen = cp.asnumpy(slots).reshape(FRAMES, SAMPLE).sum(
        axis=1, dtype=np.int64)[:captured]
    truth = np.asarray(truth, dtype=np.int64)
    if len(set(truth.tolist())) < 2:
        pytest.skip("frames are identical -- nothing to distinguish")
    bad = int((seen != truth).sum())
    assert bad == 0, (
        f"{bad} of {captured} frames were read after the producer had already "
        f"overwritten them; the consumer handshake did not hold")
