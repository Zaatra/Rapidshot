"""Convert-first across adapters — ordering B of ROADMAP § 6.1, end to end.

§ 6.1 measured this ordering twice and found it winning, but nothing could
*select* it: the converter writes a buffer and `CrossAdapterTransfer` takes a
texture. These tests cover the path that closes that, and the property that
matters most about it — **what arrives is what was produced**.

A cross-adapter copy that silently truncates or reorders is the exact failure
class § 11 exists for: the destination buffer is the right size and full of
plausible floats either way. So the check is byte equality against the source,
not a tolerance.

Machine A has one hardware adapter, so the destination here is WARP. That
makes the *timing* unrepresentative, which is why none is asserted — but it
does not make the bytes unrepresentative, and the bytes are the contract.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

OUT = 64


@pytest.fixture(scope="module")
def live_frame():
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
def transfer_pair(live_frame):
    """A converter and a transfer, or a skip on a single-adapter machine."""
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float16", sampling="nearest"
    )
    try:
        transfer = rapidshot.TensorTransfer(converter)
    except RuntimeError as exc:
        if "only one adapter" in str(exc):
            pytest.skip("single-adapter machine; nothing to transfer to")
        raise
    return converter, transfer


# --------------------------------------------------------------------------
# the property the whole path depends on
# --------------------------------------------------------------------------


def test_what_arrives_is_what_was_produced(transfer_pair, live_frame):
    """Byte equality, not a tolerance.

    A truncating or misaligned copy still yields a correctly-sized buffer of
    plausible half-floats. Nothing about shape, dtype or range would catch it.
    """
    converter, transfer = transfer_pair
    converter.process(live_frame)
    produced = converter.process(live_frame)  # stable frame, converted twice
    source = produced.numpy()

    transfer.transfer()
    arrived = transfer.read_back_destination()

    assert arrived.shape == source.shape
    assert arrived.dtype == source.dtype
    np.testing.assert_array_equal(arrived, source)


def test_transfer_before_process_does_not_raise(live_frame):
    """Documented behaviour: an unwritten buffer is zeros, not an error.

    Pinned because the docstring promises it, and a caller who forgets
    `process()` should get a wrong-but-explicable result rather than a crash
    deep in D3D12.

    Its own converter, not the module fixture: by the time this runs the
    shared one has been processed, so it would not be "before process".
    """
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype="float16", sampling="nearest"
    )
    try:
        transfer = rapidshot.TensorTransfer(converter)
    except RuntimeError as exc:
        if "only one adapter" in str(exc):
            pytest.skip("single-adapter machine; nothing to transfer to")
        raise
    transfer.transfer()
    arrived = np.asarray(transfer.read_back_destination())
    assert arrived.size > 0
    assert not arrived.any(), "an unwritten buffer arrived non-zero"


# --------------------------------------------------------------------------
# the payload, which is the reason to choose this ordering at all
# --------------------------------------------------------------------------


def test_payload_is_the_converted_size_not_the_frame(transfer_pair, live_frame):
    """§ 6.1 in one assertion: B moves the tensor, not the frame."""
    converter, transfer = transfer_pair
    frame_bytes = live_frame.width * live_frame.height * 4

    assert transfer.total_bytes == converter.output_byte_size
    assert transfer.total_bytes == OUT * OUT * 3 * 2
    assert transfer.total_bytes < frame_bytes / 10


@pytest.mark.parametrize(
    "dtype,layout,bpp",
    [("uint8", "nhwc", 4), ("float16", "nchw", 6), ("float32", "nchw", 12)],
)
def test_dtype_decides_what_crosses(live_frame, dtype, layout, bpp):
    """The representation is chosen at the converter and the bus follows it."""
    converter = rapidshot.GpuConverter(
        live_frame, (OUT, OUT), dtype=dtype, layout=layout
    )
    try:
        transfer = rapidshot.TensorTransfer(converter)
    except RuntimeError as exc:
        if "only one adapter" in str(exc):
            pytest.skip("single-adapter machine; nothing to transfer to")
        raise
    assert transfer.total_bytes == OUT * OUT * bpp


# --------------------------------------------------------------------------
# identity, which is what a consumer needs to find the tensor
# --------------------------------------------------------------------------


def test_destination_is_a_different_adapter_than_the_source(transfer_pair):
    converter, transfer = transfer_pair
    assert transfer.source != transfer.destination
    assert len(transfer.destination_luid) == 8
    # The converter's LUID is the capture adapter; the destination must differ,
    # or the transfer moved nothing anywhere.
    assert transfer.destination_luid != converter._impl.adapter_luid()


def test_destination_handles_are_non_null(transfer_pair):
    """A consumer imports by handle, and a null one is a plausible integer."""
    _, transfer = transfer_pair
    assert transfer.shared_destination_handle != 0
    assert transfer.destination_resource_address != 0


def test_repr_names_the_payload_and_both_adapters(transfer_pair):
    _, transfer = transfer_pair
    text = repr(transfer)
    assert "MB" in text
    assert transfer.destination in text
