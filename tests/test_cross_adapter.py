"""Tests for the cross-adapter transfer API (ROADMAP.md 6.1).

What can be tested here is the contract around the transfer: that it refuses
what it cannot handle, and refuses it with a message that says what to do.

The transfer *itself* cannot be unit tested. It needs a genuinely duplicated
surface: D3D11 refuses SHARED_NTHANDLE without SHARED_KEYEDMUTEX, and a
keyed-mutex resource reads as zeros until acquired, so a synthetic texture
exercises none of the path. Correctness lives in
``examples/verify_cross_adapter.py``, which compares every transferred byte
against a source-side readback of the same frame and must be run on real
hardware before a release.
"""

import pytest

from rapidshot import native

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
