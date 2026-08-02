"""Tests for the optional native GPU-interop shim.

The contract that matters most here is that the extension is *optional*: the
library must install and capture without a Rust toolchain, and callers who need
GPU interop must get an actionable message rather than an ImportError from deep
in the stack.
"""

import pytest

from rapidshot import native


def test_availability_is_reported_not_raised():
    """Importing rapidshot.native must never fail, built or not."""
    assert isinstance(native.is_available(), bool)


def test_build_info_matches_availability():
    info = native.build_info()
    if native.is_available():
        assert isinstance(info, dict)
        assert "version" in info and "stage" in info
    else:
        assert info is None


def test_require_gives_build_instructions_when_absent(monkeypatch):
    """A missing extension must explain how to build it."""
    monkeypatch.setattr(native, "_ext", None)
    monkeypatch.setattr(native, "_import_error", ImportError("simulated"))

    with pytest.raises(RuntimeError) as excinfo:
        native.require()

    message = str(excinfo.value)
    assert "cargo build --release" in message
    assert "rustup.rs" in message
    # It must also make clear that the rest of the library still works.
    assert "without it" in message


def test_require_returns_the_module_when_present():
    if not native.is_available():
        pytest.skip("native extension not built")
    assert native.require() is not None


def test_texture_address_rejects_a_released_frame():
    """A released Frame must never yield a dangling pointer to native code."""
    from rapidshot.frame import Frame, FrameReleasedError

    frame = Frame(texture=object(), on_release=lambda: None, region=(0, 0, 4, 4))
    frame.release()

    with pytest.raises(FrameReleasedError):
        native._texture_address(frame)


def test_texture_address_rejects_null_pointer():
    import ctypes

    from rapidshot.frame import Frame

    frame = Frame(texture=ctypes.c_void_p(0), on_release=lambda: None,
                  region=(0, 0, 4, 4))
    with pytest.raises(ValueError, match="null"):
        native._texture_address(frame)


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_native_rejects_null_pointer_directly():
    with pytest.raises(ValueError, match="null"):
        native.require().describe_texture(0)


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_build_info_reports_the_expected_stage():
    info = native.build_info()
    assert info["stage"].startswith("6-"), "native shim should identify its stage"


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_no_d3d11_buffer_configuration_is_shareable():
    """
    Encodes the finding that redirected Stage 6 milestone 3b.

    D3D11 can only share 2D non-mipmapped textures — never buffers. Six
    configurations were probed (structured / raw / plain, each with NT-handle
    and legacy sharing) and none produced a buffer D3D12 could open. That is why
    the conversion shader has to run on the D3D12 device rather than writing in
    D3D11 and sharing across.

    If this ever starts passing, the simpler D3D11 route has become available
    and the design should be revisited.
    """
    result = native.probe_shareable_buffers()
    assert result["d3d12_available"], "no D3D12 device; probe is inconclusive"

    candidates = result["candidates"]
    assert candidates, "probe returned no candidates"
    usable = [name for name, info in candidates.items() if info.get("usable")]
    assert not usable, (
        f"a D3D11 buffer configuration is now shareable ({usable}); the D3D12 "
        "port of the conversion shader may no longer be necessary"
    )

    # Every candidate must have been genuinely attempted, so an all-fail result
    # cannot be produced by the probe silently doing nothing.
    for name, info in candidates.items():
        assert "created" in info, f"{name} was not attempted"


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_d3d12_probe_rejects_null_pointer():
    with pytest.raises(ValueError, match="null"):
        native.require().probe_d3d12_sharing(0)


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_cross_adapter_probe_reports_the_whole_chain():
    """Cross-adapter sharing must be measured, not assumed (ROADMAP.md 6.1).

    A small frame and few iterations: this asserts the mechanism works and that
    the probe reports honestly, not that it is fast. Timing numbers belong in
    benchmarks/, where the noise floor is measured first.
    """
    result = native.probe_cross_adapter(width=256, height=256, iterations=3)

    assert result["adapters"], "probe returned no adapters"
    if not result.get("supported"):
        # A single-adapter machine is a legitimate outcome, but it has to say so
        # rather than silently reporting nothing.
        assert result.get("reason"), "unsupported without a reason"
        pytest.skip(f"cross-adapter sharing unavailable: {result['reason']}")

    # Every step of the chain has to be reported, so a "supported" verdict
    # cannot come from a probe that stopped early.
    assert result["opened_on_destination"], "shared heap did not open on the second adapter"
    assert result["placed_on_destination"], "no resource could be placed on the shared heap"
    assert result["iterations"] == 3
    assert result["copy_ms_min"] > 0

    # The WARP caveat must survive into the result: a software destination
    # proves the mechanism and nothing about the cost.
    assert result["representative"] is not result["destination_is_software"]


@pytest.mark.skipif(not native.is_available(), reason="native extension not built")
def test_d3d12_probe_exposed_through_the_python_wrapper():
    """The wrapper must exist and refuse a released frame."""
    from rapidshot.frame import Frame, FrameReleasedError

    assert hasattr(native, "probe_d3d12_sharing")
    frame = Frame(texture=object(), on_release=lambda: None, region=(0, 0, 4, 4))
    frame.release()
    with pytest.raises(FrameReleasedError):
        native.probe_d3d12_sharing(frame)
