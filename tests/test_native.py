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


def _mock_cross_adapter(inner):
    transfer = native.CrossAdapterTransfer.__new__(native.CrossAdapterTransfer)
    transfer._inner = inner
    return transfer


def test_failed_signal_with_fallback_drain_does_not_quarantine_frame():
    """A mocked private-fence drain makes ReleaseFrame safe despite the error."""
    import ctypes
    from rapidshot.frame import Frame

    class DrainedFailure:
        submission_quarantined = False

        def transfer_async(self, _texture, _source_id):
            raise RuntimeError("injected shared Signal failure; fallback drained")

    released = []
    frame = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))
    with pytest.raises(RuntimeError, match="fallback drained"):
        _mock_cross_adapter(DrainedFailure()).transfer_async(frame)

    frame.release()
    assert released == [True]
    assert frame.released


def test_untrackable_failed_signal_quarantines_frame():
    """Both mocked Signal calls failing must prevent DXGI ReleaseFrame."""
    import ctypes
    from rapidshot.frame import Frame, FrameQuarantinedError

    class UntrackableFailure:
        submission_quarantined = True

        def transfer_async(self, _texture, _source_id):
            raise RuntimeError("injected primary and fallback Signal failures")

    released = []
    frame = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))
    with pytest.raises(RuntimeError, match="primary and fallback"):
        _mock_cross_adapter(UntrackableFailure()).transfer_async(frame)

    with pytest.raises(FrameQuarantinedError, match="restart the process"):
        frame.release()
    assert released == []
    assert not frame.released


@pytest.mark.parametrize("failure", [
    "injected CreateEventW failure",
    "injected SetEventOnCompletion failure",
])
def test_async_drain_failure_quarantines_instead_of_releasing(failure):
    """A failed completion wait must never hand a live GPU surface to DXGI."""
    import ctypes
    from rapidshot.frame import Frame, FrameQuarantinedError

    class FailedWait:
        submission_quarantined = False

        def transfer_async(self, _texture, _source_id):
            return 17

        def wait_shared_fence(self, value):
            assert value == 17
            raise RuntimeError(failure)

    released = []
    frame = Frame(ctypes.c_void_p(1), lambda: released.append(True), (0, 0, 4, 4))
    transfer = _mock_cross_adapter(FailedWait())
    assert transfer.transfer_async(frame) == 17

    with pytest.raises(FrameQuarantinedError, match="restart the process"):
        frame.release()
    assert released == []
    assert not frame.released

    # The failed drain becomes a persistent quarantine, not a one-shot error
    # that allows a later release to invalidate the surface.
    with pytest.raises(FrameQuarantinedError, match=failure):
        frame.release()


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


# -- version-gated features ------------------------------------------------
#
# `native = ["rapidshot-native>=0.1.0"]` shipped while 2.6 was calling
# `GpuConverter12` and `TensorTransfer`, neither of which 0.1.0 exports. The
# extra resolved, and the transform path then died on `AttributeError: module
# '_rapidshot_native' has no attribute 'GpuConverter12'` -- a message naming
# neither the cause nor the cure. `pyproject.toml`'s floor is the real fix;
# `require_feature` is the backstop, and these are its tests.


class _OldExtension:
    """A wheel that predates the 2.6 symbols, as 0.1.0 actually was."""

    def build_info(self):
        return {"version": "0.1.0", "stage": "6-m3b-d3d12-preprocess"}


def test_require_feature_returns_the_symbol_when_present(monkeypatch):
    sentinel = object()

    class New:
        GpuConverter12 = sentinel

        def build_info(self):
            return {"version": "0.2.0", "stage": "x"}

    monkeypatch.setattr(native, "_ext", New())
    assert native.require_feature("GpuConverter12") is sentinel


@pytest.mark.parametrize("symbol", ["GpuConverter12", "TensorTransfer"])
def test_require_feature_names_the_version_that_provides_it(monkeypatch, symbol):
    """The whole point: say which version, not which attribute is missing."""
    monkeypatch.setattr(native, "_ext", _OldExtension())
    monkeypatch.setattr(native, "_ext_source", "rapidshot-native wheel")

    with pytest.raises(RuntimeError) as excinfo:
        native.require_feature(symbol)

    message = str(excinfo.value)
    assert symbol in message
    assert "0.2.0" in message, "must name the version that provides the feature"
    assert "0.1.0" in message, "must name the version that is installed"
    assert "pip install --upgrade" in message, "must say how to fix it"
    assert "AttributeError" not in message


def test_require_feature_still_explains_an_absent_extension(monkeypatch):
    """No extension at all is a different problem and keeps the build hint."""
    monkeypatch.setattr(native, "_ext", None)
    monkeypatch.setattr(native, "_import_error", ImportError("no module"))

    with pytest.raises(RuntimeError) as excinfo:
        native.require_feature("GpuConverter12")
    assert "not installed" in str(excinfo.value)


def test_require_feature_on_an_unknown_symbol_does_not_invent_a_version(monkeypatch):
    """A symbol missing from the table is a bug here, not the user's old wheel,
    so it must not claim some version would fix it."""
    monkeypatch.setattr(native, "_ext", _OldExtension())

    with pytest.raises(RuntimeError) as excinfo:
        native.require_feature("SomethingNeverShipped")
    message = str(excinfo.value)
    assert "SomethingNeverShipped" in message
    assert "0.2.0" not in message


def test_every_gated_feature_exists_in_the_built_extension():
    """The table must not gate a symbol the current extension lacks -- that
    would turn a working install into a spurious upgrade demand."""
    if not native.is_available():
        pytest.skip("native extension not built")
    extension = native.require()
    missing = [name for name in native._FEATURE_SINCE
               if not hasattr(extension, name)]
    assert not missing, (
        f"_FEATURE_SINCE gates {missing}, which this extension does not export; "
        "rebuild it (python native/install_dev.py) or fix the table"
    )


def test_extension_version_tracks_availability():
    version = native.extension_version()
    if native.is_available():
        assert isinstance(version, str) and version
    else:
        assert version is None


def test_extension_version_survives_a_build_info_that_raises(monkeypatch):
    """An extension too old to report a version must not crash the check."""

    class Hostile:
        def build_info(self):
            raise RuntimeError("no build_info in this build")

    monkeypatch.setattr(native, "_ext", Hostile())
    assert native.extension_version() is None
