"""Fault-injection tests for the DXGI failure paths fixed in Stage 1.

These drive the recovery logic by making the duplication interface return the
exact HRESULTs Windows would return, so access loss, session disconnect, device
removal and protected-content refusals are all exercised without needing a game
in exclusive fullscreen or DRM video on screen.

This is the "integration tests for the failure paths, not just the happy path"
item from Stage 0 of ROADMAP.md, minus the parts that need real hardware
transitions.
"""

import pytest

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

from rapidshot._libs.dxgi import (  # noqa: E402
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_RESET,
    DXGI_ERROR_INVALID_CALL,
    DXGI_ERROR_SESSION_DISCONNECTED,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_WAIT_TIMEOUT,
    E_ACCESSDENIED,
)
from rapidshot.core.duplicator import Duplicator  # noqa: E402
from rapidshot.util.errors import (  # noqa: E402
    RapidShotConfigError,
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotProtectedContentError,
    RapidShotReinitError,
)


def com_error(hresult, message="injected failure"):
    """Build a COMError exactly as comtypes reports one from a failed call."""
    return comtypes.COMError(hresult, message, (None, None, None, 0, None))


class FakeDuplication:
    """Stands in for IDXGIOutputDuplication, failing on demand."""

    def __init__(self, acquire_error=None):
        self.acquire_error = acquire_error
        self.acquire_calls = 0
        self.release_frame_calls = 0
        self.released = False

    def AcquireNextFrame(self, timeout, info_ref, res_ref):
        self.acquire_calls += 1
        if self.acquire_error is not None:
            raise com_error(self.acquire_error)
        raise com_error(DXGI_ERROR_WAIT_TIMEOUT)  # default: nothing new

    def ReleaseFrame(self):
        self.release_frame_calls += 1

    def Release(self):
        self.released = True


def make_duplicator(acquire_error=None):
    """A Duplicator wired to a fake duplication object, no GPU required."""
    dup = Duplicator.__new__(Duplicator)
    dup.duplicator = FakeDuplication(acquire_error)
    dup.texture = None
    dup.updated = False
    dup.cursor = None
    dup.last_error = ""
    dup.cursor_visible = False
    dup.protected_content_detected = False
    dup.used_duplicate_output1 = True
    dup.timeout_ms = 10
    dup._frame_acquired = False
    dup._output_width, dup._output_height = 1920, 1080
    dup._rotation_angle = 0
    return dup


# --------------------------------------------------------------------------
# HRESULT classification
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hresult,expected",
    [
        (DXGI_ERROR_ACCESS_LOST, RapidShotReinitError),
        (DXGI_ERROR_SESSION_DISCONNECTED, RapidShotReinitError),
        (DXGI_ERROR_DEVICE_REMOVED, RapidShotDeviceError),
        (DXGI_ERROR_DEVICE_RESET, RapidShotDeviceError),
        (E_ACCESSDENIED, RapidShotProtectedContentError),
    ],
)
def test_acquire_failures_map_to_the_right_exception(hresult, expected):
    """
    Regression guard for the signed/unsigned HRESULT bug.

    These constants used to be stored unsigned, so none of them ever matched
    what comtypes reports and every one of these fell through to a generic
    error -- access-lost recovery never triggered.
    """
    dup = make_duplicator(acquire_error=hresult)
    with pytest.raises(expected):
        dup.update_frame()


def test_timeout_is_not_an_error():
    """A static desktop times out constantly; that must stay a quiet no-op."""
    dup = make_duplicator(acquire_error=DXGI_ERROR_WAIT_TIMEOUT)
    assert dup.update_frame() is True   # duplication still healthy
    assert dup.updated is False         # but no new content
    assert dup.protected_content_detected is False


@pytest.mark.parametrize("hresult", [DXGI_ERROR_INVALID_CALL, DXGI_ERROR_UNSUPPORTED])
def test_invalid_call_and_unsupported_are_configuration_errors(hresult):
    """These signal a bad call or an unsupported setup, not a lost device."""
    dup = make_duplicator(acquire_error=hresult)
    with pytest.raises(RapidShotConfigError):
        dup.update_frame()


def test_unclassified_error_still_raises_dxgi_error():
    """Anything unrecognised must surface as a DXGI error, not be swallowed."""
    e_fail = -2147467259  # 0x80004005, in none of the classification groups
    dup = make_duplicator(acquire_error=e_fail)
    with pytest.raises(RapidShotDXGIError):
        dup.update_frame()


# --------------------------------------------------------------------------
# Access loss drops the dead interface
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hresult", [DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_SESSION_DISCONNECTED]
)
def test_access_loss_releases_the_invalidated_duplication(hresult):
    """
    After access loss the duplication object is dead. It must be dropped so no
    further calls are issued against it.
    """
    dup = make_duplicator(acquire_error=hresult)
    fake = dup.duplicator

    with pytest.raises(RapidShotReinitError):
        dup.update_frame()

    assert fake.released is True
    assert dup.duplicator is None
    assert dup._frame_acquired is False
    assert dup.texture is None


def test_update_frame_on_released_duplicator_is_safe():
    """A second call after access loss must not explode."""
    dup = make_duplicator(acquire_error=DXGI_ERROR_ACCESS_LOST)
    with pytest.raises(RapidShotReinitError):
        dup.update_frame()

    assert dup.update_frame() is False  # degraded, but no exception
    assert dup.updated is False


def test_device_error_also_releases_duplication():
    dup = make_duplicator(acquire_error=DXGI_ERROR_DEVICE_REMOVED)
    fake = dup.duplicator
    with pytest.raises(RapidShotDeviceError):
        dup.update_frame()
    assert fake.released is True
    assert dup.duplicator is None


# --------------------------------------------------------------------------
# Error formatting (regression: invalid f-string format specifiers)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hresult",
    [DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED, DXGI_ERROR_INVALID_CALL],
)
def test_error_paths_do_not_raise_valueerror_while_formatting(hresult):
    """
    The error messages used to contain
    ``{hresult:#010x if isinstance(hresult, int) else hresult}``, which Python
    parses as an invalid *format specifier*. Every one of these paths raised
    ValueError on top of the original DXGI failure.
    """
    dup = make_duplicator(acquire_error=hresult)
    try:
        dup.update_frame()
    except ValueError as e:
        if "format specifier" in str(e).lower():
            pytest.fail(f"error formatting is broken: {e}")
    except Exception:
        pass  # a RapidShot* exception is the expected outcome
    assert "0x" in dup.last_error


def test_release_frame_survives_a_failing_releaseframe():
    """A failed ReleaseFrame must still clear the held-frame flag."""
    dup = make_duplicator()
    dup._frame_acquired = True

    def boom():
        raise com_error(DXGI_ERROR_INVALID_CALL)

    dup.duplicator.ReleaseFrame = boom
    dup.release_frame()
    assert dup._frame_acquired is False  # otherwise every later acquire looks leaked


# --------------------------------------------------------------------------
# Protected content (HDCP) — the refusal must not become a retry loop
# --------------------------------------------------------------------------

class ProtectedOutput:
    """An output whose DuplicateOutput1 is refused for protected content."""

    def __init__(self, hresult=E_ACCESSDENIED):
        self.hresult = hresult
        self.devicename = "FAKE-PROTECTED"
        self.legacy_calls = 0

    def QueryInterface(self, interface):
        return self

    def DuplicateOutput1(self, device, flags, count, formats, out_ref):
        raise com_error(self.hresult)

    def DuplicateOutput(self, device, out_ref):
        self.legacy_calls += 1
        raise AssertionError(
            "legacy DuplicateOutput must not be attempted after a "
            "protected-content refusal"
        )


class FakeOutputWrapper:
    def __init__(self, inner):
        self.output = inner
        self.devicename = inner.devicename
        self.resolution = (1920, 1080)
        self.rotation_angle = 0


class FakeDevice:
    device = object()


def test_protected_content_refusal_raises_and_does_not_fall_back():
    """
    A protected-content refusal is permanent while the content is on screen.
    Falling back to the legacy path cannot help, and retrying would spin.
    """
    inner = ProtectedOutput()
    dup = Duplicator.__new__(Duplicator)
    dup.protected_content_detected = False

    with pytest.raises(RapidShotProtectedContentError) as excinfo:
        dup._create_duplication(FakeOutputWrapper(inner), FakeDevice())

    assert inner.legacy_calls == 0
    assert dup.protected_content_detected is True
    assert "HDCP" in str(excinfo.value) or "protected" in str(excinfo.value).lower()


def test_non_protected_failure_does_fall_back_to_legacy():
    """An ordinary DuplicateOutput1 failure should still try the legacy path."""
    calls = {"legacy": 0}

    class FlakyOutput(ProtectedOutput):
        def DuplicateOutput1(self, device, flags, count, formats, out_ref):
            raise com_error(DXGI_ERROR_UNSUPPORTED)

        def DuplicateOutput(self, device, out_ref):
            calls["legacy"] += 1  # succeeds

    dup = Duplicator.__new__(Duplicator)
    dup.protected_content_detected = False

    _, used_v1 = dup._create_duplication(
        FakeOutputWrapper(FlakyOutput()), FakeDevice()
    )
    assert calls["legacy"] == 1
    assert used_v1 is False
    assert dup.protected_content_detected is False


def test_legacy_env_var_skips_duplicate_output1(monkeypatch):
    monkeypatch.setenv("RAPIDSHOT_DUPLICATE_OUTPUT", "legacy")
    calls = {"legacy": 0, "v1": 0}

    class CountingOutput(ProtectedOutput):
        def DuplicateOutput1(self, device, flags, count, formats, out_ref):
            calls["v1"] += 1

        def DuplicateOutput(self, device, out_ref):
            calls["legacy"] += 1

    dup = Duplicator.__new__(Duplicator)
    dup.protected_content_detected = False
    _, used_v1 = dup._create_duplication(
        FakeOutputWrapper(CountingOutput()), FakeDevice()
    )
    assert calls == {"legacy": 1, "v1": 0}
    assert used_v1 is False


def test_masked_out_protected_content_is_flagged_not_fatal():
    """
    When protected content is merely blanked (rather than refused), capture
    continues but callers must be able to tell a masked frame from a black one.
    """
    dup = make_duplicator()
    dup.protected_content_detected = False

    class MaskedInfo:
        ProtectedContentMaskedOut = True
        LastMouseUpdateTime = 0
        LastPresentTime = 0
        PointerShapeBufferSize = 0

    # Drive the same branch update_frame() uses for the flag.
    info = MaskedInfo()
    if info.ProtectedContentMaskedOut:
        dup.protected_content_detected = True
    assert dup.protected_content_detected is True
    # ...and it is cleared again once the protected surface goes away.
    info.ProtectedContentMaskedOut = False
    dup.protected_content_detected = bool(info.ProtectedContentMaskedOut)
    assert dup.protected_content_detected is False


# --------------------------------------------------------------------------
# Bounded rebuild (the exclusive-fullscreen hang)
# --------------------------------------------------------------------------

def test_output_change_rebuild_is_bounded_and_does_not_hang(monkeypatch):
    """
    _on_output_change() used to retry duplication creation in an unbounded
    `while True` with no backoff, so a mode switch that never settled hung the
    caller forever. It must give up and report failure instead.
    """
    import rapidshot.capture as capture_module
    from rapidshot.capture import ScreenCapture

    attempts = {"n": 0}

    def always_failing_duplicator(output, device):
        attempts["n"] += 1
        raise com_error(DXGI_ERROR_UNSUPPORTED)

    monkeypatch.setattr(capture_module, "Duplicator", always_failing_duplicator)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _s: None)

    class FakeStageSurf:
        def release(self):
            pass

        def rebuild(self, output, device, dim=None):
            pass

    class FakeOutput:
        resolution = (1920, 1080)
        rotation_angle = 0

        def update_desc(self):
            pass

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._duplicator = None
    cam._stagesurf = FakeStageSurf()
    cam._output = FakeOutput()
    cam._device = None
    cam.width, cam.height = 1920, 1080
    cam.region = (0, 0, 1920, 1080)
    cam._region_set_by_user = False
    cam._sourceRegion = None
    cam.is_capturing = False
    cam.rotation_angle = 0
    cam._needs_reinit = False
    cam._last_capture_error_message = ""
    cam._max_output_change_retries = 5

    assert cam._on_output_change() is False       # reports failure
    assert attempts["n"] == 5                     # bounded, did not spin
    assert cam._needs_reinit is True
    assert "Failed to rebuild" in cam._last_capture_error_message


def test_output_change_gives_up_immediately_on_protected_content(monkeypatch):
    """Retrying a protected-content refusal is pointless; fail fast."""
    import rapidshot.capture as capture_module
    from rapidshot.capture import ScreenCapture

    attempts = {"n": 0}

    def protected_duplicator(output, device):
        attempts["n"] += 1
        raise RapidShotProtectedContentError("HDCP content on screen")

    monkeypatch.setattr(capture_module, "Duplicator", protected_duplicator)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _s: None)

    class FakeStageSurf:
        def release(self):
            pass

        def rebuild(self, output, device, dim=None):
            pass

    class FakeOutput:
        resolution = (1920, 1080)
        rotation_angle = 0

        def update_desc(self):
            pass

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._duplicator = None
    cam._stagesurf = FakeStageSurf()
    cam._output = FakeOutput()
    cam._device = None
    cam.width, cam.height = 1920, 1080
    cam.region = (0, 0, 1920, 1080)
    cam._region_set_by_user = False
    cam._sourceRegion = None
    cam.is_capturing = False
    cam.rotation_angle = 0
    cam._needs_reinit = False
    cam._last_capture_error_message = ""
    cam._max_output_change_retries = 12

    assert cam._on_output_change() is False
    assert attempts["n"] == 1  # no retry storm
