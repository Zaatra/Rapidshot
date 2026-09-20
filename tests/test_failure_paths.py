"""Fault-injection tests for the DXGI failure paths fixed in Stage 1.

These drive the recovery logic by making the duplication interface return the
exact HRESULTs Windows would return, so access loss, session disconnect, device
removal and protected-content refusals are all exercised without needing a game
in exclusive fullscreen or DRM video on screen.

This is the "integration tests for the failure paths, not just the happy path"
item from Stage 0 of ROADMAP.md, minus the parts that need real hardware
transitions.
"""

import gc
import weakref

import pytest

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

from test_capture_paths import pipeline  # noqa: E402,F401  (fixture)

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
    """Stands in for IDXGIOutputDuplication, failing on demand.

    `log` outlives the fake, so a test can still read what happened to it after
    the library has dropped its last reference.
    """

    def __init__(self, acquire_error=None, log=None):
        self.acquire_error = acquire_error
        self.acquire_calls = 0
        self.release_frame_calls = 0
        self.log = {"release_calls": 0} if log is None else log
        self.log.setdefault("release_calls", 0)

    def AcquireNextFrame(self, timeout, info_ref, res_ref):
        self.acquire_calls += 1
        if self.acquire_error is not None:
            raise com_error(self.acquire_error)
        raise com_error(DXGI_ERROR_WAIT_TIMEOUT)  # default: nothing new

    def ReleaseFrame(self):
        self.release_frame_calls += 1

    def Release(self):
        # A real comtypes pointer has this, and the library must no longer call
        # it: comtypes issues Release itself when the pointer is dropped, so an
        # explicit call decrements the COM refcount twice for one reference.
        self.log["release_calls"] += 1


def make_duplicator(acquire_error=None, log=None):
    """A Duplicator wired to a fake duplication object, no GPU required."""
    dup = Duplicator.__new__(Duplicator)
    dup.duplicator = FakeDuplication(acquire_error, log=log)
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

    "Dropped" is the whole contract: for a comtypes pointer, letting go of the
    last reference *is* the COM release, and calling Release() as well
    decrements the refcount twice. So this checks that nothing still holds the
    interface -- which the old assertion could not, since it only proved a
    method had been called.
    """
    log = {}
    dup = make_duplicator(acquire_error=hresult, log=log)
    fake = weakref.ref(dup.duplicator)

    with pytest.raises(RapidShotReinitError):
        dup.update_frame()

    assert dup.duplicator is None
    gc.collect()
    assert fake() is None, (
        "something still holds the invalidated duplication, so its COM "
        "reference was never released")
    assert log["release_calls"] == 0, (
        "an explicit Release() on top of dropping the pointer over-releases it")
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
    log = {}
    dup = make_duplicator(acquire_error=DXGI_ERROR_DEVICE_REMOVED, log=log)
    fake = weakref.ref(dup.duplicator)
    with pytest.raises(RapidShotDeviceError):
        dup.update_frame()
    assert dup.duplicator is None
    gc.collect()
    assert fake() is None
    assert log["release_calls"] == 0


# --------------------------------------------------------------------------
# Error formatting (regression: invalid f-string format specifiers)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "hresult, expected",
    [
        (DXGI_ERROR_ACCESS_LOST, RapidShotReinitError),
        (DXGI_ERROR_DEVICE_REMOVED, RapidShotDeviceError),
        (DXGI_ERROR_INVALID_CALL, RapidShotConfigError),
    ],
)
def test_error_paths_do_not_raise_valueerror_while_formatting(hresult, expected):
    """
    The error messages used to contain
    ``{hresult:#010x if isinstance(hresult, int) else hresult}``, which Python
    parses as an invalid *format specifier*. Every one of these paths raised
    ValueError on top of the original DXGI failure.

    pytest.raises names the exact type, so a ValueError from the formatter --
    or the wrong RapidShot* classification -- fails here instead of being
    absorbed.
    """
    dup = make_duplicator(acquire_error=hresult)
    with pytest.raises(expected) as excinfo:
        dup.update_frame()
    assert excinfo.value.hresult == hresult
    assert f"{hresult & 0xFFFFFFFF:#010x}" in dup.last_error


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


# Masked-out (blanked rather than refused) protected content is covered through
# update_frame itself in test_duplicator_paths.py; the test that lived here set
# the flag by hand and asserted it had been set.


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

    def always_failing_duplicator(output, device, timeout_ms=10):
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
    cam._timeout_ms = 10                          # __init__ always sets this
    cam._all_devices = [None]                     # ditto: adapters to try

    assert cam._on_output_change() is False       # reports failure
    assert attempts["n"] == 5                     # bounded, did not spin
    assert cam._needs_reinit is True
    assert "Failed to rebuild" in cam._last_capture_error_message


def test_stage_surface_failure_releases_duplicator_before_retry(monkeypatch):
    """A retry must not coexist with the duplicator from the failed attempt."""
    import rapidshot.capture as capture_module
    from rapidshot.capture import ScreenCapture

    built = []

    class FakeDuplicator:
        def __init__(self, output=None, device=None, timeout_ms=10):
            assert not built or built[-1].released, (
                "a new duplicator was created while the previous attempt was live")
            self.released = False
            built.append(self)

        def release(self):
            assert not self.released, "partial duplicator released twice"
            self.released = True

    class FakeStageSurf:
        def __init__(self):
            self.rebuild_calls = 0
            self.release_calls = 0
            self.has_resource = True

        def release(self):
            self.release_calls += 1
            self.has_resource = False

        def rebuild(self, output=None, device=None):
            assert not self.has_resource, (
                "partial stage surface survived into the next retry")
            self.rebuild_calls += 1
            self.has_resource = True
            if self.rebuild_calls < 3:
                raise com_error(DXGI_ERROR_DEVICE_RESET)

    class FakeOutput:
        devicename = "FAKE"
        resolution = (1920, 1080)
        rotation_angle = 0

        def update_desc(self):
            pass

    monkeypatch.setattr(capture_module, "Duplicator", FakeDuplicator)
    monkeypatch.setattr(capture_module.time, "sleep", lambda _seconds: None)

    stage = FakeStageSurf()
    cam = ScreenCapture.__new__(ScreenCapture)
    cam._duplicator = None
    cam._stagesurf = stage
    cam._output = FakeOutput()
    cam._device = None
    cam._all_devices = [None]
    cam._timeout_ms = 10
    cam.width, cam.height = 1920, 1080
    cam.region = (0, 0, 1920, 1080)
    cam._region_set_by_user = False
    cam._sourceRegion = None
    cam.is_capturing = False
    cam.rotation_angle = 0
    cam._needs_reinit = False
    cam._last_capture_error_message = ""
    cam._max_output_change_retries = 3

    assert cam._on_output_change() is True
    assert len(built) == 3
    assert [duplicator.released for duplicator in built] == [True, True, False]
    assert cam._duplicator is built[-1]
    assert stage.rebuild_calls == 3
    assert stage.release_calls == 3  # initial teardown + two failed attempts
    assert stage.has_resource is True


def test_output_change_gives_up_immediately_on_protected_content(monkeypatch):
    """Retrying a protected-content refusal is pointless; fail fast."""
    import rapidshot.capture as capture_module
    from rapidshot.capture import ScreenCapture

    attempts = {"n": 0}

    def protected_duplicator(output, device, timeout_ms=10):
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
    # _on_output_change rebuilds the Duplicator and carries the caller's
    # acquire timeout across; __init__ always sets this, but these fixtures
    # build the object with __new__.
    cam._timeout_ms = 10
    # Candidate adapters. Just the one here, so the rebuild tries exactly one
    # per attempt, which is what the attempt counts below are asserting.
    cam._all_devices = [None]

    assert cam._on_output_change() is False
    assert attempts["n"] == 1  # no retry storm


# --------------------------------------------------------------------------
# Duplication must try every adapter, not just the one that owns the output
# --------------------------------------------------------------------------

def _capture_stub(primary, fallbacks):
    """A ScreenCapture with just enough state to run _build_duplicator."""
    from rapidshot.capture import ScreenCapture

    class FakeOutput:
        devicename = r"\\.\DISPLAY1"

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._output = FakeOutput()
    cam._device = primary
    cam._all_devices = [primary] + list(fallbacks)   # ordered candidates
    cam._timeout_ms = 10
    return cam


def test_duplication_falls_back_to_another_adapter(monkeypatch):
    """The adapter owning the output is not always the one DDA accepts.

    Measured 2026-08-21 on a hybrid laptop: the display-owning adapter refused
    DuplicateOutput with DXGI_ERROR_UNSUPPORTED. Refusing there used to end
    capture, even where another adapter would have been granted duplication.
    """
    import rapidshot.capture as capture_module
    from rapidshot.util.errors import RapidShotConfigError

    good, bad = object(), object()
    tried = []

    def picky_duplicator(output, device, timeout_ms=10):
        tried.append(device)
        if device is bad:
            raise RapidShotConfigError("refused", hresult=DXGI_ERROR_UNSUPPORTED)
        return "duplicator-on-good"

    monkeypatch.setattr(capture_module, "Duplicator", picky_duplicator)
    cam = _capture_stub(primary=bad, fallbacks=[good])

    assert cam._build_duplicator() == "duplicator-on-good"
    assert tried == [bad, good]           # primary first, then the fallback
    # The stage surface is built on self._device and must land on the same
    # adapter as the duplicated texture, so the winner has to be recorded.
    assert cam._device is good


def test_the_original_adapter_stays_a_candidate_after_a_fallback_wins(monkeypatch):
    """Winning once must not remove the loser from the pool.

    `_build_duplicator` reassigns `self._device` to whichever adapter was
    granted duplication. If the candidate list were "everything except the
    starting device", the original would vanish the moment a fallback won --
    and a later rebuild where the fallback starts refusing and the original is
    valid again would fail with a working adapter available. Ordering is a
    preference; the set has to stay whole.
    """
    import rapidshot.capture as capture_module
    from rapidshot.util.errors import RapidShotConfigError

    original, fallback = object(), object()
    refuse = {original}

    def picky_duplicator(output, device, timeout_ms=10):
        if device in refuse:
            raise RapidShotConfigError("refused", hresult=DXGI_ERROR_UNSUPPORTED)
        return f"duplicator-{'original' if device is original else 'fallback'}"

    monkeypatch.setattr(capture_module, "Duplicator", picky_duplicator)
    cam = _capture_stub(primary=original, fallbacks=[fallback])

    assert cam._build_duplicator() == "duplicator-fallback"
    assert cam._device is fallback

    # The situation reverses -- a MUX flip, an output change -- and the
    # original becomes the only adapter that will duplicate.
    refuse.clear()
    refuse.add(fallback)
    assert cam._build_duplicator() == "duplicator-original"
    assert cam._device is original


def test_prefer_integrated_reaches_an_output_less_igpu(monkeypatch):
    """The flag has to outrank the display-owning adapter, or it does nothing.

    On a hybrid laptop the iGPU usually owns no output, so it is never the
    device the factory selects. If it were merely ranked ahead of the *other*
    fallbacks it would still sit behind the display-owning adapter -- which on
    a working system duplicates successfully, so the integrated adapter would
    never be tried at all. The flag would be inert in exactly the topology it
    exists for.
    """
    import rapidshot.capture as capture_module

    igpu, dgpu = object(), object()
    tried = []

    def duplicator(output, device, timeout_ms=10):
        tried.append(device)
        return "duplicator"

    monkeypatch.setattr(capture_module, "Duplicator", duplicator)
    # As the factory ranks them with prefer_integrated=True: iGPU first, even
    # though dgpu is the adapter that owns the output.
    cam = _capture_stub(primary=dgpu, fallbacks=[])
    cam._all_devices = [igpu, dgpu]

    assert cam._build_duplicator() == "duplicator"
    assert tried == [igpu], "the integrated adapter was not tried first"
    assert cam._device is igpu


def test_capture_cache_separates_adapter_preferences(monkeypatch):
    """Opposite duplication orders must not share a cached capture.

    On the Optimus topology this option exists for, the iGPU owns no output.
    Both calls therefore retain the same public device/output indices; only the
    candidate order differs. A two-component cache key returned the first
    capture and never tried the requested order on the second call.
    """
    import weakref
    import rapidshot

    class Device:
        def __init__(self, description):
            self.desc = type("Desc", (), {"Description": description})()

    class Output:
        devicename = "DISPLAY1"

        def update_desc(self):
            pass

    class Capture:
        def __init__(self, **kwargs):
            self.candidates = tuple(kwargs["candidate_devices"])

    dgpu, igpu = Device("NVIDIA GeForce"), Device("Intel Graphics")
    factory = object.__new__(rapidshot.RapidshotFactory)
    factory.devices = [dgpu]
    factory.outputs = [[Output()]]
    factory.all_devices = [dgpu, igpu]
    factory.output_metadata = {"DISPLAY1": (None, True)}
    factory._screencapture_instances = weakref.WeakValueDictionary()

    monkeypatch.setattr(rapidshot, "ScreenCapture", Capture)
    monkeypatch.setattr(rapidshot.time, "sleep", lambda _seconds: None)

    display_first = factory.create(output_idx=0, prefer_integrated=False)
    integrated_first = factory.create(output_idx=0, prefer_integrated=True)

    assert display_first is not integrated_first
    assert display_first.candidates == (dgpu, igpu)
    assert integrated_first.candidates == (igpu, dgpu)
    assert factory.create(output_idx=0, prefer_integrated=True) is integrated_first


def test_create_does_not_return_a_released_camera(monkeypatch):
    """`camera.release(); camera = rapidshot.create()` must get a new camera.

    The factory caches cameras weakly, so a released one stays in the cache
    while anything references it -- and in that idiom the old object is still
    bound when create() runs. It used to be returned, and since a released
    camera never captures again, every grab() yielded None with no error.
    """
    import weakref
    import rapidshot

    class Output:
        devicename = "DISPLAY1"

        def update_desc(self):
            pass

    class Capture:
        def __init__(self, **kwargs):
            self.released = False

        def release(self):
            self.released = True

    device = type("Device", (), {"desc": type("Desc", (), {"Description": "GPU"})()})()
    factory = object.__new__(rapidshot.RapidshotFactory)
    factory.devices = [device]
    factory.outputs = [[Output()]]
    factory.all_devices = [device]
    factory.output_metadata = {"DISPLAY1": (None, True)}
    factory._screencapture_instances = weakref.WeakValueDictionary()
    monkeypatch.setattr(rapidshot, "ScreenCapture", Capture)
    monkeypatch.setattr(rapidshot.time, "sleep", lambda _seconds: None)

    camera = factory.create(output_idx=0)
    assert factory.create(output_idx=0) is camera, "a live camera is still shared"

    camera.release()
    replacement = factory.create(output_idx=0)
    assert replacement is not camera
    assert replacement.released is False
    assert factory.create(output_idx=0) is replacement


def _fake_factory(monkeypatch):
    """A factory over one fake output whose cameras are plain objects."""
    import weakref
    import rapidshot

    class Output:
        devicename = "DISPLAY1"

        def update_desc(self):
            pass

    class Capture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.released = False

        def release(self):
            self.released = True

    device = type("Device", (), {"desc": type("Desc", (), {"Description": "GPU"})()})()
    factory = object.__new__(rapidshot.RapidshotFactory)
    factory.devices = [device]
    factory.outputs = [[Output()]]
    factory.all_devices = [device]
    factory.output_metadata = {"DISPLAY1": (None, True)}
    factory._screencapture_instances = weakref.WeakValueDictionary()
    monkeypatch.setattr(rapidshot, "ScreenCapture", Capture)
    monkeypatch.setattr(rapidshot.time, "sleep", lambda _seconds: None)
    return factory


@pytest.mark.parametrize("change", [
    {"output_color": "BGRA"},
    {"region": (0, 0, 100, 100)},
    {"nvidia_gpu": True},
    {"pool_output": False},
    {"timeout_ms": 0},
    {"max_buffer_len": 8},
    # Must differ from the constructor default, or there is no mismatch
    # to detect: this was 2 until 2026-09-14, when 2 became the default.
    {"pool_size_frames": 3},
])
def test_create_refuses_to_return_a_camera_built_with_other_settings(monkeypatch, change):
    """The cache is keyed by output; the settings must match too.

    It used to be keyed by (device, output, prefer_integrated) alone, so
    `create(output_color="BGRA")` after an RGB camera on the same output got
    the RGB camera back -- wrong channel count, and nothing said so.
    """
    import rapidshot

    factory = _fake_factory(monkeypatch)
    camera = factory.create(output_idx=0)
    with pytest.raises(rapidshot.ConfigurationError) as err:
        factory.create(output_idx=0, **change)
    (name,) = change
    assert name in str(err.value) and "release()" in str(err.value)
    assert factory.create(output_idx=0) is camera, "matching settings still share"


def test_settings_can_change_once_the_camera_is_released(monkeypatch):
    factory = _fake_factory(monkeypatch)
    rgb = factory.create(output_idx=0, output_color="RGB")
    rgb.release()
    bgra = factory.create(output_idx=0, output_color="BGRA")
    assert bgra is not rgb
    assert bgra.kwargs["output_color"] == "BGRA"


def test_a_repeat_request_matches_even_after_the_cupy_fallback(monkeypatch):
    """nvidia_gpu is compared as requested, not as the fallback rewrote it."""
    import builtins

    factory = _fake_factory(monkeypatch)
    real_import = builtins.__import__

    def no_cupy(name, *args, **kwargs):
        if name == "cupy":
            raise ImportError("no cupy here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    first = factory.create(output_idx=0, nvidia_gpu=True)
    assert first.kwargs["nvidia_gpu"] is False          # fell back
    assert factory.create(output_idx=0, nvidia_gpu=True) is first


def test_a_successful_adapter_moves_to_the_front(monkeypatch):
    """Winner first on the next rebuild -- reordering only, never removing."""
    import rapidshot.capture as capture_module
    from rapidshot.util.errors import RapidShotConfigError

    first, second = object(), object()

    def picky(output, device, timeout_ms=10):
        if device is first:
            raise RapidShotConfigError("refused", hresult=DXGI_ERROR_UNSUPPORTED)
        return "duplicator"

    monkeypatch.setattr(capture_module, "Duplicator", picky)
    cam = _capture_stub(primary=first, fallbacks=[second])

    assert cam._build_duplicator() == "duplicator"
    assert cam._all_devices[0] is second, "the winner was not promoted"
    assert set(map(id, cam._all_devices)) == {id(first), id(second)}, (
        "promotion dropped an adapter instead of reordering")


def test_duplication_does_not_retry_a_non_adapter_refusal(monkeypatch):
    """A desktop refusal applies to every adapter equally.

    It is also a RapidShotConfigError, so retrying on the type alone would
    burn through every adapter and then replace an already-actionable message
    ("you are not on the input desktop") with a generic one.
    """
    import rapidshot.capture as capture_module
    from rapidshot._libs.dxgi import E_ACCESSDENIED
    from rapidshot.util.errors import RapidShotConfigError

    tried = []

    def refusing_duplicator(output, device, timeout_ms=10):
        tried.append(device)
        raise RapidShotConfigError("not on the input desktop", hresult=E_ACCESSDENIED)

    monkeypatch.setattr(capture_module, "Duplicator", refusing_duplicator)
    cam = _capture_stub(primary=object(), fallbacks=[object(), object()])

    with pytest.raises(RapidShotConfigError, match="input desktop"):
        cam._build_duplicator()
    assert len(tried) == 1                # gave up after the first, as it should


def test_duplication_failure_explains_itself(monkeypatch):
    """Every adapter refusing must produce a diagnosis, not an HRESULT."""
    import rapidshot.capture as capture_module
    from rapidshot.util.errors import RapidShotConfigError

    def always_refusing(output, device, timeout_ms=10):
        raise RapidShotConfigError("refused", hresult=DXGI_ERROR_UNSUPPORTED)

    monkeypatch.setattr(capture_module, "Duplicator", always_refusing)
    cam = _capture_stub(primary=object(), fallbacks=[object()])

    with pytest.raises(RapidShotConfigError) as excinfo:
        cam._build_duplicator()
    message = str(excinfo.value)
    assert "No adapter" in message
    assert "What each adapter reported" in message


# --------------------------------------------------------------------------
# The "nothing is arriving" warning must describe elapsed time, not attempts
# --------------------------------------------------------------------------

def _quiet_warning_fired(pipeline, monkeypatch, clock_values, updated_flags):
    """Run real grab() calls over a scripted clock; return the warnings logged.

    This used to re-implement the warning logic inside the test and drive the
    copy, so capture.py itself was never run: its three lines could be deleted
    with every one of these tests still passing. Now each step is a grab() on a
    ScreenCapture over the fake pipeline, and the clock is read wherever the
    library reads it.
    """
    import rapidshot.capture as capture_module
    from test_capture_paths import FakeDuplicator

    cam, _, _, _ = pipeline(pool_output=False)
    warnings = []
    monkeypatch.setattr(capture_module.logger, "warning",
                        lambda msg, *a, **k: warnings.append(msg))
    clock = {"now": 0.0}
    monkeypatch.setattr(capture_module.time, "perf_counter", lambda: clock["now"])

    FakeDuplicator.script = ["frame" if updated else "timeout" for updated in updated_flags]
    FakeDuplicator.position = 0
    for now, updated in zip(clock_values, updated_flags):
        clock["now"] = now
        frame = cam.grab()
        assert (frame is not None) == updated
    return [w for w in warnings if "No screen updates" in w]


def test_polling_misses_do_not_warn_while_frames_are_arriving(pipeline, monkeypatch):
    """A run of empty acquires is normal and must not be reported as a still screen.

    With `timeout_ms=0` the capture loop makes tens of thousands of calls a
    second and ~97% return nothing, so counting *consecutive* misses fired this
    warning seven times while capture was running at 117 fps. The question is
    how long it has been since a frame, not how many times we asked.
    """
    # 600 polls across 3 seconds, a frame every 20th -- i.e. 5 ms apart, a
    # perfectly healthy 200 fps with a 95% miss rate.
    clock = [i * 0.005 for i in range(600)]
    flags = [(i % 20 == 0) for i in range(600)]
    assert _quiet_warning_fired(pipeline, monkeypatch, clock, flags) == []


def test_a_genuinely_still_screen_still_warns(pipeline, monkeypatch):
    """The warning must survive: a real stall is worth reporting."""
    clock = [0.0] + [1.0 + i * 0.5 for i in range(12)]
    flags = [True] + [False] * 12
    fired = _quiet_warning_fired(pipeline, monkeypatch, clock, flags)
    assert fired, "a still screen should still produce a warning"
    assert "No screen updates for 2.0s" in fired[0]


def test_still_screen_warning_is_rate_limited(pipeline, monkeypatch):
    """Once every couple of seconds, not once per poll."""
    clock = [0.0] + [1.0 + i * 0.01 for i in range(1200)]   # 12s of polling
    flags = [True] + [False] * 1200
    fired = _quiet_warning_fired(pipeline, monkeypatch, clock, flags)
    assert 1 <= len(fired) <= 8, f"expected a handful of warnings, got {len(fired)}"


def test_a_screen_still_from_the_first_grab_warns_too(pipeline, monkeypatch):
    """No frame has ever arrived: the clock starts at the first empty grab."""
    fired = _quiet_warning_fired(pipeline, monkeypatch, [0.0, 1.0, 2.5], [False] * 3)
    assert fired == ["No screen updates for 2.5s. Desktop Duplication only reports "
                     "changed content, so a still screen produces no frames by design."]


# --------------------------------------------------------------------------
# timeout_ms is public, validated, and survives a duplication rebuild
# --------------------------------------------------------------------------

def test_timeout_ms_defaults_to_blocking():
    """The default blocks rather than polls: 4x less CPU for ~7% fewer frames."""
    from rapidshot.core.duplicator import Duplicator
    assert Duplicator.timeout_ms == 10


@pytest.mark.parametrize("bad", [-1, 1.5, "10", None, True])
def test_timeout_ms_rejects_nonsense(bad):
    """A silently-ignored bad value would look like the setting having no effect.

    `True` is in here deliberately: bool is a subclass of int, so a naive
    isinstance check accepts it and the duplicator would block for 1 ms.
    """
    from rapidshot.capture import ScreenCapture

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._duplicator = None
    with pytest.raises(ValueError):
        ScreenCapture.timeout_ms.fset(cam, bad)


def test_timeout_ms_setter_reaches_the_live_duplicator():
    """Setting it must take effect on the next acquire, not the next rebuild."""
    from rapidshot.capture import ScreenCapture

    class FakeDup:
        timeout_ms = 10

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._timeout_ms = 10
    cam._duplicator = FakeDup()

    ScreenCapture.timeout_ms.fset(cam, 0)
    assert cam._timeout_ms == 0
    assert cam._duplicator.timeout_ms == 0
    assert ScreenCapture.timeout_ms.fget(cam) == 0


def test_timeout_ms_survives_a_duplication_rebuild(monkeypatch):
    """The rebuild path must carry the caller's timeout across.

    `_on_output_change` constructs a fresh Duplicator. Dropping the setting
    there would reset it to the default on the first resolution change or
    display reconnect -- a regression that only ever shows up as "it got slower
    after I unplugged a monitor".
    """
    import rapidshot.capture as capture_module
    from rapidshot.capture import ScreenCapture

    built = {}

    class FakeDuplicator:
        def __init__(self, output=None, device=None, timeout_ms=10):
            built["timeout_ms"] = timeout_ms
            self.timeout_ms = timeout_ms

    class FakeStageSurf:
        def rebuild(self, output=None, device=None):
            pass

        def release(self):
            pass

    class FakeOutput:
        devicename = "FAKE"
        resolution = (1920, 1080)
        rotation_angle = 0

        def update_desc(self):
            pass

    monkeypatch.setattr(capture_module, "Duplicator", FakeDuplicator)

    cam = ScreenCapture.__new__(ScreenCapture)
    cam._timeout_ms = 0                      # the caller asked for polling
    cam._all_devices = [None]                # __init__ sets this; __new__ does not
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

    assert cam._on_output_change() is True
    assert built["timeout_ms"] == 0, "rebuild reset the caller's timeout"
