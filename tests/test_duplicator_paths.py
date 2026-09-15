"""The Duplicator branches that live capture on a healthy desktop never reaches.

Measured with line coverage over the full suite: the paths below ran zero times.
They are the ones that only fire when something has already gone wrong -- a
duplication built against a device that has since been lost, a cursor shape the
driver will not hand over, a ReleaseFrame that fails -- plus the move-rect
reader, which DWM never exercises on a composited desktop (see
test_move_rects.py: zero move rects across 3,768 frames of dragging and
scrolling).

Everything is driven through the real Duplicator methods against fakes that
raise the HRESULTs Windows would, so no GPU is needed.
"""
import ctypes

import pytest

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot.core.duplicator as duplicator_module  # noqa: E402
from rapidshot._libs.dxgi import (  # noqa: E402
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_DEVICE_HUNG,
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_INVALID_CALL,
    DXGI_ERROR_MODE_CHANGE_IN_PROGRESS,
    DXGI_ERROR_MORE_DATA,
    DXGI_ERROR_NOT_FOUND,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_ERROR_WAIT_TIMEOUT,
    DXGI_OUTDUPL_MOVE_RECT,
    E_ACCESSDENIED,
    IDXGIResource,
    RECT,
)
from rapidshot.core.duplicator import (  # noqa: E402
    CURSOR_ERRORS,
    Cursor,
    Duplicator,
    _format_hresult,
)
from rapidshot.util.errors import (  # noqa: E402
    RapidShotConfigError,
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotError,
    RapidShotProtectedContentError,
    RapidShotReinitError,
)

E_FAIL = -2147467259  # 0x80004005: in none of the classification groups
E_NOINTERFACE = -2147467262


def com_error(hresult, message="injected failure"):
    return comtypes.COMError(hresult, message, (None, None, None, 0, None))


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeDuplication:
    """IDXGIOutputDuplication, scripted per test.

    `acquire` is called with the real DXGI_OUTDUPL_FRAME_INFO so a test can
    fill it in the way the driver would, or raise.
    """

    def __init__(self, acquire=None):
        self.acquire = acquire
        self.calls = []
        self.release_frame = lambda: None
        self.pointer_shape = lambda size, buf, required, info: 0
        self.move_rects = None
        self.dirty_rects = None

    def AcquireNextFrame(self, timeout, info_ref, res_ref):
        self.calls.append("AcquireNextFrame")
        if self.acquire is None:
            raise com_error(DXGI_ERROR_WAIT_TIMEOUT)
        self.acquire(info_ref._obj)

    def ReleaseFrame(self):
        self.calls.append("ReleaseFrame")
        self.release_frame()

    def GetFramePointerShape(self, size, buf, required, info):
        self.calls.append("GetFramePointerShape")
        return self.pointer_shape(size, buf, required, info)

    def GetFrameMoveRects(self, size, buf, used):
        self.calls.append(("GetFrameMoveRects", size))
        return self.move_rects(size, buf, used)

    def GetFrameDirtyRects(self, size, buf, used):
        self.calls.append(("GetFrameDirtyRects", size))
        if self.dirty_rects is None:
            used._obj.value = 0
            return 0
        return self.dirty_rects(size, buf, used)


def make_duplicator(duplication=None):
    """A Duplicator with the dataclass defaults, wired to a fake, no GPU."""
    dup = Duplicator.__new__(Duplicator)
    dup.duplicator = duplication if duplication is not None else FakeDuplication()
    dup.texture = None
    dup.updated = False
    dup.cursor = Cursor()
    dup.last_error = ""
    dup.cursor_visible = False
    dup.protected_content_detected = False
    dup.used_duplicate_output1 = True
    dup.timeout_ms = 10
    dup.last_present_time = 0
    dup.accumulated_frames = 0
    dup.dirty_rects = None
    dup.move_rects = None
    dup.rects_coalesced = False
    dup._frame_acquired = False
    dup._output_width, dup._output_height = 1920, 1080
    dup._rotation_angle = 0
    return dup


class FakeOutput:
    """The comtypes IDXGIOutput inside an Output wrapper."""

    def __init__(self, v1_error=None, legacy_error=None, has_output5=True):
        self.v1_error = v1_error
        self.legacy_error = legacy_error
        self.has_output5 = has_output5
        self.calls = []

    def QueryInterface(self, interface):
        if not self.has_output5:
            raise com_error(E_NOINTERFACE)
        return self

    def DuplicateOutput1(self, device, flags, count, formats, out_ref):
        self.calls.append("DuplicateOutput1")
        if self.v1_error is not None:
            raise com_error(self.v1_error)

    def DuplicateOutput(self, device, out_ref):
        self.calls.append("DuplicateOutput")
        if self.legacy_error is not None:
            raise com_error(self.legacy_error)


class FakeOutputWrapper:
    def __init__(self, inner, resolution=(2560, 1600), rotation=90):
        self.output = inner
        self.devicename = r"\\.\DISPLAY9"
        self.resolution = resolution
        self.rotation_angle = rotation


class FakeDevice:
    device = object()


class _Blocked:
    blocked_reason = "the input desktop is Winlogon, not Default"


class _Clear:
    blocked_reason = None


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------

def test_construction_records_the_outputs_geometry():
    dup = Duplicator(output=FakeOutputWrapper(FakeOutput()), device=FakeDevice())

    assert dup.used_duplicate_output1 is True
    assert dup.get_output_dimensions() == (2560, 1600)
    assert dup.get_rotation_angle() == 90


@pytest.mark.parametrize("hresult,expected", [
    (DXGI_ERROR_ACCESS_LOST, RapidShotReinitError),
    (DXGI_ERROR_DEVICE_REMOVED, RapidShotDeviceError),
    (DXGI_ERROR_UNSUPPORTED, RapidShotConfigError),
    (E_FAIL, RapidShotDXGIError),
])
def test_a_failed_construction_is_classified_not_leaked_as_comerror(hresult, expected):
    """Callers decide between rebuild-duplication, rebuild-device and give-up
    by exception type, so construction has to classify exactly like acquire."""
    inner = FakeOutput(v1_error=DXGI_ERROR_UNSUPPORTED, legacy_error=hresult)

    with pytest.raises(expected) as excinfo:
        Duplicator(output=FakeOutputWrapper(inner), device=FakeDevice())

    assert excinfo.value.hresult == hresult
    assert isinstance(excinfo.value.__cause__, comtypes.COMError)
    assert inner.calls == ["DuplicateOutput1", "DuplicateOutput"]


def test_pre_1607_output_goes_straight_to_the_legacy_path():
    """No IDXGIOutput5 is an old OS or driver, not a failure."""
    inner = FakeOutput(has_output5=False)
    dup = Duplicator.__new__(Duplicator)
    dup.protected_content_detected = False

    _, used_v1 = dup._create_duplication(FakeOutputWrapper(inner), FakeDevice())

    assert used_v1 is False
    assert inner.calls == ["DuplicateOutput"]


def test_access_denied_on_a_blocked_desktop_blames_the_desktop(monkeypatch):
    """The DuplicateOutput1 branch of the desktop fix: a locked workstation must
    not be reported as protected content, and must not fall back either."""
    monkeypatch.setattr(duplicator_module, "describe_desktop_access", _Blocked)
    inner = FakeOutput(v1_error=E_ACCESSDENIED)
    dup = Duplicator.__new__(Duplicator)
    dup.protected_content_detected = False

    with pytest.raises(RapidShotConfigError) as excinfo:
        dup._create_duplication(FakeOutputWrapper(inner), FakeDevice())

    assert "Winlogon" in str(excinfo.value)
    assert not isinstance(excinfo.value, RapidShotProtectedContentError)
    assert dup.protected_content_detected is False
    assert inner.calls == ["DuplicateOutput1"]


# --------------------------------------------------------------------------
# HRESULT classification, called directly
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hresult,expected", [
    (DXGI_ERROR_MODE_CHANGE_IN_PROGRESS, RapidShotReinitError),
    (DXGI_ERROR_DEVICE_HUNG, RapidShotDeviceError),
    (DXGI_ERROR_INVALID_CALL, RapidShotConfigError),
    (E_FAIL, RapidShotDXGIError),
])
def test_map_com_error_classifies_every_group(hresult, expected):
    dup = make_duplicator()

    error = dup._map_com_error(com_error(hresult, "boom"), "while testing")

    assert type(error) is expected
    assert error.hresult == hresult
    assert "while testing" in str(error) and "boom" in str(error)


def test_map_com_error_prefers_the_desktop_explanation(monkeypatch):
    monkeypatch.setattr(duplicator_module, "describe_desktop_access", _Blocked)
    dup = make_duplicator()

    error = dup._map_com_error(com_error(E_ACCESSDENIED), "ctx")

    assert type(error) is RapidShotConfigError
    assert dup.protected_content_detected is False


def test_map_com_error_flags_real_protected_content(monkeypatch):
    monkeypatch.setattr(duplicator_module, "describe_desktop_access", _Clear)
    dup = make_duplicator()

    error = dup._map_com_error(com_error(E_ACCESSDENIED), "ctx")

    assert type(error) is RapidShotProtectedContentError
    assert dup.protected_content_detected is True


def test_hresults_are_formatted_unsigned_and_other_values_pass_through():
    assert _format_hresult(DXGI_ERROR_ACCESS_LOST) == "0x887a0026"
    assert _format_hresult(None) == "None"
    assert _format_hresult("weird") == "weird"


# --------------------------------------------------------------------------
# update_frame
# --------------------------------------------------------------------------

def _presenting(info):
    info.LastPresentTime = 123456789
    info.AccumulatedFrames = 3
    info.RectsCoalesced = 1


def test_masked_protected_content_is_flagged_and_cleared_by_update_frame():
    """Replaces a test that set the flag itself and asserted it was set.

    Blanked protected content does not fail the acquire; the only signal is a
    bit in the frame info, and it must clear once the content is gone.
    """
    state = {"masked": True}

    def acquire(info):
        info.ProtectedContentMaskedOut = state["masked"]

    dup = make_duplicator(FakeDuplication(acquire))

    assert dup.update_frame() is True
    assert dup.protected_content_detected is True

    dup.release_frame()
    state["masked"] = False
    dup.update_frame()
    assert dup.protected_content_detected is False


def test_a_cursor_only_update_keeps_the_last_shape_but_moves_the_pointer():
    """DXGI hands over a shape only when it changes. A failed read must not
    wipe the shape already held, or the cursor vanishes until the next change."""
    def acquire(info):
        info.LastMouseUpdateTime = 42
        info.PointerShapeBufferSize = 64
        info.PointerPosition.Position.x = 300
        info.PointerPosition.Position.y = 400
        info.PointerPosition.Visible = True

    duplication = FakeDuplication(acquire)

    def not_found(*args):
        raise com_error(DXGI_ERROR_NOT_FOUND)

    duplication.pointer_shape = not_found
    dup = make_duplicator(duplication)
    previous_shape = b"previous shape"
    dup.cursor.Shape = previous_shape

    assert dup.update_frame() is True

    assert dup.updated is False, "no present time means no new frame content"
    assert dup.cursor.Shape is previous_shape
    assert (dup.cursor.PointerPositionInfo.Position.x,
            dup.cursor.PointerPositionInfo.Position.y) == (300, 400)
    assert dup.cursor_visible  # a Win32 BOOL, so 1 rather than True


def test_access_lost_while_reading_the_cursor_rebuilds_duplication():
    """get_frame_pointer_shape re-raises ACCESS_LOST so update_frame's recovery
    runs; swallowing it there would leave capture calling a dead interface."""
    def acquire(info):
        info.LastMouseUpdateTime = 1
        info.PointerShapeBufferSize = 64

    duplication = FakeDuplication(acquire)

    def lost(*args):
        raise com_error(DXGI_ERROR_ACCESS_LOST)

    duplication.pointer_shape = lost
    dup = make_duplicator(duplication)

    with pytest.raises(RapidShotReinitError):
        dup.update_frame()

    assert dup.duplicator is None


def test_move_rects_are_read_before_dirty_rects():
    """Both live in one metadata buffer with the move rects at the front; the
    order matches Microsoft's sample and is load-bearing on some drivers."""
    def acquire(info):
        _presenting(info)
        info.TotalMetadataBufferSize = 256

    duplication = FakeDuplication(acquire)
    duplication.move_rects = lambda size, buf, used: setattr(used._obj, "value", 0)

    class GoodResource(IDXGIResource):
        _iid_ = comtypes.GUID("{7f5b2a4e-0000-4000-8000-000000000001}")

        def QueryInterface(self, interface):
            return "texture"

    dup = make_duplicator(duplication)
    original = duplicator_module.IDXGIResource
    duplicator_module.IDXGIResource = GoodResource
    try:
        assert dup.update_frame() is True
    finally:
        duplicator_module.IDXGIResource = original

    metadata_calls = [c[0] for c in duplication.calls if isinstance(c, tuple)]
    assert metadata_calls == ["GetFrameMoveRects", "GetFrameDirtyRects"]
    assert dup.updated is True
    assert dup.texture == "texture"
    assert (dup.last_present_time, dup.accumulated_frames) == (123456789, 3)
    assert dup.rects_coalesced is True
    assert (dup.move_rects, dup.dirty_rects) == ([], [])


def test_a_texture_query_failure_skips_the_frame_but_keeps_duplication(monkeypatch):
    """One unusable surface is not a dead duplication: report no update, keep
    the interface, and leave the frame acquired so the caller releases it."""
    class BadResource(IDXGIResource):
        _iid_ = comtypes.GUID("{7f5b2a4e-0000-4000-8000-000000000002}")

        def QueryInterface(self, interface):
            raise com_error(E_NOINTERFACE)

    monkeypatch.setattr(duplicator_module, "IDXGIResource", BadResource)
    dup = make_duplicator(FakeDuplication(_presenting))

    assert dup.update_frame() is True

    assert dup.updated is False
    assert dup.duplicator is not None
    assert dup._frame_acquired is True
    assert "Failed to query texture interface" in dup.last_error


def test_only_frames_with_new_content_advance_the_frame_serial(monkeypatch):
    """Capture uses (instance_id, frame_serial) to tell whether a frame directly
    follows the one its dirty-rect accumulator holds."""
    class GoodResource(IDXGIResource):
        _iid_ = comtypes.GUID("{7f5b2a4e-0000-4000-8000-000000000004}")

        def QueryInterface(self, interface):
            return "texture"

    monkeypatch.setattr(duplicator_module, "IDXGIResource", GoodResource)
    steps = iter(["cursor", "present", "present"])

    def acquire(info):
        if next(steps) == "present":
            info.LastPresentTime = 1
        else:
            info.LastMouseUpdateTime = 1

    dup = make_duplicator(FakeDuplication(acquire))

    counts = []
    for _ in range(3):
        try:
            dup.update_frame()
        finally:
            dup.release_frame()
        counts.append(dup.frame_serial)

    assert counts[0] == 0, "a cursor-only acquire changes no pixels"
    assert counts[1:] == [1, 2]

    dup.duplicator.acquire = None           # timeout: nothing acquired at all
    dup.update_frame()
    assert dup.frame_serial == 2


def test_a_frame_whose_texture_is_unusable_still_counts(monkeypatch):
    """Its dirty rects were consumed by acquiring it, so the next frame's rects
    no longer chain to the one before -- even though nothing was captured."""
    class BadResource(IDXGIResource):
        _iid_ = comtypes.GUID("{7f5b2a4e-0000-4000-8000-000000000003}")

        def QueryInterface(self, interface):
            raise com_error(E_NOINTERFACE)

    monkeypatch.setattr(duplicator_module, "IDXGIResource", BadResource)
    dup = make_duplicator(FakeDuplication(_presenting))

    dup.update_frame()

    assert dup.updated is False
    assert dup.frame_serial == 1


def test_a_python_exception_during_acquire_is_wrapped():
    def acquire(info):
        raise ValueError("driver shim returned garbage")

    dup = make_duplicator(FakeDuplication(acquire))

    with pytest.raises(RapidShotError) as excinfo:
        dup.update_frame()

    assert isinstance(excinfo.value.__cause__, ValueError)
    assert dup.updated is False
    assert "driver shim returned garbage" in dup.last_error


# --------------------------------------------------------------------------
# release_frame / release
# --------------------------------------------------------------------------

def test_release_frame_without_a_frame_does_not_call_dxgi():
    """An unmatched ReleaseFrame is itself DXGI_ERROR_INVALID_CALL."""
    dup = make_duplicator()

    dup.release_frame()

    assert "ReleaseFrame" not in dup.duplicator.calls


def test_release_frame_drops_the_texture_before_releasing():
    """DXGI refuses the next acquire while any reference to the old surface is
    outstanding, so the texture has to be gone by the time ReleaseFrame runs."""
    dup = make_duplicator()
    dup._frame_acquired = True
    dup.texture = "surface"
    seen = {}
    dup.duplicator.release_frame = lambda: seen.setdefault("texture", dup.texture)

    dup.release_frame()

    assert seen == {"texture": None}
    assert dup._frame_acquired is False


def test_release_frame_after_the_duplication_is_gone_just_clears_state():
    dup = make_duplicator()
    dup.duplicator = None
    dup._frame_acquired = True
    dup.texture = "surface"

    dup.release_frame()

    assert dup._frame_acquired is False
    assert dup.texture is None


@pytest.mark.parametrize("failure,expected_prefix", [
    (com_error(E_FAIL, "release refused"), "Failed to release frame: "),
    (RuntimeError("wrapper exploded"), "Unexpected Python error releasing frame: "),
])
def test_release_frame_failures_are_recorded_not_raised(failure, expected_prefix):
    """Cleanup must not throw, but a real failure must stay visible."""
    dup = make_duplicator()
    dup._frame_acquired = True

    def fail():
        raise failure

    dup.duplicator.release_frame = fail
    dup.release_frame()

    assert dup.last_error == expected_prefix + str(failure)
    assert dup._frame_acquired is False


def test_already_released_is_not_recorded_as_an_error():
    dup = make_duplicator()
    dup._frame_acquired = True

    def already():
        raise com_error(DXGI_ERROR_INVALID_CALL)

    dup.duplicator.release_frame = already
    dup.release_frame()

    assert dup.last_error == ""


def test_release_lets_go_of_a_held_frame_first():
    dup = make_duplicator()
    dup._frame_acquired = True
    duplication = dup.duplicator

    dup.release()

    assert duplication.calls == ["ReleaseFrame"]
    assert dup.duplicator is None
    assert dup._frame_acquired is False


# --------------------------------------------------------------------------
# get_frame (compatibility wrapper)
# --------------------------------------------------------------------------

def test_get_frame_returns_none_on_a_static_desktop():
    assert make_duplicator().get_frame() is None


def test_get_frame_releases_what_it_acquired():
    """A wrapper that returned without ReleaseFrame would stall every later
    acquire with DXGI_ERROR_INVALID_CALL."""
    dup = make_duplicator()

    def fake_update():
        dup._frame_acquired = True
        dup.updated = True
        dup.texture = "surface"
        dup.cursor_visible = True
        return True

    dup.update_frame = fake_update
    info = dup.get_frame()

    assert (info.width, info.height, info.cursor_visible) == (1920, 1080, True)
    assert info.rect == "surface"
    assert dup.duplicator.calls == ["ReleaseFrame"]


def test_get_frame_releases_a_cursor_only_acquire():
    dup = make_duplicator()

    def fake_update():
        dup._frame_acquired = True
        dup.updated = False
        return True

    dup.update_frame = fake_update

    assert dup.get_frame() is None
    assert dup.duplicator.calls == ["ReleaseFrame"]


# --------------------------------------------------------------------------
# move and dirty rect readers
# --------------------------------------------------------------------------

class Info:
    def __init__(self, total=0, pointer=0):
        self.TotalMetadataBufferSize = total
        self.PointerShapeBufferSize = pointer


MOVE = ctypes.sizeof(DXGI_OUTDUPL_MOVE_RECT)


def acquired(**hooks):
    duplication = FakeDuplication()
    for name, hook in hooks.items():
        setattr(duplication, name, hook)
    dup = make_duplicator(duplication)
    dup._frame_acquired = True
    return dup


def write_moves(moves):
    def hook(size, buf, used):
        # DXGI refuses rather than overrun; a fake that wrote anyway would
        # corrupt memory through the ctypes pointer instead of failing a test.
        if len(moves) * MOVE > size:
            used._obj.value = len(moves) * MOVE
            raise com_error(DXGI_ERROR_MORE_DATA)
        for i, (sx, sy, left, top, right, bottom) in enumerate(moves):
            buf[i].SourcePoint.x, buf[i].SourcePoint.y = sx, sy
            rect = buf[i].DestinationRect
            rect.left, rect.top, rect.right, rect.bottom = left, top, right, bottom
        used._obj.value = len(moves) * MOVE
        return 0
    return hook


def test_move_rects_are_read_field_by_field():
    """Source point first, then the destination -- a swapped layout here would
    patch pixels from the wrong place without any error."""
    moves = [(10, 20, 30, 40, 50, 60), (-5, 7, 0, 0, 100, 200)]
    dup = acquired(move_rects=write_moves(moves))

    assert dup.get_frame_move_rects(Info(total=16 * MOVE)) == moves


def test_move_rects_report_only_what_dxgi_used():
    dup = acquired(move_rects=write_moves([(1, 2, 3, 4, 5, 6)]))

    assert len(dup.get_frame_move_rects(Info(total=32 * MOVE))) == 1


def test_move_rects_need_an_acquired_frame():
    dup = acquired(move_rects=write_moves([(1, 2, 3, 4, 5, 6)]))
    dup._frame_acquired = False

    assert dup.get_frame_move_rects(Info(total=MOVE)) is None


def test_no_move_metadata_is_an_empty_answer():
    dup = acquired(move_rects=write_moves([(1, 2, 3, 4, 5, 6)]))

    assert dup.get_frame_move_rects(Info(total=0)) == []
    assert dup.duplicator.calls == []


def test_move_rects_retry_once_at_the_size_dxgi_asks_for():
    attempts = []

    def hook(size, buf, used):
        attempts.append(size)
        if len(attempts) == 1:
            used._obj.value = 3 * MOVE
            raise com_error(DXGI_ERROR_MORE_DATA)
        return write_moves([(1, 2, 3, 4, 5, 6)] * 3)(size, buf, used)

    dup = acquired(move_rects=hook)

    assert len(dup.get_frame_move_rects(Info(total=MOVE))) == 3
    assert attempts == [MOVE, 3 * MOVE]


@pytest.mark.parametrize("required", [0, MOVE])
def test_move_rects_give_up_rather_than_loop(required):
    """MORE_DATA with no size, or MORE_DATA twice, is unknown -- not a hang."""
    attempts = []

    def hook(size, buf, used):
        attempts.append(size)
        used._obj.value = required
        raise com_error(DXGI_ERROR_MORE_DATA)

    dup = acquired(move_rects=hook)

    assert dup.get_frame_move_rects(Info(total=MOVE)) is None
    assert len(attempts) <= 2


@pytest.mark.parametrize("hresult", [DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_DEVICE_REMOVED])
def test_move_rects_let_dead_duplication_errors_through(hresult):
    def hook(size, buf, used):
        raise com_error(hresult)

    dup = acquired(move_rects=hook)

    with pytest.raises(comtypes.COMError):
        dup.get_frame_move_rects(Info(total=MOVE))


def test_other_move_rect_failures_mean_unknown():
    def hook(size, buf, used):
        raise com_error(E_FAIL)

    dup = acquired(move_rects=hook)

    assert dup.get_frame_move_rects(Info(total=MOVE)) is None
    assert "0x80004005" in dup.last_error


def test_dirty_rects_give_up_when_dxgi_names_no_size():
    def hook(size, buf, used):
        used._obj.value = 0
        raise com_error(DXGI_ERROR_MORE_DATA)

    dup = acquired(dirty_rects=hook)

    assert dup.get_frame_dirty_rects(Info(total=ctypes.sizeof(RECT))) is None


def test_other_dirty_rect_failures_mean_unknown():
    def hook(size, buf, used):
        raise com_error(E_FAIL)

    dup = acquired(dirty_rects=hook)

    assert dup.get_frame_dirty_rects(Info(total=ctypes.sizeof(RECT))) is None
    assert "0x80004005" in dup.last_error


@pytest.mark.parametrize("reader,hook", [
    ("get_frame_move_rects", "move_rects"),
    ("get_frame_dirty_rects", "dirty_rects"),
])
def test_an_implausible_metadata_size_is_refused_before_allocating(reader, hook):
    """The size is allocated from directly; a corrupt one was a multi-gigabyte
    allocation on the capture thread."""
    calls = []
    dup = acquired(**{hook: lambda *args: calls.append(args)})

    result = getattr(dup, reader)(Info(total=duplicator_module.MAX_METADATA_BUFFER_BYTES + 1))

    assert result is None, "unknown, not an empty list"
    assert calls == []
    assert "exceeds" in dup.last_error


@pytest.mark.parametrize("reader,hook", [
    ("get_frame_move_rects", "move_rects"),
    ("get_frame_dirty_rects", "dirty_rects"),
])
def test_an_implausible_size_asked_for_on_retry_is_refused_too(reader, hook):
    attempts = []

    def more_data(size, buf, used):
        attempts.append(size)
        used._obj.value = 0xFFFFFFF0
        raise com_error(DXGI_ERROR_MORE_DATA)

    dup = acquired(**{hook: more_data})

    assert getattr(dup, reader)(Info(total=64)) is None
    assert len(attempts) == 1


def test_metadata_at_the_limit_is_still_read():
    limit = duplicator_module.MAX_METADATA_BUFFER_BYTES
    dup = acquired(move_rects=write_moves([(1, 2, 3, 4, 5, 6)]))

    assert len(dup.get_frame_move_rects(Info(total=limit - limit % MOVE))) == 1


# --------------------------------------------------------------------------
# cursor shape reader
# --------------------------------------------------------------------------

def test_no_shape_buffer_means_no_call():
    dup = acquired()

    assert dup.get_frame_pointer_shape(Info(pointer=0)) == (
        False, False, CURSOR_ERRORS["NO_SHAPE"])
    assert dup.duplicator.calls == []


def test_a_shape_is_returned_with_its_info():
    def hook(size, buf, required, info):
        info._obj.Width, info._obj.Height = 32, 64
        return 0

    dup = acquired(pointer_shape=hook)

    info, shape, error = dup.get_frame_pointer_shape(Info(pointer=128))

    assert (info.Width, info.Height, error) == (32, 64, "")
    assert len(shape) == 128


def test_an_implausible_shape_size_is_refused_before_allocating():
    calls = []
    dup = acquired(pointer_shape=lambda *args: calls.append(args) or 0)

    result = dup.get_frame_pointer_shape(
        Info(pointer=duplicator_module.MAX_POINTER_SHAPE_BUFFER_BYTES + 1))

    assert result[:2] == (False, False)
    assert "exceeds" in result[2] and dup.last_error == result[2]
    assert calls == []


def test_a_shape_at_the_limit_is_still_read():
    dup = acquired(pointer_shape=lambda *args: 0)

    _, shape, error = dup.get_frame_pointer_shape(
        Info(pointer=duplicator_module.MAX_POINTER_SHAPE_BUFFER_BYTES))

    assert error == "" and len(shape) == duplicator_module.MAX_POINTER_SHAPE_BUFFER_BYTES


def test_a_failing_hresult_return_is_reported():
    dup = acquired(pointer_shape=lambda *args: -1)

    result = dup.get_frame_pointer_shape(Info(pointer=16))

    assert result[:2] == (False, False)
    assert "-1" in result[2]
    assert dup.last_error == result[2]


def test_shape_not_found_is_quiet_and_explained():
    def hook(*args):
        raise com_error(DXGI_ERROR_NOT_FOUND)

    dup = acquired(pointer_shape=hook)

    result = dup.get_frame_pointer_shape(Info(pointer=16))

    assert result[:2] == (False, False)
    assert "not found" in result[2]
    assert "0x887a0002" in result[2]


@pytest.mark.parametrize("failure", [com_error(E_FAIL), OSError("mapping failed")])
def test_other_shape_failures_are_returned_not_raised(failure):
    """Only ACCESS_LOST escapes; a bad cursor read must not stop frame capture."""
    def hook(*args):
        raise failure

    dup = acquired(pointer_shape=hook)

    result = dup.get_frame_pointer_shape(Info(pointer=16))

    assert result[:2] == (False, False)
    assert result[2] == dup.last_error != ""


# --------------------------------------------------------------------------
# small accessors
# --------------------------------------------------------------------------

def test_repr_reports_state_without_touching_com():
    dup = make_duplicator()
    assert repr(dup) == "<Duplicator Initialized:True Cursor:not available>"

    dup.duplicator = None
    dup.cursor.Shape = b"x"
    assert repr(dup) == "<Duplicator Initialized:False Cursor:available>"


def test_get_last_error_returns_the_recorded_message():
    dup = make_duplicator()
    dup.last_error = "something specific"
    assert dup.get_last_error() == "something specific"
