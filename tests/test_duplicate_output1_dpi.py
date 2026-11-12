"""DuplicateOutput1 needs a per-monitor DPI aware caller.

It answers DXGI_ERROR_UNSUPPORTED (0x887A0004) to anything less -- measured
2026-11-11: an unaware or system-aware process refused every time, per-monitor
v1 and v2 accepted, and so did a thread-only v2 context. ``core/output.py``
asks for per-monitor awareness process-wide, but a host that fixed another
awareness first keeps it, and then every capture silently took the legacy path
with no 10-bit or HDR formats. So the Duplicator raises the calling thread for
that one call and puts the thread's own setting back, as rapidshot-core does.

The tests make their own thread DPI unaware, which is what such a host looks
like from inside, and restore it afterwards.
"""
import ctypes
import logging
import sys

import pytest

if sys.platform != "win32":
    pytest.skip("Windows-only", allow_module_level=True)

comtypes = pytest.importorskip("comtypes", reason="COM is Windows-only")

import rapidshot  # noqa: E402
import rapidshot.core.duplicator as duplicator_module  # noqa: E402
from rapidshot._libs.dxgi import DXGI_ERROR_UNSUPPORTED  # noqa: E402
from rapidshot.core.duplicator import Duplicator  # noqa: E402

LOGGER = "rapidshot.core.duplicator"
DPI_AWARENESS_CONTEXT_UNAWARE = -1
DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4


def com_error(hresult):
    return comtypes.COMError(hresult, "injected", (None, None, None, 0, None))


def _thread_context(context, expected_awareness):
    api = duplicator_module._thread_dpi()
    if api is None:
        pytest.skip("this Windows has no per-thread DPI awareness (before 10 1607)")
    _, set_context, _ = api
    previous = set_context(ctypes.c_void_p(context))
    assert previous, "could not set the test thread's DPI awareness"
    assert duplicator_module._thread_dpi_awareness() == expected_awareness
    return set_context, previous


@pytest.fixture
def unaware_thread():
    set_context, previous = _thread_context(DPI_AWARENESS_CONTEXT_UNAWARE, 0)
    yield
    set_context(ctypes.c_void_p(previous))


@pytest.fixture
def per_monitor_thread():
    set_context, previous = _thread_context(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, 2)
    yield
    set_context(ctypes.c_void_p(previous))


class WindowsLikeOutput:
    """IDXGIOutput5 that refuses DuplicateOutput1, as Windows does, unless the
    calling thread is per-monitor DPI aware. Records what it saw."""

    def __init__(self, refuse_anyway=False):
        self.refuse_anyway = refuse_anyway
        self.seen = []
        self.calls = []

    def QueryInterface(self, interface):
        return self

    def DuplicateOutput1(self, device, flags, count, formats, out_ref):
        awareness = duplicator_module._thread_dpi_awareness()
        self.seen.append(awareness)
        self.calls.append("DuplicateOutput1")
        if self.refuse_anyway or awareness != 2:
            raise com_error(DXGI_ERROR_UNSUPPORTED)

    def DuplicateOutput(self, device, out_ref):
        self.calls.append("DuplicateOutput")


class OutputWrapper:
    def __init__(self, inner):
        self.output = inner
        self.devicename = r"\\.\DISPLAY9"
        self.resolution = (1920, 1080)
        self.rotation_angle = 0


class FakeDevice:
    device = object()


def test_an_unaware_thread_gets_duplicate_output1_and_keeps_its_setting(
    unaware_thread, caplog
):
    inner = WindowsLikeOutput()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        dup = Duplicator(output=OutputWrapper(inner), device=FakeDevice())

    assert dup.used_duplicate_output1 is True
    assert inner.calls == ["DuplicateOutput1"]
    assert inner.seen == [2], "the call was not made per-monitor DPI aware"
    assert duplicator_module._thread_dpi_awareness() == 0, "the thread was not put back"
    assert "This thread is DPI unaware; it was made per-monitor DPI aware" in caplog.text


def test_a_refusal_for_another_reason_does_not_blame_dpi(unaware_thread, caplog):
    inner = WindowsLikeOutput(refuse_anyway=True)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        dup = Duplicator(output=OutputWrapper(inner), device=FakeDevice())

    assert dup.used_duplicate_output1 is False
    assert inner.calls == ["DuplicateOutput1", "DuplicateOutput"]
    assert "which offers no 10-bit or HDR formats" in caplog.text
    assert "could not be made per-monitor" not in caplog.text
    assert duplicator_module._thread_dpi_awareness() == 0


def test_a_thread_already_per_monitor_is_left_alone(per_monitor_thread, caplog):
    inner = WindowsLikeOutput()
    with caplog.at_level(logging.INFO, logger=LOGGER):
        Duplicator(output=OutputWrapper(inner), device=FakeDevice())
    assert inner.seen == [2]
    assert "made per-monitor DPI aware" not in caplog.text


def test_a_real_duplicator_from_an_unaware_thread(unaware_thread, caplog, desktop_duplication):
    """The same on the real screen, where the PC allows DuplicateOutput1 at
    all. A machine may refuse it for its own reasons (a basic display driver);
    DPI must then not be one of them."""
    try:
        camera = rapidshot.create()
    except Exception as e:  # no display, no adapter
        pytest.skip(f"no capturable display: {e}")
    output, device = camera._output, camera._device
    # DXGI allows one duplication of an output per device.
    camera.release()

    with caplog.at_level(logging.INFO, logger=LOGGER):
        dup = Duplicator(output=output, device=device)
    try:
        assert duplicator_module._thread_dpi_awareness() == 0
        if not dup.used_duplicate_output1:
            assert "could not be made per-monitor" not in caplog.text, caplog.text
    finally:
        dup.release()
