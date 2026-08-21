"""Telling "not on the input desktop" apart from "protected content".

``DuplicateOutput`` reports both as a bare ``E_ACCESSDENIED``. RapidShot used to
call every one of them protected content, so a locked workstation, an open UAC
prompt, and a Session 0 service all advised the user to "close or move the
protected player window" — a window that does not exist. The Session 0 case is
not hypothetical: it is exactly what the downstream consumer in ROADMAP § 12
runs as.

The refusal is reproduced for real, not injected: a thread moved to a fresh
desktop with ``CreateDesktop`` gets the identical HRESULT from DXGI. That runs
in a subprocess because ``SetThreadDesktop`` is sticky and would poison capture
for the rest of the process's life.
"""
import json
import subprocess
import sys
import textwrap

import pytest

from rapidshot._libs.dxgi import (
    DXGI_ERROR_ACCESS_DENIED,
    DXGI_ERROR_CANNOT_PROTECT_CONTENT,
    E_ACCESSDENIED,
)
from rapidshot.core.duplicator import _desktop_refusal
from rapidshot.util.desktop import describe_desktop_access
from rapidshot.util.errors import RapidShotConfigError

REPO_ROOT = str(__import__("pathlib").Path(__file__).resolve().parent.parent)


# --------------------------------------------------------------------------
# the diagnostic itself, on a normal desktop
# --------------------------------------------------------------------------

def test_normal_session_is_the_input_desktop():
    state = describe_desktop_access()
    if state.thread_desktop is None:
        pytest.skip("no window station (headless or non-Windows)")
    assert state.is_input_desktop is True
    assert state.blocked_reason is None


def test_no_desktop_blame_when_the_desktop_is_fine():
    """The regression that matters: real protected content must still be
    reported as protected content, not silently reattributed to the desktop."""
    if describe_desktop_access().thread_desktop is None:
        pytest.skip("no window station")
    assert _desktop_refusal(E_ACCESSDENIED) is None


@pytest.mark.parametrize("hresult", [DXGI_ERROR_ACCESS_DENIED,
                                     DXGI_ERROR_CANNOT_PROTECT_CONTENT])
def test_protected_specific_codes_are_left_alone(hresult):
    """Only the generic E_ACCESSDENIED is ambiguous.

    These two are specific to protected content, so the desktop check must not
    apply to them even on a desktop that would otherwise explain a refusal.
    """
    assert _desktop_refusal(hresult) is None


def test_config_error_carries_the_hresult():
    state = describe_desktop_access()
    if state.thread_desktop is None:
        pytest.skip("no window station")
    # Force the blocked path without touching the real desktop.
    import rapidshot.core.duplicator as dup

    class _Blocked:
        blocked_reason = "test reason"

    original = dup.describe_desktop_access
    dup.describe_desktop_access = lambda: _Blocked()
    try:
        error = dup._desktop_refusal(E_ACCESSDENIED, "context")
    finally:
        dup.describe_desktop_access = original

    assert isinstance(error, RapidShotConfigError)
    assert "test reason" in str(error)
    assert "context" in str(error)
    assert error.hresult == E_ACCESSDENIED


# --------------------------------------------------------------------------
# the real refusal, in a subprocess on a non-input desktop
# --------------------------------------------------------------------------

CHILD = textwrap.dedent(
    r"""
    import ctypes, json, sys
    sys.path.insert(0, %(root)r)

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.CreateDesktopW.restype = ctypes.c_void_p
    user32.CreateDesktopW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                                      ctypes.c_void_p, ctypes.c_ulong,
                                      ctypes.c_ulong, ctypes.c_void_p]
    user32.SetThreadDesktop.argtypes = [ctypes.c_void_p]

    out = {}
    desktop = user32.CreateDesktopW("rapidshot_pytest_probe", None, None, 0,
                                    0x10000000, None)
    if not desktop or not user32.SetThreadDesktop(ctypes.c_void_p(desktop)):
        print(json.dumps({"skip": "could not create or attach a desktop"}))
        raise SystemExit(0)

    from rapidshot.util.desktop import describe_desktop_access
    state = describe_desktop_access()
    out["thread_desktop"] = state.thread_desktop
    out["input_desktop"] = state.input_desktop
    out["is_input_desktop"] = state.is_input_desktop
    out["blocked_reason"] = state.blocked_reason

    import logging
    logging.disable(logging.CRITICAL)
    import rapidshot
    try:
        rapidshot.create(output_color="BGRA")
        out["created"] = True
    except BaseException as e:
        out["created"] = False
        out["message"] = str(e)
    print(json.dumps(out))
    """
) % {"root": REPO_ROOT}


@pytest.fixture(scope="module")
def child_result():
    if sys.platform != "win32":
        pytest.skip("Windows only")
    proc = subprocess.run([sys.executable, "-c", CHILD],
                          capture_output=True, text=True, timeout=180)
    line = next((l for l in reversed(proc.stdout.splitlines())
                 if l.startswith("{")), None)
    if line is None:
        pytest.skip(f"probe produced no result: {proc.stderr[-300:]}")
    data = json.loads(line)
    if "skip" in data:
        pytest.skip(data["skip"])
    return data


def test_non_input_desktop_is_detected(child_result):
    assert child_result["is_input_desktop"] is False
    assert child_result["thread_desktop"] == "rapidshot_pytest_probe"
    assert child_result["input_desktop"] not in (None, "rapidshot_pytest_probe")


def test_non_input_desktop_reason_names_the_real_cause(child_result):
    reason = child_result["blocked_reason"] or ""
    assert "input" in reason.lower()
    assert "rapidshot_pytest_probe" in reason
    # The old message's advice, which sent people looking for a window that
    # was not there. Its absence is the whole point of the fix.
    assert "protected" not in reason.lower()
    assert "player window" not in reason.lower()


def test_capture_from_a_non_input_desktop_fails(child_result):
    """It must still fail — the fix is about the message, not about making
    capture work from a desktop that genuinely cannot see the screen."""
    assert child_result["created"] is False
