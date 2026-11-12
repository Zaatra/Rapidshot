"""The native Desktop Duplication (``core/native_duplicator.py``).

The parity that matters most is not here: the whole suite runs a second time
with ``RAPIDSHOT_DUPLICATOR=native``, so every capture test -- dirty rects,
rebuilds, cursors, the regions, the lock-screen refusal in
``test_desktop_refusal.py`` -- drives the native class too. This file covers
what that run cannot: that the switch selects it, that each error kind becomes
the class ``ScreenCapture`` acts on, and that it reads the same things the
Python class reads from the same screen.
"""
import ctypes
import json
import os
import subprocess
import sys

import pytest

import rapidshot
from rapidshot import capture
from rapidshot._libs.d3d11 import D3D11_TEXTURE2D_DESC
from rapidshot._libs.dxgi import (
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_UNSUPPORTED,
    DXGI_OUTDUPL_POINTER_POSITION,
    DXGI_OUTDUPL_POINTER_SHAPE_INFO,
    E_ACCESSDENIED,
)
from rapidshot.core import native_duplicator
from rapidshot.core.backend import CaptureBackend
from rapidshot.core.duplicator import Duplicator
from rapidshot.core.native_duplicator import NativeDuplicator
from rapidshot.util.errors import (
    RapidShotConfigError,
    RapidShotDeviceError,
    RapidShotDXGIError,
    RapidShotProtectedContentError,
    RapidShotReinitError,
)

pytestmark = pytest.mark.skipif(
    not native_duplicator.is_available(), reason="extension has no DesktopDuplicator"
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def target(desktop_duplication):
    """The primary output and its device, from an ordinary camera.

    The camera is released first: DXGI allows one duplication of an output per
    device, and the camera's own would make every one built here fail with
    E_INVALIDARG. The output and device outlive it.
    """
    camera = rapidshot.create()
    output, device = camera._output, camera._device
    camera.release()
    yield output, device


def held_frame(backend, attempts=60):
    """Acquire until a frame with content is held; the first one after
    creation always carries the whole desktop."""
    for _ in range(attempts):
        assert backend.update_frame(), backend.last_error
        if backend.updated:
            return backend
        if backend._frame_acquired:
            backend.release_frame()
    pytest.fail("no frame with content")


def texture_desc(texture):
    desc = D3D11_TEXTURE2D_DESC()
    texture.GetDesc(ctypes.byref(desc))
    return desc.Width, desc.Height, desc.Format


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def test_the_python_class_is_the_default(monkeypatch):
    monkeypatch.delenv(native_duplicator.NATIVE_DUPLICATOR_ENV, raising=False)
    assert capture._duplicator_class() is Duplicator


def test_the_environment_selects_the_native_class(monkeypatch, desktop_duplication):
    monkeypatch.setenv(native_duplicator.NATIVE_DUPLICATOR_ENV, "native")
    assert capture._duplicator_class() is NativeDuplicator
    camera = rapidshot.create()
    try:
        assert isinstance(camera._duplicator, NativeDuplicator)
        frame = None
        for _ in range(60):
            frame = camera.grab()
            if frame is not None:
                break
        assert frame is not None and frame.ndim == 3
    finally:
        camera.release()


# --------------------------------------------------------------------------
# the contract and parity with the Python class
# --------------------------------------------------------------------------


def test_it_meets_the_capture_backend_contract(target):
    output, device = target
    backend = NativeDuplicator(output=output, device=device, timeout_ms=10)
    try:
        assert isinstance(backend, CaptureBackend)
    finally:
        backend.release()


def test_it_reads_what_the_python_class_reads(target):
    output, device = target
    seen = {}
    for cls in (Duplicator, NativeDuplicator):
        backend = held_frame(cls(output=output, device=device, timeout_ms=50))
        try:
            seen[cls] = dict(
                desc=texture_desc(backend.texture),
                output1=backend.used_duplicate_output1,
                serial=backend.frame_serial,
                rects_are_a_list=isinstance(backend.dirty_rects, list),
                moves_are_a_list=isinstance(backend.move_rects, list),
            )
            assert backend.last_present_time > 0
            assert backend.accumulated_frames >= 1
        finally:
            backend.release_frame()
            backend.release()
    assert seen[Duplicator] == seen[NativeDuplicator]


def test_the_cursor_is_the_python_class_shapes(target):
    output, device = target
    backend = held_frame(NativeDuplicator(output=output, device=device, timeout_ms=50))
    try:
        cursor = backend.cursor
        assert isinstance(cursor.PointerPositionInfo, DXGI_OUTDUPL_POINTER_POSITION)
        assert isinstance(cursor.PointerShapeInfo, DXGI_OUTDUPL_POINTER_SHAPE_INFO)
        if cursor.Shape is not None:
            info = cursor.PointerShapeInfo
            assert len(cursor.Shape) >= info.Pitch * (info.Height // (2 if info.Type == 1 else 1))
    finally:
        backend.release()


def test_release_is_idempotent_and_a_released_backend_reports_unhealthy(target):
    output, device = target
    backend = held_frame(NativeDuplicator(output=output, device=device, timeout_ms=50))
    backend.release_frame()
    backend.release_frame()
    backend.release()
    backend.release()
    assert backend.duplicator is None
    assert backend.update_frame() is False
    assert backend.texture is None


def test_the_timeout_reaches_the_native_side(target):
    output, device = target
    backend = NativeDuplicator(output=output, device=device, timeout_ms=10)
    try:
        backend.timeout_ms = 3
        assert backend._native.timeout_ms == 3
    finally:
        backend.release()


def test_the_legacy_switch_is_honoured(target, monkeypatch):
    monkeypatch.setenv("RAPIDSHOT_DUPLICATE_OUTPUT", "legacy")
    output, device = target
    backend = NativeDuplicator(output=output, device=device, timeout_ms=10)
    try:
        assert backend.used_duplicate_output1 is False
    finally:
        backend.release()


# --------------------------------------------------------------------------
# errors become the classes ScreenCapture acts on
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,hresult,cls",
    [
        ("reinit", DXGI_ERROR_ACCESS_LOST, RapidShotReinitError),
        ("device", DXGI_ERROR_DEVICE_REMOVED, RapidShotDeviceError),
        ("protected_content", E_ACCESSDENIED, RapidShotProtectedContentError),
        ("config", DXGI_ERROR_UNSUPPORTED, RapidShotConfigError),
        ("dxgi", -1, RapidShotDXGIError),
    ],
)
def test_each_error_kind_becomes_its_class_with_its_hresult(kind, hresult, cls):
    ext = native_duplicator._ext()
    error = native_duplicator._translate(ext.DuplicationError(kind, hresult, "message"))
    assert type(error) is cls
    assert error.hresult == hresult
    assert "message" in str(error)


def test_the_native_hresults_are_the_python_constants():
    """A kind is only useful if the HRESULT beside it compares equal to the
    constants ScreenCapture checks (DXGI_ERROR_UNSUPPORTED picks the next
    adapter); both sides write them as signed 32-bit values."""
    ext = native_duplicator._ext()
    e = ext.DuplicationError("config", DXGI_ERROR_UNSUPPORTED, "x")
    assert native_duplicator._translate(e).hresult == DXGI_ERROR_UNSUPPORTED < 0


def test_capture_from_a_non_input_desktop_names_the_desktop_not_drm(desktop_duplication):
    """The real refusal: a thread on a desktop that is not receiving input
    gets a bare E_ACCESSDENIED, which must read as a desktop problem."""
    script = (
        "import ctypes, json, sys\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "u = ctypes.WinDLL('user32', use_last_error=True)\n"
        "u.CreateDesktopW.restype = ctypes.c_void_p\n"
        "u.CreateDesktopW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p,"
        " ctypes.c_ulong, ctypes.c_ulong, ctypes.c_void_p]\n"
        "u.SetThreadDesktop.argtypes = [ctypes.c_void_p]\n"
        "d = u.CreateDesktopW('rapidshot_native_probe', None, None, 0, 0x10000000, None)\n"
        "if not d or not u.SetThreadDesktop(ctypes.c_void_p(d)):\n"
        "    print(json.dumps({'skip': 1})); raise SystemExit(0)\n"
        "import logging; logging.disable(logging.CRITICAL)\n"
        "import rapidshot\n"
        "try:\n"
        "    rapidshot.create(output_color='BGRA'); print(json.dumps({'created': True}))\n"
        "except BaseException as e:\n"
        "    print(json.dumps({'created': False, 'type': type(e).__name__, 'message': str(e)}))\n"
    )
    env = dict(os.environ, **{native_duplicator.NATIVE_DUPLICATOR_ENV: "native"})
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          timeout=180, env=env)
    line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("{")), None)
    assert line is not None, proc.stderr[-500:]
    result = json.loads(line)
    if "skip" in result:
        pytest.skip("could not create or attach a desktop")
    assert result["created"] is False
    assert "rapidshot_native_probe" in result["message"]
    assert "player window" not in result["message"].lower()
