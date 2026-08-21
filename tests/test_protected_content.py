"""Protected-content detection, with a real trigger instead of a fake HRESULT.

ROADMAP listed the HDCP path as fault-injection tested only, on the reasoning
that the trigger is OPM/HDCP-protected playback and that needs licensed DRM
content. That turns out to be too pessimistic: `SetWindowDisplayAffinity` with
`WDA_MONITOR` makes the compositor blank a window out of captured frames, and
**DXGI reports it through the same `ProtectedContentMaskedOut` flag** the HDCP
path reads. Verified 2026-08-06 on Machine B.

Scope, precisely. This covers the *masked-out* branch -- duplication succeeds
and the protected region arrives blanked. It does **not** cover the refusal
branch, where `DuplicateOutput` itself is denied and
`RapidShotProtectedContentError` is raised at construction; that one still has
no local trigger and remains fault-injection tested.

Needs a desktop session and screen activity, so it skips in CI.
"""
import ctypes
import time

import numpy as np
import pytest

import rapidshot

tk = pytest.importorskip("tkinter", reason="needs tkinter to make a window")

WDA_NONE = 0x00000000
WDA_MONITOR = 0x00000001

WINDOW = {"x": 300, "y": 200, "w": 600, "h": 440}

user32 = ctypes.windll.user32
user32.SetWindowDisplayAffinity.argtypes = [ctypes.c_void_p, ctypes.c_uint]
user32.SetWindowDisplayAffinity.restype = ctypes.c_int
user32.GetParent.argtypes = [ctypes.c_void_p]
user32.GetParent.restype = ctypes.c_void_p


@pytest.fixture
def excluded_window(tk_root):
    """A visible window the OS blanks out of captured frames.

    Yields a callable that flips the affinity, so a test can compare the same
    window captured both ways rather than trusting one reading. Built from the
    session-wide root in conftest, since a second ``Tk()`` in one process
    raises and turns into a misleading "no desktop session" skip.
    """
    root = tk.Toplevel(tk_root)
    root.overrideredirect(True)
    root.geometry(f"{WINDOW['w']}x{WINDOW['h']}+{WINDOW['x']}+{WINDOW['y']}")
    root.attributes("-topmost", True)
    canvas = tk.Canvas(root, width=WINDOW["w"], height=WINDOW["h"],
                       bg="#f0c020", highlightthickness=0)
    canvas.pack()
    root.update()

    hwnd = user32.GetParent(root.winfo_id()) or root.winfo_id()

    def set_affinity(value):
        ok = user32.SetWindowDisplayAffinity(ctypes.c_void_p(hwnd), value)
        if not ok:
            pytest.skip("SetWindowDisplayAffinity unsupported here")
        # Repaint so Desktop Duplication has changed content to report; an
        # idle screen produces no frames at all (ROADMAP section 2).
        for i in range(12):
            canvas.create_rectangle(10 + i * 4, 10, WINDOW["w"] - 10,
                                    WINDOW["h"] - 10, fill="#ff2020",
                                    outline="")
            root.update()
            time.sleep(0.02)

    try:
        yield set_affinity
    finally:
        user32.SetWindowDisplayAffinity(ctypes.c_void_p(hwnd), WDA_NONE)
        try:
            root.destroy()
        except tk.TclError:
            pass


def sample(frames=5):
    """Capture, returning (any protected flag seen, mean of the window area)."""
    camera = rapidshot.create(output_color="BGRA")
    flags, means = [], []
    try:
        for _ in range(500):
            frame = camera.grab_frame()
            if frame is None:
                continue
            flags.append(bool(frame.protected_content))
            frame.release()
            if len(flags) >= frames:
                break
        for _ in range(300):
            buf = camera.grab()
            if buf is None:
                continue
            arr = np.asarray(buf)
            top, left = WINDOW["y"], WINDOW["x"]
            bottom = min(top + WINDOW["h"], arr.shape[0])
            right = min(left + WINDOW["w"], arr.shape[1])
            means.append(float(arr[top:bottom, left:right, :3].mean()))
            release = getattr(buf, "release", None)
            if release:
                release()
            if len(means) >= 3:
                break
    finally:
        camera.release()
        rapidshot.reset()

    if not flags or not means:
        pytest.skip("no frames captured — the screen must be changing")
    return any(flags), sum(means) / len(means)


def test_display_affinity_sets_the_protected_flag(excluded_window):
    """`Frame.protected_content` must follow the real OS state, both ways.

    Checked against a baseline in the same run rather than asserting True in
    isolation: a flag that is always set would pass that weaker test.
    """
    excluded_window(WDA_NONE)
    baseline_flag, baseline_mean = sample()

    excluded_window(WDA_MONITOR)
    excluded_flag, excluded_mean = sample()

    assert not baseline_flag, (
        "protected_content was set with no protected content on screen")
    assert excluded_flag, (
        "protected_content was not set while the OS was blanking a window "
        "out of captured frames")
    assert baseline_mean > 1.0, "the window was not visible in the baseline"
    assert excluded_mean < 1.0, (
        f"the excluded window still captured with mean {excluded_mean:.1f}; "
        f"the OS should have blanked it")


def test_protected_capture_still_returns_usable_frames(excluded_window):
    """Masked-out is not a failure: capture continues, the region is blank.

    Worth pinning because the sibling branch -- `DuplicateOutput` refusing
    outright -- *is* an error, and conflating the two would either turn a
    normal blanked region into an exception or swallow a real refusal.
    """
    excluded_window(WDA_MONITOR)
    camera = rapidshot.create(output_color="RGB")
    try:
        got = None
        for _ in range(500):
            got = camera.grab()
            if got is not None:
                break
        if got is None:
            pytest.skip("no frames captured — the screen must be changing")
        arr = np.asarray(got)
        assert arr.ndim == 3 and arr.shape[2] == 3
        assert arr.dtype == np.uint8
        release = getattr(got, "release", None)
        if release:
            release()
    finally:
        camera.release()
        rapidshot.reset()
