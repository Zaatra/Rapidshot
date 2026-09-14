"""Shared fixtures.

The only thing here is a single Tk root for the whole session. Two test files
need real on-screen windows — `test_protected_content.py` and
`test_exclusive_fullscreen.py` — and Tkinter does not tolerate a second
``Tk()`` after the first has been destroyed: the later call raises ``TclError``
and the test skips itself with "no desktop session", which is both wrong and
quiet. Run either file alone and it passes; run the suite and the second one
vanishes.

A skip that only appears in a full run is worse than a failure, so the root is
created once, kept withdrawn, and handed out for ``Toplevel`` windows.

``motion`` runs ``benchmarks/motion_source.py`` for tests that need a stream of
*changed* frames. Module-scoped on purpose: its window is topmost over part of
the primary display, and leaving it up for the whole session would change what
every later live test captures.
"""
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

try:
    import tkinter as tk
except ImportError:  # pragma: no cover - tkinter is optional
    tk = None


@pytest.fixture(scope="session")
def tk_root():
    """One hidden Tk root for the session; tests make Toplevels from it."""
    if tk is None:
        pytest.skip("tkinter not available")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no desktop session")
    root.withdraw()
    try:
        yield root
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


MOTION_SOURCE = Path(__file__).resolve().parent.parent / "benchmarks" / "motion_source.py"
#: A rectangle inside the motion window (900x700 at +200+120 on the primary
#: display), as (left, top, right, bottom).
MOTION_INSIDE = (240, 160, 1080, 780)


@pytest.fixture(scope="module")
def motion():
    """An animating window, or a skip when there is no desktop to draw on."""
    proc = subprocess.Popen(
        [sys.executable, str(MOTION_SOURCE), "--parent-controlled"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    deadline = time.monotonic() + 15
    ready = False
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        try:
            event = json.loads(line).get("event")
        except ValueError:
            continue
        if event in ("ready", "error"):
            ready = event == "ready"
            break
    if not ready:
        proc.kill()
        pytest.skip("motion source did not start (no desktop session?)")
    # It keeps reporting its rate; an undrained pipe fills and blocks it, and a
    # blocked source can neither animate nor notice the stop signal.
    threading.Thread(target=proc.stdout.read, daemon=True).start()
    try:
        yield MOTION_INSIDE
    finally:
        try:
            proc.stdin.close()          # closing the pipe is its stop signal
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
