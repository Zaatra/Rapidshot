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
"""
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
