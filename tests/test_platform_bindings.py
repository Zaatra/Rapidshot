"""ctypes declarations, the off-pool staging buffer, and library output.

Six small things, one theme: code that worked by accident.

The timer's `WaitForSingleObject` had no `restype`, so ctypes read its DWORD
return as a signed 32-bit int. `WAIT_FAILED` is 0xFFFFFFFF, which comes back as
-1, so the capture thread's `res == WAIT_FAILED` check could never be true and a
failing timer wait was indistinguishable from a normal tick.
`CreateWaitableTimerExW` returns a pointer-sized HANDLE, which the same default
would truncate.

`IDXGIOutputDuplication::GetDesc` was declared with no parameter, so calling it
would have let DXGI write a 36-byte struct through whatever the argument
register happened to hold. Nothing calls it, which is the only reason that never
fired.

And `grab(region=...)` for a shape the pool does not cover allocated a fresh
staging buffer per frame, paying the first-touch page faults every time.
"""

import ctypes
import logging

import numpy as np
import pytest

pytest.importorskip("comtypes")

from rapidshot.util import timer as timer_module  # noqa: E402


# --------------------------------------------------------------------------
# ctypes prototypes
# --------------------------------------------------------------------------

TIMER_FUNCTIONS = [
    "CreateWaitableTimerExW",
    "SetWaitableTimer",
    "WaitForSingleObject",
    "CancelWaitableTimer",
    "CloseHandle",
]


def _timer_kernel32():
    """The module's private kernel32 handle.

    Stored under its literal name: a leading double underscore is only mangled
    inside a class body, not at module level.
    """
    return vars(timer_module)["__kernel32"]


@pytest.mark.parametrize("name", TIMER_FUNCTIONS)
def test_timer_functions_are_declared(name):
    function = getattr(_timer_kernel32(), name)

    assert function.argtypes is not None, f"{name} has no argtypes"
    assert function.restype is not None, f"{name} has no restype"


def test_wait_failed_is_comparable():
    """The check the capture thread makes. As a signed int it read as -1."""
    invalid_handle = 0xDEAD
    result = timer_module.wait_for_timer(invalid_handle, 0)

    assert result == timer_module.WAIT_FAILED, (
        f"a failed wait returned {result!r}, which the capture thread's "
        "`res == WAIT_FAILED` check would not recognise")


def test_the_timer_does_not_mutate_the_shared_kernel32():
    """Prototypes go on a private handle, not the process-wide windll cache.

    `ctypes.windll` hands out one cached object per DLL for the whole process,
    so declaring a function there changes it for every other library that
    reaches for the same name.
    """
    assert _timer_kernel32() is not ctypes.windll.kernel32


def test_a_real_timer_round_trips():
    handle = timer_module.create_high_resolution_timer()
    assert handle, "no handle returned"
    try:
        timer_module.set_periodic_timer(handle, 5)
        assert timer_module.wait_for_timer(handle, 1000) == 0  # WAIT_OBJECT_0
        timer_module.cancel_timer(handle)
    finally:
        timer_module.close_timer(handle)


def test_outdupl_desc_matches_the_windows_layout():
    from rapidshot._libs.dxgi import DXGI_MODE_DESC, DXGI_OUTDUPL_DESC

    assert ctypes.sizeof(DXGI_MODE_DESC) == 28
    assert ctypes.sizeof(DXGI_OUTDUPL_DESC) == 36


def test_duplication_get_desc_declares_its_parameter():
    """Declared bare, it was callable and would have written through nothing."""
    from rapidshot._libs.dxgi import DXGI_OUTDUPL_DESC, IDXGIOutputDuplication

    # A comtypes STDMETHOD entry is (restype, name, argtypes, ...).
    get_desc = next(m for m in IDXGIOutputDuplication._methods_
                    if m[1] == "GetDesc")

    assert get_desc[2] == [ctypes.POINTER(DXGI_OUTDUPL_DESC)], (
        f"GetDesc declares {get_desc[2]}; DXGI writes a DXGI_OUTDUPL_DESC "
        "through that argument")


def test_the_dead_monitor_name_helper_is_gone():
    from rapidshot.util import io as io_module

    assert not hasattr(io_module, "get_monitor_name_by_handle")


# --------------------------------------------------------------------------
# Library output goes to the logger, not stdout
# --------------------------------------------------------------------------

def test_a_dependency_check_writes_nothing_to_stdout(capsys):
    from rapidshot.processor import base as base_module

    processor = base_module.Processor.__new__(base_module.Processor)
    processor._check_dependencies()

    assert capsys.readouterr().out == "", "a library wrote to stdout"


def test_pool_messages_go_to_the_logger(caplog, capsys):
    from rapidshot.memory_pool import NumpyMemoryPool

    pool = NumpyMemoryPool((4, 4, 4), np.uint8, 2)
    with caplog.at_level(logging.DEBUG, logger="rapidshot.memory_pool"):
        pool.initialize_pool()      # already initialised: used to print

    assert capsys.readouterr().out == "", "the pool wrote to stdout"


def test_no_module_in_the_package_calls_print():
    """The whole point: a library that prints cannot be silenced."""
    import ast
    import pathlib

    # Parsed, not grepped: `profiling.py` documents its own usage with a
    # `print(profiler.report())` line inside a docstring, which is an example
    # for the reader rather than output from the library.
    #
    # Programs are not library code: `python -m rapidshot` and the benchmark
    # harness in `_bench/` exist to write to a terminal, and nothing a user
    # imports reaches them -- which the test below holds, so this exemption
    # cannot quietly widen.
    root = pathlib.Path(__file__).resolve().parent.parent / "rapidshot"
    offenders = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] in PROGRAMS:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"):
                offenders.append(f"{path.relative_to(root)}:{node.lineno}")

    assert offenders == [], f"print() in library code: {offenders}"


#: Parts of the package that are command-line programs rather than library code.
PROGRAMS = ("__main__.py", "_bench")


def test_no_library_module_imports_the_programs():
    """What keeps the print() exemption above honest.

    `__main__.py` may reach into `_bench` (that is `rapidshot benchmark`), but
    only inside a function; a module-level import anywhere would put printing
    code on the path of `import rapidshot`.
    """
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent / "rapidshot"
    offenders = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        if relative.parts[0] == "_bench":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            if not any("_bench" in name or name.endswith("__main__") for name in names):
                continue
            if relative.parts[0] == "__main__.py" and node.col_offset > 0:
                continue          # inside a function: only runs as a command
            offenders.append(f"{relative}:{node.lineno}")
    assert offenders == [], f"library code imports a program: {offenders}"


# --------------------------------------------------------------------------
# The off-pool staging buffer
# --------------------------------------------------------------------------

class _Processor:
    def __init__(self, converts):
        self.converts_output = converts


def _camera(converts_output):
    from rapidshot.capture import ScreenCapture

    camera = ScreenCapture.__new__(ScreenCapture)
    camera.nvidia_gpu = False
    camera._scratch_staging = None
    camera._processor = _Processor(converts_output)
    return camera


def test_a_converting_grab_reuses_the_scratch_buffer():
    camera = _camera(converts_output=True)

    first = camera._scratch_staging_buffer(700, 800)
    second = camera._scratch_staging_buffer(700, 800)

    assert first is second, "a fresh buffer per frame pays the page faults again"
    assert first.shape == (700, 800, 4)


def test_a_changed_region_shape_gets_a_new_buffer():
    camera = _camera(converts_output=True)

    first = camera._scratch_staging_buffer(700, 800)
    second = camera._scratch_staging_buffer(400, 400)

    assert second is not first
    assert second.shape == (400, 400, 4)
    assert camera._scratch_staging is second


def test_bgra_never_reuses_the_buffer():
    """With no conversion the staging buffer *is* the frame handed back, so
    reusing it would give two callers the same memory."""
    camera = _camera(converts_output=False)

    first = camera._scratch_staging_buffer(700, 800)
    second = camera._scratch_staging_buffer(700, 800)

    assert first is not second, (
        "BGRA returns this buffer to the caller; reusing it aliases frames")
    assert camera._scratch_staging is None, "nothing should be cached for BGRA"


def test_the_scratch_buffer_is_the_right_dtype():
    camera = _camera(converts_output=True)

    assert camera._scratch_staging_buffer(8, 8).dtype == np.uint8
