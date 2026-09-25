"""Prebuilt native GPU-interop extension for RapidShot.

This distribution carries no Python logic of its own: the compiled module sits
alongside this file and `rapidshot.native` imports it. Installing it changes
nothing about how RapidShot is used -- `rapidshot.native.is_available()` simply
starts returning True.

    pip install rapidshot rapidshot-native

Built from `native/` in the RapidShot repository as an abi3 wheel, so one binary
covers every supported Python version rather than one per minor release.
"""

from __future__ import annotations

__all__ = ["__version__", "extension_path", "latency_source_path"]


def _version() -> str:
    """Read the version from installed metadata rather than repeating it.

    The number lives in `native/Cargo.toml`, which is what maturin stamps onto
    the wheel; writing it out here as well would create exactly the drift that
    `rapidshot/_version.py` exists to prevent -- two declarations that nothing
    checks against each other until a release is being cut.

    Falls back to "unknown" rather than raising: a source tree that was never
    pip-installed has no metadata, and failing to import over a version string
    would be worse than not knowing it.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return "unknown"
    try:
        return version("rapidshot-native")
    except PackageNotFoundError:  # pragma: no cover - running from a checkout
        return "unknown"


__version__ = _version()


def extension_path() -> str:
    """Absolute path to the compiled module.

    Useful when diagnosing a failed import: it distinguishes "the wheel is not
    installed" from "the wheel is installed and the DLL will not load", which
    otherwise produce the same ImportError out of `rapidshot.native`.
    """
    from pathlib import Path

    here = Path(__file__).resolve().parent
    for name in ("_rapidshot_native.pyd", "_rapidshot_native.so"):
        candidate = here / name
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"no compiled extension next to {here}")


def latency_source_path() -> str:
    """Absolute path to the benchmark's controlled test source, `latency_source.exe`.

    The pixel-age benchmark needs a source that stamps a frame ID into every
    `Present()` and logs when it happened, so that every capture library is timed
    on one clock. That source is a Rust binary built from the same crate
    (`native/src/bin/latency_source.rs`) and shipped in this wheel so that a
    tester with no toolchain can produce the same table the README quotes.

    It is a standalone executable, not something `rapidshot` imports: nothing in
    the capture path depends on it.
    """
    from pathlib import Path

    candidate = Path(__file__).resolve().parent / "latency_source.exe"
    if candidate.is_file():
        return str(candidate)
    raise FileNotFoundError(f"latency_source.exe is not in {candidate.parent}; "
                            "rapidshot-native 0.2.1 or later ships it")
