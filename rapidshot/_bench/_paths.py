"""Where the benchmark harness is running from, and where it may write.

The harness moved here from ``benchmarks/`` so that ``rapidshot benchmark`` can
reach it from a plain ``pip install``, and every module in it used to assume a
repository checkout: ``ROOT / "build"`` for results, ``ROOT / "native" /
"target"`` for the test source, and a script path to re-run itself as a worker.
In a wheel none of those exist. This module is the one place that decides.

``REPO`` is the checkout root when there is one and ``None`` in an installed
wheel. ``WORK`` is where results go: ``build/`` in a checkout, as before, and a
per-user directory otherwise -- never the current directory, which is wherever
the tester happened to open a terminal.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

#: ``rapidshot/_bench/_paths.py`` -> the directory containing ``rapidshot``.
PACKAGE_PARENT = Path(__file__).resolve().parents[2]


def _checkout(root: Path):
    """``root`` if it is a RapidShot checkout rather than site-packages."""
    if (root / "native" / "Cargo.toml").is_file() and (root / "benchmarks").is_dir():
        return root
    return None


REPO = _checkout(PACKAGE_PARENT)

if REPO is not None:
    WORK = REPO / "build"
else:
    WORK = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "rapidshot" / "benchmark"


def worker_command(module: str, *args: str) -> list:
    """Run ``module`` (a ``rapidshot._bench`` module name) as a worker process.

    ``-m`` rather than a file path: a module inside a package cannot be run as a
    script without losing its relative imports.
    """
    return [sys.executable, "-u", "-m", module, *args]


def worker_env(base=None) -> dict:
    """The environment a worker needs to import the same ``rapidshot`` as its parent.

    From a checkout that is not pip-installed, ``python -m rapidshot._bench...``
    only resolves if the checkout is on ``PYTHONPATH``; otherwise a worker could
    import an installed copy and measure different code from the parent's.
    """
    env = dict(os.environ if base is None else base)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PACKAGE_PARENT) + (os.pathsep + existing if existing else "")
    return env
