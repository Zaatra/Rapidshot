"""Moved to ``rapidshot._bench.memory_profile``, so ``rapidshot benchmark`` can run it from a wheel.

This file keeps ``python benchmarks/memory_profile.py ...`` and ``import memory_profile`` (with
``benchmarks/`` on the path) working: as a script it runs the moved module's own
``__main__`` block, and as an import it *is* the moved module, so monkeypatching
an attribute here patches what the harness actually uses.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("rapidshot._bench.memory_profile", run_name="__main__", alter_sys=True)
else:
    from rapidshot._bench import memory_profile as _module
    sys.modules[__name__] = _module
