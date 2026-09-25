"""Moved to ``rapidshot._bench.section7_adapters``, so ``rapidshot benchmark`` can run it from a wheel.

This file keeps ``python benchmarks/section7_adapters.py ...`` and ``import section7_adapters`` (with
``benchmarks/`` on the path) working: as a script it runs the moved module's own
``__main__`` block, and as an import it *is* the moved module, so monkeypatching
an attribute here patches what the harness actually uses.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("rapidshot._bench.section7_adapters", run_name="__main__", alter_sys=True)
else:
    from rapidshot._bench import section7_adapters as _module
    sys.modules[__name__] = _module
