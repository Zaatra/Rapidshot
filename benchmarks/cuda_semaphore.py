"""Moved to ``rapidshot._bench.cuda_semaphore``, so ``rapidshot benchmark`` can run it from a wheel.

This file keeps ``python benchmarks/cuda_semaphore.py ...`` and ``import cuda_semaphore`` (with
``benchmarks/`` on the path) working: as a script it runs the moved module's own
``__main__`` block, and as an import it *is* the moved module, so monkeypatching
an attribute here patches what the harness actually uses.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    runpy.run_module("rapidshot._bench.cuda_semaphore", run_name="__main__", alter_sys=True)
else:
    from rapidshot._bench import cuda_semaphore as _module
    sys.modules[__name__] = _module
