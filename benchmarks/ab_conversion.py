"""Interleaved A/B of the old vs new pixel conversions.

Comparing two separate benchmark runs is vulnerable to machine drift between
them. Holding both implementations in one process and alternating A/B/A/B means
any drift hits both arms equally, so the ratio stays honest even on a loaded
machine. This is the measurement of record for the conversion speedup.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rapidshot.processor.numpy_processor import NumpyProcessor  # noqa: E402

H, W = 1080, 1920
ROUNDS = 40
rng = np.random.default_rng(20260727)
SRC = rng.integers(0, 256, (H, W, 4), dtype=np.uint8)


# --- the implementations that shipped before the change -------------------

def old_rgb(src, dst):
    dst[:] = src[..., 2::-1]


def old_bgr(src, dst):
    dst[:] = src[..., :3]


def old_rgba(src, dst):
    dst[:] = src[..., [2, 1, 0, 3]]


def old_gray(src, dst):
    # Q14 fixed point with uint32 intermediates
    b = src[..., 0].astype(np.uint32)
    g = src[..., 1].astype(np.uint32)
    r = src[..., 2].astype(np.uint32)
    y = r * 4899
    y += g * 9617
    y += b * 1868
    y += 8192
    y >>= 14
    dst[:] = y.astype(np.uint8)[..., np.newaxis]


# --- current implementations ----------------------------------------------

def make_new(mode):
    proc = NumpyProcessor(mode)
    return lambda src, dst: proc.convert_into(src, dst)


CASES = [
    ("RGB", 3, old_rgb),
    ("BGR", 3, old_bgr),
    ("RGBA", 4, old_rgba),
    ("GRAY", 1, old_gray),
]


def main() -> int:
    print(f"\nInterleaved A/B, {W}x{H} ({SRC.nbytes / 1e6:.1f} MB), "
          f"{ROUNDS} alternating rounds\n")
    print(f"{'mode':<8}{'old (min)':>12}{'new (min)':>12}{'speedup':>10}"
          f"{'old GB/s':>10}{'new GB/s':>10}")
    print("-" * 62)

    all_ok = True
    for mode, channels, old_fn in CASES:
        new_fn = make_new(mode)
        dst_old = np.empty((H, W, channels), np.uint8)
        dst_new = np.empty((H, W, channels), np.uint8)

        # Correctness first: the change must be output-preserving.
        old_fn(SRC, dst_old)
        new_fn(SRC, dst_new)
        max_dev = int(np.abs(dst_old.astype(np.int16)
                             - dst_new.astype(np.int16)).max())
        # GRAY moved from Q14 to Q8 fixed point, so one level of deviation is
        # expected and documented. Everything else must be bit-identical.
        limit = 1 if mode == "GRAY" else 0
        ok = max_dev <= limit
        all_ok &= ok

        t_old, t_new = [], []
        for _ in range(ROUNDS):
            t0 = time.perf_counter()
            old_fn(SRC, dst_old)
            t_old.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            new_fn(SRC, dst_new)
            t_new.append(time.perf_counter() - t0)

        o = min(t_old) * 1000
        n = min(t_new) * 1000
        mb = SRC.nbytes / 1e9
        flag = "" if ok else "  <-- OUTPUT DIFFERS"
        print(f"{mode:<8}{o:>11.2f}m{n:>11.2f}m{o / n:>9.2f}x"
              f"{mb / (o / 1000):>10.2f}{mb / (n / 1000):>10.2f}{flag}")
        if mode == "GRAY":
            print(f"{'':8}(GRAY max deviation {max_dev} level, Q14 -> Q8, expected)")

    print("-" * 62)
    print("\ncorrectness:", "all outputs match" if all_ok else "MISMATCH — investigate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
