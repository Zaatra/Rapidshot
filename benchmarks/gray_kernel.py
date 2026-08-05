"""How much is left in GRAY, and would SIMD get it? (ROADMAP.md sections 3, 10)

GRAY is the slowest colour mode by a wide margin -- 13.7-14.9 ms per 1080p
frame, which means it cannot keep up with a 60 Hz display. ROADMAP.md section 10
nominates a SIMD kernel as the clearest remaining CPU win. Before writing one in
Rust -- which only helps users who build the optional extension -- this measures
two things:

  * how much of the cost is reachable from pure NumPy, by trying four
    formulations of the same arithmetic;
  * what a real hand-written SIMD kernel actually achieves, using OpenCV's
    BGRA->GRAY as a stand-in ceiling.

The second number is the one that decides whether the Rust work is worth it. If
OpenCV is only modestly faster than the best NumPy formulation, there is no
prize; if it is several times faster, the prize is real and measurable.

Every NumPy candidate is checked byte-exact against the shipped
`bgra_to_gray`. OpenCV uses different rounding, so it is compared for speed
only and its worst-case pixel deviation is reported separately.

Reps are paced to a frame period, matching perf_suite.py -- see the duty-cycle
note in ROADMAP.md section 2. GRAY fills ~95% of a 60 Hz frame, so it is
measured under effectively sustained load, which is the honest condition.

    python benchmarks/gray_kernel.py
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from perf_suite import FRAME_H, FRAME_W, make_frame, time_it  # noqa: E402

from rapidshot.processor.numpy_processor import (  # noqa: E402
    _LUMA_B,
    _LUMA_G,
    _LUMA_R,
    _LUMA_ROUND,
    _LUMA_SHIFT,
    bgra_to_gray,
)

try:
    import cv2
except ImportError:
    cv2 = None


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------

def gray_2_0_0(src):
    """The formulation 2.0.0 shipped, kept inline as the fixed reference point.

    Deliberately duplicated rather than imported. This row is what every speedup
    below is quoted against, so it has to keep measuring the same thing even as
    the library's own GRAY path changes -- pointing it at `bgra_to_gray` would
    silently re-baseline the whole table the next time that function improves,
    which is exactly what happened once already.

    Every candidate is checked byte-exact against this.
    """
    luma = src[..., 2].astype(np.uint16)
    luma *= _LUMA_R
    luma += _LUMA_ROUND
    luma += src[..., 1].astype(np.uint16) * _LUMA_G
    luma += src[..., 0].astype(np.uint16) * _LUMA_B
    luma >>= _LUMA_SHIFT
    return luma.astype(np.uint8)[..., np.newaxis]


def gray_current(src):
    """The library's present convenience form, which allocates its own buffers."""
    return bgra_to_gray(src)


class Scratch:
    """Persistent intermediates.

    The shipped path allocates two full-frame uint16 temporaries per call --
    `src[..., 1].astype(uint16) * 150` materialises one, and channel 0 another.
    The processor already keeps a persistent accumulator for dirty rects, so
    holding scratch here is consistent with how the class already works.
    """

    def __init__(self, h=FRAME_H, w=FRAME_W):
        self.acc = np.empty((h, w), np.uint16)
        self.tmp = np.empty((h, w), np.uint16)
        self.out = np.empty((h, w, 1), np.uint8)
        # uint32 scratch for the SWAR candidate
        self.u32a = np.empty((h, w), np.uint32)
        self.u32b = np.empty((h, w), np.uint32)


def gray_scratch(src, s):
    """Same arithmetic, zero allocation: every op writes into reused buffers."""
    acc, tmp, out = s.acc, s.tmp, s.out
    np.multiply(src[..., 2], _LUMA_R, out=acc, dtype=np.uint16, casting="unsafe")
    acc += _LUMA_ROUND
    np.multiply(src[..., 1], _LUMA_G, out=tmp, dtype=np.uint16, casting="unsafe")
    acc += tmp
    np.multiply(src[..., 0], _LUMA_B, out=tmp, dtype=np.uint16, casting="unsafe")
    acc += tmp
    acc >>= _LUMA_SHIFT
    np.copyto(out[..., 0], acc, casting="unsafe")
    return out


def gray_swar(src, s):
    """Read the frame as uint32 so every access is a full cache line.

    Extracting channels from a BGRA frame with `src[..., 2]` walks memory with
    stride 4, touching one useful byte per four. Viewing the frame as uint32
    reads contiguously and masks the channels out arithmetically. The trade is
    that all intermediates are uint32, doubling the bytes touched per pixel
    versus the uint16 formulation.
    """
    u32 = src.view(np.uint32)[..., 0]
    a, b, out = s.u32a, s.u32b, s.out

    np.right_shift(u32, 16, out=a)          # R in low byte (plus alpha above)
    a &= 0xFF
    a *= _LUMA_R
    a += _LUMA_ROUND

    np.right_shift(u32, 8, out=b)           # G
    b &= 0xFF
    b *= _LUMA_G
    a += b

    np.bitwise_and(u32, 0xFF, out=b)        # B
    b *= _LUMA_B
    a += b

    a >>= _LUMA_SHIFT
    np.copyto(out[..., 0], a, casting="unsafe")
    return out


_LUT_R = np.arange(256, dtype=np.uint16) * _LUMA_R + _LUMA_ROUND
_LUT_G = np.arange(256, dtype=np.uint16) * _LUMA_G
_LUT_B = np.arange(256, dtype=np.uint16) * _LUMA_B


def gray_lut(src, s):
    """Replace the multiplies with three 256-entry table lookups.

    Trades three multiplies for three gathers. A gather from a table that fits
    in L1 is cheap per element, but it is still a gather, and NumPy's fancy
    indexing allocates.
    """
    acc, tmp, out = s.acc, s.tmp, s.out
    np.take(_LUT_R, src[..., 2], out=acc)
    np.take(_LUT_G, src[..., 1], out=tmp)
    acc += tmp
    np.take(_LUT_B, src[..., 0], out=tmp)
    acc += tmp
    acc >>= _LUMA_SHIFT
    np.copyto(out[..., 0], acc, casting="unsafe")
    return out


def gray_cv2(src, dst):
    """OpenCV's hand-written SIMD kernel: the ceiling this is all measured against."""
    cv2.cvtColor(src, cv2.COLOR_BGRA2GRAY, dst=dst)
    return dst


# ---------------------------------------------------------------------------
# component breakdown
# ---------------------------------------------------------------------------

def breakdown(src, s, reps):
    """Where the time goes, so the candidates can be reasoned about."""
    acc, tmp = s.acc, s.tmp
    u32 = src.view(np.uint32)[..., 0]
    parts = [
        ("memcpy 8.3MB (bandwidth control)", lambda: src.copy()),
        ("strided gather ch2 -> uint16", lambda: np.multiply(
            src[..., 2], 1, out=acc, dtype=np.uint16, casting="unsafe")),
        ("contiguous uint32 read -> uint32", lambda: np.right_shift(u32, 0, out=s.u32a)),
        ("one uint16 multiply in place", lambda: acc.__imul__(_LUMA_R)),
        ("one uint16 add in place", lambda: acc.__iadd__(tmp)),
        ("uint16 -> uint8 narrow + write", lambda: np.copyto(
            s.out[..., 0], acc, casting="unsafe")),
    ]
    print("\ncomponent breakdown (min of paced reps)")
    print("-" * 58)
    for name, fn in parts:
        ms = min(time_it(fn, reps)) * 1000.0
        print(f"  {name:<40} {ms:7.3f} ms")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(reps=40):
    src = make_frame()
    s = Scratch()
    reference = gray_2_0_0(src)

    cands = [
        ("2.0.0 formulation", lambda: gray_2_0_0(src)),
        ("bgra_to_gray (allocs)", lambda: gray_current(src)),
        ("scratch (no alloc)", lambda: gray_scratch(src, s)),
        ("swar uint32", lambda: gray_swar(src, s)),
        ("lut (np.take)", lambda: gray_lut(src, s)),
    ]

    # The native kernel is the headline result, so it belongs in the same table
    # under the same pacing rather than in a separate script.
    from rapidshot import native
    native_dst = None
    if native.is_available():
        native_dst = np.empty((FRAME_H, FRAME_W), np.uint8)

        def _native():
            native.bgra_to_gray_into(src, native_dst)
            return native_dst[..., np.newaxis]

        cands.append(("native rust kernel", _native))

    # Correctness first: a fast wrong answer is worthless (section 11).
    print("byte-exactness vs the 2.0.0 formulation")
    print("-" * 58)
    exact = {}
    for name, fn in cands:
        got = fn()
        ok = got.shape == reference.shape and np.array_equal(got, reference)
        exact[name] = ok
        if not ok:
            diff = np.abs(got.astype(np.int32).reshape(reference.shape)
                          - reference.astype(np.int32))
            print(f"  {name:<24} MISMATCH  max |delta| = {diff.max()}")
        else:
            print(f"  {name:<24} exact")

    cv_dst = None
    if cv2 is not None:
        cv_dst = np.empty((FRAME_H, FRAME_W), np.uint8)
        cvg = gray_cv2(src, cv_dst)
        d = np.abs(cvg.astype(np.int32) - reference[..., 0].astype(np.int32))
        print(f"  {'cv2 (different rounding)':<24} max |delta| = {d.max()}, "
              f"mean {d.mean():.4f}")
        cands.append(("cv2 SIMD (ceiling)", lambda: gray_cv2(src, cv_dst)))

    print(f"\nthroughput, 1920x1080 BGRA, {reps} paced reps")
    print("-" * 58)
    base = None
    rows = []
    for name, fn in cands:
        samples = time_it(fn, reps, name=f"gray.{name}")
        ms = min(samples) * 1000.0
        med = sorted(samples)[len(samples) // 2] * 1000.0
        if base is None:
            base = ms
        rows.append((name, ms, med, base / ms, exact.get(name)))

    print(f"  {'candidate':<24} {'min':>8} {'median':>8} {'vs today':>9}  exact")
    for name, ms, med, sp, ok in rows:
        mark = "yes" if ok else ("--" if ok is None else "NO")
        print(f"  {name:<24} {ms:7.3f}  {med:7.3f}  {sp:8.2f}x  {mark}")

    print(f"\n  60 Hz frame budget is {1000.0/60.0:.2f} ms")
    for name, ms, _med, _sp, _ok in rows:
        verdict = "fits" if ms < 1000.0 / 60.0 else "OVER BUDGET"
        print(f"  {name:<24} {ms:7.3f} ms  {verdict}")

    breakdown(src, s, reps)


if __name__ == "__main__":
    sys.exit(main())
