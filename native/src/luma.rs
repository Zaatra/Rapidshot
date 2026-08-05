//! BGRA -> 8-bit luma, byte-exact with the NumPy path in
//! `rapidshot/processor/numpy_processor.py`.
//!
//! GRAY is the slowest colour mode by a wide margin. `benchmarks/gray_kernel.py`
//! measured the shipped NumPy formulation at 15.5 ms per 1080p frame and a
//! reused-scratch rewrite of the same arithmetic at 8.5 ms, against a
//! single-threaded hand-written SIMD reference (OpenCV) at **0.34 ms** -- which
//! is memory-bandwidth-bound, not compute-bound, at 24 GB/s. So roughly 25x
//! remains on the table beyond what NumPy can express.
//!
//! # Why this is byte-exact rather than "close enough"
//!
//! Rec. 601 luma in Q8 fixed point:
//!
//! ```text
//! Y = (R*77 + G*150 + B*29 + 128) >> 8
//! ```
//!
//! The whole intermediate fits in u16 -- the maximum is 255*(77+150+29) + 128 =
//! 65408, just inside 65535 -- which is what makes a 16-bit-lane SIMD
//! formulation possible at all, and it is why the NumPy path chose Q8. The
//! `+128` is round-to-nearest; without it every pixel biases dark.
//!
//! Matching the NumPy result exactly matters more than it looks. OpenCV's kernel
//! deviates by up to 1 LSB (mean 0.13) because it rounds differently, and
//! ROADMAP.md section 11 records what that class of difference costs: a fast
//! wrong answer is worthless, and a conversion that is "nearly" right is exactly
//! the kind of defect no test of speed, shape or stability catches. Producing
//! the identical answer means the accelerated path is verifiable against the
//! NumPy reference pixel for pixel, and can be swapped in without changing what
//! any consumer sees.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Rec. 601 luma coefficients in Q8, matching `numpy_processor._LUMA_*`.
const LUMA_R: u32 = 77;
const LUMA_G: u32 = 150;
const LUMA_B: u32 = 29;
const LUMA_ROUND: u32 = 128;
const LUMA_SHIFT: u32 = 8;

/// One pixel, in the exact arithmetic the NumPy path performs.
#[inline(always)]
fn luma_px(b: u8, g: u8, r: u8) -> u8 {
    let acc = (r as u32) * LUMA_R + (g as u32) * LUMA_G + (b as u32) * LUMA_B + LUMA_ROUND;
    (acc >> LUMA_SHIFT) as u8
}

/// One row of BGRA to one row of luma.
///
/// Written as a `chunks_exact(4)` zip rather than an index loop so that the
/// bounds checks fold away and LLVM can see a fixed-stride gather it is able to
/// vectorise. The u32 accumulator is deliberate: it cannot overflow, so no
/// saturating behaviour is needed, and after inlining LLVM narrows the lanes
/// itself.
#[inline]
fn luma_row(src: &[u8], dst: &mut [u8]) {
    for (px, out) in src.chunks_exact(4).zip(dst.iter_mut()) {
        *out = luma_px(px[0], px[1], px[2]);
    }
}

// ---------------------------------------------------------------------------
// AVX2 path
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    /// 8 pixels per iteration, exactly 8 bytes out, byte-identical to `luma_px`.
    ///
    /// # Why not `maddubs`
    ///
    /// The instruction that looks made for a weighted byte sum is
    /// `_mm256_maddubs_epi16`, and it is **unusable here**. It accumulates into
    /// *signed* i16 with saturation, while `b*29 + g*150` reaches 45,645 --
    /// well past 32,767. It would clamp on bright pixels and corrupt them
    /// silently: no test of speed, shape or stability would notice, and the
    /// error only appears in the highlights. This is the trap ROADMAP.md section
    /// 10 warns about.
    ///
    /// Widening to u16 first and using `_mm256_madd_epi16` accumulates into i32
    /// instead, where the Q8 total cannot overflow at all. That costs one unpack
    /// per half and buys correctness that does not depend on the input.
    ///
    /// # Shape of the computation
    ///
    /// Per 128-bit lane the widened pixels are `[B,G,R,A, B,G,R,A]` as u16, so
    /// pairing adjacent lanes for `madd` yields `(B*29 + G*150)` and
    /// `(R*77 + A*0)` as i32. `phaddd` then sums those two per pixel -- and
    /// conveniently interleaves the low and high unpacks back into pixel order,
    /// giving all eight lumas in sequence with no cross-lane fixup.
    #[target_feature(enable = "avx2")]
    pub unsafe fn row_luma(src: &[u8], dst: &mut [u8]) {
        let zero = _mm256_setzero_si256();
        let (b, g, r) = (
            super::LUMA_B as i16,
            super::LUMA_G as i16,
            super::LUMA_R as i16,
        );
        let coef = _mm256_setr_epi16(b, g, r, 0, b, g, r, 0, b, g, r, 0, b, g, r, 0);
        let round = _mm256_set1_epi32(super::LUMA_ROUND as i32);

        let px = dst.len();
        let vec_px = px & !7usize;

        let mut i = 0;
        while i < vec_px {
            let v = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            // unpacklo takes pixels 0,1 of each lane -> 0,1 and 4,5;
            // unpackhi takes 2,3 of each lane -> 2,3 and 6,7.
            let lo = _mm256_madd_epi16(_mm256_unpacklo_epi8(v, zero), coef);
            let hi = _mm256_madd_epi16(_mm256_unpackhi_epi8(v, zero), coef);
            // Lane 0 becomes [l0,l1,l2,l3], lane 1 becomes [l4,l5,l6,l7].
            let h = _mm256_hadd_epi32(lo, hi);
            let y = _mm256_srli_epi32::<{ super::LUMA_SHIFT as i32 }>(_mm256_add_epi32(h, round));
            // Values are 0..255 here, so neither pack saturates.
            let packed = _mm256_packus_epi16(_mm256_packus_epi32(y, y), _mm256_packus_epi32(y, y));
            let out = dst.as_mut_ptr().add(i);
            // Exactly 8 bytes: the low dword of each 128-bit lane. Writing a
            // whole vector would overrun the row, and for a dirty-rect patch
            // that means clobbering live pixels of the next row.
            (out as *mut u32)
                .write_unaligned(_mm_cvtsi128_si32(_mm256_castsi256_si128(packed)) as u32);
            (out.add(4) as *mut u32)
                .write_unaligned(_mm_cvtsi128_si32(_mm256_extracti128_si256::<1>(packed)) as u32);
            i += 8;
        }
        // Tail: fewer than 8 pixels left.
        while i < px {
            let s = i * 4;
            *dst.get_unchecked_mut(i) = super::luma_px(
                *src.get_unchecked(s),
                *src.get_unchecked(s + 1),
                *src.get_unchecked(s + 2),
            );
            i += 1;
        }
    }
}

/// Pick the row implementation once per image.
///
/// x86_64 guarantees only SSE2, so AVX2 is detected at runtime rather than
/// assumed; a wheel built with it unconditionally would fault on older
/// hardware. `is_x86_feature_detected!` caches its answer and this runs once per
/// frame, so the check costs nothing measurable.
fn row_impl() -> fn(&[u8], &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return |s: &[u8], d: &mut [u8]| unsafe { avx2::row_luma(s, d) };
        }
    }
    luma_row
}

/// BGRA -> luma over a whole image, honouring both row pitches.
///
/// Neither side is necessarily tightly packed. A mapped DXGI staging surface is
/// padded to a pitch wider than `width * 4`, and the destination may be a
/// sub-rectangle of the dirty-rect accumulator, whose rows are strided by the
/// full frame width. Rows are therefore addressed individually on both sides
/// rather than treating either as one contiguous run.
fn luma_image(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    pitch: usize,
    dst_pitch: usize,
) {
    let row_bytes = width * 4;
    let row_fn = row_impl();
    for row in 0..height {
        let s = row * pitch;
        let d = row * dst_pitch;
        row_fn(&src[s..s + row_bytes], &mut dst[d..d + width]);
    }
}

/// Convert a BGRA image to 8-bit luma, writing into a caller-owned buffer.
///
/// Byte-identical to `rapidshot.processor.numpy_processor.bgra_to_gray`, which
/// is asserted by the test suite over all 2^24 BGR combinations.
///
/// Both addresses come from Python as integers, matching how the rest of this
/// extension takes buffers. The caller owns both allocations and must keep them
/// alive across the call; the sizes are validated against `width`/`height`
/// before anything is dereferenced, so a short buffer is refused rather than
/// overrun.
///
/// # Safety
///
/// `src_ptr` must point to at least `pitch * height` readable bytes and
/// `dst_ptr` to at least `width * height` writable bytes. This is why
/// `src_len`/`dst_len` are required rather than inferred: they are what the
/// validation below has to work with.
#[pyfunction]
#[pyo3(signature = (src_ptr, src_len, dst_ptr, dst_len, width, height, pitch=None,
                    dst_pitch=None))]
#[allow(clippy::too_many_arguments)]
pub fn bgra_to_gray_into(
    py: Python<'_>,
    src_ptr: usize,
    src_len: usize,
    dst_ptr: usize,
    dst_len: usize,
    width: usize,
    height: usize,
    pitch: Option<usize>,
    dst_pitch: Option<usize>,
) -> PyResult<()> {
    if width == 0 || height == 0 {
        return Err(PyValueError::new_err(
            "width and height must both be non-zero",
        ));
    }
    if src_ptr == 0 || dst_ptr == 0 {
        return Err(PyValueError::new_err(
            "source and destination must not be null",
        ));
    }

    let pitch = pitch.unwrap_or(width * 4);
    let dst_pitch = dst_pitch.unwrap_or(width);
    let row_bytes = width
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("width * 4 overflows"))?;
    if pitch < row_bytes {
        return Err(PyValueError::new_err(format!(
            "pitch {pitch} is smaller than a {width}px BGRA row ({row_bytes} bytes)"
        )));
    }
    if dst_pitch < width {
        return Err(PyValueError::new_err(format!(
            "dst_pitch {dst_pitch} is smaller than a {width}px luma row"
        )));
    }

    // Each side only has to contain its final row's real bytes, not that row's
    // trailing pad: a mapped surface is not required to carry it, and demanding
    // pitch * height would reject a legitimate buffer -- including, on the
    // destination side, a sub-rectangle that ends at the last row of its parent.
    let need_src = pitch
        .checked_mul(height - 1)
        .and_then(|v| v.checked_add(row_bytes))
        .ok_or_else(|| PyValueError::new_err("source extent overflows"))?;
    let need_dst = dst_pitch
        .checked_mul(height - 1)
        .and_then(|v| v.checked_add(width))
        .ok_or_else(|| PyValueError::new_err("destination extent overflows"))?;

    if src_len < need_src {
        return Err(PyValueError::new_err(format!(
            "source buffer is {src_len} bytes; {width}x{height} at pitch {pitch} needs {need_src}"
        )));
    }
    if dst_len < need_dst {
        return Err(PyValueError::new_err(format!(
            "destination buffer is {dst_len} bytes; {width}x{height} luma needs {need_dst}"
        )));
    }

    // Releasing the GIL is half the point of doing this here: a 1080p
    // conversion is hundreds of microseconds of pure memory traffic that touches
    // no Python objects, so a capture thread should not hold the interpreter for
    // it. `detach` is PyO3 0.29's name for what was `allow_threads` -- the GIL
    // API was renamed in 0.25, and this is the first place in the crate to
    // release it, so there was no local precedent to copy.
    py.detach(|| {
        let src = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, need_src) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, need_dst) };
        luma_image(src, dst, width, height, pitch, dst_pitch);
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference the Python side is held to, spelled out independently.
    fn reference(b: u8, g: u8, r: u8) -> u8 {
        let acc = (r as f64) * 77.0 + (g as f64) * 150.0 + (b as f64) * 29.0 + 128.0;
        (acc as u32 >> 8) as u8
    }

    #[test]
    fn matches_reference_for_every_bgr_triple() {
        for b in 0..=255u8 {
            for g in 0..=255u8 {
                for r in 0..=255u8 {
                    assert_eq!(luma_px(b, g, r), reference(b, g, r), "bgr=({b},{g},{r})");
                }
            }
        }
    }

    #[test]
    fn intermediate_never_leaves_u16_range() {
        let max = 255u32 * LUMA_R + 255 * LUMA_G + 255 * LUMA_B + LUMA_ROUND;
        assert_eq!(max, 65408);
        assert!(max <= u16::MAX as u32);
    }

    #[test]
    fn honours_row_pitch() {
        let (w, h, pitch) = (3usize, 2usize, 3 * 4 + 8);
        let mut src = vec![0u8; pitch * h];
        for row in 0..h {
            for x in 0..w {
                let o = row * pitch + x * 4;
                src[o] = 10; // B
                src[o + 1] = 20; // G
                src[o + 2] = 30; // R
                src[o + 3] = 255;
            }
            // Padding after the real pixels must not be read as image data.
            src[row * pitch + w * 4..row * pitch + pitch].fill(0xFF);
        }
        let mut dst = vec![0u8; w * h];
        luma_image(&src, &mut dst, w, h, pitch, w);
        let want = luma_px(10, 20, 30);
        assert!(dst.iter().all(|&v| v == want), "{dst:?}");
    }

    /// The AVX2 kernel must agree with the scalar one on **every** BGR triple.
    ///
    /// This is the test that earns the `unsafe`, and the reason it is exhaustive
    /// rather than sampled: the failure mode this design exists to avoid --
    /// signed saturation in a 16-bit accumulator -- only shows up in bright
    /// pixels. A random sample would very likely pass while highlights were
    /// being clamped. 16.7M pixels through a vector kernel costs milliseconds,
    /// so there is no reason to sample.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_matches_scalar_for_every_bgr_triple() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not present; scalar path is the only one in use");
            return;
        }
        const CHUNK: usize = 1 << 16; // a multiple of 8: all vector, no tail
        let mut src = vec![0u8; CHUNK * 4];
        let mut got = vec![0u8; CHUNK];
        let mut want = vec![0u8; CHUNK];

        let mut base = 0usize;
        while base < (1usize << 24) {
            for k in 0..CHUNK {
                let v = base + k;
                let (b, g, r) = (
                    (v & 0xFF) as u8,
                    ((v >> 8) & 0xFF) as u8,
                    ((v >> 16) & 0xFF) as u8,
                );
                src[k * 4] = b;
                src[k * 4 + 1] = g;
                src[k * 4 + 2] = r;
                src[k * 4 + 3] = 0xFF; // alpha must not reach the result
                want[k] = luma_px(b, g, r);
            }
            unsafe { avx2::row_luma(&src, &mut got) };
            assert_eq!(got, want, "mismatch in chunk starting at {base}");
            base += CHUNK;
        }
    }

    /// Every width from 1 to 64 covers all eight possible tail lengths.
    ///
    /// A tail bug is invisible at any width that happens to be a multiple of 8,
    /// which includes the 1920 this library actually runs at -- so testing only
    /// realistic frame widths would prove nothing about the tail.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_agrees_with_scalar_at_every_width() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        for width in 1..=64usize {
            let src: Vec<u8> = (0..width * 4).map(|i| (i * 7 % 251) as u8).collect();
            let mut want = vec![0u8; width];
            luma_row(&src, &mut want);
            let mut got = vec![0u8; width];
            // Through the dispatcher, so this also proves it selects the vector
            // path when the CPU has it.
            luma_image(&src, &mut got, width, 1, width * 4, width);
            assert_eq!(got, want, "width {width}");
        }
    }

    /// A vector store must not spill into the next row of a strided destination.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_stores_never_cross_a_row_boundary() {
        for width in [1usize, 7, 8, 9, 15, 16, 17, 31, 33] {
            let (parent_w, height) = (width + 9, 4usize);
            let mut parent = vec![0xAAu8; parent_w * height];
            let src: Vec<u8> = (0..width * 4 * height)
                .map(|i| (i * 13 % 251) as u8)
                .collect();
            luma_image(&src, &mut parent, width, height, width * 4, parent_w);
            for row in 0..height {
                let tail = row * parent_w + width;
                assert!(
                    parent[tail..tail + 9].iter().all(|&b| b == 0xAA),
                    "width {width} row {row}: a store ran past the row end"
                );
            }
        }
    }

    #[test]
    fn writes_a_strided_destination_without_touching_neighbours() {
        // A dirty-rect patch writes a sub-rectangle of the accumulator, so the
        // destination rows are strided by the parent's width. Everything outside
        // the rectangle must be left exactly as it was.
        let (parent_w, parent_h) = (16usize, 8usize);
        let (w, h) = (5usize, 3usize);
        let (x0, y0) = (4usize, 2usize);

        let mut src = vec![0u8; w * 4 * h];
        for px in src.chunks_exact_mut(4) {
            px[0] = 10;
            px[1] = 20;
            px[2] = 30;
            px[3] = 255;
        }

        let mut parent = vec![0xAAu8; parent_w * parent_h];
        let offset = y0 * parent_w + x0;
        luma_image(&src, &mut parent[offset..], w, h, w * 4, parent_w);

        let want = luma_px(10, 20, 30);
        for y in 0..parent_h {
            for x in 0..parent_w {
                let inside = (y0..y0 + h).contains(&y) && (x0..x0 + w).contains(&x);
                let expected = if inside { want } else { 0xAA };
                assert_eq!(parent[y * parent_w + x], expected, "at ({x},{y})");
            }
        }
    }
}
