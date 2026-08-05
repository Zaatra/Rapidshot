//! BGRA -> RGB / BGR / RGBA channel reordering, byte-exact with the NumPy path.
//!
//! These are the *common* modes -- RGB is what most CV consumers hand to a model
//! -- and they were running far below what the memory system allows. From
//! `benchmarks/baseline.json`, recorded on the dev machine at 1920x1080:
//!
//! ```text
//! convert.BGRA   0.209 ms   33.2 GB/s   <- pure copy, no reorder: the ceiling
//! convert.RGB    1.817 ms    4.0 GB/s
//! convert.BGR    1.818 ms    4.0 GB/s
//! convert.RGBA   2.437 ms    2.9 GB/s
//! ```
//!
//! The BGRA row is the control that makes the case: the same 8.29 MB moves at
//! 33 GB/s when nothing is reordered, so the gap is not the memory system. It is
//! that the NumPy path assigns one channel at a time --
//! `dst[..., 0] = src[..., 2]` and so on -- and each of those is a separate
//! strided gather/scatter pass over the whole frame. Three passes touching one
//! useful byte in four, where one pass could read each cache line once.
//!
//! GRAY was the slowest mode and got fixed first (see `luma.rs`), but it was
//! never the most used one. The absolute saving available here is larger.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Which reordering to apply. Mirrors the strings `NumpyProcessor` already uses.
#[derive(Copy, Clone)]
enum Mode {
    /// BGRA -> RGB: reverse the colour channels, drop alpha.
    Rgb,
    /// BGRA -> BGR: keep channel order, drop alpha.
    Bgr,
    /// BGRA -> RGBA: swap red and blue, preserve alpha.
    Rgba,
}

impl Mode {
    fn parse(name: &str) -> PyResult<Self> {
        match name {
            "RGB" => Ok(Mode::Rgb),
            "BGR" => Ok(Mode::Bgr),
            "RGBA" => Ok(Mode::Rgba),
            other => Err(PyValueError::new_err(format!(
                "unsupported mode {other:?}; expected RGB, BGR or RGBA"
            ))),
        }
    }

    fn channels(self) -> usize {
        match self {
            Mode::Rgb | Mode::Bgr => 3,
            Mode::Rgba => 4,
        }
    }
}

// Each row function is a single pass: read four bytes, write three or four.
// Written as `chunks_exact` zips so the bounds checks fold away and LLVM sees a
// fixed-stride pattern it can widen. Kept as three separate functions rather
// than one with a branch inside the loop, so the match happens once per row
// instead of once per pixel.

#[inline]
fn row_rgb(src: &[u8], dst: &mut [u8]) {
    for (px, out) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        out[0] = px[2];
        out[1] = px[1];
        out[2] = px[0];
    }
}

#[inline]
fn row_bgr(src: &[u8], dst: &mut [u8]) {
    for (px, out) in src.chunks_exact(4).zip(dst.chunks_exact_mut(3)) {
        out[0] = px[0];
        out[1] = px[1];
        out[2] = px[2];
    }
}

#[inline]
fn row_rgba(src: &[u8], dst: &mut [u8]) {
    for (px, out) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
        out[0] = px[2];
        out[1] = px[1];
        out[2] = px[0];
        out[3] = px[3];
    }
}

// ---------------------------------------------------------------------------
// AVX2 paths
// ---------------------------------------------------------------------------
//
// `pshufb` is precisely the instruction for this: an arbitrary byte permutation
// within a 128-bit lane, chosen by a mask. The scalar loops above reach 10-13
// GB/s against a 33 GB/s ceiling because the autovectoriser will not invent that
// permutation for itself -- RGB is the worst case, since reversing each triple
// defeats it entirely and it degrades to per-byte stores.
//
// Two details make this safe rather than merely fast:
//
//  * **Exact stores, no overrun.** The usual trick for a 3-byte output is to
//    store a full 16 or 32 bytes and let the next iteration overwrite the
//    surplus. That would write past the end of the final group of a row, which
//    here can be the end of the caller's buffer or -- worse, because it would
//    corrupt live pixels rather than crash -- the next row of a sub-rectangle
//    being patched into the accumulator. So the 3-byte modes store exactly 24
//    bytes as 16 + 8, and pitch and tail handling need no special cases.
//  * **Byte permutation only.** No arithmetic, so unlike a luma kernel there is
//    no accumulator width to reason about and no saturating-add trap. The
//    output is bit-identical to the scalar path by construction, and the tests
//    assert that across every width from 1 to 64.

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use core::arch::x86_64::*;

    /// Within each 128-bit lane: gather 4 pixels' worth of 3 bytes into the low
    /// 12, zeroing the top 4. `-1` sets the mask's high bit, which zeroes.
    macro_rules! lane_mask {
        ($a:expr, $b:expr, $c:expr) => {
            _mm256_setr_epi8(
                $a,
                $b,
                $c,
                $a + 4,
                $b + 4,
                $c + 4,
                $a + 8,
                $b + 8,
                $c + 8,
                $a + 12,
                $b + 12,
                $c + 12,
                -1,
                -1,
                -1,
                -1,
                $a,
                $b,
                $c,
                $a + 4,
                $b + 4,
                $c + 4,
                $a + 8,
                $b + 8,
                $c + 8,
                $a + 12,
                $b + 12,
                $c + 12,
                -1,
                -1,
                -1,
                -1,
            )
        };
    }

    /// Shared body for the two 3-byte modes: 8 pixels in, 24 bytes out.
    ///
    /// After the per-lane shuffle, lane 0 holds 12 useful bytes then 4 zeros and
    /// lane 1 the same, so the halves must be compacted against each other
    /// before storing. `vpermd` does it at dword granularity: taking dwords
    /// 0,1,2 then 4,5,6 lays 24 contiguous bytes into the low half.
    #[target_feature(enable = "avx2")]
    unsafe fn row_3byte(src: &[u8], dst: &mut [u8], shuf: __m256i, order: [usize; 3]) {
        let perm = _mm256_setr_epi32(0, 1, 2, 4, 5, 6, 7, 7);
        let px = dst.len() / 3;
        let vec_px = px & !7usize;

        let mut i = 0;
        while i < vec_px {
            let v = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            let c = _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(v, shuf), perm);
            let out = dst.as_mut_ptr().add(i * 3);
            // Exactly 24 bytes: 16 from the low lane, 8 from the high one.
            _mm_storeu_si128(out as *mut __m128i, _mm256_castsi256_si128(c));
            _mm_storel_epi64(out.add(16) as *mut __m128i, _mm256_extracti128_si256(c, 1));
            i += 8;
        }
        // Tail: fewer than 8 pixels left. `order` states the channel mapping
        // directly so it cannot drift out of step with `shuf`.
        while i < px {
            let s = i * 4;
            let d = i * 3;
            *dst.get_unchecked_mut(d) = *src.get_unchecked(s + order[0]);
            *dst.get_unchecked_mut(d + 1) = *src.get_unchecked(s + order[1]);
            *dst.get_unchecked_mut(d + 2) = *src.get_unchecked(s + order[2]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_rgb(src: &[u8], dst: &mut [u8]) {
        row_3byte(src, dst, lane_mask!(2, 1, 0), [2, 1, 0])
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn row_bgr(src: &[u8], dst: &mut [u8]) {
        row_3byte(src, dst, lane_mask!(0, 1, 2), [0, 1, 2])
    }

    /// RGBA is the easy one: 4 bytes in, 4 bytes out, so the lanes need no
    /// compaction and a single 32-byte store is exact.
    #[target_feature(enable = "avx2")]
    pub unsafe fn row_rgba(src: &[u8], dst: &mut [u8]) {
        let shuf = _mm256_setr_epi8(
            2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8, 11, 14, 13, 12, 15, 2, 1, 0, 3, 6, 5, 4, 7, 10, 9, 8,
            11, 14, 13, 12, 15,
        );
        let px = dst.len() / 4;
        let vec_px = px & !7usize;

        let mut i = 0;
        while i < vec_px {
            let v = _mm256_loadu_si256(src.as_ptr().add(i * 4) as *const __m256i);
            _mm256_storeu_si256(
                dst.as_mut_ptr().add(i * 4) as *mut __m256i,
                _mm256_shuffle_epi8(v, shuf),
            );
            i += 8;
        }
        while i < px {
            let s = i * 4;
            *dst.get_unchecked_mut(s) = *src.get_unchecked(s + 2);
            *dst.get_unchecked_mut(s + 1) = *src.get_unchecked(s + 1);
            *dst.get_unchecked_mut(s + 2) = *src.get_unchecked(s);
            *dst.get_unchecked_mut(s + 3) = *src.get_unchecked(s + 3);
            i += 1;
        }
    }
}

/// Pick the row implementation once per image.
///
/// x86_64 only guarantees SSE2, so AVX2 has to be detected at runtime rather
/// than assumed -- a wheel built with it unconditionally would fault on older
/// hardware. `is_x86_feature_detected!` caches its answer, and this runs once
/// per frame rather than per row, so the check costs nothing measurable.
fn row_impl(mode: Mode) -> fn(&[u8], &mut [u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return match mode {
                Mode::Rgb => |s: &[u8], d: &mut [u8]| unsafe { avx2::row_rgb(s, d) },
                Mode::Bgr => |s: &[u8], d: &mut [u8]| unsafe { avx2::row_bgr(s, d) },
                Mode::Rgba => |s: &[u8], d: &mut [u8]| unsafe { avx2::row_rgba(s, d) },
            };
        }
    }
    match mode {
        Mode::Rgb => row_rgb,
        Mode::Bgr => row_bgr,
        Mode::Rgba => row_rgba,
    }
}

/// Reorder a whole image, honouring both row pitches.
///
/// Neither side is necessarily tightly packed: a mapped DXGI staging surface is
/// padded, and the destination may be a sub-rectangle of the dirty-rect
/// accumulator whose rows are strided by the full frame width.
fn swizzle_image(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    pitch: usize,
    dst_pitch: usize,
    mode: Mode,
) {
    let src_row = width * 4;
    let dst_row = width * mode.channels();
    let row_fn = row_impl(mode);
    for row in 0..height {
        let s = row * pitch;
        let d = row * dst_pitch;
        row_fn(&src[s..s + src_row], &mut dst[d..d + dst_row]);
    }
}

/// Reorder a BGRA image into `mode`, writing into a caller-owned buffer.
///
/// Byte-identical to what `NumpyProcessor.convert_into` produces for the same
/// mode, which the test suite asserts on both sides of the FFI boundary.
///
/// Addresses arrive from Python as integers, matching the rest of this
/// extension. Sizes are validated against the geometry before anything is
/// dereferenced, so a short buffer is refused rather than overrun.
///
/// # Safety
///
/// `src_ptr` must point to at least `pitch * (height - 1) + width * 4` readable
/// bytes, and `dst_ptr` to `dst_pitch * (height - 1) + width * channels`
/// writable bytes. The explicit lengths are what the validation below checks
/// that requirement against.
#[pyfunction]
#[pyo3(signature = (src_ptr, src_len, dst_ptr, dst_len, width, height, mode,
                    pitch=None, dst_pitch=None))]
#[allow(clippy::too_many_arguments)]
pub fn bgra_swizzle_into(
    py: Python<'_>,
    src_ptr: usize,
    src_len: usize,
    dst_ptr: usize,
    dst_len: usize,
    width: usize,
    height: usize,
    mode: &str,
    pitch: Option<usize>,
    dst_pitch: Option<usize>,
) -> PyResult<()> {
    let mode = Mode::parse(mode)?;
    let channels = mode.channels();

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

    let src_row = width
        .checked_mul(4)
        .ok_or_else(|| PyValueError::new_err("width * 4 overflows"))?;
    let dst_row = width
        .checked_mul(channels)
        .ok_or_else(|| PyValueError::new_err("width * channels overflows"))?;
    let pitch = pitch.unwrap_or(src_row);
    let dst_pitch = dst_pitch.unwrap_or(dst_row);

    if pitch < src_row {
        return Err(PyValueError::new_err(format!(
            "pitch {pitch} is smaller than a {width}px BGRA row ({src_row} bytes)"
        )));
    }
    if dst_pitch < dst_row {
        return Err(PyValueError::new_err(format!(
            "dst_pitch {dst_pitch} is smaller than a {width}px row ({dst_row} bytes)"
        )));
    }

    // Each side only has to hold its final row's real bytes, not that row's
    // trailing pad -- including, on the destination side, a sub-rectangle that
    // ends at the last row of its parent.
    let need_src = pitch
        .checked_mul(height - 1)
        .and_then(|v| v.checked_add(src_row))
        .ok_or_else(|| PyValueError::new_err("source extent overflows"))?;
    let need_dst = dst_pitch
        .checked_mul(height - 1)
        .and_then(|v| v.checked_add(dst_row))
        .ok_or_else(|| PyValueError::new_err("destination extent overflows"))?;

    if src_len < need_src {
        return Err(PyValueError::new_err(format!(
            "source buffer is {src_len} bytes; {width}x{height} at pitch {pitch} \
             needs {need_src}"
        )));
    }
    if dst_len < need_dst {
        return Err(PyValueError::new_err(format!(
            "destination buffer is {dst_len} bytes; {width}x{height}x{channels} \
             needs {need_dst}"
        )));
    }

    // Pure memory traffic touching no Python objects, so the interpreter should
    // not be held for it. See the note in `luma.rs`: `detach` is PyO3 0.29's
    // name for what used to be `allow_threads`.
    py.detach(|| {
        let src = unsafe { std::slice::from_raw_parts(src_ptr as *const u8, need_src) };
        let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut u8, need_dst) };
        swizzle_image(src, dst, width, height, pitch, dst_pitch, mode);
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bgra(n: usize) -> Vec<u8> {
        (0..n * 4).map(|i| (i * 7 % 251) as u8).collect()
    }

    #[test]
    fn rgb_reverses_colour_channels_and_drops_alpha() {
        let src = bgra(5);
        let mut dst = vec![0u8; 5 * 3];
        swizzle_image(&src, &mut dst, 5, 1, 20, 15, Mode::Rgb);
        for i in 0..5 {
            assert_eq!(dst[i * 3], src[i * 4 + 2], "R at {i}");
            assert_eq!(dst[i * 3 + 1], src[i * 4 + 1], "G at {i}");
            assert_eq!(dst[i * 3 + 2], src[i * 4], "B at {i}");
        }
    }

    #[test]
    fn bgr_preserves_order_and_drops_alpha() {
        let src = bgra(5);
        let mut dst = vec![0u8; 5 * 3];
        swizzle_image(&src, &mut dst, 5, 1, 20, 15, Mode::Bgr);
        for i in 0..5 {
            assert_eq!(&dst[i * 3..i * 3 + 3], &src[i * 4..i * 4 + 3], "px {i}");
        }
    }

    #[test]
    fn rgba_swaps_red_and_blue_and_keeps_alpha() {
        let src = bgra(5);
        let mut dst = vec![0u8; 5 * 4];
        swizzle_image(&src, &mut dst, 5, 1, 20, 20, Mode::Rgba);
        for i in 0..5 {
            assert_eq!(dst[i * 4], src[i * 4 + 2], "R at {i}");
            assert_eq!(dst[i * 4 + 1], src[i * 4 + 1], "G at {i}");
            assert_eq!(dst[i * 4 + 2], src[i * 4], "B at {i}");
            assert_eq!(dst[i * 4 + 3], src[i * 4 + 3], "A at {i}");
        }
    }

    #[test]
    fn honours_both_pitches_and_leaves_neighbours_alone() {
        // Source padded past its pixels; destination is a sub-rectangle of a
        // wider parent. Everything outside the rectangle must survive.
        let (w, h) = (4usize, 3usize);
        let (parent_w, parent_h) = (9usize, 6usize);
        let (x0, y0) = (3usize, 2usize);
        let src_pitch = w * 4 + 12;

        let mut src = vec![0xFFu8; src_pitch * h];
        for row in 0..h {
            for x in 0..w {
                let o = row * src_pitch + x * 4;
                src[o] = 11;
                src[o + 1] = 22;
                src[o + 2] = 33;
                src[o + 3] = 44;
            }
        }

        let parent_pitch = parent_w * 3;
        let mut parent = vec![0xAAu8; parent_pitch * parent_h];
        let offset = y0 * parent_pitch + x0 * 3;
        swizzle_image(
            &src,
            &mut parent[offset..],
            w,
            h,
            src_pitch,
            parent_pitch,
            Mode::Rgb,
        );

        for y in 0..parent_h {
            for x in 0..parent_w {
                let base = y * parent_pitch + x * 3;
                let inside = (y0..y0 + h).contains(&y) && (x0..x0 + w).contains(&x);
                let want: [u8; 3] = if inside { [33, 22, 11] } else { [0xAA; 3] };
                assert_eq!(&parent[base..base + 3], &want, "at ({x},{y})");
            }
        }
    }

    /// The AVX2 kernels must agree with the scalar ones at every width.
    ///
    /// This is the test that earns the `unsafe`. Widths 1..=64 cover: no vector
    /// iterations at all, exactly one, several, and every possible tail length
    /// from 0 to 7 -- which is where a lane-compaction or tail bug hides, since
    /// it would be invisible at any width that happens to be a multiple of 8.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_agrees_with_scalar_at_every_width() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("AVX2 not present; scalar path is the only one in use");
            return;
        }
        for &(mode, channels, scalar) in &[
            (Mode::Rgb, 3usize, row_rgb as fn(&[u8], &mut [u8])),
            (Mode::Bgr, 3, row_bgr as fn(&[u8], &mut [u8])),
            (Mode::Rgba, 4, row_rgba as fn(&[u8], &mut [u8])),
        ] {
            for width in 1..=64usize {
                let src = bgra(width);
                let mut want = vec![0u8; width * channels];
                scalar(&src, &mut want);

                let mut got = vec![0u8; width * channels];
                // Goes through the dispatcher, so this also proves the
                // dispatcher selects the vector path when the CPU has it.
                swizzle_image(&src, &mut got, width, 1, width * 4, width * channels, mode);
                assert_eq!(got, want, "mode {} width {width}", mode.channels());
            }
        }
    }

    /// A vector store must not spill into the next row of a strided destination.
    ///
    /// The conventional 3-byte trick stores 16 or 32 bytes and lets the next
    /// iteration overwrite the surplus; here that would corrupt live pixels of
    /// the row below when patching a sub-rectangle, which is silent rather than
    /// fatal. The kernels store exactly 24 bytes, and this pins that down.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn vector_stores_never_cross_a_row_boundary() {
        for width in [1usize, 7, 8, 9, 15, 16, 17, 31, 33] {
            let (parent_w, height) = (width + 5, 4usize);
            let parent_pitch = parent_w * 3;
            let mut parent = vec![0xAAu8; parent_pitch * height];

            let src_pitch = width * 4;
            let src = bgra(width * height);
            swizzle_image(
                &src,
                &mut parent,
                width,
                height,
                src_pitch,
                parent_pitch,
                Mode::Rgb,
            );

            // The 5 pixels of slack at the end of every row must be untouched.
            for row in 0..height {
                let tail = row * parent_pitch + width * 3;
                assert!(
                    parent[tail..tail + 15].iter().all(|&b| b == 0xAA),
                    "width {width} row {row}: a store ran past the row end"
                );
            }
        }
    }

    #[test]
    fn rejects_unknown_mode() {
        assert!(Mode::parse("GRAY").is_err());
        assert!(Mode::parse("BGRA").is_err());
        assert!(Mode::parse("RGB").is_ok());
    }
}
