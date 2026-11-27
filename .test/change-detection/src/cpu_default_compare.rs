use std::slice;

#[allow(clippy::too_many_arguments)]
fn scan_tiles(
    previous: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
    previous_stride: usize,
    current_stride: usize,
    tile_size: usize,
    candidates: &[u32],
    mut changed: impl FnMut(u32),
) {
    let columns = width.div_ceil(tile_size);
    for &tile in candidates {
        let tile_index = tile as usize;
        let left = tile_index % columns * tile_size;
        let top = tile_index / columns * tile_size;
        let bytes = (left + tile_size).min(width) * 4 - left * 4;
        let offset = left * 4;
        if (top..(top + tile_size).min(height)).any(|row| {
            let a = row * previous_stride + offset;
            let b = row * current_stride + offset;
            previous[a..a + bytes] != current[b..b + bytes]
        }) {
            changed(tile);
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compare_tiles(
    previous: &[u8],
    current: &[u8],
    width: usize,
    height: usize,
    previous_stride: usize,
    current_stride: usize,
    tile_size: usize,
    candidates: &[u32],
    changed: &mut Vec<u32>,
) {
    changed.clear();
    scan_tiles(
        previous,
        current,
        width,
        height,
        previous_stride,
        current_stride,
        tile_size,
        candidates,
        |tile| changed.push(tile),
    );
}

/// Scratch benchmark ABI. The caller owns and bounds the two read-only frames,
/// the non-overlapping candidate list, and storage for at least that many IDs.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn rs_compare_tiles(
    previous: *const u8,
    current: *const u8,
    width: usize,
    height: usize,
    previous_stride: usize,
    current_stride: usize,
    tile_size: usize,
    candidates: *const u32,
    candidate_count: usize,
    output: *mut u32,
    output_capacity: usize,
) -> usize {
    fn frame_len(width_bytes: usize, rows: usize, pitch: usize) -> Option<usize> {
        if pitch < width_bytes {
            return None;
        }
        rows.checked_sub(1)?
            .checked_mul(pitch)?
            .checked_add(width_bytes)
            .filter(|&length| length <= isize::MAX as usize)
    }
    let Some(width_bytes) = width.checked_mul(4) else {
        return usize::MAX;
    };
    if width == 0
        || height == 0
        || tile_size == 0
        || tile_size > 4096
        || previous.is_null()
        || current.is_null()
        || candidate_count > output_capacity
    {
        return usize::MAX;
    }
    let Some(tiles) = width
        .div_ceil(tile_size)
        .checked_mul(height.div_ceil(tile_size))
    else {
        return usize::MAX;
    };
    if candidate_count > tiles || candidate_count > isize::MAX as usize / size_of::<u32>() {
        return usize::MAX;
    }
    let (Some(previous_len), Some(current_len)) = (
        frame_len(width_bytes, height, previous_stride),
        frame_len(width_bytes, height, current_stride),
    ) else {
        return usize::MAX;
    };
    if candidate_count == 0 {
        return 0;
    }
    if candidates.is_null() || output.is_null() {
        return usize::MAX;
    }
    let ids = slice::from_raw_parts(candidates, candidate_count);
    if ids.windows(2).any(|pair| pair[0] >= pair[1]) || ids[candidate_count - 1] as usize >= tiles {
        return usize::MAX;
    }
    let before = slice::from_raw_parts(previous, previous_len);
    let after = slice::from_raw_parts(current, current_len);
    let mut written = 0;
    scan_tiles(
        before,
        after,
        width,
        height,
        previous_stride,
        current_stride,
        tile_size,
        ids,
        |id| {
            output.add(written).write(id);
            written += 1;
        },
    );
    written
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strided_rows_skip_padding_and_compare_alpha_on_partial_tiles() {
        let mut a = vec![0; 60];
        let mut b = vec![0; 72];
        a[12..20].fill(25);
        b[12..24].fill(83);
        b[2 * 24 + 2 * 4 + 3] = 1;
        let mut changes = Vec::new();
        compare_tiles(&a, &b, 3, 3, 20, 24, 2, &[0, 1, 2, 3], &mut changes);
        assert_eq!(changes, [3]);
        compare_tiles(&a, &b, 3, 3, 20, 24, 2, &[0, 1, 2], &mut changes);
        assert!(changes.is_empty());
    }
}
