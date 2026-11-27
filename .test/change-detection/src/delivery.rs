use std::{mem, slice};

use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};

use crate::fingerprint::{Fingerprint, Output};
use crate::{gpu::GpuFrame, AnyResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Fingerprint,
    HardwareDelivery,
    ShaderDelivery,
    Independent,
    Fused,
    CpuExact,
    Verify,
}

impl Mode {
    fn retains_hashes(self) -> bool {
        matches!(
            self,
            Self::Fingerprint | Self::Independent | Self::Fused | Self::Verify
        )
    }

    fn retains_pixels(self) -> bool {
        matches!(self, Self::CpuExact | Self::Verify)
    }
}

pub fn retained_history_bytes(mode: Mode, width: u32, height: u32, tile_count: u32) -> usize {
    let pixels = usize::from(mode.retains_pixels()) * width as usize * height as usize;
    let hashes = usize::from(mode.retains_hashes()) * tile_count as usize;
    pixels * size_of::<u32>() + hashes * size_of::<u64>()
}

#[derive(Clone, Copy)]
struct Grid {
    width: usize,
    height: usize,
    tile_size: usize,
}

impl Grid {
    fn differs<'a>(
        self,
        tile: u32,
        previous: &[u32],
        current_row: impl Fn(usize) -> &'a [u32],
    ) -> bool {
        let columns = self.width.div_ceil(self.tile_size);
        let x = tile as usize % columns * self.tile_size;
        let y = tile as usize / columns * self.tile_size;
        let end_x = (x + self.tile_size).min(self.width);
        let end_y = (y + self.tile_size).min(self.height);
        (y..end_y).any(|row| {
            let start = row * self.width;
            previous[start + x..start + end_x] != current_row(row)[x..end_x]
        })
    }
}

struct TextureReadback {
    context: ID3D11DeviceContext,
    staging: ID3D11Texture2D,
    grid: Grid,
}

impl TextureReadback {
    fn new(device: &ID3D11Device, context: &ID3D11DeviceContext, grid: Grid) -> AnyResult<Self> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: grid.width as u32,
            Height: grid.height as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..Default::default()
        };
        let mut staging = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut staging))? };
        Ok(Self {
            context: context.clone(),
            staging: staging.ok_or("D3D11 did not create a staging texture for delivery")?,
            grid,
        })
    }

    fn submit(&self, frame: &GpuFrame) {
        unsafe { self.context.CopyResource(&self.staging, &frame.texture) };
    }

    fn map(&self) -> AnyResult<MappedTexture<'_>> {
        let mut data = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut data))?
        };
        let mapped = MappedTexture { owner: self, data };
        if mapped.data.RowPitch as usize % size_of::<u32>() != 0
            || (mapped.data.RowPitch as usize) < self.grid.width * size_of::<u32>()
        {
            return Err("D3D11 returned an invalid row pitch for the delivered frame".into());
        }
        Ok(mapped)
    }
}

struct MappedTexture<'a> {
    owner: &'a TextureReadback,
    data: D3D11_MAPPED_SUBRESOURCE,
}

impl MappedTexture<'_> {
    fn row(&self, y: usize) -> &[u32] {
        unsafe {
            let data = self
                .data
                .pData
                .cast::<u8>()
                .add(y * self.data.RowPitch as usize);
            slice::from_raw_parts(data.cast(), self.owner.grid.width)
        }
    }

    fn deliver(&self, pixels: &mut [u32]) {
        for (y, row) in pixels.chunks_exact_mut(self.owner.grid.width).enumerate() {
            row.copy_from_slice(self.row(y));
        }
    }
}

impl Drop for MappedTexture<'_> {
    fn drop(&mut self) {
        unsafe { self.owner.context.Unmap(&self.owner.staging, 0) };
    }
}

pub struct Delivery {
    fingerprint: Fingerprint,
    texture: TextureReadback,
    grid: Grid,
    all_tiles: Vec<u32>,
    previous_hashes: Vec<u64>,
    current_hashes: Vec<u64>,
    previous_pixels: Vec<u32>,
    delivered_pixels: Vec<u32>,
    mode: Option<Mode>,
    has_delivery: bool,
    second_map_waited: Option<bool>,
    cpu_digest_update_us: Option<f64>,
}

impl Delivery {
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> AnyResult<Self> {
        let fingerprint = Fingerprint::new(device, context, width, height, tile_size)?;
        let grid = Grid {
            width: width as usize,
            height: height as usize,
            tile_size: tile_size as usize,
        };
        let texture = TextureReadback::new(device, context, grid)?;
        let tile_count = width.div_ceil(tile_size) * height.div_ceil(tile_size);
        Ok(Self {
            fingerprint,
            texture,
            grid,
            all_tiles: (0..tile_count).collect(),
            current_hashes: vec![0; tile_count as usize],
            previous_hashes: Vec::new(),
            previous_pixels: Vec::new(),
            delivered_pixels: Vec::new(),
            mode: None,
            has_delivery: false,
            second_map_waited: None,
            cpu_digest_update_us: None,
        })
    }

    pub fn reset(
        &mut self,
        mode: Mode,
        previous_gpu: &GpuFrame,
        previous_cpu: Option<&[u32]>,
    ) -> AnyResult<()> {
        self.mode = None;
        self.has_delivery = false;
        self.second_map_waited = None;
        self.cpu_digest_update_us = None;
        let pixel_count = self.grid.width * self.grid.height;
        self.delivered_pixels.resize(
            if mode == Mode::Fingerprint {
                0
            } else {
                pixel_count
            },
            0,
        );
        if mode.retains_pixels() {
            let previous = previous_cpu
                .filter(|pixels| pixels.len() == pixel_count)
                .ok_or("exact CPU checking needs a full previous frame at reset")?;
            self.previous_pixels.resize(pixel_count, 0);
            self.previous_pixels.copy_from_slice(previous);
        } else {
            self.previous_pixels = Vec::new();
        }
        if mode.retains_hashes() {
            self.previous_hashes.resize(self.all_tiles.len(), 0);
            self.fingerprint
                .submit(Output::Hash, previous_gpu, &self.all_tiles)?;
            self.fingerprint
                .read(Output::Hash, &mut [], &mut self.previous_hashes, false)?;
        } else {
            self.previous_hashes = Vec::new();
        }
        self.mode = Some(mode);
        Ok(())
    }

    pub fn run(
        &mut self,
        mode: Mode,
        current: &GpuFrame,
        candidates: &[u32],
        changed: &mut Vec<u32>,
    ) -> AnyResult<()> {
        if self.mode != Some(mode) {
            return Err("reset the delivery arm before running it".into());
        }
        if candidates.len() > self.all_tiles.len()
            || candidates
                .last()
                .is_some_and(|&tile| tile as usize >= self.all_tiles.len())
        {
            return Err("delivery candidates are outside the tile grid".into());
        }
        debug_assert!(candidates.windows(2).all(|pair| pair[0] < pair[1]));
        self.has_delivery = false;
        self.second_map_waited = None;
        self.cpu_digest_update_us = None;
        changed.clear();
        changed.reserve(candidates.len());
        let hashes = &mut self.current_hashes[..candidates.len()];

        match mode {
            Mode::Fingerprint => {
                self.fingerprint.submit(Output::Hash, current, candidates)?;
                self.fingerprint
                    .read(Output::Hash, &mut [], hashes, false)?;
            }
            Mode::HardwareDelivery | Mode::CpuExact => {
                self.texture.submit(current);
                let mapped = self.texture.map()?;
                if mode == Mode::CpuExact {
                    changed.extend(candidates.iter().copied().filter(|&tile| {
                        self.grid
                            .differs(tile, &self.previous_pixels, |y| mapped.row(y))
                    }));
                }
                mapped.deliver(&mut self.delivered_pixels);
            }
            Mode::ShaderDelivery => {
                self.fingerprint.submit(Output::Pixels, current, &[])?;
                self.fingerprint.read(
                    Output::Pixels,
                    &mut self.delivered_pixels,
                    &mut [],
                    false,
                )?;
            }
            Mode::Independent => {
                // Both GPU copies are submitted first. The image Map follows the
                // later copy. Probe whether the second synchronized result is ready.
                self.fingerprint.submit(Output::Hash, current, candidates)?;
                self.texture.submit(current);
                self.texture.map()?.deliver(&mut self.delivered_pixels);
                if !candidates.is_empty() {
                    self.second_map_waited =
                        Some(self.fingerprint.read(Output::Hash, &mut [], hashes, true)?);
                }
            }
            Mode::Fused | Mode::Verify => {
                self.fingerprint
                    .submit(Output::Fused, current, candidates)?;
                self.fingerprint
                    .read(Output::Fused, &mut self.delivered_pixels, hashes, false)?;
            }
        }

        if mode.retains_hashes() {
            let start = std::time::Instant::now();
            for (&tile, &hash) in candidates.iter().zip(hashes.iter()) {
                let history = &mut self.previous_hashes[tile as usize];
                let tile_changed = verified_change(*history, hash, || {
                    mode == Mode::Verify
                        && self.grid.differs(tile, &self.previous_pixels, |y| {
                            let start = y * self.grid.width;
                            &self.delivered_pixels[start..start + self.grid.width]
                        })
                });
                *history = hash;
                if tile_changed {
                    changed.push(tile);
                }
            }
            self.cpu_digest_update_us = Some(start.elapsed().as_secs_f64() * 1_000_000.0);
        }
        if mode.retains_pixels() {
            // The current packed delivery becomes the preallocated baseline for
            // the next chronological comparison. There is no extra frame copy.
            mem::swap(&mut self.previous_pixels, &mut self.delivered_pixels);
        }
        self.has_delivery = mode != Mode::Fingerprint;
        Ok(())
    }

    pub fn delivered(&self) -> Option<&[u32]> {
        self.mode.filter(|_| self.has_delivery).map(|mode| {
            if mode.retains_pixels() {
                &self.previous_pixels[..]
            } else {
                &self.delivered_pixels[..]
            }
        })
    }

    pub fn fingerprints(&self) -> &[u64] {
        &self.previous_hashes
    }

    pub fn second_map_waited(&self) -> Option<bool> {
        self.second_map_waited
    }

    pub fn cpu_digest_update_us(&self) -> Option<f64> {
        self.cpu_digest_update_us
    }
}

fn verified_change(previous: u64, current: u64, check_pixels: impl FnOnce() -> bool) -> bool {
    previous != current || check_pixels()
}

#[cfg(test)]
mod tests {
    use super::{verified_change, Grid};

    #[test]
    fn equal_fingerprints_still_check_packed_pixels_in_an_edge_tile() {
        let grid = Grid {
            width: 3,
            height: 2,
            tile_size: 2,
        };
        let old = [1, 2, 3, 4, 5, 6];
        let swapped = [1, 2, 6, 4, 5, 3];
        assert!(verified_change(7, 7, || grid
            .differs(1, &old, |y| { &swapped[y * 3..y * 3 + 3] })));
        assert!(!verified_change(7, 7, || grid
            .differs(0, &old, |y| { &swapped[y * 3..y * 3 + 3] })));
        assert!(verified_change(7, 8, || panic!(
            "changed fingerprint should skip the pixel check"
        )));
    }
}
