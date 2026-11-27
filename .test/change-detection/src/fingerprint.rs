use std::{mem::size_of, ptr, slice};

use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D_SRV_DIMENSION_BUFFER};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::{Common::DXGI_FORMAT_R32_UINT, DXGI_ERROR_WAS_STILL_DRAWING};

use crate::{gpu::GpuFrame, AnyResult};

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Output {
    Hash,
    Pixels,
    Fused,
}

struct Readback {
    buffer: ID3D11Buffer,
    view: ID3D11UnorderedAccessView,
    staging: ID3D11Buffer,
}

impl Readback {
    fn new(device: &ID3D11Device, words: u32) -> AnyResult<Self> {
        let buffer = create_buffer(
            device,
            D3D11_BUFFER_DESC {
                ByteWidth: words * size_of::<u32>() as u32,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_UNORDERED_ACCESS.0 as u32,
                ..Default::default()
            },
            None,
        )?;
        let mut desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
            ..Default::default()
        };
        desc.Anonymous.Buffer = D3D11_BUFFER_UAV {
            FirstElement: 0,
            NumElements: words,
            Flags: 0,
        };
        let mut view = None;
        unsafe { device.CreateUnorderedAccessView(&buffer, Some(&desc), Some(&mut view))? };
        let view = view.ok_or("D3D11 did not create a fingerprint output view")?;
        let staging = create_buffer(
            device,
            D3D11_BUFFER_DESC {
                ByteWidth: words * size_of::<u32>() as u32,
                Usage: D3D11_USAGE_STAGING,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                ..Default::default()
            },
            None,
        )?;
        Ok(Self {
            buffer,
            view,
            staging,
        })
    }
}

pub(crate) struct Fingerprint {
    context: ID3D11DeviceContext,
    hash: ID3D11ComputeShader,
    copy: ID3D11ComputeShader,
    fused: ID3D11ComputeShader,
    geometry: ID3D11Buffer,
    selection_buffer: ID3D11Buffer,
    selection_view: ID3D11ShaderResourceView,
    selection: Vec<u32>,
    hash_result: Readback,
    packed_result: Readback,
    pixels: u32,
    tiles: u32,
}

impl Fingerprint {
    pub(crate) fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> AnyResult<Self> {
        if width == 0 || height == 0 || tile_size == 0 {
            return Err("fingerprint dimensions and tile size must be positive".into());
        }
        let columns = width.div_ceil(tile_size);
        let tiles = columns
            .checked_mul(height.div_ceil(tile_size))
            .filter(|&count| count <= D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION)
            .ok_or("fingerprint has too many tiles for one D3D11 dispatch")?;
        let pixels = width
            .checked_mul(height)
            .ok_or("frame pixel count overflow")?;
        let words = pixels
            .checked_add(tiles * 2)
            .filter(|&count| count <= 1 << D3D11_REQ_BUFFER_RESOURCE_TEXEL_COUNT_2_TO_EXP)
            .ok_or("packed frame and fingerprint buffer exceeds D3D11 limits")?;
        let geometry_words = [width, height, columns, tile_size];
        let geometry_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: geometry_words.as_ptr().cast(),
            ..Default::default()
        };
        let geometry = create_buffer(
            device,
            D3D11_BUFFER_DESC {
                ByteWidth: size_of::<[u32; 4]>() as u32,
                Usage: D3D11_USAGE_IMMUTABLE,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                ..Default::default()
            },
            Some(&geometry_data),
        )?;
        let selection_buffer = create_buffer(
            device,
            D3D11_BUFFER_DESC {
                ByteWidth: tiles * size_of::<u32>() as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                ..Default::default()
            },
            None,
        )?;
        let mut selection_desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R32_UINT,
            ViewDimension: D3D_SRV_DIMENSION_BUFFER,
            ..Default::default()
        };
        selection_desc.Anonymous.Buffer = D3D11_BUFFER_SRV {
            Anonymous1: D3D11_BUFFER_SRV_0 { FirstElement: 0 },
            Anonymous2: D3D11_BUFFER_SRV_1 { NumElements: tiles },
        };
        let mut selection_view = None;
        unsafe {
            device.CreateShaderResourceView(
                &selection_buffer,
                Some(&selection_desc),
                Some(&mut selection_view),
            )?;
        }
        Ok(Self {
            context: context.clone(),
            hash: create_shader(device, c"Fingerprint")?,
            copy: create_shader(device, c"CopyPixels")?,
            fused: create_shader(device, c"Fused")?,
            geometry,
            selection_buffer,
            selection_view: selection_view
                .ok_or("D3D11 did not create a fingerprint selection view")?,
            selection: vec![0; tiles as usize],
            hash_result: Readback::new(device, tiles * 2)?,
            packed_result: Readback::new(device, words)?,
            pixels,
            tiles,
        })
    }

    fn readback(&self, output: Output) -> &Readback {
        if output == Output::Hash {
            &self.hash_result
        } else {
            &self.packed_result
        }
    }

    /// Enqueue the work and the copy to staging without synchronizing the GPU.
    pub(crate) fn submit(
        &mut self,
        output: Output,
        frame: &GpuFrame,
        candidates: &[u32],
    ) -> AnyResult<()> {
        if output == Output::Hash && candidates.is_empty() {
            return Ok(());
        }
        if output == Output::Fused {
            self.selection.fill(0);
            for (ordinal, &tile) in candidates.iter().enumerate() {
                self.selection[tile as usize] = ordinal as u32 + 1;
            }
        }
        let hashes = if output == Output::Pixels {
            0
        } else {
            candidates.len() as u32
        };
        let (shader, groups, words) = match output {
            Output::Hash => (&self.hash, hashes, hashes * 2),
            Output::Pixels => (&self.copy, self.tiles, self.pixels),
            Output::Fused => (&self.fused, self.tiles, self.pixels + hashes * 2),
        };
        let readback = self.readback(output);
        unsafe {
            if output != Output::Pixels {
                let selected = if output == Output::Hash {
                    candidates
                } else {
                    &self.selection
                };
                let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
                self.context.Map(
                    &self.selection_buffer,
                    0,
                    D3D11_MAP_WRITE_DISCARD,
                    0,
                    Some(&mut mapped),
                )?;
                ptr::copy_nonoverlapping(selected.as_ptr(), mapped.pData.cast(), selected.len());
                self.context.Unmap(&self.selection_buffer, 0);
            }
            self.context.CSSetShader(shader, None);
            self.context.CSSetShaderResources(
                0,
                Some(&[
                    Some(frame.view.clone()),
                    (output != Output::Pixels).then(|| self.selection_view.clone()),
                ]),
            );
            self.context
                .CSSetConstantBuffers(0, Some(&[Some(self.geometry.clone())]));
            self.context
                .CSSetUnorderedAccessViews(0, 1, Some(&Some(readback.view.clone())), None);
            self.context.Dispatch(groups, 1, 1);
            self.context.CSSetShaderResources(0, Some(&[None, None]));
            self.context
                .CSSetUnorderedAccessViews(0, 1, Some(&None), None);

            let region = D3D11_BOX {
                right: words * size_of::<u32>() as u32,
                bottom: 1,
                back: 1,
                ..Default::default()
            };
            self.context.CopySubresourceRegion(
                &readback.staging,
                0,
                0,
                0,
                0,
                &readback.buffer,
                0,
                Some(&region),
            );
        }
        Ok(())
    }

    /// Retrieve the enqueued data with one Map. Return whether a nonblocking Map had to wait.
    pub(crate) fn read(
        &self,
        output: Output,
        pixels: &mut [u32],
        hashes: &mut [u64],
        probe_wait: bool,
    ) -> AnyResult<bool> {
        let pixel_words = if output == Output::Hash {
            0
        } else {
            self.pixels as usize
        };
        if pixels.len() != pixel_words
            || hashes.len() > self.tiles as usize
            || (output == Output::Pixels && !hashes.is_empty())
        {
            return Err("fingerprint readback requested the wrong layout".into());
        }
        let words = pixel_words + hashes.len() * 2;
        if words == 0 {
            return Ok(false);
        }
        let readback = self.readback(output);
        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            let flags = if probe_wait {
                D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32
            } else {
                0
            };
            let map = self.context.Map(
                &readback.staging,
                0,
                D3D11_MAP_READ,
                flags,
                Some(&mut mapped),
            );
            let waited = if let Err(error) = map {
                if !probe_wait || error.code() != DXGI_ERROR_WAS_STILL_DRAWING {
                    return Err(error.into());
                }
                self.context
                    .Map(&readback.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
                true
            } else {
                false
            };
            let data = slice::from_raw_parts(mapped.pData.cast::<u32>(), words);
            let (image, fingerprints) = data.split_at(pixel_words);
            pixels.copy_from_slice(image);
            for (fingerprint, words) in hashes.iter_mut().zip(fingerprints.chunks_exact(2)) {
                *fingerprint = u64::from(words[0]) | (u64::from(words[1]) << 32);
            }
            self.context.Unmap(&readback.staging, 0);
            Ok(waited)
        }
    }
}

fn create_buffer(
    device: &ID3D11Device,
    desc: D3D11_BUFFER_DESC,
    data: Option<&D3D11_SUBRESOURCE_DATA>,
) -> AnyResult<ID3D11Buffer> {
    let mut buffer = None;
    unsafe { device.CreateBuffer(&desc, data.map(ptr::from_ref), Some(&mut buffer))? };
    buffer.ok_or_else(|| "D3D11 did not create a fingerprint buffer".into())
}

fn create_shader(device: &ID3D11Device, entry: &std::ffi::CStr) -> AnyResult<ID3D11ComputeShader> {
    let source = include_bytes!("fingerprint.hlsl");
    let mut bytecode: Option<ID3DBlob> = None;
    let mut diagnostics: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            None,
            None,
            None,
            PCSTR(entry.as_ptr().cast()),
            PCSTR(c"cs_5_0".as_ptr().cast()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut bytecode,
            Some(&mut diagnostics),
        )
    };
    if let (Err(error), Some(diagnostics)) = (&result, diagnostics) {
        let bytes = unsafe {
            slice::from_raw_parts(
                diagnostics.GetBufferPointer().cast(),
                diagnostics.GetBufferSize(),
            )
        };
        return Err(format!(
            "{} shader failed ({error}): {}",
            entry.to_string_lossy(),
            String::from_utf8_lossy(bytes)
        )
        .into());
    }
    result?;
    let blob = bytecode.ok_or("D3D11 shader compiler returned no fingerprint bytecode")?;
    let mut shader = None;
    unsafe {
        let bytes = slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize());
        device.CreateComputeShader(bytes, None, Some(&mut shader))?;
    }
    shader.ok_or_else(|| "D3D11 did not create a fingerprint shader".into())
}
