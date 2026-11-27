use std::error::Error;
use std::{ptr, slice};

use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D_SRV_DIMENSION_BUFFER};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

pub struct GpuFrame {
    pub(crate) view: ID3D11ShaderResourceView,
    pub(crate) texture: ID3D11Texture2D,
}

pub struct Gpu {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    width: u32,
    height: u32,
    tile_count: u32,
    shader: ID3D11ComputeShader,
    geometry: ID3D11Buffer,
    candidates: ID3D11Buffer,
    pub(crate) candidate_view: ID3D11ShaderResourceView,
    pub(crate) output: ID3D11Buffer,
    output_view: ID3D11UnorderedAccessView,
    staging: ID3D11Buffer,
    seed_done: ID3D11Query,
}

impl Gpu {
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        width: u32,
        height: u32,
        tile_size: u32,
    ) -> Result<Self> {
        if width == 0
            || height == 0
            || width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || tile_size == 0
        {
            return Err("invalid D3D11 comparison dimensions or tile size".into());
        }
        let columns = width.div_ceil(tile_size);
        let tile_count = columns * height.div_ceil(tile_size);
        if tile_count > D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION {
            return Err("too many comparison tiles for a single D3D11 dispatch".into());
        }

        let shader = create_shader(device)?;
        let byte_width = tile_count * size_of::<u32>() as u32;
        let geometry_words = [width, height, columns, tile_size];
        let geometry_data = D3D11_SUBRESOURCE_DATA {
            pSysMem: geometry_words.as_ptr().cast(),
            ..Default::default()
        };
        let geometry = create_buffer(
            device,
            &D3D11_BUFFER_DESC {
                ByteWidth: size_of_val(&geometry_words) as u32,
                Usage: D3D11_USAGE_IMMUTABLE,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                ..Default::default()
            },
            Some(&geometry_data),
        )?;
        let candidates = create_buffer(
            device,
            &D3D11_BUFFER_DESC {
                ByteWidth: byte_width,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                ..Default::default()
            },
            None,
        )?;
        let candidate_view = unsafe {
            let mut desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D_SRV_DIMENSION_BUFFER,
                ..Default::default()
            };
            desc.Anonymous.Buffer = D3D11_BUFFER_SRV {
                Anonymous1: D3D11_BUFFER_SRV_0 { FirstElement: 0 },
                Anonymous2: D3D11_BUFFER_SRV_1 {
                    NumElements: tile_count,
                },
            };
            let mut view = None;
            device.CreateShaderResourceView(&candidates, Some(&desc), Some(&mut view))?;
            view.ok_or("D3D11 returned no candidate view")?
        };
        let output = create_buffer(
            device,
            &D3D11_BUFFER_DESC {
                ByteWidth: byte_width,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_SHADER_RESOURCE).0 as u32,
                ..Default::default()
            },
            None,
        )?;
        let output_view = unsafe {
            let mut desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_R32_UINT,
                ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
                ..Default::default()
            };
            desc.Anonymous.Buffer = D3D11_BUFFER_UAV {
                FirstElement: 0,
                NumElements: tile_count,
                Flags: 0,
            };
            let mut view = None;
            device.CreateUnorderedAccessView(&output, Some(&desc), Some(&mut view))?;
            view.ok_or("D3D11 returned no comparison result view")?
        };
        let staging = create_buffer(
            device,
            &D3D11_BUFFER_DESC {
                ByteWidth: byte_width,
                Usage: D3D11_USAGE_STAGING,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                ..Default::default()
            },
            None,
        )?;

        Ok(Self {
            device: device.clone(),
            context: context.clone(),
            width,
            height,
            tile_count,
            shader,
            geometry,
            candidates,
            candidate_view,
            output,
            output_view,
            staging,
            seed_done: crate::consumer::query(device, D3D11_QUERY_EVENT)?,
        })
    }

    /// Upload a frame outside the timed path. The SRV holds the texture alive.
    pub fn upload(&self, pixels: &[u32]) -> Result<GpuFrame> {
        if pixels.len() != (self.width * self.height) as usize {
            return Err("BGRA8 pixel count does not match comparison dimensions".into());
        }
        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_IMMUTABLE,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            ..Default::default()
        };
        let data = D3D11_SUBRESOURCE_DATA {
            pSysMem: pixels.as_ptr().cast(),
            SysMemPitch: self.width * size_of::<u32>() as u32,
            ..Default::default()
        };
        let (texture, view) = unsafe {
            let mut texture = None;
            self.device
                .CreateTexture2D(&desc, Some(&data), Some(&mut texture))?;
            let texture = texture.ok_or("D3D11 returned no comparison texture")?;
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))?;
            (
                texture,
                view.ok_or("D3D11 returned no comparison texture view")?,
            )
        };
        Ok(GpuFrame { view, texture })
    }

    pub(crate) fn history(&self) -> Result<GpuFrame> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: self.width,
            Height: self.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R32_UINT,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            ..Default::default()
        };
        unsafe {
            let mut texture = None;
            self.device
                .CreateTexture2D(&desc, None, Some(&mut texture))?;
            let texture = texture.ok_or("D3D11 returned no history texture")?;
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))?;
            Ok(GpuFrame {
                view: view.ok_or("D3D11 returned no history view")?,
                texture,
            })
        }
    }

    pub(crate) fn retain(&self, destination: &GpuFrame, source: &GpuFrame) {
        unsafe {
            self.context
                .CopyResource(&destination.texture, &source.texture)
        };
    }

    pub(crate) fn seed_history(&self, destination: &GpuFrame, source: &GpuFrame) -> Result<()> {
        self.retain(destination, source);
        unsafe {
            self.context.End(&self.seed_done);
            self.context.Flush();
        }
        let complete: windows::core::BOOL =
            crate::consumer::query_data(&self.context, &self.seed_done)?;
        if !complete.as_bool() {
            return Err("D3D11 could not confirm that the history seed finished".into());
        }
        Ok(())
    }

    pub(crate) fn staging_buffer(&self) -> Result<ID3D11Buffer> {
        create_buffer(
            &self.device,
            &D3D11_BUFFER_DESC {
                ByteWidth: self.tile_count * size_of::<u32>() as u32,
                Usage: D3D11_USAGE_STAGING,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                ..Default::default()
            },
            None,
        )
    }

    /// Compare sorted, unique tile indexes and return the changed subset in order.
    pub fn compare(
        &self,
        previous: &GpuFrame,
        current: &GpuFrame,
        candidates: &[u32],
        changed: &mut Vec<u32>,
    ) -> Result<()> {
        changed.clear();
        self.submit(previous, current, candidates)?;
        if !candidates.is_empty() {
            self.enqueue_readback(&self.staging, candidates.len());
            self.read_flags(&self.staging, candidates, changed, false)?;
        }
        Ok(())
    }

    pub(crate) fn submit(
        &self,
        previous: &GpuFrame,
        current: &GpuFrame,
        candidates: &[u32],
    ) -> Result<()> {
        let Some(&last_tile) = candidates.last() else {
            return Ok(());
        };
        if candidates.len() > self.tile_count as usize || last_tile >= self.tile_count {
            return Err("comparison candidates are outside the tile grid".into());
        }
        debug_assert!(candidates.windows(2).all(|tiles| tiles[0] < tiles[1]));
        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context.Map(
                &self.candidates,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            )?;
            ptr::copy_nonoverlapping(candidates.as_ptr(), mapped.pData.cast(), candidates.len());
            self.context.Unmap(&self.candidates, 0);

            self.context.CSSetShader(&self.shader, None);
            self.context.CSSetShaderResources(
                0,
                Some(&[
                    Some(previous.view.clone()),
                    Some(current.view.clone()),
                    Some(self.candidate_view.clone()),
                ]),
            );
            self.context
                .CSSetConstantBuffers(0, Some(&[Some(self.geometry.clone())]));
            self.context.CSSetUnorderedAccessViews(
                0,
                1,
                Some(&Some(self.output_view.clone())),
                None,
            );
            self.context.Dispatch(candidates.len() as u32, 1, 1);
            self.context
                .CSSetShaderResources(0, Some(&[None, None, None]));
            self.context
                .CSSetUnorderedAccessViews(0, 1, Some(&None), None);
        }
        Ok(())
    }

    pub(crate) fn enqueue_readback(&self, staging: &ID3D11Buffer, count: usize) {
        let region = D3D11_BOX {
            right: (count * size_of::<u32>()) as u32,
            bottom: 1,
            back: 1,
            ..Default::default()
        };
        unsafe {
            self.context
                .CopySubresourceRegion(staging, 0, 0, 0, 0, &self.output, 0, Some(&region));
        }
    }

    pub(crate) fn clear_output(&self) {
        unsafe {
            self.context
                .ClearUnorderedAccessViewUint(&self.output_view, &[0; 4])
        };
    }

    pub(crate) fn read_flags(
        &self,
        staging: &ID3D11Buffer,
        candidates: &[u32],
        changed: &mut Vec<u32>,
        nonblocking: bool,
    ) -> Result<bool> {
        changed.clear();
        if candidates.is_empty() {
            return Ok(true);
        }
        changed.reserve(candidates.len());
        let flags = if nonblocking {
            D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32
        } else {
            0
        };
        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            match self
                .context
                .Map(staging, 0, D3D11_MAP_READ, flags, Some(&mut mapped))
            {
                Err(error)
                    if nonblocking
                        && error.code()
                            == windows::Win32::Graphics::Dxgi::DXGI_ERROR_WAS_STILL_DRAWING =>
                {
                    return Ok(false)
                }
                result => result?,
            }
            let flags = slice::from_raw_parts(mapped.pData.cast::<u32>(), candidates.len());
            changed.extend(
                candidates
                    .iter()
                    .zip(flags)
                    .filter_map(|(&tile, &flag)| (flag != 0).then_some(tile)),
            );
            self.context.Unmap(staging, 0);
        }
        Ok(true)
    }
}

pub(crate) fn create_buffer(
    device: &ID3D11Device,
    desc: &D3D11_BUFFER_DESC,
    data: Option<&D3D11_SUBRESOURCE_DATA>,
) -> Result<ID3D11Buffer> {
    let mut buffer = None;
    unsafe { device.CreateBuffer(desc, data.map(ptr::from_ref), Some(&mut buffer))? };
    buffer.ok_or_else(|| "D3D11 returned no comparison buffer".into())
}

fn create_shader(device: &ID3D11Device) -> Result<ID3D11ComputeShader> {
    compile_shader(device, include_bytes!("compare.hlsl"), c"Compare")
}

pub(crate) fn compile_shader(
    device: &ID3D11Device,
    source: &[u8],
    entry: &std::ffi::CStr,
) -> Result<ID3D11ComputeShader> {
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
            "comparison shader compilation failed ({error}): {}",
            String::from_utf8_lossy(bytes)
        )
        .into());
    }
    result?;
    let blob = bytecode.ok_or("D3D11 shader compiler returned no comparison bytecode")?;
    let mut shader = None;
    unsafe {
        let bytes = slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize());
        device.CreateComputeShader(bytes, None, Some(&mut shader))?;
    }
    shader.ok_or_else(|| "D3D11 returned no comparison shader".into())
}
