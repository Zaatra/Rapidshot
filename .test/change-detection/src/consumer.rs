use std::time::{Duration, Instant};
use std::{mem::size_of, ptr, slice, thread};

use windows::core::Interface;
use windows::Win32::Foundation::S_FALSE;
use windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_BUFFER;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_TYPELESS, DXGI_FORMAT_R32_UINT};

use crate::{
    gpu::{compile_shader, create_buffer},
    AnyResult, Extent, Gpu, GpuFrame, TILE_SIZE,
};

pub struct Consumer {
    context: ID3D11DeviceContext,
    compact: ID3D11ComputeShader,
    score: ID3D11ComputeShader,
    params: ID3D11Buffer,
    flags: ID3D11ShaderResourceView,
    work: ID3D11Buffer,
    work_srv: ID3D11ShaderResourceView,
    work_uav: ID3D11UnorderedAccessView,
    args: ID3D11Buffer,
    args_uav: ID3D11UnorderedAccessView,
    scores: ID3D11Buffer,
    scores_uav: ID3D11UnorderedAccessView,
    staging: ID3D11Buffer,
    extent: Extent,
}

fn srv(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    words: u32,
) -> AnyResult<ID3D11ShaderResourceView> {
    let mut desc = D3D11_SHADER_RESOURCE_VIEW_DESC {
        Format: DXGI_FORMAT_R32_UINT,
        ViewDimension: D3D_SRV_DIMENSION_BUFFER,
        ..Default::default()
    };
    desc.Anonymous.Buffer = D3D11_BUFFER_SRV {
        Anonymous1: D3D11_BUFFER_SRV_0 { FirstElement: 0 },
        Anonymous2: D3D11_BUFFER_SRV_1 { NumElements: words },
    };
    let mut view = None;
    unsafe { device.CreateShaderResourceView(buffer, Some(&desc), Some(&mut view))? };
    view.ok_or_else(|| "missing GPU buffer shader view".into())
}

fn uav(
    device: &ID3D11Device,
    buffer: &ID3D11Buffer,
    words: u32,
    raw: bool,
) -> AnyResult<ID3D11UnorderedAccessView> {
    let mut desc = D3D11_UNORDERED_ACCESS_VIEW_DESC {
        Format: if raw {
            DXGI_FORMAT_R32_TYPELESS
        } else {
            DXGI_FORMAT_R32_UINT
        },
        ViewDimension: D3D11_UAV_DIMENSION_BUFFER,
        ..Default::default()
    };
    desc.Anonymous.Buffer = D3D11_BUFFER_UAV {
        FirstElement: 0,
        NumElements: words,
        Flags: if raw {
            D3D11_BUFFER_UAV_FLAG_RAW.0 as u32
        } else {
            0
        },
    };
    let mut view = None;
    unsafe { device.CreateUnorderedAccessView(buffer, Some(&desc), Some(&mut view))? };
    view.ok_or_else(|| "missing GPU buffer writable view".into())
}

fn buffer(
    device: &ID3D11Device,
    words: u32,
    binds: D3D11_BIND_FLAG,
    misc: u32,
) -> AnyResult<ID3D11Buffer> {
    create_buffer(
        device,
        &D3D11_BUFFER_DESC {
            ByteWidth: words * size_of::<u32>() as u32,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: binds.0 as u32,
            MiscFlags: misc,
            ..Default::default()
        },
        None,
    )
}

impl Consumer {
    pub fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        gpu: &Gpu,
        extent: Extent,
    ) -> AnyResult<Self> {
        let source = include_bytes!("consumer.hlsl");
        let count = extent.tile_count();
        let work = buffer(
            device,
            count,
            D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_UNORDERED_ACCESS,
            0,
        )?;
        let args = buffer(
            device,
            3,
            D3D11_BIND_UNORDERED_ACCESS,
            (D3D11_RESOURCE_MISC_DRAWINDIRECT_ARGS | D3D11_RESOURCE_MISC_BUFFER_ALLOW_RAW_VIEWS).0
                as u32,
        )?;
        let scores = buffer(device, count, D3D11_BIND_UNORDERED_ACCESS, 0)?;
        Ok(Self {
            context: context.clone(),
            compact: compile_shader(device, source, c"Compact")?,
            score: compile_shader(device, source, c"Score")?,
            params: buffer(device, 8, D3D11_BIND_CONSTANT_BUFFER, 0)?,
            flags: srv(device, &gpu.output, count)?,
            work_srv: srv(device, &work, count)?,
            work_uav: uav(device, &work, count, false)?,
            args_uav: uav(device, &args, 3, true)?,
            scores_uav: uav(device, &scores, count, false)?,
            work,
            args,
            scores,
            staging: gpu.staging_buffer()?,
            extent,
        })
    }

    fn set_parameters(&self, count: usize) {
        let words = [
            self.extent.width,
            self.extent.height,
            self.extent.tile_columns(),
            TILE_SIZE,
            count as u32,
            0,
            0,
            0,
        ];
        unsafe {
            self.context
                .UpdateSubresource(&self.params, 0, None, words.as_ptr().cast(), 0, 0);
            self.context
                .CSSetConstantBuffers(0, Some(&[Some(self.params.clone())]));
        }
    }

    fn score(&self, current: &GpuFrame, direct_groups: Option<u32>) {
        unsafe {
            self.context.CSSetShader(&self.score, None);
            self.context.CSSetShaderResources(
                2,
                Some(&[Some(current.view.clone()), Some(self.work_srv.clone())]),
            );
            self.context.CSSetUnorderedAccessViews(
                2,
                1,
                Some(&Some(self.scores_uav.clone())),
                None,
            );
            match direct_groups {
                Some(groups) => self.context.Dispatch(groups, 1, 1),
                None => self.context.DispatchIndirect(&self.args, 0),
            }
            self.context.CSSetShaderResources(2, Some(&[None, None]));
            self.context
                .CSSetUnorderedAccessViews(2, 1, Some(&None), None);
        }
    }

    pub fn indirect(&self, gpu: &Gpu, current: &GpuFrame, count: usize) {
        self.set_parameters(count);
        unsafe {
            self.context.CSSetShader(&self.compact, None);
            self.context.CSSetShaderResources(
                0,
                Some(&[Some(gpu.candidate_view.clone()), Some(self.flags.clone())]),
            );
            self.context.CSSetUnorderedAccessViews(
                0,
                2,
                Some([Some(self.work_uav.clone()), Some(self.args_uav.clone())].as_ptr()),
                None,
            );
            self.context.Dispatch(1, 1, 1);
            self.context.CSSetShaderResources(0, Some(&[None, None]));
            self.context
                .CSSetUnorderedAccessViews(0, 2, Some([None, None].as_ptr()), None);
        }
        self.score(current, None);
    }

    pub fn direct(&self, current: &GpuFrame, tiles: &[u32]) {
        if tiles.is_empty() {
            return;
        }
        self.set_parameters(tiles.len());
        let region = word_region(tiles.len());
        unsafe {
            self.context.UpdateSubresource(
                &self.work,
                0,
                Some(&region),
                tiles.as_ptr().cast(),
                0,
                0,
            )
        };
        self.score(current, Some(tiles.len() as u32));
    }

    pub fn clear_scores(&self) {
        unsafe {
            self.context
                .ClearUnorderedAccessViewUint(&self.scores_uav, &[u32::MAX; 4])
        };
    }

    pub fn verify(&self, expected: &[u32], pixels: &[u32], indirect: bool) -> AnyResult<()> {
        if indirect {
            let groups = self.read_words(&self.args, 3)?;
            if groups != [expected.len() as u32, 1, 1] {
                return Err(format!(
                    "unexpected indirect arguments {groups:?} for {} tiles",
                    expected.len()
                )
                .into());
            }
        }
        if !expected.is_empty() && self.read_words(&self.work, expected.len())? != expected {
            return Err("GPU consumer work list differs from the CPU oracle".into());
        }
        let scores = self.read_words(&self.scores, self.extent.tile_count() as usize)?;
        for (index, actual) in scores.into_iter().enumerate() {
            let tile = index as u32;
            let wanted = if expected.binary_search(&tile).is_ok() {
                self.score_cpu(tile, pixels)
            } else {
                u32::MAX
            };
            if actual != wanted {
                return Err(format!(
                    "GPU consumer score differs on tile {tile}: {actual} vs {wanted}"
                )
                .into());
            }
        }
        Ok(())
    }

    fn score_cpu(&self, tile: u32, pixels: &[u32]) -> u32 {
        let columns = self.extent.tile_columns();
        let x0 = tile % columns * TILE_SIZE;
        let y0 = tile / columns * TILE_SIZE;
        let mut sum = 0u32;
        for y in y0..(y0 + TILE_SIZE).min(self.extent.height) {
            for x in x0..(x0 + TILE_SIZE).min(self.extent.width) {
                let p = pixels[(y * self.extent.width + x) as usize];
                sum = sum.wrapping_add((p & 255) * 3 + ((p >> 8) & 255) * 6 + ((p >> 16) & 255));
            }
        }
        sum
    }

    fn read_words(&self, buffer: &ID3D11Buffer, count: usize) -> AnyResult<Vec<u32>> {
        unsafe {
            self.context.CopySubresourceRegion(
                &self.staging,
                0,
                0,
                0,
                0,
                buffer,
                0,
                Some(&word_region(count)),
            );
            let mut map = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut map))?;
            let result = slice::from_raw_parts(map.pData.cast::<u32>(), count).to_vec();
            self.context.Unmap(&self.staging, 0);
            Ok(result)
        }
    }
}

fn word_region(words: usize) -> D3D11_BOX {
    D3D11_BOX {
        right: (words * size_of::<u32>()) as u32,
        bottom: 1,
        back: 1,
        ..Default::default()
    }
}

pub(crate) fn query(device: &ID3D11Device, kind: D3D11_QUERY) -> AnyResult<ID3D11Query> {
    let mut value = None;
    let desc = D3D11_QUERY_DESC {
        Query: kind,
        MiscFlags: 0,
    };
    unsafe { device.CreateQuery(&desc, Some(&mut value))? };
    value.ok_or_else(|| "missing D3D11 timing query".into())
}

pub(crate) fn query_data<T: Default>(
    context: &ID3D11DeviceContext,
    query: &ID3D11Query,
) -> AnyResult<T> {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut value = T::default();
    loop {
        let status = unsafe {
            (context.vtable().GetData)(
                context.as_raw(),
                query.as_raw(),
                ptr::from_mut(&mut value).cast(),
                size_of::<T>() as u32,
                D3D11_ASYNC_GETDATA_DONOTFLUSH.0 as u32,
            )
        };
        if status != S_FALSE {
            status.ok()?;
            return Ok(value);
        }
        if Instant::now() >= deadline {
            return Err("D3D11 GPU completion deadline (2 s) exceeded".into());
        }
        thread::yield_now();
    }
}
