use std::ffi::c_void;
use std::io::{self, Write};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::{Duration, Instant};
use std::{ptr, slice};

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIFactory1};
use windows::Win32::System::Threading::GetCurrentThreadId;

use crate::cpu_default_compare::compare_tiles;
use crate::{AnyResult, Gpu, GpuFrame};

#[repr(C)]
pub struct ReplayTiming {
    pub delivery_us: f64,
    pub decision_us: f64,
    pub wall_us: f64,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    DirtyOnly,
    CpuTiled,
    CpuFull,
    GpuTiled,
    GpuFull,
}

impl Mode {
    fn parse(mode: u32) -> AnyResult<Self> {
        match mode {
            0 => Ok(Self::DirtyOnly),
            1 => Ok(Self::CpuTiled),
            2 => Ok(Self::CpuFull),
            3 => Ok(Self::GpuTiled),
            4 => Ok(Self::GpuFull),
            _ => Err(format!("unknown replay mode {mode}").into()),
        }
    }

    fn uses_gpu(self) -> bool {
        matches!(self, Self::GpuTiled | Self::GpuFull)
    }
}

#[derive(Clone, Copy)]
struct Geometry {
    width: usize,
    height: usize,
    tile_size: usize,
    row_bytes: usize,
    frame_bytes: usize,
    tile_count: usize,
}

impl Geometry {
    fn new(width: u32, height: u32, tile_size: u32) -> AnyResult<Self> {
        if width == 0
            || height == 0
            || width > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || height > D3D11_REQ_TEXTURE2D_U_OR_V_DIMENSION
            || tile_size == 0
        {
            return Err("invalid replay frame or tile dimensions".into());
        }
        let (width, height, tile_size) = (width as usize, height as usize, tile_size as usize);
        let tile_count = width.div_ceil(tile_size) * height.div_ceil(tile_size);
        if tile_count > D3D11_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION as usize {
            return Err("too many tiles for the GPU replay".into());
        }
        let row_bytes = width * size_of::<u32>();
        Ok(Self {
            width,
            height,
            tile_size,
            row_bytes,
            frame_bytes: row_bytes * height,
            tile_count,
        })
    }

    fn validate_candidates(self, candidates: &[u32]) -> AnyResult<()> {
        if candidates.len() > self.tile_count
            || candidates
                .last()
                .is_some_and(|&tile| tile as usize >= self.tile_count)
            || candidates.windows(2).any(|tiles| tiles[0] >= tiles[1])
        {
            return Err("replay candidates must be in range, sorted, and unique".into());
        }
        Ok(())
    }
}

struct Readback {
    context: ID3D11DeviceContext,
    staging: ID3D11Texture2D,
}

impl Readback {
    fn new(
        device: &ID3D11Device,
        context: &ID3D11DeviceContext,
        geometry: Geometry,
    ) -> AnyResult<Self> {
        let description = D3D11_TEXTURE2D_DESC {
            Width: geometry.width as u32,
            Height: geometry.height as u32,
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
        unsafe { device.CreateTexture2D(&description, None, Some(&mut staging))? };
        Ok(Self {
            context: context.clone(),
            staging: staging.ok_or("D3D11 returned no replay delivery texture")?,
        })
    }

    fn deliver(&self, source: &GpuFrame, geometry: Geometry, output: &mut [u8]) -> AnyResult<()> {
        let mut data = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe {
            self.context.CopyResource(&self.staging, &source.texture);
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut data))?;
        }
        let mapping = Mapping { owner: self, data };
        let pitch = mapping.data.RowPitch as usize;
        if pitch < geometry.row_bytes || mapping.data.pData.is_null() {
            return Err("invalid replay delivery mapping".into());
        }
        for (y, row) in output.chunks_exact_mut(geometry.row_bytes).enumerate() {
            let source = unsafe { mapping.data.pData.cast::<u8>().add(y * pitch) };
            row.copy_from_slice(unsafe { slice::from_raw_parts(source, geometry.row_bytes) });
        }
        Ok(())
    }
}

struct Mapping<'a> {
    owner: &'a Readback,
    data: D3D11_MAPPED_SUBRESOURCE,
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        unsafe { self.owner.context.Unmap(&self.owner.staging, 0) };
    }
}

struct Frame {
    gpu: GpuFrame,
    pixels: Vec<u32>,
}

struct Replay {
    owner: u32,
    geometry: Geometry,
    gpu: Gpu,
    gpu_previous: GpuFrame,
    gpu_flags: ID3D11Buffer,
    readback: Readback,
    frames: Vec<Frame>,
    all_tiles: Vec<u32>,
    accepted_pixels: Vec<u8>,
    candidate_pixels: Vec<u8>,
    changed: Vec<u32>,
    mode: Option<Mode>,
    ready: bool,
}

impl Replay {
    fn new(geometry: Geometry, raw: &[u8], adapter_index: u32) -> AnyResult<Self> {
        let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
        let index = if adapter_index == u32::MAX {
            0
        } else {
            adapter_index
        };
        let adapter = unsafe { factory.EnumAdapters1(index)? };
        let mut device = None;
        let mut context = None;
        unsafe {
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )?;
        }
        let device = device.ok_or("D3D11 returned no replay device")?;
        let context = context.ok_or("D3D11 returned no replay context")?;
        let gpu = Gpu::new(
            &device,
            &context,
            geometry.width as u32,
            geometry.height as u32,
            geometry.tile_size as u32,
        )?;
        let gpu_previous = gpu.history()?;
        let gpu_flags = gpu.staging_buffer()?;
        let readback = Readback::new(&device, &context, geometry)?;
        let mut frames = Vec::with_capacity(raw.len() / geometry.frame_bytes);
        for frame in raw.chunks_exact(geometry.frame_bytes) {
            let mut pixels = vec![0_u32; geometry.width * geometry.height];
            // The FFI input can be byte aligned. The retained CPU copy is also
            // the aligned upload input and supplies untimed history resets.
            unsafe {
                ptr::copy_nonoverlapping(frame.as_ptr(), pixels.as_mut_ptr().cast(), frame.len());
            }
            frames.push(Frame {
                gpu: gpu.upload(&pixels)?,
                pixels,
            });
        }
        Ok(Self {
            owner: unsafe { GetCurrentThreadId() },
            geometry,
            gpu,
            gpu_previous,
            gpu_flags,
            readback,
            frames,
            all_tiles: (0..geometry.tile_count as u32).collect(),
            accepted_pixels: vec![0; geometry.frame_bytes],
            candidate_pixels: vec![0; geometry.frame_bytes],
            changed: Vec::with_capacity(geometry.tile_count),
            mode: None,
            ready: false,
        })
    }

    fn reset(&mut self, previous: usize) -> AnyResult<()> {
        self.ready = false;
        self.mode = None;
        let frame = self
            .frames
            .get(previous)
            .ok_or("invalid previous replay frame")?;
        unsafe {
            ptr::copy_nonoverlapping(
                frame.pixels.as_ptr().cast::<u8>(),
                self.accepted_pixels.as_mut_ptr(),
                self.geometry.frame_bytes,
            );
        }
        self.gpu.seed_history(&self.gpu_previous, &frame.gpu)?;
        self.ready = true;
        Ok(())
    }

    fn decide(&mut self, mode: Mode, index: usize, candidates: &[u32]) -> AnyResult<ReplayTiming> {
        if !self.ready || self.mode.is_some_and(|prior| prior != mode) {
            return Err("reset the replay before its first decision or a mode change".into());
        }
        self.geometry.validate_candidates(candidates)?;
        let frame = &self
            .frames
            .get(index)
            .ok_or("invalid current replay frame")?
            .gpu;
        let candidates = match mode {
            Mode::CpuFull | Mode::GpuFull => &self.all_tiles[..],
            _ => candidates,
        };
        self.ready = false;
        let wall_start = Instant::now();
        let uses_gpu = mode.uses_gpu();
        if uses_gpu && !candidates.is_empty() {
            self.gpu.submit(&self.gpu_previous, frame, candidates)?;
            self.gpu.retain(&self.gpu_previous, frame);
            self.gpu.enqueue_readback(&self.gpu_flags, candidates.len());
        }
        let delivery_start = Instant::now();
        self.readback
            .deliver(frame, self.geometry, &mut self.candidate_pixels)?;
        let delivery_end = Instant::now();
        self.changed.clear();
        match mode {
            Mode::DirtyOnly => self.changed.extend_from_slice(candidates),
            Mode::CpuTiled | Mode::CpuFull => compare_tiles(
                &self.accepted_pixels,
                &self.candidate_pixels,
                self.geometry.width,
                self.geometry.height,
                self.geometry.row_bytes,
                self.geometry.row_bytes,
                self.geometry.tile_size,
                candidates,
                &mut self.changed,
            ),
            Mode::GpuTiled | Mode::GpuFull => {
                self.gpu
                    .read_flags(&self.gpu_flags, candidates, &mut self.changed, false)?;
            }
        }
        if !self.changed.is_empty() {
            std::mem::swap(&mut self.accepted_pixels, &mut self.candidate_pixels);
        }
        let completed = Instant::now();
        self.ready = true;
        self.mode = Some(mode);
        Ok(ReplayTiming {
            delivery_us: micros(delivery_end - delivery_start),
            decision_us: micros((delivery_start - wall_start) + (completed - delivery_end)),
            wall_us: micros(completed - wall_start),
        })
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn ffi<T: Copy>(name: &str, error_value: T, function: impl FnOnce() -> AnyResult<T>) -> T {
    match catch_unwind(AssertUnwindSafe(function)) {
        Ok(Ok(result)) => result,
        Ok(Err(error)) => {
            let _ = writeln!(io::stderr().lock(), "{name}: {error}");
            error_value
        }
        Err(_) => {
            let _ = writeln!(io::stderr().lock(), "{name}: unexpected replay panic");
            error_value
        }
    }
}

unsafe fn owner(handle: *const c_void) -> AnyResult<*mut Replay> {
    let replay = handle.cast::<Replay>().cast_mut();
    if replay.is_null() {
        return Err("replay handle is null".into());
    }
    if unsafe { ptr::addr_of!((*replay).owner).read() != GetCurrentThreadId() } {
        return Err("replay handle belongs to another thread".into());
    }
    Ok(replay)
}

// Caller-owned pointers are valid and nonaliasing for the duration of each call.
// The handle is used on the thread that created it.
#[no_mangle]
pub unsafe extern "C" fn rs_replay_init(
    width: u32,
    height: u32,
    tile_size: u32,
    frames: *const u8,
    frame_count: usize,
    adapter_index: u32,
) -> *mut c_void {
    ffi("rs_replay_init", ptr::null_mut(), || {
        let geometry = Geometry::new(width, height, tile_size)?;
        let bytes = geometry
            .frame_bytes
            .checked_mul(frame_count)
            .filter(|&bytes| bytes != 0 && bytes <= isize::MAX as usize)
            .ok_or("invalid replay input length")?;
        if frames.is_null() {
            return Err("replay input frames are null".into());
        }
        let source = unsafe { slice::from_raw_parts(frames, bytes) };
        Ok(Box::into_raw(Box::new(Replay::new(geometry, source, adapter_index)?)).cast())
    })
}

#[no_mangle]
pub unsafe extern "C" fn rs_replay_reset(handle: *mut c_void, previous: usize) -> i32 {
    ffi("rs_replay_reset", -1, || {
        let replay = unsafe { &mut *owner(handle)? };
        replay.reset(previous)?;
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rs_replay_decide(
    handle: *mut c_void,
    mode: u32,
    current: usize,
    candidates: *const u32,
    candidate_count: usize,
    output_tiles: *mut u32,
    output_count: *mut usize,
    timings: *mut ReplayTiming,
) -> i32 {
    ffi("rs_replay_decide", -1, || {
        let replay = unsafe { &mut *owner(handle)? };
        if output_tiles.is_null() || output_count.is_null() || timings.is_null() {
            return Err("replay output buffers are null".into());
        }
        if candidate_count > replay.geometry.tile_count
            || (candidate_count != 0 && candidates.is_null())
        {
            return Err("invalid replay candidates buffer".into());
        }
        let candidates = if candidate_count == 0 {
            &[]
        } else {
            unsafe { slice::from_raw_parts(candidates, candidate_count) }
        };
        let measurement = replay.decide(Mode::parse(mode)?, current, candidates)?;
        unsafe {
            ptr::copy_nonoverlapping(replay.changed.as_ptr(), output_tiles, replay.changed.len());
            output_count.write(replay.changed.len());
            timings.write(measurement);
        }
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rs_replay_pixels(
    handle: *const c_void,
    output: *mut u8,
    bytes: usize,
) -> i32 {
    ffi("rs_replay_pixels", -1, || {
        let replay = unsafe { &*owner(handle)? };
        if output.is_null() || bytes != replay.geometry.frame_bytes || !replay.ready {
            return Err("replay pixel output buffer or history is invalid".into());
        }
        unsafe { ptr::copy_nonoverlapping(replay.accepted_pixels.as_ptr(), output, bytes) };
        Ok(0)
    })
}

#[no_mangle]
pub unsafe extern "C" fn rs_replay_close(handle: *mut c_void) {
    if !handle.is_null() {
        ffi("rs_replay_close", (), || {
            unsafe { drop(Box::from_raw(owner(handle)?)) };
            Ok(())
        });
    }
}
