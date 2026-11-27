//! Benchmark-only D3D11 partial readback and owned tiled history.
#[path = "../gpu.rs"]
#[allow(dead_code)]
mod gpu;
#[path = "../idea_gpu_live.rs"]
mod live;

use serde::Serialize;
use std::collections::BTreeSet;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use windows::core::Interface;
use windows::Win32::Foundation::{HANDLE, HMODULE, S_FALSE};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_R32_UINT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
    DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter1, IDXGIAdapter3,
    IDXGIFactory1, IDXGIOutput1,
};
use windows::Win32::System::Threading::GetCurrentProcess;

#[link(name = "kernel32")]
extern "system" {
    fn QueryProcessCycleTime(process: HANDLE, value: *mut u64) -> i32;
}

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
const WIDTH: usize = 1920;
const HEIGHT: usize = 1080;
const TILE: usize = 128;
const COUNT: usize = WIDTH.div_ceil(TILE) * HEIGHT.div_ceil(TILE);
const LOAD: usize = WIDTH * HEIGHT * 4;

// The exact-comparison shader's existing GPU module asks for these private query helpers.
mod consumer {
    use super::*;

    pub(crate) fn query(device: &ID3D11Device, kind: D3D11_QUERY) -> Result<ID3D11Query> {
        let desc = D3D11_QUERY_DESC { Query: kind, MiscFlags: 0 };
        let mut result = None;
        unsafe { device.CreateQuery(&desc, Some(&mut result))? };
        result.ok_or_else(|| "D3D11 event query was not created".into())
    }

    pub(crate) fn query_data<T: Default>(context: &ID3D11DeviceContext, query: &ID3D11Query) -> Result<T> {
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut value = T::default();
        loop {
            let code = unsafe {
                (Interface::vtable(context).GetData)(
                    Interface::as_raw(context),
                    Interface::as_raw(query),
                    (&mut value as *mut T).cast(),
                    std::mem::size_of::<T>() as u32,
                    0,
                )
            };
            if code != S_FALSE {
                code.ok()?;
                return Ok(value);
            }
            if Instant::now() >= deadline {
                return Err("D3D11 seed query timed out".into());
            }
            std::thread::yield_now();
        }
    }
}

fn process_cycles() -> Option<u64> {
    let mut cycles = 0;
    (unsafe { QueryProcessCycleTime(GetCurrentProcess(), &mut cycles) } != 0).then_some(cycles)
}

fn elapsed_us(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000_000.0
}

fn adapter(index: u32) -> Result<IDXGIAdapter1> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    Ok(unsafe { factory.EnumAdapters1(index)? })
}

fn device(adapter: &IDXGIAdapter1) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;
    unsafe {
        D3D11CreateDevice(
            adapter, D3D_DRIVER_TYPE_UNKNOWN, HMODULE::default(), D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            None, D3D11_SDK_VERSION, Some(&mut device), None, Some(&mut context),
        )?;
    }
    Ok((device.ok_or("D3D11 did not create a device")?, context.ok_or("D3D11 did not create a context")?))
}

#[derive(Clone, Copy, Serialize)]
struct Rect {
    left: usize,
    top: usize,
    right: usize,
    bottom: usize,
}

impl Rect {
    fn tile(tile: u32) -> Self {
        let columns = WIDTH.div_ceil(TILE);
        let left = tile as usize % columns * TILE;
        let top = tile as usize / columns * TILE;
        Self { left, top, right: (left + TILE).min(WIDTH), bottom: (top + TILE).min(HEIGHT) }
    }

    fn full() -> Self {
        Self { left: 0, top: 0, right: WIDTH, bottom: HEIGHT }
    }

    fn bytes(self) -> usize {
        (self.right - self.left) * (self.bottom - self.top) * 4
    }

    fn raw(self) -> D3D11_BOX {
        D3D11_BOX {
            left: self.left as u32, top: self.top as u32, front: 0, right: self.right as u32,
            bottom: self.bottom as u32, back: 1,
        }
    }
}

fn stage(device: &ID3D11Device, width: usize, height: usize, format: DXGI_FORMAT) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width as u32, Height: height as u32, MipLevels: 1, ArraySize: 1, Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 }, Usage: D3D11_USAGE_STAGING,
        CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32, ..Default::default()
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    texture.ok_or_else(|| "D3D11 staging allocation returned no texture".into())
}

fn copy_regions(
    context: &ID3D11DeviceContext, destination: &ID3D11Texture2D, source: &ID3D11Texture2D, tiles: &[u32],
) {
    for &tile in tiles {
        let rect = Rect::tile(tile);
        unsafe {
            context.CopySubresourceRegion(
                destination, 0, rect.left as u32, rect.top as u32, 0, source, 0, Some(&rect.raw()),
            )
        };
    }
}

struct Mapping<'a> {
    context: &'a ID3D11DeviceContext,
    texture: &'a ID3D11Texture2D,
    data: D3D11_MAPPED_SUBRESOURCE,
}

impl<'a> Mapping<'a> {
    fn new(context: &'a ID3D11DeviceContext, texture: &'a ID3D11Texture2D, width: usize) -> Result<Self> {
        let mut data = D3D11_MAPPED_SUBRESOURCE::default();
        unsafe { context.Map(texture, 0, D3D11_MAP_READ, 0, Some(&mut data))? };
        let mapped = Self { context, texture, data };
        if mapped.data.pData.is_null() || (mapped.data.RowPitch as usize) < width * 4 {
            return Err("D3D11 returned an invalid row pitch".into());
        }
        Ok(mapped)
    }

    fn row(&self, y: usize, left: usize, right: usize) -> &[u32] {
        unsafe {
            let row = self.data.pData.cast::<u8>().add(y * self.data.RowPitch as usize).cast::<u32>();
            std::slice::from_raw_parts(row.add(left), right - left)
        }
    }
}

impl Drop for Mapping<'_> {
    fn drop(&mut self) {
        unsafe { self.context.Unmap(self.texture, 0) };
    }
}

#[derive(Clone)]
struct History {
    pages: Vec<Arc<Vec<u32>>>,
}

impl History {
    fn seed(input: &[u32]) -> Self {
        let pages = (0..COUNT as u32).map(|id| {
            let rect = Rect::tile(id);
            let mut page = Vec::with_capacity(rect.bytes() / 4);
            for y in rect.top..rect.bottom {
                page.extend_from_slice(&input[y * WIDTH + rect.left..y * WIDTH + rect.right]);
            }
            Arc::new(page)
        }).collect();
        Self { pages }
    }

    fn patch<F>(&mut self, tiles: &[u32], exact_known: bool, mut source: F) -> Vec<u32>
    where
        F: FnMut(usize, usize, usize, &mut dyn FnMut(&[u32])),
    {
        let mut changed = Vec::with_capacity(tiles.len());
        for &id in tiles {
            let rect = Rect::tile(id);
            let span = rect.right - rect.left;
            let previous = &self.pages[id as usize];
            let differs = exact_known || (rect.top..rect.bottom).any(|y| {
                let offset = (y - rect.top) * span;
                let mut differs = false;
                source(y, rect.left, rect.right, &mut |row| {
                    differs = row != &previous[offset..offset + span];
                });
                differs
            });
            if differs {
                let mut bytes = Vec::with_capacity(rect.bytes() / 4);
                for y in rect.top..rect.bottom {
                    source(y, rect.left, rect.right, &mut |row| bytes.extend_from_slice(row));
                }
                self.pages[id as usize] = Arc::new(bytes);
                changed.push(id);
            }
        }
        changed
    }

    fn materialize(&self, rect: Rect, output: &mut Vec<u32>) {
        output.clear();
        let columns = WIDTH.div_ceil(TILE);
        for y in rect.top..rect.bottom {
            let mut x = rect.left;
            while x < rect.right {
                let tile = y / TILE * columns + x / TILE;
                let bounds = Rect::tile(tile as u32);
                let end = rect.right.min(bounds.right);
                let start = (y - bounds.top) * (bounds.right - bounds.left) + x - bounds.left;
                output.extend_from_slice(&self.pages[tile][start..start + end - x]);
                x = end;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Arm {
    FullFrame,
    CandidateTiles,
    CandidateRoi,
    CandidateFull,
    GpuExactTiles,
}

#[derive(Serialize)]
struct Sample {
    round: usize,
    layout: &'static str,
    arm: Arm,
    iteration: usize,
    frame_bytes: usize,
    candidate_count: usize,
    changed_count: usize,
    gpu_readback_payload_bytes: usize,
    gpu_history_write_bytes: usize,
    gpu_flag_bytes: usize,
    gpu_copy_operations: usize,
    cpu_frame_write_bytes: usize,
    cpu_history_patch_bytes: usize,
    cpu_materialize_write_bytes: usize,
    returned_cpu_payload_bytes: usize,
    seed_history_retained_bytes: usize,
    prior_page_payload_bytes_retained: usize,
    lease_page_references: usize,
    history_page_payload_allocations: usize,
    materialized_rect: Option<Rect>,
    submit_us: f64,
    map_us: f64,
    patch_us: f64,
    materialize_us: f64,
    total_us: f64,
    process_cycles: Option<u64>,
}

struct Fixture {
    layout: &'static str,
    base: Vec<u32>,
    edited: Vec<u32>,
    candidates: Vec<u32>,
    expected: Vec<u32>,
    roi: Rect,
}

impl Fixture {
    fn new(layout: &'static str, hints: &[u32], changes: &[u32]) -> Self {
        let base: Vec<u32> = (0..WIDTH * HEIGHT)
            .map(|i| 0xff00_0000 | (i as u32).wrapping_mul(0x45d9f3b) & 0x00ff_ffff)
            .collect();
        let mut edited = base.clone();
        for &id in changes {
            let rect = Rect::tile(id);
            let width = (rect.right - rect.left).min(48);
            let height = (rect.bottom - rect.top).min(48);
            for y in rect.top..rect.top + height {
                for x in rect.left..rect.left + width {
                    edited[y * WIDTH + x] ^= 0x0100_7f23;
                }
            }
        }
        let expected = (0..COUNT as u32).filter(|&id| {
            let rect = Rect::tile(id);
            (rect.top..rect.bottom).any(|y| {
                let row = y * WIDTH;
                base[row + rect.left..row + rect.right] != edited[row + rect.left..row + rect.right]
            })
        }).collect();
        let tiles = Rect::tile(changes[changes.len() / 2]);
        let margin = 35;
        let roi = Rect {
            left: tiles.left.saturating_sub(margin), top: tiles.top.saturating_sub(margin),
            right: (tiles.right + margin).min(WIDTH), bottom: (tiles.bottom + margin).min(HEIGHT),
        };
        Self { layout, base, edited, candidates: hints.to_vec(), expected, roi }
    }
}

fn rect_matches(pixels: &[u32], expected: &[u32], rect: Rect) -> bool {
    pixels.chunks_exact(rect.right - rect.left).enumerate().all(|(row, actual)| {
        let from = (row + rect.top) * WIDTH + rect.left;
        actual == &expected[from..from + rect.right - rect.left]
    }) && pixels.len() * 4 == rect.bytes()
}

fn replay(adapter_index: u32, rounds: usize, reps: usize, budget: Duration) -> Result<()> {
    let started = Instant::now();
    let adapter = adapter(adapter_index)?;
    let (device, context) = device(&adapter)?;
    let staged = stage(&device, WIDTH, HEIGHT, DXGI_FORMAT_R32_UINT)?;
    let gpu = gpu::Gpu::new(&device, &context, WIDTH as u32, HEIGHT as u32, TILE as u32)?;
    let gpu_history = gpu.history()?;
    let flags = gpu.staging_buffer()?;
    let sparse = Fixture::new("sparse",
                              &[0, 1, 16, 17, 18, 31, 32, 33, 46, 47, 48, 134],
                              &[1, 16, 17, 31, 32, 47, 48, 134]);
    let scattered_hints: Vec<u32> = (0..COUNT as u32).filter(|v| v % 2 == 0).collect();
    let scattered_changes: Vec<u32> = scattered_hints.iter().copied().filter(|v| v % 6 != 2).collect();
    let scatter = Fixture::new("scattered", &scattered_hints, &scattered_changes);
    let fixtures = [sparse, scatter];
    let textures: Vec<_> = fixtures.iter().map(|f| Ok((gpu.upload(&f.base)?, gpu.upload(&f.edited)?)))
        .collect::<Result<_>>()?;
    let mut rows = Vec::new();
    let mut rng = 0x1b66_2de1_u64;
    'schedules: for round in 0..rounds {
        for (fixture, pair) in fixtures.iter().zip(&textures) {
            let mut order = [Arm::FullFrame, Arm::CandidateTiles, Arm::CandidateRoi, Arm::CandidateFull,
                             Arm::GpuExactTiles];
            for i in (1..order.len()).rev() {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                order.swap(i, rng as usize % (i + 1));
            }
            for arm in order {
                if started.elapsed() >= budget {
                    break 'schedules;
                }
                let mut history = History::seed(&fixture.base);
                let mut prior = history.clone();
                let mut full = vec![0; WIDTH * HEIGHT];
                let mut materialized = Vec::with_capacity(LOAD / 4);
                let mut gpu_changes = Vec::with_capacity(COUNT);
                if matches!(arm, Arm::GpuExactTiles) {
                    gpu.seed_history(&gpu_history, &pair.0)?;
                }
                for iteration in 0..reps {
                    if started.elapsed() >= budget {
                        break 'schedules;
                    }
                    let (current, expected) = if iteration % 2 == 0 {
                        (&pair.1, &fixture.edited)
                    } else {
                        (&pair.0, &fixture.base)
                    };
                    let previous = if iteration % 2 == 0 { &fixture.base } else { &fixture.edited };
                    let cycles = process_cycles();
                    let wall = Instant::now();
                    // Hold the prior page table across the mutation. It is
                    // verified untimed and never shared with a mutable pool.
                    let mut gpu_flag_bytes = 0;
                    let mut gpu_history_bytes = 0;
                    let full_frame = matches!(arm, Arm::FullFrame);
                    if matches!(arm, Arm::GpuExactTiles) {
                        gpu.submit(&gpu_history, current, &fixture.candidates)?;
                        gpu.enqueue_readback(&flags, fixture.candidates.len());
                        gpu.read_flags(&flags, &fixture.candidates, &mut gpu_changes, false)?;
                        gpu_flag_bytes = fixture.candidates.len() * 4;
                        copy_regions(&context, &gpu_history.texture, &current.texture, &gpu_changes);
                        gpu_history_bytes = gpu_changes.iter().map(|&id| Rect::tile(id).bytes()).sum();
                    }
                    let gpu_selected = if matches!(arm, Arm::GpuExactTiles) {
                        gpu_changes.as_slice()
                    } else {
                        fixture.candidates.as_slice()
                    };
                    if full_frame {
                        unsafe { context.CopyResource(&staged, &current.texture) };
                    } else {
                        copy_regions(&context, &staged, &current.texture, gpu_selected);
                    }
                    let submit_us = elapsed_us(wall);
                    let map_started = Instant::now();
                    let mapped = Mapping::new(&context, &staged, WIDTH)?;
                    let map_us = elapsed_us(map_started);
                    let patch_started = Instant::now();
                    let changed = if full_frame {
                        for y in 0..HEIGHT {
                            full[y * WIDTH..(y + 1) * WIDTH].copy_from_slice(mapped.row(y, 0, WIDTH));
                        }
                        history.patch(gpu_selected, false, |y, left, right, output| {
                            output(&full[y * WIDTH + left..y * WIDTH + right]);
                        })
                    } else {
                        history.patch(gpu_selected, matches!(arm, Arm::GpuExactTiles), |y, left, right, output| {
                            output(mapped.row(y, left, right));
                        })
                    };
                    drop(mapped);
                    let patch_us = elapsed_us(patch_started);
                    let materialization = match arm {
                        Arm::CandidateRoi => Some(fixture.roi),
                        Arm::CandidateFull => Some(Rect::full()),
                        _ => None,
                    };
                    let materialize_start = Instant::now();
                    let lease_pages = if materialization.is_some() {
                        let lease = history.clone();
                        lease.materialize(materialization.unwrap(), &mut materialized);
                        lease.pages.len()
                    } else if full_frame {
                        0
                    } else {
                        let lease: Vec<_> = changed.iter().map(|&id| history.pages[id as usize].clone()).collect();
                        std::hint::black_box(&lease);
                        lease.len()
                    };
                    let materialize_us = elapsed_us(materialize_start);
                    let total_us = elapsed_us(wall);
                    let process_cycles = process_cycles().zip(cycles).map(|(end, begin)| end.saturating_sub(begin));
                    if changed != fixture.expected || (full_frame && &full != expected) {
                        return Err(format!("Incorrect {} {:?} round {} iteration {}: {:?}", fixture.layout, arm, round, iteration, changed).into());
                    }
                    if let Some(rect) = materialization {
                        if !rect_matches(&materialized, expected, rect) {
                            return Err("Materialized bytes did not match the current full-frame oracle".into());
                        }
                    }
                    for &id in &changed {
                        let rect = Rect::tile(id);
                        if !rect_matches(&history.pages[id as usize], expected, rect) {
                            return Err("Changed page bytes did not match the current full-frame oracle".into());
                        }
                    }
                    let verify = Rect::tile(fixture.expected[0]);
                    prior.materialize(verify, &mut materialized);
                    if !rect_matches(&materialized, previous, verify) {
                        return Err("A prior page lease changed after the current image was applied".into());
                    }
                    prior = history.clone();
                    let patch_bytes: usize = changed.iter().map(|&id| Rect::tile(id).bytes()).sum();
                    let materialized_bytes = materialization.map_or(0, Rect::bytes);
                    rows.push(Sample {
                        round: round + 1, layout: fixture.layout, arm, iteration, frame_bytes: LOAD,
                        candidate_count: fixture.candidates.len(), changed_count: changed.len(),
                        gpu_readback_payload_bytes: if full_frame { LOAD } else {
                            gpu_selected.iter().map(|&id| Rect::tile(id).bytes()).sum()
                        },
                        gpu_history_write_bytes: gpu_history_bytes, gpu_flag_bytes,
                        gpu_copy_operations: if full_frame { 1 } else {
                            gpu_selected.len() + if gpu_flag_bytes > 0 { 1 + gpu_changes.len() } else { 0 }
                        },
                        cpu_frame_write_bytes: if full_frame { LOAD } else { 0 },
                        cpu_history_patch_bytes: patch_bytes, cpu_materialize_write_bytes: materialized_bytes,
                        returned_cpu_payload_bytes: if full_frame { LOAD } else if materialized_bytes > 0 {
                            materialized_bytes
                        } else { patch_bytes },
                        seed_history_retained_bytes: LOAD, prior_page_payload_bytes_retained: patch_bytes,
                        lease_page_references: lease_pages, history_page_payload_allocations: changed.len(),
                        materialized_rect: materialization, submit_us, map_us, patch_us,
                        materialize_us, total_us, process_cycles,
                    });
                }
            }
        }
    }
    let description = unsafe { adapter.GetDesc1()? };
    let adapter_name = String::from_utf16_lossy(&description.Description).trim_end_matches('\0').to_owned();
    println!("{}", serde_json::json!({
        "mode": "d3d11_uploaded_bgra_bytes_as_r32_uint", "samples": rows, "rounds": rounds,
        "repetitions_per_case": reps, "elapsed_s": started.elapsed().as_secs_f64(),
        "adapter": adapter_name, "staging_allocation_logical_bytes": LOAD,
        "complete": rows.len() == rounds * reps * 10,
    }));
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let telemetry = std::env::var("RAPIDSHOT_IDEA_GPU_TELEMETRY").is_ok_and(|v| v == "1");
    match args.first().map(String::as_str) {
        Some("replay") if args.len() == 5 => {
            let index = args[1].parse()?;
            if telemetry {
                println!("READY replay");
                io::stdout().flush()?;
                wait_for("GO")?;
            }
            sampled_run(telemetry, index, || {
                replay(index, args[2].parse()?, args[3].parse()?, Duration::from_secs_f64(args[4].parse()?))
            })
        }
        Some("live") if args.len() == 6 => {
            let index = args[2].parse()?;
            sampled_run(telemetry, index, || {
                live::run(&args[1], index, args[3].parse()?, args[4].parse()?, args[5].parse()?)
            })
        }
        _ => Err("usage: idea_gpu replay ADAPTER ROUNDS REPS BUDGET_S | live ARM ADAPTER OUTPUT DURATION_S TIMEOUT_MS".into()),
    }
}

fn wait_for(expected: &str) -> Result<()> {
    let mut line = String::new();
    if io::stdin().read_line(&mut line)? == 0 || line.trim() != expected {
        return Err(format!("Missing {expected} handshake").into());
    }
    Ok(())
}

fn segment(adapter: &IDXGIAdapter3, group: DXGI_MEMORY_SEGMENT_GROUP) -> serde_json::Value {
    let mut result = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    match unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut result) } {
        Ok(()) => serde_json::json!({
            "current_usage_bytes": result.CurrentUsage, "budget_bytes": result.Budget,
            "current_reservation_bytes": result.CurrentReservation,
        }),
        Err(error) => serde_json::json!({"unavailable_reason": error.to_string()}),
    }
}

fn video_memory_samples(index: u32, finish: Arc<AtomicBool>) -> serde_json::Value {
    let describe = || -> Result<(IDXGIAdapter3, String)> {
        let base = adapter(index)?;
        let luid = unsafe { base.GetDesc1()? }.AdapterLuid;
        let value = ((luid.HighPart as u64) << 32) | luid.LowPart as u64;
        Ok((base.cast()?, format!("0x{value:016x}")))
    };
    let (adapter, luid) = match describe() {
        Ok(adapter) => adapter,
        Err(error) => return serde_json::json!({"unavailable_reason": error.to_string(), "samples": []}),
    };
    let start = Instant::now();
    let mut rows = Vec::new();
    loop {
        rows.push(serde_json::json!({
            "elapsed_us": elapsed_us(start),
            "local": segment(&adapter, DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
            "non_local": segment(&adapter, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
        }));
        if finish.load(Ordering::Acquire) {
            break;
        }
        std::thread::park_timeout(Duration::from_millis(100));
    }
    serde_json::json!({"adapter_luid": luid, "node": 0, "sample_interval_ms": 100, "samples": rows})
}

fn sampled_run(enabled: bool, index: u32, trial: impl FnOnce() -> Result<()>) -> Result<()> {
    if !enabled {
        return trial();
    }
    let finished = Arc::new(AtomicBool::new(false));
    let signal = Arc::clone(&finished);
    let sampler = std::thread::spawn(move || video_memory_samples(index, signal));
    let outcome = trial();
    finished.store(true, Ordering::Release);
    sampler.thread().unpark();
    let video = sampler.join().map_err(|_| "Native video-memory sampler panicked")?;
    if outcome.is_ok() {
        println!("VIDEO_MEMORY {video}");
        io::stdout().flush()?;
        // The parent samples actual working set/private commit and PDH while
        // this process remains alive. It acknowledges after reading the result.
        wait_for("DONE")?;
    }
    outcome
}
