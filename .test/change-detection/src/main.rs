mod consumer;
mod delivery;
mod experiment;
mod fingerprint;
mod followup;
mod gpu;
mod live;
mod runner;
mod tile_cache;

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::json;
use windows::core::Interface;
use windows::Win32::Foundation::{HMODULE, RECT};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_MODE_ROTATION_IDENTITY;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::UI::HiDpi::{
    SetThreadDpiAwarenessContext, DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2,
};

use gpu::{Gpu, GpuFrame};
use runner::ReportContext;

type AnyResult<T> = Result<T, Box<dyn std::error::Error>>;
type Pixels = Rc<[u32]>;

const TILE_SIZE: u32 = 128;
const BASELINE: &str = "4b20e5c2230322dc68ffcec3934bd4a33286075d";

struct Case {
    name: String,
    source: String,
    pairs: Vec<FramePair>,
}

struct FramePair {
    previous: Rc<GpuFrame>,
    current: Rc<GpuFrame>,
    metadata_tiles: Vec<u32>,
    hybrid_tiles: Vec<u32>,
    expected: Vec<u32>,
    fallback: Option<&'static str>,
}

struct InputCase {
    name: &'static str,
    source: &'static str,
    pairs: Vec<InputPair>,
}

struct InputPair {
    previous: Pixels,
    current: Pixels,
    presented: bool,
    continuous: bool,
    dirty: Option<Vec<RECT>>,
    moves: Option<Vec<RECT>>,
}

#[derive(Clone, Copy)]
struct Extent {
    width: u32,
    height: u32,
}

impl Extent {
    fn tile_columns(self) -> u32 {
        self.width.div_ceil(TILE_SIZE)
    }

    fn tile_count(self) -> u32 {
        self.tile_columns() * self.height.div_ceil(TILE_SIZE)
    }

    fn screen_rect(self) -> RECT {
        RECT {
            left: 0,
            top: 0,
            right: self.width as i32,
            bottom: self.height as i32,
        }
    }

    fn tiles_for_rects(self, rects: impl Iterator<Item = RECT>) -> Option<Vec<u32>> {
        let mut tiles = vec![false; self.tile_count() as usize];
        for rect in rects {
            if rect.left < 0
                || rect.top < 0
                || rect.right <= rect.left
                || rect.bottom <= rect.top
                || rect.right > self.width as i32
                || rect.bottom > self.height as i32
            {
                return None;
            }
            let first_x = rect.left as u32 / TILE_SIZE;
            let last_x = (rect.right as u32 - 1) / TILE_SIZE;
            let first_y = rect.top as u32 / TILE_SIZE;
            let last_y = (rect.bottom as u32 - 1) / TILE_SIZE;
            for y in first_y..=last_y {
                for x in first_x..=last_x {
                    tiles[(y * self.tile_columns() + x) as usize] = true;
                }
            }
        }
        Some(marked_tiles(&tiles))
    }

    fn cpu_changed_tiles(self, previous: &[u32], current: &[u32]) -> Vec<u32> {
        assert_eq!(previous.len(), (self.width * self.height) as usize);
        assert_eq!(current.len(), previous.len());
        let mut tiles = vec![false; self.tile_count() as usize];
        for (index, (old, new)) in previous.iter().zip(current).enumerate() {
            if old != new {
                let x = index as u32 % self.width;
                let y = index as u32 / self.width;
                tiles[(y / TILE_SIZE * self.tile_columns() + x / TILE_SIZE) as usize] = true;
            }
        }
        marked_tiles(&tiles)
    }
}

fn marked_tiles(tiles: &[bool]) -> Vec<u32> {
    tiles
        .iter()
        .enumerate()
        .filter_map(|(index, &marked)| marked.then_some(index as u32))
        .collect()
}

fn prepare_cases(
    gpu: &Gpu,
    extent: Extent,
    inputs: &[InputCase],
    all_tiles: &[u32],
) -> AnyResult<Vec<Case>> {
    let mut uploaded = HashMap::<*const [u32], Rc<GpuFrame>>::new();
    let mut frame = |pixels: &Pixels| -> AnyResult<Rc<GpuFrame>> {
        let key = Rc::as_ptr(pixels);
        if let Some(frame) = uploaded.get(&key) {
            return Ok(Rc::clone(frame));
        }
        let frame = Rc::new(gpu.upload(pixels)?);
        uploaded.insert(key, Rc::clone(&frame));
        Ok(frame)
    };

    let mut cases = Vec::new();
    for case in inputs {
        let mut pairs = Vec::new();
        for pair in &case.pairs {
            let metadata = pair.dirty.as_ref().zip(pair.moves.as_ref());
            let (metadata_tiles, hybrid_tiles, fallback) = match (pair.presented, metadata) {
                (false, _) if pair.continuous => (vec![], vec![], None),
                (_, None) => (
                    all_tiles.to_vec(),
                    all_tiles.to_vec(),
                    Some("metadata_unavailable"),
                ),
                (_, Some((dirty, moves))) => {
                    let tiles = extent.tiles_for_rects(dirty.iter().chain(moves).copied());
                    match tiles {
                        None => (
                            all_tiles.to_vec(),
                            all_tiles.to_vec(),
                            Some("invalid_rectangle"),
                        ),
                        Some(tiles) if tiles.is_empty() => (
                            all_tiles.to_vec(),
                            all_tiles.to_vec(),
                            Some("present_without_rectangles"),
                        ),
                        Some(tiles) if !pair.continuous => {
                            (tiles, all_tiles.to_vec(), Some("frame_gap"))
                        }
                        Some(tiles) if !moves.is_empty() => {
                            (tiles, all_tiles.to_vec(), Some("move_rectangle"))
                        }
                        Some(tiles) => (tiles.clone(), tiles, None),
                    }
                }
            };
            pairs.push(FramePair {
                previous: frame(&pair.previous)?,
                current: frame(&pair.current)?,
                expected: extent.cpu_changed_tiles(&pair.previous, &pair.current),
                metadata_tiles,
                hybrid_tiles,
                fallback,
            });
        }
        if !pairs.is_empty() {
            cases.push(Case {
                name: case.name.into(),
                source: case.source.into(),
                pairs,
            });
        }
    }
    Ok(cases)
}

fn fixed_pair(previous: &Pixels, current: &Pixels, presented: bool, dirty: Vec<RECT>) -> InputPair {
    InputPair {
        previous: Rc::clone(previous),
        current: Rc::clone(current),
        presented,
        continuous: true,
        dirty: Some(dirty),
        moves: Some(vec![]),
    }
}

fn solid_frame(extent: Extent, color: u32) -> Pixels {
    vec![color; (extent.width * extent.height) as usize].into()
}

fn add_fixtures(cases: &mut Vec<InputCase>, extent: Extent, fixtures_only: bool) {
    let previous = solid_frame(extent, 0xff32_5476);
    let full = extent.screen_rect();
    let small = RECT {
        left: 11,
        top: 17,
        right: 37,
        bottom: 49,
    };
    let mut sparse = previous.to_vec();
    for y in small.top..small.bottom {
        let start = y as usize * extent.width as usize + small.left as usize;
        sparse[start..start + (small.right - small.left) as usize].fill(0xffcc_8844);
    }
    let sparse: Pixels = sparse.into();
    let motion = solid_frame(extent, 0xff85_6237);

    if !cases.iter().any(|case| case.name == "idle_live") {
        cases.push(InputCase {
            name: "idle_duplicate_replay",
            source: "Synthetic idle baseline. No image presentation, so latency is per hypothetical decision.",
            pairs: vec![fixed_pair(&previous, &previous, false, vec![])],
        });
    }
    if !cases.iter().any(|case| case.name == "excluded_cover_live") {
        cases.push(InputCase {
            name: "excluded_cover_wakeup_replay",
            source: "Synthetic excluded-cover wake-up. Identical pixels and a conservative, partial DXGI-style dirty rectangle.",
            pairs: vec![fixed_pair(&previous, &previous, true, vec![RECT {
                left: extent.width as i32 / 3,
                top: extent.height as i32 / 3,
                right: extent.width as i32 * 2 / 3,
                bottom: extent.height as i32 * 2 / 3,
            }])],
        });
    }
    if fixtures_only {
        cases.push(InputCase {
            name: "sparse_fixture",
            source: "Synthetic small visible update with a one-tile dirty mask.",
            pairs: vec![fixed_pair(&previous, &sparse, true, vec![small])],
        });
        cases.push(InputCase {
            name: "full_motion_fixture",
            source: "Synthetic full-screen visible update and full-screen metadata.",
            pairs: vec![fixed_pair(&previous, &motion, true, vec![full])],
        });
    }

    // Exercise partial final rows/columns, disjoint candidates, empty metadata and
    // lost continuity independently of what the compositor chooses to emit.
    let mut edge = sparse.to_vec();
    *edge.last_mut().unwrap() ^= 0x0100_0000;
    let edge: Pixels = edge.into();
    let bottom_right = RECT {
        left: full.right - 1,
        top: full.bottom - 1,
        ..full
    };
    let mut missing = fixed_pair(&previous, &edge, true, vec![]);
    missing.dirty = None;
    let mut gap = fixed_pair(&previous, &edge, true, vec![small]);
    gap.continuous = false;
    let mut moved = fixed_pair(&previous, &edge, true, vec![small]);
    moved.moves = Some(vec![bottom_right]);
    cases.push(InputCase {
        name: "metadata_fallback_checks",
        source: "Synthetic cases for edge pixels, overlapping rectangles, presented empty metadata, missing metadata, move rectangles and a capture gap.",
        pairs: vec![
            fixed_pair(&previous, &edge, true, vec![small, small, bottom_right]),
            fixed_pair(&previous, &edge, true, vec![]),
            missing, moved, gap,
        ],
    });
}

struct Display {
    label: String,
    bounds: RECT,
    output: IDXGIOutput1,
    adapter: String,
    device: ID3D11Device,
    context: ID3D11DeviceContext,
}

fn display() -> AnyResult<Display> {
    unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let adapter = unsafe { factory.EnumAdapters1(0)? };
    let adapter_name = unsafe { adapter.GetDesc1()? };
    // Prefer the attached non-primary display on this adapter for the test windows.
    let primary = unsafe { adapter.EnumOutputs(0)? };
    let secondary = unsafe { adapter.EnumOutputs(1) }.ok();
    let output = secondary
        .filter(|item| unsafe { item.GetDesc() }.is_ok_and(|d| d.AttachedToDesktop.as_bool()))
        .unwrap_or(primary);
    let description = unsafe { output.GetDesc()? };
    if description.Rotation != DXGI_MODE_ROTATION_IDENTITY {
        return Err("This harness expects an unrotated output".into());
    }
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
    let name = |value: &[u16]| {
        String::from_utf16_lossy(value)
            .trim_end_matches('\0')
            .to_owned()
    };
    Ok(Display {
        label: name(&description.DeviceName),
        bounds: description.DesktopCoordinates,
        output: output.cast()?,
        adapter: name(&adapter_name.Description),
        device: device.ok_or("D3D11CreateDevice returned no device")?,
        context: context.ok_or("D3D11CreateDevice returned no device context")?,
    })
}

fn main() -> AnyResult<()> {
    let mut reps = 120;
    let mut rounds = 5;
    let mut seed = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos() as u64;
    let mut output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results");
    let mut fixtures_only = false;
    let mut followup = false;
    let mut explicit_output = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--reps" => reps = args.next().ok_or("Missing --reps value")?.parse()?,
            "--rounds" => rounds = args.next().ok_or("Missing --rounds value")?.parse()?,
            "--seed" => seed = args.next().ok_or("Missing --seed value")?.parse()?,
            "--output" => {
                output = PathBuf::from(args.next().ok_or("Missing --output value")?);
                explicit_output = true;
            }
            "--fixtures-only" => fixtures_only = true,
            "--followup" => followup = true,
            _ => return Err(format!("Unknown argument {arg}").into()),
        }
    }
    if reps == 0 || rounds == 0 {
        return Err("Reps and rounds must be positive".into());
    }
    let display = display()?;
    let extent = Extent {
        width: (display.bounds.right - display.bounds.left).try_into()?,
        height: (display.bounds.bottom - display.bounds.top).try_into()?,
    };
    if (extent.width, extent.height) != (1920, 1080) {
        return Err(format!(
            "Expected a 1920x1080 output, found {}x{}",
            extent.width, extent.height
        )
        .into());
    }
    eprintln!(
        "{} ({}) at {}x{}",
        display.adapter, display.label, extent.width, extent.height
    );
    let gpu = Gpu::new(
        &display.device,
        &display.context,
        extent.width,
        extent.height,
        TILE_SIZE,
    )?;
    let (mut inputs, live_results) = if fixtures_only {
        (
            vec![],
            json!({"capture": "skipped for fixture smoke check"}),
        )
    } else {
        live::capture_cases(&display, extent)?
    };
    add_fixtures(&mut inputs, extent, fixtures_only);
    if followup {
        if !explicit_output {
            output = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("results-followup");
        }
        experiment::add_validation_cases(&mut inputs, extent);
    }
    let all_tiles: Vec<u32> = (0..extent.tile_count()).collect();
    let cases = prepare_cases(&gpu, extent, &inputs, &all_tiles)?;
    if followup {
        return experiment::run(
            &display,
            extent,
            &gpu,
            &inputs,
            &cases,
            &all_tiles,
            rounds,
            reps,
            seed,
            &output,
            live_results,
        );
    }
    drop(inputs);
    eprintln!("Validating {} workloads, seed {seed}", cases.len());
    let report = runner::run(&gpu, &cases, &all_tiles, rounds, seed, reps)?;
    runner::write_report(
        &report,
        &output,
        ReportContext {
            commit: BASELINE.into(),
            adapter: display.adapter,
            width: extent.width,
            height: extent.height,
            tile_size: TILE_SIZE,
            live: live_results,
        },
    )?;
    eprintln!("Saved results in {}", output.display());
    Ok(())
}
