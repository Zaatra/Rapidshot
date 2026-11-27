use std::time::{Duration, Instant};

use rapidshot_core::duplicator::Duplicator;
use serde::Serialize;
use serde_json::{json, Value};
use windows::core::w;
use windows::Win32::Foundation::{COLORREF, HINSTANCE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dwm::DwmFlush;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::WindowsAndMessaging::*;

use crate::{AnyResult, Display, Extent, InputCase, InputPair, Pixels};

const STEPS: usize = 8;
const BACKGROUND: u32 = 0x32_5476;
const SPARSE: RECT = RECT {
    left: 24,
    top: 30,
    right: 68,
    bottom: 78,
};
const COVER: RECT = RECT {
    left: 700,
    top: 360,
    right: 1220,
    bottom: 720,
};
const UNDER_COVER: RECT = RECT {
    left: 840,
    top: 450,
    right: 915,
    bottom: 520,
};
const WINDOW_CLASS: windows::core::PCWSTR = w!("RapidShotChangeDetectionBenchmarkWindow");

struct Scene {
    background: u32,
    patch: Option<(RECT, u32)>,
}

struct SceneWindow {
    handle: HWND,
    scene: Box<Scene>,
}

unsafe extern "system" fn window_proc(
    window: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if message == WM_PAINT {
        let scene = GetWindowLongPtrW(window, GWLP_USERDATA) as *const Scene;
        let mut paint = PAINTSTRUCT::default();
        let dc = BeginPaint(window, &mut paint);
        if let Some(scene) = scene.as_ref() {
            fill(dc, &paint.rcPaint, scene.background);
            if let Some((rect, color)) = &scene.patch {
                fill(dc, rect, *color);
            }
        }
        let _ = EndPaint(window, &paint);
        return LRESULT(0);
    }
    DefWindowProcW(window, message, wparam, lparam)
}

unsafe fn fill(dc: HDC, rect: &RECT, color: u32) {
    let colorref = (color & 0xff) << 16 | (color & 0xff00) | ((color >> 16) & 0xff);
    let brush = CreateSolidBrush(COLORREF(colorref));
    FillRect(dc, rect, brush);
    let _ = DeleteObject(brush.into());
}

impl SceneWindow {
    fn new(bounds: RECT, color: u32, exclude: bool) -> AnyResult<Self> {
        let module = unsafe { GetModuleHandleW(None)? };
        let instance = HINSTANCE(module.0);
        let class = WNDCLASSW {
            lpfnWndProc: Some(window_proc),
            hInstance: instance,
            lpszClassName: WINDOW_CLASS,
            ..Default::default()
        };
        // A second window reuses the class already registered by the first.
        unsafe { RegisterClassW(&class) };
        let mut window = Self {
            handle: unsafe {
                CreateWindowExW(
                    WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST,
                    WINDOW_CLASS,
                    w!("RapidShot benchmark"),
                    WS_POPUP,
                    bounds.left,
                    bounds.top,
                    bounds.right - bounds.left,
                    bounds.bottom - bounds.top,
                    None,
                    None,
                    Some(instance),
                    None,
                )?
            },
            scene: Box::new(Scene {
                background: color,
                patch: None,
            }),
        };
        unsafe {
            SetWindowLongPtrW(
                window.handle,
                GWLP_USERDATA,
                (&mut *window.scene as *mut Scene) as isize,
            );
            if exclude {
                SetWindowDisplayAffinity(window.handle, WDA_EXCLUDEFROMCAPTURE)?;
            }
            let _ = ShowWindow(window.handle, SW_SHOWNOACTIVATE);
        }
        window.redraw(None)?;
        Ok(window)
    }

    fn redraw(&self, rect: Option<&RECT>) -> AnyResult<()> {
        unsafe {
            let flags = RDW_INVALIDATE | RDW_UPDATENOW;
            if !RedrawWindow(Some(self.handle), rect.map(std::ptr::from_ref), None, flags).as_bool()
            {
                return Err(windows::core::Error::from_thread().into());
            }
            let mut message = MSG::default();
            while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&message);
                DispatchMessageW(&message);
            }
            DwmFlush()?;
        }
        Ok(())
    }

    fn change(&mut self, color: u32, patch: Option<RECT>) -> AnyResult<()> {
        self.scene.patch = patch.map(|rect| (rect, color));
        if patch.is_none() {
            self.scene.background = color;
        }
        self.redraw(patch.as_ref())
    }
}

impl Drop for SceneWindow {
    fn drop(&mut self) {
        unsafe {
            let _ = DestroyWindow(self.handle);
        }
    }
}

struct Snapshot {
    pixels: Pixels,
    serial: u64,
    accumulated: u32,
    dirty: Option<Vec<RECT>>,
    moves: Option<Vec<RECT>>,
    coalesced: bool,
}

enum Acquisition {
    Frame(Snapshot),
    NoImage,
    Timeout,
}

struct Sampler<'a> {
    capture: Duplicator,
    staging: ID3D11Texture2D,
    context: &'a ID3D11DeviceContext,
    extent: Extent,
}

impl<'a> Sampler<'a> {
    fn new(display: &'a Display, extent: Extent) -> AnyResult<Self> {
        let mut staging = None;
        let desc = D3D11_TEXTURE2D_DESC {
            Width: extent.width,
            Height: extent.height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_STAGING,
            CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
            ..Default::default()
        };
        unsafe {
            display
                .device
                .CreateTexture2D(&desc, None, Some(&mut staging))?
        };
        Ok(Self {
            capture: Duplicator::new(&display.output, &display.device, 45)?,
            staging: staging.ok_or("No staging texture")?,
            context: &display.context,
            extent,
        })
    }

    fn sample(&mut self) -> AnyResult<Acquisition> {
        if !self.capture.update_frame()? {
            return Err(format!(
                "DXGI duplicator became unhealthy: {}",
                self.capture.last_error
            )
            .into());
        }
        if !self.capture.updated {
            let status = if self.capture.frame_acquired() {
                Acquisition::NoImage
            } else {
                Acquisition::Timeout
            };
            self.capture.release_frame();
            return Ok(status);
        }
        let pixels = self.read_pixels();
        self.capture.release_frame();
        Ok(Acquisition::Frame(Snapshot {
            pixels: pixels?,
            serial: self.capture.frame_serial,
            accumulated: self.capture.accumulated_frames,
            dirty: self.capture.dirty_rects.as_ref().map(|rects| {
                rects
                    .iter()
                    .map(|&(left, top, right, bottom)| RECT {
                        left,
                        top,
                        right,
                        bottom,
                    })
                    .collect()
            }),
            moves: self.capture.move_rects.as_ref().map(|rects| {
                rects
                    .iter()
                    .flat_map(|&(x, y, left, top, right, bottom)| {
                        [
                            RECT {
                                left,
                                top,
                                right,
                                bottom,
                            },
                            RECT {
                                left: x,
                                top: y,
                                right: x + right - left,
                                bottom: y + bottom - top,
                            },
                        ]
                    })
                    .collect()
            }),
            coalesced: self.capture.rects_coalesced,
        }))
    }

    fn read_pixels(&self) -> AnyResult<Pixels> {
        let source = self.capture.texture().ok_or("DXGI returned no texture")?;
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { source.GetDesc(&mut desc) };
        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM
            || (desc.Width, desc.Height) != (self.extent.width, self.extent.height)
        {
            return Err(format!(
                "Expected an SDR BGRA8 texture, got format {} {}x{}",
                desc.Format.0, desc.Width, desc.Height
            )
            .into());
        }
        let mut pixels = Vec::with_capacity((desc.Width * desc.Height) as usize);
        unsafe {
            self.context.CopyResource(&self.staging, source);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
            for y in 0..desc.Height as usize {
                let row = (mapped.pData as *const u8)
                    .add(y * mapped.RowPitch as usize)
                    .cast::<u32>();
                pixels.extend_from_slice(std::slice::from_raw_parts(row, desc.Width as usize));
            }
            self.context.Unmap(&self.staging, 0);
        }
        Ok(pixels.into())
    }
}

#[derive(Default, Serialize)]
struct Observations {
    steps: usize,
    image_frames: usize,
    identical_image_frames: usize,
    changed_image_frames: usize,
    coalesced_frames: usize,
    no_image_acquisitions: usize,
    timeouts: usize,
    scene_mismatches: usize,
    replay_pairs: usize,
}

fn pixel(snapshot: &Snapshot, extent: Extent, x: i32, y: i32) -> u32 {
    snapshot.pixels[(y as u32 * extent.width + x as u32) as usize] & 0xff_ffff
}

fn center(rect: RECT) -> (i32, i32) {
    ((rect.left + rect.right) / 2, (rect.top + rect.bottom) / 2)
}

fn prime(sampler: &mut Sampler<'_>, point: (i32, i32), color: u32) -> AnyResult<Snapshot> {
    let deadline = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline {
        if let Acquisition::Frame(frame) = sampler.sample()? {
            if pixel(&frame, sampler.extent, point.0, point.1) == color {
                return Ok(frame);
            }
        }
    }
    Err(format!("The controlled DXGI scene did not reach color 0x{color:06x} at {point:?}").into())
}

fn track(
    sampler: &mut Sampler<'_>,
    prior: &mut Snapshot,
    verify: ((i32, i32), u32),
    pairs: &mut Vec<InputPair>,
    observations: &mut Observations,
    wait_for_change: bool,
) -> AnyResult<()> {
    observations.steps += 1;
    let attempts = if wait_for_change { 5 } else { 1 };
    for _ in 0..attempts {
        let next = match sampler.sample()? {
            Acquisition::Frame(next) => next,
            Acquisition::NoImage => {
                observations.no_image_acquisitions += 1;
                continue;
            }
            Acquisition::Timeout => {
                observations.timeouts += 1;
                continue;
            }
        };
        observations.image_frames += 1;
        observations.coalesced_frames += usize::from(next.coalesced);
        let ((x, y), color) = verify;
        if pixel(&next, sampler.extent, x, y) != color {
            observations.scene_mismatches += 1;
            continue;
        }
        let identical = prior.pixels == next.pixels;
        if identical {
            observations.identical_image_frames += 1;
        } else {
            observations.changed_image_frames += 1;
        }
        let current = if identical {
            prior.pixels.clone()
        } else {
            next.pixels.clone()
        };
        pairs.push(InputPair {
            previous: prior.pixels.clone(),
            current: current.clone(),
            presented: true,
            continuous: next.serial == prior.serial + 1 && next.accumulated <= 1,
            dirty: next.dirty.clone(),
            moves: next.moves.clone(),
        });
        *prior = Snapshot {
            pixels: current,
            ..next
        };
        observations.replay_pairs += 1;
        return Ok(());
    }
    Ok(())
}

fn translated(rect: RECT, desktop: RECT) -> RECT {
    RECT {
        left: rect.left + desktop.left,
        right: rect.right + desktop.left,
        top: rect.top + desktop.top,
        bottom: rect.bottom + desktop.top,
    }
}

fn case(name: &'static str, source: &'static str) -> InputCase {
    InputCase {
        name,
        source,
        pairs: Vec::new(),
    }
}

pub(super) fn capture_cases(
    display: &Display,
    extent: Extent,
) -> AnyResult<(Vec<InputCase>, Value)> {
    let mut sampler = Sampler::new(display, extent)?;
    let mut base = SceneWindow::new(display.bounds, BACKGROUND, false)?;
    let near_corner = center(SPARSE);
    let under_cover = center(UNDER_COVER);
    let mut prior = prime(&mut sampler, near_corner, BACKGROUND)?;
    let mut idle = case(
        "idle_live",
        "Live DXGI image acquisitions while the desktop test window was static.",
    );
    let mut idle_status = Observations::default();
    for _ in 0..STEPS {
        track(
            &mut sampler,
            &mut prior,
            (near_corner, BACKGROUND),
            &mut idle.pairs,
            &mut idle_status,
            false,
        )?;
    }

    let mut sparse = case(
        "sparse_live",
        "Live DXGI BGRA8 trace of a changing 44x48 patch on a static background.",
    );
    let mut sparse_status = Observations::default();
    let mut full = case(
        "full_motion_live",
        "Live DXGI BGRA8 trace of visible full-screen motion.",
    );
    let mut full_status = Observations::default();
    for step in 0..STEPS {
        let color = if step % 2 == 0 { 0x87_3514 } else { 0x18_7654 };
        base.change(color, Some(SPARSE))?;
        track(
            &mut sampler,
            &mut prior,
            (near_corner, color),
            &mut sparse.pairs,
            &mut sparse_status,
            true,
        )?;
    }
    for step in 0..STEPS {
        let color = if step % 2 == 0 { 0x42_1675 } else { 0x75_4816 };
        base.change(color, None)?;
        track(
            &mut sampler,
            &mut prior,
            (near_corner, color),
            &mut full.pairs,
            &mut full_status,
            true,
        )?;
    }
    if sparse.pairs.is_empty() || full.pairs.is_empty() {
        return Err(format!(
            "Could not verify live sparse or full-screen motion: {}, {} pairs",
            sparse.pairs.len(),
            full.pairs.len()
        )
        .into());
    }

    base.change(BACKGROUND, None)?;
    let mut prior = prime(&mut sampler, under_cover, BACKGROUND)?;
    let mut excluded = case("excluded_cover_live", "Live DXGI acquisitions during animation of a WDA_EXCLUDEFROMCAPTURE cover over static content.");
    let mut beneath = case("under_excluded_cover_live", "Live DXGI acquisitions while visible content changes underneath a WDA_EXCLUDEFROMCAPTURE cover.");
    let mut excluded_status = Observations::default();
    let mut beneath_status = Observations::default();
    let mut affinity_error = None;
    match SceneWindow::new(translated(COVER, display.bounds), 0x8b_35aa, true) {
        Ok(mut cover) => {
            for step in 0..STEPS {
                let color = if step % 2 == 0 { 0x35_ab22 } else { 0x8b_35aa };
                cover.change(color, None)?;
                track(
                    &mut sampler,
                    &mut prior,
                    (under_cover, BACKGROUND),
                    &mut excluded.pairs,
                    &mut excluded_status,
                    false,
                )?;
            }
            for step in 0..STEPS {
                let color = if step % 2 == 0 { 0x79_b321 } else { 0xc3_7733 };
                base.change(color, Some(UNDER_COVER))?;
                track(
                    &mut sampler,
                    &mut prior,
                    (under_cover, color),
                    &mut beneath.pairs,
                    &mut beneath_status,
                    true,
                )?;
            }
        }
        Err(error) => affinity_error = Some(error.to_string()),
    }
    let observations = json!({
        "output": display.label,
        "captured_format": "B8G8R8A8_UNORM",
        "idle": idle_status,
        "sparse": sparse_status,
        "full_motion": full_status,
        "excluded_cover": excluded_status,
        "under_excluded_cover": beneath_status,
        "cover_setup_error": affinity_error,
    });
    eprintln!("Live DXGI observations: {observations}");
    let cases = vec![idle, sparse, full, excluded, beneath]
        .into_iter()
        .filter(|case| !case.pairs.is_empty())
        .collect();
    Ok((cases, observations))
}
