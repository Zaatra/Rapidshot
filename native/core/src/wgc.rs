//! Windows.Graphics.Capture: monitors and single windows (ROADMAP § 7.3).
//!
//! The design follows measurements, not the documentation's defaults. Machine
//! B, 2026-10-22, `build/spike-2.7/FINDINGS.md`:
//!
//! - **Copy, then close the WGC frame at once.** Pool surfaces are shareable
//!   (`SHARED | SHARED_NTHANDLE`), and reading one directly is safe only while
//!   its frame stays open: after `Close()` WGC rewrote the surface within 40 ms
//!   in 50 of 50 trials. Keeping frames open to cover a consumer pins one of
//!   the pool's two buffers, and the pool is first-in-first-out, so a 30 ms
//!   hold delivered frames 25 ms older. The copy costs 0.75 ms of GPU time at
//!   2560x1600 BGRA.
//! - **Drain the pool as frames arrive; three buffers.** A frame left in the
//!   pool holds a buffer, and with none free WGC drops new frames for good:
//!   a window changed 3 times between grabs came back showing its 2nd state
//!   and stayed there (2026-10-29). The FrameArrived handler keeps only the
//!   newest frame, so the pool always has room; the third buffer covers a
//!   change landing while a frame is being copied.
//! - **One owned texture, not a ring.** `SHARED | SHARED_NTHANDLE` (0x802)
//!   creates, opens on D3D12 and read back 150/150 byte-exact. A single texture
//!   keeps its address from frame to frame, so a `GpuConverter`'s cache keyed
//!   on (address, source id) keeps hitting -- a ring would make it reopen the
//!   surface every frame. The camera never holds two frames from one backend,
//!   and `Frame.release()` drains GPU readers before the next acquire, so the
//!   next copy cannot overwrite a texture still being read.
//! - **Readers of the copy still need the capture-order fence.** With no
//!   ordering, D3D12 read the previous frame 7 times in 150; behind the shared
//!   fence in `capture_order.rs`, 150/150 were current. Nothing changes there:
//!   the copy is recorded on the capture device's immediate context, which is
//!   the context that fence is signalled on.
//! - **Set `MinUpdateInterval`.** The default (16 ms on build 26300) capped a
//!   165 Hz panel at 55 fps; 1 ms gave 164.
//! - **`SystemRelativeTime` is not a capture time.** It is QPC-based but falls
//!   about 3.4 ms *after* acquisition and 17.6 ms after the source's
//!   `Present()` -- a target presentation time. The frame's time is the QPC
//!   at which WGC signalled its arrival instead: acquisition time alone would
//!   hide however long the frame waited in the pool. Desktop Duplication's
//!   `LastPresentTime` falls 0.1 ms before acquisition, so the two backends'
//!   times are comparable.
//! - **A closed window is detected by polling, not by `Closed`.** That event
//!   needs the creating thread to pump messages, which a capture thread
//!   blocked in a wait does not. A closed window's item reports a 0x0 size and
//!   `IsWindow` turns false; both are checked when a wait times out.
//! - **WGC uses the device from its own threads.** Creating the frame pool
//!   turns on D3D11 multithread protection for it. Calls are serialised by
//!   D3D's lock; nothing here assumes another thread is not between two calls.

use std::sync::{Arc, Mutex, MutexGuard};

use windows::core::{w, IInspectable, Interface};
use windows::Foundation::{TimeSpan, TypedEventHandler};
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::{
    CloseHandle, E_INVALIDARG, HANDLE, HMODULE, HWND, RPC_E_CHANGED_MODE,
};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D, D3D11_BIND_RENDER_TARGET,
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows::Win32::Graphics::Gdi::HMONITOR;
use windows::Win32::System::LibraryLoader::{GetModuleHandleExW, GET_MODULE_HANDLE_EX_FLAG_PIN};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForSingleObject};
use windows::Win32::System::WinRT::Direct3D11::{
    CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess,
};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RoInitialize, RO_INIT_MULTITHREADED};
use windows::Win32::UI::WindowsAndMessaging::IsWindow;

type Result<T> = windows::core::Result<T>;

/// `D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE`: what
/// WGC gives its own pool surfaces, and what the D3D12 consumers need.
const OWNED_MISC_FLAGS: u32 = 0x2 | 0x800;

/// What to capture.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    Monitor(isize),
    Window(isize),
}

/// Pixel format of the capture, and so of the owned copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// `B8G8R8A8_UNORM` (87): sRGB, what Desktop Duplication gives an SDR desktop.
    Bgra8,
    /// `R16G16B16A16_FLOAT` (10): linear scRGB, 1.0 = 80 nits.
    Fp16,
}

impl Format {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "bgra" | "bgra8" => Some(Self::Bgra8),
            "fp16" | "float16" => Some(Self::Fp16),
            _ => None,
        }
    }

    fn winrt(self) -> DirectXPixelFormat {
        match self {
            Self::Bgra8 => DirectXPixelFormat::B8G8R8A8UIntNormalized,
            Self::Fp16 => DirectXPixelFormat::R16G16B16A16Float,
        }
    }

    pub fn dxgi(self) -> u32 {
        match self {
            Self::Bgra8 => 87,
            Self::Fp16 => 10,
        }
    }
}

/// One acquired frame, copied into the session's owned texture.
#[derive(Debug)]
pub struct Acquired {
    /// Borrowed address of the owned `ID3D11Texture2D`; valid until the next
    /// `acquire` or `close`.
    pub texture: usize,
    pub width: u32,
    pub height: u32,
    pub dxgi_format: u32,
    /// QPC when WGC signalled this frame's arrival: the frame's time. Later
    /// than that is time it spent waiting in the pool, which is part of its
    /// age, so the acquisition time alone would understate it. Never later
    /// than `acquired_qpc`.
    pub arrived_qpc: i64,
    /// QPC at acquisition.
    pub acquired_qpc: i64,
    /// `SystemRelativeTime` in QPC ticks, for diagnostics only (see module docs).
    pub system_relative_qpc: i64,
    /// New-content frames delivered by this session, starting at 1.
    pub serial: u64,
    /// Older frames closed unread to reach this one.
    pub skipped: u32,
}

#[derive(Debug)]
pub enum Outcome {
    Frame(Acquired),
    Timeout,
    /// The source's size changed; the session must be rebuilt at the new size.
    Resized(u32, u32),
    /// The window is gone.
    Closed,
}

/// Make this thread usable for WinRT calls. A thread COM already initialised
/// as single-threaded -- comtypes does that to Python's main thread -- keeps
/// its apartment; the objects used here are agile, so either works.
pub fn ensure_winrt() -> Result<()> {
    match unsafe { RoInitialize(RO_INIT_MULTITHREADED) } {
        Ok(()) => Ok(()),
        Err(e) if e.code() == RPC_E_CHANGED_MODE => Ok(()),
        Err(e) => Err(e),
    }
}

/// Whether `GraphicsCapture.dll` is pinned in this process.
static GRAPHICS_CAPTURE_PINNED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Keep `GraphicsCapture.dll` mapped until the process exits; returns whether
/// it is. Called by `WgcSession::new` once capture has started, when the DLL
/// is certainly loaded. Idempotent.
///
/// **WGC outlives `close()`.** It finishes a session on its own threads, and
/// a process that unloads the DLL meanwhile -- `CoUninitialize` frees
/// in-process WinRT servers, as comtypes does from Python's atexit -- runs
/// that work in unmapped memory. Every such crash Windows recorded (Event
/// 1000) faulted in `GraphicsCapture.dll_unloaded`, offset 0x1106c. On an HDR
/// desktop (Arc 130V, 2026-11-12), 120 processes each, from Python: a session
/// left open crashed 6 times after a frame and 74 before one; with this same
/// call made first, 0 and 0. The wait in `close()` narrows the window; only
/// the pin closes it, for Rust callers as for Python ones. A failure is not an
/// error: capture works without it, and only an exit mid-teardown is at risk.
pub fn pin_graphics_capture() -> bool {
    use std::sync::atomic::Ordering;
    if GRAPHICS_CAPTURE_PINNED.load(Ordering::Acquire) {
        return true;
    }
    let mut module = HMODULE::default();
    let result = unsafe {
        GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_PIN,
            w!("GraphicsCapture.dll"),
            &mut module,
        )
    };
    if result.is_err() {
        return false;
    }
    GRAPHICS_CAPTURE_PINNED.store(true, Ordering::Release);
    true
}

/// Whether Windows.Graphics.Capture is available at all.
pub fn is_supported() -> bool {
    ensure_winrt().is_ok() && GraphicsCaptureSession::IsSupported().unwrap_or(false)
}

fn create_item(target: Target) -> Result<GraphicsCaptureItem> {
    let interop = windows::core::factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
    unsafe {
        match target {
            Target::Monitor(handle) => interop.CreateForMonitor(HMONITOR(handle as *mut _)),
            Target::Window(handle) => interop.CreateForWindow(HWND(handle as *mut _)),
        }
    }
}

/// Distinct targets kept at once. One per display or captured window is the
/// realistic count; past this the oldest goes, costing the handle noted below.
const MAX_ITEMS: usize = 64;

/// Capture items, one per target, reused by every session and size query.
///
/// **Each `GraphicsCaptureItem` costs the process one handle that releasing
/// the item does not return** -- measured 2026-10-29 at exactly +1.00 per item
/// over 200, for monitors and windows alike, with the WinRT device, pool,
/// handler, session and StartCapture each adding nothing. RapidShot made two
/// per camera (a size query and the session), so a program creating cameras
/// in a loop grew by ~2 handles each. Reusing the item per target bounds that
/// by the number of distinct targets.
static ITEMS: Mutex<Vec<(isize, bool, GraphicsCaptureItem)>> = Mutex::new(Vec::new());

/// Drop every cached item. For process exit: a WinRT capture object still
/// referenced when comtypes runs `CoUninitialize` crashes the interpreter on
/// the way out (2026-10-22 for open sessions, and again for cached items), so
/// Python's atexit hook calls this first. Returns how many were dropped.
pub fn release_items() -> usize {
    let mut items = ITEMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let count = items.len();
    items.clear();
    count
}

/// The cached item for `target`, created on first use. A cached item is
/// checked before it is handed out: handle values are recycled, so a closed
/// window's HWND -- or a disconnected display's HMONITOR -- can come back
/// naming something else, and its old item reports a 0x0 size.
fn item_for(target: Target) -> Result<GraphicsCaptureItem> {
    let (handle, is_window) = match target {
        Target::Monitor(h) => (h, false),
        Target::Window(h) => (h, true),
    };
    let mut items = ITEMS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(index) = items
        .iter()
        .position(|(h, w, _)| *h == handle && *w == is_window)
    {
        let item = &items[index].2;
        if !target_gone(target, item) {
            return Ok(item.clone());
        }
        items.remove(index);
    }
    let item = create_item(target)?;
    if items.len() >= MAX_ITEMS {
        items.remove(0);
    }
    items.push((handle, is_window, item.clone()));
    Ok(item)
}

/// The size WGC will capture `target` at, without starting a capture. What a
/// caller lays regions out against.
pub fn item_size(target: Target) -> Result<(u32, u32)> {
    ensure_winrt()?;
    let size = item_for(target)?.Size()?;
    Ok((size.Width.max(0) as u32, size.Height.max(0) as u32))
}

fn qpc_now() -> i64 {
    let mut value = 0i64;
    unsafe {
        let _ = QueryPerformanceCounter(&mut value);
    }
    value
}

fn qpc_frequency() -> i64 {
    let mut value = 0i64;
    unsafe {
        let _ = QueryPerformanceFrequency(&mut value);
    }
    value
}

/// The auto-reset event FrameArrived signals. Shared with the handler, which
/// WGC may still be running when the session closes, so the handle is closed
/// only when the last owner lets go -- never under a running handler.
struct Event(isize);

impl Event {
    fn new() -> Result<Self> {
        let handle = unsafe { CreateEventW(None, false, false, None)? };
        Ok(Self(handle.0 as isize))
    }

    fn handle(&self) -> HANDLE {
        HANDLE(self.0 as *mut _)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.handle());
        }
    }
}

/// The newest frame the pool has delivered, kept by WGC's own thread.
///
/// **Why the pool is drained as frames arrive, not when they are asked for.**
/// A pool frame nobody has taken holds a buffer, and with every buffer held
/// WGC drops new frames -- and never delivers them later. A caller that
/// grabbed less often than the source changed therefore got stale pixels:
/// measured 2026-10-29 with a 2-buffer pool, a window changed 3 times
/// between grabs came back showing its 2nd colour, and stayed that way while
/// the window held still. So the FrameArrived handler takes every frame as it
/// lands and keeps only the newest, closing the one it replaces; the pool
/// always has a buffer free for the next.
struct Latest {
    frame: Option<Direct3D11CaptureFrame>,
    /// QPC when `frame` was taken from the pool, on WGC's thread: its
    /// arrival, within the handler's latency.
    arrived_qpc: i64,
    /// Frames closed unread since the last acquire, because a newer one came.
    replaced: u32,
}

fn lock(latest: &Mutex<Latest>) -> MutexGuard<'_, Latest> {
    // Every mutation is a whole-field assignment; a panic leaves it coherent.
    latest
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Everything that holds a COM or WinRT reference. `close()` drops all of it,
/// so a closed session that outlives COM -- one still referenced from Python
/// when the interpreter exits, after comtypes has run `CoUninitialize` --
/// releases nothing late. Closing only the session and pool was not enough:
/// such a process crashed on exit (Machine B, 2026-10-22, every time), and
/// one that dropped its session before exiting did not.
struct Live {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    item: GraphicsCaptureItem,
    pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    arrived: Arc<Event>,
    arrived_token: i64,
    latest: Arc<Mutex<Latest>>,
    owned: Option<(ID3D11Texture2D, u32, u32)>,
}

pub struct WgcSession {
    /// None once closed.
    live: Option<Live>,
    target: Target,
    format: Format,
    size: SizeInt32,
    serial: u64,
    qpc_frequency: i64,
    /// The target went away (a closed window).
    gone: bool,
    /// References WGC itself took on the capture device for this session;
    /// `close()` waits for it to give them back (see there).
    wgc_device_refs: u32,
    /// Settings that were requested but that this Windows build refused.
    pub unsupported: Vec<&'static str>,
}

/// Pool buffers by default: one being copied by `acquire`, one holding the
/// newest frame, and one free for the frame after it. With two, a change that
/// lands while a frame is being copied finds no free buffer and is lost.
pub const DEFAULT_BUFFERS: i32 = 3;

/// Longest `close()` waits for WGC to finish tearing a session down.
const TEARDOWN_LIMIT: std::time::Duration = std::time::Duration::from_millis(1000);

/// Consecutive 1 ms samples the device's reference count must hold at its
/// target before the teardown counts as finished.
const SETTLED_SAMPLES: u32 = 10;

impl WgcSession {
    pub fn new(
        device: &ID3D11Device,
        target: Target,
        format: Format,
        buffers: i32,
        cursor: bool,
        border: bool,
        min_update_interval_100ns: i64,
    ) -> Result<Self> {
        if !(2..=8).contains(&buffers) {
            return Err(windows::core::Error::new(
                E_INVALIDARG,
                format!("buffers must be 2-8, got {buffers}"),
            ));
        }
        ensure_winrt()?;
        if !GraphicsCaptureSession::IsSupported()? {
            return Err(windows::core::Error::new(
                E_INVALIDARG,
                "Windows.Graphics.Capture is not supported on this system",
            ));
        }
        let item = item_for(target)?;
        let size = item.Size()?;
        if size.Width <= 0 || size.Height <= 0 {
            return Err(windows::core::Error::new(
                E_INVALIDARG,
                "the capture target has no size (a closed window?)",
            ));
        }

        let refs_before = refcount(device);
        let dxgi: IDXGIDevice = device.cast()?;
        let winrt_device: IDirect3DDevice =
            unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi)? }.cast()?;
        let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
            &winrt_device,
            format.winrt(),
            buffers,
            size,
        )?;

        let arrived = Arc::new(Event::new()?);
        let latest = Arc::new(Mutex::new(Latest {
            frame: None,
            arrived_qpc: 0,
            replaced: 0,
        }));
        let (signal, slot) = (arrived.clone(), latest.clone());
        // The pool arrives as the sender rather than being captured: a clone
        // held by its own handler would be a reference cycle.
        let arrived_token = pool.FrameArrived(&TypedEventHandler::<
            Direct3D11CaptureFramePool,
            IInspectable,
        >::new(move |sender, _| {
            if let Some(pool) = sender.as_ref() {
                while let Ok(frame) = pool.TryGetNextFrame() {
                    let mut latest = lock(&slot);
                    if let Some(older) = latest.frame.replace(frame) {
                        let _ = older.Close();
                        latest.replaced += 1;
                    }
                    latest.arrived_qpc = qpc_now();
                }
            }
            unsafe { SetEvent(signal.handle())? };
            Ok(())
        }))?;

        let session = pool.CreateCaptureSession(&item)?;
        let mut unsupported = Vec::new();
        if session.SetIsCursorCaptureEnabled(cursor).is_err() {
            unsupported.push("cursor");
        }
        if session.SetIsBorderRequired(border).is_err() {
            unsupported.push("border");
        }
        if session
            .SetMinUpdateInterval(TimeSpan {
                Duration: min_update_interval_100ns,
            })
            .is_err()
        {
            unsupported.push("min_update_interval");
        }
        session.StartCapture()?;
        // Before anything can close the session: from here WGC has teardown
        // work to finish on its own threads, in this DLL.
        pin_graphics_capture();
        // Everything WGC holds on the device now is its own; ours are taken
        // below. `dxgi` is still alive, so it is discounted.
        let wgc_device_refs = refcount(device).saturating_sub(refs_before + 1);
        drop(dxgi);

        let context = unsafe { device.GetImmediateContext()? };
        Ok(Self {
            live: Some(Live {
                device: device.clone(),
                context,
                item,
                pool,
                session,
                arrived,
                arrived_token,
                latest,
                owned: None,
            }),
            target,
            format,
            size,
            serial: 0,
            qpc_frequency: qpc_frequency(),
            gone: false,
            wgc_device_refs,
            unsupported,
        })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.size.Width as u32, self.size.Height as u32)
    }

    pub fn format(&self) -> Format {
        self.format
    }

    /// Wait up to `timeout_ms` for a frame newer than the last one acquired,
    /// and take it.
    pub fn acquire(&mut self, timeout_ms: u32) -> Result<Outcome> {
        if self.gone {
            return Ok(Outcome::Closed);
        }
        let Some(live) = self.live.as_mut() else {
            return Ok(Outcome::Closed);
        };
        ensure_winrt()?;

        let take = |live: &Live| {
            let mut latest = lock(&live.latest);
            let frame = latest.frame.take()?;
            let skipped = std::mem::take(&mut latest.replaced);
            Some((frame, latest.arrived_qpc, skipped))
        };
        let mut taken = take(live);
        if taken.is_none() {
            let deadline =
                std::time::Instant::now() + std::time::Duration::from_millis(timeout_ms as u64);
            loop {
                let left = deadline.saturating_duration_since(std::time::Instant::now());
                unsafe { WaitForSingleObject(live.arrived.handle(), left.as_millis() as u32) };
                taken = take(live);
                if taken.is_some() || left.is_zero() {
                    break;
                }
            }
        }
        let Some((frame, arrived_qpc, skipped)) = taken else {
            if target_gone(self.target, &live.item) {
                self.gone = true;
                return Ok(Outcome::Closed);
            }
            return Ok(Outcome::Timeout);
        };
        let acquired_qpc = qpc_now();
        let arrived_qpc = if arrived_qpc > 0 {
            arrived_qpc.min(acquired_qpc)
        } else {
            acquired_qpc
        };

        let content = frame.ContentSize()?;
        if content.Width != self.size.Width || content.Height != self.size.Height {
            let _ = frame.Close();
            return Ok(Outcome::Resized(
                content.Width.max(0) as u32,
                content.Height.max(0) as u32,
            ));
        }

        let result = copy_out(live, self.format, &frame);
        let system_relative = frame.SystemRelativeTime().map(|t| t.Duration).unwrap_or(0);
        // Back to the pool at once: the copy is all that is kept.
        let _ = frame.Close();
        let (texture, width, height) = result?;

        self.serial += 1;
        Ok(Outcome::Frame(Acquired {
            texture,
            width,
            height,
            dxgi_format: self.format.dxgi(),
            arrived_qpc,
            acquired_qpc,
            system_relative_qpc: (system_relative as i128 * self.qpc_frequency as i128 / 10_000_000)
                as i64,
            serial: self.serial,
            skipped,
        }))
    }

    /// Stop capturing and release every COM reference, then wait for WGC to
    /// finish tearing the session down. Idempotent. Returns how long the
    /// wait took, in milliseconds.
    ///
    /// **`Close()` returns before WGC has finished.** It completes the
    /// teardown on its own threads, and a process that starts exiting in the
    /// meantime crashes: a camera left open at interpreter exit crashed 8
    /// times in 10 and one released just before exit 1 in 10, whether the
    /// session lived in the main thread's apartment or a worker's, while half
    /// a second between close and exit made both 0 in 10 (Machine B,
    /// 2026-10-29). The signal that the teardown is over is the capture
    /// device's reference count: WGC holds references on it while the session
    /// runs, takes more as it tears down, and lets them all go 2-6 ms after
    /// `Close()`. So this waits until the count has fallen by the references
    /// WGC took at creation, bounded by `TEARDOWN_LIMIT`. (WGC drops the
    /// FrameArrived delegate synchronously, so that is no signal.)
    pub fn close(&mut self) -> f64 {
        let Some(live) = self.live.take() else {
            return 0.0;
        };
        let Live {
            device,
            context,
            item,
            pool,
            session,
            arrived,
            arrived_token,
            latest,
            owned,
        } = live;
        // Our own references first, so what remains is the caller's and WGC's.
        // `device` itself is kept: it guarantees the device outlives the wait.
        drop(owned);
        drop(context);
        let ours_released = refcount(&device);
        let target = ours_released.saturating_sub(self.wgc_device_refs);

        let _ = pool.RemoveFrameArrived(arrived_token);
        let _ = session.Close();
        if let Some(frame) = lock(&latest).frame.take() {
            let _ = frame.Close();
        }
        let _ = pool.Close();
        drop((session, pool, item, arrived, latest));

        // Settled, not merely touched: WGC takes references during the
        // teardown as well as dropping them, so the count can dip to the
        // target and climb again. Measured: a loop that stopped at the first
        // dip returned with WGC still holding 1-3 references.
        let started = std::time::Instant::now();
        let mut settled = 0;
        while settled < SETTLED_SAMPLES && started.elapsed() < TEARDOWN_LIMIT {
            std::thread::sleep(std::time::Duration::from_millis(1));
            settled = if refcount(&device) <= target {
                settled + 1
            } else {
                0
            };
        }
        started.elapsed().as_secs_f64() * 1000.0
    }
}

/// A COM object's reference count, by AddRef and Release.
fn refcount(device: &ID3D11Device) -> u32 {
    unsafe {
        let raw = device.as_raw();
        let unknown = &Interface::vtable(device).base__;
        (unknown.AddRef)(raw);
        (unknown.Release)(raw)
    }
}

impl Drop for WgcSession {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn target_gone(target: Target, item: &GraphicsCaptureItem) -> bool {
    if let Target::Window(handle) = target {
        if !unsafe { IsWindow(Some(HWND(handle as *mut _))) }.as_bool() {
            return true;
        }
    }
    matches!(item.Size(), Ok(s) if s.Width <= 0 || s.Height <= 0)
}

fn copy_out(
    live: &mut Live,
    format: Format,
    frame: &Direct3D11CaptureFrame,
) -> Result<(usize, u32, u32)> {
    let access: IDirect3DDxgiInterfaceAccess = frame.Surface()?.cast()?;
    let surface: ID3D11Texture2D = unsafe { access.GetInterface()? };
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { surface.GetDesc(&mut desc) };

    let reuse = matches!(&live.owned, Some((_, w, h)) if *w == desc.Width && *h == desc.Height);
    if !reuse {
        live.owned = Some((
            create_owned(&live.device, format, &desc)?,
            desc.Width,
            desc.Height,
        ));
    }
    let (owned, width, height) = live.owned.as_ref().expect("set above");
    unsafe { live.context.CopyResource(owned, &surface) };
    Ok((owned.as_raw() as usize, *width, *height))
}

fn create_owned(
    device: &ID3D11Device,
    format: Format,
    source: &D3D11_TEXTURE2D_DESC,
) -> Result<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: source.Width,
        Height: source.Height,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT(format.dxgi() as i32),
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
        CPUAccessFlags: 0,
        MiscFlags: OWNED_MISC_FLAGS,
    };
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    Ok(texture.expect("CreateTexture2D reported success"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::System::LibraryLoader::LoadLibraryW;

    #[test]
    fn formats_round_trip_to_the_dxgi_codes_the_converter_accepts() {
        assert_eq!(Format::parse("bgra").map(Format::dxgi), Some(87));
        assert_eq!(Format::parse("fp16").map(Format::dxgi), Some(10));
        assert_eq!(Format::parse("rgba"), None);
    }

    #[test]
    fn owned_texture_flags_are_shared_with_an_nt_handle_and_no_keyed_mutex() {
        assert_eq!(OWNED_MISC_FLAGS & 0x800, 0x800, "SHARED_NTHANDLE");
        assert_eq!(OWNED_MISC_FLAGS & 0x2, 0x2, "SHARED");
        assert_eq!(OWNED_MISC_FLAGS & 0x100, 0, "no keyed mutex");
    }

    #[test]
    fn graphics_capture_is_pinned_once_it_is_loaded() {
        // Loaded here as a session loads it. Where it cannot load (a Windows
        // without WGC), there is nothing to pin and nothing to test.
        let loaded = unsafe { LoadLibraryW(w!("GraphicsCapture.dll")) };
        if loaded.is_err() {
            return;
        }
        assert!(pin_graphics_capture());
        assert!(pin_graphics_capture(), "idempotent");
    }
}
