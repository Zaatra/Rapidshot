//! Desktop Duplication: one acquisition stream on one output.
//!
//! A port of `rapidshot/core/duplicator.py`, kept behaviour-for-behaviour so
//! the Python package and the Rust remote-support helper share one
//! implementation. Each rule below was learned on real hardware; the Python
//! file has the measurements behind them.
//!
//! **What it must keep doing** (each is a test in `tests/test_native_duplicator.py`
//! or below):
//!
//! 1. Prefer `IDXGIOutput5::DuplicateOutput1` with BGRA8, RGBA8, R10G10B10A2
//!    and RGBA16F offered, so HDR and 10-bit desktops duplicate; fall back to
//!    `DuplicateOutput` when IDXGIOutput5 is missing or DuplicateOutput1 fails
//!    for any reason but protected content, or when `RAPIDSHOT_DUPLICATE_OUTPUT`
//!    is `legacy`, `0` or `duplicateoutput`.
//! 2. A bare `E_ACCESSDENIED` is a desktop refusal (lock screen, UAC, Session 0)
//!    when `desktop::describe_desktop_access` says so, and protected content
//!    otherwise. The DXGI-specific denials are always protected content.
//! 3. HRESULTs keep the classes `ScreenCapture` acts on: access lost, session
//!    disconnected, mode change and the abandoned keyed mutex rebuild the
//!    duplication; device removed/reset/hung and driver internal error rebuild
//!    the device; `INVALID_CALL`/`UNSUPPORTED` are configuration errors that
//!    carry their HRESULT (so the caller can try the next adapter); anything
//!    else is a plain DXGI error.
//! 4. A timeout is healthy (`Ok(true)`, nothing updated). A rebuild-class or
//!    device-class failure drops the duplication before reporting, so nothing
//!    is issued against an interface DXGI has already invalidated.
//! 5. An acquire with `LastPresentTime == 0` (cursor only) holds a frame that
//!    must still be released, and is not new content.
//! 6. A new-content frame counts `frame_serial` before anything else can fail,
//!    and reads move rects then dirty rects while the frame is held. `None`
//!    means unreadable; empty means none reported. `MORE_DATA` retries once at
//!    the size DXGI asks for. Metadata above 16 MiB, or a pointer shape above
//!    16 MiB, is refused before allocating.
//! 7. The previous frame's texture is dropped before the next is taken:
//!    DXGI refuses the next acquire while any reference to it is outstanding.
//! 8. A cursor shape that cannot be read keeps the shape already held; only
//!    access lost propagates from that read.
//! 9. `release_frame` is idempotent and always clears the held flag;
//!    `INVALID_CALL` from `ReleaseFrame` means "already released".
//!
//! Messages are the Python file's, except that a COM error is rendered the way
//! windows-rs renders it ("text (0x887A0026)") rather than comtypes' tuple.

use std::ffi::c_void;
use std::mem::size_of;

use windows::core::{Interface, HRESULT};
use windows::Win32::Foundation::{E_ACCESSDENIED, RECT};
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
};
use windows::Win32::Graphics::Dxgi::*;

use crate::desktop::describe_desktop_access;

/// Upper bounds on buffers sized from driver-reported fields. A corrupt frame
/// info after a driver fault would otherwise become a multi-gigabyte
/// allocation on the capture thread.
pub const MAX_METADATA_BUFFER_BYTES: u32 = 16 * 1024 * 1024;
pub const MAX_POINTER_SHAPE_BUFFER_BYTES: u32 = 16 * 1024 * 1024;

/// Offered to DuplicateOutput1, in preference order.
pub const DUPLICATE_OUTPUT1_FORMATS: [DXGI_FORMAT; 4] = [
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
];

const DUPLICATE_OUTPUT_ENV: &str = "RAPIDSHOT_DUPLICATE_OUTPUT";

const RECOVERABLE: [HRESULT; 3] = [
    DXGI_ERROR_ACCESS_LOST,
    DXGI_ERROR_SESSION_DISCONNECTED,
    DXGI_ERROR_MODE_CHANGE_IN_PROGRESS,
];
// The abandoned keyed mutex (`ABANDONED_MUTEX_EXCEPTION` in the Python
// bindings) has the same value as DXGI_ERROR_ACCESS_LOST, so it is covered.
const DEVICE: [HRESULT; 4] = [
    DXGI_ERROR_DEVICE_REMOVED,
    DXGI_ERROR_DEVICE_RESET,
    DXGI_ERROR_DEVICE_HUNG,
    DXGI_ERROR_DRIVER_INTERNAL_ERROR,
];
const PROTECTED: [HRESULT; 3] = [
    DXGI_ERROR_ACCESS_DENIED,
    DXGI_ERROR_CANNOT_PROTECT_CONTENT,
    E_ACCESSDENIED,
];

/// A redrawn region, `(left, top, right, bottom)` in desktop coordinates.
pub type DirtyRect = (i32, i32, i32, i32);

/// A moved region: where it came from, then the rectangle it now occupies --
/// `(source_x, source_y, left, top, right, bottom)`.
pub type MoveRect = (i32, i32, i32, i32, i32, i32);

/// Which recovery a failure calls for; the Python exception it becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// `RapidShotReinitError`: rebuild the duplication.
    Reinit,
    /// `RapidShotDeviceError`: rebuild the device.
    Device,
    /// `RapidShotProtectedContentError`.
    ProtectedContent,
    /// `RapidShotConfigError`.
    Config,
    /// `RapidShotDXGIError`.
    Dxgi,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DupError {
    pub kind: ErrorKind,
    /// As a signed 32-bit value, the way the Python constants are written.
    pub hresult: Option<i32>,
    pub message: String,
}

impl std::fmt::Display for DupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DupError {}

/// Severity of a note for the caller's log. The Python adapter writes these
/// to the `rapidshot.core.duplicator` logger, where the Python class wrote them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Debug,
    Info,
    Warning,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PointerPosition {
    pub x: i32,
    pub y: i32,
    pub visible: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PointerShape {
    /// `DXGI_OUTDUPL_POINTER_SHAPE_TYPE`: 1 monochrome, 2 colour, 4 masked colour.
    pub kind: u32,
    pub width: u32,
    pub height: u32,
    pub pitch: u32,
    pub hot_spot: (i32, i32),
    pub data: Vec<u8>,
}

fn format_hresult(code: HRESULT) -> String {
    format!("{:#010x}", code.0 as u32)
}

fn legacy_duplication_forced() -> bool {
    std::env::var(DUPLICATE_OUTPUT_ENV)
        .map(|v| {
            matches!(
                v.trim().to_lowercase().as_str(),
                "legacy" | "0" | "duplicateoutput"
            )
        })
        .unwrap_or(false)
}

/// A desktop-specific error when that is what a bare `E_ACCESSDENIED` means.
fn desktop_refusal(code: HRESULT, detail: &str) -> Option<DupError> {
    if code != E_ACCESSDENIED {
        return None;
    }
    let reason = describe_desktop_access().blocked_reason?;
    let prefix = if detail.is_empty() {
        String::new()
    } else {
        format!("{detail} — ")
    };
    Some(DupError {
        kind: ErrorKind::Config,
        hresult: Some(code.0),
        message: format!("{prefix}Desktop duplication was denied: {reason}"),
    })
}

/// What a pointer-shape read produced.
enum ShapeRead {
    Shape(PointerShape),
    /// Nothing new; the message says why. The held shape stays.
    Unchanged(String),
}

pub struct Duplicator {
    duplication: Option<IDXGIOutputDuplication>,
    texture: Option<ID3D11Texture2D>,
    frame_acquired: bool,
    log: Vec<(Level, String)>,

    pub used_duplicate_output1: bool,
    pub timeout_ms: u32,
    pub updated: bool,
    pub frame_serial: u64,
    pub last_present_time: i64,
    pub accumulated_frames: u32,
    pub dirty_rects: Option<Vec<DirtyRect>>,
    pub move_rects: Option<Vec<MoveRect>>,
    pub rects_coalesced: bool,
    pub protected_content_detected: bool,
    pub cursor_visible: bool,
    pub pointer_position: PointerPosition,
    pub pointer_shape: Option<PointerShape>,
    pub last_error: String,
}

impl Duplicator {
    /// Duplicate `output` on `device`.
    pub fn new(
        output: &IDXGIOutput1,
        device: &ID3D11Device,
        timeout_ms: u32,
    ) -> Result<Self, DupError> {
        let mut dup = Duplicator {
            duplication: None,
            texture: None,
            frame_acquired: false,
            log: Vec::new(),
            used_duplicate_output1: false,
            timeout_ms,
            updated: false,
            frame_serial: 0,
            last_present_time: 0,
            accumulated_frames: 0,
            dirty_rects: None,
            move_rects: None,
            rects_coalesced: false,
            protected_content_detected: false,
            cursor_visible: false,
            pointer_position: PointerPosition::default(),
            pointer_shape: None,
            last_error: String::new(),
        };
        let duplication = dup.create_duplication(output, device)?;
        dup.duplication = Some(duplication);
        dup.note(
            Level::Info,
            format!(
                "Duplicator initialized (DuplicateOutput1={})",
                if dup.used_duplicate_output1 {
                    "True"
                } else {
                    "False"
                }
            ),
        );
        Ok(dup)
    }

    fn note(&mut self, level: Level, message: impl Into<String>) {
        self.log.push((level, message.into()));
    }

    /// Notes recorded since the last call, oldest first.
    pub fn take_log(&mut self) -> Vec<(Level, String)> {
        std::mem::take(&mut self.log)
    }

    /// The desktop image of the held frame, valid until `release_frame`.
    pub fn texture(&self) -> Option<&ID3D11Texture2D> {
        self.texture.as_ref()
    }

    pub fn frame_acquired(&self) -> bool {
        self.frame_acquired
    }

    pub fn is_released(&self) -> bool {
        self.duplication.is_none()
    }

    fn create_duplication(
        &mut self,
        output: &IDXGIOutput1,
        device: &ID3D11Device,
    ) -> Result<IDXGIOutputDuplication, DupError> {
        if legacy_duplication_forced() {
            self.note(
                Level::Info,
                format!(
                    "{DUPLICATE_OUTPUT_ENV} requests the legacy path; skipping DuplicateOutput1."
                ),
            );
        } else {
            match output.cast::<IDXGIOutput5>() {
                Err(e) => self.note(
                    Level::Info,
                    format!(
                        "IDXGIOutput5 unavailable (HRESULT {}); falling back to legacy DuplicateOutput.",
                        format_hresult(e.code())
                    ),
                ),
                Ok(output5) => {
                    match unsafe { output5.DuplicateOutput1(device, 0, &DUPLICATE_OUTPUT1_FORMATS) } {
                        Ok(duplication) => {
                            self.used_duplicate_output1 = true;
                            return Ok(duplication);
                        }
                        Err(e) => {
                            let code = e.code();
                            if PROTECTED.contains(&code) {
                                // A bare E_ACCESSDENIED means one of two unrelated
                                // things; ask the window station before blaming
                                // protected content for it.
                                if let Some(desktop) = desktop_refusal(code, "") {
                                    return Err(desktop);
                                }
                                self.protected_content_detected = true;
                                return Err(DupError {
                                    kind: ErrorKind::ProtectedContent,
                                    hresult: Some(code.0),
                                    message: "Desktop duplication was denied because protected \
                                              (HDCP/DRM) content is on screen. Close or move the \
                                              protected player window and retry."
                                        .to_owned(),
                                });
                            }
                            self.note(
                                Level::Warning,
                                format!(
                                    "DuplicateOutput1 failed (HRESULT {}); falling back to legacy DuplicateOutput.",
                                    format_hresult(code)
                                ),
                            );
                        }
                    }
                }
            }
        }
        unsafe { output.DuplicateOutput(device) }.map_err(|e| {
            let message = format!("Failed to initialize duplicator: {e}");
            self.note(Level::Warning, message.clone());
            self.last_error = message;
            self.map_error(&e, "Failed to initialize duplicator")
        })
    }

    /// The one place an HRESULT becomes an error class.
    fn map_error(&mut self, e: &windows::core::Error, context: &str) -> DupError {
        let code = e.code();
        let detail = format!("{context}: {e}");
        let (kind, message) = if PROTECTED.contains(&code) {
            if let Some(desktop) = desktop_refusal(code, &detail) {
                return desktop;
            }
            self.protected_content_detected = true;
            (
                ErrorKind::ProtectedContent,
                format!("{detail} (protected/HDCP content is blocking duplication)"),
            )
        } else if DEVICE.contains(&code) {
            (ErrorKind::Device, detail)
        } else if RECOVERABLE.contains(&code) {
            (ErrorKind::Reinit, detail)
        } else if code == DXGI_ERROR_INVALID_CALL || code == DXGI_ERROR_UNSUPPORTED {
            // Carries the HRESULT so the caller can tell "this adapter will not
            // duplicate this output" from a desktop refusal.
            (ErrorKind::Config, detail)
        } else {
            (ErrorKind::Dxgi, detail)
        };
        DupError {
            kind,
            hresult: Some(code.0),
            message,
        }
    }

    /// Try to acquire the next frame. `Ok(true)` while healthy, a timeout
    /// included; read `updated` for whether there is new content. `Ok(false)`
    /// once released.
    pub fn update_frame(&mut self) -> Result<bool, DupError> {
        self.updated = false;
        self.last_error.clear();
        self.frame_acquired = false;

        if self.duplication.is_none() {
            self.last_error = "update_frame called on a released duplicator".to_owned();
            self.note(Level::Debug, self.last_error.clone());
            return Ok(false);
        }

        match self.acquire_and_read() {
            Ok(healthy) => Ok(healthy),
            Err(e) => {
                let code = e.code();
                self.last_error = format!(
                    "COMError in update_frame: {e} (HRESULT: {})",
                    format_hresult(code)
                );
                if code == DXGI_ERROR_WAIT_TIMEOUT {
                    // Normal on a static desktop.
                    self.note(Level::Debug, "Frame acquisition timed out.");
                    self.updated = false;
                    return Ok(true);
                }
                self.note(Level::Warning, self.last_error.clone());
                if RECOVERABLE.contains(&code) {
                    let reason = if code == DXGI_ERROR_SESSION_DISCONNECTED {
                        "Session disconnected (RDP/fast user switch)"
                    } else {
                        "Access lost"
                    };
                    self.release_duplication();
                    return Err(DupError {
                        kind: ErrorKind::Reinit,
                        hresult: Some(code.0),
                        message: format!("{reason}, re-initialization needed: {e}"),
                    });
                }
                if DEVICE.contains(&code) {
                    self.release_duplication();
                    return Err(DupError {
                        kind: ErrorKind::Device,
                        hresult: Some(code.0),
                        message: format!("Device error, re-initialization needed: {e}"),
                    });
                }
                Err(self.map_error(&e, "Unexpected DXGI error in update_frame"))
            }
        }
    }

    /// Everything between AcquireNextFrame and handing back the texture. A COM
    /// error from any of it is classified by `update_frame`, as one handler
    /// covered all of it in Python.
    fn acquire_and_read(&mut self) -> windows::core::Result<bool> {
        let duplication = self.duplication.clone().expect("checked by update_frame");
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;
        unsafe { duplication.AcquireNextFrame(self.timeout_ms, &mut info, &mut resource) }?;
        self.frame_acquired = true;
        self.note(Level::Debug, "Frame acquired successfully");

        // Protected content is blanked by the compositor rather than failing
        // the acquire; surface it once.
        if info.ProtectedContentMaskedOut.as_bool() {
            if !self.protected_content_detected {
                self.note(
                    Level::Warning,
                    "Protected (HDCP/DRM) content is on screen; the affected region is blanked \
                     out by the OS in captured frames.",
                );
            }
            self.protected_content_detected = true;
        } else {
            self.protected_content_detected = false;
        }

        if info.LastMouseUpdateTime > 0 {
            match self.read_pointer_shape(&duplication, &info)? {
                ShapeRead::Shape(shape) => self.pointer_shape = Some(shape),
                ShapeRead::Unchanged(why) => {
                    self.note(Level::Debug, format!("Cursor shape not updated: {why}"))
                }
            }
            let p = info.PointerPosition;
            self.pointer_position = PointerPosition {
                x: p.Position.x,
                y: p.Position.y,
                visible: p.Visible.as_bool(),
            };
            self.cursor_visible = p.Visible.as_bool();
        }

        if info.LastPresentTime == 0 {
            self.note(Level::Debug, "No new frame content");
            self.updated = false;
            return Ok(true);
        }

        self.last_present_time = info.LastPresentTime;
        self.accumulated_frames = info.AccumulatedFrames;
        // Counted before anything below can fail: once acquired, this frame's
        // dirty rects are consumed whether or not the texture is usable.
        self.frame_serial += 1;
        // Move rects first: they occupy the front of the same metadata buffer.
        self.move_rects = self.read_move_rects(&duplication, &info)?;
        self.dirty_rects = self.read_dirty_rects(&duplication, &info)?;
        self.rects_coalesced = info.RectsCoalesced.as_bool();

        // Drop the previous image before taking the new one: DXGI refuses the
        // next acquire while a reference to the prior surface is outstanding.
        self.texture = None;
        match resource.map(|r| r.cast::<ID3D11Texture2D>()) {
            Some(Ok(texture)) => {
                self.texture = Some(texture);
                self.updated = true;
            }
            Some(Err(e)) => {
                self.last_error = format!("Failed to query texture interface: {e}");
                self.note(Level::Warning, self.last_error.clone());
                self.updated = false;
            }
            None => {
                self.last_error =
                    "Failed to query texture interface: no desktop resource".to_owned();
                self.note(Level::Warning, self.last_error.clone());
                self.updated = false;
            }
        }
        Ok(true)
    }

    fn refuse_metadata_size(&mut self, kind: &str, capacity: u32) {
        self.last_error = format!(
            "{kind} rect metadata of {capacity} bytes exceeds the \
             {MAX_METADATA_BUFFER_BYTES}-byte limit; treating it as unreadable"
        );
        self.note(Level::Warning, self.last_error.clone());
    }

    /// Read one kind of metadata rect. `None` is unreadable, empty is none.
    /// Rebuild- and device-class failures propagate to `update_frame`.
    fn read_rects<T: Default + Clone>(
        &mut self,
        kind: &str,
        capacity: u32,
        mut read: impl FnMut(u32, *mut T, *mut u32) -> windows::core::Result<()>,
    ) -> windows::core::Result<Option<Vec<T>>> {
        if self.duplication.is_none() || !self.frame_acquired {
            return Ok(None);
        }
        if capacity == 0 {
            return Ok(Some(Vec::new()));
        }
        let rect_size = size_of::<T>() as u32;
        let mut capacity = capacity;
        for _ in 0..2 {
            if capacity > MAX_METADATA_BUFFER_BYTES {
                self.refuse_metadata_size(kind, capacity);
                return Ok(None);
            }
            let count = (capacity / rect_size).max(1);
            let mut buffer = vec![T::default(); count as usize];
            let mut used = 0u32;
            match read(count * rect_size, buffer.as_mut_ptr(), &mut used) {
                Ok(()) => {
                    buffer.truncate((used / rect_size) as usize);
                    return Ok(Some(buffer));
                }
                Err(e) if e.code() == DXGI_ERROR_MORE_DATA => {
                    // DXGI wrote nothing and said what it needs; retry once.
                    capacity = used;
                    if capacity == 0 {
                        return Ok(None);
                    }
                }
                Err(e) => {
                    let code = e.code();
                    self.last_error = format!(
                        "COMError in get_frame_{kind}_rects: {e} (HRESULT: {})",
                        format_hresult(code)
                    );
                    self.note(Level::Debug, self.last_error.clone());
                    if RECOVERABLE.contains(&code) || DEVICE.contains(&code) {
                        return Err(e);
                    }
                    return Ok(None);
                }
            }
        }
        Ok(None)
    }

    fn read_move_rects(
        &mut self,
        duplication: &IDXGIOutputDuplication,
        info: &DXGI_OUTDUPL_FRAME_INFO,
    ) -> windows::core::Result<Option<Vec<MoveRect>>> {
        let rects = self.read_rects::<DXGI_OUTDUPL_MOVE_RECT>(
            "move",
            info.TotalMetadataBufferSize,
            |size, buffer, used| unsafe { duplication.GetFrameMoveRects(size, buffer, used) },
        )?;
        Ok(rects.map(|v| {
            v.iter()
                .map(|m| {
                    let d = m.DestinationRect;
                    (
                        m.SourcePoint.x,
                        m.SourcePoint.y,
                        d.left,
                        d.top,
                        d.right,
                        d.bottom,
                    )
                })
                .collect()
        }))
    }

    fn read_dirty_rects(
        &mut self,
        duplication: &IDXGIOutputDuplication,
        info: &DXGI_OUTDUPL_FRAME_INFO,
    ) -> windows::core::Result<Option<Vec<DirtyRect>>> {
        let rects = self.read_rects::<RECT>(
            "dirty",
            info.TotalMetadataBufferSize,
            |size, buffer, used| unsafe { duplication.GetFrameDirtyRects(size, buffer, used) },
        )?;
        Ok(rects.map(|v| {
            v.iter()
                .map(|r| (r.left, r.top, r.right, r.bottom))
                .collect()
        }))
    }

    fn read_pointer_shape(
        &mut self,
        duplication: &IDXGIOutputDuplication,
        info: &DXGI_OUTDUPL_FRAME_INFO,
    ) -> windows::core::Result<ShapeRead> {
        let size = info.PointerShapeBufferSize;
        if size == 0 {
            return Ok(ShapeRead::Unchanged("No cursor shape available".to_owned()));
        }
        if size > MAX_POINTER_SHAPE_BUFFER_BYTES {
            let message = format!(
                "pointer shape of {size} bytes exceeds the {MAX_POINTER_SHAPE_BUFFER_BYTES}-byte limit"
            );
            self.note(Level::Warning, message.clone());
            self.last_error = message.clone();
            return Ok(ShapeRead::Unchanged(message));
        }
        let mut data = vec![0u8; size as usize];
        let mut required = 0u32;
        let mut shape_info = DXGI_OUTDUPL_POINTER_SHAPE_INFO::default();
        match unsafe {
            duplication.GetFramePointerShape(
                size,
                data.as_mut_ptr() as *mut c_void,
                &mut required,
                &mut shape_info,
            )
        } {
            Ok(()) => {
                self.note(
                    Level::Debug,
                    format!(
                        "Cursor shape acquired: {}x{}, Type: {}",
                        shape_info.Width, shape_info.Height, shape_info.Type
                    ),
                );
                Ok(ShapeRead::Shape(PointerShape {
                    kind: shape_info.Type,
                    width: shape_info.Width,
                    height: shape_info.Height,
                    pitch: shape_info.Pitch,
                    hot_spot: (shape_info.HotSpot.x, shape_info.HotSpot.y),
                    data,
                }))
            }
            Err(e) => {
                let code = e.code();
                self.last_error = format!(
                    "COMError in get_frame_pointer_shape: {e} (HRESULT: {})",
                    format_hresult(code)
                );
                self.note(Level::Warning, self.last_error.clone());
                if code == DXGI_ERROR_ACCESS_LOST {
                    return Err(e);
                }
                if code == DXGI_ERROR_NOT_FOUND {
                    return Ok(ShapeRead::Unchanged(format!(
                        "Cursor shape not found (HRESULT: {})",
                        format_hresult(code)
                    )));
                }
                Ok(ShapeRead::Unchanged(self.last_error.clone()))
            }
        }
    }

    /// Give back the held frame, if any. Idempotent.
    pub fn release_frame(&mut self) {
        if !self.frame_acquired {
            self.note(Level::Debug, "ReleaseFrame called with no active frame");
            return;
        }
        // The texture is only valid until ReleaseFrame; drop our reference
        // first so DXGI sees none outstanding.
        self.texture = None;
        let Some(duplication) = self.duplication.clone() else {
            self.frame_acquired = false;
            return;
        };
        match unsafe { duplication.ReleaseFrame() } {
            Ok(()) => self.note(Level::Debug, "Frame released"),
            Err(e) if e.code() == DXGI_ERROR_INVALID_CALL => {
                self.note(Level::Debug, format!("Frame already released: {e}"))
            }
            Err(e) => {
                self.note(
                    Level::Warning,
                    format!(
                        "Failed to release frame: {e} (HRESULT: {})",
                        format_hresult(e.code())
                    ),
                );
                self.last_error = format!("Failed to release frame: {e}");
            }
        }
        // Always cleared: a failed release must not be retried against the
        // same frame, and a stuck flag makes every later acquire look like a leak.
        self.frame_acquired = false;
    }

    /// Drop the duplication without releasing a frame: DXGI has already
    /// invalidated it, so nothing more may be issued against it.
    fn release_duplication(&mut self) {
        self.duplication = None;
        self.frame_acquired = false;
        self.updated = false;
        self.texture = None;
    }

    /// Release the held frame and the duplication. Idempotent.
    pub fn release(&mut self) {
        if self.duplication.is_some() {
            if self.frame_acquired {
                self.release_frame();
            }
            self.duplication = None;
            self.frame_acquired = false;
            self.note(Level::Info, "Duplicator resources released.");
        }
    }
}

impl Drop for Duplicator {
    fn drop(&mut self) {
        self.release();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
    use windows::Win32::Graphics::Direct3D11::{
        D3D11CreateDevice, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION,
    };

    /// Tests run in parallel, and DXGI allows one duplication of an output at a
    /// time: every test that duplicates holds this while it does.
    static DUPLICATION: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn exclusive() -> std::sync::MutexGuard<'static, ()> {
        DUPLICATION.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn primary() -> (IDXGIOutput1, ID3D11Device) {
        unsafe {
            let factory: IDXGIFactory1 = CreateDXGIFactory1().unwrap();
            let adapter = factory.EnumAdapters1(0).unwrap();
            let output: IDXGIOutput1 = adapter.EnumOutputs(0).unwrap().cast().unwrap();
            let mut device = None;
            D3D11CreateDevice(
                &adapter,
                D3D_DRIVER_TYPE_UNKNOWN,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
            .unwrap();
            (output, device.unwrap())
        }
    }

    #[test]
    fn hresult_classes_match_the_python_sets() {
        // ABANDONED_MUTEX_EXCEPTION is -0x7785ffda in the Python bindings.
        assert_eq!(DXGI_ERROR_ACCESS_LOST.0, -0x7785ffda);
        assert!(RECOVERABLE.contains(&DXGI_ERROR_MODE_CHANGE_IN_PROGRESS));
        assert!(DEVICE.contains(&DXGI_ERROR_DRIVER_INTERNAL_ERROR));
        assert!(PROTECTED.contains(&E_ACCESSDENIED));
    }

    #[test]
    fn an_access_lost_error_is_a_rebuild_and_unsupported_carries_its_hresult() {
        let _only = exclusive();
        let (output, device) = primary();
        let mut dup = Duplicator::new(&output, &device, 10).unwrap();
        let lost = windows::core::Error::from_hresult(DXGI_ERROR_ACCESS_LOST);
        assert_eq!(dup.map_error(&lost, "x").kind, ErrorKind::Reinit);
        let unsupported = windows::core::Error::from_hresult(DXGI_ERROR_UNSUPPORTED);
        let e = dup.map_error(&unsupported, "x");
        assert_eq!(
            (e.kind, e.hresult),
            (ErrorKind::Config, Some(DXGI_ERROR_UNSUPPORTED.0))
        );
        let other = windows::core::Error::from_hresult(DXGI_ERROR_NOT_FOUND);
        assert_eq!(dup.map_error(&other, "x").kind, ErrorKind::Dxgi);
    }

    #[test]
    fn acquires_releases_and_stays_healthy_on_a_static_desktop() {
        let _only = exclusive();
        let (output, device) = primary();
        let mut dup = Duplicator::new(&output, &device, 50).unwrap();
        let mut got_content = false;
        for _ in 0..40 {
            assert!(dup.update_frame().unwrap(), "{}", dup.last_error);
            if dup.updated {
                got_content = true;
                assert!(dup.texture().is_some());
                assert!(dup.frame_serial >= 1);
            }
            dup.release_frame();
            assert!(!dup.frame_acquired());
            // Releasing twice is a no-op, not a DXGI call.
            dup.release_frame();
        }
        // The first acquire after creation always carries the whole desktop.
        assert!(got_content);
        dup.release();
        assert!(dup.is_released());
        assert_eq!(dup.update_frame(), Ok(false));
        dup.release();
    }
}
