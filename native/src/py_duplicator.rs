//! `DesktopDuplicator`: the core crate's Desktop Duplication, for Python.
//!
//! `rapidshot/core/native_duplicator.py` wraps it into a `CaptureBackend`. This
//! side only adapts: every call goes through a mutex, the acquire releases the
//! GIL while it waits (comtypes released it around the same COM call), errors
//! become `DuplicationError(kind, hresult, message)` for the adapter to turn
//! into the `RapidShot*Error` the class names, and the per-call notes are
//! handed back for the Python logger.

use std::sync::Mutex;

use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::ID3D11Device;
use windows::Win32::Graphics::Dxgi::{IDXGIOutput, IDXGIOutput1};

use rapidshot_core::duplicator::{DirtyRect, DupError, Duplicator, ErrorKind, Level, MoveRect};

create_exception!(
    _rapidshot_native,
    DuplicationError,
    pyo3::exceptions::PyRuntimeError,
    "A Desktop Duplication failure: args are (kind, hresult, message)."
);

/// The core Duplicator, movable between threads.
///
/// DXGI and D3D11 objects are free-threaded -- the Python Duplicator already
/// drives the same interfaces from its capture thread through comtypes -- but
/// windows-rs does not mark them `Send`. Every use is serialised by the mutex
/// in `DesktopDuplicator`, so no two threads touch them at once.
struct Shared(Duplicator);

// SAFETY: see above; access is only ever through `DesktopDuplicator::inner`.
unsafe impl Send for Shared {}

#[pyclass(name = "DesktopDuplicator")]
pub struct DesktopDuplicator {
    inner: Mutex<Shared>,
}

fn kind_name(kind: ErrorKind) -> &'static str {
    match kind {
        ErrorKind::Reinit => "reinit",
        ErrorKind::Device => "device",
        ErrorKind::ProtectedContent => "protected_content",
        ErrorKind::Config => "config",
        ErrorKind::Dxgi => "dxgi",
    }
}

fn duplication_error(e: DupError) -> PyErr {
    DuplicationError::new_err((kind_name(e.kind), e.hresult, e.message))
}

fn level_name(level: Level) -> &'static str {
    match level {
        Level::Debug => "debug",
        Level::Info => "info",
        Level::Warning => "warning",
    }
}

impl DesktopDuplicator {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Shared>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("DesktopDuplicator lock poisoned"))
    }
}

#[pymethods]
impl DesktopDuplicator {
    /// Duplicate the output at `output_ptr` (an `IDXGIOutput`) on the D3D11
    /// device at `device_ptr`. Both are borrowed; this takes its own references.
    #[new]
    #[pyo3(signature = (output_ptr, device_ptr, timeout_ms=10))]
    fn new(output_ptr: usize, device_ptr: usize, timeout_ms: u32) -> PyResult<Self> {
        if output_ptr == 0 || device_ptr == 0 {
            return Err(PyValueError::new_err(
                "output and device pointers must not be null",
            ));
        }
        let raw_output = output_ptr as *mut core::ffi::c_void;
        let raw_device = device_ptr as *mut core::ffi::c_void;
        let output = unsafe { IDXGIOutput::from_raw_borrowed(&raw_output) }
            .ok_or_else(|| PyValueError::new_err("could not read the output pointer"))?
            .cast::<IDXGIOutput1>()
            .map_err(|e| {
                PyRuntimeError::new_err(format!("the output is not an IDXGIOutput1: {e}"))
            })?;
        let device = unsafe { ID3D11Device::from_raw_borrowed(&raw_device) }
            .ok_or_else(|| PyValueError::new_err("could not read the device pointer"))?
            .clone();
        let duplicator =
            Duplicator::new(&output, &device, timeout_ms).map_err(duplication_error)?;
        Ok(Self {
            inner: Mutex::new(Shared(duplicator)),
        })
    }

    /// `CaptureBackend.update_frame`. Releases the GIL while it waits.
    fn update_frame(&self, py: Python<'_>) -> PyResult<bool> {
        let result = py.detach(|| match self.inner.lock() {
            Ok(mut guard) => Ok(guard.0.update_frame()),
            Err(_) => Err(()),
        });
        match result {
            Ok(Ok(healthy)) => Ok(healthy),
            Ok(Err(e)) => Err(duplication_error(e)),
            Err(()) => Err(PyRuntimeError::new_err("DesktopDuplicator lock poisoned")),
        }
    }

    fn release_frame(&self) -> PyResult<()> {
        self.lock()?.0.release_frame();
        Ok(())
    }

    fn release(&self) -> PyResult<()> {
        self.lock()?.0.release();
        Ok(())
    }

    /// Notes since the last call, as `(level, message)`, oldest first.
    fn take_log(&self) -> PyResult<Vec<(&'static str, String)>> {
        Ok(self
            .lock()?
            .0
            .take_log()
            .into_iter()
            .map(|(level, message)| (level_name(level), message))
            .collect())
    }

    /// The held frame's `ID3D11Texture2D`, borrowed until `release_frame`; 0
    /// when none is held.
    #[getter]
    fn texture_address(&self) -> PyResult<usize> {
        Ok(self.lock()?.0.texture().map_or(0, |t| t.as_raw() as usize))
    }

    #[getter]
    fn timeout_ms(&self) -> PyResult<u32> {
        Ok(self.lock()?.0.timeout_ms)
    }

    #[setter]
    fn set_timeout_ms(&self, value: u32) -> PyResult<()> {
        self.lock()?.0.timeout_ms = value;
        Ok(())
    }

    #[getter]
    fn updated(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.updated)
    }

    #[getter]
    fn frame_acquired(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.frame_acquired())
    }

    #[getter]
    fn released(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.is_released())
    }

    #[getter]
    fn frame_serial(&self) -> PyResult<u64> {
        Ok(self.lock()?.0.frame_serial)
    }

    #[getter]
    fn last_present_time(&self) -> PyResult<i64> {
        Ok(self.lock()?.0.last_present_time)
    }

    #[getter]
    fn accumulated_frames(&self) -> PyResult<u32> {
        Ok(self.lock()?.0.accumulated_frames)
    }

    #[getter]
    fn dirty_rects(&self) -> PyResult<Option<Vec<DirtyRect>>> {
        Ok(self.lock()?.0.dirty_rects.clone())
    }

    #[getter]
    fn move_rects(&self) -> PyResult<Option<Vec<MoveRect>>> {
        Ok(self.lock()?.0.move_rects.clone())
    }

    #[getter]
    fn rects_coalesced(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.rects_coalesced)
    }

    #[getter]
    fn protected_content_detected(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.protected_content_detected)
    }

    #[getter]
    fn cursor_visible(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.cursor_visible)
    }

    #[getter]
    fn used_duplicate_output1(&self) -> PyResult<bool> {
        Ok(self.lock()?.0.used_duplicate_output1)
    }

    #[getter]
    fn last_error(&self) -> PyResult<String> {
        Ok(self.lock()?.0.last_error.clone())
    }

    /// `(x, y, visible)` of the last pointer update.
    #[getter]
    fn pointer_position(&self) -> PyResult<(i32, i32, bool)> {
        let p = self.lock()?.0.pointer_position;
        Ok((p.x, p.y, p.visible))
    }

    /// `(type, width, height, pitch, hot_x, hot_y, bytes)` of the cursor shape
    /// held, or None before one has been read.
    #[getter]
    #[allow(clippy::type_complexity)]
    fn pointer_shape<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(u32, u32, u32, u32, i32, i32, Bound<'py, PyBytes>)>> {
        let guard = self.lock()?;
        Ok(guard.0.pointer_shape.as_ref().map(|s| {
            (
                s.kind,
                s.width,
                s.height,
                s.pitch,
                s.hot_spot.0,
                s.hot_spot.1,
                PyBytes::new(py, &s.data),
            )
        }))
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DesktopDuplicator>()?;
    m.add("DuplicationError", m.py().get_type::<DuplicationError>())?;
    Ok(())
}
