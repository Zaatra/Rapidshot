//! Python entry points for the core crate's CPU kernels and diagnostic probes.
//!
//! Each one only adapts: parse the Python arguments, release the GIL where the
//! work touches no Python objects, call `rapidshot_core`, and turn its result
//! into the Python value or exception it always produced. The signatures and
//! messages are the ones 2.7.1 shipped.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rapidshot_core::cross_adapter::{self, ProbeError};
use rapidshot_core::report::{Report, Value};
use rapidshot_core::{luma, swizzle};

/// Convert a BGRA image to 8-bit luma, writing into a caller-owned buffer.
/// See `rapidshot_core::luma::bgra_to_gray_into`.
#[pyfunction]
#[pyo3(signature = (src_ptr, src_len, dst_ptr, dst_len, width, height, pitch=None,
                    dst_pitch=None))]
#[allow(clippy::too_many_arguments)]
pub fn bgra_to_gray_into(
    py: Python<'_>,
    src_ptr: usize,
    src_len: usize,
    dst_ptr: usize,
    dst_len: usize,
    width: usize,
    height: usize,
    pitch: Option<usize>,
    dst_pitch: Option<usize>,
) -> PyResult<()> {
    py.detach(|| {
        luma::bgra_to_gray_into(
            src_ptr, src_len, dst_ptr, dst_len, width, height, pitch, dst_pitch,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Reorder a BGRA image into `mode`, writing into a caller-owned buffer.
/// See `rapidshot_core::swizzle::bgra_swizzle_into`.
#[pyfunction]
#[pyo3(signature = (src_ptr, src_len, dst_ptr, dst_len, width, height, mode,
                    pitch=None, dst_pitch=None))]
#[allow(clippy::too_many_arguments)]
pub fn bgra_swizzle_into(
    py: Python<'_>,
    src_ptr: usize,
    src_len: usize,
    dst_ptr: usize,
    dst_len: usize,
    width: usize,
    height: usize,
    mode: &str,
    pitch: Option<usize>,
    dst_pitch: Option<usize>,
) -> PyResult<()> {
    let mode = mode.to_owned();
    py.detach(|| {
        swizzle::bgra_swizzle_into(
            src_ptr, src_len, dst_ptr, dst_len, width, height, &mode, pitch, dst_pitch,
        )
    })
    .map_err(PyValueError::new_err)
}

/// Probe cross-adapter sharing, and time the capture-side copy into it.
#[pyfunction]
#[pyo3(signature = (width = 1920, height = 1080, iterations = 50))]
pub fn probe_cross_adapter(
    py: Python<'_>,
    width: u32,
    height: u32,
    iterations: usize,
) -> PyResult<Py<PyDict>> {
    let report =
        cross_adapter::probe_cross_adapter(width, height, iterations).map_err(probe_error)?;
    to_dict(py, &report)
}

/// Time a cross-adapter buffer copy of `size_bytes`.
#[pyfunction]
#[pyo3(signature = (size_bytes, iterations = 50))]
pub fn probe_cross_adapter_buffer(
    py: Python<'_>,
    size_bytes: u64,
    iterations: usize,
) -> PyResult<Py<PyDict>> {
    let report =
        cross_adapter::probe_cross_adapter_buffer(size_bytes, iterations).map_err(probe_error)?;
    to_dict(py, &report)
}

fn probe_error(e: ProbeError) -> PyErr {
    match e {
        ProbeError::Value(m) => PyValueError::new_err(m),
        ProbeError::Runtime(m) => PyRuntimeError::new_err(m),
    }
}

/// A `Report` as the `dict` the probes used to build directly.
pub fn to_dict(py: Python<'_>, report: &Report) -> PyResult<Py<PyDict>> {
    Ok(fill(py, report)?.unbind())
}

fn fill<'py>(py: Python<'py>, report: &Report) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    for (key, value) in &report.fields {
        match value {
            Value::Bool(v) => out.set_item(key, v)?,
            Value::Int(v) => out.set_item(key, v)?,
            Value::Float(v) => out.set_item(key, v)?,
            Value::Str(v) => out.set_item(key, v)?,
            Value::Missing => out.set_item(key, py.None())?,
            Value::List(items) => {
                let list = PyList::empty(py);
                for item in items {
                    list.append(fill(py, item)?)?;
                }
                out.set_item(key, list)?;
            }
        }
    }
    Ok(out)
}
