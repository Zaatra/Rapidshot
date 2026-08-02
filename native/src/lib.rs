//! Native GPU interop shim for Rapidshot (Stage 6).
//!
//! Scope discipline matters here. Profiling showed the DXGI capture loop costs
//! ~0.003 ms/frame in Python binding overhead, so it stays in Python. What
//! Python *cannot* do is hand a D3D11 texture to DirectML: ONNX Runtime's
//! `OrtDmlApi::CreateGPUAllocationFromD3DResource` has no Python binding, and
//! the `data_ptr()` of a DML `OrtValue` is a host-side opaque handle that cannot
//! be synthesised from a texture. That single gap is what this crate exists to
//! close — nothing else belongs here.
//!
//! What this crate does: receive the `ID3D11Texture2D` that
//! `frame.d3d11_texture` exposes, share it to a D3D12 device, and run a compute
//! shader that converts it to a linear NCHW float32 tensor resident on the
//! DirectML device. Callers bind that resource to ONNX Runtime themselves —
//! `output_resource_address` is the contract. Owning ORT's bindings would couple
//! this library to its ABI and release cadence for one optional feature.
//!
//! The diagnostics here (`probe_*`) exist because each answered a question that
//! changed the design, and would otherwise have to be rediscovered.

mod cross_adapter;
mod preprocess;
mod preprocess12;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use std::sync::Mutex;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D, D3D11_TEXTURE2D_DESC};

/// Human-readable name for the DXGI formats desktop duplication can produce.
fn format_name(format: u32) -> &'static str {
    match format {
        87 => "B8G8R8A8_UNORM",
        28 => "R8G8B8A8_UNORM",
        24 => "R10G10B10A2_UNORM",
        10 => "R16G16B16A16_FLOAT",
        _ => "other",
    }
}

/// Borrow a raw COM pointer as an `ID3D11Texture2D` and run `f` against it.
///
/// The Python side owns this reference for the lifetime of its `Frame`, and
/// releasing it here would corrupt that refcount — the precise failure mode
/// that stalled capture before Stage 1. `from_raw_borrowed` yields a reference
/// that does not run `Release` on drop.
///
/// This takes a closure rather than returning the reference because the
/// borrowed handle is tied to a local raw pointer; handing it back out would
/// let it outlive the value it borrows from.
unsafe fn with_texture<T>(
    pointer: usize,
    f: impl FnOnce(&ID3D11Texture2D) -> PyResult<T>,
) -> PyResult<T> {
    if pointer == 0 {
        return Err(PyValueError::new_err(
            "texture pointer is null; the Frame was probably already released",
        ));
    }
    let raw = pointer as *mut core::ffi::c_void;
    let texture = ID3D11Texture2D::from_raw_borrowed(&raw).ok_or_else(|| {
        PyRuntimeError::new_err("could not interpret the pointer as an ID3D11Texture2D")
    })?;
    f(texture)
}

/// Read a texture's description.
unsafe fn texture_desc(pointer: usize) -> PyResult<D3D11_TEXTURE2D_DESC> {
    with_texture(pointer, |texture| {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        texture.GetDesc(&mut desc);
        Ok(desc)
    })
}

/// Read back an `ID3D11Texture2D`'s description.
///
/// Takes the integer address of the texture — obtained on the Python side from
/// a live `Frame` — and returns its dimensions, format and binding flags.
///
/// The caller must keep the `Frame` alive (i.e. call this inside the `with`
/// block); this borrows the texture and never releases it.
#[pyfunction]
fn describe_texture(py: Python<'_>, pointer: usize) -> PyResult<Py<PyDict>> {
    let desc = unsafe { texture_desc(pointer)? };

    let out = PyDict::new(py);
    out.set_item("width", desc.Width)?;
    out.set_item("height", desc.Height)?;
    out.set_item("mip_levels", desc.MipLevels)?;
    out.set_item("array_size", desc.ArraySize)?;
    out.set_item("format", desc.Format.0)?;
    out.set_item("format_name", format_name(desc.Format.0 as u32))?;
    out.set_item("sample_count", desc.SampleDesc.Count)?;
    out.set_item("usage", desc.Usage.0)?;
    out.set_item("bind_flags", desc.BindFlags)?;
    out.set_item("cpu_access_flags", desc.CPUAccessFlags)?;
    out.set_item("misc_flags", desc.MiscFlags)?;
    Ok(out.into())
}

/// Report whether the texture can be shared with another device (e.g. D3D12
/// for DirectML).
///
/// The duplicated desktop surface turns out to carry
/// `SHARED_NTHANDLE | SHARED_KEYEDMUTEX | RESTRICT_SHARED_RESOURCE`
/// (`misc_flags = 0x2900`), so it *is* shareable and opens on D3D12 directly —
/// no intermediate copy needed. That was not a given, hence this reporting every
/// flag by name rather than a single verdict.
#[pyfunction]
fn texture_sharing_info(py: Python<'_>, pointer: usize) -> PyResult<Py<PyDict>> {
    const SHARED: u32 = 0x2;
    const SHARED_KEYEDMUTEX: u32 = 0x100;
    const GDI_COMPATIBLE: u32 = 0x200;
    const SHARED_NTHANDLE: u32 = 0x800;
    const RESTRICTED_CONTENT: u32 = 0x1000;
    const RESTRICT_SHARED_RESOURCE: u32 = 0x2000;
    const RESTRICT_SHARED_RESOURCE_DRIVER: u32 = 0x4000;
    const GUARDED: u32 = 0x8000;

    let desc = unsafe { texture_desc(pointer)? };
    let misc = desc.MiscFlags;

    let out = PyDict::new(py);
    out.set_item("misc_flags", misc)?;
    out.set_item("shared", misc & SHARED != 0)?;
    out.set_item("shared_nthandle", misc & SHARED_NTHANDLE != 0)?;
    out.set_item("keyed_mutex", misc & SHARED_KEYEDMUTEX != 0)?;
    out.set_item("gdi_compatible", misc & GDI_COMPATIBLE != 0)?;
    out.set_item("restricted_content", misc & RESTRICTED_CONTENT != 0)?;
    out.set_item(
        "restrict_shared_resource",
        misc & RESTRICT_SHARED_RESOURCE != 0,
    )?;
    out.set_item(
        "restrict_shared_resource_driver",
        misc & RESTRICT_SHARED_RESOURCE_DRIVER != 0,
    )?;
    out.set_item("guarded", misc & GUARDED != 0)?;

    // Names of every flag actually set, so unexpected combinations are visible
    // rather than hidden behind a bitmask.
    let mut names: Vec<&str> = Vec::new();
    for (bit, name) in [
        (SHARED, "SHARED"),
        (SHARED_KEYEDMUTEX, "SHARED_KEYEDMUTEX"),
        (GDI_COMPATIBLE, "GDI_COMPATIBLE"),
        (SHARED_NTHANDLE, "SHARED_NTHANDLE"),
        (RESTRICTED_CONTENT, "RESTRICTED_CONTENT"),
        (RESTRICT_SHARED_RESOURCE, "RESTRICT_SHARED_RESOURCE"),
        (
            RESTRICT_SHARED_RESOURCE_DRIVER,
            "RESTRICT_SHARED_RESOURCE_DRIVER",
        ),
        (GUARDED, "GUARDED"),
    ] {
        if misc & bit != 0 {
            names.push(name);
        }
    }
    out.set_item("flags", names)?;

    // Whether the surface can be opened on another device at all. Being
    // shareable is necessary but not sufficient: RESTRICT_SHARED_RESOURCE
    // narrows who may open the handle, so this is a hint for the interop work
    // rather than a guarantee.
    let shareable = misc & (SHARED | SHARED_NTHANDLE) != 0;
    out.set_item("shareable", shareable)?;
    out.set_item("needs_intermediate_copy", !shareable)?;
    Ok(out.into())
}

/// Return the address of the `ID3D11Device` that owns a texture.
///
/// The DML interop has to run against the same device the capture used, so
/// being able to recover it from the texture avoids threading a second handle
/// through the Python API.
///
/// The address is **borrowed**: the `AddRef` from `GetDevice` is balanced by the
/// `Release` when this function returns. That is deliberate — leaking a
/// reference Python would never drop is worse — and the device stays alive
/// regardless, because both the texture and the Python-side `Device` hold their
/// own references. Do not use the address after the capture session is released.
#[pyfunction]
fn get_device_pointer(pointer: usize) -> PyResult<usize> {
    unsafe {
        with_texture(pointer, |texture| {
            let device: ID3D11Device = texture.GetDevice().map_err(|e| {
                PyRuntimeError::new_err(format!("texture reported no owning device: {e}"))
            })?;
            Ok(device.as_raw() as usize)
        })
    }
}

/// Probe whether the captured surface can be opened on a D3D12 device.
///
/// This is the question milestone 2 exists to answer. DirectML runs on D3D12,
/// so getting the duplicated desktop surface onto a D3D12 device is the
/// precondition for zero-copy inference. Milestone 1 found the surface already
/// carries `SHARED_NTHANDLE`, which suggests it can be opened directly rather
/// than staged through an intermediate copy — but it also carries
/// `RESTRICT_SHARED_RESOURCE`, which narrows who may open the handle, so this
/// has to be tested rather than assumed.
///
/// Every step reports separately: a failure here should say *which* part of the
/// chain broke, not just that the chain broke.
#[pyfunction]
fn probe_d3d12_sharing(py: Python<'_>, pointer: usize) -> PyResult<Py<PyDict>> {
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device, ID3D12Resource};
    use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice, IDXGIResource1};

    const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;

    let out = PyDict::new(py);
    let mut shared_handle = HANDLE::default();

    let result: PyResult<()> = unsafe {
        with_texture(pointer, |texture| {
            // 1. The texture must expose IDXGIResource1 to produce an NT handle.
            let resource: IDXGIResource1 = match texture.cast() {
                Ok(r) => {
                    out.set_item("idxgiresource1", true)?;
                    r
                }
                Err(e) => {
                    out.set_item("idxgiresource1", false)?;
                    out.set_item("failed_at", "QueryInterface(IDXGIResource1)")?;
                    out.set_item("error", format!("{e}"))?;
                    return Ok(());
                }
            };

            // 2. Create the shared NT handle.
            shared_handle = match resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None)
            {
                Ok(h) => {
                    out.set_item("create_shared_handle", true)?;
                    out.set_item("handle", h.0 as usize)?;
                    h
                }
                Err(e) => {
                    out.set_item("create_shared_handle", false)?;
                    out.set_item("failed_at", "IDXGIResource1::CreateSharedHandle")?;
                    out.set_item("error", format!("{e}"))?;
                    out.set_item(
                        "interpretation",
                        "The surface cannot produce an NT handle, so the interop \
                         must copy into an intermediate shared resource first.",
                    )?;
                    return Ok(());
                }
            };

            // 3. Find the adapter the capture device is on. DirectML has to run
            //    on the same adapter or the handle is meaningless.
            let d3d11_device: ID3D11Device = texture
                .GetDevice()
                .map_err(|e| PyRuntimeError::new_err(format!("GetDevice failed: {e}")))?;
            let dxgi_device: IDXGIDevice = d3d11_device
                .cast()
                .map_err(|e| PyRuntimeError::new_err(format!("cast to IDXGIDevice failed: {e}")))?;
            let adapter: IDXGIAdapter = match dxgi_device.GetAdapter() {
                Ok(a) => {
                    out.set_item("got_adapter", true)?;
                    a
                }
                Err(e) => {
                    out.set_item("got_adapter", false)?;
                    out.set_item("failed_at", "IDXGIDevice::GetAdapter")?;
                    out.set_item("error", format!("{e}"))?;
                    return Ok(());
                }
            };

            // 4. Create a D3D12 device on that same adapter.
            let mut d3d12: Option<ID3D12Device> = None;
            if let Err(e) = D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut d3d12) {
                out.set_item("d3d12_device", false)?;
                out.set_item("failed_at", "D3D12CreateDevice")?;
                out.set_item("error", format!("{e}"))?;
                out.set_item(
                    "interpretation",
                    "No D3D12 device on the capture adapter, so DirectML cannot \
                     run there at all.",
                )?;
                return Ok(());
            }
            let d3d12 = d3d12.expect("D3D12CreateDevice reported success");
            out.set_item("d3d12_device", true)?;

            // 5. The actual question: open the D3D11 surface on D3D12.
            let mut resource12: Option<ID3D12Resource> = None;
            match d3d12.OpenSharedHandle(shared_handle, &mut resource12) {
                Ok(()) => {
                    let resource12 = resource12.ok_or_else(|| {
                        PyRuntimeError::new_err(
                            "OpenSharedHandle reported success but returned nothing",
                        )
                    })?;
                    let desc = resource12.GetDesc();
                    out.set_item("open_on_d3d12", true)?;
                    out.set_item("d3d12_width", desc.Width)?;
                    out.set_item("d3d12_height", desc.Height)?;
                    out.set_item("d3d12_format", desc.Format.0)?;
                    out.set_item("zero_copy_possible", true)?;
                    out.set_item(
                        "interpretation",
                        "The captured surface opens directly on D3D12. DirectML \
                         can bind it without an intermediate copy.",
                    )?;
                }
                Err(e) => {
                    out.set_item("open_on_d3d12", false)?;
                    out.set_item("failed_at", "ID3D12Device::OpenSharedHandle")?;
                    out.set_item("error", format!("{e}"))?;
                    out.set_item("zero_copy_possible", false)?;
                    out.set_item(
                        "interpretation",
                        "Likely RESTRICT_SHARED_RESOURCE refusing a cross-API \
                         open. Fall back to copying the surface into an \
                         intermediate resource created with sharing flags we \
                         control -- still GPU-local, still far cheaper than the \
                         CPU round-trip.",
                    )?;
                }
            }
            Ok(())
        })
    };

    // The handle is ours; leaking one per probe would exhaust the process.
    if !shared_handle.is_invalid() {
        unsafe {
            let _ = CloseHandle(shared_handle);
        }
    }
    result?;
    Ok(out.into())
}

/// GPU preprocessor: captured texture -> NCHW float32 tensor, on the GPU.
///
/// Built once for a given output size and reused per frame, so steady-state
/// capture does no allocation. Replaces the staging read, colour conversion,
/// resize and normalisation that the CPU path spends ~8 ms/frame on at 1080p.
#[pyclass(name = "GpuPreprocessor", unsendable)]
struct GpuPreprocessor {
    inner: Mutex<preprocess::Preprocessor>,
}

#[pymethods]
impl GpuPreprocessor {
    /// Build a preprocessor bound to the device that owns `texture_ptr`.
    ///
    /// Args:
    ///     texture_ptr: address of a live ID3D11Texture2D (from a Frame)
    ///     out_width / out_height: model input size; the shader resizes to this
    #[new]
    fn new(texture_ptr: usize, out_width: u32, out_height: u32) -> PyResult<Self> {
        if out_width == 0 || out_height == 0 {
            return Err(PyValueError::new_err("output dimensions must be non-zero"));
        }
        let device = unsafe {
            with_texture(texture_ptr, |texture| {
                texture.GetDevice().map_err(|e| {
                    PyRuntimeError::new_err(format!("could not get the capture device: {e}"))
                })
            })?
        };
        let inner = preprocess::Preprocessor::new(device, out_width, out_height)
            .map_err(|e| PyRuntimeError::new_err(format!("preprocessor setup failed: {e}")))?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Convert one frame. Stays entirely on the GPU.
    ///
    /// Args:
    ///     texture_ptr: address of a live ID3D11Texture2D
    ///     scale / bias: applied as `value * scale + bias` after the 0..1 fetch.
    ///         Defaults give 0..1; pass scale=2.0, bias=-1.0 for -1..1.
    ///     bgr: emit BGR channel order instead of RGB
    #[pyo3(signature = (texture_ptr, scale=1.0, bias=0.0, bgr=false))]
    fn process(&self, texture_ptr: usize, scale: f32, bias: f32, bgr: bool) -> PyResult<()> {
        let inner = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("preprocessor lock poisoned by an earlier panic")
        })?;
        unsafe {
            with_texture(texture_ptr, |texture| {
                inner
                    .dispatch(texture, scale, bias, if bgr { 1 } else { 0 })
                    .map_err(|e| PyRuntimeError::new_err(format!("dispatch failed: {e}")))
            })
        }
    }

    /// Copy the tensor back to the CPU as a flat NCHW float list.
    ///
    /// Verification and debugging only — reading back reintroduces exactly the
    /// CPU round-trip this path exists to avoid.
    fn read_back(&self) -> PyResult<Vec<f32>> {
        let inner = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("preprocessor lock poisoned by an earlier panic")
        })?;
        inner
            .read_back()
            .map_err(|e| PyRuntimeError::new_err(format!("readback failed: {e}")))
    }

    /// Shape of the produced tensor, as (1, 3, H, W).
    #[getter]
    fn shape(&self) -> PyResult<(u32, u32, u32, u32)> {
        let inner = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("preprocessor lock poisoned by an earlier panic")
        })?;
        Ok((1, 3, inner.out_height, inner.out_width))
    }

    /// Address of the GPU output buffer, for milestone 3b's DirectML binding.
    #[getter]
    fn output_buffer_address(&self) -> PyResult<usize> {
        let inner = self.inner.lock().map_err(|_| {
            PyRuntimeError::new_err("preprocessor lock poisoned by an earlier panic")
        })?;
        Ok(inner.output_buffer_address())
    }
}

/// Check that ONNX Runtime can be reached from the native shim.
///
/// This is the safe half of the DirectML binding. `OrtApiBase` has exactly two
/// members and its layout is fixed forever, so traversing it carries no risk —
/// unlike `OrtApi`, which has hundreds of function pointers and would need the
/// header to index correctly.
///
/// Loading the DLL by name at runtime (rather than linking against it) keeps
/// ONNX Runtime an optional dependency: users who never do GPU inference are not
/// asked to install it.
#[pyfunction]
#[pyo3(signature = (dll_path=None))]
fn probe_onnxruntime(py: Python<'_>, dll_path: Option<String>) -> PyResult<Py<PyDict>> {
    use windows::core::{PCSTR, PCWSTR};
    use windows::Win32::Foundation::FreeLibrary;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};

    /// The two-member root of the ONNX Runtime C API. Stable by contract.
    #[repr(C)]
    struct OrtApiBase {
        get_api: Option<unsafe extern "system" fn(u32) -> *const std::ffi::c_void>,
        get_version_string: Option<unsafe extern "system" fn() -> *const i8>,
    }

    let out = PyDict::new(py);
    let name = dll_path.unwrap_or_else(|| "onnxruntime.dll".to_string());
    out.set_item("dll", &name)?;

    let wide: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    let module = match unsafe { LoadLibraryW(PCWSTR(wide.as_ptr())) } {
        Ok(m) => m,
        Err(e) => {
            out.set_item("loaded", false)?;
            out.set_item("error", format!("{e}"))?;
            out.set_item(
                "hint",
                "onnxruntime.dll was not found on the DLL search path. It ships \
                 with the onnxruntime Python package; add that package's \
                 directory to PATH, or pass an explicit path.",
            )?;
            return Ok(out.into());
        }
    };
    out.set_item("loaded", true)?;

    let symbol = unsafe { GetProcAddress(module, PCSTR(b"OrtGetApiBase\0".as_ptr())) };
    let Some(symbol) = symbol else {
        out.set_item("ort_get_api_base", false)?;
        unsafe {
            let _ = FreeLibrary(module);
        }
        return Ok(out.into());
    };
    out.set_item("ort_get_api_base", true)?;

    let get_api_base: unsafe extern "system" fn() -> *const OrtApiBase =
        unsafe { std::mem::transmute(symbol) };
    let base = unsafe { get_api_base() };
    if base.is_null() {
        out.set_item("api_base", false)?;
        unsafe {
            let _ = FreeLibrary(module);
        }
        return Ok(out.into());
    }
    out.set_item("api_base", true)?;

    unsafe {
        if let Some(get_version) = (*base).get_version_string {
            let ptr = get_version();
            if !ptr.is_null() {
                let text = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
                out.set_item("version", text)?;
            }
        }
        // Probe which API versions this runtime supports. GetApi returns null
        // for versions it does not implement, so this is a safe query — and it
        // is what makes a pinned struct layout safe: the C API is append-only,
        // so requesting version N guarantees N's layout for members up to N.
        let mut supported: Vec<u32> = Vec::new();
        if let Some(get_api) = (*base).get_api {
            for version in 1..=25u32 {
                if !get_api(version).is_null() {
                    supported.push(version);
                }
            }
        }
        out.set_item("max_api_version", supported.last().copied().unwrap_or(0))?;
        out.set_item("supported_api_versions", supported)?;
        let _ = FreeLibrary(module);
    }
    Ok(out.into())
}

/// GPU preprocessor running on D3D12, where DirectML can reach the output.
///
/// Functionally identical to `GpuPreprocessor`, but the tensor lands in a D3D12
/// `DEFAULT`-heap buffer on the DirectML device instead of a D3D11 buffer that
/// can never get there — D3D11 shares only 2D textures, never buffers.
#[pyclass(name = "GpuPreprocessor12", unsendable)]
struct GpuPreprocessor12 {
    inner: Mutex<preprocess12::Preprocessor12>,
}

#[pymethods]
impl GpuPreprocessor12 {
    #[new]
    fn new(texture_ptr: usize, out_width: u32, out_height: u32) -> PyResult<Self> {
        if out_width == 0 || out_height == 0 {
            return Err(PyValueError::new_err("output dimensions must be non-zero"));
        }
        let inner = unsafe {
            with_texture(texture_ptr, |texture| {
                preprocess12::Preprocessor12::new(texture, out_width, out_height).map_err(|e| {
                    PyRuntimeError::new_err(format!("D3D12 preprocessor setup failed: {e}"))
                })
            })?
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    #[pyo3(signature = (texture_ptr, scale=1.0, bias=0.0, bgr=false))]
    fn process(&self, texture_ptr: usize, scale: f32, bias: f32, bgr: bool) -> PyResult<()> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preprocessor lock poisoned"))?;
        unsafe {
            with_texture(texture_ptr, |texture| {
                inner
                    .process(texture, scale, bias, if bgr { 1 } else { 0 })
                    .map_err(|e| PyRuntimeError::new_err(format!("D3D12 dispatch failed: {e}")))
            })
        }
    }

    /// Copy the tensor back to the CPU. Verification only — doing this in
    /// production reintroduces the round-trip the whole path exists to avoid.
    fn read_back(&self) -> PyResult<Vec<f32>> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preprocessor lock poisoned"))?;
        inner
            .read_back()
            .map_err(|e| PyRuntimeError::new_err(format!("D3D12 readback failed: {e}")))
    }

    #[getter]
    fn shape(&self) -> PyResult<(u32, u32, u32, u32)> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preprocessor lock poisoned"))?;
        Ok((1, 3, inner.out_height, inner.out_width))
    }

    /// Address of the `ID3D12Resource` holding the tensor. This is what
    /// `OrtDmlApi::CreateGPUAllocationFromD3DResource` will take.
    #[getter]
    fn output_resource_address(&self) -> PyResult<usize> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preprocessor lock poisoned"))?;
        Ok(inner.output_resource_address())
    }

    /// GPU virtual address of the output buffer.
    #[getter]
    fn output_gpu_address(&self) -> PyResult<u64> {
        let inner = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("preprocessor lock poisoned"))?;
        Ok(inner.output_gpu_address())
    }
}

/// Moves a captured frame to a second GPU (ROADMAP.md § 6.1).
///
/// On a hybrid laptop Desktop Duplication only runs against the adapter driving
/// the display, so a GPU-resident frame lands on the iGPU while the caller's
/// model lives on the dGPU. This carries it across, via a shared cross-adapter
/// heap, without a CPU round-trip.
///
/// Build one per capture session and reuse it: the heap and both placed
/// resources are allocated in the constructor, so only the copy is per-frame.
#[pyclass(name = "CrossAdapterTransfer", unsendable)]
struct CrossAdapterTransfer {
    inner: Mutex<cross_adapter::CrossAdapterTransfer>,
}

#[pymethods]
impl CrossAdapterTransfer {
    #[new]
    fn new(texture_ptr: usize) -> PyResult<Self> {
        let inner = unsafe {
            with_texture(texture_ptr, |texture| {
                cross_adapter::CrossAdapterTransfer::new(texture).map_err(|e| {
                    PyRuntimeError::new_err(format!("cross-adapter setup failed: {e}"))
                })
            })?
        };
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Copy one frame across. Blocks until the source GPU has finished, so the
    /// frame is readable on the destination adapter when this returns.
    fn transfer(&self, texture_ptr: usize) -> PyResult<()> {
        let inner = self.lock()?;
        unsafe {
            with_texture(texture_ptr, |texture| {
                inner
                    .transfer(texture)
                    .map_err(|e| PyRuntimeError::new_err(format!("transfer failed: {e}")))
            })
        }
    }

    /// Transfer a frame and return a source-side copy of the same bytes, taken
    /// in the same command list.
    ///
    /// Verification only. This is the reference `read_back_destination` is
    /// compared against: the duplicated surface is live, so two separately
    /// submitted copies genuinely see different pixels, and comparing against a
    /// CPU capture does not work either — Desktop Duplication reports only
    /// changed content, so consecutive frames differ by construction.
    fn transfer_with_reference(&self, texture_ptr: usize) -> PyResult<Vec<u8>> {
        let inner = self.lock()?;
        unsafe {
            with_texture(texture_ptr, |texture| {
                inner
                    .transfer_with_reference(texture)
                    .map_err(|e| PyRuntimeError::new_err(format!("reference transfer failed: {e}")))
            })
        }
    }

    /// Read the frame back through the destination device.
    ///
    /// Verification only: in production a consumer on that adapter binds
    /// `destination_resource_address` and never touches the CPU. Rows are
    /// `row_pitch` bytes apart, which is padded to D3D12's 256-byte alignment
    /// and so is not always `width * 4`.
    fn read_back_destination(&self) -> PyResult<Vec<u8>> {
        let inner = self.lock()?;
        inner
            .read_back_destination()
            .map_err(|e| PyRuntimeError::new_err(format!("destination readback failed: {e}")))
    }

    /// Address of the `ID3D12Resource` on the destination adapter. Borrowed,
    /// not owned: valid only while this object is alive.
    #[getter]
    fn destination_resource_address(&self) -> PyResult<usize> {
        Ok(self.lock()?.destination_resource_address())
    }

    /// Address of the destination `ID3D12Device`, for a consumer that must
    /// build its own resources on the same device.
    #[getter]
    fn destination_device_address(&self) -> PyResult<usize> {
        Ok(self.lock()?.destination_device_address())
    }

    #[getter]
    fn source(&self) -> PyResult<String> {
        Ok(self.lock()?.source_name.clone())
    }

    #[getter]
    fn destination(&self) -> PyResult<String> {
        Ok(self.lock()?.destination_name.clone())
    }

    /// True when the only second adapter is WARP. The path is exercised in
    /// full, but timings from it say nothing about a real iGPU-to-dGPU move.
    #[getter]
    fn destination_is_software(&self) -> PyResult<bool> {
        Ok(self.lock()?.destination_is_software)
    }

    #[getter]
    fn total_bytes(&self) -> PyResult<u64> {
        Ok(self.lock()?.total_bytes())
    }

    #[getter]
    fn row_pitch(&self) -> PyResult<u32> {
        Ok(self.lock()?.row_pitch())
    }

    #[getter]
    fn width(&self) -> PyResult<u32> {
        Ok(self.lock()?.width)
    }

    #[getter]
    fn height(&self) -> PyResult<u32> {
        Ok(self.lock()?.height)
    }
}

impl CrossAdapterTransfer {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, cross_adapter::CrossAdapterTransfer>> {
        self.inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("cross-adapter transfer lock poisoned"))
    }
}

/// Find a buffer configuration that a compute shader can write AND that can be
/// shared with D3D12 for DirectML.
///
/// Milestone 3a hit a wall: D3D11 rejects `BUFFER_STRUCTURED` combined with
/// `SHARED_NTHANDLE` (E_INVALIDARG), so the shader's output buffer cannot reach
/// D3D12 as-is. Rather than guess which alternative works, this tries each
/// candidate and reports how far it gets — buffer creation, shared-handle
/// creation, then opening on D3D12.
///
/// Every candidate includes `BIND_UNORDERED_ACCESS`, since a configuration the
/// shader cannot write to is useless regardless of how well it shares.
#[pyfunction]
fn probe_shareable_buffers(py: Python<'_>) -> PyResult<Py<PyDict>> {
    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D11::{
        ID3D11Buffer, D3D11_BIND_SHADER_RESOURCE, D3D11_BIND_UNORDERED_ACCESS, D3D11_BUFFER_DESC,
        D3D11_USAGE_DEFAULT,
    };
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device, ID3D12Resource};
    use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice, IDXGIResource1};

    const SHARED: u32 = 0x2;
    const ALLOW_RAW_VIEWS: u32 = 0x20;
    const STRUCTURED: u32 = 0x40;
    const SHARED_KEYEDMUTEX: u32 = 0x100;
    const SHARED_NTHANDLE: u32 = 0x800;
    const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;

    // (label, misc flags, structure stride)
    let candidates: [(&str, u32, u32); 6] = [
        (
            "structured + NTHANDLE + keyedmutex",
            STRUCTURED | SHARED_NTHANDLE | SHARED_KEYEDMUTEX,
            4,
        ),
        ("structured + SHARED (legacy)", STRUCTURED | SHARED, 4),
        (
            "raw + NTHANDLE + keyedmutex",
            ALLOW_RAW_VIEWS | SHARED_NTHANDLE | SHARED_KEYEDMUTEX,
            0,
        ),
        ("raw + SHARED (legacy)", ALLOW_RAW_VIEWS | SHARED, 0),
        (
            "plain + NTHANDLE + keyedmutex",
            SHARED_NTHANDLE | SHARED_KEYEDMUTEX,
            0,
        ),
        ("plain + SHARED (legacy)", SHARED, 0),
    ];

    let device = create_test_device()
        .map_err(|e| PyRuntimeError::new_err(format!("could not create a D3D11 device: {e}")))?;

    // A D3D12 device on the same adapter, to test the open step.
    let d3d12: Option<ID3D12Device> = unsafe {
        (|| {
            let dxgi: IDXGIDevice = device.cast().ok()?;
            let adapter: IDXGIAdapter = dxgi.GetAdapter().ok()?;
            let mut dev: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut dev).ok()?;
            dev
        })()
    };

    let out = PyDict::new(py);
    out.set_item("d3d12_available", d3d12.is_some())?;
    let results = PyDict::new(py);

    for (label, misc, stride) in candidates {
        let entry = PyDict::new(py);
        let desc = D3D11_BUFFER_DESC {
            ByteWidth: 4096,
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
            CPUAccessFlags: 0,
            MiscFlags: misc,
            StructureByteStride: stride,
        };

        let mut buffer: Option<ID3D11Buffer> = None;
        let created = unsafe { device.CreateBuffer(&desc, None, Some(&mut buffer)) };
        match created {
            Ok(()) => entry.set_item("created", true)?,
            Err(e) => {
                entry.set_item("created", false)?;
                entry.set_item("error", format!("{e}"))?;
                entry.set_item("usable", false)?;
                results.set_item(label, entry)?;
                continue;
            }
        }
        let buffer = buffer.expect("CreateBuffer reported success");

        // Can it produce a handle D3D12 can open?
        let mut handle = HANDLE::default();
        let shared_ok = unsafe {
            match buffer.cast::<IDXGIResource1>() {
                Ok(res) => match res.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None) {
                    Ok(h) => {
                        handle = h;
                        true
                    }
                    Err(e) => {
                        entry.set_item("share_error", format!("{e}"))?;
                        false
                    }
                },
                Err(e) => {
                    entry.set_item("share_error", format!("no IDXGIResource1: {e}"))?;
                    false
                }
            }
        };
        entry.set_item("shared_handle", shared_ok)?;

        let mut opened = false;
        if shared_ok {
            if let Some(ref dev12) = d3d12 {
                let mut res12: Option<ID3D12Resource> = None;
                match unsafe { dev12.OpenSharedHandle(handle, &mut res12) } {
                    Ok(()) => opened = res12.is_some(),
                    Err(e) => entry.set_item("open_error", format!("{e}"))?,
                }
            }
            unsafe {
                let _ = CloseHandle(handle);
            }
        }
        entry.set_item("opened_on_d3d12", opened)?;
        // Usable means: the shader can write it AND DirectML can see it.
        entry.set_item("usable", opened)?;
        results.set_item(label, entry)?;
    }

    out.set_item("candidates", results)?;
    Ok(out.into())
}

/// Create a D3D11 device for test fixtures and probes, preferring real hardware.
///
/// Falls back to WARP, Microsoft's software rasteriser, so these paths still run
/// on machines with no GPU — CI runners in particular. WARP is slow but
/// functionally complete for compute, which is all the correctness tests need.
fn create_test_device() -> windows::core::Result<ID3D11Device> {
    use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP};
    use windows::Win32::Graphics::Direct3D11::{D3D11CreateDevice, D3D11_SDK_VERSION};

    let mut last_error = None;
    for driver in [D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_WARP] {
        let mut device: Option<ID3D11Device> = None;
        let result = unsafe {
            D3D11CreateDevice(
                None,
                driver,
                None,
                Default::default(),
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                None,
            )
        };
        match result {
            Ok(()) => {
                if let Some(device) = device {
                    return Ok(device);
                }
            }
            Err(e) => last_error = Some(e),
        }
    }
    Err(last_error.unwrap_or_else(|| {
        windows::core::Error::new(
            windows::Win32::Foundation::E_FAIL,
            "no D3D11 device could be created, with hardware or WARP",
        )
    }))
}

/// A texture with caller-supplied contents, for deterministic testing.
///
/// Verifying the preprocessor against live capture is unreliable: Desktop
/// Duplication only reports *changed* content, so an idle screen produces no
/// frames, and two captures taken moments apart are not guaranteed identical.
/// This creates a texture whose exact contents Python chose, so the shader's
/// output can be compared against a reference with no ambiguity.
#[pyclass(name = "TestTexture", unsendable)]
struct TestTexture {
    // Field order matters: the texture must drop before the device that made it.
    texture: ID3D11Texture2D,
    #[allow(dead_code)]
    device: ID3D11Device,
    #[pyo3(get)]
    width: u32,
    #[pyo3(get)]
    height: u32,
}

#[pymethods]
impl TestTexture {
    /// Build a BGRA8 texture from raw bytes (4 per pixel, BGRA order).
    ///
    /// `shared` adds the sharing flags the real duplicated desktop surface
    /// carries, which is required to exercise the D3D12 path but changes how the
    /// resource must be used — hence being able to turn it off.
    #[new]
    #[pyo3(signature = (width, height, data, shared=false, keyed_mutex=true))]
    fn new(
        width: u32,
        height: u32,
        data: Vec<u8>,
        shared: bool,
        keyed_mutex: bool,
    ) -> PyResult<Self> {
        use windows::Win32::Graphics::Direct3D11::{
            D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        };
        use windows::Win32::Graphics::Dxgi::Common::{
            DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC,
        };

        let expected = (width as usize) * (height as usize) * 4;
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "expected {expected} bytes for {width}x{height} BGRA, got {}",
                data.len()
            )));
        }

        let device = create_test_device().map_err(|e| {
            PyRuntimeError::new_err(format!("could not create a D3D11 device: {e}"))
        })?;

        let desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: if shared {
                // SHARED_NTHANDLE, optionally with SHARED_KEYEDMUTEX
                0x800u32 | if keyed_mutex { 0x100 } else { 0 }
            } else {
                0
            },
        };
        // Shared textures cannot be created with initial data — D3D11 ignores
        // pInitialData for resources carrying sharing flags, leaving the surface
        // zeroed. Create it empty and upload separately.
        let texture = unsafe {
            let mut texture = None;
            device
                .CreateTexture2D(&desc, None, Some(&mut texture))
                .map_err(|e| PyRuntimeError::new_err(format!("CreateTexture2D failed: {e}")))?;
            texture.expect("CreateTexture2D reported success")
        };

        unsafe {
            let context = device
                .GetImmediateContext()
                .map_err(|e| PyRuntimeError::new_err(format!("GetImmediateContext failed: {e}")))?;
            context.UpdateSubresource(&texture, 0, None, data.as_ptr() as *const _, width * 4, 0);
            // Make the upload visible before any other device opens the shared
            // handle; otherwise readers can observe the zeroed surface.
            context.Flush();
        }

        Ok(Self {
            texture,
            device,
            width,
            height,
        })
    }

    /// Address of the texture, for passing to the preprocessor.
    #[getter]
    fn pointer(&self) -> usize {
        use windows::core::Interface;
        self.texture.as_raw() as usize
    }
}

/// Build identity, so Python can report which native shim it loaded.
#[pyfunction]
fn build_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("version", env!("CARGO_PKG_VERSION"))?;
    out.set_item("stage", "6-m3b-d3d12-preprocess")?;
    Ok(out.into())
}

#[pymodule]
fn _rapidshot_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(describe_texture, m)?)?;
    m.add_function(wrap_pyfunction!(texture_sharing_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_device_pointer, m)?)?;
    m.add_function(wrap_pyfunction!(probe_d3d12_sharing, m)?)?;
    m.add_function(wrap_pyfunction!(probe_shareable_buffers, m)?)?;
    m.add_function(wrap_pyfunction!(cross_adapter::probe_cross_adapter, m)?)?;
    m.add_function(wrap_pyfunction!(
        cross_adapter::probe_cross_adapter_buffer,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(probe_onnxruntime, m)?)?;
    m.add_function(wrap_pyfunction!(build_info, m)?)?;
    m.add_class::<GpuPreprocessor>()?;
    m.add_class::<GpuPreprocessor12>()?;
    m.add_class::<TestTexture>()?;
    m.add_class::<CrossAdapterTransfer>()?;
    Ok(())
}
