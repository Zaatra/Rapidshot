//! Cross-adapter sharing probe (ROADMAP.md § 6.1).
//!
//! On a hybrid-GPU system Desktop Duplication only runs against the adapter
//! that drives the display, so a GPU-resident frame is produced on the iGPU
//! while the caller's model usually lives on the dGPU. Getting it there means a
//! cross-adapter copy.
//!
//! Two things are worth correcting before reading the numbers:
//!
//! * **Cross-adapter shared heaps live in system memory**, not in either
//!   adapter's VRAM. This is not peer-to-peer VRAM-to-VRAM DMA. The win is that
//!   the GPU's copy engine moves the bytes instead of CPU cores.
//! * The mechanism is `D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER`, not
//!   `IDXGIAdapter3` (which is video-memory budgeting).
//!
//! This probe measures the half of the trip Rapidshot is responsible for: the
//! copy from a GPU-local texture on the *capture* adapter into the shared heap.
//! What the consumer adapter then pays to read it is its own device's cost and
//! cannot be measured meaningfully from here.
//!
//! A cross-adapter *buffer* is used rather than a row-major texture, because
//! `CrossAdapterRowMajorTextureSupported` is optional and false on plenty of
//! hardware. The buffer path works either way; the capability is reported so
//! the difference is visible.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use std::cell::Cell;

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, IDXGIAdapter, IDXGIAdapter1, IDXGIDevice, IDXGIFactory1, IDXGIResource1,
    DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_ERROR_NOT_FOUND,
};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

/// Generic access rights for a shared heap handle.
const GENERIC_ALL: u32 = 0x1000_0000;

const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;

struct Adapter {
    index: usize,
    adapter: IDXGIAdapter1,
    description: String,
    is_software: bool,
    output_count: usize,
    /// Identifies the adapter across APIs. Compared rather than the index,
    /// because enumeration order is not a stable identity.
    luid: (u32, i32),
}

/// DXGI descriptions are a fixed-size UTF-16 buffer padded with NULs.
fn trim_description(raw: &[u16]) -> String {
    let end = raw.iter().position(|&c| c == 0).unwrap_or(raw.len());
    String::from_utf16_lossy(&raw[..end])
}

impl Adapter {
    /// Only an adapter with an output can run Desktop Duplication, so this is
    /// the one that must hold the captured frame.
    fn drives_display(&self) -> bool {
        self.output_count > 0
    }
}

fn enumerate_adapters() -> windows::core::Result<Vec<Adapter>> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let mut adapters = Vec::new();
    let mut index = 0u32;
    loop {
        let adapter: IDXGIAdapter1 = match unsafe { factory.EnumAdapters1(index) } {
            Ok(a) => a,
            Err(e) if e.code() == DXGI_ERROR_NOT_FOUND => break,
            Err(e) => return Err(e),
        };
        let desc: DXGI_ADAPTER_DESC1 = unsafe { adapter.GetDesc1()? };

        let mut output_count = 0usize;
        while unsafe { adapter.EnumOutputs(output_count as u32) }.is_ok() {
            output_count += 1;
        }

        adapters.push(Adapter {
            index: index as usize,
            adapter,
            description: trim_description(&desc.Description),
            is_software: (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) != 0,
            output_count,
            luid: (desc.AdapterLuid.LowPart, desc.AdapterLuid.HighPart),
        });
        index += 1;
    }
    Ok(adapters)
}

/// Does this device support row-major cross-adapter textures?
///
/// Reported, not depended on: when false the shared resource has to be a
/// buffer, which is why this probe uses one unconditionally.
fn row_major_texture_supported(device: &ID3D12Device) -> Option<bool> {
    let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
    let ok = unsafe {
        device.CheckFeatureSupport(
            D3D12_FEATURE_D3D12_OPTIONS,
            &mut options as *mut _ as *mut core::ffi::c_void,
            std::mem::size_of::<D3D12_FEATURE_DATA_D3D12_OPTIONS>() as u32,
        )
    };
    ok.ok()
        .map(|_| options.CrossAdapterRowMajorTextureSupported.as_bool())
}

fn make_device(adapter: &IDXGIAdapter1) -> windows::core::Result<ID3D12Device> {
    let mut device: Option<ID3D12Device> = None;
    unsafe { D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, &mut device)? };
    Ok(device.expect("D3D12CreateDevice reported success"))
}

/// A GPU-local BGRA texture standing in for a captured frame.
fn make_local_texture(
    device: &ID3D12Device,
    width: u32,
    height: u32,
) -> windows::core::Result<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Alignment: 0,
        Width: width as u64,
        Height: height,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource reported success"))
}

/// Describe the buffer that lives in the cross-adapter heap.
///
/// `ALLOW_CROSS_ADAPTER` is what makes it placeable in a heap created with
/// `SHARED_CROSS_ADAPTER`; without it CreatePlacedResource fails.
fn cross_adapter_buffer_desc(size: u64) -> D3D12_RESOURCE_DESC {
    D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
    }
}

/// Probe cross-adapter sharing, and time the capture-side copy into it.
///
/// Args:
///     width, height: frame size to simulate (defaults to 1920x1080).
///     iterations: timed copies. The reported figure is the minimum, which is
///         the least noise-contaminated estimate of the real cost — see the
///         benchmark noise note in ROADMAP.md § 2.
#[pyfunction]
#[pyo3(signature = (width = 1920, height = 1080, iterations = 50))]
pub fn probe_cross_adapter(
    py: Python<'_>,
    width: u32,
    height: u32,
    iterations: usize,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);

    let adapters = enumerate_adapters()
        .map_err(|e| PyRuntimeError::new_err(format!("could not enumerate adapters: {e}")))?;

    let listing = PyList::empty(py);
    for a in &adapters {
        let entry = PyDict::new(py);
        entry.set_item("index", a.index)?;
        entry.set_item("description", &a.description)?;
        entry.set_item("software", a.is_software)?;
        entry.set_item("outputs", a.output_count)?;
        listing.append(entry)?;
    }
    out.set_item("adapters", listing)?;

    // Source = the adapter capture is bound to: it drives a display. Fall back
    // to the first hardware adapter so the probe still says something on a
    // headless box.
    let source = adapters
        .iter()
        .find(|a| a.drives_display() && !a.is_software)
        .or_else(|| adapters.iter().find(|a| !a.is_software))
        .or_else(|| adapters.first());
    let Some(source) = source else {
        out.set_item("supported", false)?;
        out.set_item("reason", "no DXGI adapters")?;
        return Ok(out.into());
    };
    // Destination = any other adapter. A hardware one is the real case; WARP is
    // the only second adapter on a single-GPU machine and still exercises the
    // whole share/open path.
    let dest = adapters
        .iter()
        .find(|a| a.index != source.index && !a.is_software)
        .or_else(|| adapters.iter().find(|a| a.index != source.index));
    let Some(dest) = dest else {
        out.set_item("supported", false)?;
        out.set_item("reason", "only one adapter on this system")?;
        return Ok(out.into());
    };

    out.set_item("source", &source.description)?;
    out.set_item("destination", &dest.description)?;
    out.set_item("destination_is_software", dest.is_software)?;
    // A WARP destination proves the mechanism but says nothing about the cost
    // of a real iGPU->dGPU transfer.
    out.set_item("representative", !dest.is_software)?;

    let source_device = make_device(&source.adapter)
        .map_err(|e| PyRuntimeError::new_err(format!("D3D12 device on source failed: {e}")))?;
    let dest_device = match make_device(&dest.adapter) {
        Ok(d) => d,
        Err(e) => {
            out.set_item("supported", false)?;
            out.set_item("reason", format!("D3D12 device on destination failed: {e}"))?;
            return Ok(out.into());
        }
    };

    out.set_item(
        "source_row_major_texture",
        row_major_texture_supported(&source_device),
    )?;
    out.set_item(
        "destination_row_major_texture",
        row_major_texture_supported(&dest_device),
    )?;

    match run_probe(
        &out,
        &source_device,
        &dest_device,
        width,
        height,
        iterations,
    ) {
        Ok(()) => {}
        Err(e) => {
            out.set_item("supported", false)?;
            out.set_item("reason", format!("{e}"))?;
        }
    }

    Ok(out.into())
}

fn run_probe(
    out: &Bound<'_, PyDict>,
    source_device: &ID3D12Device,
    dest_device: &ID3D12Device,
    width: u32,
    height: u32,
    iterations: usize,
) -> windows::core::Result<()> {
    let local = make_local_texture(source_device, width, height)?;
    let local_desc = unsafe { local.GetDesc() };

    // How many bytes the frame occupies once laid out linearly, including the
    // 256-byte row-pitch alignment D3D12 requires for copies.
    let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
    let mut total_bytes = 0u64;
    unsafe {
        source_device.GetCopyableFootprints(
            &local_desc,
            0,
            1,
            0,
            Some(&mut footprint),
            None,
            None,
            Some(&mut total_bytes),
        );
    }

    let heap_desc = D3D12_HEAP_DESC {
        SizeInBytes: total_bytes,
        Properties: D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
            MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
            CreationNodeMask: 1,
            VisibleNodeMask: 1,
        },
        Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
        Flags: D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
    };
    let mut source_heap: Option<ID3D12Heap> = None;
    unsafe { source_device.CreateHeap(&heap_desc, &mut source_heap)? };
    let source_heap = source_heap.expect("CreateHeap reported success");
    let _ = out.set_item("heap_bytes", total_bytes);

    // Share the *heap*, not the resource: cross-adapter placement is a heap
    // property, and each device then places its own resource over it.
    let handle: HANDLE =
        unsafe { source_device.CreateSharedHandle(&source_heap, None, GENERIC_ALL, None)? };

    let mut dest_heap: Option<ID3D12Heap> = None;
    let opened = unsafe { dest_device.OpenSharedHandle(handle, &mut dest_heap) };
    unsafe {
        let _ = CloseHandle(handle);
    }
    match opened {
        Ok(()) => {
            let _ = out.set_item("opened_on_destination", dest_heap.is_some());
        }
        Err(e) => {
            let _ = out.set_item("opened_on_destination", false);
            let _ = out.set_item("open_error", format!("{e}"));
            return Err(e);
        }
    }

    let buffer_desc = cross_adapter_buffer_desc(total_bytes);
    let mut shared_src: Option<ID3D12Resource> = None;
    unsafe {
        source_device.CreatePlacedResource(
            &source_heap,
            0,
            &buffer_desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut shared_src,
        )?;
    }
    let shared_src = shared_src.expect("CreatePlacedResource reported success");

    // Place a resource on the destination too. It is not used for timing, but
    // if this fails the sharing is not actually usable and the numbers below
    // would be measuring a copy to nowhere.
    if let Some(dest_heap) = dest_heap.as_ref() {
        let mut shared_dst: Option<ID3D12Resource> = None;
        let placed = unsafe {
            dest_device.CreatePlacedResource(
                dest_heap,
                0,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut shared_dst,
            )
        };
        let _ = out.set_item("placed_on_destination", placed.is_ok());
        if let Err(e) = placed {
            let _ = out.set_item("place_error", format!("{e}"));
        }
    }

    // A copy queue, because this is exactly what the DMA engine is for.
    let queue: ID3D12CommandQueue = unsafe {
        source_device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_COPY,
            Priority: 0,
            Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
            NodeMask: 0,
        })?
    };
    let allocator: ID3D12CommandAllocator =
        unsafe { source_device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)? };
    let list: ID3D12GraphicsCommandList = unsafe {
        source_device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)?
    };
    unsafe { list.Close()? };
    let fence: ID3D12Fence = unsafe { source_device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
    let event = unsafe { CreateEventW(None, false, false, None)? };

    // pResource is a ManuallyDrop, so the reference each location holds has to
    // be released by hand once the copies are done.
    let mut dst_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(shared_src.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    };
    let mut src_location = D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(local.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };

    let mut fence_value = 0u64;
    let mut timings_ms: Vec<f64> = Vec::with_capacity(iterations);

    // One untimed pass so the first-copy cost (allocation, driver warm-up)
    // does not land in the sample.
    for i in 0..=iterations {
        let start = std::time::Instant::now();
        unsafe {
            allocator.Reset()?;
            list.Reset(&allocator, None)?;
            list.CopyTextureRegion(&dst_location, 0, 0, 0, &src_location, None);
            list.Close()?;
            queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);

            fence_value += 1;
            queue.Signal(&fence, fence_value)?;
            if fence.GetCompletedValue() < fence_value {
                fence.SetEventOnCompletion(fence_value, event)?;
                WaitForSingleObject(event, INFINITE);
            }
        }
        if i > 0 {
            timings_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    unsafe {
        let _ = CloseHandle(event);
        core::mem::ManuallyDrop::drop(&mut dst_location.pResource);
        core::mem::ManuallyDrop::drop(&mut src_location.pResource);
    }

    timings_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = timings_ms.first().copied().unwrap_or(0.0);
    let median = timings_ms.get(timings_ms.len() / 2).copied().unwrap_or(0.0);

    let _ = out.set_item("supported", true);
    let _ = out.set_item("iterations", timings_ms.len());
    let _ = out.set_item("copy_ms_min", min);
    let _ = out.set_item("copy_ms_median", median);
    if min > 0.0 {
        let mb = total_bytes as f64 / 1_048_576.0;
        let _ = out.set_item("throughput_mb_s", mb / (min / 1000.0));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Cross-adapter transfer
// ---------------------------------------------------------------------------

/// One command queue plus everything needed to submit to it and wait.
struct Submitter {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    event: HANDLE,
    value: Cell<u64>,
}

impl Submitter {
    /// A copy queue: moving bytes is exactly what the DMA engine is for.
    fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let queue: ID3D12CommandQueue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COPY,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            })?
        };
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)? };
        let list: ID3D12GraphicsCommandList =
            unsafe { device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, &allocator, None)? };
        unsafe { list.Close()? };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let event = unsafe { CreateEventW(None, false, false, None)? };
        Ok(Self {
            queue,
            allocator,
            list,
            fence,
            event,
            value: Cell::new(0),
        })
    }

    /// Reset the list so a caller can record into it.
    fn begin(&self) -> windows::core::Result<&ID3D12GraphicsCommandList> {
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
        }
        Ok(&self.list)
    }

    /// Submit what was recorded and block until the GPU has finished it.
    fn end_and_wait(&self) -> windows::core::Result<()> {
        unsafe {
            self.list.Close()?;
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            let value = self.value.get() + 1;
            self.value.set(value);
            self.queue.Signal(&self.fence, value)?;
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.event)?;
                WaitForSingleObject(self.event, INFINITE);
            }
        }
        Ok(())
    }
}

impl Drop for CrossAdapterTransfer {
    fn drop(&mut self) {
        // Release the cached view of the capture texture, and the handle it
        // was opened from, before the devices go.
        if let Some((_, handle)) = self.cached_shared.borrow_mut().take() {
            if !handle.is_invalid() {
                unsafe {
                    let _ = CloseHandle(handle);
                }
            }
        }
        // The heap outlives this handle, but the handle must not outlive the
        // object that documents itself as owning it. Same contract the Stage 6
        // preprocessor uses for `shared_output_handle`.
        if !self.shared_fence_handle.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.shared_fence_handle);
            }
        }
        if !self.shared_fence_event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.shared_fence_event);
            }
        }
        if !self.shared_destination_handle.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.shared_destination_handle);
            }
        }
    }
}

impl Drop for Submitter {
    fn drop(&mut self) {
        unsafe {
            let _ = CloseHandle(self.event);
        }
    }
}

/// Moves a captured frame from the capture adapter to a second adapter.
///
/// The heap, the shared handle and both placed resources are created once and
/// reused, because none of them depend on the frame: only the copy is per-frame
/// work. Rebuilding them per frame would cost far more than the copy itself.
///
/// Synchronisation is CPU-side: `transfer` blocks until the source GPU has
/// finished writing before returning, so anything the caller then submits on
/// the destination adapter is correctly ordered. A fully asynchronous pipeline
/// would instead want a shared fence
/// (`D3D12_FENCE_FLAG_SHARED | SHARED_CROSS_ADAPTER`); that is a later
/// optimisation, not a correctness gap.
/// One row of `probe_shared_handles`: which object, what CUDA import type it
/// would need, and whether `CreateSharedHandle` was permitted.
pub type SharedHandleAttempt = (String, u32, Result<(), String>);

/// A candidate way of minting a shared handle, for the probe's table.
type HandleMaker<'a> = dyn Fn() -> windows::core::Result<HANDLE> + 'a;

pub struct CrossAdapterTransfer {
    src_device: ID3D12Device,
    dst_device: ID3D12Device,
    src: Submitter,
    dst: Submitter,
    // Kept alive: the placed resources below live inside these heaps.
    _src_heap: ID3D12Heap,
    _dst_heap: ID3D12Heap,
    src_buffer: ID3D12Resource,
    dst_buffer: ID3D12Resource,
    readback: ID3D12Resource,
    src_readback: ID3D12Resource,
    /// A stable copy of the live surface, used only by the verification path.
    snapshot: ID3D12Resource,
    /// NT handle for the destination heap, so a consumer on that adapter can
    /// import the transferred frame. Borrowed, like the Stage 6 preprocessor's
    /// output handle: created once here and closed in `Drop`.
    shared_destination_handle: HANDLE,
    /// The captured texture most recently opened on the source device, keyed by
    /// raw pointer. Desktop Duplication reuses its surfaces -- a 12-minute soak
    /// saw exactly one across 211,726 frames -- so this hits on essentially
    /// every frame. It is a cache, not an assumption: a different pointer
    /// reopens rather than reusing a resource describing a surface the caller
    /// is no longer capturing.
    ///
    /// Measured 2026-08-22, Intel iGPU -> RTX 4060: opening cost 300.8 us and
    /// closing 110.2 us per frame, together 13.4% of a transfer. Section 6.1
    /// previously recorded that overhead as noise, which it was on the hardware
    /// available then; it is not noise here.
    cached_texture: Cell<usize>,
    cached_shared: std::cell::RefCell<Option<(ID3D12Resource, HANDLE)>>,
    /// A fence the destination adapter can also see, so a consumer there can
    /// wait on the copy GPU-side instead of the calling thread blocking.
    ///
    /// `transfer()` still blocks; this is what `transfer_async()` signals.
    /// Created with `SHARED | SHARED_CROSS_ADAPTER` on the source device and
    /// opened on the destination, which is the only fence configuration both
    /// adapters can observe.
    shared_fence: ID3D12Fence,
    shared_fence_handle: HANDLE,
    shared_fence_value: Cell<u64>,
    /// Dedicated event for CPU waits on `shared_fence`. Separate from the
    /// Submitter's, which `end_and_wait` owns -- sharing one would let a
    /// blocking transfer and an async wait clobber each other's signal.
    shared_fence_event: HANDLE,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    total_bytes: u64,
    pub width: u32,
    pub height: u32,
    pub source_name: String,
    pub destination_name: String,
    pub destination_is_software: bool,
}

impl CrossAdapterTransfer {
    /// Build a transfer path for textures shaped like the one given.
    ///
    /// The source adapter is not a choice: it is whichever adapter owns the
    /// texture, because that is where Desktop Duplication put it.
    pub fn new(texture: &ID3D11Texture2D) -> windows::core::Result<Self> {
        let mut desc = Default::default();
        unsafe { texture.GetDesc(&mut desc) };
        let (width, height) = (desc.Width, desc.Height);

        // Fail at setup rather than on the first frame. A texture that cannot
        // be shared with D3D12 will never work on this path, and finding that
        // out while building the transfer is far easier to act on than a
        // failure part-way through a capture session.
        {
            let resource: IDXGIResource1 = texture.cast().map_err(|e| {
                windows::core::Error::new(
                    e.code(),
                    "texture does not expose IDXGIResource1, so it cannot be \
                     shared with D3D12; it must be created with \
                     D3D11_RESOURCE_MISC_SHARED_NTHANDLE",
                )
            })?;
            let handle =
                unsafe { resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None) }
                    .map_err(|e| {
                        windows::core::Error::new(
                            e.code(),
                            "texture is not shareable, so it cannot cross to another \
                             adapter; it must carry D3D11_RESOURCE_MISC_SHARED_NTHANDLE",
                        )
                    })?;
            unsafe {
                let _ = CloseHandle(handle);
            }
        }

        let src_adapter: IDXGIAdapter1 = unsafe {
            let device: ID3D11Device = texture.GetDevice()?;
            let dxgi: IDXGIDevice = device.cast()?;
            let adapter: IDXGIAdapter = dxgi.GetAdapter()?;
            adapter.cast()?
        };
        let src_desc = unsafe { src_adapter.GetDesc1()? };
        let src_luid = (src_desc.AdapterLuid.LowPart, src_desc.AdapterLuid.HighPart);

        let adapters = enumerate_adapters()?;
        // Any adapter that is not the capture one. A hardware adapter is the
        // real case; a software one still exercises the whole path, which is
        // the only way to test this without a second GPU.
        let dst = adapters
            .iter()
            .find(|a| a.luid != src_luid && !a.is_software)
            .or_else(|| adapters.iter().find(|a| a.luid != src_luid))
            .ok_or_else(|| {
                windows::core::Error::new(
                    windows::Win32::Foundation::E_FAIL,
                    "no second adapter: this system has only the capture adapter, \
                     so there is nowhere to transfer to",
                )
            })?;

        let src_device = make_device(&src_adapter)?;
        let dst_device = make_device(&dst.adapter)?;

        // Lay the frame out linearly, including the 256-byte row alignment
        // D3D12 requires for copies. This is the size of the shared heap.
        let texture_desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Alignment: 0,
            Width: width as u64,
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            Flags: D3D12_RESOURCE_FLAG_NONE,
        };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut total_bytes = 0u64;
        unsafe {
            src_device.GetCopyableFootprints(
                &texture_desc,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut total_bytes),
            );
        }

        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: total_bytes,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Flags: D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
        };
        let mut src_heap: Option<ID3D12Heap> = None;
        unsafe { src_device.CreateHeap(&heap_desc, &mut src_heap)? };
        let src_heap = src_heap.expect("CreateHeap reported success");

        // The *heap* is shared, not the resource: cross-adapter placement is a
        // heap property, and each device then places its own resource over it.
        let handle: HANDLE =
            unsafe { src_device.CreateSharedHandle(&src_heap, None, GENERIC_ALL, None)? };
        let mut dst_heap: Option<ID3D12Heap> = None;
        let opened = unsafe { dst_device.OpenSharedHandle(handle, &mut dst_heap) };
        unsafe {
            let _ = CloseHandle(handle);
        }
        opened?;
        let dst_heap = dst_heap.expect("OpenSharedHandle reported success");

        // A buffer, not a row-major texture: CrossAdapterRowMajorTextureSupported
        // is optional and false on plenty of hardware, and the buffer path needs
        // no branch. See probe_cross_adapter().
        let buffer_desc = cross_adapter_buffer_desc(total_bytes);
        let mut src_buffer: Option<ID3D12Resource> = None;
        unsafe {
            src_device.CreatePlacedResource(
                &src_heap,
                0,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut src_buffer,
            )?;
        }
        let mut dst_buffer: Option<ID3D12Resource> = None;
        unsafe {
            dst_device.CreatePlacedResource(
                &dst_heap,
                0,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut dst_buffer,
            )?;
        }

        // Readback lives on the destination device on purpose: reading it is
        // what proves the bytes actually crossed, rather than that a copy was
        // submitted without error.
        let readback = make_readback_buffer(&dst_device, total_bytes)?;
        // The reference for that readback: the same texture, read through the
        // source device without going near the shared heap.
        let src_readback = make_readback_buffer(&src_device, total_bytes)?;
        let snapshot = make_local_texture(&src_device, width, height)?;

        // Share the *heap*, not the placed resource. `probe_shared_handles`
        // measured both: `CreateSharedHandle` on either placed buffer fails
        // with E_INVALIDARG, on either heap it succeeds. So a consumer imports
        // this as CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP (4), not the
        // D3D12_RESOURCE (5) that Stage 6's committed output resource uses.
        //
        // Minted on `dst_device` rather than `src_device` although both work.
        // The consumer runs on the destination adapter, so a handle created by
        // the same device is the narrower assumption; using the source's would
        // additionally require the consumer's runtime to accept a handle made
        // by a different vendor's device. That held on the one Intel -> NVIDIA
        // pair measured, which is not enough to depend on it.
        let shared_destination_handle =
            unsafe { dst_device.CreateSharedHandle(&dst_heap, None, GENERIC_ALL, None)? };

        // A fence both adapters can observe. SHARED alone is not enough across
        // adapters -- SHARED_CROSS_ADAPTER is what makes the destination able
        // to open it, exactly as the heap above needed it.
        let shared_fence: ID3D12Fence = unsafe {
            src_device.CreateFence(
                0,
                D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER,
            )?
        };
        let shared_fence_handle =
            unsafe { src_device.CreateSharedHandle(&shared_fence, None, GENERIC_ALL, None)? };
        let shared_fence_event = unsafe { CreateEventW(None, false, false, None)? };

        Ok(Self {
            src: Submitter::new(&src_device)?,
            dst: Submitter::new(&dst_device)?,
            src_device,
            dst_device,
            _src_heap: src_heap,
            _dst_heap: dst_heap,
            src_buffer: src_buffer.expect("CreatePlacedResource reported success"),
            dst_buffer: dst_buffer.expect("CreatePlacedResource reported success"),
            readback,
            src_readback,
            snapshot,
            shared_destination_handle,
            cached_texture: Cell::new(0),
            cached_shared: std::cell::RefCell::new(None),
            shared_fence,
            shared_fence_handle,
            shared_fence_value: Cell::new(0),
            shared_fence_event,
            footprint,
            total_bytes,
            width,
            height,
            source_name: trim_description(&src_desc.Description),
            destination_name: dst.description.clone(),
            destination_is_software: dst.is_software,
        })
    }

    /// Copy one captured frame into the shared heap.
    ///
    /// Blocks until the source GPU has finished, so the frame is readable from
    /// the destination adapter by the time this returns.
    pub fn transfer(&self, texture: &ID3D11Texture2D) -> windows::core::Result<()> {
        self.copy_from(texture, false)
    }

    /// Copy the texture into the shared heap, optionally mirroring the same
    /// copy into the source-side readback buffer.
    ///
    /// Both copies go into one command list on purpose: submitted separately
    /// they would see different content, because the duplicated surface is
    /// live.
    /// Open the capture texture on the source device, reusing the last one.
    ///
    /// Keyed on the raw pointer rather than assumed stable: a changed surface
    /// reopens. The NT handle is kept alongside the resource because it must
    /// outlive nothing in particular -- `OpenSharedHandle` does not take
    /// ownership -- but closing it per frame is exactly the 110 us this cache
    /// exists to remove, so it is closed when the cache entry is replaced.
    fn open_capture_texture(
        &self,
        texture: &ID3D11Texture2D,
    ) -> windows::core::Result<ID3D12Resource> {
        let key = texture.as_raw() as usize;
        if self.cached_texture.get() == key {
            if let Some((resource, _)) = self.cached_shared.borrow().as_ref() {
                return Ok(resource.clone());
            }
        }

        let dxgi: IDXGIResource1 = texture.cast()?;
        let handle = unsafe { dxgi.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None)? };
        let mut shared: Option<ID3D12Resource> = None;
        let opened = unsafe { self.src_device.OpenSharedHandle(handle, &mut shared) };
        if opened.is_err() {
            unsafe {
                let _ = CloseHandle(handle);
            }
            opened?;
        }
        let shared = shared.expect("OpenSharedHandle reported success");

        // Replace the previous entry, closing the handle it held.
        if let Some((_, old_handle)) = self.cached_shared.borrow_mut().take() {
            if !old_handle.is_invalid() {
                unsafe {
                    let _ = CloseHandle(old_handle);
                }
            }
        }
        *self.cached_shared.borrow_mut() = Some((shared.clone(), handle));
        self.cached_texture.set(key);
        Ok(shared)
    }

    fn copy_from(
        &self,
        texture: &ID3D11Texture2D,
        with_reference: bool,
    ) -> windows::core::Result<()> {
        let mut desc = Default::default();
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Width != self.width || desc.Height != self.height {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                "texture size does not match the one this transfer was built for;                  build a new transfer after a resolution change",
            ));
        }

        let shared = self.open_capture_texture(texture)?;

        if with_reference {
            return self.copy_via_snapshot(&shared);
        }

        let mut src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: core::mem::ManuallyDrop::new(Some(shared)),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let mut destination = placed_location(&self.src_buffer, self.footprint);

        let list = self.src.begin()?;
        unsafe { list.CopyTextureRegion(&destination, 0, 0, 0, &src_location, None) };
        let submitted = self.src.end_and_wait();

        // pResource is a ManuallyDrop, so every reference is released by hand
        // rather than at end of scope.
        unsafe {
            core::mem::ManuallyDrop::drop(&mut destination.pResource);
            core::mem::ManuallyDrop::drop(&mut src_location.pResource);
        }
        submitted
    }

    /// Transfer a frame *and* return a source-side copy of the same bytes.
    ///
    /// Verification only. Both copies are recorded into one command list and
    /// submitted together, so they are guaranteed to see identical source
    /// content — the duplicated surface is live and does change between two
    /// separately submitted copies, which showed up as ~2000 differing bytes in
    /// a single screen region.
    ///
    /// Comparing against a CPU capture instead does not work at all: Desktop
    /// Duplication reports only *changed* content, so two consecutive frames
    /// differ by construction and there is no stable screen to compare with.
    ///
    /// The layout matches `read_back_destination` exactly, including the
    /// 256-byte row padding, so the two are comparable byte for byte.
    pub fn transfer_with_reference(
        &self,
        texture: &ID3D11Texture2D,
    ) -> windows::core::Result<Vec<u8>> {
        self.copy_from(texture, true)?;
        map_to_vec(&self.src_readback, self.total_bytes)
    }

    /// Freeze the live surface, then feed both the shared heap and the
    /// reference from the frozen copy.
    ///
    /// The duplicated surface is not stable: DXGI keeps writing to it, and
    /// Rapidshot does not hold its keyed mutex during these copies. Two
    /// CopyTextureRegion calls execute one after another on the copy engine, so
    /// even inside a single command list they can see different pixels — this
    /// showed up as ~2100 bytes differing in one screen region, reproducibly at
    /// the same offset. Reading the live surface exactly once removes the race.
    fn copy_via_snapshot(&self, shared: &ID3D12Resource) -> windows::core::Result<()> {
        let list = self.src.begin()?;
        unsafe {
            transition(
                list,
                &self.snapshot,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            list.CopyResource(&self.snapshot, shared);
            transition(
                list,
                &self.snapshot,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );

            let mut source = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(self.snapshot.clone())),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let mut destinations = [
                placed_location(&self.src_buffer, self.footprint),
                placed_location(&self.src_readback, self.footprint),
            ];
            for destination in destinations.iter() {
                list.CopyTextureRegion(destination, 0, 0, 0, &source, None);
            }
            transition(
                list,
                &self.snapshot,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );

            let submitted = self.src.end_and_wait();
            for destination in destinations.iter_mut() {
                core::mem::ManuallyDrop::drop(&mut destination.pResource);
            }
            core::mem::ManuallyDrop::drop(&mut source.pResource);
            submitted
        }
    }

    /// Read the frame back through the *destination* device.
    ///
    /// Verification, not production: a consumer on the destination adapter
    /// should bind `destination_resource_address()` instead and never touch the
    /// CPU. This exists because "the copy was submitted without error" is not
    /// evidence that the right bytes arrived on the other adapter.
    pub fn read_back_destination(&self) -> windows::core::Result<Vec<u8>> {
        let list = self.dst.begin()?;
        unsafe { list.CopyBufferRegion(&self.readback, 0, &self.dst_buffer, 0, self.total_bytes) };
        self.dst.end_and_wait()?;
        map_to_vec(&self.readback, self.total_bytes)
    }

    /// Address of the `ID3D12Resource` holding the frame on the destination
    /// adapter. Borrowed, not owned: valid while this object is alive.
    /// Submit a transfer without blocking, returning the fence value to wait on.
    ///
    /// `transfer()` blocks until the source GPU finishes, which on this path is
    /// mostly the copy itself executing -- measured 2026-08-22 at 2.67 ms of a
    /// 2.70 ms transfer, Intel iGPU to RTX 4060 at 2560x1600. Blocking returns
    /// a frame that is immediately readable; this returns the calling thread
    /// instead and leaves synchronisation to the caller.
    ///
    /// **It still waits for the *previous* submission** before recording, and
    /// that is not laziness: `Submitter::begin` resets the command allocator,
    /// and resetting one the GPU is still reading is corruption. So this
    /// pipelines to depth one -- frame N's copy overlaps whatever the caller
    /// does next, and the cost is paid at the start of frame N+1 only if the
    /// GPU has not finished by then.
    ///
    /// The returned value is for `wait_shared_fence`. A consumer on the
    /// destination adapter can instead wait GPU-side on the fence opened from
    /// `shared_fence_handle()`, which is the point of it being cross-adapter.
    pub fn transfer_async(&self, texture: &ID3D11Texture2D) -> windows::core::Result<u64> {
        let mut desc = Default::default();
        unsafe { texture.GetDesc(&mut desc) };
        if desc.Width != self.width || desc.Height != self.height {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                "texture size does not match the one this transfer was built for",
            ));
        }

        // The allocator reset below is only safe once the GPU is done with it.
        self.wait_shared_fence(self.shared_fence_value.get())?;

        let shared = self.open_capture_texture(texture)?;
        let mut src_location = D3D12_TEXTURE_COPY_LOCATION {
            pResource: core::mem::ManuallyDrop::new(Some(shared)),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let mut destination = placed_location(&self.src_buffer, self.footprint);

        let result = (|| -> windows::core::Result<u64> {
            let list = self.src.begin()?;
            unsafe { list.CopyTextureRegion(&destination, 0, 0, 0, &src_location, None) };
            unsafe {
                list.Close()?;
                self.src
                    .queue
                    .ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
            }
            let value = self.shared_fence_value.get() + 1;
            self.shared_fence_value.set(value);
            unsafe { self.src.queue.Signal(&self.shared_fence, value)? };
            Ok(value)
        })();

        unsafe {
            core::mem::ManuallyDrop::drop(&mut destination.pResource);
            core::mem::ManuallyDrop::drop(&mut src_location.pResource);
        }
        result
    }

    /// Block until the shared fence reaches `value`. 0 returns immediately.
    pub fn wait_shared_fence(&self, value: u64) -> windows::core::Result<()> {
        if let Some(event) = self.arm_shared_fence_wait(value)? {
            unsafe { WaitForSingleObject(HANDLE(event as *mut core::ffi::c_void), INFINITE) };
        }
        Ok(())
    }

    /// Arm the wait and hand back the event, or `None` if already complete.
    ///
    /// Split out so a caller holding the Python GIL can release it around the
    /// blocking half. `py.detach` needs a `Send` closure, and this type holds
    /// `Cell`/`RefCell`, so a reference to it cannot cross that boundary --
    /// but a raw event handle can, and waiting on one touches no state here.
    /// That keeps the GIL release provably sound rather than asserted.
    pub fn arm_shared_fence_wait(&self, value: u64) -> windows::core::Result<Option<isize>> {
        if value == 0 {
            return Ok(None);
        }
        unsafe {
            if self.shared_fence.GetCompletedValue() >= value {
                return Ok(None);
            }
            self.shared_fence
                .SetEventOnCompletion(value, self.shared_fence_event)?;
        }
        Ok(Some(self.shared_fence_event.0 as isize))
    }

    /// NT handle for the cross-adapter fence. Borrowed, closed on `Drop`.
    ///
    /// A consumer on the destination adapter opens this with
    /// `ID3D12Device::OpenSharedHandle` and waits on it from its own queue, so
    /// the copy and the consuming work overlap without a CPU round-trip.
    pub fn shared_fence_handle(&self) -> isize {
        self.shared_fence_handle.0 as isize
    }

    /// Split one `transfer()` into its phases, to size what a shared fence
    /// would actually buy.
    ///
    /// Section 6.1 deferred the fence on a measurement taken against WARP:
    /// the blocking wait looked free because the copy dominated. Real hybrid
    /// hardware runs the Intel -> NVIDIA direction at roughly half that
    /// throughput, so the share is not the same number and the argument has to
    /// be re-derived rather than carried over.
    ///
    /// Phases, in the order `copy_from` performs them:
    ///
    /// - `open`: CreateSharedHandle + OpenSharedHandle on the capture texture.
    ///   Cached per texture now; it cost 268 us a frame before that.
    /// - `record`: allocator reset + CopyTextureRegion.
    /// - `submit`: Close + ExecuteCommandLists.
    /// - `signal`: Signal on the queue.
    /// - `wait`: SetEventOnCompletion + WaitForSingleObject. The only phase a
    ///   shared fence removes, and 85% of the transfer before it did.
    /// - `close`: CloseHandle, 114 us a frame before caching.
    ///
    /// Returns `(min, median)` microseconds per phase. **Read `wait` as the
    /// median, never the minimum**: it is bimodal, because whenever the GPU
    /// has already finished, `GetCompletedValue` clears immediately and the
    /// sample is ~0. Section 10 records that trap costing a previous probe its
    /// conclusion.
    ///
    /// `use_cache` selects which open path is measured, so the per-frame
    /// handle cache can be A/B'd inside one harness rather than across two
    /// runs -- comparing separate harnesses is what section 2 warns produces
    /// verdicts about conditions rather than code.
    pub fn probe_transfer_phases(
        &self,
        texture: &ID3D11Texture2D,
        iterations: u32,
        use_cache: bool,
    ) -> windows::core::Result<[(f64, f64); 6]> {
        let count = iterations.max(1) as usize;
        let mut phases: [Vec<f64>; 6] = Default::default();

        for _ in 0..count {
            let t0 = std::time::Instant::now();
            let (shared, handle) = if use_cache {
                (self.open_capture_texture(texture)?, HANDLE::default())
            } else {
                let resource: IDXGIResource1 = texture.cast()?;
                let handle =
                    unsafe { resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None)? };
                let mut shared: Option<ID3D12Resource> = None;
                unsafe { self.src_device.OpenSharedHandle(handle, &mut shared)? };
                (shared.expect("OpenSharedHandle reported success"), handle)
            };
            let t1 = std::time::Instant::now();

            let mut src_location = D3D12_TEXTURE_COPY_LOCATION {
                pResource: core::mem::ManuallyDrop::new(Some(shared)),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            let mut destination = placed_location(&self.src_buffer, self.footprint);
            let list = self.src.begin()?;
            unsafe { list.CopyTextureRegion(&destination, 0, 0, 0, &src_location, None) };
            let t2 = std::time::Instant::now();

            unsafe {
                list.Close()?;
                self.src
                    .queue
                    .ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
            }
            let t3 = std::time::Instant::now();

            let value = self.src.value.get() + 1;
            self.src.value.set(value);
            unsafe { self.src.queue.Signal(&self.src.fence, value)? };
            let t4 = std::time::Instant::now();

            unsafe {
                if self.src.fence.GetCompletedValue() < value {
                    self.src.fence.SetEventOnCompletion(value, self.src.event)?;
                    WaitForSingleObject(self.src.event, INFINITE);
                }
            }
            let t5 = std::time::Instant::now();

            unsafe {
                core::mem::ManuallyDrop::drop(&mut destination.pResource);
                core::mem::ManuallyDrop::drop(&mut src_location.pResource);
                // Only the uncached path owns a handle here; the cached one
                // belongs to the cache and is closed when it is replaced.
                if !handle.is_invalid() {
                    let _ = CloseHandle(handle);
                }
            }
            let t6 = std::time::Instant::now();

            let us = |a: std::time::Instant, b: std::time::Instant| {
                b.duration_since(a).as_secs_f64() * 1e6
            };
            phases[0].push(us(t0, t1));
            phases[1].push(us(t1, t2));
            phases[2].push(us(t2, t3));
            phases[3].push(us(t3, t4));
            phases[4].push(us(t4, t5));
            phases[5].push(us(t5, t6));
        }

        fn min_and_median(mut xs: Vec<f64>) -> (f64, f64) {
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (xs[0], xs[xs.len() / 2])
        }

        let mut out = [(0.0, 0.0); 6];
        for (slot, samples) in out.iter_mut().zip(phases) {
            *slot = min_and_median(samples);
        }
        Ok(out)
    }

    /// Which objects will yield an NT handle for the transferred frame.
    ///
    /// Diagnostic. It exists to settle empirically which handle a consumer on
    /// the destination adapter can obtain, rather than picking one from
    /// documentation that does not say -- in particular whether
    /// `CreateSharedHandle` succeeds on a placed resource inside a heap that
    /// was itself obtained from `OpenSharedHandle`. Measured 2026-08-22 on an
    /// Intel iGPU -> RTX 4060 pair: **only the heaps can be shared.** Both
    /// placed resources fail with `E_INVALIDARG`, which is why
    /// `shared_destination_handle` shares `dst_heap` and a CUDA consumer
    /// imports it as `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP` (4) rather
    /// than the `D3D12_RESOURCE` (5) the Stage 6 preprocessor uses.
    ///
    /// Run this first on any new adapter pairing; the answer is a property of
    /// the driver pair, and this project has verified exactly one.
    ///
    /// **Leaks nothing.** Every handle it mints is closed before returning, so
    /// the result is the outcome of the attempt rather than a usable handle.
    /// For a handle you can actually import, use `shared_destination_handle`.
    ///
    /// Returns `(label, cuda_handle_type, outcome)` per candidate, where
    /// `cuda_handle_type` is what the handle would import as: 4 = D3D12_HEAP,
    /// 5 = D3D12_RESOURCE.
    pub fn probe_shared_handles(&self) -> Vec<SharedHandleAttempt> {
        let attempts: [(&str, u32, &HandleMaker<'_>); 4] = [
            ("dst_resource", 5, &|| unsafe {
                self.dst_device
                    .CreateSharedHandle(&self.dst_buffer, None, GENERIC_ALL, None)
            }),
            ("dst_heap", 4, &|| unsafe {
                self.dst_device
                    .CreateSharedHandle(&self._dst_heap, None, GENERIC_ALL, None)
            }),
            ("src_resource", 5, &|| unsafe {
                self.src_device
                    .CreateSharedHandle(&self.src_buffer, None, GENERIC_ALL, None)
            }),
            ("src_heap", 4, &|| unsafe {
                self.src_device
                    .CreateSharedHandle(&self._src_heap, None, GENERIC_ALL, None)
            }),
        ];

        attempts
            .iter()
            .map(|(label, kind, make)| {
                let outcome = match make() {
                    Ok(handle) => {
                        // Closed immediately: this reports whether the call is
                        // permitted, and a probe that hands back live handles
                        // makes the caller responsible for cleanup it did not
                        // ask for.
                        unsafe {
                            let _ = CloseHandle(handle);
                        }
                        Ok(())
                    }
                    Err(e) => Err(format!("{e}")),
                };
                (label.to_string(), *kind, outcome)
            })
            .collect()
    }

    /// NT handle for the destination heap the transferred frame lives in.
    ///
    /// **Borrowed, not owned.** It is closed when this object is dropped, and
    /// the heap goes with it -- so a consumer holding only the integer, or a
    /// mapped pointer derived from it, has nothing that looks wrong afterwards.
    /// Keep the transfer alive for as long as anything reads the frame.
    ///
    /// Import it as a **heap** (`CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP`,
    /// 4), sized `total_bytes`, offset 0 -- not as a resource. Measured
    /// 2026-08-22, Intel iGPU -> RTX 4060: the imported buffer reads
    /// byte-identical to `read_back_destination()` and observes later
    /// `transfer()` calls without re-importing.
    /// Raw pointer of the capture texture currently cached, or 0.
    ///
    /// Exposed for tests. Reopening per frame cost 411 us here and produces
    /// output that looks correct either way, so a change that quietly disabled
    /// the cache would be a silent regression -- the assertion has to be on the
    /// key, not on pixels.
    pub fn cached_texture_address(&self) -> usize {
        if self.cached_shared.borrow().is_some() {
            self.cached_texture.get()
        } else {
            0
        }
    }

    pub fn shared_destination_handle(&self) -> isize {
        self.shared_destination_handle.0 as isize
    }

    pub fn destination_resource_address(&self) -> usize {
        self.dst_buffer.as_raw() as usize
    }

    pub fn destination_device_address(&self) -> usize {
        self.dst_device.as_raw() as usize
    }

    pub fn total_bytes(&self) -> u64 {
        self.total_bytes
    }

    /// Bytes per row in the shared buffer. Padded to D3D12's 256-byte copy
    /// alignment, so this is not always `width * 4`.
    pub fn row_pitch(&self) -> u32 {
        self.footprint.Footprint.RowPitch
    }
}

/// Record a resource state transition into a command list.
fn transition(
    list: &ID3D12GraphicsCommandList,
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) {
    let barrier = D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: core::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[barrier]) };
}

/// A copy destination addressing a linear buffer through a texture footprint.
fn placed_location(
    resource: &ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
) -> D3D12_TEXTURE_COPY_LOCATION {
    D3D12_TEXTURE_COPY_LOCATION {
        pResource: core::mem::ManuallyDrop::new(Some(resource.clone())),
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: footprint,
        },
    }
}

/// Copy a mapped readback resource into an owned Vec.
fn map_to_vec(resource: &ID3D12Resource, size: u64) -> windows::core::Result<Vec<u8>> {
    let mut mapped: *mut core::ffi::c_void = std::ptr::null_mut();
    let mut out = vec![0u8; size as usize];
    unsafe {
        resource.Map(0, None, Some(&mut mapped))?;
        std::ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), out.len());
        resource.Unmap(0, None);
    }
    Ok(out)
}

fn make_readback_buffer(device: &ID3D12Device, size: u64) -> windows::core::Result<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_READBACK,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COPY_DEST,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource reported success"))
}

/// Time a cross-adapter copy of an arbitrary buffer.
///
/// The frame probe above copies a *texture*; the Stage 6 tensor is a buffer, so
/// deciding whether to convert before or after the transfer needs this shape
/// measured too. Same heap mechanism, `CopyBufferRegion` instead of
/// `CopyTextureRegion`.
#[pyfunction]
#[pyo3(signature = (size_bytes, iterations = 50))]
pub fn probe_cross_adapter_buffer(
    py: Python<'_>,
    size_bytes: u64,
    iterations: usize,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    if size_bytes == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "size_bytes must be non-zero",
        ));
    }

    let adapters = enumerate_adapters()
        .map_err(|e| PyRuntimeError::new_err(format!("could not enumerate adapters: {e}")))?;
    let source = adapters
        .iter()
        .find(|a| a.drives_display() && !a.is_software)
        .or_else(|| adapters.iter().find(|a| !a.is_software))
        .or_else(|| adapters.first());
    let Some(source) = source else {
        out.set_item("supported", false)?;
        out.set_item("reason", "no DXGI adapters")?;
        return Ok(out.into());
    };
    let dest = adapters
        .iter()
        .find(|a| a.index != source.index && !a.is_software)
        .or_else(|| adapters.iter().find(|a| a.index != source.index));
    let Some(dest) = dest else {
        out.set_item("supported", false)?;
        out.set_item("reason", "only one adapter on this system")?;
        return Ok(out.into());
    };

    out.set_item("source", &source.description)?;
    out.set_item("destination", &dest.description)?;
    out.set_item("representative", !dest.is_software)?;
    out.set_item("size_bytes", size_bytes)?;

    let run = || -> windows::core::Result<(f64, f64)> {
        let src_device = make_device(&source.adapter)?;
        let dst_device = make_device(&dest.adapter)?;

        // The tensor as it exists on the capture adapter: an ordinary
        // GPU-local buffer, exactly what the conversion shader writes.
        let local = make_default_buffer(&src_device, size_bytes)?;

        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: size_bytes,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
            },
            Alignment: D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT as u64,
            Flags: D3D12_HEAP_FLAG_SHARED | D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
        };
        let mut src_heap: Option<ID3D12Heap> = None;
        unsafe { src_device.CreateHeap(&heap_desc, &mut src_heap)? };
        let src_heap = src_heap.expect("CreateHeap reported success");

        let handle = unsafe { src_device.CreateSharedHandle(&src_heap, None, GENERIC_ALL, None)? };
        let mut dst_heap: Option<ID3D12Heap> = None;
        let opened = unsafe { dst_device.OpenSharedHandle(handle, &mut dst_heap) };
        unsafe {
            let _ = CloseHandle(handle);
        }
        opened?;

        let buffer_desc = cross_adapter_buffer_desc(size_bytes);
        let mut shared: Option<ID3D12Resource> = None;
        unsafe {
            src_device.CreatePlacedResource(
                &src_heap,
                0,
                &buffer_desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut shared,
            )?;
        }
        let shared = shared.expect("CreatePlacedResource reported success");

        let submitter = Submitter::new(&src_device)?;
        let mut timings = Vec::with_capacity(iterations);
        // One untimed pass so first-copy driver warm-up stays out of the sample.
        for i in 0..=iterations {
            let start = std::time::Instant::now();
            let list = submitter.begin()?;
            unsafe { list.CopyBufferRegion(&shared, 0, &local, 0, size_bytes) };
            submitter.end_and_wait()?;
            if i > 0 {
                timings.push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
        timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Ok((
            timings.first().copied().unwrap_or(0.0),
            timings.get(timings.len() / 2).copied().unwrap_or(0.0),
        ))
    };

    match run() {
        Ok((min, median)) => {
            out.set_item("supported", true)?;
            out.set_item("copy_ms_min", min)?;
            out.set_item("copy_ms_median", median)?;
            if min > 0.0 {
                let mb = size_bytes as f64 / 1_048_576.0;
                out.set_item("throughput_mb_s", mb / (min / 1000.0))?;
            }
        }
        Err(e) => {
            out.set_item("supported", false)?;
            out.set_item("reason", format!("{e}"))?;
        }
    }
    Ok(out.into())
}

/// A GPU-local buffer on the given device.
fn make_default_buffer(device: &ID3D12Device, size: u64) -> windows::core::Result<ID3D12Resource> {
    let props = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
        MemoryPoolPreference: D3D12_MEMORY_POOL_UNKNOWN,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Alignment: 0,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_NONE,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource reported success"))
}
