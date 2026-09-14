//! Ordering D3D12 reads of a captured surface behind the capture itself.
//!
//! **The race.** `AcquireNextFrame` returns once the copy into the duplication
//! surface has been *submitted* on the capture device's D3D11 queue, not once
//! it has *run*. A D3D12 queue reading that surface through a shared handle is
//! a different queue with no implied order, so it can overtake the copy and
//! read the previous frame's pixels. No error, correct shape, plausible
//! content.
//!
//! Measured 2026-09-14 against `benchmarks/motion_source.py`, first read
//! immediately after acquisition versus a settled read of the same held frame:
//!
//! | Path | Stale first reads, before |
//! | --- | --- |
//! | `GpuConverter` | 7–11 / 100 |
//! | `GpuPreprocessor12` | 7–15 / 150 |
//! | `CrossAdapterTransfer` (to WARP) | 65 / 150 |
//!
//! `ID3D11DeviceContext::Flush` alone does not help (13–14 / 100): it submits
//! without waiting. A 5 ms sleep does (0 / 100), which confirms the cause and
//! is not a fix.
//!
//! **The fix is ordering, not waiting.** A D3D12 fence shared with the capture
//! device is signalled into its D3D11 command stream — behind the copy — and
//! the reading queue is told, with `ID3D12CommandQueue::Wait`, not to execute
//! anything further until the fence is reached. The GPU does the waiting; the
//! CPU never sleeps for it.
//!
//! **Thread safety rests on an invariant, and it is worth stating.** The
//! capture device is created without D3D11 multithread protection, so its
//! immediate context must not be used from two threads at once. `order` is
//! only ever called while the caller holds a live `Frame` — the texture it
//! takes comes from one — and while a frame is live the camera refuses every
//! call that would touch that context (`grab`, `grab_frame`, continuous
//! capture). Code that breaks that invariant breaks more than this.

use std::cell::{Cell, RefCell};

use windows::core::Interface;
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D11::{
    ID3D11Device, ID3D11Device5, ID3D11DeviceContext4, ID3D11Fence, ID3D11Texture2D,
};
use windows::Win32::Graphics::Direct3D12::{
    ID3D12CommandQueue, ID3D12Device, ID3D12Fence, D3D12_FENCE_FLAG_SHARED,
};

const GENERIC_ALL: u32 = 0x1000_0000;

pub(crate) struct CaptureOrder {
    fence: ID3D12Fence,
    handle: HANDLE,
    value: Cell<u64>,
    /// The capture device's immediate context and its view of `fence`, keyed
    /// on the D3D11 device address. Holding the context keeps that device
    /// alive, so the address cannot be recycled while it is the key; a rebuilt
    /// capture brings a new device, which re-opens the fence.
    sync: RefCell<Option<(usize, ID3D11DeviceContext4, ID3D11Fence)>>,
    /// Set by `quarantine`: the handle must then outlive this value too.
    handle_leaked: Cell<bool>,
}

impl CaptureOrder {
    /// A fence on the device whose queue will read captured surfaces.
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_SHARED)? };
        let handle = unsafe { device.CreateSharedHandle(&fence, None, GENERIC_ALL, None)? };
        Ok(Self {
            fence,
            handle,
            value: Cell::new(0),
            sync: RefCell::new(None),
            handle_leaked: Cell::new(false),
        })
    }

    /// Make `queue`'s next submission wait, on the GPU, for the capture work
    /// that filled `texture`. Call after opening the surface and before
    /// `ExecuteCommandLists`.
    ///
    /// If a later step fails before anything is submitted, the queued wait is
    /// harmless: the fence is reached as soon as the D3D11 stream runs, and the
    /// next submission would have needed the same ordering anyway.
    pub(crate) fn order(
        &self,
        queue: &ID3D12CommandQueue,
        texture: &ID3D11Texture2D,
    ) -> windows::core::Result<()> {
        let device11: ID3D11Device = unsafe { texture.GetDevice()? };
        let key = device11.as_raw() as usize;
        let mut sync = self.sync.borrow_mut();
        if sync.as_ref().map(|(k, _, _)| *k) != Some(key) {
            let device5: ID3D11Device5 = device11.cast().map_err(|e| {
                windows::core::Error::new(
                    e.code(),
                    "the capture device does not expose ID3D11Device5 (Direct3D 11.4, \
                     Windows 10 1703+), which is needed to order GPU reads after capture",
                )
            })?;
            let mut fence11: Option<ID3D11Fence> = None;
            unsafe { device5.OpenSharedFence(self.handle, &mut fence11)? };
            let context: ID3D11DeviceContext4 = unsafe { device11.GetImmediateContext()? }.cast()?;
            *sync = Some((key, context, fence11.expect("OpenSharedFence reported success")));
        }
        let (_, context, fence11) = sync.as_ref().expect("populated above");

        let value = self.value.get() + 1;
        unsafe {
            context.Signal(fence11, value)?;
            // The signal sits behind the copy in the D3D11 stream, but nothing
            // reaches the GPU until the stream is submitted. Without this the
            // D3D12 queue would wait for whatever next flushes it.
            context.Flush();
            self.value.set(value);
            queue.Wait(&self.fence, value)?;
        }
        Ok(())
    }

    /// Keep every object a queued `Wait` may name alive past this value's drop.
    /// For owners that quarantine rather than release after an untrackable
    /// submission.
    pub(crate) fn quarantine(&self) {
        std::mem::forget(self.fence.clone());
        if let Some((_, context, fence11)) = self.sync.borrow().as_ref() {
            std::mem::forget(context.clone());
            std::mem::forget(fence11.clone());
        }
        // The handle is left open for the same reason.
        self.handle_leaked.set(true);
    }
}

impl Drop for CaptureOrder {
    fn drop(&mut self) {
        *self.sync.borrow_mut() = None;
        if !self.handle_leaked.get() && !self.handle.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.handle);
            }
        }
    }
}
