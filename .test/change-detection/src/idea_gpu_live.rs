//! Live DXGI controls for the benchmark binary. Python owns the marker presenter.
use super::*;
use std::sync::mpsc::{sync_channel, TrySendError};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_MODE_ROTATION_IDENTITY};
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO};

const MARKER: Rect = Rect { left: 332, top: 31, right: 524, bottom: 43 };
const MARKER_WIDTH: usize = MARKER.right - MARKER.left;
const MARKER_HEIGHT: usize = MARKER.bottom - MARKER.top;

#[link(name = "kernel32")]
extern "system" {
    fn QueryPerformanceCounter(value: *mut i64) -> i32;
    fn QueryPerformanceFrequency(value: *mut i64) -> i32;
}

fn qpc() -> i64 {
    let mut value = 0;
    unsafe { QueryPerformanceCounter(&mut value) };
    value
}

#[derive(Clone, Copy)]
struct Meta {
    present: i64,
    acquired: i64,
    count: u32,
}

#[derive(Serialize)]
struct Frame {
    present_qpc: i64,
    acquired_qpc: i64,
    producer_qpc: Option<i64>,
    delivered_qpc: Option<i64>,
    accumulated: u32,
    stamp: Option<u16>,
}

fn stamp(row: &[u32]) -> Option<u16> {
    if row.len() < MARKER_WIDTH {
        return None;
    }
    let mut word = 0_u32;
    for &pixel in row[3..MARKER_WIDTH].iter().step_by(6) {
        let pixel = pixel.to_le_bytes();
        let dark = pixel[..3].iter().all(|v| *v < 44);
        let light = pixel[..3].iter().all(|v| *v > 208);
        if !light && !dark {
            return None;
        }
        word = (word << 1) | u32::from(light);
    }
    let value = (word >> 8 & 0xffff) as u16;
    let parity = (value as u8) ^ ((value >> 8) as u8) ^ 0x4b;
    (word >> 24 == 0xa6 && word as u8 == parity).then_some(value)
}

fn result(meta: Meta, producer_qpc: Option<i64>, marker: Option<u16>) -> Frame {
    Frame {
        present_qpc: meta.present, acquired_qpc: meta.acquired, producer_qpc,
        delivered_qpc: producer_qpc.map(|_| qpc()), accumulated: meta.count, stamp: marker,
    }
}

fn read(
    context: &ID3D11DeviceContext, texture: &ID3D11Texture2D, marker_only: bool,
) -> Result<Vec<u32>> {
    let (width, height) = if marker_only { (MARKER_WIDTH, MARKER_HEIGHT) } else { (WIDTH, HEIGHT) };
    let mapped = Mapping::new(context, texture, width)?;
    let mut pixels = Vec::with_capacity(width * height);
    for y in 0..height {
        pixels.extend_from_slice(mapped.row(y, 0, width));
    }
    Ok(pixels)
}

fn marker(pixels: &[u32], marker_only: bool) -> Option<u16> {
    let start = if marker_only {
        MARKER_WIDTH * (MARKER_HEIGHT / 2)
    } else {
        (MARKER.top + MARKER_HEIGHT / 2) * WIDTH + MARKER.left
    };
    stamp(&pixels[start..start + MARKER_WIDTH])
}

pub(super) fn run(arm: &str, adapter_index: u32, output_index: u32, duration_s: f64, timeout: u32) -> Result<()> {
    let (marker_only, bare, ring) = match arm {
        "bare" => (false, true, false),
        "marker" => (true, false, false),
        "sync" => (false, false, false),
        "two_slot" => (false, false, true),
        _ => return Err("Choose bare, marker, sync, or two_slot".into()),
    };
    if !duration_s.is_finite() || duration_s <= 0.0 || duration_s > 10.0 || timeout > 100 {
        return Err("Live duration must be positive and bounded, and timeout cannot exceed 100 ms".into());
    }
    let adapter = adapter(adapter_index)?;
    let (device, context) = device(&adapter)?;
    let output: IDXGIOutput1 = unsafe { adapter.EnumOutputs(output_index)? }.cast()?;
    let description = unsafe { output.GetDesc()? };
    let desktop = description.DesktopCoordinates;
    if !description.AttachedToDesktop.as_bool() || description.Rotation != DXGI_MODE_ROTATION_IDENTITY
        || (desktop.right - desktop.left, desktop.bottom - desktop.top) != (WIDTH as i32, HEIGHT as i32)
    {
        return Err("Expected an attached unrotated 1920×1080 DXGI output".into());
    }
    let name = String::from_utf16_lossy(&description.DeviceName).trim_end_matches('\0').to_owned();
    let duplication = unsafe { output.DuplicateOutput(&device)? };
    let staged: Vec<_> = (0..if bare { 0 } else if ring { 2 } else { 1 }).map(|_| {
        let (w, h) = if marker_only { (MARKER_WIDTH, MARKER_HEIGHT) } else { (WIDTH, HEIGHT) };
        stage(&device, w, h, DXGI_FORMAT_B8G8R8A8_UNORM)
    }).collect::<Result<_>>()?;
    let (sender, receiver) = sync_channel::<(Meta, i64, Vec<u32>)>(2);
    let consumer = std::thread::spawn(move || {
        let mut frames = Vec::new();
        while let Ok((meta, readable, pixels)) = receiver.recv() {
            frames.push(result(meta, Some(readable), marker(&pixels, false)));
        }
        frames
    });
    println!("READY {name}");
    io::stdout().flush()?;
    let mut signal = String::new();
    if io::stdin().read_line(&mut signal)? == 0 || signal.trim() != "GO" {
        return Err("No start signal received".into());
    }
    let duration = Duration::from_secs_f64(duration_s);
    let mut times_out = 0;
    let mut pointer_only = 0;
    let mut pending = None;
    let mut slot = 0;
    let mut captured = Vec::new();
    let mut frame_rows = Vec::new();
    let mut dropped = Vec::new();
    let mut staged_count = 0;
    let begin_cycles = process_cycles();
    let ready = Instant::now();
    let start_qpc = qpc();
    let mut publish = |index: usize, meta: Meta| -> Result<()> {
        let pixels = read(&context, &staged[index], marker_only)?;
        let ready = qpc();
        if ring {
            match sender.try_send((meta, ready, pixels)) {
                Err(TrySendError::Full((meta, _, pixels))) => {
                    dropped.push(result(meta, None, marker(&pixels, false)));
                }
                Err(TrySendError::Disconnected(_)) => return Err("Capture consumer stopped unexpectedly".into()),
                Ok(()) => (),
            }
        } else {
            frame_rows.push(result(meta, Some(ready), marker(&pixels, marker_only)));
        }
        Ok(())
    };
    while ready.elapsed() < duration {
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource = None;
        match unsafe { duplication.AcquireNextFrame(timeout, &mut info, &mut resource) } {
            Err(error) if error.code() == DXGI_ERROR_WAIT_TIMEOUT => {
                times_out += 1;
                if let Some((index, meta)) = pending.take() {
                    publish(index, meta)?;
                }
                continue;
            }
            outcome => outcome?,
        }
        let meta = Meta { present: info.LastPresentTime, acquired: qpc(), count: info.AccumulatedFrames };
        let write = (|| -> Result<bool> {
            if meta.present <= 0 || meta.count == 0 || bare {
                return Ok(false);
            }
            let source: ID3D11Texture2D = resource.as_ref().ok_or("DXGI did not return a resource")?.cast()?;
            let mut description = D3D11_TEXTURE2D_DESC::default();
            unsafe { source.GetDesc(&mut description) };
            if description.Format != DXGI_FORMAT_B8G8R8A8_UNORM
                || (description.Width as usize, description.Height as usize) != (WIDTH, HEIGHT)
            {
                return Err("Expected full unrotated BGRA8 SDR texture".into());
            }
            unsafe {
                if marker_only {
                    context.CopySubresourceRegion(&staged[slot], 0, 0, 0, 0, &source, 0, Some(&MARKER.raw()));
                } else {
                    context.CopyResource(&staged[slot], &source);
                }
            }
            Ok(true)
        })();
        drop(resource);
        unsafe { duplication.ReleaseFrame()? };
        let written = write?;
        if meta.present <= 0 || meta.count == 0 {
            pointer_only += 1;
            continue;
        }
        captured.push(result(meta, None, None));
        if written {
            staged_count += 1;
            if ring {
                if let Some((index, previous)) = pending.replace((slot, meta)) {
                    publish(index, previous)?;
                }
                slot ^= 1;
            } else {
                publish(slot, meta)?;
            }
        }
    }
    if let Some((index, meta)) = pending {
        publish(index, meta)?;
    }
    drop(sender);
    let downstream = consumer.join().map_err(|_| "Native capture consumer panicked")?;
    frame_rows.extend(downstream);
    let cycles = process_cycles().zip(begin_cycles).map(|(a, b)| a.saturating_sub(b));
    let end_qpc = qpc();
    let mut frequency = 0;
    unsafe { QueryPerformanceFrequency(&mut frequency) };
    let unique_present_times = captured.iter().map(|frame| frame.present_qpc).collect::<BTreeSet<_>>().len();
    let payload = if marker_only { MARKER.bytes() } else { LOAD };
    println!("{}", serde_json::json!({
        "arm": arm, "monitor": name, "start_qpc": start_qpc, "end_qpc": end_qpc, "qpc_hz": frequency,
        "elapsed_s": ready.elapsed().as_secs_f64(), "unique_present_times": unique_present_times,
        "native_acquisitions": captured, "frames": frame_rows, "queue_drops": dropped,
        "pointer_only": pointer_only, "timeouts": times_out, "process_cycles": cycles,
        "staged_count": staged_count, "gpu_copy_payload_bytes": staged_count * payload,
        "cpu_copy_payload_bytes": staged_count * payload, "staging_slots": staged.len(),
        "staging_allocation_logical_bytes": staged.len() * payload,
    }));
    Ok(())
}
