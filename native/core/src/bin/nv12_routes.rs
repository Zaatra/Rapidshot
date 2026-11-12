//! Measure the two routes from a captured frame to an encoder's NV12 input.
//!
//! - **converter**: `Converter12` writing `Nv12Texture` on D3D12. Its input
//!   must be a shareable texture, and a Desktop Duplication surface is not, so
//!   in a real pipeline the frame is first copied into one on the GPU; that
//!   copy is timed as part of the route, and also reported without it (WGC
//!   hands back such a copy already). The NV12 texture is D3D11's, on the
//!   capture device, so it is read there as an encoder would read it and
//!   checked byte for byte against D3D12's readback of the same memory.
//! - **video processor**: one `VideoProcessorBlt` from the desktop surface to an
//!   NV12 texture on the capture device.
//!
//! Each runs on the same held frame, for a full-frame conversion and for a
//! crop-and-scale. Times are submit-to-GPU-done as the CPU sees it (median,
//! p95, max over the timed runs); CPU is this process's user+kernel time per
//! frame. Accuracy is against an exact CPU reference (BT.709, limited range,
//! left-sited chroma) for the full frame; for the scaled case the two filters
//! differ by design, so only how far apart the routes are is reported.
//!
//! Not measured: GPU engine time, and power. CPU per frame comes from
//! `GetProcessTimes`, which ticks every 15.6 ms, so over the default 300 runs
//! it resolves about 0.05 ms per frame; less reads as 0.
//!
//! Usage: `nv12_routes [--iterations N] [--json PATH]`

use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Foundation::{FILETIME, HMODULE};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

use rapidshot_core::converter12::{
    Converter12, Crop, Matrix, OutputFormat, Sampling, ToneMap, YuvOptions,
};
use rapidshot_core::duplicator::Duplicator;
use rapidshot_core::video_processor::{read_back_nv12, SourceRect, VideoProcessorNv12};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

fn cpu_seconds() -> f64 {
    let (mut c, mut e, mut k, mut u) = (
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
        FILETIME::default(),
    );
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }.unwrap();
    let t = |f: FILETIME| ((f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64) as f64 / 1e7;
    t(k) + t(u)
}

struct Timing {
    median_ms: f64,
    p95_ms: f64,
    max_ms: f64,
    cpu_ms_per_frame: f64,
}

fn time_it(iterations: usize, mut run: impl FnMut() -> Res<()>) -> Res<Timing> {
    for _ in 0..20 {
        run()?;
    }
    let mut samples = Vec::with_capacity(iterations);
    let cpu0 = cpu_seconds();
    for _ in 0..iterations {
        let t0 = Instant::now();
        run()?;
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let cpu = (cpu_seconds() - cpu0) * 1000.0 / iterations as f64;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let at = |p: f64| samples[((samples.len() - 1) as f64 * p).round() as usize];
    Ok(Timing {
        median_ms: at(0.5),
        p95_ms: at(0.95),
        max_ms: *samples.last().unwrap(),
        cpu_ms_per_frame: cpu,
    })
}

/// How close `got` is to `want`, per plane: share exact, share within one
/// code, largest difference, mean absolute difference.
struct Accuracy {
    y_exact: f64,
    y_within_1: f64,
    y_max: i32,
    y_mean: f64,
    uv_exact: f64,
    uv_within_1: f64,
    uv_max: i32,
    uv_mean: f64,
}

fn accuracy(got: &[u8], want: &[u8], width: usize, height: usize) -> Accuracy {
    let split = width * height;
    let stats = |a: &[u8], b: &[u8]| {
        let (mut exact, mut within, mut max, mut sum) = (0usize, 0usize, 0i32, 0f64);
        for (x, y) in a.iter().zip(b) {
            let d = (*x as i32 - *y as i32).abs();
            exact += (d == 0) as usize;
            within += (d <= 1) as usize;
            max = max.max(d);
            sum += d as f64;
        }
        let n = a.len() as f64;
        (exact as f64 / n, within as f64 / n, max, sum / n)
    };
    let (ye, yw, ym, ya) = stats(&got[..split], &want[..split]);
    let (ue, uw, um, ua) = stats(&got[split..], &want[split..]);
    Accuracy {
        y_exact: ye,
        y_within_1: yw,
        y_max: ym,
        y_mean: ya,
        uv_exact: ue,
        uv_within_1: uw,
        uv_max: um,
        uv_mean: ua,
    }
}

/// Exact NV12 from BGRA bytes: BT.709, limited range, left-sited chroma -- the
/// rule `tests/test_gpu_converter_yuv.py` checks the converter against.
fn reference_nv12(bgra: &[u8], width: usize, height: usize) -> Vec<u8> {
    let (kr, kb) = (0.2126f64, 0.0722f64);
    let rgb = |x: usize, y: usize| {
        let i = (y * width + x) * 4;
        [
            bgra[i + 2] as f64 / 255.0,
            bgra[i + 1] as f64 / 255.0,
            bgra[i] as f64 / 255.0,
        ]
    };
    let luma = |c: [f64; 3]| kr * c[0] + (1.0 - kr - kb) * c[1] + kb * c[2];
    let code = |v: f64| (v + 0.5).floor().clamp(0.0, 255.0) as u8;
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    for y in 0..height {
        for x in 0..width {
            out.push(code(16.0 + 219.0 * luma(rgb(x, y))));
        }
    }
    let sited = |cx: usize, y: usize| {
        let l = rgb(cx.saturating_sub(1), y);
        let c = rgb(cx, y);
        let r = rgb(cx + 1, y);
        [0, 1, 2].map(|i| 0.25 * l[i] + 0.5 * c[i] + 0.25 * r[i])
    };
    for cy in (0..height).step_by(2) {
        for cx in (0..width).step_by(2) {
            let (a, b) = (sited(cx, cy), sited(cx, cy + 1));
            let c = [0, 1, 2].map(|i| 0.5 * (a[i] + b[i]));
            let yv = luma(c);
            out.push(code(128.0 + 224.0 * (c[2] - yv) / (2.0 * (1.0 - kb))));
            out.push(code(128.0 + 224.0 * (c[0] - yv) / (2.0 * (1.0 - kr))));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn texture(
    device: &ID3D11Device,
    width: u32,
    height: u32,
    format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT,
    usage: D3D11_USAGE,
    bind: u32,
    cpu: u32,
    misc: u32,
) -> Res<ID3D11Texture2D> {
    let desc = D3D11_TEXTURE2D_DESC {
        Width: width,
        Height: height,
        MipLevels: 1,
        ArraySize: 1,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Usage: usage,
        BindFlags: bind,
        CPUAccessFlags: cpu,
        MiscFlags: misc,
    };
    let mut t = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut t))? };
    Ok(t.unwrap())
}

fn read_bgra(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
    src: &ID3D11Texture2D,
    width: u32,
    height: u32,
) -> Res<Vec<u8>> {
    let staging = texture(
        device,
        width,
        height,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        D3D11_USAGE_STAGING,
        0,
        D3D11_CPU_ACCESS_READ.0 as u32,
        0,
    )?;
    let mut out = Vec::with_capacity((width * height * 4) as usize);
    unsafe {
        context.CopyResource(&staging, src);
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context.Map(&staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        for row in 0..height as usize {
            let p = (mapped.pData as *const u8).add(row * mapped.RowPitch as usize);
            out.extend_from_slice(std::slice::from_raw_parts(p, width as usize * 4));
        }
        context.Unmap(&staging, 0);
    }
    Ok(out)
}

struct Scenario {
    name: &'static str,
    crop: Option<(u32, u32, u32, u32)>,
    out: (u32, u32),
    sampling: Sampling,
}

fn main() -> Res<()> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let iterations: usize = arg("--iterations").map_or(300, |v| v.parse().unwrap());
    let json_path = arg("--json");

    // Capture device: the adapter that owns the primary output, with video support
    // so its video processor is reachable.
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let adapter = unsafe { factory.EnumAdapters1(0)? };
    let adapter_desc = unsafe { adapter.GetDesc1()? };
    let gpu = String::from_utf16_lossy(&adapter_desc.Description)
        .trim_end_matches('\0')
        .to_owned();
    let output: IDXGIOutput1 = unsafe { adapter.EnumOutputs(0)?.cast()? };
    let mut device = None;
    unsafe {
        D3D11CreateDevice(
            &adapter,
            D3D_DRIVER_TYPE_UNKNOWN,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT,
            None,
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            None,
        )?;
    }
    let device: ID3D11Device = device.unwrap();
    let context = unsafe { device.GetImmediateContext()? };

    // One frame with content, held for the whole run, through the native port.
    let mut dup = Duplicator::new(&output, &device, 100)?;
    let frame = loop {
        dup.update_frame()?;
        if dup.updated {
            break dup
                .texture()
                .cloned()
                .expect("updated frames carry a texture");
        }
        dup.release_frame();
    };
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { frame.GetDesc(&mut desc) };
    if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
        return Err(format!("desktop is {:?}, not BGRA8 (an HDR desktop?)", desc.Format).into());
    }
    let (sw, sh) = (desc.Width, desc.Height);
    let source_bytes = read_bgra(&device, &context, &frame, sw, sh)?;

    // The shareable copy the converter needs.
    let shared_src = texture(
        &device,
        sw,
        sh,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        D3D11_USAGE_DEFAULT,
        (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32,
        0,
        (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32,
    )?;

    let crop_w = (sw * 5 / 6) & !1;
    let crop_h = (sh * 5 / 6) & !1;
    let scenarios = [
        Scenario {
            name: "full frame, no scaling",
            crop: None,
            out: (sw & !1, sh & !1),
            sampling: Sampling::Nearest,
        },
        Scenario {
            name: "crop 5/6 and scale to 1280x720",
            crop: Some(((sw - crop_w) / 2, (sh - crop_h) / 2, crop_w, crop_h)),
            out: (1280, 720),
            sampling: Sampling::Bilinear,
        },
    ];

    let mut report = Vec::new();
    println!("GPU: {gpu}\nsource: {sw}x{sh} BGRA8, {iterations} timed runs per row\n");
    println!("| case | route | median ms | p95 ms | max ms | CPU ms/frame |");
    println!("|---|---|---|---|---|---|");
    for s in &scenarios {
        let (ow, oh) = s.out;
        let crop = s.crop.map(|(x, y, w, h)| Crop {
            x,
            y,
            width: w,
            height: h,
        });
        let regions: Vec<Crop> = crop.into_iter().collect();

        // -- converter -------------------------------------------------------
        let converter = Converter12::new(
            &shared_src,
            ow,
            oh,
            s.sampling,
            OutputFormat::Nv12Texture,
            YuvOptions {
                matrix: Matrix::Bt709,
                full_range: false,
            },
            1,
        )?;
        let with_copy = time_it(iterations, || {
            unsafe { context.CopyResource(&shared_src, &frame) };
            converter.process(&shared_src, 1, 1.0, 0.0, 0, ToneMap::None, &regions)?;
            Ok(())
        })?;
        unsafe { context.CopyResource(&shared_src, &frame) };
        let process_only = time_it(iterations, || {
            converter.process(&shared_src, 1, 1.0, 0.0, 0, ToneMap::None, &regions)?;
            Ok(())
        })?;
        let converter_bytes = converter.read_back()?;

        // What an encoder on the capture device reads: the D3D11 texture itself.
        let opened = converter
            .output_texture()
            .cloned()
            .expect("an Nv12Texture converter has a D3D11 texture");
        let staging = texture(
            &device,
            ow,
            oh,
            DXGI_FORMAT_NV12,
            D3D11_USAGE_STAGING,
            0,
            D3D11_CPU_ACCESS_READ.0 as u32,
            0,
        )?;
        let encoder_view = read_back_nv12(&context, &staging, &opened, ow, oh)?;
        let encoder_sees_same = encoder_view == converter_bytes;

        // -- video processor ---------------------------------------------------
        let mut vp = VideoProcessorNv12::new(&device, sw, sh, ow, oh)?;
        let vp_crop = s.crop.map(|(x, y, w, h)| SourceRect {
            left: x as i32,
            top: y as i32,
            right: (x + w) as i32,
            bottom: (y + h) as i32,
        });
        let direct = vp.process(&frame, vp_crop).is_ok();
        let vp_input = if direct { &frame } else { &shared_src };
        let vp_timing = time_it(iterations, || {
            vp.process(vp_input, vp_crop)?;
            vp.wait()?;
            Ok(())
        })?;
        let vp_bytes = vp.read_back()?;

        let row = |route: &str, t: &Timing| {
            println!(
                "| {} | {route} | {:.3} | {:.3} | {:.3} | {:.3} |",
                s.name, t.median_ms, t.p95_ms, t.max_ms, t.cpu_ms_per_frame
            );
        };
        row("converter, incl. copy to a shareable texture", &with_copy);
        row("converter, process only", &process_only);
        row(
            if direct {
                "video processor (from the desktop surface)"
            } else {
                "video processor (from a copy: the desktop surface was refused)"
            },
            &vp_timing,
        );

        let (want, label) = if s.crop.is_none() {
            (
                reference_nv12(&source_bytes, ow as usize, oh as usize),
                "exact reference",
            )
        } else {
            (converter_bytes.clone(), "the converter")
        };
        let conv_acc = accuracy(&converter_bytes, &want, ow as usize, oh as usize);
        let vp_acc = accuracy(&vp_bytes, &want, ow as usize, oh as usize);
        report.push(serde_like(
            s.name,
            &with_copy,
            &process_only,
            &vp_timing,
            direct,
            encoder_sees_same,
            vp.conversion_supported,
            label,
            &conv_acc,
            &vp_acc,
        ));
        println!();
        println!("  accuracy against {label}:");
        for (route, a) in [("converter", &conv_acc), ("video processor", &vp_acc)] {
            println!(
                "    {route:16} Y exact {:.2}% (within 1: {:.2}%, max {}, mean {:.3}) | \
                 CbCr exact {:.2}% (within 1: {:.2}%, max {}, mean {:.3})",
                a.y_exact * 100.0,
                a.y_within_1 * 100.0,
                a.y_max,
                a.y_mean,
                a.uv_exact * 100.0,
                a.uv_within_1 * 100.0,
                a.uv_max,
                a.uv_mean
            );
        }
        println!(
            "  the D3D11 texture an encoder reads holds the converter's bytes exactly: {encoder_sees_same}"
        );
        println!(
            "  driver says it supports BT.709 limited left-sited NV12: {:?}\n",
            vp.conversion_supported
        );
        println!("| case | route | median ms | p95 ms | max ms | CPU ms/frame |");
        println!("|---|---|---|---|---|---|");
    }
    dup.release_frame();

    if let Some(path) = json_path {
        let body = format!(
            "{{\"gpu\": {:?}, \"source\": [{sw}, {sh}], \"iterations\": {iterations}, \"cases\": [{}]}}",
            gpu,
            report.join(", ")
        );
        std::fs::write(&path, body)?;
        println!("wrote {path}");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn serde_like(
    name: &str,
    with_copy: &Timing,
    process_only: &Timing,
    vp: &Timing,
    direct: bool,
    encoder_sees_same: bool,
    supported: Option<bool>,
    against: &str,
    conv: &Accuracy,
    vpa: &Accuracy,
) -> String {
    let t = |t: &Timing| {
        format!(
            "{{\"median_ms\": {:.4}, \"p95_ms\": {:.4}, \"max_ms\": {:.4}, \"cpu_ms_per_frame\": {:.4}}}",
            t.median_ms, t.p95_ms, t.max_ms, t.cpu_ms_per_frame
        )
    };
    let a = |a: &Accuracy| {
        format!(
            "{{\"y_exact\": {:.5}, \"y_within_1\": {:.5}, \"y_max\": {}, \"y_mean\": {:.4}, \
             \"uv_exact\": {:.5}, \"uv_within_1\": {:.5}, \"uv_max\": {}, \"uv_mean\": {:.4}}}",
            a.y_exact,
            a.y_within_1,
            a.y_max,
            a.y_mean,
            a.uv_exact,
            a.uv_within_1,
            a.uv_max,
            a.uv_mean
        )
    };
    format!(
        "{{\"case\": {name:?}, \"converter_with_copy\": {}, \"converter_process_only\": {}, \
         \"video_processor\": {}, \"video_processor_direct\": {direct}, \
         \"encoder_reads_converter_exactly\": {encoder_sees_same}, \
         \"driver_supports_conversion\": {}, \"accuracy_against\": {against:?}, \
         \"converter_accuracy\": {}, \"video_processor_accuracy\": {}}}",
        t(with_copy),
        t(process_only),
        t(vp),
        supported.map_or("null".to_owned(), |b| b.to_string()),
        a(conv),
        a(vpa)
    )
}
