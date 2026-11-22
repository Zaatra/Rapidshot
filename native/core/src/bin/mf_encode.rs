//! Hardware H.264 encoding of captured frames, in place on the GPU.
//!
//! ROADMAP § 7.3a step 4. The PyAV measurements showed the encoder is not the
//! bottleneck: getting NV12 to it is -- the readback to system memory alone
//! was 6.5 ms of a 25.7 ms screen-to-packet. `Converter12` already writes NV12
//! into a D3D11 texture on the capture device; this feeds that texture to a
//! hardware Media Foundation encoder on the same device, with no readback and
//! no CPU copy, and measures it.
//!
//! - `mf_encode --list`: the hardware H.264 encoders, their adapters, and
//!   whether each activates.
//! - `mf_encode --seconds 5 --out run.h264 --times run.json [--verify DIR]`:
//!   capture the primary display for that long and encode every new frame.
//!   `run.json` holds, per frame, the QPC times of Windows' present, the
//!   capture, the finished NV12 texture, the submission, and the packet
//!   coming out; the Python harness decodes `run.h264` and judges it. With
//!   `--verify`, every 10th frame's NV12 is also read back to DIR -- which
//!   costs time, so a verify run is for correctness, not for timing.
//!
//! Hardware MFTs are asynchronous: "need input" and "have output" arrive as
//! events. A thread of its own waits for them, so a packet is timed when the
//! encoder produces it rather than when the capture loop next looks. Media
//! Foundation needs a multithreaded COM apartment (measured in
//! `encode_latency.py`: FFmpeg's MF encoder refuses an STA thread).
//!
//! The converter rewrites its NV12 texture every frame, and the encoder may
//! still be reading the last one, so each frame is first copied on the GPU
//! into a ring slot the encoder owns until it is done with it.
use std::collections::VecDeque;
use std::fs;
use std::io::Write;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use rapidshot_core::converter12::{
    Converter12, Crop, Matrix, OutputFormat, Sampling, ToneMap, YuvOptions,
};
use rapidshot_core::duplicator::Duplicator;
use windows::core::{IUnknown, Interface, PWSTR};
use windows::Win32::Foundation::{HMODULE, LUID, VARIANT_FALSE, VARIANT_TRUE};
use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_UNKNOWN;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Media::MediaFoundation::*;
use windows::Win32::System::Com::{CoInitializeEx, CoTaskMemFree, COINIT_MULTITHREADED};
use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};
use windows::Win32::System::Variant::{VARIANT, VT_BOOL, VT_UI4};

type Res<T> = Result<T, Box<dyn std::error::Error>>;

const RING: usize = 4;
const VERIFY_EVERY: u64 = 10;

fn qpc() -> i64 {
    let mut v = 0i64;
    unsafe {
        let _ = QueryPerformanceCounter(&mut v);
    }
    v
}

fn variant_bool(value: bool) -> VARIANT {
    let mut v = VARIANT::default();
    unsafe {
        let inner = &mut *v.Anonymous.Anonymous;
        inner.vt = VT_BOOL;
        inner.Anonymous.boolVal = if value { VARIANT_TRUE } else { VARIANT_FALSE };
    }
    v
}

fn variant_u32(value: u32) -> VARIANT {
    let mut v = VARIANT::default();
    unsafe {
        let inner = &mut *v.Anonymous.Anonymous;
        inner.vt = VT_UI4;
        inner.Anonymous.ulVal = value;
    }
    v
}

fn luid_u64(luid: LUID) -> u64 {
    ((luid.HighPart as u32 as u64) << 32) | luid.LowPart as u64
}

/// A COM pointer used from the event thread. Media Foundation objects here
/// live in the multithreaded apartment, where that is allowed; windows-rs
/// cannot know it, so the claim is made once, here.
struct Mta<T>(T);
unsafe impl<T> Send for Mta<T> {}
unsafe impl<T> Sync for Mta<T> {}

struct HwEncoder {
    name: String,
    activate: IMFActivate,
}

fn hardware_h264_encoders() -> Res<Vec<HwEncoder>> {
    let input = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Video,
        guidSubtype: MFVideoFormat_NV12,
    };
    let output = MFT_REGISTER_TYPE_INFO {
        guidMajorType: MFMediaType_Video,
        guidSubtype: MFVideoFormat_H264,
    };
    let mut list: *mut Option<IMFActivate> = std::ptr::null_mut();
    let mut count = 0u32;
    unsafe {
        MFTEnumEx(
            MFT_CATEGORY_VIDEO_ENCODER,
            MFT_ENUM_FLAG_HARDWARE | MFT_ENUM_FLAG_SORTANDFILTER,
            Some(&input),
            Some(&output),
            &mut list,
            &mut count,
        )?;
    }
    let mut found = Vec::new();
    for i in 0..count as usize {
        let Some(activate) = (unsafe { (*list.add(i)).take() }) else {
            continue;
        };
        let name = unsafe {
            let mut text = PWSTR::null();
            let mut len = 0u32;
            if activate
                .GetAllocatedString(&MFT_FRIENDLY_NAME_Attribute, &mut text, &mut len)
                .is_ok()
            {
                let s = text.to_string().unwrap_or_default();
                CoTaskMemFree(Some(text.0 as *const _));
                s
            } else {
                "(unnamed)".into()
            }
        };
        found.push(HwEncoder { name, activate });
    }
    unsafe { CoTaskMemFree(Some(list as *const _)) };
    Ok(found)
}

fn transform_luid(transform: &IMFTransform) -> Option<u64> {
    let attributes = unsafe { transform.GetAttributes() }.ok()?;
    let mut blob = [0u8; 8];
    let mut size = 0u32;
    if unsafe { attributes.GetBlob(&MFT_ENUM_ADAPTER_LUID, &mut blob, Some(&mut size)) }.is_ok()
        && size == 8
    {
        return Some(u64::from_le_bytes(blob));
    }
    unsafe { attributes.GetUINT64(&MFT_ENUM_ADAPTER_LUID) }.ok()
}

fn list() -> Res<()> {
    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let mut adapters = Vec::new();
    let mut i = 0;
    while let Ok(adapter) = unsafe { factory.EnumAdapters1(i) } {
        let desc = unsafe { adapter.GetDesc1()? };
        adapters.push((
            luid_u64(desc.AdapterLuid),
            String::from_utf16_lossy(&desc.Description)
                .trim_end_matches('\0')
                .to_owned(),
        ));
        i += 1;
    }
    println!("adapters:");
    for (luid, name) in &adapters {
        println!("  {luid:#018x}  {name}");
    }
    println!("hardware H.264 encoders (NV12 in), in Media Foundation's order:");
    for e in hardware_h264_encoders()? {
        match unsafe { e.activate.ActivateObject::<IMFTransform>() } {
            Ok(transform) => {
                let adapter = transform_luid(&transform)
                    .and_then(|l| adapters.iter().find(|(a, _)| *a == l))
                    .map_or("no adapter LUID reported".to_owned(), |(_, n)| n.clone());
                let aware = unsafe { transform.GetAttributes() }
                    .and_then(|a| unsafe { a.GetUINT32(&MF_SA_D3D11_AWARE) })
                    .unwrap_or(0)
                    != 0;
                println!(
                    "  {} | activates | adapter: {adapter} | D3D11-aware: {aware}",
                    e.name
                );
            }
            Err(err) => println!("  {} | REFUSES to activate: {err}", e.name),
        }
        unsafe {
            let _ = e.activate.ShutdownObject();
        }
    }
    Ok(())
}

struct Settings {
    width: u32,
    height: u32,
    fps: u32,
    bitrate: u32,
    /// CODECAPI_AVEncCommonQualityVsSpeed, 0 (fastest) to 100; None leaves the encoder's default.
    speed: Option<u32>,
    low_latency: bool,
}

/// Which codec properties the encoder accepted; reported, because a refused
/// low-latency switch would otherwise be measured as if it applied.
fn configure(transform: &IMFTransform, s: &Settings) -> Res<Vec<(String, String)>> {
    let mut applied = Vec::new();
    unsafe {
        let attributes = transform.GetAttributes()?;
        attributes.SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1)?;
        let _ = attributes.SetUINT32(&MF_LOW_LATENCY, s.low_latency as u32);

        let out = MFCreateMediaType()?;
        out.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        out.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_H264)?;
        out.SetUINT32(&MF_MT_AVG_BITRATE, s.bitrate)?;
        out.SetUINT64(
            &MF_MT_FRAME_SIZE,
            ((s.width as u64) << 32) | s.height as u64,
        )?;
        out.SetUINT64(&MF_MT_FRAME_RATE, ((s.fps as u64) << 32) | 1)?;
        out.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        out.SetUINT32(&MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main.0 as u32)?;
        transform.SetOutputType(0, &out, 0)?;

        let input = MFCreateMediaType()?;
        input.SetGUID(&MF_MT_MAJOR_TYPE, &MFMediaType_Video)?;
        input.SetGUID(&MF_MT_SUBTYPE, &MFVideoFormat_NV12)?;
        input.SetUINT64(
            &MF_MT_FRAME_SIZE,
            ((s.width as u64) << 32) | s.height as u64,
        )?;
        input.SetUINT64(&MF_MT_FRAME_RATE, ((s.fps as u64) << 32) | 1)?;
        input.SetUINT32(&MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive.0 as u32)?;
        transform.SetInputType(0, &input, 0)?;

        if let Ok(codec) = transform.cast::<ICodecAPI>() {
            let mut set = |name: &str, key: &windows::core::GUID, value: VARIANT| {
                let result = codec.SetValue(key, &value);
                applied.push((
                    name.to_owned(),
                    match result {
                        Ok(()) => "applied".to_owned(),
                        Err(e) => format!("refused: {e}"),
                    },
                ));
            };
            set(
                &format!("AVLowLatencyMode={}", s.low_latency),
                &CODECAPI_AVLowLatencyMode,
                variant_bool(s.low_latency),
            );
            set(
                "AVEncCommonRateControlMode=CBR",
                &CODECAPI_AVEncCommonRateControlMode,
                variant_u32(eAVEncCommonRateControlMode_CBR.0 as u32),
            );
            set(
                "AVEncCommonMeanBitRate",
                &CODECAPI_AVEncCommonMeanBitRate,
                variant_u32(s.bitrate),
            );
            set(
                "AVEncMPVDefaultBPictureCount=0",
                &CODECAPI_AVEncMPVDefaultBPictureCount,
                variant_u32(0),
            );
            set(
                "AVEncMPVGOPSize",
                &CODECAPI_AVEncMPVGOPSize,
                variant_u32(s.fps * 2),
            );
            if let Some(speed) = s.speed {
                set(
                    &format!("AVEncCommonQualityVsSpeed={speed}"),
                    &CODECAPI_AVEncCommonQualityVsSpeed,
                    variant_u32(speed),
                );
            }
        } else {
            applied.push(("ICodecAPI".into(), "not exposed".into()));
        }
    }
    Ok(applied)
}

#[derive(Default)]
struct Credits {
    need_input: u32,
    drained: bool,
    error: Option<String>,
}

struct Packet {
    sample_time: i64,
    out_qpc: i64,
    bytes: Vec<u8>,
}

/// Wait for encoder events: count "need input", take every output.
fn event_loop(
    transform: Mta<IMFTransform>,
    credits: Arc<(Mutex<Credits>, Condvar)>,
    packets: Arc<Mutex<Vec<Packet>>>,
) {
    let transform = transform.0;
    let generator: IMFMediaEventGenerator = match transform.cast() {
        Ok(g) => g,
        Err(e) => {
            credits.0.lock().unwrap().error = Some(format!("no event generator: {e}"));
            credits.1.notify_all();
            return;
        }
    };
    let provides_samples = unsafe { transform.GetOutputStreamInfo(0) }
        .map(|i| i.dwFlags & MFT_OUTPUT_STREAM_PROVIDES_SAMPLES.0 as u32 != 0)
        .unwrap_or(true);
    let output_size = unsafe { transform.GetOutputStreamInfo(0) }
        .map(|i| i.cbSize)
        .unwrap_or(0);
    loop {
        let event = match unsafe { generator.GetEvent(MEDIA_EVENT_GENERATOR_GET_EVENT_FLAGS(0)) } {
            Ok(e) => e,
            Err(e) => {
                credits.0.lock().unwrap().error = Some(format!("GetEvent: {e}"));
                credits.1.notify_all();
                return;
            }
        };
        let kind = unsafe { event.GetType() }.unwrap_or(0);
        if kind == METransformNeedInput.0 as u32 {
            credits.0.lock().unwrap().need_input += 1;
            credits.1.notify_all();
        } else if kind == METransformHaveOutput.0 as u32 {
            let mut buffer = MFT_OUTPUT_DATA_BUFFER {
                dwStreamID: 0,
                pSample: ManuallyDrop::new(None),
                dwStatus: 0,
                pEvents: ManuallyDrop::new(None),
            };
            if !provides_samples {
                let sample = unsafe { MFCreateSample() }.ok();
                if let (Some(sample), Ok(media)) = (sample.as_ref(), unsafe {
                    MFCreateMemoryBuffer(output_size.max(4 << 20))
                }) {
                    unsafe {
                        let _ = sample.AddBuffer(&media);
                    }
                }
                buffer.pSample = ManuallyDrop::new(sample);
            }
            let mut status = 0u32;
            let result = unsafe {
                transform.ProcessOutput(0, std::slice::from_mut(&mut buffer), &mut status)
            };
            let out_qpc = qpc();
            let sample = unsafe { ManuallyDrop::take(&mut buffer.pSample) };
            unsafe {
                drop(ManuallyDrop::take(&mut buffer.pEvents));
            }
            match result {
                Ok(()) => {
                    if let Some(sample) = sample {
                        let sample_time = unsafe { sample.GetSampleTime() }.unwrap_or(-1);
                        let mut bytes = Vec::new();
                        if let Ok(media) = unsafe { sample.ConvertToContiguousBuffer() } {
                            let mut data = std::ptr::null_mut();
                            let mut len = 0u32;
                            if unsafe { media.Lock(&mut data, None, Some(&mut len)) }.is_ok() {
                                bytes = unsafe { std::slice::from_raw_parts(data, len as usize) }
                                    .to_vec();
                                let _ = unsafe { media.Unlock() };
                            }
                        }
                        packets.lock().unwrap().push(Packet {
                            sample_time,
                            out_qpc,
                            bytes,
                        });
                    }
                }
                Err(e) if e.code() == MF_E_TRANSFORM_STREAM_CHANGE => {
                    // The encoder settled its output type: take it as offered.
                    if let Ok(t) = unsafe { transform.GetOutputAvailableType(0, 0) } {
                        let _ = unsafe { transform.SetOutputType(0, &t, 0) };
                    }
                }
                Err(e) => {
                    credits.0.lock().unwrap().error = Some(format!("ProcessOutput: {e}"));
                    credits.1.notify_all();
                    return;
                }
            }
        } else if kind == METransformDrainComplete.0 as u32 {
            credits.0.lock().unwrap().drained = true;
            credits.1.notify_all();
            return;
        }
    }
}

fn texture_like(device: &ID3D11Device, like: &ID3D11Texture2D) -> Res<ID3D11Texture2D> {
    let mut desc = D3D11_TEXTURE2D_DESC::default();
    unsafe { like.GetDesc(&mut desc) };
    desc.MiscFlags = 0;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE.0 as u32;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.CPUAccessFlags = 0;
    let mut texture = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture))? };
    Ok(texture.unwrap())
}

fn encode(args: &[String]) -> Res<()> {
    let arg = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let seconds: f64 = arg("--seconds").map_or(5.0, |v| v.parse().unwrap());
    let fps: u32 = arg("--fps").map_or(60, |v| v.parse().unwrap());
    let bitrate: u32 = arg("--bitrate").map_or(20_000_000, |v| v.parse().unwrap());
    let speed: Option<u32> = arg("--quality-vs-speed").map(|v| v.parse().unwrap());
    let low_latency = !args.iter().any(|a| a == "--no-low-latency");
    let out_path = arg("--out").unwrap_or_else(|| "mf_encode.h264".into());
    let times_path = arg("--times").unwrap_or_else(|| "mf_encode.json".into());
    let verify_dir = arg("--verify");
    // Encode copies of one converted frame and capture nothing: the encoder's
    // own speed, with nothing else on the GPU.
    let encode_only = args.iter().any(|a| a == "--encode-only");

    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1()? };
    let adapter = unsafe { factory.EnumAdapters1(0)? };
    let adapter_desc = unsafe { adapter.GetDesc1()? };
    let gpu = String::from_utf16_lossy(&adapter_desc.Description)
        .trim_end_matches('\0')
        .to_owned();
    let capture_luid = luid_u64(adapter_desc.AdapterLuid);
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
    // The encoder calls into this device from its own threads.
    let multithread: ID3D11Multithread = device.cast()?;
    unsafe {
        let _ = multithread.SetMultithreadProtected(true);
    }
    let context = unsafe { device.GetImmediateContext()? };

    // The encoder on the capture adapter.
    let mut chosen = None;
    for e in hardware_h264_encoders()? {
        if let Ok(t) = unsafe { e.activate.ActivateObject::<IMFTransform>() } {
            let luid = transform_luid(&t);
            if luid.is_none() || luid == Some(capture_luid) {
                chosen = Some((e.name.clone(), t, e.activate));
                break;
            }
            let _ = unsafe { e.activate.ShutdownObject() };
        }
    }
    let (encoder_name, transform, activate) =
        chosen.ok_or("no hardware H.264 encoder activates on the capture adapter")?;
    // An asynchronous MFT refuses every call until unlocked
    // (MF_E_TRANSFORM_ASYNC_LOCKED), including the D3D manager below.
    unsafe {
        transform
            .GetAttributes()?
            .SetUINT32(&MF_TRANSFORM_ASYNC_UNLOCK, 1)?;
    }

    let mut token = 0u32;
    let mut manager = None;
    unsafe { MFCreateDXGIDeviceManager(&mut token, &mut manager)? };
    let manager: IMFDXGIDeviceManager = manager.unwrap();
    unsafe {
        manager.ResetDevice(&device, token)?;
        transform.ProcessMessage(MFT_MESSAGE_SET_D3D_MANAGER, manager.as_raw() as usize)?;
    }

    // One frame to size everything by.
    let mut dup = Duplicator::new(&output, &device, 20)?;
    let first = loop {
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
    unsafe { first.GetDesc(&mut desc) };
    dup.release_frame();
    if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
        return Err(format!("desktop is {:?}, not BGRA8", desc.Format).into());
    }
    let (width, height) = (desc.Width & !1, desc.Height & !1);

    let shared_src = {
        let mut d = desc;
        d.Usage = D3D11_USAGE_DEFAULT;
        d.BindFlags = (D3D11_BIND_SHADER_RESOURCE.0 | D3D11_BIND_RENDER_TARGET.0) as u32;
        d.CPUAccessFlags = 0;
        d.MiscFlags = (D3D11_RESOURCE_MISC_SHARED_NTHANDLE.0 | D3D11_RESOURCE_MISC_SHARED.0) as u32;
        let mut t = None;
        unsafe { device.CreateTexture2D(&d, None, Some(&mut t))? };
        t.unwrap()
    };
    let converter = Converter12::new(
        &shared_src,
        width,
        height,
        Sampling::Nearest,
        OutputFormat::Nv12Texture,
        YuvOptions {
            matrix: Matrix::Bt709,
            full_range: false,
        },
        1,
    )?;
    let nv12 = converter
        .output_texture()
        .ok_or("the converter made no NV12 texture")?
        .clone();
    let ring: Vec<ID3D11Texture2D> = (0..RING)
        .map(|_| texture_like(&device, &nv12))
        .collect::<Res<_>>()?;

    let settings = Settings {
        width,
        height,
        fps,
        bitrate,
        speed,
        low_latency,
    };
    let applied = configure(&transform, &settings)?;

    let credits = Arc::new((Mutex::new(Credits::default()), Condvar::new()));
    let packets = Arc::new(Mutex::new(Vec::<Packet>::new()));
    unsafe {
        transform.ProcessMessage(MFT_MESSAGE_NOTIFY_BEGIN_STREAMING, 0)?;
        transform.ProcessMessage(MFT_MESSAGE_NOTIFY_START_OF_STREAM, 0)?;
    }
    let events = {
        let t = Mta(transform.clone());
        let c = credits.clone();
        let p = packets.clone();
        std::thread::spawn(move || event_loop(t, c, p))
    };

    if let Some(dir) = &verify_dir {
        fs::create_dir_all(dir)?;
    }
    let region: Vec<Crop> = Vec::new();
    // Per frame: pts, present, captured, nv12 ready, submitted.
    let mut frames: Vec<(u64, i64, i64, i64, i64)> = Vec::new();
    let mut pts = 0u64;
    let mut recent: VecDeque<usize> = VecDeque::new();
    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let mut failure = None;
    let mut converted_once = false;
    while Instant::now() < deadline {
        let (captured, present);
        if encode_only && converted_once {
            captured = qpc();
            present = 0;
        } else {
            if let Err(e) = dup.update_frame() {
                failure = Some(format!("capture: {e:?}"));
                break;
            }
            if !dup.updated {
                dup.release_frame();
                continue;
            }
            captured = qpc();
            present = dup.last_present_time;
            let frame = dup
                .texture()
                .cloned()
                .expect("updated frames carry a texture");
            unsafe { context.CopyResource(&shared_src, &frame) };
            dup.release_frame();
            converter.process(&shared_src, 1, 1.0, 0.0, 0, ToneMap::None, &region)?;
            converted_once = true;
        }
        let ready = qpc();
        if let Some(dir) = &verify_dir {
            if pts.is_multiple_of(VERIFY_EVERY) {
                fs::write(format!("{dir}/frame_{pts:06}.nv12"), converter.read_back()?)?;
            }
        }

        // Wait for the encoder to ask for input.
        {
            let (lock, cv) = &*credits;
            let mut c = lock.lock().unwrap();
            while c.need_input == 0 && c.error.is_none() {
                c = cv.wait(c).unwrap();
            }
            if let Some(e) = &c.error {
                failure = Some(e.clone());
                break;
            }
            c.need_input -= 1;
        }
        let slot = (pts as usize) % RING;
        recent.push_back(slot);
        unsafe { context.CopyResource(&ring[slot], &nv12) };
        let surface: IUnknown = ring[slot].cast()?;
        let media =
            unsafe { MFCreateDXGISurfaceBuffer(&ID3D11Texture2D::IID, &surface, 0, false)? };
        if let Ok(max) = unsafe { media.GetMaxLength() } {
            let _ = unsafe { media.SetCurrentLength(max) };
        }
        let sample = unsafe { MFCreateSample()? };
        unsafe {
            sample.AddBuffer(&media)?;
            sample.SetSampleTime((pts as i64) * 10_000_000 / fps as i64)?;
            sample.SetSampleDuration(10_000_000 / fps as i64)?;
            transform.ProcessInput(0, &sample, 0)?;
        }
        frames.push((pts, present, captured, ready, qpc()));
        pts += 1;
    }

    unsafe {
        let _ = transform.ProcessMessage(MFT_MESSAGE_NOTIFY_END_OF_STREAM, 0);
        let _ = transform.ProcessMessage(MFT_MESSAGE_COMMAND_DRAIN, 0);
    }
    {
        let (lock, cv) = &*credits;
        let c = lock.lock().unwrap();
        let _ = cv.wait_timeout_while(c, Duration::from_secs(5), |c| {
            !c.drained && c.error.is_none()
        });
    }
    let drained_ok = credits.0.lock().unwrap().drained;
    if drained_ok {
        let _ = events.join();
    }
    let error = credits.0.lock().unwrap().error.clone();

    let packets = std::mem::take(&mut *packets.lock().unwrap());
    let mut stream = fs::File::create(&out_path)?;
    for p in &packets {
        stream.write_all(&p.bytes)?;
    }
    let mut freq = 0i64;
    unsafe {
        let _ = QueryPerformanceFrequency(&mut freq);
    }
    let mut json = String::new();
    json.push_str(&format!(
        "{{\"gpu\":{:?},\"encoder\":{:?},\"width\":{width},\"height\":{height},\"fps\":{fps},\
         \"bitrate\":{bitrate},\"qpc_frequency\":{freq},\"ring\":{RING},\"drained\":{drained_ok},\
         \"error\":{},\"failure\":{},\"applied\":[",
        gpu,
        encoder_name,
        error
            .as_ref()
            .map_or("null".to_owned(), |e| format!("{e:?}")),
        failure
            .as_ref()
            .map_or("null".to_owned(), |e| format!("{e:?}")),
    ));
    json.push_str(
        &applied
            .iter()
            .map(|(k, v)| format!("[{k:?},{v:?}]"))
            .collect::<Vec<_>>()
            .join(","),
    );
    json.push_str("],\"frames\":[");
    json.push_str(
        &frames
            .iter()
            .map(|(p, a, b, c, d)| format!("[{p},{a},{b},{c},{d}]"))
            .collect::<Vec<_>>()
            .join(","),
    );
    json.push_str("],\"packets\":[");
    json.push_str(
        &packets
            .iter()
            .map(|p| format!("[{},{},{}]", p.sample_time, p.out_qpc, p.bytes.len()))
            .collect::<Vec<_>>()
            .join(","),
    );
    json.push_str("]}");
    fs::write(&times_path, json)?;
    println!(
        "{encoder_name} on {gpu}: {} frames, {} packets, {} bytes, drained {drained_ok}{}{}",
        frames.len(),
        packets.len(),
        packets.iter().map(|p| p.bytes.len()).sum::<usize>(),
        error.map_or(String::new(), |e| format!(", encoder error: {e}")),
        failure.map_or(String::new(), |e| format!(", stopped: {e}")),
    );
    for (k, v) in &applied {
        println!("  {k}: {v}");
    }
    unsafe {
        let _ = activate.ShutdownObject();
    }
    drop(recent);
    Ok(())
}

fn main() -> Res<()> {
    unsafe { CoInitializeEx(None, COINIT_MULTITHREADED).ok()? };
    unsafe { MFStartup(MF_VERSION, MFSTARTUP_FULL)? };
    let args: Vec<String> = std::env::args().collect();
    let result = if args.iter().any(|a| a == "--list") {
        list()
    } else {
        encode(&args)
    };
    unsafe {
        let _ = MFShutdown();
    }
    result
}
