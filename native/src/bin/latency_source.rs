//! Controlled desktop input. No Tk, GDI drawing, or exclusive display mode.
//! argv: width height fps workload present-log. EOF on stdin ends ownership.
use std::io::{BufRead, Write};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use windows::core::{w, PCSTR, BOOL};
use windows::Win32::Foundation::*;
use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D::Fxc::*;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::System::Performance::*;
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::Win32::UI::HiDpi::*;

const SHADER: &str = r#"
cbuffer Params : register(b0) { uint frame; uint mode; uint width; uint height; };
float4 VS(uint id: SV_VertexID): SV_Position {
    float2 p = float2((id << 1) & 2, id & 2);
    return float4(p * float2(2,-2) + float2(-1,1), 0, 1);
}
float4 PS(float4 p: SV_Position): SV_Target {
    uint x = (uint)p.x, y = (uint)p.y;
    if (y < 16 && x < 384) {
        uint k = x / 8;
        uint check = (frame ^ (frame >> 8) ^ (frame >> 16) ^ (frame >> 24) ^ 167) & 255;
        uint bit = k < 8 ? (167 >> k) & 1 : (k < 40 ? (frame >> (k-8)) & 1 : (check >> (k-40)) & 1);
        return float4(bit,bit,bit,1);
    }
    if (mode == 2) {
        return float4(((x+frame*7)%256)/255.0, ((y+frame*11)%256)/255.0, ((x/7+y/5+frame*13)%256)/255.0, 1);
    }
    uint row = (y + (mode == 1 ? frame*3 : 0)) % 200;
    bool text = row > 22 && row < 34 && x > 60 && (x % 640) < 510;
    bool panel = (x / 320) % 2 == 0;
    return text ? float4(.65,.70,.75,1) : (panel ? float4(.08,.10,.14,1) : float4(.12,.14,.18,1));
}
"#;

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if msg == WM_CLOSE || (msg == WM_KEYDOWN && wp.0 == 27) {
        PostQuitMessage(0);
        return LRESULT(0);
    }
    DefWindowProcW(hwnd, msg, wp, lp)
}

unsafe fn compile(entry: PCSTR, target: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code = None;
    let mut errors = None;
    let result = D3DCompile(SHADER.as_ptr().cast(), SHADER.len(), None, None, None,
                          entry, target, D3DCOMPILE_OPTIMIZATION_LEVEL3, 0,
                          &mut code, Some(&mut errors));
    if let Some(errors) = errors {
        eprintln!("{}", String::from_utf8_lossy(std::slice::from_raw_parts(
            errors.GetBufferPointer().cast(), errors.GetBufferSize())));
    }
    result?;
    Ok(code.unwrap())
}

unsafe fn qpc() -> windows::core::Result<i64> {
    let mut value = 0;
    QueryPerformanceCounter(&mut value)?;
    Ok(value)
}

unsafe fn run() -> Result<(), Box<dyn std::error::Error>> {
    // Not fatal. This returns E_ACCESSDENIED when the process already has a DPI
    // awareness context, which is a benign "nothing to do" -- and which of the
    // two happens depends on how the process was launched, not on anything this
    // program controls. Treating it as fatal made the source die instantly under
    // subprocess launch while working from a shell, with a bare "Access is
    // denied" that named neither the call nor the reason.
    //
    // Awareness matters here only so the window is not silently scaled, which
    // would corrupt the encoded frame IDs the benchmark decodes. If it was
    // already set, that requirement is already met; if it was set to something
    // unscaled-hostile, the readiness check downstream fails loudly on a size
    // mismatch rather than reporting wrong latencies.
    if let Err(error) = SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) {
        eprintln!("note: DPI awareness already set ({error}); continuing");
    }
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 6 { return Err("expected width height fps workload present-log".into()); }
    let width: u32 = args[1].parse()?;
    let height: u32 = args[2].parse()?;
    let fps: f64 = args[3].parse()?;
    let mode = match args[4].as_str() { "static" => 0, "scroll" => 1, "motion" => 2,
                                      _ => return Err("invalid workload".into()) };
    if width < 384 || height < 16 || !fps.is_finite() || !(1.0..=240.0).contains(&fps) {
        return Err("invalid dimensions or fps outside 1..240".into());
    }
    let mut frequency = 0;
    QueryPerformanceFrequency(&mut frequency)?;
    let mut log = std::fs::File::create(&args[5])?;
    let stopped = Arc::new(AtomicBool::new(false));
    let parent_stopped = stopped.clone();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = std::io::stdin().lock().read_line(&mut line);
        parent_stopped.store(true, Ordering::SeqCst);
    });
    let class = WNDCLASSW { lpfnWndProc: Some(wndproc), lpszClassName: w!("RapidshotLatency"),
                           ..Default::default() };
    eprintln!("step: RegisterClassW");
    if RegisterClassW(&class) == 0 { return Err(windows::core::Error::from_thread().into()); }
    eprintln!("step: CreateWindowExW");
    let hwnd = CreateWindowExW(WS_EX_TOPMOST, w!("RapidshotLatency"), w!("Rapidshot latency source - Esc stops"),
                              WS_POPUP, 0, 0, width as i32, height as i32, None, None, None, None)?;
    eprintln!("step: window created");
    // Window destruction also happens on every error after this point.
    struct Window(HWND);
    impl Drop for Window { fn drop(&mut self) { unsafe { let _ = DestroyWindow(self.0); } } }
    let _window = Window(hwnd);
    let desc = DXGI_SWAP_CHAIN_DESC {
        BufferDesc: DXGI_MODE_DESC { Width: width, Height: height,
                                    Format: DXGI_FORMAT_B8G8R8A8_UNORM, ..Default::default() },
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
        BufferCount: 2, OutputWindow: hwnd, Windowed: BOOL(1),
        SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD, ..Default::default()
    };
    let (mut swap, mut device, mut context) = (None, None, None);
    // Step markers: a bare HRESULT from this sequence names no call, and on a
    // hybrid laptop several of these can fail for reasons that look identical.
    eprintln!("step: D3D11CreateDeviceAndSwapChain");
    D3D11CreateDeviceAndSwapChain(None, D3D_DRIVER_TYPE_HARDWARE, HMODULE::default(),
        D3D11_CREATE_DEVICE_BGRA_SUPPORT, Some(&[D3D_FEATURE_LEVEL_11_0]), D3D11_SDK_VERSION,
        Some(&desc), Some(&mut swap), Some(&mut device), None, Some(&mut context))?;
    eprintln!("step: device created");
    let (swap, device, context) = (swap.unwrap(), device.unwrap(), context.unwrap());
    let back: ID3D11Texture2D = swap.GetBuffer(0)?;
    let mut rtv = None;
    device.CreateRenderTargetView(&back, None, Some(&mut rtv))?;
    let vs = compile(PCSTR(c"VS".as_ptr().cast()), PCSTR(c"vs_5_0".as_ptr().cast()))?;
    let ps = compile(PCSTR(c"PS".as_ptr().cast()), PCSTR(c"ps_5_0".as_ptr().cast()))?;
    let (mut vertex, mut pixel, mut buffer) = (None, None, None);
    device.CreateVertexShader(std::slice::from_raw_parts(vs.GetBufferPointer().cast(), vs.GetBufferSize()), None, Some(&mut vertex))?;
    device.CreatePixelShader(std::slice::from_raw_parts(ps.GetBufferPointer().cast(), ps.GetBufferSize()), None, Some(&mut pixel))?;
    device.CreateBuffer(&D3D11_BUFFER_DESC { ByteWidth: 16, Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32, ..Default::default() }, None, Some(&mut buffer))?;
    let buffer = buffer.unwrap();
    context.VSSetShader(vertex.as_ref(), None);
    context.PSSetShader(pixel.as_ref(), None);
    context.PSSetConstantBuffers(0, Some(&[Some(buffer.clone())]));
    context.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context.RSSetViewports(Some(&[D3D11_VIEWPORT { Width: width as f32, Height: height as f32,
                                                  MaxDepth: 1., ..Default::default() }]));
    let _ = ShowWindow(hwnd, SW_SHOW);
    let start = Instant::now();
    let mut report = start;
    let mut count = 0;
    let mut id: u32 = 0;
    loop {
        let mut message = MSG::default();
        while PeekMessageW(&mut message, None, 0, 0, PM_REMOVE).as_bool() {
            if message.message == WM_QUIT { stopped.store(true, Ordering::SeqCst); }
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if stopped.load(Ordering::SeqCst) || start.elapsed().as_secs() > 3600 { break; }
        id = id.checked_add(1).ok_or("frame ID exhausted")?;
        context.UpdateSubresource(&buffer, 0, None, [id, mode, width, height].as_ptr().cast(), 0, 0);
        context.OMSetRenderTargets(Some(&[rtv.clone()]), None);
        context.Draw(3, 0);
        let before = qpc()?;
        let status = swap.Present(1, DXGI_PRESENT(0));
        let after = qpc()?;
        if status != windows::core::HRESULT(0) { return Err(format!("Present not visible/successful: {status:?}").into()); }
        writeln!(log, "{{\"event\":\"present\",\"id\":{id},\"qpc_before\":{before},\"qpc_after\":{after}}}")?;
        if id == 1 {
            log.sync_data()?;
            println!("{{\"event\":\"ready\",\"qpc_frequency\":{frequency},\"width\":{width},\"height\":{height}}}");
        }
        count += 1;
        if report.elapsed().as_secs_f64() >= 2. {
            println!("{{\"event\":\"rate\",\"updates_per_second\":{}}}", count as f64 / report.elapsed().as_secs_f64());
            std::io::stdout().flush()?;
            log.sync_data()?;
            count = 0;
            report = Instant::now();
        }
        // Present(1) prevents an uncapped redraw loop. This adds requested pacing.
        let target = start + Duration::from_secs_f64(id as f64 / fps);
        if let Some(delay) = target.checked_duration_since(Instant::now()) { std::thread::sleep(delay); }
    }
    log.sync_data()?;
    Ok(())
}

fn main() {
    if let Err(error) = unsafe { run() } {
        eprintln!("latency source: {error}");
        std::process::exit(1);
    }
}
