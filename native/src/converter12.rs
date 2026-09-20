//! `GpuConverter` — the production transform path (ROADMAP § 7.2).
//!
//! This is deliberately **a second path rather than an edit** to
//! `preprocess12.rs`. Three reasons, in the order they matter:
//!
//! 1. Bilinear sampling changes the numbers. `tests/test_gpu_preprocess.py`
//!    pins the nearest-neighbour output exactly, and that contract is worth
//!    keeping: it is the only thing standing between a silent resampling
//!    change and a model whose accuracy moved for reasons nobody logged.
//! 2. A/B comparisons between the two must state which sampling they timed.
//!    Substituting bilinear underneath the old name would make every stored
//!    recording ambiguous after the fact — the failure ROADMAP § 3 spends a
//!    page on.
//! 3. The old path emits FP32 NCHW and nothing else. Rather than grow a
//!    format switch through a struct built around one layout, the variants
//!    are compile-time here and each one gets its own PSO.
//!
//! **What this adds over `Preprocessor12`:**
//!
//! | | `Preprocessor12` | `Converter12` |
//! | --- | --- | --- |
//! | Sampling | nearest (`Load`) | nearest **or** bilinear (`SampleLevel`) |
//! | Output | FP32 NCHW | FP32 NCHW, **FP16 NCHW**, **BGRA8 NHWC**, **NV12**, **P010** |
//! | Root signature | `NumStaticSamplers: 0` | one static linear sampler |
//!
//! The BGRA8 output is not a tensor at all — it is a *resized frame*, and it
//! exists because § 6.1's re-opened ordering question made payload size
//! load-bearing. At 2560×1600 → 640² it is 1.64 MB against the frame's
//! 16.38 MB, and § 6.1 records that converting before the bus wins at every
//! size once such a representation exists. Until this module there was no
//! way to produce one: `GpuPreprocessor12` emits FP32 NCHW only, which is why
//! the § 6.1 convert column is pessimistic on the cheap rows.

use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D11::ID3D11Texture2D;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM,
};
use windows::Win32::Graphics::Dxgi::IDXGIResource1;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

const GENERIC_ALL: u32 = 0x1000_0000;
/// Size of the shader's `Params` cbuffer, in 32-bit values.
const CONSTANT_COUNT: usize = 8;
/// `sizeof(Rect)` in the shader: four `uint`s.
const RECT_BYTES: u64 = 16;

/// A source rectangle in texel coordinates of the captured surface.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Crop {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Matches the private copy in `preprocess12`; see the note there.
const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;

/// Every format `DuplicateOutput1` can hand back — see `DUPLICATE_OUTPUT1_FORMATS`
/// in `core/duplicator.py`, which requests exactly these four.
///
/// **They need no shader change, and that is worth explaining rather than
/// assuming.** A `Texture2D<float4>` read returns channels in RGBA order for
/// all four: the hardware swizzles `B8G8R8A8` on read (which is why `.x` is
/// red despite blue being first in memory), `R8G8B8A8` and `R10G10B10A2` are
/// already in that order, and `R16G16B16A16_FLOAT` likewise. So `shade()`
/// sees red in `.x` regardless, and the only thing that has to follow the
/// source is the **SRV format** — declaring BGRA8 over a 10-bit surface
/// reinterprets the bits rather than converting them, which is a plausible
/// tensor built from the wrong numbers.
///
/// What does differ is **range**. The two UNORM 8-bit formats and the 10-bit
/// one all normalise to 0..1 on read. `R16G16B16A16_FLOAT` is scRGB and is
/// *not* bounded by 1.0 — an HDR highlight legitimately reads above it. That
/// is passed through for float outputs rather than clamped, because clamping
/// would discard the very thing the format exists to carry; the `uint8`
/// output has no choice and saturates.
const SUPPORTED_SOURCE_FORMATS: [DXGI_FORMAT; 4] = [
    DXGI_FORMAT_B8G8R8A8_UNORM,
    DXGI_FORMAT_R8G8B8A8_UNORM,
    DXGI_FORMAT_R10G10B10A2_UNORM,
    DXGI_FORMAT_R16G16B16A16_FLOAT,
];

/// Human-readable name for an error message, so a caller is not left mapping
/// a bare DXGI integer themselves.
fn format_name(format: DXGI_FORMAT) -> &'static str {
    match format {
        DXGI_FORMAT_B8G8R8A8_UNORM => "B8G8R8A8_UNORM",
        DXGI_FORMAT_R8G8B8A8_UNORM => "R8G8B8A8_UNORM",
        DXGI_FORMAT_R10G10B10A2_UNORM => "R10G10B10A2_UNORM (10-bit)",
        DXGI_FORMAT_R16G16B16A16_FLOAT => "R16G16B16A16_FLOAT (HDR scRGB)",
        _ => "unsupported",
    }
}

/// How the source is sampled when the output is smaller than the frame.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sampling {
    /// `Texture2D.Load` — what `Preprocessor12` does. Drops pixels rather than
    /// averaging them; at 2560×1600 → 640² that is fifteen of every sixteen.
    Nearest,
    /// `Texture2D.SampleLevel` through a linear static sampler. Filters the
    /// four nearest texels, which is what desktop content — small text, thin
    /// borders — needs to survive a 4× reduction.
    Bilinear,
}

/// What the converter writes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OutputFormat {
    /// float32, planar CHW. Byte-compatible with `Preprocessor12`'s output.
    Fp32Nchw,
    /// float16, planar CHW. Half the bytes, and what production inference
    /// mostly consumes — so the cast the caller used to pay disappears.
    Fp16Nchw,
    /// float32, interleaved HWC — what TensorFlow/Keras, TFLite and most
    /// channels-last exports take. Same values as `Fp32Nchw`, other order.
    Fp32Nhwc,
    /// float16, interleaved HWC.
    Fp16Nhwc,
    /// BGRA8, interleaved HWC. A *resized frame*, not a tensor: no scale, no
    /// bias, no channel reorder. The cheapest thing to put on a bus.
    Bgra8Nhwc,
    /// 8-bit 4:2:0: a full-resolution Y plane, then interleaved Cb/Cr at half
    /// resolution in both axes. The encoder input § 7.3 needs.
    Nv12,
    /// 10-bit 4:2:0 in the same layout as NV12, 16-bit little-endian samples
    /// with the value in the **high** ten bits (`code << 6`) — what DXGI, NVENC
    /// and FFmpeg's `p010le` all mean by P010.
    P010,
}

impl OutputFormat {
    /// Exact payload size. NV12 is 1.5 bytes per pixel, so this cannot be a
    /// per-pixel multiple.
    fn byte_size(self, width: u32, height: u32) -> u64 {
        let pixels = width as u64 * height as u64;
        match self {
            OutputFormat::Fp32Nchw | OutputFormat::Fp32Nhwc => pixels * 12,
            OutputFormat::Fp16Nchw | OutputFormat::Fp16Nhwc => pixels * 6,
            OutputFormat::Bgra8Nhwc => pixels * 4,
            OutputFormat::Nv12 => pixels * 3 / 2,
            OutputFormat::P010 => pixels * 3,
        }
    }

    fn define(self) -> u32 {
        match self {
            OutputFormat::Fp32Nchw => 0,
            OutputFormat::Fp16Nchw => 1,
            OutputFormat::Bgra8Nhwc => 2,
            OutputFormat::Nv12 => 3,
            OutputFormat::P010 => 4,
            OutputFormat::Fp32Nhwc => 5,
            OutputFormat::Fp16Nhwc => 6,
        }
    }

    pub fn is_yuv(self) -> bool {
        matches!(self, OutputFormat::Nv12 | OutputFormat::P010)
    }

    /// The FP16 kernels cover two output pixels per thread in x.
    pub fn is_fp16(self) -> bool {
        matches!(self, OutputFormat::Fp16Nchw | OutputFormat::Fp16Nhwc)
    }
}

/// Y'CbCr matrix for the YUV outputs. Ignored by every other output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Matrix {
    /// Kr 0.2126, Kb 0.0722. The HD default and what an encoder assumes when
    /// the stream does not say otherwise at these resolutions.
    Bt709,
    /// Kr 0.299, Kb 0.114. SD content, and some legacy decoders.
    Bt601,
}

/// How the YUV outputs are quantised. `Converter12` carries both so a caller
/// can state it rather than inherit it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct YuvOptions {
    pub matrix: Matrix,
    /// Limited ("TV", Y 16–235) when false, which is what encoders expect by
    /// default. Full ("PC", 0–255) when true.
    pub full_range: bool,
}

impl Default for YuvOptions {
    fn default() -> Self {
        Self {
            matrix: Matrix::Bt709,
            full_range: false,
        }
    }
}

/// One source, three output variants, two sampling modes, selected by
/// `#define` at compile time so no branch survives into the inner loop.
///
/// `RWByteAddressBuffer` rather than `RWStructuredBuffer<float>`: FP16 needs
/// to pack two halves into one 32-bit store, and BGRA8 four bytes, neither of
/// which a float-strided structured buffer can express.
const SHADER_BODY: &str = r#"
Texture2D<float4> Source : register(t0);
RWByteAddressBuffer Output : register(u0);
SamplerState LinearSampler : register(s0);

cbuffer Params : register(b0)
{
    uint OutWidth;
    uint OutHeight;
    uint SrcWidth;
    uint SrcHeight;
    float Scale;
    float Bias;
    uint  ChannelOrder;   // 0 = RGB, 1 = BGR
    uint  Plane;           // YUV outputs only: 0 = luma plane, 1 = chroma plane
};

// Source rectangles in texel coordinates, one per batch slot. A single crop is
// Regions[0]; no crop is the whole surface there -- which is what makes nearest
// output at full crop identical to the uncropped path. Multi-ROI dispatches
// once with z = region count, so tid.z is both the rectangle and the slot.
struct Rect { uint x; uint y; uint w; uint h; };
StructuredBuffer<Rect> Regions : register(t1);

// The rectangle this invocation samples. A static rather than a parameter
// threaded through fetch(): statics are per-invocation in a compute shader,
// and every kernel sets it before its first fetch.
static Rect g_crop;

// The hardware swizzles BGRA formats, so .x is RED here despite blue being
// first in memory. Getting this backwards produces BGR labelled RGB, which no
// test of speed or shape would catch.
float4 fetch(uint x, uint y)
{
#if SAMPLING == 1
    // Output pixel x covers crop range [x*C/O, (x+1)*C/O]; its centre is
    // crop.x + (x+0.5)*C/O texels.
    float2 origin = float2(g_crop.x, g_crop.y);
    float2 extent = float2(g_crop.w, g_crop.h);
    float2 pos = origin + (float2(x, y) + 0.5f) * extent / float2(OutWidth, OutHeight);
    // Clamp to the crop's outermost texel centres, so the filter never reaches
    // a texel outside the crop. Without this, upscaling a crop blends in the
    // row and column beyond its edge -- the sampler's own clamp only applies
    // at the edge of the whole surface. The result is exactly "crop, then
    // resize with clamp-to-edge", which is what a crop means.
    pos = clamp(pos, origin + 0.5f, origin + extent - 0.5f);
    return Source.SampleLevel(LinearSampler, pos / float2(SrcWidth, SrcHeight), 0.0f);
#else
    uint sx = g_crop.x + ((g_crop.w == OutWidth ) ? x : (x * g_crop.w / OutWidth ));
    uint sy = g_crop.y + ((g_crop.h == OutHeight) ? y : (y * g_crop.h / OutHeight));
    return Source.Load(int3(sx, sy, 0));
#endif
}

// Channel order applied, scale/bias applied. RGB when ChannelOrder == 0.
float3 shade(float4 texel)
{
    float c0 = (ChannelOrder == 0) ? texel.x : texel.z;
    float c1 = texel.y;
    float c2 = (ChannelOrder == 0) ? texel.z : texel.x;
    return float3(c0, c1, c2) * Scale + Bias;
}

#if OUTPUT == 1
// FP16 NCHW. Two horizontally adjacent output pixels per thread, because a
// ByteAddressBuffer stores 32 bits at a time and a half is 16. OutWidth is
// validated even on the Rust side, so a pair never straddles a row and every
// plane starts 4-byte aligned.
[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint x0 = tid.x * 2;
    if (x0 >= OutWidth || tid.y >= OutHeight)
        return;
    g_crop = Regions[tid.z];

    float3 a = shade(fetch(x0,     tid.y));
    float3 b = shade(fetch(x0 + 1, tid.y));

    uint plane = OutWidth * OutHeight;
    uint idx   = tid.y * OutWidth + x0;

    [unroll] for (uint c = 0; c < 3; ++c)
    {
        uint packed = (f32tof16(b[c]) << 16) | f32tof16(a[c]);
        Output.Store(((tid.z * 3 + c) * plane + idx) * 2, packed);
    }
}

#elif OUTPUT == 5
// FP32 NHWC -- the NCHW values, interleaved. Three stores per pixel, each a
// whole float, so nothing straddles a dword.
[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth || tid.y >= OutHeight)
        return;
    g_crop = Regions[tid.z];

    float3 v = shade(fetch(tid.x, tid.y));

    uint base = ((tid.z * OutHeight + tid.y) * OutWidth + tid.x) * 3;
    [unroll] for (uint c = 0; c < 3; ++c)
        Output.Store((base + c) * 4, asuint(v[c]));
}

#elif OUTPUT == 6
// FP16 NHWC. A pixel is three halves -- 6 bytes, not a dword multiple -- so,
// as in FP16 NCHW, a thread owns two adjacent pixels: 12 bytes, exactly three
// stores laid out [a0 a1] [a2 b0] [b1 b2]. The even width the Rust side
// already requires for FP16 means a pair never straddles a row.
[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    uint x0 = tid.x * 2;
    if (x0 >= OutWidth || tid.y >= OutHeight)
        return;
    g_crop = Regions[tid.z];

    float3 a = shade(fetch(x0,     tid.y));
    float3 b = shade(fetch(x0 + 1, tid.y));

    uint base = ((tid.z * OutHeight + tid.y) * OutWidth + x0) * 6;
    Output.Store(base,     f32tof16(a.x) | (f32tof16(a.y) << 16));
    Output.Store(base + 4, f32tof16(a.z) | (f32tof16(b.x) << 16));
    Output.Store(base + 8, f32tof16(b.y) | (f32tof16(b.z) << 16));
}

#elif OUTPUT == 2
// BGRA8 NHWC -- a resized frame. Deliberately ignores Scale, Bias and
// ChannelOrder: this output is meant to be indistinguishable from what grab()
// hands back, only smaller, so a consumer can treat it as an ordinary frame.
[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth || tid.y >= OutHeight)
        return;
    g_crop = Regions[tid.z];

    float4 t = fetch(tid.x, tid.y);

    uint r = (uint)(saturate(t.x) * 255.0f + 0.5f);
    uint g = (uint)(saturate(t.y) * 255.0f + 0.5f);
    uint bl = (uint)(saturate(t.z) * 255.0f + 0.5f);
    uint a = (uint)(saturate(t.w) * 255.0f + 0.5f);

    uint idx = (tid.z * OutHeight + tid.y) * OutWidth + tid.x;
    Output.Store(idx * 4, bl | (g << 8) | (r << 16) | (a << 24));
}

#elif OUTPUT == 3 || OUTPUT == 4
// NV12 (8-bit) / P010 (10-bit), 4:2:0, in the standard planar layout: the Y
// plane, then Cb/Cr interleaved at half resolution. Two dispatches, selected by
// Plane, because the planes have different sample grids.
//
// A ByteAddressBuffer stores 32 bits at a time and an NV12 sample is 8, so a
// thread owns one *dword of the plane*, not one pixel. The dword index is laid
// out over a 2D dispatch as d = tid.y * OutWidth + tid.x, and each byte inside
// it maps back to its own (x, y). That is what lets a plane whose width is not
// a multiple of four be written without a dword ever straddling two threads.
//
// The source is taken as gamma-encoded RGB with BT.709 primaries -- true of a
// UNORM desktop surface. Linear scRGB (R16G16B16A16_FLOAT) is refused on the
// Rust side: putting linear light through a Y'CbCr matrix is wrong, and the
// right fix (tone-map to SDR, or PQ + BT.2020 for HDR10) is a decision.

#if MATRIX == 1
static const float KR = 0.299f;
static const float KB = 0.114f;
#else
static const float KR = 0.2126f;
static const float KB = 0.0722f;
#endif

#if OUTPUT == 3
static const float MAXCODE = 255.0f;
static const float DEPTH_SCALE = 1.0f;     // 2^(8-8)
#else
static const float MAXCODE = 1023.0f;
static const float DEPTH_SCALE = 4.0f;     // 2^(10-8)
#endif

uint quantize(float v)
{
    return (uint)clamp(floor(v + 0.5f), 0.0f, MAXCODE);
}

float3 rgb_at(uint x, uint y)
{
    return saturate(fetch(x, y).xyz);
}

float luma_of(float3 c)
{
    return KR * c.x + (1.0f - KR - KB) * c.y + KB * c.z;
}

uint luma_code(float3 c)
{
#if FULL_RANGE == 1
    return quantize(luma_of(c) * MAXCODE);
#else
    return quantize((16.0f + 219.0f * luma_of(c)) * DEPTH_SCALE);
#endif
}

// Chroma for 4:2:0 sample k: the 2x2 block of output pixels it covers,
// averaged, then through the matrix. Centre-sited (JPEG / MPEG-1 style).
// Returns (Cb, Cr) codes.
uint2 chroma_codes(uint k)
{
    uint cw = OutWidth / 2;
    uint cx = (k % cw) * 2;
    uint cy = (k / cw) * 2;
    float3 c = 0.25f * (rgb_at(cx, cy) + rgb_at(cx + 1, cy)
                      + rgb_at(cx, cy + 1) + rgb_at(cx + 1, cy + 1));
    float y = luma_of(c);
    float pb = (c.z - y) / (2.0f * (1.0f - KB));
    float pr = (c.x - y) / (2.0f * (1.0f - KR));
#if FULL_RANGE == 1
    float mid = (MAXCODE + 1.0f) * 0.5f;
    return uint2(quantize(mid + pb * MAXCODE), quantize(mid + pr * MAXCODE));
#else
    return uint2(quantize((128.0f + 224.0f * pb) * DEPTH_SCALE),
                 quantize((128.0f + 224.0f * pr) * DEPTH_SCALE));
#endif
}

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth)
        return;
    // One frame, never a batch: the Rust side refuses batch > 1 for YUV.
    g_crop = Regions[0];
    uint d = tid.y * OutWidth + tid.x;

    uint luma_samples = OutWidth * OutHeight;
    uint chroma_samples = (OutWidth / 2) * (OutHeight / 2);

#if OUTPUT == 3
    if (Plane == 0)
    {
        if (d >= luma_samples / 4)     // even x even, so an exact multiple
            return;
        uint packed = 0;
        [unroll] for (uint i = 0; i < 4; ++i)
        {
            uint s = d * 4 + i;
            packed |= luma_code(rgb_at(s % OutWidth, s / OutWidth)) << (8 * i);
        }
        Output.Store(d * 4, packed);
    }
    else
    {
        // 2 bytes per chroma sample, so the plane need not fill its last dword;
        // the tail is padding inside the allocation, never part of the payload.
        uint k0 = d * 2;
        if (k0 >= chroma_samples)
            return;
        uint2 a = chroma_codes(k0);
        uint packed = a.x | (a.y << 8);
        if (k0 + 1 < chroma_samples)
        {
            uint2 b = chroma_codes(k0 + 1);
            packed |= (b.x << 16) | (b.y << 24);
        }
        Output.Store(luma_samples + d * 4, packed);
    }
#else
    if (Plane == 0)
    {
        if (d >= luma_samples / 2)
            return;
        uint s0 = d * 2;
        uint y0 = luma_code(rgb_at(s0 % OutWidth, s0 / OutWidth));
        uint y1 = luma_code(rgb_at((s0 + 1) % OutWidth, (s0 + 1) / OutWidth));
        Output.Store(d * 4, (y0 << 6) | ((y1 << 6) << 16));
    }
    else
    {
        if (d >= chroma_samples)
            return;
        uint2 c = chroma_codes(d);
        Output.Store(luma_samples * 2 + d * 4, (c.x << 6) | ((c.y << 6) << 16));
    }
#endif
}

#else
// FP32 NCHW -- byte-identical in layout to Preprocessor12's output, so the
// only difference a caller can observe is the sampling mode.
[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth || tid.y >= OutHeight)
        return;
    g_crop = Regions[tid.z];

    float3 v = shade(fetch(tid.x, tid.y));

    uint plane = OutWidth * OutHeight;
    uint idx   = tid.y * OutWidth + tid.x;

    [unroll] for (uint c = 0; c < 3; ++c)
        Output.Store(((tid.z * 3 + c) * plane + idx) * 4, asuint(v[c]));
}
#endif
"#;

fn compile_variant(
    sampling: Sampling,
    output: OutputFormat,
    yuv: YuvOptions,
) -> windows::core::Result<ID3DBlob> {
    // Prepending `#define`s rather than threading `D3D_SHADER_MACRO` through
    // FFI: same result, no CString lifetimes to get wrong.
    let source = format!(
        "#define SAMPLING {}\n#define OUTPUT {}\n#define MATRIX {}\n#define FULL_RANGE {}\n{}",
        if sampling == Sampling::Bilinear { 1 } else { 0 },
        output.define(),
        if yuv.matrix == Matrix::Bt601 { 1 } else { 0 },
        if yuv.full_range { 1 } else { 0 },
        SHADER_BODY
    );

    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr() as *const _,
            source.len(),
            None,
            None,
            None,
            PCSTR(c"CSMain".as_ptr().cast()),
            PCSTR(c"cs_5_1".as_ptr().cast()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(error) = &result {
        if let Some(errors) = errors {
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer() as *const u8,
                    errors.GetBufferSize(),
                )
            };
            return Err(windows::core::Error::new(
                error.code(),
                format!(
                    "converter shader compilation failed ({sampling:?}, {output:?}): {}",
                    String::from_utf8_lossy(text)
                ),
            ));
        }
        result?;
    }
    code.ok_or_else(|| {
        windows::core::Error::new(
            windows::Win32::Foundation::E_FAIL,
            "converter shader compiled but produced no bytecode",
        )
    })
}

pub struct Converter12 {
    device: ID3D12Device,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    fence_value: std::cell::Cell<u64>,
    fence_event: HANDLE,
    root_signature: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    output: ID3D12Resource,
    readback: ID3D12Resource,
    /// Upload-heap `StructuredBuffer<Rect>`, `batch` entries. Written from the
    /// CPU before each dispatch; safe because `process` waits for the GPU
    /// before returning, so no dispatch is ever reading it during a write.
    regions: ID3D12Resource,
    /// Orders each dispatch behind the capture device's own work; see
    /// `capture_order.rs` for the race it closes.
    capture_order: super::capture_order::CaptureOrder,
    shared_output_handle: HANDLE,
    cached_texture: std::cell::Cell<(usize, u64)>,
    cached_shared: std::cell::RefCell<Option<ID3D12Resource>>,
    cached_src_size: std::cell::Cell<(u32, u32)>,
    /// Format of the surface currently open, for reporting. Zero until a
    /// frame has been processed.
    cached_src_format: std::cell::Cell<DXGI_FORMAT>,
    pub out_width: u32,
    pub out_height: u32,
    pub sampling: Sampling,
    pub format: OutputFormat,
    pub yuv: YuvOptions,
    /// Most regions one `process` call can convert; the output buffer holds
    /// this many slots.
    pub batch: u32,
}

impl Converter12 {
    pub fn new(
        d3d11_texture: &ID3D11Texture2D,
        out_width: u32,
        out_height: u32,
        sampling: Sampling,
        format: OutputFormat,
        yuv: YuvOptions,
        batch: u32,
    ) -> windows::core::Result<Self> {
        if out_width == 0 || out_height == 0 {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                "output size must be non-zero",
            ));
        }
        if batch == 0 {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                "batch must be at least 1",
            ));
        }
        if format.is_yuv() && batch != 1 {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                "NV12/P010 output is one frame for an encoder, not a batch; \
                 use batch=1",
            ));
        }
        // The shader addresses the output with 32-bit byte offsets. Past 4 GB
        // they wrap, and the wrapped writes land inside the buffer -- on top of
        // earlier slots, not out of bounds -- so nothing would ever fault.
        let total = format.byte_size(out_width, out_height) as u128 * batch as u128;
        if total.next_multiple_of(4) > u32::MAX as u128 {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                format!(
                    "{batch} x {out_width}x{out_height} is {total} bytes of output; \
                     the shader's 32-bit offsets cap it below 4 GB"
                ),
            ));
        }
        // 4:2:0 has one chroma sample per 2x2 block. An odd dimension leaves a
        // row or column with no block to belong to; every encoder refuses it,
        // so refuse here rather than invent a convention for the edge.
        if format.is_yuv() && (!out_width.is_multiple_of(2) || !out_height.is_multiple_of(2)) {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                format!(
                    "NV12/P010 output requires even width and height (got \
                     {out_width}x{out_height}); 4:2:0 subsamples in 2x2 blocks"
                ),
            ));
        }
        // The FP16 kernel packs two adjacent pixels into one 32-bit store, so
        // an odd width would leave a half-written dword at the end of every
        // row. Refuse rather than emit a tensor whose last column is garbage:
        // the shape would still be right, which is exactly the kind of wrong
        // § 11 says to fail loudly on. Every realistic model input is even.
        if format.is_fp16() && !out_width.is_multiple_of(2) {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                format!(
                    "float16 output requires an even width (got {out_width}); \
                     the kernel packs two pixels per 32-bit store"
                ),
            ));
        }

        // Fail at setup rather than on the first frame, as Preprocessor12 does.
        {
            let resource: IDXGIResource1 = d3d11_texture.cast().map_err(|e| {
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
                            "texture is not shareable, so the converter cannot reach it; \
                             it must be created with D3D11_RESOURCE_MISC_SHARED_NTHANDLE",
                        )
                    })?;
            unsafe {
                let _ = CloseHandle(handle);
            }
        }

        let device = super::preprocess12::device_for_texture(d3d11_texture)?;

        let queue: ID3D12CommandQueue = unsafe {
            device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                Priority: 0,
                Flags: D3D12_COMMAND_QUEUE_FLAG_NONE,
                NodeMask: 0,
            })?
        };
        let allocator: ID3D12CommandAllocator =
            unsafe { device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)? };
        let list: ID3D12GraphicsCommandList = unsafe {
            device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)?
        };
        unsafe { list.Close()? };
        let fence: ID3D12Fence = unsafe { device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let fence_event = unsafe { CreateEventW(None, false, false, None)? };
        let capture_order = super::capture_order::CaptureOrder::new(&device)?;

        let ranges = [
            // t0 = the captured surface, t1 = the region rectangles.
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 2,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 2,
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: CONSTANT_COUNT as u32,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                        NumDescriptorRanges: ranges.len() as u32,
                        pDescriptorRanges: ranges.as_ptr(),
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            },
        ];

        // The one structural difference from Preprocessor12's root signature.
        // Declared for both sampling modes so there is a single signature to
        // reason about; an unused static sampler costs nothing.
        let samplers = [D3D12_STATIC_SAMPLER_DESC {
            Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
            AddressU: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressV: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            AddressW: D3D12_TEXTURE_ADDRESS_MODE_CLAMP,
            MipLODBias: 0.0,
            MaxAnisotropy: 0,
            ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
            BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
            MinLOD: 0.0,
            MaxLOD: D3D12_FLOAT32_MAX,
            ShaderRegister: 0,
            RegisterSpace: 0,
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];

        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: samplers.len() as u32,
            pStaticSamplers: samplers.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
        };

        let root_signature: ID3D12RootSignature = unsafe {
            let mut blob: Option<ID3DBlob> = None;
            let mut error: Option<ID3DBlob> = None;
            D3D12SerializeRootSignature(
                &root_desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut error),
            )?;
            let blob = blob.expect("root signature serialised");
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer() as *const u8,
                blob.GetBufferSize(),
            );
            device.CreateRootSignature(0, bytes)?
        };

        let bytecode = compile_variant(sampling, format, yuv)?;
        let pso: ID3D12PipelineState = unsafe {
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(root_signature.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytecode.GetBufferPointer(),
                    BytecodeLength: bytecode.GetBufferSize(),
                },
                NodeMask: 0,
                CachedPSO: D3D12_CACHED_PIPELINE_STATE::default(),
                Flags: D3D12_PIPELINE_STATE_FLAG_NONE,
            };
            device.CreateComputePipelineState(&desc)?
        };

        let heap: ID3D12DescriptorHeap = unsafe {
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 3,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
        };
        let descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        // The shader writes whole dwords, so the allocation is rounded up to
        // one. Only NV12 can need it (1.5 bytes/pixel); the padding is never
        // part of the payload — `output_byte_size` stays exact, and readback
        // and cross-adapter transfer copy only that.
        let byte_size =
            (format.byte_size(out_width, out_height) * batch as u64).next_multiple_of(4);

        let output = super::preprocess12::create_buffer(
            &device,
            byte_size,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_HEAP_FLAG_SHARED,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        let readback = super::preprocess12::create_buffer(
            &device,
            byte_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_HEAP_FLAG_NONE,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let shared_output_handle =
            unsafe { device.CreateSharedHandle(&output, None, GENERIC_ALL, None)? };

        // RAW rather than structured: the shader stores 32 bits at a time at
        // byte addresses it computes itself, which is what lets one kernel
        // emit f32, packed f16 or packed bytes.
        let heap_start = unsafe { heap.GetCPUDescriptorHandleForHeapStart() }.ptr;

        let regions = super::preprocess12::create_buffer(
            &device,
            batch as u64 * RECT_BYTES,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_HEAP_FLAG_NONE,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut regions_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN,
            ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            ..Default::default()
        };
        regions_desc.Anonymous.Buffer = D3D12_BUFFER_SRV {
            FirstElement: 0,
            NumElements: batch,
            StructureByteStride: RECT_BYTES as u32,
            Flags: D3D12_BUFFER_SRV_FLAG_NONE,
        };
        unsafe {
            device.CreateShaderResourceView(
                &regions,
                Some(&regions_desc),
                D3D12_CPU_DESCRIPTOR_HANDLE {
                    ptr: heap_start + descriptor_size as usize,
                },
            );
        }

        let uav_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
            ptr: heap_start + 2 * descriptor_size as usize,
        };
        let mut uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
            Format: windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R32_TYPELESS,
            ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
            ..Default::default()
        };
        uav_desc.Anonymous.Buffer = D3D12_BUFFER_UAV {
            FirstElement: 0,
            NumElements: (byte_size / 4) as u32,
            StructureByteStride: 0,
            CounterOffsetInBytes: 0,
            Flags: D3D12_BUFFER_UAV_FLAG_RAW,
        };
        unsafe {
            device.CreateUnorderedAccessView(&output, None, Some(&uav_desc), uav_handle);
        }

        Ok(Self {
            device,
            queue,
            allocator,
            list,
            fence,
            fence_value: std::cell::Cell::new(0),
            fence_event,
            root_signature,
            pso,
            heap,
            output,
            readback,
            regions,
            capture_order,
            shared_output_handle,
            cached_texture: std::cell::Cell::new((0, 0)),
            cached_shared: std::cell::RefCell::new(None),
            cached_src_size: std::cell::Cell::new((0, 0)),
            cached_src_format: std::cell::Cell::new(DXGI_FORMAT(0)),
            out_width,
            out_height,
            sampling,
            format,
            yuv,
            batch,
        })
    }

    fn wait_for_gpu(&self) -> windows::core::Result<()> {
        let value = self.fence_value.get() + 1;
        self.fence_value.set(value);
        unsafe {
            self.queue.Signal(&self.fence, value)?;
            if self.fence.GetCompletedValue() < value {
                self.fence.SetEventOnCompletion(value, self.fence_event)?;
                WaitForSingleObject(self.fence_event, INFINITE);
            }
        }
        Ok(())
    }

    /// Convert one captured frame. Same caching rules as `Preprocessor12`:
    /// the shared handle is opened per *texture*, not per frame, and the key
    /// pairs the pointer with the duplicator that produced it so a recycled
    /// COM address cannot silently reuse a stale resource.
    ///
    /// `regions` are in texel coordinates of the surface, one per output slot,
    /// all converted by **one** dispatch; empty means one slot holding the
    /// whole surface. A rectangle reaching outside the surface is refused
    /// rather than clamped — a clamped crop is a plausible image of the wrong
    /// region.
    pub fn process(
        &self,
        d3d11_texture: &ID3D11Texture2D,
        source_id: u64,
        scale: f32,
        bias: f32,
        channel_order: u32,
        regions: &[Crop],
    ) -> windows::core::Result<()> {
        if regions.len() > self.batch as usize {
            return Err(windows::core::Error::new(
                windows::Win32::Foundation::E_INVALIDARG,
                format!(
                    "{} regions given, but this converter was built for at most {}",
                    regions.len(),
                    self.batch
                ),
            ));
        }
        let key = (d3d11_texture.as_raw() as usize, source_id);
        if self.cached_texture.get() != key || self.cached_shared.borrow().is_none() {
            self.open_texture(d3d11_texture, key)?;
        }
        let borrowed = self.cached_shared.borrow();
        let shared = borrowed
            .as_ref()
            .expect("open_texture populates the cache or returns Err");
        let (src_width, src_height) = self.cached_src_size.get();

        let whole = [Crop {
            x: 0,
            y: 0,
            width: src_width,
            height: src_height,
        }];
        let regions = if regions.is_empty() {
            &whole[..]
        } else {
            regions
        };
        for (index, crop) in regions.iter().enumerate() {
            let fits = crop.width > 0
                && crop.height > 0
                && crop.x as u64 + crop.width as u64 <= src_width as u64
                && crop.y as u64 + crop.height as u64 <= src_height as u64;
            if !fits {
                return Err(windows::core::Error::new(
                    windows::Win32::Foundation::E_INVALIDARG,
                    format!(
                        "region {index}: crop {}x{} at ({}, {}) does not fit inside \
                         the {}x{} surface",
                        crop.width, crop.height, crop.x, crop.y, src_width, src_height
                    ),
                ));
            }
        }
        unsafe {
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            // An empty read range: the CPU never reads this buffer back.
            self.regions.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut mapped),
            )?;
            let rects = mapped as *mut u32;
            for (index, crop) in regions.iter().enumerate() {
                let slot = rects.add(index * 4);
                *slot = crop.x;
                *slot.add(1) = crop.y;
                *slot.add(2) = crop.width;
                *slot.add(3) = crop.height;
            }
            self.regions.Unmap(0, None);
        }
        let count = regions.len() as u32;

        self.capture_order.order(&self.queue, d3d11_texture)?;

        let mut constants: [u32; CONSTANT_COUNT] = [
            self.out_width,
            self.out_height,
            src_width,
            src_height,
            scale.to_bits(),
            bias.to_bits(),
            channel_order,
            0,
        ];

        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, &self.pso)?;

            transition(
                &self.list,
                shared,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            );

            self.list.SetComputeRootSignature(&self.root_signature);
            self.list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            self.list.SetComputeRoot32BitConstants(
                0,
                CONSTANT_COUNT as u32,
                constants.as_ptr() as *const _,
                0,
            );
            self.list
                .SetComputeRootDescriptorTable(1, self.heap.GetGPUDescriptorHandleForHeapStart());

            if self.format.is_yuv() {
                // One thread per dword of each plane, laid out OutWidth dwords
                // to a dispatch row; the shader maps each byte back to (x, y).
                let pixels = self.out_width as u64 * self.out_height as u64;
                let (luma_dwords, chroma_dwords) = match self.format {
                    OutputFormat::Nv12 => (pixels / 4, (pixels / 2).div_ceil(4)),
                    _ => (pixels / 2, pixels / 4),
                };
                for (pass, dwords) in [(0u32, luma_dwords), (1u32, chroma_dwords)] {
                    constants[7] = pass;
                    self.list.SetComputeRoot32BitConstants(
                        0,
                        CONSTANT_COUNT as u32,
                        constants.as_ptr() as *const _,
                        0,
                    );
                    let rows = dwords.div_ceil(self.out_width as u64) as u32;
                    self.list
                        .Dispatch(self.out_width.div_ceil(8), rows.div_ceil(8), 1);
                }
            } else {
                // FP16 covers two output pixels per thread in x. z is the
                // region: numthreads has z = 1, so exactly `count` slices run
                // and tid.z never exceeds the rectangles written above.
                let threads_x = if self.format.is_fp16() {
                    self.out_width.div_ceil(2)
                } else {
                    self.out_width
                };
                self.list
                    .Dispatch(threads_x.div_ceil(8), self.out_height.div_ceil(8), count);
            }

            transition(
                &self.list,
                shared,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            );

            self.list.Close()?;
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
        }
        self.wait_for_gpu()
    }

    fn open_texture(
        &self,
        d3d11_texture: &ID3D11Texture2D,
        key: (usize, u64),
    ) -> windows::core::Result<()> {
        let resource: IDXGIResource1 = d3d11_texture.cast()?;
        let handle = unsafe { resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None)? };

        let result = (|| -> windows::core::Result<()> {
            let mut shared: Option<ID3D12Resource> = None;
            unsafe { self.device.OpenSharedHandle(handle, &mut shared)? };
            let shared = shared.expect("OpenSharedHandle reported success");

            let desc = unsafe { shared.GetDesc() };
            if !SUPPORTED_SOURCE_FORMATS.contains(&desc.Format) {
                return Err(windows::core::Error::new(
                    windows::Win32::Foundation::E_INVALIDARG,
                    format!(
                        "captured surface is DXGI format {}, which this converter \
                         does not handle. Supported: B8G8R8A8_UNORM (87), \
                         R8G8B8A8_UNORM (28), R10G10B10A2_UNORM (24), \
                         R16G16B16A16_FLOAT (10) -- the four DuplicateOutput1 \
                         requests. Use grab() and convert on the CPU.",
                        desc.Format.0
                    ),
                ));
            }
            if self.format.is_yuv() && desc.Format == DXGI_FORMAT_R16G16B16A16_FLOAT {
                return Err(windows::core::Error::new(
                    windows::Win32::Foundation::E_INVALIDARG,
                    "captured surface is R16G16B16A16_FLOAT (HDR, linear scRGB), \
                     which NV12/P010 output does not convert. A Y'CbCr matrix \
                     expects gamma-encoded values, and choosing between tone \
                     mapping to SDR and PQ + BT.2020 for HDR10 is a decision this \
                     path has not made. Use a float32/float16 output, which passes \
                     scRGB through.",
                ));
            }
            self.cached_src_format.set(desc.Format);

            self.cached_src_size.set((desc.Width as u32, desc.Height));

            let srv_handle = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
            let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                // Follows the surface. Hard-coding BGRA8 here would
                // reinterpret a 10-bit or HDR surface rather than convert it.
                Format: desc.Format,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                ..Default::default()
            };
            srv_desc.Anonymous.Texture2D = D3D12_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
                PlaneSlice: 0,
                ResourceMinLODClamp: 0.0,
            };
            unsafe {
                self.device
                    .CreateShaderResourceView(&shared, Some(&srv_desc), srv_handle)
            };

            *self.cached_shared.borrow_mut() = Some(shared);
            self.cached_texture.set(key);
            Ok(())
        })();

        unsafe {
            let _ = CloseHandle(handle);
        }

        if result.is_err() {
            *self.cached_shared.borrow_mut() = None;
            self.cached_texture.set((0, 0));
        }
        result
    }

    /// Raw output bytes. Verification only — the production path keeps the
    /// tensor on the GPU and hands over `shared_output_handle`.
    ///
    /// Returns bytes rather than a typed vector deliberately: ROADMAP § 10
    /// records that returning `Vec<f32>` here made PyO3 build 1.2M Python
    /// floats per call and turned this into an allocator benchmark.
    pub fn read_back(&self) -> windows::core::Result<Vec<u8>> {
        let count = self.output_byte_size() as usize;
        unsafe {
            self.allocator.Reset()?;
            self.list.Reset(&self.allocator, None)?;
            transition(
                &self.list,
                &self.output,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            );
            self.list.CopyResource(&self.readback, &self.output);
            transition(
                &self.list,
                &self.output,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            self.list.Close()?;
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
        }
        self.wait_for_gpu()?;

        let mut out = vec![0u8; count];
        unsafe {
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(mapped as *const u8, out.as_mut_ptr(), count);
            self.readback.Unmap(0, None);
        }
        Ok(out)
    }

    /// The whole buffer: every batch slot, exact (no dword padding). What a
    /// cross-adapter transfer or a CUDA import must cover.
    pub fn output_byte_size(&self) -> u64 {
        self.format.byte_size(self.out_width, self.out_height) * self.batch as u64
    }

    pub fn shared_output_handle(&self) -> isize {
        self.shared_output_handle.0 as isize
    }

    pub fn output_resource_address(&self) -> usize {
        self.output.as_raw() as usize
    }

    pub fn output_gpu_address(&self) -> u64 {
        unsafe { self.output.GetGPUVirtualAddress() }
    }

    /// Name of the source format currently open, or "none" before the first
    /// frame. Diagnostic: an HDR desktop and an SDR one differ in nothing a
    /// caller can otherwise see from the output.
    pub fn source_format(&self) -> &'static str {
        let format = self.cached_src_format.get();
        if format.0 == 0 {
            "none"
        } else {
            format_name(format)
        }
    }

    pub fn adapter_luid(&self) -> [u8; 8] {
        super::preprocess12::adapter_luid_for(&self.device)
    }

    /// The D3D12 device the output lives on. Needed by `TensorTransfer`, which
    /// must build its cross-adapter heap on *this* device rather than pick one.
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// The output buffer itself, as the source of a cross-adapter copy.
    pub(crate) fn output(&self) -> &ID3D12Resource {
        &self.output
    }
}

impl Drop for Converter12 {
    fn drop(&mut self) {
        let _ = self.wait_for_gpu();
        *self.cached_shared.borrow_mut() = None;
        if !self.shared_output_handle.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.shared_output_handle);
            }
        }
        if !self.fence_event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.fence_event);
            }
        }
    }
}

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
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: std::mem::ManuallyDrop::new(Some(resource.clone())),
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    };
    unsafe { list.ResourceBarrier(&[barrier]) };
}
#[cfg(test)]
mod tests {
    use super::*;

    /// The declarations inside a shader's `cbuffer Params { ... };`.
    fn cbuffer_fields(shader: &str) -> Vec<&str> {
        let start = shader.find("cbuffer Params").expect("cbuffer Params");
        let open = start + shader[start..].find('{').expect("opening brace") + 1;
        let close = open + shader[open..].find("};").expect("closing brace");
        shader[open..close]
            .lines()
            .map(|line| line.split("//").next().unwrap_or("").trim())
            .filter(|line| line.ends_with(';'))
            .collect()
    }

    const ALL_FORMATS: [OutputFormat; 7] = [
        OutputFormat::Fp32Nchw,
        OutputFormat::Fp16Nchw,
        OutputFormat::Fp32Nhwc,
        OutputFormat::Fp16Nhwc,
        OutputFormat::Bgra8Nhwc,
        OutputFormat::Nv12,
        OutputFormat::P010,
    ];

    #[test]
    fn byte_sizes_match_the_figures_the_docs_quote() {
        // TensorTransfer's docstring and ROADMAP 6.1 quote these for 640x640.
        assert_eq!(OutputFormat::Fp32Nchw.byte_size(640, 640), 4_915_200);
        assert_eq!(OutputFormat::Fp16Nchw.byte_size(640, 640), 2_457_600);
        assert_eq!(OutputFormat::Bgra8Nhwc.byte_size(640, 640), 1_638_400);
    }

    #[test]
    fn layout_does_not_change_the_byte_count() {
        assert_eq!(
            OutputFormat::Fp32Nchw.byte_size(64, 32),
            OutputFormat::Fp32Nhwc.byte_size(64, 32)
        );
        assert_eq!(
            OutputFormat::Fp16Nchw.byte_size(64, 32),
            OutputFormat::Fp16Nhwc.byte_size(64, 32)
        );
    }

    #[test]
    fn yuv_sizes_are_a_luma_plane_plus_half_resolution_chroma() {
        // NV12: w*h luma bytes, then (w/2)*(h/2) Cb/Cr pairs of one byte each.
        let (w, h) = (1920u64, 1080u64);
        assert_eq!(
            OutputFormat::Nv12.byte_size(1920, 1080),
            w * h + (w / 2) * (h / 2) * 2
        );
        // P010: the same samples, two bytes each.
        assert_eq!(
            OutputFormat::P010.byte_size(1920, 1080),
            2 * (w * h + (w / 2) * (h / 2) * 2)
        );
    }

    #[test]
    fn sizes_do_not_overflow_at_large_dimensions() {
        assert_eq!(
            OutputFormat::Fp32Nchw.byte_size(u32::MAX, 2),
            u32::MAX as u64 * 2 * 12
        );
    }

    #[test]
    fn every_format_has_its_own_shader_define() {
        let mut defines: Vec<u32> = ALL_FORMATS.iter().map(|f| f.define()).collect();
        defines.sort_unstable();
        defines.dedup();
        assert_eq!(defines, (0..7).collect::<Vec<u32>>());
    }

    #[test]
    fn the_shader_branches_only_on_defines_that_exist() {
        let defines: Vec<u32> = ALL_FORMATS.iter().map(|f| f.define()).collect();
        let mut rest = SHADER_BODY;
        let mut seen = 0;
        while let Some(at) = rest.find("OUTPUT == ") {
            rest = &rest[at + "OUTPUT == ".len()..];
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            let value: u32 = digits.parse().expect("a number after OUTPUT ==");
            assert!(defines.contains(&value), "shader tests OUTPUT == {value}");
            seen += 1;
        }
        assert!(seen > 0, "the shader selects its output by define");
    }

    #[test]
    fn yuv_and_fp16_classification() {
        let yuv: Vec<_> = ALL_FORMATS.iter().filter(|f| f.is_yuv()).collect();
        let fp16: Vec<_> = ALL_FORMATS.iter().filter(|f| f.is_fp16()).collect();
        assert_eq!(yuv, [&OutputFormat::Nv12, &OutputFormat::P010]);
        assert_eq!(fp16, [&OutputFormat::Fp16Nchw, &OutputFormat::Fp16Nhwc]);
    }

    #[test]
    fn the_root_constants_match_the_shader_cbuffer() {
        assert_eq!(cbuffer_fields(SHADER_BODY).len(), CONSTANT_COUNT);
    }

    #[test]
    fn a_region_record_is_four_uints() {
        assert!(SHADER_BODY.contains("struct Rect { uint x; uint y; uint w; uint h; };"));
        assert_eq!(RECT_BYTES, 4 * std::mem::size_of::<u32>() as u64);
    }

    #[test]
    fn every_supported_source_format_has_a_name() {
        for format in SUPPORTED_SOURCE_FORMATS {
            assert_ne!(format_name(format), "unsupported", "{format:?}");
        }
        assert_eq!(format_name(DXGI_FORMAT(0)), "unsupported");
    }

    #[test]
    fn yuv_defaults_to_limited_range_bt709() {
        let options = YuvOptions::default();
        assert_eq!(options.matrix, Matrix::Bt709);
        assert!(!options.full_range);
    }
}
