//! GPU preprocessing: captured BGRA8 texture -> linear NCHW float32 tensor.
//!
//! This is the step that makes zero-copy inference actually pay off. Milestone 2
//! established that the duplicated desktop surface can be opened on a D3D12
//! device, but DirectML does not consume textures — it consumes *linear buffers*
//! in NCHW float32. The desktop surface is a tiled BGRA8 texture, so a
//! conversion is unavoidable.
//!
//! Doing it here, on the GPU, is what removes the CPU round-trip. Measured on
//! the CPU path for a 1920x1080 frame:
//!
//! * 2.27 ms reading the mapped staging surface
//! * 1.74 ms BGRA -> RGB conversion
//! * 3.80 ms resize / normalise / transpose to NCHW
//!
//! All three collapse into one dispatch that never leaves the GPU. Because the
//! shader is already touching every pixel, folding the resize and normalisation
//! in costs essentially nothing extra — which is why Stage 6b is not a separate
//! piece of work.

use windows::core::PCSTR;
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_B8G8R8A8_UNORM;

/// Compute shader source.
///
/// Reading through a `Texture2D<float4>` SRV means the hardware's texture unit
/// handles the tiled layout, the UNORM->float conversion *and* the BGRA channel
/// swizzle in one step, so the shader sees normalised 0..1 values already in
/// semantic RGBA order. No manual unpacking, and no channel reversal — see the
/// channel-order comment in the shader body for why that last part matters.
const SHADER_SOURCE: &str = r#"
Texture2D<float4> Source : register(t0);
RWStructuredBuffer<float> Output : register(u0);

cbuffer Params : register(b0)
{
    uint OutWidth;
    uint OutHeight;
    uint SrcWidth;
    uint SrcHeight;
    float Scale;      // multiply after the 0..1 texture fetch
    float Bias;       // then add
    uint  ChannelOrder; // 0 = RGB, 1 = BGR
    uint  _pad;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth || tid.y >= OutHeight)
        return;

    // Nearest-neighbour sampling. Deliberate: it is exact and cheap, and any
    // filtering choice belongs to the caller's model, not to the capture layer.
    uint sx = (SrcWidth  == OutWidth ) ? tid.x : (tid.x * SrcWidth  / OutWidth );
    uint sy = (SrcHeight == OutHeight) ? tid.y : (tid.y * SrcHeight / OutHeight);

    // The hardware presents texture components *semantically*, not in memory
    // order: for DXGI_FORMAT_B8G8R8A8_UNORM it swizzles on the fly, so .x is
    // RED even though blue is stored first. Verified empirically -- an earlier
    // version swapped these by hand and produced BGR while reporting RGB, which
    // is silently wrong model input rather than an visible error.
    float4 texel = Source.Load(int3(sx, sy, 0));   // .x=R .y=G .z=B .w=A

    float c0 = (ChannelOrder == 0) ? texel.x : texel.z;   // RGB->R, BGR->B
    float c1 = texel.y;                                   // G either way
    float c2 = (ChannelOrder == 0) ? texel.z : texel.x;   // RGB->B, BGR->R

    // NCHW: each channel is a contiguous plane, which is the layout ONNX
    // Runtime and DirectML expect for image models.
    uint plane = OutWidth * OutHeight;
    uint idx   = tid.y * OutWidth + tid.x;

    Output[0 * plane + idx] = c0 * Scale + Bias;
    Output[1 * plane + idx] = c1 * Scale + Bias;
    Output[2 * plane + idx] = c2 * Scale + Bias;
}
"#;

/// Constant buffer layout. Must stay 16-byte aligned to match HLSL packing.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Params {
    pub out_width: u32,
    pub out_height: u32,
    pub src_width: u32,
    pub src_height: u32,
    pub scale: f32,
    pub bias: f32,
    pub channel_order: u32,
    pub _pad: u32,
}

/// Everything needed to run the conversion, built once and reused per frame.
pub struct Preprocessor {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    shader: ID3D11ComputeShader,
    pub out_width: u32,
    pub out_height: u32,
    output: ID3D11Buffer,
    uav: ID3D11UnorderedAccessView,
    constants: ID3D11Buffer,
    staging: ID3D11Buffer,
}

fn compile_shader() -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;

    let result = unsafe {
        D3DCompile(
            SHADER_SOURCE.as_ptr() as *const _,
            SHADER_SOURCE.len(),
            None,
            None,
            None,
            PCSTR(b"CSMain\0".as_ptr()),
            PCSTR(b"cs_5_0\0".as_ptr()),
            D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut code,
            Some(&mut errors),
        )
    };

    if result.is_err() {
        if let Some(errors) = errors {
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer() as *const u8,
                    errors.GetBufferSize(),
                )
            };
            let text = String::from_utf8_lossy(text);
            return Err(windows::core::Error::new(
                result.unwrap_err().code(),
                format!("shader compilation failed: {text}"),
            ));
        }
        result?;
    }
    code.ok_or_else(|| {
        windows::core::Error::new(
            windows::Win32::Foundation::E_FAIL,
            "shader compiled but produced no bytecode",
        )
    })
}

impl Preprocessor {
    /// Build the pipeline for a fixed output size.
    ///
    /// The buffers are allocated once and reused, so the per-frame path is just
    /// a dispatch — no allocation, which is the whole point of avoiding the CPU.
    pub fn new(
        device: ID3D11Device,
        out_width: u32,
        out_height: u32,
    ) -> windows::core::Result<Self> {
        let context = unsafe { device.GetImmediateContext()? };

        let bytecode = compile_shader()?;
        let shader = unsafe {
            let slice = std::slice::from_raw_parts(
                bytecode.GetBufferPointer() as *const u8,
                bytecode.GetBufferSize(),
            );
            let mut shader = None;
            device.CreateComputeShader(slice, None, Some(&mut shader))?;
            shader.expect("CreateComputeShader reported success")
        };

        let element_count = out_width * out_height * 3;
        let byte_size = element_count * 4;

        // Output buffer.
        //
        // Note the absence of sharing flags — deliberately, because none work.
        // D3D11 can share only 2D non-mipmapped textures, never buffers: all six
        // configurations were probed (structured / raw / plain, each with
        // NT-handle and legacy sharing) and none produced a buffer D3D12 could
        // open. See `probe_shareable_buffers`.
        //
        // That is why `preprocess12.rs` exists: the D3D12 version of this shader
        // writes a buffer already resident on the DirectML device, so no sharing
        // step is needed. This D3D11 path remains useful for callers who want
        // GPU-side preprocessing without DirectML.
        let output = unsafe {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: byte_size,
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: (D3D11_BIND_UNORDERED_ACCESS.0 | D3D11_BIND_SHADER_RESOURCE.0) as u32,
                CPUAccessFlags: 0,
                MiscFlags: D3D11_RESOURCE_MISC_BUFFER_STRUCTURED.0 as u32,
                StructureByteStride: 4,
            };
            let mut buffer = None;
            device.CreateBuffer(&desc, None, Some(&mut buffer))?;
            buffer.expect("CreateBuffer reported success")
        };

        let uav = unsafe {
            let mut desc = D3D11_UNORDERED_ACCESS_VIEW_DESC::default();
            desc.Format = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_UNKNOWN;
            desc.ViewDimension = D3D11_UAV_DIMENSION_BUFFER;
            desc.Anonymous.Buffer = D3D11_BUFFER_UAV {
                FirstElement: 0,
                NumElements: element_count,
                Flags: 0,
            };
            let mut uav = None;
            device.CreateUnorderedAccessView(&output, Some(&desc), Some(&mut uav))?;
            uav.expect("CreateUnorderedAccessView reported success")
        };

        let constants = unsafe {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: std::mem::size_of::<Params>() as u32,
                Usage: D3D11_USAGE_DYNAMIC,
                BindFlags: D3D11_BIND_CONSTANT_BUFFER.0 as u32,
                CPUAccessFlags: D3D11_CPU_ACCESS_WRITE.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut buffer = None;
            device.CreateBuffer(&desc, None, Some(&mut buffer))?;
            buffer.expect("CreateBuffer reported success")
        };

        // Readback buffer, used only by the verification path. The production
        // path never touches it — that is the entire point.
        let staging = unsafe {
            let desc = D3D11_BUFFER_DESC {
                ByteWidth: byte_size,
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
                StructureByteStride: 0,
            };
            let mut buffer = None;
            device.CreateBuffer(&desc, None, Some(&mut buffer))?;
            buffer.expect("CreateBuffer reported success")
        };

        Ok(Self {
            device,
            context,
            shader,
            out_width,
            out_height,
            output,
            uav,
            constants,
            staging,
        })
    }

    /// Run the conversion for one captured texture. GPU-only; nothing is read back.
    pub fn dispatch(
        &self,
        texture: &ID3D11Texture2D,
        scale: f32,
        bias: f32,
        channel_order: u32,
    ) -> windows::core::Result<()> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { texture.GetDesc(&mut desc) };

        let srv = unsafe {
            let mut view_desc = D3D11_SHADER_RESOURCE_VIEW_DESC::default();
            // The duplicated surface is BGRA8_UNORM; state it explicitly so the
            // fetch returns normalised floats rather than raw integers.
            view_desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
            view_desc.ViewDimension =
                windows::Win32::Graphics::Direct3D::D3D_SRV_DIMENSION_TEXTURE2D;
            view_desc.Anonymous.Texture2D = D3D11_TEX2D_SRV {
                MostDetailedMip: 0,
                MipLevels: 1,
            };
            let mut srv = None;
            self.device
                .CreateShaderResourceView(texture, Some(&view_desc), Some(&mut srv))?;
            srv.expect("CreateShaderResourceView reported success")
        };

        let params = Params {
            out_width: self.out_width,
            out_height: self.out_height,
            src_width: desc.Width,
            src_height: desc.Height,
            scale,
            bias,
            channel_order,
            _pad: 0,
        };

        unsafe {
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context.Map(
                &self.constants,
                0,
                D3D11_MAP_WRITE_DISCARD,
                0,
                Some(&mut mapped),
            )?;
            std::ptr::copy_nonoverlapping(
                &params as *const Params as *const u8,
                mapped.pData as *mut u8,
                std::mem::size_of::<Params>(),
            );
            self.context.Unmap(&self.constants, 0);

            self.context.CSSetShader(&self.shader, None);
            self.context
                .CSSetShaderResources(0, Some(&[Some(srv.clone())]));
            self.context
                .CSSetUnorderedAccessViews(0, 1, Some(&Some(self.uav.clone())), None);
            self.context
                .CSSetConstantBuffers(0, Some(&[Some(self.constants.clone())]));

            // 8x8 threadgroups, rounded up to cover the output.
            let groups_x = (self.out_width + 7) / 8;
            let groups_y = (self.out_height + 7) / 8;
            self.context.Dispatch(groups_x, groups_y, 1);

            // Unbind so the next frame's SRV creation is not blocked by a stale
            // binding on the same resource.
            self.context.CSSetShaderResources(0, Some(&[None]));
            self.context
                .CSSetUnorderedAccessViews(0, 1, Some(&None), None);
        }
        Ok(())
    }

    /// Copy the result to the CPU. Verification only — the production path
    /// keeps the tensor on the GPU.
    pub fn read_back(&self) -> windows::core::Result<Vec<f32>> {
        let count = (self.out_width * self.out_height * 3) as usize;
        let mut out = vec![0f32; count];
        unsafe {
            self.context.CopyResource(&self.staging, &self.output);
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            self.context
                .Map(&self.staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(mapped.pData as *const f32, out.as_mut_ptr(), count);
            self.context.Unmap(&self.staging, 0);
        }
        Ok(out)
    }

    /// Address of the output buffer, for milestone 3b's DirectML binding.
    pub fn output_buffer_address(&self) -> usize {
        use windows::core::Interface;
        self.output.as_raw() as usize
    }
}
