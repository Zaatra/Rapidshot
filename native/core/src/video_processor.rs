//! NV12 through the GPU's video processor: the other route to an encoder's
//! input, measured against `Converter12`'s `Nv12Texture` output.
//!
//! The video processor is a fixed-function block, the one video players use
//! to scale and convert, driven through D3D11. One `VideoProcessorBlt` takes
//! the captured BGRA texture -- the desktop image itself, on the device that
//! captured it -- crops it, scales it and converts it to NV12 into a texture on
//! that same device, which is where a hardware encoder reads it.
//!
//! What it cannot promise is the converter's exactness. The matrix, range,
//! chroma siting and scaling filter are the driver's; this asks for BT.709,
//! limited range, left-sited chroma (`DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709`)
//! and records whether the driver says it supports that conversion, but the
//! bytes differ by vendor and driver version. `tests` and the `nv12_routes`
//! benchmark measure how far they are from the converter on this machine.

use windows::core::{Interface, BOOL};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::*;
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709, DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_NV12, DXGI_RATIONAL, DXGI_SAMPLE_DESC,
};

/// A source rectangle in texels of the input texture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRect {
    pub left: i32,
    pub top: i32,
    pub right: i32,
    pub bottom: i32,
}

pub struct VideoProcessorNv12 {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    video_context: ID3D11VideoContext1,
    video_device: ID3D11VideoDevice,
    enumerator: ID3D11VideoProcessorEnumerator,
    processor: ID3D11VideoProcessor,
    output: ID3D11Texture2D,
    output_view: ID3D11VideoProcessorOutputView,
    staging: Option<ID3D11Texture2D>,
    /// The input view for the last texture seen, keyed by its address. A
    /// desktop surface keeps its address across frames, so this is one view
    /// for the life of a capture rather than one per frame.
    cached_input: Option<(usize, ID3D11VideoProcessorInputView)>,
    pub input_width: u32,
    pub input_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    /// Whether the driver says it supports exactly the conversion asked for
    /// (`CheckVideoProcessorFormatConversion`). `None` when it cannot be asked.
    pub conversion_supported: Option<bool>,
}

fn invalid(message: &str) -> windows::core::Error {
    windows::core::Error::new(windows::Win32::Foundation::E_INVALIDARG, message)
}

impl VideoProcessorNv12 {
    /// A processor converting `input_width` x `input_height` BGRA textures on
    /// `device` into an `out_width` x `out_height` NV12 texture.
    pub fn new(
        device: &ID3D11Device,
        input_width: u32,
        input_height: u32,
        out_width: u32,
        out_height: u32,
    ) -> windows::core::Result<Self> {
        if out_width == 0
            || out_height == 0
            || !out_width.is_multiple_of(2)
            || !out_height.is_multiple_of(2)
        {
            return Err(invalid(
                "NV12 output needs a non-zero, even width and height",
            ));
        }
        let video_device: ID3D11VideoDevice = device.cast()?;
        let context = unsafe { device.GetImmediateContext()? };
        let video_context: ID3D11VideoContext1 = context.cast()?;

        let rate = DXGI_RATIONAL {
            Numerator: 60,
            Denominator: 1,
        };
        let content = D3D11_VIDEO_PROCESSOR_CONTENT_DESC {
            InputFrameFormat: D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            InputFrameRate: rate,
            InputWidth: input_width,
            InputHeight: input_height,
            OutputFrameRate: rate,
            OutputWidth: out_width,
            OutputHeight: out_height,
            Usage: D3D11_VIDEO_USAGE_OPTIMAL_SPEED,
        };
        let enumerator = unsafe { video_device.CreateVideoProcessorEnumerator(&content)? };
        let supports = |format: DXGI_FORMAT, flag: D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT| -> bool {
            unsafe { enumerator.CheckVideoProcessorFormat(format) }
                .map(|f| f & flag.0 as u32 != 0)
                .unwrap_or(false)
        };
        if !supports(
            DXGI_FORMAT_B8G8R8A8_UNORM,
            D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_INPUT,
        ) {
            return Err(invalid("this video processor does not take BGRA input"));
        }
        if !supports(
            DXGI_FORMAT_NV12,
            D3D11_VIDEO_PROCESSOR_FORMAT_SUPPORT_OUTPUT,
        ) {
            return Err(invalid("this video processor does not write NV12"));
        }
        let conversion_supported = enumerator
            .cast::<ID3D11VideoProcessorEnumerator1>()
            .ok()
            .and_then(|e| unsafe {
                e.CheckVideoProcessorFormatConversion(
                    DXGI_FORMAT_B8G8R8A8_UNORM,
                    DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
                    DXGI_FORMAT_NV12,
                    DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
                )
                .ok()
            })
            .map(|b: BOOL| b.as_bool());
        let processor = unsafe { video_device.CreateVideoProcessor(&enumerator, 0)? };

        let output = unsafe {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: out_width,
                Height: out_height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_DEFAULT,
                BindFlags: D3D11_BIND_RENDER_TARGET.0 as u32,
                CPUAccessFlags: 0,
                MiscFlags: 0,
            };
            let mut texture = None;
            device.CreateTexture2D(&desc, None, Some(&mut texture))?;
            texture.expect("CreateTexture2D reported success")
        };
        let output_view = unsafe {
            let desc = D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC {
                ViewDimension: D3D11_VPOV_DIMENSION_TEXTURE2D,
                Anonymous: D3D11_VIDEO_PROCESSOR_OUTPUT_VIEW_DESC_0 {
                    Texture2D: D3D11_TEX2D_VPOV { MipSlice: 0 },
                },
            };
            let mut view = None;
            video_device.CreateVideoProcessorOutputView(
                &output,
                &enumerator,
                &desc,
                Some(&mut view),
            )?;
            view.expect("CreateVideoProcessorOutputView reported success")
        };

        unsafe {
            video_context.VideoProcessorSetStreamFrameFormat(
                &processor,
                0,
                D3D11_VIDEO_FRAME_FORMAT_PROGRESSIVE,
            );
            // No driver "enhancements": they change the pixels by heuristic.
            video_context.VideoProcessorSetStreamAutoProcessingMode(&processor, 0, false);
            video_context.VideoProcessorSetStreamColorSpace1(
                &processor,
                0,
                DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709,
            );
            video_context.VideoProcessorSetOutputColorSpace1(
                &processor,
                DXGI_COLOR_SPACE_YCBCR_STUDIO_G22_LEFT_P709,
            );
            let whole = RECT {
                left: 0,
                top: 0,
                right: out_width as i32,
                bottom: out_height as i32,
            };
            video_context.VideoProcessorSetStreamDestRect(&processor, 0, true, Some(&whole));
            video_context.VideoProcessorSetOutputTargetRect(&processor, true, Some(&whole));
        }

        Ok(Self {
            device: device.clone(),
            context,
            video_context,
            video_device,
            enumerator,
            processor,
            output,
            output_view,
            staging: None,
            cached_input: None,
            input_width,
            input_height,
            out_width,
            out_height,
            conversion_supported,
        })
    }

    /// The NV12 texture, on the input's device: what an encoder reads.
    pub fn output(&self) -> &ID3D11Texture2D {
        &self.output
    }

    fn input_view(
        &mut self,
        texture: &ID3D11Texture2D,
    ) -> windows::core::Result<ID3D11VideoProcessorInputView> {
        let key = texture.as_raw() as usize;
        if let Some((cached, view)) = &self.cached_input {
            if *cached == key {
                return Ok(view.clone());
            }
        }
        let desc = D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC {
            FourCC: 0,
            ViewDimension: D3D11_VPIV_DIMENSION_TEXTURE2D,
            Anonymous: D3D11_VIDEO_PROCESSOR_INPUT_VIEW_DESC_0 {
                Texture2D: D3D11_TEX2D_VPIV {
                    MipSlice: 0,
                    ArraySlice: 0,
                },
            },
        };
        let mut view = None;
        unsafe {
            self.video_device.CreateVideoProcessorInputView(
                texture,
                &self.enumerator,
                &desc,
                Some(&mut view),
            )?;
        }
        let view = view.expect("CreateVideoProcessorInputView reported success");
        self.cached_input = Some((key, view.clone()));
        Ok(view)
    }

    /// Convert `texture` (or the `crop` of it) into the NV12 output, scaled to
    /// the output size. Submits the work and returns; `wait` blocks until the
    /// GPU has finished it.
    pub fn process(
        &mut self,
        texture: &ID3D11Texture2D,
        crop: Option<SourceRect>,
    ) -> windows::core::Result<()> {
        let view = self.input_view(texture)?;
        let source = crop.unwrap_or(SourceRect {
            left: 0,
            top: 0,
            right: self.input_width as i32,
            bottom: self.input_height as i32,
        });
        let rect = RECT {
            left: source.left,
            top: source.top,
            right: source.right,
            bottom: source.bottom,
        };
        unsafe {
            self.video_context.VideoProcessorSetStreamSourceRect(
                &self.processor,
                0,
                true,
                Some(&rect),
            );
            let stream = D3D11_VIDEO_PROCESSOR_STREAM {
                Enable: true.into(),
                // Borrowed: the call reads the pointer and the ManuallyDrop
                // must not keep a reference nothing would release.
                pInputSurface: std::mem::transmute_copy(&view),
                ..Default::default()
            };
            self.video_context.VideoProcessorBlt(
                &self.processor,
                &self.output_view,
                0,
                &[stream],
            )?;
        }
        Ok(())
    }

    /// Block until everything submitted on this device has finished.
    pub fn wait(&self) -> windows::core::Result<()> {
        wait_for_device(&self.device, &self.context)
    }

    /// The output as tightly packed NV12 bytes (Y rows, then Cb/Cr rows), the
    /// layout `Converter12` reads back, so the two can be compared directly.
    pub fn read_back(&mut self) -> windows::core::Result<Vec<u8>> {
        if self.staging.is_none() {
            let desc = D3D11_TEXTURE2D_DESC {
                Width: self.out_width,
                Height: self.out_height,
                MipLevels: 1,
                ArraySize: 1,
                Format: DXGI_FORMAT_NV12,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            let mut texture = None;
            unsafe {
                self.device
                    .CreateTexture2D(&desc, None, Some(&mut texture))?
            };
            self.staging = texture;
        }
        let staging = self.staging.clone().expect("created above");
        read_back_nv12(
            &self.context,
            &staging,
            &self.output,
            self.out_width,
            self.out_height,
        )
    }
}

/// Copy an NV12 texture through `staging` and pack it as Y rows then Cb/Cr rows.
pub fn read_back_nv12(
    context: &ID3D11DeviceContext,
    staging: &ID3D11Texture2D,
    texture: &ID3D11Texture2D,
    width: u32,
    height: u32,
) -> windows::core::Result<Vec<u8>> {
    let (width, height) = (width as usize, height as usize);
    let mut out = Vec::with_capacity(width * height * 3 / 2);
    unsafe {
        context.CopyResource(staging, texture);
        let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
        context.Map(staging, 0, D3D11_MAP_READ, 0, Some(&mut mapped))?;
        let base = mapped.pData as *const u8;
        let pitch = mapped.RowPitch as usize;
        // D3D11 maps NV12 as one block: the Y plane, then the Cb/Cr plane
        // starting `height` rows down, at the same pitch.
        for row in 0..height + height / 2 {
            out.extend_from_slice(std::slice::from_raw_parts(base.add(row * pitch), width));
        }
        context.Unmap(staging, 0);
    }
    Ok(out)
}

/// Block until the GPU has finished everything submitted on `device`.
pub fn wait_for_device(
    device: &ID3D11Device,
    context: &ID3D11DeviceContext,
) -> windows::core::Result<()> {
    let desc = D3D11_QUERY_DESC {
        Query: D3D11_QUERY_EVENT,
        MiscFlags: 0,
    };
    let mut query = None;
    unsafe { device.CreateQuery(&desc, Some(&mut query))? };
    let query = query.expect("CreateQuery reported success");
    unsafe {
        context.End(&query);
        let mut done = BOOL(0);
        loop {
            // S_FALSE (not yet) is success with `done` still false.
            context.GetData(
                &query,
                Some(&mut done as *mut BOOL as *mut _),
                std::mem::size_of::<BOOL>() as u32,
                0,
            )?;
            if done.as_bool() {
                return Ok(());
            }
            std::thread::yield_now();
        }
    }
}
