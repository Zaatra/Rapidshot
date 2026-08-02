//! The conversion shader, on D3D12 — where DirectML can reach the result.
//!
//! Why this exists alongside `preprocess.rs`: the D3D11 version produces a
//! correct tensor, but that tensor can never reach DirectML. D3D11 can only
//! share 2D non-mipmapped textures, never buffers, so a D3D11-written buffer
//! has no route to a D3D12/DML device — verified across six configurations in
//! `probe_shareable_buffers`.
//!
//! Running the same shader on D3D12 removes the problem instead of working
//! around it. The captured *texture* does share (milestone 2 proved it opens on
//! D3D12), and once the shader runs there, its output buffer is already resident
//! on the DirectML device. There is no sharing step left to fail.
//!
//! The HLSL is identical to the D3D11 path; only the host-side plumbing differs.

use windows::core::{Interface, PCSTR};
use windows::Win32::Foundation::{CloseHandle, HANDLE};
use windows::Win32::Graphics::Direct3D::Fxc::{D3DCompile, D3DCOMPILE_OPTIMIZATION_LEVEL3};
use windows::Win32::Graphics::Direct3D::{ID3DBlob, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_UNKNOWN};
use windows::Win32::Graphics::Dxgi::{IDXGIAdapter, IDXGIDevice, IDXGIResource1};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject, INFINITE};

const DXGI_SHARED_RESOURCE_READ: u32 = 0x8000_0000;

/// Same conversion as the D3D11 path. Params arrive as root constants rather
/// than a constant buffer, which removes a resource and an upload per frame.
const SHADER_SOURCE: &str = r#"
Texture2D<float4> Source : register(t0);
RWStructuredBuffer<float> Output : register(u0);

cbuffer Params : register(b0)
{
    uint OutWidth;
    uint OutHeight;
    uint SrcWidth;
    uint SrcHeight;
    float Scale;
    float Bias;
    uint  ChannelOrder;   // 0 = RGB, 1 = BGR
    uint  _pad;
};

[numthreads(8, 8, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= OutWidth || tid.y >= OutHeight)
        return;

    uint sx = (SrcWidth  == OutWidth ) ? tid.x : (tid.x * SrcWidth  / OutWidth );
    uint sy = (SrcHeight == OutHeight) ? tid.y : (tid.y * SrcHeight / OutHeight);

    // The hardware swizzles BGRA formats, so .x is RED here despite blue being
    // first in memory. Getting this backwards produces BGR labelled RGB, which
    // no test of speed or shape would catch.
    float4 texel = Source.Load(int3(sx, sy, 0));

    float c0 = (ChannelOrder == 0) ? texel.x : texel.z;
    float c1 = texel.y;
    float c2 = (ChannelOrder == 0) ? texel.z : texel.x;

    uint plane = OutWidth * OutHeight;
    uint idx   = tid.y * OutWidth + tid.x;

    Output[0 * plane + idx] = c0 * Scale + Bias;
    Output[1 * plane + idx] = c1 * Scale + Bias;
    Output[2 * plane + idx] = c2 * Scale + Bias;
}
"#;

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
                    "shader compilation failed: {}",
                    String::from_utf8_lossy(text)
                ),
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

pub struct Preprocessor12 {
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
    descriptor_size: u32,
    output: ID3D12Resource,
    readback: ID3D12Resource,
    pub out_width: u32,
    pub out_height: u32,
}

impl Preprocessor12 {
    /// Build a D3D12 pipeline on the same adapter as the given D3D11 texture.
    ///
    /// Sharing only works within one adapter, so the D3D12 device must be
    /// created on whichever adapter the capture is running on.
    pub fn new(
        d3d11_texture: &ID3D11Texture2D,
        out_width: u32,
        out_height: u32,
    ) -> windows::core::Result<Self> {
        // Fail at setup rather than on the first frame: a texture that cannot be
        // shared will never work on this path, and finding that out during
        // construction is far easier to act on than a dispatch-time error.
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
                            "texture is not shareable, so DirectML cannot reach it; \
                             it must carry D3D11_RESOURCE_MISC_SHARED_NTHANDLE",
                        )
                    })?;
            unsafe {
                let _ = CloseHandle(handle);
            }
        }

        let adapter: IDXGIAdapter = unsafe {
            let d3d11_device: ID3D11Device = d3d11_texture.GetDevice()?;
            let dxgi: IDXGIDevice = d3d11_device.cast()?;
            dxgi.GetAdapter()?
        };

        let mut device: Option<ID3D12Device> = None;
        unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device)? };
        let device = device.expect("D3D12CreateDevice reported success");

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

        // Root signature: 8 root constants at b0, plus a table holding the SRV
        // and UAV. Root constants avoid a constant buffer resource entirely.
        let ranges = [
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 0,
            },
            D3D12_DESCRIPTOR_RANGE {
                RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_UAV,
                NumDescriptors: 1,
                BaseShaderRegister: 0,
                RegisterSpace: 0,
                OffsetInDescriptorsFromTableStart: 1,
            },
        ];
        let params = [
            D3D12_ROOT_PARAMETER {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: 8,
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
        let root_desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: params.len() as u32,
            pParameters: params.as_ptr(),
            NumStaticSamplers: 0,
            pStaticSamplers: std::ptr::null(),
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

        let bytecode = compile_shader()?;
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
                NumDescriptors: 2,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?
        };
        let descriptor_size = unsafe {
            device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };

        let byte_size = (out_width * out_height * 3 * 4) as u64;

        // Output lives in the DEFAULT heap: device-local, and exactly the kind
        // of resource DirectML binds to.
        let output = create_buffer(
            &device,
            byte_size,
            D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )?;
        // Readback is for verification only; the production path never uses it.
        let readback = create_buffer(
            &device,
            byte_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;

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
            descriptor_size,
            output,
            readback,
            out_width,
            out_height,
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

    /// Convert one captured frame, entirely on the D3D12 device.
    pub fn process(
        &self,
        d3d11_texture: &ID3D11Texture2D,
        scale: f32,
        bias: f32,
        channel_order: u32,
    ) -> windows::core::Result<()> {
        // Share the captured texture across to D3D12. Textures are the one
        // resource type D3D11 can share, which is why the shader lives here.
        let resource: IDXGIResource1 = d3d11_texture.cast()?;
        let handle = unsafe { resource.CreateSharedHandle(None, DXGI_SHARED_RESOURCE_READ, None)? };

        let result = (|| -> windows::core::Result<()> {
            let mut shared: Option<ID3D12Resource> = None;
            unsafe { self.device.OpenSharedHandle(handle, &mut shared)? };
            let shared = shared.expect("OpenSharedHandle reported success");

            let desc = unsafe { shared.GetDesc() };
            let src_width = desc.Width as u32;
            let src_height = desc.Height;

            // Descriptors: SRV over the shared texture, UAV over the output.
            let cpu_start = unsafe { self.heap.GetCPUDescriptorHandleForHeapStart() };
            let srv_handle = cpu_start;
            let uav_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: cpu_start.ptr + self.descriptor_size as usize,
            };

            let mut srv_desc = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT_B8G8R8A8_UNORM,
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

            let element_count = self.out_width * self.out_height * 3;
            let mut uav_desc = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: DXGI_FORMAT_UNKNOWN,
                ViewDimension: D3D12_UAV_DIMENSION_BUFFER,
                ..Default::default()
            };
            uav_desc.Anonymous.Buffer = D3D12_BUFFER_UAV {
                FirstElement: 0,
                NumElements: element_count,
                StructureByteStride: 4,
                CounterOffsetInBytes: 0,
                Flags: D3D12_BUFFER_UAV_FLAG_NONE,
            };
            unsafe {
                self.device.CreateUnorderedAccessView(
                    &self.output,
                    None,
                    Some(&uav_desc),
                    uav_handle,
                )
            };

            let constants: [u32; 8] = [
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

                // A resource opened from a shared handle arrives in COMMON.
                transition(
                    &self.list,
                    &shared,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                );

                self.list.SetComputeRootSignature(&self.root_signature);
                self.list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                self.list
                    .SetComputeRoot32BitConstants(0, 8, constants.as_ptr() as *const _, 0);
                self.list.SetComputeRootDescriptorTable(
                    1,
                    self.heap.GetGPUDescriptorHandleForHeapStart(),
                );

                let groups_x = self.out_width.div_ceil(8);
                let groups_y = self.out_height.div_ceil(8);
                self.list.Dispatch(groups_x, groups_y, 1);

                // Hand the texture back in the state D3D11 expects.
                transition(
                    &self.list,
                    &shared,
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    D3D12_RESOURCE_STATE_COMMON,
                );

                self.list.Close()?;
                self.queue
                    .ExecuteCommandLists(&[Some(self.list.cast::<ID3D12CommandList>()?)]);
            }
            self.wait_for_gpu()
        })();

        unsafe {
            let _ = CloseHandle(handle);
        }
        result
    }

    /// Copy the tensor to the CPU. Verification only.
    pub fn read_back(&self) -> windows::core::Result<Vec<f32>> {
        let count = (self.out_width * self.out_height * 3) as usize;
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

        let mut out = vec![0f32; count];
        unsafe {
            let mut mapped: *mut std::ffi::c_void = std::ptr::null_mut();
            self.readback.Map(0, None, Some(&mut mapped))?;
            std::ptr::copy_nonoverlapping(mapped as *const f32, out.as_mut_ptr(), count);
            self.readback.Unmap(0, None);
        }
        Ok(out)
    }

    /// Address of the D3D12 output buffer — what DirectML will bind to.
    pub fn output_resource_address(&self) -> usize {
        self.output.as_raw() as usize
    }

    /// GPU virtual address of the output buffer.
    pub fn output_gpu_address(&self) -> u64 {
        unsafe { self.output.GetGPUVirtualAddress() }
    }
}

impl Drop for Preprocessor12 {
    fn drop(&mut self) {
        // Never destroy resources the GPU is still reading.
        let _ = self.wait_for_gpu();
        if !self.fence_event.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.fence_event);
            }
        }
    }
}

fn create_buffer(
    device: &ID3D12Device,
    size: u64,
    heap_type: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> windows::core::Result<ID3D12Resource> {
    let heap_props = D3D12_HEAP_PROPERTIES {
        Type: heap_type,
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
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
    };
    let mut resource: Option<ID3D12Resource> = None;
    unsafe {
        device.CreateCommittedResource(
            &heap_props,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut resource,
        )?;
    }
    Ok(resource.expect("CreateCommittedResource reported success"))
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
