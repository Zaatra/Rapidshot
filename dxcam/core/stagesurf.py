import ctypes
from dataclasses import dataclass, InitVar
from typing import Tuple, Optional
from dxcam._libs.d3d11 import *
from dxcam._libs.dxgi import *
from dxcam.core.device import Device
from dxcam.core.output import Output


@dataclass
class StageSurface:
    """
    Staging surface for efficient copying from GPU to CPU memory.
    """
    width: ctypes.c_uint32 = 0
    height: ctypes.c_uint32 = 0
    dxgi_format: ctypes.c_uint32 = DXGI_FORMAT_B8G8R8A8_UNORM
    desc: D3D11_TEXTURE2D_DESC = D3D11_TEXTURE2D_DESC()
    texture: ctypes.POINTER(ID3D11Texture2D) = None
    interface: Optional[ctypes.POINTER(IDXGISurface)] = None
    output: InitVar[Output] = None
    device: InitVar[Device] = None

    def __post_init__(self, output, device) -> None:
        """
        Initialize the staging surface.
        
        Args:
            output: Output associated with the surface
            device: Device for creating the surface
        """
        self.rebuild(output, device)

    def release(self):
        """
        Release resources.
        """
        if self.texture is not None:
            self.width = 0
            self.height = 0
            self.texture.Release()
            self.texture = None
            self.interface = None

    def rebuild(self, output: Output, device: Device, dim: Optional[Tuple[int, int]] = None):
        """
        Rebuild the staging surface.
        
        Args:
            output: Output associated with the surface
            device: Device for creating the surface
            dim: Optional dimensions (width, height) override
        """
        # Set dimensions
        if dim is not None:
            self.width, self.height = dim
        else:
            self.width, self.height = output.surface_size

        # Only rebuild if texture doesn't exist yet
        if self.texture is None:
            self.desc.Width = self.width
            self.desc.Height = self.height
            self.desc.Format = self.dxgi_format
            self.desc.MipLevels = 1
            self.desc.ArraySize = 1
            self.desc.SampleDesc.Count = 1
            self.desc.SampleDesc.Quality = 0
            self.desc.Usage = D3D11_USAGE_STAGING
            self.desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ
            self.desc.MiscFlags = 0
            self.desc.BindFlags = 0
            
            self.texture = ctypes.POINTER(ID3D11Texture2D)()
            device.device.CreateTexture2D(
                ctypes.byref(self.desc),
                None,
                ctypes.byref(self.texture),
            )
            
            # Cache the surface interface for improved performance
            self.interface = self.texture.QueryInterface(IDXGISurface)

    def map(self) -> DXGI_MAPPED_RECT:
        """
        Map the surface to system memory.
        
        Returns:
            Mapped rectangle
        """
        rect = DXGI_MAPPED_RECT()
        if self.interface:
            # Use cached interface for better performance
            self.interface.Map(ctypes.byref(rect), 1)
        else:
            # Fall back to querying interface
            self.texture.QueryInterface(IDXGISurface).Map(ctypes.byref(rect), 1)
        return rect

    def unmap(self):
        """
        Unmap the surface from system memory.
        """
        if self.interface:
            # Use cached interface for better performance
            self.interface.Unmap()
        else:
            # Fall back to querying interface
            self.texture.QueryInterface(IDXGISurface).Unmap()

    def __repr__(self) -> str:
        """
        String representation.
        
        Returns:
            String representation
        """
        return "<{} Initialized:{} Size:{} Format:{}>".format(
            self.__class__.__name__,
            self.texture is not None,
            (self.width, self.height),
            "DXGI_FORMAT_B8G8R8A8_UNORM",
        )