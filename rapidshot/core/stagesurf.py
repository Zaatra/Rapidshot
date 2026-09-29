import ctypes
from dataclasses import dataclass, field, InitVar
from typing import Tuple, Optional

import numpy as np

from rapidshot.core import hdr
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *
from rapidshot.core.device import Device
from rapidshot.core.output import Output


@dataclass
class StageSurface:
    """
    Staging surface for efficient copying from GPU to CPU memory.
    """
    width: ctypes.c_uint32 = 0
    height: ctypes.c_uint32 = 0
    dxgi_format: ctypes.c_uint32 = DXGI_FORMAT_B8G8R8A8_UNORM
    # A factory, not a default instance: a ctypes struct default is created
    # once and shared, so every StageSurface wrote into the same description.
    desc: D3D11_TEXTURE2D_DESC = field(default_factory=D3D11_TEXTURE2D_DESC)
    texture: ctypes.POINTER(ID3D11Texture2D) = None
    interface: Optional[ctypes.POINTER(IDXGISurface)] = None
    #: The display's HDR state and SDR white, used only when the surface is
    #: not BGRA8. Set by the capture before :meth:`map`.
    color: Optional[hdr.DisplayColor] = None
    _converted: Optional[np.ndarray] = field(default=None, repr=False)
    _mapped: bool = field(default=False, repr=False)
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

        Dropping the reference is the release: comtypes releases the COM
        pointer when the Python object goes away, so calling ``Release()``
        here as well decremented the count twice for one reference. See
        :meth:`Device.release`.
        """
        if self.texture is not None:
            self.width = 0
            self.height = 0
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

    @staticmethod
    def format_of(texture) -> int:
        """The DXGI format of the texture about to be copied in."""
        desc = D3D11_TEXTURE2D_DESC()
        texture.GetDesc(ctypes.byref(desc))
        return desc.Format

    def ensure(self, output: Output, device: Device, dim: Tuple[int, int],
               source_format: int) -> None:
        """Rebuild if the size or the source's format changed.

        The copy into a staging texture of another format fails without an
        error and leaves it zeroed: an HDR desktop, duplicated as FP16 or
        R10G10B10A2, read back as a black frame through a BGRA8 surface.
        """
        if (self.texture is not None and (self.width, self.height) == tuple(dim)
                and self.dxgi_format == source_format):
            return
        self.release()
        self.dxgi_format = source_format
        self.rebuild(output=output, device=device, dim=dim)

    def _map_raw(self) -> DXGI_MAPPED_RECT:
        rect = DXGI_MAPPED_RECT()
        if self.interface:
            # Use cached interface for better performance
            self.interface.Map(ctypes.byref(rect), 1)
        else:
            # Fall back to querying interface
            self.texture.QueryInterface(IDXGISurface).Map(ctypes.byref(rect), 1)
        return rect

    def _unmap_raw(self) -> None:
        if self.interface:
            # Use cached interface for better performance
            self.interface.Unmap()
        else:
            # Fall back to querying interface
            self.texture.QueryInterface(IDXGISurface).Unmap()

    def map(self) -> DXGI_MAPPED_RECT:
        """
        Map the surface to system memory, as 8-bit BGRA whatever it holds.

        A BGRA8 surface is mapped as it is. Any other supported format is
        converted into a BGRA8 buffer this surface keeps until :meth:`unmap`,
        and the rectangle returned points at that -- so every processor keeps
        reading the one layout it knows. See :mod:`rapidshot.core.hdr`.

        Returns:
            Mapped rectangle
        """
        rect = self._map_raw()
        if self.dxgi_format == DXGI_FORMAT_B8G8R8A8_UNORM:
            self._mapped = True
            return rect
        try:
            rows = np.ctypeslib.as_array(
                ctypes.cast(rect.pBits, ctypes.POINTER(ctypes.c_uint8)),
                shape=(self.height, rect.Pitch))
            self._converted = hdr.to_bgra8(rows, self.width, self.height,
                                           self.dxgi_format, self.color)
        finally:
            self._unmap_raw()
        converted = DXGI_MAPPED_RECT()
        converted.Pitch = self.width * 4
        converted.pBits = self._converted.ctypes.data
        return converted

    def unmap(self):
        """
        Unmap the surface from system memory.
        """
        if self._mapped:
            self._mapped = False
            self._unmap_raw()
        # A converted frame was unmapped as soon as it was converted, and its
        # BGRA8 buffer stays readable until the next map().

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
            "DXGI_FORMAT_" + hdr.FORMAT_NAMES.get(self.dxgi_format, str(self.dxgi_format)),
        )