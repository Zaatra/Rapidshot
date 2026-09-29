"""What an HDR desktop hands back, and how to turn it into the 8-bit sRGB the
CPU capture paths promise.

With Windows HDR on, the desktop is composed as linear scRGB -- 1.0 is 80 nits,
and SDR white sits at the user's "SDR content brightness", 200-240 nits being
typical -- and ``DuplicateOutput1`` returns it as ``R16G16B16A16_FLOAT`` or, on
some platforms, ``R10G10B10A2_UNORM``. Every CPU path downstream reads 8-bit
BGRA, so an HDR surface copied into a BGRA8 staging texture came back black.

The conversion here is the one OBS and RustDesk ship for HDR-to-SDR display
capture: divide by the SDR white level, clip to [0, 1], apply the sRGB transfer
function. SDR content drawn on the HDR desktop comes back as the bytes it was
drawn with; anything brighter than SDR white clips to white. That is SDR
normalisation, not tone mapping, and it is said so rather than dressed up.

Both formats go through lookup tables: 1024 entries for a 10-bit channel, and
65536 for FP16, indexed by the half-float's raw bits. So a 4K frame costs a
table gather per channel, not a transcendental per pixel.
"""
from __future__ import annotations

import ctypes
from ctypes import wintypes
import functools
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DXGI_FORMAT_R16G16B16A16_FLOAT = 10
DXGI_FORMAT_R10G10B10A2_UNORM = 24
DXGI_FORMAT_R8G8B8A8_UNORM = 28
DXGI_FORMAT_B8G8R8A8_UNORM = 87

FORMAT_NAMES = {
    DXGI_FORMAT_R16G16B16A16_FLOAT: "R16G16B16A16_FLOAT",
    DXGI_FORMAT_R10G10B10A2_UNORM: "R10G10B10A2_UNORM",
    DXGI_FORMAT_R8G8B8A8_UNORM: "R8G8B8A8_UNORM",
    DXGI_FORMAT_B8G8R8A8_UNORM: "B8G8R8A8_UNORM",
}

#: scRGB 1.0, and what Windows reports SDR white relative to.
NOMINAL_WHITE_NITS = 80.0


def is_supported(dxgi_format: int) -> bool:
    return dxgi_format in FORMAT_NAMES


# ---------------------------------------------------------------------------
# The display's colour state
# ---------------------------------------------------------------------------

class DisplayColor:
    """HDR on or off, and the SDR white level, for one display."""

    __slots__ = ("hdr", "sdr_white_nits")

    def __init__(self, hdr: bool = False, sdr_white_nits: float = NOMINAL_WHITE_NITS):
        self.hdr = bool(hdr)
        self.sdr_white_nits = float(sdr_white_nits)

    @property
    def sdr_white_scale(self) -> float:
        """SDR white in scRGB units: 3.0 at 240 nits."""
        return self.sdr_white_nits / NOMINAL_WHITE_NITS

    def __repr__(self) -> str:
        return f"<DisplayColor hdr={self.hdr} sdr_white={self.sdr_white_nits:g} nits>"


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _PATH_SOURCE_INFO(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("statusFlags", wintypes.UINT)]


class _RATIONAL(ctypes.Structure):
    _fields_ = [("Numerator", wintypes.UINT), ("Denominator", wintypes.UINT)]


class _PATH_TARGET_INFO(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", wintypes.UINT),
                ("modeInfoIdx", wintypes.UINT), ("outputTechnology", wintypes.UINT),
                ("rotation", wintypes.UINT), ("scaling", wintypes.UINT),
                ("refreshRate", _RATIONAL), ("scanLineOrdering", wintypes.UINT),
                ("targetAvailable", wintypes.BOOL), ("statusFlags", wintypes.UINT)]


class _PATH_INFO(ctypes.Structure):
    _fields_ = [("sourceInfo", _PATH_SOURCE_INFO), ("targetInfo", _PATH_TARGET_INFO),
                ("flags", wintypes.UINT)]


class _MODE_INFO(ctypes.Structure):
    # The union is 48 bytes (its largest member, DISPLAYCONFIG_TARGET_MODE);
    # nothing here reads it, so it is carried as bytes.
    _fields_ = [("infoType", wintypes.UINT), ("id", wintypes.UINT),
                ("adapterId", _LUID), ("mode", ctypes.c_byte * 48)]


class _HEADER(ctypes.Structure):
    _fields_ = [("type", wintypes.UINT), ("size", wintypes.UINT),
                ("adapterId", _LUID), ("id", wintypes.UINT)]


class _SOURCE_DEVICE_NAME(ctypes.Structure):
    _fields_ = [("header", _HEADER), ("viewGdiDeviceName", wintypes.WCHAR * 32)]


class _ADVANCED_COLOR_INFO(ctypes.Structure):
    _fields_ = [("header", _HEADER), ("flags", wintypes.UINT),
                ("colorEncoding", wintypes.UINT), ("bitsPerColorChannel", wintypes.UINT),
                ("activeColorMode", wintypes.UINT)]


class _SDR_WHITE_LEVEL(ctypes.Structure):
    _fields_ = [("header", _HEADER), ("SDRWhiteLevel", wintypes.ULONG)]


_QDC_ONLY_ACTIVE_PATHS = 0x2
_GET_SOURCE_NAME = 1
_GET_ADVANCED_COLOR_INFO = 9
_GET_SDR_WHITE_LEVEL = 11
_GET_ADVANCED_COLOR_INFO_2 = 15      # Windows 11 24H2; separates HDR from ACM


def display_color(gdi_device_name: str) -> DisplayColor:
    """HDR state and SDR white for the display GDI calls ``gdi_device_name``.

    Falls back to SDR at nominal white when anything here fails: a machine
    that cannot be asked is treated the way every capture path treated it
    before HDR was considered.
    """
    try:
        return _query(gdi_device_name)
    except Exception as exc:  # noqa: BLE001 -- best effort; SDR is the safe answer
        logger.debug("display colour query failed for %s: %s", gdi_device_name, exc)
        return DisplayColor()


def _query(gdi_device_name: str) -> DisplayColor:
    user32 = ctypes.WinDLL("user32")
    sizes = wintypes.UINT(), wintypes.UINT()
    if user32.GetDisplayConfigBufferSizes(_QDC_ONLY_ACTIVE_PATHS, ctypes.byref(sizes[0]),
                                          ctypes.byref(sizes[1])):
        return DisplayColor()
    paths = (_PATH_INFO * sizes[0].value)()
    modes = (_MODE_INFO * sizes[1].value)()
    if user32.QueryDisplayConfig(_QDC_ONLY_ACTIVE_PATHS, ctypes.byref(sizes[0]), paths,
                                 ctypes.byref(sizes[1]), modes, None):
        return DisplayColor()
    for path in paths[:sizes[0].value]:
        source = _SOURCE_DEVICE_NAME()
        source.header.type, source.header.size = _GET_SOURCE_NAME, ctypes.sizeof(source)
        source.header.adapterId, source.header.id = path.sourceInfo.adapterId, path.sourceInfo.id
        if user32.DisplayConfigGetDeviceInfo(ctypes.byref(source)):
            continue
        if source.viewGdiDeviceName.lower() != gdi_device_name.lower():
            continue
        return DisplayColor(_hdr_enabled(user32, path), _sdr_white_nits(user32, path))
    return DisplayColor()


def _hdr_enabled(user32, path) -> bool:
    info = _ADVANCED_COLOR_INFO()
    info.header.adapterId, info.header.id = path.targetInfo.adapterId, path.targetInfo.id
    info.header.type, info.header.size = _GET_ADVANCED_COLOR_INFO_2, ctypes.sizeof(info)
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(info)) == 0:
        return bool(info.flags & 32)            # highDynamicRangeUserEnabled
    info.header.type = _GET_ADVANCED_COLOR_INFO
    info.header.size = ctypes.sizeof(info) - ctypes.sizeof(wintypes.UINT)
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(info)) == 0:
        return bool(info.flags & 2)             # advancedColorEnabled, pre-24H2
    return False


def _sdr_white_nits(user32, path) -> float:
    level = _SDR_WHITE_LEVEL()
    level.header.type, level.header.size = _GET_SDR_WHITE_LEVEL, ctypes.sizeof(level)
    level.header.adapterId, level.header.id = path.targetInfo.adapterId, path.targetInfo.id
    if user32.DisplayConfigGetDeviceInfo(ctypes.byref(level)) == 0 and level.SDRWhiteLevel:
        # "A multiplier for 80 nits, times 1000": 1000 is 80 nits.
        return level.SDRWhiteLevel / 1000.0 * NOMINAL_WHITE_NITS
    return NOMINAL_WHITE_NITS


# ---------------------------------------------------------------------------
# Conversion to BGRA8
# ---------------------------------------------------------------------------

def _srgb_encode(linear: np.ndarray) -> np.ndarray:
    """sRGB OETF of values already clipped to [0, 1], rounded to 8 bits."""
    encoded = np.where(linear <= 0.0031308, linear * 12.92,
                       1.055 * np.power(linear, 1 / 2.4) - 0.055)
    return np.clip(np.rint(encoded * 255.0), 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=16)
def _lut_10bit(linear: bool, scale_milli: int) -> np.ndarray:
    codes = np.arange(1024, dtype=np.float64) / 1023.0
    if not linear:
        return np.clip(np.rint(codes * 255.0), 0, 255).astype(np.uint8)
    return _srgb_encode(np.clip(codes / (scale_milli / 1000.0), 0.0, 1.0))


@functools.lru_cache(maxsize=16)
def _lut_fp16(linear: bool, scale_milli: int) -> np.ndarray:
    values = np.arange(65536, dtype=np.uint32).astype(np.uint16).view(np.float16)
    values = np.nan_to_num(values.astype(np.float64), nan=0.0, posinf=65504.0, neginf=0.0)
    if linear:
        values = values / (scale_milli / 1000.0)
        return _srgb_encode(np.clip(values, 0.0, 1.0))
    return np.clip(np.rint(np.clip(values, 0.0, 1.0) * 255.0), 0, 255).astype(np.uint8)


def to_bgra8(rows: np.ndarray, width: int, height: int, dxgi_format: int,
             color: Optional[DisplayColor] = None) -> np.ndarray:
    """``(height, width, 4)`` uint8 BGRA from mapped staging rows.

    ``rows`` is the mapped surface as ``(height, pitch)`` bytes. 10-bit and
    FP16 are linear scRGB only while HDR is on; otherwise they carry the
    desktop's gamma-encoded values and are only rescaled. Alpha is opaque,
    as the BGRA8 duplication surface's is.
    """
    color = color or DisplayColor()
    scale_milli = max(1, int(round(color.sdr_white_scale * 1000)))
    out = np.empty((height, width, 4), dtype=np.uint8)
    out[..., 3] = 255
    if dxgi_format == DXGI_FORMAT_B8G8R8A8_UNORM:
        out[...] = rows[:, : width * 4].reshape(height, width, 4)
        return out
    if dxgi_format == DXGI_FORMAT_R8G8B8A8_UNORM:
        rgba = rows[:, : width * 4].reshape(height, width, 4)
        out[..., 0], out[..., 1], out[..., 2] = rgba[..., 2], rgba[..., 1], rgba[..., 0]
        return out
    if dxgi_format == DXGI_FORMAT_R10G10B10A2_UNORM:
        packed = np.ascontiguousarray(rows[:, : width * 4]).view(np.uint32).reshape(height, width)
        lut = _lut_10bit(color.hdr, scale_milli)
        out[..., 2] = lut[packed & 0x3FF]
        out[..., 1] = lut[(packed >> 10) & 0x3FF]
        out[..., 0] = lut[(packed >> 20) & 0x3FF]
        return out
    if dxgi_format == DXGI_FORMAT_R16G16B16A16_FLOAT:
        halves = np.ascontiguousarray(rows[:, : width * 8]).view(np.uint16).reshape(height, width, 4)
        lut = _lut_fp16(color.hdr, scale_milli)
        out[..., 2] = lut[halves[..., 0]]
        out[..., 1] = lut[halves[..., 1]]
        out[..., 0] = lut[halves[..., 2]]
        return out
    raise ValueError(f"no conversion from DXGI format {dxgi_format} to BGRA8")


def bytes_per_pixel(dxgi_format: int) -> int:
    return 8 if dxgi_format == DXGI_FORMAT_R16G16B16A16_FLOAT else 4


def clipped_at_nominal_white(dxgi_format: int, color: DisplayColor) -> bool:
    """True when this capture cannot show anything above 80 nits.

    A 10-bit UNORM surface holding linear scRGB tops out at 1.0, which is 80
    nits: everything brighter, SDR white included whenever it is set above 80,
    comes back clipped. Seen on an Intel Comet Lake desktop, a platform
    Microsoft lists as lacking full Advanced Color support.
    """
    return (color.hdr and dxgi_format == DXGI_FORMAT_R10G10B10A2_UNORM
            and color.sdr_white_scale > 1.0)
