import ctypes
import logging
from typing import Tuple
from dataclasses import dataclass
from rapidshot._libs.d3d11 import *
from rapidshot._libs.dxgi import *

logger = logging.getLogger(__name__)

# DPI awareness is a process-wide setting, and Windows lets it be set once.
# RapidShot needs it because a DPI-unaware process is fed virtualised desktop
# coordinates: on a scaled display `DXGI_OUTPUT_DESC.DesktopCoordinates` would
# then disagree with the size of the texture Desktop Duplication actually hands
# back, and every region would map to the wrong pixels.
#
# But claiming it is the host application's decision, not a screen-capture
# library's, and this used to be attempted on every Output construction with
# the result thrown away. So: try once, and say what happened rather than
# silently either winning or losing. A Python process is already per-monitor
# aware from python.exe's manifest, which is why this has always been a no-op
# there -- it matters for an embedded interpreter in a host that set nothing.
_PROCESS_DPI_PER_MONITOR = 2
_E_ACCESSDENIED = -2147024891
_dpi_awareness_attempted = False


def _ensure_process_dpi_awareness() -> None:
    """Ask for per-monitor DPI awareness once per process, and report."""
    global _dpi_awareness_attempted
    if _dpi_awareness_attempted:
        return
    _dpi_awareness_attempted = True

    try:
        shcore = ctypes.windll.shcore
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long
        hresult = shcore.SetProcessDpiAwareness(_PROCESS_DPI_PER_MONITOR)
    except Exception as e:  # pragma: no cover - shcore missing (pre-8.1)
        logger.debug(f"Could not set process DPI awareness: {e}")
        return

    if hresult == 0:
        logger.debug("Process DPI awareness set to per-monitor.")
        return

    if hresult == _E_ACCESSDENIED:
        # Already set -- by python.exe's manifest, or by the host application.
        # Either way it cannot be changed, and if the host chose something
        # other than per-monitor, capture regions on a scaled secondary display
        # may not line up.
        current = ctypes.c_int(-1)
        try:
            shcore.GetProcessDpiAwareness(None, ctypes.byref(current))
        except Exception:  # pragma: no cover
            pass
        if current.value == _PROCESS_DPI_PER_MONITOR:
            logger.debug("Process is already per-monitor DPI aware.")
        else:
            logger.warning(
                f"Process DPI awareness is already set to {current.value} and "
                "cannot be changed; RapidShot wants per-monitor (2). On a "
                "display with scaling, desktop coordinates may not match the "
                "captured texture.")
        return

    logger.debug(f"SetProcessDpiAwareness returned {hresult:#x}.")


@dataclass
class Output:
    output: ctypes.POINTER(IDXGIOutput1)
    rotation_mapping: tuple = (0, 0, 90, 180, 270)
    desc: DXGI_OUTPUT_DESC = None

    def __post_init__(self):
        _ensure_process_dpi_awareness()
        self.desc = DXGI_OUTPUT_DESC()
        self.update_desc()

    def update_desc(self):
        if self.desc is None:
            self.desc = DXGI_OUTPUT_DESC()
        self.output.GetDesc(ctypes.byref(self.desc))

    @property
    def hmonitor(self) -> wintypes.HMONITOR:
        return self.desc.Monitor

    @property
    def devicename(self) -> str:
        return self.desc.DeviceName

    @property
    def resolution(self) -> Tuple[int, int]:
        return (
            (self.desc.DesktopCoordinates.right - self.desc.DesktopCoordinates.left),
            (self.desc.DesktopCoordinates.bottom - self.desc.DesktopCoordinates.top),
        )

    @property
    def surface_size(self) -> Tuple[int, int]:
        if self.rotation_angle in (90, 270):
            return self.resolution[1], self.resolution[0]
        else:
            return self.resolution

    @property
    def attached_to_desktop(self) -> bool:
        return bool(self.desc.AttachedToDesktop)

    @property
    def rotation_angle(self) -> int:
        return self.rotation_mapping[self.desc.Rotation]

    def __repr__(self) -> str:
        return "<{} Name:{} Resolution:{} Rotation:{}>".format(
            self.__class__.__name__,
            self.devicename,
            self.resolution,
            self.rotation_angle,
        )