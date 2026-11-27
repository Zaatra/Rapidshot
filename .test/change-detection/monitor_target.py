"""Resolve the benchmark monitor by its actual EDID name."""

import ctypes as ct
from ctypes import wintypes as wt
import winreg


class DisplayDevice(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD), ("name", wt.WCHAR * 32), ("label", wt.WCHAR * 128),
        ("flags", wt.DWORD), ("hardware_id", wt.WCHAR * 128), ("key", wt.WCHAR * 128),
    ]


class DisplayMode(ct.Structure):
    _fields_ = [
        ("name", wt.WCHAR * 32), ("spec", wt.WORD), ("driver", wt.WORD),
        ("size", wt.WORD), ("extra", wt.WORD), ("fields", wt.DWORD),
        ("position", wt.POINT), ("rotation", wt.DWORD), ("fixed", wt.DWORD),
        ("print_fields", wt.SHORT * 5), ("form", wt.WCHAR * 32), ("pixels", wt.WORD),
        ("bpp", wt.DWORD), ("width", wt.DWORD), ("height", wt.DWORD),
        ("flags", wt.DWORD), ("refresh_hz", wt.DWORD), ("tail", wt.DWORD * 8),
    ]


def edid_label(device_interface):
    parts = device_interface.split("#")
    if len(parts) < 3:
        return None
    path = "SYSTEM\\CurrentControlSet\\Enum\\DISPLAY\\{}\\{}\\Device Parameters".format(parts[1], parts[2])
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
            edid = winreg.QueryValueEx(key, "EDID")[0]
    except OSError:
        return None
    for offset in (54, 72, 90, 108):
        if edid[offset:offset + 5] == b"\x00\x00\x00\xfc\x00":
            return edid[offset + 5:offset + 18].decode("ascii", errors="replace").strip()
    return None


def windows_displays():
    user = ct.WinDLL("user32", use_last_error=True)
    user.EnumDisplayDevicesW.argtypes = [wt.LPCWSTR, wt.DWORD, ct.POINTER(DisplayDevice), wt.DWORD]
    user.EnumDisplaySettingsW.argtypes = [wt.LPCWSTR, wt.DWORD, ct.POINTER(DisplayMode)]
    index = 0
    result = []
    while True:
        display = DisplayDevice(size=ct.sizeof(DisplayDevice))
        if not user.EnumDisplayDevicesW(None, index, ct.byref(display), 0):
            break
        index += 1
        if not display.flags & 1:
            continue
        monitor = DisplayDevice(size=ct.sizeof(DisplayDevice))
        mode = DisplayMode(size=ct.sizeof(DisplayMode))
        if not user.EnumDisplayDevicesW(display.name, 0, ct.byref(monitor), 1):
            continue
        if not user.EnumDisplaySettingsW(display.name, 0xffffffff, ct.byref(mode)):
            continue
        left, top = mode.position.x, mode.position.y
        result.append({"output": display.name, "monitor": edid_label(monitor.hardware_id) or monitor.label,
                       "physical_refresh_hz": mode.refresh_hz, "primary": bool(display.flags & 4),
                       "bounds": [left, top, left + mode.width, top + mode.height]})
    return result


def select_monitor(label="HP 324pe"):
    from rapidshot import get_factory

    displays = windows_displays()
    matches = [display for display in displays if display["monitor"].casefold() == label.casefold()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {label!r} monitor, found {matches!r} among {displays!r}")
    selected = matches[0]
    factory = get_factory()
    for device, outputs in enumerate(factory.outputs):
        for index, output in enumerate(outputs):
            if (output.devicename == selected["output"] and output.attached_to_desktop
                    and output.resolution == (1920, 1080) and output.rotation_angle == 0):
                return {**selected, "device_index": device, "output_index": index,
                        "adapter": str(factory.devices[device].desc.Description)}
    raise RuntimeError(f"{label} was not an attached unrotated DXGI 1080p output")


if __name__ == "__main__":
    import json
    print(json.dumps(select_monitor()))
