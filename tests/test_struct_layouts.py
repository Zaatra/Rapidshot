"""Every ctypes structure in rapidshot/_libs against the Windows SDK layout.

A mis-declared structure does not raise. DXGI writes the size the SDK says into
memory ctypes sized differently, and the result is a wrong field, a truncated
string, or a write past the end of the buffer -- found, if at all, a long way
from the declaration. These are the x64 sizes and offsets from the SDK headers
(dxgi.h, dxgi1_2.h, dxgi1_6.h, d3d11.h, winuser.h), written out independently
of the declarations they check. All of them matched when this was added.
"""
import ctypes

import pytest

pytest.importorskip("comtypes", reason="COM is Windows-only")

if ctypes.sizeof(ctypes.c_void_p) != 8:          # pragma: no cover - 32-bit Python
    pytest.skip("layouts below are the x64 ones", allow_module_level=True)

import rapidshot._libs.d3d11 as d3d11  # noqa: E402
import rapidshot._libs.dxgi as dxgi  # noqa: E402
import rapidshot._libs.user32 as user32  # noqa: E402

# (module, structure, sdk size, {field: sdk offset})
LAYOUTS = [
    (d3d11, "DXGI_SAMPLE_DESC", 8, {"Count": 0, "Quality": 4}),
    (d3d11, "D3D11_BOX", 24, {"left": 0, "top": 4, "front": 8, "right": 12, "bottom": 16, "back": 20}),
    (d3d11, "D3D11_TEXTURE2D_DESC", 44, {
        "Width": 0, "Height": 4, "MipLevels": 8, "ArraySize": 12, "Format": 16,
        "SampleDesc": 20, "Usage": 28, "BindFlags": 32, "CPUAccessFlags": 36, "MiscFlags": 40}),
    (dxgi, "LUID", 8, {"LowPart": 0, "HighPart": 4}),
    (dxgi, "DXGI_ADAPTER_DESC1", 312, {
        "Description": 0, "VendorId": 256, "DeviceId": 260, "SubSysId": 264, "Revision": 268,
        "DedicatedVideoMemory": 272, "DedicatedSystemMemory": 280, "SharedSystemMemory": 288,
        "AdapterLuid": 296, "Flags": 304}),
    (dxgi, "DXGI_ADAPTER_DESC2", 320, {
        "AdapterLuid": 296, "Flags": 304,
        "GraphicsPreemptionGranularity": 308, "ComputePreemptionGranularity": 312}),
    (dxgi, "DXGI_ADAPTER_DESC3", 320, {
        "AdapterLuid": 296, "Flags": 304,
        "GraphicsPreemptionGranularity": 308, "ComputePreemptionGranularity": 312}),
    (dxgi, "DXGI_OUTPUT_DESC", 96, {
        "DeviceName": 0, "DesktopCoordinates": 64, "AttachedToDesktop": 80, "Rotation": 84, "Monitor": 88}),
    (dxgi, "DXGI_OUTDUPL_POINTER_POSITION", 12, {"Position": 0, "Visible": 8}),
    (dxgi, "DXGI_OUTDUPL_POINTER_SHAPE_INFO", 24, {
        "Type": 0, "Width": 4, "Height": 8, "Pitch": 12, "HotSpot": 16}),
    (dxgi, "DXGI_OUTDUPL_FRAME_INFO", 48, {
        "LastPresentTime": 0, "LastMouseUpdateTime": 8, "AccumulatedFrames": 16,
        "RectsCoalesced": 20, "ProtectedContentMaskedOut": 24, "PointerPosition": 28,
        "TotalMetadataBufferSize": 40, "PointerShapeBufferSize": 44}),
    (dxgi, "POINT", 8, {"x": 0, "y": 4}),
    (dxgi, "DXGI_RATIONAL", 8, {"Numerator": 0, "Denominator": 4}),
    (dxgi, "DXGI_MODE_DESC", 28, {
        "Width": 0, "Height": 4, "RefreshRate": 8, "Format": 16, "ScanlineOrdering": 20, "Scaling": 24}),
    (dxgi, "DXGI_OUTDUPL_DESC", 36, {"ModeDesc": 0, "Rotation": 28, "DesktopImageInSystemMemory": 32}),
    (dxgi, "DXGI_OUTDUPL_MOVE_RECT", 24, {"SourcePoint": 0, "DestinationRect": 8}),
    (dxgi, "DXGI_MAPPED_RECT", 16, {"Pitch": 0, "pBits": 8}),
    (user32, "DISPLAY_DEVICE", 840, {
        "cb": 0, "DeviceName": 4, "DeviceString": 68, "StateFlags": 324, "DeviceID": 328, "DeviceKey": 584}),
    (user32, "MONITORINFOEXW", 104, {
        "cbSize": 0, "rcMonitor": 4, "rcWork": 20, "dwFlags": 36, "szDevice": 40}),
]


@pytest.mark.parametrize("module,name,size,offsets", LAYOUTS,
                         ids=[entry[1] for entry in LAYOUTS])
def test_layout_matches_the_sdk(module, name, size, offsets):
    struct = getattr(module, name)

    assert ctypes.sizeof(struct) == size
    for field, offset in offsets.items():
        assert getattr(struct, field).offset == offset, field


def test_every_structure_is_checked():
    """A structure added later without a layout entry fails here, not in DXGI."""
    import inspect

    declared = {
        (module.__name__, name)
        for module in (d3d11, dxgi, user32)
        for name, obj in vars(module).items()
        if inspect.isclass(obj)
        and issubclass(obj, (ctypes.Structure, ctypes.Union))
        and obj.__module__ == module.__name__
    }
    checked = {(module.__name__, name) for module, name, _, _ in LAYOUTS}

    assert declared == checked
