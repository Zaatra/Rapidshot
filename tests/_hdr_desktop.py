"""Skips for live tests whose check only holds on an SDR desktop.

With Windows HDR on, the desktop duplicates as linear scRGB (FP16, or 10-bit
``R10G10B10A2``) instead of 8-bit ``B8G8R8A8_UNORM`` sRGB. A live test whose
reference reads that surface as 8-bit bytes, or which drives a path that
refuses scRGB by design, then fails on its reference or on the refusal and
says nothing about the code it is there to check. Each such test calls
:func:`skip_on_an_hdr_desktop` with one of the reasons below, so an HDR run
reports what was not checked instead of a wall of failures.

What *is* checked on an HDR desktop: ``tests/test_gpu_converter_hdr.py``
(the converter's sRGB against ``grab()``), the live refusal at the end of
``tests/test_gpu_converter_yuv.py``, and everything that passes unchanged.
A test is skipped here only for one of these reasons, never because it fails.
"""
import pytest

#: GpuPreprocessor (D3D11) and GpuPreprocessor12 predate HDR capture and
#: accept 8-bit BGRA only; GpuConverter replaces them for every other format.
LEGACY_PREPROCESSOR = ("GpuPreprocessor and GpuPreprocessor12 accept only "
                       "B8G8R8A8_UNORM, by design")
#: The reference is the surface as 8-bit sRGB bytes -- a BGRA8 staging copy,
#: or bytes compared exactly with float output -- which an HDR surface is not.
BGRA8_REFERENCE = "the reference is the surface as 8-bit sRGB bytes, which an HDR surface is not"
#: GpuConverter filters an HDR frame in linear light before the tone map, as
#: documented; a reference that filters the sRGB bytes differs at every edge.
LINEAR_LIGHT_FILTERING = ("the converter filters an HDR frame in linear light; "
                          "the reference filters sRGB bytes")
#: NV12/P010 refuse linear light unless given a tone map; see the live
#: refusal test in test_gpu_converter_yuv.py.
YUV_WITHOUT_TONE_MAP = "NV12/P010 without a tone map refuse an HDR frame, by design"


def is_hdr(frame) -> bool:
    """Whether ``frame`` holds linear scRGB from an HDR desktop."""
    return getattr(frame, "color_space", "srgb") == "scrgb"


def skip_on_an_hdr_desktop(frame, why: str, *release) -> None:
    """Skip the calling test when ``frame`` is HDR, naming ``why``.

    ``release`` are frames or cameras to release first: a skip raised while a
    frame is held leaves the duplicator unable to acquire the next one, and
    every later live test in the session then fails to start.
    """
    if not is_hdr(frame):
        return
    reason = f"HDR desktop ({frame.pixel_format}): {why}"
    for thing in release:
        thing.release()
    pytest.skip(reason)
