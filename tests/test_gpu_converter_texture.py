"""GpuConverter(pixel_format="nv12", output="texture"): the encoder's NV12.

The texture output is written by a different kernel from the buffer output,
so its bytes are checked against the buffer's on the same held frame -- the
exact comparison `nv12_routes` makes in Rust, here through the Python API. The
buffer path is itself pinned to an exact CPU reference in
`test_gpu_converter_yuv.py`, so equal to it is exact.

Measured on an Intel UHD at 2560x1600 when this was added (2026-11-17):
`process` 2.9 against 7.5 ms p50 on live frames, readback 6.5 against 3.0 ms.
"""

import numpy as np
import pytest

rapidshot = pytest.importorskip("rapidshot")
from rapidshot import native  # noqa: E402

pytestmark = pytest.mark.skipif(
    not native.is_available() or rapidshot.GpuConverter is None,
    reason="native extension not built")


@pytest.fixture
def held():
    try:
        camera = rapidshot.create(output_color="BGRA")
    except Exception as exc:  # noqa: BLE001 - no capture on this machine
        pytest.skip(f"capture unavailable: {type(exc).__name__}: {exc}")
    frame = None
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        camera.release()
        pytest.skip("no frame captured -- the screen must be changing")
    with frame:
        yield camera, frame
    camera.release()


def _texture_converter(frame, size, **kw):
    try:
        return rapidshot.GpuConverter(frame, size, pixel_format="nv12",
                                      output="texture", **kw)
    except RuntimeError as exc:
        if "typed UAV" in str(exc) or "NV12" in str(exc):
            pytest.skip(f"this GPU cannot write NV12 textures: {exc}")
        raise


@pytest.mark.parametrize("matrix, full_range", [("bt709", False), ("bt601", True)])
def test_texture_output_holds_the_buffer_outputs_bytes(held, matrix, full_range):
    camera, frame = held
    size = (frame.width - frame.width % 2, frame.height - frame.height % 2)
    buffer = rapidshot.GpuConverter(frame, size, pixel_format="nv12",
                                    matrix=matrix, full_range=full_range)
    texture = _texture_converter(frame, size, matrix=matrix, full_range=full_range)
    expected = buffer.process(frame).numpy().copy()
    got = texture.process(frame).numpy()
    assert got.shape == expected.shape == (size[1] * 3 // 2, size[0])
    assert np.array_equal(got, expected), (
        f"{np.count_nonzero(got != expected)} of {got.size} NV12 bytes differ")


def test_texture_output_scaled_matches_the_buffer_output(held):
    camera, frame = held
    texture = _texture_converter(frame, (1280, 720))
    buffer = rapidshot.GpuConverter(frame, (1280, 720), pixel_format="nv12")
    assert np.array_equal(texture.process(frame).numpy(), buffer.process(frame).numpy())


def test_texture_output_describes_itself(held):
    _camera, frame = held
    texture = _texture_converter(frame, (640, 360))
    texture.process(frame)
    assert texture.pixel_format == "nv12"
    assert texture.output == "texture"
    assert texture.output_texture_address != 0
    buffer = rapidshot.GpuConverter(frame, (640, 360), pixel_format="nv12")
    assert buffer.output == "buffer"
    with pytest.raises(RuntimeError, match="output='texture'"):
        buffer.output_texture_address


def test_cuda_exports_refuse_a_texture(held):
    _camera, frame = held
    tensor = _texture_converter(frame, (640, 360)).process(frame)
    for export in (tensor.to_cupy, tensor.to_dlpack, tensor.to_torch):
        with pytest.raises(RuntimeError, match="linear array"):
            export()


@pytest.mark.parametrize("kwargs", [
    dict(output="texture"),
    dict(output="texture", pixel_format="p010"),
    dict(output="texture", dtype="float16"),
    dict(output="disk", pixel_format="nv12"),
])
def test_texture_output_is_refused_where_it_does_not_apply(held, kwargs):
    _camera, frame = held
    with pytest.raises(ValueError, match="output"):
        rapidshot.GpuConverter(frame, (640, 360), **kwargs)
