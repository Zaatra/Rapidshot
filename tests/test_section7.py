"""Headless section 7 tests. Never construct capture, Tk, CUDA or ORT devices."""
import io
import json
from pathlib import Path
import sys
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import benchmark_contract as contract
import section7
import section7_adapters
import agent_pipeline


def marker(frame_id):
    bits = contract.MAGIC | frame_id << 8 | contract.checksum(frame_id) << 40
    image = np.zeros((16, 384, 4), dtype=np.uint8)
    for i in range(48):
        image[:, i*8:(i+1)*8, :3] = 255 if (bits >> i) & 1 else 0
    return image


@pytest.mark.parametrize("frame_id", [0, 1, 255, 256, 65537, 0xFFFFFFFF])
def test_marker_round_trip(frame_id):
    assert contract.decode_marker(marker(frame_id), np) == frame_id


def test_marker_rejects_corruption():
    image = marker(123)
    image[:, 80:88, :3] ^= 255
    assert contract.decode_marker(image, np) is None
    image = marker(123)
    image[8, 4, :3] = [255, 0, 0]
    assert contract.decode_marker(image, np) is None


@pytest.mark.parametrize("shape,size", [((7, 11), 5), ((1, 1), 3), ((2, 3), 7), ((13, 3), 4)])
def test_bilinear_matches_independent_fraction_reference(shape, size):
    source = np.random.default_rng(17).integers(0, 256, (*shape, 4), dtype=np.uint8)
    actual = contract.canonical_rgb(source, np, size)
    expected = np.empty((size, size, 3), np.uint8)
    for y in range(size):
        fy = max(Fraction((2*y+1)*shape[0]-size, 2*size), 0)
        y0, wy = int(fy), fy-int(fy)
        y1 = min(y0+1, shape[0]-1)
        for x in range(size):
            fx = max(Fraction((2*x+1)*shape[1]-size, 2*size), 0)
            x0, wx = int(fx), fx-int(fx)
            x1 = min(x0+1, shape[1]-1)
            for k in range(3):
                channel = 2-k
                a,b,c,d = [int(source[yy,xx,channel]) for yy,xx in ((y0,x0),(y0,x1),(y1,x0),(y1,x1))]
                v = (a*(1-wx)+b*wx)*(1-wy)+(c*(1-wx)+d*wx)*wy
                expected[y,x,k] = int(v+Fraction(1,2))
    np.testing.assert_array_equal(actual, expected)
    tensor = contract.normalized_tensor(actual, np)
    assert tensor.dtype == np.float16
    assert tensor.shape == (1,3,size,size)
    assert np.isfinite(tensor).all()


def test_present_log_handles_partial_records(tmp_path):
    path = tmp_path / "presents.jsonl"
    line = json.dumps({"event":"present", "id":7, "qpc_before":100, "qpc_after":120})
    path.write_text(line[:15])
    reader = contract.PresentLog(path, 1000)
    try:
        assert reader.age(7, 130) is None
        with path.open("a") as stream:
            stream.write(line[15:]+"\n")
        assert reader.age(7, 130) == 30
        with pytest.raises(ValueError):
            reader.age(7, 99)
    finally:
        reader.close()


@pytest.mark.parametrize("row", [{}, {"path":"wrong"}, {"path":"mss", "verified":False}])
def test_worker_result_rejects_invalid_rows(row):
    assert "error" in section7.parse_result("mss", True, 0, json.dumps(row), "")


def test_worker_nonzero_exit_cannot_report_success():
    row = section7.parse_result("mss", True, 7, '{"path":"mss","verified":true}', "")
    assert "error" in row


def test_cleanup_does_not_unmap_after_failed_sync():
    calls = []
    adapter = object.__new__(section7_adapters.Adapter)
    adapter.closed = False
    adapter.sync = lambda: (_ for _ in ()).throw(RuntimeError("device sync failed"))
    adapter.sem = adapter.view = adapter.cam = SimpleNamespace(close=lambda: calls.append("close"))
    with pytest.raises(RuntimeError, match="device sync failed"):
        adapter.close()
    assert calls == []


def test_cleanup_is_idempotent():
    calls = []
    adapter = object.__new__(section7_adapters.Adapter)
    adapter.closed = False
    adapter.sync = lambda: calls.append("sync")
    adapter.sem = adapter.view = None
    adapter.cam = SimpleNamespace(release=lambda: calls.append("release"))
    adapter.close()
    adapter.close()
    assert calls == ["sync", "release"]


def test_png_is_api_ready_and_lossless():
    pytest.importorskip("PIL")
    import base64
    from PIL import Image
    rgb = np.random.default_rng(2).integers(0,256,(19,23,3),dtype=np.uint8)
    value, size = agent_pipeline.encode(rgb)
    assert value.startswith("data:image/png;base64,")
    payload = base64.b64decode(value.split(",",1)[1])
    assert len(payload) == size
    np.testing.assert_array_equal(np.asarray(Image.open(io.BytesIO(payload))), rgb)


def test_inference_requires_pinned_model_before_launch():
    with pytest.raises(SystemExit) as error:
        section7.main("inference", [])
    assert error.value.code == 2


def test_dll_directory_handles_are_retained(tmp_path, monkeypatch):
    import ai_pipeline
    target = tmp_path / "Lib/site-packages/nvidia/cublas/bin"
    target.mkdir(parents=True)
    (target / "example.dll").touch()
    handle = object()
    monkeypatch.setattr(ai_pipeline.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(ai_pipeline.sys, "base_prefix", str(tmp_path))
    monkeypatch.setattr(ai_pipeline.os, "add_dll_directory", lambda path: handle, raising=False)
    monkeypatch.setattr(ai_pipeline, "_DLL_HANDLES", [])
    monkeypatch.setenv("PATH", "")
    assert ai_pipeline.enable_cuda_dlls() == [str(target)]
    assert ai_pipeline._DLL_HANDLES == [handle]
