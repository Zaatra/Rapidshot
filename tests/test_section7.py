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
    adapter.sem = adapter.view = adapter.marker_view = None
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


@pytest.mark.parametrize("shape, accepted", [
    (["batch", 3, "height", "width"], True),   # official Ultralytics yolo11n.onnx
    ([1, 3, 640, 640], True),                  # a pinned export
    ([None, 3, None, None], True),             # unnamed symbolic axes
    ([1, 3, 320, 320], False),                 # concrete and wrong
    (["batch", 1, "height", "width"], False),  # wrong channel count
    ([1, 3, 640], False),
    (None, False),
])
def test_inference_accepts_the_published_model_shape(shape, accepted):
    import ai_pipeline
    assert ai_pipeline.accepts_input_shape(shape) is accepted


def test_fp16_tensor_recovers_every_rgb8_level_exactly():
    # Verification compares in RGB8 by undoing the normalisation, which is only
    # sound if FP16 keeps every k/255 distinguishable.
    levels = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
    tensor = contract.normalized_tensor(levels, np)
    recovered = np.rint(tensor[0].astype(np.float32) * 255).transpose(1, 2, 0)
    assert np.array_equal(recovered.astype(np.uint8), levels)


@pytest.mark.parametrize("height, width", [(1600, 2560), (1080, 1920), (37, 53)])
def test_pipeline_resize_stays_within_the_verification_tolerance(height, width):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(height * width)
    bgra = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    deviation = np.abs(contract.pipeline_rgb(bgra, np).astype(np.int16)
                       - contract.canonical_rgb(bgra, np).astype(np.int16))
    assert deviation.max() <= contract.PIPELINE_TOLERANCE_RGB8


def test_symbolic_axes_are_pinned_in_the_session_not_the_file():
    import ai_pipeline
    assert ai_pipeline.symbolic_overrides(["batch", 3, "height", "width"]) == {
        "batch": 1, "height": 640, "width": 640}
    assert ai_pipeline.symbolic_overrides([1, 3, 640, 640]) == {}
    # An unnamed axis cannot be overridden by name; it is left to ORT.
    assert ai_pipeline.symbolic_overrides([None, 3, "h", "w"]) == {"h": 640, "w": 640}


# -- the shared clock and the model pin ----------------------------------


def test_qpc_reports_a_frequency_and_a_counter_that_advances():
    """Pixel age is a difference of two QPC readings taken in different
    processes, so the frequency has to be read rather than assumed -- it is not
    nanoseconds and is not fixed across machines."""
    now, hz = contract.qpc_clock()
    assert hz > 0
    first = now()
    assert now() >= first
    assert isinstance(first, int)


def test_qpc_refuses_a_frequency_it_cannot_read(monkeypatch):
    """Returning 0 would make every later division a crash, or worse a silent
    infinity in a latency figure."""
    class Kernel:
        def __getattr__(self, name):
            def call(pointer):
                pointer._obj.value = 0
                return 1
            call.argtypes = None
            return call

    monkeypatch.setattr(contract.ctypes, "WinDLL", lambda *a, **kw: Kernel(), raising=False)
    with pytest.raises(OSError, match="QueryPerformanceFrequency"):
        contract.qpc_clock()


def test_sha256_matches_the_published_digest(tmp_path):
    """The model is pinned by digest, so this is what decides whether the
    benchmark ran the weights it claims to have run."""
    path = tmp_path / "w.bin"
    path.write_bytes(b"abc")
    assert contract.sha256(path) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_sha256_reads_a_file_larger_than_one_block(tmp_path):
    """Read in 1 MiB blocks, so a model-sized file exercises the loop rather
    than a single read -- and a loop that dropped a block would still return a
    plausible-looking digest."""
    import hashlib

    payload = bytes(range(256)) * 12_000          # ~3 MB, several blocks
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert contract.sha256(path) == hashlib.sha256(payload).hexdigest()


# -- the guard that stops live work on failing hardware -------------------


class FakeLogs:
    def __init__(self):
        self.events = []
        self.directory = Path(".")

    def event(self, name, **fields):
        self.events.append((name, fields))


def guard(monkeypatch, readings):
    """A HealthGuard whose WHEA query returns each reading in turn."""
    values = list(readings)
    monkeypatch.setattr(section7.HealthGuard, "query",
                        staticmethod(lambda: values.pop(0)))
    logs = FakeLogs()
    return section7.HealthGuard(logs), logs


def test_the_baseline_whea_state_is_recorded_at_construction(monkeypatch):
    """Without a baseline there is nothing to compare against, and a machine
    that already had records would look healthy."""
    g, logs = guard(monkeypatch, [{"latest": 5, "count": 5}] * 2)
    assert g.baseline == {"latest": 5, "count": 5}
    assert logs.events[0][0] == "health-baseline"


def test_a_new_whea_record_stops_live_benchmarks(monkeypatch):
    """A run that continues past a machine-check exception produces numbers
    from hardware that has just reported a fault, and they are
    indistinguishable from good ones."""
    g, logs = guard(monkeypatch, [{"latest": 5, "count": 5},
                                  {"latest": 6, "count": 6}])
    with pytest.raises(section7.MotionError, match="WHEA"):
        g.check(force=True)
    assert logs.events[-1][0] == "hardware-error-or-log-change"


def test_an_unchanged_log_lets_the_run_continue(monkeypatch):
    g, _ = guard(monkeypatch, [{"latest": 5, "count": 5}] * 2)
    g.check(force=True)


def test_checks_are_throttled_unless_forced(monkeypatch):
    """The query shells out to PowerShell, which costs far more than a frame.
    Polling it per frame would change what is being measured."""
    values = [{"latest": 5, "count": 5}, {"latest": 9, "count": 9}]
    calls = []

    def query():
        calls.append(1)
        return values.pop(0) if values else {"latest": 9, "count": 9}

    monkeypatch.setattr(section7.HealthGuard, "query", staticmethod(query))
    monkeypatch.setattr(section7.time, "monotonic", lambda: 100.0)
    g = section7.HealthGuard(FakeLogs())
    g.last = 100.0
    g.check()                      # within 5 s of `last`: must not re-query
    assert len(calls) == 1


# -- the display mode, and the source that follows it ---------------------


def test_the_primary_display_mode_is_reported():
    """Read rather than assumed: ENUM_CURRENT_SETTINGS (0xFFFFFFFF) is the
    mode in force, and the benchmark paces its source from it. A stored or
    default value would describe some other machine."""
    mode = section7.display_mode()
    assert mode["width"] > 0 and mode["height"] > 0
    assert mode["refresh_hz"] > 0
    assert set(mode) == {"width", "height", "refresh_hz"}


def test_a_display_query_that_fails_is_not_silently_zero(monkeypatch):
    """Zero would flow into the source's fps argument, which the latency source
    refuses with 'fps outside 1..240' -- an error about the wrong thing, one
    layer away from the cause."""
    class User:
        def __getattr__(self, name):
            def call(device, index, mode):
                return 0
            call.argtypes = None
            return call

    monkeypatch.setattr(section7.ctypes, "WinDLL", lambda *a, **kw: User(), raising=False)
    with pytest.raises(OSError, match="EnumDisplaySettingsW"):
        section7.display_mode()


class FakeSource:
    """Stands in for the built binary: a Path instance refuses attribute
    patching, and `str()` resolves on the type rather than the instance."""

    def __init__(self, exists=True):
        self.exists = exists

    def is_file(self):
        return self.exists

    def __str__(self):
        return "latency_source.exe"


def test_the_source_is_launched_with_the_requested_mode(monkeypatch, tmp_path):
    """Resolution, refresh and workload all reach the binary, because a run
    labelled 2560x1600 at 165 Hz that quietly launched something else is worse
    than no run."""
    commands = []

    class Proc:
        pid = 4242

        def __init__(self, *args, **kwargs):
            commands.append(list(args[0]))
            self.stdin = None

        def poll(self):
            return None

    monkeypatch.setattr(section7.subprocess, "Popen", Proc)
    monkeypatch.setattr(section7, "SOURCE", FakeSource())
    logs = section7.RunLogs(tmp_path)
    args = SimpleNamespace(width=2560, height=1600, motion_fps=165.0,
                           workload="scroll")
    guard = SimpleNamespace(check=lambda force=False: None)
    source = section7.VisualSource(logs, args, guard)
    source.ready = True          # readiness itself is covered by MotionSource
    source.start()
    command = commands[0]
    assert command[1:5] == ["2560", "1600", "165.0", "scroll"]
    assert command[5].endswith("presents.jsonl")


def test_a_silent_source_does_not_hang_the_run(monkeypatch, tmp_path):
    class Proc:
        pid = 4243

        def __init__(self, *args, **kwargs):
            self.stdin = None

        def poll(self):
            return None

    # Unbounded: construction and every `check()` read the clock too, so a
    # fixed list runs out before the deadline is reached.
    ticks = {"n": 0}

    def monotonic():
        ticks["n"] += 1
        return 0.0 if ticks["n"] <= 2 else 1000.0

    monkeypatch.setattr(section7.subprocess, "Popen", Proc)
    monkeypatch.setattr(section7, "SOURCE", FakeSource())
    monkeypatch.setattr(section7.time, "monotonic", monotonic)
    monkeypatch.setattr(section7.time, "sleep", lambda _s: None)
    logs = section7.RunLogs(tmp_path)
    args = SimpleNamespace(width=900, height=700, motion_fps=60.0, workload="static")
    source = section7.VisualSource(logs, args,
                                   SimpleNamespace(check=lambda force=False: None))
    with pytest.raises(section7.MotionError, match="readiness timeout"):
        source.start()


def test_the_source_defers_to_the_health_guard(monkeypatch, tmp_path):
    """`VisualSource.check` consults the guard before anything else, so a WHEA
    record that appears mid-run stops the source rather than being measured
    through."""
    logs = section7.RunLogs(tmp_path)
    args = SimpleNamespace(width=900, height=700, motion_fps=60.0, workload="static")
    calls = []

    def refuse(force=False):
        calls.append("guard")
        raise section7.MotionError("WHEA log changed")

    source = section7.VisualSource(logs, args, SimpleNamespace(check=refuse))
    with pytest.raises(section7.MotionError, match="WHEA"):
        source.check()
    assert calls == ["guard"]


def test_a_local_source_build_wins_over_the_wheel(monkeypatch, tmp_path):
    built = tmp_path / "latency_source.exe"
    built.write_bytes(b"MZ")
    wheel = tmp_path / "wheel" / "latency_source.exe"
    monkeypatch.setitem(sys.modules, "rapidshot_native",
                        SimpleNamespace(latency_source_path=lambda: str(wheel)))
    assert section7.find_source(built) == built


def test_the_wheel_source_is_used_when_nothing_was_built(monkeypatch, tmp_path):
    wheel = tmp_path / "latency_source.exe"
    monkeypatch.setitem(sys.modules, "rapidshot_native",
                        SimpleNamespace(latency_source_path=lambda: str(wheel)))
    assert section7.find_source(tmp_path / "missing.exe") == wheel


@pytest.mark.parametrize("module", [
    None,                                            # rapidshot-native not installed
    SimpleNamespace(),                               # 0.2.0: no latency_source_path
    SimpleNamespace(latency_source_path=lambda: (_ for _ in ()).throw(FileNotFoundError())),
])
def test_without_any_source_the_error_names_the_local_build(monkeypatch, tmp_path, module):
    monkeypatch.setitem(sys.modules, "rapidshot_native", module)
    missing = tmp_path / "missing.exe"
    assert section7.find_source(missing) == missing
