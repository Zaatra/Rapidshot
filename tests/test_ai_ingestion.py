"""Headless benchmark regression tests: no capture, CUDA, or Tk windows."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import ai_ingestion as bench


class FakeGPU:
    def __init__(self, events):
        self.cuda = SimpleNamespace(runtime=SimpleNamespace(
            deviceSynchronize=lambda: events.append("sync"), memGetInfo=lambda: (10, 20)))

    def __getattr__(self, name):
        return getattr(np, name)

    def asnumpy(self, array):
        return np.array(array, copy=True)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def harness(monkeypatch):
    events, clock = [], Clock()
    gpu = FakeGPU(events)
    monkeypatch.setitem(sys.modules, "cupy", gpu)
    monkeypatch.setitem(sys.modules, "psutil", None)
    monkeypatch.setattr(bench.time, "perf_counter", clock)
    monkeypatch.setattr(bench, "stage", lambda *a, **kw: None)
    monkeypatch.setattr(bench, "OUT", 8)
    monkeypatch.setattr(bench, "TARGET_SHAPE", (1, 3, 8, 8))

    def install(produce, close=None):
        close = close or (lambda: events.append("close"))
        monkeypatch.setitem(bench.ADAPTERS, "mss", lambda *a, **kw: (produce, close, {}))

    return SimpleNamespace(events=events, clock=clock, gpu=gpu, install=install)


def patterned(height, width):
    y, x = np.indices((height, width))
    image = np.empty((height, width, 4), dtype=np.uint8)
    image[..., 0] = x * 19 % 256
    image[..., 1] = y * 17 % 256
    image[..., 2] = (x * 31 + y * 7) % 256
    image[..., 3] = 255
    return image


@pytest.mark.parametrize("shape", [(25, 40), (3, 5), (1, 1), (17, 13), (16, 16)])
def test_gpu_resize_matches_full_image_reference(harness, shape):
    image = patterned(*shape)
    actual = bench._gpu_to_tensor(image, harness.gpu)
    reference = bench.reference_tensor(image, np)
    assert actual.shape == bench.TARGET_SHAPE
    assert actual.dtype == np.dtype(bench.TARGET_DTYPE)
    np.testing.assert_allclose(actual, reference, atol=1 / 255, rtol=0)


@pytest.mark.parametrize("dtype", bench.DTYPES)
def test_target_dtype_is_honoured_and_stays_within_tolerance(harness, monkeypatch, dtype):
    """Both settings must produce their own dtype and still match the float64
    reference. FP16 quantises to about 0.0005 near 1.0, well inside the 1/255
    the comparison allows -- so switching the target cannot quietly start
    measuring a different picture."""
    monkeypatch.setattr(bench, "TARGET_DTYPE", dtype)
    image = patterned(21, 29)
    actual = bench._gpu_to_tensor(image, harness.gpu)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(actual, bench.reference_tensor(image, np),
                               atol=1 / 255, rtol=0)
    assert bench.validate_tensor(actual, image, np)["verified"] is True


def test_resize_includes_bottom_of_1600_row_screen(harness, monkeypatch):
    monkeypatch.setattr(bench, "OUT", 640)
    monkeypatch.setattr(bench, "TARGET_SHAPE", (1, 3, 640, 640))
    image = np.zeros((1600, 1, 4), np.uint8)
    image[1300:, :, :3] = 255
    result = bench._gpu_to_tensor(image, harness.gpu)
    assert np.all(result[:, :, -1, :] == 1)
    assert np.all(result[:, :, 0, :] == 0)


@pytest.mark.parametrize("corrupt", ["dtype", "black", "channels", "nan", "shape", "range"])
def test_verification_rejects_incorrect_tensor(harness, corrupt):
    image = patterned(13, 17)
    result = bench.reference_tensor(image, np).astype(bench.TARGET_DTYPE)
    if corrupt == "dtype":
        result = result.astype(np.float64)
    elif corrupt == "black":
        result[:] = 0
    elif corrupt == "channels":
        result = result[:, ::-1]
    elif corrupt == "nan":
        result.flat[0] = np.nan
    elif corrupt == "shape":
        result = result[:, :, :-1]
    else:
        result.flat[0] = 2
    validation = bench.validate_tensor(result, image, np)
    assert validation["verified"] is False
    assert "error" in validation
    json.dumps(validation, allow_nan=False)


def test_verification_accepts_same_frame_bilinear(harness):
    image = patterned(13, 17)
    harness.install(lambda: bench.Sample(bench._gpu_to_tensor(image, harness.gpu), image))
    result = bench.verify_path("mss")
    assert result["verified"] is True
    assert harness.events[-2:] == ["sync", "close"]
    assert harness.events.count("close") == 1


def test_warmup_synchronizes_before_next_produce(harness):
    def produce():
        if harness.events:
            assert harness.events[-1] == "sync"
        harness.events.append("produce")
        harness.clock.now += 0.6
        return bench.Sample(None)

    harness.install(produce)
    result = bench.run_path("mss", 1, 3)
    assert "error" not in result
    assert harness.events[:6] == ["produce", "sync"] * 3
    assert result["frames"] == 2
    assert result["elapsed_seconds"] == pytest.approx(1.2)
    assert result["fps"] == 1.7
    assert harness.events.count("close") == 1


def test_cpu_percent_uses_actual_duration(harness, monkeypatch):
    proc = SimpleNamespace(cpu_times=lambda: SimpleNamespace(user=harness.clock.now, system=0),
                           memory_info=lambda: SimpleNamespace(rss=1024))
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(Process=lambda: proc))

    def produce():
        harness.clock.now += 0.6
        return bench.Sample(None)

    harness.install(produce)
    result = bench.run_path("mss", 1, 0)
    assert result["cpu_percent"] == 100.0
    assert result["elapsed_seconds"] == pytest.approx(1.2)


@pytest.mark.parametrize("mode", ["verify", "measure", "empty"])
def test_failure_cleans_up_exactly_once(harness, mode):
    def produce():
        harness.clock.now += 21
        if mode == "empty":
            return None
        raise RuntimeError("capture failed")

    harness.install(produce)
    result = bench.verify_path("mss") if mode == "verify" else bench.run_path("mss", 1, 1)
    assert "error" in result
    assert harness.events.count("close") == 1
    assert harness.events[-2:] == ["sync", "close"]


def test_cleanup_failure_is_reported(harness):
    image = patterned(3, 5)
    def close():
        raise RuntimeError("close failed")
    harness.install(lambda: bench.Sample(
        bench.reference_tensor(image, np).astype(bench.TARGET_DTYPE), image), close)
    result = bench.verify_path("mss")
    assert result["error"] == "adapter cleanup failed"
    assert "close failed" in result["cleanup_errors"][0]


def transfer(**overrides):
    data = dict(width=2, height=3, row_pitch=16, total_bytes=64,
                bytes_per_pixel=4, dxgi_format=87)
    data.update(overrides)
    return SimpleNamespace(**data)


def test_pitched_image_ignores_padding_and_allocation_tail():
    raw = np.arange(64, dtype=np.uint8)
    image = bench._pitched_bgra(raw, transfer())
    assert image.shape == (3, 2, 4)
    np.testing.assert_array_equal(image[2].ravel(), raw[32:40])


@pytest.mark.parametrize("fields", [{"dxgi_format": 28}, {"dxgi_format": 24},
                                    {"dxgi_format": 10, "bytes_per_pixel": 8},
                                    {"row_pitch": 4}, {"height": 0}, {"total_bytes": 16}])
def test_transfer_rejects_wrong_format_or_layout(fields):
    with pytest.raises(ValueError):
        bench._validate_transfer(transfer(**fields))


def test_transfer_rejects_truncated_reference():
    with pytest.raises(ValueError, match="shorter"):
        bench._pitched_bgra(np.zeros(32, np.uint8), transfer())


def success_row():
    return dict(path="mss", frames=1, fps=1, ms_p50=1, ms_p95=1, ms_p99=1,
                elapsed_seconds=1)


@pytest.mark.parametrize("status", [1, -1])
def test_worker_failure_cannot_be_hidden_by_json(status):
    result = bench._worker_result("mss", False, status, json.dumps(success_row()), "failure")
    assert "error" in result


@pytest.mark.parametrize("output", ["", "not json", "[]", '{"path":"other"}',
                                     '{"path":"mss","fps":1}'])
def test_invalid_worker_result_fails(output):
    assert "error" in bench._worker_result("mss", False, 0, output, "")


class Process:
    pid = 123
    def __init__(self, returncode=None):
        self.returncode = returncode
        self.stdin = io.BytesIO()
        self.terminated = self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 1

    def kill(self):
        self.killed = True
        self.returncode = 1

    def wait(self, timeout):
        if self.returncode is None:
            raise bench.subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


@pytest.fixture
def process_env(monkeypatch, tmp_path):
    clock, process = Clock(), Process()
    monkeypatch.setattr(bench.time, "monotonic", clock)
    monkeypatch.setattr(bench.time, "sleep", clock.sleep)
    monkeypatch.setattr(bench.subprocess, "Popen", lambda *a, **kw: process)
    return SimpleNamespace(clock=clock, process=process, logs=bench.RunLogs(tmp_path))


def test_worker_timeout_is_error_row_and_child_is_stopped(process_env):
    result = bench.spawn("mss", 1, 1, False, logs=process_env.logs)
    assert "time limit" in result["error"]
    assert process_env.process.terminated
    assert Path(result["stderr_log"]).exists()


def test_motion_failure_during_worker_stops_worker(process_env):
    checks = []
    def check():
        checks.append(1)
        if len(checks) > 1:
            raise bench.MotionError("animation stopped")
    result = bench.spawn("mss", 1, 1, False, logs=process_env.logs,
                         motion=SimpleNamespace(check=check))
    assert result["motion_failed"] is True
    assert process_env.process.terminated


def test_dead_motion_never_becomes_ready(process_env):
    process_env.process.returncode = 1
    motion = bench.MotionSource(process_env.logs)
    try:
        with pytest.raises(bench.MotionError, match="exited"):
            motion.start()
    finally:
        motion.close()
    assert motion.stdout.closed and motion.stderr.closed


def test_motion_readiness_timeout_cleans_up(process_env):
    motion = bench.MotionSource(process_env.logs, startup_timeout=0.1)
    try:
        with pytest.raises(bench.MotionError, match="ready"):
            motion.start()
    finally:
        motion.close()
    assert process_env.process.terminated
    assert process_env.process.stdin.closed


def test_motion_telemetry_handles_partial_lines_and_saves_rates(tmp_path):
    motion = bench.MotionSource(bench.RunLogs(tmp_path))
    motion.proc = Process()
    motion.reader = io.StringIO('{"event":"ready"}\n{"event":"rate",')
    motion.check()
    assert motion.ready and not motion.rates
    motion.reader = io.StringIO('"updates_per_second":120}\n')
    motion.check()
    assert motion.summary()["minimum_updates_per_second"] == 120


def test_startup_interrupt_closes_motion(monkeypatch, tmp_path):
    closed = []
    class Motion:
        def __init__(self, *args): pass
        def start(self): raise KeyboardInterrupt
        def close(self): closed.append(True)
        def summary(self): return {"minimum_updates_per_second": None}
    monkeypatch.setattr(bench, "MotionSource", Motion)
    assert bench.main(["--with-motion", "--log-dir", str(tmp_path)]) == 130
    assert closed == [True]


def test_parent_reports_failures_and_continues_workers(monkeypatch, tmp_path):
    called = []
    def spawn(path, *args, **kwargs):
        called.append(path)
        return {"path": path, "error": "simulated failure"}
    monkeypatch.setattr(bench, "spawn", spawn)
    out = tmp_path / "results.json"
    assert bench.main(["--paths", "mss", "dxcam", "--out", str(out)]) == 1
    assert called == ["mss", "dxcam"]
    assert len(json.loads(out.read_text())["results"]) == 2


def test_parent_stops_after_motion_failure(monkeypatch, tmp_path):
    called = []
    def spawn(path, *args, **kwargs):
        called.append(path)
        return {"path": path, "error": "motion died", "motion_failed": True}
    monkeypatch.setattr(bench, "spawn", spawn)
    assert bench.main(["--paths", "mss", "dxcam", "--log-dir", str(tmp_path)]) == 1
    assert called == ["mss"]


def test_worker_entrypoint_returns_nonzero(monkeypatch):
    monkeypatch.setattr(bench, "verify_path", lambda p: {"path": p, "error": "invalid"})
    assert bench.main(["--worker", "mss", "--verify"]) == 1


def test_atomic_write_failure_preserves_previous_result(monkeypatch, tmp_path):
    out = tmp_path / "result.json"
    out.write_text('{"preserved":true}')
    def failed_replace(*args):
        raise OSError("simulated replace failure")
    monkeypatch.setattr(bench.os, "replace", failed_replace)
    with pytest.raises(OSError):
        bench.save_results(out, {"new": True})
    assert json.loads(out.read_text()) == {"preserved": True}
    assert not list(tmp_path.glob("*.tmp"))


def test_mss_reference_survives_source_reuse(harness, monkeypatch):
    pixels = patterned(3, 5)
    raw = bytearray(pixels.tobytes())
    capture = SimpleNamespace(monitors=[{}, {}],
                              grab=lambda monitor: SimpleNamespace(raw=raw, height=3, width=5),
                              close=lambda: harness.events.append("close"))
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: capture))
    monkeypatch.setattr(bench, "_cpu_to_tensor", lambda image, cp, np: image.copy())
    produce, close, _ = bench._adapter_mss(harness.gpu, np, verify=True)
    sample = produce()
    raw[:] = bytes(len(raw))
    np.testing.assert_array_equal(sample.reference, pixels)
    close()
    assert harness.events == ["close"]


def test_mss_setup_failure_closes_capture(monkeypatch):
    calls = []
    capture = SimpleNamespace(monitors=[], close=lambda: calls.append("close"))
    monkeypatch.setitem(sys.modules, "mss", SimpleNamespace(mss=lambda: capture))
    with pytest.raises(IndexError):
        bench._adapter_mss(None, np)
    assert calls == ["close"]


def test_cupy_pool_release_follows_sync(harness, monkeypatch):
    original = patterned(3, 5)
    frame = SimpleNamespace(array=original.copy())
    def release():
        assert harness.events[-1] == "sync"
        harness.events.append("release")
        frame.array[:] = 0
    frame.release = release
    camera = SimpleNamespace(grab=lambda: frame, release=lambda: harness.events.append("close"))
    monkeypatch.setitem(sys.modules, "rapidshot", SimpleNamespace(create=lambda **kw: camera))
    produce, close, _ = bench._adapter_rapidshot_cupy(harness.gpu, np, verify=True)
    sample = produce()
    np.testing.assert_array_equal(sample.reference, original)
    assert bench.validate_tensor(sample.tensor, sample.reference, np)["verified"]
    close()
    assert harness.events[-2:] == ["release", "close"]


def test_cross_adapter_reference_and_actual_height(harness, monkeypatch):
    expected = patterned(3, 2)
    storage = np.zeros(64, dtype=np.uint8)
    trans = transfer()
    trans.shared_destination_handle = 7
    class Frame:
        pixels = expected.copy()
        def __enter__(self): return self
        def __exit__(self, *args):
            self.pixels[:] = 0  # Simulate surface reuse immediately on release.
            harness.events.append("frame-release")
    frame = Frame()
    def copy_with_reference(value):
        assert value is frame
        storage[:48].reshape(3, 16)[:, :8] = value.pixels.reshape(3, 8)
        harness.events.append("same-frame-copy")
        return storage.tobytes()
    trans.transfer_with_reference = copy_with_reference
    class View:
        def __init__(self, owner, shape, device):
            assert owner._transfer is trans
            assert shape == (16,)
            self.array = storage.view(np.float32)
        def close(self): harness.events.append("view-close")
    camera = SimpleNamespace(grab_frame=lambda: frame,
                             release=lambda: harness.events.append("camera-close"))
    monkeypatch.setitem(sys.modules, "rapidshot", SimpleNamespace(
        create=lambda: camera, native=SimpleNamespace(cross_adapter_transfer=lambda f: trans)))
    monkeypatch.setitem(sys.modules, "gpu_tensor_to_cupy", SimpleNamespace(CudaTensor=View))
    result = bench.verify_path("rapidshot-xadapter")
    assert result["verified"]
    assert result["reference_shape"] == [3, 2, 4]
    assert harness.events.index("same-frame-copy") < harness.events.index("frame-release")
    assert harness.events[-2:] == ["view-close", "camera-close"]


def test_cross_adapter_no_frame_does_not_enter_none(harness, monkeypatch):
    camera = SimpleNamespace(grab_frame=lambda: None, release=lambda: None)
    monkeypatch.setitem(sys.modules, "rapidshot", SimpleNamespace(create=lambda: camera,
                                                                native=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "gpu_tensor_to_cupy", SimpleNamespace(CudaTensor=None))
    produce, close, _ = bench._adapter_rapidshot_xadapter(harness.gpu, np)
    assert produce() is None
    close()


def test_ready_start_retains_pid_and_rates(process_env, monkeypatch):
    def popen(*args, **kwargs):
        kwargs["stdout"].write(b'{"event":"ready"}\n')
        kwargs["stdout"].flush()
        return process_env.process
    monkeypatch.setattr(bench.subprocess, "Popen", popen)
    motion = bench.MotionSource(process_env.logs)
    try:
        motion.start()
        assert motion.ready
        parent_log = (process_env.logs.directory / "parent.log").read_text()
        assert '"child_pid": 123' in parent_log
    finally:
        motion.close()


def test_motion_progress_timeout_is_fatal(process_env):
    motion = bench.MotionSource(process_env.logs)
    motion.proc = process_env.process
    motion.ready = True
    process_env.clock.now = 31
    with pytest.raises(bench.MotionError, match="progress"):
        motion.check()


def test_worker_interrupt_stops_process(process_env, monkeypatch):
    def interrupted(seconds): raise KeyboardInterrupt
    monkeypatch.setattr(bench.time, "sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        bench.spawn("mss", 1, 1, False, logs=process_env.logs)
    assert process_env.process.terminated


# -- what a path is allowed to refuse ------------------------------------
#
# A path that cannot run here and a path that is broken need opposite
# reactions: one is a skip, the other fails the run. The distinction is
# `PathUnavailable`, and it only earns its keep if the harness actually treats
# the two differently -- so that is what these check, not the exception itself.


@pytest.mark.parametrize("image, why", [
    (np.zeros((4, 4, 3), np.uint8), "three channels"),
    (np.zeros((4, 4), np.uint8), "no channel axis"),
    (np.zeros((0, 4, 4), np.uint8), "empty"),
    (np.zeros((4, 4, 4), np.float32), "not uint8"),
])
def test_bgra_input_is_checked_before_it_is_resized(image, why):
    """A wrong-shaped source produces a correctly shaped tensor of nonsense,
    which is the failure mode this whole file exists to make impossible."""
    with pytest.raises(ValueError):
        bench._check_bgra(image)


def transfer_stub(**overrides):
    fields = {"total_bytes": 3 * bench.OUT * bench.OUT * 2,
              "destination_is_software": False}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def test_a_transfer_carrying_the_wrong_payload_is_a_bug_not_a_skip(harness):
    """Size is how this path knows it moved the tensor rather than the frame.
    Getting it wrong means the benchmark is timing something else."""
    monkey = transfer_stub(total_bytes=16 * 1024 * 1024)
    with pytest.raises(ValueError, match="bytes"):
        bench._validate_tensor_transfer(monkey, np)


def test_a_target_that_cannot_be_viewed_is_refused(harness, monkeypatch):
    """The CUDA import maps 4-byte words and the result is viewed back to its
    own dtype, so a target size that is not a whole number of words cannot be
    read at all. An odd output square in FP16 is 6*OUT^2 bytes, which is 2 mod
    4 -- reachable only because this is checked before the arrival is."""
    monkeypatch.setattr(bench, "OUT", 3)
    monkeypatch.setattr(bench, "TARGET_DTYPE", "float16")
    with pytest.raises(ValueError, match="4-byte words"):
        bench._validate_tensor_transfer(transfer_stub(total_bytes=54), np)


def test_a_warp_destination_is_a_skip_not_a_failure(harness):
    """'no host-to-device transfer' is only meaningful against a real second
    GPU. Against WARP it measures a system-memory copy, so the row would be a
    confident lie rather than an error."""
    with pytest.raises(bench.PathUnavailable, match="WARP"):
        bench._validate_tensor_transfer(transfer_stub(destination_is_software=True), np)


def test_a_transfer_of_exactly_the_tensor_is_accepted(harness):
    monkeypatch_dtype = bench.TARGET_DTYPE
    size = 3 * bench.OUT * bench.OUT * np.dtype(monkeypatch_dtype).itemsize
    bench._validate_tensor_transfer(transfer_stub(total_bytes=size), np)


def test_an_unavailable_path_is_skipped_rather_than_failed(harness):
    """`run_path` reports `skipped`; nothing writes `error`, because the run
    should not go red for a machine that simply cannot host the path."""
    def produce():
        raise bench.PathUnavailable("capture adapter has no CUDA device")

    harness.install(produce)
    result = bench.run_path("mss", 1, 1)
    assert "error" not in result
    assert "no CUDA device" in result["skipped"]
    assert harness.events.count("close") == 1


def test_an_unavailable_path_is_skipped_during_verification_too(harness):
    def produce():
        raise bench.PathUnavailable("capture adapter has no CUDA device")

    harness.install(produce)
    result = bench.verify_path("mss")
    assert "error" not in result
    assert "no CUDA device" in result["skipped"]


def test_an_ordinary_failure_is_still_an_error(harness):
    """The skip must not swallow real breakage -- that would turn every bug on
    a GPU path into a quiet green run."""
    def produce():
        raise RuntimeError("the adapter is broken")

    harness.install(produce)
    result = bench.run_path("mss", 1, 1)
    assert "skipped" not in result
    assert "the adapter is broken" in result["error"]


# -- what the worker is told ----------------------------------------------


def test_the_worker_is_told_which_harness_and_which_dtype(process_env):
    """A worker is a fresh process. `ai_ingestion.py` as a script dispatches to
    section7's pixel-age harness, so without `--call-duration` the worker runs
    a different benchmark than the parent thinks; without `--dtype` it measures
    the module default while the parent reports the requested one."""
    commands = []
    original = bench.subprocess.Popen

    def popen(*args, **kwargs):
        commands.append(list(args[0]))
        return original(*args, **kwargs)

    bench.subprocess.Popen = popen
    try:
        bench.spawn("mss", 1, 1, False, logs=process_env.logs, dtype="float32")
    finally:
        bench.subprocess.Popen = original
    command, = commands
    assert "--call-duration" in command
    assert command[command.index("--dtype") + 1] == "float32"
    assert command[command.index("--worker") + 1] == "mss"


def test_the_dtype_flag_is_omitted_when_not_asked_for(process_env):
    """So a caller that does not care keeps whatever the module defaults to,
    rather than having the parent's default silently pinned into the child."""
    commands = []
    original = bench.subprocess.Popen

    def popen(*args, **kwargs):
        commands.append(list(args[0]))
        return original(*args, **kwargs)

    bench.subprocess.Popen = popen
    try:
        bench.spawn("mss", 1, 1, False, logs=process_env.logs)
    finally:
        bench.subprocess.Popen = original
    assert "--dtype" not in commands[0]


# -- the adapters that had no tests --------------------------------------
#
# Their conversion is shared and covered above; what was not covered is each
# one's *control flow* -- what it does when no frame arrives, and whether the
# frame gets released. A leaked frame is the expensive one: DXGI cannot acquire
# the next surface while a reference is outstanding, so capture stops dead
# rather than slowing down.


def test_dxcam_reports_no_frame_rather_than_an_empty_one(harness, monkeypatch):
    """DXcam returns None when nothing changed. Turning that into a Sample
    would report a capture rate the screen never produced."""
    camera = SimpleNamespace(grab=lambda: None,
                             release=lambda: harness.events.append("close"))
    monkeypatch.setitem(sys.modules, "dxcam",
                        SimpleNamespace(create=lambda **kw: camera))
    produce, close, meta = bench._adapter_dxcam(harness.gpu, np)
    assert produce() is None
    assert meta["h2d_bytes_per_frame"] == 3 * bench.OUT * bench.OUT
    close()
    assert harness.events[-1] == "close"


def test_rapidshot_cpu_reports_no_frame_rather_than_an_empty_one(harness, monkeypatch):
    camera = SimpleNamespace(grab=lambda: None,
                             release=lambda: harness.events.append("close"))
    monkeypatch.setitem(sys.modules, "rapidshot",
                        SimpleNamespace(create=lambda **kw: camera))
    produce, close, meta = bench._adapter_rapidshot_cpu(harness.gpu, np)
    assert produce() is None
    assert meta["h2d_bytes_per_frame"] == 3 * bench.OUT * bench.OUT
    close()


def test_rapidshot_cpu_releases_the_frame_when_conversion_raises(harness, monkeypatch):
    """The `finally` that makes this safe. Without it one bad frame stops
    capture for the rest of the run, and the benchmark reports whatever it had
    managed before that rather than an error."""
    released = []
    frame = SimpleNamespace(release=lambda: released.append("release"))
    camera = SimpleNamespace(grab=lambda: frame, release=lambda: None)
    monkeypatch.setitem(sys.modules, "rapidshot",
                        SimpleNamespace(create=lambda **kw: camera))

    def explode(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(bench, "_cpu_sample", explode)
    produce, close, _ = bench._adapter_rapidshot_cpu(harness.gpu, np)
    with pytest.raises(RuntimeError, match="conversion failed"):
        produce()
    assert released == ["release"]
    close()


def test_rapidshot_cpu_releases_the_frame_it_converted(harness, monkeypatch):
    released = []
    frame = SimpleNamespace(release=lambda: released.append("release"))
    camera = SimpleNamespace(grab=lambda: frame, release=lambda: None)
    monkeypatch.setitem(sys.modules, "rapidshot",
                        SimpleNamespace(create=lambda **kw: camera))
    monkeypatch.setattr(bench, "_cpu_sample", lambda *a, **kw: bench.Sample("t"))
    produce, close, _ = bench._adapter_rapidshot_cpu(harness.gpu, np)
    assert produce().tensor == "t"
    assert released == ["release"]
    close()


def test_async_transfer_waits_before_the_frame_is_released(harness, monkeypatch):
    """The hazard async introduces over the blocking call, and the reason the
    wait sits inside the `with`. `transfer_async` returns before the copy that
    reads the surface has run; releasing first hands the surface back to DXGI
    to overwrite while the copy is still reading it.
    """
    storage = np.zeros(64, dtype=np.uint8)
    trans = transfer()
    trans.shared_destination_handle = 7
    trans.transfer_async = lambda frame: harness.events.append("submit") or 11
    trans.wait_shared_fence = lambda value: harness.events.append(f"wait({value})")

    class Frame:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            harness.events.append("frame-release")

    class View:
        def __init__(self, owner, shape, device):
            self.array = storage.view(np.float32)

        def close(self):
            harness.events.append("view-close")

    camera = SimpleNamespace(grab_frame=lambda: Frame(),
                             release=lambda: harness.events.append("camera-close"))
    monkeypatch.setitem(sys.modules, "rapidshot", SimpleNamespace(
        create=lambda: camera,
        native=SimpleNamespace(cross_adapter_transfer=lambda f: trans)))
    monkeypatch.setitem(sys.modules, "gpu_tensor_to_cupy",
                        SimpleNamespace(CudaTensor=View))

    produce, close, meta = bench._adapter_rapidshot_xadapter_async(harness.gpu, np)
    produce()
    assert meta["h2d_bytes_per_frame"] == 0
    order = harness.events
    assert order.index("submit") < order.index("wait(11)") < order.index("frame-release")
    close()


# -- the 2.6 paths, and the small helpers everything leans on -------------


def converter_env(harness, monkeypatch, *, to_cupy=None, software=False):
    """A fake rapidshot whose converter and transfer record what was asked."""
    calls = []

    class Tensor:
        def to_cupy(self):
            if to_cupy is not None:
                raise to_cupy
            return np.zeros(bench.TARGET_SHAPE, dtype=bench.TARGET_DTYPE)

    class Converter:
        def __init__(self, frame, size, **kwargs):
            calls.append(("converter", size, kwargs))

        def process(self, frame):
            calls.append(("process",))
            return Tensor()

    class Transfer:
        total_bytes = 3 * bench.OUT * bench.OUT * np.dtype(bench.TARGET_DTYPE).itemsize
        destination_is_software = software
        shared_destination_handle = 9

        def __init__(self, converter):
            calls.append(("transfer-built",))

        def transfer(self):
            calls.append(("transfer",))

    class Frame:
        width = height = 8

        def __enter__(self):
            return self

        def __exit__(self, *args):
            calls.append(("frame-release",))
            harness.events.append("frame-release")

    class View:
        def __init__(self, owner, shape, device):
            self.array = np.zeros(shape, dtype=np.float32)

        def close(self):
            harness.events.append("view-close")

    camera = SimpleNamespace(grab_frame=lambda: Frame(),
                             release=lambda: harness.events.append("camera-close"))

    class Required(RuntimeError):
        pass

    monkeypatch.setitem(sys.modules, "rapidshot", SimpleNamespace(
        create=lambda: camera, GpuConverter=Converter, TensorTransfer=Transfer,
        CrossAdapterRequired=Required, native=SimpleNamespace()))
    monkeypatch.setitem(sys.modules, "gpu_tensor_to_cupy",
                        SimpleNamespace(CudaTensor=View))
    return SimpleNamespace(calls=calls, Required=Required)


def test_the_converter_is_asked_for_the_benchmarks_own_target(harness, monkeypatch):
    """Every path must produce the same tensor or the comparison measures
    different work, so the converter is configured from the module's target
    rather than from anything of its own."""
    env = converter_env(harness, monkeypatch)
    produce, close, meta = bench._adapter_rapidshot_converter_xadapter(harness.gpu, np)
    produce()
    (_, size, kwargs), = [c for c in env.calls if c[0] == "converter"]
    assert size == (bench.OUT, bench.OUT)
    assert kwargs["dtype"] == bench.TARGET_DTYPE
    assert kwargs["layout"] == "nchw" and kwargs["normalize"] is True
    assert meta["h2d_bytes_per_frame"] == 0
    close()


def test_the_converted_payload_crosses_after_it_is_converted(harness, monkeypatch):
    """Ordering B is the whole point: convert first, move the small result."""
    env = converter_env(harness, monkeypatch)
    produce, close, _ = bench._adapter_rapidshot_converter_xadapter(harness.gpu, np)
    produce()
    names = [c[0] for c in env.calls]
    # One list, so the order is actually comparable: convert, then move the
    # result, and only then let the frame go -- the copy reads the surface.
    assert names.index("process") < names.index("transfer") < names.index("frame-release")
    close()


def test_a_warp_destination_skips_the_converter_path(harness, monkeypatch):
    converter_env(harness, monkeypatch, software=True)
    produce, close, _ = bench._adapter_rapidshot_converter_xadapter(harness.gpu, np)
    with pytest.raises(bench.PathUnavailable, match="WARP"):
        produce()
    close()


def test_the_direct_export_returns_the_tensor_where_it_already_is(harness, monkeypatch):
    """No transfer at all, which is what makes this the fastest shape of the
    path -- and only possible when capture and CUDA are the same adapter."""
    converter_env(harness, monkeypatch)
    produce, close, meta = bench._adapter_rapidshot_converter(harness.gpu, np)
    assert produce().tensor.shape == bench.TARGET_SHAPE
    assert meta["h2d_bytes_per_frame"] == 0
    close()


def test_the_direct_export_reports_cross_adapter_as_a_skip(harness, monkeypatch):
    env = converter_env(harness, monkeypatch)
    required = sys.modules["rapidshot"].CrossAdapterRequired("no CUDA device here")

    class Tensor:
        def to_cupy(self):
            raise required

    monkeypatch.setattr(sys.modules["rapidshot"], "GpuConverter",
                        lambda frame, size, **kw: SimpleNamespace(
                            process=lambda f: Tensor()))
    produce, close, _ = bench._adapter_rapidshot_converter(harness.gpu, np)
    with pytest.raises(bench.PathUnavailable, match="no CUDA device"):
        produce()
    close()


@pytest.mark.parametrize("camera, expected", [
    (SimpleNamespace(release=lambda: None), None),
    (SimpleNamespace(stop=lambda: None), None),
])
def test_a_camera_is_closed_by_whichever_method_it_offers(camera, expected):
    """mss stops, the others release. Picking wrong leaks the device."""
    assert bench._close_camera(camera) is expected


def test_a_camera_with_no_cleanup_is_refused_loudly():
    """Silently skipping cleanup leaks a duplication device, and the next
    camera on that output fails for an unrelated-looking reason."""
    with pytest.raises(RuntimeError, match="cleanup"):
        bench._close_camera(SimpleNamespace())


@pytest.mark.parametrize("seconds, warmup", [
    (0, 1), (-1, 1), (float("inf"), 1), (float("nan"), 1), (1, -1)])
def test_nonsense_durations_are_refused_before_a_camera_is_opened(seconds, warmup):
    with pytest.raises(ValueError):
        bench._validate_options(seconds, warmup)


def test_a_child_that_ignores_terminate_is_killed():
    """A source that will not stop holds the display and the next path
    measures against it."""
    events = []

    class Proc:
        def __init__(self):
            self.alive = True

        def poll(self):
            return None if self.alive else 0

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout=None):
            if "kill" not in events:
                raise bench.subprocess.TimeoutExpired("cmd", timeout)
            return 0

        def kill(self):
            events.append("kill")

    bench.stop_process(Proc())
    assert events == ["terminate", "kill"]


def test_an_already_finished_child_is_left_alone():
    proc = SimpleNamespace(poll=lambda: 0,
                           terminate=lambda: pytest.fail("terminated a dead child"))
    bench.stop_process(proc)
    bench.stop_process(None)


def test_children_inherit_utf8_and_no_console():
    """A console window steals focus mid-run, which changes what is on screen
    and therefore what is being measured."""
    options = bench._child_options()
    assert options["env"]["PYTHONIOENCODING"] == "utf-8"
    assert "creationflags" in options
