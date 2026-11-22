"""GpuConverter must let other Python threads run while the GPU works.

`GpuConverter12.process()` and `read_back()` held the GIL through their GPU
waits. Measured 2026-11-17 on an Intel UHD at 2560x1600 NV12: a second Python
thread made 36% of its idle progress during `process()`, and an encoder thread
beside a capture thread saw NVENC's 4.2 ms call take 13.5 ms. After the fix,
100% and 96% (with readback). The older GpuPreprocessor12 had the same fix in
2026-08 for the same reason.

Needs a real capture and the native extension; skips without them. The bound
is loose on purpose -- 0.36 against 1.0 measured -- so load on the machine
does not make it flaky.
"""

import threading
import time

import pytest

rapidshot = pytest.importorskip("rapidshot")
from rapidshot import native  # noqa: E402

pytestmark = pytest.mark.skipif(
    not native.is_available() or rapidshot.GpuConverter is None,
    reason="native extension not built")


def _held_frame(camera, tries=600):
    for _ in range(tries):
        frame = camera.grab_frame()
        if frame is not None:
            return frame
    pytest.skip("no frame captured -- the screen must be changing")


def _other_thread_progress(work, seconds=0.6):
    count, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            count[0] += 1

    thread = threading.Thread(target=spin, daemon=True)
    thread.start()
    time.sleep(0.1)

    def rate(job):
        start_count, start = count[0], time.perf_counter()
        while time.perf_counter() - start < seconds:
            job()
        return (count[0] - start_count) / (time.perf_counter() - start)

    try:
        idle = rate(lambda: time.sleep(0.005))
        busy = rate(work)
    finally:
        stop.set()
        thread.join()
    return busy / idle


@pytest.mark.parametrize("readback", [False, True], ids=["process", "process+numpy"])
def test_another_thread_keeps_running_during_a_conversion(readback):
    try:
        camera = rapidshot.create(output_color="BGRA")
    except Exception as exc:  # noqa: BLE001 - no capture on this machine
        pytest.skip(f"capture unavailable: {type(exc).__name__}: {exc}")
    try:
        frame = _held_frame(camera)
        with frame:
            converter = rapidshot.GpuConverter(
                frame, (camera.width, camera.height), pixel_format="nv12")
            converter.process(frame)

            def work():
                tensor = converter.process(frame)
                if readback:
                    tensor.numpy()

            share = _other_thread_progress(work)
            converter.close()
    finally:
        camera.release()
    assert share > 0.36, (
        f"another Python thread made {share:.0%} of its idle progress during "
        "GpuConverter work: the GPU wait is holding the GIL")
