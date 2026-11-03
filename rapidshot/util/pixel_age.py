"""Pixel age: how old a captured frame's pixels are when they reach the caller.

The test source (``latency_source.exe``) stamps a frame ID into a 48-cell
marker in its top-left corner and logs the QPC time of every ``Present()``.
Decoding the marker from a captured frame and looking its ID up in that log
gives the frame's age at the moment it was read. The benchmark harness
(``_bench``) and the capture planner (``rapidshot.planner``) measure with the
same pieces, which live here so library code never imports the harness.
"""
import ctypes
import json
import math
from pathlib import Path

MARKER_BITS = 48
CELL = 8
MARKER_HEIGHT = 16
MAGIC = 0xA7


def qpc_clock():
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    counter, frequency = kernel.QueryPerformanceCounter, kernel.QueryPerformanceFrequency
    counter.argtypes = frequency.argtypes = [ctypes.POINTER(ctypes.c_longlong)]
    value = ctypes.c_longlong()
    if not frequency(ctypes.byref(value)) or value.value <= 0:
        raise OSError("QueryPerformanceFrequency failed")
    hz = value.value

    def now():
        tick = ctypes.c_longlong()
        if not counter(ctypes.byref(tick)):
            raise OSError("QueryPerformanceCounter failed")
        return tick.value
    return now, hz


def checksum(frame_id):
    return ((frame_id >> 24) ^ (frame_id >> 16) ^ (frame_id >> 8) ^ frame_id ^ MAGIC) & 255


def decode_marker(pixels, xp):
    """Read 48 tiny cells only. GPU paths copy 48 booleans, not an image."""
    if pixels.shape[0] < MARKER_HEIGHT or pixels.shape[1] < MARKER_BITS * CELL:
        return None
    cells = pixels[8, xp.arange(MARKER_BITS) * CELL + CELL // 2, :3]
    # Reject coloured/obscured cells rather than guessing a plausible ID.
    white, black = xp.all(cells > 240, axis=1), xp.all(cells < 15, axis=1)
    values = xp.stack((white, white | black))
    if hasattr(xp, "asnumpy"):
        values = xp.asnumpy(values)
    if not values[1].all():
        return None
    bits = sum(int(bit) << i for i, bit in enumerate(values[0]))
    frame_id = (bits >> 8) & 0xFFFFFFFF
    if bits & 255 != MAGIC or bits >> 40 != checksum(frame_id):
        return None
    return frame_id


def percentiles(values):
    if not values:
        return None
    values = sorted(values)
    n = len(values)
    mean = sum(values) / n
    return {"count": n, "p50": values[min(n - 1, int(n * .5))],
            "p95": values[min(n - 1, int(n * .95))],
            "p99": values[min(n - 1, int(n * .99))],
            "jitter_stdev": math.sqrt(sum((v - mean) ** 2 for v in values) / n)}


class PresentLog:
    def __init__(self, path, frequency):
        self.reader = Path(path).open(encoding="utf-8")
        self.pending, self.frames, self.frequency = "", {}, frequency

    def refresh(self):
        self.pending += self.reader.read()
        while "\n" in self.pending:
            line, self.pending = self.pending.split("\n", 1)
            row = json.loads(line)
            if row.get("event") == "present":
                self.frames[row["id"]] = row

    def age(self, frame_id, ready):
        self.refresh()
        row = self.frames.get(frame_id)
        if row is None:
            return None
        delta = ready - row["qpc_before"]
        if delta < 0:
            raise ValueError("captured marker predates its source timestamp")
        return delta * 1000 / self.frequency

    def close(self):
        self.reader.close()
