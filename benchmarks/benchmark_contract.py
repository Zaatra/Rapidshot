"""Pure section 7 contracts. Importing this module never touches the desktop."""
import ctypes
import hashlib
import json
import math
from pathlib import Path

OUT = 640
SHAPE = (1, 3, OUT, OUT)
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


def pipeline_rgb(bgra, xp, size=OUT):
    """The resize a deployment would actually run, per backend.

    NumPy inputs go through cv2's fixed-point bilinear; CuPy inputs use the
    exact contract, which is already ~1.2 ms there and needs no substitute.
    Same algorithm, same output size, same full-image extent -- only the
    rounding differs, and `canonical_rgb` remains the reference every path is
    verified against.

    Using `canonical_rgb` in the measured loop instead would charge NumPy paths
    76-86 ms for what cv2 does in 0.72 ms. That is not a fairer comparison; it
    is a slower one applied only to the competitors.
    """
    if getattr(xp, "__name__", "") == "numpy":
        import cv2

        if len(bgra.shape) != 3 or bgra.shape[2] != 4 or str(bgra.dtype) != "uint8":
            raise ValueError("expected uint8 HxWx4 BGRA")
        small = cv2.resize(bgra, (size, size), interpolation=cv2.INTER_LINEAR)
        return xp.ascontiguousarray(small[:, :, 2::-1])
    return canonical_rgb(bgra, xp, size)


def canonical_rgb(bgra, xp, size=OUT):
    """Exact rational half-pixel bilinear, round-half-up to RGB8.

    Integer arithmetic fixes CPU/GPU interpolation and rounding differences.
    No FMA or hardware texture-sampler precision enters this contract.
    """
    if len(bgra.shape) != 3 or bgra.shape[2] != 4 or str(bgra.dtype) != "uint8":
        raise ValueError("expected uint8 HxWx4 BGRA")
    h, w = bgra.shape[:2]
    if min(h, w, size) <= 0:
        raise ValueError("empty image or output")
    den = 2 * size
    yn = xp.maximum((2 * xp.arange(size, dtype=xp.int64) + 1) * h - size, 0)
    xn = xp.maximum((2 * xp.arange(size, dtype=xp.int64) + 1) * w - size, 0)
    y0, x0 = yn // den, xn // den
    y1, x1 = xp.minimum(y0 + 1, h - 1), xp.minimum(x0 + 1, w - 1)
    wy, wx = (yn % den)[:, None, None], (xn % den)[None, :, None]
    a = bgra[y0[:, None], x0[None, :], :3].astype(xp.int64)
    b = bgra[y0[:, None], x1[None, :], :3].astype(xp.int64)
    c = bgra[y1[:, None], x0[None, :], :3].astype(xp.int64)
    d = bgra[y1[:, None], x1[None, :], :3].astype(xp.int64)
    value = ((a * (den - wx) + b * wx) * (den - wy)
             + (c * (den - wx) + d * wx) * wy)
    rounded = ((value + den * den // 2) // (den * den)).astype(xp.uint8)
    return xp.ascontiguousarray(rounded[:, :, ::-1])


def normalized_tensor(rgb, xp):
    chw = xp.ascontiguousarray(rgb.transpose(2, 0, 1))
    return (chw.astype(xp.float32) / xp.float32(255)).astype(xp.float16)[None]


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


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
