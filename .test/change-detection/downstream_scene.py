"""Repeatable dashboard, independent template truth, and nonactivating test windows."""

from __future__ import annotations

from contextlib import ExitStack
import ctypes as ct
from ctypes import wintypes as wt
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

Rect = tuple[int, int, int, int]
Point = tuple[int, int]
ICON_SIZE = 36
TILE_SIZE = 128
STAMP_ORIGIN = (332, 31)
STAMP_CELL = 6
STAMP_BITS = 32
STAMP_HEIGHT = 12
STAMP_MAGIC = 0xA6
STEPS = 13


@dataclass(frozen=True)
class Step:
    workload: str
    stamp: int
    objects: dict[str, Point]
    covers: bool = False
    cover_phase: int | None = None
    visible_update: bool = True
    cover_rect: Rect | None = None
    motion: int = 0


@lru_cache(maxsize=1)
def templates() -> dict[str, np.ndarray]:
    """Original, exact grayscale assets used by both GDI and the actual task."""
    yy, xx = np.indices((ICON_SIZE, ICON_SIZE))

    def circle(cx: int, cy: int, radius: int) -> np.ndarray:
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2

    def asset(background: int, mask: np.ndarray, foreground: int) -> np.ndarray:
        result = np.full((ICON_SIZE, ICON_SIZE), background, np.uint8)
        result[mask] = foreground
        return result

    lines = ((xx >= 8) & (xx <= 27) & (abs(2 * yy - xx - 20) <= 2)) | (
        (yy >= 10) & (yy <= 25) & (abs(2 * xx + yy - 55) <= 2)
    )
    network = asset(36, lines | circle(9, 14, 4) | circle(24, 12, 5) | circle(20, 25, 5), 234)

    case = (xx >= 6) & (xx <= 27) & (yy >= 10) & (yy <= 25)
    case_border = case & ((xx < 9) | (xx > 24) | (yy < 13) | (yy > 22))
    terminal = (xx >= 28) & (xx <= 30) & (yy >= 15) & (yy <= 20)
    power = asset(47, case_border | terminal, 221)
    power[13:23, 10:18] = 137

    triangle = (yy >= 6) & (yy < 30) & (abs(xx - 18) * 5 <= (yy - 5) * 3)
    alert = asset(53, triangle, 243)
    alert[14:22, 17:20] = 53
    alert[24:27, 17:20] = 53

    ring = circle(18, 18, 12) & ~circle(18, 18, 9)
    ring &= ((xx < 18) & (yy > 14)) | ((xx > 18) & (yy < 22))
    arrows = (abs(xx - 21) + abs(yy - 7) < 5) | (abs(xx - 15) + abs(yy - 29) < 5)
    sync = asset(63, ring | arrows, 230)
    icons = {"network": network, "power": power, "alert": alert, "sync": sync}
    for icon in icons.values():
        icon.setflags(write=False)
    return icons


def scenes(
    width: int = 1920, height: int = 1080, visible_steps: int = STEPS - 1,
    passive_steps: int = STEPS - 1,
) -> dict[str, list[Step]]:
    if width < 1024 or height < 640:
        raise ValueError("the dashboard needs at least 1024 × 640 pixels")
    if not 1 <= visible_steps <= 64 or not 1 <= passive_steps <= 64:
        raise ValueError("dashboard steps must be between 1 and 64")
    devices = {
        "network": (width * 39 // 100, height * 32 // 100),
        "power": (width * 65 // 100, height * 34 // 100),
        "alert": (width * 54 // 100, height * 52 // 100),
        "sync": (width * 77 // 100, height * 70 // 100),
    }
    ax, ay = devices["alert"]
    cover = (ax - 100, ay - 50, ax + 160, ay + 120)
    tile_edge = ((devices["network"][0] + TILE_SIZE // 2) // TILE_SIZE) * TILE_SIZE
    sparse = []
    dense = []
    for i in range(visible_steps + 1):
        objects = dict(devices)
        objects["network"] = (
            tile_edge - ICON_SIZE - 8 + i % 4 * 24,
            objects["network"][1] + (i % 3) * 4,
        )
        if i % 6 in (2, 3):
            del objects["alert"]
        if i % 7 == 5:
            del objects["power"]
        if i % 4 == 3:
            x, y = objects["sync"]
            objects["sync"] = (x - 47, y + 13)
        sparse.append(Step("sparse", 0x100 + i, objects))

        locations = dict(
            zip(
                devices,
                (
                    (width * 31 // 100 + i % 5 * 15, height * 30 // 100 + i % 3 * 14),
                    (width * 72 // 100 - i % 4 * 18, height * 28 // 100 + i % 6 * 11),
                    (width * 42 // 100 + i % 4 * 23, height * 66 // 100 - i % 3 * 10),
                    (width * 78 // 100 - i % 5 * 21, height * 70 // 100 + i % 3 * 12),
                ),
            )
        )
        if i % 5 == 2:
            del locations["sync"]
        dense.append(Step("dense", 0x200 + i, locations, motion=i + 1))

    return {
        "sparse": sparse,
        "dense": dense,
        "idle": [Step("idle", 0x300, dict(devices), visible_update=i == 0)
                 for i in range(passive_steps + 1)],
        "excluded_cover": [
            Step(
                "excluded_cover",
                0x400,
                dict(devices),
                covers=True,
                cover_phase=i,
                visible_update=i == 0,
                cover_rect=cover,
            )
            for i in range(passive_steps + 1)
        ],
    }


def _stamp_rect(origin: Point) -> Rect:
    x, y = origin
    return x, y, x + STAMP_BITS * STAMP_CELL, y + STAMP_HEIGHT


def _stamp(image: np.ndarray, origin: Point, value: int) -> None:
    parity = (value & 255) ^ (value >> 8) ^ 0x4B
    word = (STAMP_MAGIC << 24) | (value << 8) | parity
    x, y = origin
    for bit in range(STAMP_BITS):
        color = 242 if ((word >> (31 - bit)) & 1) else 8
        image[y : y + STAMP_HEIGHT, x : x + STAMP_CELL, :3] = color
        x += STAMP_CELL


def _read_stamp(frame: np.ndarray, origin: Point) -> int | None:
    x, y, right, bottom = _stamp_rect(origin)
    if frame.ndim != 3 or frame.shape[2] < 3 or frame.shape[0] < bottom or frame.shape[1] < right:
        return None
    row = frame[y + STAMP_HEIGHT // 2, x + STAMP_CELL // 2 : right : STAMP_CELL, :3]
    dark = (row < 44).all(axis=1)
    light = (row > 208).all(axis=1)
    if not (dark | light).all():
        return None
    word = 0
    for bit in light:
        word = (word << 1) | int(bit)
    value, parity = (word >> 8) & 0xFFFF, word & 255
    if word >> 24 != STAMP_MAGIC or parity != ((value & 255) ^ (value >> 8) ^ 0x4B):
        return None
    return value


def read_stamp(frame: np.ndarray) -> int | None:
    return _read_stamp(frame, STAMP_ORIGIN)


def read_cover_stamp(step: Step, frame: np.ndarray) -> int | None:
    if not step.covers or step.cover_rect is None:
        return None
    x, y, _, _ = step.cover_rect
    return _read_stamp(frame, (x + 17, y + 34))


def capture_cover_state(step: Step, frame: np.ndarray) -> str:
    """Distinguish a transparent/excluded, visible, or masked cover by its pixels."""
    if not step.covers:
        return "none"
    if read_cover_stamp(step, frame) is not None:
        return "visible"
    x, y = step.objects["alert"]
    alert = templates()["alert"]
    actual = frame[y : y + ICON_SIZE, x : x + ICON_SIZE, :3]
    if actual.shape == (ICON_SIZE, ICON_SIZE, 3) and (actual == alert[:, :, None]).all():
        return "excluded"
    return "occluded"


def expected_detections(step: Step, cover_state: str = "none") -> dict[str, Point]:
    if cover_state not in {"none", "excluded", "visible", "occluded"}:
        raise ValueError(f"unknown cover state {cover_state!r}")
    if step.cover_rect is None or cover_state in {"excluded", "none"}:
        return dict(step.objects)
    left, top, right, bottom = step.cover_rect
    return {
        name: (x, y)
        for name, (x, y) in step.objects.items()
        if x + ICON_SIZE <= left or y + ICON_SIZE <= top or x >= right or y >= bottom
    }


def _fill(image: np.ndarray, rect: Rect, rgb: tuple[int, int, int]) -> None:
    left, top, right, bottom = rect
    image[top:bottom, left:right, :3] = rgb[::-1]


def _text(image: np.ndarray, text: str, position: Point, size: float = 0.56) -> None:
    import cv2

    cv2.putText(
        image, text, position, cv2.FONT_HERSHEY_SIMPLEX, size, (206, 209, 219, 255), 1, cv2.LINE_8
    )


@lru_cache(maxsize=4)
def _static_dashboard(width: int, height: int) -> np.ndarray:
    image = np.full((height, width, 4), 255, np.uint8)
    _fill(image, (0, 0, width, height), (24, 27, 37))
    _fill(image, (0, 0, width, 72), (33, 38, 53))
    _text(image, "RAPIDSHOT / OPERATIONS", (24, 45), 0.66)
    _text(image, "LIVE INGEST MONITOR", (width - 226, 45), 0.48)
    left = width * 17 // 100
    _fill(image, (18, 92, left - 20, height - 27), (30, 36, 48))
    for i, label in enumerate(
        ("OVERVIEW", "Capture routes", "Display adapters", "Event history", "Queue status")
    ):
        y = 132 + i * 50
        if i == 0:
            _fill(image, (27, y - 22, left - 31, y + 15), (44, 59, 77))
        _text(image, label, (42, y))
    columns = (left + 6, width * 56 // 100, width - 30)
    rows = (107, height * 51 // 100, height - 25)
    for i, label in enumerate(
        ("INPUT SIGNALS", "RESOURCE HEALTH", "PRESENTATION TIMELINE", "DELIVERY EVENTS")
    ):
        row, column = divmod(i, 2)
        x0, x1 = columns[column] + 10, columns[column + 1] - 10
        y0, y1 = rows[row] + 8, rows[row + 1] - 8
        _fill(image, (x0, y0, x1, y1), (31, 38, 51))
        _text(image, label, (x0 + 19, y0 + 34))
        for stripe in range(y0 + 65, y1 - 12, 33):
            _fill(image, (x0 + 25, stripe, x1 - 22, stripe + 1), (44, 54, 68))
    _text(image, "Capture available    /    SDR BGRA8", (36, height - 52), 0.46)
    image.setflags(write=False)
    return image


def _dense_dashboard(image: np.ndarray, phase: int) -> None:
    height, width = image.shape[:2]
    for y in range(84, height - 30, 29):
        for x in range(18, width - 30, 44):
            value = (x // 11 + y // 7 + phase * 17) % 96
            _fill(
                image,
                (x, y, min(x + 39, width), min(y + 20, height)),
                (43 + value, 48 + value // 2, 78 + (value * 3) % 103),
            )


def _cover_image(step: Step) -> np.ndarray:
    if step.cover_rect is None or step.cover_phase is None:
        raise ValueError("cover paint requires cover bounds and phase")
    left, top, right, bottom = step.cover_rect
    width, height = right - left, bottom - top
    image = np.full((height, width, 4), 255, np.uint8)
    _fill(image, (0, 0, width, height), (92, 39, 61))
    _fill(image, (4, 4, width - 4, height - 4), (55, 32, 48))
    _text(image, "CAPTURE-EXCLUDED PREVIEW", (15, 24), 0.45)
    offset = 20 + step.cover_phase % 8 * 22
    _fill(image, (offset, 98, offset + 44, 132), (184, 78, 123))
    _stamp(image, (17, 34), step.cover_phase)
    return image


def render(
    step: Step, width: int = 1920, height: int = 1080, *, include_cover: bool = False
) -> np.ndarray:
    """The compositor input with BGRA8 bytes. GDI capture may choose another alpha."""
    image = _static_dashboard(width, height).copy()
    if step.motion:
        _dense_dashboard(image, step.motion)
    _stamp(image, STAMP_ORIGIN, step.stamp)
    for name, (x, y) in step.objects.items():
        image[y : y + ICON_SIZE, x : x + ICON_SIZE, :3] = templates()[name][:, :, None]
    if include_cover and step.cover_rect is not None:
        left, top, right, bottom = step.cover_rect
        image[top:bottom, left:right] = _cover_image(step)
    return image


def _changed_regions(previous: Step, current: Step) -> list[Rect]:
    regions = [_stamp_rect(STAMP_ORIGIN)] if previous.stamp != current.stamp else []
    for name in sorted(previous.objects.keys() | current.objects.keys()):
        old, new = previous.objects.get(name), current.objects.get(name)
        if old != new:
            for placement in (old, new):
                if placement is not None:
                    x, y = placement
                    regions.append((x, y, x + ICON_SIZE, y + ICON_SIZE))
    return regions


class _BitmapHeader(ct.Structure):
    _fields_ = [
        ("size", wt.DWORD),
        ("width", wt.LONG),
        ("height", wt.LONG),
        ("planes", wt.WORD),
        ("bit_count", wt.WORD),
        ("compression", wt.DWORD),
        ("image_size", wt.DWORD),
        ("x_resolution", wt.LONG),
        ("y_resolution", wt.LONG),
        ("colors_used", wt.DWORD),
        ("important_colors", wt.DWORD),
    ]


def _open_windows():
    # The independent capture workers import scene metadata too. Loading the
    # RapidShot COM declarations in a DXcam worker would invalidate that arm.
    from live_cpu_default import PaintStruct, Windows, _bind

    class RasterWindows(Windows):
        def __init__(self):
            super().__init__()
            self.images: dict[int, np.ndarray] = {}
            gdi = ct.WinDLL("gdi32", use_last_error=True)
            self.gdi_flush = _bind(gdi, "GdiFlush", wt.BOOL)
            self.dib_to_device = _bind(
                gdi,
                "SetDIBitsToDevice",
                ct.c_int,
                wt.HDC,
                ct.c_int,
                ct.c_int,
                wt.DWORD,
                wt.DWORD,
                ct.c_int,
                ct.c_int,
                wt.UINT,
                wt.UINT,
                wt.LPCVOID,
                ct.POINTER(_BitmapHeader),
                wt.UINT,
            )

        def _window_proc(self, handle, message, wparam, lparam):
            image = self.images.get(handle)
            if message != 0x000F or image is None:
                return super()._window_proc(handle, message, wparam, lparam)
            paint = PaintStruct()
            dc = self.begin_paint(handle, ct.byref(paint))
            if not dc:
                self.paint_error = ct.WinError(ct.get_last_error())
                return 0
            try:
                height, width = image.shape[:2]
                header = _BitmapHeader(
                    size=ct.sizeof(_BitmapHeader),
                    width=width,
                    height=-height,
                    planes=1,
                    bit_count=32,
                    image_size=image.nbytes,
                )
                if not self.dib_to_device(
                    dc, 0, 0, width, height, 0, 0, 0, height, image.ctypes.data, ct.byref(header), 0
                ):
                    raise OSError("GDI did not paint the benchmark dashboard")
            except Exception as error:
                self.paint_error = error
            finally:
                self.end_paint(handle, ct.byref(paint))
            return 0

        def repaint(self, handle: int, image: np.ndarray, regions: list[Rect] | None,
                    wait_for_desktop: bool):
            self.images[handle] = image
            if regions == []:
                return
            if regions:
                for rect in regions[:-1]:
                    if not self.redraw(handle, ct.byref(wt.RECT(*rect)), None, 0x0001):
                        raise ct.WinError(ct.get_last_error())
            rect = regions[-1] if regions else None
            if wait_for_desktop:
                self.present(handle, rect)
                return
            rect_pointer = None if rect is None else ct.byref(wt.RECT(*rect))
            if not self.redraw(handle, rect_pointer, None, 0x0001 | 0x0100):
                raise ct.WinError(ct.get_last_error())
            message = wt.MSG()
            while self.peek(ct.byref(message), handle, 0, 0, 1):
                self.translate(ct.byref(message))
                self.dispatch(ct.byref(message))
            if self.paint_error is not None:
                raise self.paint_error
            if not self.gdi_flush():
                raise ct.WinError(ct.get_last_error())

    return RasterWindows()


class ScenePlayer:
    def __init__(self, bounds: Rect):
        self.bounds = bounds
        left, top, right, bottom = bounds
        self.width, self.height = right - left, bottom - top
        self.stack = ExitStack()
        self.cover_stack = ExitStack()
        self.previous: Step | None = None
        self.overlay = None
        self.rasters: dict[int, np.ndarray] = {}
        self.covers: dict[int, np.ndarray] = {}

    def prepare(self, source):
        for steps in source.values():
            for step in steps:
                if step.visible_update:
                    self.rasters[id(step)] = render(step, self.width, self.height)
                if step.covers:
                    self.covers[id(step)] = _cover_image(step)

    def __enter__(self):
        try:
            self.windows = self.stack.enter_context(_open_windows())
            self.base = self.stack.enter_context(self.windows.scene(self.bounds, 0x181B25))
            self.stack.callback(self.cover_stack.close)
        except BaseException:
            self.stack.close()
            raise
        return self

    def __exit__(self, *exc):
        self.stack.__exit__(*exc)

    def present(self, step: Step, *, wait_for_desktop: bool = True, restart: bool = False) -> None:
        previous = self.previous
        switched = restart or previous is None or previous.workload != step.workload
        if switched and self.overlay is not None:
            self.windows.images.pop(self.overlay.handle, None)
            self.cover_stack.close()
            self.overlay = None
        if switched or step.visible_update:
            regions = None if switched or step.motion else _changed_regions(previous, step)
            image = self.rasters.get(id(step))
            if image is None:
                image = render(step, self.width, self.height)
            self.windows.repaint(self.base.handle, image, regions, wait_for_desktop)
        if step.covers:
            if step.cover_rect is None or step.cover_phase is None:
                raise ValueError("cover step has no bounds or phase")
            left, top, _, _ = self.bounds
            x1, y1, x2, y2 = step.cover_rect
            new_overlay = self.overlay is None
            if new_overlay:
                bounds = left + x1, top + y1, left + x2, top + y2
                self.overlay = self.cover_stack.enter_context(
                    self.windows.scene(bounds, 0x5C273D, excluded=True)
                )
            if new_overlay or previous.cover_phase != step.cover_phase:
                regions = (
                    None if new_overlay else [_stamp_rect((17, 34)), (17, 98, x2 - x1 - 15, 132)]
                )
                image = self.covers.get(id(step))
                if image is None:
                    image = _cover_image(step)
                self.windows.repaint(self.overlay.handle, image, regions, wait_for_desktop)
        self.previous = step
