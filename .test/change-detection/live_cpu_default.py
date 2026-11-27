"""Controlled scenes captured by the unmodified synchronous RapidShot CPU path."""

from __future__ import annotations

import ctypes as ct
from ctypes import wintypes as wt
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import os
import time
from typing import Callable

import numpy as np
import rapidshot
from rapidshot.core.hdr import DXGI_FORMAT_B8G8R8A8_UNORM


WIDTH, HEIGHT = 1920, 1080
FULL = (0, 0, WIDTH, HEIGHT)
SPARSE = (20, 30, 68, 78)
COVER = (700, 360, 1220, 720)
UNDER = (840, 450, 915, 520)
SCENES = ("sparse", "full_motion", "excluded_cover", "under_cover")
PAIRS = 10
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class LiveFrame:
    pixels: np.ndarray
    dirty_rects: tuple[Rect, ...] | None
    move_rects: tuple[tuple[int, ...], ...] | None
    serial: int
    source_id: int
    accumulated_frames: int
    grab_us: float
    rects_coalesced: bool
    continuous: bool
    region: Rect = FULL


FrameHook = Callable[[str, LiveFrame, LiveFrame | None], bool]


class PaintStruct(ct.Structure):
    _fields_ = [
        ("dc", wt.HDC), ("erase", wt.BOOL), ("rect", wt.RECT),
        ("restore", wt.BOOL), ("inc_update", wt.BOOL), ("reserved", ct.c_byte * 32),
    ]


WndProc = ct.WINFUNCTYPE(ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)


class WndClass(ct.Structure):
    _fields_ = [
        ("style", wt.UINT), ("wndproc", WndProc), ("class_extra", ct.c_int),
        ("window_extra", ct.c_int), ("instance", wt.HINSTANCE), ("icon", wt.HANDLE),
        ("cursor", wt.HANDLE), ("background", wt.HBRUSH), ("menu", wt.LPCWSTR),
        ("name", wt.LPCWSTR),
    ]


def _bind(library, name, result, *args):
    function = getattr(library, name)
    function.restype = result
    function.argtypes = args
    return function


class Windows:
    """Nonactivating GDI windows. The test never moves the user's cursor."""

    def __init__(self):
        user = ct.WinDLL("user32", use_last_error=True)
        gdi = ct.WinDLL("gdi32", use_last_error=True)
        kernel = ct.WinDLL("kernel32", use_last_error=True)
        dwm = ct.WinDLL("dwmapi", use_last_error=True)
        self.set_dpi = _bind(user, "SetThreadDpiAwarenessContext", wt.HANDLE, wt.HANDLE)
        self.get_module = _bind(kernel, "GetModuleHandleW", wt.HMODULE, wt.LPCWSTR)
        self.register = _bind(user, "RegisterClassW", wt.ATOM, ct.POINTER(WndClass))
        self.unregister = _bind(user, "UnregisterClassW", wt.BOOL, wt.LPCWSTR, wt.HINSTANCE)
        self.create = _bind(
            user, "CreateWindowExW", wt.HWND, wt.DWORD, wt.LPCWSTR, wt.LPCWSTR, wt.DWORD,
            ct.c_int, ct.c_int, ct.c_int, ct.c_int, wt.HWND, wt.HMENU, wt.HINSTANCE, wt.LPVOID,
        )
        self.destroy = _bind(user, "DestroyWindow", wt.BOOL, wt.HWND)
        self.show = _bind(user, "ShowWindow", wt.BOOL, wt.HWND, ct.c_int)
        self.affinity = _bind(user, "SetWindowDisplayAffinity", wt.BOOL, wt.HWND, wt.DWORD)
        self.redraw = _bind(user, "RedrawWindow", wt.BOOL, wt.HWND, ct.POINTER(wt.RECT), wt.HANDLE, wt.UINT)
        self.peek = _bind(user, "PeekMessageW", wt.BOOL, ct.POINTER(wt.MSG), wt.HWND, wt.UINT, wt.UINT, wt.UINT)
        self.translate = _bind(user, "TranslateMessage", wt.BOOL, ct.POINTER(wt.MSG))
        self.dispatch = _bind(user, "DispatchMessageW", ct.c_ssize_t, ct.POINTER(wt.MSG))
        self.flush = _bind(dwm, "DwmFlush", ct.c_long)
        self.default_proc = _bind(user, "DefWindowProcW", ct.c_ssize_t, wt.HWND, wt.UINT, wt.WPARAM, wt.LPARAM)
        self.begin_paint = _bind(user, "BeginPaint", wt.HDC, wt.HWND, ct.POINTER(PaintStruct))
        self.end_paint = _bind(user, "EndPaint", wt.BOOL, wt.HWND, ct.POINTER(PaintStruct))
        self.fill_rect = _bind(user, "FillRect", ct.c_int, wt.HDC, ct.POINTER(wt.RECT), wt.HBRUSH)
        self.brush = _bind(gdi, "CreateSolidBrush", wt.HBRUSH, wt.DWORD)
        self.delete = _bind(gdi, "DeleteObject", wt.BOOL, wt.HANDLE)
        self.scenes: dict[int, Scene] = {}
        self.name = f"RapidShotCpuDefaultBenchmark-{os.getpid()}"
        self.instance = self.get_module(None)
        self.callback = WndProc(self._window_proc)
        self.paint_error: Exception | None = None

    def __enter__(self):
        self.previous_dpi = self.set_dpi(ct.c_void_p(-4))
        if not self.previous_dpi:
            raise ct.WinError(ct.get_last_error())
        klass = WndClass(wndproc=self.callback, instance=self.instance, name=self.name)
        if not self.register(ct.byref(klass)):
            error = ct.WinError(ct.get_last_error())
            self.set_dpi(self.previous_dpi)
            raise error
        return self

    def __exit__(self, *_):
        try:
            self.unregister(self.name, self.instance)
        finally:
            self.set_dpi(self.previous_dpi)

    def _fill(self, dc, rect: wt.RECT, rgb: int):
        colorref = ((rgb & 0xFF) << 16) | (rgb & 0xFF00) | (rgb >> 16)
        brush = self.brush(colorref)
        if not brush:
            raise ct.WinError(ct.get_last_error())
        try:
            if not self.fill_rect(dc, ct.byref(rect), brush):
                raise ct.WinError(ct.get_last_error())
        finally:
            self.delete(brush)

    def _window_proc(self, handle, message, wparam, lparam):
        scene = self.scenes.get(handle)
        if message != 0x000F or scene is None:  # WM_PAINT
            return self.default_proc(handle, message, wparam, lparam)
        paint = PaintStruct()
        dc = self.begin_paint(handle, ct.byref(paint))
        if not dc:
            self.paint_error = ct.WinError(ct.get_last_error())
            return 0
        try:
            self._fill(dc, paint.rect, scene.background)
            if scene.patch is not None:
                rect, rgb = scene.patch
                self._fill(dc, wt.RECT(*rect), rgb)
        except Exception as error:
            self.paint_error = error
        finally:
            self.end_paint(handle, ct.byref(paint))
        return 0

    def present(self, handle, rect: Rect | None):
        region = None if rect is None else ct.byref(wt.RECT(*rect))
        if not self.redraw(handle, region, None, 0x0001 | 0x0100):  # RDW_INVALIDATE | RDW_UPDATENOW
            raise ct.WinError(ct.get_last_error())
        message = wt.MSG()
        while self.peek(ct.byref(message), handle, 0, 0, 1):  # PM_REMOVE
            self.translate(ct.byref(message))
            self.dispatch(ct.byref(message))
        if self.paint_error is not None:
            raise self.paint_error
        result = self.flush()
        if result < 0:
            raise OSError(f"DwmFlush failed with HRESULT {result & 0xFFFFFFFF:#010x}")

    @contextmanager
    def scene(self, bounds: Rect, background: int, excluded: bool = False):
        left, top, right, bottom = bounds
        ex_style = 0x08000000 | 0x00000080 | 0x00000008  # NOACTIVATE | TOOLWINDOW | TOPMOST
        handle = self.create(
            ex_style, self.name, "RapidShot benchmark", 0x80000000,  # WS_POPUP
            left, top, right - left, bottom - top, None, None, self.instance, None,
        )
        if not handle:
            raise ct.WinError(ct.get_last_error())
        scene = Scene(self, handle, background)
        self.scenes[handle] = scene
        try:
            if excluded and not self.affinity(handle, 0x11):  # WDA_EXCLUDEFROMCAPTURE
                raise ct.WinError(ct.get_last_error())
            self.show(handle, 4)  # SW_SHOWNOACTIVATE
            self.present(handle, None)
            yield scene
        finally:
            self.destroy(handle)
            del self.scenes[handle]


@dataclass
class Scene:
    windows: Windows
    handle: int
    background: int
    patch: tuple[Rect, int] | None = None

    def paint(self, color: int, rect: Rect | None = None):
        previous = self.patch
        self.patch = None if rect is None else (rect, color)
        if rect is None:
            self.background = color
        # A changed patch location needs its old pixels cleared as well.
        repaint = None if previous is not None and previous[0] != rect else rect
        self.windows.present(self.handle, repaint)


def _secondary_display():
    factory = rapidshot.get_factory()
    options = []
    for dev, outputs in enumerate(factory.outputs):
        adapter = str(factory.devices[dev].desc.Description)
        if "intel" not in adapter.lower():
            continue
        for index, output in enumerate(outputs):
            metadata = factory.output_metadata.get(output.devicename)
            if (output.attached_to_desktop and output.resolution == (WIDTH, HEIGHT)
                    and output.rotation_angle == 0 and metadata is not None and not metadata[1]):
                coord = output.desc.DesktopCoordinates
                bounds = (coord.left, coord.top, coord.right, coord.bottom)
                options.append((dev, index, output.devicename, adapter, bounds))
    if not options:
        raise RuntimeError("Expected an attached secondary unrotated Intel 1920x1080 display")
    return options[0]


def _center(rect: Rect) -> tuple[int, int]:
    left, top, right, bottom = rect
    return (left + right) // 2, (top + bottom) // 2


def _translated(rect: Rect, monitor: Rect) -> Rect:
    left, top, _, _ = monitor
    return rect[0] + left, rect[1] + top, rect[2] + left, rect[3] + top


def _rects(source):
    return None if source is None else tuple(tuple(int(v) for v in rect) for rect in source)


def _cursor_state(duplicator):
    info = getattr(getattr(duplicator, "cursor", None), "PointerPositionInfo", None)
    if info is None:
        return None
    return int(info.Position.x), int(info.Position.y), bool(info.Visible)


def _release(owner):
    release = getattr(owner, "release", None)
    if release is not None:
        release()


@dataclass
class Count:
    scene_steps: int = 0
    verified_frames: int = 0
    published_frames: int = 0
    no_image_polls: int = 0
    cursor_only_polls: int = 0
    rejected_marker_frames: int = 0
    accumulated_frames: int = 0
    coalesced_frames: int = 0
    discontinuities: int = 0


class Recorder:
    def __init__(self, camera, on_frame: FrameHook | None, retain_trace: bool):
        self.camera = camera
        self.on_frame = on_frame
        self.retain_trace = retain_trace
        self.trace: dict[str, list[LiveFrame]] = {}
        self.previous_owner = None
        self.previous_delivered: LiveFrame | None = None
        self.previous_observed: tuple[int, int] | None = None

    def close(self):
        _release(self.previous_owner)
        self.previous_owner = self.previous_delivered = self.previous_observed = None

    def begin(self, name: str):
        self.close()
        self.name = name
        self.count = Count()
        self.trace[name] = []

    def _grab(self):
        prior_cursor = _cursor_state(self.camera._duplicator)
        start = time.perf_counter_ns()
        owner = self.camera.grab()
        grab_us = (time.perf_counter_ns() - start) / 1_000
        if owner is None:
            self.count.no_image_polls += 1
            if _cursor_state(self.camera._duplicator) != prior_cursor:
                self.count.cursor_only_polls += 1
        return owner, grab_us

    def _pixels(self, owner) -> np.ndarray:
        if self.camera._stagesurf.dxgi_format != DXGI_FORMAT_B8G8R8A8_UNORM:
            raise RuntimeError("Expected the native SDR BGRA8 staging path")
        pixels = np.asarray(owner)
        if (pixels.shape != (HEIGHT, WIDTH, 4) or pixels.dtype != np.uint8
                or not pixels.flags.c_contiguous):
            raise RuntimeError(f"Unexpected RapidShot delivery: {pixels.shape} {pixels.dtype}")
        view = pixels.view()
        view.setflags(write=False)
        return view

    @staticmethod
    def _matches(pixels, markers):
        for (x, y), rgb in markers:
            color = tuple(int(channel) for channel in pixels[y, x, :3])
            if color != (rgb & 0xFF, (rgb >> 8) & 0xFF, rgb >> 16):
                return False
        return True

    def observe(self, markers, max_polls: int) -> bool:
        self.count.scene_steps += 1
        for _ in range(max_polls):
            owner, grab_us = self._grab()
            if owner is None:
                continue
            try:
                pixels = self._pixels(owner)
                if not self._matches(pixels, markers):
                    self.count.rejected_marker_frames += 1
                    continue
                dup = self.camera._duplicator
                source, serial = int(dup.instance_id), int(dup.frame_serial)
                accumulated = int(dup.accumulated_frames)
                coalesced = bool(dup.rects_coalesced)
                continuous = self.previous_observed == (source, serial - 1) and accumulated <= 1
                if self.previous_observed is not None and not continuous:
                    self.count.discontinuities += 1
                self.previous_observed = (source, serial)
                self.count.verified_frames += 1
                self.count.accumulated_frames += int(accumulated > 1)
                self.count.coalesced_frames += int(coalesced)
                frame = LiveFrame(
                    pixels=pixels, dirty_rects=_rects(dup.dirty_rects), move_rects=_rects(dup.move_rects),
                    serial=serial, source_id=source, accumulated_frames=accumulated, grab_us=grab_us,
                    rects_coalesced=coalesced, continuous=continuous,
                )
                publish = self.on_frame is None or self.on_frame(self.name, frame, self.previous_delivered)
                if self.retain_trace:
                    snapshot = pixels.copy()
                    snapshot.setflags(write=False)
                    self.trace[self.name].append(replace(frame, pixels=snapshot))
                if publish:
                    _release(self.previous_owner)
                    self.previous_owner, self.previous_delivered = owner, frame
                    owner = None
                    self.count.published_frames += 1
                return True
            finally:
                _release(owner)
        return False

    def idle(self, polls: int):
        for _ in range(polls):
            owner, _ = self._grab()
            if owner is not None:
                self.count.verified_frames += 1
                _release(owner)


def _run_scene(recorder: Recorder, base: Scene, windows: Windows, bounds: Rect, name: str):
    recorder.begin(name)
    far = (WIDTH - 54, HEIGHT - 54)
    near = _center(SPARSE)
    under = _center(UNDER)
    inside_cover = (COVER[0] + 35, COVER[1] + 35)
    background = (0x345675, 0x2B5678, 0x375371, 0x2E5574)[SCENES.index(name)]
    colors = (0x873514, 0x187654)
    base.paint(background)

    def collect(cover: Scene | None):
        initial = ((near, background), (under, background), (inside_cover, background), (far, background))
        if not recorder.observe(initial, 50):
            raise RuntimeError(f"No verified baseline reached for {name} on the secondary display")
        missed = 0
        for step in range(PAIRS + 6):
            color = colors[step % len(colors)]
            if name == "sparse":
                base.paint(color, SPARSE)
                markers = ((near, color), (far, background))
            elif name == "full_motion":
                base.paint(color)
                markers = ((near, color), (far, color), (under, color))
            elif name == "excluded_cover":
                cover.paint(color)
                markers = initial
            else:
                base.paint(color, UNDER)
                markers = ((under, color), (inside_cover, background), (far, background))
            if not recorder.observe(markers, 3 if name == "excluded_cover" else 12):
                missed += 1
            if recorder.count.verified_frames - 1 >= PAIRS:
                break
        return {**asdict(recorder.count), "pairs": recorder.count.verified_frames - 1, "missed_scene_steps": missed}

    if name in ("excluded_cover", "under_cover"):
        with windows.scene(_translated(COVER, bounds), 0x8B35AA, excluded=True) as cover:
            return collect(cover)
    return collect(None)


def capture_trace(
    on_frame: FrameHook | None = None,
    order: tuple[str, ...] | list[str] | None = None,
    retain_trace: bool = True,
) -> tuple[dict[str, list[LiveFrame]], dict]:
    """Capture verified scenes. The hook's borrowed views are valid only inside the call."""
    scene_order = tuple(SCENES if order is None else order)
    if len(scene_order) != len(SCENES) or set(scene_order) != set(SCENES):
        raise ValueError(f"Expected a permutation of {SCENES}")
    with Windows() as windows:
        dev, index, output, adapter, bounds = _secondary_display()
        camera = rapidshot.create(
            device_idx=dev, output_idx=index, backend="dxgi", output_color="BGRA",
            nvidia_gpu=False, pool_output=True,
        )
        recorder = Recorder(camera, on_frame, retain_trace)
        try:
            if camera.region != FULL or camera.rotation_angle != 0:
                raise RuntimeError(f"Unexpected live camera geometry {camera.region} at {camera.rotation_angle} degrees")
            stats = {}
            with windows.scene(bounds, 0x24455F) as base:
                recorder.begin("idle")
                recorder.idle(2)  # Discard initial present before timed idle observations.
                recorder.begin("idle")
                recorder.idle(6)
                stats["idle"] = asdict(recorder.count)
                for name in scene_order:
                    stats[name] = _run_scene(recorder, base, windows, bounds, name)
            recorder.trace.pop("idle", None)
            return recorder.trace, {
                "adapter": adapter, "output": output, "bounds": bounds, "order": scene_order,
                "duplicator": type(camera._duplicator).__name__, "captured_format": "B8G8R8A8_UNORM",
                "retained_trace": retain_trace, "workloads": stats,
            }
        finally:
            recorder.close()
            camera.release()
