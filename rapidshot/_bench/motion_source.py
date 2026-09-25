"""Animated input for capture benchmarks; importing this module opens no window.

    python benchmarks/motion_source.py 60
    python benchmarks/motion_source.py 60 --fps 120
    python benchmarks/motion_source.py 60 --window 900x700+200+120

**It covers the whole screen by default, and that is a correction.** It used to
animate a hardcoded 900x700 window at +200+120. A full-screen capture benchmark
driven by it therefore measured a screen of which **15% changed** on a
2560x1600 display and 30% on a 1080p one -- so Desktop Duplication reported a
small dirty rectangle, every capture path did far less work than it would on a
real workload, and the size of the discount depended on the monitor. That is
not a workload anyone would recognise, and two machines running the identical
command were not running the same benchmark.

``--window`` keeps the old behaviour for the few tests that need a source
smaller than the screen, and the animated rectangle is reported in the
``ready`` event either way so a consumer can record what fraction of its
capture was actually moving.

The default remains uncapped for comparison with historical runs. --fps is an
optional diagnostic limit. Inspect the achieved rate: a source slower than the
display can become the benchmark's limiting factor. The AI harness uses
--parent-controlled and keeps stdin open for the run's lifetime. Closing that
pipe (including parent exit) stops the animation.

This is a Tk source, kept for the harnesses built around it. `section7.py` uses
the D3D source in `native/src/bin/latency_source.rs`, which is the one with the
frame-ID marker the pixel-age clock needs.
"""

import argparse
import json
import math
import os
import re
import sys
import threading
import time


def emit(event, **fields):
    print(json.dumps({"event": event, "pid": os.getpid(), **fields}), flush=True)
    try:
        os.fsync(sys.stdout.fileno())
    except (OSError, ValueError):
        pass  # Pipes and consoles need not support fsync.


def watch_parent(stream, stopped):
    try:
        stream.readline()  # STOP, EOF, or a broken pipe end ownership.
    finally:
        stopped.set()


def frame_delay(started, fps, now):
    return max(0.0, started + 1.0 / fps - now) if fps else 0.0


def parse_window(text):
    """``WxH+X+Y`` -> ``(width, height, left, top)``.

    Explicit rather than four separate flags: a window is one thing, and three
    of four numbers supplied is a mistake nobody wants to debug from a
    screenshot.
    """
    match = re.fullmatch(r"(\d+)x(\d+)([+-]-?\d+)([+-]-?\d+)", text.strip())
    if not match:
        raise ValueError(f"expected WxH+X+Y, got {text!r}")
    width, height, left, top = match.groups()
    # X geometry writes a negative offset either way round: "-100" and "+-100"
    # both mean -100, and a string split on "+" gets one of them wrong.
    return (int(width), int(height),
            int(left.replace("+-", "-").lstrip("+")),
            int(top.replace("+-", "-").lstrip("+")))


def screen_size(root):
    """The screen in physical pixels.

    ``python.exe`` declares per-monitor DPI awareness in its manifest, so Tk's
    screen metrics are already physical and need no correction. Checked on a
    150%-scaled 2560x1600 panel: `winfo_screenwidth` reports 2560, not 1707.
    """
    return root.winfo_screenwidth(), root.winfo_screenheight()


def animate(duration, fps=0.0, display2=False, stopped=None, window=None):
    emit("starting", fps_limit=fps)
    import tkinter as tk

    stopped = stopped if stopped is not None else threading.Event()
    root = None
    try:
        root = tk.Tk()
        root.title("motion")
        root.overrideredirect(True)
        if window is None:
            # The whole screen. A capture benchmark that animates a corner of
            # the display is measuring a mostly-still desktop, whatever it says
            # in the results file.
            width, height = screen_size(root)
            left = top = 0
        else:
            width, height, left, top = window
        if display2:
            # Historical flag: push the window onto a second display by
            # offsetting past the first. Only meaningful with --window.
            left += screen_size(root)[0]
        root.geometry(f"{width}x{height}+{left}+{top}")
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", stopped.set)
        canvas = tk.Canvas(root, width=width, height=height, highlightthickness=0,
                           bg="#101018")
        canvas.pack()
        # Bars scale with the canvas so a full-screen source is not 28 slivers
        # on a 2560px display; roughly one per 90 px, as the original had.
        count = max(12, min(96, width // 90))
        bars = [canvas.create_rectangle(0, 0, 0, 0, outline="") for _ in range(count)]
        span = width / count
        frames = interval_frames = 0
        start = last_report = time.perf_counter()
        while not stopped.is_set():
            now = time.perf_counter()
            if duration is not None and now - start >= duration:
                break
            phase = (frames + 1) * 0.15
            for i, bar in enumerate(bars):
                x = (i * span + (phase * 40) % width) % width
                bar_height = 60 + (i * 37 + int(phase * 60)) % max(80, height - 120)
                canvas.coords(bar, x, (height - bar_height) / 2, x + span - 6,
                              (height + bar_height) / 2)
                shade = (i * 9 + int(phase * 30)) % 256
                canvas.itemconfig(bar, fill=f"#{shade:02x}{255 - shade:02x}c0")
            root.update_idletasks()
            root.update()
            frames += 1
            interval_frames += 1
            finished = time.perf_counter()
            if frames == 1:
                # The animated rectangle travels with the readiness event, so a
                # consumer can record what fraction of its capture was moving
                # instead of assuming all of it was.
                emit("ready", fps_limit=fps, rect=[left, top, left + width,
                                                   top + height],
                     screen=list(screen_size(root)), fullscreen=window is None)
            if finished - last_report >= 2.0:
                emit("rate", updates_per_second=interval_frames / (finished - last_report),
                     frames=frames)
                last_report, interval_frames = finished, 0
            delay = frame_delay(now, fps, finished)
            if delay:
                stopped.wait(delay)
        elapsed = time.perf_counter() - start
        emit("stopped", frames=frames, elapsed_seconds=elapsed,
             updates_per_second=frames / elapsed if elapsed else 0.0)
    finally:
        if root is not None:
            root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seconds", nargs="?", type=float, default=30.0)
    parser.add_argument("--display2", action="store_true")
    parser.add_argument("--window", type=parse_window, metavar="WxH+X+Y",
                        help="animate a window instead of the whole screen. The "
                             "default is full screen; a partial window means a "
                             "capture benchmark measures a mostly-still desktop.")
    parser.add_argument("--fps", type=float, default=0.0, help="0 is uncapped")
    parser.add_argument("--parent-controlled", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    if not math.isfinite(args.fps) or args.fps < 0:
        parser.error("fps must be finite and nonnegative")
    stopped = threading.Event()
    if args.parent_controlled:
        threading.Thread(target=watch_parent, args=(sys.stdin, stopped), daemon=True).start()
    try:
        animate(None if args.parent_controlled else args.seconds, args.fps,
                args.display2, stopped, args.window)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit("error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
