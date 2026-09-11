"""Animated input for capture benchmarks; importing this module opens no window.

    python benchmarks/motion_source.py 60
    python benchmarks/motion_source.py 60 --display2 --fps 120

The default remains uncapped for comparison with historical runs. --fps is an
optional diagnostic limit. Inspect the achieved rate: a source
slower than the display can become the benchmark's limiting factor.
The AI harness uses --parent-controlled and keeps stdin open for the run's
lifetime. Closing that pipe (including parent exit) stops the animation.
"""

import argparse
import json
import math
import os
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


def animate(duration, fps=0.0, display2=False, stopped=None):
    emit("starting", fps_limit=fps)
    import tkinter as tk

    stopped = stopped if stopped is not None else threading.Event()
    root = None
    try:
        root = tk.Tk()
        root.title("motion")
        root.overrideredirect(True)
        root.geometry(f"900x700+{2120 if display2 else 200}+120")
        root.attributes("-topmost", True)
        root.protocol("WM_DELETE_WINDOW", stopped.set)
        canvas = tk.Canvas(root, width=900, height=700, highlightthickness=0, bg="#101018")
        canvas.pack()
        bars = [canvas.create_rectangle(0, 0, 0, 0, outline="") for _ in range(28)]
        frames = interval_frames = 0
        start = last_report = time.perf_counter()
        while not stopped.is_set():
            now = time.perf_counter()
            if duration is not None and now - start >= duration:
                break
            phase = (frames + 1) * 0.15
            for i, bar in enumerate(bars):
                x = (i * (900 / 28) + (phase * 40) % 900) % 900
                height = 60 + (i * 37 + int(phase * 60)) % 580
                canvas.coords(bar, x, (700 - height) / 2, x + 900 / 28 - 6,
                              (700 + height) / 2)
                shade = (i * 9 + int(phase * 30)) % 256
                canvas.itemconfig(bar, fill=f"#{shade:02x}{255 - shade:02x}c0")
            root.update_idletasks()
            root.update()
            frames += 1
            interval_frames += 1
            finished = time.perf_counter()
            if frames == 1:
                emit("ready", fps_limit=fps)
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
                args.display2, stopped)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        emit("error", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
