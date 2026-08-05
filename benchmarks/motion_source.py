"""On-screen motion for the capture benchmarks, fast enough not to be the limit.

Run this alongside `compare_libraries.py`. Desktop Duplication only reports
frames whose *content changed*, so with a still desktop every library returns
stale buffers instantly and reports meaningless four-figure FPS -- and with a
motion source that is merely slow, every library reports the motion source.

That second failure is the subtle one, and it happened here. An earlier
generator ticked on `root.after(33, ...)`: about 30 changes a second. Every
library measured 40-50 fps and the obvious conclusion -- "they are all at the
display's refresh ceiling, throughput is a tie" -- was wrong. The panel runs at
100 Hz; the benchmark was measuring tkinter's timer. With this generator the
same libraries reach 114-169 fps and separate clearly. **A benchmark's motion
source is part of the measurement apparatus and has to be calibrated like one.**

So this drives `update()` in a tight loop with no scheduled delay, repaints a
large multi-bar area that DWM cannot coalesce away, and **reports its own
achieved rate**. Check that number before trusting any capture figure: if the
source is not comfortably above the display's refresh rate, the capture results
describe this script.

    python benchmarks/motion_source.py 60
    python benchmarks/motion_source.py 60 --display2

Achieved ~610 updates/s on the dev machine, against a 100 Hz panel.
"""
import sys
import time
import tkinter as tk

DURATION = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
# DISPLAY1 (100 Hz) is at x=0; DISPLAY2 (60 Hz) starts at x=1920.
ORIGIN_X = 1920 if "--display2" in sys.argv else 0

W, H = 900, 700
root = tk.Tk()
root.title("motion")
root.overrideredirect(True)                    # no title bar to repaint around
root.geometry(f"{W}x{H}+{ORIGIN_X + 200}+120")
root.attributes("-topmost", True)
canvas = tk.Canvas(root, width=W, height=H, highlightthickness=0, bg="#101018")
canvas.pack()

# A grid of bars, all of which change every frame: a large, unambiguous delta
# that the compositor has to carry through rather than optimise away.
BARS = 28
bars = [canvas.create_rectangle(0, 0, 0, 0, outline="") for _ in range(BARS)]

frames = 0
start = time.perf_counter()
last_report = start
phase = 0.0

try:
    while True:
        now = time.perf_counter()
        if now - start >= DURATION:
            break
        phase += 0.15
        for i, bar in enumerate(bars):
            x = (i * (W / BARS) + (phase * 40) % W) % W
            h = 60 + (i * 37 + int(phase * 60)) % (H - 120)
            canvas.coords(bar, x, (H - h) / 2, x + W / BARS - 6, (H + h) / 2)
            shade = (i * 9 + int(phase * 30)) % 256
            canvas.itemconfig(bar, fill=f"#{shade:02x}{(255 - shade):02x}c0")
        root.update_idletasks()
        root.update()
        frames += 1
        if now - last_report >= 2.0:
            print(f"  motion source: {frames / (now - start):.1f} updates/s",
                  flush=True)
            last_report = now
except tk.TclError:
    pass

elapsed = time.perf_counter() - start
print(f"MOTION SOURCE ACHIEVED {frames / elapsed:.1f} updates/s "
      f"over {elapsed:.1f}s ({frames} updates)", flush=True)
try:
    root.destroy()
except Exception:
    pass
