"""UNSUPPORTED -- do not quote this number. See ./README.md for why.

Targets d3dshot, an abandoned package. Device creation is inside the timed
window, and throughput depends on whatever happened to be on screen.

Supported entry points: perf_suite.py, compare_libraries.py, ai_ingestion.py,
section7.py, section7_suite.py, memory_profile.py.
"""
import sys as _sys
print("UNSUPPORTED BENCHMARK -- see benchmarks/unsupported/README.md", file=_sys.stderr)

import time
import d3dshot

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[D3DShot] FPS benchmark"
start_time = time.perf_counter()

fps = 0

# Create a screencapture instance using d3dshot
screencapture = d3dshot.create(capture_output="numpy")

start = time.perf_counter()
while fps < 1000:
    frame = screencapture.screenshot(region)
    if frame is not None:
        fps += 1

end_time = time.perf_counter() - start_time

print(f"{title}: {fps/end_time}")
screencapture.stop()