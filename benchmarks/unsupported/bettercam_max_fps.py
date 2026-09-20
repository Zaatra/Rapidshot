"""UNSUPPORTED -- do not quote this number. See ./README.md for why.

Device creation is inside the timed window, and throughput depends on
whatever happened to be on screen. Single pass, mean only, no machine record.

Supported entry points: perf_suite.py, compare_libraries.py, ai_ingestion.py,
section7.py, section7_suite.py, memory_profile.py.
"""
import sys as _sys
print("UNSUPPORTED BENCHMARK -- see benchmarks/unsupported/README.md", file=_sys.stderr)

import time
import bettercam

TOP = 0
LEFT = 0
RIGHT = 1920
BOTTOM = 1080
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[bettercam] FPS benchmark"

start_time = time.perf_counter()

fps = 0
screencapture = bettercam.create()
start = time.perf_counter()
while fps < 1000:
    frame = screencapture.grab(region=region)
    if frame is not None:
        fps += 1

end_time = time.perf_counter() - start_time

print(f"{title}: {fps/end_time}")
del screencapture