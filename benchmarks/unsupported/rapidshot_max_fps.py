"""UNSUPPORTED -- do not quote this number. See ./README.md for why.

Device creation, a warm-up grab and a 100 ms sleep are all inside the timed
window -- a handicap the competitor scripts do not carry, so these numbers
cannot be compared with theirs in either direction.

Supported entry points: perf_suite.py, compare_libraries.py, ai_ingestion.py,
section7.py, section7_suite.py, memory_profile.py.
"""
import sys as _sys
print("UNSUPPORTED BENCHMARK -- see benchmarks/unsupported/README.md", file=_sys.stderr)

import time
import rapidshot
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Rapidshot Max FPS Benchmark')
parser.add_argument('--gpu', action='store_true', help='Use NVIDIA GPU acceleration')
parser.add_argument('--color', default='RGB', choices=['RGB', 'BGRA', 'RGBA', 'BGR', 'GRAY'],
                    help='Output color format')
parser.add_argument('--region', nargs=4, type=int, default=[0, 0, 1920, 1080],
                    help='Capture region (left top right bottom)')
args = parser.parse_args()

# Setup capture region
LEFT, TOP, RIGHT, BOTTOM = args.region
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[Rapidshot] Max FPS Benchmark"

# Print configuration
gpu_mode = "GPU" if args.gpu else "CPU"
print(f"Starting {title} ({gpu_mode} mode, {args.color} format)")
print(f"Region: {region}")

# Benchmark code
start_time = time.perf_counter()
fps = 0

# Create screencapture with specified options
screencapture = rapidshot.create(output_color=args.color, nvidia_gpu=args.gpu)

# Warm-up (some systems need this for stable measurements)
_ = screencapture.grab(region=region)
time.sleep(0.1)

# Run benchmark
while fps < 1000:
    frame = screencapture.grab(region=region)
    if frame is not None:
        fps += 1
        # For very fast iterations, add a micro-sleep to prevent 100% CPU usage
        # time.sleep(0.001)

end_time = time.perf_counter() - start_time
fps_rate = fps / end_time

print(f"{title} Results:")
print(f"- Total frames: {fps}")
print(f"- Time elapsed: {end_time:.2f} seconds")
print(f"- Average FPS: {fps_rate:.2f}")

# Clean up
del screencapture
rapidshot.clean_up()