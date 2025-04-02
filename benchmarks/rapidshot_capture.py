import time
import rapidshot
import argparse
import numpy as np

# Parse command line arguments
parser = argparse.ArgumentParser(description='Rapidshot Capture Benchmark')
parser.add_argument('--gpu', action='store_true', help='Use NVIDIA GPU acceleration')
parser.add_argument('--color', default='RGB', choices=['RGB', 'BGRA', 'RGBA', 'BGR', 'GRAY'],
                    help='Output color format')
parser.add_argument('--video-mode', action='store_true', help='Enable video mode')
parser.add_argument('--target-fps', type=int, default=60, help='Target FPS for capture')
parser.add_argument('--frames', type=int, default=1000, help='Number of frames to capture')
parser.add_argument('--region', nargs=4, type=int, default=[0, 0, 1920, 1080],
                    help='Capture region (left top right bottom)')
parser.add_argument('--as-numpy', action='store_true', help='Always convert to numpy arrays')
args = parser.parse_args()

# Setup capture region
LEFT, TOP, RIGHT, BOTTOM = args.region
region = (LEFT, TOP, RIGHT, BOTTOM)
title = "[Rapidshot] Capture Benchmark"

# Print configuration
gpu_mode = "GPU" if args.gpu else "CPU"
video_mode = "Enabled" if args.video_mode else "Disabled"
print(f"Starting {title} ({gpu_mode} mode, {args.color} format)")
print(f"Region: {region}, Video mode: {video_mode}, Target FPS: {args.target_fps}")

# Create screencapture with specified options
screencapture = rapidshot.create(output_idx=0, output_color=args.color, nvidia_gpu=args.gpu)

# Start performance timing
start_time = time.perf_counter()

# Start capture
screencapture.start(region=region, target_fps=args.target_fps, video_mode=args.video_mode)

# Get frames
frame_times = []
frames = []
for i in range(args.frames):
    frame_start = time.perf_counter()
    image = screencapture.get_latest_frame(as_numpy=args.as_numpy)
    frame_times.append(time.perf_counter() - frame_start)
    
    # Optionally store frame shapes or other metrics
    if i == 0:  # Just keep the first frame for verification
        frames.append(image.shape)

# Stop capture
screencapture.stop()

# Calculate metrics
end_time = time.perf_counter()
total_time = end_time - start_time
fps_rate = args.frames / total_time
avg_frame_time = np.mean(frame_times) * 1000  # in ms

print(f"{title} Results:")
print(f"- Total frames: {args.frames}")
print(f"- Time elapsed: {total_time:.2f} seconds")
print(f"- Average FPS: {fps_rate:.2f}")
print(f"- Average frame retrieval time: {avg_frame_time:.2f} ms")
if frames:
    print(f"- Frame shape: {frames[0]}")

# Clean up
del screencapture
rapidshot.clean_up()