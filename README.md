# RapidShot

A high-performance screencapture library for Windows using the Desktop Duplication API. This is a merged version combining features from multiple DXCam forks, designed to deliver ultra-fast capture capabilities with advanced functionality.

## Features

- **Ultra-fast capture**: 240Hz+ capturing capability
- **Multi-backend support**: NumPy, PIL, and CUDA/CuPy backends
- **Cursor capture**: Capture mouse cursor position and shape
- **Direct3D support**: Capture Direct3D exclusive full-screen applications without interruption
- **NVIDIA GPU acceleration**: GPU-accelerated processing using CuPy
- **Multi-monitor setup**: Support for multiple GPUs and monitors
- **Flexible output formats**: RGB, RGBA, BGR, BGRA, and grayscale support
- **Region-based capture**: Efficient capture of specific screen regions
- **Rotation handling**: Automatic handling of rotated displays

## Installation

> **Note:** The package is installed as `rapidshot` and imported as `import rapidshot`.

### Basic Installation

```bash
pip install rapidshot
```

### With OpenCV Support (recommended)

```bash
pip install rapidshot[cv2]
```

### With NVIDIA GPU Acceleration

```bash
pip install rapidshot[gpu]
```

### With All Dependencies

```bash
pip install rapidshot[all]
```

## Quick Start

### Basic Screencapture

```python
import rapidshot

# Create a ScreenCapture instance on the primary monitor
screencapture = rapidshot.create()

# Take a screencapture
frame = screencapture.grab()

# Display the screencapture
from PIL import Image
Image.fromarray(frame).show()
```

### Region-based Capture

```python
# Define a specific region
left, top = (1920 - 640) // 2, (1080 - 640) // 2
right, bottom = left + 640, top + 640
region = (left, top, right, bottom)

# Capture only this region
frame = screencapture.grab(region=region)  # 640x640x3 numpy.ndarray
```

### Continuous Capture

```python
# Start capturing at 60 FPS
screencapture.start(target_fps=60)

# Get the latest frame
for i in range(1000):
    image = screencapture.get_latest_frame()  # Blocks until new frame is available
    # Process the frame...

# Stop capturing
screencapture.stop()
```

### Video Recording

```python
import rapidshot
import cv2

# Create a ScreenCapture instance with BGR color format for OpenCV
screencapture = rapidshot.create(output_color="BGR")

# Start capturing at 30 FPS in video mode
screencapture.start(target_fps=30, video_mode=True)

# Create a video writer
writer = cv2.VideoWriter(
    "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
)

# Record for 10 seconds (300 frames at 30 FPS)
for i in range(300):
    writer.write(screencapture.get_latest_frame())

# Clean up
screencapture.stop()
writer.release()
```

### NVIDIA GPU Acceleration

```python
# Create a ScreenCapture instance with NVIDIA GPU acceleration
screencapture = rapidshot.create(nvidia_gpu=True)

# Screenshots will be processed on the GPU for improved performance
frame = screencapture.grab()
```

### Cursor Capture

RapidShot provides comprehensive cursor capture capabilities, allowing you to track cursor position, visibility, and shape in your screen captures.

```python
# Take a screenshot
frame = screencapture.grab()

# Get cursor information
cursor = screencapture.grab_cursor()

# Check if cursor is visible in the capture area
if cursor.PointerPositionInfo.Visible:
    # Get cursor position
    x, y = cursor.PointerPositionInfo.Position.x, cursor.PointerPositionInfo.Position.y
    print(f"Cursor position: ({x}, {y})")
    
    # Cursor shape information is also available
    if cursor.Shape is not None:
        width = cursor.PointerShapeInfo.Width
        height = cursor.PointerShapeInfo.Height
        print(f"Cursor size: {width}x{height}")
```

#### Advanced Cursor Handling

The cursor information provided by RapidShot can be used in various ways:

1. **Overlay cursor on captured image:**

```python
import numpy as np
import cv2

def overlay_cursor(frame, cursor):
    """Overlay cursor on captured frame."""
    if not cursor.PointerPositionInfo.Visible or cursor.Shape is None:
        return frame
    
    # Create an overlay from cursor shape data
    shape_type = cursor.PointerShapeInfo.Type
    width = cursor.PointerShapeInfo.Width
    height = cursor.PointerShapeInfo.Height
    
    # Different processing based on cursor type (monochrome, color, or masked)
    if shape_type & DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MONOCHROME:
        # Process monochrome cursor
        # ...
    elif shape_type & DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR:
        # Process color cursor
        # ...
    elif shape_type & DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR:
        # Process masked color cursor
        # ...
    
    # Position the cursor on the frame at its current coordinates
    x, y = cursor.PointerPositionInfo.Position.x, cursor.PointerPositionInfo.Position.y
    
    # Ensure cursor is within frame boundaries
    # ...
    
    # Blend cursor with frame
    # ...
    
    return frame_with_cursor

# Usage example
frame = screencapture.grab()
cursor = screencapture.grab_cursor()
composite_image = overlay_cursor(frame, cursor)
```

2. **Track cursor movements:**

```python
import time

# Record cursor positions over time
positions = []
screencapture = rapidshot.create()

for i in range(100):
    cursor = screencapture.grab_cursor()
    if cursor.PointerPositionInfo.Visible:
        positions.append((
            time.time(),
            cursor.PointerPositionInfo.Position.x,
            cursor.PointerPositionInfo.Position.y
        ))
    time.sleep(0.05)  # Sample at 20Hz

# Analyze cursor movement
# ...
```

## Multiple Monitors / GPUs

```python
# Show available devices and outputs
print(rapidshot.device_info())
print(rapidshot.output_info())

# Create ScreenCapture instances for specific devices/outputs
capture1 = rapidshot.create(device_idx=0, output_idx=0)  # First monitor on first GPU
capture2 = rapidshot.create(device_idx=0, output_idx=1)  # Second monitor on first GPU
capture3 = rapidshot.create(device_idx=1, output_idx=0)  # First monitor on second GPU
```

## Advanced Usage

### Custom Buffer Size

```python
# Create a ScreenCapture instance with a larger frame buffer
screencapture = rapidshot.create(max_buffer_len=256)
```

### Different Color Formats

```python
# RGB (default)
screencapture_rgb = rapidshot.create(output_color="RGB")

# RGBA (with alpha channel)
screencapture_rgba = rapidshot.create(output_color="RGBA")

# BGR (OpenCV format)
screencapture_bgr = rapidshot.create(output_color="BGR")

# Grayscale
screencapture_gray = rapidshot.create(output_color="GRAY")
```

### Resource Management

```python
# Release resources when done
screencapture.release()

# Or automatically released when object is deleted
del screencapture

# Clean up all resources
rapidshot.clean_up()

# Reset the library completely
rapidshot.reset()
```

## Benchmarks and Performance Comparison

RapidShot includes benchmark utilities to compare its performance against other popular screen capture libraries. The benchmark scripts are located in the `benchmarks/` directory and are designed to provide objective performance measurements.

### Benchmark Structure

- **FPS Benchmarks**: Measure the maximum frame rate achievable by each library
  - `rapidshot_max_fps.py` - Tests RapidShot's maximum FPS
  - `bettercam_max_fps.py` - Tests BetterCam's maximum FPS
  - `dxcam_max_fps.py` - Tests DXCam's maximum FPS
  - `d3dshot_max_fps.py` - Tests D3DShot's maximum FPS
  - `mss_max_fps.py` - Tests MSS's maximum FPS

- **Capture Benchmarks**: Test the continuous capture performance
  - `rapidshot_capture.py` - Tests RapidShot's continuous capture
  - `bettercam_capture.py` - Tests BetterCam's continuous capture
  - `dxcam_capture.py` - Tests DXCam's continuous capture

### Running Benchmarks

To run a benchmark comparison:

```bash
# Run RapidShot benchmark
python benchmarks/rapidshot_max_fps.py

# Run with GPU acceleration
python benchmarks/rapidshot_max_fps.py --gpu

# Test with different color formats
python benchmarks/rapidshot_max_fps.py --color BGRA
```

### Benchmark Results

The table below shows typical performance results across different libraries:

| Library         | Average FPS | GPU-accelerated FPS |
|-----------------|-------------|---------------------|
| RapidShot       | 240+        | 300+                |
| Original DXCam  | 210         | N/A                 |
| Python-MSS      | 75          | N/A                 |
| D3DShot         | 118         | N/A                 |

## System Requirements

- **Operating System:** Windows 10 or newer
- **Python:** 3.7+
- **GPU:** Compatible GPU for NVIDIA acceleration (for GPU features)
- **RAM:** 8 GB+ (depending on the resolution and number of screencapture instances used)

### Troubleshooting

- **ImportError with CuPy:** Ensure you have compatible CUDA drivers installed.
- **Black screens when capturing:** Verify the application isn't running in exclusive fullscreen mode.
- **Low performance:** Experiment with different backends (NUMPY vs. CUPY) to optimize performance.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

RapidShot is a merged version of the following projects:

- Original DXcam by ra1nty
- dxcampil - PIL-based version
- DXcam-AI-M-BOT - Cursor support version
- BetterCam - GPU acceleration version