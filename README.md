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
## System Requirements

- **Operating System:** Windows 10 or newer
- **Python:** 3.7+
- **GPU:** Compatible GPU for NVIDIA acceleration (for GPU features)
- **RAM:** 8 GB+ (depending on the resolution and number of screencapture instances used)

| Library         | Average FPS | GPU-accelerated FPS |
|-----------------|-------------|---------------------|
| RapidShot       | 240+        | 300+                |
| Original DXCam  | 210         | N/A                 |
| Python-MSS      | 75          | N/A                 |
| D3DShot         | 118         | N/A                 |

### Troubleshooting

- **ImportError with CuPy:** Ensure you have compatible CUDA drivers installed.
- **Black screens when capturing:** Verify the application isn’t running in exclusive fullscreen mode.
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