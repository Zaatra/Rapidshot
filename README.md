# RapidShot

[![CI](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml/badge.svg)](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rapidshot)](https://pypi.org/project/rapidshot/)
[![Python](https://img.shields.io/pypi/pyversions/rapidshot)](https://pypi.org/project/rapidshot/)
[![License](https://img.shields.io/pypi/l/rapidshot)](LICENSE)

[![grab_frame](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/grab-frame.json)](ROADMAP.md#3-measured-baseline)
[![grab](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/grab.json)](ROADMAP.md#3-measured-baseline)
[![measured on](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/measured-on.json)](benchmarks/baseline.json)

A high-performance screencapture library for Windows using the Desktop Duplication API. This is a merged version combining features from multiple DXCam forks, designed to deliver ultra-fast capture capabilities with advanced functionality.

## Features

- **Capture at your display's refresh rate**: Desktop Duplication delivers at
  most one frame per refresh, and RapidShot keeps up with that ceiling — about
  240 fps on a 240 Hz monitor. No capture library can exceed it.
- **GPU-resident frames**: `grab_frame()` hands back a frame that never leaves
  the GPU — 21× faster than `grab()`, for consumers that feed a model directly
- **Only process what changed**: frames carry the compositor's dirty-rect
  metadata, typically under 1% of the screen on a normal desktop
- **Hybrid GPU laptops**: move a frame to the discrete GPU that Desktop
  Duplication cannot capture from
- **Multi-backend support**: NumPy, PIL, and CUDA/CuPy backends
- **Cursor capture**: Capture mouse cursor position and shape
- **Direct3D support**: Capture Direct3D exclusive full-screen applications without interruption
- **NVIDIA GPU acceleration**: GPU-accelerated processing using CuPy
- **Multi-monitor setup**: Support for multiple GPUs and monitors
- **Flexible output formats**: RGB, RGBA, BGR, BGRA, and grayscale support
- **Region-based capture**: Efficient capture of specific screen regions
- **Rotation handling**: Automatic handling of rotated displays
- **Actionable diagnostics**: headless machines and hybrid GPU setups are
  detected and explained rather than failing opaquely

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
import numpy as np
import rapidshot

# Create a ScreenCapture instance on the primary monitor
screencapture = rapidshot.create()

# Take a screencapture
frame = screencapture.grab()

# Display the screencapture
from PIL import Image
Image.fromarray(np.asarray(frame)).show()

# Hand the buffer back when done (see "Frame buffers" below)
frame.release()
```

> **New in 2.0:** `grab()` returns a pooled buffer that you `release()` when
> done. It indexes and converts like the array it wraps, so most code needs only
> the added `release()`. See [Frame buffers](#frame-buffers).

### Region-based Capture

```python
# Define a specific region
left, top = (1920 - 640) // 2, (1080 - 640) // 2
right, bottom = left + 640, top + 640
region = (left, top, right, bottom)

# Capture only this region
frame = screencapture.grab(region=region)  # 640x640x3 frame; release() when done
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
frame.release()
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
        pass  # Process monochrome cursor
    elif shape_type & DXGI_OUTDUPL_POINTER_SHAPE_TYPE_COLOR:
        pass  # Process color cursor
    elif shape_type & DXGI_OUTDUPL_POINTER_SHAPE_TYPE_MASKED_COLOR:
        pass  # Process masked color cursor
    
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

### Frame buffers

`grab()` returns a `PooledBuffer` — a reused buffer rather than a freshly
allocated array. Allocating one per frame costs ~1.6 ms on a 1080p RGB frame,
because the page faults on first touch cost more than the conversion itself;
reusing buffers makes `grab()` **1.3–2.1× faster**.

It behaves like the array it wraps, so most code is unchanged:

```python
frame = camera.grab()
if frame is not None:
    frame.shape, frame.dtype, frame.ndim   # as before
    pixel = frame[y, x]                    # indexing works
    arr = np.asarray(frame)                # zero-copy, for cv2 / PIL / models
    frame.release()                        # the one new line
```

**Release when done.** The buffer goes back to the pool and is handed to the
next capture, so anything still holding it would see the wrong frame. Reading it
after release raises `BufferReleasedError` rather than returning stale pixels.
To keep data beyond the release, `frame.copy()`.

Forgetting to release is not fatal: the pool runs dry and capture falls back to
allocating, which is slower but always correct. It will never hand you a buffer
another caller is reading.

**Migrating from 1.x.** Add `release()`, and wrap in `np.asarray()` anywhere a
true `ndarray` is required (`isinstance` checks, `Image.fromarray`). Or keep the
old behaviour outright:

```python
camera = rapidshot.create(pool_output=False)   # returns plain ndarrays
```

BGRA already worked this way before 2.0 — it does no conversion, so its staging
buffer was always returned pooled.

### Only process what changed

`grab_frame()` frames carry the compositor's own dirty-rect metadata, so a
consumer can skip regions that did not change:

```python
with camera.grab_frame() as frame:
    if frame.dirty_rects is None or not frame.dirty_rects:
        process_everything(frame)          # unknown, or no metadata reported
    else:
        for left, top, right, bottom in frame.dirty_rects:
            process_region(frame, left, top, right, bottom)
```

Coordinates are relative to the frame, so they index straight into the captured
image even when `region=` is in use.

Two things to get right. An **empty list does not mean nothing changed** — it
means no rects were reported, which a mode change or a coalescing driver can
also produce while the image differs completely; treat it as "assume everything
changed". And `None` means the metadata could not be read at all. Check
`frame.rects_coalesced` too: when true the driver merged rects, so they
over-estimate what actually changed.

### Hybrid GPU laptops and headless machines

`device_info()` only lists adapters that drive a display, because only those can
capture. `topology_info()` lists every adapter and says what that means:

```python
print(rapidshot.topology_info())
```

```
Topology: hybrid
  Adapter[0] (Intel(R) UHD Graphics) (Intel) (128MB VRAM) (1 output)
  Adapter[1] (NVIDIA GeForce RTX 4070 Laptop GPU) (NVIDIA) (8192MB VRAM) (0 outputs)

  Hybrid GPU system detected. Capture runs on Intel(R) UHD Graphics, which
  drives the display. NVIDIA GeForce RTX 4070 Laptop GPU has no outputs, so
  Desktop Duplication cannot run against it at all (DXGI_ERROR_UNSUPPORTED).
  ...
```

On an Optimus/switchable laptop the discrete GPU has no outputs, so Desktop
Duplication cannot run against it — capture is always bound to the adapter that
drives the display. `grab()` is unaffected. A GPU-resident frame is: it lives on
the capture adapter, so feeding it to a model on the *other* adapter needs a
cross-adapter copy. `rapidshot.native` provides one:

```python
from rapidshot import native

with camera.grab_frame() as frame:
    transfer = native.cross_adapter_transfer(frame)   # build once, reuse

with camera.grab_frame() as frame:
    transfer.transfer(frame)
    # Now bind transfer.destination_resource_address on the other adapter.
    print(transfer.source, "->", transfer.destination)
```

Costs about 0.87 ms per 1080p frame — roughly a third of what reading the same
frame to the CPU costs, so crossing adapters beats leaving the GPU. The shared
heap lives in **system memory**, not either adapter's VRAM; this is not
peer-to-peer VRAM-to-VRAM DMA, and the win is that a GPU copy engine moves the
bytes instead of CPU cores.

On a machine with no monitor attached there is no desktop to duplicate at all,
and `rapidshot.create()` raises `HeadlessError` explaining that a virtual
display driver (IDD) is needed. `topology_info()` still works there — it probes
DXGI directly rather than going through capture.

> A virtual display's advertised refresh rate does **not** raise capture rate.
> Desktop Duplication is driven by presents, not by refresh: a 500 Hz virtual
> display does not make an application render 500 fps.

## Advanced Usage

### Custom Buffer Size

```python
# Create a ScreenCapture instance with a larger frame buffer
screencapture = rapidshot.create(max_buffer_len=256)
```

### Different Color Formats

```python
# RGB (default)                      -> (H, W, 3)
screencapture_rgb = rapidshot.create(output_color="RGB")

# RGBA (with alpha channel)          -> (H, W, 4)
screencapture_rgba = rapidshot.create(output_color="RGBA")

# BGR (OpenCV format)                -> (H, W, 3)
screencapture_bgr = rapidshot.create(output_color="BGR")

# BGRA (raw, no conversion)          -> (H, W, 4)
screencapture_bgra = rapidshot.create(output_color="BGRA")

# Grayscale (Rec. 601 luma)          -> (H, W, 1)
screencapture_gray = rapidshot.create(output_color="GRAY")
```

All conversions are pure NumPy — OpenCV is not required. An unsupported
`output_color` raises `ValueError` at creation time.

### Capturing Into Your Own Buffer

`shot()` writes straight into a buffer you own, avoiding a per-frame
allocation. The buffer receives pixels in the instance's `output_color`, so
size it with `bytes_per_frame()`:

```python
import numpy as np

screencapture = rapidshot.create(output_color="RGB", region=(0, 0, 640, 480))

buffer = np.zeros((480, 640, screencapture.channels), dtype=np.uint8)
if screencapture.shot(buffer):
    print("captured", buffer.shape)   # (480, 640, 3)
```

The destination size is checked before anything is written, so an undersized
buffer raises `ValueError` instead of corrupting memory. NumPy arrays,
`ctypes` arrays, `bytearray` and `memoryview` all report their own size. A raw
pointer cannot, so it must be paired with an explicit `buffer_size`:

```python
import ctypes

screencapture.shot(
    ctypes.c_void_p(buffer.ctypes.data),
    buffer_size=buffer.nbytes,
)
```

### GPU-Resident Capture (no CPU round-trip)

`grab()` brings every frame down to the CPU — a staging read plus a color
conversion, about 4.5 ms per 1080p frame. If you are handing the pixels to a GPU
consumer (an inference runtime, a hardware encoder), `grab_frame()` skips all of
that and gives you the Direct3D texture directly:

```python
with screencapture.grab_frame() as frame:
    texture = frame.d3d11_texture        # ID3D11Texture2D, valid in here only
    print(frame.timestamp, frame.accumulated_frames)
```

Measured at 1920x1080: **0.21 ms/frame versus 4.53 ms — about 21x faster.**
Numbers come from `benchmarks/baseline.json`; see ROADMAP.md section 3 for how
they are measured and why they moved.

> **The `with` block is not optional.** Direct3D cannot capture the next frame
> while a reference to the previous one is outstanding, so an unreleased `Frame`
> stalls capture entirely. Use the context manager (or call `frame.release()`),
> and copy anything you need out before the block ends. `grab()`, `shot()` and
> `grab_frame()` all raise a clear error if a frame is still outstanding.

Frame metadata stays readable after release: `timestamp` / `timestamp_qpc` (when
the compositor presented the frame), `accumulated_frames` (greater than 1 means
the OS dropped frames because your loop fell behind), `protected_content`,
`cursor_visible`, `region`, `width`, `height`, `rotation_angle`.

### GPU Inference: Handing Frames to DirectML

Rapidshot can convert a captured frame into a **model-ready NCHW float32 tensor
that never leaves the GPU** — no staging read, no colour conversion, no
resize on the CPU. This needs the optional native extension (see below).

```python
from rapidshot import native

with screencapture.grab_frame() as frame:
    pre = native.GpuPreprocessor12(frame, 640, 640)   # build once, reuse
    pre.process(frame)                                 # one GPU dispatch

    print(pre.shape)                        # (1, 3, 640, 640)
    resource = pre.output_resource_address  # ID3D12Resource*
    gpu_va = pre.output_gpu_address         # GPU virtual address
```

`process()` resizes, normalises, converts BGRA→RGB and transposes to NCHW in a
single compute shader. On the CPU that same work costs about **8 ms per 1080p
frame**; here it is one dispatch and the result stays in VRAM. Optional
arguments cover the usual normalisation ranges (`scale=2.0, bias=-1.0` for
−1..1) and channel order (`bgr=True`).

**Where Rapidshot stops.** The output is an `ID3D12Resource` on the DirectML
device — exactly what ONNX Runtime's DirectML provider consumes. Rapidshot
deliberately does **not** bind it to a session: that would couple this library
to ONNX Runtime's ABI and release cadence for the sake of an optional feature.
Consuming it is a few lines on your side:

```cpp
const OrtDmlApi* dml = nullptr;
Ort::GetApi().GetExecutionProviderApi(
    "DML", ORT_API_VERSION, reinterpret_cast<const void**>(&dml));

void* allocation = nullptr;
dml->CreateGPUAllocationFromD3DResource(d3d12_resource, &allocation);

Ort::MemoryInfo info("DML", OrtDeviceAllocator, 0, OrtMemTypeDefault);
auto tensor = Ort::Value::CreateTensor(
    info, allocation, byte_size,
    shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

// bind with Ort::IoBinding, run, then:
dml->FreeGPUAllocation(allocation);
```

> **Note for Python users:** `OrtDmlApi` has no Python binding — it is reachable
> only from C/C++. That is a gap in ONNX Runtime, not in Rapidshot. If you need
> this from Python today you will need a small native shim of your own;
> `native.probe_onnxruntime()` and `native.onnxruntime_dll_path()` are provided
> to help locate and validate the runtime.

### Building the Optional Native Extension

Everything above except GPU-tensor interop works with no toolchain.
`pip install rapidshot` never requires Rust.

```bash
cd native && cargo build --release
```

```bash
python native/install_dev.py
```

Requires [Rust](https://rustup.rs) and the MSVC C++ build tools. Check
availability at runtime with `rapidshot.native.is_available()`.

### Benchmarking Your Changes

```bash
python benchmarks/perf_suite.py --out baseline.json
python benchmarks/perf_suite.py --out after.json --compare baseline.json
```

The suite compares minimum samples, calibrates against a control benchmark to
divide out machine drift, and pools samples across rounds. Run
`--self-test` to measure your machine's noise floor before trusting a result.

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

**Run them yourself.** This README used to carry a table of cross-library FPS
figures with no hardware, method or date attached, claiming 240+ for RapidShot
and 300+ with GPU acceleration. Both were unsupportable: Desktop Duplication
returns at most one frame per display refresh, so 300 fps needs a 300 Hz
monitor, and CuPy acceleration changes the *conversion* cost, not the rate at
which frames arrive.

Published FPS claims in this space contradict each other badly — DXcam's README
reports DXcam at 239 fps, BetterCam's reports the same library at 39 — because
they come from different hardware with no shared harness. A number measured on
someone else's machine tells you nothing about yours.

What is measured, reproducibly, is the per-frame cost of RapidShot's own paths
(1920×1080, Intel iGPU, from `benchmarks/baseline.json`):

| Path | Per frame | Implied ceiling |
| --- | --- | --- |
| `grab()` — staging read + colour conversion | 4.53 ms | ~220 fps |
| `grab_frame()` — texture stays on the GPU | **0.21 ms** | far above any display |

The `grab_frame()` figure is not a capture rate. It is what the *calling thread*
pays per frame; the display still delivers only one frame per refresh. What it
means is that capture stops being your bottleneck.

See `ROADMAP.md` section 3 for the full breakdown and how these are measured.

## System Requirements

- **Operating System:** Windows 10 or newer. Windows only — Desktop Duplication
  has no cross-platform equivalent.
- **Python:** 3.9+ (`pip` will refuse to install on anything older)
- **GPU:** Any GPU that drives a display. A CUDA-capable NVIDIA GPU is needed
  only for the optional CuPy acceleration.
- **RAM:** 8 GB+ (depending on the resolution and number of screencapture instances used)

### Troubleshooting

- **ImportError with CuPy:** Ensure you have compatible CUDA drivers installed.
- **Black screens when capturing:** Verify the application isn't running in exclusive fullscreen mode.
- **Low performance:** Experiment with different backends (NUMPY vs. CUPY) to optimize performance.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). It is mostly
a list of the things that will waste your time otherwise: live capture tests
need something moving on screen, CI cannot verify them at all, and a naive
benchmark comparison here once produced eleven false regressions on identical
code.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

RapidShot is a merged version of the following projects:

- Original DXcam by ra1nty
- dxcampil - PIL-based version
- DXcam-AI-M-BOT - Cursor support version
- BetterCam - GPU acceleration version