# RapidShot

[![CI](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml/badge.svg)](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rapidshot)](https://pypi.org/project/rapidshot/)
[![Python](https://img.shields.io/pypi/pyversions/rapidshot)](https://pypi.org/project/rapidshot/)
[![License](https://img.shields.io/pypi/l/rapidshot)](LICENSE)

**RapidShot captures the Windows desktop and turns it into model-ready GPU
tensors without the pixels ever reaching the CPU — and still hands you a NumPy
array when that is what you want.**

```text
Windows desktop → native GPU frame → fused preprocessing → Torch / CuPy / DLPack
```

It is built on DXGI Desktop Duplication, began as a merge of several DXcam
forks, and keeps a broadly familiar API. The capture path, the colour pipeline
and the GPU interop have since been rewritten.

## Why this instead of DXcam

If you want a screenshot as fast as possible and nothing else, use
[DXcam](https://github.com/ra1nty/DXcam) or
[BetterCam](https://github.com/RootKit-Org/BetterCam). They are good, they are
smaller, and RapidShot does not win the capture-only frame-rate column in most
cells ([Performance](#performance)).

RapidShot is for the case where the frame has somewhere to go. It owns the whole
path from the duplicated surface to a `(1, 3, 640, 640)` FP16 tensor your model
can consume, in one GPU dispatch, with no readback and no `ctypes`.

## What changed in 2.6

Four things, each with the measurement behind it:

- **Memory: `grab()` uses 58% less than RapidShot 2.5**, and `grab_frame()` 63%
  less. Read that as *against RapidShot 2.5*, not against DXcam — the
  cross-library figure is in [Performance](#performance), and it is a much
  smaller gap than 58%.
- **Fused GPU preprocessing with native FP16** — crop, resize, colour
  conversion, normalisation and layout in a **single** D3D12 dispatch, emitting
  `float16` as well as `float32` and `uint8`. Multi-ROI batches in one dispatch
  too.
- **One-call framework export** — `tensor.to_torch()`, `.to_cupy()`,
  `.to_dlpack()`, zero-copy. This replaces roughly sixty lines of `ctypes` that
  the 2.5 README asked you to copy.
- **Convert-before-transfer on hybrid GPUs** — preprocess on the capture
  adapter, then move the finished tensor rather than the whole frame: **2.46 MB
  instead of 16.38 MB** at 2560×1600. Verified end to end on real Optimus
  hardware (Intel UHD → RTX 4060), CUDA importing the transferred buffer — see
  [Hybrid GPU laptops](#hybrid-gpu-laptops). Measured: **162 fps against DXcam's
  96, pixels 10 ms younger, at under a fifth of the CPU**, and through YOLO11n
  **70 fps against 45, detections 7.8 ms sooner** — see
  [Desktop to model](#c-desktop-to-model).

**2.6.1** adds `rapidshot benchmark`, which runs those measurements on your own
machine — see [Run it on your machine](#run-it-on-your-machine).

Upgrading from 2.5? See [Migrating from 2.5](#migrating-from-25) — the one
breaking item is a new minimum native-extension version.

## Quick start

A screenshot, as a NumPy array. No GPU knowledge, no native extension:

```python
import rapidshot

camera = rapidshot.create()          # primary output, RGB
frame = camera.grab()                # (H, W, 3) uint8, or None if nothing changed
if frame is not None:
    print(frame.shape)
    frame.release()                  # return the pooled buffer
camera.release()
```

A model-ready FP16 tensor on the GPU, never touching the CPU. Needs
`rapidshot[native]`:

```python
import rapidshot

camera = rapidshot.create()
stream = rapidshot.TensorStream(camera, size=(640, 640),
                                dtype="float16", layout="NCHW")

for tensor in stream:
    prediction = model(tensor.to_torch())
```

Use `TensorStream` rather than writing that loop by hand. Its docstring lists
the four ways a hand-written version goes quietly wrong — a held frame stalls
capture, the reused output buffer races the model, `None` means both "nothing
changed" and "capture has died", and a rebuilt capture can land on a new
adapter. It closes all four.

> **On a hybrid laptop, `to_torch()` above raises `CrossAdapterRequired`.** That
> is not a bug. On an Optimus-style machine the display — and therefore capture
> — lives on the integrated GPU, while CUDA only sees the discrete one, so there
> is nothing for CUDA to import until the tensor has crossed. Add one object:
> [Hybrid GPU laptops](#hybrid-gpu-laptops). `rapidshot.diagnose()` tells you
> which kind of machine you are on before you write any of this.

For explicit control, drive the converter yourself:

```python
camera = rapidshot.create()

with camera.grab_frame() as frame:            # GPU texture, no CPU copy
    converter = rapidshot.GpuConverter(       # build once, reuse every frame
        frame, (640, 640), dtype="float16", layout="NCHW",
    )
    tensor = converter.process(frame)
    x = tensor.to_torch()                     # zero-copy
    prediction = model(x)
```

`GpuConverter` takes the frame first: it needs the source surface to size its
resources. Build it once outside your loop and call `process()` per frame — the
output buffer is reused, which is why `TensorStream` synchronises before
overwriting it.

`to_cupy()` and `to_dlpack()` are the same tensor by other routes:

```python
array   = tensor.to_cupy()     # cupy.ndarray, zero-copy
capsule = tensor.to_dlpack()   # any framework implementing DLPack
```

All three alias the converter's buffer, so the next `process()` overwrites them.
Clone if you need one to outlive the next frame.

## Choose your path

| Goal | API | CPU pixels? | Native required? |
| --- | --- | --- | --- |
| NumPy screenshot | `camera.grab()` | Yes | No |
| GPU frame, no readback | `camera.grab_frame()` | No | No |
| Model-ready FP16 tensor | `rapidshot.GpuConverter` | No | **Yes** |
| Capture-to-model loop | `rapidshot.TensorStream` | No | **Yes** |
| PyTorch | `tensor.to_torch()` | No | **Yes** |
| CuPy | `tensor.to_cupy()` | No | **Yes** |
| Any DLPack framework | `tensor.to_dlpack()` | No | **Yes** |
| Encoder input (NV12 / P010) | `GpuConverter(pixel_format=…)` | No | **Yes** |
| Hybrid GPU (Optimus) | `GpuConverter` + `TensorTransfer` | No | **Yes** |

## Installation

```bash
pip install rapidshot
```

That is the whole library for CPU and D3D11 capture — `grab()`, `grab_frame()`,
regions, colour conversion, dirty rects, video recording. **No toolchain, no
compiler, no GPU requirement.**

The 2.6 GPU tensor features need the prebuilt native extension:

```bash
pip install "rapidshot[native]"
```

This installs **`rapidshot-native >= 0.2.0`**, which is the minimum that exports
`GpuConverter` and `TensorTransfer` — both were added after `rapidshot-native`
0.1.0. If you somehow end up with an older extension, RapidShot says so:

```text
RuntimeError: GpuConverter12 requires rapidshot-native >= 0.2.0; the installed
extension is version 0.1.0, from rapidshot-native wheel.

Upgrade the prebuilt wheel:
    pip install --upgrade 'rapidshot-native>=0.2.0'
```

Other extras: `rapidshot[gpu_cuda12]` or `[gpu_cuda13]` for CuPy (pick one —
the `cupy-cudaNNx` wheels are mutually exclusive), `rapidshot[benchmark]` for
the [benchmark command](#run-it-on-your-machine), and `rapidshot[all]` for
everything that is not CUDA-version-specific.

Check what your machine actually has:

```python
import rapidshot
print(rapidshot.diagnose())
```

It reports the adapters, outputs, whether the native extension loaded and from
where, and which optional dependencies are present — before you write any
capture code.

## The 2.6 pipeline

```text
                              ┌── NumPy / OpenCV        grab()
                              │
   DXGI ──→ D3D11 Frame ──────┼── raw GPU texture       grab_frame()
                              │
                              └── GpuConverter
                                       │
                          crop · resize · RGB/BGR
                          normalize · FP16 · layout
                              (one GPU dispatch)
                                       │
                          ┌────────────┼────────────┐
                          ↓            ↓            ↓
                       Torch         CuPy        DLPack
```

On a hybrid laptop the tensor crosses adapters *after* preprocessing, so what
moves is a small tensor rather than a full frame — see
[Hybrid GPU laptops](#hybrid-gpu-laptops).

## Capturing to NumPy

```python
camera = rapidshot.create(output_color="BGR")          # RGB, BGR, RGBA, BGRA, GRAY
frame  = camera.grab(region=(0, 0, 1920, 1080))        # left, top, right, bottom
```

`grab()` returns `None` when nothing has changed — Desktop Duplication reports
only *changed* content, so an idle desktop legitimately produces no frames.

Continuous capture runs on a background thread:

```python
camera.start(target_fps=60, video_mode=True)
for _ in range(120):
    frame = camera.get_latest_frame()
    ...
camera.stop()
```

One camera per output; create several for several monitors. `rapidshot.create()`
returns the same camera for the same output if you ask twice.

### Frame buffers and who owns them

`grab()` returns a `PooledBuffer` — a reused buffer, not a fresh array.
Allocating per frame costs about 1.6 ms on a 1080p RGB frame, because first-touch
page faults cost more than the conversion.

```python
frame = camera.grab()
if frame is not None:
    frame.shape, frame.dtype, frame.ndim
    pixel = frame[y, x]
    arr = np.asarray(frame)             # zero-copy, for cv2 / PIL / a model
    frame.release()                     # the one new line
```

**Release when you are done.** The buffer goes back to the pool and on to the
next capture. Reading it after release raises `BufferReleasedError` rather than
returning stale pixels. To keep the data, use `frame.copy()` or
`np.array(frame, copy=True)`.

Forgetting to release is never *corrupting* — the pool will not hand out a
buffer someone is still reading. With a conversion (`RGB`, `BGR`, `RGBA`,
`GRAY`) an exhausted pool falls back to allocating. `BGRA` converts nothing, so
the frame you hold *is* the staging buffer and there is nothing to fall back to:
`grab()` returns `None` until one comes back.

Prefer plain arrays? `rapidshot.create(pool_output=False)`.

## GPU-resident frames

`grab_frame()` hands back the Direct3D texture itself. Nothing is copied to
system memory.

```python
with camera.grab_frame() as frame:      # the context manager is the safe form
    tex = frame.d3d11_texture           # ID3D11Texture2D, valid until release
    print(frame.width, frame.height, frame.timestamp_qpc)
```

**A held frame blocks capture.** DXGI will not produce the next frame while a
reference is outstanding, so release promptly — `with` does it for you.

Frames also carry the compositor's metadata: `dirty_rects` and `move_rects` in
frame coordinates, `changed_fraction`, `accumulated_frames`, `cursor`,
`rotation_angle`, `protected_content`, and `timestamp_qpc` from the present
itself.

## Model-ready tensors

`GpuConverter` is one compute dispatch that does all of it:

| Option | Values |
| --- | --- |
| `size` | output `(width, height)` |
| `dtype` | `"float32"`, `"float16"`, `"uint8"` |
| `layout` | `"nchw"` (default), `"nhwc"` |
| `sampling` | `"bilinear"` (default), `"nearest"` |
| `normalize` | `True` → 0..1, `False` → 0..255 |
| `bgr` | channel order |
| `crop` | `(left, top, right, bottom)`, frame coordinates, applied before resize |
| `batch` / `regions=` | multi-ROI, **one dispatch**, returns `(N, …)` |
| `pixel_format` | `"nv12"` / `"p010"` for encoder input |

Multi-ROI in a single dispatch:

```python
converter = rapidshot.GpuConverter(frame, (224, 224), batch=4)
tensor = converter.process(frame, regions=[
    (0, 0, 400, 300), (800, 40, 1000, 240),
])                                          # (2, 3, 224, 224)
```

Payload sizes at 640², which is what crosses a bus if anything has to:

| dtype | shape | bytes |
| --- | --- | --- |
| `uint8` (resized BGRA) | `(1, 640, 640, 4)` | 1.64 MB |
| `float16` | `(1, 3, 640, 640)` | **2.46 MB** |
| `float32` | `(1, 3, 640, 640)` | 4.92 MB |

against 8.29 MB for a 1080p frame or 16.38 MB at 2560×1600.

## Hybrid GPU laptops

On an Optimus-style laptop the display is driven by the integrated GPU, so
capture happens there, while CUDA lives on the discrete GPU. Something has to
cross.

```text
2.5 — move the frame, convert on arrival
   2560×1600 BGRA (16.38 MB) → cross-adapter transfer → resize/RGB/normalize → model

2.6 — convert first, move the result
   2560×1600 capture → crop/resize/RGB/normalize/FP16 → 2.46 MB tensor
                     → cross-adapter transfer → model
```

```python
converter = rapidshot.GpuConverter(frame, (640, 640), dtype="float16")
transfer  = rapidshot.TensorTransfer(converter)
converter.process(frame)
transfer.transfer()
```

This is also the answer to `CrossAdapterRequired` from the quick start: once the
tensor is on the CUDA adapter, the consumer imports it there.

**Verified on real hybrid hardware, 2026-09-20** — Intel UHD Graphics capture,
RTX 4060 destination, 2560×1600:

- `examples/verify_cross_adapter.py` moved 5 frames Intel → RTX 4060, each
  16,384,000 bytes, every one byte-exact against a source-side readback.
- The converted-tensor path was verified end to end with **CUDA importing the
  transferred buffer** from the shared D3D12 heap on the discrete GPU
  (`benchmarks/ai_ingestion.py --paths rapidshot-converter-xadapter --verify`),
  max deviation **1 RGB8 level** — the documented bilinear rounding tolerance.

Why that last point is stated so specifically: `TensorTransfer` used to hand the
consumer the *source* device's shared handle. Against a WARP destination that
still "passed" — CUDA imported the capture GPU's own memory and the path
reported a crossing that never happened. WARP is the configuration that hid the
bug, so a WARP result is not evidence for this path. The check above is the one
that counts, and it needs two real GPUs.

**Measured on this laptop, 2026-09-25:** 162 fps to a CUDA tensor, pixels
27.6 ms old at the median, 2.0 ms of CPU per frame — against DXcam's 95.6 fps,
37.7 ms and 11.2 ms. Carried on through YOLO11n with NMS: 69.5 fps, detections
41.4 ms after `Present()`, 12.9 ms of CPU per frame, against DXcam's 45.0 fps,
49.2 ms and 20.9 ms. The full tables, including the full-frame transfer this
replaces, are in [Desktop to model](#c-desktop-to-model).

## Correctness guarantees

These are the disciplines the code actually enforces, not aspirations:

- **An independent reference comes before any performance claim.** Conversion
  kernels are checked byte-for-byte against NumPy; the GPU converter against a
  D3D11 staging readback of the same texture (`tests/test_gpu_converter.py`);
  NV12/P010 against a CPU reference *and* the published inverse matrices, so
  the kernel and its reference cannot share a wrong constant.
- **Frame lifetime is enforced, not documented.** Reading a released frame or a
  released buffer raises rather than returning stale pixels. Generation
  tracking invalidates frames from a duplicator that has been rebuilt.
- **GPU reads are ordered on the GPU.** `AcquireNextFrame` returns when the
  copy is *submitted*, not complete. Every D3D12 path that reads the capture
  surface waits on a shared fence — without it, measured stale-frame rates ran
  from 7/150 to 65/150 (`native/src/capture_order.rs`).
- **Producer and consumer both have a barrier.** `GpuTensor.sync()` blocks
  until CUDA work on the buffer is done, and `set_consumer_fence()` stops the
  producer overwriting a buffer a consumer is still reading.
- **WARP is not a GPU.** A software adapter proves an API works and nothing
  about what it costs, and every table and message that uses one says so.

## Performance

Three different questions, deliberately separated — DXcam can win the first
while RapidShot wins the others.

### A. Capture alone

RapidShot does not win this column in most cells, and says so. Full tables,
methodology and caveats: [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

### B. Memory

[`benchmarks/memory-baseline-machineB.json`](benchmarks/memory-baseline-machineB.json)
— Machine B (RTX 4060 laptop, Intel UHD capture, 2560×1600 at 165 Hz), still
desktop, 10 s after a 2 s warm-up, **three passes**, working set:

| | RapidShot 2.5 | RapidShot 2.6 | vs 2.5 | DXcam |
| --- | ---: | ---: | ---: | ---: |
| `grab()` (RGB) | 416.0 MB | **174.8 MB** | **−58%** | 103.1 MB |
| `grab_frame()` (GPU) | 282.1 MB | **104.4 MB** | **−63%** | 103.1 MB |

Read the two comparisons separately. **Against RapidShot 2.5 the reduction is
58% and 63%.** Against DXcam, `grab_frame()` is now **1.01×** its working set,
where 2.5 was 2.7×; `grab()` remains **1.70×**, because it also holds converted
output buffers DXcam has no equivalent of. Throughput was unchanged across the
change: 165 fps for `grab_frame()`, ~100 for `grab()`, every difference inside
the run-to-run spread.

None of it came from a redesign — a lazy CuPy import (that one import was
178.8 MB resident, paid by every caller including those with no NVIDIA GPU), a
staging pool sized to what a converting `grab()` can actually use, and
`pool_size_frames` defaulting to 2 instead of 4.

### C. Desktop to model

Pixel **age**, not call duration: a source encodes a frame ID into the image and
records every `Present()`, so each library is timed on one clock by how old its
pixels were when they became a `(1, 3, 640, 640)` FP16 tensor on CUDA.
Recorded 2026-09-25 on Machine B, 2560×1600 at 165 Hz, medians across 3 passes,
8 s per path, every path verified to produce the correct tensor before being
timed. The hybrid detector table was recorded against **2.6.1 with the
`rapidshot-native` 0.2.1 wheel**, the others against **2.6.0 with 0.2.0**, both
from PyPI. The library and extension code are the same in both: 2.6.1 added the
benchmark command and shipped its test source.

**Hybrid laptop** — Intel UHD captures, RTX 4060 runs CUDA
([`section7-ingestion-machineB-hybrid-2.6.0.json`](benchmarks/section7-ingestion-machineB-hybrid-2.6.0.json)):

| | unique frames/s | pixel age p50 / p95 | CPU per frame |
| --- | ---: | ---: | ---: |
| mss | 29.1 | 56.5 / 64.3 ms | 19.5 ms |
| DXcam (DXGI) | 95.6 | 37.7 / 43.4 ms | 11.2 ms |
| DXcam (WGC) | 93.3 | 40.8 / 49.0 ms | 10.8 ms |
| RapidShot `grab()` | 123.1 | 35.1 / 40.2 ms | 8.8 ms |
| RapidShot `grab()`, `nvidia_gpu=True` | 125.6 | 34.8 / 38.8 ms | 5.2 ms |
| RapidShot, full frame across adapters (2.5 path) | 80.9 | 34.9 / 37.5 ms | 5.2 ms |
| **RapidShot 2.6, `GpuConverter` + `TensorTransfer`** | **162.2** | **27.6 / 29.9 ms** | **2.0 ms** |

Converting on the capture GPU and moving only the 2.46 MB tensor gives **1.7×
DXcam's frames, pixels 10 ms younger, at under a fifth of its CPU** — and its
worst pass beat DXcam's best on all three columns. 162 fps is the 165 Hz panel,
so that column is a floor. Moving the *whole frame* across adapters is still
slower than DXcam, at 81 fps; on a hybrid laptop, convert first.

**NVIDIA driving the display** — the MUX in discrete-only mode, so capture and
CUDA share one adapter
([`section7-ingestion-machineB-dgpu-2.6.0.json`](benchmarks/section7-ingestion-machineB-dgpu-2.6.0.json)):

| | unique frames/s | pixel age p50 / p95 | CPU per frame |
| --- | ---: | ---: | ---: |
| mss | 40.7 | 46.0 / 52.1 ms | 14.1 ms |
| DXcam (DXGI) | 124.0 | 35.1 / 38.5 ms | 10.1 ms |
| DXcam (WGC) | 122.9 | 35.6 / 39.2 ms | 10.4 ms |
| RapidShot `grab()` | 161.8 | 31.2 / 33.2 ms | 8.7 ms |
| RapidShot `grab()`, `nvidia_gpu=True` | 151.6 | 33.9 / 35.1 ms | 6.6 ms |
| **RapidShot 2.6, `GpuConverter`** | **164.3** | **25.9 / 26.6 ms** | **1.2 ms** |

The `GpuConverter` row is two passes, not three, and it read frames as fast as
the source presented them, so its frame rate is a floor rather than its ceiling.
The two tables are different machine configurations; compare within one, not
across.

**Through a detector** — the same clock carried to *usable detections*: YOLO11n
on the RTX 4060 through ONNX Runtime's CUDA provider, NMS included, with a scene
of signage and a keyboard panning under the marker so the model has something
to find.

Hybrid laptop
([`section7-inference-machineB-hybrid-2.6.1.json`](benchmarks/section7-inference-machineB-hybrid-2.6.1.json)):

| | unique frames/s | detections ready p50 / p95 | CPU per frame |
| --- | ---: | ---: | ---: |
| mss | 23.6 | 62.9 / 65.7 ms | 26.3 ms |
| DXcam (DXGI) | 45.0 | 49.2 / 52.7 ms | 20.9 ms |
| RapidShot `grab()` | 51.3 | 46.5 / 50.7 ms | 19.2 ms |
| RapidShot `grab()`, `nvidia_gpu=True` | 52.0 | 46.2 / 50.5 ms | 16.4 ms |
| RapidShot, full frame across adapters (2.5 path) | 50.9 | 47.0 / 51.1 ms | 15.7 ms |
| **RapidShot 2.6, `GpuConverter` + `TensorTransfer`** | **69.5** | **41.4 / 46.0 ms** | **12.9 ms** |

1.55× DXcam's frames through the model, detections 7.8 ms sooner, 38% less CPU
per frame — and its worst pass beat DXcam's best on all three columns. Unlike
the capture-only table, the full-frame crossing is not behind DXcam here: it
lands with the other full-frame paths at about 51 fps, so at this rate the
crossing is no longer what limits it. Every path found the same scene: 8.5–9.3
detections per frame.

Discrete-only mode
([`section7-inference-machineB-dgpu-2.6.0.json`](benchmarks/section7-inference-machineB-dgpu-2.6.0.json)):

| | unique frames/s | detections ready p50 / p95 | CPU per frame |
| --- | ---: | ---: | ---: |
| mss | 23.7 | 67.6 / 71.0 ms | 32.1 ms |
| DXcam (DXGI) | 37.7 | 53.8 / 57.7 ms | 28.0 ms |
| RapidShot `grab()` | 41.3 | 51.6 / 56.2 ms | 26.2 ms |
| **RapidShot 2.6, `GpuConverter`** | **50.5** | **46.9 / 51.2 ms** | **19.2 ms** |

34% more frames through the model than DXcam, detections 6.9 ms sooner, and a
third less CPU, with every pass ahead of DXcam's best. Every path found the
same scene: 8.4–9.7 detections per frame.

Compare within a table, not across the two. The model itself takes about 11 ms
per frame in hybrid mode but about 18 ms in discrete-only, where the RTX 4060
is also drawing the desktop, so the hybrid rows are faster for a reason that
has nothing to do with capture. Three passes are consistent, not the five
paired runs the harness itself requires before calling a difference a verdict
([ROADMAP](ROADMAP.md) § 7.0c).

### Run it on your machine

```bash
pip install "rapidshot[benchmark]"
rapidshot benchmark
```

The same harness as the tables above: every installed library and every
RapidShot path this hardware supports, verified first, then three 8-second
passes timed by pixel age. About five minutes; a test pattern fills the screen.
It writes a Markdown report and a JSON file with no hostname, username or
device IDs in them. Please
[open a benchmark issue](https://github.com/Zaatra/Rapidshot/issues/new?template=benchmark.md)
with both. Add `cupy-cuda12x` or `cupy-cuda13x` to include the GPU paths,
`--full` for CPU per tensor and memory as well (about ten minutes), and
`--check` to see what would run without running it. The report also says
whether HDR was on and whether frames can cross between your GPUs.

### What none of this shows

One machine, one resolution, one refresh rate, one driver. No AMD part has ever
run these. Live capture rates depend on what the screen is doing and are not
comparable across recordings — see `docs/BENCHMARKS.md` for why a single live
recording is not a measurement.

## Tested hardware

Nothing in RapidShot branches on GPU vendor. This is what has actually been run,
which is a different claim:

**Verified**

| Configuration | Notes |
| --- | --- |
| Intel iGPU, single adapter | Capture, conversion, D3D12 preprocess |
| NVIDIA dGPU, single adapter | With and without the native extension |
| NVIDIA dGPU driving the display (MUX) | Capture + CUDA on one adapter; this is where the byte-equal tensor export is verified |
| Cross-adapter **frame** transfer, Intel → RTX 4060 | Byte-exact, 5 frames at 2560×1600 |
| Cross-adapter **tensor** transfer + CUDA import | `TensorTransfer` on real Optimus, verified 2026-09-20 |
| Hybrid, capture to YOLO11n detections | All six hybrid paths verified, then timed, 2026-09-25 |
| Python 3.9 – 3.14 | CI matrix |
| `rapidshot-native` 0.2.0 / 0.2.1 | 0.2.0 is the minimum for the 2.6 GPU features; 0.2.1 adds the benchmark's test source around the same extension |

**Not verified**

| Configuration | Why it matters |
| --- | --- |
| Any AMD GPU | The BGRA swizzle rule is confirmed on Intel and NVIDIA and follows from the DXGI format rather than driver discretion — but it is the one rule a silent mismatch would corrupt rather than crash |
| Hybrid with an AMD adapter | Cross-adapter capability flags are unknown; the buffer path was chosen so nothing depends on them |
| Headless / virtual display (IDD) | Diagnostics exist; capture on one has not been run |

## Migrating from 2.5

- **`rapidshot-native >= 0.2.0` is now required** for the GPU tensor features.
  `pip install --upgrade "rapidshot[native]"`. An older extension gives a named
  error telling you this, not a missing-attribute traceback.
- **`GpuConverter` replaces `GpuPreprocessor12`** as the recommended transform.
  `GpuPreprocessor12` still works and is unchanged; `GpuConverter` adds
  bilinear sampling (the old path decimates with `Load()`), FP16, NHWC, crop,
  multi-ROI and NV12/P010. `sampling="nearest"` reproduces the old path bit for
  bit.
- **`to_torch()` / `to_cupy()` / `to_dlpack()` replace the `ctypes` recipe.**
  `examples/gpu_tensor_to_cupy.py` remains as the worked explanation of what
  they do.
- **`pool_size_frames` now defaults to 2, from 4.** Externally visible: with
  `BGRA` (which does no conversion) an exhausted pool returns `None` sooner.
  Pass `pool_size_frames=4` to restore the old behaviour.
- **Nothing was renamed or removed.**

Everything else, including the bugs fixed, is in [`CHANGELOG.md`](CHANGELOG.md).

## Advanced: native interop

For consumers that bind GPU resources directly — DirectML, ONNX Runtime, a
custom CUDA kernel — the underlying handles are reachable:

- `GpuPreprocessor12` — the 2.3-era NCHW float32 preprocessor.
- `converter.output_resource_address` — the `ID3D12Resource` address, which is
  the vendor-neutral route for DirectML and ONNX Runtime. It is on the
  converter, not the tensor, and sees the whole buffer including every batch
  slot; `converter.output_byte_size` is its size.
- `native.CrossAdapterTransfer` — move a whole *frame* across adapters
  (ordering A), with `transfer_async()`, `shared_fence_handle`,
  `set_consumer_fence()` and `wait_for_consumer()`.
- `native.probe_d3d12_sharing()`, `probe_shareable_buffers()`,
  `texture_sharing_info()` — capability probes.

`examples/gpu_tensor_to_cupy.py` and `examples/verify_cross_adapter.py` are the
worked, commented versions. You should not need any of this to use
`GpuConverter`.

## Diagnostics and troubleshooting

```python
import rapidshot
print(rapidshot.diagnose())                 # adapters, outputs, extension, deps
print(rapidshot.diagnose(probe_gpu=True))   # also probe D3D12 sharing
```

From a shell, `rapidshot diagnose` prints the same report.

| Symptom | Cause |
| --- | --- |
| `grab()` returns `None` | Nothing changed on screen. This is normal, not an error. |
| `None` forever in `BGRA` | Pool exhausted — release your frames, or raise `pool_size_frames`. |
| `AttributeError` on `GpuConverter` | Native extension older than 0.2.0. Upgrade it. |
| `CrossAdapterRequired` | Capture is on the iGPU and CUDA is on the dGPU — the hybrid case. |
| `RapidShotProtectedContentError` | HDCP/DRM content; the OS blanks the region. |
| Capture stops after a resolution change | Handled automatically; see `on_output_change`. |
| Nothing captures at all | No desktop session (a CI runner, a service, an RDP session with no console). |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The one rule worth stating here:
**a correctness check against an independent reference comes before any
performance claim.** A fast wrong answer is worthless.

Plans and their reasoning are in [ROADMAP.md](ROADMAP.md); measurements and
their methodology in [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

[DXcam](https://github.com/ra1nty/DXcam) and
[BetterCam](https://github.com/RootKit-Org/BetterCam), whose forks RapidShot
began as a merge of.
