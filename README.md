# RapidShot

[![CI](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml/badge.svg)](https://github.com/Zaatra/Rapidshot/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/rapidshot)](https://pypi.org/project/rapidshot/)
[![Python](https://img.shields.io/pypi/pyversions/rapidshot)](https://pypi.org/project/rapidshot/)
[![License](https://img.shields.io/pypi/l/rapidshot)](LICENSE)

[![BGRA to RGB](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/convert-rgb.json)](ROADMAP.md#3-measured-baseline)
[![BGRA to GRAY](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/convert-gray.json)](ROADMAP.md#3-measured-baseline)
[![shot to buffer](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/shot.json)](ROADMAP.md#3-measured-baseline)
[![measured on](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/Zaatra/Rapidshot/main/.github/badges/measured-on.json)](benchmarks/baseline.json)

<sub>Badge figures are generated from [`benchmarks/baseline.json`](benchmarks/baseline.json)
and checked in CI, so they cannot drift from the recording. They are **synthetic
conversion benchmarks with the optional native extension built** — not
end-to-end capture rates, and not what a plain `pip install` produces. Read
[Measurements](#measurements) before quoting any of them.</sub>

Windows screen capture through DXGI Desktop Duplication, with a route from a
captured frame to GPU memory that does not pass through the CPU.

Concretely, it can hand you a frame as a NumPy array, as a `cupy.ndarray`, as a
Direct3D texture that never leaves the GPU, as an NCHW float32 tensor produced
by one compute dispatch, or as a copy that has been moved to a second adapter on
a hybrid laptop. Frames carry the compositor's dirty-rect metadata. Colour
conversion has byte-exact AVX2 kernels behind an optional extension.

It began as a merge of several DXcam forks and keeps a broadly familiar API. The
capture path, the colour pipeline and the GPU interop have since been rewritten.

**What this library is not.** It is not the fastest at `grab()` alone — see
[Against other libraries](#against-other-libraries), where it loses the
capture-only frame-rate column in most cells. It pulls ahead once the frame has
somewhere to go: taking pixels to a model-ready CUDA tensor, and on through a
trained YOLO11n, it returned more unique frames, younger pixels and less CPU per
frame than DXcam on the one machine measured so far
([Desktop to model](#desktop-to-model)). It uses
noticeably more memory than DXcam, BetterCam or mss. If capture is the only
thing your machine is doing and you want the lowest per-call latency, DXcam and
BetterCam are good and you should use them.

---

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Frame buffers](#frame-buffers)
- [Coming from DXcam](#coming-from-dxcam)
- [Trading CPU for frames](#trading-cpu-for-frames)
- [Only process what changed](#only-process-what-changed)
- [Colour formats](#colour-formats)
- [Capturing into your own buffer](#capturing-into-your-own-buffer)
- [GPU-resident capture](#gpu-resident-capture)
- [The GPU tensor](#the-gpu-tensor)
- [Hybrid GPU laptops](#hybrid-gpu-laptops)
- [Surviving display changes](#surviving-display-changes)
- [Headless machines](#headless-machines)
- [The optional native extension](#the-optional-native-extension)
- [Profiling your own loop](#profiling-your-own-loop)
- [Measurements](#measurements)
- [Against other libraries](#against-other-libraries)
- [System requirements](#system-requirements)
- [Diagnostics](#diagnostics)
- [Troubleshooting](#troubleshooting)

## Installation

Installed as `rapidshot`, imported as `import rapidshot`.

```bash
pip install rapidshot
```

That is pure Python and needs no toolchain. For the GPU tensor, cross-adapter
transfer and the AVX2 conversion kernels, add the prebuilt extension — still no
toolchain:

```bash
pip install rapidshot[native]
```

One `abi3` wheel covers Python 3.9 onward, Windows x86-64. See
[the native extension](#the-optional-native-extension) for what it adds and how
to build it yourself instead.

To see what you ended up with — whether the extension loaded, which adapters
can capture, which optional dependencies are importable:

```python
import rapidshot
print(rapidshot.diagnose())
```

See [Diagnostics](#diagnostics).

```bash
pip install rapidshot[cv2]         # OpenCV, for your own downstream use
pip install rapidshot[gpu_cuda13]  # CuPy for CUDA 13
pip install rapidshot[gpu_cuda12]  # CUDA 12
pip install rapidshot[gpu]         # CUDA 11
pip install rapidshot[all]
```

The CuPy wheels are mutually exclusive, so there is no catch-all extra — check
your toolkit with `nvidia-smi`. RapidShot does not use OpenCV for colour
conversion; since 2.3.0 the GPU path is pure CuPy and stays on the device.

## Quick start

```python
import numpy as np
import rapidshot

camera = rapidshot.create()

frame = camera.grab()
if frame is not None:
    print(frame.shape)          # (H, W, 3), RGB by default
    frame.release()             # hand the buffer back -- see Frame buffers
```

`grab()` returns `None` when nothing has changed on screen. Desktop Duplication
reports compositor *presents*, so a still desktop produces no frames at all.
This surprises people benchmarking against a static screen; it is not a fault.

### Region capture

```python
region = (760, 340, 1160, 740)          # left, top, right, bottom
frame = camera.grab(region=region)      # 400x400
```

### Continuous capture

```python
camera.start(target_fps=60)
for _ in range(1000):
    image = camera.get_latest_frame()   # blocks until a new frame arrives
camera.stop()
```

### Recording to a video file

```python
import cv2
import rapidshot

camera = rapidshot.create(output_color="BGR")   # OpenCV's channel order
camera.start(target_fps=30, video_mode=True)

writer = cv2.VideoWriter(
    "video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
)
for _ in range(300):
    writer.write(camera.get_latest_frame())

camera.stop()
writer.release()
```

`video_mode=True` repeats the last frame when the screen is idle, so the output
keeps a constant frame rate rather than stalling.

### Converting on the GPU

```python
camera = rapidshot.create(output_color="RGB", nvidia_gpu=True)
frame = camera.grab()                    # a cupy.ndarray, still in VRAM
```

Conversion runs in CuPy and is byte-identical to the CPU path, so enabling it
changes no pixel. It returns a bare `cupy.ndarray` rather than a pooled buffer,
so there is no `release()` to call and `pool_size_frames` does not apply.

> Requires a working CuPy, which means CUDA headers as well as the wheel. If
> CuPy imports but fails at its first JIT, `grab()` returns `None` and the CuPy
> error goes to stderr — capture does **not** silently fall back to the CPU
> processor in that case. The fallback covers CuPy being absent, not CuPy being
> broken. See [Troubleshooting](#troubleshooting).

### Cursor

Frames from [`grab_frame()`](#gpu-resident-capture) carry a snapshot of the
cursor as of that frame:

```python
frame = camera.grab_frame()
if frame is not None:
    with frame:
        cursor = frame.cursor             # a copy -- still valid after release
    if cursor.visible:
        x, y = cursor.position            # desktop coordinates
        hx, hy = cursor.hotspot           # click point within the shape
        if cursor.shape is not None:
            w, h = cursor.shape_size
```

`cursor.shape` is the raw pointer image as DXGI sent it, `cursor.shape_pitch`
bytes per row, in the encoding `cursor.shape_type` names: `1` monochrome (an AND
mask stacked above an XOR mask, so it is twice the cursor's height), `2` colour
BGRA, `4` masked colour. `shape` is `None` until DXGI has sent one.

Compositing the cursor onto a frame is left to the caller: doing it correctly
means handling those three encodings separately, and the right blend depends on
what you are compositing onto.

`camera.grab_cursor()` still returns the raw DXGI structures
(`PointerPositionInfo`, `PointerShapeInfo`, `Shape`) for code that already
uses them.

## Frame buffers

`grab()` returns a `PooledBuffer` — a reused buffer, not a fresh array.
Allocating one per frame costs about 1.6 ms on a 1080p RGB frame, because the
page faults on first touch cost more than the conversion; reusing them makes
`grab()` 1.3–2.1x faster.

It behaves like the array it wraps:

```python
frame = camera.grab()
if frame is not None:
    frame.shape, frame.dtype, frame.ndim
    pixel = frame[y, x]
    arr = np.asarray(frame)             # zero-copy, for cv2 / PIL / a model
    frame.release()                     # the one new line
```

**Release when done.** The buffer returns to the pool and is handed to the next
capture, so anything still holding it would see the wrong frame. Reading it
after release raises `BufferReleasedError` rather than returning stale pixels.
To keep the data, `frame.copy()` or `np.array(frame, copy=True)`.

> Before 2.5.0, `np.array(frame, copy=True)` returned a **view** under NumPy 2 —
> the buffer accepted NumPy's `copy` argument and ignored it — so the "copy"
> was overwritten by the next capture, with nothing raising. `frame.copy()` and
> `np.asarray(frame).copy()` were always correct.

Forgetting is not fatal: the pool runs dry and capture falls back to allocating,
which is slower but always correct. It will never hand you a buffer another
caller is reading.

**Coming from 1.x**: add `release()`, and wrap in `np.asarray()` where a true
`ndarray` is required (`isinstance` checks, `Image.fromarray`). Or opt out:

```python
camera = rapidshot.create(pool_output=False)   # plain ndarrays, as before
```

BGRA already worked this way before 2.0 — it does no conversion, so its staging
buffer was always returned pooled.

## Coming from DXcam

Change one import:

```python
import rapidshot.dxcam_compat as dxcam     # was: import dxcam

camera = dxcam.create(output_color="BGR")
frame = camera.grab()                      # a plain numpy.ndarray, as before
```

The shim provides `create()`, `device_info()`, `output_info()`, `reset()` and
`clean_up()`, and a camera with `grab()`, `shot()`, `start()` /
`get_latest_frame()` / `stop()`, `release()`, `grab_view()` /
`get_latest_frame_view()`, and the attributes DXcam code reads — `width`,
`height`, `channel_size`, `region`, `is_capturing`, `latest_frame_time`.
Keywords `create()` does not recognise are passed through to
`rapidshot.create()`, so `nvidia_gpu=True` keeps working.

**It costs one copy per frame, on purpose.** DXcam code never releases a frame,
and RapidShot's buffers [must be released](#frame-buffers), so the shim copies
each frame out and releases it at once. That copy is the price of not touching
your code.

`grab_view()` is the exception. DXcam's "valid until the next grab" contract is
exactly a pooled buffer's lifetime, so it is genuinely zero-copy. The same
caveat as DXcam applies: a view kept past the next grab is not protected, and
silently shows whatever frame its buffer holds next. Copy what you need to keep.

Migrate a call site at a time: `camera.rapidshot_camera` is the RapidShot camera
underneath, and any attribute the shim does not define is forwarded to it. Once
nothing uses the shim, `import rapidshot` and add `release()` to drop the copy.
Code that reaches into DXcam internals such as `camera._duplicator` is not
portable, and the shim does not pretend otherwise.

## Trading CPU for frames

Each capture waits up to `timeout_ms` for the compositor to present. That single
number is most of the difference between this library and the polling ones:

```python
camera = rapidshot.create(timeout_ms=0)   # poll instead of waiting
camera.timeout_ms = 10                    # or change it on a live camera
```

Measured on a 100 Hz output against a source presenting at ~610 updates/s
(ROADMAP.md section 3):

| `timeout_ms` | frames/s | hit rate | CPU |
| --- | --- | --- | --- |
| 0 (poll) | 127.8 | 2.4% | 68.6% |
| 1 | 119.2 | 74.5% | 19.5% |
| **10 (default)** | 118.9 | 100% | **15.7%** |

The frame rate barely moves across that range while CPU moves by more than four
times. Polling is what DXcam does, and the frames it buys are real — just
expensive. Use `0` if capture is the only thing running; leave the default if it
is one stage of a pipeline that needs its cores.

Frames beyond the display's refresh rate were never shown as distinct images.
They are extra temporal samples for a model, and redundant for a recorder.

## Only process what changed

Frames from `grab_frame()` carry the compositor's dirty-rect metadata:

```python
frame = camera.grab_frame()               # None when nothing changed
if frame is not None:
    with frame:
        if not frame.dirty_rects or frame.changed_fraction > 0.5:
            process_everything(frame)
        else:
            for left, top, right, bottom in frame.dirty_rects:
                process_region(frame, left, top, right, bottom)
```

Coordinates are relative to the frame, so they index straight into the captured
image even when `region=` is in use.

`changed_fraction` is the share of the frame the rects cover, from 0.0 to 1.0,
with overlaps counted once — drivers do report overlapping rects, and summing
their areas can exceed the frame. It follows the rules below: `None` when the
metadata could not be read, `1.0` when the list is empty.

Two things to get right. An **empty list does not mean nothing changed** — it
means no rects were reported, which a mode change or a coalescing driver can
also produce while the image differs completely. Treat it as "assume everything
changed". **`None` means the metadata could not be read at all.** And check
`frame.rects_coalesced`: when true the driver merged rects, so they
over-estimate what actually changed.

## Colour formats

```python
rapidshot.create(output_color="RGB")    # (H, W, 3) -- default
rapidshot.create(output_color="RGBA")   # (H, W, 4)
rapidshot.create(output_color="BGR")    # (H, W, 3) -- OpenCV order
rapidshot.create(output_color="BGRA")   # (H, W, 4) -- raw, no conversion
rapidshot.create(output_color="GRAY")   # (H, W, 1) -- Rec. 601 luma
```

An unsupported value raises `ValueError` at creation. Conversion uses the
native AVX2 kernels when the extension is present and NumPy otherwise; the two
are byte-identical. OpenCV is never required.

## Capturing into your own buffer

`shot()` writes straight into memory you own, avoiding a per-frame allocation:

```python
import numpy as np

camera = rapidshot.create(output_color="RGB", region=(0, 0, 640, 480))
buffer = np.zeros((480, 640, camera.channels), dtype=np.uint8)
if camera.shot(buffer):
    print("captured", buffer.shape)
```

The destination size is checked before anything is written, so an undersized
buffer raises `ValueError` instead of corrupting memory. NumPy arrays, `ctypes`
arrays, `bytearray` and `memoryview` all report their own size. A raw pointer
cannot, so it needs an explicit `buffer_size`:

```python
import ctypes
camera.shot(ctypes.c_void_p(buffer.ctypes.data), buffer_size=buffer.nbytes)
```

## GPU-resident capture

`grab()` brings every frame to the CPU — a staging read plus a conversion. If
you are handing pixels to a GPU consumer, `grab_frame()` skips that and gives
you the Direct3D texture:

```python
frame = camera.grab_frame()              # None when nothing changed
if frame is not None:
    with frame:
        texture = frame.d3d11_texture    # ID3D11Texture2D, valid inside the block
        print(frame.timestamp, frame.accumulated_frames)
```

> **The `with` block is not optional.** Direct3D cannot capture the next frame
> while a reference to the previous one is outstanding, so an unreleased frame
> stalls capture entirely. Use the context manager or call `frame.release()`,
> and copy out anything you need before the block ends. `grab()`, `shot()` and
> `grab_frame()` all raise a clear error if a frame is still outstanding.
>
> Check for `None` *before* the `with`: `grab_frame()` returns `None` when
> nothing changed, and `with None` raises `TypeError`.

Metadata stays readable after release: `timestamp` / `timestamp_qpc` (when the
compositor presented the frame), `accumulated_frames` (greater than 1 means the
OS coalesced presents because your loop fell behind), `protected_content`,
`cursor_visible`, `region`, `width`, `height`, `rotation_angle`, and:

| | |
| --- | --- |
| `sequence` | This frame's index within the camera, from 1. Continues across recoveries, so it identifies one frame in a log. |
| `generation` | How many times capture had rebuilt itself when the frame was taken. See [Surviving display changes](#surviving-display-changes). |
| `age_ms` | How old the pixels are *now*, measured from the present timestamp — not how long a call took. It grows while you hold the frame, so read it just before handing off. `0.0` means no present time was reported. |
| `changed_fraction` | Share of the frame covered by dirty rects. See [Only process what changed](#only-process-what-changed). |
| `cursor` | Cursor position, hotspot and shape. See [Cursor](#cursor). |

## The GPU tensor

Requires the [native extension](#the-optional-native-extension). One compute
dispatch resizes, normalises, converts BGRA to RGB and transposes to NCHW, and
the result stays in VRAM:

```python
from rapidshot import native

with camera.grab_frame() as frame:
    pre = native.GpuPreprocessor12(frame, 640, 640)   # build once, reuse
    pre.process(frame)                                # one dispatch

    print(pre.shape)                        # (1, 3, 640, 640)
    resource = pre.output_resource_address  # ID3D12Resource*
    gpu_va = pre.output_gpu_address
    handle = pre.shared_output_handle       # shared NT handle, for CUDA
```

Optional arguments cover the usual normalisation ranges (`scale=2.0, bias=-1.0`
for −1..1) and channel order (`bgr=True`).

### Reading it from CuPy

`shared_output_handle` is imported by CUDA with `cudaImportExternalMemory`, so
the frame reaches a CUDA consumer without touching the CPU.
**`examples/gpu_tensor_to_cupy.py` is a complete working version** of the ~60
lines of `ctypes` this takes, and verifies the resulting `cupy.ndarray` is
byte-identical to a readback of the same dispatch.

`CudaTensor` lives in that example rather than in the package — copy it into
your project:

```python
with CudaTensor(pre, (1, 3, 640, 640)) as view:
    tensor = view.array                            # a cupy.ndarray in VRAM
    while capturing:
        frame = camera.grab_frame()                # a *new* frame each pass
        if frame is None:
            continue
        with frame:
            view.sync()                            # queued CUDA work must finish
            pre.process(frame)                     # overwrites the tensor buffer
        model(tensor)
```

Three things that bite:

- **A fresh `grab_frame()` each pass.** The preprocessor and the CUDA import are
  built once; the frame is not. A released frame raises `FrameReleasedError`,
  and reusing a live one just re-processes the same image.
- **`sync()` is not optional.** `process()` waits on its D3D12 fence but knows
  nothing about CUDA work you queued against the same memory. A kernel still
  reading the tensor when the next dispatch lands sees a half-overwritten
  frame — no error, just wrong numbers.
- **Release before `model()`.** DXGI cannot acquire the next frame while a
  reference to the previous surface is outstanding.

Keep the `CudaTensor` alive as long as you use the array: it owns the CUDA
import and the preprocessor that owns the VRAM. With more than one CUDA device,
pass `device=N` — the tensor can only be imported by the device owning the
adapter that captured the frame.

### DirectML / ONNX Runtime

The output is an `ID3D12Resource`, which is what ONNX Runtime's DirectML
provider consumes. RapidShot deliberately does not bind it to a session — that
would couple this library to ONNX Runtime's ABI and release cadence for an
optional feature:

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

> `OrtDmlApi` has no Python binding — it is reachable only from C/C++. That is a
> gap in ONNX Runtime, not in RapidShot. From Python today you need a small
> native shim of your own; `native.probe_onnxruntime()` and
> `native.onnxruntime_dll_path()` help locate and validate the runtime.

## Hybrid GPU laptops

On an Optimus/switchable laptop the discrete GPU usually drives no display, and
Desktop Duplication cannot run against an adapter that drives no display. Capture
is therefore bound to the integrated GPU. `device_info()` lists only adapters
that can capture; `topology_info()` lists every adapter and explains the
consequence:

```python
print(rapidshot.topology_info())
```

```
Topology: hybrid
  Adapter[0] (Intel(R) UHD Graphics) (Intel) (128MB VRAM) (1 output)
  Adapter[1] (NVIDIA GeForce RTX 4060 Laptop GPU) (NVIDIA) (7956MB VRAM) (0 outputs)
  ...
```

`grab()` is unaffected. A GPU-resident frame is: it lives on the capture
adapter, so feeding it to a model on the *other* adapter needs a cross-adapter
copy.

```python
from rapidshot import native

with camera.grab_frame() as frame:
    transfer = native.cross_adapter_transfer(frame)   # build once, reuse

with camera.grab_frame() as frame:
    transfer.transfer(frame)
    print(transfer.source, "->", transfer.destination)
    # bind transfer.destination_resource_address on the other adapter
```

The shared heap lives in **system memory**, not either adapter's VRAM. This is
not peer-to-peer VRAM-to-VRAM DMA; the win is that a GPU copy engine moves the
bytes instead of CPU cores.

### Asynchronous transfer

`transfer()` blocks until the copy completes, which on a hybrid laptop is most
of its cost. `transfer_async()` returns the calling thread instead:

```python
value = transfer.transfer_async(frame)
transfer.wait_shared_fence(value)          # CPU wait, or...
handle = transfer.shared_fence_handle      # ...import into CUDA, wait GPU-side
```

**Measure the GPU-side wait for your own consumer before adopting it.** Against
a synthetic CuPy consumer, Intel iGPU to RTX 4060 at 2560×1600, the CPU-side
async wait bought nothing — the calling thread was never the constraint — and
importing `shared_fence_handle` with `cuImportExternalSemaphore` to wait on it
in a stream was worth **7–14%** (two runs disagreed by a factor of two on the
margin). In the end-to-end [pixel-age benchmark](#desktop-to-model) on the same
laptop it did not pay: no younger pixels than the blocking transfer, and more
CPU per frame. Which one you see depends on what else your stream is doing.

`transfer()` remains the default and still holds the GIL for its copy.
`wait_shared_fence()` releases it.

### Waiting for the consumer

Every transfer writes the **same destination buffer**, and the shared fence only
reports that the *copy* finished. A consumer still reading frame N can be
overwritten by the copy for frame N+1. Hand the producer a fence the consumer
signals when it has finished reading:

```python
transfer.set_consumer_fence(consumer_fence_handle)   # once
transfer.wait_for_consumer(value)                    # queued before the next copy
```

The wait is enqueued on the source queue, so it orders ahead of the next copy
without blocking the caller.

**This is not theoretical.** Over a 60-frame loop whose consumer ran slower than
the producer: **28 of 60 frames wrong without the handshake, 0 with it.** With a
consumer that keeps up, the same loop is clean either way over 100 frames —
which is why the hazard stays invisible until a real workload arrives.

`shared_fence_submitted` and `shared_fence_completed` expose what was queued
versus what the GPU has reached, if you need to watch the handshake work.

## Surviving display changes

A mode change, a monitor arriving or leaving, exclusive fullscreen or a device
reset invalidates Desktop Duplication. RapidShot rebuilds and carries on, with
bounded retries and backoff. It also tells you it did:

```python
camera.generation            # 0 until the first rebuild, +1 per successful one
camera.recovery_count        # the same count, read as a health counter
camera.last_recovery_reason  # e.g. "DXGI device error during update_frame", or None
```

Every frame from `grab_frame()` is stamped with the generation it came from.
That matters for anything built from one frame and reused: a rebuilt duplicator
may differ in size, rotation or format, so a `GpuPreprocessor12` or a
cross-adapter transfer made before the rebuild describes a surface that is gone.
Rebuild it when the generation moves:

```python
pre, built_for = None, None

frame = camera.grab_frame()
if frame is not None:
    with frame:
        if frame.generation != built_for:
            pre = native.GpuPreprocessor12(frame, 640, 640)
            built_for = frame.generation
        pre.process(frame)
```

`grab()` returns pixels without this metadata; compare `camera.generation`
between calls instead.

The generation moves only on a *successful* rebuild, so a failed attempt that is
about to be retried does not make you throw a cache away for nothing. A
`recovery_count` that keeps climbing on an otherwise idle machine means capture
is being torn down and rebuilt repeatedly, and is worth investigating.

## Headless machines

With no monitor attached there is no desktop to duplicate, and
`rapidshot.create()` raises `HeadlessError` explaining that a virtual display
driver (IDD) is needed. `topology_info()` still works — it probes DXGI directly
rather than going through capture.

> A virtual display's advertised refresh rate does **not** raise capture rate.
> Desktop Duplication is driven by presents, not refresh: a 500 Hz virtual
> display does not make an application render 500 fps.

## The optional native extension

Everything above except the GPU tensor and cross-adapter transfer works without
it. `pip install rapidshot` never requires Rust.

**Install the prebuilt wheel:**

```bash
pip install rapidshot-native
```

Windows x86-64, built as an `abi3` wheel so one binary serves Python 3.9 and
every later version — including ones released after the build. AVX2 is detected
at runtime with scalar fallbacks, so it runs on pre-AVX2 hardware too.

Nothing else changes: RapidShot finds it automatically and
`rapidshot.native.is_available()` starts returning `True`.

**Or build it from source**, which needs [Rust](https://rustup.rs) 1.88+ and the
MSVC C++ build tools:

```bash
cd native && cargo build --release
python native/install_dev.py
```

A build made this way **takes precedence** over an installed wheel, so
rebuilding does what you expect when both are present. `native.build_info()`
reports which one is loaded under `source`.

`rapidshot-native` is versioned independently of `rapidshot`, because the Rust
changes on its own schedule. `rapidshot` declares the minimum it needs, so pip
resolves a working pair; if you pin, pin both.

## Profiling your own loop

The figures below describe this project's machines. For yours:

```python
from rapidshot.profiling import Profiler

profiler = Profiler("inference loop")
with profiler:
    for _ in range(1000):
        with profiler.time("grab"):
            frame = camera.grab_frame()
        profiler.observe(frame)                # None is counted, not skipped
        if frame is None:
            continue
        with frame:
            with profiler.time("process"):
                process(frame)

print(profiler.report())    # a table, for a human
profiler.summary()          # a dict, for asserting on
profiler.json()             # a string, for storing beside a baseline
```

It reports the minimum and percentiles, **never a mean**. Background load can
only make a sample slower, so the minimum is the least contaminated estimate and
the tail is what a real-time consumer feels; a mean hides both. Any stage with
fewer than 30 samples is flagged `low_confidence` — a p99 over a dozen samples
is just the largest of them.

`observe()` records what wall clock cannot show. `coalesced_updates_missed`
counts display updates the OS folded together because the loop fell behind, so
a loop that looks fast while dropping most of what it should capture says so.
If capture rebuilt mid-run, `recoveries_during_run` says that too: timings
either side of a recovery describe different duplicators and should not be
pooled.

Stages nest, and a sample is recorded even when its block raises. One profiler
per thread — it is deliberately not thread-safe.

## Measurements

Every figure below comes from a committed recording, named so you can check it.
Reproduce with:

```bash
python benchmarks/perf_suite.py --rounds 5 --reps 25 --compare auto
```

`--compare auto` selects the baseline recorded on *your* machine and fails if
there is none, rather than comparing you against someone else's hardware.
Run `--self-test` first to see your machine's noise floor.

### RapidShot's own paths

Synthetic, deterministic, 1920×1080. These move when the library changes and
not otherwise, which is why they are the badge figures.

From [`benchmarks/baseline.json`](benchmarks/baseline.json) — Intel iGPU,
**with** the native extension — against
[`benchmarks/baseline-nonative.json`](benchmarks/baseline-nonative.json), the
same machine **without** it:

| | with extension | without |
| --- | --- | --- |
| BGRA→RGB | **0.30 ms** | 1.93 ms |
| BGRA→GRAY | **0.26 ms** | 6.88 ms |
| `shot()` → your buffer | **0.31 ms** | 1.96 ms |

From [`benchmarks/baseline-rtx4060-hybrid.json`](benchmarks/baseline-rtx4060-hybrid.json)
— a second machine, Intel iGPU capture with an RTX 4060 present, recorded at
2.4.0:

| | per frame |
| --- | --- |
| BGRA→RGB | 0.22 ms |
| BGRA→GRAY | 0.25 ms |
| CPU resize + normalise + NCHW | 3.12 ms |
| GPU dispatch (submission cost only) | 0.001 ms |
| GPU dispatch + forced readback | 2.56 ms |

**The dispatch row is submission cost, not GPU execution.** The calling thread
returns in a microsecond because the GPU works asynchronously; the readback row
is the honest worst case, and it is *slower* than the CPU arm. That is the
point — the GPU path wins only when the tensor is consumed on the GPU. Pulling
it back to the CPU gives up the entire advantage.

The two machines are not comparable to each other: different CPU, GPU and panel.
Each answers for itself, which is why both recordings are kept.

### End-to-end capture is not a stable number

`grab()` converts only the regions that changed, so its cost tracks what is
happening on screen. Across seven recordings of unchanged code, `grab_frame()`
alone spanned **0.17–0.83 ms** and `grab()` **1.65–4.53 ms**. These are reported
for shape, not precision, and are deliberately not badged.

Neither is a capture rate. They are what the *calling thread* pays per frame;
frames still arrive only as fast as the compositor presents them.

Sustained-load check: 211,726 frames of `grab_frame()` → GPU tensor in 12
minutes, zero errors, VRAM flat, no throughput decay.

## Against other libraries

Two capture-only recordings on different machines, then one that follows the
frame to a model-ready tensor. **They are not comparable to each other** —
different panels, different resolutions, different RapidShot versions,
different workloads — and none predicts your hardware.

Reproduce with:

```bash
python benchmarks/compare_libraries.py --motion --with-motion
```

Every library runs in its own process: all three DXGI libraries declare the same
COM interfaces, and whichever imports first breaks the others.

### Machine A — 1080p, 100 Hz, RapidShot 2.1.0

[`benchmarks/library-comparison.json`](benchmarks/library-comparison.json),
recorded 2026-08-06. Fullscreen BGRA, median of three runs:

| | frames/s | CPU | memory |
| --- | --- | --- | --- |
| RapidShot | 99.9 | **13.6%** | 124.4 MB |
| RapidShot (`timeout_ms=0`) | 100.1 | 95.9% | 125.2 MB |
| DXcam | 100.5 | 79.7% | **87.1 MB** |
| BetterCam | 100.5 | 74.4% | **80.0 MB** |
| mss | 48.1 | 35.0% | **60.4 MB** |

**Nobody beats the compositor.** Every DXGI library lands at ~100 fps here
because that is where the ceiling is. A frame-rate win in this space is almost
always measuring something else — most often a still desktop returning stale
buffers instantly.

### Machine B — 2560×1600, RapidShot 2.4.0

[`benchmarks/library-comparison-machineB.json`](benchmarks/library-comparison-machineB.json),
recorded 2026-08-22 with `cv2 5.0.0` (32 threads) and `cupy 14.2.0`. Fullscreen.

**Medians pooled across independent full-matrix runs**, each of which is itself
the median of 3–5 repeats, so most cells rest on 9–15 measurements. The `n`
column says how many runs each row pools; the per-run minimum and maximum are in
the JSON. Pooling mattered — on a single run the AVX2 and NumPy builds looked
identical, and across four they do not.

Each library is configured for its **best** available path rather than its
default. DXcam ships a NumPy processor beside its cv2 one, BetterCam has a CuPy
path, and RapidShot has a CuPy path, an unpooled mode, and the toolchain-free
NumPy build that a plain `pip install` actually produces:

| | n | BGRA fps / CPU | RGB fps / CPU | memory |
| --- | --- | --- | --- | --- |
| RapidShot (AVX2) | 5 | 153.2 / **35.5%** | 124.4 / **65.2%** | 238 MB |
| RapidShot (NumPy, no extension) | 3 | 153.7 / 33.3% | 113.0 / 70.0% | 238 MB |
| RapidShot (CuPy) | 2 | — | 124.2 / 66.8% | 260 MB |
| RapidShot (`pool_output=False`) | 2 | — | 102.5 / 73.3% | **189 MB** |
| RapidShot (`timeout_ms=0`) | 4 | 151.8 / 53.5% | 122.4 / 73.2% | 234 MB |
| DXcam (cv2) | 4 | 129.1 / 60.0% | **146.8** / 450.8% | **102 MB** |
| DXcam (NumPy) | 3 | 136.4 / 61.2% | **150.5** / 2692.2% | **91 MB** |
| BetterCam (cv2) | 4 | 154.6 / 53.5% | **142.6** / 458.6% | **98 MB** |
| mss | 4 | 32.9 / 42.2% | 21.9 / 49.8% | **83 MB** |

**RapidShot loses the RGB frame-rate column.** DXcam and BetterCam return
142–151 frames per second against RapidShot's 124. It also uses roughly 2.4x
their memory. Both are real, and neither is argued away here.

**What it wins is CPU time.** Isolating conversion by subtracting each library's
BGRA cost from its RGB cost:

| | CPU cost of BGRA→RGB |
| --- | --- |
| **RapidShot (AVX2)** | **+29.7 points** |
| **RapidShot (NumPy)** | **+36.7 points** |
| BetterCam (cv2) | +405.1 points |
| DXcam (cv2) | +390.8 points |
| DXcam (NumPy) | +2631.0 points |

**Read that carefully, because the obvious reading is wrong.** Those are
CPU-*time* percentages summed across cores, not wall-clock. OpenCV 5.0 defaults
to one thread per logical core — 32 on this machine — so `cvtColor` spends many
cores to finish quickly. Measured directly here, BGRA→RGB at 2560×1600 costs
2.22 ms multithreaded and 4.36 ms pinned to a single thread. DXcam still
returned *more* frames per second than RapidShot while doing it. The defensible
claim is "the same work for far less total CPU", not "faster", and the exact
multiple will move with your core count and OpenCV version. A machine with 8
logical cores will not reproduce these ratios.

### What the extension is actually worth

The AVX2 kernels are **6.4x** the NumPy path on the synthetic conversion
benchmark (1.93 ms against 0.30 ms, both from Machine A), and **1.10x**
end-to-end here — 124.4 against 113.0 fps, and +29.7 against +36.7 points of
conversion CPU. Both numbers are honest and they measure different things: at
2560×1600 the staging read dominates `grab()`, so making conversion six times
cheaper moves the total by about a tenth.

Since `pip install rapidshot[native]` costs nothing but a download, this is no
longer much of a decision — but it is worth knowing what you are getting. The
extension earns its keep through the GPU tensor and cross-adapter transfer,
which have no alternative at all. `grab()` gets a modest gain, not a
transformation, and the `rapidshot-numpy` row above is what you fall back to
without it.

### What buffer pooling is worth

`pool_output=False` returns plain `ndarray`s and costs **1.21x** on fullscreen
RGB — 124.4 against 102.5 fps — while saving about **50 MB**. At region size
both sit at the compositor ceiling (~165 fps) and the difference disappears
entirely.

This is the first live-capture measurement of that trade; the 1.3–2.1x quoted
elsewhere for pooling came from a synthetic benchmark and is not reproduced in a
capture loop. If memory matters more to you than fullscreen frame rate,
`pool_output=False` is a reasonable trade rather than a downgrade.

### Caveats

This table is noisier than it looks, and the harness says so itself.

- Several cells disagreed by more than 10% across their runs; the spreads are
  recorded per row in the JSON.
- `bettercam-gpu` fails on RGB with `Expected Ptr<cv::UMat>` — its CuPy path
  hands a device array to OpenCV 5, which rejects it. A BetterCam/OpenCV-5
  incompatibility, recorded rather than hidden.
- The GPU rows pool only runs taken *after* CuPy had its CUDA headers. Without
  them CuPy fails at its first JIT and those rows record no frames at all —
  which is a packaging trap, not a capture result. See
  [Troubleshooting](#troubleshooting).
- The two machines are **not** comparable to each other. Machine A ran a 1080p
  100 Hz panel at 2.1.0; this one is 2560×1600. Most of the frame-rate
  difference between the tables is the panel.

**Published FPS claims in this space contradict each other badly** — DXcam's
README reports DXcam at 239 fps, BetterCam's reports the same library at 39 —
because they come from different hardware with no shared harness. A number
measured on someone else's machine tells you nothing about yours, this page
included.

### Desktop to model

The tables above stop at `grab()`. This one follows the frame to where most of
RapidShot's work is aimed: a `(1, 3, 640, 640)` FP16 tensor on CUDA, ready for
a model.

It measures **pixel age**, not call duration. A test source encodes an
incrementing frame ID into the image and records the time of every `Present()`;
each path decodes the ID from what it captured, so every library is timed on
one clock by how old its pixels were. RapidShot's own present timestamps would
have given it an advantage no other library could match, so they are not used.

[`benchmarks/section7-ingestion-machineB.json`](benchmarks/section7-ingestion-machineB.json),
recorded 2026-09-11 — Machine B, Intel iGPU capture with an RTX 4060 doing the
CUDA work, 2560×1600 at 165 Hz. **Medians across 3 passes, 8 s per path**; every
path was verified to produce the correct tensor before it was timed:

| | unique frames/s | pixel age p50 / p95 | CPU per frame |
| --- | --- | --- | --- |
| mss | 33.0 | 57.6 / 60.7 ms | 15.3 ms |
| DXcam (DXGI) | 108.2 | 36.6 / 39.6 ms | 8.9 ms |
| DXcam (WGC) | 106.0 | 40.7 / 45.1 ms | 8.7 ms |
| RapidShot `grab()` | **140.5** | **33.6 / 36.1 ms** | 7.7 ms |
| RapidShot `grab()`, `nvidia_gpu=True` | 128.8 | 34.4 / 36.6 ms | **4.4 ms** |
| RapidShot cross-adapter | 80.2 | 35.1 / 40.2 ms | 4.9 ms |
| RapidShot cross-adapter, async | 80.8 | 34.6 / 40.3 ms | 5.0 ms |
| RapidShot cross-adapter, GPU-side wait | 81.4 | 36.0 / 37.0 ms | 11.0 ms |

**`grab()` and `nvidia_gpu=True` beat DXcam on every column, on every pass.**
`grab()` returns **30% more unique frames** with pixels **8% younger**;
`nvidia_gpu=True` does it at **half DXcam's CPU**. Each one's worst pass still
beats DXcam's best.

**The cross-adapter paths are a trade, not a win.** They are capped at about 80
frames a second on this laptop — 25% fewer than DXcam — most likely by the 16 MB
copy each frame makes through system memory, since all three variants hit the
same ceiling. In return the blocking and async variants
cost 45% less CPU than DXcam with no host-to-device copy, and still return
younger pixels. The GPU-side-wait variant gains nothing here and costs more CPU
than DXcam. [Hybrid GPU laptops](#hybrid-gpu-laptops) covers the cross-adapter
path.

This replaces a single 5-second pass from the day before, which had the
cross-adapter paths matching DXcam's frame rate. They reproduced within 2%; the
paths that read frames back to the CPU all ran 38–56% faster in the second
session, for reasons not established. That is why every figure here is a median
of three.

Read it with these caveats:

- **One machine, and a hybrid one.** The direct single-adapter GPU path could
  not run here; on a machine whose NVIDIA GPU drives the display it is expected
  to beat every row above, and it has not been measured.
- **Every path drops source frames.** The source presented at a median 165/s
  and no path keeps up, so these are throughput and latency under load — not a
  best case. It dipped briefly below `grab()`'s rate, so that row's frame rate
  is, if anything, understated.
- **Age starts at `Present()` submission**, so it includes the compositor's
  queue but not scan-out to the panel.
- **This stops at the tensor.** The next table carries the frame through a
  trained model.

#### Through a trained model

The same clock, carried one step further: each path's tensor goes into
**YOLO11n** — the official Ultralytics weights — on the RTX 4060 through ONNX
Runtime's CUDA provider, and age is taken when the forward pass completes.

[`benchmarks/section7-inference-machineB.json`](benchmarks/section7-inference-machineB.json),
recorded 2026-09-11 on the same machine. **Medians across 3 passes, 8 s per
path**; every path was verified to produce the correct tensor before it was
timed.

| | unique frames/s | pixel age p50 / p95 | CPU per frame |
| --- | --- | --- | --- |
| mss | 26.8 | 53.7 / 64.0 ms | 20.3 ms |
| DXcam (DXGI) | 68.1 | 42.2 / 45.8 ms | 13.4 ms |
| DXcam (WGC) | 65.8 | 45.4 / 49.8 ms | 14.6 ms |
| RapidShot `grab()` | 80.3 | 39.3 / 43.3 ms | 12.7 ms |
| RapidShot `grab()`, `nvidia_gpu=True` | **81.8** | 38.8 / 42.5 ms | **8.9 ms** |
| RapidShot cross-adapter | 75.2 | 39.6 / 42.6 ms | 8.9 ms |
| RapidShot cross-adapter, async | 76.2 | 39.6 / 42.8 ms | 9.1 ms |
| RapidShot cross-adapter, GPU-side wait | 81.0 | **38.3 / 41.1 ms** | 11.9 ms |

**Every RapidShot path beat DXcam (its default DXGI backend) on every pass** —
more frames, younger pixels and less CPU per frame, with RapidShot's worst pass
still ahead of DXcam's best.
At the medians: up to **20% more frames**, pixels up to **3.9 ms (9%) younger**
when the model finishes, and up to **a third less CPU** per frame.

The lead is smaller than at the tensor (20% more frames here, 30% there), which
is expected: this loop runs capture and inference back to back, so YOLO11n's
4–5 ms per frame paces every path alike. RapidShot's own paths finish within
1.3 ms of each other, inside their pass-to-pass spread — pick between them on
frames and CPU rather than latency.

Caveats specific to this table: it times the raw forward pass, with no NMS or
postprocessing; the model file is an export of the official weights made with
Ultralytics' own exporter, because the published `yolo11n.onnx` targets an ONNX
opset that ONNX Runtime 1.30 cannot run entirely on the GPU; and it is one
machine. [ROADMAP.md](ROADMAP.md) section 7.0 has the spreads, the model's
provenance, and what is still unmeasured.

Reproduce with the test source built and the model exported first:

```bash
cargo build --release --bin latency_source --manifest-path native/Cargo.toml
python benchmarks/section7.py --seconds 5 --out results.json
python benchmarks/prepare_model.py
python benchmarks/ai_pipeline.py --seconds 8 --model build/section7/model/yolo11n.onnx --model-sha256 <sha256 printed by prepare_model.py>
```

### What these tables do not show

Dirty-rect metadata and GPU-resident frames from `grab_frame()` have no column,
because the other libraries have no equivalent to measure them against. The
single-dispatch NCHW tensor from `GpuPreprocessor12` is not in the tables above
either: on this hybrid machine CUDA cannot import it, which is the case
cross-adapter transfer exists for.

## System requirements

- **OS:** Windows 10 or newer. Windows only — Desktop Duplication has no
  cross-platform equivalent.
- **Python:** 3.9+. Tested through 3.14.
- **GPU:** any GPU that drives a display. A CUDA-capable NVIDIA GPU is needed
  only for the CuPy path.
- **RAM:** 8 GB+, depending on resolution and how many instances you create.

### Tested configurations

Nothing in RapidShot branches on GPU vendor. This is what has actually been run,
which is a different claim:

| Configuration | State |
| --- | --- |
| Intel iGPU, single adapter | Verified |
| NVIDIA dGPU, single adapter, native extension built | Verified |
| NVIDIA dGPU, single adapter, no extension | Verified |
| Hybrid Intel iGPU + NVIDIA dGPU (Optimus) | **Verified** — 2026-08-22 |
| Cross-adapter transfer, Intel iGPU → RTX 4060 | **Verified** byte-exact |
| Any AMD GPU | **Not tested** |
| Hybrid with an AMD adapter | **Not tested** |
| Headless with a virtual display | **Not tested** |

The Optimus row is verified by `examples/verify_cross_adapter.py`, which moved
5 captured frames from an Intel iGPU to an RTX 4060 — 16,384,000 bytes each at
2560×1600 — every one byte-exact against a source-side readback.

The AMD rows are genuinely untested. The BGRA swizzle rule is verified on Intel
and NVIDIA drivers and follows from the DXGI format rather than driver
discretion, so the risk is low — but it is the one rule a silent mismatch would
corrupt rather than crash. AMD's cross-adapter capability flags are unknown; the
buffer path was chosen so nothing depends on them.

## Diagnostics

```python
import rapidshot
print(rapidshot.diagnose())
```

One report: the RapidShot version, whether the native extension loaded and from
where (a local build and the wheel can both be installed, and differ), the
adapter topology and whether it is hybrid, and which optional dependencies —
NumPy, comtypes, CuPy, OpenCV, PIL, ONNX Runtime — are importable, with versions.
Most "it does not work" reports are answerable from this alone, so paste it into
any issue you open.

It never raises. Each section is independent, and one that fails reports its
error while the rest still print — the machine where this matters most is the
one where something is already broken.

`rapidshot.capabilities()` returns the same report as a dict, for code to branch
on. Pass `probe_gpu=True` to either to add the cross-adapter probe; it is off by
default because it creates D3D devices and allocates a shared heap, and a
diagnostic should not be able to destabilise the thing it is diagnosing.

## Troubleshooting

Run [`rapidshot.diagnose()`](#diagnostics) first; it answers most of these.

- **CuPy fails at its first JIT** with `Failed to find CUDA headers`. The wheel
  is not enough — `pip install cupy-cuda13x[ctk]` installs no headers against
  `cuda-toolkit` 13.3.x, because that version dropped the extras it requests and
  pip only warns. Pin it instead:

  ```bash
  pip install "cuda-toolkit[cudart,nvrtc]==13.2.*"
  ```

  Until this is fixed, `nvidia_gpu=True` returns no frames — CuPy imports, so
  the CPU fallback does not trigger.

- **"Desktop duplication was denied."** The message names the cause. A
  non-input desktop, a locked workstation, an open UAC prompt and a Session 0
  service cannot capture the user's screen. Protected (HDCP/DRM) content is a
  separate case and is reported as such.

- **Black frames.** Protected content is blanked by the OS — check
  `frame.protected_content`. Exclusive fullscreen is handled: capture detects
  the transition, rebuilds and continues.

- **`grab()` keeps returning `None`.** Usually nothing is changing on screen.
  Desktop Duplication reports presents, not refreshes, so a still desktop
  produces no frames. This is the single most common cause of a benchmark
  reading zero.

- **On a hybrid laptop, every adapter refuses `DuplicateOutput`.** Set NVIDIA
  Control Panel → Manage 3D Settings → Preferred graphics processor →
  Integrated graphics. Without it the NVIDIA driver can claim the display while
  the firmware reports Optimus, and every adapter — including WARP — returns
  `DXGI_ERROR_UNSUPPORTED`.

- **Unstable benchmark numbers.** On a hybrid P-core/E-core CPU, pin to the
  performance cores. `benchmarks/perf_suite.py` does this itself; unpinned it
  reported false regressions up to 2.57x against unchanged code.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). It is mostly a list of things that will
waste your time otherwise: live capture tests need something moving on screen,
CI cannot verify them at all, and a naive benchmark comparison here once
produced eleven false regressions on identical code.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

A merge of, and successor to:

- Original DXcam by ra1nty
- dxcampil — PIL-based version
- DXcam-AI-M-BOT — cursor support version
