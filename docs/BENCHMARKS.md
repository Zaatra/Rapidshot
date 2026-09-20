# RapidShot benchmarks

Everything measured, with the methodology and the caveats, moved out of
`README.md` so the README can describe what the library *is* rather than read as
a benchmark paper. Nothing here has been re-edited or re-run for 2.6 — it is the
2.1 / 2.4 / 2.5-era record, kept because a measurement is only worth what its
context is worth, and deleting the context would leave numbers nobody can check.

**For the current 2.6 figures**, see [Performance](../README.md#performance) in
the README. The one number restated there and here is memory, and 2.6 changed
it substantially — the tables below predate that change.

The raw recordings are committed beside this file's sources in
[`benchmarks/`](../benchmarks/); every table names the JSON it came from.

---

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

- **One machine.** Every row above was taken with the laptop's MUX in hybrid
  mode: capture on the Intel iGPU, CUDA on the RTX 4060. The direct
  single-adapter path cannot run in that mode at all — it fails with
  `CrossAdapterRequired`, because no CUDA device owns the adapter the frame was
  captured on.

  **It has since been measured, with the MUX switched to discrete-only** so the
  NVIDIA card drives the display (2026-09-13, same laptop, 2560x1600). Verified
  byte-exact first, then timed:

  | | unique frames/s | pixel age p50 | CPU per frame | host-to-device |
  | --- | --- | --- | --- | --- |
  | RapidShot direct, single adapter | 89.4 | **31.4 ms** | 6.7 ms | **0 bytes** |

  **The pixels are the youngest measured anywhere on this page** — 2.2 ms ahead
  of the best hybrid row and 5.2 ms ahead of DXcam — which is what you would
  expect from the only path that never crosses an adapter.

  Two things stop this being a like-for-like row in the table above, and both
  matter more than the headline. It is **one 8-second pass, not a median of
  three**. And it ran in **a different machine configuration**: with the NVIDIA
  card driving the display, the compositor's present behaviour is not the same,
  so the frame-rate column cannot be compared across the two modes. 89.4 against
  `grab()`'s 140.5 is not a regression, it is a different machine. Pixel age and
  CPU per frame are the columns that travel.
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
