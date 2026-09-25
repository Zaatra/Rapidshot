# Changelog

All notable changes to Rapidshot are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are grouped by the [ROADMAP.md](ROADMAP.md) stage that motivated them, so
each release can be traced back to the plan it implements.

## [Unreleased]

### Added — `rapidshot benchmark`

- **One command reproduces the README's desktop-to-model table on any
  machine.** `pip install "rapidshot[benchmark]"` then `rapidshot benchmark`
  (also `python -m rapidshot benchmark`): it finds which capture libraries are
  installed and which RapidShot paths the hardware supports, verifies each one
  produces the correct tensor, times three 8-second passes by pixel age, and
  writes a Markdown report plus a JSON file for a GitHub issue. `--check` says
  what would run and why without capturing; `--full` adds CPU per tensor and
  memory.
  Every path it does not measure is listed with the command that would enable
  it, rather than left out.
- **Capabilities beside the timings.** Every report records whether each
  display supports and has on HDR -- the Windows 11 24H2 query, which tells
  HDR apart from SDR auto colour management -- next to the DXGI format a
  capture actually received, and whether a frame can cross to a second
  adapter, flagged when the only second adapter is WARP and the copy time
  therefore says nothing about real hardware. `--full` adds memory: working
  set, what capture adds over the same process before its first frame, and
  growth as a slope, per library, on a static, a scrolling and a full-motion
  screen. Every harness recording now carries the display's advanced-colour
  state too.
- **The report is sanitised for posting.** No hostname hash, username, profile
  paths, or device instance IDs -- the first live run found the adapter device
  path carrying the same per-machine instance ID as the PnP ID, and both are
  cut to vendor, device and subsystem. GPU and driver names, laptop model,
  resolution and refresh stay, because they are what a hardware matrix needs.
- **The `benchmark` extra** installs everything the comparison needs:
  `rapidshot-native>=0.2.1` (for the test source), `psutil`, `mss` and `dxcam`.
- **The harness moved into the package**, as the private `rapidshot._bench`,
  because `benchmarks/` never reaches a wheel. Every moved module keeps a
  stand-in in `benchmarks/`, so `python benchmarks/section7.py ...` and the
  tests' `import section7` work unchanged; the stand-in *is* the moved module,
  so monkeypatching it patches what the harness uses. In a wheel the harness
  writes to `%LOCALAPPDATA%\rapidshot\benchmark` and records a content
  fingerprint instead of asking git, which in site-packages could report the
  commit of whatever project contains the environment.
- A `rapidshot` console script with `benchmark` and `diagnose` commands.

### Added — `rapidshot-native` 0.2.1

- **The wheel ships the benchmark's test source.** `latency_source.exe`, the
  controlled source that stamps a frame ID into every `Present()` so that every
  capture library is timed on one clock, was only available to someone who
  could build it with Cargo. It is now in the wheel, found through
  `rapidshot_native.latency_source_path()`, and the benchmark harnesses use it
  when no local build exists. The extension itself is unchanged from 0.2.0.
- **Recordings name the extension that ran.** Results now carry
  `native_loaded` — `rapidshot.native.build_info()`, including which route
  provided it — alongside the installed-package list, which reports the wheel
  even when an in-tree development build is what actually loads.

### Documentation

- **2.6.0's performance is now measured, and the README quotes it.** 2.6.0
  shipped claiming correctness but no hybrid throughput, because the only
  recording was 2.5-era. Re-recorded 2026-09-25 against 2.6.0 and the
  `rapidshot-native` 0.2.0 wheel on Machine B, in both MUX modes, three passes
  per path, every path verified first. On the hybrid laptop,
  `GpuConverter` + `TensorTransfer` reaches **162.2 fps to a CUDA tensor, pixels
  27.6 ms old, 2.0 ms CPU per frame, against DXcam's 95.6 fps, 37.7 ms and
  11.2 ms**; the full-frame cross-adapter path it replaces still runs at 80.9
  fps. Through YOLO11n with NMS, discrete-only: 50.5 fps and detections 46.9 ms
  after `Present()`, against DXcam's 37.7 fps and 53.8 ms. Five recordings in
  `benchmarks/*-machineB-*-2.6.0.json`; the README's Desktop to model section
  replaces the 2026-09-11 table, which stays in `docs/BENCHMARKS.md` as the
  2.5-era record.

## [2.6.0] - 2026-09-20

### Highlights

Four things, each with the measurement behind it:

- **~58% lower `grab()` memory** — 416.0 -> 174.8 MB at 2560x1600.
- **~63% lower `grab_frame()` memory** — 282.1 -> 104.4 MB, which is 1.01x
  DXcam's 103.1 MB. Both figures are the measured Machine B configuration
  (RTX 4060 laptop, Intel UHD capture, 2560x1600 at 165 Hz), recorded three
  times in `benchmarks/memory-baseline-machineB.json`, at unchanged throughput.
  Neither came from a redesign: a lazy CuPy import, a staging pool sized to what
  a converting `grab()` can use, and `pool_size_frames` 4 -> 2.
- **Fused one-dispatch GPU preprocessing with FP16** — crop, resize, colour
  conversion, normalisation and layout in a single D3D12 dispatch, emitting
  native `float16` as well as `float32` and `uint8`. Multi-ROI batches in one
  dispatch too. This is a fused *kernel*, not a chainable pipeline builder;
  the `.crop().resize()` graph in ROADMAP § 7.2 is still unbuilt.
- **One-call Torch / CuPy / DLPack interop** — `tensor.to_torch()`,
  `.to_cupy()`, `.to_dlpack()`, zero-copy, replacing roughly sixty lines of
  `ctypes` in `examples/gpu_tensor_to_cupy.py`. Verified byte-equal against a
  readback on hardware where capture and CUDA share an adapter.
- **Convert-before-transfer on hybrid GPUs** — preprocess on the capture
  adapter, then move the finished tensor instead of the whole frame: **2.46 MB
  rather than 16.38 MB** at 2560×1600. Verified on real Optimus hardware
  (Intel UHD Graphics → RTX 4060) on 2026-09-20, including **CUDA importing
  the transferred buffer** from the shared D3D12 heap on the discrete GPU.

**Correctness is claimed; a hybrid throughput figure is not.** The hybrid
performance recording is 2.5-era and will be re-recorded, so no latency or
frames-per-second number for this path appears in this release.

### Stage 7.2 — GPU transform and framework interop (2.6)

**Capture to model input, or to encoder input, without leaving the GPU — and
without silently handing back the previous frame.** 2.6 adds a general
transform (`GpuConverter`), a way to move its result to another adapter
(`TensorTransfer`), and an iterator over the whole path (`TensorStream`).
Building the last of those exposed a race that 2.3.0–2.5.0 users of
`GpuPreprocessor12` and `CrossAdapterTransfer` have been hitting: see
**Fixed**, which matters more than anything added here.

Everything below was verified against live capture on the Intel-only
development machine (Core Ultra 5 235, Intel iGPU, no discrete GPU). **Two
things could not be, and were release gates:** the CUDA exports, and any
transfer to a *hardware* second adapter — this machine's only second adapter is
WARP.

**Both were taken to an RTX 4060 laptop (Machine B).** The transfer gate is met
(2026-09-14): a converted tensor crosses from the Intel iGPU to the discrete
GPU, and the stale-read fix holds on a hardware destination. The CUDA gate found
a bug that had made the exports dead on every machine — see **Fixed**.

**The CUDA gate is now met too** (2026-09-20). It needed capture and CUDA on the
same adapter, and an earlier revision of this section said Optimus never gives
that — true of Optimus, but Machine B has a MUX. In discrete-only mode the RTX
4060 drives the display, so capture and CUDA share an adapter and
`test_to_cupy_is_byte_equal_to_the_readback` runs: the exported CuPy array is
byte-equal to a readback of the same tensor. `to_torch()` and `to_dlpack()` pass
alongside it, and `to_cupy()` works as a process's first CUDA call, which is the
bug **Fixed** records.

**The last gate closed on 2026-09-20.** `TensorTransfer`'s consumer handle is
minted on the destination device, and that change had faced only `cargo test
--lib` and a WARP destination — precisely the configuration that hid the bug it
fixes, since with WARP the consumer imported the *capture* GPU's own memory and
the path reported a crossing that never happened. Re-run in hybrid mode on real
hardware: Intel UHD Graphics capture, RTX 4060 destination, CUDA importing the
transferred tensor, verified against an independent reference at a maximum
deviation of 1 RGB8 level — the documented bilinear rounding tolerance. The four
cross-adapter tests that skip on a single-GPU machine all ran and passed.

#### Added

**`GpuConverter` — a D3D12 transform path, added alongside `GpuPreprocessor12`
rather than replacing it.** The old path samples with `Texture2D.Load()`, which
scaling 2560×1600 to 640² discards roughly fifteen of every sixteen pixels
rather than filtering them — aliasing exactly the small text and thin borders
desktop capture is usually pointed at. `GpuConverter` defaults to bilinear
through a static sampler; `sampling="nearest"` reproduces the old path **bit for
bit**, asserted against it on live capture. Kept separate on purpose: bilinear
changes the numbers `tests/test_gpu_preprocess.py` pins, and an A/B that cannot
say which sampling it timed is not a measurement.

- **Outputs.** `float32` and `float16` tensors in `layout="nchw"` or `"nhwc"`,
  and `uint8` as a resized BGRA frame — the cheapest thing to put on a
  cross-adapter bus. FP16 is half the bytes and what production inference
  mostly consumes. NHWC is the NCHW values in the other order and is asserted
  bit-identical to the transposed NCHW output. `normalize=` and `bgr=` as
  before; note `normalize` is not a division — the capture format is UNORM, so
  0..1 arrives free and 0..255 is what has to be reconstructed.
- **`pixel_format="nv12"` / `"p010"`** — 4:2:0 in the standard planar layout,
  what an encoder takes, returned as `(H*3/2, W)`. P010 carries the value in
  the high ten bits. `matrix="bt709"` (default) or `"bt601"`, limited range by
  default or `full_range=True`. Chroma is centre-sited (the 2×2 block mean);
  H.264/HEVC default to left-sited, a half-pixel shift that matters for video
  quality and not for ML. Verified against a CPU reference and, separately, by
  decoding with the *published* inverse coefficients so the kernel and the
  reference cannot share a wrong constant.
- **`crop=(left, top, right, bottom)`**, in frame coordinates — the convention
  of `Frame.region` and `dirty_rects` — at construction and per call. Applied
  before the resize. With bilinear sampling the filter is clamped to the crop's
  edge texels, so upscaling a small crop does not blend in the pixels beyond
  it. A crop outside the frame is refused rather than clamped. Refused on
  rotated displays, which are not handled.
- **Multi-ROI.** `process(frame, regions=[...])` converts every region in **one
  dispatch** into an `(N, …)` batch; `batch=` sets the capacity, and each call
  may pass fewer. Every slot is asserted bit-identical to converting that region
  alone. One batched call against N single-region calls, float16 bilinear:

  | Regions × size | batched | N × single |
  | --- | --- | --- |
  | 4 × 224² | 0.20 ms | 0.51 ms |
  | 16 × 224² | 0.43 ms | 2.11 ms |
  | 8 × 640² | 1.15–1.20 ms | 2.02–2.43 ms |

- **All four `DuplicateOutput1` formats accepted as input** — `B8G8R8A8`,
  `R8G8B8A8`, `R10G10B10A2` and `R16G16B16A16_FLOAT` — where the first draft
  refused everything but BGRA8. The view follows the surface format; declaring
  BGRA8 over a 10-bit surface reinterprets the bits rather than converting
  them. Float outputs pass scRGB values above 1.0 through; `uint8` saturates;
  NV12/P010 refuse the linear-light HDR format rather than put it through a
  gamma-domain matrix. `converter.source_format` reports which one arrived.
  **Only BGRA8 has actually been exercised** — this is an SDR desktop.

**`GpuTensor` — one-call framework export.** `to_torch()`, `to_cupy()` and
`to_dlpack()` replace the ~60 lines of `ctypes` in
`examples/gpu_tensor_to_cupy.py`, which stays as the worked version. The import
is cached, so a capture loop pays nothing per frame to keep the view. Deliberately
**not** on `Frame`: a frame is a captured surface, not a tensor. **Unverified —
no CUDA device on this machine.**

**`TensorTransfer` — convert first, then move the small result.** § 6.1
measured that converting before crossing adapters wins at 2560×1600 and at
1080p up to 640² FP16, but there was no way to do it: the converter writes a
buffer and the frame transfer takes a texture. At 640² FP16 this moves
2,457,600 bytes instead of the frame's 8,294,400. Asserted **byte-equal** across
the boundary rather than within a tolerance. The destination adapter is matched
by LUID, not index. **Exercised only against WARP.**

**`TensorStream` — the whole path as an iterator.**

```python
for tensor in rapidshot.TensorStream(camera, (640, 640), dtype="float16"):
    model(tensor.to_torch())
```

The value is in what it closes, each of which is quietly wrong when written by
hand: the frame goes back to DXGI before the tensor is yielded, so holding a
tensor cannot stall capture; `tensor.sync()` runs before the buffer is
overwritten; a permanently failed or released camera raises instead of looking
like an idle screen, since `grab_frame()` returns `None` for both; and a frame
from a new duplicator rebuilds the converter once, while any other failure is
raised rather than retried. Takes every converter option, including `regions`,
which may change between iterations. Its control flow is tested without a GPU;
its output is tested against an independent conversion of the same live frame.

**`cross_adapter_ordering_v3.py` removes v2's stated caveat, and finds the
ordering is resolution-dependent.** v2 charged every row FP32's conversion cost
because that was the only kernel that existed. The cheap representations are
genuinely cheaper to produce (640²: BGRA8 0.27 ms, FP16 0.31 ms, FP32 0.38 ms),
but at **1080p** convert-first wins only up to 640² FP16, where v2's 2560×1600
measurement had it winning everywhere. Both tables are right about their own
resolution. See ROADMAP § 6.1.

#### Changed

- **`rapidshot-native` 0.2.0 is now required, and 2.6 will not install against
  0.1.0.** `GpuConverter12` and `TensorTransfer` were both added to the Rust
  crate *after* the `native-v0.1.0` tag, so 0.1.0 exports neither — while
  `pyproject.toml` floored the `native` extra at `>=0.1.0`. The resolver was
  happy and the transform path then died on `AttributeError: module
  '_rapidshot_native' has no attribute 'GpuConverter12'`, which names neither
  the cause nor the cure. Three changes, because one was not enough:

  - the `native` extra floors at `rapidshot-native>=0.2.0`;
  - `rapidshot-native` joins the `all` extra, held back until now only because
    naming an unpublished distribution there would have broken
    `pip install rapidshot[all]` for everyone. It carries a
    `platform_system == 'Windows'` marker, without which `rapidshot[all]` would
    be uninstallable on Linux and macOS;
  - `native.require_feature()` gates every version-dependent symbol and reports
    *"GpuConverter12 requires rapidshot-native >= 0.2.0; the installed
    extension is version 0.1.0"* plus the upgrade command, for anyone who gets
    an old wheel past the floor by pinning it or by leaving a stale local build
    in the package directory — which `native.py` prefers over the wheel by
    design. A test asserts the table never gates a symbol the built extension
    lacks, so the backstop cannot itself invent an upgrade demand.

  **`native-v0.2.0` must be published before the RapidShot tag that floors
  against it**, or `pip install rapidshot[all]` resolves to nothing for as long
  as the gap lasts. `RELEASING.md` records that ordering.

- **`to_dlpack()` and `to_torch()` no longer emit a deprecation warning.** Both
  went through CuPy's `toDlpack()`, which is deprecated and raised a
  `VisibleDeprecationWarning` on every call — a library should not ship a
  warning it can avoid. `to_dlpack()` now uses `__dlpack__()` and returns the
  same PyCapsule as before; `to_torch()` hands Torch the array itself and lets
  it call `__dlpack__`, which also avoids a single-use capsule. Checked against
  CuPy 14.1.1 and Torch 2.11: `cupy.from_dlpack` and `torch.from_dlpack` accept
  both forms, so this is not an API change for callers.

- **CuPy `GRAY` is 6-9x faster, and byte-identical.** The Q8 luma was a chain
  of CuPy expressions: a kernel launch per line and a full-size uint16
  temporary for each, roughly six passes over an 8 MB frame to do arithmetic
  that needs one. It is now a single `ElementwiseKernel`. Measured on an RTX
  4060 against the form it replaces: **0.262 -> 0.042 ms** at 1080p, **0.629 ->
  0.068** at 1600p, **1.601 -> 0.190** at 4K.

  End to end the gain is small and worth stating plainly: `grab()` on that
  machine is capped by the 165 Hz panel, so GRAY capture went 162.8 -> 164.8
  fps, about 1%. What this buys is headroom -- for higher-rate sources, for
  more than one display, and for a GPU that is also running inference -- not
  frame rate.

  The portable array-expression form is kept as `_gray_chained` and is still
  what runs under any module without `ElementwiseKernel`. `CupyProcessor` is
  deliberately NumPy-substitutable so it can be tested without a GPU, and the
  fused kernel had broken that. The two are asserted byte-equal on a real
  device, at odd sizes as well as round ones.


- **`import rapidshot` no longer configures logging.** It used to attach a
  stdout handler and a DEBUG-level rotating file under `~/.rapidshot/logs` in
  every process that imported it -- per-frame debug messages included, 55 MB in
  two days on the development machine. The package now adds only a
  `NullHandler`, as a library should; warnings still reach stderr through
  Python's default handler. Call `rapidshot.util.logging.setup_logging()` to get
  the old console and file output back.

- **Removed unused DXGI factory wrappers** from `rapidshot._libs.dxgi`
  (`CreateDXGIFactory1`, `CreateDXGIFactory6`, `CreateLatestDXGIFactory`).
  Nothing called them, `dxgi.dll` does not export `CreateDXGIFactory6`, and
  binding them at import changed `restype` on `ctypes.windll`'s process-wide
  function object. `core/device.py` and `core/output.py` now load their DLLs on
  private handles for the same reason.

- **Device creation no longer falls back to another adapter.** If
  `D3D11CreateDevice` failed on an adapter, `Device` retried on the *default*
  adapter and then on WARP, REFERENCE and SOFTWARE, while still reporting the
  original adapter's description -- a device living on one adapter under
  another's name, which the factory then offered for duplication. Creation now
  stays on the adapter, varying only what that adapter may legitimately refuse
  (the debug layer, and feature level 11.1 on a runtime that predates it). An
  adapter that will not open is recorded in `RapidshotFactory.device_failures`
  and named in the `HeadlessError`, with its HRESULT -- now printed as
  `0x80004005` rather than `-0x7fffbffb`.

- **Removed code nothing could reach.** `PillowProcessor` (selectable only if
  importing NumPy failed, which the package cannot survive) with the PIL
  backend enum member and the Pillow version warning; the uncalled
  `NumpyProcessor.process_cvtcolor`; `Device.create()`; and
  `util.io.enum_dxgi_adapters_with_preference()`. **The `pil` extra is gone and
  `all` no longer installs Pillow** -- nothing in the library imports it.
  `pip install rapidshot[pil]` now warns that the extra does not exist rather
  than failing. `capabilities()` still reports a Pillow install, as an optional
  consumer of frames.

- **Memory: a RapidShot process is 58% smaller, and none of it was a
  redesign.** At 2560x1600 the RGB `grab()` path went from 416.0 MB to
  174.8 MB, and `grab_frame()` from 282.1 MB to 104.4 MB -- **1.01x DXcam's
  103.1 MB**, against the 1.25x that ROADMAP § 7.2 sets as the target. Three
  changes, each measured on its own with `benchmarks/memory_profile.py` and
  each recorded in `benchmarks/memory-baseline-machineB.json`:

  | | `grab()` | `grab_frame()` |
  | --- | ---: | ---: |
  | before | 416.0 MB | 282.1 MB |
  | CuPy imported on first use | 238.1 MB | 104.6 MB |
  | staging pool sized to what a converting grab can use | 199.3 MB | 104.5 MB |
  | `pool_size_frames` 4 -> 2 | **174.8 MB** | **104.4 MB** |

  Throughput is unchanged throughout: 165 fps for `grab_frame()` and ~100-139
  for `grab()` before and after, every difference inside the run-to-run spread.

- **CuPy is imported on first use rather than at `import rapidshot`.**
  `rapidshot/capture.py` imported it at module scope to set `CUPY_AVAILABLE`,
  so **every** caller paid **178.8 MB** resident for it -- including on
  machines with no NVIDIA GPU, and for callers who only ever touch `grab()`.
  `import rapidshot` now costs 20.3 MB rather than 184.8 MB. Nothing in that
  module needs CuPy unless `nvidia_gpu=True`, and `capture.CUPY_AVAILABLE` and
  `capture.cp` still read correctly for anything that imported them; they just
  pay for the import at that point. For scale, the native extension is 1.3 MB.

- **`pool_size_frames` defaults to 2, from 4.** Each buffer is a full frame, so
  the step is 12.3 MB at 2560x1600 and the scaling is exactly linear. Measured
  before changing it: 196.3 MB at 4 against 171.7 MB at 2, at 134.4 and 139.1
  fps -- a difference inside the run-to-run spread. Holding a rolling window of
  1, 3 and 6 frames did not separate them either, and **neither size could be
  made to exhaust**: 25 frames were held at both, because a converting mode
  falls back to allocating rather than refusing. Pass `pool_size_frames=4` to
  restore the old sizing; raise it only if you genuinely hold several frames at
  once. This is the second time the default has moved on a measurement, after
  10 -> 4 in 2.4.0.

#### Fixed

- **The only test checking `GpuConverter` against a CPU reference had never run,
  on any machine.** `test_nearest_output_matches_a_cpu_reference` fetched its
  source pixels from `live_frame.frame_buffer` and skipped when that was absent
  — and `Frame` has no `frame_buffer`. It is a GPU texture wrapper; the
  attribute has never existed on it. So the check skipped silently every time,
  everywhere, while reading as coverage in the suite.

  It now reads the frame's own `ID3D11Texture2D` back through a D3D11 staging
  surface, which is independent of everything under test: no D3D12, no compute
  shader, no cross-adapter machinery. On Machine B the converter's nearest
  output matches that reference **exactly** (max deviation 0.0), and a
  negative control confirms the assertion still fires — a channel-swapped
  reference is rejected at 1.0. Given how many silent wrong-pixel paths this
  area has produced (§ 10, and four more closed in this release), a correctness
  test that cannot fail was worse than none.

- **A corrupt surface pitch crashed the process rather than raising.** `pitch`
  sizes a ctypes array over the mapped surface, and only its lower bound was
  checked -- so a value larger than the mapping described a region the surface
  could not contain. This is not a clean failure: the test written for it took
  the interpreter down with a **Windows fatal exception: access violation** in
  `_read_rows` before the fix, which is what moved this off the "hardening
  against values that cannot occur" list.

  Both bounds now live in one place, `processor.base.check_surface_pitch`, used
  by `shot()` and by `process()` in both processors. Padding beyond 64 KiB is
  refused: drivers pad a row to an alignment boundary -- tens of bytes,
  occasionally a few hundred -- and a 4K BGRA row is 30,720 bytes, so the bound
  clears any real one by orders of magnitude. It completes a set: the other two
  driver-reported sizes, `MAX_METADATA_BUFFER_BYTES` and
  `MAX_POINTER_SHAPE_BUFFER_BYTES`, were already capped.

- **Cursor shape metadata was handed out without being checked against its
  buffer.** `Pitch`, `Width` and `Height` are the driver's, and `CursorInfo`
  reports them beside the shape bytes -- so the library was vouching for a
  description it had never verified. Nothing inside RapidShot indexes the
  buffer with them, which is exactly why a mismatch was invisible here and a
  crash in whoever walked the buffer by `shape_pitch`. A description that does
  not fit is no longer handed out at all: "no shape available" is something a
  consumer already handles, where a half-described buffer is what produces the
  over-read.

- **A missing CUDA driver was reported as a missing DLL.** `ctypes.WinDLL("nvcuda.dll")`
  on a machine with no NVIDIA driver raises `OSError: [WinError 126] The
  specified module could not be found`, which names a file rather than the
  situation. `rapidshot.converter` already separates the two answers a caller
  must tell apart -- `CrossAdapterRequired`, routine on a hybrid laptop, and an
  import failure, which is a bug -- and a driver that is simply not installed
  is a third case that was getting the least useful message of the three.


- **`shot()` on a rotated display trusted its own docstring about the
  destination size.** The direct path validates the caller's buffer before
  capturing; the rotated path then wrote `frame.nbytes` into it on the strength
  of a comment saying the two must match. They do while nothing else is wrong,
  and `describe_destination` was already returning the size -- which this threw
  away into `_`. The size is now re-checked, and a mismatch is refused the way
  `shot()` refuses one two checks earlier.

- **A rotated 1-pixel region handed back the pooled buffer itself.**
  `np.ascontiguousarray` returns its argument unchanged when the view is
  already contiguous, and every rotation of a 1x1 region is a no-op view -- so
  the caller received the pool's own memory while `is_still_pooled_buffer` said
  `False`, which is precisely the aliasing the pool exists to prevent. It now
  copies. `CupyProcessor` documents this exact case and has always used
  `.copy()`; the NumPy path had the bug its comment describes.


- **`process()` did not refuse a pitch narrower than a row; `shot()` always
  had.** The guard existed on the path almost nobody takes and was missing from
  the one every `grab()` goes through. A pitch smaller than a row makes the
  strided view span past the end of the mapped surface, so the last rows read
  whatever follows it -- with nothing in the result's shape, dtype or range to
  show it. Added to both the NumPy and CuPy processors, with the same message
  `shot()` uses.

- **`process()` read rows one at a time whenever a region was offset.** The
  vectorised branch was guarded by `pitch == row_bytes and start == 0`, and
  everything else fell to a Python loop. Padding is one way to miss that
  condition; the other is `start`, the byte offset of a region's left edge --
  so **every region camera took the loop, whatever the pitch**. One strided
  slice covers all of it: where `pitch == row_bytes` and `start == 0`,
  `start:end` *is* `:row_bytes`, so the special case was subsumed rather than
  removed. The CuPy path did the same thing and also allocated a fresh buffer
  per frame.

  Measured on the read in isolation, 2560x1600: full frame padded **1.662 ->
  0.797 ms**, region at origin **0.396 -> 0.170**, region offset **0.385 ->
  0.152**, region offset and padded **0.368 -> 0.167**. The contiguous
  full-frame fast path is unchanged at 0.59 ms.

  **It does not move end-to-end capture on this machine, and that is worth
  saying rather than burying.** Instrumented over 400 frames of a 1280x800
  region capture, `_read_patch` ran 399 times: with dirty rects available the
  frame is patched a band at a time, so the loop was running over a handful of
  rows rather than 800, and `grab()` is refresh-capped at 165 fps regardless.
  CPU per frame measured 1.17 / 1.14 / 1.04 ms across before-and-after runs --
  all noise. The change is kept because it is strictly less work, simpler than
  the branch it replaces, and faster on the full-read path that runs whenever
  dirty-rect metadata is unavailable; not on an end-to-end claim the numbers
  here do not support.


- **Unreadable move metadata patched the frame anyway, showing stale pixels.**
  The duplicator distinguishes two answers deliberately: `[]` means the frame
  carried no move rects, `None` means they could not be *read*. `_dirty_rects_for`
  collapsed both with a truthiness check, so an unreadable frame was patched by
  dirty rect alone -- and since DXGI does not repeat moved regions in the dirty
  rects, any region the compositor had moved kept showing the previous frame,
  with nothing anywhere to indicate it.

  Unknown now means "there may have been moves", so the whole frame is
  converted. That is already how unreadable *dirty* metadata was handled three
  lines below; the same uncertainty was being answered two opposite ways inside
  one function.

  **This reverses a previously tested decision.** The old behaviour was
  asserted by `test_unreadable_move_metadata_does_not_force_a_full_convert`, on
  the reasoning that the dirty rects are probably still usable. What settled it
  is the cost: an ordinary frame with no move metadata returns `[]`, so every
  `None` is a genuine metadata error and rare, and the full convert it now
  forces is correspondingly rare. Measured on Windows 11 at 2560x1600, across
  3,768 frames of window dragging and page scrolling, DWM reported zero move
  rects with the metadata readable on every frame.

  One test fake had to be corrected with it: `FakeDuplicator` left `move_rects`
  at `None` for its lifetime, where a real duplicator reassigns it on every
  acquire. It was modelling a display whose move metadata failed on every
  frame.


- **`shot()` copied a padded surface one row at a time.** When the driver pads
  a row -- common on non-power-of-two widths, and on modes this machine's
  display does not happen to use -- the BGRA path ran a `ctypes.memmove` per
  row inside a Python loop: 1600 calls a frame at 1600p against one for the
  contiguous case. It is now a single strided copy. Measured: **0.775 -> 0.313
  ms** at 1080p (padding now costs nothing at all, 0.98x the contiguous case,
  from 2.38x), **1.833 -> 1.036** at 1600p, **4.082 -> 3.235** at 4K. The
  converting modes already worked this way, which is why *they* showed no
  padding penalty while BGRA showed 2.4-2.7x.

- **`TensorStream` burned a core waiting on a still screen.** What paces its
  wait loop is the camera blocking inside `AcquireNextFrame` for `timeout_ms`.
  With `timeout_ms=0` -- documented and supported -- it does not block, and the
  loop measured **10,302,950 `grab_frame()` calls per second**, one core fully
  consumed re-asking a question whose answer had not changed. It now yields
  after a call that returned without blocking: **1,804 calls/s**, a 5,710x
  reduction, CPU per wall-second **1.0 -> 0.078**.

  The blocking default is untouched and pays nothing, because the yield is
  reached only when the call came back in under a millisecond. Windows rounds
  any non-zero sleep up to about half a millisecond, so a polling caller waits
  ~0.5 ms rather than 10 -- still 18x tighter than the default it opted out of.


- **`grab()` could return a frame that was almost entirely the previous one,
  after a `shot()` or `grab_frame()`.** Converting modes keep an accumulator and
  patch only each frame's dirty rects onto it. Those rects describe the change
  since the last frame the duplicator acquired, so a frame taken by `shot()` or
  `grab_frame()` in between -- whose rects were never applied -- left the next
  `grab()` 99.5% out of date in the reproduction, with no error. The duplicator
  now numbers every frame with new content, and `grab()` patches only onto an
  accumulator holding exactly the previous frame from the same duplicator;
  anything else converts in full. That covers both entry points, a duplicator
  rebuilt after an output change, and a grab that failed after acquiring.

- **`video_mode` could copy a frame it no longer owned into the queue.** When the
  screen is idle the capture thread duplicates the last frame, copying outside
  the capture lock. A consumer calling `get_latest_frame_buffer()` in that
  window takes ownership of the frame, and anything it drew on it could come
  back out of the queue as a captured frame. The copy is now checked, under the
  lock that publishes it, against the frame still being the producer's to copy,
  and dropped if not; on the CuPy path only after the device copy has actually
  run. Not a torn frame from recycling -- only the capture thread writes
  staging buffers, and it is the one copying.

- **`MemoryPool.release_all_buffers()` gave a held buffer to a second owner.** It
  marked every buffer available, including ones a caller still held, so the
  next checkout returned memory that caller was still reading. Held buffers are
  now detached and replaced with new allocations, atomically.

- **`shot()` raised for failures `grab()` reports as None.** Its contract is
  False on a failed capture, but only access loss was handled: protected
  content, other DXGI errors and a failed copy or map escaped as exceptions, and
  a camera that had given up or was waiting on recovery was used as though
  healthy. `shot()` now handles each case as `grab()` does -- recovery first,
  rebuild on access loss, False with the reason recorded otherwise -- and only
  an unusable destination still raises. The new `last_capture_error` property
  says why the last capture failed, for either call.

- **`max_buffer_len=0` failed capture for good.** It built a zero-length queue
  whose first eviction raised `IndexError` inside the capture thread. It is now
  rejected as a positive int, at construction and again at `start()`.

- **A CUDA driver failure was reported as the hybrid-laptop case.** `cuInit`'s
  return code was ignored, and a device the driver could not describe counted
  as "not on this adapter", so a broken driver raised `CrossAdapterRequired`
  and told the caller to transfer the frame. Both now raise `RuntimeError`
  naming the call and code; a device that fails does not hide a matching one
  after it. Every call through `nvcuda.dll` declares its argument types, and
  `close()` logs a failed `cuMemFree` or `cuDestroyExternalMemory` rather than
  dropping the code.

- **Driver-reported buffer sizes were allocated from unchecked.** A corrupt
  `TotalMetadataBufferSize`, `PointerShapeBufferSize` or `MORE_DATA` size would
  have been a multi-gigabyte allocation on the capture thread. Sizes over
  16 MiB are refused before allocating: the rects count as unknown, and the
  cursor keeps the shape it has.

- **The CuPy path allocated a host output pool it never used.** With
  `pool_output`, a pool of `pool_size_frames` full frames was built and a buffer
  checked out and returned every frame, though that backend allocates its own
  result. The pool is now used only by a backend that can write into it.

- **`shot()` on the GPU backend failed quietly, over and over.** The CuPy
  backend cannot write into caller memory, which `shot()` discovered only after
  acquiring a frame -- then returned False and scheduled a rebuild that could
  never help. It now raises `NotImplementedError` before capturing, pointing
  at `grab()`.

- **Releasing a pooled buffer twice raised.** A second `release()` raised
  `ValueError` from the pool, or `RuntimeError` if the pool had since been
  destroyed; it is now a no-op. A release through a stale reference after the
  buffer was checked out again is still not detectable, since the wrapper is
  the same object.

- **`repr()` of a GPU pooled buffer raised `AttributeError`.** It read the
  address through `.ctypes`, which CuPy arrays do not have.

- **`version_below()` treated `"4.5"` as older than `"4.5.0"`.** Missing
  components now count as zero, so a two-part version string no longer
  triggers the too-old warning.

- **A frame held across a capture rebuild broke.** A resolution change or
  device loss destroys the staging pool, which deleted the array out of every
  buffer -- including ones a caller still held from `grab()`. Reading such a
  frame raised `AttributeError` and releasing it raised `RuntimeError`. Held
  buffers are now detached: still readable, and released without error.

- **DXGI error messages printed HRESULTs signed**, as `-0x7785ffda` rather than
  the `0x887a0026` people search for.

- **`_create_dxgi_factory1()` ignored `CreateDXGIFactory1`'s result**, so a
  failure surfaced later as an unexplained "NULL COM pointer access".

- **Every `StageSurface` shared one `D3D11_TEXTURE2D_DESC`**: a ctypes struct
  used as a dataclass default is created once.

- **`pointer_to_address()` raised for a NULL typed pointer** instead of
  returning None, because `hasattr` does not swallow ctypes'
  `ValueError("NULL pointer access")`.

- **CuPy's install hint recommended `cupy-cuda10x`**, plus Linux and macOS
  variants of a Windows-only library. It now names this package's
  `gpu_cuda13` / `gpu_cuda12` / `gpu` extras.

- **`benchmarks/compare_libraries.py` reported per-call time as per-frame
  time.** The interval clock advanced on every `grab()`, hit or miss, so a
  polling library showed `ms_p50` 1.59 ms at ~100 fps while RapidShot's blocking
  default showed 9.96 ms at the same rate. Intervals now run frame to frame and
  records carry `"ms_basis": "frame_interval"`. **The `ms_p50`, `ms_p99` and
  `ms_jitter_stdev` columns in the committed `library-comparison*.json` predate
  this and are per-call times**; README quotes only frame rate, CPU and memory
  from those runs, which were correct.

- **The native extension's Rust formatting had drifted**, so CI's
  `cargo fmt --check` failed on the committed code before reaching anything
  else. CI also never ran `cargo test`: clippy and build do not compile
  `#[cfg(test)]` code, so the Rust unit tests had never executed. There is now a
  `cargo test --lib` step, and new tests pin the shader constant layouts, output
  byte sizes and DXGI format names.

- **`shot(0, buffer_size=n)` crashed the process.** `pointer_to_address(0)`
  returns 0 rather than None, and every null check was `is None`, so a null
  destination passed validation and reached `ctypes.memmove` into address 0 --
  an access violation, not an exception. `shot()` now refuses it before any
  capture work, and both copy paths below it check for 0 as well.
  `pointer_to_address()` also no longer lets `ctypes.ArgumentError` escape for
  objects it cannot read.

- **`grab()` failed on every display rotated 90 or 270 degrees.** Staging
  buffers were sized from the desktop region, but the staging surface holds the
  panel's orientation, and the processor refuses a buffer of the wrong shape.
  Until processing errors were made to raise (below) that refusal came back as
  a black frame; afterwards it would have been `None` and a recovery loop.
  Buffers are now sized by `ScreenCapture._staging_shape()`.

- **Rotated frames were turned the wrong way.** `region_to_memory_region`
  followed Microsoft's Desktop Duplication sample, where 90 degrees is
  clockwise; both processors rotated counter-clockwise with `rot90(k=1)`. So a
  full-screen grab on a 90/270 display came back upside down and a region grab
  read the wrong part of the screen. Both now turn clockwise. The tests that
  pinned the old direction compared against `np.rot90` itself; they now build
  the expected image pixel by pixel from Microsoft's mapping.

- **`shot()` ignored rotation.** It wrote the staging surface as-is, so on a
  rotated display it did not match `grab()`, which is what it documents: every
  pixel misplaced at 180 degrees, a transposed image at 90 and 270.

- **A processing error returned a black frame.** `NumpyProcessor.process()`
  zeroed the staging buffer and returned it flagged as a fresh array, so an RGB
  or GRAY caller received a 4-channel BGRA frame that aliased a buffer already
  back in the pool, and no recovery was scheduled. It now raises, as
  `CupyProcessor` already did, and invalidates a half-patched dirty-rect
  accumulator. `grab()`'s handler also returns the converted-output buffer it
  had checked out, which would otherwise have leaked once per failed frame.

- **A recovery forgot the region given to `start()`.** Both rebuild paths reset
  to the constructor's region, so continuous capture of a region came back from
  a device loss or resolution change capturing the whole screen; and a
  constructor region that no longer fit made `_on_output_change` raise
  `ValueError`. The requested region is now restored whenever it fits, with the
  full screen used -- and the request kept -- while it does not.

- **`dxcam_compat` reported `latest_frame_time` as 0 on every real camera.** It
  read `last_present_time` from the `ScreenCapture`, which never had one; its
  test passed because the fake camera did. `ScreenCapture.last_present_time`
  now exists.

- **`create()` accepted negative device and output indices.** `-1` selected the
  last one under a cache key different from its positive index, so one output
  could be given two cameras.

- **`output_info()` raised `TypeError`** for an output missing from the display
  metadata, such as one attached after the factory was built. `create()`
  already tolerated that case.

- **`grab()` held three BGRA staging buffers it could never reach.** A
  converting mode releases its staging buffer inside the same `grab()` -- the
  caller receives the *output* buffer -- and `_grab_locked` runs under the
  duplication lock, so exactly one staging buffer is ever in flight. The pool
  held `pool_size_frames` of them regardless: **4 x 16.4 MB at 2560x1600, three
  of them unreachable**, allocated at `create()` for every RGB camera.
  `ScreenCapture._staging_pool_size()` now sizes that pool to 1 when the frame
  the caller receives is not the staging buffer. **BGRA is unchanged and keeps
  the full count** -- it converts nothing, so the staging buffer *is* the
  returned frame, and in video mode the capture thread checks out more to fill
  `_pooled_frames_deque`. Throughput is unchanged, because serialised grabs
  never used the other three.

- **A rebuilt memory pool used a default two releases stale.** The rebuild path
  read `pool_size_frames` with a fallback of **10** -- the default before 2.4.0
  -- so any camera that did not pass the argument explicitly got a pool five
  times the size of the one it replaced when the frame shape changed.


- **`GpuPreprocessor12` and `CrossAdapterTransfer` sometimes returned the
  previous frame.** `AcquireNextFrame` returns once the copy into the
  duplication surface has been *submitted* on the capture device's D3D11 queue,
  not once it has run. A D3D12 queue reading that surface is a different queue
  with no implied order, and could overtake the copy. No error, correct shape,
  plausible content. Measured against a moving source, first read immediately
  after acquisition against a settled re-read of the same held frame:

  | Path | Stale, before | After |
  | --- | --- | --- |
  | `GpuPreprocessor12` | 7–15 / 150 | **0 / 150** |
  | `CrossAdapterTransfer` (to WARP) | 65 / 150 | **0 / 150** |
  | `GpuConverter` (never released with it) | 7–11 / 100 | **0 / 150** |

  Fixed by ordering, not waiting: a fence shared with the capture device is
  signalled into its D3D11 command stream behind the copy, and the reading
  queue waits on it on the GPU. `ID3D11DeviceContext::Flush` alone does not
  help (13–14 / 100 stale) because it submits without waiting. `transfer_async`
  stays non-blocking. Cost, same build with and without, 640², P-core-pinned:
  `GpuPreprocessor12.process` 0.44–0.48 → 0.40–0.41 ms, `GpuConverter.process`
  0.38–0.41 → 0.36–0.37 ms, `CrossAdapterTransfer.transfer` 0.50–0.51 →
  0.52–0.53 ms. One shared implementation, `native/src/capture_order.rs`.

  **Why the "byte-exact" verification never caught it.** `transfer_with_reference`
  compares the destination with a source-side copy taken from *the same
  snapshot*, so a stale read is stale on both sides and the comparison passes.
  The ~2,000 differing bytes that motivated that snapshot were this race,
  recorded at the time as a surface DXGI keeps writing to. The new regression
  tests compare a first read with a settled re-read instead, and each fails
  when its ordering call is removed. **The cross-adapter half was measured only
  against WARP**; a hardware destination is a release gate.

- **`GpuPreprocessor12` and `GpuPreprocessor` resized the entire monitor on a region camera.** A
  camera created with `region=` hands back frames whose texture is still the
  whole output, and the preprocessor sized its sampling from the texture — so
  its tensor was the whole monitor, correctly shaped. Verified before the fix:
  on a `(101, 51, 421, 291)` region camera the output was bit-identical to the
  whole 1920×1080 surface resized, and matched the region not at all. It now
  converts the region. **Full-frame output is unchanged bit for bit** — the
  shader's crop reduces to the old expression — and `tests/test_gpu_preprocess.py`
  passes unmodified. **This changes the output for anyone using it with a
  region camera**, which is the fix: that output was never the region. The raw
  extension (`native.require().GpuPreprocessor12`) works in texels and still
  converts the whole surface unless given `crop=`; the `native.GpuPreprocessor12`
  wrapper passes the region. On a rotated display the region is still not
  translated. `GpuConverter` had the same bug and was fixed before release.

  **The D3D11 `native.GpuPreprocessor` had the same bug, fixed the same way** —
  verified before the fix, bit-identical to the whole monitor on the same region
  camera. Its constant buffer grew from 32 to 48 bytes to carry the crop, which
  keeps it on the 16-byte boundary D3D11 requires. Whole-texture output is
  unchanged: the existing exact-reference tests pass unmodified, and new tests on
  synthetic textures pin the crop against a known pattern with no screen needed.
  On a region frame the D3D11 and D3D12 preprocessors now produce one tensor.

- **`to_cupy()`, `to_torch()` and `to_dlpack()` never worked, on any machine.**
  Never released — the bug and the feature are both new in 2.6 — but it is
  recorded here because of how it hid. Choosing the CUDA device is an equality
  test between the tensor's adapter LUID and the device's, and the device side
  was read from CuPy's `getDeviceProperties()["luid"]`. CuPy converts that
  fixed-size `char luid[8]` field as though it were a C string, so the value
  stops at the first zero byte — and a LUID almost always has several. The
  RTX 4060 here reports `3234010000000000` and CuPy returns three bytes, so
  **the comparison could never succeed on any adapter.** Every export raised
  `CrossAdapterRequired`, whose message advises transferring the frame to the
  CUDA adapter — impossible to act on when the tensor is already on it, as it
  is on any single-adapter NVIDIA desktop.

  Fixed by asking the driver instead: `cuDeviceGetLuid` from `nvcuda.dll`
  returns the whole 8-byte field. `examples/gpu_tensor_to_cupy.py` had always
  done this correctly, and `TensorTransfer.destination_luid` already documented
  it as the thing to pair with; the regression was in re-implementing the
  example rather than calling it. `tests/test_gpu_tensor_export.py` now pins
  the library's comparison against the example's, so the two cannot diverge
  unnoticed again, and two of its tests fail against the shipped version.


## [2.5.0] - 2026-09-13

**Everything that made RapidShot worth choosing, without the toolchain.** 2.4.0
made the hybrid path work end to end; this release makes it reachable. The GPU
tensor, cross-adapter transfer and AVX2 kernels -- every figure in ROADMAP
section 3 that justifies choosing this library -- sat behind a Rust toolchain
and the MSVC build tools, which most users will never install. Now:

- `pip install rapidshot[native]` -- one prebuilt `abi3` wheel for Python 3.9
  onward, no Rust.
- `import rapidshot.dxcam_compat as dxcam` -- an existing DXcam project
  switches by changing one import.
- `rapidshot.diagnose()` -- one command that answers most "it does not work"
  reports.

**Recovery you can observe.** Capture has long survived mode changes, monitors
coming and going and device resets, but never said so. `camera.generation`,
`recovery_count` and `last_recovery_reason` now do, and every frame is stamped
with the generation it came from, so a cached preprocessor knows when to
rebuild.

**The first measurement of what the GPU work is for.** Screen pixels to a
model-ready CUDA tensor, every library timed against one clock by decoding a
frame ID out of the captured pixels. On a hybrid 2560x1600 laptop, medians of
three passes: `grab()` returns 30% more unique frames than DXcam with pixels 8%
younger, and `nvidia_gpu=True` does it at half DXcam's CPU -- having lost the
capture-only frame-rate comparison to it. The cross-adapter path is a trade on
that hardware: 45% less CPU, but capped near 80 frames a second, fewer than
DXcam. Carried on through a trained YOLO11n on the RTX 4060, the lead narrows
but holds: every RapidShot path beat DXcam on every one of three passes, with up
to 20% more frames, pixels up to 9% younger when inference completes, and up to
a third less CPU per frame. One machine; ROADMAP section 7.0 carries the
caveats.

**A silent-corruption fix for anyone on NumPy 2.** `np.array(frame, copy=True)`
returned a view of a pooled buffer that the next capture overwrote.

### Fixed

- **The cursor position was reported in desktop coordinates.** DXGI reports the
  pointer against the whole duplicated output, and `Frame.cursor.position`
  passed that through unchanged — so on any off-origin region capture it named
  a point somewhere else entirely, wrong in exactly the case nobody checks by
  hand. It is now in frame coordinates, the same rule `dirty_rects` follows. A
  position outside the frame is kept and shifted, not dropped or clamped: a
  cursor whose hotspot sits just past the edge still draws pixels inside the
  region, and clamping would claim the pointer is somewhere it is not.
  `position is None` still means DXGI reported no position, which is a
  different answer from `visible=False`. Verified live against `GetCursorPos`:
  a full-output capture returns the desktop position unchanged, and a
  `(300, 200, 900, 700)` region turns a pointer at `(1328, 784)` into
  `(1028, 584)`.

- **Moved regions were invisible to the dirty-rect path.** DXGI reports regions
  the compositor *moved* -- a scroll, a window drag -- separately from the ones
  it redrew, and does not repeat them in the dirty rects. `GetFrameMoveRects`
  was declared but never called, so patching a frame by dirty rect alone could
  leave a moved region showing the previous contents, and `changed_fraction`
  reported a scroll as no change at all. Move rects are now read alongside the
  dirty rects, exposed as `frame.move_rects`, counted in `changed_fraction`,
  and a frame carrying any of them falls back to converting the whole frame
  rather than patching.

  **Measured first, as the review asked.** On Windows 11 at 2560x1600, across
  3,768 frames of window dragging and real page scrolling, DWM reported **zero**
  move rects, with the metadata readable on every frame -- a fully composited
  desktop leaves nothing for a screen-to-screen blit to optimise. So the
  full-convert fallback is correctness insurance for configurations where that
  is not true, not a path this hardware takes. The extra metadata read costs
  7.9 us, 0.13% of a 6.05 ms `grab()`.

- **Continuous BGRA capture stopped after `pool_size_frames` frames.** The
  frame queue was bounded by `max_buffer_len` (64) while backed by a pool of 4,
  and nothing returned a buffer until the queue reached 64 -- which it could
  never do. The queue is now bounded by what the pool can spare, leaving one
  buffer free for the next capture. Other colour modes are unaffected and keep
  the full `max_buffer_len`: their frames come from the output pool, which falls
  back to allocating. Verified live with nobody reading at all: 173 frames in
  three seconds at `pool_size_frames=4` where it previously produced 4 and
  froze, and 116 at `pool_size_frames=2`.

- **`grab()` on BGRA has no allocating fallback**, which the README claimed it
  did. With a conversion the output pool running dry falls back to allocating;
  BGRA hands back the staging buffer itself, so once every buffer is out
  `grab()` returns None until one is released. Measured: 30 unreleased BGRA
  grabs at `pool_size_frames=2` gave 2 frames and then None, where RGB kept
  going. Documented rather than changed -- the fallback would have to hand back
  an unpooled array, which is the aliasing hazard pooling exists to prevent.

- **A failing timer wait looked like a normal tick.** `util/timer.py` declared
  none of its Windows functions, so ctypes read `WaitForSingleObject`'s DWORD
  return as a signed 32-bit int: `WAIT_FAILED` is `0xFFFFFFFF`, which comes
  back as `-1`, so the capture thread's `res == WAIT_FAILED` check could never
  be true. `CreateWaitableTimerExW` returns a pointer-sized HANDLE that the
  same default would truncate. Both are now declared, on a privately opened
  `kernel32` rather than `ctypes.windll`, whose per-DLL cache is shared with
  every other library in the process -- `util/io.py` was setting `argtypes` on
  that shared object on every call, and now declares its functions once on its
  own handles. The dead `get_monitor_name_by_handle` is gone.

- **`IDXGIOutputDuplication::GetDesc` was declared without its parameter**, so
  calling it would have let DXGI write a 36-byte `DXGI_OUTDUPL_DESC` through
  whatever the argument register happened to hold. The vtable slot was the
  right size, so nothing else was affected, and nothing calls it yet -- which
  is the only reason this never fired. `DXGI_OUTDUPL_DESC`, `DXGI_MODE_DESC`
  and `DXGI_RATIONAL` are now declared and the call works: verified live,
  returning 2560x1600 at 165 Hz for this display.

- **`grab(region=...)` allocated a staging buffer per frame** whenever the
  region's shape did not match the pool's. The allocation is cheap but filling
  it is not: every page faults on first touch, measured at 0.15 ms for a
  400x400 region and 1.9 ms at 2560x1600 -- 77-94% of the cost of writing the
  buffer at all, the same effect `pool_output` exists to avoid. The buffer is
  now reused across grabs, but **only for converting colour modes**, where it
  is a pure intermediate. BGRA does no conversion, so that buffer *is* the
  frame handed back, and reusing it would give successive callers the same
  memory; BGRA keeps allocating.

- **Library warnings went to stdout.** `processor/base.py` and
  `memory_pool.py` used `print()` for stale-dependency warnings, backend
  fallbacks and pool errors, which no application can silence or route. They
  use the module logger now, and a test parses the package with `ast` to keep
  `print()` out of it.

- **`Output.__post_init__` claimed process-wide DPI awareness silently**, on
  every `Output` constructed, discarding the result. RapidShot does need it --
  a DPI-unaware process is fed virtualised desktop coordinates, which would
  disagree with the size of the texture Desktop Duplication hands back -- but
  it is a process-wide setting that Windows allows to be set once, and
  claiming it is the host application's decision. It is now attempted once per
  process, and says what happened: a host that has already set a different
  awareness gets a warning naming the consequence instead of silently wrong
  coordinates on a scaled display. A Python process is already per-monitor
  aware from `python.exe`'s manifest, so this has always been a no-op there;
  it matters for an embedded interpreter.

- **`examples/verify_cross_adapter.py` described a check it no longer makes.**
  Its docstring said the comparison was against the CPU capture path, and that
  it took two CPU frames per attempt and compared only when they matched. It
  actually compares the destination readback against a source-side readback
  from the same command list, and uses a CPU capture only for a mean-colour
  sanity check it prints rather than asserts. The docstring now says that.

- **Every COM pointer was released twice.** `Device.release()`,
  `StageSurface.release()` and `Duplicator.release()` each called `.Release()`
  on a comtypes pointer and then dropped it -- but comtypes issues `Release`
  itself when the Python object goes away, so one reference produced two
  decrements. Measured on this machine against a real D3D11 device: an explicit
  `Release()` followed by dropping the pointer took the refcount down by two,
  dropping it alone by one, and `Device.release()` as a whole moved it by three
  where two is correct. Over-releasing frees the object while other holders
  still have valid pointers, so what happens next depends on who touches it
  first -- which is why this survived. The duplicator had the same fault on the
  intermediate `IDXGIResource` in `update_frame()`, where it corrupted the
  desktop surface's refcount outright; that one was fixed in 2.4.0, this is the
  rest of them. Dropping the reference is now the release. Verified live: 25
  create/grab/release cycles left the process handle count flat.

- **The factory could be built more than once.** `get_factory()` and the
  `Singleton` metaclass both tested for an existing instance and then created
  one, with nothing in between. Constructing `RapidshotFactory` enumerates DXGI
  adapters and opens a D3D11 device per adapter, so the losing thread's devices
  stayed open: with eight threads calling `create()` at once, eight factories
  were built and seven discarded. Both are now double-checked under a lock, and
  `reset()` holds it across the teardown so a concurrent `get_factory()` cannot
  be handed the factory being reset.

- **`Frame.release()` is now serialised.** Two threads could both pass the
  released check, run every drain twice and hand the same DXGI frame back
  twice. This is a real race in the language, but not one that could be
  provoked here: 24,000 concurrent releases with the GIL switch interval at a
  nanosecond produced no double release, because CPython's swap-and-take made
  the hand-off effectively atomic. The lock makes it correct by construction
  rather than by accident of the GIL -- which is what a free-threaded build
  removes. It costs 0.68 us per construct-and-release, against the 7.6 ms CPU
  path this type exists to avoid.

- **`stop()` tidied up after a capture thread that was still running.** It
  waited ten seconds for the thread and then carried on regardless: closing the
  timer handle that thread closes again on its way out -- a second
  `CloseHandle` on a handle Windows may already have reissued to something
  unrelated -- and setting the frame queue to `None` underneath a thread still
  appending to it, which faults it into the catch-all that marks the camera
  permanently failed. The same double close happened whenever `stop()` was
  called *from* the capture thread, where the join is skipped entirely.
  `stop()` now leaves both to the thread in those cases and returns `False`
  rather than `None`, `start()` refuses to launch a second thread alongside a
  surviving one, and `release()` says out loud that it is tearing down
  resources still in use. Verified live: a capture thread wedged behind the
  duplication lock left the timer handle and queue untouched, refused a
  restart, and exited cleanly once unblocked.

- **Queued frames survived a mid-capture rebuild.** `_initialize_resources()`
  cleared the continuous-mode queue only when `continuous_mode` was True -- a
  flag nothing has ever set, so the block never ran -- and did it after the
  pool those buffers belong to had already been destroyed, so the check-ins
  would have been refused and the buffers dropped rather than recycled. Frames
  captured before a display change therefore stayed queued, and
  `get_latest_frame()` handed them out as current. The drain now happens before
  the pool is destroyed, is shared with `stop()` and `_rebuild_frame_buffer()`,
  and no longer consults the dead flag, which is removed. Verified live on a
  real rebuild: every check-in preceded the pool's destruction.

- **`get_latest_frame()` handed back memory the pool could recycle.** It
  returned the newest queued buffer's array without taking the frame out of the
  queue, so the producer stayed free to evict that entry and release it, and
  the next capture then wrote over the frame the caller was still reading --
  silently, with nothing raising. `stop()` did the same to every queued buffer
  at once. The frame is now copied out of the queue and is the caller's to
  keep. `get_latest_frame_buffer()` is the zero-copy alternative: it hands over
  the buffer itself, along with the duty to `release()` it, and the DXcam
  shim's `get_latest_frame_view()` now uses it -- which also makes that view's
  retirement real, where before it held a plain array whose absent `release()`
  made `_retire_view()` a no-op.

  This also unblocks continuous BGRA capture, which stalled after
  `pool_size_frames` frames: the queue is bounded by `max_buffer_len` (64) but
  backed by a pool of 4, and nothing returned a buffer until the queue reached
  64, which it could never do. Verified live -- twelve reads at
  `pool_size_frames=2` gave two distinct frames before and twelve after, and no
  frame changed while capture kept running or after `stop()`. A consumer that
  stops reading can still fill the queue and stall the producer; the queue
  bound itself is not fixed.

- **A rotated display corrupted frames on the CuPy path.** `cp.rot90` returns a
  view, and `process()` passed it on two different wrong ways. At 180° the
  shape is unchanged, so the pooled buffer was assigned from a view of itself
  -- an overlapping device copy, which CuPy's elementwise kernel tears, unlike
  NumPy's overlap-safe assignment. At 90° and 270° the view was returned with
  the pooled flag cleared, so `_grab()` checked the buffer back in while the
  caller still pointed into it. Rotation now copies, as the NumPy path does.
  `.copy()` rather than `ascontiguousarray`, which hands a contiguous view
  straight back -- true of a 1x1 region, where both flips are no-ops. Covered
  by tests that run NumPy in CuPy's place, so they need no GPU; not yet
  verified on a physically rotated display or a real device.

- **`native.probe_onnxruntime()` no longer loads a DLL by bare name.** With no
  argument it passed `"onnxruntime.dll"` to `LoadLibraryW`, which walks the DLL
  search path. On Windows 11 that found the OS's own System32 copy (1.17), not
  the installed package's, and printed a line per unsupported API version; on
  machines without that copy the search reaches the working directory and
  `PATH`, where a planted DLL would run. `capabilities()` and `diagnose()` call
  it, so any diagnostic did that search. It now probes the DLL shipped with the
  installed `onnxruntime` package, loads nothing if there is none, and only
  loads an explicit path that names an existing file, resolved to an absolute
  one. Verified live: the report now shows 1.30.0 from the package.

- **`shot(region=...)` changed the camera's region for good.** It validated the
  region with the method that also assigns it, so a one-off shot of a corner
  became the region every later `grab()` and `start()` captured. `shot()` now
  validates without adopting it, as `grab()` already did.

- **`start(delay=...)` documented milliseconds and slept seconds.**
  `time.sleep(delay)` takes seconds, so `delay=500` meant as half a second
  waited over eight minutes. The code is unchanged -- seconds is what it always
  did, what DXcam means, and what the DXcam shim passes through -- and the
  docstring now says so. A negative or non-numeric delay raises `ValueError`.

- **Single-shot capture raced the capture thread.** `grab()` was meant to
  redirect to `get_latest_frame()` while `start()` ran, but the flag it tested,
  `continuous_mode`, was never set anywhere -- so it called `AcquireNextFrame`
  on the same duplication as the capture thread. `shot()` and `grab_frame()`
  had no check at all, and nothing serialised duplicator use between threads:
  `_capture_lock` guarded only the frame deque. All three now raise
  `RuntimeError` while continuous capture runs (a redirect would hand code
  written for `grab()` a plain array with no `release()`), and every use of the
  duplicator, staging surface and context -- grab, shot, `grab_frame`,
  rebuild, `release()` -- goes through one re-entrant lock. Verified live: two
  threads grabbing at once captured ~280 frames with no errors and no rebuilds.

- **Rotated displays mapped the capture region outside the texture.**
  `region_to_memory_region` reversed the 90° and 270° axes by the wrong
  dimension of the native-orientation texture. On any non-square panel a
  full-screen region became a box with a negative or out-of-range edge -- a
  1080x1920 portrait desktop at 270° gave a left edge of -840 -- and
  `CopySubresourceRegion` silently skipped the copy, so capture returned a
  stale frame. The existing test used only interior regions, where the error
  stays in bounds, and took its expected values from the same formula. New
  tests check that a full-screen region maps to the whole texture, that every
  valid region stays inside it, and that every texel round-trips through the
  texture-to-desktop mapping Microsoft's Desktop Duplication sample uses. Not
  yet verified on a physically rotated display.

- **`create()` returned a camera built with different settings.** Cameras were
  cached by device, output and `prefer_integrated` alone, so
  `create(output_color="BGRA")` after an RGB camera on the same output got the
  RGB camera back, as did a different region, `nvidia_gpu`, pool or timeout
  setting. A repeat request with the same settings still shares the camera; a
  different one now raises `ConfigurationError` naming what differs.

- **`rapidshot.create()` could return a camera that had already been released**,
  which never captures again: every `grab()` returned `None`, with no error. The
  factory caches cameras weakly, so a released one stayed cached for as long as
  anything referenced it -- and in `camera.release(); camera = rapidshot.create()`
  the old object is still bound when `create()` runs. `clean_up()` released
  cameras without evicting them either. Any second camera in a process was
  affected, including one made through the DXcam shim. Present in 2.4.0 and
  earlier; found by a live smoke test of the shim. `ScreenCapture.released` now
  says whether `release()` has run, and `create()` builds a new camera in place
  of a released one.

- **Pillow 12 triggered a warning that Pillow was too old.** The processor's
  dependency checks compared version strings, and `"12.3.0" < "9.0.0"` is True
  as text. NumPy, Pillow, OpenCV and CuPy versions are now compared
  numerically; OpenCV 10 would have hit the same false alarm.

- **`np.array(frame, copy=True)` returned a view of a pooled buffer.**
  `PooledBuffer.__array__` accepted NumPy's `copy` argument and ignored it.
  NumPy 2 forwards `copy` and trusts the answer, so an explicit copy request
  handed back a view of a buffer the pool was about to reuse: the caller held
  what looked like its own array, the next capture overwrote it, and nothing
  raised -- exactly the failure pooling is documented to prevent. Verified
  aliasing on numpy 2.5.1.

  It predates everything else in this release and affects anyone who has
  written `np.array(frame, copy=True)` against a pooled buffer since NumPy 2.
  `frame.copy()` and `np.asarray(frame).copy()` were always correct. Found by
  the DXcam compatibility layer's tests, whose whole safety story is copying a
  frame out before releasing it. Regression tests cover the copy, the
  zero-copy default and dtype conversion.

- **The release performance gate had stopped gating anything.**
  `RELEASING.md` step 4 named `benchmarks/baseline.json`, which is Machine A.
  Run on Machine B the suite compared correctly, detected the hardware
  mismatch, and declined to gate -- so the release step printed a full table in
  which every verdict was indicative and no result could fail. The suite was
  right each time; the instructions pointed it at the wrong file. Nothing was
  broken and nothing was verified, which is the failure mode worth naming: a
  gate that always passes still gets quoted as evidence that it passed.

  `--compare auto` now selects the committed baseline recorded on the current
  host and **exits non-zero if there is none**, rather than falling back to
  another machine's file. `RELEASING.md` and the pull request template point at
  it. Matching an explicit path still works and still reports indicative
  verdicts, which is the right behaviour for a deliberate cross-machine look.

### Added

- **`rapidshot-native`, the prebuilt extension.** `pip install
  rapidshot[native]` and the GPU tensor, cross-adapter transfer and AVX2
  kernels are present, with no Rust toolchain and no MSVC build tools.

  This was the largest gap between what the library can do and what a `pip
  install` could reach: every headline measurement sat behind a build step most
  users will not perform, so the features they justify went unused.

  **One wheel, not a matrix.** The crate already carried `pyo3/abi3-py39`, so a
  single `cp39-abi3-win_amd64` wheel serves Python 3.9 and every later version
  -- including CPython releases that do not exist yet. A per-version matrix
  would have produced six near-identical wheels and a maintenance burden that
  grows with each Python release; CI fails the build if the abi3 tag is ever
  lost, because a version-locked wheel installs happily and breaks silently on
  the next interpreter.

  **A separate distribution on a separate tag** (`native-v*`, published by
  `release-native.yml`). `rapidshot` stays `py3-none-any`, so the guard that
  fails the main release if a compiled artefact appears inside it is preserved
  rather than argued with -- that guard exists because a `*.pyd` glob once
  swept a locally built extension into a platform-neutral wheel. The two
  version independently: the Rust changes on its own schedule, and lockstep
  would republish an identical binary under a new number every release.

  **A local build takes precedence over an installed wheel.** Both present is
  the normal state for a contributor who installed the wheel first and later
  built from source; if the wheel won, every `cargo build` would appear to do
  nothing. `native.build_info()` reports which is loaded under `source`.

  `BUILD_HINT` now leads with the wheel. It previously offered only "install
  Rust and the MSVC build tools", which is part of why the extension went
  unused -- the easy route was not mentioned because it did not exist.

  Verified by installing both wheels into a clean environment **outside the
  repository**: `rapidshot.native.is_available()` returns True, the AVX2
  swizzle is byte-exact against NumPy, and `probe_cross_adapter()` reports
  `representative: true` from the Intel iGPU to the RTX 4060.

  **Published 2026-09-13 as `native-v0.1.0`**, after this release was drafted:
  PyPI serves `rapidshot_native-0.1.0-cp39-abi3-win_amd64.whl`, verified by
  installing it from PyPI into a clean environment. It stays out of the `all`
  extra for now -- adding it only takes effect on a `rapidshot` release, so it
  rides with the next one. Note the wheel needs **this** release: discovery
  lives in `rapidshot.native`, which 2.4.0 predates, so the wheel reports
  `is_available() == False` against the previously published RapidShot.

- **`rapidshot.dxcam_compat`: DXcam code runs by changing one import.**
  `import rapidshot.dxcam_compat as dxcam` provides `create()`,
  `device_info()`, `output_info()`, `reset()`, `clean_up()`, and a camera with
  DXcam's methods and the attributes DXcam code reads.

  `grab()` returns a plain `ndarray`, which costs one copy per frame, on
  purpose: DXcam callers never release a frame, and a pooled buffer must be
  released, so the shim copies out and releases at once. `grab_view()` and
  `get_latest_frame_view()` are genuinely zero-copy, because DXcam's "valid
  until the next grab" contract is exactly a pooled buffer's lifetime. As with
  DXcam, a view kept past the next grab silently shows whatever frame its
  buffer holds next; measured against live capture, not assumed. `camera.rapidshot_camera`
  exposes the camera underneath, so a project can migrate one call site at a
  time.

- **`rapidshot.capabilities()` and `rapidshot.diagnose()`.** One report --
  version, where the native extension was loaded from, adapter topology and
  whether it is hybrid, what the extension can do here, optional dependency
  versions -- in place of six scattered probes a user had to run and paste in
  the right order. `capabilities()` returns a dict; `diagnose()` renders it for
  an issue report.

  Neither raises. Each section is independent and reports its own failure,
  because the machine where this matters most is the one where something is
  already broken. The cross-adapter probe is opt-in (`probe_gpu=True`): it
  creates D3D devices and allocates a shared heap, and a diagnostic should not
  be able to destabilise the machine it is diagnosing.

- **Recovery is observable, not only survivable.** The rebuild machinery --
  bounded retries, backoff -- already existed, but nothing told a caller it had
  run. A consumer holding a `GpuPreprocessor12` or `CrossAdapterTransfer` built
  from an earlier frame had no signal that the duplicator underneath had been
  replaced and might now differ in size, rotation or format.

  `camera.generation` and `camera.recovery_count` count successful rebuilds,
  `camera.last_recovery_reason` says why the latest one was scheduled, and all
  seven rebuild triggers record a cause. The generation moves only on success,
  so a failed attempt that will be retried does not invalidate anyone's cache
  for nothing.

- **`Frame` gains `sequence`, `generation`, `changed_fraction`, `age_ms` and
  `cursor`.** `generation` pins each frame to one side of a rebuild, and
  `sequence` identifies it for the camera's whole life, continuing across
  recoveries.

  `changed_fraction` unions the dirty rects rather than summing them. Drivers
  do report overlapping regions, and a summed area can exceed the frame --
  sending a consumer that thresholds on "more than 90% changed" down the
  full-frame path for a frame that barely moved. An empty list reads as `1.0`,
  because no rects is no information.

  `age_ms` is measured from the present timestamp, so it is how old the pixels
  are rather than how long a call took. `cursor` is a `CursorInfo` snapshot --
  position, hotspot, raw shape and its encoding -- where only visibility was
  surfaced before, and anything more meant the raw COM structures from
  `grab_cursor()`.

- **`rapidshot.profiling.Profiler`.** Stage timings and frame health from a
  capture loop, as `report()` for a human, `summary()` for asserting on, and
  `json()` for storing beside a baseline. It reports percentiles and a minimum,
  never a mean -- background load can only make a sample slower, so the
  minimum is the least contaminated estimate and the tail is what a real-time
  consumer feels. Any stage under 30 samples is labelled `low_confidence`. It
  also records `coalesced_updates_missed` and mid-run recoveries, because a
  loop can look fast while dropping most of what it was meant to capture, and
  wall clock alone cannot tell the two apart.

- **`CrossAdapterTransfer` exposes what the extension already exported:**
  `transfer_async_with_reference()` with `read_back_source()`,
  `probe_transfer_phases()`, and `submission_quarantined`. The Python wrapper
  listed its methods explicitly with no passthrough, so an async transfer's
  pixels could not be verified from Python at all.
  `transfer_async_with_reference()` defers the frame's release until its fence
  completes, as `transfer_async()` does.

- **The version number is declared once**, in `rapidshot/_version.py`.
  `pyproject.toml` reads it through `[tool.setuptools.dynamic]`, `setup.py`
  parses it out of the AST, and `rapidshot.__version__` re-exports it. It was
  previously written out in all three; the release workflow checked each
  against the git tag but never against each other, so two could agree while
  the third drifted, and nothing failed until a release was being cut.

  `_version.py` is deliberately import-free, so the build backend can read the
  literal without importing `rapidshot` -- which needs Windows COM, and would
  turn a version read into a release-day failure with no local reproduction.
  `tests/test_version.py` fails if a second declaration appears anywhere in the
  package, if `pyproject.toml` or `setup.py` goes back to repeating it, or if
  `_version.py` grows an import.

- **`native_extension` in the recorded machine block.** `baseline.json` is
  recorded with the optional native extension and `baseline-nonative.json`
  without it, and comparing across that line reports every conversion row
  6-20x slower on every run -- the hazard `ci.yml` already routes around by
  hand. Nothing in a recording said which side it came from, so the two were
  distinguishable only by filename, and a chooser cannot read a naming
  convention. Absent on older recordings, which is treated as unknown: it
  cannot disqualify a baseline on its own, and it cannot break a tie either.

  Ambiguity fails loudly rather than picking one. Two equally valid baselines
  resolved by directory order would make a verdict depend on `glob()`, which is
  the kind of invisible coupling this suite exists to remove.

### Documentation

- **The 2.5 APIs are in the README.** None of `capabilities()` /
  `diagnose()`, the DXcam shim, the profiler, observable recovery or the new
  `Frame` fields appeared there. Each now has a section or a row, with the
  behaviour that matters when using it: the shim's one copy per frame and why
  `grab_view()` avoids it, the generation check that decides when to rebuild a
  cached preprocessor, and why the profiler will not report a mean.

- **The README compares libraries past `grab()` for the first time.** A new
  *Desktop to model* section carries the pixel-age table, and the introduction
  no longer says only that RapidShot loses the frame-rate column -- true of
  capture alone, and not of the workload the GPU paths exist for. It keeps the
  caveats that bound it: one hybrid machine, submission age rather than photon
  age, and nothing yet past the tensor.

- **ROADMAP section 7.0 agreed with itself only in places.** Its open-items
  list still named the DXcam WGC backend, the GPU-side semaphore path, the
  per-stage breakdown and the QPC frequency as missing, when the pixel-age
  recording contains all four; it called the agent benchmark unbuilt when
  `section7.py --category agent` exists; and it gave the inference table's
  stand-in model as the harness default after the fallback had been removed.
  The list now matches the recordings, the call-duration caveats sit with the
  tables they describe, and the source rate is given as a recorded median
  rather than a single figure.

- **The README's CUDA loop could never run its own `None` check.** It sat
  inside `with camera.grab_frame() as frame:`, but `grab_frame()` returns
  `None` when nothing changed and `with None` raises `TypeError`, so the loop
  crashed on the first still frame. The examples now check before the `with`,
  and the GPU-resident section says why.

- **ROADMAP sections 7.0 and 7.1 record what was measured and delivered**,
  including one correction made in the open: an earlier revision claimed no
  `cuImportExternalSemaphore` glue existed and made building it the top
  follow-up, when `tests/test_cross_adapter.py` already had working producer
  and consumer implementations and section 6.1 had verified the mechanism a
  fortnight earlier.

- **`ROADMAP.md` section 7 rewritten as a staged 2.5 → 3.0 plan.** It was a flat
  bullet list of "later stages" with no ordering rationale, which is how a
  roadmap turns into a wish list.

  The plan now leads with **section 7.0: build the AI-ingestion benchmark before
  any of the features below it.** The only cross-library benchmark this project
  has measures `grab()` — OS pixels to a CPU array — which is the path where
  RapidShot's advantage is smallest and where it loses the frame-rate column
  outright. Nothing measures *present → model-ready CUDA tensor*, which is what
  the GPU work exists for. Until that number exists, ordering `GpuConverter`,
  Torch interop, FP16, multi-ROI and DLPack against each other is guesswork.

  Also recorded: measure pixel *age* rather than call duration, via a controlled
  visual latency source that encodes a frame ID into the pixels so every library
  can be timed against one clock — `Frame.timestamp_qpc` gives RapidShot an
  advantage the others cannot match, which makes it useless for a fair
  comparison without one.

  Several proposals were corrected rather than adopted as written: the hardware
  encoder cannot branch off `GpuPreprocessor12` (it emits an NCHW float32 ML
  tensor, not an encoder input); ROI scheduling does not reduce DXGI capture
  work, only downstream processing; dirty rects carry no window identity, so
  they cannot track windows; multi-monitor "synchronisation" is alignment within
  a tolerance, since monitors present independently; a Windows `HANDLE` cannot
  be sent over a socket; and QPC is a sub-microsecond interval timer, not a
  nanosecond-accurate synchronised clock.

  The native C ABI is recorded as a **reversal with its reason stated**: section
  8 rejected a native core on the measurement that the Python/COM binding costs
  0.003 ms/frame. That measurement stands. The new argument is embedding, not
  speed — a C/C++ host such as OBS cannot put Python in its capture path at any
  speed. Section 8 gained explicit rejections for a YAML/DSL pipeline layer, A/V
  production features, Game Capture-style API injection, and adaptive
  backpressure.

- **`README.md` rewritten.** It carried numbers that were never in the files it
  cited, and claims nothing measured supported.

  The worst was the compatibility table: it listed **hybrid / switchable
  graphics as "Not tested"**, disclaiming the configuration 2.4.0 exists to
  serve, six days after `verify_cross_adapter.py` moved five frames Intel to
  RTX 4060 byte-exact on exactly that hardware.

  The "Second machine, RTX 4060" table attributed four of its six figures to
  `baseline-rtx4060.json`, which never contained them at any point in its
  history -- BGRA to RGB quoted 0.198 ms against a recorded 0.235, and the GPU
  dispatch row quoted 0.070 ms for a benchmark that measures *submission* cost
  and recorded 0.001. Elsewhere: "6 recordings, 0.17-0.77 ms" in one paragraph
  and "seven recordings, 0.17-0.83 ms" in another; "9.4 ms" for a no-extension
  GRAY conversion the file records as 6.878; conversion described as "pure
  NumPy" two sections after the AVX2 kernels it actually uses; a cursor example
  that could not run, referencing undefined constants and returning a variable
  it never assigns; and benchmarking instructions still teaching
  `--compare baseline.json`, the pattern that stopped the release gate gating.

  Cut: "built for feeding models, not for saving screenshots", "comprehensive
  cursor capture capabilities", "designed to provide objective performance
  measurements", "colour conversion is essentially free" (it is 0.30 ms), and
  the inherited fork boilerplate that advertised "NVIDIA GPU acceleration" two
  bullets below the entry explaining what that actually is.

  Every figure now names the committed file it came from.


- **`ROADMAP.md` still said 2.3.0 was the tagged release** and listed the
  asynchronous shared fence as the piece of section 6.1 still outstanding,
  which 2.4.0 shipped. It also described Machine B as being in discrete-only
  mode, which stopped being true on 2026-08-22 when the MUX was switched to
  Optimus -- the change the section 6.1 verification depended on. Release
  status, the per-release table, section 6.1's state and the machine
  description are now current, and 2.4.0 is recorded as confirmed live rather
  than as unverifiable from the tree.

- **`README.md` documented `transfer()` and nothing after it.** The async path,
  the GPU-side wait through `shared_fence_handle`, and the consumer handshake
  were all shipped and none appeared, so the front page described a blocking
  copy as the whole story. Added, with the measurements attached: 7-14% for the
  GPU-side wait, and 28 of 60 frames wrong without the handshake.

- **Recorded the one thing not to do next.** Section 6.1 now says that another
  round of D3D12 synchronisation work needs a measurement or a user report
  first, and names prebuilt native wheels as the larger return -- every figure
  in section 3 that makes this library worth choosing sits behind an extension
  a `pip install` cannot currently reach.

### Measured

- **The direct single-adapter GPU path, which had never run.** Every ingestion
  figure in the README was taken on a hybrid laptop, where the direct path fails
  with `CrossAdapterRequired` -- no CUDA device owns the adapter the frame was
  captured on. Re-measured with the MUX in discrete-only mode so the RTX 4060
  drives the display: verified byte-exact, then **31.4 ms pixel age p50, 6.7 ms
  CPU per frame, zero host-to-device bytes**. That is the youngest pixel age
  measured anywhere in this project -- 2.2 ms ahead of the best hybrid path and
  5.2 ms ahead of DXcam. One 8-second pass in a different machine configuration,
  so its frame-rate column is not comparable with the hybrid table; pixel age
  and CPU per frame are.

- **The hybrid ingestion figures reproduced.** A single 8-second pass per path
  landed within a few percent of the committed medians of three: `grab()` at
  +37% unique frames over DXcam against the recorded 30%, pixel age 7.4% younger
  against the recorded 8%, and the cross-adapter path at 59% less CPU against
  the recorded 45%.

### Benchmarks

- **Pixel age, measured against one clock.** `native/src/bin/latency_source.rs`
  is a D3D11 source that encodes an incrementing frame ID into the image and
  records `frame_id -> QPC` at every `Present()`. Each path decodes the ID from
  what it captured, so latency is how old the pixels were -- not how long a
  call took -- and every library is timed on the same clock. `Frame`'s own
  present timestamp would have given RapidShot an advantage no other library
  could match. Run through `benchmarks/section7.py`; recorded in
  `benchmarks/section7-ingestion-machineB.json`.

  Machine B, 2560x1600 at 165 Hz, pixels to a `(1, 3, 640, 640)` FP16 tensor on
  CUDA, medians across 3 passes of 8 s per path. `grab()` returns **140.5**
  unique frames per second against DXcam's 108.2 (+30%) with pixels **3.0 ms
  (8%) younger**, and `nvidia_gpu=True` returns 128.8 at **4.4 ms of CPU per
  frame against 8.9**; both beat DXcam on frames, age and CPU on every pass.
  The cross-adapter paths are capped near **80 frames a second**, 25% fewer
  than DXcam, while costing 45% less CPU and still returning younger pixels;
  the GPU-side semaphore variant gains no latency there and costs more CPU than
  DXcam.

  A single 5 s pass the day before had the cross-adapter paths level with
  DXcam and the semaphore path as the lowest-latency one. The cross-adapter
  paths reproduced within 2%; everything reading frames back to the CPU ran
  38-56% faster in the second session, for reasons not established. The 3-pass
  recording replaces it, and the claims that did not survive were withdrawn.

- **Call duration from capture to tensor, and through inference**
  (`benchmarks/ai_ingestion.py`, `benchmarks/ai_pipeline.py`). Every path is
  verified against a float64 NumPy reference taken from the same frame, and the
  GPU paths feed ONNX Runtime through `io_binding` so they are not charged a
  round trip they exist to avoid. Capture is **80-96% of the end-to-end
  budget**, which settles that capture-path work is worth doing. Cross-adapter
  costs 0.45x DXcam's CPU per frame and moves zero bytes host-to-device, at
  2.7 ms more latency than RapidShot's CPU path; `transfer_async()` does not
  close that gap, as section 6.1 predicted.

  These are call durations, so they are lower bounds on present-to-inference
  latency, and the inference figures use a FLOP-calibrated stand-in rather than
  a trained model -- superseded by the YOLO11n measurement below.

- **Present to inference with a trained YOLO11n, measured as pixel age**
  (`benchmarks/section7-inference-machineB.json`). The official Ultralytics
  weights, exported with Ultralytics' own exporter and run entirely on the RTX
  4060; age is taken when the forward pass completes. Medians across 3 passes of
  8 s per path, every path verified before timing.

  Every RapidShot path beat DXcam on every pass: **10-20% more unique frames,
  pixels 2.6-3.9 ms (6-9%) younger, 5-33% less CPU per frame**, and each path's
  worst pass still beats DXcam's best. Against DXcam's WGC backend and mss the
  frame and age leads hold on every pass too; on CPU one pair's pass ranges
  overlap, and the medians still favour RapidShot. The lead is smaller than at
  the tensor (20% more frames, against 30%) because the loop is synchronous and
  the model's 4-5 ms paces every path. RapidShot's own paths finish within
  1.3 ms of each other, inside their spread.

  The published `yolo11n.onnx` was tried first and is not the recorded model:
  it targets opset 22, for which ONNX Runtime 1.30 has no CUDA `MaxPool`
  kernel, so 7 of its nodes run on the CPU and ORT's spinning thread pool
  dominates the CPU column. It ranked the paths the same way in a trial pass.

  Three harness bugs meant the inference category had never completed a run:
  a provider check no ONNX Runtime session could pass, an input-shape check
  that rejected the published model's symbolic axes, and a verify step that
  still demanded bit-identical tensors after the loop moved to the idiomatic
  resize, failing every CPU path. All three are fixed, and the relaxations are
  recorded in every result.

- **Two apparatus bugs found on the way, both of which flattered or hid a
  result.** The motion source defaulted to 60 fps on a 165 Hz panel, so every
  path reported about 60 unique frames a second and looked like a tie;
  `section7.py` now follows the display's physical refresh and prints the
  achieved rate. And to make every path emit a bit-identical tensor, the CPU
  arms ran an exact rational resize costing 76-86 ms in NumPy against 0.72 ms
  for the `cv2` call a real caller writes -- charging mss, DXcam and RapidShot's
  own CPU path roughly seventy times their true conversion cost, in the GPU
  paths' favour. Each backend now uses its idiomatic resize, within 1/255 of
  the exact reference every path is still verified against.

- **Each library now competes in its best configuration, not its default.**
  `compare_libraries.py` measured RapidShot with its native kernels against
  DXcam and BetterCam on their cv2 path, which is one library's best against
  another's default. Added `rapidshot-numpy` (what a toolchain-free
  `pip install` actually gets), `rapidshot-gpu`, `rapidshot-unpooled`,
  `dxcam-numpy` (DXcam ships a NumPy processor beside its cv2 one) and
  `bettercam-gpu`.

- **The environment block records `cv2`, `cupy` and `cv2_threads`.** It did not,
  and without them a CPU figure is uninterpretable: OpenCV 5.0 defaults to one
  thread per logical core, so `cvtColor` reports ~450% CPU on this 32-thread
  machine while finishing *faster* in wall-clock than the single-threaded
  alternative. The 2026-08-06 recording cannot be reconciled with the current
  one for exactly this reason -- it never recorded which OpenCV it ran.

- **Recorded `benchmarks/library-comparison-machineB.json`**, Machine B at
  2560x1600 on 2.4.0, pooled across five independent full-matrix runs. Kept
  separate from `library-comparison.json` (Machine A, 1080p/100 Hz, 2.1.0)
  because the panels differ and most of the frame-rate gap between them is the
  panel, not the code.

  Pooling changed a conclusion, which is the argument for doing it: on one run
  the AVX2 and NumPy builds were indistinguishable end to end, and across five
  the extension is consistently worth **1.10x** -- real, and nothing like the
  **6.4x** it shows on the synthetic conversion benchmark. At 2560x1600 the
  staging read dominates `grab()`, so making conversion six times cheaper moves
  the total by a tenth. Quoting the synthetic ratio as a user-facing number was
  overselling it.

- **Buffer pooling measured against live capture for the first time**, at
  **1.21x** on fullscreen RGB for about 50 MB -- and nothing at all at region
  size, where both sit at the compositor ceiling. The 1.3-2.1x quoted elsewhere
  came from a synthetic benchmark and does not reproduce in a capture loop.


- **Re-recorded `benchmarks/baseline-rtx4060-hybrid.json` at 2.4.0.** It was
  recorded at 2.3.0, so the only same-machine baseline this host had was a
  version stale. 2.4.0 measured against the old recording was `~ same` on every
  synthetic row, and for the first time with **no NOT COMPARABLE rows** -- both
  sides now sit after the 2.3.0 redefinitions, where the 2.1.0 `baseline.json`
  suppressed two.

  Noise floor verified on the host first: `--self-test` reported 0 of 16
  benchmarks exceeding 1.30x with no code change, max drift 8%, so the default
  threshold clears the measurement error here.

- `baseline-rtx4060.json` (same machine, discrete-only) is **still at 2.3.0**.
  Re-recording it needs the MUX switched out of hybrid, which is a firmware
  change, not a benchmark flag.

## [2.4.0] - 2026-08-22

**The hybrid path works end to end.** Capture runs on the integrated GPU, the
frame crosses to the discrete one, and a CUDA consumer reads it there without
the CPU touching the pixels or the synchronisation. That was the point of
ROADMAP section 6.1 and it had never run on real hardware; a MUX switch on the
development machine made it testable, and the configuration promptly broke
capture entirely, which is where most of this release came from.

**Capture no longer gives up when the display-owning adapter refuses.**
Duplication now tries every adapter rather than assuming the one that owns the
output can duplicate it, keeps adapters with no outputs as candidates, and
explains an all-adapter refusal instead of printing an HRESULT. None of those
four defects would have surfaced without a machine in a state nobody had seen.

**Measurements that had been carried forward stopped being true.** Two entries
in the roadmap said not to bother with things worth ~10% and ~14%, both
recorded accurately on hardware where they were noise and both wrong here. The
harness gained a matching correction: rows whose meaning changed between
releases are no longer compared across that change, and caveats now apply to
improvements as well as regressions, because nobody investigates good news.

### Added

- **`CrossAdapterTransfer.transfer_async()` and the cross-adapter shared
  fence.** `transfer()` blocks until the copy completes, which on a hybrid
  laptop is 85% of it. The async path submits and returns, leaving
  synchronisation to `wait_shared_fence(value)` or to a GPU-side wait on
  `shared_fence_handle` from the destination adapter.

  Measured against a real GPU consumer (BGRA to float32, normalise, reduce over
  the 16.4 MB frame in CuPy), Intel iGPU to RTX 4060 at 2560x1600, two
  interleaved runs:

  | strategy | run 1 | run 2 |
  | --- | --- | --- |
  | blocking + CPU wait | 16.57 ms | 14.81 ms |
  | async + CPU fence wait | -0.9% | +2.5% |
  | async + **GPU semaphore** | **+14.2%** | **+7.5%** |

  **The CPU-side async wait buys nothing against a GPU consumer.** The gain is
  in `shared_fence_handle`: CUDA imports the D3D12 fence with
  `cuImportExternalSemaphore` and waits on it in a stream, so no CPU is
  involved in the handoff. Verified 6/6 frames byte-exact through a GPU-side
  wait, with the fence created on the Intel device and imported by CUDA on the
  NVIDIA one.

  Quote 7-14%, not a point estimate: live capture varies and two runs disagreed
  by a factor of two on the margin. An earlier synthetic benchmark reported 42%
  using a CPU busy-loop as the consumer; that is real for a CPU-bound caller
  and does not survive a GPU-bound one, which is what this feature exists for.

  `transfer()` still blocks and remains the default. The async path pipelines
  to depth one: the command allocator cannot be reset while the GPU reads it,
  so it waits for the previous submission before recording.
- **`set_consumer_fence()` / `wait_for_consumer()`** let the producer wait for
  an asynchronous consumer before reusing the shared destination buffer. Every
  transfer writes the same buffer and `shared_fence` only reports that the copy
  finished, so a consumer still reading frame N could be overwritten by the
  copy for N+1. The wait is queued on the source queue, so it orders ahead of
  the next copy without blocking the caller.

  Feasibility was the open question and is now measured: CUDA on an NVIDIA
  dGPU can signal a D3D12 fence created by an Intel iGPU's device, and the
  producer observes it.

  Measured at scale over a 60-frame loop whose consumer was slower than the
  producer: **28 of 60 frames wrong without the handshake, 0 with it.** With a
  consumer that keeps up the same loop is clean either way over 100 frames,
  which is why the hazard stays invisible until a real workload arrives.

  `transfer_async()` and `shared_fence_handle` now carry that warning and name
  the remedy -- the mechanism existed but said nothing at the API a caller
  actually reads.

  **Reproduced deterministically**, after four failed attempts: gate the
  consumer's read behind a semaphore, let frame B's copy complete while the
  read is provably still pending, then open the gate. Unguarded, the consumer
  read frame B after waiting for frame A -- 3/3 runs. With the handshake it
  reads A. Covered by a test.

  The earlier attempts all failed for one reason worth recording: CuPy's
  allocator synchronises the calling thread, so anything allocating inside the
  gated region either hides the race or self-deadlocks. Making the consumer
  slower was the wrong axis -- a 527 ms consumer showed nothing.

  A buffer ring was rejected as the alternative: it widens the window rather
  than closing it, since the producer wraps after N frames.
- `shared_fence_submitted` / `shared_fence_completed` expose what was queued
  versus what the GPU has reached. Diagnostic, and the instrument the
  handshake was built with.
- **`wait_shared_fence()` releases the GIL.** It previously held the
  interpreter for the entire wait, so the calling thread it handed back could
  not run Python -- a second thread made zero progress across a 7 ms wait.
  Armed under the GIL, waited with it released; the longest stall fell to ~1 ms
  and the other thread ran throughout. `transfer()` still holds the GIL for its
  copy (~2.7 ms): its wait sits between COM calls on interior-mutable state and
  cannot be released without restructuring resource lifetimes.
- **`probe_transfer_phases()`**, splitting a transfer into open / record /
  submit / signal / wait / close, as `probe_dispatch_phases` already did for
  the Stage 6 preprocessor. It takes a `use_cache` flag so the handle cache
  below can be A/B'd inside one harness rather than across two runs.

### Changed

- **The captured texture is opened once per texture, not once per frame.**
  `CreateSharedHandle` + `OpenSharedHandle` + `CloseHandle` ran on every
  transfer, costing 268 us + 114 us -- **10.3% of a transfer**, removed. Keyed
  on the raw pointer, so a changed surface reopens rather than reusing a stale
  resource. Output is byte-identical either way, which is why
  `cached_texture_address` is exposed and the test asserts on the key rather
  than on pixels.

  ROADMAP 6.1 previously said not to chase this, on three runs where the
  overhead was indistinguishable from noise. That was accurate for the hardware
  it was measured on and wrong as a general instruction.


- **`CrossAdapterTransfer.shared_destination_handle`** completes the Optimus
  path. Capture runs on the integrated GPU, so the Stage 6 tensor lands on an
  adapter CUDA cannot see; the frame already crossed adapters, and this NT
  handle is how a consumer on the destination adapter reaches it. Verified end
  to end 2026-08-22 on an Intel iGPU to RTX 4060 pair: capture on the iGPU,
  transfer, one CUDA import, then 8/8 frames read byte-exact through the
  imported view with no re-import and no CPU round-trip of the frame.

  **Import it as a heap, not a resource.** The transferred buffers are *placed*
  resources, and `CreateSharedHandle` refuses those with E_INVALIDARG on both
  devices -- so the shared object is the heap, imported as
  `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP` (4). That differs from
  `GpuPreprocessor12.shared_output_handle`, a committed resource imported as
  type 5, and the difference is not cosmetic: copying the Stage 6 pattern here
  does not work. The handle is borrowed -- created once, closed when the
  transfer is dropped -- matching the Stage 6 convention.
- **`CrossAdapterTransfer.probe_shared_handles()`**, a diagnostic for bringing
  up a new adapter pairing. Which objects can be shared is a property of the
  driver pair and this project has verified exactly one, so the decision above
  is re-checkable rather than folklore. It leaks nothing: every handle it mints
  is closed before returning, which a test asserts by process handle count.

### Verified

- **Cross-adapter transfer verified on a real Optimus laptop, Intel to NVIDIA.**
  `examples/verify_cross_adapter.py` carried 5 captured frames from an Intel
  iGPU to an RTX 4060 -- 16,384,000 bytes each at 2560x1600, every one
  byte-exact against a source-side readback. `probe_cross_adapter()` reports
  `representative: true` with hardware on both ends for the first time. This
  had been open since the project began; the mechanism was previously verified
  only against WARP as the destination. Copy cost is 1.39 ms min / 5.68 GB/s,
  roughly half the reverse direction's throughput -- the iGPU is the weaker
  part and is also driving the display, so quote this figure for Optimus rather
  than the NVIDIA-to-Intel one.

  In this direction NVIDIA is the *destination* and reports
  `destination_row_major_texture: false`, so the row-major branch that was
  avoided on principle would have been the one that broke, in the exact
  configuration the feature exists for.

### Changed

- `examples/gpu_tensor_to_cupy.py` raises `CrossAdapterRequired` rather than a
  bare `RuntimeError` when no CUDA device owns the capturing adapter. A caller
  needs to tell "this frame is on the wrong GPU, transfer it first" -- routine
  on any Optimus laptop -- apart from "the import is broken", which is a bug.
- `tests/test_cuda_interop.py`'s four example tests skip when CUDA cannot see
  the capturing adapter, instead of failing. They assumed capture and CUDA
  always share an adapter, which was true of every machine this project had
  run on and is false on the hybrid systems it targets. The check runs through
  the shipped example rather than reimplementing the LUID comparison.

### Benchmarks

- **Recorded `benchmarks/baseline-rtx4060-hybrid.json`**, Machine B in Optimus
  with capture on the Intel iGPU. Kept separate from `baseline-rtx4060.json`
  (same machine, discrete-only): the recorded `gpu` differs, so the two refuse
  to gate against each other, which is correct -- they measure different
  capture adapters.
- **A caveat now applies in both directions.** `live`, `duty-cycle sensitive`
  and `sub-ms` were attached only to regressions, so a live row could print a
  bare `FASTER 8.70x` -- observed on identical code that read `SLOWER 2.26x`
  on the next run. The FASTER branch carried a comment claiming the qualifiers
  applied "in both directions on purpose" while implementing two of five. A
  spurious improvement is as misleading as a spurious regression and harder to
  notice, because nobody investigates good news.

- **The suite reported a landed code change as a performance win.** Comparing a
  2.3.0 run against the 2.1.0 `baseline.json` showed
  `pipeline.gpu_plus_readback` as **FASTER 6.08x**, which reads as a hardware
  result. It is not: 2.3.0 made `read_back` return bytes rather than a
  `Vec<f32>` that PyO3 expanded into 1.2 million Python floats per call, so the
  row measures something different than it did when that baseline was recorded.
  `perf_suite.py` now knows which rows were redefined in which release, marks
  them `NOT COMPARABLE` against any older baseline, and excludes them from the
  verdict **in both directions** -- a spurious improvement is the dangerous one,
  because nobody investigates good news. `pipeline.cpu_to_nchw` (redefined in
  2.2.0) is covered by the same table. A baseline with no version stamp fails
  closed and is treated as older.

### Fixed

- **Capture gave up when the display-owning adapter refused to duplicate.**
  RapidShot picked the adapter that enumerates the output, called
  `DuplicateOutput`, and failed if that refused. Which adapter Desktop
  Duplication accepts depends on where the desktop is actually composed, not on
  which adapter enumerates the output, so on hybrid systems this failed on
  machines that had a working adapter available. Duplication now tries every
  adapter and uses the first that succeeds, carrying the stage surface onto
  whichever device won. A refusal that is not adapter-specific -- a desktop
  refusal, protected content -- still propagates immediately rather than being
  retried against adapters that would refuse it for the same reason.
- **Adapters with no outputs were discarded during enumeration**, so they could
  neither be selected nor tried. On a hybrid laptop the integrated GPU often
  owns no output, which made `prefer_integrated=True` unable to find it and
  `device_idx` unable to name it -- the parameter was unreachable in exactly
  the configuration it exists for. They are retained now, and
  `prefer_integrated` orders them first when duplication is set up.
- **`DXGI_ERROR_UNSUPPORTED` surfaced as a raw COM string.** A hybrid system
  where no adapter can duplicate reported "The specified device interface or
  feature level is not supported on this system", which names neither the
  cause nor a fix. It now explains that the OS-level hybrid graphics path is
  not running, lists what to check (GPU mode set consistently and rebooted,
  current iGPU driver, control panel not forcing every app onto the discrete
  GPU), and prints the adapters it tried with what each reported. Same
  treatment `HeadlessError` already gave the no-display case.
- **Initialisation failures replaced their own diagnosis with "Check logs for
  details."** `_initialize_resources` logged the real exception and returned
  False; the caller then raised a generic error. The specific cause is now
  re-raised, so a diagnosis that had already been made reaches the caller
  instead of only the log.
- **`topology_info()` asserted that capture works on the display-owning
  adapter.** It printed "Capture runs on <adapter>" -- a prediction stated as
  fact, and false on a machine where that adapter refuses. It now describes the
  arrangement and says outright that whether capture works is only known once
  it has been tried.
- **The `gpu` extras installed OpenCV, which the GPU path stopped using in
  2.3.0.** `CupyProcessor` converted colour by copying the frame off the device
  and running `cv2` on the CPU; that was replaced with pure CuPy, but the extras
  still pulled 60+ MB of OpenCV that nothing on the path calls. `[cv2]` still
  exists for callers who want it.
- **Added `gpu_cuda13`.** The extras stopped at CUDA 12 while development runs
  CUDA 13.2 / CuPy 14.1.1, so there was no extra that matched the machine the
  CUDA interop was verified on. `gpu` still means CUDA 11 rather than being
  repointed, so existing installs do not silently change wheels.
- `setup.py`'s `extras_require` had drifted from `pyproject.toml` — it was
  missing `gpu_cuda12` entirely. Both are release artifacts (`RELEASING.md`
  tracks the version in three places), so a stale copy is not harmless.
- `ROADMAP.md` § 1 still announced 2.2.0 as the current release, and the README
  steered every NVIDIA user to the CUDA 11 extra without saying so.

## [2.3.0] - 2026-08-06

**First release verified on NVIDIA hardware.** Development had been on an Intel
iGPU with no CUDA-capable GPU, so every CuPy and CUDA path shipped untested. Run
on an RTX 4060, the suite went from 273 passed / 9 skipped to **323 passed / 1 skipped**, and two shipped bugs turned up — one of them silently returning
wrong pixels.

**Capture can now feed CUDA directly.** The GPU tensor exposes a shared NT
handle, so CuPy or PyTorch can map it with `cudaImportExternalMemory` and read
it in place: `grab_frame()` → `cupy.ndarray` with no CPU round-trip.
`examples/gpu_tensor_to_cupy.py` is a complete working consumer.

### Fixed

- **`create(nvidia_gpu=True)` returned wrong pixels for every colour mode
  except BGRA.** Conversion went through OpenCV, which is not a dependency; the
  resulting failure was logged and the *unconverted* 4-channel BGRA buffer
  returned as success. Callers asking for RGB got the wrong shape and the wrong
  channel order with no exception. Conversion is now pure CuPy, runs on the
  device, and is byte-identical to the NumPy path.
- **Every `E_ACCESSDENIED` from `DuplicateOutput` was reported as
  protected content.** A locked workstation, an open UAC prompt, a non-input
  desktop and a Session 0 service all advised closing a protected player window
  that did not exist. The real cause is now identified and named.
- `CupyProcessor.process()` no longer swallows failures and returns a
  possibly-invalid buffer; it raises. Unsupported colour modes are rejected at
  construction rather than on the first frame that arrives.

### Added

- `GpuPreprocessor12.shared_output_handle` and `.output_byte_size` for
  importing the tensor into CUDA or another D3D12 device.
- `GpuPreprocessor12.probe_dispatch_phases()` — per-phase dispatch timing.
- `examples/gpu_tensor_to_cupy.py`, verified byte-identical to a readback.
- `rapidshot/util/desktop.py` — reports which desktop is receiving input.
- 42 tests, including five paths previously reachable only by fault injection:
  protected content, the `DuplicateOutput` refusal, real exclusive fullscreen,
  the access-loss rebuild, and the preprocessor's cache-miss branch.
- `benchmarks/baseline-rtx4060.json`.

### Changed

- **`to_nchw()` is 1.6–1.8× faster at 640×640, bit-identical.** Its gather ran
  at ~1% of memory bandwidth and was 73% of the call; two sequential `take`
  calls replace two-dimensional advanced indexing.
- **D3D12 dispatch is 2.4× faster.** Opening the captured texture is cached per
  texture instead of repeated per frame, and the UAV moved to construction.
  Keyed on the texture pointer, so a changed surface reopens.
- **`read_back()` is 6.3× faster.** It returns bytes rather than a `Vec<f32>`
  that PyO3 turned into 1.2 million Python floats per call.
- The GPU tensor's output heap is `D3D12_HEAP_FLAG_SHARED`.
- `benchmarks/perf_suite.py` pins itself to the performance cores on a hybrid
  CPU and records the fact. Unpinned, it reported false regressions up to 2.57×
  against unchanged code.

### Notes

- `pip install cupy-cuda13x[ctk]` installs no CUDA headers against
  `cuda-toolkit` 13.3.x. Use `pip install "cuda-toolkit[cudart,nvrtc]==13.2.*"`.
- Cross-adapter transfer with an NVIDIA source: 0.68 ms per 1080p frame. NVIDIA
  does **not** support cross-adapter row-major textures, confirming the buffer
  path was required rather than merely safe.
- Hybrid (Optimus) topology remains unverified: the test machine's MUX is set
  to discrete-only, so it reports as a single-adapter system.

## [2.2.0] - 2026-08-06

**Capture that gets out of your way.** No library beats the compositor -- every
DXGI capture library lands at ~100 fps on a 100 Hz display, RapidShot included.
What separates them is the bill. RapidShot delivers those frames on **13.6% CPU
against DXcam's 79.7% and BetterCam's 74.4%**, and converts colour for about a
**third** of what they spend, which is the AVX2 kernels rather than a scheduling
choice. If capture is one stage of a pipeline that needs its cores for inference,
that difference is the whole point.

**The last mile got shorter.** `to_nchw()` turns a frame into model input in one
call -- `(1, 3, H, W)` float32, correct channel order, **7.20 ms → 4.05 ms**
against the version most people write, bit-identical output. And two knobs that
were previously ours to assume are now yours to set: `timeout_ms` picks your
point on the CPU-versus-latency curve, and `pool_size_frames` your memory
footprint.

**Memory dropped 60 MB per camera**, from 174 MB to 114 MB, at no measurable cost
in frame rate. It remains roughly 40 MB above DXcam's, and `pool_size_frames`
takes it lower still if that matters more to you than buffer reuse.

**And you can now check all of it yourself.** This release adds a reproducible
cross-library comparison against DXcam, BetterCam and mss -- error bars, isolated
processes, a calibrated motion source -- along with the two results that do not
flatter us. Nobody should take a performance claim on trust, including ours.

Pick DXcam or BetterCam if capture is your whole program and you want the lowest
possible per-call latency. Pick RapidShot if it is one stage of something larger.

No breaking changes. The `pool_size_frames` default drops from 10 to 4; pass 10
explicitly to restore the previous behaviour.

### Benchmarks

- **The CPU arm of the preprocess comparison was a strawman; fixing it cost the
  GPU path 1.78× of its apparent win.** `pipeline.cpu_to_nchw` is not library
  code — RapidShot ships no CPU preprocess — it is the reference the GPU tensor
  path is measured against, and Stage 6 was promoted on that comparison. It
  widened the 640×640×4 gather to float32 *before* scaling (6.55 MB of traffic
  where 1.6 MB suffices), divided in a second pass, stacked channels into a
  fresh array in a third, and allocated ~11 MB per call. Writing each channel
  once into a preallocated destination is **6.84 → 3.90 ms, bit-identical**.
  The conclusion holds — GPU dispatch is 2 µs against 3.90 ms — but the margin
  was overstated for as long as the control arm was the first implementation
  rather than the best one. Both baselines re-recorded, since this changes what
  the row measures; recordings before 2026-08-05 are not comparable on it.
  A variant that looked 7× faster was discarded for sampling the wrong pixels:
  the resize indices are not a uniform stride (1080/640 = 1.6875), so a strided
  slice silently reads a different image. Caught only because every variant is
  checked against the original's output before its timing is believed.

### Added

- **`rapidshot.to_nchw()` converts a frame to model input** — `(1, 3, H, W)`
  float32 scaled to 0-1 — in one call, with the fast implementation rather than
  the obvious one. Measured at 1920x1080 to 640x640: hand-written naive
  **7.20 ms**, `to_nchw()` **5.18 ms**, `to_nchw(out=buf)` **4.05 ms** reusing a
  destination. Output is bit-identical to the hand-written version, so this is
  purely about which implementation you end up with.

  Deliberately narrow: output size and channel order, nothing else. No mean/std
  normalisation, no float16, no NHWC, no letterboxing — those are dictated by the
  model rather than chosen for speed, and measured within ~1.5x of each other
  (2.5-5.1 ms), so there is nothing to gain by guessing. Two costs are documented
  rather than hidden: `source_order` is **required**, because an `(H, W, 3)`
  array cannot say whether it holds RGB or BGR and feeding a model the wrong one
  fails silently; and the resize is nearest-neighbour, not the bilinear most
  detection models were trained with.

- **`pool_size_frames` is now public, and its default drops from 10 to 4.** It
  was settable on `ScreenCapture` but the factory never forwarded it, so the
  largest tunable part of the process footprint was unreachable through
  `rapidshot.create()`. Measured per-camera cost, one configuration per process
  (measuring several in one interpreter cannot work — CPython does not return
  freed arenas, so each config inherits the previous one's):

  | `pool_size_frames` | camera cost | fps |
  | --- | --- | --- |
  | 10 (old default) | 173.6 MB | 97.1 |
  | **4 (new default)** | **113.6 MB** | 95.4 |
  | 2 | 84.3 MB | 98.8 |

  **60 MB saved for −1.8% frame rate**, which is inside run-to-run noise. This
  closes most of the memory gap against DXcam (87 MB) measured in
  `benchmarks/compare_libraries.py`. Running the pool dry remains safe rather
  than an error — verified by holding six frames against a two-buffer pool, all
  readable and all carrying different content, so it falls back to allocating
  rather than recycling a buffer someone is still reading. `0` and `bool` are
  rejected: an empty pool is a permanent fallback, not a smaller pool.

- **`timeout_ms` is now public** — as `rapidshot.create(timeout_ms=...)` and as a
  settable `ScreenCapture.timeout_ms` property that takes effect on the next
  acquire. It controls how long each acquire waits for the compositor, and it is
  the single parameter separating this library's behaviour from the poll-based
  ones. Measured on a 100 Hz output against a source presenting at ~610
  updates/s:

  | timeout | fps | hit rate | CPU |
  | --- | --- | --- | --- |
  | 0 (poll) | 127.8 | 2.4% | 68.6% |
  | 10 (default) | 118.9 | 100% | 15.7% |

  Frame rate barely moves; CPU moves 4×. Polling is what DXcam does — it calls
  `AcquireNextFrame(0)` and reached 134 fps at 66% CPU in the same comparison, so
  the extra frames are real but expensive. The default stays at 10 ms because a
  capture stage in a pipeline usually wants its cores back more than it wants the
  last 7% of frames. Negative, non-integer and `bool` values are rejected —
  `bool` explicitly, since it is a subclass of `int` and would otherwise silently
  configure a 1 ms wait. The setting is carried across a duplication rebuild, so
  a resolution change or display reconnect no longer resets it.

### Fixed

- **The "no screen updates" warning fired while capture was running at 117 fps.**
  It counted *consecutive empty acquires*, which makes the threshold depend on
  `timeout_ms`: at the 10 ms default, 100 misses mean a second of still screen,
  but while polling with `timeout_ms=0` the same 100 misses take under 20 ms and
  are entirely normal at a 97% miss rate. It now measures how long it has
  actually been since a frame, reports that in seconds, and is rate-limited —
  false warnings during a 6 s polling capture went from 7 to 0.

### Documentation

- **ROADMAP.md § 4 corrected: capture rate is bounded by the compositor's present
  rate, not the display's refresh rate.** The entry claimed "at most one frame
  per display refresh"; § 6.2 already contradicted it ("DDA is driven by presents,
  not refresh") and the two sat in the same document. Measured on a 100 Hz panel:
  705 frames in 6 s, **705 distinct `LastPresentTime` values** (zero repeats),
  inter-present gaps down to **1.01 ms**, and `AccumulatedFrames` showing the
  compositor produced ~188 presents/s while the capture loop caught 117.5. A
  library reporting more frames per second than the refresh rate is not
  necessarily lying. README claims updated to match.

- ROADMAP.md § 4 records that **DWM does not emit move rects** — 2,205 frames of
  live capture under a workload built to produce them returned zero, so
  `move_rects` is deferred rather than pending, and the latent correctness hole
  it would close is documented instead of lost.
- ROADMAP.md § 6.1 records that the cross-adapter **CPU-side wait is not the
  cost** (0.83 ms min / 1.01 ms median copy at 9.5 GB/s, indistinguishable from
  `transfer()`), so a shared fence is a latency change, not a throughput one —
  and cannot be validated against WARP.
- ROADMAP.md § 6.3 records the **dirty-fraction distribution**, which is
  workload-dependent: 0.7–0.8% for a small animated window, but median 68% for a
  dragged and scrolling one, where the optimisation decays toward 1.0×.

### Benchmarks

- **A cross-library comparison against DXcam, BetterCam and mss**
  (`benchmarks/compare_libraries.py`), with each library in its own process --
  a correctness requirement, not tidiness: all three DXGI libraries declare the
  same COM interfaces and whichever imports first breaks the others.
  `benchmarks/motion_source.py` supplies on-screen motion and reports its own
  achieved rate, because a source slower than the display silently becomes the
  ceiling and every library then reports *it* rather than themselves.
- **Every measured quantity carries an error bar**, taken across repeated runs
  and printed next to the value. Frame rate alone was not enough: CPU moved
  between runs by far more than frame rate did, and CPU is the column the
  headline claim rests on. CPU is now derived from `cpu_times()` rather than
  sampled, which removes the sampling error entirely -- though the remaining
  variation is the machine rather than the instrument, so the column is
  directional and the README says so.
- **`benchmarks/compare_recordings.py`** diffs stored recordings against each
  other, normalising each by its own control row first.

### Documentation

- **The README now states where RapidShot loses.** It is not faster than DXcam
  or BetterCam -- every DXGI library sits at ~100 fps because the compositor is
  the ceiling -- and it uses the most memory of the four (124 MB against
  BetterCam's 80 MB). Most of the CPU advantage is a default rather than an
  achievement, which the `timeout_ms=0` row in the table makes unavoidable. What
  survives as engineering is the colour conversion, which costs about a third of
  DXcam's CPU for the same work.

## [2.1.0] - 2026-08-05

**Colour conversion stops being the CPU bottleneck.** It was the dominant cost in
every capture that leaves the GPU; all five modes now run at 69–100% of what the
dev machine's memory system can move, so there is little left to win there.

**Read the two figures separately, because they differ by a lot.** A plain
`pip install rapidshot` needs no toolchain and gains **GRAY 1.5–1.8×** — the other
modes are unchanged, deliberately. The large numbers below (6–37×) need the
*optional* native extension, which requires Rust plus the MSVC build tools. Both
paths produce byte-identical output, so the extension is purely a speed choice and
never a behaviour one.

No breaking changes. `grab()` still returns a `PooledBuffer` as it has since 2.0.0.

### Performance — Stage 1b (pixel path)

- **GRAY is 1.5–1.8× faster with no toolchain, and 24× faster with the optional
  native extension.** Output is byte-identical in both cases, asserted over all
  16,777,216 BGR combinations. GRAY had been the one colour mode that could not
  keep up with a 60 Hz display at 1080p (13.7–14.9 ms against a 16.67 ms frame);
  it now costs 8.5–11 ms in pure NumPy and 0.70 ms through the native kernel.
  - The NumPy path stopped allocating. The previous formulation built a
    full-frame `uint16` temporary per channel, and those page faults cost more
    than the arithmetic they carried; intermediates are now reused across frames.
  - **An AVX2 kernel takes GRAY to 0.26 ms** — 37× the NumPy path, 59× the 2.0.0
    formulation, and 96% of the memory system's 33.2 GB/s limit. It beats
    single-threaded OpenCV (0.34 ms) while being byte-exact, where OpenCV differs
    by up to 1 LSB. GRAY went from the slowest colour mode to the fastest.
    The obvious instruction, `_mm256_maddubs_epi16`, is unusable: it saturates as
    signed i16 while `b*29 + g*150` reaches 45,645, so it would clamp and corrupt
    bright pixels silently. Widening to u16 and using `_mm256_madd_epi16`
    accumulates into i32 where nothing can overflow. Because that failure mode
    hides in highlights, the correctness test is **exhaustive over all 2²⁴ BGR
    triples through the vector path**, not sampled.
  - `native/src/luma.rs` adds a byte-exact Rust kernel, used automatically by
    `NumpyProcessor.convert_into` when the extension is present and silently
    declined when it is not — so `pip install rapidshot` is unaffected, per the
    "optional means optional" principle in ROADMAP.md § 11. It releases the GIL
    for the duration of the conversion, and addresses both source and
    destination by their own row pitches, so dirty-rect patches into the
    accumulator go through it without being copied first.
  - New `benchmarks/gray_kernel.py` compares four NumPy formulations and the
    native kernel against a hand-written SIMD reference, and records two
    negative results worth keeping: a lookup-table formulation is **1.5× slower**
    than the multiplies it replaces, and a `uint32` SWAR formulation that reads
    contiguously is no better than the strided `uint16` one.

- **Native kernels for RGB, BGR and RGBA too** (`native/src/swizzle.rs`), byte-exact
  and used automatically when the optional extension is present: **RGB 6.4×,
  BGR 6.5×, RGBA 7.5×** at 1920x1080, putting all three at 69–79% of the memory
  system's measured 33.2 GB/s limit. BGRA is untouched — a straight copy already
  running at that limit, and the control proving the old 2–3 ms figures were never
  a memory constraint but three strided passes doing one pass's work.
  - The reorder modes use an AVX2 `pshufb` permutation, detected at runtime with
    the scalar loops as fallback (x86_64 guarantees only SSE2). The
    autovectorised loops alone reached just 2.5–7.2×, and RGB was the worst
    because reversing each triple defeats the vectoriser — `pshufb` took it from
    0.82 ms to 0.32 ms.
  - The 3-byte kernels store **exactly 24 bytes**, not a full vector. The usual
    store-32-and-overwrite trick would run past a row end, which for a
    dirty-rect patch is the next row of live pixels — silent corruption rather
    than a crash.
  - Correctness is asserted over every distinct BGRA quad per mode, on odd shapes
    down to 1×1, on strided sub-rectangles that must leave the rest of the
    accumulator intact, and vector-against-scalar at every width from 1 to 64 —
    which covers all eight tail lengths, where such a bug would otherwise hide.
  - The NumPy fallbacks are unchanged, so a toolchain-free install performs
    exactly as before.

### Benchmarks

- **Two committed recordings instead of one**, both taken 2026-08-05 back-to-back
  so they are comparable to each other rather than separated by machine drift:
  - `benchmarks/baseline.json` — **with** the native extension. What the library
    can do on stated hardware; feeds the README badges.
  - `benchmarks/baseline-nonative.json` — **without** it. What a plain
    `pip install rapidshot` gets, and what CI's compare step now points at, since
    the runner builds no extension. Aimed at `baseline.json` it would report a
    6–20× "regression" on every conversion row forever.
  - The 07-30 recording is preserved as `benchmarks/baseline-2026-07-30.json`.
- **New `benchmarks/compare_recordings.py`** diffs stored recordings against each
  other, normalising each by its own `control.memcopy` row first so a recording
  taken on a cold machine is not credited for it. Measured gains against 07-30:
  **GRAY 20.8×, RGBA 7.0×, RGB 6.2×, BGR 6.0×**; BGRA unchanged (it is a copy),
  and `pipeline.cpu_to_nchw` unchanged (untouched).
- **Badges now come from deterministic synthetic rows only.** `BGRA→RGB`,
  `BGRA→GRAY` and `shot()→buffer` replace the former `grab()` and
  `grab_frame()` badges, which were fed from live capture and could not be
  trusted: across seven recordings on code that only ever got faster,
  `live.grab_frame_gpu` spanned 0.17–0.83 ms — a 4.9× swing on a path that
  performs no conversion at all, so the badge reported the desktop rather than
  the library. The live figures are still measured and still in ROADMAP.md § 3
  with their range stated; they are simply no longer advertised as if stable.
- **A cross-machine comparison no longer reports verdicts as if they were real.**
  `control.memcopy` measures memory bandwidth, so drift-normalising by it only
  works for benchmarks that are also bandwidth-bound. On a CI runner it reported
  `pipeline.cpu_to_nchw` (float32 resize/normalise/transpose, compute-bound) as a
  1.34× regression against untouched code, while calling every conversion row
  1.4× *faster* on a machine that was uniformly slower — one control standing in
  for workloads it does not resemble, wrong in both directions at once.
  `print_comparison` now detects a baseline recorded on different hardware, marks
  every verdict indicative, and gates nothing. Improvements are flagged as
  loudly as regressions.
- The regression summary claimed "the 10% threshold" while the default was 1.30,
  telling anyone reading a failure that a change had to be 10% to count when it
  actually had to be 30%. It now quotes the threshold in force.
- Also recorded there: a minimum is monotonically non-increasing in sample count,
  so raising `--live-seconds` makes live numbers look better on unchanged code.
  Live rows are comparable only at the same setting.

### Documentation

- ROADMAP.md § 4 records that **DWM does not emit move rects** — 2,205 frames of
  live capture under a workload built to produce them returned zero, so
  `move_rects` is deferred rather than pending, and the latent correctness hole
  it would close is documented instead of lost.
- ROADMAP.md § 6.1 records that the cross-adapter **CPU-side wait is not the
  cost** (0.83 ms min / 1.01 ms median copy at 9.5 GB/s, indistinguishable from
  `transfer()`), so a shared fence is a latency change, not a throughput one —
  and cannot be validated against WARP.
- ROADMAP.md § 6.3 records the **dirty-fraction distribution**, which is
  workload-dependent: 0.7–0.8% for a small animated window, but median 68% for a
  dragged and scrolling one, where the optimisation decays toward 1.0×.

## [2.0.0] - 2026-08-02

**Upgrade if you are on any earlier version.** rapidshot 1.1.0 does not import
at all on Python 3.11 or newer — `cursor: Cursor = Cursor()` trips the dataclass
mutable-default check that was broadened in 3.11 — and patching that one line
only gets it to return all-black frames, because the processor is handed a
texture where it expects a mapped staging surface. 2.0.0 is the first version
that captures a frame on a current Python.

**One breaking change:** `grab()` now returns a `PooledBuffer` you must
`release()`. It indexes and converts like the array it wraps
(`frame[y, x]`, `np.asarray(frame)`, `.shape`, `.dtype`), so migration is
usually one added line — or pass `pool_output=False` to keep 1.x behaviour
exactly. Details under *BREAKING* below.

Also the first release published through PyPI Trusted Publishing with signed
Sigstore attestations, so the artifacts can be verified back to the workflow
that built them.

Everything below is the detail.

### Fixed — SBOM generation was broken and would have been misleading

- The release workflow used `--outfile`, which `cyclonedx-py` does not accept —
  the flag is `--output-file`/`-o`. The step failed outright, so no release
  could complete.
- The deeper problem was what it would have produced once fixed. `cyclonedx_py
  environment` documents *everything installed*, and at that point the build
  environment also holds `build`, `twine`, `cyclonedx-bom` and its ~30
  dependencies. The SBOM would have listed `lxml`, `jsonschema` and `arrow` as
  though rapidshot depended on them — worse than no SBOM, because a consumer
  would scan it for vulnerabilities in packages the library never touches.
- Now generated against the verification venv, which holds the built wheel and
  its runtime dependencies only: **rapidshot, numpy, comtypes, pip**. A check
  fails the release if build tooling appears.

### Added — Python 3.14 declared and tested

- The package already installed and worked on 3.14 (`requires-python` has no
  upper bound), but claimed only 3.9–3.13, so anyone checking the classifiers
  would have concluded it was unsupported. Verified before claiming it: the full
  suite passes on 3.14.6, live capture works, dirty rects come through, and the
  `abi3-py39` native extension loads — which is the point of abi3, now
  demonstrated across five minor versions.
- Classifiers, README and the CI matrix all updated together; CI now tests
  3.9, 3.11, 3.13 and 3.14.

### Fixed — CI lint gates were reporting success while failing

- **`cargo clippy -- -D warnings` was failing on 13 lints and had been all
  along.** Both it and `cargo fmt --check` were marked `continue-on-error`, so
  the job reported success and the failure surfaced only as an annotation. A
  gate that cannot fail is not a gate; both now gate, and all 13 lints are
  fixed rather than silenced:
  - `unwrap_err()` after `is_err()` in both shader compilers — now binds the
    error with `if let Err(error) = &result`
  - five `PCSTR(b"...\0".as_ptr())` byte strings — now `c"..."` literals, which
    is what makes the nul termination a property of the type rather than of
    remembering to type it
  - two `D3D11_*_VIEW_DESC::default()` followed by field assignment — now
    struct-literal initialisation
  - four hand-rolled `(x + 7) / 8` — now `div_ceil(8)`
- Verified the shader path still works after the string changes: all 21 GPU
  preprocessing tests pass, and those compare exact pixel output against a
  NumPy reference, so a broken entry-point or target string would fail them.
- **Actions bumped off deprecated Node 20**: `checkout@v4→v5`,
  `setup-python@v5→v6`, `upload-artifact@v4→v5`, `download-artifact@v4→v5`.
  GitHub was already force-running them on Node 24 and warning about it.

### Added — performance badges that cannot drift

- README now carries CI status, PyPI version, supported Python versions and
  licence badges (all live from their own sources), plus per-frame cost badges
  for `grab()` and `grab_frame()` and one naming the hardware they were
  measured on.
- **The performance badges are generated from `benchmarks/baseline.json`, and
  CI fails if they disagree with it** (`benchmarks/make_badges.py --check`).
  Re-recording the baseline without regenerating them breaks the build rather
  than leaving the front page quietly wrong — which is exactly how "240Hz+" and
  "Python 3.7+" survived in the README long after both stopped being true.
- **They are deliberately not measured in CI.** GitHub runners have no desktop
  session, so Desktop Duplication does not run and the two figures worth
  showing do not exist there; what a runner *can* time is synthetic conversion
  work on a shared VM of unknown generation. A badge fed from that would report
  the runner rather than the library, and this project has recorded 1.9x swings
  on identical code even on dedicated hardware. The source of truth is a
  measurement on stated hardware, committed and re-recorded deliberately.

### Fixed — README claimed things that were not true

- **The cross-library FPS table is gone.** It reported RapidShot at "240+" and
  "300+ GPU-accelerated" against DXCam 210, MSS 75 and D3DShot 118, with no
  hardware, method or date attached. Both RapidShot figures were unsupportable:
  Desktop Duplication returns at most one frame per display refresh, so 300 fps
  needs a 300 Hz monitor, and CuPy acceleration changes the conversion cost
  rather than the rate frames arrive. ROADMAP.md section 4 already records that
  published FPS claims in this space contradict each other — the README was
  producing exactly that. Replaced with the reproducible per-frame costs from
  `benchmarks/baseline.json`, the hardware they were measured on, and a note
  that `grab_frame()`'s 0.21 ms is what the calling thread pays, not a capture
  rate.
- The Contributing section now links `CONTRIBUTING.md` instead of saying
  "feel free to submit a Pull Request".

- **"Python: 3.7+" in System Requirements.** `requires-python` is `>=3.9`, so
  pip refuses to install on 3.7 or 3.8 — the README was promising an install
  that cannot happen. Now 3.9+, matching `pyproject.toml`, `setup.py`, the CI
  matrix and the classifiers, which are all cross-checked.
- **"Ultra-fast capture: 240Hz+ capturing capability."** Desktop Duplication
  returns at most one frame per display refresh (ROADMAP.md section 4), so
  exceeding the refresh rate is not possible and the `+` claimed something the
  API cannot do. Replaced with what is actually true: RapidShot keeps up with
  that ceiling, and no capture library can beat it.
- The feature list omitted the headline 2.0 capabilities — GPU-resident frames,
  dirty-rect metadata, cross-adapter transfer, topology diagnostics — which are
  the reasons to choose this library over a plain DXcam fork.

### Added — community health files

- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and a pull
  request template, completing GitHub's community standards checklist.
- `CONTRIBUTING.md` is deliberately specific rather than boilerplate. It leads
  with the traps that have each cost someone a session — live tests need
  something moving on screen, synthetic textures cannot exercise the D3D12 path,
  two Python interpreters are often on PATH — and with the measurement rules:
  run `--self-test` before trusting a comparison, never benchmark per-frame work
  back-to-back, interleave when comparing two implementations.
- The PR template asks for the things that are expensive to catch in review:
  whether live capture was checked on real hardware (CI cannot), the benchmark
  comparison table when performance is touched, and what keeps the buffer
  lifetime sound when `frame.py` or `memory_pool.py` is touched.

### Security — CI workflow now runs with least-privilege permissions

- `ci.yml` set no `permissions:` block, so its `GITHUB_TOKEN` inherited the
  repository default — which for repositories created before February 2023 is
  **read-write**. Every step in all four jobs could therefore have pushed
  commits or opened issues on the strength of a checkout. Now scoped to
  `contents: read`, which is all any of them need;
  `actions/upload-artifact` uses its own artifact service rather than this
  token. Found by CodeQL (`actions/missing-workflow-permissions`), four alerts,
  one per job. `release.yml` already set it.

### Changed — dependencies brought to current

- **`windows` crate 0.58 -> 0.62.2.** One breaking change across four minor
  versions: `D3D11CreateDevice`'s software-rasteriser parameter went from
  `Option<HMODULE>` to a bare `HMODULE`, so `None` becomes `HMODULE::default()`
  — the same null handle, meaning "no software module".
- **Build requirements trimmed** to `setuptools>=64`. `setuptools_scm` was
  declared but never configured; the version is static in `pyproject.toml`, so
  it was downloaded on every build and did nothing. `wheel` has not been needed
  since PEP 517 builds became the default.
- **`pip install rapidshot[all]` no longer forces CUDA 11.** The `all` extra
  pulled `cupy-cuda11x`, which conflicts with `cupy-cuda12x` and cannot be
  installed alongside it — so a CUDA 12 user asking for "everything" got the
  wrong CuPy. `all` now covers only what is not CUDA-version-specific; `gpu`
  and `gpu_cuda12` remain the explicit choices.
- Runtime dependency lower bounds are unchanged. `numpy>=1.19`,
  `comtypes>=1.1`, `pillow>=8.0` and `opencv-python>=4.5` are minimums, not
  pins: raising them without a reason would force every downstream user to
  upgrade for nothing. The only OpenCV API used is a handful of colour-code
  constants, which are stable from 4.5 through 5.0.
- **No measurable performance change.** The full suite reports `~ same` on
  every benchmark against the 2026-07-30 baseline, which is the expected
  result: PyO3 and windows-rs are compile-time bindings and do not sit in the
  per-frame path.

### Security — PyO3 upgraded 0.23.5 to 0.29.0

- Clears three Dependabot advisories against the optional native extension:
  an out-of-bounds read in `nth`/`nth_back` for `PyList`/`PyTuple` iterators,
  a missing `Sync` bound on `PyCFunction::new_closure`, and a buffer overflow
  risk in `PyString::from_object`.
- **Rapidshot used none of the affected APIs**, and the iterator bug was
  introduced in PyO3 0.24.0 — which 0.23.5 predates — so the practical exposure
  was nil and the advisories matched on version range alone. Upgraded anyway:
  the alerts are otherwise permanent, and the crate's API surface here is small
  enough that staying current is cheap.
- No source changes were needed; the crate compiles unmodified across six minor
  versions. Verified by rebuilding, running the suite, and re-running the
  byte-exact cross-adapter transfer check on live capture.
- The abi3 target moves from `abi3-py38` to `abi3-py39`, matching
  `requires-python = ">=3.9"`. Python 3.8 is end-of-life.

### BREAKING — `grab()` returns a pooled buffer (2.0.0)

`grab()` now returns a `PooledBuffer` for every colour mode, not a freshly
allocated `ndarray`. **Callers must call `release()` when done with a frame.**

Why: allocating the output array costs ~1.6 ms per 1080p frame in page faults on
first touch — more than the colour conversion it feeds. Reusing buffers makes
`grab()` **1.3–2.1× faster** across RGB, RGBA and GRAY, with pixels verified
byte-identical.

**What still works.** `PooledBuffer` behaves like the array it wraps:
`frame[y, x]`, `frame.shape`, `frame.dtype`, `frame.ndim`, `frame.size`,
`len(frame)`, and `np.asarray(frame)` — the last is zero-copy, so OpenCV, PIL
and model input paths keep working at no cost.

**What breaks.**

- Code that never released frames now needs `frame.release()`. Without it the
  pool drains and capture falls back to allocating: slower, never incorrect.
- `isinstance(frame, np.ndarray)` is False. Use `np.asarray(frame)`.
- APIs that require a true `ndarray` — `Image.fromarray(frame)` — need
  `np.asarray(frame)` first.
- Reading a frame after `release()` raises `BufferReleasedError` instead of
  returning stale pixels. That is deliberate: the buffer belongs to a later
  frame by then, and silently reading it is the failure this replaces.

**Opting out.** `rapidshot.create(pool_output=False)` restores the 1.x
behaviour exactly.

BGRA is unaffected — it did no conversion and already returned a pooled buffer.

Added `PooledBuffer.copy()` for data that must outlive the release, and
`BufferReleasedError` in `rapidshot.memory_pool`.

### Added — Topology detection: headless and hybrid GPU (Stages 6.1, 6.2)

- **`rapidshot.topology_info()`** reports which adapters exist, which drive a
  display, and what that implies for capture. Unlike `device_info()` it includes
  adapters that *cannot* capture — that is the whole point, since those are the
  ones that explain the failure.
- **Headless machines now get an actionable error.** With no monitor attached
  there is no desktop to duplicate, and the old message was
  `"No usable graphics devices found. Check your display configuration."` — true
  but unactionable, and identical to the message shown when devices exist but
  fail to open. The new `HeadlessError` (a subclass of `DeviceError`, so existing
  handlers keep working) names the fix: install a virtual display driver (IDD).
  It also carries the caveat that a virtual display's advertised refresh rate
  does **not** raise capture rate — Desktop Duplication is driven by presents,
  not refresh.
- **Hybrid GPU systems are detected and reported.** On an Optimus laptop the
  dGPU has no outputs, so Desktop Duplication cannot run against it
  (`DXGI_ERROR_UNSUPPORTED`) and capture is bound to the iGPU. Nothing was
  broken before, but nothing said so either: a GPU-resident frame produced on
  the capture adapter cannot be consumed on the inference adapter without a
  cross-adapter copy, and that mismatch was invisible until it failed.
- **A software adapter is not a second GPU.** The Microsoft Basic Render Driver
  (WARP) has zero outputs exactly like an Optimus dGPU, so the naive check
  reports every ordinary desktop as hybrid. `DXGI_ADAPTER_FLAG_SOFTWARE`
  separates them; a test pins this.
- Adapters that fail `D3D11CreateDevice` are recorded rather than logged and
  discarded, so "no usable device" can distinguish *no display* from *every
  device refused to open*.
- New `rapidshot/util/topology.py`. The classification is pure data over
  already-read adapter descriptions, so headless and hybrid machines are tested
  without owning either; one live test asserts the real machine agrees with raw
  DXGI enumeration.

### Fixed — the benchmark suite measured per-frame work as a burn loop (Stage 0)

- **`perf_suite.py` sampled back-to-back, which is not how capture runs.**
  Sustained heavy vector work holds the CPU in a lower-power state, and GRAY has
  two modes because of it. Measured on identical code: **16.27 ms back-to-back,
  9.16 ms with a 16 ms idle gap, 9.91 ms in bursts with a 200 ms gap.** The
  suite reported whichever mode a run happened to land in — GRAY read 8.75,
  13.48, 15.08 and 15.70 ms across four runs of unchanged code, a 1.8x spread
  that flagged `SLOWER 1.65x`.
- **Reps are now paced to a frame period, not to a fixed idle gap.** A fixed gap
  is wrong in both directions, because a benchmark's real duty cycle follows
  from its own cost: RGB takes 1.8 ms of a 16.7 ms frame (~11% duty cycle,
  mostly idle) while GRAY takes 15.9 ms (~95%, effectively sustained). Sleeping
  out the remainder of a 60 Hz frame reproduces both from one rule. A 16 ms
  fixed gap handed GRAY a 50% duty cycle and reported 9.16 ms — a number a
  capture loop never achieves. Sub-millisecond benchmarks (the memcpy control,
  COM call overhead) stay unpaced: pacing them would cost more than they measure.
- **Duty-cycle sensitivity is measured, not assumed.** Each paced benchmark is
  also run back-to-back and the ratio recorded; above 1.25x it is annotated in
  the JSON, warned about on stdout, and **excluded from the regression gate** —
  the same treatment live benchmarks get, for the same reason. The comparison
  honours the *baseline's* flag as well as the current run's, because the
  detector only fires when a paced sample reached the fast mode: a run stuck in
  the slow mode looks self-consistent, goes unflagged, and would otherwise be
  gated against a baseline that got lucky. That exact case failed a run at
  `SLOWER 1.65x` before the fix.
- The control benchmark cannot detect any of this and reported "machine state
  comparable" throughout, because a memcpy is not heavy enough to trigger the
  power state that causes it.
- New `tests/test_perf_harness.py` — 14 tests driving the harness with a fake
  clock, since this logic decides whether a regression is believed.

### Changed — `benchmarks/baseline.json` re-recorded (2026-07-30)

- Numbers from the old and new pacing models are not comparable, so the baseline
  was re-recorded and verified with an immediate second run reading all
  `~ same`. The previous recording is preserved as
  `benchmarks/baseline-2026-07-27.json`.
- **GRAY's honest cost is 13.7–14.9 ms, not 9.16 ms.** The fast mode is a
  transient the CPU sustains for a second or two, and a capture loop never sees
  it. This means GRAY cannot keep up with a 60 Hz display at 1080p — a real
  limitation the old harness concealed.
- Other figures also moved (RGB 1.45 → 1.76 ms, `grab()` 4.82 → 4.53 ms,
  `grab_frame()` 0.17 → 0.21 ms). How much is the harness correction and how
  much is machine state three days apart cannot be separated after the fact,
  which is the argument for re-recording whenever the harness changes.

### Added — pooled output buffers, `rapidshot.create(pool_output=True)`

- **`grab()` gets 1.3–2.1× faster** for converted colour modes (RGB, RGBA,
  GRAY), measured by interleaved A/B on one camera instance. Never slower.
  Pixels verified byte-identical to the allocating path across all 6,220,800
  values of a 1080p RGB frame.
- The win is not the conversion, it is the allocation. A fresh output array
  costs ~1.6 ms per 1080p frame in page faults on first touch — more than the
  conversion it feeds. Reusing a buffer removes it.
- **Off by default, on purpose.** With it on, `grab()` returns a `PooledBuffer`
  that must be released, exactly as BGRA already does. Turning that on silently
  would hand existing callers a buffer they never give back, which the pool
  would then recycle underneath them — the same class of bug as the recycled
  pool buffers fixed in Stage 1b. Making it the default belongs in a major
  version.
- BGRA is untouched: there is no conversion, so the staging buffer is already
  returned with no copy and nothing to pool.
- Pool exhaustion falls back to allocating for that frame. Blocking capture, or
  recycling a buffer a caller is still reading, would both be worse. Verified by
  holding 25 frames without releasing: capture degrades to plain arrays and
  keeps running.
- Two bugs found by the tests before release: a mis-shaped `output_target` was
  swallowed by `process()`'s catch-all and silently ignored (it now validates
  before that handler, since it is a caller bug rather than a capture fault),
  and BGRA was checking out pool buffers it never used — the wrapper keeps the
  validated colour string while backends set theirs to `None` for BGRA, so the
  obvious `color_mode is None` test never matched.
- Interleaving matters: separate runs put GRAY at 0.93× and BGRA at 0.63×, both
  pure cross-run noise on a machine that has swung 50% between consecutive
  measurements. Only the interleaved comparison is trustworthy.

### Measured — dirty-rect read strategy, and where the time actually goes

- **Reading each dirty rect's columns instead of its whole rows makes no
  difference.** New `benchmarks/dirty_rect_read_strategy.py` compares the two
  against a real mapped staging surface with fixed rect shapes; they land within
  noise at tall-and-narrow (0.8% area, 11.5% rows), square, and wide-and-short.
  `_read_patch_columns` is kept only so the comparison can be reproduced.
- **A live `grab()` comparison was tried first and had to be thrown out.** The
  same two implementations measured 2.26×, 1.56× and 0.87× on three consecutive
  runs, because each frame's cost depends on what happened to change on screen
  at that moment. Timing live capture measures the desktop, not the code; the
  benchmark now holds one captured frame's staging surface mapped and drives
  fixed rects over it.
- **The read was never the bottleneck: allocating the output array is.**
  `np.empty` for a 1080p RGB frame is free (0.010 ms — pages are not committed),
  but filling it costs 1.785 ms against 0.158 ms for a buffer that already
  exists. **~1.6 ms per frame is page-fault overhead**, paid in every converted
  colour mode whether dirty rects are used or not, and larger than the
  conversion it feeds. Recorded in ROADMAP.md § 10 as the biggest remaining CPU
  win.

### Added — dirty-rect accumulated conversion (Stage 6.3)

- `grab()` now converts only the regions DXGI marked dirty, taking the rest of
  the frame from a persistent accumulator. **`grab()` goes from 4.56 ms to
  2.97 ms — 1.5× — on live capture** with `output_color="RGB"`.
- **The 12–15× projected from `dirty_rect_pipeline.py` did not materialise, and
  the reasons are recorded rather than buried.** `process()` is only 69% of
  `grab()`, so Amdahl caps the result; the staging read shrinks with dirty
  *rows* (11.5% live) rather than dirty *area* (0.8% live); and the pipeline
  benchmark modelled the mapped staging surface with an ordinary RAM buffer,
  which understated precisely the component that shrinks least.
- Falls back to a full conversion when the metadata is missing, the rect list is
  empty, the dirty area exceeds 90%, a rect is out of range, rotation is in
  play, or the output is BGRA (which returns the pool buffer with no copy at
  all, and would only be made slower by an accumulator).
- The accumulator is invalidated whenever the previous frame stops being a sound
  base: shape change, captured region moving, or any frame that bypassed it.
  Region identity is tracked separately from shape, because two same-sized
  regions would otherwise blend.
- The returned frame is a copy, never a view into the accumulator — a view would
  be rewritten by the next capture, aliasing exactly like the recycled pool
  buffers fixed in Stage 1b.
- 18 tests covering the correctness properties, plus live verification that the
  fast path engages, that consecutive frames differ, and that untouched regions
  stay byte-identical.
- **Fixed a regression introduced by this change before release:** adding
  `dirty_rects` to the NumPy backend alone broke `grab()` completely, because
  the `Processor` wrapper it actually dispatches through has its own fixed
  signature. `_grab()`'s catch-all turned the TypeError into a silent `None`
  plus a re-init loop, and **the entire test suite still passed** — nothing
  exercised that seam. The wrapper now forwards the argument only to backends
  that accept it, and three tests cover the dispatch path.

### Added — `frame.dirty_rects` (Stage 6.3)

- **The compositor already computes which regions changed, and Rapidshot was
  discarding it.** `GetFrameDirtyRects` was declared in `_libs/dxgi.py` with no
  argtypes, which made it callable but unusable — comtypes cannot marshal the
  out-parameters without them — and nothing ever called it.
- `frame.dirty_rects` returns `(left, top, right, bottom)` tuples **in frame
  coordinates**. DXGI reports them relative to the whole duplicated output, so
  passing them through unchanged would index outside the frame whenever a
  region is in use. `Frame` clips to the region and translates; rects that miss
  it entirely are dropped, rects that straddle its edge are clipped. Verified on
  live capture with an off-origin region.
- **`[]` and `None` mean different things.** Empty means no rects were reported;
  `None` means the metadata could not be read. A consumer that skips unchanged
  regions has to tell them apart, or it silently skips everything on a frame
  whose metadata failed. An empty list is *not* a claim that nothing changed —
  a mode change or a coalescing driver can report none while the image differs
  completely.
- `frame.rects_coalesced` reports when the driver merged rects rather than
  listing them, which makes the regions an over-estimate and a weaker basis for
  skipping work.
- The buffer-growth path is handled: `GetFrameDirtyRects` writes nothing and
  reports the size it needs, so it retries once at that size and then gives up
  rather than looping on a driver that always asks for more.
- Measured on real capture: an animated window on an otherwise still desktop
  reports **one rect covering 0.7–0.8% of the frame**.
- 15 new tests covering the coordinate mapping and, by fault injection, the
  buffer growth and error paths.

### Measured — does region-limited conversion pay? (Stage 6.3)

- **Yes, and there is no losing regime.** New `benchmarks/dirty_rect_savings.py`
  measures converting only the dirty rects against converting the whole frame,
  1080p BGRA→RGB: **168× faster** at the 0.8% dirty figure live capture
  produces, 9.9× at 15%, and still 1.3× at 80% dirty.
- Cost scales linearly with dirty area — the strided-view penalty that could
  have eaten the saving never materialises. Per-rect overhead is ~1 µs, so
  hundreds of rects remain affordable (64 rects at 0.8% dirty is still 27×).
- Output is verified against the full-frame conversion inside every rect and
  confirmed to touch nothing outside them, before any timing is reported.
- **The blocker is design, not performance.** Converting part of a frame
  requires the rest of the destination to already hold the previous frame, but
  `grab()` fills a fresh pool buffer and the pool recycles — a partially written
  buffer would carry another frame's pixels, which is the Stage 1b aliasing bug
  by another route. The fix is a persistent accumulator plus a copy-out, which
  is full-frame work the current path never pays.
- **So the accumulator was measured end to end too**, in
  `benchmarks/dirty_rect_pipeline.py`: staging read plus conversion plus
  copy-out, against the current path. **12–15× faster** at the 0.8% dirty figure
  live capture produces, 6.8–7.2× at 10%, 1.8× at 50%, and **1.05–1.09× slower**
  at 100% dirty. The copy-out alone is 0.14–0.17 ms, so ~15× is the floor no
  dirty-rect scheme can beat on this path.
- The only losing case is a fully dirty frame — video, a fullscreen game — and
  it costs under 10%. A fallback to the current path when metadata is missing or
  the dirty area exceeds ~90% removes it. Output verified byte-identical to the
  current path.
- Recorded caveat: the staging-read half was measured against a ctypes buffer in
  ordinary RAM, matching `perf_suite.py`'s fixture. A real mapped staging
  surface is uncached and reads ~10× slower, and it is the component that
  shrinks with dirty rows — so this measurement probably *understates* the gain
  on real hardware.

### Measured — convert-then-transfer vs transfer-then-convert (Stage 6.1)

- **Settled in favour of transferring the frame.** New
  `benchmarks/cross_adapter_ordering.py` measures the capture-side cost of both
  orderings against real capture, across six model input sizes.
- Transferring the 8.29 MB frame costs **0.70–0.98 ms** regardless of model
  size. Converting first and transferring the tensor costs 0.52–0.59 ms at 320²,
  0.90–0.99 ms at 640², and 2.83–3.03 ms at 1280².
- So converting first wins only below 416², **640² is a tie**, and transferring
  the frame wins clearly above it. The tie breaks toward transferring the frame
  for three reasons the timings do not capture: the conversion then runs on the
  consumer's GPU (the faster one on a hybrid system), it leaves the
  display-driving iGPU free, and it keeps the model's preprocessing with the
  model's owner.
- New `native.probe_cross_adapter_buffer(size_bytes)` times a cross-adapter copy
  of an arbitrary buffer. The frame probe copies a *texture*; the tensor is a
  buffer, so deciding this needed that shape measured too.
- **A single run suggested a ~0.8 ms per-frame shared-handle overhead. It was
  noise.** `transfer()` measured 1.48 ms against a 0.66 ms raw copy in one run,
  then 0.70–0.98 ms in three more. Recorded because the false conclusion was
  attractive and would have sent someone optimising a cost that does not exist.

### Added — cross-adapter frame transfer (Stage 6.1)

- **`native.cross_adapter_transfer(frame)`** carries a captured frame from the
  capture adapter to a second GPU through a shared cross-adapter heap, and
  exposes the `ID3D12Resource` it lands in via
  `destination_resource_address`. On a hybrid laptop this is the missing link:
  Desktop Duplication only runs against the adapter driving the display, so a
  GPU-resident frame was previously stranded on the iGPU while the model lived
  on the dGPU.
- The heap, shared handle and both placed resources are allocated once in the
  constructor; only the copy is per-frame work.
- Shareability is validated at construction, not on the first frame — a texture
  that cannot reach D3D12 will never work on this path, and finding out during
  setup is far easier to act on than a failure mid-session. A test pins this.
- **The duplicated surface is live, and that is not a detail.** Rapidshot does
  not hold its keyed mutex during the copy, so two copies of "the same" frame
  observe different pixels — even when recorded into a single command list,
  because they execute in sequence on the copy engine. This showed up as ~2,100
  bytes differing in one screen region, reproducibly at the same offset. The
  verification path therefore snapshots the surface once and feeds both
  comparands from the snapshot.
- **Verified byte-exact on real capture.** `examples/verify_cross_adapter.py`
  transfers frames and compares all 8,294,400 bytes against a source-side
  readback of the same snapshot, then sanity-checks the result against an
  independent CPU capture so a self-consistent artefact cannot pass. Comparing
  against a CPU capture *directly* does not work at all: Desktop Duplication
  reports only changed content, so two consecutive frames differ by
  construction and no stable screen exists to compare with.
- Synchronisation is CPU-side: `transfer` blocks until the source GPU has
  finished. Correct, but it serialises the two adapters; a shared fence
  (`D3D12_FENCE_FLAG_SHARED | SHARED_CROSS_ADAPTER`) is the later optimisation.
- Verified Intel → WARP only, since the dev machine has no second hardware GPU.
  `destination_is_software` reports this so a caller cannot mistake it for a
  real iGPU-to-dGPU result.

### Added — `native.probe_cross_adapter()` (Stage 6.1)

- Verifies the whole cross-adapter chain rather than assuming any link:
  `CreateHeap` with `D3D12_HEAP_FLAG_SHARED | SHARED_CROSS_ADAPTER` →
  `CreateSharedHandle` → `OpenSharedHandle` on the second device →
  `CreatePlacedResource` on both sides. Each step is reported, so a "supported"
  verdict cannot come from a probe that stopped early.
- **Measured: 0.87 ms** to copy a 1080p BGRA frame (8.29 MB) into the shared
  heap, including copy-queue submission and the fence wait — roughly a third of
  the 2.27 ms it costs to read the same frame to the CPU. Moving a frame between
  adapters is cheaper than taking it off the GPU.
- Measured Intel iGPU → WARP, because the dev machine has no second hardware
  GPU. The result carries `representative: false` when the destination is a
  software adapter, so the number cannot be quoted as an iGPU→dGPU cost by
  accident. The *source* side is representative: an Optimus iGPU has no
  dedicated VRAM either, so it is the same system-memory copy.
- Uses a cross-adapter **buffer**, not a row-major texture:
  `CrossAdapterRowMajorTextureSupported` is an optional capability, and the
  buffer path works without branching on it. Both devices' capability is
  reported regardless.

### Fixed — `pip install rapidshot` shipped a broken package

- **The published wheel contained 5 modules instead of 25.** `pyproject.toml`
  declared `packages = ["rapidshot"]`, which names only the top-level package —
  `rapidshot.core`, `.processor`, `.util` and `._libs` were all omitted. A clean
  install failed on import:

  ```
  ModuleNotFoundError: No module named 'rapidshot.util'
  ```

  Replaced with `[tool.setuptools.packages.find] include = ["rapidshot*"]`.
  Verified by building a wheel, installing it into a fresh virtualenv, and
  importing **from outside the source tree** — the only way to see this, since
  from a checkout the subpackages are simply on `sys.path` and everything works.
- `setup.py` used `find_packages()` and was therefore correct, which is precisely
  why the bug survived: the two build configurations disagreed, and the wrong one
  is the one modern tooling uses.
- Added `[tool.setuptools.package-data]` so a built native extension ships
  alongside the package when present.

### Added — Continuous integration (Stage 0)

- **`.github/workflows/ci.yml`** with four jobs, split so a failure names its own
  cause:
  - **tests** — Python 3.9 / 3.11 / 3.13, **without** the native extension, and
    asserting `native.is_available()` is False. If this job ever needs Rust, the
    "optional extension" promise has been broken.
  - **native** — builds the Rust crate, runs the tests it unlocks, plus `fmt`
    and `clippy`.
  - **benchmarks** — measures the runner's noise floor with `--self-test` before
    reporting numbers, publishes JSON as an artifact, and runs the interleaved
    A/B as a *correctness* gate (it fails if the optimised conversions stop
    matching the originals bit for bit).
  - **quality** — README code blocks must parse, distributions build, `twine
    check` passes, and the built wheel is installed into a clean virtualenv and
    imported from outside the source tree. That last step is what catches the
    packaging bug above.
- **WARP fallback** for the test device: `TestTexture` and the buffer probe now
  fall back to Microsoft's software rasteriser when no GPU adapter is present, so
  the 18 shader-correctness tests run on GPU-less CI runners. WARP is slow but
  functionally complete, so correctness is genuinely checked; timings are not.

  **What CI cannot check**, stated in the workflow so results are read correctly:
  live desktop capture. GitHub runners have no desktop session, so Desktop
  Duplication has nothing to duplicate. Those tests skip themselves and must be
  run on real hardware before a release.

### Stage 6 delivered — GPU-resident capture to a DirectML-ready tensor

Rapidshot now takes a captured frame from the desktop to a **model-ready NCHW
float32 tensor on the DirectML device, without the CPU touching it**. The CPU
path for the same work measures ~8 ms per 1080p frame.

Documented in the README, with the ONNX Runtime call sequence a caller needs.

**Scope decision: the ORT session binding is deliberately not included.** Every
route to `OrtDmlApi` costs something permanent — vendoring ~5000 lines of header
plus hand-counted struct offsets, adding libclang via bindgen, or requiring ONNX
Runtime built from source for the `ort` crate. All of them couple Rapidshot's
core to ONNX Runtime's ABI and release cadence: a standing maintenance cost paid
by every user for one optional feature. The capture library's job is to produce
the frame in the right place and format. `output_resource_address` is the
documented contract; binding it is roughly fifteen lines on the consumer's side.

If demand appears, a separate `rapidshot-directml` package or a Python-side
`ctypes` binding both preserve that boundary — neither needs deciding now.

### Added — ONNX Runtime reachability probe

- **`native.probe_onnxruntime(dll_path)`** loads `onnxruntime.dll` at runtime
  (rather than linking against it, keeping ORT optional) and reports its version
  and supported C API versions. Confirms ORT 1.24.4 with C API versions 1–24 is
  reachable from the native shim.
- **`native.onnxruntime_dll_path()`** locates the DLL that ships with the
  installed Python package. Use it — resolving by name goes through the DLL
  search path, which on this machine finds an unrelated **1.17.1** installed by
  another application while the Python package ships **1.24.4**. The ONNX Runtime
  C API's struct layout is version-dependent, so silently binding to the wrong
  runtime is a genuine hazard rather than a cosmetic mismatch.
- Only `OrtApiBase` is traversed, which has exactly two members and a layout
  fixed by contract. The much larger `OrtApi` struct is deliberately left alone —
  see the roadmap for the open decision on reaching `GetExecutionProviderApi`
  safely, which needs the header the Python package does not ship.

### Added — D3D12 conversion shader: the tensor now lands on the DirectML device

- **`native.GpuPreprocessor12`** — the same BGRA -> NCHW float32 conversion,
  running on **D3D12** instead of D3D11. This is the step that makes the tensor
  reachable by DirectML at all: D3D11 can share only 2D textures, never buffers,
  so a D3D11-written tensor had no route to a D3D12 device. Running the shader on
  D3D12 removes the problem rather than working around it — the captured
  *texture* does share, and the output buffer is then already resident where
  DirectML binds.
- Verified against **real captured frames**: output is non-empty (76.9% non-zero)
  and **matches the exhaustively-unit-tested D3D11 path exactly** (max difference
  0.00e+00), with 411 sustained dispatches and no errors. `output_gpu_address`
  and `output_resource_address` expose exactly what
  `OrtDmlApi::CreateGPUAllocationFromD3DResource` will consume.
- Uses root constants rather than a constant buffer, removing a resource and an
  upload per frame. Command allocator, list, fence, PSO, root signature and
  descriptor heap are built once and reused.
- Shareability is validated **at construction**, not on the first dispatch, so an
  unusable texture is reported immediately with a message naming the missing flag.

  **A trap worth recording:** the synthetic `TestTexture` fixture cannot exercise
  this path. D3D11 refuses `SHARED_NTHANDLE` unless paired with
  `SHARED_KEYEDMUTEX`, and a keyed-mutex resource reads as zeros until its mutex
  is acquired — on *both* APIs. That briefly looked like a D3D12 porting bug; the
  giveaway was the already-verified D3D11 path failing on the same texture. The
  real duplicated desktop surface has its mutex managed by DXGI and needs no
  explicit acquire, so D3D12 correctness is established against live capture plus
  exact agreement with the D3D11 path, and the synthetic tests stay on D3D11.

### Added — Buffer-sharing probe, and the finding that redirected milestone 3b

- **`native.probe_shareable_buffers()`** — tests six D3D11 buffer configurations
  (structured / raw / plain, each with NT-handle and legacy sharing) for whether
  a compute shader can write them *and* D3D12 can open them.

  **Result: none of them.** Every NT-handle variant fails at creation with
  `E_INVALIDARG`; every legacy variant creates but cannot produce a shared
  handle. This is documented behaviour rather than a flag mistake —
  [only 2D non-mipmapped textures can be shared in D3D11](https://learn.microsoft.com/en-us/windows/win32/api/d3d11/nf-d3d11-id3d11device-opensharedresource).

  So milestone 3a's assumption — write the tensor in D3D11, share it to D3D12
  for DirectML — is impossible, and so is the reverse direction. **The
  conversion shader has to run on the D3D12 device instead**, which milestone 2
  already showed the captured texture can reach. That removes the sharing step
  entirely rather than working around it.

  Found by a 15-minute probe rather than partway through the ONNX Runtime FFI
  work, which is the entire argument for building this in milestones.

### Added — GPU preprocessing shader, milestone 3a (Stage 6 + 6b)

- **`native.GpuPreprocessor`** — a D3D11 compute shader converting the captured
  BGRA8 texture into a linear **NCHW float32** tensor, with resize and
  normalisation folded into the same pass. This is what DirectML consumes, and
  it collapses three CPU stages (staging read, color conversion, resize/
  normalise/transpose) into one dispatch that never leaves the GPU.
- Because the shader already touches every pixel, folding the resize and
  normalisation in costs essentially nothing — **Stage 6b is therefore delivered
  as part of Stage 6 rather than as separate work.**
- Options: `scale`/`bias` for arbitrary normalisation ranges (default 0..1, pass
  `scale=2, bias=-1` for -1..1) and `bgr` for channel order. Buffers are
  allocated once and reused, so the per-frame path allocates nothing.

  **A silent correctness bug, caught only because the output was checked against
  a reference:** the shader originally reversed the channels by hand, assuming a
  BGRA texture presents blue in `.x`. It does not — the hardware swizzles
  `DXGI_FORMAT_B8G8R8A8_UNORM` so `.x` is *red*. The result was a tensor labelled
  RGB that actually contained BGR. Nothing about its speed, stability or shape
  would have revealed this; it would simply have corrupted every model's input.
  Diagnosis: 46% of pixels matched exactly, which turned out to be precisely the
  greyscale fraction of the screen (where R==G==B), and per-channel correlation
  then showed plane 0 matching blue at 99.99%.

- **18 deterministic correctness tests** (`tests/test_gpu_preprocess.py`) using a
  new `TestTexture` with contents the test chooses, comparing against a NumPy
  reference to within 1e-5. Live capture was tried first and rejected as a
  verification basis: Desktop Duplication only reports *changed* content, so an
  idle screen produces no frames at all, and consecutive captures are not
  guaranteed identical.
- Benchmarks `pipeline.cpu_to_nchw`, `pipeline.gpu_dispatch` and
  `pipeline.gpu_plus_readback` added to the suite, using a synthetic texture so
  they are deterministic and need no desktop session.

  **Reported honestly:** dispatch submission is ~1.4 us, but the GPU executes
  asynchronously, so that is what the calling thread pays — not total work.
  Forcing a readback costs *more* than the CPU arm. That is the real lesson and
  it is printed alongside the numbers: this path wins only when the tensor is
  consumed on the GPU. Pulling it back to the CPU gives up the entire advantage.

### Changed — GRAY conversion, rounding folded

- The `+128` round-to-nearest term is now folded into the first channel's
  accumulation instead of running as its own `luma += 128` pass, avoiding a full
  read-modify-write over a 4 MB intermediate. **Bit-identical output, 1.20x
  faster**; GRAY's overall speedup against the original implementation rises
  from 1.48x to **1.85x**. Found by re-running the interleaved A/B after the
  rounding term was added and noticing the speedup had dropped.

### Added — Native GPU interop, milestone 2: D3D12 sharing confirmed (Stage 6)

- **`native.probe_d3d12_sharing(frame)`** — and it answers the question Stage 6
  hinged on. The full chain succeeds on real capture:

  | Step | Result |
  | --- | --- |
  | `QueryInterface(IDXGIResource1)` | OK |
  | `CreateSharedHandle` | OK |
  | Recover capture adapter | OK |
  | `D3D12CreateDevice` on that adapter | OK |
  | **`ID3D12Device::OpenSharedHandle`** | **OK — D3D12 sees 1920x1080, B8G8R8A8** |

  **The duplicated desktop surface opens directly on a D3D12 device.** DirectML
  runs on D3D12, so this is the precondition for zero-copy inference — and it is
  met without an intermediate copy. `RESTRICT_SHARED_RESOURCE`, which was the
  main risk to this approach, does not block a cross-API open in-process.
  Milestone 3 can bind the captured surface itself rather than a copy of it.
- Each step of the probe reports separately, with an `interpretation` naming the
  fallback, so a failure on other hardware identifies which link broke rather
  than just that the chain did.
- The shared NT handle is closed on every path, including early returns —
  leaking one per probe would exhaust the process handle table.
- Verified stable across 30 repeated probes with capture healthy afterwards.

### Changed — Benchmark harness trustworthiness

- **Live benchmarks no longer gate a change.** They depend on what is happening
  on screen, which is not a controlled input: three consecutive runs of identical
  code produced minimums of 0.346, 0.174 and 0.141 ms — a 2.5x swing. They are
  still reported, marked `(live: informational)`.
- **Sub-millisecond benchmarks no longer gate either.** Below ~0.5 ms the OS
  scheduler's granularity dominates the code under test, and drift normalisation
  amplifies it; these swing ~1.15x between back-to-back runs of identical code
  while millisecond-scale benchmarks hold within 1.02x. Marked
  `(sub-ms: informational)`.
- Both changes exist so the suite does not cry wolf. A harness that reports
  regressions that are not real gets ignored, which is worse than having none.

### Added — Native GPU interop shim, milestone 1 (Stage 6)

- **`native/` — a Rust + PyO3 extension crate**, deliberately scoped to the one
  thing Python cannot do. Profiling put the Python/COM capture loop at
  ~0.003 ms/frame, so it stays in Python; the extension exists solely because
  ONNX Runtime's `CreateGPUAllocationFromD3DResource` has no Python binding.
- **The extension is optional.** `pip install rapidshot` needs no Rust toolchain,
  and everything except GPU-tensor interop is unaffected. `rapidshot.native`
  reports availability rather than raising on import, and `require()` explains
  how to build it instead of surfacing a bare `ImportError`.
- **Milestone 1 proves the plumbing end to end**: a comtypes `ID3D11Texture2D`
  from `frame.d3d11_texture` crosses the FFI boundary and D3D11 COM calls
  succeed from Rust. Verified against live capture — Rust read back the correct
  1920x1080 / `B8G8R8A8_UNORM` description, and **401 FFI crossings in 4 s with
  zero errors** confirmed the borrow does not disturb COM reference counts.
- API: `native.describe_texture(frame)`, `native.texture_sharing_info(frame)`,
  `native.device_address(frame)`, `native.build_info()`. All take a live `Frame`
  and refuse a released one, so a dangling pointer can never reach native code.
- `native/install_dev.py` copies the built artifact into the package and
  verifies it imports, rather than reporting success on a copy Python cannot load.
- Built with `abi3-py39`, so one wheel will cover every supported Python version.

  **Finding that shapes milestone 2:** the duplicated desktop surface already
  carries `SHARED_NTHANDLE | SHARED_KEYEDMUTEX | RESTRICT_SHARED_RESOURCE`
  (`misc_flags = 0x2900`). It is therefore *already shareable*, which was not
  assumed — the interop may be able to open it on the DirectML device via its NT
  handle instead of staging through an intermediate shared resource.
  `RESTRICT_SHARED_RESOURCE` narrows who may open that handle, so this needs
  verifying rather than relying on; it is a promising lead, not a settled design.

### Added — Frame object with explicit GPU texture lifetime (Stage 3, first slice)

- **`ScreenCapture.grab_frame(region=None)`** returns a
  `rapidshot.frame.Frame` holding the `ID3D11Texture2D` DXGI produced, with **no
  staging read and no color conversion**. Measured against the CPU path on real
  capture at 1920x1080:

  | Path | Per frame | Ceiling |
  | --- | --- | --- |
  | `grab()` — CPU staging read + convert | 4.82 ms | ~208 FPS |
  | `grab_frame()` — texture stays on GPU | 0.17 ms | ~5879 FPS |

  **28x faster, 4.65 ms/frame saved.** That gap is the CPU round-trip Stage 6
  exists to eliminate — and it is already available today to consumers that want
  the frame on the GPU (inference runtimes, hardware encoders).

- **The texture lifetime is explicit, because it has to be.** DXGI refuses the
  next `AcquireNextFrame` with `DXGI_ERROR_INVALID_CALL` while any reference to
  the previous desktop surface is outstanding — this stalls capture completely
  rather than degrading. `Frame` is therefore a context manager:

  ```python
  with camera.grab_frame() as frame:
      run_model(frame.d3d11_texture)   # valid only inside the block
  ```

  - Touching `d3d11_texture` after release raises `FrameReleasedError` naming the
    cause, instead of letting DXGI fail opaquely on a later call.
  - `grab()`, `shot()` and `grab_frame()` all refuse to start while a Frame is
    outstanding, with a message that says what to do about it.
  - Garbage collection releases as a safety net and logs a warning; by then
    capture has already been blocked for an unbounded period.
  - `release()` is idempotent, and runs on exception via the context manager.

- **Frame metadata**, readable before and after release: `timestamp_qpc` /
  `timestamp` (the compositor's `LastPresentTime`, so it measures capture latency
  rather than call time), `accumulated_frames` (>1 means the OS dropped frames
  because the consumer fell behind), `protected_content`, `cursor_visible`,
  `region`, `width`, `height`, `rotation_angle`.

- `Duplicator` now records `last_present_time` and `accumulated_frames` from
  `DXGI_OUTDUPL_FRAME_INFO`; both were computed and discarded before.

- 13 tests covering the lifetime contract, plus 14 live checks including a
  sustained loop (401 frames, 0 errors) that would have stalled under the
  pre-Stage-1 texture handling.

### Added — Performance measurement (Stage 0)

- **`benchmarks/perf_suite.py`** — reproducible performance suite with JSON
  output and before/after comparison (`--compare baseline.json`). Every change
  from here on is measured rather than asserted.
  - **Minimum-sample comparison.** Background load can only make a benchmark
    slower, never faster, so the minimum is the least-contaminated statistic.
  - **`control.memcopy`** — a benchmark whose implementation never changes. Any
    movement in it between runs is machine drift, which is divided back out of
    every other comparison.
  - **`--rounds N`** pools samples across repeated runs of the whole suite, so a
    benchmark only needs one quiet moment in the session rather than one quiet
    run. This mattered: the first methodology reported **11 false regressions of
    up to 1.9x when comparing identical code to itself** on this machine (55%
    background CPU). With pooled rounds the false-positive count is **0** at the
    1.30x threshold.
  - **`--self-test`** measures that noise floor on demand, so the significance
    threshold can be justified rather than guessed.
- **`benchmarks/ab_conversion.py`** — interleaved A/B harness holding the old and
  new implementations in one process and alternating between them, so machine
  drift hits both arms equally. This is the measurement of record for the
  conversion speedups below, and it verifies output equivalence alongside timing.
- `benchmarks/baseline.json` — reference numbers for future comparison.

### Fixed — Frame aliasing (data corruption)

- **`grab()` returned a view into a buffer it had already recycled.** For every
  color mode except BGRA, the converter returned a lazy NumPy slice
  (`src[..., 2::-1]`) rather than a materialised array. `process()` then reported
  `is_still_pooled_buffer=False`, which makes `_grab()` check that pooled buffer
  straight back into the pool — while the caller was still holding a view into
  it. **The next capture silently rewrote the caller's frame in place.**
  Confirmed with a direct reproduction: a frame captured as all-10 pixels became
  all-200 after the following capture. `process()` now materialises into a
  freshly owned, C-contiguous array whenever the result is not the pooled buffer.
- The same aliasing applied to rotated output: `np.rot90` returns a view, which
  was returned directly. It is now materialised with `np.ascontiguousarray`.
- This bug is also why `process.RGB` previously benchmarked at 0.23 ms while the
  conversion alone cost 4 ms — the work was deferred to whoever read the array.
  The honest figure is ~3 ms; the old number measured nothing.

### Changed — Pixel conversion performance

Measured by interleaved A/B at 1920x1080, outputs verified identical:

| Mode | Before | After | Speedup |
| --- | --- | --- | --- |
| RGBA | 9.02 ms | 2.72 ms | **3.31x** |
| BGR | 4.11 ms | 1.61 ms | **2.55x** |
| RGB | 4.15 ms | 1.83 ms | **2.26x** |
| GRAY | 17.28 ms | 11.67 ms | **1.48x** |

- **RGB/BGR/RGBA**: replaced reversed-stride and fancy-index gathers
  (`src[..., 2::-1]`, `src[..., [2,1,0,3]]`) with per-channel contiguous copies.
- **GRAY**: moved the Rec. 601 luma from Q14/uint32 to Q8/uint16 fixed point
  (`(R*77 + G*150 + B*29 + 128) >> 8`), halving memory traffic. The whole
  intermediate stays in uint16 — the maximum is 65408, just inside the limit. Max
  deviation is one level, the same approximation OpenCV uses for 8-bit input, and
  the added rounding term removes the darkening bias that plain truncation causes.
- GRAY remains by far the slowest mode (~11.7 ms/frame). Still the strided-gather
  pattern; a candidate for a SIMD kernel if it ever matters.

### Added — Color pipeline correctness

- **`ScreenCapture.channels`** and **`ScreenCapture.bytes_per_frame(region)`**,
  so callers can size a `shot()` destination buffer correctly instead of
  guessing at the pixel format.
- **`shot()` accepts sized buffer objects.** NumPy arrays, `ctypes` arrays,
  `bytearray` and `memoryview` all report their own length, which is validated
  before any write. Raw pointers carry no size and are now only accepted
  together with the new `buffer_size` argument.
- `processor.base.COLOR_MODE_CHANNELS`, `validate_color_mode()` and
  `channels_for_color_mode()` as the single source of truth for how many
  channels each output mode produces.
- `util.ctypes_helpers.describe_destination()` resolves any supported
  destination object to `(address, size_in_bytes)`.
- Test suite for the color pipeline (`tests/test_color_modes.py`, 29 tests),
  including a sentinel-guard test that fails if `shot()` writes a single byte
  past the destination.

### Fixed — Color pipeline correctness

- **`shot()` overran the caller's buffer and crashed the process.** It always
  wrote `width * height * 4` bytes of raw BGRA regardless of the instance's
  `output_color` and regardless of the real buffer size. A capture created with
  `output_color="RGB"` writing into a correctly-sized 3-channel buffer overran
  it by a third of a frame — an access violation (`0xC0000005`), with no bounds
  check anywhere on the path. `shot()` now writes in the configured color mode
  and validates the destination size *before* capturing.
- **The size check runs up front, not opportunistically.** Validating inside the
  processor would only have caught bad buffers on calls that happened to receive
  new frame content, so on a static desktop an undersized buffer returned
  `False` for a while and only raised once something on screen changed.
  `ScreenCapture.shot()` validates before doing any capture work, so the error is
  deterministic.
- **`output_color="GRAY"` did not convert.** The GRAY branch routed through
  OpenCV, an optional dependency; when `cv2` was missing the `ImportError` was
  swallowed and the converter was replaced with an identity function, so callers
  silently received unconverted `(H, W, 4)` BGRA frames labelled as grayscale.
  GRAY is now implemented in pure NumPy using Rec. 601 luma in Q14 fixed point
  (matching OpenCV's `COLOR_BGRA2GRAY` to within one level) and returns
  `(H, W, 1)`. No conversion path depends on OpenCV any more.
- **`output_color="RGBA"` returned BGRA data.** The converter was
  `lambda img: img.copy()` on the premise that "OpenCV's BGRA2RGBA also just
  copies" — it does not; it swaps red and blue. RGBA output had its red and blue
  channels transposed.
- **Unsupported color modes failed silently.** An unrecognised `output_color`
  fell through to an identity converter deep in the pipeline and produced
  unconverted BGRA. `Processor` now validates at construction and raises
  `ValueError` listing the supported modes.
- `shot()` used `self.shot_w`/`self.shot_h` for the copy dimensions, which are
  only refreshed when a region is passed explicitly and are in screen space
  rather than the surface's memory space. It now uses the mapped region's own
  dimensions, so a rotated display no longer reads with the wrong row stride.
- `NumpyProcessor.shot()` swallowed every exception and logged it, while
  `_shot()` returned `True` regardless — a failed capture reported success.
  Errors now propagate.
- `shot()` validates that the mapped surface pitch is at least one full row
  wide before reading, rather than trusting it.

### Changed — Color pipeline correctness

- `Processor.process2()` takes an optional `buffer_size` and returns the
  backend's success flag instead of discarding it.
- README documents the channel count of each color mode and the `shot()` buffer
  contract.
- Known gap: the CuPy backend still routes RGBA/GRAY through OpenCV (with a
  device-to-host copy). Unlike the NumPy path it raises a clear `ImportError`
  when `cv2` is missing rather than failing silently, and it has no `shot()`
  implementation — `process2()` raises `NotImplementedError` there.

### Added — Stage 1: DXGI engine correctness

- **`IDXGIOutput5.DuplicateOutput1` is now the default duplication path.** The
  duplicator queries `IDXGIOutput5` and calls `DuplicateOutput1` with an explicit
  supported-format list (`B8G8R8A8_UNORM`, `R8G8B8A8_UNORM`, `R10G10B10A2_UNORM`,
  `R16G16B16A16_FLOAT`), which is what allows HDR and 10-bit desktops to be
  duplicated instead of failing. Falls back automatically to the legacy
  `IDXGIOutput1.DuplicateOutput` when `IDXGIOutput5` is unavailable.
- **`RAPIDSHOT_DUPLICATE_OUTPUT` environment variable.** Set it to `legacy` (or
  `0`, or `duplicateoutput`) to force the pre-1.5 `DuplicateOutput` path when a
  driver misbehaves on `DuplicateOutput1`.
- **Protected-content (HDCP/DRM) handling.** New
  `RapidShotProtectedContentError` is raised when duplication is denied because
  protected content is on screen, instead of surfacing an opaque COM error. The
  per-frame `ProtectedContentMaskedOut` flag is also read and logged once, so a
  blanked region is distinguishable from a genuinely black frame. Because this is
  an OS refusal rather than a transient fault, it deliberately does *not* trigger
  the re-initialization retry loop.
- `IDXGIOutput2` through `IDXGIOutput5` interface definitions.
- `Duplicator.used_duplicate_output1` and `Duplicator.protected_content_detected`
  attributes for callers and tests that need to know which path is active.

### Fixed — Stage 1: DXGI engine correctness

- **The package could not be imported at all.** `rapidshot/core/duplicator.py`
  imported `DXGI_ERROR_DEVICE_REMOVED`, `DXGI_ERROR_DEVICE_RESET`,
  `DXGI_ERROR_INVALID_CALL`, `DXGI_ERROR_UNSUPPORTED` and `ID3D11Texture2D` from
  `rapidshot._libs.dxgi`, where none of them were defined, and referenced the
  `logging` module without importing it. Any `import rapidshot` that reached the
  capture path raised `ImportError`.
- **Every DXGI error comparison silently failed.** HRESULT constants were written
  as unsigned literals (`0x887A0026`) while `comtypes` reports
  `COMError.args[0]` as a *signed* 32-bit int (`-2005270522`), so no constant
  ever compared equal to a real error code. Access-lost detection, device-loss
  detection, and timeout detection were all dead code. All constants now go
  through a `_hresult()` normalizer and are grouped into
  `DXGI_RECOVERABLE_ERRORS`, `DXGI_DEVICE_ERRORS`, and
  `DXGI_PROTECTED_CONTENT_ERRORS`.
- **Capture stalled permanently after the second frame.** The duplicator kept the
  previous frame's `ID3D11Texture2D` reference alive across acquisitions. DXGI
  refuses `AcquireNextFrame` with `DXGI_ERROR_INVALID_CALL` while any reference
  to the prior desktop surface is outstanding, so the third and every subsequent
  grab failed. The stale reference is now dropped before the next acquire and in
  `release_frame()`.
- **Double-release of the acquired desktop resource.** `update_frame()` called
  `res.Release()` by hand on a `comtypes` COM pointer, which already releases on
  scope exit — corrupting the surface's reference count. The manual call is gone.
- **The error-reporting paths crashed with `ValueError`.** Four f-strings used
  `{hresult:#010x if isinstance(hresult, int) else hresult}`, which Python parses
  as an invalid *format specifier*, not a conditional. Any DXGI error that
  reached these lines raised `ValueError` on top of the original failure. Replaced
  with a `_format_hresult()` helper.
- **`ScreenCapture` never assigned `self._output` / `self._device`,** so
  `_initialize_resources()` failed with `AttributeError` and no instance could
  ever be constructed.
- **A 10 ms acquire timeout was treated as a display-mode change.** `shot()`
  triggered a full `_on_output_change()` rebuild whenever no new frame arrived
  within the acquire timeout — the normal state of a static desktop. It now
  distinguishes "duplication is healthy but there is no new content" from
  "duplication is broken".
- **Exclusive-fullscreen transitions could hang the caller forever.**
  `_on_output_change()` retried duplication creation in an unbounded
  `while True` loop with no backoff, spinning the CPU until the mode switch
  settled or hanging indefinitely if it never did. It now retries with
  exponential backoff up to a bounded budget and returns a success flag. The
  stale stage surface — still sized for the pre-switch resolution, which is what
  produced the black-screen-in-fullscreen symptom — is now released before the
  rebuild rather than being reused.
- **Continuous mode never delivered a single frame.** `capture.py` referenced
  `PooledBuffer` without importing it, so the capture thread raised `NameError`
  on its first frame and marked capture permanently failed. Additionally, the
  deque only accepted `PooledBuffer` instances, but `_grab()` correctly returns a
  plain array whenever color conversion changes the channel count — i.e. for
  every mode except `BGRA`. Both are fixed; the deque now holds either, and
  `get_latest_frame()`, `stop()`, and the `video_mode` duplication path handle
  both shapes.
- **`_rebuild_frame_buffer()` was dead code** referencing four attributes that do
  not exist on the class (`self.__lock`, `self.channel_size`,
  `self.__frame_buffer`, `self.__head`). It now actually drains queued buffers
  and resizes the memory pool for the new resolution.
- Access-lost, session-disconnect (`DXGI_ERROR_SESSION_DISCONNECTED`, i.e. RDP
  and fast user switching) and mode-change-in-progress failures now release the
  invalidated duplication interface before raising, so no further calls are
  issued against a dead interface.
- `release_frame()` clears its acquired-frame flag in a `finally` block, so a
  failed `ReleaseFrame` no longer leaves the duplicator permanently convinced a
  frame is outstanding.
- `release()` drops any still-held frame before releasing the duplication object,
  so DXGI no longer keeps the desktop surface pinned afterwards.
- `Duplicator`'s `cursor` and `texture` fields used mutable dataclass defaults,
  which are shared across *all* instances. Replaced with `default_factory`.

### Changed — Stage 1: DXGI engine correctness

- **Reduced lock hold time in continuous mode.** The capture thread no longer
  checks evicted buffers back into the memory pool while holding
  `_capture_lock`. `PooledBuffer.release()` takes the pool's own lock, so the
  previous nesting stalled every `get_latest_frame()` consumer for the duration
  of a pool round-trip. The same change applies to `stop()` and to the
  re-initialization path.
- `Duplicator.update_frame()` now has a documented `bool` return meaning
  "duplication is still healthy" (a timeout counts as healthy). Callers must read
  `.updated` to learn whether a frame is actually present. Previously it returned
  `True`/`None` inconsistently, which is what made the timeout path look like a
  fatal error to `shot()`.
- `RapidShotConfigError` accepts an `hresult` argument, matching the other
  DXGI error types.
- `ScreenCapture.shot()` documents that it always writes raw **BGRA** regardless
  of `output_color`, and that an undersized buffer is overrun without a bounds
  check. *(Superseded within this same Unreleased block — `shot()` now honors
  `output_color` and validates the destination; see "Color pipeline
  correctness" above.)*

### Notes

- Region-aware `CopySubresourceRegion` — listed under Stage 1 in the roadmap —
  was already implemented in `_grab()` and `_shot()`; verified working rather
  than rewritten.
- The gap noted here during Stage 1 — `output_color="GRAY"` returning
  unconverted 4-channel BGRA — has since been fixed; see "Color pipeline
  correctness" above.

---

## [1.1.0]

Baseline release. This changelog was empty before Stage 1; earlier history is not
reconstructed here.
