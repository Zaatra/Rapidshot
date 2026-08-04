# Rapidshot Roadmap

**Goal:** be the capture layer the AI-agent/CV ecosystem reaches for on Windows — not a faster copy of DXcam.

This document is written to be read cold. It states where the project actually is, what to do next, and which questions are already settled so they are not re-litigated. Everything marked ✅ has been implemented and verified; see `CHANGELOG.md` for detail.

---

## 1. Start here

**Current state.** Rapidshot captures the desktop via DXGI Desktop Duplication and can hand a frame to a GPU consumer as a **model-ready NCHW float32 tensor that never touches the CPU**. The CPU path for the same work costs ~8 ms per 1080p frame. Core capture is pure Python; an *optional* Rust extension provides GPU interop only.

### Release status — read this before anything else

**2.0.0 is finished and verified, but not yet published.** The version on PyPI is 1.1.0 from April 2025, and it **does not work**: it fails to import on Python 3.11+ (`cursor: Cursor = Cursor()` trips the dataclass mutable-default check broadened in 3.11), and patching that one line only gets it to return all-black frames, because the processor is handed a texture where it expects a mapped staging surface. Every current-Python user is broken today. Shipping 2.0.0 is therefore the highest-value action available, ahead of any feature work below.

What remains, in order — the full procedure is in `RELEASING.md`:

1. `git push origin main` (commits may be outstanding; check `git log origin/main..HEAD`)
2. Repo settings that files alone cannot enable: **Settings → Security → private vulnerability reporting** (or the link in `SECURITY.md` 404s), and a branch protection rule with *Require review from Code Owners* (or `CODEOWNERS` is only a routing hint)
3. `git tag -a v2.0.0 -m "RapidShot 2.0.0"` and push the tag — that triggers `release.yml`, which builds, runs four wheel gates, publishes to PyPI via Trusted Publishing, then creates the GitHub Release
4. Approve the `pypi` deployment when the workflow pauses for it
5. Afterwards, consider **yanking 1.1.0** so nobody new lands on a version that returns black frames

PyPI Trusted Publishing and the `pypi` GitHub environment are already configured, restricted to `v*` tags.

**Next feature task once released:** § 6.3 — finish Stage 3 (Frame metadata). § 6.1 is complete: hybrid and headless systems are reported clearly, a captured frame crosses to a second adapter at **0.70–0.98 ms per 1080p frame** verified byte-exact, and the convert-first-or-transfer-first question has been measured and settled in favour of transferring the frame. What § 6.1 still lacks is validation on real hybrid hardware and an asynchronous shared fence, both noted there.

**Before changing anything performance-related**, read § 3 (measured baseline) and § 4 (settled questions). Several intuitive-sounding optimisations have already been measured and rejected.

---

## 2. Working on this project

### Environment

- **Windows only.** DXGI Desktop Duplication has no cross-platform equivalent.
- **Two Python interpreters may be on PATH.** Check `sys.executable` before concluding a dependency is missing. Dev dependencies (`comtypes`, `numpy`, `pytest`) live in the 3.13 install.
- **Native extension is optional.** `pip install rapidshot` never needs a toolchain. Building it needs Rust plus the MSVC C++ build tools, and `cargo` is often not on a fresh shell's PATH — prefix with `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.
- Dev machine for the numbers below: Intel iGPU, 2× 1920×1080. **No NVIDIA GPU**, so CUDA/CuPy paths are untested locally.

### Commands

```bash
python -m pytest tests/ -q
```

```bash
(cd native && cargo build --release) && python native/install_dev.py
```

```bash
python benchmarks/perf_suite.py --rounds 5 --reps 25 --out after.json --compare benchmarks/baseline.json
```

```bash
python benchmarks/ab_conversion.py
```

### Testing gotchas that will waste your time otherwise

- **Live capture tests need screen activity.** Desktop Duplication only reports *changed* content, so an idle screen produces zero frames and tests fail for reasons unrelated to the code.
- **Synthetic textures cannot test the D3D12 path.** D3D11 refuses `SHARED_NTHANDLE` without `SHARED_KEYEDMUTEX`, and a keyed-mutex resource reads as zeros until acquired — on *both* APIs. The real duplicated surface has its mutex managed by DXGI. Use live capture.
- **Benchmark noise is severe on a loaded machine.** Naive comparison once reported 11 false regressions up to 1.9× on *identical* code. The suite compensates with pooled rounds, minimum-sample comparison, and a control benchmark; run `--self-test` to measure the current noise floor before trusting any result.
- **Pace benchmarks to a frame period, never to a fixed gap or a burn loop.** Sustained heavy vector work holds the CPU in a lower power state, and GRAY has two modes because of it — **16.27 ms back-to-back, 9.16 ms with a 16 ms gap, 9.91 ms in bursts with a 200 ms gap.** The trap is that *both* extremes are wrong. A benchmark's real duty cycle follows from its own cost: RGB takes 1.8 ms of a 16.7 ms frame (~11% duty cycle, mostly idle) while GRAY takes 15.9 ms (~95%, effectively sustained). `perf_suite.py` therefore sleeps out the remainder of a 60 Hz frame after each rep, which reproduces both from one rule; a fixed gap handed GRAY a 50% duty cycle and reported a number no capture loop achieves. The memcpy control cannot catch any of this — memcpy is not heavy enough to trigger it, so it reports "machine state comparable" throughout.
- **Re-record `baseline.json` whenever the harness changes how it drives benchmarks**, and verify immediately with a second run that should read all `~ same`. Numbers from different pacing models are not comparable, and after the fact you cannot separate a harness change from machine drift.
- **CI cannot verify live capture.** GitHub runners have no desktop session. Those tests skip themselves and must be run on real hardware before a release.

---

## 3. Measured baseline

All figures 1920×1080 BGRA (8.3 MB/frame), measured on the dev machine. Stored in `benchmarks/baseline.json`, **re-recorded 2026-07-30** after the harness was corrected to pace reps to a frame period (§ 2). The previous recording is kept as `benchmarks/baseline-2026-07-27.json`; do not compare across the two, because they drove the benchmarks differently.

| Stage of the loop | Cost/frame |
| --- | --- |
| Python → COM binding overhead (~6 calls) | **0.003 ms** |
| `CopySubresourceRegion` (GPU) | 0.016 ms |
| Read from mapped staging surface | 2.27 ms |
| Pixel conversion, RGB/BGR (NumPy, post-optimisation) | 1.76–1.80 ms |
| Pixel conversion, GRAY | 13.7–14.9 ms — see § 10 |
| Preprocess for a model (resize/normalise/CHW → 640×640) | 6.23 ms |
| **CPU total, capture → model input** | **~8 ms** |

Capture path comparison, real capture:

| Path | Per frame |
| --- | --- |
| `grab()` — CPU staging read + convert | 4.53 ms |
| `grab_frame()` — texture stays on GPU | **0.21 ms** |

Absolute figures rose against the 2026-07-27 recording (RGB 1.45 → 1.76 ms, GRAY 9.16 → 13.72 ms). Part of that is the harness correction, which stopped flattering GRAY in particular; part may be machine state three days apart. The two causes cannot be separated after the fact, which is the argument for re-recording a baseline whenever the harness changes rather than carrying one across.

Cross-adapter transfer, 1080p BGRA (8.29 MB) into a `SHARED_CROSS_ADAPTER` heap, copy-queue submission and fence wait included:

| | Per frame |
| --- | --- |
| Copy into the shared heap | **0.87 ms** min, 0.94 ms median (~9 GB/s) |

Measured Intel iGPU → WARP, since the dev machine has no second hardware GPU. The **source** side is representative — on an Optimus laptop capture also runs on an iGPU with no dedicated VRAM, so it is the same system-memory copy. What the *consumer* adapter pays to read the heap is its own device's cost and is not measured here. Reproduce with `native.probe_cross_adapter()`.

For scale: this is about a third of what reading the same frame to the CPU costs (2.27 ms staging read), so moving a frame between adapters is cheaper than leaving the GPU.

---

## 4. Settled questions — do not re-litigate

**Python is not the bottleneck.** CPU pixel work is ~2,500× the entire Python/COM binding overhead. At 240 FPS the binding cost is 0.07% of the frame budget.

**Capture rate is hardware-bounded.** Desktop Duplication returns at most one frame per display refresh. DXcam already achieves 239 FPS on a 240 Hz monitor *in pure Python with ctypes* — essentially 100% of the physical ceiling. No language change moves this.

**A native capture core is not worth building.** It addresses 0.003 ms of an 8 ms frame, and `windows-capture` already ships exactly that (Rust + PyO3, DXGI + WGC) — so it is neither a differentiator nor a measurable win. See § 8.

**Published FPS claims in this space are mutually contradictory.** DXcam's README reports DXcam at 239 FPS; BetterCam's reports DXcam at 39 and itself at 123; other sources quote BetterCam near 290. Different hardware, no shared harness. Plan against `benchmarks/`, not anyone's marketing.

**D3D11 cannot share buffers — only 2D non-mipmapped textures.** Six configurations were probed (structured/raw/plain × NT-handle/legacy); none produced a buffer D3D12 could open. This is why the conversion shader runs on D3D12. `native.probe_shareable_buffers()` re-checks it; a regression test fails if this ever changes.

**HLSL presents BGRA textures semantically.** For `DXGI_FORMAT_B8G8R8A8_UNORM` the hardware swizzles so `.x` is **red**, despite blue being first in memory. Reversing channels by hand produces BGR labelled RGB — silently wrong model input that no test of speed or shape catches.

**A software adapter is not a second GPU.** The Microsoft Basic Render Driver (WARP) reports zero outputs, exactly like the dGPU on an Optimus laptop, so the obvious "adapter with no outputs = discrete GPU" check calls every ordinary desktop a hybrid system. `DXGI_ADAPTER_FLAG_SOFTWARE` is what separates them. The flip side is useful: WARP is a real second D3D12 device, which is what makes the cross-adapter path testable on a single-GPU machine at all.

**ONNX Runtime's `OrtDmlApi` has no Python binding.** `CreateGPUAllocationFromD3DResource` is reachable only from C/C++. Also: `IOBinding.bind_input` performs **no pointer validation** — it accepts `0xdeadbeef` without complaint, so "bind succeeded" proves nothing.

---

## 5. Completed

| Stage | Delivered |
| --- | --- |
| **0 — Infrastructure** ✅ | `benchmarks/perf_suite.py` (JSON, `--compare`, drift-calibrated, noise-floor self-test), `benchmarks/ab_conversion.py` (interleaved A/B), `.github/workflows/ci.yml` (4 jobs), populated `CHANGELOG.md` |
| **1 — DXGI correctness** ✅ | The package did not import at all before this. Fixed 5 undefined imports, signed/unsigned HRESULT comparisons (every error check was dead code), a leaked texture reference that stalled capture after 2 frames, invalid f-string format specifiers in the error paths, `DuplicateOutput1` with env-var fallback, access-lost/session-disconnect recovery, bounded fullscreen rebuild, HDCP handling |
| **1b — Pixel path** ✅ | RGBA 3.59×, BGR 2.58×, RGB 2.52×, GRAY 1.85×, outputs verified identical. Also fixed a frame-aliasing data-corruption bug (`grab()` returned views into recycled pool buffers) |
| **3 (slice) — Frame object** ✅ | `grab_frame()` returns a GPU-resident `Frame` with an explicit texture lifetime; **21× faster than `grab()`** on the current baseline (4.53 → 0.21 ms). Guards against the `INVALID_CALL` stall with a clear error |
| **6 + 6b — GPU tensor** ✅ | `GpuPreprocessor12` produces an `ID3D12Resource` on the DirectML device: BGRA→NCHW float32 with resize and normalisation in one dispatch. Verified exact against a NumPy reference and against real capture |
| **6.2 — Headless diagnostics** ✅ | `HeadlessError` replaces `"No usable graphics devices found. Check your display configuration."` with the actual fix (install an IDD virtual display), and distinguishes *no display* from *every device refused to open* — previously the same message |
| **6.1 (detect + measure)** ✅ | `rapidshot.topology_info()` classifies headless / single / hybrid / multi-adapter and says what a hybrid system means for the GPU tensor path. `native.probe_cross_adapter()` verifies the whole `SHARED_CROSS_ADAPTER` chain (heap → shared handle → open on the second device → placed resource on both) and times the capture-side copy |
| **6.3 (dirty rects)** ✅ | `frame.dirty_rects` in frame coordinates, plus `rects_coalesced`. The compositor already computed this and Rapidshot was discarding it — `GetFrameDirtyRects` was declared without argtypes, making it callable but unusable. Live capture reports **0.7–0.8% of the frame** dirty for a moving window |
| **6.3 (region-limited conversion)** ✅ | `grab()` converts only the dirty regions into a persistent accumulator, **12–15× faster** at the dirty fraction live capture produces. Falls back to a full conversion when metadata is missing, the area exceeds 90%, rotation is in play, or the mode is BGRA |
| **Pooled output (2.0 breaking change)** ✅ | `grab()` returns a `PooledBuffer` the caller releases. Allocating per frame cost ~1.6 ms in page faults — more than the conversion — so reuse is **1.3–2.1× on `grab()`**. See § 10 |
| **Stage 0 — release infrastructure** ✅ | PyPI Trusted Publishing with Sigstore attestations, SBOM, four wheel gates, GitHub Release automation, `py.typed` with the public API annotated, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, PR template, least-privilege CI tokens, performance badges generated from `baseline.json` with a drift guard |
| **6.1 (frame transfer)** ✅ | `native.cross_adapter_transfer(frame)` carries a captured frame to a second adapter and exposes the `ID3D12Resource` it lands in. Heap and placed resources are allocated once; only the copy is per-frame. Verified byte-exact on real capture by `examples/verify_cross_adapter.py` — 8,294,400 bytes per frame, against a source-side readback of the same snapshot |

**Also fixed:** `pip install rapidshot` shipped a broken package — `pyproject.toml` listed `packages = ["rapidshot"]`, so the wheel contained 5 modules instead of 25 and failed with `ModuleNotFoundError: No module named 'rapidshot.util'`. Invisible from a source checkout. Now guarded by CI.

**Test coverage:** 217 unit tests plus five live suites (Stage 1 regression, shot/color, Frame lifetime, native interop, D3D12 capture).

---

## 6. Next up, in order

### 6.1 — Cross-adapter transfer (hybrid GPU laptops)

**This is a coverage gap, not an optimisation.** [Desktop Duplication cannot run against the discrete GPU on a hybrid system](https://support.microsoft.com/en-us/help/3019314/error-generated-when-desktop-duplication-api-capable-application-is-ru) — it fails with `DXGI_ERROR_UNSUPPORTED`. So on an Optimus laptop you capture on the iGPU while inference runs on the dGPU, and the Stage 6 tensor has nowhere to go. That is a large share of consumer NVIDIA hardware.

Detection, measurement and the **BGRA frame transfer** are done (see § 5). `native.cross_adapter_transfer(frame)` carries a captured frame to the second adapter and exposes the `ID3D12Resource` it lands in.

Settled, do not re-derive:

- The chain works: heap with `D3D12_HEAP_FLAG_SHARED | SHARED_CROSS_ADAPTER` → `CreateSharedHandle` → `OpenSharedHandle` on the second device → `CreatePlacedResource` on both sides.
- [Cross-adapter shared resources live in system memory](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps). This is **not** VRAM-to-VRAM peer-to-peer DMA, and `IDXGIAdapter3` is not the mechanism (that is video-memory budgeting). The win is that a GPU copy engine moves the bytes instead of CPU cores.
- **Use a buffer, not a row-major texture.** `CrossAdapterRowMajorTextureSupported` is an optional capability; the buffer path works regardless and needs no branch.
- **Allocate once, copy per frame.** Nothing but the copy depends on the frame.
- **The duplicated surface is live.** Rapidshot does not hold its keyed mutex during the copy, so two copies of "the same" frame — even recorded into a single command list — genuinely observe different pixels. This appeared as ~2,100 bytes differing in one screen region, reproducibly at the same offset, and it is why the verification snapshots the surface once and feeds both its comparands from that. Anything that reads a captured texture twice and expects agreement is wrong.
- Cost: **0.87 ms** per 1080p frame on the capture side (§ 3).

**Ordering: transfer the frame, do not convert first.** This was measured, not reasoned about — `benchmarks/cross_adapter_ordering.py`, three runs, capture-side cost on the Intel iGPU:

| Model input | Tensor | Convert (iGPU) | Transfer tensor | **B total** | vs A |
| --- | --- | --- | --- | --- | --- |
| 320² | 1.23 MB | 0.38–0.45 ms | 0.14–0.16 ms | **0.52–0.59 ms** | B saves 0.1–0.5 ms |
| 416² | 2.08 MB | 0.46–0.49 ms | 0.18–0.20 ms | **0.64–0.67 ms** | B saves 0.05–0.3 ms |
| 640² | 4.92 MB | 0.54–0.60 ms | 0.34–0.40 ms | **0.90–0.99 ms** | tie |
| 832² | 8.30 MB | 0.72–0.83 ms | 0.59–0.61 ms | **1.32–1.42 ms** | A saves 0.4–0.7 ms |
| 1280² | 19.7 MB | 1.29–1.32 ms | 1.54–1.70 ms | **2.83–3.03 ms** | A saves 2.0 ms |

Ordering A — transfer the 8.29 MB frame — costs **0.70–0.98 ms** regardless of model size.

So B wins only below 416², **640² is a tie**, and A wins clearly above it. Three further points settle it in A's favour at the tie:

- In A the conversion runs on the *consumer's* GPU, which on a hybrid system is the faster one by assumption. These figures therefore understate A.
- B spends iGPU time on every frame; the iGPU is also driving the display.
- A matches the principle in § 11: Rapidshot produces frames and does not own its consumers' pipelines. Handing over BGRA leaves the model's preprocessing to the model's owner.

Remaining work:

- **A shared fence.** Synchronisation is currently CPU-side: `transfer` blocks until the source GPU finishes. Correct, but it serialises the two adapters. `D3D12_FENCE_FLAG_SHARED | SHARED_CROSS_ADAPTER` would let them overlap. Likely worth more than the ordering choice ever was.
- **Real hybrid hardware.** WARP proves the mechanism; it cannot prove an Intel→NVIDIA transfer behaves the same. `examples/verify_cross_adapter.py` is the check to run there.
- Do **not** chase per-frame shared-handle caching on this evidence. One run suggested `transfer()` cost 1.48 ms against a 0.66 ms raw copy, implying ~0.8 ms of per-frame handle overhead; three further runs put it at 0.70–0.98 ms, consistent with the raw copy. The apparent overhead was noise. (The capture texture pointer *is* stable across frames, so caching remains possible if a real profile ever justifies it.)

### 6.2 — Headless / virtual display diagnostics ✅

Done. [With no monitor attached there is no desktop to duplicate](https://github.com/FreeRDP/FreeRDP/issues/5825) — `DuplicateOutput` fails outright, which blocks every cloud-VM deployment. `HeadlessError` now names the fix (install an IDD virtual display driver) instead of saying "check your display configuration".

Kept here because the caveat is easy to lose: a virtual display's advertised refresh rate does **not** raise capture rate. DDA is driven by presents, not refresh — a 500 Hz virtual display does not make an application render 500 fps. This is in the error text and the README.

Not yet exercised on a genuinely headless machine — the logic is tested by describing that topology, not by having one.

### 6.3 — Finish Stage 3 (Frame metadata)

The lifetime slice is done, `dirty_rects` is done, and `py.typed` now ships with the module-level API annotated. **Still missing:** `move_rects` (the COM signature is declared and ready — see § 6.3's settled notes), cursor data on `Frame`, normalised timestamps, and `Protocol`-typed interfaces for the public surface.

Design this **before** a second backend exists — retrofitting a DXGI-shaped API to fit WGC later is more expensive than designing one abstraction all backends fill.

**Settled by `dirty_rects`, and it applies to the rest:**

- **Frame metadata must be in frame coordinates, not desktop coordinates.** DXGI reports rects relative to the whole duplicated output; a `Frame` may cover a region of it. Passing raw values through would make `dirty_rects` index outside the frame whenever a region is off-origin — wrong only in the case nobody checks by hand. `Frame` clips and translates. `move_rects` and cursor position need the same treatment.
- **Empty and unknown are different answers.** `[]` means no rects were reported; `None` means the metadata could not be read. A consumer skipping unchanged regions must distinguish them or it silently skips everything on a frame whose metadata failed. An empty list does *not* mean nothing changed — a mode change or a coalescing driver can report none while the image differs completely.
- **`RectsCoalesced` matters.** When set, the driver merged rects, so they over-estimate what changed. Surfaced as `frame.rects_coalesced`.
- The COM signatures for `GetFrameDirtyRects`/`GetFrameMoveRects` were declared in `_libs/dxgi.py` without argtypes, so they were callable but unusable — comtypes could not marshal the out-parameters. `GetFrameMoveRects` is now declared correctly and is the next piece to wire up.

Measured on real capture: an animated window moving across an otherwise still desktop reports **one rect covering 0.7–0.8% of the frame**. That is the size of the prize for consumers that can act on it — and the number § 7's GPU-side change detection (6c) has to beat.

**Region-limited conversion pays, and by a lot** — `benchmarks/dirty_rect_savings.py`, 1080p BGRA→RGB:

| Dirty | 1 rect | 8 rects | 64 rects |
| --- | --- | --- | --- |
| 0.8% (the live figure) | **168×** | 106× | 27× |
| 5% | 30× | 26× | 15× |
| 15% | 9.9× | 9.6× | 6.5× |
| 50% | 2.6× | 2.5× | 2.3× |
| 80% | 1.3× | 1.3× | 1.3× |

- Cost scales linearly with dirty area; the strided-view penalty never appears.
- Per-rect overhead is **~1 µs**, so hundreds of rects stay affordable.
- **There is no losing regime.** Even at 80% dirty it is still 1.3× faster, so the optimisation never needs a "give up and do the whole frame" threshold on cost grounds.
- Tall narrow rects cost ~1.4× wide ones at equal area, as expected from cache behaviour. Not enough to change any decision.
- Output verified against the full-frame conversion inside every rect, and confirmed to touch nothing outside them.

**The design consequence is the hard part, not the speed.** Converting only part of the frame means the rest of the destination buffer must already hold the *previous* frame. `grab()` currently fills a fresh pool buffer completely, and the pool recycles — so a partially-written buffer would contain some other frame's pixels. That is the frame-aliasing bug of Stage 1b returning by a different route. Two options:

- Keep a persistent accumulator and copy it out per frame. The copy-out is full-frame work the current path never pays.
- Return a view valid only until the next `grab()`. No copy, but it reintroduces exactly the aliasing hazard that was already fixed once.

**The accumulator was then measured end to end** — `benchmarks/dirty_rect_pipeline.py`, both complete pipelines including the copy-out, so the 168× above does not mislead:

| Dirty | Accumulator | vs today |
| --- | --- | --- |
| 0.8% (the live figure) | 0.16–0.18 ms | **12–15×** |
| 3% | 0.18–0.21 ms | 11–13× |
| 10% | 0.33 ms | 6.8–7.2× |
| 25% | 0.62 ms | 3.7× |
| 50% | 1.30 ms | 1.8× |
| 75% | 1.88 ms | 1.2× |
| 100% | 2.46 ms | **1.05–1.09× slower** |

- The copy-out alone is 0.14–0.17 ms, which is the accumulator's floor: **no dirty-rect scheme can beat ~15× on this path**, however little changed.
- Break-even sits at essentially 100% dirty. The only losing case is a fully-dirty frame — video playback, a fullscreen game — and it costs just **5–9%**.
- That regression is trivially avoidable: fall back to the current path when the metadata is missing or the dirty area exceeds ~90%. Given the tiny penalty, the guard is about predictability rather than necessity.
- Verified: the accumulator produces a frame byte-identical to the current path.

**Then it was built, and the real gain is 1.5×, not 12×.** Measured on live capture with an animated window, `output_color="RGB"`:

| | Today | With dirty rects | |
| --- | --- | --- | --- |
| `grab()` | 4.56 ms | **2.97 ms** | **1.5×** |
| of which `process()` | 3.16 ms | 1.90 ms | 1.7× |

Three reasons the projection missed, all worth keeping in mind before trusting the next one:

- **`process()` is only 69% of `grab()`.** Acquire, `CopySubresourceRegion`, and map/unmap are untouched, so Amdahl caps the end-to-end result no matter how good the conversion gets.
- **The staging read must touch whole rows.** The live figures are **0.8% dirty area but 11.5% dirty rows** — a tall narrow rect spans many rows while covering little of them. The read shrinks with rows, not area, and it is the expensive half.
- **The pipeline benchmark used a RAM proxy for the mapped surface**, as `perf_suite.py`'s fixture does. A real mapped staging surface is uncached and far slower, so the component that shrinks least was the one modelled most optimistically. The 12–15× figure was measuring the wrong thing, not measuring it wrongly.

Still worth having: 1.6 ms/frame, no regression on the median, correctness verified live. But quote 1.5×, not 12×.

**Reading rect columns instead of whole rows: measured, no difference.** The idea was to touch 0.8% of the surface instead of 11.5%. Against a real mapped staging surface with fixed rect shapes, both strategies land within noise — tall-and-narrow, square, and wide-and-short alike. `_read_patch_columns` is kept only so `benchmarks/dirty_rect_read_strategy.py` can reproduce that; rows stays the default as the simpler of two equals.

**Two lessons from getting there, both worth more than the result:**

- **Timing `grab()` over live frames cannot compare two implementations.** Each frame's cost depends on what happened to change on screen at that instant, so consecutive runs of the same comparison gave **2.26×, 1.56× and 0.87×**. That is measuring the desktop. A controlled comparison needs fixed rects over one captured frame with its staging surface still mapped — which is what the benchmark now does.
- **The read was never the bottleneck.** The patch path costs ~1.8 ms even for a rect touching 1% of rows, because the output array is allocated fresh every frame. See § 10.

### 6.4 — Stage 3b: async streaming with backpressure

Pull-based `grab()` is wrong for ML consumption loops that cannot always keep up.

```python
async for frame in camera.stream(maxsize=4, drop_policy="oldest"):
    await model.infer(frame)
```

Ring buffer with producer/consumer semantics; drop per policy rather than queuing unboundedly or blocking capture.

### 6.5 — Stage 4: WGC backend

Pure catch-up — DXcam and `windows-capture` both have it — so scope to parity. Real advantages over DXGI worth having: per-window (HWND) capture, and cross-GPU capture without the capture process running on the display's adapter (which relates directly to § 6.1).

`Direct3D11CaptureFramePool.CreateFreeThreaded`, `SystemRelativeTime`, `ContentSize`, `Recreate()` on resize/device-loss, plus cursor/dirty-rect/rotation/HDR handling.

### 6.6 — End-to-end demo

Capture → GPU tensor → a small ONNX model → bounding boxes, in ~20 lines. This artifact drives adoption more than any individual stage. Blocked only on a consumer writing the ORT glue (§ 8).

---

## 7. Later stages

- **6c — GPU-side change detection.** Lower value than it appears: DDA *already* reports only changed content and provides dirty rects computed by the compositor. Real residual value is deduplicating presents that are reported as changed but visually identical, and sub-dirty-rect granularity. **Do § 6.3's dirty rects first and measure** before building this.
- **5 — Backend auto-selection.** Needs ≥2 backends, so it follows § 6.5.
- **7 — Hardware encode.** NVENC / AMF / QSV behind one API. *NVFBC caveat:* deprecated for general use on Windows 10+ (frozen at Capture SDK 7.1); treat it as a Linux-only path.
- **8 — `rapidshot.stream` network streaming.** WebRTC transport, DataChannel input, browser viewer. This is what changes the product category from "screenshot library" to "capture-and-stream infrastructure."
- **9 — Remote-support primitives.** `WDA_EXCLUDEFROMCAPTURE`, adaptive bitrate hook.
- **10 — AI inference layer.** `frame.to_ort(session)`, a `ScreenDetector` convenience API, multi-source synchronised capture.
- **11 — Ecosystem.** OpenCV `VideoCapture` wrapper, PyTorch `IterableDataset`, LangChain tool, OBS source plugin; propose an open Python screen-capture specification.
- **Stage 0 remainder.** Done for the 2.0.0 release: `py.typed` (with the public API annotated, so the marker is not a lie — a package that ships it while `create()` returns `Any` is worse than one that ships nothing), `SECURITY.md` routed through GitHub private reporting, `.github/CODEOWNERS`, `release.yml` doing PyPI Trusted Publishing with Sigstore attestations and a CycloneDX SBOM, and `RELEASING.md`. CodeQL runs through GitHub's **default setup**, enabled in the repository — do not add a `codeql.yml`: an advanced configuration and the default setup cannot coexist, and the advanced one fails at the SARIF upload with "CodeQL analyses from advanced configurations cannot be processed when the default setup is enabled". **Still outstanding:** `GOVERNANCE.md` (needs a decision, not a file), OpenSSF Scorecard, and hosted docs.

---

## 8. Deferred and out of scope

**Stage 2 — native capture core: do not build.** Measurement says 0.003 ms/frame on a hardware-bounded path, and `windows-capture` already ships it. Revisit only if § 9 (cross-platform) is committed to, or if a profile on a real workload contradicts § 3.

**ONNX Runtime session binding: deliberately not ours.** Rapidshot produces the `ID3D12Resource`; binding it is ~15 lines for a caller who already has ORT set up, documented in the README. Every route to `OrtDmlApi` costs something permanent — vendoring ~5000 lines of header plus hand-counted struct offsets, bindgen adding libclang as a third build dependency, or the `ort` crate requiring ONNX Runtime built from source. All couple Rapidshot's core to ORT's ABI and release cadence for one optional feature.

Progress that exists and is worth keeping:

- `native.probe_onnxruntime()` confirms ORT is reachable from Rust (1.24.4, C API versions 1–24). Only `OrtApiBase` is traversed — two members, layout fixed by contract, no risk.
- `OrtDmlApi` indices from the header: `CreateGPUAllocationFromD3DResource` = **2**, `FreeGPUAllocation` = **3**, `GetD3D12ResourceFromAllocation` = **4**.
- **Gotcha:** resolving `onnxruntime.dll` by name uses the DLL search path and can find an unrelated version (on the dev machine, 1.17.1 rather than the package's 1.24.4). Struct layout is version-dependent, so this matters. Use `native.onnxruntime_dll_path()`.
- **Never hardcode an unverified `OrtApi` offset.** If revisited: pin to a specific `ORT_API_VERSION` (the C API is append-only, so version N guarantees N's layout up to N) *and* validate at runtime that the pointer falls inside `onnxruntime.dll`'s address range.

If demand appears: a separate `rapidshot-directml` package, or a Python-side `ctypes` binding — same fragility but patchable without a rebuild and auditable without a toolchain.

---

## 9. Stage 4b — Cross-platform (a project-scale bet)

macOS ScreenCaptureKit and Linux PipeWire + XDG Portal. This is the one stage that genuinely justifies a native core, and that decision would drive Stage 2 rather than the reverse.

Realistic cost is roughly six months with a native-graphics-fluent co-maintainer — **that estimate belongs here, not to a Windows-only core**, which is a few hundred lines. Most of the effort is permission-flow edge cases: XDG portal tokens invalidating on fullscreen toggle, macOS Screen Recording permission needing restart-after-grant handling.

---

## 10. Known debt

- **Pooled output is the default since 2.0**, and it is a breaking change: `grab()` returns a `PooledBuffer` the caller must `release()`. Allocating the output array cost ~1.6 ms per 1080p frame in page faults — more than the conversion it feeds — so reusing buffers is **1.3–2.1× on `grab()`** across RGB, RGBA and GRAY, pixels identical. The wrapper indexes and converts like the array it wraps (`frame[y, x]`, `np.asarray(frame)` zero-copy), so the migration is usually one added `release()`; `pool_output=False` restores 1.x behaviour. Use after release raises rather than returning another frame's pixels.
- **Exclusive-fullscreen and HDCP paths are fault-injection tested only.** The logic is verified against injected HRESULTs; neither has been exercised by its real trigger.
- **Hybrid and headless topologies are classified but never observed.** The dev machine is a single Intel iGPU driving two monitors. Both branches are tested by describing those machines; neither has been run on one.
- **Cross-adapter sharing verified against WARP only.** The mechanism works and the source-side cost is measured, but no Intel→NVIDIA transfer has been performed.
- **GRAY is by far the slowest colour mode, and worse than previously recorded.** It is bimodal: a ~9.2 ms fast mode and a ~15 ms slow one. The fast mode is a transient the CPU sustains for a second or two, so a *capture loop never sees it* — at 60 Hz GRAY fills ~95% of a frame and runs effectively sustained. The old 9.16 ms figure came from a harness that drove it in short bursts; the honest number is **13.7–14.9 ms**, which means GRAY cannot keep up with a 60 Hz display at 1080p. A SIMD kernel is now the clearest remaining CPU win, and this is the strongest argument yet for writing one.
- **CuPy/CUDA paths untested** — no NVIDIA GPU available. Worth a spike: `gfx2cuda` + CuPy may give a **pure-Python** D3D11→CUDA tensor path, which would mean CUDA users get GPU tensors with no build step at all.
- **`shot()` writes BGRA regardless of `output_color`**, and overruns an undersized buffer without bounds checks. Documented; not yet fixed.
- **No hosted docs**, no release automation.

---

## 11. Principles

**Measure before ordering.** Two stages in this document were reordered by profiling: Stage 2 was dropped after measuring 0.003 ms/frame, and Stage 6 was promoted after measuring ~8 ms. Both had been sequenced on intuition.

**Rapidshot produces frames; it does not own its consumers' bindings.** Every time the alternative was considered — a native capture core, vendored ONNX Runtime headers, an `ort` crate dependency — the cost was permanent coupling paid by all users for the benefit of an optional feature.

**Optional means optional.** `pip install rapidshot` must never require a toolchain. CI enforces this: the main test job asserts the native extension is *absent*.

**A fast wrong answer is worthless.** The GPU shader was briefly producing BGR labelled RGB, which no test of speed, shape or stability would have caught. Correctness checks against an independent reference come before performance claims.

---

## 12. Downstream consumer track (separate codebase)

`inventory-agent` — not part of Rapidshot, noted because its later phases consume Rapidshot's output.

- **Independent, can start anytime:** Named Pipe IPC between the Session-0 service (`agent.exe`) and a per-user tray helper (`agent-tray.exe`); the "Request Help → ticket" flow.
- **Gated on Stage 8/9:** remote-view streaming (tray helper captures, encodes, pushes over WebRTC to a relay; browser dashboard renders and returns input for `SendInput`). Building it before Rapidshot reaches Stage 7/8 means redoing it — JPEG-over-WebSocket now, WebRTC later.
- Consent UI, persistent "IT is viewing your screen" banner, session audit log, idle timeout, WiX installer updates are hardening on top of whichever streaming approach is live.
