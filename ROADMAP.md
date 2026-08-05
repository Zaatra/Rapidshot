# Rapidshot Roadmap

**Goal:** be the capture layer the AI-agent/CV ecosystem reaches for on Windows — not a faster copy of DXcam.

This document is written to be read cold. It states where the project actually is, what to do next, and which questions are already settled so they are not re-litigated. Everything marked ✅ has been implemented and verified; see `CHANGELOG.md` for detail.

---

## 1. Start here

**Current state.** Rapidshot captures the desktop via DXGI Desktop Duplication and can hand a frame to a GPU consumer as a **model-ready NCHW float32 tensor that never touches the CPU**. The CPU path for the same work costs ~8 ms per 1080p frame. Core capture is pure Python; an *optional* Rust extension provides GPU interop only.

### Release status

**2.0.0 is published.** PyPI serves `rapidshot 2.0.0`, and the GitHub Release for `v2.0.0` (4 August 2026) carries its three assets. PyPI Trusted Publishing and the `pypi` GitHub environment are configured and restricted to `v*` tags, so a pushed tag is what cuts a release; the full procedure is in `RELEASING.md`.

**1.1.0 is yanked**, reason *Newer Version*. It was the only thing on PyPI from April 2025 to now and it did not work: it failed to import on Python 3.11+ (`cursor: Cursor = Cursor()` trips the dataclass mutable-default check broadened in 3.11), and patching that one line only got it to return all-black frames, because the processor was handed a texture where it expected a mapped staging surface. Yanking is not deletion — an existing `rapidshot==1.1.0` pin still resolves, which is the intent; only new unpinned installs are steered away.

One release item cannot be verified from the repository and should be confirmed in the GitHub UI: **Settings → Security → private vulnerability reporting** must be enabled (or the link in `SECURITY.md` 404s), and branch protection needs *Require review from Code Owners* (or `CODEOWNERS` is only a routing hint).

**Next feature task:** § 6.3 — finish Stage 3 (Frame metadata). § 6.1 is complete: hybrid and headless systems are reported clearly, a captured frame crosses to a second adapter at **0.70–0.98 ms per 1080p frame** verified byte-exact, and the convert-first-or-transfer-first question has been measured and settled in favour of transferring the frame. What § 6.1 still lacks is validation on real hybrid hardware and an asynchronous shared fence, both noted there.

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
- **The control benchmark only rescues comparisons that resemble it, and cannot rescue one across machines at all.** `control.memcopy` measures memory bandwidth, so dividing by its movement normalises benchmarks that are *also* bandwidth-bound and quietly mis-normalises everything else. On a CI runner this reported `pipeline.cpu_to_nchw` — float32 resize/normalise/transpose, compute-bound and sensitive to vector width and NumPy version — as a **1.34× regression against a code path nobody had touched**, while simultaneously calling every conversion row **1.4× faster on a machine that was uniformly slower**. One control standing in for workloads it does not resemble, wrong in both directions at once. `print_comparison` now detects a baseline recorded on different hardware (processor / platform / GPU), marks every verdict *indicative*, and gates nothing; a spurious improvement is flagged as loudly as a spurious regression, because nobody investigates good news. To compare code against code, re-record on the machine you are testing on.
- **CI cannot verify live capture.** GitHub runners have no desktop session. Those tests skip themselves and must be run on real hardware before a release.

---

## 3. Measured baseline

All figures 1920×1080 BGRA (8.3 MB/frame), measured on the dev machine. Stored in `benchmarks/baseline.json`, **re-recorded 2026-08-05** after the GRAY work in § 10. Earlier recordings are kept as `benchmarks/baseline-2026-07-30.json` and `benchmarks/baseline-2026-07-27.json`; do not compare across recordings casually — the 07-27 one drove the benchmarks differently, and the notes below apply to the current one.

**There are two committed recordings, and which one you want depends on the question:**

| File | Extension | What it answers |
| --- | --- | --- |
| `baseline.json` | **built** | What the library can do on stated hardware. Feeds the README badges. |
| `baseline-nonative.json` | absent | What `pip install rapidshot` gets, and what CI compares against — CI has no toolchain. |

Both were recorded back-to-back on 2026-08-05 with `--rounds 5 --reps 25`, the invocation § 2 documents, so they are directly comparable to each other rather than separated by machine drift. Two consequences worth knowing:

- **CI's compare step points at `baseline-nonative.json`.** Aimed at `baseline.json` it would report a 6–20× "regression" on every conversion row forever, since the runner builds no extension — which is how a benchmark suite teaches people to ignore it.
- **`pipeline.gpu_dispatch` and `pipeline.gpu_plus_readback` appear only in `baseline.json`**; they need the extension.
- The **live rows were recorded against a defined synthetic workload**: a 420×300 window moved at ~30 Hz. See the dirty-fraction note in § 6.3 — a small moving window is the *favourable* end of that distribution.

Use `python benchmarks/compare_recordings.py` to diff any two stored recordings; it normalises each by its own control row before quoting a ratio.

| Stage of the loop | Cost/frame |
| --- | --- |
| Python → COM binding overhead (~6 calls) | **0.003 ms** |
| `CopySubresourceRegion` (GPU) | 0.016 ms |
| Read from mapped staging surface | 2.27 ms |
| Pixel conversion, RGB/BGR (NumPy, post-optimisation) | 1.76–1.84 ms |
| Pixel conversion, GRAY (NumPy) | 6.9–10.5 ms — was 13.7–14.9, see § 10 |
| Preprocess for a model (resize/normalise/CHW → 640×640) | 6.23 ms |
| **CPU total, capture → model input** | **~8 ms** |

Capture path comparison, real capture:

| Path | Per frame |
| --- | --- |
| `grab()` — CPU staging read + convert | 4.27 ms — **see the caveat below** |
| `grab_frame()` — texture stays on GPU | 0.77 ms — **see the caveat below** |

**Neither live figure should be quoted without this caveat, and both are unreliable.** Across six recordings in a single session, on code that only ever got faster, the minima ranged:

| Row | Observed range | Spread |
| --- | --- | --- |
| `live.grab_with_frame` | 1.65 – 4.53 ms | 2.7× |
| `live.grab_frame_gpu` | 0.17 – 0.77 ms | 4.5× |

`grab_frame()` does no conversion at all — the texture never leaves the GPU — so its 4.5× spread is *purely* measurement, not code. The honest figure for it remains **0.17–0.21 ms**, which five of six recordings agree on; `baseline.json` happens to hold the outlier.

Three traps sit behind this, all worth keeping:

- **Live cost tracks the screen, not the library.** Region-limited conversion only converts the dirty part, so `grab()` depends on what moved. § 6.3 records the same effect from the other side: median dirty fraction was 0.8% for a small animated window and 68% for a dragged one.
- **A minimum is monotonically non-increasing in sample count.** Raising `--live-seconds` from 3 to 15 moved `grab()` from 3.08 to 1.65 ms on unchanged code — more samples simply find a luckier frame. So live rows are comparable only at identical `--live-seconds`, and "sample longer" makes the number flattering rather than truer. Use the default.
- **Re-recording until the number looks good is cherry-picking**, and it is tempting precisely because the spread is this wide. The recording stands as taken.

**The `grab`/`grab_frame` badges inherit all of this, and that is a known defect.** The `BGRA→RGB` badge was added because `convert.*` rows are synthetic and deterministic: it moves when the library changes and not otherwise, which is what a badge is for. Replacing the two live badges with synthetic rows is the obvious fix and has not been done.

Between the 07-27 and 07-30 recordings the absolute figures *rose* (RGB 1.45 → 1.76 ms, GRAY 9.16 → 13.72 ms). Part of that was the harness correction, which stopped flattering GRAY in particular; part was machine state three days apart. The two causes could not be separated after the fact, which is the argument for re-recording a baseline whenever the harness changes rather than carrying one across — and for recording *why* alongside the numbers, as the provenance list above now does.

### Native conversion kernels

Every colour mode now has a byte-exact Rust kernel, used automatically when the optional extension is present and declined cleanly when it is not. Measured 2026-08-05 by `perf_suite --synthetic-only --rounds 5 --reps 25 --compare`, against the native-absent baseline, with the control's 1.11× drift divided out:

| Mode | NumPy | Native | Gain | GB/s | Share of the 33.2 GB/s ceiling |
| --- | --- | --- | --- | --- | --- |
| BGRA | 0.22 ms | *(unchanged)* | — | 33.2 | **100%** — a straight copy; nothing to win |
| GRAY | 9.39 ms | **0.26 ms** | **37.2×** | 31.8 | **96%** |
| BGR | 1.91 ms | **0.30 ms** | 6.5× | 27.3 | 82% |
| RGB | 1.90 ms | **0.31 ms** | 6.3× | 26.6 | 80% |
| RGBA | 2.61 ms | **0.36 ms** | 7.5× | 22.9 | 69% |

The `convert.BGRA` row is the control that made this worth doing at all: the same 8.29 MB moves at 33.2 GB/s when nothing is reordered, so the old 2–3 ms figures were never a memory-system limit. They were three separate strided gather/scatter passes — `dst[..., 0] = src[..., 2]` and so on — where one pass can read each cache line once.

**The three reorder modes use `pshufb`; that is what took them from 30–65% of the ceiling to 69–79%.** The autovectorised Rust loops first reached only 2.5–7.2×, and RGB was the worst of them despite being the mode most CV consumers hand to a model. The reason was visible by contrast with BGR: both write three bytes per pixel, but BGR keeps channel order and LLVM widens it into a clean load-4/store-3, whereas RGB has to *reverse* each triple, which defeated the vectoriser entirely and degraded to per-byte stores. `_mm256_shuffle_epi8` expresses exactly that permutation, and RGB went 0.82 → 0.32 ms.

Two things about that kernel are worth not rediscovering:

- **It stores exactly 24 bytes, not 32.** The conventional trick for a 3-byte output is to store a full vector and let the next iteration overwrite the surplus. That would run past the end of a row — which for a dirty-rect patch is not the end of the buffer but *the next row of live pixels*, so it corrupts silently rather than crashing. Storing 16 + 8 costs one extra instruction and removes the hazard, and the tests assert no write lands past a row end.
- **`pshufb` works per 128-bit lane**, so after the shuffle each half holds 12 useful bytes followed by 4 zeros. `vpermd` compacts the halves (dwords 0,1,2 then 4,5,6) before the store. Skipping that step yields output that looks right for the first four pixels of every group and is wrong afterwards.

AVX2 is detected at runtime — x86_64 guarantees only SSE2, so a wheel that assumed it would fault on older hardware — with the scalar loops as the fallback. The tests compare vector against scalar at **every width from 1 to 64**, which is what covers all eight possible tail lengths; a tail bug is invisible at any width that happens to be a multiple of 8 — including the 1920 this library actually runs at.

**GRAY has an AVX2 kernel too, and it is now the fastest of the five at 96% of the ceiling** — 0.26 ms, which beats single-threaded OpenCV's 0.34 ms while being byte-exact where OpenCV is off by up to 1 LSB. It went from the slowest mode by an order of magnitude to essentially free. Its design is dictated entirely by one hazard:

- **`_mm256_maddubs_epi16` is unusable, despite looking purpose-built.** It accumulates into *signed* i16 with saturation, and `b*29 + g*150` reaches 45,645 — past 32,767. It would clamp on bright pixels and corrupt them silently; the error appears only in highlights, so no test of speed, shape or stability would catch it, and a randomly sampled correctness test would very likely pass.
- The way around it is to widen to u16 first and use **`_mm256_madd_epi16`, which accumulates into i32** where the Q8 total cannot overflow. One unpack per half buys correctness that does not depend on the input.
- `phaddd` then sums the two partial products per pixel and, usefully, interleaves the low and high unpacks back into pixel order — so all eight lumas emerge in sequence with no cross-lane fixup.
- Because of that hazard the correctness test is **exhaustive over all 2²⁴ BGR triples through the vector path**, not sampled. It costs about 20 ms.

That leaves RGBA the least efficient at 69%, and it is the one mode where the autovectoriser was already close, so there is little left to win anywhere in conversion. **The remaining CPU costs are elsewhere: see § 10.**

None of this changes `benchmarks/baseline.json`, which is deliberately recorded with the extension absent (§ 3 provenance above). The NumPy fallbacks were left byte-for-byte identical, so the no-toolchain install performs exactly as the baseline records.

Cross-adapter transfer, 1080p BGRA (8.29 MB) into a `SHARED_CROSS_ADAPTER` heap, copy-queue submission and fence wait included:

| | Per frame |
| --- | --- |
| Copy into the shared heap | **0.87 ms** min, 0.94 ms median (~9 GB/s) |

Measured Intel iGPU → WARP, since the dev machine has no second hardware GPU. The **source** side is representative — on an Optimus laptop capture also runs on an iGPU with no dedicated VRAM, so it is the same system-memory copy. What the *consumer* adapter pays to read the heap is its own device's cost and is not measured here. Reproduce with `native.probe_cross_adapter()`.

For scale: this is about a third of what reading the same frame to the CPU costs (2.27 ms staging read), so moving a frame between adapters is cheaper than leaving the GPU.

---

## 4. Settled questions — do not re-litigate

**Python is not the bottleneck.** CPU pixel work is ~2,500× the entire Python/COM binding overhead. At 240 FPS the binding cost is 0.07% of the frame budget.

**Capture rate is bounded by the compositor's present rate, not by the display's refresh rate.** This entry previously said "at most one frame per display refresh", which is **wrong**, and § 6.2 already contradicted it in passing ("DDA is driven by presents, not refresh") — the two sat in the same document for months.

Measured 2026-08-05 on the 100 Hz primary output, against a source presenting unthrottled at ~610 updates/s:

| | |
| --- | --- |
| Frames returned | 705 in 6.00 s — **117.5 fps on a 100 Hz panel** |
| Distinct `LastPresentTime` values | 705 — **zero repeats** |
| Inter-present gaps | min **1.010 ms**, p50 9.181 ms; 48.4% shorter than 9 ms |
| `AccumulatedFrames` | `{1: 301, 2: 386, 3: 17, 5: 1}` → ~**188 presents/s** |

Every acquire carried a distinct QPC present timestamp, so these are neither duplicates nor cursor-only updates, and gaps of 1 ms are impossible if presents were tied to scanout. DXGI reports *presents* — desktop composition updates — and DWM composes when content changes rather than when the panel scans out. The coalescing figure is the sharper one: 386 of 705 acquires carried **two** presents, so the compositor produced ~188/s and the capture loop was *missing* presents, not inventing them.

What still holds: **Python is not what limits this** — DXcam reaches 239 FPS in pure ctypes, and on this machine both libraries sit at 117–169 fps, bounded by the compositor rather than by either implementation. What does not hold is the ceiling's *value*: it is not the refresh rate, and a library reporting more than the refresh rate is not necessarily lying.

Two caveats before generalising. The dev machine runs **mixed refresh rates** (100 Hz + 60 Hz), where DWM's composition clock is known to behave irregularly; a single-monitor measurement has not been taken. And frames above the panel's refresh were never *displayed* as distinct images — useful as extra temporal samples for a model, redundant for a recorder.

**A native capture core is not worth building.** It addresses 0.003 ms of an 8 ms frame, and `windows-capture` already ships exactly that (Rust + PyO3, DXGI + WGC) — so it is neither a differentiator nor a measurable win. See § 8.

**Published FPS claims in this space are mutually contradictory.** DXcam's README reports DXcam at 239 FPS; BetterCam's reports DXcam at 39 and itself at 123; other sources quote BetterCam near 290. Different hardware, no shared harness. Plan against `benchmarks/`, not anyone's marketing.

**D3D11 cannot share buffers — only 2D non-mipmapped textures.** Six configurations were probed (structured/raw/plain × NT-handle/legacy); none produced a buffer D3D12 could open. This is why the conversion shader runs on D3D12. `native.probe_shareable_buffers()` re-checks it; a regression test fails if this ever changes.

**HLSL presents BGRA textures semantically.** For `DXGI_FORMAT_B8G8R8A8_UNORM` the hardware swizzles so `.x` is **red**, despite blue being first in memory. Reversing channels by hand produces BGR labelled RGB — silently wrong model input that no test of speed or shape catches.

**DWM does not emit move rects.** Measured 2026-08-05 on the dev machine (Intel iGPU, Windows 11): 2,205 frames of live capture while a 700×500 window was dragged across the screen at 30 Hz with its text view scrolling — the two classic move-rect producers. `GetFrameMoveRects` returned `S_OK` on every frame and reported **zero** move rects, while dirty rects arrived on all of them (4,071 rects). `TotalMetadataBufferSize` ranged 16–144 bytes; a `RECT` is 16 bytes and a `DXGI_OUTDUPL_MOVE_RECT` is 24, so the smallest frames do not reserve room for a single move rect. Under DWM composition this metadata is effectively vestigial.

The spec requirement is real even so, and is recorded here as a latent hole rather than a closed one: [MSDN states that to produce a visually accurate copy an application must process all move rects before it processes dirty rects](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-getframemoverects). The § 6.3 accumulator patches dirty rects only, so against a source that *does* report moves it would leave stale pixels at the move destinations. This is verified unobservable here, not proven impossible everywhere — one GPU, one driver, one OS build. Treat `move_rects` as **not worth implementing** until a source that emits them is found: the code path cannot be exercised on available hardware, and synthetic metadata proves nothing (§ 2).

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

- **A shared fence.** Synchronisation is currently CPU-side: `transfer` blocks until the source GPU finishes (`end_and_wait` in `native/src/cross_adapter.rs`). Correct, but it serialises the two adapters, and `D3D12_FENCE_FLAG_SHARED | SHARED_CROSS_ADAPTER` would let them overlap. **Measured 2026-08-05: the blocking wait is not the cost.** `probe_cross_adapter()` reports 0.83 ms min / 1.01 ms median for the copy at 9.5 GB/s, indistinguishable from `transfer()`'s 0.70–0.98 ms — so the wait adds essentially nothing per frame. The fence buys *overlap*: latency and pipelining, not throughput. It also cannot be validated here, because the probe reports `representative: False` and `destination_is_software: True`, and overlap against WARP predicts nothing about how a real discrete GPU schedules two queues. **Do this on hybrid hardware, not before.**
- **Real hybrid hardware.** WARP proves the mechanism; it cannot prove an Intel→NVIDIA transfer behaves the same. `examples/verify_cross_adapter.py` is the check to run there.
- Do **not** chase per-frame shared-handle caching on this evidence. One run suggested `transfer()` cost 1.48 ms against a 0.66 ms raw copy, implying ~0.8 ms of per-frame handle overhead; three further runs put it at 0.70–0.98 ms, consistent with the raw copy. The apparent overhead was noise. (The capture texture pointer *is* stable across frames, so caching remains possible if a real profile ever justifies it.)

### 6.2 — Headless / virtual display diagnostics ✅

Done. [With no monitor attached there is no desktop to duplicate](https://github.com/FreeRDP/FreeRDP/issues/5825) — `DuplicateOutput` fails outright, which blocks every cloud-VM deployment. `HeadlessError` now names the fix (install an IDD virtual display driver) instead of saying "check your display configuration".

Kept here because the caveat is easy to lose: a virtual display's advertised refresh rate does **not** raise capture rate. DDA is driven by presents, not refresh — a 500 Hz virtual display does not make an application render 500 fps. This is in the error text and the README.

Not yet exercised on a genuinely headless machine — the logic is tested by describing that topology, not by having one.

### 6.3 — Finish Stage 3 (Frame metadata)

The lifetime slice is done, `dirty_rects` is done, and `py.typed` now ships with the module-level API annotated. **Still missing:** cursor data on `Frame`, normalised timestamps, and `Protocol`-typed interfaces for the public surface.

`move_rects` is **deferred, not pending.** The COM signature is declared correctly and is ready to call, but DWM never emits a move rect (§ 4), so there is nothing to wire it to and no way to test what was wired.

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

**And quote it with its workload.** The 0.7–0.8% dirty figure is a small animated window on an otherwise still desktop. Measured 2026-08-05 over 999 live frames with a 700×500 window dragged across the screen while its text scrolled, the dirty fraction was **median 0.68, mean 0.74, max 1.00** — about 85× the headline figure. Read against the table above that is roughly 1.2×, and any frame past `DIRTY_AREA_LIMIT` (0.9) falls back to a full conversion outright. So the optimisation is worth 1.5× on incidental desktop animation and decays toward 1.0× under sustained drag or scroll — which is what a screen-share or agent-driving workload actually produces. Neither figure is wrong; the distribution is the honest answer, and a single number quoted without its workload will mislead whoever reads it next.

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
- **GRAY was by far the slowest colour mode; both halves are now fixed.** It used to be bimodal — a ~9.2 ms fast mode the CPU sustains for a second or two and a ~15 ms slow one — and because a capture loop never sees the transient, the honest figure was **13.7–14.9 ms**, filling ~95% of a 60 Hz frame. Two changes landed 2026-08-05, measured by `benchmarks/gray_kernel.py`:
  - **NumPy path: ~16 ms → 8.5–11 ms (1.5–1.8×), byte-identical.** The old formulation allocated a full-frame uint16 temporary per channel; those page faults cost more than the arithmetic. Reusing persistent intermediates removed them. This is what `pip install rapidshot` gets — no toolchain, no new dependency.
  - **Native kernel: → 0.70 ms (24× on the same machine), byte-identical.** `native/src/luma.rs`, exercised through `NumpyProcessor.convert_into` when the optional extension is present. GRAY now costs 4% of a 60 Hz frame instead of ~95%.
  Byte-exactness is asserted over all 2²⁴ BGR triples on both sides of the FFI boundary — it is what lets the accelerated path be swapped in without changing any consumer's pixels, and it is the one thing OpenCV's kernel cannot offer (off by up to 1 LSB, mean 0.13, because it rounds differently).
  - **AVX2 kernel: → 0.26 ms, byte-identical, 96% of the memory ceiling.** GRAY is now the *fastest* of the five modes, having been the slowest by an order of magnitude, and it beats single-threaded OpenCV (0.34 ms) while OpenCV is off by up to 1 LSB. **37× against the NumPy path, 59× against the 2.0.0 formulation.** § 3 records the `maddubs` saturation hazard that dictates the design and why the correctness test is exhaustive over all 2²⁴ triples rather than sampled.
  **Conversion is finished as an optimisation target.** All five modes sit at 69–100% of the 33.2 GB/s the memory system delivers; RGBA at 69% is the weakest and worth ~0.1 ms. `benchmarks/baseline.json` and the badges were re-recorded after this landed. GRAY's *NumPy* path remains duty-cycle sensitive (6.9–10.5 ms across runs in one session), so the suite flags it informational rather than gating on it — quote a range for that one.
- **CuPy/CUDA paths untested** — no NVIDIA GPU available. Worth a spike: `gfx2cuda` + CuPy may give a **pure-Python** D3D11→CUDA tensor path, which would mean CUDA users get GPU tensors with no build step at all.
- **`shot()` writes BGRA regardless of `output_color`**, and overruns an undersized buffer without bounds checks. Documented; not yet fixed.
- **`pipeline.cpu_to_nchw` is now the largest CPU cost in the baseline at 6.4–6.8 ms**, and it is untouched — bigger than any single colour conversion now that the native kernels have landed (§ 3). Stage 6's GPU path bypasses it, but every CPU-only consumer still pays it in full.
- **The staging map is 2.1 ms and it is pure GPU-wait, not work.** Re-profiled 2026-08-05, because the old stage table predates the native kernels and no longer ranks anything correctly. A `grab()` returning a frame is p50 **9.86 ms**, split: acquire + GPU copy 6.44 ms (mostly the 10 ms blocking timeout — its *min* is 0.12 ms), **map staging surface 2.17 ms**, read + convert 0.58 ms (BGRA) / 1.14 ms (RGB), unmap 0.006 ms. Conversion is now 6–11% of a frame; the map is 3.6× it.

  Inserting a delay between `CopySubresourceRegion` and `Map` collapses the map, which identifies the cost exactly:

  | delay before map | map p50 | fps |
  | --- | --- | --- |
  | 0 ms | 2.199 ms | 99.7 |
  | 1 ms | 0.805 ms | 99.5 |
  | 2 ms | **0.022 ms** | 99.8 |
  | 5 ms | 0.018 ms | 99.5 |

  Given 2 ms the GPU finishes the copy and `Map` returns in 22 µs — 100× faster. **But look at the fps column: it does not move.** The loop is already bounded by the compositor at ~100 Hz, so the map's 2.1 ms is absorbed by time the thread would otherwise spend blocked in acquire. Pipelining it away — double-buffered staging surfaces, mapping frame N-1 while the GPU fills N — would therefore buy **no throughput on this display**. It would buy back 2.1 ms of *calling-thread time per frame* (21% of the budget) for the consumer to use, and it would matter on a faster panel where 9.86 ms no longer fits the frame period.

  **Not built, deliberately.** The obvious implementation returns the previous frame, which is a semantic change to `grab()`, and the honest version needs a fence plus a restructured acquire/copy/map order that collides with the live-frame guard in § 5. Revisit when either a >144 Hz panel or a profile showing consumer starvation makes the 2.1 ms actually cost something.
- **No hosted docs.** Release automation exists as of 2.0.0 (§ 5); `GOVERNANCE.md` and OpenSSF Scorecard remain outstanding (§ 7).

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
