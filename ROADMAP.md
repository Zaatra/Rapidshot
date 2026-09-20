# Rapidshot Roadmap

**Goal:** be the capture layer the AI-agent/CV ecosystem reaches for on Windows — not a faster copy of DXcam.

This document is written to be read cold. It states where the project actually is, what to do next, and which questions are already settled so they are not re-litigated. Everything marked ✅ has been implemented and verified; see `CHANGELOG.md` for detail.

---

## 1. Start here

**Current state.** Rapidshot captures the desktop via DXGI Desktop Duplication and can hand a frame to a GPU consumer as a **model-ready NCHW float32 tensor that never touches the CPU**. The CPU path for the same work costs ~8 ms per 1080p frame on a toolchain-free install. Core capture is pure Python; an *optional* Rust extension provides GPU interop **and byte-exact AVX2 conversion kernels** — with it, colour conversion drops to 0.26–0.36 ms and stops being the dominant CPU cost (§ 3, § 10).

### Release status

**2.4.0 is tagged and published** (`v2.4.0` at `9d40b70`, 22 August 2026), following **2.3.0** (`v2.3.0` at `b3a01f2`, 21 August), **2.2.0** (`403849e`, 6 August), **2.1.0** (5 August) and **2.0.0** (4 August). PyPI Trusted Publishing and the `pypi` GitHub environment are configured and restricted to `v*` tags, so a pushed tag is what cuts a release; the full procedure is in `RELEASING.md`.

**2.4.0 is confirmed live**, which earlier releases in this list never were from inside the repository: PyPI serves 2.4.0 as the current version with both the wheel and the sdist, and the GitHub Release carries wheel, sdist and SBOM. The release ran from the tag with no manual step.

Note that `CHANGELOG.md` dates 2.3.0 as 2026-08-06, which is when the Machine B work below was done, not when it shipped — the tag is two weeks later. Every measurement in this document dated 2026-08-06 belongs to that session and is correct as written.

What each release delivered, in one line each — `CHANGELOG.md` has the detail:

| | |
| --- | --- |
| **2.0.0** | Pooled output (breaking), release infrastructure, `py.typed` |
| **2.1.0** | Native AVX2 conversion kernels for all five colour modes; conversion finished as an optimisation target |
| **2.2.0** | `to_nchw()`, public `timeout_ms` and `pool_size_frames` (−60 MB/camera), the cross-library comparison, and the `cpu_to_nchw` strawman correction |
| **2.3.0** | First release verified on NVIDIA hardware. CUDA interop for the GPU tensor; two shipped bugs fixed — `nvidia_gpu=True` returning wrong pixels for every mode but BGRA, and every `E_ACCESSDENIED` misreported as protected content; five "untestable" paths given real tests |
| **2.4.0** | The hybrid path end to end: capture on the iGPU, cross-adapter to the dGPU, CUDA consumer, no CPU in the pixels or the synchronisation. `transfer_async()` with a GPU-side shared fence, and `set_consumer_fence()` / `wait_for_consumer()` closing the buffer-reuse race — **28 of 60 frames wrong without the handshake, 0 with it** |

**All of them published. Checked against PyPI on 2026-09-13**, which is the authority the repository is not: 2.0.0, 2.1.0, 2.2.0, 2.3.0 and 2.4.0 each serve two files, and 1.1.0 is present and yanked as intended. This entry previously said the outcome could not be verified from the tree and asked for a manual UI pass; the tree is still not the authority, but the index is one query away.

**1.1.0 is yanked**, reason *Newer Version*. It was the only thing on PyPI from April 2025 to now and it did not work: it failed to import on Python 3.11+ (`cursor: Cursor = Cursor()` trips the dataclass mutable-default check broadened in 3.11), and patching that one line only got it to return all-black frames, because the processor was handed a texture where it expected a mapped staging surface. Yanking is not deletion — an existing `rapidshot==1.1.0` pin still resolves, which is the intent; only new unpinned installs are steered away.

Both of the related settings were checked the same way on 2026-09-13:

- **Private vulnerability reporting is enabled.** `SECURITY.md`'s link works.
- **`main` has no branch protection at all.** Not "missing Code Owners review" — the API returns `Branch not protected`, so there are no required reviews and no required status checks. `CODEOWNERS` is therefore only a routing hint, CI is advisory rather than blocking, and anything can be pushed straight to `main`. That is how 39 commits landed on it in one go on 2026-09-13. Worth deciding deliberately rather than by default: for a single-maintainer project it is a defensible choice, but the repository currently documents guarantees it does not have.

### Where 2.6 stands — 2026-09-14

**Read this before anything below it; several older statements in this section are now wrong, and are corrected here rather than silently.**

**§ 7.2's feature list is built.** `GpuConverter` (bilinear or nearest; FP32/FP16 in NCHW or NHWC; a resized BGRA8 frame; NV12 and P010; crop; multi-ROI batches in one dispatch; all four `DuplicateOutput1` input formats), `GpuTensor.to_torch()` / `to_cupy()` / `to_dlpack()`, `TensorTransfer` (convert first, then cross adapters), and `TensorStream`. The fused `.crop().resize()…` graph is not built, and § 7.2 says to build it only if it fuses. `CHANGELOG.md` `[Unreleased]` has the detail; § 7.2 below has the status per item.

**Building `TensorStream` found a correctness bug in paths that have shipped since 2.3.0.** `GpuPreprocessor12` and `CrossAdapterTransfer` sometimes read **the previous frame**: `AcquireNextFrame` returns when the capture device's copy into the surface is *submitted*, and a D3D12 queue could overtake it. 7–15 of 150 first reads were stale for the preprocessor, 65 of 150 for the transfer. Fixed for all three paths with a fence ordering the D3D12 queue behind the D3D11 copy on the GPU (`native/src/capture_order.rs`); 0 of 150 after, at no measurable cost to dispatch and ~0.02 ms per transfer. § 4 records the rule, § 10 the history. **This corrects two claims:** the § 6.1 "verified byte-exact" results compared the destination against a copy of *the same snapshot*, so they could not detect a stale read, and the instruction below not to touch D3D12 synchronisation "without a measurement" — the measurement arrived.

**Release gates for 2.6.** Everything above was verified on the Intel-only development machine (Core Ultra 5 235, Intel iGPU, WARP as its only second adapter). **Machine B — RTX 4060 laptop, Optimus, driver 616.92 — ran what it could on 2026-09-14**; each gate carries its outcome. Full suite there: 931 passed, 12 skipped, 0 failed.

1. **The CUDA exports** — `to_cupy()`, `to_torch()`, `to_dlpack()` — **were broken in a way no reasoning would have caught, are fixed, and are still not fully run.** See the box below. The device-by-LUID selection they turn on is now verified on real CUDA hardware, in both directions. The byte-equal export of a converted tensor is **still open**: it needs capture and CUDA on the *same* adapter, which Optimus never gives. Either flip this laptop's MUX to discrete, or use an NVIDIA desktop. `tests/test_gpu_tensor_export.py` runs the remaining three checks automatically wherever that holds.
2. **The stale-read fix on a hardware destination — met.** All three `tests/test_capture_order_shipped_paths.py` regressions pass on Machine B, and `TensorTransfer` reports `'Intel(R) UHD Graphics' -> 'NVIDIA GeForce RTX 4060 Laptop GPU'` with `destination_is_software == False`. **The D3D11↔D3D12 shared fence works on this Intel/NVIDIA driver pair** — `OpenSharedFence` did not refuse, so the fence fallback in § 10 is not needed to ship, though a driver that refuses still fails outright.
3. **The three non-BGRA input formats** need an HDR or 10-bit desktop; only `B8G8R8A8` has been exercised. **Still open** — Machine B's panel is 8-bit.

> **The CUDA exports had never worked, on any machine, and the error said otherwise.** `_device_for_adapter` matched adapters using CuPy's `getDeviceProperties()["luid"]`, which converts the fixed-size `char luid[8]` field as though it were a C string — so it stops at the first zero byte, of which a LUID almost always has several. The RTX 4060 reports `3234010000000000`; CuPy hands back three bytes. The comparison could therefore never succeed **on any adapter**, so `to_cupy()`, `to_torch()` and `to_dlpack()` raised `CrossAdapterRequired` unconditionally — including on a single-adapter NVIDIA desktop, where the advice it gives ("transfer the frame to the CUDA adapter first") is impossible to act on because the tensor is already there. Fixed by reading `cuDeviceGetLuid` from `nvcuda.dll`, which is what `examples/gpu_tensor_to_cupy.py` had always done correctly and what `TensorTransfer.destination_luid`'s docstring already told callers to pair with. **The library broke by re-implementing an example it documents itself as mirroring**, so `tests/test_gpu_tensor_export.py` now pins the two implementations against each other rather than only checking one; two of its tests fail against the shipped comparison. The lesson is § 5's: a path nobody has run is not verified, and an exception that names a plausible cause is the easiest way for a dead path to look alive.

**Region cameras: fixed on every GPU path.** `GpuConverter`, `GpuPreprocessor12` and the D3D11 `GpuPreprocessor` all converted the whole monitor on a region camera; all three now convert the region, with whole-frame output unchanged bit for bit (§ 10). This changes output for anyone using the two shipped preprocessors on a region camera — which is the fix.

**Next feature task:** § 6.3 — finish Stage 3 (Frame metadata). It is smaller again as of 2026-09-13: **timestamps are done** (`Frame.timestamp_qpc` and `Frame.timestamp`), **cursor data now reaches `Frame`** (`Frame.cursor` carries a `CursorInfo` with position, hotspot, shape bytes, shape type, size and pitch) with **one piece outstanding — `CursorInfo.position` is still in desktop coordinates**, which § 6.3 requires be translated to frame coordinates like `dirty_rects`, and **`Protocol`-typed interfaces are not started** — there is still no `Protocol` anywhere in the package. So § 6.3 is now two tasks: translate the cursor position, and design the `Protocol` interfaces. § 6.1 is complete: hybrid and headless systems are reported clearly, a captured frame crosses to a second adapter at **0.70–0.98 ms per 1080p frame** verified byte-exact, and the convert-first-or-transfer-first question **has been re-opened** (2026-09-13): the measurement that settled it in favour of transferring the frame omitted one of that ordering's costs and only tested the most expensive payload, and at 2560x1600 converting first wins at every size — see the box in § 6.1. **§ 6.1's validation on real hybrid hardware is now done** (2026-08-22, Intel→NVIDIA, byte-exact — see above), **and the asynchronous shared fence shipped in 2.4.0**, which was the last piece outstanding there. § 6.1 is closed in both directions: `transfer_async()` submits without blocking, `shared_fence_handle` lets a CUDA consumer wait on the GPU, and `set_consumer_fence()` / `wait_for_consumer()` close the reverse hazard where the producer overwrites a buffer the consumer is still reading.

**Do not start another round of D3D12 synchronisation work without a measurement or a user report asking for it.** That seam has been through feature work, hardware validation, adversarial review, four rounds of fixes, targeted regression tests and a clean final pass. The next thing to do here is nothing; the returns are elsewhere.

> **The measurement arrived, 2026-09-14, and the rule did its job.** Every round listed above ordered the *transfer* against its *consumer*. None ordered the *read of the capture surface* against the *capture itself*, and that is where the bug was: 65 of 150 first transfers carried the previous frame. It was found by a test that compared a stream's output with an independent conversion, not by reasoning about the seam — which is the argument for the rule, not against it. See the 2.6 box above and § 4.

**The next task is § 7.0 — build the AI-ingestion benchmark — and it comes before § 6.3.** § 6.3 (finish `Frame` metadata) is still worth doing and is now folded into § 7.1, but it is no longer the front of the queue. § 7 explains why: every remaining proposal is a performance claim, and the only benchmark this project has measures `grab()` — the path where RapidShot's advantage is smallest and where it loses the frame-rate column outright. Until *present → model-ready tensor* is measured, feature ordering is guesswork, and § 11 has been right about that twice already.

**The largest of those returns was distribution, and it is now built.** § 6.1's cross-adapter transfer, § 6.6's CUDA interop, the AVX2 kernels of § 3 and 2.4.0's synchronisation all sat behind an extension the consumer had to build, costing a Rust toolchain, MSVC build tools and `native/install_dev.py`. Every measurement in § 3 that makes this library worth choosing was one a `pip install` could not reach.

`rapidshot-native` closes that: `pip install rapidshot[native]` and the extension is present, no toolchain. One **abi3** wheel covers Python 3.9 and every later version, which is what makes this a single artefact rather than a matrix that grows with each CPython release — the crate already carried `pyo3/abi3-py39`, so the work was packaging rather than porting.

It is a **separate distribution on a separate tag**, published by `release-native.yml`. `rapidshot` stays `py3-none-any`, so the guard that fails the main release if a compiled artefact appears in it (§ 10) is preserved rather than negotiated with. The two version independently: the Rust moves on its own schedule, and lockstep would republish an identical binary under a new number on every RapidShot release.

**Published 2026-09-13: `rapidshot-native` 0.1.0 is on PyPI**, tagged `native-v0.1.0`, as `rapidshot_native-0.1.0-cp39-abi3-win_amd64.whl`. Confirmed by installing it from PyPI into a clean environment: it imports, and `rapidshot.native.build_info()` reports `source: "rapidshot-native wheel"`.

**It does nothing until `rapidshot` 2.5.0 ships, and that is the next step.** Wheel discovery lives in `rapidshot.native`, which PyPI's 2.4.0 predates, so today `pip install rapidshot rapidshot-native` yields `is_available() == False`. The first dry run of `release-native.yml` failed on exactly this and it was the workflow's fault, not the wheel's: it installed `rapidshot` from PyPI to prove discoverability, which tests a new wheel against whatever discovery code was last released. It now installs the checkout instead (`--no-deps`, still run from `RUNNER_TEMP`, so the repository never reaches `sys.path` and the wheel is still what gets imported).

`rapidshot-native` remains deliberately absent from the `all` extra. Adding it only takes effect on a `rapidshot` release, so it rides with the next one rather than needing its own; `RELEASING.md` records the ordering.

**Hardware changed on 2026-08-06, and with it the state of the project.** A second development machine with an **NVIDIA RTX 4060** joined, and things this document called untestable are now tested. The headline: **RapidShot has a real GPU consumer for the first time** — capture → GPU tensor → `cupy.ndarray`, no CPU round-trip, verified byte-exact (§ 6.6).

In the same pass it confirmed the BGRA-swizzle rule on a second vendor's driver (§ 4), ran the eight CuPy tests that had always skipped (§ 5), exercised cross-adapter transfer with a discrete GPU as source (§ 6.1), found that NVIDIA lacks cross-adapter row-major texture support (§ 6.1), and corrected two benchmark claims this document was making (§ 10). It also introduced a new way to get numbers wrong — see the P-core pinning gotcha in § 2.

**It found one real bug, and it was a bad one.** `create(output_color="RGB", nvidia_gpu=True)` returned a 4-channel BGRA array and reported success: the CuPy processor converted colour through OpenCV, which is not a dependency, and swallowed the resulting failure. Invisible on Machine A because the path could not run there, and untested because the processor had no tests. Fixed, with the byte-exactness tests that should have existed (§ 10).

**Five paths this document called untestable were not**, and the second bug came out of testing them. Protected content, the `DuplicateOutput` refusal, real exclusive fullscreen, the access-loss rebuild, and the `process()` cache-miss branch all now have real tests (§ 5, § 10). In every case the entry had assumed the most obvious trigger was the only one — DRM content, an actual game, an actual device reset — and a cheaper one existed: a window display-affinity flag, a non-input desktop, a ctypes swapchain, a camera teardown.

Reaching the refusal branch is what exposed the second bug: **every `E_ACCESSDENIED` was reported as protected content**, so a locked workstation, a UAC prompt and a Session 0 service all advised closing a protected player window that did not exist. Worth remembering the next time something here is written off: **"untestable" is a claim about imagination at least as often as about hardware, and the paths nobody can test are where the bugs are.**

**Machine B ran as a real Optimus system on 2026-08-22, and § 6.1 is closed.** The MUX was switched from discrete-only to Optimus, which took most of a day to get working and produced more than the verification it was chasing.

**The verification, first.** `examples/verify_cross_adapter.py` transferred **5 captured frames from the Intel iGPU to the RTX 4060**, 16,384,000 bytes each at 2560×1600, every one byte-exact against a source-side readback. `probe_cross_adapter()` reports `representative: true` with a hardware adapter on both ends for the first time. This is the configuration § 6.1 was written for and had never run on.

**What took the day was one setting, and it is worth naming loudly** because nothing else moved the needle and several plausible things were tried and ruled out:

> **NVIDIA Control Panel → Manage 3D Settings → Preferred graphics processor → Integrated graphics.**

Until that was set, the NVIDIA driver claimed the display output while the firmware said Optimus, and **every adapter — Intel, NVIDIA and even WARP — refused `DuplicateOutput` with `DXGI_ERROR_UNSUPPORTED`**. Capture did not work at all. § 2 records the full elimination and the diagnostic signals; the short version is that the firmware was correct throughout and the Windows-side graphics stack was not following it.

**The library was wrong too, and that was worth more than the verification.** In that broken state RapidShot assumed the display-owning adapter was the one that could duplicate, discarded render-only adapters so `prefer_integrated` could not reach them, printed a raw HRESULT that named neither cause nor fix, and had `topology_info()` assert "Capture runs on {adapter}" about an adapter that demonstrably could not. All four are fixed (§ 5, § 10). None would have been found without a machine in a state nobody had seen.

**Before changing anything performance-related**, read § 3 (measured baseline) and § 4 (settled questions). Several intuitive-sounding optimisations have already been measured and rejected.

> ### Four performance items, measured on Machine B — 2026-09-15
>
> Measured here rather than on Machine A, because three of the four are about
> paths Machine A cannot reach: a discrete GPU, and a display that pads its
> surface pitch. **Three were real and are fixed; one was measured and left
> alone**, which is the outcome this section exists to make respectable.
>
> | item | before | after |
> | --- | ---: | ---: |
> | `shot()` on a padded surface, 1600p | 1.833 ms | **1.036 ms** |
> | CuPy `GRAY`, 1600p | 0.629 ms | **0.068 ms** |
> | `TensorStream` waiting on a still screen | 10,302,950 calls/s | **1,804 calls/s** |
> | `grab(timeout_ms=0)` idle cost | 4.3x the default | *unchanged* |
>
> **The one left alone is the interesting one.** `timeout_ms=0` costs a full
> core on a still screen, which looks exactly like the `TensorStream` spin
> beside it. It is not the same thing: the documentation already says "0 polls,
> which costs roughly 4x the CPU for about 7% more frames", and Machine B
> measured **4.3x the calls for 6% more frames** -- the claim is accurate and
> the default is already the blocking one. A caller who asks for polling is
> asking for precisely that trade, and quietly inserting a sleep would be
> overriding an explicit request. `TensorStream` was different because nobody
> asked *it* to spin: the spin was emergent, from a wrapper whose loop had no
> pacing of its own once the call beneath it stopped blocking.
>
> **The GRAY result also needs its end-to-end figure stated, not just its
> microbenchmark.** The kernel is 6-9x faster; GRAY capture on this machine
> went 162.8 to 164.8 fps, about 1%, because `grab()` is capped by the 165 Hz
> panel. The win is headroom, not frame rate, and a table showing only the
> 9x would be the kind of claim § 11 keeps having to correct.


---

## 2. Working on this project

### Environment

- **Windows only.** DXGI Desktop Duplication has no cross-platform equivalent.
- **Two Python interpreters may be on PATH.** Check `sys.executable` before concluding a dependency is missing. Dev dependencies (`comtypes`, `numpy`, `pytest`) live in the 3.13 install.
- **Native extension is optional.** `pip install rapidshot` never needs a toolchain. Building it needs Rust plus the MSVC C++ build tools, and `cargo` is often not on a fresh shell's PATH — prefix with `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"`.

**There are now two development machines, and every number in this document belongs to one of them.**

| | **Machine A** — original | **Machine B** — added 2026-08-06 |
| --- | --- | --- |
| GPU | Intel iGPU, no NVIDIA | NVIDIA RTX 4060 Laptop (8 GB) |
| CPU | Intel Core Ultra 5 235, **6 P-cores + 8 E-cores** | Intel i9-14900HX, **8 P-cores + 16 E-cores** |
| Display | 2× 1920×1080 | 1× **2560×1600** |
| Topology | single (iGPU) | single (dGPU) — see below |
| CUDA / CuPy | untested, no NVIDIA GPU | CuPy 14.1.1, CUDA 13.2, driver 595.95 |

Everything in § 3 and most of § 10 was measured on **A**. Machine B has its own recording (`benchmarks/baseline-rtx4060.json`) and its own hazards, both below.

### Vendor and configuration coverage

**Nothing in the library branches on GPU vendor.** `util/topology.py` says so where the vendor IDs are defined — *"for human-readable reporting only. Never branch capture behaviour on these: vendor is not a capability"* — and a grep for vendor comparisons in the capture and classification paths returns nothing. The native extension contains no vendor-specific code either; "CUDA" appears in it only in comments naming who may consume the shared handle, and the D3D12 device is created on whatever adapter the capture texture is already on.

So the matrix below is about what has been **run**, not about what is supported by design. Silence here reads as "supported" if nobody writes it down:

| Configuration | State |
| --- | --- |
| Intel iGPU, single adapter | Verified — Machine A |
| NVIDIA dGPU, single adapter, extension built | Verified — Machine B, extensively |
| NVIDIA dGPU, single adapter, **no extension** | Verified 2026-08-06: 269 passed, 44 skipped, no failures |
| **Any AMD GPU** | **Never run.** No AMD hardware has touched this project |
| **Intel + NVIDIA Optimus** | **Verified 2026-08-22 on Machine B**, and again 2026-09-14 for the 2.6 paths: 931 passed, 12 skipped, 0 failed, driver 616.92. Capture on the iGPU, cross-adapter transfer to the dGPU byte-exact; `TensorTransfer` and the capture-ordering fence both hold against the discrete GPU. Needed one NVIDIA control-panel setting to reach; see above |
| **AMD switchable hybrid** | **Never run.** No AMD hardware has touched this project |
| Headless / IDD virtual display | Never run — § 6.2 |

**Machine B has now run as a genuine Optimus system, and getting there took a day. The setting that mattered is one line; everything else is the elimination that found it.**

**The fix.** On a hybrid laptop where the firmware says Optimus but capture fails, set:

> **NVIDIA Control Panel → Manage 3D Settings → Preferred graphics processor → Integrated graphics**

Then re-check `topology_info()`. Correct Optimus has the **iGPU owning the output and the dGPU at zero**:

```
Topology: hybrid
  Adapter[0] (Intel(R) UHD Graphics) (Intel) (128MB VRAM) (1 output)
  Adapter[1] (NVIDIA GeForce RTX 4060 Laptop GPU) (NVIDIA) (7956MB VRAM) (0 outputs)
  Adapter[2] (Microsoft Basic Render Driver) (Microsoft) (0MB VRAM) (software) (0 outputs)
```

If those are the other way round — dGPU holding the output, iGPU at zero — the machine is in the broken state below regardless of what the firmware says.

**The broken state, so it is recognisable.** Every adapter refuses `DuplicateOutput` with `DXGI_ERROR_UNSUPPORTED` (0x887A0004), **including WARP**. That last part is the diagnostic: a merely suboptimal routing leaves one adapter capturable, so all three refusing means nothing is composing a desktop DDA can attach to. Corroborating signals, both absent in the broken state and both worth checking: `EnableMsHybrid` unset on both adapter class keys, and `OptimusProfilesInstalled: 0x0` under `nvlddmkm`.

**What was ruled out, each by measurement rather than argument.** Recorded because every one of these is a plausible-sounding fix that does nothing, and the next person will be tempted by the same list:

| Suspected | Tested by | Result |
| --- | --- | --- |
| Stale Intel driver | updated 32.0.101.5542 → 32.0.101.7088 | no change |
| Firmware not really in Optimus | BIOS `Display mode` **and** PredatorSense `GPU Operating` | both already correct |
| Non-console session | `query session`, `describe_desktop_access()` | console, `is_input=True` |
| DWM not composing | process check | running in session 1 |
| DPI virtualisation | DPI-aware process reads 2560×1600 | refused identically |
| Stale NVIDIA driver config | **clean** reinstall, "perform clean installation" ticked | no change |
| NVIDIA install order | installed into a working iGPU-owned arrangement | reclaimed the display anyway |

**Two controls set the MUX and they are separate**: BIOS `Display mode` (Optimus / Nvidia GPU only / Auto Select) and PredatorSense `GPU Operating`. Both must say Optimus. Earlier revisions of this document named one or the other as *the* setting; both exist, and both being correct is what ruled the firmware out.

**Removing the NVIDIA driver entirely also works, and is the diagnostic that proved causation.** With it uninstalled the iGPU took the output immediately and capture worked (299 passed, 0 failed). Reinstalling it took the display back. That established the NVIDIA driver as the cause before the actual setting was found — useful as a bisection step, not as a fix, since it costs CUDA.

**Do not read the broken state as "RapidShot does not work on Optimus."** It works: § 6.1 is verified on this machine. The refusal was the operating system's. What it fairly establishes is that RapidShot *handled* the situation badly, which is § 10's adapter-selection entry.

**What is genuinely NVIDIA-only, and it is one thing:** CuPy, because CUDA is. On an AMD machine `import cupy` fails, `CUPY_AVAILABLE` is False, and `nvidia_gpu=True` warns and falls back to the CPU processor — that path already exists in `capture.py`. `examples/gpu_tensor_to_cupy.py` is NVIDIA-only for the same reason.

**AMD consumers are not shut out of the GPU tensor.** `GpuPreprocessor12` runs on D3D12 and produces an `ID3D12Resource` on whatever adapter captured the frame; the consumer-side route is **DirectML** via `output_resource_address`, which is what § 8 and the README already document and is vendor-neutral. CUDA import is one of two consumers, not the only one.

**The unknowns on AMD, stated precisely.** The BGRA swizzle rule (§ 4) is verified on Intel and NVIDIA drivers and unverified on a third — low risk, since it follows from the DXGI format rather than driver discretion, but it is the one rule a silent mismatch would corrupt rather than crash. And AMD's cross-adapter capability flags are unknown; the buffer path was chosen precisely so nothing depends on them, a choice NVIDIA has already vindicated by not supporting cross-adapter row-major textures at all (§ 6.1).

**Machine B is an Acer Predator PHN16-72 with a MUX switch, set to Optimus since 2026-08-22**, so the Intel iGPU drives the display and the RTX 4060 owns no output — the genuine hybrid topology § 6.1 needs, and `topology_info()` reports `hybrid`. It spent the preceding fortnight in **discrete-only**, where the iGPU is disabled at firmware level and absent from Device Manager entirely, making it a *single-adapter NVIDIA* system that reported `single`. Switching between the two is a PredatorSense setting plus a reboot. Measurements dated before 2026-08-22 were taken in discrete-only; benchmark recordings keep the two apart by the `gpu` they record, which is why `baseline-rtx4060.json` and `baseline-rtx4060-hybrid.json` refuse to gate against each other. The two modes are separately useful: discrete-only puts capture and CUDA on one adapter with no cross-adapter step in the path, which is the cleaner place to prove an end-to-end GPU consumer; hybrid is the only place § 6.1 can be validated.

**Two toolchain traps cost time on Machine B, both worth not rediscovering:**

- **Smart App Control blocks the native build outright.** Enforced SAC (`VerifiedAndReputablePolicyState = 1`) refuses to execute pyo3's freshly compiled build script — `error: failed to run custom build command for pyo3 ... An Application Control policy has blocked this file. (os error 4551)`. Its user-mode enforcement covers DLLs too, so the resulting `.pyd` would also be blocked at import, and building elsewhere and copying the artifact over does not help. SAC has no exclusion list. The only route to the extension is turning it off — which **cannot be undone without reinstalling Windows**. Worth noting that this is a real machine on which `pip install rapidshot` works fine and the extension cannot be built at all, which is § 11's "optional means optional" earning its keep rather than a hypothetical.
- **`pip install cupy-cuda13x[ctk]` silently installs no headers.** The `[ctk]` extra requests `cuda-toolkit[cudart,nvrtc,...]==13.*`, and `cuda-toolkit` 13.3.1 no longer provides those extras — pip prints them as warnings, succeeds, and CuPy then fails at the first JIT with `Failed to find CUDA headers`. Pin it instead, and note the wheels are named `nvidia-cuda-runtime`, **not** `nvidia-cuda-runtime-cu13` (that name exists on PyPI as an empty 0.0.1 placeholder, which is a good way to conclude the package is missing when it is not):

  ```bash
  pip install "cuda-toolkit[cudart,nvrtc,cublas,cufft,curand,cusolver,cusparse]==13.2.*"
  ```

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

The suite pins itself to the performance cores on a hybrid CPU and says so; pass `--no-pin` to opt out, and read the gotcha below before doing that.

```bash
python benchmarks/ab_conversion.py
```

### Testing gotchas that will waste your time otherwise

- **Live capture tests need screen activity.** Desktop Duplication only reports *changed* content, so an idle screen produces zero frames and tests fail for reasons unrelated to the code. For tests that need a *run* of changed frames, use the `motion` fixture in `tests/conftest.py`, which drives `benchmarks/motion_source.py` and yields a rectangle inside its window. It is module-scoped on purpose — the window is topmost and would change what every later test captures.
- **A frame held for a whole test module hides every race with the capture.** Reading a frame long after acquiring it gives the capture device's copy into the surface all the time it needs, so a GPU read that would overtake it on a live stream never does. Every converter test before 2026-09-14 used one fixture frame per module, which is how `GpuPreprocessor12` and `CrossAdapterTransfer` shipped reading the previous frame 5–43% of the time with a green suite. Test a GPU path's **first read of a new frame, immediately after acquisition, against a moving source**, and compare it with a *settled re-read of the same held frame* — never with a second copy of the same read, which is stale in the same way (§ 10). Build the object under test on a frame you do not check: construction is slow enough to let that frame settle.
- **Synthetic textures cannot test the D3D12 path.** D3D11 refuses `SHARED_NTHANDLE` without `SHARED_KEYEDMUTEX`, and a keyed-mutex resource reads as zeros until acquired — on *both* APIs. The real duplicated surface has its mutex managed by DXGI. Use live capture. Constructing `GpuPreprocessor12` over a `TestTexture` fails at the constructor with `D3D12 preprocessor setup failed: texture is not shareable ... (0x80070057)`, which is the guard working as designed — but it means **`baseline.json` has no D3D12 row at all**, and any D3D12-versus-D3D11 comparison has to be run over one live frame. See § 10.
- **Benchmark noise is severe on a loaded machine.** Naive comparison once reported 11 false regressions up to 1.9× on *identical* code. The suite compensates with pooled rounds, minimum-sample comparison, and a control benchmark; run `--self-test` to measure the current noise floor before trusting any result.
- **On a hybrid P-core/E-core CPU, pin the benchmark process to the P-cores or the numbers are meaningless.** Measured 2026-08-06 on Machine B (8 P-cores, 16 E-cores) with `--self-test`, which compares the suite *to itself with no code change*:

  | | Unpinned | Pinned to P-cores |
  | --- | --- | --- |
  | Rows exceeding the 1.30× threshold | 2 | **0** |
  | Worst false verdict | `SLOWER 2.57×` | every row `~ same` |
  | Spurious `FASTER` verdicts | 3.20× on `gpu_dispatch` | none |
  | `pipeline.cpu_to_nchw` across runs | 5.5 – 16.0 ms | 5.06 – 5.31 ms |

  Windows moves benchmark threads onto E-cores under no particular provocation, and an E-core reads as a 2–3× regression on exactly the compute-bound rows that matter. **The control benchmark does not rescue this** — `control.memcopy` reported "machine state comparable, 1.01×" in the same run that called `shot.RGB` 2.57× slower, because the control got scheduled well and the others did not. That is the § 3 lesson about one control standing in for workloads it does not resemble, arriving by a new route.

  **Machine A is hybrid too — this document said otherwise until 2026-09-14.** It is an Intel Core Ultra 5 235: 6 P-cores and 8 E-cores, `performance_core_mask()` returning `0x3c03` (6 of 14 logical processors). The earlier claim that Machine A had no E-cores was never measured; it was inferred from the fact that the effect had not been *seen* there, and what actually explains that is when the field arrived — `baseline.json` and `baseline-nonative.json` were recorded 2026-08-05, a day before `cpu_topology` existed, so both recorded `None` and neither could report the machine was hybrid. An unpinned Machine A recording is therefore subject to exactly the same 2–3× scheduling noise as Machine B, and the 2.1.0 recordings should be assumed to carry it.

  **`perf_suite.py` now pins itself**, so a bare invocation is correct again. It reads `EfficiencyClass` from `GetSystemCpuSetInformation`, restricts the process to the highest class when a machine has more than one, and prints what it did. A uniform CPU is left alone. The recording carries `cpu_topology`, `pinned_to_performance_cores` and `affinity_mask` in its `machine` block, because a pinned and an unpinned recording are not comparable and nothing in the numbers alone distinguishes them. `--no-pin` disables it. Leaving this to the invocation was the wrong default: a suite that silently produces 2.5× noise unless the caller remembers a `start /affinity` prefix is a suite that teaches people to ignore it.

  **Check the `machine` block after recording.** That provenance shipped broken twice in one sitting — first because `GetProcessAffinityMask` needs explicit `argtypes` (without them ctypes coerces the process pseudo-handle to a C int and raises `OverflowError`, which a broad `except` then swallowed into a silent `None`), and then because `machine_info()` was called *before* the pin, so it faithfully recorded the state the process started in rather than the one it benchmarked in. Both produced a recording that looked complete and described the wrong run. The `except` now records `cpu_topology: "unknown (<error>)"` rather than dropping the keys, since a missing field is indistinguishable from a recording made before the field existed.
- **Pace benchmarks to a frame period, never to a fixed gap or a burn loop.** Sustained heavy vector work holds the CPU in a lower power state, and GRAY has two modes because of it — **16.27 ms back-to-back, 9.16 ms with a 16 ms gap, 9.91 ms in bursts with a 200 ms gap.** The trap is that *both* extremes are wrong. A benchmark's real duty cycle follows from its own cost: RGB takes 1.8 ms of a 16.7 ms frame (~11% duty cycle, mostly idle) while GRAY takes 15.9 ms (~95%, effectively sustained). `perf_suite.py` therefore sleeps out the remainder of a 60 Hz frame after each rep, which reproduces both from one rule; a fixed gap handed GRAY a 50% duty cycle and reported a number no capture loop achieves. The memcpy control cannot catch any of this — memcpy is not heavy enough to trigger it, so it reports "machine state comparable" throughout.
- **Re-record `baseline.json` whenever the harness changes how it drives benchmarks**, and verify immediately with a second run that should read all `~ same`. Numbers from different pacing models are not comparable, and after the fact you cannot separate a harness change from machine drift.
- **The control benchmark only rescues comparisons that resemble it, and cannot rescue one across machines at all.** `control.memcopy` measures memory bandwidth, so dividing by its movement normalises benchmarks that are *also* bandwidth-bound and quietly mis-normalises everything else. On a CI runner this reported `pipeline.cpu_to_nchw` — float32 resize/normalise/transpose, compute-bound and sensitive to vector width and NumPy version — as a **1.34× regression against a code path nobody had touched**, while simultaneously calling every conversion row **1.4× faster on a machine that was uniformly slower**. One control standing in for workloads it does not resemble, wrong in both directions at once. `print_comparison` now detects a baseline recorded on different hardware (processor / platform / GPU), marks every verdict *indicative*, and gates nothing; a spurious improvement is flagged as loudly as a spurious regression, because nobody investigates good news. To compare code against code, re-record on the machine you are testing on.
- **CI cannot verify live capture.** GitHub runners have no desktop session. Those tests skip themselves and must be run on real hardware before a release.

---

## 3. Measured baseline

All figures 1920×1080 BGRA (8.3 MB/frame), measured on **Machine A** (§ 2) unless a row says otherwise. Stored in `benchmarks/baseline.json`, **re-recorded 2026-09-14 at 2.5.0**, P-core-pinned, extension built. The recording it replaces was made 2026-08-05T12:14Z at **2.1.0**, and it had stopped gating: it predates the `cpu_topology` / `pinned_to_performance_cores` provenance, so on a hybrid CPU `print_comparison` could not tell how it had been scheduled and correctly declined to gate anything — every verdict printed *indicative only*. A gate that always passes is worse than no gate, which is § 2's own argument for `--compare auto`. Earlier recordings are kept as `benchmarks/baseline-2026-08-05.json`, `benchmarks/baseline-nonative-2026-08-05.json` (both 2.1.0), `benchmarks/baseline-2026-07-30.json` and `benchmarks/baseline-2026-07-27.json`; do not compare across recordings casually — the 07-27 one drove the benchmarks differently, and the notes below apply to the current one.

**There are two committed recordings, and which one you want depends on the question:**

| File | Extension | What it answers |
| --- | --- | --- |
| `baseline.json` | **built** | What the library can do on stated hardware. Feeds the README badges. |
| `baseline-nonative.json` | absent | What `pip install rapidshot` gets, and what CI compares against — CI has no toolchain. |
| `baseline-rtx4060.json` | **built** | Machine B, P-core-pinned, live rows included. **Not** a replacement for `baseline.json` and not wired to anything. |

Both were recorded back-to-back on 2026-09-14 with `--rounds 5 --reps 25`, the invocation § 2 documents, so they are directly comparable to each other rather than separated by machine drift. The native recording was verified immediately by a second run, which read `~ same` on every synthetic row. Three consequences worth knowing:

- **CI's compare step points at `baseline-nonative.json`.** Aimed at `baseline.json` it would report a 6–20× "regression" on every conversion row forever, since the runner builds no extension — which is how a benchmark suite teaches people to ignore it.
- **`pipeline.gpu_dispatch` and `pipeline.gpu_plus_readback` appear only in `baseline.json`**; they need the extension.
- The **live rows were recorded against a defined synthetic workload**: `benchmarks/motion_source.py --fps 30` — a **900×700** window of animated bars, every pixel of it changing each update, positioned at +200+120. Earlier revisions of this document called it *a 420×300 window moved at ~30 Hz*; `motion_source.py` has been 900×700 since it was introduced in `aa20dc8`, so that description matched no version of the script and understated the dirty area considerably. See the dirty-fraction note in § 6.3 — a fully-redrawn 900×700 region is nearer the *unfavourable* end of that distribution than the old wording implied, and live rows recorded before 2026-09-14 are not comparable to later ones on workload grounds alone.
- **`--fps` matters and its default is wrong for this purpose.** `motion_source.py` defaults to *uncapped*, which on this machine drove `live.grab_with_frame` to 5.1–7.0 ms against 1.6–2.9 ms at `--fps 30` — a 2.9× swing that reads as a regression and is nothing but the workload. Always pass `--fps 30` when recording a baseline.

Use `python benchmarks/compare_recordings.py` to diff any two stored recordings; it normalises each by its own control row before quoting a ratio.

### Machine B, for contrast — synthetic rows only

Recorded 2026-08-06 with `--synthetic-only --rounds 5 --reps 25`, P-core-pinned, extension built, **after** the § 10 fixes to `read_back` and the D3D12 dispatch. **Do not read this as a version-to-version comparison**: it is a different CPU, a different GPU, and `frame` is still the suite's synthetic 1920×1080 even though the panel is 2560×1600. Minimum times, since those are what § 3 treats as robust:

| | Machine A (Intel) | Machine B (RTX 4060) | |
| --- | --- | --- | --- |
| `control.memcopy` | 33.2 GB/s | **28.2 GB/s** | comparable memory systems |
| `convert.RGB` | 0.31 ms | **0.198 ms** | native kernels, both |
| `convert.GRAY` | 0.26 ms | **0.252 ms** | identical |
| `convert.BGRA` | 0.22 ms | **0.236 ms** | the straight-copy control |
| `pipeline.cpu_to_nchw` | 3.91 ms | **3.07 ms** | different code — see below |
| `pipeline.gpu_dispatch` | 1.6 µs | **1.3 µs** | D3D11 submit only — see § 10 |
| `pipeline.gpu_plus_readback` | 14.80 ms | **2.20 ms** | after the `read_back` fix — see § 10 |

**Two rows moved for reasons that are not hardware, and neither is comparable across the machines.** Machine A's 14.80 ms readback is mostly Python list construction, which § 10 removed. And `cpu_to_nchw` changed *implementation* after this table was first written: the gather that was 73% of it is now 2.5× faster and byte-identical, so Machine B's 3.07 ms is a different computation from Machine A's 3.91 ms. Re-recording Machine A would move both the same way. That is the general hazard of this table, stated twice concretely — **a cross-machine table ages badly precisely because the code under it keeps improving.**

### Machine B, live rows

Recorded 2026-08-06 with `--rounds 3 --reps 25 --live-seconds 3`, against `benchmarks/motion_source.py` — the same defined workload class Machine A used, so the *method* is comparable even where the numbers are not.

**Quote these as ranges over six back-to-back runs, not as the single figure in the recording.** Minimums, with the motion source running continuously throughout:

| | Machine A | Machine B, six runs | spread |
| --- | --- | --- | --- |
| `live.grab_with_frame` | 2.41 ms | **3.10 – 3.74 ms** | 1.2× |
| `live.grab_frame_gpu` | 0.16 ms | **0.099 – 0.156 ms** | 1.6× |

So `grab_frame()` is if anything slightly *faster* here than on Machine A, and `grab()` is 1.3–1.6× slower for 2.0× the pixels. Both unremarkable, which is the finding.

**Getting there required being wrong first, and the mistake is more useful than the numbers.** The first recording reported `live.grab_frame_gpu` at **1.26 ms** — 8× Machine A — and an earlier revision of this section stated that gap as a finding and offered three candidate explanations for it. The next recording read 0.115 ms. Six controlled runs then put the honest range at 0.099–0.156 ms, and the 1.26 ms was simply an outlier from a run whose screen activity differed.

This is the trap § 3 documents at length, walked into in the same session that quoted it. Two guards worth extracting:

- **A single live recording is not a measurement**, and no amount of care about *how* it was recorded fixes that. Repeat it, or quote nothing.
- **A plausible explanation is not evidence.** All three candidate causes were reasonable, and reasoning about them felt like progress. Re-running took four minutes and answered the question that reasoning could not.

One genuine sub-finding: with the motion source running *continuously*, the spread on `grab_frame_gpu` is **1.6×, not the 4.5× § 3 records** for Machine A. That earlier figure was taken across recordings made at different times, so part of what it measured was varying screen activity rather than measurement noise. A controlled source narrows this row considerably — which means § 3's "the recording stands as taken" advice is right, but the recordings can be made less lossy than they were.

Two things are worth noticing rather than acting on. The conversion kernels land within noise of Machine A despite entirely different silicon, which is what a memory-bandwidth-bound kernel at 80–100% of the ceiling should do. And `pipeline.cpu_to_nchw` is **24% slower on the faster machine**, which nobody has explained; it is not library code (§ 10), so it is a curiosity rather than a defect, but it is exactly the kind of gap that gets mistaken for a regression by someone comparing across machines — which § 2 forbids for this reason.

| Stage of the loop | Cost/frame |
| --- | --- |
| Python → COM binding overhead (~6 calls) | **0.003 ms** |
| `CopySubresourceRegion` (GPU) | 0.016 ms |
| Read from mapped staging surface | 2.27 ms |
| Pixel conversion, RGB/BGR (NumPy, post-optimisation) | 1.84–1.88 ms — 0.29 ms native, see below |
| Pixel conversion, GRAY (NumPy) | 11.05 ms — was 13.7–14.9; 0.26 ms native, see § 10 |
| Preprocess for a model (resize/normalise/CHW → 640×640) | **3.42–3.66 ms** — was 6.23, see below |
| **CPU total, capture → model input** | **~8 ms** without the extension |

Conversion and preprocess rows re-measured 2026-09-14 at 2.5.0; the staging-read and COM rows are unchanged from 2026-08-05 and were not re-measured.

The conversion row is the one the optional extension changes, and it changes it by 7–48×. **With the extension the total is no longer conversion-dominated** — it is the staging map plus the preprocess, neither of which the kernels touch. § 10 re-profiles a real `grab()` on that basis.

Capture path comparison, real capture, from the two committed recordings:

| Path | `baseline.json` (built) | `baseline-nonative.json` | |
| --- | --- | --- | --- |
| `grab()` — CPU staging read + convert | 2.86 ms | **2.28 ms** | **inverted — see below** |
| `grab_frame()` — texture stays on GPU | 0.22 ms | 0.23 ms | **see the caveat below** |

**In the 2026-09-14 pair this comparison inverted: `grab()` recorded *faster* without the native kernels than with them.** That is not possible as a code result — the extension removes 1.6 ms of conversion from exactly this path — and it is not a hardware result either; both recordings were made within two minutes of each other with the same motion source running. It is the live-row spread below, arriving on the one table the spread most easily destroys. The 2026-08-05 pair happened to read 2.41 ms against 5.93 ms and was quoted as evidence of the extension's effect; that reading was luck, and the same table taken again does not reproduce it.

**Use the synthetic `shot.*` rows for this question instead** — same staging read, same conversion, deterministic input, and they separate cleanly by 6–8×:

| Mode | `baseline.json` (built) | `baseline-nonative.json` | Gain |
| --- | --- | --- | --- |
| `shot.RGB` | **0.303 ms** | 1.948 ms | 6.4× |
| `shot.BGR` | **0.302 ms** | 1.962 ms | 6.5× |
| `shot.RGBA` | **0.361 ms** | 2.603 ms | 7.2× |
| `shot.GRAY` | **0.261 ms** | 10.989 ms | 42.1× |
| `shot.BGRA` | 0.227 ms | 0.195 ms | — (no conversion; the control) |

**Neither live figure should be quoted without this caveat, and both are unreliable.** Across six recordings in a single session, on code that only ever got faster, the minima ranged:

| Row | Observed range | Spread |
| --- | --- | --- |
| `live.grab_with_frame` | 1.65 – 4.53 ms | 2.7× |
| `live.grab_frame_gpu` | 0.17 – 0.77 ms | 4.5× |

`grab_frame()` does no conversion at all — the texture never leaves the GPU — so its 4.5× spread is *purely* measurement, not code. The honest figure for it remains **0.16–0.21 ms**, which the current recordings and five of those six agree on. (An earlier revision of this section quoted 4.27 ms and 0.77 ms, because the then-current `baseline.json` held the top of both ranges. It no longer does — which is luck, not improvement, and is exactly why the caveat outlives the numbers.)

Three traps sit behind this, all worth keeping:

- **Live cost tracks the screen, not the library.** Region-limited conversion only converts the dirty part, so `grab()` depends on what moved. § 6.3 records the same effect from the other side: median dirty fraction was 0.8% for a small animated window and 68% for a dragged one.
- **A minimum is monotonically non-increasing in sample count.** Raising `--live-seconds` from 3 to 15 moved `grab()` from 3.08 to 1.65 ms on unchanged code — more samples simply find a luckier frame. So live rows are comparable only at identical `--live-seconds`, and "sample longer" makes the number flattering rather than truer. Use the default.
- **Re-recording until the number looks good is cherry-picking**, and it is tempting precisely because the spread is this wide. The recording stands as taken.

**The `grab`/`grab_frame` badges inherit all of this, and that is a known defect.** The `BGRA→RGB` badge was added because `convert.*` rows are synthetic and deterministic: it moves when the library changes and not otherwise, which is what a badge is for. Replacing the two live badges with synthetic rows is the obvious fix and has not been done.

Between the 07-27 and 07-30 recordings the absolute figures *rose* (RGB 1.45 → 1.76 ms, GRAY 9.16 → 13.72 ms). Part of that was the harness correction, which stopped flattering GRAY in particular; part was machine state three days apart. The two causes could not be separated after the fact, which is the argument for re-recording a baseline whenever the harness changes rather than carrying one across — and for recording *why* alongside the numbers, as the provenance list above now does.

### Native conversion kernels

Every colour mode now has a byte-exact Rust kernel, used automatically when the optional extension is present and declined cleanly when it is not. **Re-measured 2026-09-14 at 2.5.0** with `compare_recordings.py benchmarks/baseline-nonative.json benchmarks/baseline.json`, P-core-pinned on both sides, with the control's 1.11× drift divided out. Ceiling is that run's `control.memcopy`, **36.4 GB/s**:

| Mode | NumPy | Native | Gain | GB/s | Share of the 36.4 GB/s ceiling |
| --- | --- | --- | --- | --- | --- |
| BGRA | 0.21 ms | *(unchanged)* | — | 36.4 | **100%** — a straight copy; nothing to win |
| GRAY | 11.05 ms | **0.26 ms** | **48.0×** | 29.6 | 81% |
| BGR | 1.84 ms | **0.29 ms** | 7.1× | 24.7 | 68% |
| RGB | 1.88 ms | **0.29 ms** | 7.1× | 24.5 | 67% |
| RGBA | 2.56 ms | **0.37 ms** | 7.7× | 21.2 | 58% |

The native times are within noise of the 2026-08-05 figures they replace — **the kernels did not change between 2.1.0 and 2.5.0** (`swizzle.rs` moved 21 lines, `luma.rs` two). The *gains* rose (GRAY 37.2× → 48.0×, RGB 6.3× → 7.1×) because the NumPy arm got slower, not because the kernels got faster, and the share-of-ceiling percentages fell for the same reason the ceiling rose: this is a different measurement session on a machine whose `control.memcopy` read 36.4 GB/s against 33.2. Both effects are the § 3 warning about cross-session comparison, landing on the one table most likely to be quoted as a speedup.

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

**`benchmarks/baseline.json` is recorded with the extension built**; the extension-absent recording is `benchmarks/baseline-nonative.json`, which is what CI compares against and what a plain `pip install rapidshot` gets (§ 3 provenance above). Earlier revisions of this section said the opposite of `baseline.json` — that predates the two files being split, and reading it that way inverts every conversion row by 7–48×. The NumPy fallbacks were left byte-for-byte identical, so the no-toolchain install performs exactly as `baseline-nonative.json` records.

Cross-adapter transfer, 1080p BGRA (8.29 MB) into a `SHARED_CROSS_ADAPTER` heap, copy-queue submission and fence wait included:

| | Per frame |
| --- | --- |
| Copy into the shared heap | **0.87 ms** min, 0.94 ms median (~9 GB/s) |

Measured Intel iGPU → WARP on Machine A, which has no second hardware GPU. (Machine B's RTX 4060 → WARP figures are in § 6.1.) The **source** side is representative — on an Optimus laptop capture also runs on an iGPU with no dedicated VRAM, so it is the same system-memory copy. What the *consumer* adapter pays to read the heap is its own device's cost and is not measured here. Reproduce with `native.probe_cross_adapter()`.

For scale: this is about a third of what reading the same frame to the CPU costs (2.27 ms staging read), so moving a frame between adapters is cheaper than leaving the GPU.

---

### Machine B, Optimus — the hybrid recording

`benchmarks/baseline-rtx4060-hybrid.json`, recorded 2026-08-21 with capture on the **Intel iGPU** and the RTX 4060 render-only. Distinct from `baseline-rtx4060.json`, which was taken on the same machine in discrete-only mode: the `gpu` field differs (`Intel(R) UHD Graphics` versus `NVIDIA GeForce RTX 4060 Laptop GPU`), so the two refuse to gate against each other and say so. **That is correct, not a limitation** — they measure different capture adapters.

Recorded to the documented procedure: noise floor checked first (`--self-test`: 0 rows exceeded 1.30×, control 0.99×), pinned to the P-cores, then verified by an immediate second run that read `~ same` on every synthetic row with the control at 1.00×.

**The verification pass found a harness bug, which is what it is for.** `live.grab_frame_gpu` reported **FASTER 8.70×** with no qualifier, and on the next run the neighbouring live row reported **SLOWER 2.26×** — on identical code, which is the 2.5× live swing § 2 already documents. Only the second was labelled. The `FASTER` branch carried a comment saying its qualifiers applied "in both directions on purpose" while implementing only two of five; `live`, `duty-cycle sensitive` and `sub-ms` were applied to regressions only. Fixed, and guarded by tests in both directions.

That asymmetry mattered more than it looks: § 10's rule is that **a spurious improvement is as misleading as a spurious regression and harder to notice, because nobody investigates good news** — and the code implemented the opposite of the rule it quoted.

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

Two caveats before generalising. Machine A runs **mixed refresh rates** (100 Hz + 60 Hz), where DWM's composition clock is known to behave irregularly; a single-monitor measurement has not been taken. And frames above the panel's refresh were never *displayed* as distinct images — useful as extra temporal samples for a model, redundant for a recorder.

**The single-monitor measurement is now takeable and has not been taken.** Machine B has exactly one output (§ 4, topology dump), so it can settle whether the compositor-bound ceiling and the sub-9 ms present gaps survive without a mixed-refresh setup. Worth doing before this section is quoted anywhere load-bearing, because the mixed-refresh caveat currently applies to every number in it.

**A native capture core is not worth building.** It addresses 0.003 ms of an 8 ms frame, and `windows-capture` already ships exactly that (Rust + PyO3, DXGI + WGC) — so it is neither a differentiator nor a measurable win. See § 8.

**Published FPS claims in this space are mutually contradictory.** DXcam's README reports DXcam at 239 FPS; BetterCam's reports DXcam at 39 and itself at 123; other sources quote BetterCam near 290. Different hardware, no shared harness. Plan against `benchmarks/`, not anyone's marketing.

**D3D11 cannot share buffers — only 2D non-mipmapped textures.** Six configurations were probed (structured/raw/plain × NT-handle/legacy); none produced a buffer D3D12 could open. This is why the conversion shader runs on D3D12. `native.probe_shareable_buffers()` re-checks it; a regression test fails if this ever changes.

**HLSL presents BGRA textures semantically.** For `DXGI_FORMAT_B8G8R8A8_UNORM` the hardware swizzles so `.x` is **red**, despite blue being first in memory. Reversing channels by hand produces BGR labelled RGB — silently wrong model input that no test of speed or shape catches.

This was derived on Intel, and it is a claim about *driver* behaviour, so it was one vendor's word for a rule the shader depends on. **Confirmed on NVIDIA 2026-08-06**: all 21 tests in `tests/test_gpu_preprocess.py` pass on Machine B, including `test_rgb_channel_order_is_actually_rgb`, which is the regression guard for precisely this. Two vendors now agree, which is as close to settled as this gets without an AMD part.

**DWM does not emit move rects.** Measured 2026-08-05 on Machine A (Intel iGPU, Windows 11): 2,205 frames of live capture while a 700×500 window was dragged across the screen at 30 Hz with its text view scrolling — the two classic move-rect producers. `GetFrameMoveRects` returned `S_OK` on every frame and reported **zero** move rects, while dirty rects arrived on all of them (4,071 rects). `TotalMetadataBufferSize` ranged 16–144 bytes; a `RECT` is 16 bytes and a `DXGI_OUTDUPL_MOVE_RECT` is 24, so the smallest frames do not reserve room for a single move rect. Under DWM composition this metadata is effectively vestigial.

**Reproduced independently 2026-09-13**, same machine class, different apparatus: 3,768 frames across two workloads — a borderless window dragged across the desktop, and a real Wikipedia page scrolled in a browser — at 2560×1600 on Windows 11. **Zero move rects**, with the metadata readable on every frame, so this is DWM reporting none rather than a failed read. Two earlier attempts that day produced numbers that were thrown away: both drove `update_frame()` directly and skipped `release_frame()` whenever a poll had no new content, which leaves a frame acquired and makes every later `AcquireNextFrame` fail with `DXGI_ERROR_INVALID_CALL`. Drive metadata experiments through `grab()`, which pairs acquire and release itself.

The spec requirement is real even so: [MSDN states that to produce a visually accurate copy an application must process all move rects before it processes dirty rects](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-getframemoverects). **That hole is now closed, without implementing the copy** (2026-09-13): the rects are read, `Frame.move_rects` surfaces them, `changed_fraction` counts them — a scroll previously reported as no change — and the accumulator falls back to converting the whole frame whenever a frame carries any. So a source that *does* report moves gets a correct frame rather than stale pixels, at the cost of the dirty-rect optimisation on those frames only.

What remains **not worth implementing** is a path that reproduces the move by copying: it cannot be exercised on available hardware, and synthetic metadata proves nothing (§ 2). Reading the rects is cheap and testable; acting on them is neither.

**A D3D12 read of a captured surface must be ordered behind the capture device's own work, on the GPU.** `AcquireNextFrame` returns once the copy into the duplication surface is *submitted* on the capture device's D3D11 queue, not once it has run, and a D3D12 queue has no implied order with it. Unordered, a first read overtook the copy and returned the previous frame 7–15 of 150 times for `GpuPreprocessor12` and 65 of 150 for `CrossAdapterTransfer` (Intel iGPU, 2026-09-14). The settled answer is a fence shared with D3D11, signalled into its immediate context behind the copy, and `ID3D12CommandQueue::Wait` on the reading queue — `native/src/capture_order.rs`, used by every path that reads the surface. Do not re-litigate the alternatives, all measured:

| Before the first read | Stale first reads | Why not |
| --- | --- | --- |
| nothing | 7–11 / 100 | the bug |
| `ID3D11DeviceContext::Flush` | 13–14 / 100 | submits without waiting for completion |
| a 5 ms sleep | 0 / 100 | hides the race; costs 5 ms; no guarantee under load |
| **shared fence + queue `Wait`** | **0 / 150 per path** (converter also 0 / 500) | ordering on the GPU; no CPU wait; no measurable dispatch cost |

It relies on one invariant, which should stay true: the capture device has no D3D11 multithread protection, so the fence is signalled only while the caller holds a live `Frame`, during which the camera refuses every call that uses that context. Any new path reading the capture surface from D3D12 needs this, and a test of the kind § 2 describes.

**A software adapter is not a second GPU.** The Microsoft Basic Render Driver (WARP) reports zero outputs, exactly like the dGPU on an Optimus laptop, so the obvious "adapter with no outputs = discrete GPU" check calls every ordinary desktop a hybrid system. `DXGI_ADAPTER_FLAG_SOFTWARE` is what separates them. The flip side is useful: WARP is a real second D3D12 device, which is what makes the cross-adapter path testable on a single-GPU machine at all.

**Confirmed against a real discrete GPU 2026-08-06.** On Machine B `topology_info()` reports `single`, not `hybrid` — the check holds when the hardware adapter is an NVIDIA dGPU rather than an Intel iGPU, which is the case it was written for and had never faced:

```
Topology: single
  Adapter[0] (NVIDIA GeForce RTX 4060 Laptop GPU) (NVIDIA) (7956MB VRAM) (1 output)
  Adapter[1] (Microsoft Basic Render Driver) (Microsoft) (0MB VRAM) (software) (0 outputs)
```

**Desktop Duplication runs fine on a discrete GPU when the dGPU is the one driving the display.** § 6.1's `DXGI_ERROR_UNSUPPORTED` is about capturing on an adapter that is *not* driving the display, not about discrete GPUs as such. With Machine B's MUX in discrete-only mode the NVIDIA card owns the desktop and DDA works normally — 79 frames in 5 s at 2560×1600, with dirty rects, timestamps and cursor state all populated. Worth stating because "DDA cannot use the dGPU" is an easy over-reading of that support article.

**ONNX Runtime's `OrtDmlApi` has no Python binding.** `CreateGPUAllocationFromD3DResource` is reachable only from C/C++. Also: `IOBinding.bind_input` performs **no pointer validation** — it accepts `0xdeadbeef` without complaint, so "bind succeeded" proves nothing.

---

## 5. Completed

| Stage | Delivered |
| --- | --- |
| **0 — Infrastructure** ✅ | `benchmarks/perf_suite.py` (JSON, `--compare`, drift-calibrated, noise-floor self-test), `benchmarks/ab_conversion.py` (interleaved A/B), `.github/workflows/ci.yml` (4 jobs), populated `CHANGELOG.md` |
| **1 — DXGI correctness** ✅ | The package did not import at all before this. Fixed 5 undefined imports, signed/unsigned HRESULT comparisons (every error check was dead code), a leaked texture reference that stalled capture after 2 frames, invalid f-string format specifiers in the error paths, `DuplicateOutput1` with env-var fallback, access-lost/session-disconnect recovery, bounded fullscreen rebuild, HDCP handling |
| **1b — Pixel path** ✅ | RGBA 3.59×, BGR 2.58×, RGB 2.52×, GRAY 1.85×, outputs verified identical. Also fixed a frame-aliasing data-corruption bug (`grab()` returned views into recycled pool buffers) |
| **3 (slice) — Frame object** ✅ | `grab_frame()` returns a GPU-resident `Frame` with an explicit texture lifetime; **15× faster than `grab()`** on the current baseline (2.41 → 0.16 ms), and 31× on the no-extension one (5.93 → 0.19 ms). Both ratios move with the live-row spread in § 3 — the point is the order of magnitude, not the figure. Guards against the `INVALID_CALL` stall with a clear error |
| **6 + 6b — GPU tensor** ✅ | `GpuPreprocessor12` produces an `ID3D12Resource` on the DirectML device: BGRA→NCHW float32 with resize and normalisation in one dispatch. Verified exact against a NumPy reference and against real capture |
| **6.2 — Headless diagnostics** ✅ | `HeadlessError` replaces `"No usable graphics devices found. Check your display configuration."` with the actual fix (install an IDD virtual display), and distinguishes *no display* from *every device refused to open* — previously the same message |
| **6.1 (detect + measure)** ✅ | `rapidshot.topology_info()` classifies headless / single / hybrid / multi-adapter and says what a hybrid system means for the GPU tensor path. `native.probe_cross_adapter()` verifies the whole `SHARED_CROSS_ADAPTER` chain (heap → shared handle → open on the second device → placed resource on both) and times the capture-side copy |
| **6.3 (dirty rects)** ✅ | `frame.dirty_rects` in frame coordinates, plus `rects_coalesced`. The compositor already computed this and Rapidshot was discarding it — `GetFrameDirtyRects` was declared without argtypes, making it callable but unusable. Live capture reports **0.7–0.8% of the frame** dirty for a moving window |
| **6.3 (region-limited conversion)** ✅ | `grab()` converts only the dirty regions into a persistent accumulator, **12–15× faster** at the dirty fraction live capture produces. Falls back to a full conversion when metadata is missing, the area exceeds 90%, rotation is in play, or the mode is BGRA |
| **Pooled output (2.0 breaking change)** ✅ | `grab()` returns a `PooledBuffer` the caller releases. Allocating per frame cost ~1.6 ms in page faults — more than the conversion — so reuse is **1.3–2.1× on `grab()`**. See § 10 |
| **Stage 0 — release infrastructure** ✅ | PyPI Trusted Publishing with Sigstore attestations, SBOM, four wheel gates, GitHub Release automation, `py.typed` with the public API annotated, `SECURITY.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CODEOWNERS`, PR template, least-privilege CI tokens, performance badges generated from `baseline.json` with a drift guard |
| **6.1 (frame transfer)** ✅ | `native.cross_adapter_transfer(frame)` carries a captured frame to a second adapter and exposes the `ID3D12Resource` it lands in. Heap and placed resources are allocated once; only the copy is per-frame. Verified byte-exact on real capture by `examples/verify_cross_adapter.py` — 8,294,400 bytes per frame, against a source-side readback of the same snapshot. **That method proves the bytes arrived intact, not that they were the current frame** — until 2026-09-14 the copy read the previous frame 43% of the time (§ 4, § 10) |
| **1b — Native conversion (2.1.0)** ✅ | `native/src/luma.rs` and `native/src/swizzle.rs`: byte-exact AVX2 kernels for all five colour modes, used automatically when the extension is present and declined cleanly when it is not. **GRAY 37×, RGBA 7.5×, BGR 6.5×, RGB 6.3×**, all at 69–100% of the memory system's 33.2 GB/s. Correctness asserted exhaustively over 2²⁴ triples for GRAY and vector-against-scalar at every width 1–64. See § 3 |
| **Consumer ergonomics (2.2.0)** ✅ | `to_nchw()` (7.20 → 4.05 ms against the version most people write, bit-identical); `timeout_ms` and `pool_size_frames` made public — the latter dropping the default from 10 to 4 for **−60 MB per camera at −1.8% fps**. The "no screen updates" warning no longer fires during healthy polling capture |
| **GPU consumer — CUDA interop** ✅ | The tensor's output heap is `SHARED`, and `GpuPreprocessor12` exposes `shared_output_handle` / `output_byte_size`. `examples/gpu_tensor_to_cupy.py` imports it via `cuImportExternalMemory` into a `cupy.ndarray`, verified byte-identical to `read_back()` and read in place by a CUDA kernel. **The first GPU consumer this project has had** (§ 6.6) |
| **`to_nchw()` gather (shipped)** ✅ | `src[np.ix_(ys, xs)]` ran at **1% of memory bandwidth** and was 73% of the preprocess cost. Two sequential `take` calls are byte-identical and **1.8× faster end to end** at 640². The same fix landed in `perf_suite`'s reference arm, which the GPU claim is measured against (§ 10) |
| **Sustained-load soak** ✅ | 12 min, **211,726 frames, zero errors, VRAM flat at +0.00 MB/min**, no throughput decay. Also established that Desktop Duplication hands back a single capture texture across the whole run (§ 10) |
| **Five "untestable" paths, tested** ✅ | Protected content via `SetWindowDisplayAffinity`, which sets the same flag HDCP does. The `DuplicateOutput` **refusal** via a non-input desktop. **Real exclusive fullscreen** via `SetFullscreenState`, which drives the access-loss path end to end. The rebuild itself against live DXGI, pinned by object identity. And a **forced cache miss** from a real camera teardown. Every one had been recorded as fault-injection-only or unreachable (§ 10) |
| **Desktop-refusal misdiagnosis** ✅ | Every `E_ACCESSDENIED` from `DuplicateOutput` was reported as "protected (HDCP/DRM) content is on screen", including on a locked workstation, at a UAC prompt, and from a Session 0 service — the last being what § 12's consumer runs as. `rapidshot/util/desktop.py` now names the real cause (§ 10) |
| **Dispatch cost attributed** ✅ | `probe_dispatch_phases()` splits a dispatch into record / submit / signal / wait. **CPU floor 15.7 µs and flat; the fence wait is 70–82%**, and the shader itself ~9 µs at 640². Turns "the floor is submission and the fence" from a plausible list into a measurement, and sizes what an async path would actually buy (§ 10) |
| **Interop lifetimes under test** ✅ | `tests/test_cuda_interop.py` — 9 tests over the shared handle's lifetime and the shipped example, which it imports rather than copies. Covers handle stability, closure with the preprocessor, later dispatches landing in an already-imported view, and the ownership chain that keeps the tensor's memory alive. Skips cleanly without CuPy, an NVIDIA GPU, or screen activity (§ 10) |
| **D3D12 dispatch, per-frame cost** ✅ | Opening the captured texture is cached per *texture* rather than repeated per frame, and the UAV moved to the constructor. **2.6× at 640×640; the fixed floor fell 185 → 65 µs.** Keyed on the texture pointer, so a changed surface reopens rather than reusing a stale one (§ 10) |
| **`read_back` marshalling** ✅ | Both preprocessors return bytes instead of `Vec<f32>`, which PyO3 was turning into 1.2M Python floats per call. **6.3×**, and `pipeline.gpu_plus_readback` now measures PCIe rather than the CPython allocator (§ 10) |
| **`CupyProcessor` correctness** ✅ | Fixed a public-API path that returned BGRA when asked for RGB and called it success. Conversions are now pure CuPy and stay on the device, failures raise, bad modes are rejected at construction, and `tests/test_cupy_processor.py` asserts byte-exactness against the NumPy path for all five modes (§ 10) |
| **Benchmark harness, hybrid CPUs** ✅ | `perf_suite.py` detects a P-core/E-core split and pins itself, recording the fact in the output. Unpinned, the suite reported false verdicts up to 2.57× against *itself* (§ 2) |
| **Cross-library comparison (2.2.0)** ✅ | `benchmarks/compare_libraries.py` measures RapidShot against DXcam, BetterCam and mss — each in its own process, since all three declare the same COM interfaces and whichever imports first breaks the others — with a calibrated motion source and error bars on every quantity. It is also what established where RapidShot *loses*, now stated in the README |
| **6.1 — Intel→NVIDIA on real hybrid hardware** ✅ | `examples/verify_cross_adapter.py` carried **5 captured frames from an Intel iGPU to an RTX 4060**, 16,384,000 bytes each at 2560×1600, every one byte-exact against a source-side readback. `probe_cross_adapter()` reports `representative: true` — hardware on both ends, in the direction § 6.1 targets. Open since the project began (§ 2, § 6.1) |
| **Adapter selection on hybrid systems** ✅ | Duplication tries every adapter instead of assuming the display-owning one, keeps render-only adapters as candidates, and explains an all-adapter refusal instead of printing an HRESULT. Found by switching Machine B to Optimus, where capture stopped working entirely (§ 2, § 10) |
| **7.2 — `GpuConverter` (unreleased, 2.6)** ✅ | Bilinear or nearest (nearest bit-identical to `GpuPreprocessor12`); FP32/FP16 NCHW and NHWC; resized BGRA8; **NV12 and P010** (BT.709/601, limited/full); **crop** in frame coordinates, honouring `Frame.region`; **multi-ROI** in one dispatch, 2.5–4.9× faster than N calls for small regions; all four `DuplicateOutput1` input formats (only BGRA8 exercised). Verified on live capture; each packing and placement check was confirmed to fail against a deliberately planted bug (§ 7.2) |
| **7.2 — `TensorTransfer`, `TensorStream` (unreleased, 2.6)** ✅ | Convert-first across adapters, byte-equal at the boundary (WARP destination only). `TensorStream` iterates capture → tensor with frame release, CUDA sync, permanent-failure detection and converter rebuild handled; control flow tested without a GPU (§ 7.2) |
| **7.2 — `GpuTensor` exports (unreleased, 2.6)** ⚠️ | `to_torch()` / `to_cupy()` / `to_dlpack()` written and **never run** — no CUDA device on the machine that built them. A release gate (§ 1) |
| **Capture-ordering fix (unreleased, 2.6)** ✅ | `GpuPreprocessor12` and `CrossAdapterTransfer` read the previous frame 5–43% of the time on a moving source, since 2.3.0. Now ordered on the GPU behind the capture copy; 0 / 150 each, regression tests fail when the ordering is removed. Hardware destination unverified (§ 4, § 10) |

**Also fixed:** `pip install rapidshot` shipped a broken package — `pyproject.toml` listed `packages = ["rapidshot"]`, so the wheel contained 5 modules instead of 25 and failed with `ModuleNotFoundError: No module named 'rapidshot.util'`. Invisible from a source checkout. Now guarded by CI.

**Test coverage, 2026-09-14, Intel-only machine: 895 passed, 17 skipped**; `cargo test --release` 27 passed. The 2.6 work added `test_gpu_converter*.py` (seven files), `test_tensor_transfer.py`, `test_tensor_stream.py`, `test_tensor_stream_logic.py` (GPU-free, so it runs in CI) and `test_capture_order_shipped_paths.py`. **On Machine B the same day: 931 passed, 12 skipped, 0 failed**, the difference being the live GPU and CUDA files that Machine A skips, plus `test_gpu_tensor_export.py`, added there for the LUID selection behind the framework exports (§ 1). Three of its seven still skip on Optimus and need capture and CUDA on one adapter. The tables below are the 2026-08-06 state and are kept for the per-machine skip analysis, which still applies.

**Test coverage (2026-08-06):** `python -m pytest tests/ -q` collects 282. The skips are environmental, not pending work, and **the count is a property of the machine, not of the code**:

| | Machine A (Intel) | Machine B (RTX 4060) |
| --- | --- | --- |
| Result | 273 passed, 9 skipped | **323 passed, 1 skipped** at best; see the note below |
| Skipped | 8 need CuPy + an NVIDIA GPU; 1 needs a second adapter | 1, `test_cross_adapter.py:77` — skipped *because* 2 adapters are present |

The eight CuPy tests now run against real CUDA hardware, and **42 new tests** arrived with the Machine B work:

| File | | Covers |
| --- | --- | --- |
| `test_cupy_processor.py` | 13 | A processor that had **no tests at all** while being reachable from the public API |
| `test_cuda_interop.py` | 9 | Shared-handle lifetimes and the shipped CuPy example, which it imports rather than copies |
| `test_desktop_refusal.py` | 8 | Telling a non-input desktop apart from protected content, with the refusal reproduced for real |
| `test_preprocess_cache.py` | 4 | The `process()` texture cache, including a **deliberately forced miss** |
| `test_output_change_recovery.py` | 4 | The access-loss rebuild, against live DXGI |
| `test_exclusive_fullscreen.py` | 2 | A real exclusive-fullscreen transition, end to end |
| `test_protected_content.py` | 2 | Protected-content detection, with a **real trigger** rather than an injected HRESULT |

**None of these can run in CI**, which has neither an NVIDIA GPU nor a desktop session. They skip there and are verified only by running the suite on Machine B — so they fall under § 5's rule rather than escaping it.

**Five of the seven files exist because a path this document had written off as untestable turned out not to be**, and one of them found a shipped bug on the way (§ 10).

**The count is not stable run to run, by design.** `test_exclusive_fullscreen.py` needs Windows to actually grant exclusive fullscreen, and it sometimes declines with `DXGI_ERROR_NOT_CURRENTLY_AVAILABLE` (0x887A0022) depending on what else is on screen. The fixture skips with that HRESULT in the message rather than failing or, worse, passing without having tested anything — so a run reporting 321/3 is healthy and a run reporting 323/1 got the grant. Re-run before concluding anything from the difference.

`tests/conftest.py` holds a single session-wide Tk root, because two files need real on-screen windows and a second `Tk()` after the first is destroyed raises `TclError`. That surfaced as those files passing alone and *skipping* in a full run with "no desktop session" — a skip that only appears under a full run is worse than a failure, since nothing draws attention to it. Note the remaining skip is the inverse of Machine A's: that test covers the single-adapter path and steps aside when a second adapter exists, so **no single machine runs the whole suite** and a green run on either one is not full coverage.

The Rust side has its own suite, and it needs `python313.dll` on `PATH` or the harness dies with `STATUS_DLL_NOT_FOUND` before running a single test — pyo3 links the interpreter. `cargo test --release` reports **14 passed**, including `luma::tests::avx2_matches_scalar_for_every_bgr_triple`, the exhaustive 2²⁴ check. Verified on Machine B 2026-08-06: the AVX2 kernels are byte-exact on Raptor Lake as well as on Machine A's part.

**Live-capture verification is mostly still hand-run, and one piece no longer is.** `tests/test_cross_adapter.py` now carries the transfer's byte-exactness check against live capture, so the thing § 6.1 rests on runs with the suite instead of depending on someone remembering to run a script. It skips cleanly without a second adapter or without screen activity.

Still hand-run, and therefore still subject to the rule below: `examples/verify_cross_adapter.py` (kept as the diagnostic script, with the richer failure output), the `live.*` rows in `benchmarks/perf_suite.py`, and `benchmarks/compare_libraries.py`. None of these can run in CI, which has no desktop session (§ 2). **Anything in that list that is not run before a release is simply not verified for that release.**

---

## 6. Next up, in order

### 6.1 — Cross-adapter transfer (hybrid GPU laptops) ✅

**This is a coverage gap, not an optimisation.** [Desktop Duplication cannot run against the discrete GPU on a hybrid system](https://support.microsoft.com/en-us/help/3019314/error-generated-when-desktop-duplication-api-capable-application-is-ru) — it fails with `DXGI_ERROR_UNSUPPORTED`. So on an Optimus laptop you capture on the iGPU while inference runs on the dGPU, and the Stage 6 tensor has nowhere to go. That is a large share of consumer NVIDIA hardware.

Detection, measurement and the **BGRA frame transfer** are done (see § 5). `native.cross_adapter_transfer(frame)` carries a captured frame to the second adapter and exposes the `ID3D12Resource` it lands in.

Settled, do not re-derive:

- The chain works: heap with `D3D12_HEAP_FLAG_SHARED | SHARED_CROSS_ADAPTER` → `CreateSharedHandle` → `OpenSharedHandle` on the second device → `CreatePlacedResource` on both sides.
- [Cross-adapter shared resources live in system memory](https://learn.microsoft.com/en-us/windows/win32/direct3d12/shared-heaps). This is **not** VRAM-to-VRAM peer-to-peer DMA, and `IDXGIAdapter3` is not the mechanism (that is video-memory budgeting). The win is that a GPU copy engine moves the bytes instead of CPU cores.
- **Use a buffer, not a row-major texture.** `CrossAdapterRowMajorTextureSupported` is an optional capability; the buffer path works regardless and needs no branch. **This was a cautious choice and is now a required one.** `probe_cross_adapter()` on Machine B reports `source_row_major_texture: false` — the RTX 4060 does **not** support cross-adapter row-major textures, while WARP on the same machine reports `true`. Machine A never showed a `false`, so the branch that was avoided on principle would have been the branch that broke first on real NVIDIA hardware.
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

> ### ⚠ Re-opened 2026-09-13 — the measurement above is wrong, and B wins everywhere
>
> Two faults, both flattering A, found by review and then measured:
>
> 1. **A's destination-side conversion was never counted.** A was timed as the
>    transfer alone. The first bullet above claims the figures "understate A"
>    because the consumer's GPU is faster — but omitting one of A's costs
>    *overstates* A, whatever the speed of the GPU that would pay it. Relative
>    speed and an uncounted term are different arguments, and that bullet
>    conflates them.
> 2. **B was only ever measured carrying FP32** (`out*out*3*4`). A 640-square
>    frame need not cross as 4.92 MB: FP16 is 2.46 MB, a plain BGRA8 resize
>    1.64 MB. The cheap representations — the entire point of shrinking before
>    the bus — were never on the table.
>
> It was also measured at 1080p (8.29 MB). **At 2560×1600 the frame is 16.38 MB
> and A's transfer alone costs 2.65 ms** (`cross_adapter_ordering_v2.py`, two
> runs, min of 30):
>
> | out | payload | MB | convert (iGPU) | transfer | **B total** |
> | --- | --- | ---: | ---: | ---: | ---: |
> | 640² | BGRA8 resize | 1.64 | 0.51–0.54 | 0.15 | **0.66–0.69 ms** |
> | 640² | FP16 NCHW | 2.46 | 0.51–0.54 | 0.17 | **0.68–0.71 ms** |
> | 640² | FP32 NCHW | 4.92 | 0.51–0.54 | 0.23 | **0.74–0.77 ms** |
> | 1280² | FP32 NCHW | 19.66 | 0.86–0.92 | 0.96–1.00 | **1.81–1.92 ms** |
>
> **B wins at every size and every representation tested — including the FP32
> the original test used — and the win is unconditional.** B's total beats A's
> *transfer alone*, so no value of A's unmeasured destination conversion can
> change the ordering. That is a bound, not an estimate.
>
> A's destination conversion still cannot be timed from here: `GpuPreprocessor12`
> is built from the source texture and runs on that adapter, and nothing in the
> native API builds one on the destination device. It does not need to be: A is
> already losing without it.
>
> **What this does not establish.** The convert column is the FP32 NCHW path for
> every row, because that is the only thing `GpuPreprocessor12` emits — so the
> BGRA8 and FP16 rows are pessimistic on convert and exact on transfer. A
> resize-only kernel does not exist yet. The § 11 argument for A is untouched by
> any of this: it is about ownership, not speed, and it is now the *only*
> argument for A rather than one of four.
>
> **Do not treat the 1080p table above as settled.** The original verdict looks
> like an artifact of one resolution and one missing term. Re-measure before
> relying on it, and prefer `cross_adapter_ordering_v2.py`.

> ### Measured again 2026-09-14 with the kernels that were missing — and the resolution mattered
>
> v2 carried one stated caveat: its convert column was the FP32 NCHW path for
> every row, because `GpuPreprocessor12` emits only that, so the BGRA8 and FP16
> rows were *pessimistic on convert and exact on transfer*. 2.6's
> `GpuConverter` emits all three, so `cross_adapter_ordering_v3.py` measures
> each row with the kernel that would actually produce it.
>
> **The caveat was real** — the cheap representations are cheaper to produce as
> well as to move (640²: BGRA8 0.27 ms, FP16 0.31 ms, FP32 0.38 ms, where v2
> charged all three 0.38). **But removing it did not make B win everywhere at
> 1080p**, and that is the finding.
>
> Machine A, Intel iGPU → WARP, **1920×1080 (8.29 MB)**, nearest sampling,
> min of 30. A's transfer alone: **0.63–0.74 ms** across two runs.
>
> | out | payload | MB | convert | transfer | **B total** | verdict |
> | --- | --- | ---: | ---: | ---: | ---: | --- |
> | 320² | BGRA8 | 0.41 | 0.19 | 0.10 | **0.28 ms** | B wins |
> | 320² | FP16 | 0.61 | 0.20 | 0.12 | **0.32 ms** | B wins |
> | 416² | BGRA8 | 0.69 | 0.23 | 0.11 | **0.34 ms** | B wins |
> | 416² | FP32 | 2.08 | 0.27 | 0.22 | **0.50 ms** | B wins |
> | **640²** | **BGRA8** | 1.64 | 0.27 | 0.20 | **0.47 ms** | **B wins** |
> | **640²** | **FP16** | 2.46 | 0.31 | 0.26 | **0.57 ms** | **B wins** |
> | 640² | FP32 | 4.92 | 0.38 | 0.48 | 0.86 ms | undecided |
> | 832² | BGRA8 | 2.77 | 0.35 | 0.28 | **0.64 ms** | B wins |
> | 1024²+ | any | ≥4.19 | ≥0.39 | ≥0.37 | ≥0.76 ms | undecided |
>
> **This does not contradict v2 and does not re-open anything.** v2 measured
> **2560×1600**, where the frame is 16.38 MB and A's transfer alone costs
> 2.65 ms; here the frame is half that and A's transfer is a quarter the cost,
> so B has far less to beat. Both tables are right about their own resolution.
> What they jointly establish is that **the ordering is resolution-dependent**,
> which the § 4 entry should say rather than naming a single winner.
>
> **The load-bearing row is 640² FP16.** At 1080p, converting first wins there
> and does *not* win as FP32 — so on this machine the FP16 kernel is the
> difference between B winning and the question being undecided at the size
> production actually uses. That is the concrete return on 2.6's dtype work,
> and it is a result the FP32-only path could not have produced.
>
> **Two limits, both stated rather than buried.** The destination is WARP —
> Machine A has no second hardware GPU — so only the source side is
> representative, the same caveat § 6.1 carries throughout. And the undecided
> rows are genuinely undecided, not A wins: A's destination-side conversion is
> still unmeasured and would count against it.

> ### The first limit is now removed: measured 2026-09-14 on a hardware destination
>
> Machine B, **Intel iGPU → RTX 4060** (driver 616.92), Optimus, P-core-pinned
> (the script does not pin itself — § 2's gotcha applies), nearest sampling,
> min of 30, **2560×1600 (16.38 MB)**. This is the first run of any ordering
> benchmark with a real GPU on both ends, and the first after the
> capture-ordering fence, so unlike every earlier table here it is measuring
> transfers that are known to carry the *current* frame.
>
> A's transfer alone: **2.75 ms min, 3.69 ms median** — and A's destination-side
> conversion is still not included and cannot be negative.
>
> | out | payload | MB | convert | transfer | **B total** | verdict |
> | --- | --- | ---: | ---: | ---: | ---: | --- |
> | 320² | BGRA8 | 0.41 | 0.25 | 0.12 | **0.37 ms** | B wins by ≥2.39 ms |
> | 320² | FP16 | 0.61 | 0.25 | 0.12 | **0.37 ms** | B wins by ≥2.38 ms |
> | 416² | BGRA8 | 0.69 | 0.35 | 0.12 | **0.47 ms** | B wins by ≥2.28 ms |
> | **640²** | **BGRA8** | 1.64 | 0.56 | 0.15 | **0.71 ms** | **B wins by ≥2.04 ms** |
> | **640²** | **FP16** | 2.46 | 0.56 | 0.17 | **0.73 ms** | **B wins by ≥2.02 ms** |
> | 640² | FP32 | 4.92 | 0.59 | 0.22 | **0.81 ms** | B wins by ≥1.95 ms |
> | 832² | FP16 | 4.15 | 0.58 | 0.22 | **0.80 ms** | B wins by ≥1.96 ms |
> | 1024² | FP16 | 6.29 | 0.55 | 0.27 | **0.81 ms** | B wins by ≥1.94 ms |
> | 1280² | FP32 | 19.66 | 0.85 | 0.84 | **1.69 ms** | B wins by ≥1.06 ms |
>
> **B wins every row, unconditionally**, including the worst one by more than a
> millisecond. At this resolution the question is not close, which is what v2
> said about 2560×1600 and what the 1080p table above does *not* say about
> 1080p. The resolution-dependence stands: **at 1080p the ordering is genuinely
> contested above 640² FP16; at 2560×1600 it is not.** Choose by frame size,
> not by rule.
>
> **It corroborates v2 closely rather than replacing it.** Same resolution and
> output sizes, and the transfer column lands on the same numbers — 640² FP16
> crosses in 0.17 ms in both, against v2's 2.65 ms and this run's 2.75 ms for
> the full frame. The convert column is ~0.05 ms higher here, which is a
> different capture adapter and a different day, not a finding.
>
> **What it does not measure.** A's destination-side conversion, still — and it
> still does not need to: A loses on its transfer alone. Sampling is nearest
> for comparability with v2, so a caller taking `GpuConverter`'s bilinear
> default pays more than the convert column here. And this is one machine's
> bus; § 6.1's caveat about the consumer adapter's own read cost is unchanged.

Remaining work:

- ~~**The transferred frame was current.**~~ **It often was not, and every check above was blind to it.** Found 2026-09-14 (§ 4, § 10): the source queue's copy could overtake the capture device's own copy into the surface, and carry the previous frame — 65 of 150 first transfers, Intel iGPU → WARP. Every "byte-exact" result in this section compared the destination with a source-side readback of *the same snapshot*, which is stale in exactly the same way, so they were correct about integrity and silent about currency. Fixed with a capture-ordering fence (0 / 150 after). **Re-verified with a hardware destination 2026-09-14** — the regression tests pass Intel → RTX 4060 on Machine B, and the ordering benchmark was re-run there post-fix (box above). The older Intel → RTX 4060 figures below still predate the fix.
- ~~**A shared fence.**~~ **Built 2026-08-22, and the 2026-08-05 conclusion did not survive the hardware it was about.** That entry said the wait "adds essentially nothing per frame" and the fence would buy "latency and pipelining, not throughput". Measured against WARP, that was right. Measured Intel→NVIDIA it is wrong in the useful direction.

  **`probe_transfer_phases()` split one transfer**, the way § 10's `probe_dispatch_phases` did for the preprocessor. Medians, 2560×1600, 200 iterations:

  | phase | before caching | share |
  | --- | --- | --- |
  | open | 300.8 µs | 9.8% |
  | record + submit + signal | 31.6 µs | 1.0% |
  | **wait** | **2634.9 µs** | **85.6%** |
  | close | 110.2 µs | 3.6% |

  The wait is 85.6% of the transfer, but it *is the copy executing* — 16.4 MB at 5.68 GB/s is ~2.88 ms — so the fence cannot make the copy faster. What it removes is the calling thread's obligation to sit through it.

  **The first measurement of this was wrong, and the way it was wrong is the point.** A synthetic consumer — a Python busy-loop standing in for "the caller does something with the thread" — reported **wall clock 6.22 → 3.59 ms, 42% faster**. That number is real for a *CPU-bound* consumer, and it does not survive contact with the consumer this feature exists for.

  Re-measured with an actual GPU consumer (BGRA → float32, normalise, reduce over the 16.4 MB frame in CuPy), interleaved, two runs:

  | strategy | run 1 | run 2 |
  | --- | --- | --- |
  | blocking + CPU wait | 16.57 ms | 14.81 ms |
  | async + **CPU** fence wait | 16.73 ms (−0.9%) | 14.45 ms (+2.5%) |
  | async + **GPU** semaphore | **14.22 ms (+14.2%)** | **13.69 ms (+7.5%)** |
  | async + GPU semaphore, default stream | 14.42 ms (+13.0%) | 13.92 ms (+6.0%) |

  **Two conclusions, and neither is 42%.**

  **The CPU-side async wait buys nothing against a GPU consumer** — −0.9% and +2.5% across two runs is noise. The synthetic benchmark overstated it because a CPU busy-loop genuinely overlaps a GPU copy, while a GPU consumer competes for the same device and is itself asynchronous, so the calling thread was never the constraint.

  **The GPU-side semaphore wait is the one that pays: 7–14%.** Quote the range, not a point estimate; live capture varies (§ 2) and two runs disagreed by a factor of two on the margin. This is `shared_fence_handle` — CUDA imports the D3D12 fence with `cuImportExternalSemaphore` and waits on it *in a stream*, so no CPU is involved in the handoff at all. Verified 2026-08-22: **6/6 frames byte-exact through a GPU-side semaphore wait**, with the fence created on the **Intel** device and imported by CUDA on the **NVIDIA** one, which is not something the documentation promises.

  Using a non-default stream made little difference here (13.0–14.2% versus 6.0–7.5% on the default stream, overlapping across runs), so the theoretical serialisation on the default stream is not visible at this frame size.

  Shipped as `transfer_async()` + `wait_shared_fence()`, with `shared_fence_handle` for the GPU-side wait. `transfer()` still blocks and remains the default: async moves synchronisation to the caller, which is a real cost for anyone who does not need it.

  **The GIL was the gap between the Rust measurement and what Python sees.** `wait_shared_fence` originally held the interpreter for the whole wait, so the calling thread it "returned" could not run Python: measured 2026-08-22, a second thread made **zero progress across a 7 ms wait**. Now armed under the GIL and waited with it released, which cut the longest stall to ~1 ms and let the other thread tick 17–18 times in the same window. Only a raw event handle crosses into the detached closure — `py.detach` needs a `Send` closure and this type holds `Cell`/`RefCell`, so a reference to it cannot, and that constraint is what keeps the release provably sound instead of asserted.

  **`transfer()` still holds the GIL** for its whole copy (~2.7 ms measured, 0 ticks from another thread). Its wait is buried between COM calls on interior-mutable state and cannot be released the same way without restructuring resource lifetimes. Documented rather than hidden; the async path is the answer for anyone who needs other Python threads to run.

  **It pipelines to depth one, deliberately.** `Submitter::begin` resets the command allocator, and resetting one the GPU is still reading is corruption, so `transfer_async` waits for the *previous* submission before recording. Deeper pipelining needs multiple allocators and has not been measured to be worth it.

- ~~**Do not chase per-frame shared-handle caching.**~~ **Wrong on this hardware, and fixed.** § 6.1 recorded the apparent handle overhead as noise, from three runs on Machine A where it was indistinguishable from the copy. On the Intel→NVIDIA path it is not noise: `CreateSharedHandle` + `OpenSharedHandle` + `CloseHandle` ran **every frame** at 268 µs + 114 µs.

  Cached per texture, keyed on the raw pointer — the same fix § 10 applied to the preprocessor, for the same reason, found the same way. A/B'd inside one harness rather than across two runs, because § 2 is explicit that comparing separate harnesses produces verdicts about conditions:

  | phase | uncached | cached |
  | --- | --- | --- |
  | open | 268.2 µs | **0.2 µs** |
  | close | 114.3 µs | **0.6 µs** |
  | total | 3010.9 µs | **2700.7 µs** (10.3% faster) |

  Byte-exactness unchanged. `cached_texture_address` is exposed because the cache produces identical output either way, so a change that disabled it would be a large regression with no visible symptom — the test asserts on the key, not on pixels.

  **The lesson is about the evidence, not the cache.** "Measured as noise" was accurate for one machine and got written down as a general instruction. A measurement's scope is part of the measurement.
- ~~**The transferred frame could not reach a GPU consumer.**~~ **Done 2026-08-22.** `CrossAdapterTransfer` exposes `shared_destination_handle`, and the whole Optimus chain now runs: capture on the Intel iGPU → `transfer()` → one CUDA import on the RTX 4060 → 8/8 frames read byte-exact through the imported view, with no re-import per frame and no CPU round-trip of the frame.

  Before this the path was two verified halves that nothing joined. `GpuPreprocessor12` builds on the *capture* adapter, so on a hybrid system its output is where CUDA cannot see it; `cross_adapter_transfer` moved the frame but exposed only a pointer, and `cuImportExternalMemory` needs a handle.

  **The obvious implementation does not work, which is the part worth keeping.** Copying Stage 6's pattern — `CreateSharedHandle` on the output resource, imported as `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE` — fails here, because the cross-adapter buffers are **placed** resources rather than committed ones:

  | Candidate | `CreateSharedHandle` |
  | --- | --- |
  | `dst_buffer` (placed resource) | **E_INVALIDARG** |
  | `src_buffer` (placed resource) | **E_INVALIDARG** |
  | `dst_heap` | ok |
  | `src_heap` | ok |

  So the shared object is the **heap**, and a consumer imports it as `CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP` (4), size `total_bytes`, offset 0. `probe_shared_handles()` is that measurement, kept as a diagnostic because the answer is a property of the driver pair and exactly one pair has been checked.

  Two choices were made from measurement rather than argument, and one was not. Both heaps import into CUDA and both read byte-exact, tracking later transfers identically (12/12, twelve distinct checksums, transfer p50 6.72 vs 6.79 ms) — so the data does not choose between them. **`dst_heap` was picked on a robustness argument, not a result**: it is minted by the device the consumer runs on, where `src_heap` additionally requires CUDA to accept a handle created by a different vendor's device. That held on the one pair measured, which is not enough to depend on. And the handle is *borrowed* — created once, closed on `Drop` — matching `shared_output_handle`; measured, closing it after import is safe and does not invalidate the mapping, so consistency decided a free choice.

- ~~**Real hybrid hardware.**~~ **Done 2026-08-22.** Machine B ran as a genuine Optimus system — Intel iGPU owning the display, RTX 4060 render-only — and `examples/verify_cross_adapter.py` passed against it:

  | | |
  | --- | --- |
  | Source → destination | **Intel(R) UHD Graphics → RTX 4060** |
  | Frames | 5, each **16,384,000 bytes** at 2560×1600 |
  | Result | every one **byte-exact** against a source-side readback |
  | `representative` / `destination_is_software` | **true** / **false** |
  | Copy cost | 1.39 ms min, 1.53 ms median, **5.68 GB/s** |

  **The buffer decision is vindicated in the case that actually matters.** In this direction NVIDIA is the *destination*, and it reports `destination_row_major_texture: false` while Intel as source reports `true`. The row-major branch avoided on principle would have been the branch that broke here — not against WARP, not hypothetically, but in the exact configuration § 6.1 exists for.

  Note the asymmetry in cost: Intel→NVIDIA is 5.68 GB/s against NVIDIA→Intel's 11.0 GB/s (§ 10). The iGPU is the weaker part and it is also driving the display, so the capture-side copy is roughly half as fast in the direction a hybrid system actually needs. That is the number to quote for Optimus, not the reverse one.

  **Partly advanced 2026-08-06, and it is worth being precise about how much.** Machine B ran `probe_cross_adapter()` with an NVIDIA dGPU as the *source*, which had never happened — Machine A's source was always the iGPU:

  | | Source | Destination | Copy, min | Copy, median | Throughput |
  | --- | --- | --- | --- | --- | --- |
  | Machine A | Intel iGPU | WARP | 0.87 ms | 0.94 ms | 9.5 GB/s |
  | Machine B | **RTX 4060** | WARP | **0.68 ms** | 0.78 ms | **11.6 GB/s** |

  The destination is still WARP, so `representative` is still `False` and `destination_is_software` still `True`. **This does not close the item.** What it establishes is that the chain — shared heap, `CreateSharedHandle`, `OpenSharedHandle`, placed resources on both sides — works with a discrete NVIDIA GPU on one end, and that the capture-side copy is if anything cheaper there. The half that remains untested is the one that matters: an Intel iGPU source with an NVIDIA dGPU *destination*, which needs the MUX switched to Hybrid (§ 2).
- Do **not** chase per-frame shared-handle caching on this evidence. One run suggested `transfer()` cost 1.48 ms against a 0.66 ms raw copy, implying ~0.8 ms of per-frame handle overhead; three further runs put it at 0.70–0.98 ms, consistent with the raw copy. The apparent overhead was noise. (The capture texture pointer *is* stable across frames, so caching remains possible if a real profile ever justifies it.)

### 6.2 — Headless / virtual display diagnostics ✅

Done. [With no monitor attached there is no desktop to duplicate](https://github.com/FreeRDP/FreeRDP/issues/5825) — `DuplicateOutput` fails outright, which blocks every cloud-VM deployment. `HeadlessError` now names the fix (install an IDD virtual display driver) instead of saying "check your display configuration".

Kept here because the caveat is easy to lose: a virtual display's advertised refresh rate does **not** raise capture rate. DDA is driven by presents, not refresh — a 500 Hz virtual display does not make an application render 500 fps. This is in the error text and the README.

Not yet exercised on a genuinely headless machine — the logic is tested by describing that topology, not by having one.

### 6.3 — Finish Stage 3 (Frame metadata)

The lifetime slice is done, `dirty_rects` is done, and `py.typed` now ships with the module-level API annotated.

**What is left, precisely** — verified against the code on 2026-08-06, because this list had drifted:

| Piece | State |
| --- | --- |
| Normalised timestamps | **Done.** `Frame.timestamp_qpc` (raw `LastPresentTime` ticks) and `Frame.timestamp` (seconds). The docstring records the part that matters: this is when the *compositor presented* the frame, not when it was captured |
| Cursor data on `Frame` | **Done** (2026-09-13). `Frame.cursor` carries a `CursorInfo`: `visible`, `position`, `hotspot`, raw `shape` bytes, `shape_type`, `shape_size`, `shape_pitch`, snapshotted per frame because the duplicator mutates its `Cursor` in place on the next acquire. `position` is in frame coordinates; a point outside the frame is kept and shifted rather than dropped or clamped, because a pointer just past the edge still draws pixels inside the region. Verified live against `GetCursorPos` on both a full-output and an off-origin region capture |
| `move_rects` | **Read and surfaced** (2026-09-13). See the row below and § 4 |
| `Protocol`-typed interfaces | **Not started.** No `Protocol` anywhere in `rapidshot/` |

Cursor position **has now had** the frame-coordinate treatment below — it is reported against the duplicated output, so on a region capture the raw value was wrong in exactly the case nobody checks by hand. The empty-versus-unknown distinction holds too: `position is None` means DXGI reported no position, `visible=False` means the pointer is hidden, and they are different answers.

So **§ 6.3 is down to one piece: the `Protocol`-typed interfaces.** Design them before a second backend exists (§ 7.3) — retrofitting a DXGI-shaped API to fit WGC later is the expensive order.

`move_rects` is **read, but nothing patches with it — deliberately.** `Duplicator.get_frame_move_rects()` calls `GetFrameMoveRects` on every frame, `Frame.move_rects` surfaces the result in frame coordinates, and `changed_fraction` counts it — without which a scroll reports as *no change at all*. What is **not** built is a copy path that reproduces the move, because DWM never emits one to test it against (§ 4). Instead the accumulator refuses to patch a frame carrying move rects and converts the whole thing, which closes § 4's latent hole without inventing an untestable code path.

Design this **before** a second backend exists — retrofitting a DXGI-shaped API to fit WGC later is more expensive than designing one abstraction all backends fill.

**Settled by `dirty_rects`, and it applies to the rest:**

- **Frame metadata must be in frame coordinates, not desktop coordinates.** DXGI reports rects relative to the whole duplicated output; a `Frame` may cover a region of it. Passing raw values through would make `dirty_rects` index outside the frame whenever a region is off-origin — wrong only in the case nobody checks by hand. `Frame` clips and translates. `move_rects` and cursor position need the same treatment.
- **Empty and unknown are different answers.** `[]` means no rects were reported; `None` means the metadata could not be read. A consumer skipping unchanged regions must distinguish them or it silently skips everything on a frame whose metadata failed. An empty list does *not* mean nothing changed — a mode change or a coalescing driver can report none while the image differs completely.
- **`RectsCoalesced` matters.** When set, the driver merged rects, so they over-estimate what changed. Surfaced as `frame.rects_coalesced`.
- The COM signatures for `GetFrameDirtyRects`/`GetFrameMoveRects` were declared in `_libs/dxgi.py` without argtypes, so they were callable but unusable — comtypes could not marshal the out-parameters. Both are now declared correctly **and both are called**: move rects are read first, which is the order [MSDN requires](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-getframemoverects) and the order Microsoft's own sample uses, since the two share one metadata buffer with the move rects at the front. The extra read costs **7.9 µs, 0.13% of a 6.05 ms `grab()`**.

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

**The GPU-consumer half is done as of 2026-08-06.** `examples/gpu_tensor_to_cupy.py` runs `grab_frame()` → `GpuPreprocessor12` → `cupy.ndarray` with no CPU round-trip, verified byte-identical to `read_back()` and read in place by a CUDA kernel. This sidesteps § 8's objection entirely: CuPy is the *caller's* dependency, not RapidShot's, and the ctypes glue lives in `examples/` rather than the package (§ 11).

Measured on Machine B, 2560×1600 → 640×640:

| | min | p50 |
| --- | --- | --- |
| dispatch only | **0.075 ms** | 0.083 ms |
| dispatch + a CUDA reduction over the tensor | 0.270 ms | 0.311 ms |

The import is paid once; per frame the CuPy array simply sees the new contents. **What remains for § 6.6 is the model**: a small ONNX detector and box drawing on top of a tensor that already exists on the right device. The hard part — getting the tensor there at all — is no longer in the way.

Note the honest framing for the headline: **0.075 ms per frame for a model-ready tensor that never touches the CPU**, against **3.07 ms** for the CPU arm on the same machine — about **40×**. Not the "2 µs" this document used to quote, which measured D3D11 submission rather than the DirectML path, and not the larger ratio the CPU arm implied before its gather was fixed (§ 10). Both corrections moved the number the same direction: **down, and toward something defensible.**

---

## 7. The 2.5 → 3.0 plan

**Positioning.** RapidShot is a **Windows GPU capture runtime: capture → transform → handoff.** Not "the fastest screenshot library" — § 3 and the README both show it losing the capture-only frame-rate column, and it pulls ahead only once the frame has somewhere to go (§ 7.0 results) — and not an ML framework. It moves pixels from the compositor to whatever consumes them (NumPy, PyTorch, an encoder, another process) with the fewest copies and the least synchronisation overhead. AI is the flagship consumer, not the definition.

That phrasing matters for scope. It keeps PyTorch interop central without making the library ML-only, and leaves room for encoders, recorders and native consumers without inviting scene graphs and audio mixers (§ 8).

### 7.0 — Build the benchmark first <- **do this before any feature below**

Every item in this section is a performance claim, and § 11 says measure before ordering. When this plan was written there was no benchmark measuring the thing RapidShot is actually for; the plan is kept as written, and the results follow it.

`benchmarks/compare_libraries.py` measures `grab()` — OS pixels to a CPU array. That is one of three categories, and the one where RapidShot's advantage is smallest:

| Category | Measures | State |
| --- | --- | --- |
| **Capture** | OS pixels → application frame | ✅ `compare_libraries.py` |
| **AI ingestion** | OS pixels → model-ready CUDA tensor | ✅ `ai_ingestion.py` (call duration) and `section7.py` (pixel age) — see results |
| **Agent** | OS pixels → API-ready compressed screenshot | ⚠ built (`section7.py --category agent`), not yet run |

**`benchmarks/ai_ingestion.py`** should measure one number: **time from a new desktop frame appearing to a `1x3x640x640` FP16 RGB normalised tensor being ready on CUDA:0.** Every path must produce a bit-identical tensor, or the comparison measures different work:

| Path | Pipeline |
| --- | --- |
| mss | capture → NumPy → resize → RGB → float → NCHW → CUDA |
| DXcam (DXGI) | capture → NumPy → same → CUDA |
| DXcam (WGC) | WGC → NumPy → same → CUDA |
| RapidShot CPU | `grab()` → same → CUDA |
| **RapidShot GPU** | `grab_frame()` → D3D12 preprocess → CUDA tensor |

Report **p50 / p95 / p99 present-to-model-ready**, unique frames/s, CPU %, **CPU->GPU bytes per frame** (the one that exposes hidden copies), GPU preprocess time, fence wait, dropped frames, RAM, VRAM, jitter. Not mean FPS.

**Measure pixel age, not call duration.** Timing `t0 = perf_counter(); frame = capture(); t1` says how long the *call* took, not how old the *pixels* are. RapidShot has an advantage here — `Frame.timestamp_qpc` exposes the compositor's `LastPresentTime` — but the other libraries expose no equivalent, so a shared clock is needed.

Build a **controlled visual latency source**: a D3D window that encodes an incrementing frame ID as a small binary pattern and records `frame_id → QPC` at every `Present()`. Each capture path decodes the pattern out of the captured pixels and recovers the ID. Then

```text
latency = model_ready_qpc - source_present_qpc[decoded_id]
```

and mss, DXcam, WGC and RapidShot are all measured against one clock. `benchmarks/motion_source.py` is the obvious thing to grow this from.

**A caveat that belongs in the harness rather than being discovered later:** QPC is *not* nanosecond-accurate and is not synchronised to an external clock. Microsoft specifies a sub-microsecond interval timer whose real resolution comes from `QueryPerformanceFrequency`. Read it once and record it, the way `perf_suite.py` records its machine block.

**Then a second benchmark, present → inference complete**, with one small fixed model (YOLO11n, or a pinned ONNX/TensorRT graph) across 1080p / 1440p / 4K, 60-240 Hz, and three workloads: static UI, scrolling, high motion. If RapidShot wins that, the defensible claim becomes *"fastest Windows desktop-to-model pipeline"* — far stronger and far more checkable than *"fastest screenshot library"*.

**Why this is 7.0 and not 7.5:** `GpuConverter`, Torch interop, FP16, multi-ROI, DLPack and more fence work all *sound* architecturally attractive. The benchmark is what says which of them moves the number that matters. Building it first is the difference between a roadmap and a wish list.

### 7.0 results — measured 2026-09-10 ✅

`benchmarks/ai_ingestion.py` exists and has run. Machine B, Intel UHD capture →
RTX 4060, 2560×1600 → `(1, 3, 640, 640)` FP32 RGB on CUDA. Medians across **3
independent passes**, 8 s per path, CUDA synchronised before every sample so
these are completion times rather than submission times. Every path resizes the
full image with half-pixel bilinear sampling and is verified against an
independent float64 NumPy reference taken from the same frame — **max error
0.5/255 on all six**, so the paths are genuinely producing the same picture.
Data in `benchmarks/ai-ingestion-machineB.json`.

| path | fps | p50 ms | p95 | p99 | CPU % | **CPU ms/frame** | H2D/frame |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mss | 30.7 | 31.26 | 37.49 | 39.05 | 51.9 | 16.91 | 1.23 MB |
| dxcam | 98.2 | 9.97 | 12.16 | 13.45 | 96.2 | 9.80 | 1.23 MB |
| **rapidshot-cpu** | **138.4** | **6.87** | **9.91** | **10.85** | 97.8 | 7.07 | 1.23 MB |
| rapidshot-cupy | 116.1 | 8.62 | 10.49 | 11.32 | 56.2 | 4.84 | n/a |
| rapidshot-xadapter | 99.7 | 9.60 | 12.60 | 13.79 | 43.9 | 4.40 | **0** |
| rapidshot-xadapter-async | 96.9 | 9.84 | 12.89 | 13.84 | 41.4 | **4.27** | **0** |

**Read CPU ms/frame, not CPU %.** A path running at 138 fps does more work per
second than one at 97, so the percentages are not comparable directly. Cost per
frame is.

**RapidShot wins this workload, having lost the capture-only one.** Against
DXcam: 31% lower p50 (6.87 vs 9.97 ms), 41% more frames, and 0.72× the CPU per
frame. `compare_libraries.py` has RapidShot *losing* the RGB frame-rate column to
DXcam; here it wins all three axes. The tensor tail is byte-identical between the
two, so the difference is capture itself — waiting on `timeout_ms` beats
polling.

**The GPU paths trade latency for cost, and the trade is steep.** Cross-adapter
costs **0.45× DXcam's CPU per frame** and moves **zero bytes host-to-device**,
but is 2.7 ms slower per frame than the CPU path. The copy through the
system-memory shared heap is what that buys.

**So the claim depends on the path.** Speed and cost come from different ones:
`rapidshot-cpu` is the fastest here and the most CPU-hungry of RapidShot's
three, and cross-adapter is the cheapest and 2.7 ms slower. No single
configuration is fastest *and* cheapest. What holds up for cross-adapter is: *a
model-ready tensor for under half DXcam's CPU per frame, with no host-to-device
transfer at all.* For an agent doing inference on the same machine, that is the
number that decides whether capture starves the model.

> ### 2.6 measured, 2026-09-14 — and one configuration is now both
>
> The table above predates 2.6 and measures none of it: it was recorded
> 2026-09-10, four days before `GpuConverter` and `TensorTransfer` existed, so
> the benchmark built to decide which features move the number could not see
> the features that shipped. Two paths were added and the target moved to the
> **FP16** § 7.0 specified all along (it had measured FP32 — the representation
> § 6.1 puts on the *losing* side of the convert-first question at 1080p).
> `benchmarks/ai-ingestion-fp16-machineB.json`, same machine and method,
> medians of 3 passes, **FP16**:
>
> | path | fps | p50 ms | p95 | p99 | CPU % | **CPU ms/frame** | H2D |
> | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
> | mss | 32.4 | 30.40 | 35.90 | 37.85 | 53.9 | 16.78 | 1.23 MB |
> | dxcam | 103.7 | 9.46 | 11.96 | 13.58 | 92.0 | 8.65 | 1.23 MB |
> | rapidshot-cpu | 139.2 | 6.61 | 10.48 | 11.90 | 100.2 | 7.20 | 1.23 MB |
> | rapidshot-cupy | 130.5 | 7.32 | 10.82 | 12.40 | 59.8 | 4.68 | 0 |
> | rapidshot-xadapter | 102.3 | 9.72 | 13.18 | 14.14 | 47.0 | 4.58 | **0** |
> | rapidshot-xadapter-async | 98.1 | 10.32 | 13.45 | 14.38 | 43.9 | 4.35 | **0** |
> | **rapidshot-converter-xadapter** | **165.0** | **6.06** | **7.21** | **7.63** | **15.2** | **0.92** | **0** |
>
> **The sentence above is now wrong, and that is the finding.** 2.6's
> convert-first path is the fastest row *and* the cheapest row at once: 19%
> more frames than `rapidshot-cpu` at a **7.8× lower CPU cost**, and against
> the whole-frame cross-adapter path it is 61% more frames for **1/5th the
> CPU** — while still moving zero bytes host-to-device. Against DXcam:
> **9.4× less CPU per frame** and 59% more frames.
>
> **Where the CPU went.** Both cross-adapter rows end with the same CUDA
> import; what differs is who resizes. The old path moves the whole 16.38 MB
> frame and resizes it on the destination with about fifteen chained CuPy
> kernels per frame, each launched from Python. The new one does it in **one
> D3D12 dispatch on the capture adapter** and moves the finished 2.46 MB
> tensor. That is the § 7.2 architecture doing exactly what it was proposed to
> do, and it is the first measurement that says so.
>
> **The tail moved more than the median.** p99 7.63 ms against 11.90 for
> `rapidshot-cpu` and 14.14 for the old transfer, with jitter 0.59 ms against
> 1.99. For anything frame-paced, that matters more than p50.
>
> **It is the same picture, not a cheaper one.** The new path verifies against
> the same independent float64 NumPy reference as every other row: **max error
> 0.000486**, which is FP16 quantisation, well inside the 2/255 the comparison
> allows. And the six pre-existing paths reproduce their 2026-09-10 FP32
> figures within a few percent, so the dtype knob did not disturb them.
>
> **`rapidshot-converter` — the one-call export, no transfer at all — skips
> here.** It needs capture and CUDA on the same adapter, which Optimus never
> gives; it is recorded as a skip rather than a failure. It should be the
> fastest row of all on a single-adapter NVIDIA desktop, and that is unmeasured.
>
> **Caveats.** One machine, not P-core-pinned (matching how the FP32 recording
> was made, § 2), and **call duration, not pixel age**. The pixel-age result is
> the next box; two claims made here earlier in the day were wrong and are
> corrected there.

> ### Pixel age, 2026-09-14 — the metric § 7.0 actually asks for
>
> **Two corrections first, because both were stated here earlier today and both
> were wrong.** `section7.py` was said to measure FP32: it does not, and never
> did — `benchmark_contract.normalized_tensor` has always ended
> `.astype(xp.float16)`, so the pixel-age harness was already on FP16. And it
> was said to measure only pre-2.6 paths: true when written, fixed since.
> `rapidshot-converter-xadapter` and `rapidshot-converter` now run there.
>
> Same run, same source, 8 s per path, 2560×1600 at 165 Hz, unique frames only:
>
> | path | unique fps | **age p50** | p95 | p99 | jitter | CPU ms/frame | RSS MB |
> | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
> | dxcam | 101.4 | 35.15 | 38.42 | 39.30 | 1.68 | 10.83 | 408.3 |
> | rapidshot-cpu | 131.5 | 32.65 | 35.48 | 36.87 | 1.48 | 7.61 | 501.2 |
> | rapidshot-cupy | 128.7 | 32.62 | 35.15 | 36.45 | 1.39 | 5.13 | 454.1 |
> | rapidshot-xadapter | 80.1 | 35.16 | 36.81 | 40.49 | 1.13 | 5.68 | 556.4 |
> | **rapidshot-converter-xadapter** | **163.6** | **28.17** | **28.82** | **29.76** | **0.38** | **1.90** | 476.8 |
>
> **It wins every latency axis and the cost axis at once.** Pixels reach the
> model **7.0 ms younger than DXcam** at p50 and **9.5 ms younger at p99**, for
> **5.7× less CPU per frame**, and it is the only path that reaches the panel's
> 165 Hz ceiling — 61% more unique frames than DXcam. Jitter is 0.38 ms against
> 1.13–1.68 for everything else, which for anything frame-paced matters more
> than the median.
>
> **Stage breakdown**, p50: capture call 2.07 ms, D3D convert 1.88 ms, transfer
> 0.36 ms, marker decode 1.23 ms. The transfer — the thing ordering B exists to
> shrink — is now the smallest timed stage in the path.
>
> **One cost this path pays that the others do not, left in the numbers.** Every
> other path carries the whole frame to where the tensor is built, so the
> frame-ID marker arrives free. This one resizes *before* anything crosses, and
> a 640-square resize destroys a marker whose cells are 8 px wide and are read
> from row 8. The marker therefore travels as its own 1:1 uint8 crop, 384×16,
> about 24 kB — a second conversion and a second transfer, both inside the
> timings above. Against the 16.38 MB the other paths move for the same
> information, it is cheap; it is still not nothing, and a deployment that
> needed no frame ID would not pay it.
>
> **Verified in this harness too**, not only in the call-duration one:
> `--verify` reports max RGB8 deviation **1** against the shared canonical
> reference, and the run's WHEA guard reported no new hardware records.
>
> **`rapidshot-converter` is `unavailable` here**, correctly: `to_cupy()` on a
> tensor sitting on the iGPU raises `CrossAdapterRequired`, which the worker
> already records as unavailable rather than as a failure. It is the path that
> should win outright on a single-adapter NVIDIA desktop, and it is unmeasured.

> ### Memory, measured for the first time — 2026-09-14
>
> **Re-record before quoting: see § 7.0e.** This was taken with the source
> animating a 900×700 window against a full-screen capture, so 15% of the
> captured area was moving and the frame rates especially describe a
> mostly-still desktop.
>
> § 7.2 has a memory target (*within 1.25× DXcam*) and, until now, no
> instrument that could say whether it was met.
> `benchmarks/memory_pool_stress_test.py` exercises the pool and records no
> process memory at all. `benchmarks/memory_profile.py` is the instrument:
> every library in its own process, three workloads from the same controlled
> D3D source as § 7.0, working set / private bytes / commit sampled at 10 Hz,
> and growth reported as a least-squares slope because a single pair of samples
> cannot tell a leak from allocator warm-up. Recorded in
> `benchmarks/memory-baseline-machineB.json`.
>
> Steady state after a 2 s warm-up, 10 s per cell, 2560×1600. *capture* is
> steady state minus the same process before its first frame — the memory
> capture itself is responsible for, with the import bill removed so RapidShot
> is not charged for `comtypes` where DXcam has no equivalent:
>
> | library | fps | working set | private | **capture** | growth/s |
> | --- | ---: | ---: | ---: | ---: | ---: |
> | mss | 33.0 | 65.8 MB | 34.6 MB | +32.8 MB | +0.003 |
> | dxcam | 145.5 | 103.3 MB | 878.1 MB | +26.3 MB | -0.129 |
> | **rapidshot `grab()`** | 109.7 | **416.0 MB** | 1314.8 MB | **+134.3 MB** | +0.042 |
> | **rapidshot `grab_frame()`** | **165.1** | 282.1 MB | 1253.3 MB | **+0.6 MB** | +0.029 |
>
> Figures are within noise across all three workloads; the static row is
> reproduced above and scroll and motion agree to within a megabyte.
>
> **The target is missed by a wide margin, and the measurement says exactly
> where.** `grab()` is **4.0× DXcam's working set** against a 1.25× target, and
> **5.1× its capture-attributable memory**. But `grab_frame()` costs **0.6 MB**
> — essentially nothing. The surface pool is not the problem. **The ~134 MB is
> the NumPy output path**, and that is also what makes `grab()` the slowest
> RapidShot row here at 109.7 fps while `grab_frame()` reaches the 165 Hz
> ceiling. One defect, both symptoms.
>
> **Nothing leaks.** An early 10-second run showed `grab()` growing at
> +1.29 MB/s on the static workload; at 60 seconds that falls to **+0.066
> MB/s**, so it was warm-up bleeding past the cutoff, not a leak. Worth
> recording because the short run was alarming and wrong — the slope is only
> meaningful over a window much longer than the warm-up it is trying to exclude.
>
> `tests/test_memory_bounds.py` pins the bounded-growth property so the 7.2
> memory work cannot regress it while changing the machinery. It pins *growth*,
> not absolute size: an absolute threshold would encode one machine's
> resolution, and 2560×1600 BGRA is 16.4 MB a frame where 1080p is 8.3.
>
> **Where the 416 MB actually goes**, measured by stage in one process:
>
> | stage | delta | running |
> | --- | ---: | ---: |
> | interpreter + psutil | — | 19.1 MB |
> | `import numpy` | +13.2 | 32.3 MB |
> | **`import rapidshot`** | **+184.8** | 217.1 MB |
> | `rapidshot.create()` | +62.7 | 279.8 MB |
> | 200x `grab_frame()` + release | **+0.2** | 280.0 MB |
> | 200x `grab()` | **+128.8** | 408.9 MB |
>
> **Almost all of the import cost is CuPy, and nothing asked for it.**
> `rapidshot/capture.py` imports `cupy` at module scope to set
> `CUPY_AVAILABLE`, so every `import rapidshot` pays **178.8 MB** for it —
> including on machines with no NVIDIA GPU, and for callers who only ever touch
> `grab()`. The native extension, measured on its own without the package
> `__init__` that pulls CuPy in behind it, costs **1.3 MB**. (An earlier note
> in this session attributed the 182 MB to the extension; that was wrong, and
> the cause was exactly this import chain.)
>
> **So the 1.25x target is reachable, and the two largest levers are not the
> pool.** DXcam's 103.3 MB gives a target of ~129 MB. Deferring the CuPy import
> to first use recovers ~179 MB and is a one-line change; fixing `grab()`'s
> output path recovers ~129 MB. Those two alone land at ~101 MB — **under the
> target, before any buffer-lease redesign**, which would then be addressing
> the 62.7 MB of construction rather than the 308 MB above it. The pool work is
> still worth doing; it is just not where the memory is.

> ### The first lever, pulled — 2026-09-14
>
> CuPy's import is now deferred to first use (`_require_cupy()` in
> `rapidshot/capture.py`; `CUPY_AVAILABLE` and `cp` still resolve, lazily, via
> a module `__getattr__`, so anything that imported them still works). Every
> CuPy use in that module already sat behind `nvidia_gpu`, so nothing else had
> to move. `import rapidshot` went from **+184.8 MB to +20.3 MB**.
>
> Same command, same machine, immediately before and after — both recordings
> are in `benchmarks/memory-baseline-machineB.json`:
>
> | library | before | after | change | vs DXcam |
> | --- | ---: | ---: | ---: | ---: |
> | dxcam | 103.3 MB | 103.3 MB | — | 1.00x |
> | rapidshot `grab()` | 416.0 MB | **238.1 MB** | **-177.9** | 2.30x |
> | rapidshot `grab_frame()` | 282.1 MB | **104.6 MB** | **-177.5** | **1.01x** |
>
> **`grab_frame()` now meets the § 7.2 target**, at 1.01x DXcam against a 1.25x
> bound — from 2.73x, for a change that moved one import. Private bytes fell
> with it (1253 -> 981 MB). Throughput is unchanged: 165.1 fps before and
> after, 109.5 for `grab()`, so nothing was traded for it.
>
> **`grab()` is the whole of what remains.** At 2.30x it is still over, and the
> gap is exactly the +134.2 MB output path — unchanged by this, because this
> did not touch it.

> ### The second lever: three staging buffers nothing could reach — 2026-09-14
>
> `grab()` checks out a BGRA staging buffer from `memory_pool`, and for a
> **converting** mode releases it inside the same call: the caller receives the
> *output* buffer, and `_grab_locked` says so itself — "the BGRA staging buffer
> is finished with either way". `_grab_locked` runs under the duplication lock,
> so exactly one staging buffer is ever in flight. The pool held
> `pool_size_frames` of them anyway: **4 x 16.4 MB at 2560x1600, of which three
> were unreachable**, allocated at `create()` for every RGB camera.
>
> `ScreenCapture._staging_pool_size()` now sizes that pool to 1 when the frame
> the caller receives is not the staging buffer. **BGRA keeps the full count**
> and is the reason the pool is sized this way at all: it converts nothing, so
> the staging buffer *is* the returned frame, and in video mode the capture
> thread checks out more to fill `_pooled_frames_deque`. `pool_size_frames`
> keeps meaning exactly what it documents there. Verified per mode:
>
>     RGB    staging 1 x (1600,2560,4) = 16.4 MB   output 4 x (1600,2560,3) = 49.2 MB
>     BGRA   staging 4 x (1600,2560,4) = 65.5 MB   output none
>
> | | original | after lazy CuPy | **after this** | vs DXcam |
> | --- | ---: | ---: | ---: | ---: |
> | `grab()` | 416.0 MB | 238.1 MB | **199.3 MB** | 1.93x |
> | `grab_frame()` | 282.1 MB | 104.6 MB | **104.5 MB** | **1.01x** |
>
> Throughput is unchanged — 139.4 and 134.4 fps with the fix against 134.6
> without, the same within noise — because serialised grabs never used the
> other three buffers.
>
> **What is left is smaller than it looks, and most of it is not waste.**
> `grab()`'s capture-attributable memory is now +95.4 MB: 16.4 MB of staging,
> **49.2 MB of output pool**, and ~30 MB not yet attributed to anything — the
> next thing to measure rather than the next thing to cut.
>
> The output pool is the part that is a *choice*. Those four buffers exist so a
> caller can hold several frames at once, which DXcam does not offer; DXcam's
> whole capture cost is 25.7 MB, about one staging plus one output buffer.
> Dropping the default `pool_size_frames` from 4 to 2 — the documented minimum,
> since a pool of 1 leaves nothing free while a frame is held — would save
> 24.6 MB and put `grab()` near 175 MB, or ~1.70x.
>
> **So the 1.25x target is not reachable for `grab()` without giving that
> guarantee up.** 1.25x of DXcam's 103 MB is ~129 MB, and RapidShot's baseline
> before its first frame is already 104 MB. Meeting it would mean a capture
> budget of ~25 MB against DXcam's 25.7 — one staging and one output buffer,
> and no multi-frame ownership. That is a product decision, not an
> optimisation, and it should be made deliberately rather than by tuning a
> default. **`grab_frame()` already meets the target at 1.01x**, and it is the
> path the GPU pipeline uses.

> ### The third lever, after testing it rather than assuming it — 2026-09-14
>
> `pool_size_frames` moved from 4 to 2. **What the measurement changed was the
> reason.** The expectation was a trade: fewer buffers, fewer frames a caller
> could hold. There is no such trade, and the first attempt to find one was
> wrong in a way worth recording — the holding loop was bounded by
> `pool_size + 3`, so it could never observe exhaustion and simply reported its
> own limit back. Re-run with a fixed ceiling of 25, **neither size exhausted**:
> both held 25 frames and both recovered after release, because a converting
> mode falls back to allocating rather than refusing. `pool_size_frames` does
> not cap what a caller can hold. It sets where the allocating fallback begins.
>
> | pool | fps | working set | capture | output pool |
> | ---: | ---: | ---: | ---: | ---: |
> | **2** | 139.1 | **171.7 MB** | 70.6 MB | 24.6 MB |
> | 3 | 138.4 | 184.0 MB | 83.0 MB | 36.9 MB |
> | 4 | 134.4 | 196.3 MB | 95.4 MB | 49.2 MB |
>
> Exactly linear: one 12.3 MB RGB buffer per step. And because a steady-state
> loop releases each frame immediately, the pool never runs dry at *any* size —
> so fps there cannot separate them. Holding a rolling window is the only case
> that can, and it does not either: 140.8 / 134.5 / 132.6 fps at pool 2 for
> windows of 1, 3 and 6, against 136.5 / 130.7 / 134.9 at pool 4. Pool 2 is
> ahead at two of the three depths and 1.7% behind at the third, inside the
> spread seen between identical runs all session.
>
> **One thing the A/B did contradict.** `pool_output=False` drops the process to
> 146.1 MB — 49.5 MB below the default — for ~4% fps (135.6 against 140.9). So
> the output pool does buy throughput; it is the buffers *beyond two* that buy
> nothing measurable.
>
> `grab()` is now **174.8 MB, 1.70x DXcam**, from 416.0 MB and 4.04x at the
> start of the day. `grab_frame()` is unchanged at 104.4 MB and 1.01x. The
> paragraph above still stands: closing the last 0.45x means giving up
> multi-frame ownership, and that is a decision about what `grab()` promises.
>
> **The 62.7 MB of construction is now the second-largest item**, not the
> fourth. A buffer-lease redesign would be aimed there, and it is worth
> measuring what it is holding before designing what replaces it.
>
> **Not yet measured:** VRAM attributable to capture. CUDA's `memGetInfo`
> reports the RTX 4060, and capture runs on the Intel iGPU, so the instrument
> points at the wrong adapter on this machine. DXGI's
> `QueryVideoMemoryInfo` is the right call and is not currently exposed.

The broader claim § 7.0 was aiming at — "fastest Windows desktop-to-model
pipeline" — is not earned by this table. It needs present-to-inference measured
as pixel age with a trained model, which was done afterwards (*Present to
inference, with a trained YOLO11n*, below): on this machine every RapidShot path
returned more frames with younger pixels than every other library, on every
pass. One machine is still not the general claim.

Which to reach for:

| If you are | Use | Because |
| --- | --- | --- |
| latency-bound, cores to spare | `grab()` + CPU tensor | 6.87 ms p50 |
| running inference on the same box | cross-adapter | 4.40 CPU ms/frame, no upload |
| in between | `nvidia_gpu=True` | 8.62 ms at 4.84 CPU ms/frame |

**mss is not in this race**: 3.1× DXcam's latency and 1.7× its CPU per frame. It
is GDI-based, not DXGI.

#### The async transfer does not help, and that is the predicted result

`transfer_async()` was expected to hide the ~2.7 ms cross-adapter copy. Measured,
it does not: **96.9 fps / 9.84 ms against the blocking path's 99.7 / 9.60** —
marginally worse on both, inside run-to-run noise, with a slight CPU saving
(4.27 vs 4.40 ms/frame).

That is exactly what § 6.1 already recorded: *"The CPU-side async wait buys
nothing against a GPU consumer"* — the calling thread was never the constraint.
This benchmark now confirms it end to end rather than on a synthetic consumer.

**The variant that should help — the GPU-side semaphore wait — is not a path in
this benchmark.** It was measured afterwards, in the pixel-age benchmark below,
where it is the lowest-latency path of all. It was missing here for a reason
other than the one first written.

**Correction.** An earlier revision of this section claimed no
`cuImportExternalSemaphore` glue existed and made building it the top follow-up.
That was wrong twice over: `tests/test_cross_adapter.py` already contains
working producer *and* consumer semaphore implementations (import, wait, signal,
destroy), and § 6.1 above already records the GPU-side wait verified on
2026-08-22 at **6/6 frames byte-exact**, fence created on the Intel device and
imported by CUDA on the NVIDIA one. The capability was measured a fortnight
before this benchmark was written. What was missing was only its use *here* —
`benchmarks/cuda_semaphore.py` now factors the test-suite implementation out for
reuse. Treat this as a reminder that § 5 and § 6 are the record of what already
works, and are worth reading before declaring something absent.

**Resolved since: `transfer_async_with_reference` was unreachable from Python.**
The extension exported it, but `rapidshot.native`'s `CrossAdapterTransfer`
wrapper listed its methods explicitly with no `__getattr__` passthrough, so an
async transfer's pixels could not be verified from Python at all. The wrapper
now exposes it with `read_back_source()` (2.5.0), and `ai_ingestion.py` and
`section7_adapters.py` verify the async paths with it rather than with the
blocking `transfer_with_reference`.

#### Call duration through inference — measured 2026-09-10 ⚠ superseded

**Superseded by the trained-YOLO11n pixel-age measurement below.** Kept because
its capture-share conclusion still stands; do not quote its figures.

**These are call durations, not present-to-inference latency.** The heading here
first read "present → inference complete", which this benchmark does not
measure: it times from *asking* for a frame to the forward pass completing, and
says nothing about how old the pixels were when they arrived. § 7.0 asks
specifically for pixel age against a shared clock; that was measured afterwards
for ingestion only (below), and this table has not been re-run that way. Read
every figure in it as a lower bound on true present-to-inference latency.

`benchmarks/ai_pipeline.py` feeds each path's tensor straight into ONNX Runtime's
CUDA provider through `io_binding` on the CuPy device pointer, so the tensor
never leaves the GPU and the GPU paths are not charged for a round trip they
exist to avoid. Medians across 3 passes, 8 s per path. Data in
`benchmarks/ai-pipeline-machineB.json`.

| path | fps | tensor ms | infer ms | total ms | **capture %** |
| --- | --- | --- | --- | --- | --- |
| mss | 25.5 | 35.46 | 1.56 | 37.10 | 95.6% |
| dxcam | 76.6 | 10.69 | 1.57 | 12.54 | 84.9% |
| **rapidshot-cpu** | **105.0** | **7.57** | 1.57 | **9.31** | **79.7%** |
| rapidshot-cupy | 101.9 | 8.23 | 1.39 | 9.65 | 85.4% |
| rapidshot-xadapter | 88.0 | 9.85 | 1.41 | 11.30 | 87.4% |
| rapidshot-xadapter-async | 86.6 | 10.08 | 1.46 | 11.54 | 87.3% |

**Capture is 80–96% of the end-to-end budget.** That is the answer § 7.0 was
built to get, and it settles the ordering question: optimising the capture path
is worth real time, because there is very little else in the frame budget.
RapidShot's best path completes a forward pass in **9.31 ms against DXcam's
12.54** — 26% faster end to end, not merely at the tensor boundary. *Forward
pass*, not "a detection": the graph used here is untrained and has no detection
head, so it produces an output tensor of the right shape and no detections at
all. Nothing in this table justifies a claim about detection latency.

**The model is a stand-in and this conclusion survives it anyway.** The recorded
run used a conv stack the harness generated when no `--model` was given, scaled
to **6.28 GFLOPs**, matching
YOLO11n's arithmetic volume at 640×640 but not its layer structure — no concats,
no upsampling, no detection-head semantics, no trained weights. Real YOLO11n has
many small memory-bound layers and is expected to run **slower** than the 1.4–1.6
ms measured here. The conclusion is robust to that:

| if real inference costs | rapidshot-cpu total | capture share |
| --- | --- | --- |
| 1.5 ms (measured stand-in) | 9.3 ms | 80% |
| 5 ms (plausible YOLO11n) | 12.6 ms | 60% |
| 15 ms (a much larger model) | 22.6 ms | 33% |

Capture stays the majority of the budget until inference costs roughly 3× what
this stand-in does, and stays material well past that.

**The harness that produced this table no longer exists in that form.** The
stand-in fallback has been removed: `ai_pipeline.py` is now a front end to
`section7.py --category inference`, which measures pixel age rather than call
duration and refuses to run without `--model` and `--model-sha256`.
`benchmarks/prepare_model.py` exports pinned YOLO11n weights with a recorded
hash (it needs `ultralytics` and torch). **Quote no specific figure from this
table** until that run exists.

**What this says about § 7.2's ordering.** `GpuConverter`, DLPack and multi-ROI
are all justified by removing copies and conversions from the capture path — and
the capture path is where 80% of the time is. That supports the ranking as
written. The same table put the cross-adapter path's 2.3 ms deficit against the
CPU path at ~20% of a frame budget, and that closing it with the GPU-side
semaphore wait would be a real win. Measured as pixel age (below), the
cross-adapter paths trail the CPU path by 1.1–2.4 ms p50 and return ~80 frames a
second against its 140, and the semaphore wait does not close the gap — it is
the slowest of the three cross-adapter variants there, at more CPU. On hybrid
hardware the case for the cross-adapter path is CPU cost, not latency.

**Caveats that travel with the call-duration tables above.** One machine, hybrid
Intel→NVIDIA. On a single-adapter NVIDIA box the direct `GpuPreprocessor12 →
CUDA` path is available and would likely beat all six — it raises
`CrossAdapterRequired` here, because CUDA cannot import a resource owned by the
capture adapter. Frame rate is bounded by compositor presents, not by the
pipeline. The motion source ran uncapped at ≥498 updates/s and is not the
limiter; **a capped source made every path report 82–86 fps and looked like a
tie**, which is the failure `benchmarks/motion_source.py` was written to warn
about.

#### Pixel age, measured against a shared clock — 2026-09-11 ✅

The measurement § 7.0 actually asked for. `native/src/bin/latency_source.rs` is a
D3D11 source that encodes an incrementing frame ID into the image and records
`frame_id → QPC` at every `Present()`; each path decodes the ID out of the
captured pixels, so latency is **how old the pixels were**, not how long the call
took, and every library is measured against one clock.

Machine B, 2560×1600 at 165 Hz, pixels to a `(1, 3, 640, 640)` **FP16** tensor on
CUDA — the contract § 7.0 specified, where the call-duration run above used
FP32. **Medians across 3 passes, 8 s per path**; the bracket is the max–min spread
across passes. Every path passed verification first (CPU paths within 1 RGB8
level of the exact reference, GPU paths bit-exact).
`benchmarks/section7-ingestion-machineB.json` has every pass, with per-stage
timings (capture call, GPU preprocess, fence wait) and the QPC frequency.

Age is measured from the QPC taken immediately before `Present()`, so it is
*submission* age: it includes the compositor's queue, not scan-out to the panel.

| path | unique fps | age p50 | p95 | p99 | CPU ms/frame | source frames dropped |
| --- | --- | --- | --- | --- | --- | --- |
| mss | 33.0 [0.9%] | 57.64 [1.0%] | 60.72 | 64.63 | 15.27 | 788 of 1051 |
| dxcam (DXGI) | 108.2 [2.5%] | 36.55 [1.1%] | 39.58 | 40.55 | 8.94 | 458 of 1322 |
| dxcam (WGC) | 106.0 [2.1%] | 40.73 [0.6%] | 45.08 | 48.19 | 8.73 | 476 of 1321 |
| **rapidshot-cpu** | **140.5** [4.2%] | **33.57** [1.6%] | **36.13** | **38.05** | 7.73 | **199** of 1321 |
| **rapidshot-cupy** | 128.8 [5.1%] | 34.42 [0.5%] | 36.56 | 39.10 | **4.44** | 293 of 1320 |
| rapidshot-xadapter | 80.2 [2.4%] | 35.13 [3.5%] | 40.20 | 42.56 | 4.92 | 681 of 1319 |
| rapidshot-xadapter-async | 80.8 [1.8%] | 34.64 [3.3%] | 40.34 | 42.55 | 4.95 | 673 of 1318 |
| rapidshot-xadapter-semaphore | 81.4 [0.1%] | 35.96 [0.2%] | 36.98 | 42.01 | 11.02 | 670 of 1320 |
| rapidshot-direct | — | unavailable: `CrossAdapterRequired` on hybrid hardware |

**`grab()` and `nvidia_gpu=True` beat DXcam on every axis, on every pass.**
`rapidshot-cpu` returns **30% more unique frames** than DXcam, pixels **3.0 ms
(8%) younger**, at 14% less CPU per frame; `rapidshot-cupy` returns 19% more
frames, 5.8% younger pixels, at **half DXcam's CPU**. Both hold the stricter
test: each one's worst pass beats DXcam's best on frames, age and CPU, and the
same is true against DXcam's WGC backend.

**The cross-adapter paths are capped at ~80 frames a second** — about 25% fewer
than DXcam — while still returning younger pixels on every pass (1.6–5.2%) and,
blocking or async, costing ~45% less CPU. The cap is the same across all three
variants and both recording sessions, which points at the copy itself: a full
2560×1600 BGRA frame is 16 MB through the system-memory shared heap. The
GPU-side semaphore wait buys no latency here (35.96 ms p50, against 35.13
blocking) and costs 23% *more* CPU than DXcam; on this hardware it is the one
configuration with nothing to recommend it.

**This supersedes the single 5 s pass recorded 2026-09-10**, and the change is
worth stating rather than smoothing over. That pass had RapidShot at 100.4
unique frames against DXcam's 76.3, and the semaphore path as the lowest-latency
path at 33.95 ms. Today every path that reads frames back to the CPU runs 38–56%
faster in absolute terms (DXcam 76.3 → 108.2, RapidShot 100.4 → 140.5, WGC
68.0 → 106.0) while the cross-adapter paths reproduced within
2%, so the cross-adapter paths went from matching DXcam's frame rate to trailing
it, and the semaphore path lost its latency lead. The cause of the session-to-
session shift is not established. The ranking of `grab()` and `nvidia_gpu=True`
above DXcam survived it; the claims that did not are the reason not to quote
single passes.

**The source was briefly the limit for the fastest path.** It presented at a
median 165.1/s, but one 2-second window dipped to 128.7/s — below
`rapidshot-cpu`'s average — so that path's frame rate is, if anything,
understated. Everything drops source frames; RapidShot's CPU path drops the
fewest (199 of ~1,320).

#### Present to inference, with a trained YOLO11n — measured 2026-09-11 ✅

The same controlled source and pixel-age clock, now carried through a trained
model: each path's FP16 tensor goes into **YOLO11n** on the RTX 4060 through ONNX
Runtime's CUDA provider, and age is taken when the forward pass completes.
**Medians across 3 passes, 8 s per path**; the bracket is the max–min spread
across passes as a percentage of the median. Every path passed verification
before timing (CPU paths within 1 RGB8 level of the exact reference, GPU paths
bit-exact). `benchmarks/section7-inference-machineB.json`, with every pass and
the model's provenance.

| path | unique fps | age p50 | age p95 | YOLO11n ms | CPU ms/frame |
| --- | --- | --- | --- | --- | --- |
| mss | 26.8 [0.7%] | 53.66 [1.7%] | 63.95 | 4.99 | 20.25 |
| dxcam (DXGI) | 68.1 [0.3%] | 42.18 [0.2%] | 45.76 | 5.44 | 13.36 |
| dxcam (WGC) | 65.8 [2.3%] | 45.41 [1.5%] | 49.80 | 5.50 | 14.62 |
| rapidshot-cpu | 80.3 [0.8%] | 39.34 [0.4%] | 43.34 | 4.88 | 12.73 |
| **rapidshot-cupy** | **81.8** [1.6%] | 38.75 [1.7%] | 42.53 | 4.17 | **8.89** |
| rapidshot-xadapter | 75.2 [3.2%] | 39.57 [0.7%] | 42.60 | 4.18 | 8.91 |
| rapidshot-xadapter-async | 76.2 [3.7%] | 39.63 [1.4%] | 42.81 | 4.21 | 9.12 |
| **rapidshot-xadapter-semaphore** | 81.0 [2.4%] | **38.32** [2.3%] | **41.07** | 4.08 | 11.90 |
| rapidshot-direct | — | unavailable: `CrossAdapterRequired` on hybrid hardware |

**Every RapidShot path beat DXcam on every pass**, on all three axes that
matter: **10–20% more unique frames, pixels 2.6–3.9 ms (6–9%) younger at
inference completion, and 5–33% less CPU per frame.** The check that makes that
more than a median: each RapidShot path's *worst* pass against DXcam's *best*
still wins every one of those comparisons. Against DXcam's WGC backend and mss
the same holds for frames and pixel age; on CPU per frame one pair overlaps —
`rapidshot-cpu`'s worst pass (12.81 ms) against WGC's best (12.39) — although the
medians still favour RapidShot (12.73 against 14.62). The best configurations against
DXcam: `nvidia_gpu=True` at +20% frames and a third less CPU; the GPU-side
semaphore wait at 9% younger pixels (10% at p95).

**The gap narrows once a model is in the loop, as it should.** Tensor-only,
RapidShot returned 30% more unique frames than DXcam; here it is 20%. The loop is
capture → preprocess → infer, run synchronously, so YOLO11n's ~4–5 ms per frame
paces every path and dilutes a capture-side advantage. A pipelined consumer
would recover some of it; this benchmark does not measure one.

**RapidShot's own paths are roughly tied on latency** — within 1.3 ms of each
other at p50, which is inside their pass-to-pass spread. Choose between them on
frames and CPU, not latency. (A single trial pass before this run suggested the
cross-adapter paths lost their latency lead; three passes do not support that.)

**YOLO11n itself took 4.1–4.2 ms after GPU-side paths and 4.9–5.5 ms after
CPU-side ones.** Same model, same GPU, same stream binding; the likely cause is
CPU contention delaying kernel launches in the busier CPU paths, but it has not
been isolated, so it is recorded rather than claimed as a RapidShot effect. It
sits inside the age column either way.

Caveats that travel with this table:

- **One machine, hybrid Intel→NVIDIA.** The direct single-adapter path could not
  run.
- **Age is submission age** (QPC before `Present()`), not photon age.
- **Raw detector output, no NMS or postprocessing timed.** Forward pass only.
- **The model file is our export of the official weights, not the published
  `yolo11n.onnx`.** The weights are the Ultralytics v8.3.0 release asset (byte
  size matches; GitHub publishes no digest for it), exported with Ultralytics'
  own exporter at opset 17 by `prepare_model.py`, and all 239 nodes run on CUDA.
  The published ONNX (digest-verified) is opset 22, for which ONNX Runtime 1.30
  has no CUDA `MaxPool` kernel: 7 of its 246 nodes fall back to the CPU, and ORT's
  spinning CPU thread pool then dominates the CPU column and contends with
  capture. In a single trial pass it ranked the paths the same way (+23% frames
  and 9.5% younger pixels against DXcam), but its absolute numbers describe that
  fallback, not capture.

This supersedes the stand-in call-duration table above for any claim about
inference.

**Three harness bugs had to be fixed before any of this could run**, and none
of them had ever been hit, because the inference category had never completed a
run:

- It required `get_providers() == ["CUDAExecutionProvider"]`, but ONNX Runtime
  always lists the CPU provider as registered, so no model could pass. The
  placement guarantee is `disable_cpu_ep_fallback`, which fails session creation
  if any node lands on the CPU; that is what caught the published file.
- It required a fixed `[1, 3, 640, 640]` input declaration, rejecting the
  published file's symbolic axes. They are now pinned with ORT's free-dimension
  overrides, recorded in every row, leaving the file untouched.
- Verification still demanded a bit-identical tensor after the measured loop
  moved to `pipeline_rgb`, so every CPU path failed before it could be timed. It
  now allows the documented 1-level tolerance and records the deviation seen.

#### Two apparatus bugs found while getting there, both worth recording

**The source default made every path report a tie.** `--motion-fps` defaulted to
60 on a 165 Hz panel, so every path returned almost exactly 60 unique frames a
second and the run looked like a throughput tie. An earlier attempt capped at 120
achieved 64 and produced the same artefact at 82–86 fps. It now defaults to the
display's physical refresh, and the achieved rate is printed so it can be
checked. This is the third distinct time this failure has appeared in this
project; `benchmarks/motion_source.py` was written to warn about it and its
warning was deleted in a rewrite.

**The shared tensor contract was a strawman CPU arm.** To make every path emit a
bit-identical tensor, both CPU and GPU ran `canonical_rgb`, an exact rational
bilinear resize. On CuPy it costs 1.16 ms; on NumPy its int64 gathers cost
76–86 ms, against **0.72 ms** for the `cv2` call any real caller writes. That
charged mss, DXcam and rapidshot-cpu roughly seventy times their true conversion
cost — and all three are competitors to the GPU paths being judged, so the bias
ran in RapidShot's favour. Bit-exactness was chasing the letter of "must produce
a bit-identical tensor" past its purpose, which is to stop a path doing *less*
work and looking fast.

Measured now with `pipeline_rgb`: idiomatic per backend, **max deviation 1/255**
against the exact contract, which remains the reference every path is verified
against. § 10 records the same correction being needed once before.

#### What is still not measured

In rough order of value. The harness for most of it exists; what is missing is
the run.

- **The agent category.** Built into `section7.py` (`--category agent`, OS
  pixels → PNG/JPEG data URL for cloud computer-use pipelines, CPU paths only)
  and never run. Ingestion and inference have both been measured as pixel age.
- **A pipelined inference loop.** The inference run is synchronous — capture,
  preprocess, infer, repeat — so the model's time paces every path. A consumer
  that overlaps capture with inference is the realistic deployment and would
  show more of the capture-side difference.
- **The published `yolo11n.onnx` fully on the GPU.** It needs an ONNX Runtime
  whose CUDA provider has an opset-22 `MaxPool` kernel; until then it runs 7
  nodes on the CPU and its absolute numbers describe that fallback.
- **The direct single-adapter path.** `rapidshot-direct` raises
  `CrossAdapterRequired` on this hybrid machine. It needs one where the NVIDIA GPU
  drives the display, and is expected to beat every path measured here.
- **Resolution and refresh sweeps**: only 2560×1600 at 165 Hz is recorded. The
  spec asks for 1080p / 1440p / 4K and 60–240 Hz. `section7_suite.py` runs the
  matrix resumably, but each physical display mode has to be selected by hand —
  it will not substitute source pacing for a real refresh rate.
- **Workload sweep**: only `motion` is recorded. `--workload static` and
  `scroll` exist; the spec asks for all three because they produce very
  different dirty fractions.

#### 7.0a — Durable results, phase 1 ✅ 2026-09-19

Everything above was measured by harnesses that could not say, after a crash,
which of their results were trustworthy. `section7.py` rewrote one whole JSON
payload after every path, `section7_suite.py` rewrote a matrix of cell statuses,
`perf_suite.py` and `compare_libraries.py` held results in memory until the end.
Each individual write was atomic; none of them was a record.

`benchmarks/result_store.py` is now that record, and every supported runner
writes to it. The unit is a **case** — benchmark x path x configuration x
workload x repeat — committed once, immutably, with its raw samples, before the
next case starts.

**The ordering is the design.** Samples, then the result, then a manifest
hashing both, then the journal record pointing at the manifest. A crash at any
point either leaves no commit record — the case reads back as `interrupted` and
is retried — or leaves one whose artifacts are all present and hash-verified.
No ordering exists in which a commit record refers to a partial result.

**What it refuses to do, and why that is the point:**

- A **truncated final journal record** is expected and accepted; the torn bytes
  are moved to a sidecar rather than deleted. **Damage in the middle** raises
  and leaves the file untouched — a journal with a hole has lost the ordering
  that makes the rest meaningful, and guessing is the silent misclassification
  the store exists to prevent.
- A **committed case is never rewritten.** A retry creates `attempt-002`, so
  "retried after a crash" and "measured twice" stay distinguishable.
- **Failed, invalid and contaminated results are retained with their reasons.**
  Discarding them hides the two things a history is most often needed for: what
  broke, and what was thrown away.
- A **resume never re-measures settled work**, and never retries a failure
  automatically — that needs `--retry-failed`. A suite that silently re-runs
  whatever failed can loop on a broken machine until it produces a number
  someone likes.

`benchmarks/result_validation.py` classifies each row before it is committed:
`passed` / `unavailable` / `failed` / `invalid` / `contaminated`. It is a
**structural and physical** check, not a statistical one — percentile ordering,
a rate against its own frame count, stage timings that fit inside the
end-to-end figure they decompose. It says nothing about confidence; that needs
repeated runs and is Phase 3. The stage check deliberately runs one way only:
it catches parts that do not fit inside the whole, and never adds stage
percentiles together to invent an end-to-end one.

**Two contamination flags are live already.** `section7.py` flags a case whose
source presented more slowly than the path returned — the tie-that-is-not-a-tie
failure this section records hitting three times. `ai_ingestion.py` and
`compare_libraries.py` flag a run with no controlled source at all.

**Covered by 74 headless tests** (`tests/test_result_store.py`,
`test_result_validation.py`, `test_benchmark_history.py`): a real killed parent
process, injected disk-full, a commit that fails at the manifest, duplicate
attempts, truncated records, mid-file corruption, altered artifacts, a
rewritten manifest, and a store overtaken by another process. None of them
crashes Windows to do it. Suite total: **1683 passing**.

**Three bugs this found in existing code**, none of which had been hit:

- Appending to a journal after a torn write spliced the new record into the
  partial line, turning a recoverable truncation into the mid-file corruption
  the store refuses to read.
- A store handle held across child invocations (`section7_suite.py` resumes the
  same run once per cell) would append with a stale sequence number, putting a
  duplicate `seq` in the journal and destroying the readability of every record
  after it. `StaleStore` now refuses.
- `memory_profile.py` and `perf_suite.py` wrote their result JSON with a plain
  `write_text`. A death mid-write left a truncated file where a recorded
  baseline used to be. Both are atomic now.

**Also removed from the supported set:** `granular_performance_test.py` and the
five `*_max_fps.py` scripts, now in `benchmarks/unsupported/` with a README
stating each defect. All five `max_fps` scripts charge device creation to the
measured window — and `rapidshot_max_fps.py` additionally charges itself a
warm-up grab and a `time.sleep(0.1)` the competitor scripts do not pay, so
their numbers cannot be compared with each other in *either* direction.
`granular_performance_test.py` reports `1 / (mean(capture) + mean(process))`
where the two means are taken over different denominators; on a static desktop
that describes its 10 ms acquire timeout rather than the capture path.

**History lives in `build/performance-history/`** — git-ignored, and
deliberately not beside the committed baselines in `benchmarks/`. Those are
release gates read by `make_badges.py --check` and `perf_suite.py --compare
auto`; conflating the two is how a baseline gets silently re-recorded by a
routine run. CI passes `--no-history` throughout: a runner is discarded and
nobody reads what it wrote.

**Not done here, and stated rather than implied.** Every run currently records
`"environment_snapshot": "incomplete (phase 2 not yet integrated)"`. There is
still no git commit, dirty-source fingerprint, native-binary hash, driver, GPU
or affinity in the metadata, and the P-core pinning in `perf_suite.py` is still
not applied by `section7.py`, `ai_ingestion.py` or `memory_profile.py` — the
gap the FP16 table above already admits to. That is Phase 2. `perf_suite.py`
also commits its cases *after* the run rather than between benchmarks: its
microbenchmarks are sub-second, the crash window is barely open, and
restructuring `merge_rounds` to commit incrementally would change how a release
gate computes its figures for the guarantee it needs least.

#### 7.0b — Verified environment, phase 2 ✅ 2026-09-19

Phase 1 made results durable. They were still anonymous: every run recorded
`"environment_snapshot": "incomplete"`, and the FP16 table above carries the
caveat *"not P-core-pinned"* only because someone remembered to type it.
`benchmarks/machine_inventory.py` and `benchmarks/telemetry.py` replace
remembering.

**One CPU detector, now reachable.** The hybrid P-core detection had lived
inside `perf_suite.py` since § 2, private to it — which is precisely why
`section7.py`, `ai_ingestion.py` and `memory_profile.py` never pinned, and why
the ingestion tables above are unpinned recordings. It moved to
`machine_inventory`, and all five runners now apply the same policy through
`prepare_run()`. `perf_suite.py` keeps its function names and every key
`print_comparison` reads out of committed baselines, so no baseline in the
repository was invalidated.

What the detector now reports, verified on this i9-14900HX: **24 physical
cores / 32 logical**, two efficiency classes — 8 P-cores *with* SMT (mask
`0xffff`) and 16 E-cores without (mask `0xffff0000`) — one processor group,
per-level cache sizes, and the affinity already in force before any policy is
applied. SMT is recorded per core, not per machine, because on this CPU it is
per class. Each class also reports whether it is `homogeneous`: an efficiency
class is a scheduling hint, not a promise that its members are identical.

**Pinning is applied, then verified, then inherited, then verified again.**
`apply_cpu_policy()` reads the mask back rather than trusting the call, refuses
to widen an inherited restriction (it intersects), refuses a partial pin on a
multi-group machine, and distinguishes *uniform* from *unreadable*. Each worker
calls `verify_affinity()` **before importing NumPy, CuPy or ONNX Runtime** —
those size their thread pools from the affinity they see at initialisation, and
a pool built for 32 cores then run on 16 is slower than either choice made
consistently. A worker on the wrong cores contaminates its case.

**Pinning narrows the distribution. It does not make timings deterministic.**
Clocks, thermals, background work and GPU contention all still move them. That
sentence is in the module docstring because the alternative reading is the one
that produces overconfident tables.

**Displays: the rounded refresh rate was hiding something.**
`EnumDisplaySettingsW` — what the harness has always used — reports this panel
as **165 Hz**. `QueryDisplayConfig` reports what the driver actually
programmed: **165.00178 Hz** (77733000/471104). Source pacing is compared
against the rounded figure, and that comparison decides whether a throughput
number describes the apparatus, so the rounding was never cosmetic. Both are
recorded now, with a flag when they disagree by more than 1 Hz.

Also recorded per output: monitor device path (the stable identity — the
friendly name is empty on this internal panel, and is not unique when two
identical monitors are attached), **owning adapter LUID and PCI path**,
rotation, scaling mode, desktop rectangle, and per-monitor DPI. This machine
reports `PCI\VEN_8086&DEV_A788` for the display adapter, which makes *"capture
is on the iGPU"* a recorded fact rather than an inference from a device
description string — and that LUID matches the one `test_gpu_tensor_export.py`
prints when it skips for want of a CUDA device on the capture adapter.

**A DPI finding worth having.** This display runs at **150% scaling**. The
harness is per-monitor DPI aware, so `physical_pixel_mapping.matches` is true —
but a DPI-unaware benchmark process here would be handed 1707×1067 by every
coordinate API while the panel is 2560×1600, and every region computed from
that would address the wrong pixels. The check is explicit now, and warns
rather than being discovered later.

**Provenance, with and without git.** Commit, branch, dirty flag, **and a
dirty-source fingerprint** — this tree is dirty in 56 files, so the commit
alone describes code that is not what ran. Plus a content fingerprint of every
Python file under `rapidshot/` and `benchmarks/`, computed from disk, so a
copied folder with no `.git` and no git installed still identifies its source —
which Phase 3 depends on. Plus the **hash and origin of the native `.pyd` that
actually loaded**: `baseline.json` is recorded with the extension and
`baseline-nonative.json` without it, and a boolean "is it available" was never
enough to tell two builds apart. Plus both GPUs with driver versions, RAM
modules with configured speed, power source and power-scheme GUID.

**Telemetry, with an explicit third state.** Providers: `psutil` for load and
memory, `CallNtPowerInformation` for per-processor MHz, and **NVML through
ctypes** — not `nvidia-smi`, because a process launch per sample is the
opposite of what a conditions sampler should do — for GPU utilisation, clocks,
temperature, power and VRAM. Measured on this machine: **6.2 ms CPU per sample,
a 0.6% duty cycle at the 1 s default**. Measured, not assumed:
`measure_overhead()` exists so nothing gets switched on by default on a guess.
Intervals below 250 ms are refused, because these counters are documented for
low-frequency collection and sampling them faster measures the sampler.

**The frequency trap, handled.** `CallNtPowerInformation` is the only Windows
API reporting a current per-processor clock, and on many machines it returns
the nominal figure forever. A flat series then looks exactly like a machine
holding its boost clock rock-steady, which is the opposite conclusion. So the
sampler watches across the whole window: a series that never varies *and*
equals the reported maximum is marked as the nominal figure being echoed back,
explicitly **not** as evidence the clock was held. On this machine it does vary
— 1466 to 2200 MHz across processors — so here it is measuring. One query could
never have established that.

**Background load is a warning, not a verdict.** The sampler is rooted at the
benchmark's own process, so the parent, every worker and the motion source
generating the workload all count as *this benchmark's* CPU. What is left over
is reported as a residual that still contains the compositor, the shell and
driver threads — flagged for a human, never used to discard a measurement here.

**Drift detection between cases**, and it had to be made cheap to be usable:
the first version re-hashed the whole source tree and shelled out to git three
times *per case*. It now costs **2 ms**, against 1.2 s for a full snapshot, and
reports `not_checked` for the sections it skipped rather than letting "not
looked at" read as "did not change". A changed display configuration, a laptop
coming off mains, a changed topology or a rebuilt package marks the case
`contaminated` and sets `comparable: false`, so no automatic comparison can
step over it.

**Three bugs this found:**

- `machine_id` was built partly from the CIM query, so a cheaper re-check that
  skipped CIM computed a *different* id for the same box, and every case
  reported that the hardware had been swapped mid-run. Anything that can be
  unavailable has to stay out of an identifier.
- Four Win32 struct offsets were wrong, and the failure was silent rather than
  loud: `PROCESSOR_RELATIONSHIP` has 20 reserved bytes before its group count,
  so every core reported **zero** logical processors while the totals still
  looked plausible. `CACHE_RELATIONSHIP` and `GROUP_RELATIONSHIP` were similar.
- `SYSTEM_CPU_SET_INFORMATION` has no frequency field at all; an earlier draft
  read one out of the bytes where `AllocationTag` lives and got zeros. Nominal
  clocks now come from the registry, labelled as nominal.

**Covered by 65 new headless tests** (`test_machine_inventory.py`,
`test_telemetry.py`), plus three in `test_benchmark_history.py` for
contamination actually reaching a case. Suite total: **1749 passing**.

One of those tests found something worth recording: running the full suite
loads this machine enough that the new background-load detector contaminated
the store-wiring tests. That is the detector working, and it is why those tests
now pass `--no-telemetry` — the detector is tested with synthetic series
instead of by hoping the machine is quiet.

**Not done here.** Sustained-frequency telemetry is validated on one machine
only; on a box where the provider echoes nominal, the series is correctly
refused rather than corrected, but that path is unexercised. Cache/CCD topology
is read where Windows exposes it and has not been checked against an AMD part —
that needs the second machine. And nothing yet *uses* `comparable: false` to
gate a regression verdict, because there is no statistical comparison to gate.
That is Phase 3.

#### 7.0c — Detections and defensible statistics, phase 3 ⚠ built, first run in § 7.0f

> **The blocking scene gap below is closed** — see § 7.0d. The rest of this
> section's "what is not done" list still stands.

Phases 1 and 2 made results durable and attributable. This phase is about what
may be *concluded* from them, and about measuring the thing the § 7.0 inference
table stops short of. **The machinery is built and tested; no live detection
run has been made, and the reasons are at the bottom.**

##### The boundary moved

The inference table above ends its clock at forward-pass completion, and says
so: *"Raw detector output, no NMS or postprocessing timed."* An application
does not receive a `(1, 84, 8400)` tensor on the GPU; it receives boxes,
classes and confidence scores it can branch on. `benchmarks/detection.py` ends
the clock there instead — confidence filtering, NMS, coordinate restoration and
**the transfer to the host** are all inside the timed call.

**So every inference figure recorded before 2026-09-19 is incomparable with
every one recorded after it.** Not smaller or larger — measuring a different
interval. `ai_pipeline.py`'s docstring says so where someone will read it.

Postprocessing runs on the device when the ORT output can be wrapped without
copying and falls back to the host otherwise, and **which one happened is
recorded in every result**. The two cost very differently, and a table mixing
them silently would credit one path with a transfer it never made.

##### Geometry is a correctness problem

Ultralytics letterboxes — one scale factor, padded to square. The existing
§ 7.0 tensor contract stretches the full frame to 640×640. Both are
defensible; comparing one against the other is not, because the model sees
differently distorted objects and the boxes come back in different coordinate
systems. `Geometry` makes the choice explicit, invertible and part of the case
identity.

The inverse is checked against hand-computed values in the tests, because the
obvious mistake here is silent: removing the pad *after* dividing by the scale
instead of before offsets every box by about 480 source pixels on this panel —
far enough to land on a different object, close enough to read as tracking
wobble.

##### What a comparison has to clear before it is a finding

`benchmarks/statistics_report.py` — `calibrate_noise()` and
`compare_paired_runs()`. The rules are enforced, not documented:

- **The unit of observation is a run, never a frame.** Consecutive frame
  timings share a thermal state, a scheduler decision and a compositor cadence;
  treating a few thousand of them as independent draws gives an interval narrow
  enough to make anything look certain. § 7.0 already shows the cost — a single
  5-second pass had the semaphore path as lowest-latency and three passes did
  not support it. A series long enough to be frames is **refused** with
  `NotIndependent` rather than quietly averaged.
- **Paired and interleaved.** Run *i* of A pairs with run *i* of B, so the
  machine's warm-up drift lands on both halves instead of on whoever went last.
  The recorded order travels into the report.
- **A t interval, not 1.96.** At five pairs the normal approximation understates
  the half-width by about 30%. An untabulated df rounds *down*, so the interval
  is never accidentally narrowed.
- **Three things must line up** before `improvement` is returned: the interval
  clears a threshold **fixed in advance**, the unchanged-code calibration for
  that metric on that configuration can resolve a difference that size, and
  there are enough pairs for the metric. Otherwise `inconclusive` — which is a
  result.
- **Tail metrics need more than five pairs.** Each run's own p99 is estimated
  from its slowest handful of frames, so five runs cannot place one. Ten is a
  floor, labelled as a floor.
- **The stopping rule is enforced.** `compare_paired_runs` takes the number of
  pairs that were *planned* and refuses every verdict if a different number
  arrives. Running until the answer looks good is the easiest way to manufacture
  a finding, and an honour system does not prevent it.

Spread is never printed as uncertainty. The max–min bracket § 7.0 reports
across passes describes how far the numbers moved; it is not a confidence
interval and the report says so in both tables.

##### The self-test grew up, and immediately said something

§ 7.0 asked for this to extend `perf_suite.py --self-test` rather than become a
separate mechanism, and it does: `--calibrate N` is the same idea carried far
enough to answer the question two passes cannot. `--self-test` now ends by
saying it is a sanity check and pointing at `--calibrate 5`.

Run at **`--rounds 1 --reps 3`** — deliberately tiny, to see what a cheap
configuration can support — five unchanged-code runs gave:

| metric | median | spread across identical runs | resolvable difference |
| --- | ---: | ---: | ---: |
| `shot.RGB.median_ms` | 0.2589 | 0.1707 (**65.9%**) | 0.0938 |
| `shot.BGRA.median_ms` | 0.3056 | 0.1705 (**55.8%**) | 0.1004 |
| `shot.GRAY.median_ms` | 0.3367 | 0.1671 (**49.6%**) | 0.0787 |
| `shot.RGBA.min_ms` | 0.2852 | 0.0537 (18.8%) | 0.0275 |

**At these settings the default 1.30× regression threshold is below the noise
floor**, so a gate run this cheaply would fire on nothing. That is not a claim
about CI, which runs `--rounds 5 --reps 25` and will be much tighter — it is
the point of keying a calibration to its configuration, and the first
measurement that shows why. Re-run `--calibrate` at CI's settings before
quoting anything about the gate.

Note `min_ms` is the steadiest column, which is the reason `print_comparison`
already prefers minimum samples.

##### Covered by 82 new headless tests

`test_statistics_report.py` and `test_detection.py`. Every way of getting an
unearned finding has a test: too few runs, no threshold, a threshold below the
machine's resolution, a tail metric at five pairs, an unpaired set, and frames
passed off as runs. All return `inconclusive`. The interval arithmetic is
checked against hand-computed values so a change to it cannot pass by agreeing
with itself. Suite total: **1829 passing**.

One of those hand-computed values was wrong in the test and right in the code,
which is the correct direction for that to happen.

##### What is not done, and why

**No live detection run has been made, and one made today would measure
nothing.** `native/src/bin/latency_source.rs` draws a frame-ID marker pattern —
exactly what the pixel-age clock needs, and containing no object YOLO11n
recognises. A detection run against it would find zero detections in every
frame, time the cheapest possible postprocess, and report a number that
describes an empty scene. The harness now warns when that happens
(`no detections in any frame`) rather than reporting it as a result, but the
warning is not a substitute for the work: **the source needs to render
reproducible scenes containing actual objects**, in static, scrolling and
motion variants, and that is the next thing to build.

Also outstanding, in rough order of value:

- **The three comparisons are specified but not built as adapters.** Standard
  Ultralytics screenshot pipeline vs RapidShot into the same detector;
  MSS/DXcam/RapidShot through matched preprocessing; identical preloaded
  tensors through YOLO11n to isolate model execution. The contract that makes
  them comparable exists; the three configurations do not.
- **Thresholds have not been chosen.** `compare_paired_runs` refuses a verdict
  without one, by design, so this is a blocking gap rather than a missing
  nicety. What counts as a practically meaningful millisecond saving for a
  capture pipeline is a product decision and should be written down before any
  comparison runs, not after.
- **A calibration at the settings anyone would actually quote.** The table
  above is from a deliberately cheap configuration.
- **Everything § 7.0 already lists as unmeasured** remains unmeasured: the
  agent category, resolution and refresh sweeps, the workload sweep, the
  published `yolo11n.onnx` fully on the GPU, and `rapidshot-direct` on a
  single-adapter machine.
- **The second machine.** Nothing here has run anywhere but Machine B.

#### 7.0d — A scene with objects in it ✅ 2026-09-19

§ 7.0c's blocking gap: `latency_source.rs` draws a frame-ID marker pattern,
which is exactly what the pixel-age clock needs and contains nothing a detector
recognises. A detection run against it finds zero objects in every frame, times
the cheapest possible postprocess, and reports a number describing an empty
screen. `benchmarks/scenes.py` is the fix, and it is **verified against the
real model rather than assumed**.

##### What a detector actually sees in a drawing

The whole thing turns on a question nobody should answer from intuition, so it
was measured first. Objects drawn procedurally with OpenCV, put through the
exported `yolo11n.onnx`:

| drawn | detected |
| --- | --- |
| stop sign | `stop_sign` **0.93** |
| clock | `clock` **0.90** |
| traffic light | `traffic_light` **0.82** |
| keyboard | `keyboard` **0.66** |
| cup | `cup` 0.64 in company, **nothing** alone |
| person, laptop, bottle, sports ball | **nothing** |

So the scene is built from the four that hold up alone. A crude rectangle is
not a laptop to a detector trained on photographs, and a scene built on the
assumption that it is would silently degrade to empty — which is the exact
failure this work exists to remove.

**The marker and the objects coexist.** Composited over a full scene, the
48-cell frame-ID pattern leaves detection counts identical and moves
confidences by at most 0.02. The pixel-age clock and the detector can share a
frame, which was not obvious and is now checked.

##### The scene

A tileable canvas at twice the capture size, panned over by the source. The
three workloads become three dirty fractions of one scene rather than three
different scenes: `static` holds still (only the marker changes, the smallest
possible dirty rectangle), `scroll` advances vertically, `motion` advances on
both axes so objects travel diagonally.

Verified at 2560×1600 against the real model, 12 sampled frames per workload:

| workload | detections per frame |
| --- | --- |
| static | 8 – 8 (mean 8.0) |
| scroll | 7 – 8 (mean 7.8) |
| motion | 6 – 9 (mean 7.8) |

`scenes.py --model` writes those detections beside the scene as a reference and
**refuses to call a scene usable** below four per frame. `section7.py` reads
that file: an inference run against an unverified or failed scene is refused
before anything is measured, with the command that fixes it. An ingestion run
without a scene is still fine — pixel age never needed objects.

##### Two bugs the verification caught, which is the point of having it

- **The first scene was too sparse.** 12 objects on the canvas put about three
  in view, below the four-per-frame floor, and the verifier refused it. Density
  went to 35.
- **Object sizes were fixed in screen pixels, so they were a different object
  to the model at every resolution.** A 110px clock is 55 model pixels at
  1280×800 and 27 at 2560×1600 — detected in one case, missed in the other,
  for a reason nothing in the configuration would reveal. Sizes are now
  declared in *model space* and divided by the letterbox scale, taken directly
  from the drawings that verified.

A third was caught by eye rather than by the model: the class assignment used
an arithmetic phase, `(row * 7 + column + row) % 4`, which looks well mixed and
reduces to `column % 4`. The first rendered scene was three vertical stripes of
identical objects. It is a seeded shuffle now.

##### The source

`latency_source.rs` takes five optional arguments: a raw BGRA path, the canvas
dimensions, and the pan step. **Without them it behaves exactly as it always
has**, so every existing recording stays reproducible and the pixel-age
workload is untouched.

The scene arrives as raw bytes plus numbers on the command line rather than as
a file the binary parses — there is no JSON reader in that crate, and four
integers do not justify adding one. The shader uses `Load`, not `Sample`: an
exact texel with no filtering, so what is presented is byte-identical to the
window `scenes.py` put through the detector. A bilinear sampler would present
something slightly different from the thing that was checked.

The scene is loaded and its length validated against the stated dimensions
*before* the window is created — this draws over the desktop during a
benchmark, and a bad scene should fail without flashing a window first. `cargo
fmt`, `clippy -D warnings` and the build are all clean.

Writing it found one more thing: the argument-count guard still demanded
exactly six, so the scene arguments would have been rejected outright. `cargo
fmt --check` surfaced the line it was on.

##### Covered by 29 new headless tests

`test_scenes.py`: model-space sizing, class mixing, canvas wrap with no seam,
objects drawn across the seam, marker layout matching the shader byte for byte,
content-derived scene identity, a tampered background refused on read, and
every one of `section7.py`'s refusals. No model is loaded — that is
`scenes.py --model`. Suite total: **1858 passing**.

##### What this still does not do

**Objects do not move relative to each other.** Panning moves everything
together, so this exercises a detector's throughput and a capture path's
dirty-rectangle behaviour, and does not exercise tracking. Independent
per-object motion needs sprite compositing in the source and is not built.

**Four classes, drawn, not photographed.** The scene is signage and a keyboard
on a flat background. It is a reproducible detection workload; it is not a
natural-image benchmark, and no mAP claim can be made from it. A
general labelled-dataset evaluation remains outside this project's scope, as
§ 7.0c's assumptions already state.

The scene pack lives in `build/scenes/`, git-ignored like the rest of the
history. Rebuild it with:

```
python benchmarks/scenes.py --out build/scenes/default \
  --model build/section7/model/yolo11n.onnx
```

§ 7.0c's other blockers are unchanged: the three comparison configurations are
still unbuilt, no practical thresholds have been chosen, and nothing has run on
a second machine.

#### 7.0e — The source was animating a corner of the screen ⚠ 2026-09-19

A review of the harness for accuracy found that **three of the five benchmarks
were driving a mostly-still desktop**, and nothing in any recording said so.

`benchmarks/motion_source.py` animated a hardcoded **900×700 window at
+200+120**. `memory_profile.py` and `compare_libraries.py` capture the **whole
screen**. On this 2560×1600 panel that is **15.4% of the captured area
moving**; on a 1080p display it is 30.4%.

Desktop Duplication reports only what changed. So every library was asked for a
fraction of the work a real workload would demand, every path looked faster
than it is, and **the size of the discount depended on the monitor** — which
means two machines running the identical command were never running the same
benchmark. That is precisely the comparison § 3 keeps Machine A and Machine B
apart to make.

##### Which recorded numbers this touches

| recording | source | animated fraction |
| --- | --- | --- |
| Pixel age, § 7.0 (`section7-ingestion-machineB.json`) | D3D `latency_source` at the display mode | **100% — unaffected** |
| Present-to-inference, § 7.0 (`section7-inference-machineB.json`) | same | **100% — unaffected** |
| Memory, § 7.0 (`memory-baseline-machineB.json`) | `memory_profile.py`, default **900×700** | **15.4%** |
| `compare_libraries.py` tables, § 3 and the README | `motion_source.py` 900×700, fullscreen scenario | **15.4%** |
| `ai_ingestion.py` call duration | `motion_source.py` 900×700 | **15.4%** |

**The pixel-age and inference tables are safe**, and that is not luck:
`section7.py` passes the physical display mode to the D3D source and refuses to
run if the requested resolution is not the current mode, so its source has
always been a full-screen borderless window at (0,0).

**The memory table is the one to re-record.** Its frame rates — `grab()` at
109.7 fps against `grab_frame()` at 165.1 — were taken with 85% of the screen
still. The memory figures are less sensitive to dirty fraction than the rates
are, but `grab()`'s output-path allocation scales with frames delivered, so
they are not immune either. Nothing in § 7.0's memory conclusions should be
quoted until it has been re-run.

##### A second bug in the same place

`compare_libraries.py`'s region scenario used the literal
`REGION = (760, 340, 1160, 740)`, commented *"400x400, centred on a 1080p
display"* — which it was, and on nothing else. On 2560×1600 it sat up and to
the left of centre, and it overlapped the old 900×700 motion window only
partly, so roughly 15% of the region was never animated at all. It is computed
from the display now, and the parent and each worker subprocess derive the same
rectangle without having to pass it.

##### What changed

- **`motion_source.py` covers the whole screen by default.** `--window WxH+X+Y`
  keeps the old behaviour for the tests that need a source smaller than the
  screen, and the bar count scales with the canvas so a 2560px display does not
  get 28 slivers.
- **The source reports the rectangle it is animating** in its `ready` event, so
  a consumer records what was moving instead of assuming all of it was.
- **`memory_profile.py` follows the display** — `--width 0 --height 0` now mean
  "the current mode", and that is the default.
- **`compare_libraries.py` centres its region on the actual display.**
- **`result_validation.coverage_reasons()` is the guard.** Every run records
  the animated and captured rectangles and the fraction between them, and a run
  below 90% is marked `contaminated` with the number in the reason. A
  deliberately small animated region stays a legitimate thing to measure;
  measuring one by accident does not.

##### One thing I suspected and was wrong about

I expected the Tk source to be DPI-virtualised as well — a 900×700 window on a
150%-scaled display becoming 1350×1050 physical. It is not: `python.exe`
declares per-monitor DPI awareness in its manifest, so Tk's screen metrics are
already physical. Checked directly: `winfo_screenwidth` reports 2560, not 1707.
Recorded because it is the first thing anyone will suspect next time.

**Covered by 26 new tests** (`test_workload_coverage.py`), including one that
reproduces the old 15.4% and 30.4% figures so the regression cannot come back
quietly. Suite total: **1884 passing**.

#### 7.0f — First live run of the new harness ✅ 2026-09-19

Everything in § 7.0a–e had been tested headless against fakes. This is the first
time any of it executed against real hardware, and it found three bugs that no
amount of synthetic testing was going to find.

Five runs on Machine B, 2560×1600 at 165 Hz, P-core pinned. **7 cases
committed, 0 corrupt, 0 interrupted.** WHEA unchanged across all of it (18
records, latest 81359, before and after).

##### What was confirmed working

- **The CPU policy reaches the workers.** Parent applies `0xffff`; every worker
  reports `matches_expected: true` from its own `GetProcessAffinityMask`
  *before* NumPy, CuPy or ORT initialise.
- **All three telemetry providers are live** — psutil, `CallNtPowerInformation`,
  NVML. CPU frequency genuinely varies (1466–2200 MHz), so on this machine that
  provider is measuring rather than echoing nominal.
- **The store holds under real conditions**: 7 cases, every one hash-verified on
  read-back.
- **The scene reaches the screen.** The D3D source loads the 65 MB BGRA canvas,
  pans it, and the frame-ID marker still decodes — 434 unique frames in 3 s with
  the scene present. The pixel-age clock and the detector share a frame, as the
  synthetic check predicted.
- **`postprocess_location: "device"`.** The CuPy wrapping of ORT's device output
  — the code § 7.0c explicitly flagged as never executed, written defensively
  with a host fallback — works. It never took the fallback.

##### Three bugs, each invisible to a headless test

**1. The telemetry attributed the benchmark's own CPU to background load.** A
worker burning **113% of a core** while the tree meter read **0.05%**, so the
entire benchmark landed in `other_cpu_percent` — the exact false positive the
design was written to avoid. `psutil.Process.cpu_percent(interval=None)` is
stateful *per object*: it reports the busy fraction since that object's
previous call, and the first call on a fresh one always returns 0.0. The
provider rebuilt its child list every sample, so every worker's only reading
was its priming zero. Process objects are cached by pid now, newly seen ones
are primed and excluded from that sample, and the count of them is reported so
a partial sample is visible rather than merely quiet.

**2. The detector was fed one picture and given boxes for another.** The
detection contract defaulted to `letterbox`; `benchmark_contract.canonical_rgb`
**stretches** the whole frame into the 640 square. So the model saw every
object squashed 1.6× vertically while postprocessing restored boxes with a
letterbox inverse. The damage, measured:

| | letterbox (wrong) | stretch (correct) |
| --- | ---: | ---: |
| detections per frame | 16.7 (max **37**) | 5.9 (max 10) |
| unique fps | 44.4 | 55.9 |
| inference + postprocess p50 | 14.73 ms | 10.10 ms |
| present → usable detections p50 | 49.63 ms | 44.56 ms |

Nothing crashed. It produced confident, plausible, wrong numbers — which is the
failure mode § 7.0c's whole apparatus exists to prevent, arriving through the
one door nobody had checked. The contract now **declares** its geometry
(`benchmark_contract.TENSOR_GEOMETRY`), `section7.py` defaults to it and
**refuses** any other value, naming what would have to change to letterbox.

**3. The scene was verified against a geometry the harness does not use.**
`scenes.py` letterboxed while the harness stretches, so its promised 6–9
detections per frame described a picture nobody presents; the stretched harness
saw 2–10, with frames below the floor. Verification now takes its geometry from
`TENSOR_GEOMETRY`. Re-verified under the real transform the scene was **refused
as too sparse** — squashing every object 1.6× costs confidence on each one — so
the density went from 7×5 to 9×6.

##### The scene, re-verified under the transform that will time it

| workload | detections per frame |
| --- | --- |
| static | 11 – 11 (mean 11.0) |
| scroll | 8 – 11 (mean 9.3) |
| motion | 5 – 13 (mean 9.2) |

And the live run agrees with it: **9.52 and 9.28 detections per frame** against
a predicted 9.2. The scene verification now predicts what the harness sees,
which is the only thing that makes it worth running.

##### The numbers, and what they are not

Present → **usable detections** (boxes, classes and scores on the host), scene
panning, YOLO11n on the RTX 4060, 4 s per path:

| path | unique fps | age p50 | inference + postprocess p50 | detections/frame |
| --- | ---: | ---: | ---: | ---: |
| rapidshot-cpu | 50.18 | **46.95 ms** | 12.32 ms | 9.52 |
| dxcam | 45.22 | 49.31 ms | 11.89 ms | 9.28 |

**This is one pass each and therefore not a finding.** By the rule in
§ 7.0c — which this project now enforces in code — a verdict needs at least
five interleaved paired runs, a threshold fixed in advance, and an
unchanged-code calibration for the metric on this configuration. None of the
three exists yet. `compare_paired_runs` would refuse this and say so. The table
is here to show the pipeline produces numbers of the right shape, not to claim
2.35 ms.

**Covered by 3 new tests** for the process-attribution bug, including one that
reproduces a busy child reading as idle. Suite total: **1887 passing**.

### 7.1 — 2.5: reliability and adoption ✅ delivered 2026-09-11

All five pieces are built and covered by tests that need no desktop, GPU or
capture hardware — which matters, because these are precisely the paths CI
cannot exercise. **445 tests pass**, up from 377.

| Piece | Delivered |
| --- | --- |
| Capture Recovery Manager | `camera.generation`, `camera.recovery_count`, `camera.last_recovery_reason`, `frame.generation`; all 7 rebuild triggers record a cause |
| Finish `Frame` | `sequence`, `generation`, `changed_fraction`, `age_ms`, `cursor` (position / hotspot / shape / encoding) |
| `capabilities()` / `diagnose()` | One report replacing six scattered probes; never raises |
| DXcam compatibility | `rapidshot.dxcam_compat`, full API parity including the zero-copy `grab_view` pair |
| Structured profiler | `rapidshot.profiling.Profiler` with `report()` / `json()` / `summary()` |

**Recovery was already survivable; it was not observable.** The rebuild
machinery existed — bounded retries, backoff, `_attempt_reinitialization` — but
nothing told a caller it had happened. A consumer holding a `GpuPreprocessor12`
or a `CrossAdapterTransfer` built from an earlier frame had no signal that the
duplicator underneath it had been replaced and might now differ in size,
rotation or format. `generation` is that signal, stamped per frame, incremented
only on a *successful* rebuild so a failed attempt that will be retried does not
invalidate anyone's cache for nothing.

**`changed_fraction` unions its rects rather than summing them.** Drivers do
report overlapping regions; summing areas can exceed the frame, and a consumer
thresholding on "more than 90% changed" would then take the full-frame path for
a frame that barely moved. An empty rect list yields `1.0`, not `0.0` — no rects
is no information, and the safe reading is that everything changed.

**The DXcam layer's cost is one copy per frame, and that is the point.** DXcam
callers never release anything; RapidShot's pooled buffer must be released and
raises if read afterwards. So the shim copies out and releases immediately. Drop
the shim and add `release()` to get the performance back. `grab_view()` is the
exception: DXcam's "valid until the next grab" contract is exactly the pooled
buffer's lifetime, so that path is genuinely zero-copy.

**The profiler reports percentiles and a minimum, never a mean**, and labels any
stage under 30 samples `low_confidence`. Both rules come from § 2: background
load can only make a sample slower, so the minimum is the least contaminated
estimate and the tail is what a real-time consumer feels — a mean hides both.
It also records `accumulated_frames`, because a loop can look fast while
dropping most of what it was meant to capture, and wall clock alone cannot tell
the two apart.

#### A live data-corruption bug, found by the compatibility layer's tests

`PooledBuffer.__array__` accepted NumPy's `copy` argument and ignored it,
returning a view. NumPy 2 forwards `copy` and **trusts the answer**, so

```python
frame = camera.grab()
mine = np.array(frame, copy=True)   # returned a VIEW of the pooled buffer
```

handed back a view of a buffer the pool was about to reuse. The caller holds
what looks like its own array, the next capture overwrites it, and nothing
raises — exactly the failure pooling is documented to prevent. Verified aliasing
on numpy 2.5.1; fixed, with regression tests covering the copy, the zero-copy
default and dtype conversion.

This predates the section 7 work entirely and affects anyone who has written
`np.array(frame, copy=True)` since NumPy 2. It was found only because the DXcam
layer's whole safety story is copying a frame out before releasing it, which
made the aliasing observable in a test rather than in someone's data.

### 7.1 — original plan

Before more GPU surface area, make the library boringly dependable and easy to adopt. This is what makes RapidShot the best *capture* library rather than only the best ML bridge.

**Capture Recovery Manager — rank 1 overall.** Less exciting than Torch interop and more important. Survive, without the application writing recovery code: `DXGI_ERROR_ACCESS_LOST`, monitor unplug/replug, resolution change, refresh-rate change, rotation, SDR<->HDR switch, sleep/wake, lock/unlock and secure-desktop switches, GPU reset, driver restart, exclusive-fullscreen transitions, window resize/destruction, adapter topology change. Expose `frame.generation`, `camera.recovery_count`, `camera.last_recovery_reason` so a consumer can *tell* that recovery happened. Parts exist already (§ 5 lists the access-loss rebuild and exclusive-fullscreen handling); the work is making it systematic and observable rather than incidental.

**DXcam compatibility layer.** Technically dull, plausibly the highest adoption return in this document: a shim letting `dxcam.create(...)` code run unchanged, so existing projects switch by changing an import.

**Finish `Frame`, do not redesign it.** Most of it exists — `timestamp_qpc`, `accumulated_frames`, `dirty_rects`, `protected_content`, `cursor_visible`, `region`, `rotation_angle`. Add `frame.sequence`, `frame.changed_fraction`, `frame.age_ms`, and surface the cursor position/shape/hotspot that `core/duplicator.py` already collects and discards (§ 6.3).

**`rapidshot.capabilities()` and `rapidshot diagnose`.** Consolidate the probes that already exist (`topology_info`, `probe_cross_adapter`, `probe_onnxruntime`, `native.build_info`) into one stable report: backend, adapter, D3D12 support, cross-adapter, CUDA interop, WGC, HDR formats, encoder codecs. A support-load feature — most "it does not work" reports are answerable from that output alone.

**A structured profiler.** `profile.report()` / `.json()` / `.summary()`, not printed text. This project argues from measurements constantly; make the instrument a first-class object.

### 7.2 — 2.6: GPU transform and framework interop

> **Status 2026-09-14 — built, not released.** Verified on live capture on the Intel-only machine; `CHANGELOG.md` `[Unreleased]` has the detail. The plan text after this box is kept as the rationale.
>
> | Item | State |
> | --- | --- |
> | `GpuConverter`, bilinear + FP16 + resized BGRA8 | ✅ nearest bit-identical to `GpuPreprocessor12` |
> | Input `BGRA8 / RGBA8 / R10G10B10A2 / RGBA16F` | ⚠️ all accepted; **only BGRA8 exercised** (SDR desktop) |
> | Output `NV12 / P010` | ✅ against a CPU reference and the published inverse matrices. Centre-sited chroma — H.264/HEVC assume left-sited, a one-line shader change for § 7.3 if the encoder needs it. HDR `RGBA16F` input refused: linear light needs a tone-map or PQ decision nobody has made |
> | NHWC float | ✅ bit-identical to NCHW transposed |
> | Crop | ✅ frame coordinates, honours `Frame.region`, filter clamped inside the crop; refused on rotated displays |
> | Multi-ROI | ✅ one dispatch, `(N, …)`, capacity via `batch=`; 4 × 224²: 0.20 vs 0.51 ms, 16 × 224²: 0.43 vs 2.11 ms, 8 × 640²: 1.15–1.20 vs 2.02–2.43 ms against N separate calls |
> | `GpuTensor` → Torch / CuPy / DLPack | ⚠️ **written, never run** — release gate |
> | Convert-first transfer (`TensorTransfer`) | ✅ byte-equal; ⚠️ **WARP destination only** — release gate |
> | `TensorStream` | ✅ |
> | Fused pipeline graph | not built — build only if it fuses, as below |
>
> **What building it taught, worth more than the features.** `TensorStream`'s end-to-end test exposed that every GPU path reading the capture surface could return the previous frame (§ 4). The converter tests had not caught it because they held one frame per module (§ 2). The same work found `GpuConverter` resizing the whole monitor on a region camera, then the same bug shipped in `GpuPreprocessor12` and the D3D11 `GpuPreprocessor`; all three are fixed (§ 10).
>
> **One design choice to know before extending it:** a single crop and a multi-ROI batch are the same code path — rectangles in an upload-heap `StructuredBuffer`, `Dispatch(x, y, N)` with `tid.z` selecting the slot — so there is no separate crop kernel to keep in step.

**`GpuConverter` — rank 2, and build it before the encoder.** A reusable colourspace/format layer: `BGRA8 / RGBA8 / R10G10B10A2 / RGBA16F → NV12 / P010 / RGB`. Capture already handles HDR formats through `DuplicateOutput1`. Built once as infrastructure it serves ML, encoding, streaming and recording; buried inside a Stage 7 encoder it serves one of them.

**The existing preprocessor downsamples with nearest-neighbour, and that is not recorded anywhere else.** `native/src/preprocess12.rs` computes `sx = tid.x * SrcWidth / OutWidth` and reads with `Source.Load()`; the root signature declares `NumStaticSamplers: 0`, so no filtering hardware is involved. Scaling 2560x1600 down to 640² drops roughly fifteen of every sixteen pixels rather than averaging them, which aliases exactly the content desktop capture is most often pointed at — small text, thin borders, cursor edges. Nobody has measured what that costs a model's accuracy, and no test would catch it: the output is the right shape, the right range and the right channel order.

Two consequences. A `GpuConverter` should sample through a static sampler (`Texture2D.SampleLevel`) rather than inherit `Load()`. And **any A/B against the current path must state which sampling it used** — substituting bilinear silently changes the work being timed, so a "faster" result could just be a different computation. It also changes the numerical contract `tests/test_gpu_preprocess.py` pins, so it is a new path rather than an edit to the existing one.

Also worth taking from the same review: `GpuPreprocessor12` emits FP32 only. Most production inference runs FP16, so the consumer pays a cast the producer could have avoided — and an FP16 tensor is half the bytes across the bus, which § 6.1's re-opened ordering question now makes load-bearing.

**`GpuTensor` → Torch / CuPy / DLPack — rank 3.** `examples/gpu_tensor_to_cupy.py` proves the path but costs the caller ~60 lines of `ctypes`. Collapse that to `tensor.to_torch()` / `.to_cupy()` / `.to_dlpack()` while keeping GPU residency.

**Do not put DLPack on `Frame`.** A `Frame` is a captured surface, not a tensor. Keep the boundary at the transform output:

```python
with camera.grab_frame() as frame:
    tensor = pipeline.process(frame)
torch_tensor = tensor.to_torch()
```

with a convenience layer on top:

```python
stream = rapidshot.TensorStream(camera, size=(640, 640),
                                dtype="float16", layout="NCHW")
for tensor in stream:
    model(tensor.to_torch())
```

**Multi-ROI batch preprocessing.** `pre.process(frame, regions=[...]) → (N, 3, H, W)` in one dispatch. Directly useful for OCR, UI agents and multi-window inference.

**A narrow pipeline graph, if it fuses.** `.crop().resize().convert().normalize().layout()` is worth building **only** if RapidShot compiles the chain into fused dispatches. It must end at the consumer boundary — never `capture → run YOLO → POST → database`. See § 8.

### 7.3 — 2.7: WGC, window capture, hardware encode

**WGC backend and true HWND capture — rank 5.** Partly catch-up: DXcam already exposes a `backend` parameter defaulting to `dxgi`, so WGC is table stakes rather than differentiation. The genuine wins are per-window capture and capturing without the process running on the display's adapter (§ 6.1). WGC is also naturally frame-pool/event oriented, which is worth stating against § 4's rejection of event-driven capture — **that rejection is correct for DDA's `AcquireNextFrame` and does not generalise to WGC.**

**Get the API distinction right.** *Source* and *backend* are different axes:

```python
rapidshot.create(source=rapidshot.Window(hwnd))
rapidshot.create(source=rapidshot.Monitor(1), backend="auto")
```

not `backend="desktop"`, which conflates the two.

**Hardware encode — rank 6.** NVENC / AMF / QSV / Media Foundation, with D3D12 video encode on Windows 11. **It cannot branch off `GpuPreprocessor12`:** that produces an NCHW float32 ML tensor (`shape` is `(1, 3, H, W)`), which is not an encoder input. The encoder path runs `Frame.d3d11_texture` → `GpuConverter` → NV12/P010 → encoder. NVFBC stays out: deprecated for general Windows use since Windows 10, frozen at Capture SDK 7.1, Linux-only in practice.

**Do not assume D3D12 Video Encode is uniformly available.** `ID3D12VideoEncodeCommandList` is standardised on recent Windows 10/11 builds, but driver support varies by vendor and driver age in a way the native codec SDKs do not — so a single D3D12 path will work on the development machine and fail on a user's. Gate it: query `ID3D12VideoDevice::CheckFeatureSupport` for `D3D12_FEATURE_VIDEO_ENCODER_CODEC` and the specific profile/level before selecting the path, and keep a fallback. `NvEncodeAPI` accepts D3D12 input surfaces directly, so a slim native NVENC branch needs no extra copy. **This is untested here** — noted from review rather than measured, and it belongs to a feature nobody has started; treat it as a design constraint to verify, not a finding. The § 2 vendor-coverage rule applies: this needs an AMD or Intel part before it can be called settled.

Muxing and audio stay downstream (FFmpeg, PyAV). See § 8.

### 7.4 — 2.8: scheduling and IPC

**Multi-ROI and dirty-aware scheduling — rank 7.** State this correctly: **ROI scheduling does not reduce DXGI capture work.** Desktop Duplication hands over the whole monitor regardless. What it avoids is *downstream* processing:

```text
monitor capture @ 120 Hz  ->  one GPU texture
    |- main view    preprocess @ 60 Hz
    |- health bar   preprocess @ 10 Hz
    |- chat         only when its rects are dirty
    `- cursor ROI   preprocess @ 120 Hz
```

Dirty rects decide whether a scheduled ROI needs work at all — the two features are worth more combined than separately.

**Finish backpressure (§ 6.4) rather than redesigning it.** Ship `latest` / `all` / `block` / `drop_oldest`. **Postpone `adaptive`**: automatic resolution and rate changes make systems unpredictable, and there is no usage data to tune against yet.

**`SharedFrameBus` — rank 8.** A real N-slot shared GPU ring: producer sequence number → producer fence → N consumers → per-consumer completion. The shared D3D12 resource and fence primitives already exist (§ 6.1, 2.4.0), so this is evolution rather than new ground.

**IPC is harder than it looks.** A Windows `HANDLE` cannot be put on a socket and used elsewhere — it needs `DuplicateHandle` or a named shared handle, plus cross-process fence ordering and resource lifetime management. Budget accordingly.

**`FrameBundle`, not a fused texture.** For multi-monitor, keep each frame's zero-copy texture and expose `FrameBundle(frames, target_qpc, max_skew_us, missing_sources)`, composing only when `.compose()` is called. And do not call it synchronisation: monitors present independently, so this is *alignment within a tolerance* using each frame's `LastPresentTime`, with `skew_us` exposed.

### 7.5 — 3.0: replay, and the native core

**`.rsrec` record/replay — rank 9.** Record frames plus metadata (QPC, dirty rects, accumulated frames, cursor, source ID, HDR format) and replay deterministically. This makes RapidShot's own test suite dramatically stronger — most of § 2's testing gotchas exist because live capture is unreproducible — and lets consumers reproduce bugs without the original desktop.

**`rapidshot-core` + a stable C ABI — rank 10, and a reversal worth naming.** § 8 says *"Stage 2 — native capture core: do not build"*, on the measurement that the Python/COM binding costs 0.003 ms/frame. **That measurement stands and is not what changed.** The new argument is not performance, it is embedding: a native application cannot host Python in its capture path. OBS is C/C++ and calls D3D11/WinRT directly; it will not take a `Python → PyO3 → Rust` dependency at any speed.

So the C ABI is justified by *who can consume the library*, not by how fast it runs:

```text
rapidshot-core (Rust)
      `- stable C ABI
            |- Python bindings
            |- C / C++ consumers
            `- Rust consumers
```

Do not start it before 7.0's benchmarks and 7.1's reliability work. An ABI frozen around an unproven design is worse than no ABI.

### Priority order

1. Capture reliability and automatic recovery (7.1)
2. `GpuConverter` (7.2)
3. `GpuTensor` → Torch / CuPy / DLPack (7.2)
4. Diagnostics and the formal benchmarks (7.0, 7.1)
5. WGC / true HWND capture (7.3)
6. Hardware encoder (7.3)
7. Multi-ROI + dirty-aware scheduling (7.4)
8. `SharedFrameBus` (7.4)
9. `.rsrec` replay (7.5)
10. Native C ABI (7.5)
11. Multi-monitor temporal alignment (7.4)

**DXcam compatibility sits outside this ranking**: unexciting engineering, possibly the highest return in the document, and cheap.

### On OBS, honestly

A recurring question, worth settling on the distinction it actually turns on. OBS has three Windows capture paths: DXGI Desktop Duplication, WGC, and **Game Capture**, which injects a graphics hook and captures inside the application's render pipeline.

**RapidShot can plausibly beat OBS's DXGI/WGC display capture. It cannot beat Game Capture**, which sits upstream of the compositor entirely:

```text
GAME
 |- render ---> OBS Game Capture   (before composition)
 v
DWM compositor
 `- RapidShot / OBS Display Capture
```

Competing there means API injection, process hooks and anti-cheat compatibility — a different project. Do not start it.

The realistic path to adoption is: win the § 7.0 benchmarks → get used by AI/CV projects → stabilise the C ABI → a plugin becomes possible. The likely outcome is that OBS adopts *techniques* rather than the library, and that is still a win.

### What still belongs from the old § 7

- **6c — GPU-side change detection.** Still lower value than it looks: DDA already reports only changed content with compositor-computed dirty rects. The residual value is deduplicating presents reported as changed but visually identical, plus sub-rect granularity. Do § 6.3's dirty rects first and measure.
- **5 — Backend auto-selection.** Needs >=2 backends; follows 7.3.
- **8 — `rapidshot.stream` network streaming.** WebRTC transport, DataChannel input, browser viewer. This is what moves the category from "screenshot library" to "capture-and-stream infrastructure".
- **9 — Remote-support primitives.** `WDA_EXCLUDEFROMCAPTURE`, adaptive bitrate hook.
- **11 — Ecosystem.** OpenCV `VideoCapture` wrapper, PyTorch `IterableDataset`, OBS source plugin (after the C ABI), an open Python screen-capture specification.
- **Stage 0 remainder.** Done for 2.0.0: `py.typed` with the public API annotated, `SECURITY.md` via GitHub private reporting, `.github/CODEOWNERS`, `release.yml` with Trusted Publishing, Sigstore attestations and a CycloneDX SBOM, `RELEASING.md`. CodeQL runs through GitHub's **default setup** — do not add a `codeql.yml`; an advanced configuration cannot coexist with it and fails at SARIF upload. **Still outstanding:** `GOVERNANCE.md` (a decision, not a file), OpenSSF Scorecard, hosted docs.

---

## 8. Deferred and out of scope

**A declarative YAML/DSL pipeline: do not build.** Proposed as a config-driven `capture → transform → consumer` description. It buys no performance, adds a large API surface, and pulls the library toward being a workflow framework — the same boundary § 11 defends when it declines to own consumers' bindings. The fluent GPU graph in § 7.2 is worth having *only* because it can be fused into dispatches; a YAML layer on top of it cannot.

**Audio, scenes, transitions, webcam composition, an audio mixer: not ours.** That is OBS's product, not a capture runtime's. RapidShot already demonstrates ordinary recording through OpenCV. The meaningful direction is § 7.3's `Frame → GpuConverter → NV12 → hardware encoder`, with muxing left to FFmpeg or PyAV. A thin optional container module is acceptable; an A/V production stack is not.

**Game Capture-style API injection: not ours.** OBS hooks D3D/Vulkan/OpenGL inside the application and captures before composition, which is why it beats every compositor-based approach on games. Matching it means process injection, hook maintenance across graphics APIs, and anti-cheat compatibility — a separate project with a separate risk profile. § 7 states the boundary; this is the entry that says do not cross it.

**`policy="adaptive"` backpressure: postponed, not rejected.** Automatic resolution and frame-rate adjustment sounds attractive and produces systems whose behaviour cannot be reasoned about. Ship the deterministic policies in § 7.4, collect usage data, then revisit.

**Tracking windows by dirty rects: technically wrong, not merely inadvisable.** Dirty rectangles report *which pixels changed*, with no notion of window identity — a rect can span two windows, or a window can move without its contents changing. Semantic per-window capture is HWND tracking or, better, § 7.3's WGC backend, which addresses windows directly.

**Stage 2 — native capture core: do not build.** Measurement says 0.003 ms/frame on a hardware-bounded path, and `windows-capture` already ships it. Revisit only if § 9 (cross-platform) is committed to, or if a profile on a real workload contradicts § 3.

**ONNX Runtime session binding: deliberately not ours.** Rapidshot produces the `ID3D12Resource`; binding it is ~15 lines for a caller who already has ORT set up, documented in the README. Every route to `OrtDmlApi` costs something permanent — vendoring ~5000 lines of header plus hand-counted struct offsets, bindgen adding libclang as a third build dependency, or the `ort` crate requiring ONNX Runtime built from source. All couple Rapidshot's core to ORT's ABI and release cadence for one optional feature.

Progress that exists and is worth keeping:

- `native.probe_onnxruntime()` confirms ORT is reachable from Rust (1.24.4, C API versions 1–24). Only `OrtApiBase` is traversed — two members, layout fixed by contract, no risk.
- `OrtDmlApi` indices from the header: `CreateGPUAllocationFromD3DResource` = **2**, `FreeGPUAllocation` = **3**, `GetD3D12ResourceFromAllocation` = **4**.
- **Gotcha:** resolving `onnxruntime.dll` by name uses the DLL search path and can find an unrelated version (on Machine A, 1.17.1 rather than the package's 1.24.4). Struct layout is version-dependent, so this matters. Use `native.onnxruntime_dll_path()`.
- **Never hardcode an unverified `OrtApi` offset.** If revisited: pin to a specific `ORT_API_VERSION` (the C API is append-only, so version N guarantees N's layout up to N) *and* validate at runtime that the pointer falls inside `onnxruntime.dll`'s address range.

If demand appears: a separate `rapidshot-directml` package, or a Python-side `ctypes` binding — same fragility but patchable without a rebuild and auditable without a toolchain.

---

## 9. Stage 4b — Cross-platform (a project-scale bet)

macOS ScreenCaptureKit and Linux PipeWire + XDG Portal. This is the one stage that genuinely justifies a native core, and that decision would drive Stage 2 rather than the reverse.

Realistic cost is roughly six months with a native-graphics-fluent co-maintainer — **that estimate belongs here, not to a Windows-only core**, which is a few hundred lines. Most of the effort is permission-flow edge cases: XDG portal tokens invalidating on fullscreen toggle, macOS Screen Recording permission needing restart-after-grant handling.

---

## 10. Known debt

- ~~**GPU reads of the capture surface were unordered.**~~ **Fixed 2026-09-14, unreleased.** `GpuPreprocessor12` (7–15 / 150 stale) and `CrossAdapterTransfer` (65 / 150 to WARP) could read the surface before the capture device's copy into it had run, and return the previous frame; shipped since 2.3.0. § 4 has the mechanism and the rejected alternatives. How it went unnoticed is the part to keep:

  - **Found by accident, from the right kind of test.** `TensorStream`'s test wrapped the camera to convert each frame independently while it was live, then compared the stream's output. The *reference* — the first reader — was the one that was wrong, which a `stream != reference` failure did not say; a third opinion from a fresh converter per frame did.
  - **Two hypotheses were measured and rejected before the right one**: a surface that changes while held (it does not — identical reads across 85 ms), and a texture cache reusing a recycled address (one address across 60 frames, long-lived converters agreeing with fresh ones).
  - **The existing verification was structurally blind.** `transfer_with_reference` snapshots the surface once and copies the snapshot to both sides, so both are stale together. Its comment blamed ~2,100 differing bytes on "DXGI keeps writing to" the surface; that was this race. The comment is corrected; the snapshot stays, because it still proves integrity.

  **Still open:** the fix is unverified with a hardware destination (§ 1 release gate). `probe_cross_adapter()` and `probe_cross_adapter_buffer()` are timing probes and are not ordered. The D3D11 `GpuPreprocessor` and the CPU `grab()` path read on the capture device's own queue and should be ordered by construction — reasoned, not measured. And the fix rests on the invariant in § 4: signalled only while a `Frame` is live. Nothing tests that invariant.

- ~~**`GpuPreprocessor12` ignores `Frame.region`.**~~ **Fixed 2026-09-14, unreleased.** A camera created with `region=` hands back frames whose texture is the whole output, and the preprocessor sized its sampling from the texture, so it resized **the entire monitor** into a correctly shaped tensor. Verified before the fix: on a `(101, 51, 421, 291)` region camera the output was bit-identical to the whole 1920×1080 surface resized. The shader now takes a crop rectangle, which with its whole-surface default reduces to the old expression — `tests/test_gpu_preprocess.py` passes unmodified — and the `native.GpuPreprocessor12` wrapper passes the frame's region, using the same translation helper as `GpuConverter` (`native._texture_crop`), so the two cannot drift. Tests in `test_gpu_converter_region.py` check the region against captured bytes, against decimation at the region's own stride, and against `GpuConverter`; each fails against a planted bug (wrapper not passing the region, shader dropping the offset, shader sizing from the texture). The raw extension still converts the whole surface unless given `crop=`, deliberately: it works in texels. Rotated displays are still not translated.

- ~~**The D3D11 `native.GpuPreprocessor` ignores `Frame.region`.**~~ **Fixed 2026-09-14, unreleased**, the same way. Verified before the fix: on the same region camera its output was bit-identical to the whole monitor resized. `preprocess.rs`'s shader takes the crop (its `Params` constant buffer grew 32 → 48 bytes, asserted at compile time to stay on D3D11's 16-byte boundary) and the wrapper passes `_texture_crop`. Because the D3D11 path accepts synthetic textures, its crop is also pinned in `test_gpu_preprocess.py` against a known pattern — identity, downscale, upscale, origin and far-corner crops, a one-texel offset shift, and refusals — which needs no screen and runs wherever the extension does. Live, `test_gpu_converter_region.py` checks it against captured bytes and against `GpuPreprocessor12`. Planted bugs (wrapper not passing the region, shader dropping the offset, stride taken from the texture) fail 2, 7 and 6 tests. It reads on the capture device's own queue, so it needed no capture ordering.

- **`test_nearest_output_matches_a_cpu_reference` never runs.** It skips unless a frame exposes `frame_buffer`, and no `Frame` does. Its intent — converter output against numbers computed on the CPU — is now covered by the identity-size slice checks in `test_gpu_converter_crop.py` and `test_gpu_converter_batch.py`. Delete it or rewrite it on those; it should not stay as a skip that reads like coverage.

- **`GpuConverter` does not handle rotated displays.** The captured surface is unrotated and frame coordinates are not translated onto it, so `crop` and `regions` are refused when `Frame.rotation_angle` is non-zero, and a rotated frame without a crop converts the unrotated surface — as `GpuPreprocessor12` always has. Not testable on either development machine.

- **`GpuConverter` input formats other than `B8G8R8A8_UNORM` have never been exercised.** `R8G8B8A8`, `R10G10B10A2` and `R16G16B16A16_FLOAT` are accepted by reasoning about how `Texture2D<float4>` reads each format, not by measurement; a D3D12 converter cannot be built over a synthetic texture (§ 2), so this needs a real HDR or 10-bit desktop.

- **Pooled output is the default since 2.0**, and it is a breaking change: `grab()` returns a `PooledBuffer` the caller must `release()`. Allocating the output array cost ~1.6 ms per 1080p frame in page faults — more than the conversion it feeds — so reusing buffers is **1.3–2.1× on `grab()`** across RGB, RGBA and GRAY, pixels identical. The wrapper indexes and converts like the array it wraps (`frame[y, x]`, `np.asarray(frame)` zero-copy), so the migration is usually one added `release()`; `pool_output=False` restores 1.x behaviour. Use after release raises rather than returning another frame's pixels. Since 2.2.0 the pool depth is public as `pool_size_frames` and defaults to **4 rather than 10** — 60 MB less per camera for a frame-rate change inside noise. Running the pool dry falls back to allocating, verified by holding six frames against a two-buffer pool.
- ~~**Exclusive-fullscreen and HDCP paths are fault-injection tested only.**~~ **Both were more testable than this entry claimed**, and it was wrong for a lazy reason: the real trigger was assumed to be the only trigger.

  **Protected content now has a real local trigger.** The assumption was that `ProtectedContentMaskedOut` requires licensed DRM playback. It does not — `SetWindowDisplayAffinity(hwnd, WDA_MONITOR)` makes the compositor blank a window out of captured frames, and **DXGI reports it through the same flag the HDCP path reads**. Measured 2026-08-06 on Machine B:

  | affinity | `Frame.protected_content` | mean of the window's pixels |
  | --- | --- | --- |
  | `WDA_NONE` | False | 113.7 |
  | `WDA_MONITOR` | **True** | **0.0** |
  | `WDA_EXCLUDEFROMCAPTURE` | **True** | **0.0** |

  `tests/test_protected_content.py` builds a window, flips its affinity, and asserts the flag follows the OS state **in both directions** — a flag that was always set would pass a one-sided test. It also pins that masked-out capture keeps returning usable frames, because the sibling branch genuinely *is* an error and conflating them would either raise on a normal blanked region or swallow a real refusal.

  **The refusal branch had a local trigger too, and reaching it found a real bug.** `DuplicateOutput` refuses with a bare `E_ACCESSDENIED` when the calling thread is not on the input desktop — reproducible with `CreateDesktop` + `SetThreadDesktop`, which disturbs nothing because creating a desktop object does not switch to it. RapidShot classified *every* `E_ACCESSDENIED` as protected content, so that produced:

  > Desktop duplication was denied because protected (HDCP/DRM) content is on screen. Close or move the protected player window and retry.

  with no protected content anywhere. The same message was being shown for a **locked workstation, an open UAC prompt, and a Session 0 service** — the last of which is exactly what § 12's `inventory-agent` runs as, so the one consumer this project already knows about was the one most likely to hit it and be sent looking for a player window that does not exist.

  Only plain `E_ACCESSDENIED` is ambiguous; `DXGI_ERROR_ACCESS_DENIED` and `DXGI_ERROR_CANNOT_PROTECT_CONTENT` are specific to protected content and are left alone. `rapidshot/util/desktop.py` now asks the window station which desktop is receiving input and reports the real cause:

  > Desktop duplication was denied: this thread is attached to desktop 'X' but 'Default' is receiving input. Desktop Duplication only works on the input desktop — a Session 0 service or a thread moved with SetThreadDesktop cannot capture the user's screen directly.

  It also names the secure desktop specifically, because `OpenInputDesktop` failing outright *is* the signal for a lock screen or UAC prompt. `tests/test_desktop_refusal.py` covers both directions, including the regression that matters: on a normal desktop, real protected content must still be reported as protected content and not quietly reattributed.

  **The exclusive-fullscreen path is now half real.** `tests/test_output_change_recovery.py` invokes `_on_output_change()` against live hardware, so the teardown and rebuild — release the duplicator, rebuild the stage surface, re-read the output, capture again — all execute for real. It pins the actual historical defect by **identity, not by symptom**: the duplicator and the stage-surface texture must both be *different objects* afterwards. Comparing pixels would not catch a surviving stage surface, since the old one returns plausible content right up until the resolution actually changes. Repeated rebuilds and the survival of a user-set region are covered too.

  **And detection is no longer injected either.** "The real trigger is a game" was wrong — the real trigger is `IDXGISwapChain::SetFullscreenState(TRUE)`, which is a page of ctypes and restores itself on exit, touching no persistent setting. Measured 2026-08-06 on Machine B, entering *and* leaving exclusive fullscreen each produced:

  ```
  COMError in update_frame: 'The keyed mutex was abandoned.' (0x887A0026)
  Access lost, re-initialization needed. Flagging for re-initialization.
  Re-initialization attempt 1 of 5 scheduled.
  ```

  So the whole chain runs against a real transition: detect → classify as recoverable → rebuild → resume. Capture produced 141 frames *during* exclusive fullscreen and 243 after, and **no error reached the caller** — absorbing it is the entire point.

  Two things about the shape of that result are worth keeping. The trigger was `ABANDONED_MUTEX_EXCEPTION`, **not** `DXGI_ERROR_ACCESS_LOST` as this entry had assumed; both land in `DXGI_RECOVERABLE_ERRORS`, so the classification was already right, but a test that had asserted on the specific HRESULT would have failed for no good reason. And because the library recovers silently, a test that only checked "did capture keep working" would pass just as well on a build where fullscreen never disturbed duplication at all — `tests/test_exclusive_fullscreen.py` therefore asserts the recovery was *reached*, via the log, and skips loudly if it was not.
- ~~**Nothing has ever been run for longer than a few seconds.**~~ **Soaked 2026-08-06 on Machine B**: 12 minutes of continuous `grab_frame()` → `GpuPreprocessor12.process()`, with a pooled `grab()` every 20th frame, against the calibrated motion source.

  | | |
  | --- | --- |
  | Frames | **211,726** in 12 min (~296/s), plus 10,558 pooled `grab()` |
  | Errors | **none** |
  | VRAM | 1191.9 MB → 1191.9 MB, **+0.00 MB/min** |
  | RSS | 540.9 → 543.3 MB, **+0.21 MB/min**, flat from t=300 s |
  | Throughput | 288 → 296 fps, **no decay** |

  No leak, no drift, no degradation. Two caveats on reading this. The RSS figure needed care: a naive first/last comparison reported +9.63 MB/min because the first sample straddled CuPy's import, and the real curve settles by t=300 s and then oscillates within 0.2 MB — **a settling curve is not a leak, and a two-point trend cannot tell them apart**.

  And the useful negative result: **exactly one distinct capture texture was seen across 211,726 frames.** That confirms § 6.1's "the capture texture pointer is stable" empirically and at scale, and it means `process()`'s cache hits essentially always — while leaving the *miss* path unexercised by any amount of normal running. A soak is the wrong instrument for that, and the answer was to build the miss deliberately rather than wait for one.

  **`tests/test_preprocess_cache.py` now forces a real miss.** `create()` returns a cached instance per output, so asking for two cameras gives one camera and one texture; a full teardown (`release()` + `rapidshot.reset()`) followed by a rebuild produces a genuinely new `IDXGIOutputDuplication` and a new texture — which is also the realistic trigger, since a device reset or mode change arrives at the same place. Verified: the cache re-keys to the new texture, the tensor is valid afterwards, and a repeat call does *not* re-open.

  That last assertion matters as much as the miss itself. Reopening per frame cost ~98 µs of fixed work and made every pixel 2.5× more expensive (above), so a change that quietly disabled the cache would be a large regression with no visible symptom — the tests assert on the cached **key**, exposed as `cached_texture_address` for exactly this purpose, rather than on output that would look correct either way.
- ~~**Capture assumed the display-owning adapter is the one that can duplicate.**~~ **Fixed 2026-08-21**, and the assumption was four separate defects wearing one coat. Found by putting Machine B into Optimus (§ 2), where every adapter refuses `DuplicateOutput` — a state in which the library should have explained itself and instead produced a COM error code.

  | Defect | Was | Now |
  | --- | --- | --- |
  | Adapter choice | picked the adapter owning the output, gave up if it refused | tries every adapter, uses the first granted duplication |
  | Enumeration | adapters with 0 outputs discarded | retained as duplication candidates |
  | `DXGI_ERROR_UNSUPPORTED` | raw COM string naming neither cause nor fix | names the cause and what to check |
  | `topology_info()` | "Capture runs on {adapter}" | describes the arrangement, does not predict |

  **Which adapter DDA accepts is not deducible from which adapter enumerates the output.** It depends on where the desktop is actually composed. That is the whole bug: the old code encoded a guess as a fact, and on the first real hybrid machine the guess was wrong.

  Three details worth keeping. **The retry is narrow on purpose** — only `DXGI_ERROR_UNSUPPORTED` and `DXGI_ERROR_INVALID_CALL`. A desktop refusal is *also* a `RapidShotConfigError`, so retrying on exception type alone would walk every adapter and then replace an already-actionable message ("you are not on the input desktop", § 10) with a generic one; that needed `_map_com_error` to stop dropping the HRESULT on this branch. **The winning device is recorded**, because the stage surface must be built on the same device as the duplicated texture — so `_on_output_change` now builds the duplicator *before* rebuilding the stage surface, not after. And **`prefer_integrated` was unreachable in exactly the configuration it exists for**: on a hybrid laptop the iGPU usually owns no output, so it had been discarded before the parameter could select it.

  Also fixed on the way: `_initialize_resources` logged the real exception and returned False, and the caller then raised `"Initial resource initialization failed. Check logs for details."` — discarding a diagnosis that had already been made, in the one case where the caller most needs it. The specific cause is re-raised now.

  **The success branch is verified by unit test, not by hardware.** `tests/test_failure_paths.py` covers falling back to a second adapter, not retrying a non-adapter refusal, and the all-refused message, all with stub adapters. No machine available can exercise a *successful* fallback, because the only hybrid machine here has no working pairing at all. Under § 5's rule that is not verified for a release until it runs on hardware.

- **RapidShot forces process-wide DPI awareness, and ignores the failure.** `Output.__post_init__` calls `SetProcessDpiAwareness(2)`, which is why coordinates are normally correct — `output_info()` reports 2560x1600 on Machine B's 150%-scaled panel whether or not the caller thought about DPI. Two problems sit under that.

  **The HRESULT is unchecked, and the failure is silent.** A host that pins DPI awareness before RapidShot loads — an app with a DPI-unaware manifest, or one that called `SetProcessDpiAwarenessContext` first — makes this call fail with `E_ACCESSDENIED`, because process DPI awareness cannot be changed once set. Measured 2026-08-21:

  | Host | `SetProcessDpiAwareness(2)` | `output_info()` resolution |
  | --- | --- | --- |
  | leaves DPI alone | `S_OK` | 2560x1600 (correct) |
  | pins DPI-unaware first | **`0x80070005`, ignored** | **1707x1067** |

  So embedding RapidShot in a DPI-unaware host silently yields coordinates 1.5x smaller than the panel, and every region the caller computes follows them down. Nothing reports this; the resolution simply reads wrong. The fix is to check the HRESULT and say so, not to try harder — the setting genuinely cannot be changed at that point, so the honest move is to tell the caller their coordinates are in scaled space.

  **And it mutates process-global state from a constructor.** Building an `Output` changes DPI behaviour for the entire host application, including its own windows. That it happens per-`Output` rather than once at import makes it repeated rather than worse, but a library reshaping its host's window layout as a side effect of enumeration is a decision that should be explicit and documented, and currently is neither.

- ~~**One destination buffer, and nothing tells the producer the consumer is done.**~~ **Reproduced and fixed 2026-08-22.** Every `transfer()` writes the same `src_buffer`/`dst_buffer` pair, and `shared_fence` only reports that the *copy* finished. A consumer that waits on it GPU-side and keeps reading asynchronously can still be reading frame N when the copy for N+1 overwrites the allocation underneath it — two frames blended, nothing raised.

  **Reproduced deterministically, after four failed attempts that are the more useful part of this entry.** The trick was to stop chasing a timing race and build a gate: queue the consumer's read behind a semaphore the test holds shut, let frame B's copy *complete* while the read is provably still pending, then open the gate. The consumer — which had waited on the producer fence for frame A — read B's bytes. 3/3 runs.

  | | consumer waited for | consumer actually read |
  | --- | --- | --- |
  | unguarded | frame A | **frame B** |
  | `wait_for_consumer` | frame A | frame A |

  **Then measured at scale, and the scale result is the one to quote.** A real
  60-frame loop with a consumer slower than the producer:

  | consumer | frames | wrong, unguarded | wrong, guarded |
  | --- | --- | --- | --- |
  | slower than producer | 60 | **28 (47%)** | **0** |
  | faster than producer | 100 | 0 | 0 |

  **Nearly half the frames, and the guarded loop is clean.** The second row is
  why this stays hidden: with a consumer that keeps up, the same loop shows
  nothing wrong over 100 frames, so the hazard is invisible until a real
  workload arrives and then corrupts silently. The handshake costs ~3% in the
  slow case and nothing measurable in the fast one, and holds across the whole
  run — no fence drift, no deadlock, no leak.

  **Every earlier attempt failed the same way, and it was never about the race being rare.** CuPy's allocator synchronises the calling thread, so anything allocating inside the gated region either hides the race (the producer can never run ahead) or self-deadlocks — one probe hung for ten minutes, blocking the CPU on a stream only that CPU could open. Making the consumer *slower* was the wrong axis entirely: a 527 ms consumer still showed nothing, because the CPU was being paced by the allocator, not by the GPU. Nothing inside the gated region may allocate. That is the reusable lesson.

  **The fix needed a signal travelling the other way, and its feasibility was the open question.** Measured: CUDA on the RTX 4060 signalled a D3D12 fence created by the **Intel** iGPU's device and the producer observed it — completed value 0 → 5000. Shipped as `set_consumer_fence(handle)` + `wait_for_consumer(value)`; the wait is queued on the source queue, so it orders ahead of the next copy without blocking the caller:

  ```
  transfer.set_consumer_fence(consumer_handle)   # once
  v = transfer.transfer_async(frame)             # frame N
  # consumer waits for v GPU-side, reads, signals its fence = N
  transfer.wait_for_consumer(N)                  # before frame N+1
  ```

  **The alternative was multiple destination buffers, rejected on correctness rather than cost.** A ring of N does not close the race, it widens it: the producer wraps after N frames, so a consumer lagging more than N corrupts again. It also costs N × 16.4 MB and forces a per-frame offset into the consumer's contract. The handshake closes it outright for one queued wait — +0.27 ms against a light consumer, +2.4% against a heavy one.

- **No AMD hardware has ever run this project**, and nothing branches on vendor, so AMD is supported by design and unverified in fact. The coverage matrix is in § 2; the one genuinely NVIDIA-bound feature is CuPy, and AMD consumers reach the same GPU tensor through DirectML.
- **Headless is classified but never observed; hybrid is now both observed and working.** The `hybrid` branch ran for real on 2026-08-22 — Intel iGPU owning the display, RTX 4060 render-only, `topology_info()` classifying it correctly, capture running on the iGPU and a frame crossing to the dGPU byte-exact (§ 6.1). Getting there also produced the lesson worth keeping: **classifying a topology correctly is not the same as being able to capture in it.** For most of a day the same machine classified as `hybrid` while every adapter refused `DuplicateOutput`, which is what § 10's adapter-selection entry came out of. The *single* branch has been observed on an Intel iGPU and on an NVIDIA dGPU. **`headless` remains tested by describing that topology rather than by having one.**
- ~~**Cross-adapter sharing verified against WARP only.**~~ **Closed 2026-08-21.** Switching Machine B to Optimus put a second *hardware* adapter in the machine, and `probe_cross_adapter()` reports `destination_is_software: false` and **`representative: true`** for the first time in the project's history — an RTX 4060 source and an Intel UHD destination, both real drivers.

  | | Source → destination | min | median | Throughput |
  | --- | --- | --- | --- | --- |
  | Machine A | Intel iGPU → WARP | 0.87 ms | 0.94 ms | 9.5 GB/s |
  | Machine B, discrete-only | RTX 4060 → WARP | 0.68 ms | 0.78 ms | 11.6 GB/s |
  | **Machine B, Optimus** | **RTX 4060 → Intel UHD** | **0.72 ms** | **0.82 ms** | **11.0 GB/s** |

  **WARP turned out to be a good proxy for the destination side** — within ~5% of a real hardware destination. The caution was still right to hold: it was unfalsifiable until now, and the answer happening to be benign is not something the earlier runs could have established.

  It also adds a third vendor data point to § 6.1's buffer decision. `destination_row_major_texture` is **true** on Intel and on WARP, and **false** on NVIDIA as source. Two of three support it, which is exactly why branching on it would have broken.

  **The direction § 6.1 targets was then measured too, on 2026-08-22.** With the machine in correct Optimus, `probe_cross_adapter()` runs **Intel→NVIDIA** at 1.39 ms min / 5.68 GB/s, and `verify_cross_adapter.py` carried 5 captured frames across byte-exact (§ 6.1). Both directions are now measured on real hardware; the earlier NVIDIA→Intel figures stand as the reverse case rather than as a substitute for it.
- **GRAY was by far the slowest colour mode; both halves are now fixed.** It used to be bimodal — a ~9.2 ms fast mode the CPU sustains for a second or two and a ~15 ms slow one — and because a capture loop never sees the transient, the honest figure was **13.7–14.9 ms**, filling ~95% of a 60 Hz frame. Two changes landed 2026-08-05, measured by `benchmarks/gray_kernel.py`:
  - **NumPy path: ~16 ms → 8.5–11 ms (1.5–1.8×), byte-identical.** The old formulation allocated a full-frame uint16 temporary per channel; those page faults cost more than the arithmetic. Reusing persistent intermediates removed them. This is what `pip install rapidshot` gets — no toolchain, no new dependency.
  - **Native kernel: → 0.70 ms (24× on the same machine), byte-identical.** `native/src/luma.rs`, exercised through `NumpyProcessor.convert_into` when the optional extension is present. GRAY now costs 4% of a 60 Hz frame instead of ~95%.
  Byte-exactness is asserted over all 2²⁴ BGR triples on both sides of the FFI boundary — it is what lets the accelerated path be swapped in without changing any consumer's pixels, and it is the one thing OpenCV's kernel cannot offer (off by up to 1 LSB, mean 0.13, because it rounds differently).
  - **AVX2 kernel: → 0.26 ms, byte-identical, 96% of the memory ceiling.** GRAY is now the *fastest* of the five modes, having been the slowest by an order of magnitude, and it beats single-threaded OpenCV (0.34 ms) while OpenCV is off by up to 1 LSB. **37× against the NumPy path, 59× against the 2.0.0 formulation.** § 3 records the `maddubs` saturation hazard that dictates the design and why the correctness test is exhaustive over all 2²⁴ triples rather than sampled.
  **Conversion is finished as an optimisation target.** All five modes sit at 69–100% of the 33.2 GB/s the memory system delivers; RGBA at 69% is the weakest and worth ~0.1 ms. `benchmarks/baseline.json` and the badges were re-recorded after this landed. GRAY's *NumPy* path remains duty-cycle sensitive (6.9–10.5 ms across runs in one session), so the suite flags it informational rather than gating on it — quote a range for that one.
- ~~**CuPy/CUDA paths untested** — no NVIDIA GPU available.~~ **Partly closed 2026-08-06.** CuPy 14.1.1 runs against CUDA 13.2 on Machine B and the eight previously-skipped tests pass (§ 5). Two things fell out of actually looking at it, and they replace the old one-line hope with a narrower, harder question:

  - **`CupyProcessor` was returning wrong data through the public API, and had no tests at all.** This is the most serious defect the new hardware surfaced, and it was invisible on Machine A because the path could not run there.

    `process()` converted colour by calling OpenCV — which **is not a RapidShot dependency**. On any machine without `cv2`, every non-BGRA mode raised. And the exception handler logged the error, then *returned the unconverted buffer as though it had succeeded*. So `create(output_color="RGB", nvidia_gpu=True)` returned a **4-channel BGRA array** — wrong shape, wrong channel order, no exception, just a log line. A caller feeding that to a model got silent garbage. This is precisely what § 11's "a fast wrong answer is worthless" exists to prevent, reappearing in the one processor nobody could execute.

    It also did the conversion in the worst possible place. Where `cucv` was absent — effectively always — it called `cp.asnumpy` to pull the frame **off** the GPU, ran OpenCV on the CPU, and pushed the result **back**. Three PCIe crossings on a code path whose entire premise is GPU residency.

    **Fixed 2026-08-06.** Every mode is now expressed in CuPy and runs on the device; OpenCV and `cucv` are gone from the path entirely. `process()` raises instead of returning a mis-shaped buffer, and an unsupported mode is rejected at construction rather than on the first frame that happens to arrive — the same reasoning as the `shot()` fix below, since a bad configuration on a static desktop otherwise looks fine until something moves. `create(nvidia_gpu=True)` now returns a `cupy.ndarray` that never leaves the GPU.

    **`tests/test_cupy_processor.py` is new and asserts the thing that matters: byte-exactness against `NumpyProcessor.convert_into` for all five modes.** `nvidia_gpu` is a performance switch, so turning it on must not change a single pixel — and OpenCV could never have satisfied that, its luma being off by up to 1 LSB. One test blocks `import cv2` outright and requires every mode to convert anyway, which is the original defect nailed down directly.
  - ~~**The Stage 6 tensor cannot reach CuPy at all**~~ — **closed 2026-08-06.** `create_buffer` allocated the output with `D3D12_HEAP_FLAG_NONE`, so there was no `CreateSharedHandle` and nothing for `cudaImportExternalMemory` to import. The output heap is now `D3D12_HEAP_FLAG_SHARED`, and `GpuPreprocessor12` exposes `shared_output_handle` and `output_byte_size`. The handle is created once and closed in `Drop` — minting one per call would leave the caller holding something it must close at a moment it cannot determine, since importers reference the handle rather than taking ownership.

    `examples/gpu_tensor_to_cupy.py` is the ~60 lines of ctypes that turn that handle into a `cupy.ndarray`. It lives in `examples/` deliberately: § 11 says RapidShot produces frames and does not own its consumers' bindings, and the same argument that keeps ONNX Runtime out of the core (§ 8) applies here. Verified byte-identical to `read_back()` for the same dispatch, with a CUDA kernel reading the tensor in place — shape and dtype would have agreed even if the import had mapped unrelated memory, so the pixel comparison is the whole check.

    `tests/test_cuda_interop.py` imports **the shipped example file** rather than a copy, so what ships is what is tested. § 5's rule — anything not run before a release is not verified for that release — otherwise puts an example verified by hand in the same category as one nobody ran.
- **Every native pyclass is `unsendable`, and Python decides which thread drops them. Not investigated.** Seen once, in a full suite run on 2026-08-21:

  ```
  RuntimeError: _rapidshot_native::TestTexture is unsendable, but is being
  dropped on another thread
  ```

  `#[pyclass(unsendable)]` tells PyO3 the object may only be touched on the thread that created it, and PyO3 enforces that **on drop** by raising. The catch is that dropping is not something the caller schedules: a Python object dies when the garbage collector gets to it, and the collector runs on whichever thread happens to trigger it. So the rule is "created and destroyed on one thread", but only the first half is under anyone's control.

  **This is not test scaffolding.** Four classes carry the annotation and three of them are shipping API:

  | Class | `native/src/lib.rs` | |
  | --- | --- | --- |
  | `GpuPreprocessor` | 377 | public |
  | `GpuPreprocessor12` | 561 | public — the Stage 6 tensor |
  | `CrossAdapterTransfer` | 770 | public — § 6.1 |
  | `TestTexture` | 1089 | test only |

  `ScreenCapture.start()` runs capture on its own thread, so a consumer that builds a preprocessor on the main thread and drops it while the capture thread is what triggers collection is an ordinary arrangement, not a contrived one.

  **The failure mode is worse than the failure.** It raises during garbage collection, where there is no caller to receive it: Python reports it as an *unraisable* exception, pytest turns it into a warning, and a plain application prints it to stderr and continues. So a real lifetime violation looks like log noise, and whatever the drop was supposed to release — a D3D12 resource, a shared NT handle, a COM interface — may not have been released. That is the same class of hazard as the shared-handle entry below: nothing looks wrong afterwards.

  **Nothing here is diagnosed yet.** The mechanism above is read off the annotation and the source, not reproduced deliberately, and the warning has been observed exactly once. Open questions, in order: is `unsendable` actually required for each of these (D3D11 devices are free-threaded by default, so it may be inherited caution rather than a real constraint); if it is, should these objects carry an explicit `close()` so release is scheduled rather than left to the collector; and does the existing ownership chain in `examples/gpu_tensor_to_cupy.py` make this reachable for a real consumer today. **Reproduce it deliberately before changing anything** — § 5's rule about paths nobody can trigger applies here, and a warning seen once in one suite run is not yet a bug that has been understood.

- **A shared handle is an integer, and every lifetime mistake around it looks like a valid number.** `shared_output_handle` is borrowed: it is closed when the preprocessor is dropped, and the D3D12 resource it names goes with it. A consumer holding the integer, or a CuPy array pointing into that VRAM, has nothing that looks wrong afterwards.

  The first version of `CudaTensor` in the example had exactly this bug — it read the preprocessor in `__init__` and kept no reference, so `CudaTensor(native.GpuPreprocessor12(frame, 640, 640))` left a live-looking array over released memory. Fixed by holding the preprocessor and passing `owner=self` to `UnownedMemory`, which makes the chain array → memory → view → preprocessor explicit.

  **The lesson is in how it was verified, not in the fix.** Measured 2026-08-06 with the ownership chain deliberately removed:

  | | with the fix | without |
  | --- | --- | --- |
  | Shared handle still open after collection | yes | **no** |
  | Reading the CuPy view | correct | **correct** |

  The handle was closed, the D3D12 resource released — and **the view still returned byte-identical data**, because nothing had claimed the freed VRAM yet. A pixel comparison passes with the bug fully present. The test therefore asserts *reachability* with a `weakref`, which is deterministic, and says in its docstring not to "fix" it later by comparing arrays. Any use-after-free test in this project that compares data instead of lifetimes is measuring luck.
- ~~**`shot()` writes BGRA regardless of `output_color`**, and overruns an undersized buffer without bounds checks.~~ **Fixed.** It writes the configured colour mode, and `_validate_destination` checks the size *before* any capture work — deferring it to the processor would have made it fire only on calls that receive new content, so an undersized buffer would return `False` on a static desktop and raise later, when something happened to move. Unsized raw pointers are rejected unless `buffer_size` is supplied. Nine tests in `tests/test_color_modes.py` cover it, including a sentinel guard that fails on a single byte past the end.
- **`pipeline.cpu_to_nchw` was a strawman, and fixing it cost the GPU comparison 1.78×.** This row is not library code — RapidShot ships no CPU preprocess — it is the reference arm the GPU tensor path is measured against, and **Stage 6 was promoted on the strength of that comparison** (§ 11). It was written naively: it widened the 640×640×4 gather to float32 *before* scaling (6.55 MB of traffic where 1.6 MB suffices), divided in a second pass, stacked channels into a fresh array in a third, and allocated ~11 MB per call. Writing each channel once into a preallocated destination is **6.84 → 3.90 ms for a bit-identical result**.

  The conclusion survives — GPU dispatch is 2 µs against 3.9 ms, so a consumer that keeps the tensor on the device still wins overwhelmingly — but the margin was overstated by 1.78× for as long as the reference was the first implementation rather than the best one. **A benchmark's control arm is part of the claim it supports.**

  One variant looked 7× faster and was wrong: replacing the fancy-index gather with a strided slice, on the assumption the indices were a uniform decimation. They are not — 1080/640 = 1.6875, so the indices step 0, 1, 3, 5, 6, 8 — and the strided version silently sampled a different set of pixels. It was caught only because every variant is checked against the original's output before its time is believed.

  ~~Still the largest single CPU cost at 3.90 ms~~ — **and it was a strawman a second time, for a second reason.** Machine B reported this row 24% slower than Machine A despite newer silicon, which looked like a hardware curiosity. Pricing the row against the bytes it moves said otherwise:

  | stage | min | MB moved | GB/s | % of this machine's ceiling |
  | --- | --- | --- | --- | --- |
  | gather, `src[np.ix_(ys, xs)]` | 3.72 ms | 3.28 | **0.88** | **1%** |
  | 3× divide → float32 | 1.17 ms | 6.14 | 5.27 | 8% |

  **The gather was 73% of the row and running at 1% of memory bandwidth.** Two-dimensional advanced indexing walks the output element by element. Two sequential one-dimensional `take` calls do not, and produce byte-identical output: **3.72 → 1.50 ms, 2.5×**.

  The bandwidth hypothesis that motivated the check was wrong, incidentally — the ratio of times (1.31×) and the ratio of memory bandwidths (0.53×) do not agree at all, which is what ruled it out. The row is gather-bound, not bandwidth-bound, and the cross-machine gap follows from memory *latency* and NumPy's inner loop rather than throughput.

  **This is shipped code, not only a benchmark.** `rapidshot.to_nchw()` used the same `np.ix_` gather, so the fix is user-facing: **1.83× at 1920×1080 → 640×640 and 1.58× at 2560×1600 → 640×640, bit-identical**, negligible at 320² where the gather is small. `_gather` in `rapidshot/preprocess.py` and `cpu_pipeline` in `perf_suite.py` now match.

  The strided-slice variant § 10 records as *wrong* was re-tested alongside and is still wrong — it reports an 18,000× speedup and samples a different image. It is kept in the variant test so that stays demonstrated rather than remembered.

  **Recordings are not comparable on this row across the change**, for the second time in its history.
- **"GPU dispatch is 2 µs" is true and is not the number a Stage 6 consumer pays.** The two halves of that comparison come from different code paths, and this document's own summary line conflates them. `pipeline.gpu_dispatch` benchmarks **`GpuPreprocessor`** — the D3D11 preprocessor — and times *submission only*; `perf_suite.py` prints the caveat plainly ("the GPU executes asynchronously, so it is not a measure of total work done"), but § 10's one-liner did not carry it. **`GpuPreprocessor12`, the path that reaches DirectML and the only one Stage 6 can use, calls `wait_for_gpu()` inside `process()`** and therefore measures completion.

  Measured 2026-08-06 on Machine B. § 2 forbids building a D3D12 preprocessor over a synthetic texture, so both preprocessors were run over one live frame to hold the source fixed, with the synthetic D3D11 row included for continuity with `baseline.json`:

  | Path | Source | min | p50 |
  | --- | --- | --- | --- |
  | D3D11, submit only | synthetic 1920×1080 | **1.3 µs** | 1.6 µs |
  | D3D11, submit only | live 2560×1600 | 2.1 µs | 2.6 µs |
  | **D3D12, blocking** | live 2560×1600 | **175.6 µs** | 254.0 µs |
  | *Machine A, `baseline.json`, D3D11 submit only* | *synthetic 1920×1080* | *1.6 µs* | *2.6 µs* |

  So the 2 µs figure **reproduces on NVIDIA**, slightly faster, and source resolution barely moves it — as it should not, being a CPU-side submission cost. But the honest figure for the D3D12 path was **~176 µs**, roughly 100× more, and after the hoist below it is **~70 µs**. The conclusion survives comfortably — 70 µs against Machine B's 3.07 ms CPU arm is ~44×, and the GPU arm additionally leaves the tensor where a model wants it. What does not survive is quoting 2 µs for the DirectML path.

  **Then the 176 µs turned out to be almost entirely avoidable, and was avoided.** Scaling the output size answered where it went — if the cost were shader work it would track the pixels written:

  | target | pixels out | before | after |
  | --- | --- | --- | --- |
  | 160×160 | 25,600 | 185.2 µs | **65.0 µs** |
  | 320×320 | 102,400 | 187.3 µs | 66.6 µs |
  | 640×640 | 409,600 | 197.0 µs | **75.7 µs** |
  | 1280×1280 | 1,638,400 | 282.1 µs | 93.1 µs |

  **64× the pixels cost 1.5× the time**, so it was never shader work — it was a fixed floor paid per call regardless of the work requested. `process()` was doing `CreateSharedHandle` → `OpenSharedHandle` → create SRV → create UAV → `CloseHandle` on **every frame**, for a texture Desktop Duplication hands back unchanged. § 6.1 had already recorded that the capture texture pointer is stable across frames and that caching was therefore possible "if a real profile ever justifies it" — this was that profile.

  Opening the texture is now per-*texture* work, cached in `preprocess12.rs` and **keyed on the raw pointer, not assumed stable**: a different pointer reopens rather than reusing a resource that describes a surface the caller is no longer capturing. The UAV moved to the constructor, where it always belonged — it describes the output buffer, which never changes. **2.6× at 640×640, and the fixed floor fell from 185 µs to 65 µs.**

  **What remains of the floor was then measured rather than guessed.** An earlier revision of this entry attributed it to "command-list reset, submission, and the fence round-trip" without saying which dominated, which is a list of suspects rather than a finding. `GpuPreprocessor12.probe_dispatch_phases()` times each phase separately; measured 2026-08-06, source 2560×1600, 200 runs:

  | target | record | submit | signal | **wait** (p50) | total | wait share |
  | --- | --- | --- | --- | --- | --- | --- |
  | 64² | 6.6 | 9.2 | 1.3 | **39.6** | 56.7 | 70% |
  | 320² | 5.9 | 8.4 | 1.3 | **42.4** | 58.0 | 73% |
  | 640² | 5.9 | 8.5 | 1.2 | **47.3** | 62.9 | 75% |
  | 1280² | 5.9 | 8.6 | 1.2 | **72.6** | 88.3 | 82% |

  **The CPU-side floor is 15.7 µs and perfectly flat**; the fence wait is 70–82% of the dispatch and the only phase that moves with output size. Two things follow. At 64×64 the shader has essentially nothing to do and the wait is *still* 39.6 µs, so ~38 µs of it is fixed GPU submit-to-completion latency — queue scheduling and fence propagation, not work. And the actual shader is the difference: roughly **9 µs at 640×640**. The tensor really is nearly free; the synchronisation around it is not.

  **The hoist also cut the marginal, per-pixel cost by 2.5×, which a fixed-cost removal cannot do.** That was flagged as an unexplained anomaly and possible clock artifact; it is neither. Measured back to back by rebuilding with the cache forced off, so both builds ran minutes apart on the same machine:

  | build | fixed floor (160²) | marginal | 640² total |
  | --- | --- | --- | --- |
  | texture cached (current) | **55–60 µs** | **21.6–23.8 µs/Mpx** | 67.5–70.0 µs |
  | forced reopen (pre-hoist) | 153.5 µs | 59.2 µs/Mpx | 169.4 µs |

  So reopening per frame cost ~98 µs of fixed work *and* made every pixel 2.5× more expensive. The leading explanation is that the old code rebuilt SRV and UAV descriptors in a **shader-visible** heap on every call, and rewriting descriptors the GPU may still be reading forces the driver to serialise against the previous dispatch — a stall that scales with dispatch duration, hence with output size. **That mechanism is inferred, not proven**; confirming it needs GPU-side profiling this project does not currently do. The measurement stands on its own either way.

  The GPU-clock hypothesis was checked and rejected: under sustained load the SM clock sits pinned at 2595 MHz and only the memory clock varies, by 14% — nowhere near enough to produce 2.5×.

  **The blocking wait stays anyway, and now for a quantified reason rather than caution.** Going async would return ~47 µs of calling-thread time per frame, but it changes when a caller may legally read the tensor, and a CUDA consumer has no way to know the dispatch finished without one. The correct mechanism exists — a shared D3D12 fence imported via `cudaImportExternalSemaphore` — and it is the *same* shared-fence work § 6.1 defers for the cross-adapter path. Do both together or neither.

  **Read `wait` as a median, never a minimum.** It is bimodal: whenever the GPU has already finished, `GetCompletedValue` clears immediately and the sample is ~0. A first version of this probe reported minima and produced `wait = 0.0 µs` at 320² sitting between 29 µs and 47 µs at neighbouring sizes — the same "a minimum is monotonically non-increasing in sample count" trap § 3 records for the live rows, arriving in a new place. The probe now returns both, and its docstring says which to read.
- **`pipeline.gpu_plus_readback` is substantially a Python-object benchmark.** `read_back()` returns `Vec<f32>`, which PyO3 marshals into a Python **list**, and `native.py:328` then re-parses that list with `np.asarray`. For a 640×640 tensor that is **1,228,800 Python float objects** built and destroyed per call. Confirmed directly — `type(read_back()) is list, len 1228800`.

  Measured on Machine B: `read_back()` costs 36.3 ms unpinned, of which `np.asarray` over an equal-length Python list accounts for **21.1 ms on its own**, before counting what PyO3 spends constructing the list. The actual 4.92 MB device-to-host copy is a minority of the row. (Pinned to P-cores the whole row drops to 14.2 ms, which is itself evidence the cost is CPU-bound rather than PCIe-bound.)

  This is the `cpu_to_nchw` strawman again in a new place, and § 10 already states the rule it breaks: **a benchmark's control arm is part of the claim it supports.** The conclusion is unharmed — the readback row exists to show that paying the round-trip loses, and it still loses — but the margin was inflated by an implementation detail of the verification helper, not by the cost of moving bytes off a GPU. `read_back` is documented "verification only", so this was never on a consumer's hot path; it is the *measurement* that was wrong, which is worse, because measurements are what decisions get made from.

  **Fixed 2026-08-06.** Both preprocessors now return raw bytes (`floats_as_bytes` in `lib.rs`) and the Python wrappers use `np.frombuffer`, which reinterprets rather than converts. **`read_back()` 14.19 → 2.24 ms (6.3×)**, and `pipeline.gpu_plus_readback` 14.19 → 2.06 ms in the recorded baseline. 2 ms for a 4.92 MB device-to-host copy is finally a number that describes PCIe rather than the CPython allocator. The row still loses to the CPU arm, which is what it exists to show — it just now loses by the right margin.
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

  **The revisit condition arrived, and the answer did not change.** Machine B sustains **165 fps**, past the >144 threshold named above. Re-measured there 2026-08-06, source 2560×1600:

  | phase | min | p50 |
  | --- | --- | --- |
  | update + GPU copy | 0.126 ms | 3.039 ms |
  | **map staging surface** | 0.507 ms | **1.469 ms** |
  | process (RGB) | 1.160 ms | 1.358 ms |
  | unmap | 0.010 ms | 0.030 ms |
  | `grab()` total | 3.256 ms | 6.107 ms |

  | delay before map | map p50 | grab p50 | fps |
  | --- | --- | --- | --- |
  | 0 ms | 1.495 ms | 6.107 ms | **165.2** |
  | 1 ms | 0.284 ms | 6.127 ms | 163.7 |
  | 2 ms | **0.062 ms** | 6.101 ms | **165.0** |
  | 5 ms | 0.057 ms | 7.109 ms | 133.8 |

  The map collapses **24×** given 2 ms of slack, so it is pure GPU wait on NVIDIA exactly as on Intel — the finding reproduces on entirely different hardware. And **fps still does not move**: 165.2 → 165.0 across the range where the map goes from 1.5 ms to 0.06 ms. The 5 ms row is the sleep exceeding the frame period, not a property of the map.

  So the conclusion holds at 165 fps and is no longer conditional on a slow panel: pipelining the map away buys **no throughput**, only ~1.5 ms of calling-thread time per frame (24% of a 6.1 ms `grab()`). Worth having for a consumer that needs the CPU, still not worth the semantic change on its own. Treat the ">144 Hz" trigger above as **discharged**.
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
