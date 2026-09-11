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

**Whether 2.1.0, 2.2.0 and 2.3.0 actually published cannot be verified from the repository** — that is the release workflow's outcome, not a file in the tree. Confirm in the GitHub UI that all three Releases exist with their assets. 2.4.0 needs no such check: PyPI serves it and the Release carries its assets, both confirmed on 2026-08-22.

**1.1.0 is yanked**, reason *Newer Version*. It was the only thing on PyPI from April 2025 to now and it did not work: it failed to import on Python 3.11+ (`cursor: Cursor = Cursor()` trips the dataclass mutable-default check broadened in 3.11), and patching that one line only got it to return all-black frames, because the processor was handed a texture where it expected a mapped staging surface. Yanking is not deletion — an existing `rapidshot==1.1.0` pin still resolves, which is the intent; only new unpinned installs are steered away.

Two further items cannot be verified from the repository either and should be confirmed in the same pass: **Settings → Security → private vulnerability reporting** must be enabled (or the link in `SECURITY.md` 404s), and branch protection needs *Require review from Code Owners* (or `CODEOWNERS` is only a routing hint).

**Next feature task:** § 6.3 — finish Stage 3 (Frame metadata). It is smaller than it was: of the three pieces listed there, **timestamps are done** (`Frame.timestamp_qpc` and `Frame.timestamp`), **cursor is half done** (`Frame.cursor_visible` exists; position and shape are captured in `core/duplicator.py` and never surfaced), and **`Protocol`-typed interfaces are not started** — there is no `Protocol` anywhere in the package. § 6.1 is complete: hybrid and headless systems are reported clearly, a captured frame crosses to a second adapter at **0.70–0.98 ms per 1080p frame** verified byte-exact, and the convert-first-or-transfer-first question has been measured and settled in favour of transferring the frame. **§ 6.1's validation on real hybrid hardware is now done** (2026-08-22, Intel→NVIDIA, byte-exact — see above), **and the asynchronous shared fence shipped in 2.4.0**, which was the last piece outstanding there. § 6.1 is closed in both directions: `transfer_async()` submits without blocking, `shared_fence_handle` lets a CUDA consumer wait on the GPU, and `set_consumer_fence()` / `wait_for_consumer()` close the reverse hazard where the producer overwrites a buffer the consumer is still reading.

**Do not start another round of D3D12 synchronisation work without a measurement or a user report asking for it.** That seam has been through feature work, hardware validation, adversarial review, four rounds of fixes, targeted regression tests and a clean final pass. The next thing to do here is nothing; the returns are elsewhere.

**The next task is § 7.0 — build the AI-ingestion benchmark — and it comes before § 6.3.** § 6.3 (finish `Frame` metadata) is still worth doing and is now folded into § 7.1, but it is no longer the front of the queue. § 7 explains why: every remaining proposal is a performance claim, and the only benchmark this project has measures `grab()` — the path where RapidShot's advantage is smallest and where it loses the frame-rate column outright. Until *present → model-ready tensor* is measured, feature ordering is guesswork, and § 11 has been right about that twice already.

**The largest of those returns was distribution, and it is now built.** § 6.1's cross-adapter transfer, § 6.6's CUDA interop, the AVX2 kernels of § 3 and 2.4.0's synchronisation all sat behind an extension the consumer had to build, costing a Rust toolchain, MSVC build tools and `native/install_dev.py`. Every measurement in § 3 that makes this library worth choosing was one a `pip install` could not reach.

`rapidshot-native` closes that: `pip install rapidshot[native]` and the extension is present, no toolchain. One **abi3** wheel covers Python 3.9 and every later version, which is what makes this a single artefact rather than a matrix that grows with each CPython release — the crate already carried `pyo3/abi3-py39`, so the work was packaging rather than porting.

It is a **separate distribution on a separate tag**, published by `release-native.yml`. `rapidshot` stays `py3-none-any`, so the guard that fails the main release if a compiled artefact appears in it (§ 10) is preserved rather than negotiated with. The two version independently: the Rust moves on its own schedule, and lockstep would republish an identical binary under a new number on every RapidShot release.

**Nothing has been published yet.** The wheel builds, installs into a clean environment outside the repository, is found by `rapidshot.native`, and passes both CI and a local end-to-end check — but no `native-v*` tag has been pushed. Until one is, `pip install rapidshot[native]` resolves to nothing. `RELEASING.md` records the ordering, including why `rapidshot-native` is deliberately absent from the `all` extra until the first wheel is live.

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
| CPU | — | Intel i9-14900HX, **8 P-cores + 16 E-cores** |
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
| **Intel + NVIDIA Optimus** | **Verified 2026-08-22 on Machine B.** Capture on the iGPU, cross-adapter transfer to the dGPU byte-exact. Needed one NVIDIA control-panel setting to reach; see above |
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

- **Live capture tests need screen activity.** Desktop Duplication only reports *changed* content, so an idle screen produces zero frames and tests fail for reasons unrelated to the code.
- **Synthetic textures cannot test the D3D12 path.** D3D11 refuses `SHARED_NTHANDLE` without `SHARED_KEYEDMUTEX`, and a keyed-mutex resource reads as zeros until acquired — on *both* APIs. The real duplicated surface has its mutex managed by DXGI. Use live capture. Constructing `GpuPreprocessor12` over a `TestTexture` fails at the constructor with `D3D12 preprocessor setup failed: texture is not shareable ... (0x80070057)`, which is the guard working as designed — but it means **`baseline.json` has no D3D12 row at all**, and any D3D12-versus-D3D11 comparison has to be run over one live frame. See § 10.
- **Benchmark noise is severe on a loaded machine.** Naive comparison once reported 11 false regressions up to 1.9× on *identical* code. The suite compensates with pooled rounds, minimum-sample comparison, and a control benchmark; run `--self-test` to measure the current noise floor before trusting any result.
- **On a hybrid P-core/E-core CPU, pin the benchmark process to the P-cores or the numbers are meaningless.** Measured 2026-08-06 on Machine B (8 P-cores, 16 E-cores) with `--self-test`, which compares the suite *to itself with no code change*:

  | | Unpinned | Pinned to P-cores |
  | --- | --- | --- |
  | Rows exceeding the 1.30× threshold | 2 | **0** |
  | Worst false verdict | `SLOWER 2.57×` | every row `~ same` |
  | Spurious `FASTER` verdicts | 3.20× on `gpu_dispatch` | none |
  | `pipeline.cpu_to_nchw` across runs | 5.5 – 16.0 ms | 5.06 – 5.31 ms |

  Windows moves benchmark threads onto E-cores under no particular provocation, and an E-core reads as a 2–3× regression on exactly the compute-bound rows that matter. **The control benchmark does not rescue this** — `control.memcopy` reported "machine state comparable, 1.01×" in the same run that called `shot.RGB` 2.57× slower, because the control got scheduled well and the others did not. That is the § 3 lesson about one control standing in for workloads it does not resemble, arriving by a new route. Machine A has no E-cores, which is why this never appeared before.

  **`perf_suite.py` now pins itself**, so a bare invocation is correct again. It reads `EfficiencyClass` from `GetSystemCpuSetInformation`, restricts the process to the highest class when a machine has more than one, and prints what it did. A uniform CPU is left alone. The recording carries `cpu_topology`, `pinned_to_performance_cores` and `affinity_mask` in its `machine` block, because a pinned and an unpinned recording are not comparable and nothing in the numbers alone distinguishes them. `--no-pin` disables it. Leaving this to the invocation was the wrong default: a suite that silently produces 2.5× noise unless the caller remembers a `start /affinity` prefix is a suite that teaches people to ignore it.

  **Check the `machine` block after recording.** That provenance shipped broken twice in one sitting — first because `GetProcessAffinityMask` needs explicit `argtypes` (without them ctypes coerces the process pseudo-handle to a C int and raises `OverflowError`, which a broad `except` then swallowed into a silent `None`), and then because `machine_info()` was called *before* the pin, so it faithfully recorded the state the process started in rather than the one it benchmarked in. Both produced a recording that looked complete and described the wrong run. The `except` now records `cpu_topology: "unknown (<error>)"` rather than dropping the keys, since a missing field is indistinguishable from a recording made before the field existed.
- **Pace benchmarks to a frame period, never to a fixed gap or a burn loop.** Sustained heavy vector work holds the CPU in a lower power state, and GRAY has two modes because of it — **16.27 ms back-to-back, 9.16 ms with a 16 ms gap, 9.91 ms in bursts with a 200 ms gap.** The trap is that *both* extremes are wrong. A benchmark's real duty cycle follows from its own cost: RGB takes 1.8 ms of a 16.7 ms frame (~11% duty cycle, mostly idle) while GRAY takes 15.9 ms (~95%, effectively sustained). `perf_suite.py` therefore sleeps out the remainder of a 60 Hz frame after each rep, which reproduces both from one rule; a fixed gap handed GRAY a 50% duty cycle and reported a number no capture loop achieves. The memcpy control cannot catch any of this — memcpy is not heavy enough to trigger it, so it reports "machine state comparable" throughout.
- **Re-record `baseline.json` whenever the harness changes how it drives benchmarks**, and verify immediately with a second run that should read all `~ same`. Numbers from different pacing models are not comparable, and after the fact you cannot separate a harness change from machine drift.
- **The control benchmark only rescues comparisons that resemble it, and cannot rescue one across machines at all.** `control.memcopy` measures memory bandwidth, so dividing by its movement normalises benchmarks that are *also* bandwidth-bound and quietly mis-normalises everything else. On a CI runner this reported `pipeline.cpu_to_nchw` — float32 resize/normalise/transpose, compute-bound and sensitive to vector width and NumPy version — as a **1.34× regression against a code path nobody had touched**, while simultaneously calling every conversion row **1.4× faster on a machine that was uniformly slower**. One control standing in for workloads it does not resemble, wrong in both directions at once. `print_comparison` now detects a baseline recorded on different hardware (processor / platform / GPU), marks every verdict *indicative*, and gates nothing; a spurious improvement is flagged as loudly as a spurious regression, because nobody investigates good news. To compare code against code, re-record on the machine you are testing on.
- **CI cannot verify live capture.** GitHub runners have no desktop session. Those tests skip themselves and must be run on real hardware before a release.

---

## 3. Measured baseline

All figures 1920×1080 BGRA (8.3 MB/frame), measured on **Machine A** (§ 2) unless a row says otherwise. Stored in `benchmarks/baseline.json`, **re-recorded 2026-08-05T12:14Z** after the GRAY work in § 10 *and* the `pipeline.cpu_to_nchw` correction in § 10 — the latter changes what that row measures, so recordings from before it are not comparable on it. Earlier recordings are kept as `benchmarks/baseline-2026-07-30.json` and `benchmarks/baseline-2026-07-27.json`; do not compare across recordings casually — the 07-27 one drove the benchmarks differently, and the notes below apply to the current one.

**There are two committed recordings, and which one you want depends on the question:**

| File | Extension | What it answers |
| --- | --- | --- |
| `baseline.json` | **built** | What the library can do on stated hardware. Feeds the README badges. |
| `baseline-nonative.json` | absent | What `pip install rapidshot` gets, and what CI compares against — CI has no toolchain. |
| `baseline-rtx4060.json` | **built** | Machine B, P-core-pinned, live rows included. **Not** a replacement for `baseline.json` and not wired to anything. |

Both were recorded back-to-back on 2026-08-05 with `--rounds 5 --reps 25`, the invocation § 2 documents, so they are directly comparable to each other rather than separated by machine drift. Two consequences worth knowing:

- **CI's compare step points at `baseline-nonative.json`.** Aimed at `baseline.json` it would report a 6–20× "regression" on every conversion row forever, since the runner builds no extension — which is how a benchmark suite teaches people to ignore it.
- **`pipeline.gpu_dispatch` and `pipeline.gpu_plus_readback` appear only in `baseline.json`**; they need the extension.
- The **live rows were recorded against a defined synthetic workload**: a 420×300 window moved at ~30 Hz. See the dirty-fraction note in § 6.3 — a small moving window is the *favourable* end of that distribution.

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
| Pixel conversion, RGB/BGR (NumPy, post-optimisation) | 1.92–1.93 ms — 0.30 ms native, see below |
| Pixel conversion, GRAY (NumPy) | 6.9–10.5 ms — was 13.7–14.9; 0.26 ms native, see § 10 |
| Preprocess for a model (resize/normalise/CHW → 640×640) | **3.90 ms** — was 6.23, see below |
| **CPU total, capture → model input** | **~8 ms** without the extension |

The conversion row is the one the optional extension changes, and it changes it by 6–37×. **With the extension the total is no longer conversion-dominated** — it is the staging map plus the preprocess, neither of which the kernels touch. § 10 re-profiles a real `grab()` on that basis.

Capture path comparison, real capture, from the two committed recordings:

| Path | `baseline.json` (built) | `baseline-nonative.json` | |
| --- | --- | --- | --- |
| `grab()` — CPU staging read + convert | 2.41 ms | 5.93 ms | **see the caveat below** |
| `grab_frame()` — texture stays on GPU | 0.16 ms | 0.19 ms | **see the caveat below** |

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

The spec requirement is real even so, and is recorded here as a latent hole rather than a closed one: [MSDN states that to produce a visually accurate copy an application must process all move rects before it processes dirty rects](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-getframemoverects). The § 6.3 accumulator patches dirty rects only, so against a source that *does* report moves it would leave stale pixels at the move destinations. This is verified unobservable here, not proven impossible everywhere — one GPU, one driver, one OS build. Treat `move_rects` as **not worth implementing** until a source that emits them is found: the code path cannot be exercised on available hardware, and synthetic metadata proves nothing (§ 2).

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
| **6.1 (frame transfer)** ✅ | `native.cross_adapter_transfer(frame)` carries a captured frame to a second adapter and exposes the `ID3D12Resource` it lands in. Heap and placed resources are allocated once; only the copy is per-frame. Verified byte-exact on real capture by `examples/verify_cross_adapter.py` — 8,294,400 bytes per frame, against a source-side readback of the same snapshot |
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

**Also fixed:** `pip install rapidshot` shipped a broken package — `pyproject.toml` listed `packages = ["rapidshot"]`, so the wheel contained 5 modules instead of 25 and failed with `ModuleNotFoundError: No module named 'rapidshot.util'`. Invisible from a source checkout. Now guarded by CI.

**Test coverage:** `python -m pytest tests/ -q` collects 282. The skips are environmental, not pending work, and **the count is a property of the machine, not of the code**:

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

Remaining work:

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
| Cursor data on `Frame` | **Half done.** `Frame.cursor_visible` exists. Position and shape are already read into `Duplicator.cursor` (`PointerPositionInfo`, `Shape`) on every frame and never reach `Frame` |
| `Protocol`-typed interfaces | **Not started.** No `Protocol` anywhere in `rapidshot/` |

Cursor position needs the frame-coordinate treatment below — it is reported against the duplicated output, so on a region capture it is wrong in exactly the case nobody checks by hand. It also needs the empty-versus-unknown distinction: a hidden cursor and an unreadable pointer are different answers.

`move_rects` is **deferred, not pending.** The COM signature is declared correctly and is ready to call, but DWM never emits a move rect (§ 4), so there is nothing to wire it to and no way to test what was wired.

Design this **before** a second backend exists — retrofitting a DXGI-shaped API to fit WGC later is more expensive than designing one abstraction all backends fill.

**Settled by `dirty_rects`, and it applies to the rest:**

- **Frame metadata must be in frame coordinates, not desktop coordinates.** DXGI reports rects relative to the whole duplicated output; a `Frame` may cover a region of it. Passing raw values through would make `dirty_rects` index outside the frame whenever a region is off-origin — wrong only in the case nobody checks by hand. `Frame` clips and translates. `move_rects` and cursor position need the same treatment.
- **Empty and unknown are different answers.** `[]` means no rects were reported; `None` means the metadata could not be read. A consumer skipping unchanged regions must distinguish them or it silently skips everything on a frame whose metadata failed. An empty list does *not* mean nothing changed — a mode change or a coalescing driver can report none while the image differs completely.
- **`RectsCoalesced` matters.** When set, the driver merged rects, so they over-estimate what changed. Surfaced as `frame.rects_coalesced`.
- The COM signatures for `GetFrameDirtyRects`/`GetFrameMoveRects` were declared in `_libs/dxgi.py` without argtypes, so they were callable but unusable — comtypes could not marshal the out-parameters. `GetFrameMoveRects` is now declared correctly, but **wiring it up is not the next piece** — § 4 measured DWM emitting zero move rects over 2,205 frames of a workload built to produce them, so it is deferred. The declaration is ready for whoever finds a source that emits them.

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
semaphore wait would be a real win. Measured as pixel age (below), the deficit
does not appear: cross-adapter pixels are *younger* than the CPU path's (35.05
against 37.55 ms p50), though it delivers fewer unique frames (81.0 against
100.4 a second). The semaphore wait takes another 1.1 ms off, at 3.4 ms more CPU
per frame.

**Caveats that travel with the call-duration tables above.** One machine, hybrid
Intel→NVIDIA. On a single-adapter NVIDIA box the direct `GpuPreprocessor12 →
CUDA` path is available and would likely beat all six — it raises
`CrossAdapterRequired` here, because CUDA cannot import a resource owned by the
capture adapter. Frame rate is bounded by compositor presents, not by the
pipeline. The motion source ran uncapped at ≥498 updates/s and is not the
limiter; **a capped source made every path report 82–86 fps and looked like a
tie**, which is the failure `benchmarks/motion_source.py` was written to warn
about.

#### Pixel age, measured against a shared clock — 2026-09-10 ✅

The measurement § 7.0 actually asked for. `native/src/bin/latency_source.rs` is a
D3D11 source that encodes an incrementing frame ID into the image and records
`frame_id → QPC` at every `Present()`; each path decodes the ID out of the
captured pixels, so latency is **how old the pixels were**, not how long the call
took, and every library is measured against one clock.

Machine B, 2560×1600 at 165 Hz, 5 s per path, pixels to a `(1, 3, 640, 640)`
**FP16** tensor on CUDA — the contract § 7.0 specified, where the call-duration
run above used FP32. The source presented at a median **164.5/s** across 26
samples, minimum 136.2 — above the fastest path throughout, which is the
condition the throughput column needs. `benchmarks/section7-ingestion-machineB.json`,
which also records per-stage timings (capture call, GPU preprocess, fence wait)
and the QPC frequency for every path.

Age is measured from the QPC taken immediately before `Present()`, so it is
*submission* age: it includes the compositor's queue, not scan-out to the panel.

| path | unique fps | age p50 | p95 | p99 | CPU ms/frame | source frames dropped |
| --- | --- | --- | --- | --- | --- | --- |
| mss | 23.8 | 61.87 | 72.91 | 78.11 | 30.07 | 561 |
| dxcam (DXGI) | 76.3 | 41.08 | 46.30 | 48.13 | 15.58 | 444 |
| dxcam (WGC) | 68.0 | 46.46 | 55.94 | 58.96 | 17.83 | 482 |
| **rapidshot-cpu** | **100.4** | 37.55 | 42.85 | 45.07 | 12.02 | **323** |
| rapidshot-cupy | 93.3 | 38.28 | 43.93 | 46.26 | 7.83 | 358 |
| rapidshot-xadapter | 81.0 | 35.05 | 39.03 | 41.87 | **6.12** | 424 |
| rapidshot-xadapter-async | 79.5 | 35.29 | 40.34 | 43.47 | 6.44 | 429 |
| **rapidshot-xadapter-semaphore** | 82.0 | **33.95** | **37.63** | 41.15 | 9.53 | 417 |
| rapidshot-direct | — | unavailable: `CrossAdapterRequired` on hybrid hardware |

**RapidShot leads on every axis that separates the paths.** 100.4 unique frames
per second against DXcam's 76.3 (+32%) and WGC's 68.0; the lowest pixel age
(33.95 ms, the GPU-side semaphore path); and the lowest CPU per frame (6.12 ms
cross-adapter, 2.5× cheaper than DXcam's 15.58). mss is not in the same class.

**The GPU-side semaphore wait is the best latency path**, marginally ahead of
blocking (33.95 vs 35.05 ms p50) — the first end-to-end confirmation of § 6.1's
mechanism inside a capture benchmark rather than a synthetic consumer.

**Everything drops source frames**, 323–561 of roughly 820 presented, because the
source runs at 165/s and no path sustains that. The drop column is therefore a
real differentiator here rather than a rounding error, and RapidShot's CPU path
drops the fewest.

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
RapidShot returned 32% more unique frames than DXcam; here it is 20%. The loop is
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

**`GpuConverter` — rank 2, and build it before the encoder.** A reusable colourspace/format layer: `BGRA8 / RGBA8 / R10G10B10A2 / RGBA16F → NV12 / P010 / RGB`. Capture already handles HDR formats through `DuplicateOutput1`. Built once as infrastructure it serves ML, encoding, streaming and recording; buried inside a Stage 7 encoder it serves one of them.

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
