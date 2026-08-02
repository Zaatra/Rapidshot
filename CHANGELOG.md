# Changelog

All notable changes to Rapidshot are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries are grouped by the [ROADMAP.md](ROADMAP.md) stage that motivated them, so
each release can be traced back to the plan it implements.

## [Unreleased]

Nothing yet.

## [2.0.0] - 2026-08-02

First release with a stable API contract, and the first to be published through
PyPI Trusted Publishing with signed attestations.

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
