# Runbook for the RTX 4060 host (Machine B)

Written for a Claude session starting cold on the RTX laptop. Everything here is
work that **cannot** be done on the Intel host (HP 324pe), where the rest of the
2.9.0 inference measurements were taken.

**Check out `bench/rtx-harness`, not `dev`.** Two branches exist on purpose:

| Branch | Contains | Lifetime |
| --- | --- | --- |
| `dev` | The 2.9.0 release work only — package, tests, docs | Keep. This is what merges |
| `bench/rtx-harness` | All of `dev`, **plus** the benchmark scripts | **Disposable.** Delete it when the runs are done |

The benchmark harness lives under `.test/`, which is gitignored, so it cannot
reach this machine without being force-added — and anything force-added onto
`dev` would follow it into `main` forever. So it sits on a throwaway branch
instead. `bench/rtx-harness` is a descendant of `dev`, so one checkout gives you
everything; commit results there if you like, copy anything worth keeping into
`docs/change-detection/`, cherry-pick that onto `dev`, and then:

```powershell
git push origin --delete bench/rtx-harness
```

That leaves no harness in any branch anyone keeps.

Read [`inference-2.9.0.md`](inference-2.9.0.md) first. It records what was
measured on Intel, and — more usefully — a list of confounds that already cost
real time. Three of them will bite on GPU too.

## What is already true, so you do not re-derive it

| Finding on Intel | Why it matters to you |
| --- | --- |
| Gating saves **58% of consumer CPU** at a 75 ms model, 401 detector calls to 101, 0 of 100 states missed | The CPU half is done. Do not repeat it |
| The *metadata* verdict beats a full-frame compare by only **1.7%** with a real model, against 28% with a 15 ms synthetic one | The mechanism matters in inverse proportion to detector cost. Expect the GPU gap to be smaller still, because a GPU inference is cheaper in wall time but the compare cost is unchanged |
| A native DXGI binding with no metadata reached within **5.3%** of RapidShot's metadata | Always keep this control in. It is what stops the claim inflating |
| At ~210 ms per inference every library saturates and misses 7–14 of 100 states | Pick detector costs that leave the consumer headroom, or you measure the model |
| INT8 dynamic quantization **drops a detection** and is only 1.09× faster at 4 threads | Do not use INT8 as an arm. On GPU, prefer FP16, and verify detections before trusting it |

## Traps that already cost time here — all three apply on GPU

1. **Desktop Duplication almost never delivers a byte-identical frame.** A
   benchmark that just captures a live screen cannot measure inference
   avoidance: the no-filter baseline ran the detector exactly as often as the
   gated arm. `inference_avoidance.py` already forces duplicates. Use it rather
   than writing a fresh harness.
2. **`ScenePlayer.present()` returns without painting when nothing moved**, so
   naively re-showing a state puts *nothing* on the wire. The forced-duplicate
   path in `inference_avoidance.py` works around this; do not "simplify" it.
3. **Thread pools that spin hide the result.** ONNX Runtime's intra-op pool
   spin-waits between runs. With it on, gating appeared to save 14% of CPU; off,
   the same configuration showed 59%. The CPU detector sets
   `session.intra_op.allow_spinning` to `0`. Check whether the CUDA provider's
   host-side pool does anything similar on your box before trusting CPU numbers.

Two more specific to GPU, both of which make results look better than reality:

4. **CUDA work is asynchronous.** Time *completed* inference, not submission.
   `CudaYoloDetector.infer` binds the output to CUDA and calls
   `synchronize_outputs()` before the clock stops. If you add arms, keep that.
5. **Silent CPU fallback counts as GPU inference.** The detector sets
   `session.disable_cpu_ep_fallback` and raises if `CUDAExecutionProvider` is
   not first in `get_providers()`. Do not relax that to make a run succeed.

## Environment

The display must be held awake or unattended runs die: Desktop Duplication is
refused with `E_ACCESSDENIED` when the display sleeps, and it looks nothing like
a sleep problem. `inference_avoidance.py` and `inference_live_benchmark.py` both
hold it awake themselves. Anything else you write must too.

Check for orphaned processes holding memory before a campaign, and record what
screen-capture software is running — though note that contention was measured on
Intel and moved `rapidshot_tight` by 4 states of 1,530, so it is not the risk it
was once assumed to be.

```powershell
git fetch origin && git checkout bench/rtx-harness
```

Then, in the repo's virtual environment:

```powershell
python -m pip install onnxruntime-gpu cupy-cuda12x
```

`onnxruntime` (CPU) and `onnx` are already expected by the scripts;
`onnxruntime-gpu` replaces the CPU wheel — do not install both into one
environment, they conflict over the same module name. `cupy` is needed only by
`rapidshot/_bench/ai_pipeline.py`, not by `inference_avoidance.py`.

The YOLOX model and its two sample images are **not** in git. Fetch and verify
them against the published SHA-256 pins:

```powershell
python .test/change-detection/whole_frame_task.py --fetch-assets
```

That writes `.test/change-detection/assets/yolox/` and refuses anything whose
size or hash differs from the pin. 35,858,002 bytes, SHA-256
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.

`monitor_target.select_monitor()` defaults to `"HP 324pe"` and will raise on
your machine. Pass `--monitor "<your monitor name>"`; the name is the one
Windows reports, and the helper requires an attached, unrotated, 1080p DXGI
output, so a 2560×1600 panel needs that constraint relaxed — read
`monitor_target.py` before fighting it.

On `bench/rtx-harness`, `.test/change-detection/*.py` is tracked deliberately so
the harness reaches this machine. Everything else under `.test/` stays ignored,
so campaign outputs do not get committed by accident. Copy results worth keeping
into `docs/change-detection/`, as the Intel ones were.

**Two things are deliberately not in git at all**, and both will stop a script
rather than degrade quietly:

- **The native tile comparator.** `cpu_default.Native` loads
  `.test/change-detection/target/release/rapidshot_cpu_default_benchmark.dll`,
  built from a Rust crate whose sources (17 `.rs` files and four `.hlsl`
  shaders) were left out rather than ship a manifest with no code behind it.
  Only `inference_gate_cost.py`, `real_trace.py` and `live_cpu_default.py` need
  it; **none of the GPU priorities below do.** If you want it, copy
  `.test/change-detection/{Cargo.toml,build.cmd,src}` from the Intel host and
  `cargo build --release` — it depends on `native/core`, which *is* tracked.
- **The vendored `windows-capture` wheel** (see Priority 3).

## Priority 1 — exact changed tiles on a non-Intel GPU

**This is the 2.9.0 feature itself, and the README says plainly that it has
never run on NVIDIA or AMD.** Every figure for `grab_with_changes()` and
`grab_tiles()` comes from one Intel integrated GPU at 1920×1080 and 100 Hz. The
tested-hardware table carries the row *"Exact changed tiles on any non-Intel
GPU — **Not run**"*, and `docs/EXACT_TILES.md` opens by calling itself
provisional for exactly this reason. This machine is the first chance to change
that, and its 2560×1600 165 Hz panel also varies the panel and the rate, which
the README flags as unvaried.

### Set the MUX to discrete-only first, or the whole exercise is wasted

**In MUX hybrid mode the Intel UHD drives the display, so Desktop Duplication
captures on Intel** — the manifests from the 2.6 runs say so in as many words:
*"MUX hybrid: Intel UHD capture, RTX 4060 Laptop CUDA"* against *"MUX
discrete-only: RTX 4060 Laptop drives the display and does the CUDA work"*.

So a change-detection campaign in hybrid mode re-measures the Intel iGPU and
proves nothing new, however many hours it takes. **Switch to discrete-only, and
record which mode every result came from.** Hybrid is still worth a run, but as
a second data point, not the headline.

### Do not try to pair the exact paths with CUDA tensors

`grab_with_changes()` and `grab_tiles()` **refuse GPU output by design** — an
unsupported backend, GPU output, capture format or active stream raises
`ValueError`, and they require a CPU BGRA8 SDR surface. That is not a gap to
work around; the comparison is CPU-side on purpose, for the reason in the next
item. If a script raises there, it is the contract talking.

### The three things worth measuring

1. **Does GPU-side comparison win once readback crosses PCIe?** On Intel,
   comparing on the GPU cost **0.77 ms against 0.48 ms** for the CPU candidate
   comparison, because the frame is going to the CPU anyway — so the shipped
   comparison is CPU-side. The roadmap's § 4 entry says this is *"worth
   retesting on a discrete card, where the readback crosses PCIe; not before"*.
   This is that card. `idea_gpu.py` is the experiment. If GPU comparison wins
   here, the shipped design is wrong for discrete GPUs and that is a finding
   worth the trip on its own.
2. **Does staging only the dirty tiles save more on a discrete card?** On
   Intel, `grab_tiles()` cut readback per result from **7.91 MiB to 1.51 MiB**
   and median result pixel age from **6.33 ms to 5.09 ms** against the
   full-frame exact path. The README states the prediction and labels it one:
   *"Staging only the dirty tiles should matter **more** on a discrete card,
   where the copy crosses PCIe, but that is a prediction."* Confirm or refute
   it.
3. **Do the frozen release gates still pass on NVIDIA capture?** All four were
   validated on Intel only: correctness, memory **≤ 12 MiB**, idle CPU **≤ +5%**
   of process cycles/s, and a downstream saving **≥ 50%**. Re-run them rather
   than assume. `run_cost_campaign.py` drives the cost matrix,
   `exact_live_check.py` the live public-API check, `quiet_cycle_gate.py` and
   `idle_path_probe.py` the idle-CPU gate. Read each script's arguments before
   launching; several take hours.

### Free wins on this hardware

- **Cross-adapter tests actually run here.** On the Intel host they skip with
  *"only one hardware adapter; destination can only be WARP"*. A hybrid Intel
  UHD + RTX 4060 machine has two, and an earlier 2.6-era run exercised 32 live
  cross-adapter tests. Run the full suite in **both** MUX modes and compare the
  skip lists — the skips are where this feature is least tested.
- **Rotation at 0/90/180/270°** was covered on a Lunar Lake laptop but not with
  the exact paths. Cheap to add.

### Still not covered by anything, including this machine

Native 4K (the 4K figures are an offline replay at 3840×2160, not a panel), HDR
(the separate UHD 630 + 4K HDR host, which only does clipped 10-bit HDR), and
**AMD capturing** — the one case where the README notes a silent BGRA swizzle
mismatch would corrupt rather than crash.

## Priority 2 — refresh the README's GPU detector table

This is the highest-value run, because the table is **2.6.0/2.6.1 data recorded
2026-09-25** and the README says so. Three releases stale. It needs no new code.

```powershell
python benchmarks/section7_suite.py --help
```

Read `rapidshot/_bench/ai_pipeline.py` first: it requires CuPy's current CUDA
stream *and* a loaded `CUDAExecutionProvider`, and sets
`disable_cpu_ep_fallback`, so it raises rather than quietly degrade. That is
intentional. The rows to refresh are "Optimus" and "NVIDIA only" in the README's
**Through a detector** section, sourced from
`benchmarks/section7-inference-machineB-hybrid-2.6.1.json` and
`benchmarks/section7-inference-machineB-dgpu-2.6.0.json`.

Nothing in 2.9.0 touched the GPU converter or tensor-transfer paths those rows
measure, so expect them to hold. **Expecting is not measuring** — if they hold,
say so and relabel them 2.9.0; if they move, the README's 1.55× and 1.34×
claims need rewriting, not footnoting.

## Priority 3 — GPU inference avoidance

The CPU answer is known. The GPU question is whether gating still pays when an
inference costs 5–15 ms instead of 75 ms.

```powershell
python .test/change-detection/inference_avoidance.py --monitor "<your monitor>" --arms rapidshot_exact dxcam_copy windows_capture_dxgi --groups none full_compare metadata --detector cuda --hz 20 --duplicates 3 --duration 20 --rounds 3 --out .test/change-detection/results-rtx/gpu-avoidance.json
```

`--detector cuda` reports its measured idle cost in the output, so you will know
the regime before interpreting anything. Predict the result before you read it:
if a CUDA YOLOX-S runs at ~8 ms, a 2 ms full-frame compare is a quarter of the
pipeline and the metadata advantage should *grow* back toward the synthetic
figures, not shrink. If it does not, find out why rather than reporting the
number.

`windows_capture_dxgi` needs the vendored wheel, which is **not** in git
(`.test/change-detection/vendor/`). Either copy that folder from the Intel host
or `pip install windows-capture==2.0.1`; if neither, drop the arm and say the
native control is missing, because without it the comparison cannot distinguish
RapidShot from what Windows exposes to anyone.

## Priority 4 — what the critique asked for and nobody has measured

In rough order of value. None of this exists yet.

- **GPU-resident preprocessing.** Verify whether RapidShot can gate frames
  *before* conversion and GPU upload, rather than assuming it does. If unchanged
  frames can be rejected before preprocessing and transfer, the saving is much
  broader than fewer model calls — and that is the strongest untested claim in
  the whole project. `CudaYoloDetector` binds its input from host memory, so it
  measures inference avoided, **not** transfer avoided. Wiring
  `GpuConverter`/`TensorTransfer` into it is the real work here.
- **GPU resource accounting**: kernel execution time, host/device transfer
  volume, VRAM, utilisation, power and energy. Nsight Systems can measure CUDA
  kernel activity and transfers. Note the Intel host exposed **no** power
  counter (`CallNtPowerInformation` gives clock MHz only), so energy is a new
  capability here, not a repeat.
- **FP16 and TensorRT** arms. Verify detections against FP32 before reporting
  either — INT8 on CPU silently lost a detection, and that is the precedent.
- **Workload variety.** Six of eight behaviours are untested against a detector
  anywhere: static desktop, document editing, burst, scrolling, cursor-only and
  changes outside the model's region of interest.
- **Five rounds.** The Intel headline has three.

## Reporting standard

This project's figures are quotable because they are hedged where they must be.
Keep that.

- State the consumer's spare headroom with any capture-versus-capture claim, or
  the result is about the model.
- Verify detections match a reference whenever precision or the model changes.
- Separate correctness audits from the timed path. `inference_avoidance.py` and
  `inference_live_benchmark.py` already do.
- An incomplete campaign is not a result. One was discarded on Intel for losing
  3 of 20 trials rather than reported with a caveat.
- When a number looks good, find the confound before publishing it. Every
  headline in `inference-2.9.0.md` that survived did so after an attempt to
  break it, and two did not survive.
