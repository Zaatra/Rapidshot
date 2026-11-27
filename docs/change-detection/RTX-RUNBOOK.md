# Runbook for the RTX 4060 host (Machine B)

Written for a Claude session starting cold on the RTX laptop, on branch `dev`.
Everything here is work that **cannot** be done on the Intel host (HP 324pe),
where the rest of the 2.9.0 inference measurements were taken.

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
git fetch origin && git checkout dev
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

`.test/change-detection/*.py` is tracked on this branch **deliberately**, so the
harness reaches this machine; `.test/` is otherwise gitignored, so campaign
outputs stay local. Copy the ones worth keeping into `docs/change-detection/`
as the Intel results were.

## Priority 1 — refresh the README's GPU detector table

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

## Priority 2 — GPU inference avoidance

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

## Priority 3 — what the critique asked for and nobody has measured

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
