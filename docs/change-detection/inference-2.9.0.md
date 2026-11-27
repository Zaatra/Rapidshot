# Inference, measured at 2.9.0

Four inference measurements were run on the release candidate, all on one host
(HP 324pe, Intel UHD, `\\.\DISPLAY2` 1920×1080 at 100 Hz, Windows 10.0.26300).
Two reproduce published figures. Two are new.

Read the last section before quoting any of it. One of these measurements
established that a headline it was designed to produce **cannot be produced on
this fixture**, and that matters more than the numbers that came out clean.

| Measurement | Status |
| --- | --- |
| YOLOX frame gate, synthetic canvases | Reproduced, slightly better |
| YOLOX over recorded application traces | **New** — never run at any version |
| Cost of the skip decision itself | **New** |
| Live capture libraries through a CPU detector | **New** |
| Inference avoided, duplicates controlled, native DXGI control | **New** |
| YOLO11n through ONNX Runtime CUDA | **Not run.** Needs the RTX 4060 host |

Every arm everywhere below fed the same verified model: OpenCV Zoo YOLOX-S
FP32, 35,858,002 bytes, SHA-256
`c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`, through
OpenCV 5.1.0's CPU graph engine, one thread, OpenCL off, 640×640 letterboxed
input, confidence 0.50, NMS IoU 0.50.

## 1. The published frame gate reproduced

Eight in-memory 1080p BGRA frames where every second frame is pixel-identical
to the one before it with separate backing memory, five shuffled rounds.

| | Published | 2.9.0 re-run |
| --- | ---: | ---: |
| Always infer, mean per frame | 275.90 ms | **256.87 ms** |
| Always infer, p50 | 264.39 ms | 256.29 ms |
| Always infer, p95 | 336.64 ms | **262.97 ms** |
| Always infer, total | 11.036 s | 10.275 s |
| Gated, mean per frame | 140.87 ms | **130.66 ms** |
| Gated, p50 | 128.30 ms | 128.36 ms |
| Gated, p95 | 327.98 ms | **266.19 ms** |
| Gated, total | 5.635 s | 5.226 s |
| Reduction in mean | 48.94% | **49.13%** |
| Skipped frames, p50 / p95 | 2.87 / 3.66 ms | 2.39 / 2.81 ms |
| Inferred frames, p50 / p95 | 266.56 / 334.54 ms | 257.58 / 266.36 ms |

Reproduced, and faster on every column. The p95 improvement is large because
this host was quieter than at publication — the display was held awake and
the campaign was not competing with an unmeasured background load.

The script enforces its own correctness and raises rather than report: gated
detections must match always-infer on class, boxes within 0.01 px and score
within 0.00001; the gate must rerun on exactly the changed frames; the dog must
be found in all three positive states; the negative must return nothing; and
the boxes must move when the photo moves. All five rounds passed.

Raw paired timings (`.test/change-detection/results-release-2.9.0/inference/yolox-frame-gate.json`, raw data, not published) · benchmark source `whole_frame_task.py`

## 2. New: the same detector over recorded application traces

The published figure above uses synthetic canvases built from two public
photographs. This runs the same detector over **real recorded desktop capture**
— three application traces recorded earlier on this host at 100 Hz, replayed
frame by frame. Decompression sits outside every timer.

Three arms per trace: infer on every delivered frame; skip frames a full-frame
byte compare proves identical; skip frames proven identical by comparing only
the candidate tiles the recorded DXGI dirty rectangles point at.

| Trace | Size | Frames | Byte-identical | Saved, full compare | Saved, candidate tiles |
| --- | --- | ---: | ---: | ---: | ---: |
| Spreadsheet | 1675 × 715 | 172 | 69.0% | 68.40% | **69.13%** |
| Browser | 1844 × 978 | 299 | 33.6% | 32.91% | **33.62%** |
| Game | 1844 × 978 | 294 | 13.3% | 13.46% | **13.28%** |

**The saving equals the fraction of byte-identical frames, to within a
twentieth of a percent in all three cases.** That is the correct theoretical
result — if the detector is the only cost that matters, skipping exactly the
identical frames saves exactly their share — confirmed on real pixels across a
5× range of content stillness.

It is also the honest limit of the claim. **The saving is a property of how
still the content is, not a property of the capture library.** Any caller
willing to pay a full-frame compare gets within 0.7 points of the same number.
What the library changes is what that decision costs, which is section 3.

Detector calls: 172→54, 299→199, 294→255. Every gated arm returned detections
identical to the always-infer reference on every frame, by the same tolerances
as section 1, and the metadata arm reached the same verdict as the full compare
on every frame of all three traces. The spreadsheet trace took three full-grid
fallbacks on `accumulated frames`; the metadata arm inferred on those rather
than trust a narrowed candidate set, which is the conservative direction.

[Report](inference/yolox-trace-replay.md) · raw data (`.test/change-detection/results-release-2.9.0/inference/yolox-trace-replay.json`, not published) · source `inference_trace_task.py`

## 3. New: what the skip decision costs

Every consecutive frame pair of the same three traces, fastest of five repeats
per pair, no inference. Candidate derivation from the recorded metadata is
inside the timer, matching `real_trace.py`.

| Trace | Whole frame, NumPy | Candidate tiles, NumPy | Candidate tiles, native |
| --- | ---: | ---: | ---: |
| Spreadsheet | 1.2340 ms | **0.0048 ms** | 0.0092 ms |
| Browser | 1.6982 ms | **0.0676 ms** | 0.2787 ms |
| Game | 1.7216 ms | **0.0822 ms** | 0.0642 ms |

As a pure gate, candidate tiles are **257× / 25× / 21× cheaper** than comparing
the whole frame. As a gate that also returns the changed-tile list, **134× / 6×
/ 27×**.

**The three are not doing equal work, and the native row being slower on two
traces is not a regression.** `np.array_equal` answers only *did anything
change* and has no early exit — NumPy compares every byte. The NumPy candidate
loop answers the same question and returns at the first tile that differs, so
on a busy frame it stops almost immediately. The native comparator answers the
harder question — *exactly which tiles changed* — which cannot exit early,
because a caller that wants to search or stage only changed tiles needs the
whole list. Compare the NumPy candidate column against the whole frame for the
cost of a gate, and the native column for the cost of a gate that also tells
you where to look.

On the browser trace the dirty rectangles covered all 120 tiles, so there was
no narrowing at all and the entire win is early exit. On the game trace they
narrowed to 18 tiles of 120 at the median, and both mechanisms contribute.

Every changed-tile set matched an independent full pixel scan. Neither
candidate arm ever called a changed pair identical, on any of 762 pairs.

[Report](inference/gate-cost.md)
· source `inference_gate_cost.py`

## 4. New: live capture libraries through one CPU detector

One worker **process** per arm, so CPU and memory are the consumer alone and
painting the fixture is charged to nobody. The paced dashboard fixture repaints
the target monitor; each worker opens its library, pulls frames for the window,
decides whether the detector must run, and runs it when it must. Every arm
feeds the detector the same thing: a whole BGRA frame.

At 2 updates a second — the only rate where a 255 ms detector is not saturated,
and therefore the only rate that discriminates between libraries:

| Arm | Through model /s | Capture age p50 | Decision p50 | Mcycles/s | CPU % | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.05 | **2.20 ms** | **0.001 ms** | **1932** | **59.3** | 216 MiB |
| RapidShot `grab()` | 2.05 | 2.13 ms | 2.034 ms | **1815** | **53.4** | 211 MiB |
| DXcam `grab()` | 2.05 | 4.02 ms | 2.343 ms | 3348 | 99.4 | 197 MiB |
| DXcam `grab_view()` | 2.05 | 2.28 ms | 2.033 ms | 3353 | 98.8 | 198 MiB |
| MSS (GDI) | 2.00 | — | 2.567 ms | 2481 | 73.2 | **179 MiB** |

Three results hold up, and one apparent result does not.

**The free gate is free, and it is right.** `grab_with_changes()` reaches its
verdict in 0.001 ms because capture already paid for it, against 2.0–2.6 ms of
full-frame compare for every other arm. Across the whole session, **608 of its
verdicts were checked against an independent full-frame compare run outside the
timed path, and none disagreed.**

**Once the detector saturates, the library stops mattering.** At 5, 10 and 30
updates a second every arm converges on 3.65–3.95 frames through the model per
second and 94–100% of a core. The spread is inside run-to-run noise. A 255 ms
single-thread model on this CPU is the bottleneck, and no capture library can
move it.

**RapidShot costs more memory.** 211–217 MiB against DXcam's 197–198 MiB and
MSS's 179 MiB — 15–20 MiB over DXcam for the private history and buffer pool,
and 37 MiB over MSS. That is the price of the free gate, and MSS is the cheapest
arm here on memory while being the only one with no present timestamp at all and
the lowest throughput at every rate.

**The CPU gap is real but it is not the change detection.** RapidShot's arms ran
at 1761–1933 Mcycles/s where DXcam's ran at 3282–3353, and a no-filter control
settles the cause: with gating switched off entirely and identical detector work
(61 calls each), RapidShot still ran at 1761 and 1755 Mcycles/s against DXcam's
3347 and 3351. The difference is **empty reads — 899 and 915 against 333,403 and
340,814.** DXcam 0.3.0's `grab()` returns immediately when no new frame is ready
and gives the caller no way to wait on one, so a polling consumer spins; the
RapidShot calls wait. Attributing that to change detection would be wrong, and a
DXcam caller can close much of it by adding a retry delay — which is exactly why
this project's own matched arms carry explicit 0.5 ms and 2 ms retry budgets. The
cost of the delay is latency.

One asymmetry worth recording: at 2 Hz, `dxcam_view()` delivered **127 frames
where `dxcam_copy()` delivered 61**, 66 of them byte-identical duplicates its
caller had to detect by hand.

[Rate sweep](inference/rate-sweep.md)
· [2 Hz confirmation, two rounds](inference/two-hz-confirmation.md)
· [no-filter control](inference/no-filter-control.md)
· source `inference_live_benchmark.py`

## 5. New: how much inference is avoided, with duplicates controlled

Section 6 explains why the measurements above could not answer this. The short
version: Desktop Duplication delivers on present, so it rarely hands a consumer
a byte-identical frame, and with nothing duplicated there is nothing to avoid.

So this fixture **re-presents the same state deliberately** — each tick either
advances the dashboard or repaints the current state identically, three repaints
per new state, which puts presents on the wire whose pixels match the last byte
for byte. 20 updates a second, 20 s per arm, **401 frames delivered of which 300
are duplicates**, and a marker in each state says which state a captured frame
holds, so an arm that skips work is still scored on whether it saw every state.

There is a trap worth recording for anyone rebuilding this.
`ScenePlayer.present()` computes its own dirty regions and `repaint()` returns
without painting when nothing moved, so re-showing a state through the normal
path puts *nothing* on the wire. The duplicate has to be forced by presenting
the already-stamped cached raster with no region hint.

Three groups, which is the comparison sections 1–4 were missing:

| Group | Policy |
| --- | --- |
| **A — no filtering** | Infer on every delivered frame |
| **B — full compare** | Compare the whole frame against a retained copy |
| **C — metadata** | Take the byte-exact verdict capture already computed |

### CPU seconds for the whole 20 s workload, at a 15 ms detector

Three rounds, every arm in its own worker process. The detector is calibrated
real arithmetic, not a sleep, with every numeric runtime pinned to one thread.

| Library | A: no filter | B: full compare | C: metadata |
| --- | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 7.38 / 7.28 / 7.23 | 4.05 / 3.59 / 3.70 | **2.75 / 2.41 / 2.67** |
| `windows-capture` 2.0.1 native DXGI | 6.88 / 5.42 / 5.38 | 3.03 / 3.42 / 2.91 | none exposed |
| DXcam `grab()` | 18.77 / 18.94 / 18.58 | 19.00 / 18.88 / 18.58 | none exposed |
| MSS (GDI) | 11.12 / 11.30 / 11.45 | 10.16 / 9.12 / 9.59 | none exposed |

**Inference eliminated: 401 calls to 101, every round, every arm — 74.8%, with
300 redundant inferences avoided and 0 of 100 states missed anywhere.** The call
counts were identical in all three rounds.

- **Gating cuts RapidShot's consumer CPU by 64%** at the median, 7.28 s to
  2.67 s.
- **The metadata verdict beats a full compare by 28%** at the median, 3.70 s to
  2.67 s. The ranges do not overlap.
- **DXcam's CPU barely moves when gating is added** (18.94 s to 18.88 s) because
  its spin-polling dominates everything else the consumer does. See section 4.

Across detector costs, the metadata advantage over a full compare shrinks as the
detector grows, because the decision is a fixed cost against a rising one:
**41% at 5 ms, 28–39% at 15 ms, 20% at 50 ms, nothing at 250 ms.**

### The native DXGI control, which is what constrains the claim

`windows-capture` 2.0.1 is a native Rust Desktop Duplication binding whose
Python wrapper exposes `acquire_frame`, `monitor_index`, `recreate` and
`switch_monitor` — **no dirty-rectangle access at all.** It is therefore the
right control for the only question that really matters here: is RapidShot's
advantage its implementation, or is it just exposing what Windows already offers
anyone who asks?

With a full compare it reaches **2.91–3.42 s** against RapidShot's metadata at
**2.41–2.75 s**. The ranges do not overlap, so the advantage is real, but it is
**12% at the median — not a different league.** The honest reading: the
dirty-rectangle *information* is Windows', not RapidShot's. What RapidShot adds
is verifying that hint byte for byte and doing it inside capture, and against a
good native baseline that is worth about an eighth of consumer CPU. The 257× and
2,000× figures from sections 3 and 4 are real measurements of the decision in
isolation, and they do **not** survive contact with a whole pipeline.

### What a slow detector does to all of it

At a 250 ms detector every arm collapses to 71–80 calls, ~19.5 CPU s, and **21
to 30 of 100 states never reach the detector at all.** Gating buys nothing,
because a consumer that slow finds every frame it collects is already new —
there are no duplicates left to skip. Any claim about this feature has to state
the consumer's spare headroom.

One exception is worth keeping: **MSS with a gate caught all 100 states where
MSS without one missed 28.** Polling GDI always returns pixels, so the gate
converts wasted duplicate inferences into coverage of real states. On a polling
source, gating buys correctness rather than just CPU.

[Three rounds at 15 ms](inference/avoidance-15ms-three-rounds.md)
· [5 ms](inference/avoidance-5ms.md)
· [50 ms](inference/avoidance-50ms.md)
· [250 ms](inference/avoidance-250ms.md)
· source `inference_avoidance.py`

### The same measurement with a real model, which changes one conclusion

The figures above use a calibrated synthetic workload. Repeating them with the
**verified YOLOX-S model through ONNX Runtime 1.32.0 on the CPU**, using
`intra_op_num_threads` as the cost dial — 210.1 ms at one thread, 117.1 ms at
two, 75.5 ms at four — with the same forced duplicates and the same four
libraries:

| Detector | Arm | No filter | Full compare | Metadata | Inference eliminated |
| --- | --- | ---: | ---: | ---: | ---: |
| 75.5 ms | RapidShot `grab_with_changes()` | 67.47 s | 28.70 s | **28.22 s** | 253 → 101 (60%) |
| 75.5 ms | native DXGI | 62.72 s | 29.81 s | — | 213 → 100 (53%) |
| 75.5 ms | DXcam `grab()` | 68.44 s | 39.25 s | — | 229 → 101 (56%) |
| 75.5 ms | MSS (GDI) | 56.72 s | 33.88 s | — | 205 → 100 (51%) |
| 117.1 ms | RapidShot `grab_with_changes()` | 37.47 s | 24.55 s | **24.20 s** | 157 → 101 (36%) |
| 117.1 ms | native DXGI | 35.95 s | 24.94 s | — | 157 → 100 (36%) |
| 210.1 ms | RapidShot `grab_with_changes()` | 19.48 s | 19.77 s | 19.77 s | none, saturated |

**Gating still pays, and pays more than before: 58% of consumer CPU at 75.5 ms,
35% at 117.1 ms**, with no state missed at either. At 210.1 ms the detector
saturates, every arm misses 7–14 of 100 states, and gating buys nothing — the
same wall section 5's 250 ms synthetic run hit.

**But the choice of gating mechanism nearly stops mattering.** Against a full
compare on the same library the metadata verdict is worth **1.7% at 75.5 ms and
1.4% at 117.1 ms** — not the 28% the 15 ms synthetic detector showed. Against
the native DXGI binding it is **5.3%** and **3.0%**, not 12%. The reason is
arithmetic: a 2 ms compare is a quarter of a 15 ms detector and under 3% of a
75 ms one. **The synthetic figures in section 5 are correct for a cheap
consumer and misleading for a real model, and the honest summary is that the
decision mechanism matters in inverse proportion to what the detector costs.**

What RapidShot still wins on at 75.5 ms is the rest of the capture path:
**28.22 s against DXcam's 39.25 s (28% less) and MSS's 33.88 s (17% less)**,
and detections 77.9 ms old at the median against DXcam's 85.1 ms.

**A confound worth recording, because it inverted the result.** ONNX Runtime's
intra-op thread pool **spin-waits between runs by default**, burning cycles on
every thread whether or not an inference is in flight. With spinning left on,
the first run of this measurement showed gating saving 14% of CPU; with
`session.intra_op.allow_spinning` set to `0`, the same configuration showed
59%. The idle pool was masking almost the entire benefit. Any CPU-seconds claim
about an ORT pipeline has to state its spinning setting.

[1 thread](inference/avoidance-real-model-t1.md)
· [2 threads](inference/avoidance-real-model-t2.md)
· [4 threads](inference/avoidance-real-model-t4.md)

### INT8: faster, smaller, and not detection-equivalent

Dynamic INT8 quantization of the same model, measured rather than assumed,
because dynamic quantization targets MatMul and Attention while YOLOX-S is
convolution-heavy:

| | FP32 | INT8 dynamic | INT8, pre-processed |
| --- | ---: | ---: | ---: |
| Model size | 35,858,002 B | 9,082,271 B | 9,084,503 B |
| 1 intra-op thread | 232.2 ms | **121.4 ms (1.91×)** | 119.5 ms (1.94×) |
| 4 intra-op threads | 68.2 ms | 62.3 ms (1.09×) | 70.7 ms (0.97×) |

**The speed-up is real at one thread and gone at four.** Dynamic quantization
does not touch convolution, so once the convolutions are parallelized there is
little left for it to win, and the pre-processed variant came out *slower* than
FP32 at four threads.

**It also changes what the model finds, so it is not a drop-in.** On the same
public fixtures the frame gate uses:

| Fixture | FP32 | INT8 | Verdict |
| --- | --- | --- | --- |
| Dog photo | 4 detections, classes 1, 2, 7, 16 | 4 detections, same classes | boxes and scores outside the 0.01 px / 0.00001 tolerance |
| Wall photo (negative) | 0 | 0 | identical |
| Dog moved 120 px | 4 detections, classes 1, 2, 7, 16 | **3 detections, classes 1, 2, 16** | **lost class 7** |

ONNX Runtime warns that quantizing without shape inference and graph cleanup
costs accuracy, so that was done too (`quant_pre_process`, with symbolic shape
inference skipped — this model's shapes are fully static, `[1, 3, 640, 640]` in
and `[1, 8400, 85]` out, so there is nothing symbolic to resolve). **It changed
the accuracy outcome not at all**: the same lost detection on the same fixture.

So INT8 is **not** used as an arm in the avoidance measurements above. Using it
would mean comparing capture libraries through a detector that silently finds
less, and the gate's own correctness claim — gated against ungated with the
*same* detector — is unaffected either way. Recovering the lost detection would
need static quantization against a calibration dataset, which is out of scope
here.

[Raw data](inference/int8-quantization.md) · source `quantize_yolox.py`

## 6. What this does not establish

**Sections 1–4 could not measure inference avoidance at all**, and section 5
exists because of it. The reason is worth stating plainly.

**Duplicate frame delivery was not controlled, and it is not stable.** At 2
updates a second, the three gated runs saw **214 identical frames of 255
delivered, then 3 of 65, then 0 of 61.** The first run's 214 skips came from
incidental screen activity, not from the fixture. A benefit that swings between
84% and 0% of delivered frames across three runs of the same configuration is
not a measurement, and the 214 figure must not be quoted.

**The no-filter baseline saved nothing, for a structural reason.** With gating
off, the detector ran 61 times; with gating on, 61 times. Desktop Duplication
only delivers on present, so it rarely hands a consumer a duplicate in the first
place — the thing the gate exists to catch mostly never arrives. The gate's
value is therefore concentrated in the cases DDA *does* deliver without a pixel
change (cursor-only presents, presents with no visible delta, `accumulated
frames` bursts), and this fixture produces too few of them to measure. MSS is
the exception that proves the mechanism: polling GDI returned 497 and 674 frames
where DDA delivered 61, and 457 and 613 of those were identical.

Both of those were then fixed — a fixture that deliberately re-presents
identical images, and a tunable detector cost — which is what section 5 reports.
The figures in sections 1–4 stand as measurements of the decision and of
capture; they are not measurements of inference avoided.

**Section 5's detector is synthetic.** It is calibrated real arithmetic rather
than a sleep, so it consumes the cycles an inference would, but it is not a
model: no convolution, no memory access pattern resembling a real network, no
cache behaviour to match. It exists to make inference cost a dial. The YOLOX
numbers in sections 1–3 are the real-model ones, at one cost.

**One bug in that detector was mine and is worth recording.** The first version
let NumPy's BLAS multithread, so it burned 3.9 cores, its calibration drifted
under load, and it reported 97 CPU seconds inside a 25 second window. Every
numeric runtime is now pinned to one thread
(`OMP`/`OPENBLAS`/`MKL`/`NUMEXPR`/`VECLIB`), and the unpinned results are
superseded, not published.

**Three rounds, not five.** Section 5's headline configuration ran three times
and its CPU ranges are non-overlapping between groups, which is enough to rank
them but not to put a confidence interval on the gap. Every other configuration
ran once.

**No alternative real model was tried.** YOLO11n, MobileNetV3, INT8 quantization
and multi-threaded inference all need `onnxruntime`, which is not installed on
this host, so the only real model measured was YOLOX-S FP32 on one thread
through OpenCV.

**No GPU inference was measured.** `rapidshot/_bench/ai_pipeline.py` requires
`CUDAExecutionProvider` first in the provider list and sets
`session.disable_cpu_ep_fallback` so no node can quietly run on the CPU and be
counted as GPU work; it raises instead of degrading. This host has an Intel iGPU
and no `onnxruntime`, `cupy` or `torch` at all. The README's **"Through a
detector"** table remains a 2.6.0/2.6.1 measurement recorded 2026-09-25 on the
RTX 4060 host, and nothing here refreshes it.

**Nothing here measures energy, VRAM, GPU kernel time or transfer volume**, and
no INT8, FP16, multi-threaded or lightweight-model configuration was tried. The
detector was one model at one precision on one thread.

**One host, one display, one seed.** Intel UHD only. No non-Intel GPU, no native
4K panel, no multi-adapter case. `bettercam` is not installed, so it never ran;
`windows-capture` 2.0.1 is vendored in the harness and did run, as section 5's
native DXGI control — which is the arm that keeps the whole comparison honest,
and the reason section 5 reports a 12% median advantage rather than the 257× and
2,000× the isolated decision microbenchmarks show.
