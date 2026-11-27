# Reproduction at 2.9.0

The figures in [the main report](report.md) were recorded while the version in
the tree read `2.8.1.dev0`. This page re-runs the live arms on the **same
machine, same monitor and the same fixture seed** after the version bump, so the
published claims are checkable against the code that ships as 2.9.0 rather than
against the code that happened to be in the tree when the campaign ran.

**Nothing here replaces the main report.** Its tables remain the record, because
they carry the frozen protocol, the strata and the raw per-trial data. This page
exists to answer one question: *did anything move?*

## What was under test

| | |
| --- | --- |
| Machine | HP 324pe, Intel integrated graphics, `\\.\DISPLAY2`, 1920×1080 at 100 Hz, primary |
| Fixture | `shared_tiles` / `sparse` at 30 updates/s, seed **412430** — the same seed as the original run, so the offered states are identical |
| Rounds | 5 shuffled trials per arm, 10.2 s each |
| Libraries | dxcam 0.3.0, mss 10.2.0, numpy 2.5.5, opencv-python 5.1.0.95 — unchanged from the original |
| Code under test | the 2.9.0 source tree (`rapidshot.__version__ == "2.9.0"`, imported from the checkout) |

**Two caveats that belong with the numbers, not under them.**

1. **The manifests of both runs record `rapidshot 2.8.1.dev0`.** The harness
   reads `importlib.metadata.version("rapidshot")`, which for an editable
   install is frozen at install time
   (`__editable__.rapidshot-2.8.1.dev0.pth`). The *imported* module is the
   repository source — verified as
   `C:\Users\mzaatra\Projects\Rapidshot\rapidshot\__init__.py` with
   `__version__ == "2.9.0"`, including `changes.py`. So the label in this
   re-run's manifest is stale while the code is current. Refreshing the
   editable install before the release record is finalised would remove the
   discrepancy; it is recorded here rather than quietly corrected.
2. **Two remote-desktop capturers were running during this re-run** —
   TeamViewer and AnyDesk. Prior experience on this machine is that a contended
   run reads 4–10 ms older pixels than a clean one, and pixel age is the metric
   these claims rest on. The harness **does not record** which screen capturers
   were active, in either run, so the original's state is unknown. This matters
   for how the comparison is read: the re-run was measurably contended and
   still matched or beat the published figures, which rules out a regression —
   but it does not establish that both runs ran under equal conditions.

## Matched tight-search arms, published against 2.9.0

All four arms run the same tight OpenCV search. 20 trials completed in
**218.9 s**, against 217.9 s for the original. Zero wrong detections in every
arm of both runs.

| Arm | Correct on time | Pixel age at result p50/p95 ms | CPU % of one core | Mcycles/s | GPU staged MiB per delivery |
| --- | ---: | ---: | ---: | ---: | ---: |
| `rapidshot_tight` — published | 1,517/1,530 | 4.60 / 8.38 | 13.79 | 467.48 | 1.51 |
| `rapidshot_tight` — **2.9.0** | **1,519/1,530** | **4.43 / 7.92** | **13.79** | 462.81 | 1.51 |
| `dxcam_view_tight` — published | 1,517/1,530 | 7.05 / 9.21 | 24.66 | 838.06 | 7.91 |
| `dxcam_view_tight` — **2.9.0** | 1,518/1,530 | 6.96 / 8.74 | 26.65 | 894.17 | 7.91 |
| `dxcam_view_tight_2ms` — published | 1,511/1,530 | 7.86 / 10.57 | 20.83 | 691.00 | 7.91 |
| `dxcam_view_tight_2ms` — **2.9.0** | 1,513/1,530 | 7.74 / 10.00 | 20.68 | 734.91 | 7.91 |
| `dxcam_stream_tight` — published | 1,177/1,530 | 15.41 / 18.13 | 22.21 | 820.23 | 7.91 |
| `dxcam_stream_tight` — **2.9.0** | 1,178/1,530 | 15.73 / 18.42 | 23.90 | 829.80 | 7.94 |

RapidShot's steady-state count was **1,514/1,515** in both runs, identical. Its
requested GPU readback and CPU pixel writes per result were identical to two
decimal places. Its CPU share was identical at 13.79%, on 1.0% fewer cycles per
second.

### The derived claims the README makes, recomputed

| Claim as published | Published run | 2.9.0 re-run | Holds? |
| --- | ---: | ---: | --- |
| Fewer bytes off the GPU per result | 5.24× | 5.24× | identical |
| CPU against the matched DXcam arm | 44.1% less | 48.3% less | holds, wider |
| Pixels fresher than the matched DXcam arm | 2.45 ms | 2.53 ms | holds, wider |
| CPU gap against DXcam's **best** wait setting | 7.04 points, 33.8% less | 6.89 points, 33.3% less | holds |

The README's headline table quotes the five-trial matched campaign (14.09%
against 24.81%, 5.09 ms against 7.30 ms), not the four-arm block above; those
are separate runs of the same fixture and the derived percentages agree to
within a point.

**Where the two runs genuinely differ is on DXcam's side, not RapidShot's.**
`dxcam_view_tight` cost 1.99 more points of a core and 6.7% more cycles per
second in the re-run, while RapidShot's share did not move at all. The polling
arm issues about 38,600 empty `grab_view()` calls per five rounds against
RapidShot's 2,300, so it is the arm with the most to lose from background load,
and two screen capturers were running. That is a plausible mechanism rather
than a measured one: this re-run cannot separate "DXcam absorbs contention
worse" from "this run was noisier for everyone", because RapidShot's own
numbers were flat.

### Resource cross-checks

From the re-run's own resource table, against what the README and roadmap state:

| Quantity | Stated | 2.9.0 re-run |
| --- | ---: | ---: |
| CPU-copied MiB per frame, RapidShot | 0.72 | **0.72** |
| CPU-copied MiB per frame, streamed DXcam owned | 15.82 | 15.85 |
| GPU-staged MiB per frame, RapidShot / DXcam | 1.51 / 7.91 | **1.51 / 7.91** |
| GPU memory peak, local | 17.74 MiB, unchanged either way | 17.75 (RapidShot) / 17.74 (DXcam) |
| Peak tile-pool outstanding / exhaustions | 1 / 0 | **1 / 0** |

One figure moved in RapidShot's favour and is **not** claimed anywhere: 3D
engine busy time was **1.50%** for `rapidshot_tight` against **2.85%** for
`dxcam_view_tight`. Earlier rounds reported these as roughly equal at about
2.6%. One observation is not a claim, and these are process busy counters
rather than timing for specific copy commands, so it is recorded here and left
out of the README.

## Six-arm search controls, published against 2.9.0

30 trials in **329.3 s**, same machine, same seed. Zero wrong detections in
every arm of both runs. "Present to result" is pixel age at the finished result.

| Arm | Correct on time | | Present to result p50 ms | | Search p50 ms | |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| | published | **2.9.0** | published | **2.9.0** | published | **2.9.0** |
| `rapidshot_plain` | 893/1530 | **983/1530** | 17.19 | 16.67 | 13.06 | 13.02 |
| `rapidshot_exact` | 1091/1530 | **1092/1530** | 15.87 | 15.55 | 13.05 | 13.04 |
| `rapidshot_tiled` | 1061/1530 | **1077/1530** | 15.22 | 14.76 | 13.18 | 13.14 |
| `rapidshot_tight` | 1517/1530 | **1518/1530** | 5.09 | 4.56 | 1.99 | 1.98 |
| `rapidshot_exact_tight` | 1504/1530 | **1485/1530** | 6.33 | 5.43 | 1.97 | 1.95 |

Pixel age and search time improved slightly in every single arm. Four of the
five deadline counts went up. **One went down**, and it is worth the space it
takes to explain, because at face value it looks like a regression in
`grab_with_changes(tight_rectangles=True)`.

### The one arm that went down, and what it actually was

`rapidshot_exact_tight` lost 19 states against the published run, and all of
them are classified **undelivered** rather than late or wrong. The published run
had 5 undelivered, all at cold start; this one had 25.

Working back through the raw trials:

1. **Four of the five rounds are identical to the published run** — one
   undelivered state each, at cold start, exactly the published pattern. Every
   extra miss is in **round 4**.
2. **Round 4's misses are contiguous**: submitted indices 162–177, 179, 180.
   About 800 ms of a 10.2 s round, in one block.
3. **The fixture did not stall.** Gaps between its own submitted updates across
   that window are 32.76–34.25 ms against a 33.33 ms nominal, and the largest
   lateness anywhere in the round is 1.34 ms. The source presented all 306
   states on schedule.
4. **Capture did not stall either.** 47 observations returned inside the
   window, with capture latency 2.1–25.7 ms and **pixel age 1.7–7.7 ms** —
   fresher than the arm's own 5.43 ms median. The worker recorded 329
   observations for 306 submitted updates in that round.
5. **What failed was the marker read.** Every observation after the first
   ~66 ms of the window carries `stamp: null`: fresh frames arrived
   continuously, and the harness could not find the fixture's state marker in
   them. Something drew over the marker region for the duration and then
   cleared.

So capture delivered healthy, fresh frames throughout; the *oracle* lost sight
of the fixture. The earlier idea-sweep campaign recorded the same phenomenon in
its own words — "a normal always-on-top overlay appeared and cleared in both
live rounds" — and this re-run had two remote-desktop clients active, both of
which draw toolbars and toasts.

**This is a fixture-visibility artifact, not a capture regression**, and the
direction of the README's claim is unaffected: staging only candidate tiles
still beat reading the full frame, by a wider margin than published
(1,518 against 1,485, where the published run had 1,517 against 1,504).

**It does expose a harness weakness worth fixing.** The classifier calls a
state `undelivered` both when capture returned nothing *and* when capture
returned a frame whose marker could not be read. Those are different failures
with different causes, and collapsing them is what made a transient overlay
look like a capture stall. Until they are separated, an `undelivered` count
from this harness is not by itself evidence about capture.

### Stock DXcam without change detection

`run_tight_controls.py` configures `dxcam_view` as the **stock full-frame
baseline** — its own docstring says so, and the generated page states "Stock
DXcam supplies full frames to the original OpenCV template search." That is the
arm behind the README's *Without change detection* row, and it reproduces:

| Quantity | Published | 2.9.0 re-run |
| --- | ---: | ---: |
| Correct and on time | 0/1,530 | **0/1,530** |
| Completed results | 696 | 708 |
| Search p50 | 68.89 ms | 67.59 ms |
| Pixel age at result p50/p95 | 89.75 / 111.94 ms | 87.19 / 109.78 ms |
| Whole-process CPU | 96.65% | 97.27% |

**Note that `dxcam_view` means two different things in this evidence tree**, and
the difference is configuration, not drift. In
[the matched controls](matched/tight-controls.md) it is DXcam *with*
benchmark-added change detection and a regional search, scoring 939/1,530 at a
13.08 ms search. Here it is stock DXcam with the original full-frame search,
scoring 0/1,530 at a 67.59 ms search. Both are real; neither contradicts the
other; any comparison has to say which one it means.

### Per-arm capture cost, re-measured

| Arm | CPU % of one core | GPU staged MiB per frame |
| --- | ---: | ---: |
| `rapidshot_tight` | **14.25** | **1.51** |
| `rapidshot_exact_tight` | 15.16 | 7.91 |
| `rapidshot_tiled` | 43.81 | 1.51 |
| `rapidshot_exact` | 43.96 | 7.91 |
| `rapidshot_plain` | 47.33 | 7.91 |
| `dxcam_view` (stock, full-frame search) | 97.27 | 7.91 |

`rapidshot_tight` measured 14.25% here, 13.79% in the four-arm block and 14.09%
in the published matched campaign — three runs of the same path within half a
point. The 1.51 against 7.91 MiB staging split between the tiled and
full-frame paths is exact in every run.

## Detector and offline surfaces, re-run at 2.9.0

These are the inference and 4K checks this machine can run. Two rounds, zero
errors. Offline replays, so they report processing time rather than pixel age.

**Detector agreement — does gating on the change map change what a real model
finds?** YOLOX (OpenCV Zoo `object_detection_yolox_2022nov`, run through
`cv2.dnn` on the CPU) on positive COCO photos composited onto a canvas, with
the detector run on *every* frame versus only when the exact change map
reported a change:

| | Published | 2.9.0 re-run |
| --- | ---: | ---: |
| Frames where both arms agreed | 20/20 | **20/20** |
| Positive detections | 13 | **13** (7 in round 1, 6 in round 2) |

`exact_skip` and `every_frame` returned the same positive frames in both
rounds. Skipping the detector when the change map says nothing changed did not
cost a single detection on this fixture.

**Three further inference measurements were run after this page was first
written**, and they have their own page: [Inference, measured at
2.9.0](inference-2.9.0.md). In short — the published frame-gate figure
reproduced at 49.13% against 48.94%; the same detector over three *recorded
application traces* saved detector time equal to the byte-identical frame
fraction to within a twentieth of a percent (69.0% → 69.13%, 33.6% → 33.62%,
13.3% → 13.28%); the skip decision over candidate tiles cost 257× / 25× / 21×
less than a whole-frame compare; and a live five-library comparison through the
same CPU detector put `grab_with_changes()`'s verdict at 0.001 ms against
2.0–2.6 ms, with 608 of its verdicts checked against an independent compare and
none disagreeing.

That page also records what those runs *failed* to establish, which is the more
important half: duplicate frame delivery was never controlled and proved
unstable across runs (214 identical frames of 255, then 3 of 65, then 0 of 61 in
the same configuration), so **inference avoidance was not measured** and the
no-filter baseline saved nothing at all. Read its last section before quoting
any of it.

**4K offline replay.** 3840×2160, 510 tiles, 4–10 tiles changed per frame,
five paired frames per round:

| | Published | 2.9.0 re-run |
| --- | ---: | ---: |
| Full search, result p50 | 277–281 ms | 280.9 / 290.6 ms (round 2 / round 1) |
| Tile search, result p50 | 10.7–11.2 ms | **10.70 / 10.95 ms** |
| Detections and tile indices vs oracle | identical | **identical, both arms, both rounds** |
| History writes per sparse update | ~0.41 MiB | **0.41 MiB mean** (0.25–0.62) |

The tiled figure is dead on. The full-search figure came out 0–10 ms *above*
the published band in round 1, which widens the gap rather than narrowing it,
and is consistent with the background contention noted above. The claim in the
README quotes the published band, which is the conservative one.

**Geometry and generation fixtures.** 6 of 6 cases correct in both rounds
(crop and source resets, generation loss, injected geometry changes).

### What could not be run here, and why

| Benchmark | Status on this machine |
| --- | --- |
| The README's **"Through a detector"** table — YOLO11n through ONNX Runtime's CUDA provider | **Cannot run.** `rapidshot/_bench/ai_pipeline.py` requires CuPy's current CUDA stream *and* a loaded `CUDAExecutionProvider`, and sets `session.disable_cpu_ep_fallback` so that no node can quietly run on the CPU and be counted as GPU inference. It raises `RuntimeError` otherwise. This host has an Intel iGPU and no CUDA; `torch`, `cupy` and `onnxruntime` are all absent from the project venv. Those rows are Machine B (RTX 4060) measurements and stay labelled as such |
| `compare_libraries.py` arms for **BetterCam** and the cost matrix's **windows-capture** arm | Not installed (`bettercam`, `windows-capture` both absent). `dxcam` 0.3.0 and `mss` 10.2.0 are present, which is what every comparison in this document used |
| Any **remote-desktop** comparison | **No such benchmark exists.** See below |

### On benchmarking against RustDesk

There is no RustDesk benchmark in this repository, and there never has been.
RustDesk appears four times in the tree and every one is a prior-art citation
for colour conversion, not a measurement:

- `rapidshot/core/hdr.py` — the HDR-to-SDR conversion "is the one OBS and
  RustDesk ship for HDR-to-SDR display" capture
- `README.md` and `CHANGELOG.md` — the same statement, describing how `grab()`
  normalises an HDR desktop
- `ROADMAP.md` — a note that OBS and RustDesk request
  `[R16G16B16A16_FLOAT, B8G8R8A8_UNORM]` from Desktop Duplication

`benchmarks/compare_libraries.py` supports exactly `rapidshot`,
`rapidshot-poll`, `rapidshot-numpy`, `rapidshot-gpu`, `rapidshot-unpooled`,
`dxcam`, `dxcam-numpy`, `bettercam`, `bettercam-gpu` and `mss` — Python capture
libraries with an importable API that can be driven in-process against a shared
motion source and a common clock.

RustDesk, TeamViewer and AnyDesk are not that. They are end-user applications
that consume Desktop Duplication internally and expose no capture API to
Python, so they cannot be dropped into this harness. "Benchmark against
RustDesk" could mean three different experiments, and they need different
apparatus:

1. **Compare capture latency with RustDesk's own** — requires instrumenting
   RustDesk's capture path or measuring its end-to-end remote latency, neither
   of which the pixel-age clock can reach. This is a research project, not a
   harness arm.
2. **Measure RapidShot while remote-desktop software is running** — contention.
   This is directly measurable with the existing harness and is the one worth
   doing, because every figure in this document was in fact recorded with
   TeamViewer and AnyDesk active and still matched or beat the published runs.
   That is suggestive but uncontrolled: it needs a paired run with the
   capturers closed to turn into a result.
3. **Measure RapidShot capturing a desktop that is being viewed remotely** —
   relevant to real users, and a different question again, since the remote
   viewer changes what the compositor presents.

None of the three is claimed anywhere, and none should be until one is run.

## Contention: does other screen-capture software change the numbers?

Earlier sections of this page flagged that every re-run was recorded with
TeamViewer and AnyDesk active, and that the harness does not record contention,
so the published campaign's conditions were unknown. That is now measured
rather than guessed.

Five complete conditions, same machine, same fixture seed, 20 trials each, all
four arms running the same tight search. The display was held awake for the
last three (see the failure below for why).

| Condition | Capturers | Display held awake | Trials |
| --- | --- | --- | --- |
| `published` | unknown | no | 20/20 |
| `contended-1` | TeamViewer + AnyDesk running | no | 20/20 |
| `clean-aborted` | killed | no | **17/20 — discarded** |
| `clean` | killed | yes | 20/20 |
| `remote-load` | killed, **plus a competing duplication consumer** | yes | 20/20 |
| `contended-2` | TeamViewer restarted | yes | 20/20 |

### RapidShot is unmoved by any of it

`rapidshot_tight` across the five complete conditions:

| Condition | Correct on time | Pixel age p50/p95 ms | CPU % of one core | Mcycles/s |
| --- | ---: | ---: | ---: | ---: |
| `published` | 1,517/1,530 | 4.60 / 8.38 | 13.79 | 467.48 |
| `contended-1` | 1,519/1,530 | 4.43 / 7.92 | 13.79 | 462.81 |
| `clean` | 1,515/1,530 | 4.62 / 8.78 | 14.55 | 488.19 |
| `remote-load` | 1,516/1,530 | 4.77 / 8.95 | 14.25 | 479.06 |
| `contended-2` | 1,517/1,530 | 5.02 / 9.55 | 13.94 | 486.22 |

Total spread: **4 states out of 1,530** (0.26%), **0.59 ms** of median pixel age,
**0.76 points** of one core. The *clean* condition is not the best one on any
column — it is marginally worse on CPU than both contended runs.

**Two conclusions, and one correction.**

- **The published figures were not contaminated.** The question that prompted
  this experiment is answered: contention from remote-desktop software does not
  move RapidShot's numbers on this fixture by more than run-to-run noise.
- **An earlier warning in this project was overstated.** The expectation,
  carried from a different harness, was that a contended run reads 4-10 ms
  older pixels than a clean one. For this workload that is wrong: the worst
  spread across all five conditions is 0.59 ms. The note applies to whatever it
  was originally measured on, not here.
- **Contention should still be recorded** — not because it mattered, but
  because establishing that it did not mattered required re-running everything.

### A competing duplication consumer on the same output

`remote-load` is the one condition that answers the remote-desktop question
directly. A second process (`remote_class_load.py`, using `dxcam` so the
competitor is not a privileged copy of RapidShot) opened Desktop Duplication on
the **same output** and pulled frames from it for the whole run, copying each to
the CPU as an encoder would:

| The competing load | |
| --- | ---: |
| Frames captured | **5,431** |
| Copied to CPU | **42,960 MiB** (42 GiB) |
| Achieved rate | 18.7 fps over 290 s |
| Empty acquisitions | 3,269 |

It ran concurrently with the whole campaign, which **proves two processes can
duplicate one output simultaneously** rather than assuming it.

Against `clean`, that load cost RapidShot **1 state out of 1,530 and 0.15 ms of
median pixel age**, and its CPU share went *down* (14.55% to 14.25%), which is
noise. The same load cost `dxcam_stream_tight` **100 states** (1,380 to 1,280)
and `dxcam_view_tight` 8 states. That asymmetry is worth noting but not
claiming: `dxcam_stream_tight` ranged 1,177-1,380 across the complete
conditions, so it is too variable to carry a conclusion.

### The condition that failed, and what it taught

`clean-aborted` lost 3 of 20 trials. All three were in round 5, all three were
DXcam arms, and all three failed identically at `DuplicateOutput` with
`E_ACCESSDENIED` — the refusal `rapidshot/util/desktop.py` exists to diagnose
correctly (secure desktop, locked workstation or UAC prompt, **not** protected
content). Nothing was locked when checked afterwards, and this host's AC
display timeout is **600 s**.

The mechanism is that the display slept during an unattended run and Desktop
Duplication was refused. The part worth recording: **TeamViewer running had
been keeping the display awake**, so killing it as "contention" is what exposed
the sleep. The software being treated as noise was holding the experiment up.
The three later conditions were run with the display explicitly held awake, and
all three completed 20/20.

An incomplete campaign is not a result, so `clean-aborted` is kept as a record
and excluded from every comparison above.

### The wrong detections in `contended-2`

`contended-2` is the only condition all day with non-zero wrong detections:
**37** for `dxcam_view_tight_2ms` and **1** for `rapidshot_tight`. Both are in
**round 2**, the same wall-clock window, and both are transient:

- DXcam's 37 are stamps 4191-4263 in runs of 15, 1 and 21 consecutive states,
  inside a single trial whose search p50 was a normal 2.00 ms but whose first
  wrong state took **53.44 ms to search** and returned its result 70.08 ms
  after the present. The search stalled, results went stale, and stale results
  match a later state — which the harness scores as wrong.
- RapidShot's 1 is a single state (4313) whose search was normal at 1.33 ms but
  whose capture took 8.59 ms to return against a 2.89 ms median, putting the
  result 10.0 ms after the present. That is the documented mixed-frame case,
  at 1 in 1,530.

One external disturbance in round 2 hit both arms; the arm with a 6.39 ms p95
search and 23.90% CPU lost 37 states, the arm with a 2.85 ms p95 search and
13.94% CPU lost one. That is a plausible headroom story and **one trial**, so it
is recorded, not claimed. TeamViewer had been restarted shortly before this
condition, which is a candidate cause and equally unproven.

### What a literal RustDesk test still needs

RustDesk is **not installed** on this machine, so nothing above is RustDesk
itself; `remote-load` is a RustDesk-*class* load, reproducing the duplication
contention and CPU readback of a 30 fps remote-desktop sender, with no encoder,
no network stack and no cursor handling.

Installing it would not by itself produce a measurement. A remote-desktop tool
captures only while a session is active, so RustDesk sitting installed and idle
adds no duplication load at all. A real comparison needs an **incoming session
from a second device**, with the viewer on the remote side. A loopback session
on this machine would not substitute: the viewer window would redraw the
captured desktop continuously, so the "sparse 30 updates/s" fixture would stop
being sparse and the controlled update rate — the whole basis of these numbers
— would be gone. Placing the viewer on the second monitor avoids changing the
captured output's pixels but still puts the decode and render load on the
machine under test, which a real remote peer would not.

The exact artefacts, if that test is wanted, from the GitHub release API for
tag `1.5.3` (published 2026-11-09):

| Asset | Size | Source |
| --- | ---: | --- |
| `rustdesk-1.5.3-x86_64.exe` | 26,233,712 bytes (25.02 MiB) | `github.com/rustdesk/rustdesk/releases/download/1.5.3/` |
| `rustdesk-1.5.3-x86_64.msi` | 26,734,592 bytes (25.50 MiB) | same |

## Verdict

No performance claim in the README, the roadmap or the changelog needs changing
on this evidence. Pixel age and search time came out equal or better in every
arm measured; the two staging figures and the pool counters are identical to
two decimal places; and the single count that fell has a traced, non-capture
cause. The figures were recorded under at-least-as-contended conditions as the
originals, which strengthens rather than weakens them.

Two things remain genuinely open, and neither is a performance question:

- **`benchmarks/baseline.json` no longer gates on this host.** `--compare auto`
  refuses it because the machine moved from Windows build 10.0.26200 to
  10.0.26300 (and numpy 2.5.3 to 2.5.5), so the synthetic suite can only report
  indicative verdicts here. Re-recording it would restore gating, and would
  overwrite the file the README's performance badges are generated from.
- **The editable install's metadata is stale**, so every manifest this harness
  writes will keep saying `2.8.1.dev0` until it is refreshed, regardless of
  what the source says.
