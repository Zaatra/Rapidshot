# RapidShot exact tile validation

<!-- Published from the run tree described in the note below. Do not edit by hand. -->

> **Published evidence, RapidShot 2.9.0.** This is the readable half of the
> change-detection run tree. The raw per-trial JSON, the memory samples and the
> saved mismatch images are 61 MB and are **not** in the repository; each
> reference to them below names its path inside
> `.test/change-detection/results-tiled-final/` on the machine that recorded
> them. Every figure here is **Intel integrated graphics at 1920x1080 and
> 100 Hz** on an HP 324pe, and the opt-in paths are CPU BGRA8.
>
> These tables were recorded while the tree read `2.8.1.dev0`. The live arms
> were re-run after the bump to 2.9.0 on the same machine and the same fixture
> seed, and came out equal or better in every arm:
> [reproduction at 2.9.0](release-2.9.0-reproduction.md).

The hardware evidence covers the HP 324pe with Intel integrated graphics at 1920 × 1080 and 100 Hz. The opt-in paths use CPU BGRA8. [Original protocol](protocol.md) · [Final extension protocol](last-checks/protocol.md)

## Same tight search at 30 updates/s

Each arm ran five shuffled trials. Both libraries feed the same tight OpenCV search. Stock DXcam supplies frames through `grab_view()`, and an independent full CPU byte scan in the local harness computes its change bounds. CPU includes capture, waiting and search. Pixel age begins at the desktop present. The deadline is measured from submission.

| Pipeline | Correct and on time | Result age p50/p95 | CPU, % of one core | GPU readback per result |
| --- | ---: | ---: | ---: | ---: |
| RapidShot candidate tiles | **1,517/1,530** | **5.09/9.07 ms** | **14.09%** | **1.51 MiB** |
| DXcam view and full CPU scan | 1,514/1,530 | 7.30/10.28 ms | 24.81% | 7.91 MiB |

The original CPU numbers describe the one-shot pipelines, including the DXcam harness's 0.5 ms retry delay. The deadline difference is three updates with the same search. Cold starts are included. RapidShot's tiled arm finished **1,514/1,515** on time after the first three offers of each round. [Same-search trial strata](matched/tight-controls.md) · [CPU, memory and copy data](matched/report.md)

### DXcam waiting and streaming controls

Twenty new trials completed in **217.9 seconds**. The four arms ran the same tight search, five shuffled windows each. Streaming used stock DXcam 0.3.0 at a 100 fps target, with owned frames and timestamped delivery. CPU covers all worker threads, including the streaming producer. The readback column is requested GPU copy payload per delivered state.

| Arm | Correct on time | Result age p50/p95 | CPU, % of one core | Million cycles/s | GPU readback per result |
| --- | ---: | ---: | ---: | ---: | ---: |
| RapidShot candidate tiles | **1,517/1,530** | **4.60/8.38 ms** | **13.79%** | 467.48 | **1.51 MiB** |
| DXcam one-shot, 0.5 ms retry | 1,517/1,530 | 7.05/9.21 ms | 24.66% | 838.06 | 7.91 MiB |
| DXcam one-shot, 2 ms retry | 1,511/1,530 | 7.86/10.57 ms | 20.83% | 691.00 | 7.91 MiB |
| DXcam streaming, blocking consumer | 1,177/1,530 | 15.41/18.13 ms | 22.21% | 820.23 | 7.91 MiB |

All four arms had zero wrong detections. A longer one-shot retry lowered DXcam CPU by **3.83 percentage points**. Empty calls fell from **38,660 to 16,584** across five rounds, and the average measured pause rose from **0.95 to 2.32 ms**. That explains part of the original CPU gap under this workload. The 100 fps streaming setup used **2.45 points** less CPU than the shorter-retry one-shot arm and finished fewer results by the deadline. Producer pacing, ownership, and frame selection all change in streaming, so the observed streaming difference includes those effects. Against the streaming configuration tested here, tiled RapidShot used **8.42 points less CPU**. Its median logical CPU pixel writes were **0.72 MiB per result**, compared with **15.82 MiB** for streamed DXcam, including the owned app return. [Wait configuration, deadlines, and raw calls](last-checks/streaming/streaming-controls.md) · [Resource and memory tables](last-checks/streaming/report.md)

## Public API soak

One capture process ran for **3,600.001 seconds**. It exercised `grab_with_changes(tight_rectangles=True)` for 1,950 seconds and `grab_tiles(tight_rectangles=True, audit_every=60)` for 1,650 seconds. It checked every reported change map against an independent comparison of consecutive delivered BGRA arrays. Scheduled full dirty-hint audits, whole-fixture matches and offered state coverage are recorded separately. The map oracle uses delivered pixels, and a captured desktop image can contain parts of different presents.

| API | Offered fixture states | Delivered frames and matching maps | Retained lease checks |
| --- | ---: | ---: | ---: |
| `grab_with_changes` | 18,755 | 16,296 | 561 |
| `grab_tiles` | 15,804 | 13,689 | 472 |

The run passed with **29,985 map matches**, **1,033 retained lease checks**, and **zero missed tiles in 187 full audits**. Two capture-resource generations rebuilt, with zero process restarts. The longest loop gap was 2.02 seconds. The full regression suite separately passed **2,477 tests**, with 27 skips. Soak outcomes (`long-soak/soak.json`, raw data, not published) · Regression log (`pytest.log`, raw data, not published)

### Memory trend and leases

The external sampler recorded 3,576 Windows memory observations. Five-minute blocks start after a five-minute warmup. The worker held the benchmark fixture, one history for each API, and its verification buffers, so absolute process memory includes those allocations.

| Measure | First post-warm-up median | Highest block median | Final block median | Windows lifetime peak |
| --- | ---: | ---: | ---: | ---: |
| Private commit | 705.21 MiB | 712.01 MiB | 708.13 MiB | 787.08 MiB |
| Working set | 308.40 MiB | 311.08 MiB | 197.65 MiB | 389.84 MiB |

The median pairwise slope of the eleven private-memory blocks was **+8.48 MiB/hour**. The last block was **2.92 MiB** above the first and below the previous four blocks. Working set had a **−149.65 MiB/hour** slope dominated by a roughly 115 MiB step around the midpoint. Private memory drift warrants disclosure, and this trace does not identify its source. The retained histories ended at a fixed **7.91 MiB each**, including 135 unique pages in the tiled history. Holding an earlier tiled snapshot added at most **19 pages and 1.12 MiB**. Ending the run with all deliberate leases released returned the tiled footprint to the 135 history pages. Run manifest and slope calculations (`long-soak/soak-run.json`, raw data, not published) · External samples (`long-soak/memory.ndjson`, raw data, not published)

## Prior deadline results

The previous tight path's five misses all fell within the first three offered states of its two rounds. Two came from initial full searches that took 68–71 ms, two markers were absent immediately afterward, and one 35.42 ms completion spent most of its time before capture returned. [Per-miss timing and the candidate-without-tight comparison](prior-forensics.md)

## Without change detection

Stock DXcam supplied full images through `grab_view()` to the original full-frame OpenCV template search. That baseline searches every delivered image and maintains zero change-history bytes. RapidShot used its own exact change detection and tight regional search. Each row represents five 10.2-second trials of the sparse 30 updates/s fixture on the HP 324pe. The rows come from separate campaigns. The large deadline gap is driven primarily by the full-frame search, which alone takes about 69 ms against a 33.33 ms budget.

| Pipeline | Correct and on time | Completed results | Search p50 | Pixel age at result p50/p95 | Whole-process CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| RapidShot candidate tiles and tight search | **1,517/1,530** | 1,525 | **1.99 ms** | **5.09/9.07 ms** | 14.09% |
| Stock DXcam and original full-frame OpenCV search | **0/1,530** | 696 | **68.89 ms** | **89.75/111.94 ms** | 96.65% |

Both rows had zero wrong detections. The 33.33 ms deadline starts at submission. Pixel age starts at DXGI present. Latency values pool eligible observations across the five rounds. CPU is the median share of one core across rounds. Stock DXcam trial records (`../results-cost-matrix/measurements/trials`, raw data, not published) · RapidShot trial records (`matched/trials`, raw data, not published)

### RapidShot capture and search controls

With tight search on both RapidShot paths, candidate tiles finished **1,517/1,530** updates on time, compared with **1,504/1,530** for exact full frame. Candidate tiles lowered median result age from **6.33 to 5.09 ms** and GPU copy payload from **7.91 to 1.51 MiB** per result. With the previous regional search on both paths, candidate tiles finished **1,061/1,530** on time and exact full frame **1,091/1,530**. [Search controls](matched/tight-controls.md) · [Raw resource tables](matched/report.md)

## Original confirmed-quiet cycle gate

The study completed all **24 planned blocks**. It retained the first 13 measured blocks across an overnight interruption, then filled the 11 unmeasured blocks after DXGI reopened. The interrupted controller's elapsed time includes the overnight pause. Its failed DXGI attempts are preserved separately. Both resumed quiet batches completed in under five minutes each. The gate includes only pairs with seeded, healthy workers and zero desktop delivery or present during their windows. Measured noisy windows stayed in the raw records and were excluded by eligibility. Interrupted attempt (`quiet/campaign.json`, raw data, not published) · Resumption ledger (`quiet-resumed/resumption.json`, raw data, not published) · Resumed batch timing (`quiet-resumed/campaign.json`, raw data, not published)

| Variant versus plain RapidShot | Eligible pairs | Median change in cycles/s | 95% median confidence interval | Gate |
| --- | ---: | ---: | ---: | --- |
| Full-frame exact | 19 | +0.62% | −2.16% to +5.77% | Inconclusive |
| Candidate tiles | 20 | −0.95% | −3.15% to +5.07% | Inconclusive |
| Candidate tiles with tight bounds | 18 | −1.90% | −3.97% to +2.06% | Pass |

The original two open intervals crossed the predeclared 5% ceiling. The intervals are distribution-free and use exact binomial order statistics with at least 95% coverage. Original cycle observations, interval ranks, and exclusions (`quiet-resumed/quiet-cycle-gate.json`, raw data, not published)

## Final quiet-cycle verdict

All 24 new blocks completed in about **13 minutes 22 seconds**, with four batches of about 3 minutes 20 seconds each. The earlier batches were preserved. The new study produced 19 eligible new pairs per open path. Scoring took place after all planned blocks finished.

| Variant | Fresh pairs and median | Fresh 95% interval | Pooled pairs and median | Pooled 95% interval | Frozen gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Full-frame exact | 19, +1.23% | −1.45% to +6.19% | **38, +1.15%** | **−0.82% to +2.84%** | **Pass** |
| Candidate tiles | 19, −2.32% | −3.80% to +2.36% | **39, −1.40%** | **−3.03% to +1.13%** | **Pass** |

The prior pass for tight candidate tiles stands at an upper bound of **+2.06%**. The two pooled upper bounds fall below the unchanged **+5%** ceiling. Each displayed interval is a fixed-look interval. The two-look directional false-pass convention is set out in the extension protocol. CPU percent from Windows process-time ticks and process cycles per second remain separate counters. All three scopes, eligible pairs, and gate decisions (`last-checks/quiet/quiet-cycle-extension.json`, raw data, not published) · Batch timings and archive hashes (`last-checks/quiet/campaign.json`, raw data, not published)

### What each path does on a timeout

All six instrumented windows qualified at a fixed 5 ms wait. The two exact arms had seeded histories before measurement. Across **3,818 confirmed `AcquireNextFrame` timeouts**, the probe observed **zero GPU copy calls, zero maps, zero processor calls, and zero history-update calls or bytes written**. The observed operations differ in pool and lease bookkeeping.

| Path | Confirmed waits across two windows | Staging pool checkouts / returns | Tile-slot acquires / releases |
| --- | ---: | ---: | ---: |
| Plain | 1,273 | 1,273/1,273 | 0/0 |
| Full-frame exact | 1,272 | 1,272/1,272 | 0/0 |
| Candidate tiles | 1,273 | 0/0 | 1,273/1,273 |

The probe took 66.9 seconds. Its instrumented windows are separate from the cycle gate. DXGI outcomes, operation counts, and raw observations (`last-checks/mechanism/idle-path-probe.json`, raw data, not published)

## Adaptive wait and shutdown

All 10 trials per arm qualified. The adaptive trials required a quiet 5 ms to 50 ms transition, a real desktop update returning the controller to 5 ms, and a second quiet transition before stop. Stop latency includes camera release. The following numbers use ten samples per arm, so the p95 in this table equals the maximum.

| Wait arm | First resumed desktop update to capture p50/p95 | Stop through release p50/p95/max |
| --- | ---: | ---: |
| Fixed 5 ms | 2.52/5.12 ms | 7.22/15.30/15.30 ms |
| Adaptive 5–50 ms | 2.40/4.33 ms | 23.60/62.12/62.12 ms |
| Fixed 50 ms | 2.22/2.59 ms | 39.41/62.54/62.54 ms |

The adaptive probe completed in 58.6 seconds. This measures the experimental wait controller used by the benchmark. Wake transitions and shutdown samples (`quiet-resumed/adaptive-stop-resume.json`, raw data, not published)

The final checks took about **18 minutes 6 seconds** across separate bounded runs. The result closes the outstanding idle-cycle gate on this system. The API decision guide is in [the exact-tiles contract](../EXACT_TILES.md#choosing-a-capture-path).
