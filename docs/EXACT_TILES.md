# Exact changed tiles

**Status: provisional, shipped in RapidShot 2.9.0.** The contract below is
what the code enforces today, and it is tested. It is marked provisional
because every measurement behind it comes from one Intel integrated GPU at
1920x1080 and 100 Hz: no NVIDIA, no AMD, no native 4K, no HDR. The shape of the
API may still change in response to the first reports from other hardware.
Measured behaviour and the release gates are in
[the published benchmark report](change-detection/report.md); the case for the
feature, the crossover where it stops paying and the cost of running it are
summarised in the [README](../README.md#only-what-changed).

`BufferReleasedError` and `PoolExhaustedError` are importable from `rapidshot`
directly, as are `ChangedFrame`, `TiledFrame`, `ChangeSet` and
`ChangeAuditStats`.

## Capture contract

`camera.grab_with_changes(region=None, *, tight_rectangles=False) -> ChangedFrame | None` is an opt-in one-shot API for DXGI CPU BGRA8 SDR capture. The `region` argument follows `grab()`. An unsupported backend, GPU output, capture format, or active stream raises `ValueError`. A released camera raises `RuntimeError`. Each camera keeps its own history.

| Outcome | Return value |
| --- | --- |
| No image delivered | `None`, with the same capture-error reporting as `grab()` |
| First image or a new source, generation, geometry, format, or region | All tiles, with reason `initial` or `reset` |
| Acquired image differs from history | Exact changed tiles, with reason `changed` |
| Acquired image is byte-identical to history | A `ChangedFrame` with empty tiles and reason `identical` |

`ChangedFrame.pixels` is a leased `numpy.ndarray` valid until the frame is released, including when the converter supplies an owned array internally. Use the frame as a context manager or call its idempotent `release()`. Accessing `pixels` through a released frame raises `BufferReleasedError`. Use `pixels.copy()` for an independently owned array. A caller may edit a live lease. RapidShot updates its private history before exposing the lease, so those edits cannot affect subsequent comparisons. Keep the frame handle alive for as long as any of its views are used.

`ChangedFrame.changes` is an immutable `ChangeSet`. Tiles are fixed at 128×128 pixels. `indices` contains ascending row-major tile IDs. `rectangles` contains the corresponding frame-local half-open `(left, top, right, bottom)` rectangles, clipped at the image edges. `size` is `(width, height)`. `reason` describes the outcome above. `present_time_qpc` contains DXGI `LastPresentTime` as Windows QPC ticks, or `None` when unavailable. `accumulated_frames` contains the DXGI count. `fallback_reason` identifies why candidate hints were bypassed, or is `None` when hints were used.

Fallback reasons are `baseline` for a new history, `rotation`, `frame_gap`, `accumulated_frames`, `move_rects`, `metadata_unavailable`, `empty_dirty_rects`, and `invalid_dirty_rects`.

All four BGRA bytes participate, so an alpha-only change counts. A source or coordinate change seeds a new baseline. Missed acquisitions, accumulated frames, moves, rotation, and missing or invalid dirty rectangles use an exact full comparison. Dirty rectangles narrow candidates when the preceding image and coordinate space are known. A comparison checks delivered pixels. A captured image can still contain regions from different desktop updates.

Calls to `grab_with_changes()` on the same camera are serialized with capture and history updates. The caller serializes camera lifecycle operations such as `start()`, `stop()`, and camera `release()` with one-shot capture. Histories remain private to their cameras. A returned lease can stay live across another call while the pool has capacity. The owner coordinates access to each lease and releases it before destroying the camera. Concurrent use of an array after its lease is released is outside the contract.

## Tile snapshots and crops

`camera.grab_tiles(region=None, *, tight_rectangles=False, audit_every=0)` returns a `TiledFrame` lease or `None`. It uses the same one-shot DXGI CPU BGRA8 SDR restrictions, region coordinates, and change reasons as `grab_with_changes()`. The two methods keep separate histories. Switching between them can retain two complete baselines and cause a full readback after a frame gap.

```python
frame = camera.grab_tiles(tight_rectangles=True, audit_every=120)
if frame is not None:
    with frame:
        for index, pixels in zip(frame.changes.indices, frame.tiles):
            process_tile(index, pixels)
        crop = frame.materialize((32, 48, 240, 160))
```

`frame.tiles` yields read-only BGRA views over immutable bytes, in `frame.changes.indices` order. Use them while the lease is held. Holding a reference to a tile view can keep those bytes allocated after release. `materialize(rect=None)` assembles the whole snapshot or a nonempty, in-bounds, half-open rectangle in frame coordinates. It returns a writable, C-contiguous array owned by the caller. A held snapshot preserves its pixels across later captures, even if the camera is released. Release the frame explicitly or use a context manager. After release, `tiles` and `materialize()` raise `BufferReleasedError`. The change map remains readable. `pool_size_frames` limits the number of live tiled snapshots. An exhausted tile pool raises `PoolExhaustedError` before acquiring an update.

RapidShot stages only candidate tiles when the dirty hints can be used. It keeps a complete immutable CPU baseline and shares unchanged pages between snapshots. A reset, frame gap, move, rotation, missing metadata, or protected masking triggers a full readback of that acquisition. Dense or full-frame consumers also pay to materialize the pixels they request.

Both exact methods accept `tight_rectangles=True`. The resulting `ChangeSet.tight_rectangles` contains one frame-local bounding rectangle per changed tile, including alpha-only changes. It is `None` when disabled and `()` for an identical image when enabled. Initial and reset frames use the full clipped tile bounds. A tight rectangle encloses all byte changes within a tile and can contain unchanged pixels between them.

`audit_every=N` schedules a full pixel comparison on every Nth eligible guided tile capture. An audit that discovers a changed tile outside the hints reports the corrected pixels and change map and disables hint-guided copies for that capture generation. `camera.change_audit_stats()` reports cumulative audited frames, frames with omissions, and omitted tile counts. Audits sample the stream, so they can miss changes that appear and revert between samples. `audit_every=0` disables sampling.

## Choosing a capture path

| Consumer | Capture path |
| --- | --- |
| Sparse updates, with search or processing limited to small regions | `grab_tiles(tight_rectangles=True)`. Request only the crops needed by the task. |
| Full images with exact change metadata | `grab_with_changes()`. Enable tight bounds when the downstream stage can use them. |
| Full images with dense motion and no need for change metadata | `grab()`. |
| Find-any tasks or repeating tile-local work | Apply ranking or a byte-verified cache in the consumer. These are downstream strategies rather than capture options. |

Tight bounds spend extra CPU to find which pixels differ inside a tile. They can pay for themselves when a later search can exclude most of a frame. Materializing an entire tiled snapshot allocates and writes the whole output. Both costs depend on the workload and should be measured with the task that consumes the frames.

## Release criteria fixed before measurement

- **Correctness:** zero changed-map discrepancies against a full independent BGRA pixel scan. Exercise alpha, edge tiles, crop and source resets, frame gaps, lease mutation, and pool exhaustion. Record mixed desktop frames separately from detector errors.
- **Memory:** at Intel 1080p, exact capture adds at most **12 MiB steady private commit** and **20 MiB peak Windows-recorded commit** against matched plain RapidShot capture with the same lease policy. Both pass and fail are measured from paired trials. Count the bytes written to RapidShot history in the same column as an explicit previous-frame snapshot elsewhere.
- **Idle CPU:** use healthy, seeded capture workers with zero delivered frames in each timed window. In matched pairs against plain RapidShot, compare the relative increase in whole-process cycles per second. The gate passes when the upper end of the 95% confidence interval for the paired median is at most **5%**, fails when the lower end exceeds 5%, and stays inconclusive otherwise or with fewer than ten eligible pairs. The study targets 20 paired rounds and preserves windows with desktop activity separately. The earlier CPU-percentage comparison stays in the historical report. On this host, its 10-second samples moved in steps of about 0.155 percentage points.
- **Downstream benefit:** the same-pixel sparse fixture saves at least **50%** of search-plus-gate time including history upkeep, with identical detections. The controlled crossover must reach **30% changed tiles** for the sparse/local claim. A scrolling claim requires crossover at or above the measured Chrome scroll fraction of about 59%.

These criteria apply to Intel Graphics at 1920×1080 and 100 Hz on the HP 324pe. A failed or unmeasurable release criterion holds the performance claim pending.

## Measured against those criteria

Every criterion above was written down before the runs that scored it. All four passed for 2.9.0. [Full report, protocols and strata](change-detection/report.md)

| Criterion | Outcome | Measured |
| --- | --- | --- |
| **Correctness** | Pass | One-hour public-API soak: **29,985 change maps** each checked against an independent comparison of consecutive delivered BGRA arrays, **1,033 retained-lease checks**, **zero missed tiles in 187 full audits**, two capture-resource generations rebuilt, zero process restarts. Fault injection covered forced pool exhaustion, access loss, a locked desktop, protected content, source and crop resets, rotation, DPI changes, frame gaps, alpha-only changes, edge tiles and mutation of a live lease. Offline, periodic full scans caught a deliberately injected omission and found **zero real omissions in 217 unique eligible trace pairs**. Mixed desktop frames are recorded separately from detector errors, as required: all 63 saved mismatches from a 100 updates/s rerun held an exact mix of two consecutive intended states, this occurred with DXcam too, and a full-frame search agreed with every saved detection |
| **Memory** | Pass | Steady private commit cost is one history frame, **7.91 MiB** at 1080p against a 12 MiB limit. History writes were **~0.7 MiB per sparse result** and 7.91 MiB when the whole frame changed, counted in the same column as an explicit previous-frame snapshot elsewhere. At 4K the history is **31.64 MiB**; no 4K gate was set, so no 4K memory claim is made |
| **Idle CPU** | Pass | Pooled, under the extension protocol's rule declared before the new data existed: `grab_with_changes()` **+1.15%** of process cycles/s (95% CI −0.82% to +2.84%, 38 pairs); `grab_tiles()` **−1.40%** (−3.03% to +1.13%, 39 pairs); `grab_tiles(tight_rectangles=True)` **−1.90%** (−3.97% to +2.06%, 18 pairs). Ceiling **+5%**, unchanged. Two paths first scored *inconclusive* on about 20 pairs each; more pairs were collected under the same frozen rule rather than moving the ceiling, and the original result is preserved in the report. Mechanism, counted rather than inferred: across **3,818 confirmed `AcquireNextFrame` timeouts** an instrumented probe observed **zero GPU copy calls, zero maps, zero processor calls, and zero history-update calls or bytes written** |
| **Downstream benefit** | Pass, with the scrolling claim dropped | On identical pixels at 8.1% changed tiles, the median paired saving in search-plus-gate time including history upkeep was **81.6% clustered and 57.7% scattered**, against a 50% threshold, with identical detections. The controlled crossover reached **30.4%** changed tiles for both layouts, meeting the 30% bar for the sparse/local claim. It did **not** reach the ~59% Chrome scroll fraction for scattered change — first loss at 40.0% — so **there is no scrolling claim**, and DXGI reported no usable move rectangles in 690 trace frames to build one from. With tight rectangles, six timed sparse pairs covered **15.9–43.4× fewer pixels** and cut detector-plus-search time by **95.4–96.4%** |

**Known limits that are not criteria failures.** A captured image can contain parts of two desktop updates and the change map reports that faithfully; it happens more on a 60 Hz panel (88 of 96 states correct against 96 of 96 at 100 Hz with the same 60 updates/s offered). Dense *scattered* change is slower than a full search past the crossover, by up to 18.3%. `tight_rectangles=True` costs capture-side CPU with no return when no search follows: 7.55 against 3.56 Mcycles per state in a capture-only run, which is why it is off by default. Private commit drifted **+8.48 MiB/hour** across the soak's five-minute blocks, ending 2.92 MiB above the first block and below the previous four, with the histories themselves fixed; the slope is not monotonic and the trace does not identify a source, so it is disclosed rather than explained. Fixture, library versions, denominator, and measured path accompany any reported number.
