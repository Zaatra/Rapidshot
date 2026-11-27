# Matched tight-search streaming controls

All arms run the same tight OpenCV search. The two DXcam one-shot arms differ only in the requested pause after an empty `grab_view()`. Stock DXcam streaming uses `start(target_fps=100, video_mode=False)` and `get_latest_frame(copy=True, with_timestamp=True)`. Producer pacing, threading, latest-slot selection, and copying all change in that arm.

One-shot DXcam copies changed deliveries into a private history. The streaming history retains the consumer's owned array. The stream counts full GPU staging and CPU ring writes per successful production and an additional full copy per consumer return. The source fixture retains metrics.

The deadline is 33.33 ms from source submission. Pixel age starts at the desktop present. Steady-state excludes the first 3 offered updates each round. Frame counts and latencies pool rounds. CPU and per-delivery copy figures are medians of round values.

| Arm | Rounds | Correct on time | Steady correct on time | Wrong | Undelivered | Pixel age at result p50/p95 ms | Submission to result p50/p95 ms | Process Mcycles/s | Process % of one core | Empty calls | Mean measured retry ms | GPU MiB per delivery |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| rapidshot_tight | 5 | 1517/1530 | 1514/1515 | 0 | 5 | 4.60/8.38 | 19.14/27.46 | 467.48 | 13.79 | 2302 | 0.68 | 1.51 |
| dxcam_view_tight | 5 | 1517/1530 | 1514/1515 | 0 | 5 | 7.05/9.21 | 20.30/28.45 | 838.06 | 24.66 | 38660 | 0.95 | 7.91 |
| dxcam_view_tight_2ms | 5 | 1511/1530 | 1508/1515 | 0 | 5 | 7.86/10.57 | 21.89/30.41 | 691.00 | 20.83 | 16584 | 2.32 | 7.91 |
| dxcam_stream_tight | 5 | 1177/1530 | 1177/1515 | 0 | 2 | 15.41/18.13 | 28.69/37.88 | 820.23 | 22.21 | 2178 | — | 7.91 |

[Resource and timing tables](report.md) · Individual misses and copy counters (`last-checks/streaming/streaming-controls.json`, raw data, not published) · Raw trials (`last-checks/streaming/trials`, raw data, not published)
