# Capture cost and frame age

Values are medians of matched-format windows. The CPU numerator covers the measured worker and its background threads. Active coverage counts each offered marker once. Passive counts use new native timestamps or visible cover updates when available. Repeated grab returns stay in the raw totals. Missing measurements appear as —.

CPU uses whole-process cycles plus Windows process time over the fixed window. GPU-engine values show the busiest observed instance of each engine type. They are process busy counters, not timing for specific copy commands. GPU memory shows peak process-reported local and non-local usage. Logical output copy bytes exclude unobserved driver traffic. CPU copies include private history and materialized task and fixture-probe crops. GPU staging and all copy stages are recorded separately in each trial.

Desktop age uses each backend’s validated presentation timestamp. GDI and the Windows Capture DXGI binding report no native timestamp. The baseline imports Python, NumPy and OpenCV without capture libraries and runs at the listed offered rate.

| Pipeline | Workload | Updates/s | Arm | Waves | New/second | Grab returns | CPU % of one core | Mcycles/s/frame | Private steady/peak and RSS steady/peak MiB | GPU local/non-local peak MiB | GPU engine busy % | CPU-copied MiB/s/frame | GPU-staged MiB/s/frame | Desktop→delivery p50/p95 ms | Desktop→result p50/p95 ms | Queue age p50 ms | Correct on time / offered | Peak pool outstanding/exhaustions |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|---:|---|
| shared_tiles | sparse | 30 | dxcam_stream_tight | 5 | 30.00 | 306.00 | 22.21 | 820.23/27.40 | 556.67/568.12/158.65/166.82 | 17.74/0.00 | 3D 3.49, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 474.61/15.82 | 237.30/7.91 | 11.15/13.00 | 15.51/18.30 | 2.44 | 1177/1530 | — |
| shared_tiles | sparse | 30 | dxcam_view_tight | 5 | 29.90 | 305.00 | 24.66 | 838.06/28.03 | 501.05/512.50/103.45/114.16 | 17.74/0.00 | 3D 2.84, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 473.06/15.82 | 236.53/7.91 | 2.64/3.95 | 7.07/9.09 | — | 1517/1530 | — |
| shared_tiles | sparse | 30 | dxcam_view_tight_2ms | 5 | 29.90 | 305.00 | 20.83 | 691.00/23.11 | 501.05/512.49/103.42/114.01 | 17.74/0.00 | 3D 3.10, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 473.06/15.82 | 236.53/7.91 | 3.51/4.43 | 7.76/9.64 | — | 1511/1530 | — |
| shared_tiles | sparse | 30 | rapidshot_tight | 5 | 29.90 | 305.00 | 13.79 | 467.48/15.63 | 512.75/531.86/99.84/118.00 | 17.75/0.00 | 3D 1.50, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 21.42/0.72 | 45.04/1.51 | 2.55/5.45 | 4.66/8.35 | — | 1517/1530 | 1/0 |

Configuration (`last-checks/streaming/manifest.json`, raw data, not published) · Raw trials (`last-checks/streaming/trials`, raw data, not published)
