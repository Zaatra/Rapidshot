# Capture cost and frame age

Values are medians of matched-format windows. The CPU numerator covers the measured worker and its background threads. Active coverage counts each offered marker once. Passive counts use new native timestamps or visible cover updates when available. Repeated grab returns stay in the raw totals. Missing measurements appear as —.

CPU uses whole-process cycles plus Windows process time over the fixed window. GPU-engine values show the busiest observed instance of each engine type. They are process busy counters, not timing for specific copy commands. GPU memory shows peak process-reported local and non-local usage. Logical output copy bytes exclude unobserved driver traffic. CPU copies include private history and materialized task and fixture-probe crops. GPU staging and all copy stages are recorded separately in each trial.

Desktop age uses each backend’s validated presentation timestamp. GDI and the Windows Capture DXGI binding report no native timestamp. The baseline imports Python, NumPy and OpenCV without capture libraries and runs at the listed offered rate.

| Pipeline | Workload | Updates/s | Arm | Waves | New/second | Grab returns | CPU % of one core | Mcycles/s/frame | Private steady/peak and RSS steady/peak MiB | GPU local/non-local peak MiB | GPU engine busy % | CPU-copied MiB/s/frame | GPU-staged MiB/s/frame | Desktop→delivery p50/p95 ms | Desktop→result p50/p95 ms | Queue age p50 ms | Correct on time / offered | Peak pool outstanding/exhaustions |
|---|---|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|---|---|---:|---:|---|
| shared_tiles | sparse | 30 | dxcam_view | 5 | 29.90 | 305.00 | 52.85 | 1753.10/58.63 | 501.57/512.68/103.18/113.87 | 17.74/0.00 | 3D 2.79, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 473.06/15.82 | 236.53/7.91 | 2.93/6.09 | 17.34/24.17 | — | 939/1530 | — |
| shared_tiles | sparse | 30 | dxcam_view_tight | 5 | 29.90 | 305.00 | 24.81 | 832.41/27.84 | 501.03/512.46/103.26/113.87 | 17.74/0.00 | 3D 2.73, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 473.06/15.82 | 236.53/7.91 | 2.74/4.83 | 7.10/10.13 | — | 1514/1530 | — |
| shared_tiles | sparse | 30 | rapidshot_exact | 5 | 29.90 | 305.00 | 46.26 | 1496.79/49.97 | 512.63/523.73/114.67/125.43 | 17.74/0.00 | 3D 2.63, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 251.19/8.40 | 236.53/7.91 | 2.86/5.29 | 15.78/22.37 | — | 1091/1530 | 1/0 |
| shared_tiles | sparse | 30 | rapidshot_exact_tight | 5 | 29.90 | 305.00 | 15.78 | 511.94/17.12 | 512.36/523.81/114.97/125.54 | 17.74/0.00 | 3D 2.64, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 251.19/8.40 | 236.53/7.91 | 3.86/6.52 | 6.17/9.41 | — | 1504/1530 | 1/0 |
| shared_tiles | sparse | 30 | rapidshot_plain | 5 | 29.90 | 305.00 | 46.87 | 1632.28/54.59 | 512.56/523.73/114.53/125.09 | 17.74/0.00 | 3D 2.66, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 473.06/15.82 | 236.53/7.91 | 2.64/5.19 | 17.15/24.67 | — | 893/1530 | 1/0 |
| shared_tiles | sparse | 30 | rapidshot_tight | 5 | 29.90 | 305.00 | 14.09 | 476.53/15.94 | 512.93/532.20/99.82/118.21 | 17.75/0.00 | 3D 1.49, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 21.42/0.72 | 45.04/1.51 | 2.99/6.17 | 5.17/8.65 | — | 1517/1530 | 1/0 |
| shared_tiles | sparse | 30 | rapidshot_tiled | 5 | 29.90 | 305.00 | 44.73 | 1489.47/49.81 | 513.49/532.32/99.67/118.29 | 17.75/0.00 | 3D 1.49, Compute 0.00, Copy 0.00, GSC 0.00, VideoDecode 0.00, VideoProcessing 0.00 | 63.58/2.13 | 45.04/1.51 | 1.83/5.25 | 15.05/23.19 | — | 1061/1530 | 1/0 |

## Exact and tiled arms minus plain RapidShot

| Arm | Pipeline | Workload | Updates/s | Paired rounds | Steady private MiB | Windows peak private MiB | CPU percentage points of one core |
|---|---|---|---:|---:|---:|---:|---:|
| rapidshot_exact | shared_tiles | sparse | 30 | 5 | 0.18 | 0.12 | -1.68 |
| rapidshot_tight | shared_tiles | sparse | 30 | 5 | 0.44 | 8.32 | -32.93 |
| rapidshot_tiled | shared_tiles | sparse | 30 | 5 | 0.90 | 8.59 | -1.99 |

Configuration (`matched/manifest.json`, raw data, not published) · Raw trials (`matched/trials`, raw data, not published)
