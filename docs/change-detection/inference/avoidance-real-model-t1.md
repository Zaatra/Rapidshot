# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — OpenCV Zoo YOLOX-S FP32 via ONNX Runtime (1 intra-op thread(s), 210.1 ms measured idle)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 88 | 88 | 0 | baseline | 19.84 | 19.48 | 13 of 100 |
| rapidshot_exact | Full-frame compare | 91 | 91 | 0 | -3.4% | 19.56 | 19.77 | 10 of 100 |
| rapidshot_exact | Change metadata | 93 | 93 | 0 | -5.7% | 19.72 | 19.77 | 8 of 100 |
| dxcam_copy | No filtering | 94 | 94 | 0 | baseline | 19.86 | 19.81 | 7 of 100 |
| dxcam_copy | Full-frame compare | 90 | 90 | 0 | -2.3% | 19.47 | 19.67 | 11 of 100 |
| mss_gdi | No filtering | 86 | 86 | 0 | baseline | 18.22 | 18.59 | 14 of 100 |
| mss_gdi | Full-frame compare | 87 | 87 | 0 | 1.1% | 18.21 | 18.95 | 13 of 100 |
| windows_capture_dxgi | No filtering | 90 | 90 | 0 | baseline | 19.19 | 19.06 | 10 of 100 |
| windows_capture_dxgi | Full-frame compare | 89 | 89 | 0 | -1.1% | 19.20 | 19.23 | 12 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 87 of 100 | 0.2240 | 251.9 ms | 290.9 ms | 328.1 ms | 0.0000 | 277.7 MiB |
| rapidshot_exact | Full-frame compare | 90 of 100 | 0.2196 | 246.2 ms | 276.8 ms | 325.0 ms | 0.1783 | 291.1 MiB |
| rapidshot_exact | Change metadata | 92 of 100 | 0.2148 | 239.0 ms | 269.0 ms | 286.2 ms | 0.0001 | 278.0 MiB |
| dxcam_copy | No filtering | 93 of 100 | 0.2130 | 238.8 ms | 263.9 ms | 276.7 ms | 0.0001 | 263.7 MiB |
| dxcam_copy | Full-frame compare | 89 of 100 | 0.2210 | 248.6 ms | 278.0 ms | 301.2 ms | 0.1732 | 273.6 MiB |
| mss_gdi | No filtering | 86 of 100 | 0.2162 | — | — | — | 0.0001 | 247.5 MiB |
| mss_gdi | Full-frame compare | 87 of 100 | 0.2179 | — | — | — | 0.1756 | 253.5 MiB |
| windows_capture_dxgi | No filtering | 90 of 100 | 0.2118 | — | — | — | 0.0000 | 271.1 MiB |
| windows_capture_dxgi | Full-frame compare | 88 of 100 | 0.2186 | — | — | — | 0.1643 | 284.6 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
