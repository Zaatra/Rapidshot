# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — OpenCV Zoo YOLOX-S FP32 via ONNX Runtime (2 intra-op thread(s), 117.1 ms measured idle)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 157 | 157 | 56 | baseline | 19.49 | 37.47 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 296 | 101 | 0 | 35.7% | 11.98 | 24.55 | 0 of 100 |
| rapidshot_exact | Change metadata | 299 | 101 | 0 | 35.7% | 12.18 | 24.20 | 0 of 100 |
| dxcam_copy | No filtering | 156 | 156 | 55 | baseline | 19.45 | 37.56 | 0 of 100 |
| dxcam_copy | Full-frame compare | 298 | 101 | 0 | 35.7% | 12.43 | 30.59 | 0 of 100 |
| mss_gdi | No filtering | 143 | 143 | 43 | baseline | 17.13 | 33.61 | 0 of 100 |
| mss_gdi | Full-frame compare | 366 | 100 | 0 | 36.3% | 11.81 | 26.61 | 0 of 100 |
| windows_capture_dxgi | No filtering | 157 | 157 | 57 | baseline | 18.77 | 35.95 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 293 | 100 | 0 | 36.3% | 12.19 | 24.94 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.3747 | 151.4 ms | 179.8 ms | 182.5 ms | 0.0001 | 277.8 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.2455 | 122.5 ms | 139.1 ms | 162.4 ms | 0.6273 | 291.6 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.2420 | 120.9 ms | 139.2 ms | 148.6 ms | 0.0002 | 280.2 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.3756 | 152.8 ms | 187.2 ms | 195.4 ms | 0.0001 | 264.5 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.3059 | 128.8 ms | 144.3 ms | 157.6 ms | 0.6319 | 272.2 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.3361 | — | — | — | 0.0002 | 246.1 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.2661 | — | — | — | 0.9569 | 254.2 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.3595 | — | — | — | 0.0001 | 273.5 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.2494 | — | — | — | 0.7749 | 285.0 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
