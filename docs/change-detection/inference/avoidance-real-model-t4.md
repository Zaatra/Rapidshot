# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — OpenCV Zoo YOLOX-S FP32 via ONNX Runtime (4 intra-op thread(s), 75.5 ms measured idle)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 253 | 253 | 152 | baseline | 19.06 | 67.47 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 399 | 101 | 0 | 60.1% | 7.43 | 28.70 | 0 of 100 |
| rapidshot_exact | Change metadata | 397 | 101 | 0 | 60.1% | 7.70 | 28.22 | 0 of 100 |
| dxcam_copy | No filtering | 229 | 229 | 128 | baseline | 19.05 | 68.44 | 0 of 100 |
| dxcam_copy | Full-frame compare | 389 | 101 | 0 | 60.1% | 7.87 | 39.25 | 0 of 100 |
| mss_gdi | No filtering | 205 | 205 | 105 | baseline | 15.64 | 56.72 | 0 of 100 |
| mss_gdi | Full-frame compare | 475 | 100 | 0 | 60.5% | 8.27 | 33.88 | 0 of 100 |
| windows_capture_dxgi | No filtering | 213 | 213 | 112 | baseline | 18.09 | 62.72 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 357 | 100 | 0 | 60.5% | 7.70 | 29.81 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.6747 | 98.5 ms | 124.1 ms | 134.5 ms | 0.0001 | 278.6 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.2870 | 79.1 ms | 86.1 ms | 96.7 ms | 1.1563 | 292.7 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.2822 | 77.9 ms | 91.0 ms | 93.0 ms | 0.0003 | 278.7 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.6844 | 109.6 ms | 146.5 ms | 150.3 ms | 0.0002 | 265.0 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.3925 | 85.1 ms | 93.1 ms | 100.2 ms | 1.0009 | 272.6 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.5672 | — | — | — | 0.0002 | 246.4 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.3387 | — | — | — | 1.2945 | 254.4 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.6272 | — | — | — | 0.0001 | 272.3 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.2981 | — | — | — | 1.0711 | 285.6 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
