# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — synthetic calibrated matmul (5.0 ms target, 41 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 401 | 401 | 300 | baseline | 2.28 | 3.33 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 401 | 101 | 0 | 74.8% | 0.39 | 2.83 | 0 of 100 |
| rapidshot_exact | Change metadata | 401 | 101 | 0 | 74.8% | 0.57 | 1.66 | 0 of 100 |
| dxcam_copy | No filtering | 401 | 401 | 300 | baseline | 2.32 | 19.17 | 0 of 100 |
| dxcam_copy | Full-frame compare | 401 | 101 | 0 | 74.8% | 0.39 | 18.95 | 0 of 100 |
| mss_gdi | No filtering | 668 | 668 | 568 | baseline | 3.45 | 8.25 | 0 of 100 |
| mss_gdi | Full-frame compare | 911 | 101 | 0 | 74.8% | 0.51 | 9.70 | 0 of 100 |
| windows_capture_dxgi | No filtering | 401 | 401 | 300 | baseline | 2.20 | 2.61 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 401 | 101 | 0 | 74.8% | 0.55 | 2.56 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.0333 | 8.1 ms | 11.8 ms | 13.2 ms | 0.0002 | 112.5 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.0283 | 9.7 ms | 12.1 ms | 16.2 ms | 1.1693 | 128.1 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.0166 | 8.5 ms | 10.5 ms | 12.2 ms | 0.0003 | 112.6 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.1917 | 9.8 ms | 11.6 ms | 12.5 ms | 0.0003 | 99.5 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.1895 | 10.2 ms | 11.3 ms | 12.6 ms | 0.9090 | 107.5 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.0825 | — | — | — | 0.0005 | 83.1 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.0970 | — | — | — | 2.3762 | 91.0 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.0261 | — | — | — | 0.0002 | 103.5 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.0256 | — | — | — | 1.1565 | 119.8 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
