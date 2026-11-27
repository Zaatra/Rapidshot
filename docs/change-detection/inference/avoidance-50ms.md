# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — synthetic calibrated matmul (50.0 ms target, 316 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 401 | 401 | 300 | baseline | 15.95 | 16.81 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 400 | 101 | 0 | 74.8% | 5.11 | 7.28 | 0 of 100 |
| rapidshot_exact | Change metadata | 401 | 101 | 0 | 74.8% | 4.98 | 5.81 | 0 of 100 |
| dxcam_copy | No filtering | 366 | 366 | 265 | baseline | 18.69 | 19.53 | 0 of 100 |
| dxcam_copy | Full-frame compare | 400 | 101 | 0 | 74.8% | 5.19 | 19.02 | 0 of 100 |
| mss_gdi | No filtering | 285 | 285 | 184 | baseline | 13.94 | 15.95 | 0 of 100 |
| mss_gdi | Full-frame compare | 700 | 100 | 0 | 75.1% | 5.05 | 12.12 | 0 of 100 |
| windows_capture_dxgi | No filtering | 334 | 334 | 233 | baseline | 16.94 | 17.34 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 400 | 100 | 0 | 75.1% | 4.98 | 6.77 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.1681 | 45.0 ms | 56.9 ms | 76.5 ms | 0.0002 | 112.1 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.0728 | 55.1 ms | 57.0 ms | 63.2 ms | 0.8695 | 128.1 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.0581 | 51.8 ms | 55.9 ms | 87.9 ms | 0.0003 | 112.3 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.1953 | 73.1 ms | 103.4 ms | 108.2 ms | 0.0003 | 99.5 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.1902 | 58.1 ms | 63.2 ms | 69.1 ms | 1.0472 | 107.3 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.1595 | — | — | — | 0.0002 | 83.0 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.1212 | — | — | — | 1.8870 | 91.3 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.1734 | — | — | — | 0.0001 | 103.5 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.0677 | — | — | — | 1.1633 | 119.7 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
