# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 1 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — synthetic calibrated matmul (250.0 ms target, 2059 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 75 | 75 | 0 | baseline | 19.78 | 19.62 | 26 of 100 |
| rapidshot_exact | Full-frame compare | 80 | 80 | 0 | -6.7% | 19.73 | 19.69 | 21 of 100 |
| rapidshot_exact | Change metadata | 78 | 78 | 0 | -4.0% | 19.76 | 19.77 | 23 of 100 |
| dxcam_copy | No filtering | 71 | 71 | 0 | baseline | 19.69 | 19.53 | 30 of 100 |
| dxcam_copy | Full-frame compare | 72 | 72 | 0 | 4.0% | 19.46 | 19.55 | 29 of 100 |
| mss_gdi | No filtering | 72 | 72 | 0 | baseline | 18.59 | 18.88 | 28 of 100 |
| mss_gdi | Full-frame compare | 138 | 100 | 0 | -33.3% | 16.77 | 18.22 | 0 of 100 |
| windows_capture_dxgi | No filtering | 77 | 77 | 0 | baseline | 19.48 | 19.41 | 24 of 100 |
| windows_capture_dxgi | Full-frame compare | 77 | 77 | 0 | -2.7% | 19.31 | 19.27 | 23 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 74 of 100 | 0.2652 | 290.4 ms | 342.2 ms | 359.5 ms | 0.0000 | 111.8 MiB |
| rapidshot_exact | Full-frame compare | 79 of 100 | 0.2492 | 282.5 ms | 303.6 ms | 327.4 ms | 0.1783 | 127.3 MiB |
| rapidshot_exact | Change metadata | 77 of 100 | 0.2567 | 283.7 ms | 320.7 ms | 356.4 ms | 0.0001 | 111.8 MiB |
| dxcam_copy | No filtering | 70 of 100 | 0.2790 | 305.1 ms | 368.8 ms | 389.5 ms | 0.0000 | 99.4 MiB |
| dxcam_copy | Full-frame compare | 71 of 100 | 0.2753 | 305.5 ms | 350.2 ms | 371.5 ms | 0.1860 | 107.5 MiB |
| mss_gdi | No filtering | 72 of 100 | 0.2622 | — | — | — | 0.0001 | 83.0 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.1822 | — | — | — | 0.3588 | 90.8 MiB |
| windows_capture_dxgi | No filtering | 76 of 100 | 0.2553 | — | — | — | 0.0000 | 103.7 MiB |
| windows_capture_dxgi | Full-frame compare | 77 of 100 | 0.2502 | — | — | — | 0.1964 | 119.8 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
