# Inference avoided, with duplicate frames controlled

HP 324pe, 1920×1080 at 100 Hz, Intel(R) Graphics. The dashboard was repainted 20 times a second for 20 s per arm, **3 identical repaints after each new state**, 3 round(s).

Each arm ran in its own worker process, so CPU and memory are the consumer alone. A state counts as *processed* when the detector ran on a frame carrying its marker; *missed* means no frame carrying it ever reached the detector.

## Round 1 — synthetic calibrated matmul (15.0 ms target, 122 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 401 | 401 | 300 | baseline | 6.26 | 7.38 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.52 | 4.05 | 0 of 100 |
| rapidshot_exact | Change metadata | 401 | 101 | 0 | 74.8% | 1.54 | 2.75 | 0 of 100 |
| dxcam_copy | No filtering | 401 | 401 | 300 | baseline | 6.38 | 18.77 | 0 of 100 |
| dxcam_copy | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.57 | 19.00 | 0 of 100 |
| mss_gdi | No filtering | 503 | 503 | 403 | baseline | 7.63 | 11.12 | 0 of 100 |
| mss_gdi | Full-frame compare | 896 | 100 | 0 | 75.1% | 1.53 | 10.16 | 0 of 100 |
| windows_capture_dxgi | No filtering | 401 | 401 | 301 | baseline | 6.42 | 6.88 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 401 | 100 | 0 | 75.1% | 1.55 | 3.03 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.0737 | 18.5 ms | 21.2 ms | 32.5 ms | 0.0002 | 112.0 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.0405 | 20.7 ms | 23.7 ms | 42.6 ms | 1.2338 | 128.2 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.0275 | 17.8 ms | 21.4 ms | 32.6 ms | 0.0003 | 112.2 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.1877 | 19.8 ms | 22.6 ms | 37.7 ms | 0.0003 | 99.5 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.1900 | 21.4 ms | 24.4 ms | 32.2 ms | 0.8646 | 107.5 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.1113 | — | — | — | 0.0006 | 83.3 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.1016 | — | — | — | 2.0761 | 91.3 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.0688 | — | — | — | 0.0002 | 103.6 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.0303 | — | — | — | 0.8429 | 119.5 MiB |

## Round 2 — synthetic calibrated matmul (15.0 ms target, 122 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 401 | 401 | 300 | baseline | 6.24 | 7.28 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.09 | 3.59 | 0 of 100 |
| rapidshot_exact | Change metadata | 401 | 101 | 0 | 74.8% | 1.52 | 2.41 | 0 of 100 |
| dxcam_copy | No filtering | 401 | 401 | 300 | baseline | 6.42 | 18.94 | 0 of 100 |
| dxcam_copy | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.49 | 18.88 | 0 of 100 |
| mss_gdi | No filtering | 501 | 501 | 401 | baseline | 7.71 | 11.30 | 0 of 100 |
| mss_gdi | Full-frame compare | 718 | 100 | 0 | 75.1% | 1.52 | 9.12 | 0 of 100 |
| windows_capture_dxgi | No filtering | 401 | 401 | 300 | baseline | 4.91 | 5.42 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.60 | 3.42 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.0728 | 18.2 ms | 22.0 ms | 27.9 ms | 0.0002 | 112.3 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.0359 | 15.7 ms | 19.4 ms | 26.6 ms | 1.0397 | 128.0 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.0241 | 17.6 ms | 19.8 ms | 21.6 ms | 0.0003 | 112.1 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.1894 | 20.0 ms | 23.3 ms | 26.9 ms | 0.0003 | 99.5 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.1888 | 21.9 ms | 23.2 ms | 33.8 ms | 1.0523 | 107.3 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.1130 | — | — | — | 0.0004 | 83.2 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.0912 | — | — | — | 1.9450 | 91.4 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.0542 | — | — | — | 0.0002 | 103.9 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.0342 | — | — | — | 1.1265 | 119.2 MiB |

## Round 3 — synthetic calibrated matmul (15.0 ms target, 123 rounds)

| Arm | Group | Delivered | Detector calls | Redundant | Avoided | Inference s | CPU s | States missed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 401 | 401 | 300 | baseline | 6.23 | 7.23 | 0 of 100 |
| rapidshot_exact | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.44 | 3.70 | 0 of 100 |
| rapidshot_exact | Change metadata | 401 | 101 | 0 | 74.8% | 1.58 | 2.67 | 0 of 100 |
| dxcam_copy | No filtering | 401 | 401 | 300 | baseline | 6.49 | 18.58 | 0 of 100 |
| dxcam_copy | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.59 | 18.58 | 0 of 100 |
| mss_gdi | No filtering | 501 | 501 | 401 | baseline | 7.71 | 11.45 | 0 of 100 |
| mss_gdi | Full-frame compare | 767 | 100 | 0 | 75.1% | 1.51 | 9.59 | 0 of 100 |
| windows_capture_dxgi | No filtering | 401 | 401 | 300 | baseline | 4.99 | 5.38 | 0 of 100 |
| windows_capture_dxgi | Full-frame compare | 401 | 101 | 0 | 74.8% | 1.09 | 2.91 | 0 of 100 |

| Arm | Group | States processed | CPU s per state | Freshness p50 | p95 | p99 | Decision s | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_exact | No filtering | 100 of 100 | 0.0723 | 18.0 ms | 21.2 ms | 38.2 ms | 0.0002 | 112.1 MiB |
| rapidshot_exact | Full-frame compare | 100 of 100 | 0.0370 | 19.5 ms | 21.8 ms | 37.6 ms | 1.1773 | 128.1 MiB |
| rapidshot_exact | Change metadata | 100 of 100 | 0.0267 | 18.0 ms | 22.3 ms | 30.5 ms | 0.0003 | 112.3 MiB |
| dxcam_copy | No filtering | 100 of 100 | 0.1858 | 21.1 ms | 24.2 ms | 29.6 ms | 0.0002 | 99.6 MiB |
| dxcam_copy | Full-frame compare | 100 of 100 | 0.1858 | 23.7 ms | 27.3 ms | 29.3 ms | 1.1975 | 107.4 MiB |
| mss_gdi | No filtering | 100 of 100 | 0.1145 | — | — | — | 0.0004 | 83.2 MiB |
| mss_gdi | Full-frame compare | 100 of 100 | 0.0959 | — | — | — | 2.1184 | 91.0 MiB |
| windows_capture_dxgi | No filtering | 100 of 100 | 0.0537 | — | — | — | 0.0002 | 103.7 MiB |
| windows_capture_dxgi | Full-frame compare | 100 of 100 | 0.0291 | — | — | — | 1.1598 | 119.6 MiB |

No energy figures appear here. This host exposes processor clock through `CallNtPowerInformation` but no power or energy counter, so CPU seconds and cycles are the resource measure.
