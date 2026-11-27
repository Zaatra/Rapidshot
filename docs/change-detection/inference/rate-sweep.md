# Live capture libraries through one CPU detector

One worker process per arm on HP 324pe (1920×1080, 100 Hz, Intel(R) Graphics). The dashboard was repainted at 2, 5, 10, 30 updates a second, 20 s per arm per rate, 1 round(s). Every arm fed the same OpenCV Zoo YOLOX-S FP32 model on the CPU, one thread, OpenCL off.

CPU and memory are the worker process only, so painting the fixture is not charged to any library. `detections ready` is the age of the captured pixels at the moment the detector's output exists, measured from the DXGI present that produced them.

## 2 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.05 | 254.93 ms | 2.20 ms | 0.001 ms | 1932.2 | 59.30 | 216.1 MiB |
| RapidShot `grab()` | 2.05 | 255.27 ms | 2.13 ms | 2.034 ms | 1815.1 | 53.36 | 210.8 MiB |
| DXcam `grab()` | 2.05 | 256.50 ms | 4.02 ms | 2.343 ms | 3348.4 | 99.37 | 196.7 MiB |
| DXcam `grab_view()` | 2.05 | 253.75 ms | 2.28 ms | 2.033 ms | 3352.9 | 98.83 | 197.7 MiB |
| MSS (GDI) | 2.00 | — | — | 2.567 ms | 2481.0 | 73.20 | 179.0 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 255 | 41 | 214 | 648 | 255 | capture change metadata (identical) |
| RapidShot `grab()` | 64 | 41 | 23 | 599 | 64 | retained full-frame compare |
| DXcam `grab()` | 41 | 41 | 0 | 220548 | 41 | retained full-frame compare |
| DXcam `grab_view()` | 41 | 41 | 0 | 227218 | 41 | retained full-frame compare |
| MSS (GDI) | 497 | 40 | 457 | 0 | 497 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 254 verdicts checked against an independent full-frame compare, **0 disagreed**.

## 5 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 3.95 | 351.28 ms | 99.89 ms | 0.001 ms | 3351.4 | 98.75 | 217.2 MiB |
| RapidShot `grab()` | 3.95 | 358.22 ms | 102.85 ms | 1.985 ms | 3352.1 | 99.06 | 209.3 MiB |
| DXcam `grab()` | 3.90 | 354.41 ms | 103.25 ms | 2.308 ms | 3353.2 | 99.37 | 197.6 MiB |
| DXcam `grab_view()` | 3.95 | 351.05 ms | 100.31 ms | 1.991 ms | 3353.0 | 99.61 | 196.7 MiB |
| MSS (GDI) | 3.70 | — | — | 2.316 ms | 3215.7 | 95.39 | 179.8 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 79 | 79 | 0 | 0 | 79 | capture change metadata (changed) |
| RapidShot `grab()` | 79 | 79 | 0 | 0 | 79 | retained full-frame compare |
| DXcam `grab()` | 78 | 78 | 0 | 0 | 78 | retained full-frame compare |
| DXcam `grab_view()` | 79 | 79 | 0 | 0 | 79 | retained full-frame compare |
| MSS (GDI) | 74 | 74 | 0 | 0 | 74 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 78 verdicts checked against an independent full-frame compare, **0 disagreed**.

## 10 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 3.80 | 312.59 ms | 52.58 ms | 0.001 ms | 3327.3 | 98.44 | 215.8 MiB |
| RapidShot `grab()` | 3.85 | 310.39 ms | 54.60 ms | 1.983 ms | 3334.1 | 99.06 | 208.9 MiB |
| DXcam `grab()` | 3.90 | 303.64 ms | 49.50 ms | 2.304 ms | 3350.6 | 99.69 | 196.3 MiB |
| DXcam `grab_view()` | 3.85 | 312.31 ms | 53.46 ms | 1.989 ms | 3341.5 | 99.06 | 196.6 MiB |
| MSS (GDI) | 3.65 | — | — | 2.314 ms | 3213.7 | 95.08 | 179.6 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 76 | 76 | 0 | 0 | 76 | capture change metadata (changed) |
| RapidShot `grab()` | 77 | 77 | 0 | 0 | 77 | retained full-frame compare |
| DXcam `grab()` | 78 | 78 | 0 | 0 | 78 | retained full-frame compare |
| DXcam `grab_view()` | 77 | 77 | 0 | 0 | 77 | retained full-frame compare |
| MSS (GDI) | 73 | 73 | 0 | 0 | 73 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 75 verdicts checked against an independent full-frame compare, **0 disagreed**.

## 30 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 3.90 | 272.03 ms | 19.40 ms | 0.001 ms | 3350.2 | 99.22 | 215.9 MiB |
| RapidShot `grab()` | 3.95 | 271.58 ms | 18.36 ms | 2.001 ms | 3350.5 | 99.14 | 209.3 MiB |
| DXcam `grab()` | 3.90 | 275.13 ms | 21.71 ms | 2.285 ms | 3349.5 | 99.30 | 196.4 MiB |
| DXcam `grab_view()` | 3.90 | 272.01 ms | 16.43 ms | 2.008 ms | 3345.2 | 98.91 | 196.8 MiB |
| MSS (GDI) | 3.65 | — | — | 2.322 ms | 3196.5 | 94.53 | 180.2 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 78 | 78 | 0 | 0 | 78 | capture change metadata (changed) |
| RapidShot `grab()` | 79 | 79 | 0 | 0 | 79 | retained full-frame compare |
| DXcam `grab()` | 78 | 78 | 0 | 0 | 78 | retained full-frame compare |
| DXcam `grab_view()` | 78 | 78 | 0 | 0 | 78 | retained full-frame compare |
| MSS (GDI) | 73 | 73 | 0 | 0 | 73 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 77 verdicts checked against an independent full-frame compare, **0 disagreed**.

MSS reaches the desktop through GDI, which exposes no present timestamp, so that arm has no pixel age and its latency columns are blank. Every other arm times from the DXGI present that produced the pixels.
