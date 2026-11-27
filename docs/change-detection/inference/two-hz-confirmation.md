# Live capture libraries through one CPU detector

One worker process per arm on HP 324pe (1920×1080, 100 Hz, Intel(R) Graphics). The dashboard was repainted at 2 updates a second, 30 s per arm per rate, 2 round(s). Every arm fed the same OpenCV Zoo YOLOX-S FP32 model on the CPU, one thread, OpenCL off.

CPU and memory are the worker process only, so painting the fixture is not charged to any library. `detections ready` is the age of the captured pixels at the moment the detector's output exists, measured from the DXGI present that produced them.

## 2 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.07 | 258.26 ms | 2.59 ms | 0.001 ms | 1806.8 | 53.96 | 215.6 MiB |
| RapidShot `grab()` | 2.03 | 260.26 ms | 2.10 ms | 2.043 ms | 1833.7 | 54.53 | 210.3 MiB |
| DXcam `grab()` | 2.03 | 272.65 ms | 3.78 ms | 2.332 ms | 3314.1 | 98.85 | 197.6 MiB |
| DXcam `grab_view()` | 2.03 | 287.20 ms | 2.50 ms | 2.254 ms | 3281.7 | 96.77 | 196.6 MiB |
| MSS (GDI) | 2.03 | — | — | 2.591 ms | 2498.1 | 74.17 | 179.1 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 65 | 62 | 3 | 876 | 65 | capture change metadata (changed) |
| RapidShot `grab()` | 61 | 61 | 0 | 862 | 61 | retained full-frame compare |
| DXcam `grab()` | 61 | 61 | 0 | 279420 | 61 | retained full-frame compare |
| DXcam `grab_view()` | 127 | 61 | 66 | 222812 | 127 | retained full-frame compare |
| MSS (GDI) | 674 | 61 | 613 | 0 | 674 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 64 verdicts checked against an independent full-frame compare, **0 disagreed**.

## 2 updates a second — round 2

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.03 | 262.73 ms | 2.51 ms | 0.001 ms | 1834.1 | 54.32 | 216.9 MiB |
| RapidShot `grab()` | 2.03 | 254.50 ms | 2.21 ms | 2.015 ms | 1773.6 | 52.34 | 207.6 MiB |
| DXcam `grab()` | 2.03 | 256.99 ms | 3.87 ms | 2.326 ms | 3348.0 | 98.85 | 197.9 MiB |
| DXcam `grab_view()` | 2.03 | 256.00 ms | 2.30 ms | 2.044 ms | 3347.0 | 99.22 | 197.9 MiB |
| MSS (GDI) | 2.00 | — | — | 2.579 ms | 2472.4 | 73.44 | 179.0 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 61 | 61 | 0 | 866 | 61 | capture change metadata (changed) |
| RapidShot `grab()` | 61 | 61 | 0 | 900 | 61 | retained full-frame compare |
| DXcam `grab()` | 61 | 61 | 0 | 332926 | 61 | retained full-frame compare |
| DXcam `grab_view()` | 61 | 61 | 0 | 337848 | 61 | retained full-frame compare |
| MSS (GDI) | 722 | 60 | 662 | 0 | 722 | retained full-frame compare |

Audit of RapidShot `grab_with_changes()`: 60 verdicts checked against an independent full-frame compare, **0 disagreed**.

MSS reaches the desktop through GDI, which exposes no present timestamp, so that arm has no pixel age and its latency columns are blank. Every other arm times from the DXGI present that produced the pixels.
