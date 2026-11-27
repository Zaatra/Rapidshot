# Live capture libraries through one CPU detector

One worker process per arm on HP 324pe (1920×1080, 100 Hz, Intel(R) Graphics). The dashboard was repainted at 2 updates a second, 30 s per arm per rate, 2 round(s). Every arm fed the same OpenCV Zoo YOLOX-S FP32 model on the CPU, one thread, OpenCL off.

CPU and memory are the worker process only, so painting the fixture is not charged to any library. `detections ready` is the age of the captured pixels at the moment the detector's output exists, measured from the DXGI present that produced them.

## 2 updates a second — round 1

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.03 | 253.90 ms | 2.35 ms | 0.000 ms | 1761.3 | 52.03 | 208.0 MiB |
| DXcam `grab()` | 2.03 | 256.35 ms | 3.83 ms | 0.298 ms | 3346.8 | 99.17 | 189.8 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 61 | 61 | 0 | 899 | 0 | none |
| DXcam `grab()` | 61 | 61 | 0 | 333403 | 0 | none |

## 2 updates a second — round 2

| Arm | Frames through model /s | Detections ready p50 | Capture age p50 | Decision p50 | Mcycles/s | CPU % of one core | Peak working set |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RapidShot `grab_with_changes()` | 2.03 | 252.73 ms | 2.36 ms | 0.000 ms | 1754.6 | 52.19 | 209.1 MiB |
| DXcam `grab()` | 2.03 | 253.86 ms | 3.96 ms | 0.287 ms | 3350.9 | 99.01 | 188.5 MiB |

| Arm | Delivered | Detector calls | Skipped | Empty reads | Retained copies | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RapidShot `grab_with_changes()` | 61 | 61 | 0 | 915 | 0 | none |
| DXcam `grab()` | 61 | 61 | 0 | 340814 | 0 | none |

MSS reaches the desktop through GDI, which exposes no present timestamp, so that arm has no pixel age and its latency columns are blank. Every other arm times from the DXGI present that produced the pixels.
