### RapidShot benchmark

- **Machine:** Acer Predator PHN16-72
- **CPU:** Intel(R) Core(TM) i9-14900HX
- **GPUs:** NVIDIA GeForce RTX 4060 Laptop GPU — capture on NVIDIA GeForce RTX 4060 Laptop GPU (single)
- **Display:** 2560×1600 at 165 Hz
- **Versions:** rapidshot 2.7.1, rapidshot-native 0.4.0 (development build), Python 3.13.15, Windows 11 (10.0.26300)
- **HDR:** off (panel does not support HDR; SDR, 8 bpc); capture receives B8G8R8A8_UNORM
- **Cross-adapter:** supported, NVIDIA GeForce RTX 4060 Laptop GPU to Microsoft Basic Render Driver, 0.74 ms per 1080p copy (to WARP: proves the mechanism, not the cost)

Screen to a (1, 3, 640, 640) FP16 tensor on CUDA; medians of 5 passes. Pixel age is from `Present()` to the tensor.

| path | unique fps | pixel age p50 / p95 | CPU per frame | passes |
| --- | ---: | ---: | ---: | ---: |
| mss | 40.6 | 33.7 / 36.5 ms | 13.6 ms | 5 |
| DXcam (DXGI) | 135.1 | 22.3 / 25.9 ms | 9.2 ms | 5 |
| DXcam (WGC) | 135.3 | 22.7 / 26.0 ms | 9.2 ms | 5 |
| RapidShot grab() | 162.0 * | 18.4 / 21.0 ms | 8.0 ms | 5 (1 flagged) |
| RapidShot grab(), WGC | 162.0 | 18.5 / 23.5 ms | 7.7 ms | 5 |
| RapidShot grab() of the window, WGC | 162.5 | 18.5 / 21.6 ms | 7.5 ms | 5 |
| RapidShot grab(), nvidia_gpu=True | 148.7 | 22.0 / 22.9 ms | 6.6 ms | 5 |
| RapidShot direct, single adapter | 87.1 | 19.1 / 20.1 ms | 6.2 ms | 5 |
| RapidShot GpuConverter | 164.6 * | 13.9 / 14.4 ms | 0.9 ms | 5 (4 flagged) |

\* read as fast as the test source presented in at least one pass, so the frame rate is a floor.

CPU per tensor (call-duration harness; its fps is bounded by its test window):

| path | CPU per frame | call p50 |
| --- | ---: | ---: |
| mss | 13.84 ms | 25.01 ms |
| DXcam (DXGI) | 9.30 ms | 7.14 ms |
| RapidShot grab() | 7.28 ms | 6.06 ms |
| RapidShot grab(), nvidia_gpu=True | 6.63 ms | 6.46 ms |
| RapidShot GpuConverter | 0.57 ms | 6.06 ms |

Memory: working set, what capture adds over the same process before its first frame, and growth as a least-squares slope after warm-up.

| library | screen | fps | working set | capture adds | growth |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | static | 38.1 | 83.7 MB | +32.8 MB | +0.001 MB/s |
| mss | scroll | 33.9 | 83.4 MB | +32.8 MB | +0.001 MB/s |
| mss | motion | 34.2 | 84.1 MB | +32.8 MB | +0.001 MB/s |
| dxcam | static | 166.6 | 104.5 MB | +25.5 MB | +0.065 MB/s |
| dxcam | scroll | 166.7 | 105.3 MB | +25.6 MB | +0.064 MB/s |
| dxcam | motion | 171.3 | 105.4 MB | +25.6 MB | +0.083 MB/s |
| rapidshot | static | 149.0 | 155.4 MB | +70.6 MB | +0.072 MB/s |
| rapidshot | scroll | 151.2 | 155.3 MB | +70.5 MB | +0.035 MB/s |
| rapidshot | motion | 153.5 | 155.3 MB | +70.5 MB | +0.034 MB/s |
| rapidshot-frame | static | 164.0 | 85.9 MB | +1.5 MB | +0.042 MB/s |
| rapidshot-frame | scroll | 164.4 | 86.3 MB | +1.4 MB | +0.050 MB/s |
| rapidshot-frame | motion | 163.7 | 86.2 MB | +1.6 MB | +0.043 MB/s |
