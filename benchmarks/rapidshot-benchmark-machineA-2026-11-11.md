### RapidShot benchmark

- **Machine:** HP HP ProDesk 4 Tower G1i Desktop AI PC
- **CPU:** Intel(R) Core(TM) Ultra 5 235
- **GPUs:** Intel(R) Graphics — capture on Intel(R) Graphics (single)
- **Display:** 1920×1080 at 100 Hz
- **Versions:** rapidshot 2.7.1, rapidshot-native 0.4.0 (development build), Python 3.14.6, Windows 11 (10.0.26300)
- **HDR:** off (panel does not support HDR; SDR, 8 bpc); capture receives B8G8R8A8_UNORM
- **Cross-adapter:** supported, Intel(R) Graphics to Microsoft Basic Render Driver, 0.65 ms per 1080p copy (to WARP: proves the mechanism, not the cost)

Screen to a (1, 3, 640, 640) FP16 tensor, **no CUDA**: in system memory after CPU capture, on the capture GPU (D3D12) after GpuConverter. Not comparable with the CUDA table in the README; medians of 5 passes. Pixel age is from `Present()` to the tensor.

| path | tensor in | unique fps | pixel age p50 / p95 | CPU per frame | passes |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | system memory | 33.3 † | 42.4 / 45.4 ms | 15.8 ms | 5 (2 flagged) |
| DXcam (DXGI) | system memory | 86.5 † | 37.2 / 42.5 ms | 11.9 ms | 5 (1 flagged) |
| DXcam (WGC) | system memory | 85.8 † | 37.9 / 42.9 ms | 11.7 ms | 5 (2 flagged) |
| RapidShot grab() | system memory | 100.0 *† | 31.1 / 32.5 ms | 9.6 ms | 5 (3 flagged) |
| RapidShot grab(), WGC | system memory | 99.0 *† | 30.8 / 31.8 ms | 9.6 ms | 5 (3 flagged) |
| RapidShot grab() of the window, WGC | system memory | 98.7 | 30.8 / 32.0 ms | 9.4 ms | 5 |
| RapidShot GpuConverter | capture GPU (D3D12) | 100.0 † | 22.4 / 22.9 ms | 0.4 ms | 5 (1 flagged) |

\* read as fast as the test source presented in at least one pass, so the frame rate is a floor.
† other programs used the CPU during at least one pass (median above 15% outside the benchmark); timings may be inflated. The harness treats this as a warning, not proof, and keeps the pass.

Not measured:
- RapidShot grab(), nvidia_gpu=True, RapidShot direct, single adapter: ends in an FP16 tensor on CUDA, which needs an NVIDIA GPU and CuPy for your CUDA version, e.g. pip install cupy-cuda12x

CPU per tensor (call-duration harness): not measured; it ends on CUDA. The CPU per frame above is the same cost, from the pixel-age frames.

Memory: working set, what capture adds over the same process before its first frame, and growth as a least-squares slope after warm-up.

| library | screen | fps | working set | capture adds | growth |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | static | 49.9 | 72.1 MB | +20.5 MB | +0.001 MB/s |
| mss | scroll | 49.9 | 71.4 MB | +20.5 MB | +0.001 MB/s |
| mss | motion | 49.9 | 71.5 MB | +20.5 MB | +0.001 MB/s |
| dxcam | static | 99.8 | 95.4 MB | +9.5 MB | +0.044 MB/s |
| dxcam | scroll | 98.9 | 95.3 MB | +10.2 MB | +0.082 MB/s |
| dxcam | motion | 99.7 | 95.1 MB | +9.9 MB | +0.044 MB/s |
| rapidshot | static | 100.0 | 119.8 MB | +29.1 MB | +0.027 MB/s |
| rapidshot | scroll | 99.8 | 120.1 MB | +28.7 MB | +0.034 MB/s |
| rapidshot | motion | 100.0 | 119.9 MB | +29.1 MB | +0.028 MB/s |
| rapidshot-frame | static | 100.2 | 92.5 MB | +1.1 MB | +0.034 MB/s |
| rapidshot-frame | scroll | 99.3 | 92.3 MB | +1.6 MB | +0.034 MB/s |
| rapidshot-frame | motion | 100.2 | 92.5 MB | +1.7 MB | +0.034 MB/s |
