### RapidShot benchmark

- **Machine:** HP HP OmniBook X Flip Laptop 16-as0xxx
- **CPU:** Intel(R) Core(TM) Ultra 5 226V
- **GPUs:** Intel(R) Arc(TM) 130V GPU (8GB) — capture on Intel(R) Arc(TM) 130V GPU (8GB) (single)
- **Display:** 1920×1200 at 60 Hz
- **Versions:** rapidshot 2.7.1, rapidshot-native 0.4.0 (development build), Python 3.12.10, Windows 11 (10.0.26300)
- **HDR:** off (panel does not support HDR; SDR, 8 bpc); capture receives B8G8R8A8_UNORM
- **Cross-adapter:** supported, Intel(R) Arc(TM) 130V GPU (8GB) to Microsoft Basic Render Driver, 0.49 ms per 1080p copy (to WARP: proves the mechanism, not the cost)

Screen to a (1, 3, 640, 640) FP16 tensor, **no CUDA**: in system memory after CPU capture, on the capture GPU (D3D12) after GpuConverter. Not comparable with the CUDA table in the README; medians of 5 passes. Pixel age is from `Present()` to the tensor.

| path | tensor in | unique fps | pixel age p50 / p95 | CPU per frame | passes |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | system memory | 29.9 | 58.8 / 75.3 ms | 17.6 ms | 5 |
| DXcam (DXGI) | system memory | 51.2 | 47.4 / 57.4 ms | 20.3 ms | 5 |
| DXcam (WGC) | system memory | 59.8 | 46.6 / 55.6 ms | 13.5 ms | 5 |
| RapidShot grab() | system memory | 59.1 | 44.3 / 45.0 ms | 10.6 ms | 5 |
| RapidShot grab(), WGC | system memory | 53.4 | 44.6 / 48.9 ms | 11.4 ms | 5 |
| RapidShot grab() of the window, WGC | system memory | 53.0 | 44.5 / 48.0 ms | 10.4 ms | 5 |
| RapidShot GpuConverter | capture GPU (D3D12) | 59.5 | 36.3 / 37.1 ms | 1.1 ms | 5 |

Not measured:
- RapidShot grab(), nvidia_gpu=True, RapidShot direct, single adapter: ends in an FP16 tensor on CUDA, which needs an NVIDIA GPU and CuPy for your CUDA version, e.g. pip install cupy-cuda12x

CPU per tensor (call-duration harness): not measured; it ends on CUDA. The CPU per frame above is the same cost, from the pixel-age frames.

Memory: working set, what capture adds over the same process before its first frame, and growth as a least-squares slope after warm-up.

| library | screen | fps | working set | capture adds | growth |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | static | 30.0 | 66.4 MB | +18.5 MB | +0.002 MB/s |
| mss | scroll | 30.0 | 66.5 MB | +18.5 MB | +0.002 MB/s |
| mss | motion | 30.0 | 66.3 MB | +18.5 MB | +0.002 MB/s |
| dxcam | static | 68.9 | 95.0 MB | +9.9 MB | +0.044 MB/s |
| dxcam | scroll | 70.6 | 94.6 MB | +9.8 MB | +0.017 MB/s |
| dxcam | motion | 68.0 | 94.9 MB | +9.5 MB | +0.058 MB/s |
| rapidshot | static | 59.6 | 122.9 MB | +31.2 MB | +0.016 MB/s |
| rapidshot | scroll | 59.9 | 123.1 MB | +31.7 MB | +0.016 MB/s |
| rapidshot | motion | 59.6 | 123.3 MB | +31.1 MB | +0.039 MB/s |
| rapidshot-frame | static | 59.9 | 93.0 MB | +1.5 MB | +0.000 MB/s |
| rapidshot-frame | scroll | 59.6 | 92.7 MB | +0.9 MB | +0.000 MB/s |
| rapidshot-frame | motion | 59.5 | 92.9 MB | +0.9 MB | +0.000 MB/s |
