### RapidShot benchmark

- **Machine:** Acer Predator PHN16-72
- **CPU:** Intel(R) Core(TM) i9-14900HX
- **GPUs:** Intel(R) UHD Graphics, NVIDIA GeForce RTX 4060 Laptop GPU — capture on Intel(R) UHD Graphics (hybrid)
- **Display:** 2560×1600 at 165 Hz
- **Versions:** rapidshot 2.7.1, rapidshot-native 0.4.0 (development build), Python 3.13.15, Windows 11 (10.0.26300)
- **HDR:** off (panel does not support HDR; SDR, 8 bpc); capture receives B8G8R8A8_UNORM
- **Cross-adapter:** supported, Intel(R) UHD Graphics to NVIDIA GeForce RTX 4060 Laptop GPU, 1.65 ms per 1080p copy

Screen to a (1, 3, 640, 640) FP16 tensor on CUDA; medians of 5 passes. Pixel age is from `Present()` to the tensor.

| path | unique fps | pixel age p50 / p95 | CPU per frame | passes |
| --- | ---: | ---: | ---: | ---: |
| mss | 33.1 | 39.2 / 41.8 ms | 13.8 ms | 5 |
| DXcam (DXGI) | 108.2 | 24.2 / 27.3 ms | 8.9 ms | 5 |
| DXcam (WGC) | 107.6 | 28.8 / 32.6 ms | 8.9 ms | 5 |
| RapidShot grab() | 132.7 | 21.9 / 23.2 ms | 7.1 ms | 5 |
| RapidShot grab(), WGC | 122.1 | 24.4 / 27.8 ms | 7.5 ms | 5 |
| RapidShot grab() of the window, WGC | 122.0 | 24.3 / 27.8 ms | 7.2 ms | 5 |
| RapidShot grab(), nvidia_gpu=True | 119.3 | 22.4 / 23.7 ms | 4.5 ms | 5 |
| RapidShot full frame across adapters | 81.2 | 22.2 / 24.8 ms | 4.6 ms | 5 |
| RapidShot GpuConverter + TensorTransfer | 162.7 * | 15.4 / 15.9 ms | 1.2 ms | 5 (4 flagged) |

\* read as fast as the test source presented in at least one pass, so the frame rate is a floor.

Not measured:
- RapidShot direct, single adapter: unavailable — CrossAdapterRequired: no CUDA device owns the adapter this frame was captured on. On a hybrid laptop that is expected: capture runs on the integrated GPU and CUDA only sees the discrete one. The frame has to cross adapters first — see native.cross_adapter_transfer() and ROADMAP § 6.1.
- RapidShot GpuConverter: unavailable — CrossAdapterRequired: no CUDA device is on the adapter holding this tensor (LUID aa46010000000000); 1 CUDA device(s) present, none matching. This is the ordinary hybrid-laptop case: capture is on the integrated GPU. Transfer the frame to the CUDA adapter first.

CPU per tensor (call-duration harness; its fps is bounded by its test window):

| path | CPU per frame | call p50 |
| --- | ---: | ---: |
| mss | 16.76 ms | 34.32 ms |
| DXcam (DXGI) | 10.10 ms | 9.71 ms |
| RapidShot grab() | 6.33 ms | 11.28 ms |
| RapidShot grab(), nvidia_gpu=True | 5.33 ms | 10.90 ms |
| RapidShot full frame across adapters | 4.59 ms | 10.82 ms |
| RapidShot GpuConverter + TensorTransfer | 0.96 ms | 10.01 ms |

Memory: working set, what capture adds over the same process before its first frame, and growth as a least-squares slope after warm-up.

| library | screen | fps | working set | capture adds | growth |
| --- | --- | ---: | ---: | ---: | ---: |
| mss | static | 38.3 | 84.1 MB | +32.8 MB | +0.001 MB/s |
| mss | scroll | 38.4 | 84.2 MB | +32.8 MB | +0.001 MB/s |
| mss | motion | 37.2 | 83.7 MB | +32.8 MB | +0.001 MB/s |
| dxcam | static | 163.7 | 104.0 MB | +23.9 MB | +0.037 MB/s |
| dxcam | scroll | 164.2 | 104.4 MB | +24.2 MB | +0.035 MB/s |
| dxcam | motion | 163.9 | 103.9 MB | +23.8 MB | +0.037 MB/s |
| rapidshot | static | 109.6 | 175.2 MB | +70.5 MB | +0.029 MB/s |
| rapidshot | scroll | 110.2 | 174.9 MB | +70.5 MB | +0.028 MB/s |
| rapidshot | motion | 110.6 | 175.2 MB | +70.5 MB | +0.026 MB/s |
| rapidshot-frame | static | 158.9 | 105.8 MB | +1.4 MB | +0.044 MB/s |
| rapidshot-frame | scroll | 160.9 | 105.7 MB | +1.4 MB | +0.043 MB/s |
| rapidshot-frame | motion | 162.4 | 105.9 MB | +1.1 MB | +0.043 MB/s |
