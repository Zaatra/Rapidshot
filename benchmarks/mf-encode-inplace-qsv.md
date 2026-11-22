# In-place hardware encoding through Media Foundation, 2026-11-17

`native/core/src/bin/mf_encode.rs`: capture (Desktop Duplication) -> copy into
a shareable texture -> `Converter12` NV12 **texture** -> GPU copy into a 4-slot
ring -> the hardware H.264 MFT on the capture adapter, fed `IMFSample`s that
wrap the textures (`MFCreateDXGISurfaceBuffer`) through an
`IMFDXGIDeviceManager`. No readback, no CPU copy of a pixel. Events ("need
input", "have output") on their own thread, MTA.

Machine B, hybrid: the Intel UHD drives the 2560x1600 @ 165 Hz panel. The
NVIDIA H.264 Encoder MFT refuses to activate (`0x8000FFFF`); the Intel Quick
Sync Video H.264 Encoder MFT activates and is used. Settings, each reported
applied by the encoder: `AVLowLatencyMode` true, CBR 20 Mbps, no B-frames,
GOP 330, `MF_LOW_LATENCY`.

Judged as the PyAV runs are: the stream is decoded with FFmpeg's software
decoder and every frame's ID read back from the decoded picture; latency is
from the test source's `Present()` of that ID to the packet coming out. A
`--verify` run read back every 10th encoder input: PSNR 36.9 dB median
(35.2 min) against it, 0 of 20 sampled frames the wrong frame.

## Five runs, default settings

| run | frames | latency p50 / p95 ms | encode (submit -> packet) p50 ms | capture -> NV12 texture p50 ms | wait for "need input" p50 ms | fps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 195 | 45.2 / 48.4 | 14.86 | 7.49 | 7.88 | 65.0 |
| 2 | 195 | 45.2 / 48.6 | 14.82 | 7.45 | 7.90 | 65.0 |
| 3 | 193 | 45.1 / 48.5 | 14.93 | 7.59 | 7.82 | 64.3 |
| 4 | 194 | 45.3 / 48.6 | 14.91 | 7.51 | 7.91 | 64.7 |
| 5 | 195 | 45.2 / 48.3 | 14.89 | 7.51 | 7.81 | 65.0 |
| **median** | | **45.2** / 48.5 | **14.89** | 7.51 | 7.88 | **65.0** |

Every frame of every run decoded, its ID read, in order.

## Why 15 ms: one variable at a time

| change | encode p50 | fps | verdict |
| --- | ---: | ---: | --- |
| none (above) | 14.89 ms | 65.0 | |
| `AVEncCommonQualityVsSpeed` = 0 (fastest) | 14.93 ms | 64.7 | accepted, no effect |
| `AVEncCommonQualityVsSpeed` = 50 | 14.80 ms | 65.0 | accepted, no effect |
| `--encode-only`: copies of one frame, no capture or conversion on the GPU | 14.91-14.94 ms | 64.7 | not contention |
| frame rate given to the encoder 60 instead of 165 | 14.85 ms | 64.7 | not pacing |
| `AVLowLatencyMode` false | 143-156 ms | 73-78 | pipelines frames: useless live |

The encoder accepts the next input only once the previous frame is out, and
needs ~15 ms a frame at 2560x1600 however it is configured. FFmpeg's
`h264_qsv` (oneVPL) on the same GPU encoded in 9.7 ms, and NVENC in 3.3-4.1
ms (`encode-latency-*.md`), so the in-place path through Media Foundation,
correct as it is, lands at 45.2 ms against 25.7 ms for texture NV12 read back
into NVENC through PyAV.
