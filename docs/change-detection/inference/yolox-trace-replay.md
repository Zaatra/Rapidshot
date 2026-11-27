# YOLOX inference over recorded application traces

Each recorded capture is replayed through the OpenCV Zoo YOLOX-S FP32 CPU
detector three times: inferring on every delivered frame, skipping frames
proven identical by a full-frame byte compare, and skipping frames proven
identical by comparing only the candidate tiles that DXGI dirty rectangles
point at. Frame decompression sits outside every timer.

OpenCV 5.1.0, 1 thread, OpenCL off. Model 35,858,002 bytes, SHA-256 `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063`.

## Spreadsheet — 1675 × 715, 172 frames, 84 tiles

118 of 171 consecutive frames were byte-identical (69.0%).

| Arm | Detector calls | Mean per frame | p95 | Total | Decision p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always infer | 172 | 255.46 ms | 268.13 ms | 43.940 s | — |
| Skip identical, full-frame compare | 54 | 80.71 ms | 255.83 ms | 13.883 s | 1.342 ms |
| Skip identical, candidate tiles only | 54 | 78.85 ms | 252.56 ms | 13.562 s | 0.004 ms |

Detector time saved: **68.40%** by full-frame compare, **69.13%** by candidate tiles. Candidate tiles per frame: 0 median, 84 p95, of 84.

Full-grid fallbacks: 3 frames (accumulated frames).

The metadata arm reached the same decision as the full compare on every frame.

## Browser — 1844 × 978, 299 frames, 120 tiles

100 of 298 consecutive frames were byte-identical (33.6%).

| Arm | Detector calls | Mean per frame | p95 | Total | Decision p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always infer | 299 | 253.61 ms | 269.13 ms | 75.829 s | — |
| Skip identical, full-frame compare | 199 | 170.15 ms | 259.02 ms | 50.874 s | 2.166 ms |
| Skip identical, candidate tiles only | 199 | 168.34 ms | 258.03 ms | 50.334 s | 0.122 ms |

Detector time saved: **32.91%** by full-frame compare, **33.62%** by candidate tiles. Candidate tiles per frame: 120 median, 120 p95, of 120.

The metadata arm reached the same decision as the full compare on every frame.

## Game — 1844 × 978, 294 frames, 120 tiles

39 of 293 consecutive frames were byte-identical (13.3%).

| Arm | Detector calls | Mean per frame | p95 | Total | Decision p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Always infer | 294 | 252.69 ms | 263.21 ms | 74.291 s | — |
| Skip identical, full-frame compare | 255 | 218.67 ms | 254.51 ms | 64.288 s | 1.847 ms |
| Skip identical, candidate tiles only | 255 | 219.12 ms | 263.97 ms | 64.423 s | 0.165 ms |

Detector time saved: **13.46%** by full-frame compare, **13.28%** by candidate tiles. Candidate tiles per frame: 18 median, 45 p95, of 120.

The metadata arm reached the same decision as the full compare on every frame.

Every gated arm produced detections identical to the always-infer reference: same classes, boxes within 0.01 px, scores within 0.00001.
