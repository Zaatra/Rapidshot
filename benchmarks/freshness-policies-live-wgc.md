# Live capture, wgc, 2560x1600 @ 165 Hz RGB, 6 s per run

Each case ran 5 times, in shuffled order (seed 225953745). Cells are medians across runs, with the min-max range in brackets where the runs differed.

## Model inference: capture at the display's rate, 16 ms of work per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 58.2 | 29.7 (28.8-30.0) | 34.3 (33.7-35.2) | 38.6 (37.1-40.5) | 0 | 0 | 285 / 0 | 276 | 4.75 (4.12-5.12) | - | - | 684 (680-687) | 765 (764-765) |
| `latest` | 59.8 | 26.3 (25.5-26.9) | 32.1 (31.3-33.2) | 34.7 (32.6-55.3) | 0 | 0 | 0 / 505 | 111 | 3.62 (3.52-3.74) | - | - | 9 (9-9) | 0 (0-2) |

## Agent with pauses: 3 ms per frame, a 250 ms pause every second

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 138.2 | 23.4 (23.0-23.5) | 359.8 (356.4-366.7) | 524.2 (517.0-536.7) | 125 (121-125) | 0 | 0 / 0 | 144 | 3.83 (3.67-4.27) | - | - | 64 (61-67) | 364 (352-365) |
| `drop_oldest, max_age_ms=20` | 121.2 | 22.5 (22.3-22.7) | 27.1 (26.9-27.1) | 40.3 (37.3-41.1) | 3 (2-3) | 121 | 0 / 0 | 131 | 3.71 (3.33-3.84) | - | - | 35 (33-35) | 364 (352-376) |
| `latest` | 121.0 | 22.5 (22.4-22.7) | 26.7 (26.6-29.9) | 33.6 (30.4-86.4) | 0 | 0 | 0 / 146 | 109 | 3.57 (3.35-3.90) | - | - | 2 (2-2) | 1 (1-1) |
| `latest, max_age_ms=20` | 122.0 | 22.5 (22.4-22.8) | 26.9 (26.6-26.9) | 32.2 (29.1-32.9) | 0 | 0 | 0 / 143 | 116 | 3.48 (3.19-3.90) | - | - | 2 (2-2) | 0 (0-1) |

Test source: 159.4-165.1 presents a second across runs.

