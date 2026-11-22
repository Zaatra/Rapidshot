# Queue policies, 1920x1080 RGB, 6 s per run

Each case ran 5 times; every run of every case went in one shuffled order (seed 826159159). Cells are medians across runs, with the min-max range in brackets where the runs differed.

## Model inference: screen and capture at 240 Hz, 16 ms of work per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 60.0 | 4.7 (4.3-4.9) | 7.8 (7.5-7.9) | 9.6 (9.3-12.4) | 0 | 0 | 966 / 0 | 52 | 0.84 (0.82-0.84) | 1345 (1341-1354) | 367 (366-368) | 387 (387-387) |
| `latest` | 60.5 | 4.5 (4.2-4.6) | 7.6 (7.4-7.7) | 8.9 (8.6-9.0) | 0 | 0 | 0 / 1019 | 58 | 0.44 (0.43-0.45) | 0 | 5 (5-5) | 0 (0-0) |

## Agent with pauses: 120 Hz, 3 ms per frame, a 250 ms pause every second

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 119.2 | 6.2 (5.3-6.7) | 306.5 (300.6-312.8) | 436.6 (429.8-457.2) | 115 (114-116) | 0 | 0 / 0 | 4 | 0.71 (0.70-0.73) | 192 (190-197) | 25 (25-26) | 167 (167-173) |
| `drop_oldest, max_age_ms=20` | 99.8 | 4.4 (4.2-4.6) | 8.6 (8.4-8.8) | 23.4 (12.8-24.8) | 3 (0-4) | 113 | 0 / 0 | 7 | 0.67 (0.65-0.68) | 115 (112-116) | 16 (15-16) | 167 (167-173) |
| `latest` | 99.5 | 4.8 (3.2-5.2) | 8.5 (8.0-8.6) | 11.8 (9.8-12.7) | 0 | 0 | 0 / 116 | 6 | 0.55 (0.49-0.56) | 0 | 1 (1-1) | 0 (0-0) |
| `latest, max_age_ms=20` | 99.8 | 4.3 (3.5-5.3) | 8.3 (8.1-8.6) | 13.6 (9.2-15.0) | 0 | 0 | 0 / 114 | 8 | 0.54 (0.52-0.60) | 0 | 1 (1-1) | 0 (0-0) |

## Capture stalls 300 ms every 2 s (a rebuild), 120 Hz, 20 ms per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 48.2 | 9.5 (9.0-10.1) | 263.4 (242.2-267.6) | 512.3 (497.2-515.1) | 27 (27-28) | 0 | 292 / 0 | 76 | 1.08 (0.96-1.16) | 620 (613-623) | 343 (342-343) | 387 (387-387) |
| `latest` | 44.0 | 8.6 (8.3-9.1) | 14.3 (14.0-14.9) | 17.0 (16.3-20.7) | 0 | 0 | 0 / 379 | 77 | 0.56 (0.54-0.63) | 0 | 4 (4-4) | 0 (0-0) |
| `latest, max_age_ms=50` | 44.2 | 8.8 (7.8-9.1) | 14.7 (13.8-15.3) | 17.8 (16.4-19.6) | 0 | 0 | 0 / 376 | 80 | 0.58 (0.53-0.60) | 0 | 4 (4-4) | 0 (0-0) |

## Mostly idle screen: 2 updates a second, capture at 60 Hz, 5 ms per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `latest` | 2.0 | 9.6 (9.1-10.3) | 15.3 (14.4-16.4) | 15.3 (14.4-16.4) | 0 | 0 | 0 / 0 | 0 | 0.92 (0.91-1.08) | 0 | 0 | 0 (0-0) |
| `latest, max_age_ms=20` | 2.0 | 9.7 (7.7-11.2) | 16.0 (14.5-17.4) | 16.0 (14.5-17.4) | 0 | 0 | 0 / 0 | 0 | 0.89 (0.85-1.01) | 0 | 0 | 0 (0-0) |

## Recording a 60 Hz screen with a 30 fps writer, read in order

Reader: `get_next_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 29.5 | 1063.1 (1061.8-1064.6) | 1077.7 (1076.8-1105.2) | 1084.2 (1083.1-1112.0) | 0 | 0 | 119 / 0 | 0 | 1.52 (1.40-1.57) | 338 (336-339) | 311 (310-311) | 386 (381-387) |
| `block` | 29.5 | 1513.4 (1494.8-1534.4) | 2179.9 (2172.2-2189.7) | 2182.7 (2177.3-2198.7) | 0 | 0 | 0 / 0 | 118 | 1.60 (1.32-1.66) | 223 (222-224) | 310 (308-311) | 375 (375-375) |

