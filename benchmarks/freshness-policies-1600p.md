# Queue policies, 2560x1600 RGB, 6 s per run

Each case ran 5 times; every run of every case went in one shuffled order (seed 122252545). Cells are medians across runs, with the min-max range in brackets where the runs differed.

## Model inference: screen and capture at 240 Hz, 16 ms of work per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 59.7 | 6.0 (5.7-6.2) | 8.9 (8.8-9.2) | 10.1 (9.8-10.2) | 0 | 0 | 973 / 0 | 47 | 1.66 (1.62-1.71) | 1351 (1346-1356) | 725 (724-727) | 764 (764-764) |
| `latest` | 60.5 | 4.7 (4.5-5.0) | 7.7 (7.4-7.8) | 9.0 (8.8-10.1) | 0 | 0 | 0 / 1021 | 57 | 0.66 (0.64-0.70) | 0 | 10 (10-10) | 0 (0-0) |

## Agent with pauses: 120 Hz, 3 ms per frame, a 250 ms pause every second

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 119.2 | 6.3 (5.2-6.5) | 317.4 (302.3-322.5) | 453.7 (439.3-465.8) | 115 (115-116) | 0 | 0 / 0 | 5 | 1.12 (0.99-1.19) | 199 (193-204) | 52 (50-53) | 329 (329-341) |
| `drop_oldest, max_age_ms=20` | 100.3 | 5.0 (4.5-5.7) | 8.8 (8.7-8.9) | 25.2 (21.2-26.5) | 2 (2-3) | 112 | 0 / 0 | 5 | 1.02 (0.95-1.07) | 115 (114-118) | 31 (31-32) | 329 (329-341) |
| `latest` | 99.7 | 5.6 (4.4-5.9) | 8.7 (8.7-8.8) | 12.9 (10.1-17.4) | 0 | 0 | 0 / 116 | 6 | 0.89 (0.87-0.94) | 0 | 2 (2-2) | 0 (0-0) |
| `latest, max_age_ms=20` | 100.2 | 5.3 (4.7-5.8) | 8.6 (8.5-9.0) | 12.5 (9.6-14.6) | 0 | 0 | 0 / 116 | 5 | 0.85 (0.79-0.94) | 0 | 2 (2-2) | 0 (0-0) |

