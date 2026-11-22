# Live capture, dxgi, 2560x1600 @ 165 Hz RGB, 6 s per run

Each case ran 5 times, in shuffled order (seed 535679894). Cells are medians across runs, with the min-max range in brackets where the runs differed.

## Model inference: capture at the display's rate, 16 ms of work per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 58.3 | 28.4 (27.9-28.6) | 33.8 (33.5-33.9) | 36.3 (35.2-37.7) | 0 | 0 | 260 / 0 | 315 | 5.50 (5.17-5.96) | - | - | 671 (668-687) | 764 (764-765) |
| `latest` | 59.8 | 26.0 (25.4-26.3) | 32.3 (30.6-32.5) | 34.1 (33.4-34.9) | 0 | 0 | 0 / 434 | 197 | 4.55 (4.25-4.85) | - | - | 9 (9-9) | 1 (0-1) |

## Agent with pauses: 3 ms per frame, a 250 ms pause every second

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 125.8 | 22.9 (22.8-23.0) | 352.1 (338.8-353.2) | 501.8 (489.9-506.7) | 117 (113-119) | 0 | 0 / 0 | 234 | 4.81 (4.64-5.19) | - | - | 56 (52-58) | 352 (341-365) |
| `drop_oldest, max_age_ms=20` | 108.5 | 22.4 (22.2-22.5) | 24.1 (23.9-24.7) | 41.8 (37.4-47.0) | 3 (1-3) | 116 | 0 / 0 | 218 | 4.77 (4.39-5.04) | - | - | 33 (32-33) | 352 (342-353) |
| `latest` | 105.7 | 22.5 (22.4-22.6) | 23.9 (23.7-26.3) | 29.5 (28.9-35.8) | 0 | 0 | 0 / 125 | 228 | 4.73 (4.42-4.98) | - | - | 2 (2-2) | 1 (0-1) |
| `latest, max_age_ms=20` | 107.5 | 22.5 (22.4-22.6) | 23.9 (23.6-26.5) | 29.0 (28.8-41.9) | 0 | 0 | 0 / 127 | 215 | 4.30 (4.13-4.53) | - | - | 2 (2-2) | 1 (1-2) |

## Recording with a 30 fps writer, read in order

Reader: `get_next_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `drop_oldest` | 29.3 | 593.3 (582.3-599.0) | 609.5 (599.4-632.2) | 615.6 (610.5-644.4) | 0 | 0 | 426 / 0 | 328 | 5.64 (5.50-5.94) | - | - | 699 (697-700) | 765 (764-765) |
| `block` | 29.3 | 2186.4 (2185.2-2189.2) | 2196.6 (2194.8-2200.2) | 2199.3 (2197.6-2203.6) | 0 | 0 | 0 / 0 | 746 | 5.73 (5.01-6.58) | - | - | 696 (696-698) | 740 (740-741) |

## Mostly idle screen: the source presents twice a second, 5 ms per frame

Reader: `get_latest_frame()`

| policy | consumed fps | age p50 ms | age p95 ms | age max ms | older than previous | expired | dropped (consumer / replaced) | capture_drops | capture CPU ms/frame | grab ms/frame | fallback allocs | queued MB mean | private MB peak |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `latest` | 2.2 | 16.8 (15.7-17.2) | 19.0 (17.7-21.1) | 19.0 (17.7-21.1) | 0 | 0 | 0 / 0 | 0 | 27.64 (20.43-33.65) | - | - | 0 (0-0) | 0 (0-0) |
| `latest, max_age_ms=20` | 2.2 | 16.8 (16.7-16.9) | 21.0 (20.2-24.5) | 21.0 (20.2-24.5) | 0 | 0 | 0 / 0 | 0 | 22.84 (14.42-51.68) | - | - | 0 | 0 (0-0) |

Test source: 2.1-165.1 presents a second across runs.

