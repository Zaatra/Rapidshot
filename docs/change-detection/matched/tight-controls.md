# Matched tight-search controls

All three tight arms use the same OpenCV matcher. RapidShot exact and DXcam can borrow crop views from their current arrays. The tiled arm materializes crops and charges those copies. DXcam reads every tile for the independent CPU comparison and copies changed full frames into its private snapshot.

Headlines include every offered update at 30/s. Steady-state excludes the first 3 offered updates of each round for every arm. A missed result is classified as wrong before late, so the categories are mutually exclusive. Deadlines start at submission. Pixel age starts at DXGI present. Latency quantiles pool eligible timestamps across rounds and exclude undelivered results.

## All updates

| Arm | Rounds | Correct on time | Late | Wrong | Undelivered | No result | Submission to result p50/p95 ms | Present to result p50/p95 ms | Search p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_plain | 5 | 893/1530 | 632 | 0 | 5 | 0 | 31.82/43.31 | 17.19/24.45 | 13.06/18.30 |
| rapidshot_exact | 5 | 1091/1530 | 435 | 0 | 4 | 0 | 29.54/41.21 | 15.87/23.27 | 13.05/18.22 |
| rapidshot_tiled | 5 | 1061/1530 | 464 | 0 | 5 | 0 | 29.65/42.20 | 15.22/24.04 | 13.18/19.13 |
| rapidshot_tight | 5 | 1517/1530 | 8 | 0 | 5 | 0 | 19.06/28.29 | 5.09/9.07 | 1.99/3.13 |
| dxcam_view | 5 | 939/1530 | 586 | 0 | 5 | 0 | 31.35/41.73 | 17.67/24.96 | 13.08/18.69 |
| rapidshot_exact_tight | 5 | 1504/1530 | 21 | 0 | 5 | 0 | 21.15/29.71 | 6.33/10.06 | 1.97/3.18 |
| dxcam_view_tight | 5 | 1514/1530 | 11 | 0 | 5 | 0 | 21.32/29.29 | 7.30/10.28 | 1.94/3.03 |

## First three offered per round

| Arm | Rounds | Correct on time | Late | Wrong | Undelivered | No result | Submission to result p50/p95 ms | Present to result p50/p95 ms | Search p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_plain | 5 | 0/15 | 10 | 0 | 5 | 0 | 62.00/88.49 | 52.88/73.79 | 42.22/68.30 |
| rapidshot_exact | 5 | 1/15 | 10 | 0 | 4 | 0 | 63.18/88.44 | 41.63/74.13 | 16.01/68.10 |
| rapidshot_tiled | 5 | 2/15 | 8 | 0 | 5 | 0 | 63.85/93.08 | 51.73/78.32 | 43.89/76.23 |
| rapidshot_tight | 5 | 3/15 | 7 | 0 | 5 | 0 | 60.01/94.73 | 47.57/76.27 | 36.73/73.01 |
| dxcam_view | 5 | 1/15 | 9 | 0 | 5 | 0 | 68.37/98.52 | 55.44/74.55 | 41.68/69.77 |
| rapidshot_exact_tight | 5 | 5/15 | 5 | 0 | 5 | 0 | 54.13/87.06 | 45.43/72.67 | 34.82/67.26 |
| dxcam_view_tight | 5 | 2/15 | 8 | 0 | 5 | 0 | 61.75/94.49 | 44.50/75.12 | 38.65/68.63 |

## Steady-state

| Arm | Rounds | Correct on time | Late | Wrong | Undelivered | No result | Submission to result p50/p95 ms | Present to result p50/p95 ms | Search p50/p95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rapidshot_plain | 5 | 893/1515 | 622 | 0 | 0 | 0 | 31.76/43.08 | 17.17/24.12 | 13.05/18.26 |
| rapidshot_exact | 5 | 1090/1515 | 425 | 0 | 0 | 0 | 29.50/41.06 | 15.86/23.04 | 13.04/18.13 |
| rapidshot_tiled | 5 | 1059/1515 | 456 | 0 | 0 | 0 | 29.62/41.79 | 15.20/23.73 | 13.17/18.93 |
| rapidshot_tight | 5 | 1514/1515 | 1 | 0 | 0 | 0 | 19.03/28.12 | 5.06/8.97 | 1.99/3.10 |
| dxcam_view | 5 | 938/1515 | 577 | 0 | 0 | 0 | 31.32/41.61 | 17.62/24.62 | 13.07/18.51 |
| rapidshot_exact_tight | 5 | 1499/1515 | 16 | 0 | 0 | 0 | 21.07/29.62 | 6.31/9.89 | 1.97/3.15 |
| dxcam_view_tight | 5 | 1512/1515 | 3 | 0 | 0 | 0 | 21.30/29.13 | 7.28/10.22 | 1.93/2.99 |

[CPU, memory and copy measurements](report.md) · Per-round strata and individual misses (`matched/strata.json`, raw data, not published) · Raw trials (`matched/trials`, raw data, not published)
