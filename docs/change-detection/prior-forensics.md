# Deadline forensics for the previous 30/s run

The five missed deadlines in the tight tile path occurred among the first three offered updates of the two trials. The scorer uses submission-to-result time against a 33.33 ms budget. Display age in the earlier report starts later, at the desktop present.

| Round | Offered index | Marker | Outcome | Submission to result | Search |
| ---: | ---: | ---: | --- | ---: | ---: |
| 1 | 0 | 4096 | Correct and late | 83.39 ms | 68.34 ms |
| 1 | 1 | 4097 | Marker absent from the worker observations | — | — |
| 2 | 0 | 4096 | Correct and late | 94.61 ms | 70.88 ms |
| 2 | 1 | 4097 | Marker absent from the worker observations | — | — |
| 2 | 2 | 4098 | Correct and late | 35.42 ms | 2.84 ms |

The first search in each round initialized the full search state. The next offered marker arrived while that first search was still running. The records are consistent with intermediate presentation states being skipped while the worker was busy. They do not locate precisely where the two absent states were lost. For the final miss, submission to desktop present took **15.27 ms**, present to capture return **17.23 ms**, and capture return to result **2.92 ms**. Round one (`../results-tiled-release/measurements/matched-30/trials/r001-shared_tiles-sparse-030-rapidshot_tight.json`, raw data, not published) · Round two (`../results-tiled-release/measurements/matched-30/trials/r002-shared_tiles-sparse-030-rapidshot_tight.json`, raw data, not published)

## Candidate tiles with the previous search strategy

Candidate tiles with the previous regional search had 398 correct on time compared with 465 for full-frame exact capture. Among common markers, there were 81 and 62 exact-only deadline successes across the two rounds, versus 45 and 31 in the opposite direction. In the exact-only group, the candidate arm's median extra submission-to-present time was **6.47 and 7.96 ms**. Candidate present-to-return was **0.50 and 0.33 ms shorter**. Its crop assembly took about **0.22 ms** at the median in these rows. The comparison arms ran in different live windows. The timings locate the measured differences in those windows, while the cause of the upstream timing shift remains unresolved. Earlier matched report and raw data (`../results-tiled-release/measurements/matched-30/report.md`, raw data, not published)
