# What it costs to decide whether the detector must run again

Every consecutive frame pair of each recorded application trace, with the
fastest of 5 repeats kept per pair. No inference runs here.
All three implementations reach the same verdict on every pair.

The three are not doing equal work, and the difference matters when reading
the tables. `np.array_equal` over the whole frame answers only *did anything
change* and has no early exit: NumPy compares every byte. The NumPy candidate
loop answers the same question but returns at the first tile that differs, so
on a busy frame it stops almost immediately. The native comparator answers the
harder question — *exactly which tiles changed* — which cannot exit early,
because the caller needs the whole list to search or stage those tiles. So the
native row is sometimes slower than the NumPy candidate row while delivering
strictly more information. Compare the candidate NumPy row against the whole
frame row for the cost of a pure gate, and the native row for the cost of a
gate that also tells you where to look.

## Spreadsheet — 1675 × 715, 171 pairs, 84 tiles

118 pairs byte-identical. Candidate tiles per pair: 0 median, 84 p95, of 84. Full-grid fallbacks: 3.

| Decision | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| Whole frame, NumPy | 1.2340 ms | 1.2774 ms | 1.3047 ms |
| Candidate tiles, NumPy | 0.0048 ms | 0.0368 ms | 0.5734 ms |
| Candidate tiles, native comparator | 0.0092 ms | 0.2233 ms | 0.2824 ms |

At p50, a pure gate over candidate tiles is **257× cheaper** than comparing the whole frame, and a gate that also returns the changed tile list is **134× cheaper**.

Every changed-tile set matched an independent full pixel scan.

Pairs the candidate arms called changed although the bytes were identical: 0 native, 0 Python. Neither ever called a changed pair identical.

## Browser — 1844 × 978, 298 pairs, 120 tiles

100 pairs byte-identical. Candidate tiles per pair: 120 median, 120 p95, of 120. Full-grid fallbacks: 0.

| Decision | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| Whole frame, NumPy | 1.6982 ms | 2.0239 ms | 2.0740 ms |
| Candidate tiles, NumPy | 0.0676 ms | 0.0942 ms | 0.1927 ms |
| Candidate tiles, native comparator | 0.2787 ms | 0.3109 ms | 0.5590 ms |

At p50, a pure gate over candidate tiles is **25× cheaper** than comparing the whole frame, and a gate that also returns the changed tile list is **6× cheaper**.

Every changed-tile set matched an independent full pixel scan.

Pairs the candidate arms called changed although the bytes were identical: 0 native, 0 Python. Neither ever called a changed pair identical.

## Game — 1844 × 978, 293 pairs, 120 tiles

39 pairs byte-identical. Candidate tiles per pair: 18 median, 45 p95, of 120. Full-grid fallbacks: 0.

| Decision | p50 | p95 | max |
| --- | ---: | ---: | ---: |
| Whole frame, NumPy | 1.7216 ms | 2.0465 ms | 2.5247 ms |
| Candidate tiles, NumPy | 0.0822 ms | 0.1074 ms | 0.2363 ms |
| Candidate tiles, native comparator | 0.0642 ms | 0.1811 ms | 0.5190 ms |

At p50, a pure gate over candidate tiles is **21× cheaper** than comparing the whole frame, and a gate that also returns the changed tile list is **27× cheaper**.

Every changed-tile set matched an independent full pixel scan.

Pairs the candidate arms called changed although the bytes were identical: 0 native, 0 Python. Neither ever called a changed pair identical.
