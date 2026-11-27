# Exact tile release validation protocol

## Live 30 updates/s comparison

Run seven arms in five shuffled 10.2-second blocks on the Intel HP 324pe at 1920 × 1080 and 100 Hz. Repeat the prior five arms and add RapidShot exact full-frame with tight rectangles and DXcam borrowed view with a full CPU byte scan and tight rectangles. The three tight arms share the same OpenCV search. Retain cold starts in the main deadline denominator and report a second stratum excluding the first three offered states in every arm and round. A result must be correct and finish before its submission-to-result deadline and the end of the window. Desktop pixel age uses valid DXGI present timestamps. Keep counts for missing, late, and wrong states separate. Preserve raw copy, lease, CPU, memory, GPU and clock observations.

## Quiet waiting

Compare the fixed 5 ms plain RapidShot, full-frame exact, candidate tiles, and candidate tiles with tight bounds. Target 20 paired blocks and allow up to 24 fixed attempts. Shuffle arm order within a block. Report the nominal and eligible counts and preserve discarded windows. A qualifying comparison requires healthy seeded capture and zero measured deliveries in both the plain and variant windows. Startup, capture failure, and geometry shifts are recorded outside or disqualify the timed comparison.

For each eligible pair, calculate `100 × (variant cycles/s ÷ plain cycles/s − 1)`. Use the exact distribution-free two-sided 95% interval for the median from the binomial order statistics. A 5% upper bound passes and a lower bound above 5% fails. A crossing interval or fewer than ten pairs is inconclusive. CPU percentage is descriptive. No conversion from cycles to elapsed CPU time is used. Test adaptive wait separately using confirmed quiet transitions, a real wake update, and timed stop through release.

## Continuous soak

Exercise the public full-frame exact and tiled APIs on the same live capture process over 60 minutes. Sample memory externally and keep worker telemetry bounded. Report both whole-process and retained lease/page memory. Compare post-warm-up five-minute memory summaries and disclose continued growth, restart or timeout. Keep the memory results separate from the shorter timed comparison.

## Release scope

The hardware performance claims cover the tested Intel graphics system. Local release preparation follows the repository release runbook once the required evidence is available. Each live benchmark batch other than the approved continuous soak stays under ten minutes.
