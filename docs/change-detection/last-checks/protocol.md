# Final RapidShot release checks

Freeze this protocol before acquiring any new timing data. The target is the HP 324pe at 1920 × 1080 and 100 Hz on Intel graphics. Archive the previously scored runs unchanged.

## Matched DXcam waiting comparison

Run five shuffled, 10.2-second windows per arm on the same sparse 30 updates/s fixture. Compare RapidShot candidate tiles and tight search with three DXcam 0.3.0 arms. Two use stock `grab_view()` and the same independent full CPU BGRA comparison and tight OpenCV task, with only the external retry delay changed between 0.5 ms and 2 ms. The third uses stock `start(target_fps=100, video_mode=False)` and owned, timestamped `get_latest_frame()` results, with the independent CPU comparison and the same tight task. Limit the campaign to 570 seconds. Keep benchmark code and synthetic outputs local.

Account for every available producer frame and consumer return separately. Charge the one-shot history copy or the owned streaming return as applicable. Report whole-process CPU and cycles including producer threads, elapsed time spent in empty retries, result deadlines, desktop pixel age, and GPU and CPU pixel writes. One-shot retry comparisons vary just the requested retry delay. Streaming changes producer pacing and frame ownership as well as how the consumer waits. The headline includes cold starts. A separate steady-state measure excludes the first three offered states per trial for every arm.

## Idle extension

The earlier study completed 24 blocks and found 19 eligible full-frame exact pairs and 20 eligible candidate-tile pairs. Tight bounds passed and remain in the earlier result. Add 24 fixed three-arm blocks, numbered 25–48, with a predetermined shuffle in each block and six blocks per batch. Each arm uses a 10.05-second measurement window at the same 5 ms timeout setting as the original trial. Retain all measured blocks. Eligibility requires a healthy initialized worker, the applicable seeded history, and zero desktop presents or deliveries during each measured window. A host interruption can be recovered by completing unmeasured work without rerunning outcomes. All batches have a 570-second cap.

Use percent change in process cycles per second within each eligible pair against plain capture. At the final look, show the original, new-only and pooled medians and exact central 95% order-statistic confidence intervals. The pooled release gate keeps the original 5% ceiling and minimum ten eligible pairs. An upper endpoint at or below 5% passes, a lower endpoint above 5% fails, and a crossing interval is inconclusive. Complete all planned blocks before using any new outcome to decide. The displayed intervals carry fixed-look two-sided coverage. Under the same sampling assumptions the two looks each have at most 2.5% directional false-pass error per path, so the union bound across the two looks is 5% per path.

## Quiet-path mechanism

Run a separate short probe with two shuffled rounds of plain, full-frame exact and candidate tiles at 5 ms. Seed before observation. Count observed `AcquireNextFrame` timeouts and other outcomes, pool and semaphore operations, staging copies and maps, and comparison or page writes only within the timed window. Exclude instrumented windows from the CPU gate. Pixel-work counts alone do not determine Python bookkeeping or total process cycles.

## Release scope

Report the observed verdict for each gate on this Intel system. The product version, tag and deployment are handled separately by the owner.
