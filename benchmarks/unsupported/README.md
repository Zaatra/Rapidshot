# Unsupported benchmarks — do not quote these numbers

These scripts are kept because deleting them would lose the record of what was
wrong with them, and because their numbers have circulated. **None of them is a
supported entry point.** They do not write to the result store, they record no
machine or environment, and each has a specific defect that makes its output
uncomparable — in several cases uncomparable with *itself* between runs.

The supported entry points are:

| What you want to measure | Use |
| --- | --- |
| Microbenchmarks, release gate | `benchmarks/perf_suite.py` |
| Capture: OS pixels → CPU array, across libraries | `benchmarks/compare_libraries.py` |
| AI ingestion: OS pixels → model-ready CUDA tensor | `benchmarks/ai_ingestion.py` |
| Pixel age against a shared clock, and inference | `benchmarks/section7.py` |
| The resumable mode/workload matrix | `benchmarks/section7_suite.py` |
| Process memory attributable to capture | `benchmarks/memory_profile.py` |

---

## `*_max_fps.py` — bettercam, dxcam, mss, d3dshot, rapidshot

Five scripts of the same shape, and the shape is the problem.

**They charge device creation to the measured interval.** Every one takes
`start_time = perf_counter()` *before* constructing the capture object, takes a
second timestamp after it into a variable named `start`, never uses `start`, and
finally divides the frame count by `perf_counter() - start_time`. DXGI
duplication setup is tens to hundreds of milliseconds; `mss.mss()` is nearly
free. So the scripts systematically penalise exactly the libraries that do real
initialisation, and the size of the penalty depends on the machine.

**The bias does not even run in a consistent direction.**
`rapidshot_max_fps.py` charges itself *more* than the competitor scripts charge
themselves: its measured window also contains a warm-up `grab()` and a literal
`time.sleep(0.1)`. At the 1000-frame default that is a ~100 ms handicap the
other four do not carry. Numbers from these five cannot be compared to each
other in either direction.

**They measure whatever happened to be on screen.** There is no controlled
motion source, so throughput is set by how much the desktop changed during the
run. DXGI returns nothing when nothing changed. This is the failure
`benchmarks/motion_source.py` exists to warn about, and which ROADMAP § 7.0
records having hit three separate times.

**They report a single mean from a single pass**, with no percentiles, no
warm-up exclusion, no verification that the libraries were asked for the same
pixels in the same format, and no record of the machine, display mode or
refresh rate.

`d3dshot_max_fps.py` additionally targets `d3dshot`, an abandoned package.

## `granular_performance_test.py`

**It reaches past the public API.** It drives `Duplicator` and `NumpyProcessor`
directly, so it measures internals rather than what a caller receives, and it
will keep drifting from the shipped path every time that path changes.

**It averages two arrays with different denominators and then adds them.**
`capture_times` gets an entry on *every* iteration, including the ones where
`AcquireNextFrame` timed out (`--timeout_ms` defaults to 10) and the ones where
no new content arrived. `process_times` gets an entry only when a frame was
actually processed. The script then reports
`fps = 1 / (mean(capture_times) + mean(process_times))`, combining a mean over
all attempts with a mean over successes. On a static desktop the first term is
dominated by 10 ms timeouts and the reported "FPS" describes the polling
interval, not the capture path.

The script's own comments say it does not know how to combine the two arrays —
*"This is complex to align post-loop without more state"* — which is accurate,
and is why the number it prints should not be used.

This is the same defect that was found and fixed once already in
`compare_libraries.py`, where the interval clock advanced on every grab, hit or
miss; `tests/test_benchmark_reporting.py` records that correction.

**It also reports means only**, over a fixed frame count rather than a fixed
duration, from a single pass, against an uncontrolled desktop.
