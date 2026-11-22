---
name: Benchmark result
about: Share what `rapidshot benchmark` measured on your machine
title: 'Benchmark: <GPU> / <resolution> @ <refresh> Hz'
labels: 'benchmark'
assignees: ''

---

Thank you. Every published RapidShot number so far comes from one developer's
Intel and NVIDIA machines, and a result from different hardware -- especially
AMD, an HDR display, or a setup unlike those in the README -- is worth more
than any code change. Results that ran cleanly are just as welcome in the
[Benchmark results discussion](https://github.com/Zaatra/Rapidshot/discussions/categories/show-and-tell);
an issue is the place for a path that failed or a colour check that did not
pass.

**How to run it**

```
pip install "rapidshot[benchmark]"
rapidshot benchmark
```

With an NVIDIA GPU, add `cupy-cuda12x` or `cupy-cuda13x` (matching your CUDA
driver) so the tensor is finished on CUDA. Without one, the benchmark finishes
it in system memory, or on the capture GPU for GpuConverter, and says so in the
report. `--full` also includes CPU per tensor and memory. It takes
about five minutes (ten with `--full`) and fills the screen with a test
pattern; please leave the machine alone while it runs.

**The report**

Paste the contents of the `rapidshot-benchmark-<date>.md` file here, and attach
the `.json` beside it. Both are sanitised: no hostname, no username, no paths
from your profile, no device instance IDs.

<!-- paste the .md here -->

**Anything unusual?**

Laptop on battery, a second monitor, another GPU-heavy program open,
a path that failed -- anything that would help read the numbers.
