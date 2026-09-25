---
name: Benchmark result
about: Share what `rapidshot benchmark` measured on your machine
title: 'Benchmark: <GPU> / <resolution> @ <refresh> Hz'
labels: 'benchmark'
assignees: ''

---

Thank you. Every published RapidShot number so far comes from one developer's
two machines, and a result from different hardware -- especially AMD, a desktop
GPU, or a panel that is not 2560x1600 at 165 Hz -- is worth more than any code
change.

**How to run it**

```
pip install "rapidshot[benchmark]"
rapidshot benchmark
```

Add `cupy-cuda12x` or `cupy-cuda13x` (matching your CUDA driver) to measure the
GPU paths too. It takes about five minutes and fills the screen with a test
pattern; please leave the machine alone while it runs.

**The report**

Paste the contents of the `rapidshot-benchmark-<date>.md` file here, and attach
the `.json` beside it. Both are sanitised: no hostname, no username, no paths
from your profile, no device instance IDs.

<!-- paste the .md here -->

**Anything unusual?**

Laptop on battery, a second monitor, HDR on, another GPU-heavy program open,
a path that failed -- anything that would help read the numbers.
