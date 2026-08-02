## What this changes

<!-- And why. The reasoning is the expensive part to reconstruct later. -->

## How it was verified

<!-- Delete what does not apply. -->

- [ ] `python -m pytest tests/ -q` passes
- [ ] New tests cover the change (fault injection is fine for paths that need
      real hardware to trigger)
- [ ] Live capture checked on real hardware — CI runners have no desktop
      session, so those tests skip there and prove nothing

## If it touches performance

<!-- Delete this whole section if it does not. -->

- [ ] `python benchmarks/perf_suite.py --self-test` run **first**, to establish
      this machine's noise floor
- [ ] Compared against `benchmarks/baseline.json`, numbers below
- [ ] If two implementations were compared, they were interleaved in one
      process — separate runs on a loaded machine have produced 2.26x, 1.56x
      and 0.87x for the same comparison

```
paste the comparison table here
```

## If it touches frames or buffers

<!-- frame.py, memory_pool.py, or the processor output path. Delete if not. -->

- [ ] Explains what keeps the buffer lifetime sound. A view into a pooled
      buffer, or a read after `release()`, yields a frame that is the right
      shape, the right dtype, and quietly wrong — that bug has shipped here
      before
- [ ] Output checked against an independent reference, not just for speed or
      shape

## Anything surprising?

<!--
If a measurement changed your mind, put it in ROADMAP.md. It will change
someone else's, and section 4 exists so nobody re-litigates a settled question
from first principles.
-->
