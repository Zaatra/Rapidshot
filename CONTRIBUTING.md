# Contributing to RapidShot

Thanks for taking an interest. This is a small project with a specific shape, so
this document is mostly about the things that will waste your time otherwise.

**Read [`ROADMAP.md`](ROADMAP.md) first.** It is written to be read cold and it
is the source of truth for where the project is going. Section 4 in particular
lists questions that are already settled by measurement — a PR that re-opens one
of those needs to beat the measurement, not the argument.

## Getting set up

Windows only. DXGI Desktop Duplication has no cross-platform equivalent, and
there is no way to work on the capture path from Linux or macOS.

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

That is the whole setup for everything except the optional Rust extension.

### The optional native extension

`pip install rapidshot` must **never** require a toolchain. That is enforced —
the CI test job asserts the extension is *absent*. If a change makes the library
need it, the change is wrong.

To build it you need Rust and the MSVC C++ build tools:

```bash
(cd native && cargo build --release) && python native/install_dev.py
```

`cargo` is often missing from a fresh shell's PATH. On PowerShell:

```powershell
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
```

## Things that will catch you out

These are all real, all cost someone a session, and all are documented in
`ROADMAP.md` section 2:

- **Live capture tests need something moving on screen.** Desktop Duplication
  only reports *changed* content, so on an idle screen `grab()` returns `None`
  forever and tests fail for reasons unrelated to your change.
- **Two Python interpreters are often on PATH.** Check `sys.executable` before
  concluding a dependency is missing.
- **Synthetic textures cannot test the D3D12 path.** D3D11 refuses
  `SHARED_NTHANDLE` without `SHARED_KEYEDMUTEX`, and a keyed-mutex resource
  reads as zeros until acquired. Use live capture.
- **CI cannot verify live capture.** GitHub runners have no desktop session, so
  those tests skip there. Run them locally before claiming a fix works.

## If your change touches performance

Measure it. This project has reordered its own roadmap twice on the strength of
profiling, and rejected optimisations that looked obviously correct.

```bash
python benchmarks/perf_suite.py --self-test    # what is this machine's noise floor?
python benchmarks/perf_suite.py --rounds 5 --reps 25 --compare benchmarks/baseline.json
```

Run `--self-test` **first**. It measures the suite against itself, so anything
it reports is pure measurement error. A naive comparison on this codebase once
produced eleven false regressions of up to 1.9x on identical code.

Two rules that follow from that:

- **Never benchmark per-frame work in a back-to-back loop.** Sustained heavy
  vector work holds the CPU in a lower power state, so a burn loop measures the
  throttled number and then calls the realistic one a regression. The suite
  paces reps to a frame period; keep it that way.
- **Re-record `baseline.json` if you change how the harness drives benchmarks**,
  and verify immediately with a second run that reads all `~ same`. Numbers from
  different pacing models are not comparable.

If you are comparing two implementations, do it **interleaved in one process**.
Separate runs on a loaded machine have produced 2.26x, 1.56x and 0.87x for the
same comparison.

## If your change touches frames or buffers

This is where bugs are expensive and invisible in review. `grab()` hands out a
pooled buffer that goes back to the pool on `release()` and is then given to the
next capture. Returning a view into one, or reading one after release, produces
a frame that is the right shape, the right dtype, and quietly wrong.

That exact bug has shipped here before. If you touch `frame.py`,
`memory_pool.py`, or the processor's output path, say so in the PR and explain
what keeps the lifetime sound.

**A fast wrong answer is worthless.** The GPU shader once produced BGR labelled
RGB — no test of speed, shape or stability caught it. Correctness checks against
an independent reference come before any performance claim.

## Pull requests

- Keep it focused. One change per PR.
- Tests for anything that could regress. Fault injection is fine and widely used
  here for paths that need real hardware to trigger naturally.
- Explain *why* in the commit message, not just what. The reasoning is the part
  that is expensive to reconstruct later.
- If you found something surprising, put it in `ROADMAP.md`. A measurement that
  changed your mind will change someone else's.

CI must be green: tests on Python 3.9/3.11/3.13 without the extension, the Rust
build with `fmt` and `clippy`, the benchmark correctness gate, and a packaging
check that installs the built wheel and imports it from outside the source tree.

## Reporting bugs

Use the issue templates. For anything security-relevant, see
[`SECURITY.md`](SECURITY.md) — report privately rather than opening an issue.

Include your Windows version, GPU, Python version, and whether the native
extension is built. `python -c "import rapidshot; print(rapidshot.topology_info())"`
covers most of the hardware side in one line.
