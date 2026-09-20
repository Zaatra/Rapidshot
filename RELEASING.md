# Releasing

Publishing runs through PyPI **Trusted Publishing**: GitHub Actions mints a
short-lived OIDC token, so there is no API token stored in the repository, in a
secret, or on anyone's laptop. Nothing to leak, nothing to rotate.

## One-time setup

1. **PyPI publisher.** At
   <https://pypi.org/manage/project/rapidshot/settings/publishing/> add a GitHub
   publisher:

   | Field | Value |
   | --- | --- |
   | Owner | `Zaatra` |
   | Repository | `Rapidshot` |
   | Workflow | `release.yml` |
   | Environment | `pypi` |

2. **GitHub environment.** Settings → Environments → new environment named
   `pypi`. Add required reviewers if a release should need a second pair of
   eyes; the workflow will wait for approval before publishing.

3. **Branch protection** on the default branch with *Require review from Code
   Owners*, or `.github/CODEOWNERS` is only a routing hint.

4. **Private vulnerability reporting.** Settings → Security → enable it, so the
   link in `SECURITY.md` works.

## Cutting a release

> **If this release raises the `rapidshot-native` floor, publish the native tag
> first.** The two normally version independently, and that is the point of
> keeping them apart — but a RapidShot release that floors the `native` extra at
> a version PyPI does not have yet makes `pip install rapidshot[native]` and
> `rapidshot[all]` unresolvable for as long as the gap lasts. 2.6.0 is the first
> release where this bites: it calls `GpuConverter12` and `TensorTransfer`, both
> added after `native-v0.1.0`, so it requires `rapidshot-native>=0.2.0`.
>
> Order: tag `native-v0.2.0`, wait for `release-native.yml` to publish, confirm
> the wheel is installable from PyPI, *then* tag the RapidShot release.

1. Update the version in **one** place — `rapidshot/_version.py`:

   ```python
   __version__ = "2.4.0"
   ```

   `pyproject.toml` reads it through `[tool.setuptools.dynamic]`, `setup.py`
   parses it, and `rapidshot.__version__` re-exports it. It used to be written
   out in all three; each was checked against the git tag at build time but
   never against the others, so two could agree while the third drifted and
   nothing failed until a release was being cut. `tests/test_version.py` fails
   if a second declaration reappears.
2. Move `CHANGELOG.md`'s `[Unreleased]` content under a new
   `## [x.y.z] - YYYY-MM-DD` heading.
3. If `benchmarks/baseline.json` was re-recorded, regenerate the README badges
   so they cannot disagree with it — CI fails otherwise:

   ```bash
   python benchmarks/make_badges.py
   ```
4. Run the full local gate:

   ```bash
   python -m pytest tests/ -q
   python benchmarks/ab_conversion.py
   python benchmarks/perf_suite.py --rounds 5 --reps 25 --compare auto
   ```

   `--compare auto` selects the committed baseline recorded on the machine you
   are running on, and **fails** if there is none. This step used to name
   `benchmarks/baseline.json`, which is one specific machine: run anywhere else
   the suite detected the mismatch and declined to gate, so the table printed
   verdicts that were all indicative and nothing could fail. Re-record this
   machine's baseline first if the release changed anything performance-facing:

   ```bash
   python benchmarks/perf_suite.py --rounds 5 --reps 25 --out benchmarks/baseline-<machine>.json
   ```

   Verify the threshold clears the host's noise before trusting a verdict —
   anything it reports here is measurement error, not a change:

   ```bash
   python benchmarks/perf_suite.py --self-test --rounds 5 --reps 25
   ```

5. **Run the live suites on real hardware.** CI runners have no desktop session,
   so every live test skips there. At minimum:

   ```bash
   python examples/verify_cross_adapter.py
   ```

6. Tag and push:

   ```bash
   git tag v2.0.0 && git push origin v2.0.0
   ```

The tag triggers `release.yml`, which builds, verifies, publishes to PyPI, and
then creates the GitHub Release — in that order, so a release is never announced
for a version that failed to upload.

The release notes come from the matching `## [x.y.z]` section of
`CHANGELOG.md`. If there is no such section the workflow **fails** rather than
publishing an empty release, so write the changelog before tagging.

## Cutting a `rapidshot-native` release

`rapidshot-native` is the prebuilt extension, published from `native/` by
`release-native.yml` on a **separate tag**. The two version independently: the
Rust changes on its own schedule, and tying them together would republish an
identical binary under a new number every time the Python side moves.

1. Bump the version in **`native/Cargo.toml`**, which is the only place it is
   written — maturin stamps the wheel from it, and
   `rapidshot_native.__version__` reads it back out of installed metadata.
2. Add a `CHANGELOG.md` entry.
3. Tag and push:

   ```bash
   git tag native-v0.1.0 && git push origin native-v0.1.0
   ```

Build it locally first if you want to see what will ship:

```bash
maturin build --release -m native/Cargo.toml --out dist
```

`-m` takes the **Cargo** manifest; maturin reads `native/pyproject.toml` from
beside it. Pointing it at the pyproject fails with *"the manifest-path must be a
path to a Cargo.toml file"*.

### One-time setup for it

A second PyPI publisher, on the `rapidshot-native` project rather than
`rapidshot`:

| Field | Value |
| --- | --- |
| Owner | `Zaatra` |
| Repository | `Rapidshot` |
| Workflow | `release-native.yml` |
| Environment | `pypi-native` |

Plus a GitHub environment named `pypi-native`. It is deliberately not the same
environment as `pypi`: an approval to publish the pure-Python wheel is not an
approval to publish a binary.

### Ordering, the first time only

`pyproject.toml` has a `native` extra pointing at `rapidshot-native`, and it is
deliberately **not** included in `all` until the first wheel is on PyPI. Naming
an unpublished distribution there would break `pip install rapidshot[all]` for
everyone, immediately, to advertise something that does not exist yet. So:

1. Ship `native-v0.1.0` and confirm PyPI serves it.
2. Then, if you want it in the catch-all, add
   `rapidshot-native>=0.1.0; platform_system == 'Windows'` to the `all` extra
   and ship it with the next `rapidshot` release.

### What it refuses to publish

- **A wheel that is not `abi3`.** Losing the `abi3-py39` feature produces a
  wheel that works on exactly the Python it was built against, installs
  happily, and breaks silently on the next interpreter. Checked by filename,
  because nothing else in the build fails when it happens.
- **A wheel `rapidshot` cannot find.** The workflow installs it into a clean
  environment *outside the repository*, then asserts
  `rapidshot.native.is_available()` and that `build_info()["source"]` names the
  wheel. A binary that imports on its own but that RapidShot does not pick up
  is a binary that does nothing for anyone.
- **A tag that disagrees with the packaged version.**

## What the workflow refuses to publish

Each of these exists because it went wrong once, or would have:

- **A wheel missing subpackages.** `packages = ["rapidshot"]` in `pyproject.toml`
  once shipped 5 modules instead of 25; `pip install rapidshot` then raised
  `ModuleNotFoundError` on import. Invisible from a source checkout, where the
  subpackages are on `sys.path` anyway — so the workflow installs the wheel into
  a fresh virtualenv and imports it **from outside the repository**.
- **A wheel containing compiled artifacts.** The native extension is optional and
  built by the consumer. A `*.pyd` package-data glob swept a locally built
  extension into a `py3-none-any` wheel, which would have shipped a Windows
  binary from the release machine to every platform.
- **A wheel without `py.typed`.** Without it, PEP 561 says to ignore the
  annotations and every `rapidshot` symbol resolves to `Any` downstream.
- **A tag that disagrees with the packaged version.**
- **An SBOM describing the build environment instead of the package.** It is
  generated against the verification venv, which holds the wheel and its runtime
  dependencies and nothing else; a check rejects it if build tooling leaks in.

## After publishing

- Check the Sigstore attestation appears on the PyPI project page.
- Check the GitHub Release exists and is marked *Latest*, with the wheel, sdist
  and SBOM attached.
- The SBOM (CycloneDX) is attached to the workflow run as a build artifact, not
  uploaded to PyPI.
- `pip install rapidshot==<version>` in a clean environment and import it.
