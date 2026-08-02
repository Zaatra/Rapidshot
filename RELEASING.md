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

1. Update the version in **three** places — they are checked against the git tag
   at build time, but not against each other:
   - `pyproject.toml`
   - `setup.py`
   - `rapidshot/__init__.py` (`__version__`)
2. Move `CHANGELOG.md`'s `[Unreleased]` content under a new
   `## [x.y.z] - YYYY-MM-DD` heading.
3. Run the full local gate:

   ```bash
   python -m pytest tests/ -q
   python benchmarks/ab_conversion.py
   python benchmarks/perf_suite.py --rounds 5 --reps 25 --compare benchmarks/baseline.json
   ```

4. **Run the live suites on real hardware.** CI runners have no desktop session,
   so every live test skips there. At minimum:

   ```bash
   python examples/verify_cross_adapter.py
   ```

5. Tag and push:

   ```bash
   git tag v2.0.0 && git push origin v2.0.0
   ```

The tag triggers `release.yml`, which builds, verifies, and publishes.

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

## After publishing

- Check the Sigstore attestation appears on the PyPI project page.
- The SBOM (CycloneDX) is attached to the workflow run as a build artifact, not
  uploaded to PyPI.
- `pip install rapidshot==<version>` in a clean environment and import it.
