"""The harness as shipped: importable from a wheel, and `rapidshot benchmark` around it.

Headless. Nothing here captures, opens a window or starts a worker; the command
line is exercised with its preflight replaced, and the report is built from
recorded-shape payloads rather than a live run.
"""
import importlib
import io
import json
from pathlib import Path
import sys

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))

from rapidshot._bench import _paths, cli  # noqa: E402

MOVED = ["section7", "section7_adapters", "benchmark_contract", "ai_ingestion",
         "machine_inventory", "result_store", "result_validation", "telemetry",
         "motion_source", "detection", "agent_pipeline", "ai_pipeline", "scenes",
         "memory_profile", "cuda_semaphore"]


# ---------------------------------------------------------------------------
# The move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", MOVED)
def test_the_old_import_name_is_the_moved_module(name):
    """Monkeypatching `section7.SOURCE` must patch what the harness reads."""
    old = importlib.import_module(name)
    new = importlib.import_module(f"rapidshot._bench.{name}")
    assert old is new


def test_the_packaged_cuda_import_is_the_example_minus_its_path_line():
    """One copy of the CUDA import, not two that can drift.

    The example stays readable documentation; the package needs it without the
    line that puts the example's parent directory on sys.path, which inside the
    package would be `rapidshot/` itself.
    """
    def text(path):
        return io.open(path, encoding="utf-8", newline="").read().replace("\r\n", "\n")
    example = text(REPO / "examples" / "gpu_tensor_to_cupy.py")
    packaged = text(REPO / "rapidshot" / "_bench" / "gpu_tensor_to_cupy.py")
    body = packaged.split("\n", 4)[4]            # drop the four-line provenance header
    block = ("# Running `python examples/gpu_tensor_to_cupy.py` puts *examples/* on the path,\n"
             "# not the repo root, so the import below fails on a source checkout even though\n"
             "# the package is right there -- the same line the other scripts use.\n"
             "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n\n")
    assert example.replace(block, "") == body


def test_no_harness_module_reaches_for_a_checkout_by_file_path():
    offenders = []
    for path in (REPO / "rapidshot" / "_bench").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for needle in ('parents[1] / "build"', 'parents[1] / "examples"', "str(Path(__file__).resolve()),"):
            if needle in source:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders


def test_a_checkout_is_recognised_and_site_packages_is_not(tmp_path):
    assert _paths._checkout(REPO) == REPO
    assert _paths._checkout(tmp_path) is None


def test_workers_import_the_parents_rapidshot():
    env = _paths.worker_env({"PYTHONPATH": "elsewhere"})
    assert env["PYTHONPATH"].split(";" if sys.platform == "win32" else ":")[:2] == [
        str(_paths.PACKAGE_PARENT), "elsewhere"]
    assert _paths.worker_command("rapidshot._bench.section7", "--worker", "mss")[1:4] == [
        "-u", "-m", "rapidshot._bench.section7"]


# ---------------------------------------------------------------------------
# Choosing what to run
# ---------------------------------------------------------------------------

def preflight(**overrides):
    base = {"rapidshot": "2.6.1", "python": "3.13.0", "windows": "10.0.26200",
            "native": {"version": "0.2.1", "source": "rapidshot-native wheel"},
            "test_source": "C:/x/latency_source.exe", "topology": "hybrid",
            "adapters": ["Intel(R) UHD Graphics", "NVIDIA GeForce RTX 4060 Laptop GPU"],
            "capture_adapters": ["Intel(R) UHD Graphics"],
            "installed": {"mss": True, "dxcam": True, "winrt": True, "cupy": True, "psutil": True},
            "display": {"width": 2560, "height": 1600, "refresh_hz": 165}}
    base.update(overrides)
    return base


def test_a_hybrid_laptop_measures_the_cross_adapter_paths():
    paths, skipped = cli.choose_paths(preflight())
    assert "rapidshot-converter-xadapter" in paths and "rapidshot-xadapter" in paths
    assert not skipped


def test_one_gpu_does_not_attempt_a_crossing():
    paths, _ = cli.choose_paths(preflight(topology="single"))
    assert not any("xadapter" in p for p in paths)
    assert "rapidshot-converter" in paths


def test_every_skip_says_how_to_fix_it():
    pre = preflight(installed={"mss": False, "dxcam": True, "winrt": False, "cupy": False,
                               "psutil": True})
    paths, skipped = cli.choose_paths(pre)
    assert paths == ["dxcam", "rapidshot-cpu"]
    assert "pip install mss" in skipped["mss"]
    assert "winrt" in skipped["dxcam-wgc"]
    assert all("CuPy" in reason for path, reason in skipped.items() if path.startswith("rapidshot"))


def test_without_the_native_wheel_the_gpu_paths_name_it():
    _, skipped = cli.choose_paths(preflight(native=None))
    assert "rapidshot[native]" in skipped["rapidshot-converter"]


@pytest.mark.parametrize("override, fragment", [
    ({"test_source": None}, "rapidshot-native>=0.2.1"),
    ({"installed": {"mss": True, "dxcam": True, "winrt": True, "cupy": True, "psutil": False}},
     "psutil"),
    ({"topology": "headless", "capture_adapters": []}, "headless"),
])
def test_what_stops_a_run_is_said_before_it_starts(override, fragment):
    assert any(fragment in problem for problem in cli.blockers(preflight(**override)))


def test_check_reports_and_exits_without_capturing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "preflight", lambda: preflight())
    ran = []
    monkeypatch.setattr(cli, "run_pixel_age", lambda *a, **k: ran.append(a))
    assert cli.main(["--check"]) == 0
    assert not ran
    assert "rapidshot-converter-xadapter" in capsys.readouterr().out


def test_a_blocked_machine_exits_nonzero_before_asking(monkeypatch):
    monkeypatch.setattr(cli, "preflight", lambda: preflight(test_source=None))
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked to start a blocked run"))
    assert cli.main([]) == 2


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def row(path, fps, age, cpu, status="passed"):
    return {"path": path, "case_status": status, "unique_fps": fps, "cpu_ms_per_unique_frame": cpu,
            "present_to_tensor_ms": {"p50": age, "p95": age + 2}}


def write_passes(tmp_path, rows_per_pass, environment=None):
    files = {"verify": tmp_path / "v.json"}
    files["verify"].write_text(json.dumps({"results": [dict(r, verified=True) for r in rows_per_pass[0]]}))
    for n, rows in enumerate(rows_per_pass, 1):
        files[f"pass{n}"] = tmp_path / f"p{n}.json"
        files[f"pass{n}"].write_text(json.dumps({"results": rows, "environment": environment}))
    return files


def test_the_summary_is_median_min_max_of_usable_passes_only(tmp_path):
    files = write_passes(tmp_path, [
        [row("dxcam", 95, 37, 11), row("rapidshot-converter-xadapter", 160, 27, 2)],
        [row("dxcam", 97, 38, 10), row("rapidshot-converter-xadapter", 164, 28, 2, "contaminated")],
        [row("dxcam", 99, 36, 12), {"path": "rapidshot-converter-xadapter", "case_status": "failed",
                                    "error": "boom"}],
    ])
    report = cli.build_report(preflight(), {}, files)
    dxcam = report["pixel_age"]["summary"]["dxcam"]
    assert dxcam["passes"] == 3
    assert dxcam["unique_fps"] == {"median": 97.0, "min": 95.0, "max": 99.0}
    converter = report["pixel_age"]["summary"]["rapidshot-converter-xadapter"]
    assert converter["passes"] == 2
    assert [s["status"] for s in report["pixel_age"]["statuses"]["rapidshot-converter-xadapter"]] == [
        "passed", "contaminated", "failed"]


def test_the_markdown_flags_a_source_limited_row_and_lists_what_was_not_measured(tmp_path):
    files = write_passes(tmp_path, [[row("dxcam", 95, 37, 11),
                                     row("rapidshot-converter", 164, 26, 1, "contaminated"),
                                     {"path": "rapidshot-direct", "case_status": "unavailable",
                                      "error": "CrossAdapterRequired: capture is on the iGPU"}]])
    text = cli.render_markdown(cli.build_report(preflight(), {"mss": "not installed: pip install mss"},
                                                files))
    assert "| RapidShot GpuConverter | 164.0 * |" in text
    assert "medians of 1 pass." in text           # the count that ran, not the default
    assert "a floor" in text
    assert "mss: not installed" in text
    assert "RapidShot direct, single adapter: unavailable" in text


def test_nothing_that_identifies_a_person_or_a_box_survives(tmp_path, monkeypatch):
    monkeypatch.setenv("USERNAME", "alice")
    home = str(Path.home())
    environment = {
        "os": {"platform": "Windows-11", "hostname_hash": "3c6c91bda91c767e"},
        "hardware": {"gpus": [{"name": "NVIDIA GeForce RTX 4060 Laptop GPU",
                               "pnp_device_id": r"PCI\VEN_10DE&DEV_28E0&SUBSYS_17311025&REV_A1\4&1B0D88EE&0&0008"}],
                     "system": {"Manufacturer": "Acer", "Model": "Predator PHN16-72"}},
        "displays": {"outputs": [{
            "monitor_device_path": r"\\?\DISPLAY#AUOCDAB#4&3632a66b&0&UID8388688",
            # Found by the first live run: the same instance ID, in `#` form.
            "adapter_device_path": r"\\?\PCI#VEN_10DE&DEV_28E0&SUBSYS_17311025&REV_A1#4&1b0d88ee&0&0008#{5b45}"}]},
        "runtime": {"python_executable": home + r"\AppData\python.exe"},
        "source": {"root": home + r"\Projects\Rapidshot", "note": "built by alice"},
    }
    files = write_passes(tmp_path, [[row("dxcam", 95, 37, 11)]], environment)
    blob = json.dumps(cli.build_report(preflight(test_source=home + r"\x\latency_source.exe"), {}, files))
    for secret in ("3c6c91bda91c767e", "1b0d88ee", "3632a66b", home.replace("\\", "\\\\").lower(), "alice"):
        assert secret not in blob.lower(), secret
    # What makes the row useful to a hardware matrix is kept.
    assert "PCI\\\\VEN_10DE&DEV_28E0&SUBSYS_17311025" in blob
    assert "Predator PHN16-72" in blob and "RTX 4060" in blob


def test_python_m_rapidshot_dispatches(monkeypatch):
    import rapidshot.__main__ as entry
    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 0)
    assert entry.main(["benchmark", "--check"]) == 0
    assert seen == [["--check"]]
    assert entry.main(["no-such-command"]) == 2


def test_a_development_build_is_labelled_as_one():
    """The first live run printed "rapidshot-native 0.1.0" for the in-tree build."""
    assert cli._native_label({"version": "0.2.1", "source": "rapidshot-native wheel",
                              "wheel_version": "0.2.1"}) == "0.2.1"
    assert cli._native_label({"version": "0.1.0", "source": "development build (x)"}) == \
        "0.1.0 (development build)"
    assert cli._native_label(None) == "absent"
