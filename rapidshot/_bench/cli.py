"""``rapidshot benchmark``: the README's desktop-to-model table, on your machine.

One command that anyone with a pip install can run, so the published numbers
stop resting on one developer's two machines. It runs the same harness the
README's recordings came from -- every path verified to produce the correct
tensor, then timed on one clock by how old its pixels were -- against whichever
capture libraries are installed, and writes a report meant to be pasted into a
GitHub issue.

    pip install "rapidshot[benchmark]"
    rapidshot benchmark            # pixel age to a tensor, ~3-5 minutes
    rapidshot benchmark --full     # plus CPU per tensor and memory
    rapidshot benchmark --check    # what would run, without running it

The report is sanitised: no hostname, no username, no file paths from your
profile, no device instance IDs. What it does contain -- GPU and driver names,
laptop model, resolution and refresh -- is what makes a row of a hardware
matrix worth anything.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

from ._paths import WORK, worker_command, worker_env

#: The 2026-09-25 recordings used three 8-second passes per path; a tester's
#: report is only comparable to them if it does the same.
PASSES = 3
SECONDS = 8.0

#: Report columns: name -> (how to read it from a result row, unit).
PIXEL_AGE = {
    "unique_fps": (lambda r: r["unique_fps"], "fps"),
    "age_p50_ms": (lambda r: r["present_to_tensor_ms"]["p50"], "ms"),
    "age_p95_ms": (lambda r: r["present_to_tensor_ms"]["p95"], "ms"),
    "cpu_ms_per_frame": (lambda r: r["cpu_ms_per_unique_frame"], "ms"),
}
CALL_DURATION = {
    "fps": (lambda r: r["fps"], "fps"),
    "call_p50_ms": (lambda r: r["ms_p50"], "ms"),
    "cpu_ms_per_frame": (lambda r: r["cpu_seconds"] * 1000 / r["frames"], "ms"),
}
USABLE = ("passed", "contaminated")
LABELS = {
    "mss": "mss",
    "dxcam": "DXcam (DXGI)",
    "dxcam-wgc": "DXcam (WGC)",
    "rapidshot-cpu": "RapidShot grab()",
    "rapidshot-cupy": "RapidShot grab(), nvidia_gpu=True",
    "rapidshot-direct": "RapidShot direct, single adapter",
    "rapidshot-xadapter": "RapidShot full frame across adapters",
    "rapidshot-converter": "RapidShot GpuConverter",
    "rapidshot-converter-xadapter": "RapidShot GpuConverter + TensorTransfer",
}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------------
# What this machine can run
# ---------------------------------------------------------------------------

def _cpu_name():
    """The marketing name; `platform.processor()` gives "Intel64 Family 6 Model 183"."""
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        return platform.processor()


def preflight() -> dict:
    """Everything decided before a single frame is captured, and why."""
    import rapidshot
    from rapidshot import native
    from rapidshot.util.topology import probe_topology
    from . import section7

    topology = probe_topology()
    info = {
        "rapidshot": rapidshot.__version__,
        "python": platform.python_version(),
        "windows": platform.version(),
        "cpu": _cpu_name(),
        "native": native.build_info() if native.is_available() else None,
        "test_source": str(section7.SOURCE) if section7.SOURCE and section7.SOURCE.is_file() else None,
        "topology": topology.kind,
        "adapters": [a.description for a in topology.adapters if not a.is_software],
        "capture_adapters": [a.description for a in topology.capture_adapters],
        "installed": {name: _installed(module) for name, module in
                      (("mss", "mss"), ("dxcam", "dxcam"), ("winrt", "winrt"),
                       ("cupy", "cupy"), ("psutil", "psutil"))},
    }
    try:
        info["display"] = section7.display_mode()
    except OSError as exc:
        info["display"] = {"error": str(exc)}
    return info


def choose_paths(pre: dict):
    """(paths to run, {path: why it is skipped}). Nothing is skipped silently."""
    have = pre["installed"]
    paths, skipped = [], {}
    if have["mss"]:
        paths.append("mss")
    else:
        skipped["mss"] = "not installed: pip install mss"
    if have["dxcam"]:
        paths.append("dxcam")
        if have["winrt"]:
            paths.append("dxcam-wgc")
        else:
            skipped["dxcam-wgc"] = 'needs pip install "dxcam[winrt]"'
    else:
        skipped["dxcam"] = "not installed: pip install dxcam"
    paths.append("rapidshot-cpu")

    gpu = ["rapidshot-cupy", "rapidshot-direct", "rapidshot-converter"]
    if pre["topology"] == "hybrid":
        # Capture on one GPU, CUDA on another: the case convert-before-transfer
        # exists for, and the only one where the cross-adapter rows mean anything.
        gpu += ["rapidshot-xadapter", "rapidshot-converter-xadapter"]
    if not pre["native"]:
        skipped.update({p: 'needs rapidshot-native: pip install "rapidshot[native]"' for p in gpu})
    elif not have["cupy"]:
        skipped.update({p: "needs CuPy for your CUDA version, e.g. pip install cupy-cuda12x"
                        for p in gpu})
    else:
        paths += gpu
    return paths, skipped


def blockers(pre: dict) -> list:
    """Reasons the benchmark cannot run at all."""
    out = []
    if sys.platform != "win32":
        out.append("Desktop Duplication is Windows-only")
    if not pre["test_source"]:
        out.append('no test source: pip install "rapidshot-native>=0.2.1" '
                   '(or "rapidshot[benchmark]", which includes it)')
    if not pre["installed"]["psutil"]:
        out.append("the harness needs psutil: pip install psutil")
    if pre["topology"] == "headless" or not pre["capture_adapters"]:
        out.append("no adapter can capture a display here (headless or remote session?)")
    return out


# ---------------------------------------------------------------------------
# Running the harness
# ---------------------------------------------------------------------------

def _run(module: str, args: list, log: Path) -> int:
    """Run one harness invocation as a worker, echoing its progress lines."""
    command = worker_command(module, *args)
    with log.open("w", encoding="utf-8", errors="replace") as sink:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace",
                                env=worker_env({**os.environ, "PYTHONIOENCODING": "utf-8"}))
        for line in proc.stdout:
            sink.write(line)
            # The harness prints a JSON row per case and a bracketed line per
            # step; only the latter is useful to watch.
            if line.startswith("[") and "]" in line[:48]:
                print("   ", line.rstrip(), flush=True)
        return proc.wait()


def run_pixel_age(run_dir: Path, paths: list, passes: int, seconds: float) -> dict:
    common = ["--category", "ingestion", "--workload", "motion", "--paths", *paths,
              "--continue-on-failure"]
    files = {}
    steps = [("verify", ["--verify"])] + [(f"pass{n}", ["--seconds", str(seconds)])
                                           for n in range(1, passes + 1)]
    for name, extra in steps:
        out = run_dir / f"pixel-age-{name}.json"
        print(f"  pixel age: {name}", flush=True)
        _run("rapidshot._bench.section7", common + extra + ["--out", str(out)],
             run_dir / f"pixel-age-{name}.log")
        files[name] = out
    return files


def run_call_duration(run_dir: Path, paths: list, passes: int, seconds: float) -> dict:
    supported = [p for p in paths if p in ("mss", "dxcam", "rapidshot-cpu", "rapidshot-cupy",
                                           "rapidshot-xadapter", "rapidshot-converter",
                                           "rapidshot-converter-xadapter")]
    common = ["--call-duration", "--with-motion", "--paths", *supported,
              "--continue-on-failure", "--log-dir", str(run_dir / "call-duration-logs")]
    files = {}
    steps = [("verify", ["--verify"])] + [(f"pass{n}", ["--seconds", str(seconds)])
                                           for n in range(1, passes + 1)]
    for name, extra in steps:
        out = run_dir / f"call-duration-{name}.json"
        print(f"  CPU per tensor: {name}", flush=True)
        _run("rapidshot._bench.ai_ingestion", common + extra + ["--out", str(out)],
             run_dir / f"call-duration-{name}.log")
        files[name] = out
    return files


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def summarise(payloads: list, metrics: dict) -> dict:
    """Median, min and max per path across passes, from usable cases only."""
    rows = {}
    for payload in payloads:
        for row in (payload or {}).get("results", []):
            if row.get("case_status") not in USABLE:
                continue
            entry = rows.setdefault(row["path"], {"passes": 0, "statuses": []})
            entry["passes"] += 1
            entry["statuses"].append(row["case_status"])
            for name, (get, _unit) in metrics.items():
                try:
                    entry.setdefault(name, []).append(float(get(row)))
                except (KeyError, TypeError, ZeroDivisionError):
                    pass
    for entry in rows.values():
        for name in metrics:
            values = entry.get(name) or []
            entry[name] = ({"median": statistics.median(values), "min": min(values),
                            "max": max(values)} if values else None)
    return rows


def statuses(payloads: list) -> dict:
    """Every path's per-pass status, including the ones that did not produce a number."""
    out = {}
    for payload in payloads:
        for row in (payload or {}).get("results", []):
            out.setdefault(row["path"], []).append(
                {"status": row.get("case_status"), "reason": (row.get("error") or "")[:200] or None})
    return out


_INSTANCE = re.compile(r"(PCI\\VEN_[0-9A-F]{4}&DEV_[0-9A-F]{4}(?:&SUBSYS_[0-9A-F]{8})?)[^\"]*", re.I)


def sanitise(value, *, secrets=None):
    """Remove what identifies a person or a particular box, keep what identifies hardware.

    Dropped: the hostname hash, adapter and monitor device paths, and the instance part of
    every PnP device ID (``PCI\\VEN_10DE&DEV_28E0&SUBSYS_...`` survives; the
    ``\\4&1B0D88EE&0&0008`` after it, which is unique to one machine, does not).
    Replaced: the user profile path, and the user name wherever else it appears.
    """
    if secrets is None:
        home = str(Path.home())
        user = os.environ.get("USERNAME") or Path.home().name
        secrets = [(home, "%USERPROFILE%")] + ([(user, "<user>")] if len(user) >= 3 else [])
    if isinstance(value, dict):
        # Every *device_path carries the same per-machine instance ID in
        # `\\?\PCI#VEN_...#4&1b0d88ee&0&0008#{...}` form.
        return {k: sanitise(v, secrets=secrets) for k, v in value.items()
                if k not in ("hostname_hash", "logs", "stdout_log", "stderr_log", "run_dir",
                             "python_executable") and not k.endswith("device_path")}
    if isinstance(value, list):
        return [sanitise(v, secrets=secrets) for v in value]
    if isinstance(value, str):
        value = _INSTANCE.sub(r"\1", value)
        for secret, replacement in secrets:
            value = re.sub(re.escape(secret), lambda _m, r=replacement: r, value, flags=re.I)
    return value


def _environment(payloads: list):
    for payload in payloads:
        if payload and payload.get("environment"):
            return payload["environment"]
    return None


def build_report(pre: dict, skipped: dict, pixel: dict, call=None) -> dict:
    pixel_passes = [_load(f) for name, f in pixel.items() if name != "verify"]
    report = {
        "schema": "rapidshot-benchmark/1",
        "passes": len(pixel_passes),
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preflight": pre,
        "skipped": skipped,
        "pixel_age": {"summary": summarise(pixel_passes, PIXEL_AGE),
                      "verify": statuses([_load(pixel["verify"])]),
                      "statuses": statuses(pixel_passes)},
        "environment": _environment(pixel_passes),
    }
    if call:
        call_passes = [_load(f) for name, f in call.items() if name != "verify"]
        report["call_duration"] = {"summary": summarise(call_passes, CALL_DURATION),
                                   "verify": statuses([_load(call["verify"])]),
                                   "statuses": statuses(call_passes)}
    return sanitise(report)


def _native_label(native: dict) -> str:
    if not native:
        return "absent"
    version = native.get("wheel_version") or native.get("version") or "?"
    return version if native.get("source") == "rapidshot-native wheel" else f"{version} (development build)"


def _cell(stat, digits=1):
    return "—" if not stat else f"{stat['median']:.{digits}f}"


def render_markdown(report: dict) -> str:
    pre = report["preflight"]
    env = report.get("environment") or {}
    system = ((env.get("hardware") or {}).get("system") or {})
    display = pre.get("display") or {}
    native = pre.get("native") or {}
    lines = [
        "### RapidShot benchmark",
        "",
        f"- **Machine:** {system.get('Manufacturer', '?')} {system.get('Model', '')}".rstrip(),
        f"- **CPU:** {pre.get('cpu') or (env.get('cpu') or {}).get('processor', '?')}",
        f"- **GPUs:** {', '.join(pre.get('adapters') or ['?'])} — capture on "
        f"{', '.join(pre.get('capture_adapters') or ['?'])} ({pre.get('topology')})",
        f"- **Display:** {display.get('width', '?')}×{display.get('height', '?')} "
        f"at {display.get('refresh_hz', '?')} Hz",
        f"- **Versions:** rapidshot {pre.get('rapidshot')}, rapidshot-native "
        f"{_native_label(native)}, "
        f"Python {pre.get('python')}, Windows {pre.get('windows')}",
        "",
        f"Screen to a (1, 3, 640, 640) FP16 tensor on CUDA; medians of {report.get('passes')} "
        f"pass{'' if report.get('passes') == 1 else 'es'}. "
        "Pixel age is from `Present()` to the tensor.",
        "",
        "| path | unique fps | pixel age p50 / p95 | CPU per frame | passes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for path, row in report["pixel_age"]["summary"].items():
        flag = " *" if "contaminated" in row["statuses"] else ""
        lines.append(f"| {LABELS.get(path, path)} | {_cell(row['unique_fps'])}{flag} | "
                     f"{_cell(row['age_p50_ms'])} / {_cell(row['age_p95_ms'])} ms | "
                     f"{_cell(row['cpu_ms_per_frame'])} ms | {row['passes']} |")
    if any("contaminated" in r["statuses"] for r in report["pixel_age"]["summary"].values()):
        lines += ["", "\\* read as fast as the test source presented, so the frame rate is a floor."]
    missing = {p: s for p, s in report["pixel_age"]["statuses"].items()
               if p not in report["pixel_age"]["summary"]}
    if missing or report["skipped"]:
        lines += ["", "Not measured:"]
        for path, reason in report["skipped"].items():
            lines.append(f"- {LABELS.get(path, path)}: {reason}")
        for path, runs in missing.items():
            reason = next((r["reason"] for r in runs if r["reason"]), runs[0]["status"])
            lines.append(f"- {LABELS.get(path, path)}: {runs[0]['status']} — {reason}")
    if "call_duration" in report:
        lines += ["", "CPU per tensor (call-duration harness; its fps is bounded by its test window):",
                  "", "| path | CPU per frame | call p50 |", "| --- | ---: | ---: |"]
        for path, row in report["call_duration"]["summary"].items():
            lines.append(f"| {LABELS.get(path, path)} | {_cell(row['cpu_ms_per_frame'], 2)} ms | "
                         f"{_cell(row['call_p50_ms'], 2)} ms |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _print_plan(pre, paths, skipped, args):
    display = pre.get("display") or {}
    print(f"RapidShot {pre['rapidshot']} benchmark", flush=True)
    print(f"  GPUs: {', '.join(pre['adapters']) or 'none'} ({pre['topology']})")
    print(f"  display: {display.get('width', '?')}x{display.get('height', '?')} "
          f"at {display.get('refresh_hz', '?')} Hz")
    print(f"  will measure: {', '.join(paths)}")
    for path, reason in skipped.items():
        print(f"  skipping {path}: {reason}")
    steps = 1 + args.passes
    estimate = steps * len(paths) * (args.seconds + 6) * (2 if args.full else 1)
    print(f"  about {max(1, round(estimate / 60))} min. A test pattern will fill the screen; "
          "leave the machine alone until it finishes.", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rapidshot benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also measure CPU per tensor with the call-duration harness")
    parser.add_argument("--check", action="store_true",
                        help="print what would run and why, then exit without capturing")
    parser.add_argument("--passes", type=int, default=PASSES)
    parser.add_argument("--seconds", type=float, default=SECONDS)
    parser.add_argument("--paths", nargs="+", help="measure only these paths")
    parser.add_argument("--out", type=Path, default=Path.cwd(),
                        help="where to write the report (default: the current directory)")
    parser.add_argument("--yes", action="store_true", help="do not wait for Enter before starting")
    args = parser.parse_args(argv)
    if args.passes < 1 or args.seconds <= 0:
        parser.error("--passes must be at least 1 and --seconds positive")

    pre = preflight()
    paths, skipped = choose_paths(pre)
    if args.paths:
        unknown = sorted(set(args.paths) - set(paths) - set(skipped))
        if unknown:
            parser.error(f"unknown path(s): {', '.join(unknown)}")
        paths = [p for p in paths if p in args.paths]
    problems = blockers(pre)
    _print_plan(pre, paths, skipped, args)
    if problems:
        for problem in problems:
            print(f"  cannot run: {problem}", file=sys.stderr)
        return 2
    if args.check:
        return 0
    if not args.yes:
        try:
            input("Press Enter to start (Ctrl+C to cancel)... ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 130

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = WORK / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    pixel = run_pixel_age(run_dir, paths, args.passes, args.seconds)
    call = run_call_duration(run_dir, paths, args.passes, args.seconds) if args.full else None

    report = build_report(pre, skipped, pixel, call)
    report["duration_s"] = round(time.monotonic() - started)
    markdown = render_markdown(report)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"rapidshot-benchmark-{stamp}.json"
    md_path = args.out / f"rapidshot-benchmark-{stamp}.md"
    json_path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print()
    print(markdown)
    print(f"Report: {md_path}")
    print(f"Full data: {json_path}")
    print("Please attach both to an issue: https://github.com/Zaatra/Rapidshot/issues/new"
          "?template=benchmark.md")
    return 0
