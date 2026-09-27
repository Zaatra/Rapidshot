"""``rapidshot benchmark``: the README's desktop-to-model table, on your machine.

One command that anyone with a pip install can run, so the published numbers
stop resting on one developer's two machines. It runs the same harness the
README's recordings came from -- every path verified to produce the correct
tensor, then timed on one clock by how old its pixels were -- against whichever
capture libraries are installed, and writes a report meant to be pasted into a
GitHub issue.

    pip install "rapidshot[benchmark]"
    rapidshot benchmark            # pixel age to a tensor, ~3-5 minutes
    rapidshot benchmark --full     # plus CPU per tensor and memory, ~10 minutes
    rapidshot benchmark --check    # what would run, without running it

The report is sanitised: no hostname, no username, no file paths from your
profile, no device instance IDs. What it does contain -- GPU and driver names,
laptop model, resolution and refresh -- is what makes a row of a hardware
matrix worth anything.
"""

from __future__ import annotations

import argparse
import contextlib
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
MEMORY = {
    "fps": (lambda r: r["fps"], "fps"),
    "working_set_mb": (lambda r: r["working_set_mb"], "MB"),
    "capture_mb": (lambda r: r["working_set_over_baseline_mb"], "MB"),
    "growth_mb_per_s": (lambda r: r["working_set_growth_mb_per_s"], "MB/s"),
}
MEMORY_WORKLOADS = ("static", "scroll", "motion")
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


def _distribution_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001 -- not installed, or no metadata
        return None


#: The DXcam the harness is verified against. Older ones are skipped, not
#: compared: 0.0.5, all Python 3.9 can install, has no WinRT backend.
DXCAM_FLOOR = (0, 3, 0)


def _dxcam_too_old(pre: dict):
    """Why the installed DXcam is not compared, or None."""
    text = (pre.get("versions") or {}).get("dxcam")
    match = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    if not match or tuple(int(g or 0) for g in match.groups()) >= DXCAM_FLOOR:
        return None
    return (f"DXcam {text} is older than the {'.'.join(map(str, DXCAM_FLOOR))} this "
            "harness is verified against: pip install -U dxcam (needs Python 3.10+)")


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
        # "11" on build 22000 and later, which `version()` alone reports as 10.0.x.
        "windows_release": platform.release(),
        "cpu": _cpu_name(),
        "native": native.build_info() if native.is_available() else None,
        "test_source": str(section7.SOURCE) if section7.SOURCE and section7.SOURCE.is_file() else None,
        "topology": topology.kind,
        "adapters": [a.description for a in topology.adapters if not a.is_software],
        "capture_adapters": [a.description for a in topology.capture_adapters],
        "installed": {name: _installed(module) for name, module in
                      (("mss", "mss"), ("dxcam", "dxcam"), ("winrt", "winrt"),
                       ("cupy", "cupy"), ("cv2", "cv2"), ("psutil", "psutil"))},
        "versions": {name: _distribution_version(name) for name in ("mss", "dxcam")},
    }
    try:
        info["display"] = section7.display_mode()
    except OSError as exc:
        info["display"] = {"error": str(exc)}
    return info


#: Why a path is skipped without CuPy: it ends on CUDA, which needs NVIDIA.
NO_CUPY = ("ends in an FP16 tensor on CUDA, which needs an NVIDIA GPU and CuPy for your "
           "CUDA version, e.g. pip install cupy-cuda12x")
NO_CV2 = "resizes with OpenCV on the CPU: pip install opencv-python"
NO_NATIVE = 'needs rapidshot-native: pip install "rapidshot[native]"'
#: Where the tensor is finished, per target. The two never share a table.
TARGETS = {
    "cuda": "ingestion",
    "no-cuda": "no-cuda",
}


def tensor_target(pre: dict) -> str:
    """"cuda" with CuPy, the README table's finish line; otherwise "no-cuda": the
    same tensor finished in system memory (CPU capture) or on the capture GPU
    (GpuConverter), which is where a machine without NVIDIA keeps it."""
    return "cuda" if pre["installed"]["cupy"] else "no-cuda"


def choose_paths(pre: dict):
    """(paths to run, {path: why it is skipped}). Nothing is skipped silently.

    mss, DXcam and grab() are CPU capture, resized with cv2 on the CPU. With CuPy
    their tensor then goes to CUDA like every GPU path's; without it, it stays in
    system memory, and GpuConverter is the one GPU path that finishes without
    CUDA -- on the capture adapter.
    """
    have = pre["installed"]
    cpu, skipped = [], {}
    if have["mss"]:
        cpu.append("mss")
    else:
        skipped["mss"] = "not installed: pip install mss"
    if have["dxcam"] and _dxcam_too_old(pre):
        skipped.update({p: _dxcam_too_old(pre) for p in ("dxcam", "dxcam-wgc")})
    elif have["dxcam"]:
        cpu.append("dxcam")
        if have["winrt"]:
            cpu.append("dxcam-wgc")
        else:
            skipped["dxcam-wgc"] = 'needs pip install "dxcam[winrt]"'
    else:
        skipped["dxcam"] = "not installed: pip install dxcam"
    cpu.append("rapidshot-cpu")
    if not have.get("cv2"):
        skipped.update({p: NO_CV2 for p in cpu})
        cpu = []

    gpu = ["rapidshot-cupy", "rapidshot-direct", "rapidshot-converter"]
    if pre["topology"] == "hybrid":
        # Capture on one GPU, CUDA on another: the case convert-before-transfer
        # exists for, and the only one where the cross-adapter rows mean anything.
        gpu += ["rapidshot-xadapter", "rapidshot-converter-xadapter"]
    if not have["cupy"]:
        skipped.update({p: NO_CUPY for p in gpu if p != "rapidshot-converter"})
        gpu = ["rapidshot-converter"]
    if not pre["native"]:
        skipped.update({p: NO_NATIVE for p in gpu})
        gpu = []
    return cpu + gpu, skipped


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
            if line.lstrip().startswith("[") and "]" in line[:48]:
                print("   ", line.rstrip(), flush=True)
        return proc.wait()


def run_pixel_age(run_dir: Path, paths: list, passes: int, seconds: float,
                  target: str = "cuda") -> dict:
    common = ["--category", TARGETS[target], "--workload", "motion", "--paths", *paths,
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


def memory_skipped(pre: dict) -> dict:
    """{library: why} for the memory rows that cannot run. None of them needs CuPy."""
    have, out = pre["installed"], {}
    for lib in ("mss", "dxcam"):
        if not have[lib]:
            out[lib] = f"not installed: pip install {lib}"
    if have["dxcam"] and not have.get("cv2"):
        # dxcam.create() converts colour with its default cv2 backend.
        out["dxcam"] = "DXcam's colour conversion needs OpenCV: pip install opencv-python"
    if have["dxcam"] and _dxcam_too_old(pre):
        out["dxcam"] = _dxcam_too_old(pre)
    return out


def memory_libraries(pre: dict) -> list:
    skipped = memory_skipped(pre)
    return ([lib for lib in ("mss", "dxcam") if lib not in skipped]
            + ["rapidshot", "rapidshot-frame"])


def run_memory(run_dir: Path, pre: dict, seconds: float) -> Path:
    """One run per library and workload; the slope, not a pass count, is the finding."""
    out = run_dir / "memory.json"
    print("  memory: static, scroll and full-motion screens", flush=True)
    _run("rapidshot._bench.memory_profile",
         ["--libraries", *memory_libraries(pre), "--workloads", *MEMORY_WORKLOADS,
          "--seconds", str(seconds), "--continue-on-failure",
          "--log-dir", str(run_dir / "memory-logs"), "--out", str(out)],
         run_dir / "memory.log")
    return out


def run_capabilities(run_dir: Path) -> Path:
    out = run_dir / "capabilities.json"
    print("  capabilities: capture format, cross-adapter", flush=True)
    _run("rapidshot._bench.capabilities", ["--out", str(out)], run_dir / "capabilities.log")
    return out


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


#: Why a pass was contaminated, by what is to blame, each with its own mark. One
#: footnote for all of them told a reader that mss at 33 fps against a 100 fps
#: source was a floor, when what had happened was other programs using the CPU.
CONTAMINATION = {
    "source": ("*", "read as fast as the test source presented in at least one pass, so "
                    "the frame rate is a floor."),
    "background": ("†", "other programs used the CPU during at least one pass (median above "
                        "15% outside the benchmark); timings may be inflated. The harness "
                        "treats this as a warning, not proof, and keeps the pass."),
    "other": ("‡", "flagged for another reason in at least one pass; see below."),
}


def contamination_kind(reason: str) -> str:
    if "source-limited" in reason or "source never reported" in reason:
        return "source"
    if "outside this benchmark's process tree" in reason:
        return "background"
    return "other"


def _marks(row) -> str:
    """The marks for a summary row. A 2.6.2 report has no reasons, so its
    contaminated rows keep the one mark that release printed."""
    flags = row.get("flags")
    if flags is None:
        return " *" if "contaminated" in row.get("statuses", []) else ""
    marks = "".join(CONTAMINATION[k][0] for k in CONTAMINATION if k in flags)
    return f" {marks}" if marks else ""


def _passes_cell(row) -> str:
    flagged = row.get("flagged") or 0
    return f"{row['passes']}" + (f" ({flagged} flagged)" if flagged else "")


def _contamination_notes(summary: dict) -> list:
    """Footnotes for the marks in use, and the reasons behind any the marks
    cannot say on their own."""
    used = {k for row in summary.values() for k in (row.get("flags") or [])}
    if not used and any("contaminated" in r.get("statuses", []) for r in summary.values()):
        used = {"source"}
    others = [f"‡ {LABELS.get(path, path)}: {reason}"
              for path, row in summary.items() for reason in row.get("reasons") or []
              if contamination_kind(reason) == "other"]
    # Escaped, or Markdown reads the leading asterisk as a list bullet.
    lines = [f"{CONTAMINATION[k][0]} {CONTAMINATION[k][1]}".replace("*", "\\*", 1)
             for k in CONTAMINATION if k in used and not (k == "other" and not others)]
    if "other" in used and not others:
        lines.append("‡ flagged in at least one pass by a harness that did not record why.")
    return (["", *lines, *others]) if lines or others else []


def summarise(payloads: list, metrics: dict) -> dict:
    """Median, min and max per path across passes, from usable cases only."""
    rows = {}
    for payload in payloads:
        for row in (payload or {}).get("results", []):
            if row.get("case_status") not in USABLE:
                continue
            entry = rows.setdefault(row["path"], {"passes": 0, "statuses": [], "flagged": 0,
                                                  "flags": [], "reasons": []})
            entry["passes"] += 1
            entry["statuses"].append(row["case_status"])
            if row["case_status"] == "contaminated":
                entry["flagged"] += 1
                for reason in row.get("case_reasons") or [""]:
                    kind = contamination_kind(reason)
                    if kind not in entry["flags"]:
                        entry["flags"].append(kind)
                    if reason and reason not in entry["reasons"]:
                        entry["reasons"].append(reason[:300])
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


def _why_not_usable(row) -> str:
    """The status and its reasons. A bare "invalid" gives a tester nothing to act
    on, and the likeliest cause -- the test pattern stopped -- only needs a rerun."""
    status = row.get("case_status") or ("failed" if "error" in row else "unusable")
    reasons = [row["error"]] if row.get("error") else list(row.get("case_reasons") or [])
    return (f"{status}: {'; '.join(reasons)}" if reasons else status)[:300]


def summarise_memory(payload) -> dict:
    """{library: {workload: {metric: value}}} for rows that measured something."""
    out = {}
    for row in (payload or {}).get("results", []):
        entry = out.setdefault(row.get("library"), {})
        if "error" in row or row.get("case_status", "passed") not in USABLE:
            entry[row.get("workload")] = {"error": _why_not_usable(row)}
            continue
        values = {}
        for name, (get, _unit) in MEMORY.items():
            try:
                values[name] = float(get(row))
            except (KeyError, TypeError):
                values[name] = None
        entry[row.get("workload")] = values
    return out


def _primary_advanced_color(displays):
    outputs = (displays or {}).get("outputs") or []
    primary = next((o for o in outputs if o.get("primary")), outputs[0] if outputs else {})
    return primary.get("advanced_color")


def hdr_state(environment, capabilities) -> dict:
    """The panel's claim (advanced colour) beside the evidence (the captured format)."""
    capabilities = capabilities or {}
    display = (_primary_advanced_color((environment or {}).get("displays"))
               or _primary_advanced_color(capabilities.get("displays")))
    return {"display": display, "capture": capabilities.get("capture")}


def statuses(payloads: list) -> dict:
    """Every path's per-pass status, including the ones that did not produce a number."""
    out = {}
    for payload in payloads:
        for row in (payload or {}).get("results", []):
            reason = row.get("error") or "; ".join(row.get("case_reasons") or [])
            out.setdefault(row["path"], []).append(
                {"status": row.get("case_status"), "reason": reason[:300] or None})
    return out


_INSTANCE = re.compile(r"(PCI\\VEN_[0-9A-F]{4}&DEV_[0-9A-F]{4}(?:&SUBSYS_[0-9A-F]{8})?)[^\"]*", re.I)


def sanitise(value, *, redactions=None):
    """Remove what identifies a person or a particular box, keep what identifies hardware.

    Dropped: the hostname hash, adapter and monitor device paths, and the instance part of
    every PnP device ID (``PCI\\VEN_10DE&DEV_28E0&SUBSYS_...`` survives; the
    ``\\4&1B0D88EE&0&0008`` after it, which is unique to one machine, does not).
    Replaced: the user profile path, and the user name wherever else it appears.
    ``redactions`` is that list of (identifying text, placeholder) pairs.
    """
    if redactions is None:
        home = str(Path.home())
        user = os.environ.get("USERNAME") or Path.home().name
        redactions = [(home, "%USERPROFILE%")] + ([(user, "<user>")] if len(user) >= 3 else [])
    if isinstance(value, dict):
        # Every *device_path carries the same per-machine instance ID in
        # `\\?\PCI#VEN_...#4&1b0d88ee&0&0008#{...}` form.
        return {k: sanitise(v, redactions=redactions) for k, v in value.items()
                if k not in ("hostname_hash", "logs", "stdout_log", "stderr_log", "run_dir",
                             "python_executable") and not k.endswith("device_path")}
    if isinstance(value, list):
        return [sanitise(v, redactions=redactions) for v in value]
    if isinstance(value, str):
        value = _INSTANCE.sub(r"\1", value)
        for identifying, placeholder in redactions:
            value = re.sub(re.escape(identifying), lambda _m, p=placeholder: p, value, flags=re.I)
    return value


def _environment(payloads: list):
    for payload in payloads:
        if payload and payload.get("environment"):
            return payload["environment"]
    return None


def build_report(pre: dict, skipped: dict, pixel=None, call=None, memory=None,
                 capabilities=None, target="cuda") -> dict:
    """``pixel`` is None when no path could run; the rest still reports."""
    pixel = pixel or {}
    pixel_passes = [_load(f) for name, f in pixel.items() if name != "verify"]
    memory_payload = _load(memory) if memory else None
    probed = _load(capabilities) if capabilities else None
    report = {
        "schema": "rapidshot-benchmark/1",
        "passes": len(pixel_passes),
        "recorded": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "preflight": pre,
        "skipped": skipped,
        "pixel_age": {"target": target,
                      "summary": summarise(pixel_passes, PIXEL_AGE),
                      "verify": statuses([_load(pixel["verify"])] if "verify" in pixel else []),
                      "statuses": statuses(pixel_passes)},
        # The memory runner records the same environment, so a machine with no
        # tensor path still says what it is.
        "environment": _environment(pixel_passes + [memory_payload]),
    }
    report["hdr"] = hdr_state(report["environment"], probed)
    report["cross_adapter"] = (probed or {}).get("cross_adapter")
    if call:
        call_passes = [_load(f) for name, f in call.items() if name != "verify"]
        report["call_duration"] = {"summary": summarise(call_passes, CALL_DURATION),
                                   "verify": statuses([_load(call["verify"])]),
                                   "statuses": statuses(call_passes)}
    if memory:
        report["memory"] = {"seconds": (memory_payload or {}).get("seconds"),
                            "summary": summarise_memory(memory_payload),
                            "skipped": memory_skipped(pre)}
    return sanitise(report)


def _native_label(native: dict) -> str:
    if not native:
        return "absent"
    version = native.get("wheel_version") or native.get("version") or "?"
    return version if native.get("source") == "rapidshot-native wheel" else f"{version} (development build)"


def _windows_label(pre: dict) -> str:
    release, version = pre.get("windows_release"), pre.get("windows")
    return f"Windows {release} ({version})" if release else f"Windows {version}"


def _hdr_line(hdr: dict) -> str:
    display, capture = hdr.get("display") or {}, hdr.get("capture") or {}
    if display:
        text = (f"{'on' if display.get('hdr_enabled') else 'off'} "
                f"(panel {'supports' if display.get('hdr_supported') else 'does not support'} HDR; "
                f"{display.get('mode', '?')}, {display.get('bits_per_channel', '?')} bpc)")
    else:
        text = "unknown"
    if capture.get("format"):
        text += f"; capture receives {capture['format']}"
    elif capture.get("error"):
        text += f"; capture format not read ({capture['error']})"
    return text


def _cross_adapter_line(probe) -> str:
    if not probe:
        return "not probed"
    if not probe.get("supported"):
        return f"not supported — {probe.get('reason') or 'no reason given'}"
    # Plain "to", not an arrow: the report is printed, and a redirected
    # Windows console is cp1252.
    line = f"supported, {probe.get('source', '?')} to {probe.get('destination', '?')}"
    if probe.get("copy_ms_median") is not None:
        line += f", {probe['copy_ms_median']:.2f} ms per 1080p copy"
    if not probe.get("representative"):
        line += " (to WARP: proves the mechanism, not the cost)"
    return line


def _num(value, digits=1, sign=False):
    return "—" if value is None else f"{value:{'+' if sign else ''}.{digits}f}"


def _cell(stat, digits=1):
    return "—" if not stat else f"{stat['median']:.{digits}f}"


#: Where a no-CUDA row's tensor ends up; every CPU capture path's is system memory.
FINISHED_IN = {"rapidshot-converter": "capture GPU (D3D12)"}


def _pixel_age_heading(report: dict) -> str:
    passes = report.get("passes")
    tail = (f"; medians of {passes} pass{'' if passes == 1 else 'es'}. "
            "Pixel age is from `Present()` to the tensor." if passes else ".")
    if report["pixel_age"].get("target") == "no-cuda":
        # Its own table: these rows answer the README's question with a
        # different finish line, and must never be read beside its CUDA rows.
        return ("Screen to a (1, 3, 640, 640) FP16 tensor, **no CUDA**: in system memory "
                "after CPU capture, on the capture GPU (D3D12) after GpuConverter. Not "
                "comparable with the CUDA table in the README" + tail)
    return "Screen to a (1, 3, 640, 640) FP16 tensor on CUDA" + tail


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
        f"Python {pre.get('python')}, {_windows_label(pre)}",
        f"- **HDR:** {_hdr_line(report.get('hdr') or {})}",
        f"- **Cross-adapter:** {_cross_adapter_line(report.get('cross_adapter'))}",
        "",
        _pixel_age_heading(report),
        "",
    ]
    no_cuda = report["pixel_age"].get("target") == "no-cuda"
    if report["pixel_age"]["summary"]:
        lines += (["| path | tensor in | unique fps | pixel age p50 / p95 | CPU per frame | passes |",
                   "| --- | --- | ---: | ---: | ---: | ---: |"] if no_cuda else
                  ["| path | unique fps | pixel age p50 / p95 | CPU per frame | passes |",
                   "| --- | ---: | ---: | ---: | ---: |"])
    else:
        lines.append("No path produced a tensor.")
    for path, row in report["pixel_age"]["summary"].items():
        where = f" {FINISHED_IN.get(path, 'system memory')} |" if no_cuda else ""
        lines.append(f"| {LABELS.get(path, path)} |{where} {_cell(row['unique_fps'])}{_marks(row)} | "
                     f"{_cell(row['age_p50_ms'])} / {_cell(row['age_p95_ms'])} ms | "
                     f"{_cell(row['cpu_ms_per_frame'])} ms | {_passes_cell(row)} |")
    lines += _contamination_notes(report["pixel_age"]["summary"])
    missing = {p: s for p, s in report["pixel_age"]["statuses"].items()
               if p not in report["pixel_age"]["summary"]}
    if missing or report["skipped"]:
        lines += ["", "Not measured:"]
        # One line per reason: without CuPy it is the same one for every path.
        by_reason = {}
        for path, reason in report["skipped"].items():
            by_reason.setdefault(reason, []).append(LABELS.get(path, path))
        for reason, labels in by_reason.items():
            lines.append(f"- {', '.join(labels)}: {reason}")
        for path, runs in missing.items():
            reason = next((r["reason"] for r in runs if r["reason"]), runs[0]["status"])
            lines.append(f"- {LABELS.get(path, path)}: {runs[0]['status']} — {reason}")
    if report["pixel_age"].get("target") == "no-cuda" and "memory" in report:
        lines += ["", "CPU per tensor (call-duration harness): not measured; it ends on CUDA. "
                  "The CPU per frame above is the same cost, from the pixel-age frames."]
    if "call_duration" in report:
        lines += ["", "CPU per tensor (call-duration harness; its fps is bounded by its test window):",
                  ""]
        if report["call_duration"]["summary"]:
            lines += ["| path | CPU per frame | call p50 |", "| --- | ---: | ---: |"]
        else:
            lines.append("No path produced a tensor.")
        for path, row in report["call_duration"]["summary"].items():
            lines.append(f"| {LABELS.get(path, path)} | {_cell(row['cpu_ms_per_frame'], 2)} ms | "
                         f"{_cell(row['call_p50_ms'], 2)} ms |")
    if "memory" in report:
        lines += ["", "Memory: working set, what capture adds over the same process before its "
                  "first frame, and growth as a least-squares slope after warm-up.", "",
                  "| library | screen | fps | working set | capture adds | growth |",
                  "| --- | --- | ---: | ---: | ---: | ---: |"]
        for library, workloads in report["memory"]["summary"].items():
            for workload, m in workloads.items():
                if "error" in m:
                    lines.append(f"| {library} | {workload} | — | — | — | {m['error']} |")
                    continue
                growth = m.get("growth_mb_per_s")
                lines.append(f"| {library} | {workload} | {_num(m.get('fps'))} | "
                             f"{_num(m.get('working_set_mb'))} MB | "
                             f"{_num(m.get('capture_mb'), sign=True)} MB | "
                             f"{'—' if growth is None else f'{growth:+.3f} MB/s'} |")
        memory_skips = report["memory"].get("skipped") or {}
        if memory_skips:
            lines += ["", "Memory not measured:"]
            lines += [f"- {library}: {reason}" for library, reason in memory_skips.items()]
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
    print(f"  will measure: {', '.join(paths) or 'no path to a tensor'}")
    if paths and tensor_target(pre) == "no-cuda":
        print("  no CUDA: tensors finish in system memory, or on the capture GPU for "
              "GpuConverter; reported in their own table")
    for path, reason in skipped.items():
        print(f"  skipping {path}: {reason}")
    if args.full:
        print(f"  memory: {', '.join(memory_libraries(pre))}")
        for library, reason in memory_skipped(pre).items():
            print(f"  skipping {library} memory: {reason}")
    steps = 1 + args.passes
    harnesses = 2 if args.full and tensor_target(pre) == "cuda" else 1
    estimate = steps * len(paths) * (args.seconds + 6) * harnesses
    if args.full:
        estimate += len(MEMORY_WORKLOADS) * len(memory_libraries(pre)) * (args.seconds + 4)
    print(f"  about {max(1, round(estimate / 60))} min. A test pattern will fill the screen; "
          "leave the machine alone until it finishes.", flush=True)


#: SetThreadExecutionState flags: ES_CONTINUOUS | ES_SYSTEM_REQUIRED |
#: ES_DISPLAY_REQUIRED while running, then ES_CONTINUOUS alone to release them.
_ES_CONTINUOUS = 0x80000000
_KEEP_DISPLAY_ON = _ES_CONTINUOUS | 0x00000001 | 0x00000002


def _set_execution_state(flags: int) -> None:
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(flags)
    except (AttributeError, OSError):
        pass  # not Windows; there is no display timeout to hold off


@contextlib.contextmanager
def display_held():
    """Keep the display on for the run. A benchmark is left alone by design, so
    the idle timeout turns the display off partway, and every test source
    launched after that exits with DXGI_STATUS_OCCLUDED, so an unattended --full
    can die partway (seen on the Intel desktop, with a 10-minute timeout)."""
    _set_execution_state(_KEEP_DISPLAY_ON)
    try:
        yield
    finally:
        _set_execution_state(_ES_CONTINUOUS)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="rapidshot benchmark", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also measure CPU per tensor (call-duration harness) and memory")
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
    if not paths and not args.full:
        # Only the capability probe would run; --full still measures memory,
        # and records the machine, HDR state and captured format beside it.
        problems.append("no path to a tensor can run here (see above); run with --full "
                        "to measure memory, HDR and capture format without them")
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
    with display_held():
        capabilities = run_capabilities(run_dir)
        target = tensor_target(pre)
        pixel = (run_pixel_age(run_dir, paths, args.passes, args.seconds, target)
                 if paths else None)
        # The call-duration harness has one finish line, CUDA; without it the
        # pixel-age table's CPU-per-frame column is the CPU cost, from the same frames.
        call = (run_call_duration(run_dir, paths, args.passes, args.seconds)
                if args.full and paths and target == "cuda" else None)
        memory = run_memory(run_dir, pre, args.seconds) if args.full else None

    report = build_report(pre, skipped, pixel, call, memory, capabilities, target)
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
