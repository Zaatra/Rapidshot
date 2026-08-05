"""Compare RapidShot against other Windows capture libraries, honestly.

Most capture-library comparisons on the internet are wrong in the same few ways.
This one tries not to be, and states its limits rather than hiding them.

**Every library runs in its own process.** This is not tidiness, it is a
correctness requirement discovered the hard way: RapidShot, DXcam and BetterCam
all declare the same DXGI COM interfaces, and comtypes keeps a process-global
registry keyed by IID, so whichever imports first wins and the others get
mismatched argtypes. `import rapidshot` followed by `import dxcam` raises
`ArgumentError: expected LP_c_ulong instance`. Measuring two of these libraries
in one interpreter measures the collision, not the libraries.

**Stale frames are labelled, not hidden.** Desktop Duplication returns a frame
only when the screen changed. A tight `grab()` loop against a still desktop
therefore reports enormous FPS for doing nothing, which is how "1000+ FPS"
numbers get published. Runs are tagged `motion=True/False`; compare only within
a tag, and treat the still-desktop numbers as a measure of what a library does
when there is nothing to do -- useful, but not a capture rate.

**The motion source is part of the apparatus, so calibrate it.** Run
`benchmarks/motion_source.py` alongside this and read the updates/s it prints. A
source slower than the display's refresh rate silently becomes the ceiling, and
every library converges on *its* rate rather than their own. That is not
hypothetical: a 30 Hz generator made all four libraries look tied at 40-50 fps on
a 100 Hz panel, which read exactly like a hardware limit. At ~610 updates/s the
same libraries spread across 114-169 fps. Check the display's real refresh rate
too (`EnumDisplaySettingsW`) rather than assuming it.

**Colour format is held equal.** mss returns BGRA and performs no conversion,
while the DXGI libraries can convert during capture. Comparing mss-BGRA against
RapidShot-RGB measures a conversion that only one side performed. Each scenario
therefore runs in BGRA (no conversion anywhere) and again in RGB, so the
conversion cost is visible as the difference rather than smuggled into a total.

    python benchmarks/compare_libraries.py                # run everything
    python benchmarks/compare_libraries.py --seconds 8
    python benchmarks/compare_libraries.py --worker rapidshot fullscreen BGRA
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

LIBRARIES = ("rapidshot", "rapidshot-poll", "dxcam", "bettercam", "mss")
SCENARIOS = ("fullscreen", "region")
COLOURS = ("BGRA", "RGB")
REGION = (760, 340, 1160, 740)          # 400x400, centred on a 1080p display


# ---------------------------------------------------------------------------
# resource sampling
# ---------------------------------------------------------------------------

class ResourceMonitor:
    """Samples CPU and RSS from a daemon thread.

    In-thread sampling would perturb the very loop being measured -- the GIL is
    held by whichever thread is running, so a `psutil` call between grabs adds
    itself to the frame time. A separate thread still contends for the GIL but
    does not serialise into the capture loop.
    """

    def __init__(self, hz: float = 20.0):
        self.interval = 1.0 / hz
        self._stop = threading.Event()
        self.cpu: List[float] = []
        self.rss: List[int] = []
        self._thread: Optional[threading.Thread] = None
        try:
            import psutil
            self._proc = psutil.Process()
            self._proc.cpu_percent(interval=None)   # prime the counter
        except Exception:
            self._proc = None

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.cpu.append(self._proc.cpu_percent(interval=None))
                self.rss.append(self._proc.memory_info().rss)
            except Exception:
                break

    def __enter__(self):
        if self._proc is not None:
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def summary(self) -> dict:
        if not self.cpu:
            return {"cpu_percent_mean": None, "cpu_percent_max": None,
                    "rss_mb_start": None, "rss_mb_end": None, "rss_mb_growth": None}
        return {
            "cpu_percent_mean": round(statistics.fmean(self.cpu), 1),
            "cpu_percent_max": round(max(self.cpu), 1),
            "rss_mb_start": round(self.rss[0] / 1e6, 1),
            "rss_mb_end": round(self.rss[-1] / 1e6, 1),
            "rss_mb_growth": round((self.rss[-1] - self.rss[0]) / 1e6, 1),
        }


# ---------------------------------------------------------------------------
# per-library adapters, each returning a zero-argument grab callable
# ---------------------------------------------------------------------------

def _adapter_rapidshot(scenario: str, colour: str, timeout_ms: int = 10):
    sys.path.insert(0, str(REPO))
    import rapidshot
    from rapidshot import native

    cam = rapidshot.create(output_color=colour, timeout_ms=timeout_ms)
    region = REGION if scenario == "region" else None

    def grab():
        frame = cam.grab(region=region) if region else cam.grab()
        if frame is None:
            return None
        # Pooled buffers must be handed back or the pool starves; releasing is
        # part of this library's per-frame cost and belongs inside the timing.
        release = getattr(frame, "release", None)
        if release:
            release()
        return True

    return grab, cam, {"native_extension": native.is_available(),
                       "timeout_ms": timeout_ms}


def _adapter_dxcam(scenario: str, colour: str):
    import dxcam
    cam = dxcam.create(output_color=colour)
    region = REGION if scenario == "region" else None

    def grab():
        frame = cam.grab(region=region) if region else cam.grab()
        return None if frame is None else True

    return grab, cam, {}


def _adapter_bettercam(scenario: str, colour: str):
    import bettercam
    cam = bettercam.create(output_color=colour)
    region = REGION if scenario == "region" else None

    def grab():
        frame = cam.grab(region=region) if region else cam.grab()
        return None if frame is None else True

    return grab, cam, {}


def _adapter_mss(scenario: str, colour: str):
    import mss
    import numpy as np

    sct = mss.mss()
    if scenario == "region":
        left, top, right, bottom = REGION
        target = {"left": left, "top": top,
                  "width": right - left, "height": bottom - top}
    else:
        target = sct.monitors[1]

    def grab():
        shot = sct.grab(target)
        arr = np.asarray(shot)           # BGRA view over the raw buffer
        if colour == "RGB":
            # mss cannot convert during capture, so the conversion every other
            # library did internally has to happen here or the comparison is
            # not measuring the same work.
            arr = arr[..., 2::-1]
            arr = np.ascontiguousarray(arr)
        return arr.shape is not None

    return grab, sct, {}


ADAPTERS = {
    "rapidshot": _adapter_rapidshot,
    # The same library polling like DXcam does. Without this row the comparison
    # conflates two different things -- how fast a library can capture, and
    # whether it chose to spend a core finding out -- and reads as a throughput
    # deficit when it is mostly a scheduling policy.
    "rapidshot-poll": lambda s, c: _adapter_rapidshot(s, c, timeout_ms=0),
    "dxcam": _adapter_dxcam,
    "bettercam": _adapter_bettercam,
    "mss": _adapter_mss,
}


# ---------------------------------------------------------------------------
# control
# ---------------------------------------------------------------------------

def control_ms() -> float:
    """A fixed workload every process runs, so runs can be compared to runs.

    Without this the suite can rank libraries within one sitting and nothing
    more: two runs of untouched code drifted 20-35% apart, which swamps any
    change worth measuring. The control's code never changes, so its movement is
    the machine -- the same calibration perf_suite.py applies.
    """
    import numpy as np
    src = np.zeros((1080, 1920, 4), np.uint8)
    dst = np.empty_like(src)
    best = float("inf")
    for _ in range(20):
        t0 = time.perf_counter()
        np.copyto(dst, src)
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    return round(best, 4)


# ---------------------------------------------------------------------------
# worker: one library, one scenario, one process
# ---------------------------------------------------------------------------

def run_worker(library: str, scenario: str, colour: str,
               seconds: float, warmup: int) -> dict:
    grab, handle, extra = ADAPTERS[library](scenario, colour)

    for _ in range(warmup):
        grab()

    deltas: List[float] = []
    misses = 0
    with ResourceMonitor() as monitor:
        start = time.perf_counter()
        deadline = start + seconds
        previous = time.perf_counter()
        while time.perf_counter() < deadline:
            got = grab()
            now = time.perf_counter()
            if got is None:
                misses += 1
            else:
                deltas.append((now - previous) * 1000.0)
            previous = now
        elapsed = time.perf_counter() - start

    try:
        del handle
    except Exception:
        pass

    if not deltas:
        return {"library": library, "scenario": scenario, "colour": colour,
                "frames": 0, "error": "no frames returned", **extra}

    ordered = sorted(deltas)

    def pct(p: float) -> float:
        return round(ordered[min(int(len(ordered) * p), len(ordered) - 1)], 3)

    return {
        "library": library,
        "scenario": scenario,
        "colour": colour,
        "control_ms": control_ms(),
        "seconds": round(elapsed, 3),
        "frames": len(deltas),
        "misses": misses,
        "fps_mean": round(len(deltas) / elapsed, 1),
        "ms_mean": round(statistics.fmean(deltas), 3),
        "ms_p50": pct(0.50),
        "ms_p95": pct(0.95),
        "ms_p99": pct(0.99),
        "ms_min": round(ordered[0], 3),
        "ms_max": round(ordered[-1], 3),
        # Jitter as the stdev of inter-frame deltas: a high mean FPS with high
        # jitter still stutters, which an average alone never shows.
        "ms_jitter_stdev": round(statistics.pstdev(deltas), 3) if len(deltas) > 1 else 0.0,
        **monitor.summary(),
        **extra,
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def environment(motion: bool) -> dict:
    info = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "motion_on_screen": motion,
        "region": list(REGION),
    }
    for module in ("numpy", "dxcam", "bettercam", "mss", "psutil", "comtypes"):
        try:
            info[module] = __import__(module).__version__
        except Exception as exc:
            info[module] = f"unavailable ({type(exc).__name__})"
    try:
        sys.path.insert(0, str(REPO))
        import rapidshot
        info["rapidshot"] = rapidshot.__version__
    except Exception as exc:
        info["rapidshot"] = f"unavailable ({exc})"
    return info


def spawn(library: str, scenario: str, colour: str, seconds: float,
          warmup: int) -> dict:
    """One library, one fresh interpreter. See the module docstring."""
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker",
           library, scenario, colour, "--seconds", str(seconds),
           "--warmup", str(warmup)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=seconds + 120)
    for line in proc.stdout.splitlines():
        if line.startswith("{"):
            return json.loads(line)
    return {"library": library, "scenario": scenario, "colour": colour,
            "frames": 0,
            "error": (proc.stderr.strip().splitlines() or ["no output"])[-1][:200]}


def aggregate(samples: List[dict]) -> dict:
    """Median across independent runs of one cell.

    Median rather than mean: one scheduling hiccup in one repeat should not move
    the reported figure, and with an odd repeat count the median is a value that
    was actually observed rather than a blend of runs.

    The spread is kept and printed, because a cell whose repeats disagree cannot
    support a verdict no matter how tidy its median looks.
    """
    good = [s for s in samples if s.get("frames")]
    if not good:
        return samples[0]
    merged = dict(good[0])
    merged["repeats"] = len(good)
    for key in ("fps_mean", "ms_p50", "ms_p95", "ms_p99", "ms_jitter_stdev",
                "cpu_percent_mean", "rss_mb_end", "control_ms", "frames",
                "misses"):
        values = [s[key] for s in good if isinstance(s.get(key), (int, float))]
        if values:
            merged[key] = round(statistics.median(values), 3)
    fps = [s["fps_mean"] for s in good]
    mid = statistics.median(fps) if fps else 0
    merged["fps_spread_pct"] = (round((max(fps) - min(fps)) / mid * 100, 1)
                                if len(fps) > 1 and mid else 0.0)
    return merged


def table(rows: List[dict]) -> str:
    head = (f"{'library':<16}{'scenario':<12}{'fmt':<6}{'fps':>8}{'p50':>9}"
            f"{'p95':>9}{'p99':>9}{'jitter':>9}{'cpu%':>7}{'rssMB':>8}{'spread':>8}")
    out = [head, "-" * len(head)]
    for r in rows:
        if r.get("error") or not r.get("frames"):
            out.append(f"{r['library']:<16}{r['scenario']:<12}{r['colour']:<6}"
                       f"  {r.get('error', 'no frames')[:60]}")
            continue
        out.append(
            f"{r['library']:<16}{r['scenario']:<12}{r['colour']:<6}"
            f"{r['fps_mean']:>8.1f}{r['ms_p50']:>9.2f}{r['ms_p95']:>9.2f}"
            f"{r['ms_p99']:>9.2f}{r['ms_jitter_stdev']:>9.2f}"
            f"{(r.get('cpu_percent_mean') or 0):>7.1f}"
            f"{(r.get('rss_mb_end') or 0):>8.1f}"
            f"{(r.get('fps_spread_pct') or 0):>7.1f}%")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", nargs=3, metavar=("LIB", "SCENARIO", "COLOUR"),
                    help="internal: run one measurement in this process")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--warmup", type=int, default=120,
                    help="frames discarded before timing; DXGI has a real "
                         "first-acquire cost and every library needs it")
    ap.add_argument("--libraries", nargs="*", default=list(LIBRARIES))
    ap.add_argument("--motion", action="store_true",
                    help="assert that something is moving on screen; recorded "
                         "in the output so still-desktop runs cannot be "
                         "mistaken for capture rates")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent runs per cell; the median is reported and "
                         "the spread shown. One 6-second sample cannot tell a "
                         "change from scheduling noise -- untouched libraries "
                         "drifted 20-35%% between single-sample runs.")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if args.worker:
        library, scenario, colour = args.worker
        print(json.dumps(run_worker(library, scenario, colour,
                                    args.seconds, args.warmup)))
        return 0

    env = environment(args.motion)
    print("Capture library comparison")
    for key in ("timestamp", "platform", "processor", "python", "rapidshot",
                "dxcam", "bettercam", "mss", "motion_on_screen"):
        print(f"  {key:<18} {env[key]}")
    if not args.motion:
        print("\n  WARNING: --motion not set. Desktop Duplication only returns")
        print("  changed frames, so these numbers describe idling, not capture.")
    print()

    rows: List[dict] = []
    total = len(SCENARIOS) * len(COLOURS) * len(args.libraries) * args.repeats
    done = 0
    for scenario in SCENARIOS:
        for colour in COLOURS:
            for library in args.libraries:
                samples = []
                for rep in range(args.repeats):
                    done += 1
                    print(f"  [{done}/{total}] {library}/{scenario}/{colour} "
                          f"rep {rep + 1}", end="\r", flush=True)
                    samples.append(spawn(library, scenario, colour,
                                         args.seconds, args.warmup))
                rows.append(aggregate(samples))
    print(" " * 70, end="\r")

    print(table(rows))
    print("\nfps = frames returned per second; p50/p95/p99 and jitter are")
    print("inter-frame deltas in ms. Compare only within one scenario+format.")

    payload = {"environment": env, "results": rows}
    out = args.out or (HERE / "library-comparison.json")
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
