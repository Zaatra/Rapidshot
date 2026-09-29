"""The memory instrument itself, which has to be trustworthy before its
numbers are.

`benchmarks/memory_profile.py` is what says whether ROADMAP § 7.2's memory
target is met, so a fault in it does not produce an obviously broken run -- it
produces a plausible number that sends the work somewhere useless. That
already happened once in miniature: a 10-second run reported `grab()` growing
at **+1.29 MB/s**, which at 60 seconds turned out to be **+0.066** -- warm-up
allocations bleeding past the cutoff and being read as a leak.

So the windowing and the slope are what these tests are mostly about:

* **Warm-up must actually be excluded**, or the first allocations of a session
  are reported as unbounded growth.
* **A slope needs enough points to be a slope.** Two samples through any two
  values give a perfectly confident straight line.
* **Capture cost is steady state minus setup**, so RapidShot is not charged for
  importing `comtypes` where DXcam has no equivalent.
* **Change detection is not frame production.** Desktop Duplication reports
  only changed content, so on the static workload `grab()` mostly returns None,
  and counting those as frames would report a busy capture loop on an idle
  screen.

Everything here is headless: no capture, no GPU, no subprocesses that do real
work.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import memory_profile as mp


class Clock:
    """Returns the current time, then advances. Deterministic runs."""

    def __init__(self, step=0.05):
        self.now, self.step = 0.0, step

    def __call__(self):
        value = self.now
        self.now += self.step
        return value


class FakeProcess:
    """psutil stand-in whose working set follows a script."""

    def __init__(self, series):
        self.series, self.calls = list(series), 0

    def memory_info(self):
        value = self.series[min(self.calls, len(self.series) - 1)]
        self.calls += 1
        return SimpleNamespace(rss=value, private=value, pagefile=value,
                               peak_wset=max(self.series[:self.calls] or [value]),
                               vms=value)

    def cpu_times(self):
        return SimpleNamespace(user=0.0, system=0.0)


# -- the slope: the figure that says "leak" or "not" ----------------------


def test_a_real_slope_is_reported():
    times = [float(i) for i in range(10)]
    values = [100e6 + i * 2e6 for i in range(10)]  # 2 MB per second
    assert mp._slope_mb_per_s(times, values) == pytest.approx(2.0)


def test_a_flat_series_reports_no_growth():
    assert mp._slope_mb_per_s([float(i) for i in range(10)], [100e6] * 10) == 0.0


def test_two_samples_are_not_a_slope():
    """Any two points lie on a line, and that line would be reported with full
    confidence. This is the difference between measuring growth and inventing
    it."""
    assert mp._slope_mb_per_s([0.0, 1.0], [100e6, 200e6]) is None
    assert mp._slope_mb_per_s([], []) is None


def test_a_series_with_no_elapsed_time_reports_nothing():
    """Every sample at the same instant gives a zero denominator; the answer is
    'cannot say', not a division by zero."""
    assert mp._slope_mb_per_s([1.0, 1.0, 1.0], [1e6, 2e6, 3e6]) is None


def test_a_falling_series_is_negative():
    """DXcam measures slightly negative in practice. Clamping at zero would
    hide a series that is still moving."""
    times = [float(i) for i in range(10)]
    assert mp._slope_mb_per_s(times, [100e6 - i * 1e6 for i in range(10)]) < 0


# -- windowing ------------------------------------------------------------


@pytest.fixture
def run(monkeypatch):
    """Drive `run_worker` over a scripted memory series and a fake clock."""
    def go(series, seconds=1.0, warmup=0.3, grab=lambda: True):
        process = FakeProcess(series)
        monkeypatch.setitem(sys.modules, "psutil",
                            SimpleNamespace(Process=lambda: process))
        monkeypatch.setattr(mp, "WARMUP_SECONDS", warmup)
        monkeypatch.setattr(mp, "SAMPLE_INTERVAL", 0.0)
        monkeypatch.setattr(mp.time, "perf_counter", Clock())
        monkeypatch.setattr(mp, "stage", lambda *a, **kw: None)
        monkeypatch.setattr(mp, "_capture_loop",
                            lambda library: (grab, lambda: None, "fake"))
        return mp.run_worker("rapidshot", seconds)
    return go


def test_warmup_allocations_are_not_reported_as_growth(run):
    """The failure this instrument already had once. A session that allocates
    hard and then settles must report the settled behaviour, not the ramp."""
    # Six values: the baseline read plus the five samples that land before
    # the 0.3 s cutoff at this clock step. Any longer and the ramp is partly
    # inside the steady window, which is the very confusion being tested.
    ramp = [100e6 + i * 5e6 for i in range(6)]        # climbing, inside warm-up
    settled = [140e6] * 40                            # flat afterwards
    result = run(ramp + settled)
    assert result["working_set_growth_mb_per_s"] == pytest.approx(0.0, abs=0.5)
    assert result["working_set_mb"] == pytest.approx(140.0, abs=1.0)


def test_a_run_too_short_to_have_a_steady_window_still_reports(run):
    """Falling back to every sample is better than reporting nothing: a short
    run is less trustworthy, not meaningless."""
    result = run([100e6] * 40, seconds=0.2, warmup=10.0)
    assert result["steady_samples"] == result["samples"]
    assert result["working_set_mb"] == pytest.approx(100.0)


def test_capture_cost_excludes_what_setup_already_paid(run):
    """Comparing raw working sets charges RapidShot for its imports. The
    number that compares across libraries is steady state minus the same
    process before its first frame."""
    result = run([100e6] * 40)
    assert result["baseline"]["working_set"] == pytest.approx(100.0)
    assert result["working_set_over_baseline_mb"] == pytest.approx(0.0)


def test_peak_is_the_highest_sample_not_the_last(run):
    """A spike inside the measured window has to survive into the report; the
    median would smooth it away."""
    result = run([100e6] * 10 + [300e6] + [100e6] * 40)
    assert result["working_set_peak_mb"] >= 300.0
    assert result["working_set_mb"] < 300.0


def test_the_steady_peak_ignores_a_warmup_spike(run):
    """Two peaks, deliberately: the steady one answers "how big does this get
    once running", and a spike while the pools are still filling is not that.
    The OS's own peak still records it, and cannot miss one between samples."""
    result = run([100e6, 100e6, 300e6] + [100e6] * 40)
    assert result["working_set_peak_mb"] == pytest.approx(100.0)
    assert result["os_peak_working_set_mb"] >= 300.0


def test_every_accounting_key_is_reported(run):
    result = run([100e6] * 40)
    for key in ("working_set", "private", "commit"):
        for suffix in ("_mb", "_peak_mb", "_growth_mb_per_s", "_over_baseline_mb"):
            assert key + suffix in result, key + suffix


# -- frames versus change detection ---------------------------------------


def test_unchanged_frames_are_not_counted_as_captured(run):
    """Desktop Duplication returns nothing when the screen has not changed. On
    the static workload that is most calls, and counting them would report a
    capture rate the screen never produced."""
    calls = {"n": 0}

    def grab():
        calls["n"] += 1
        return calls["n"] % 4 == 0        # one real frame in four

    result = run([100e6] * 60, grab=grab)
    assert result["misses"] > result["frames"] > 0
    assert result["fps"] < calls["n"] / result["elapsed_seconds"]


def test_a_failing_adapter_is_an_error_row_not_a_crash(run):
    def grab():
        raise RuntimeError("duplication refused")

    result = run([100e6] * 40, grab=grab)
    assert "duplication refused" in result["error"]


# -- the workload source --------------------------------------------------


@pytest.fixture
def logs(tmp_path):
    from ai_ingestion import RunLogs
    return RunLogs(tmp_path)


def source(logs, monkeypatch, *, lines=b"", exits=None, binary=True, steady=None):
    """``steady`` is what each launch's healthy-start check returns, in order;
    by default every launch presents steadily."""
    verdicts = list(steady or [])
    monkeypatch.setattr(mp, "presenting_steadily",
                        lambda path, fps, since=0: (verdicts.pop(0) if verdicts else True, 7))
    monkeypatch.setattr(mp, "SOURCE", SimpleNamespace(
        is_file=lambda: binary, __str__=lambda self: "latency_source.exe"))

    class Proc:
        returncode = exits

        def __init__(self, *a, **kw):
            if lines:
                kw["stdout"].write(lines)
                kw["stdout"].flush()
            self.killed = self.terminated = False

        def poll(self):
            return exits

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return exits

        def kill(self):
            self.killed = True

    monkeypatch.setattr(mp.subprocess, "Popen", Proc)
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=1.0))
    monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
    return mp.WorkloadSource(logs, "static", 900, 700, 60)


def test_the_source_is_ready_when_it_says_so(logs, monkeypatch):
    src = source(logs, monkeypatch, lines=b'{"event":"ready","width":900}\n')
    src.start()
    assert '"event": "source-ready"' in (logs.directory / "parent.log").read_text()


def test_a_source_that_dies_before_ready_is_not_waited_out(logs, monkeypatch):
    """Otherwise a source that refused its dimensions costs the full timeout
    before saying anything, once per workload."""
    src = source(logs, monkeypatch, exits=1)
    with pytest.raises(RuntimeError, match="exited with 1"):
        src.start()


def test_a_silent_source_times_out(logs, monkeypatch):
    src = source(logs, monkeypatch)
    with pytest.raises(RuntimeError, match="readiness timeout"):
        src.start()


def test_a_restarted_source_is_not_ready_on_its_predecessors_word(logs, monkeypatch):
    """The logs are appended so a restart keeps the stderr that says why the
    first source died -- which leaves the first one's ready line in stdout."""
    source(logs, monkeypatch, lines=b'{"event":"ready","width":900}\n').start()
    stderr = logs.directory / "source-static.stderr.log"
    stderr.write_bytes(b"Present not visible/successful: HRESULT(0x087A0001)\n")
    with pytest.raises(RuntimeError, match="readiness timeout"):
        source(logs, monkeypatch).start()
    assert b"0x087A0001" in stderr.read_bytes()


READY = b'{"event":"ready","width":900}' + b"\n"


def test_a_source_that_stops_after_its_first_frames_is_relaunched(logs, monkeypatch):
    """About 15% of later launches on the Intel desktop presented 3 frames and
    then had every flip held for 2 s; a relaunch cleared it every time."""
    src = source(logs, monkeypatch, lines=READY, steady=[False, True])
    src.start()
    events = (logs.directory / "parent.log").read_text()
    assert events.count('"event": "source-ready"') == 2
    assert '"event": "source-unhealthy"' in events and '"attempt": 1' in events


def test_a_source_that_never_presents_steadily_stops_the_workload(logs, monkeypatch):
    src = source(logs, monkeypatch, lines=READY, steady=[False, False, False])
    with pytest.raises(RuntimeError, match="stopped presenting after its first frames on 3"):
        src.start()
    assert (logs.directory / "parent.log").read_text().count("source-unhealthy") == 3


def test_presents_are_counted_from_the_sources_own_log(logs, monkeypatch):
    src = source(logs, monkeypatch)
    assert src.presents() == 0
    src.present_log.write_bytes(b'{"event":"present","id":1}\n{"event":"present","id":2}\n'
                                b'{"event":"other"}\n{"event":"present","id":3}\n')
    assert src.presents() == 3


@pytest.mark.parametrize("presented, wall, fps, exit_code, expected", [
    (880, 9.0, 100, None, None),
    (1, 9.0, 100, None, "presented 1 frames in 9.0 s against about 900"),
    (440, 9.0, 100, None, "presented 440 frames"),
    (900, 9.0, 100, 1, "exited (status 1)"),
    (0, 9.0, 0, None, None),
])
def test_a_case_whose_source_stopped_says_so(presented, wall, fps, exit_code, expected):
    """1 frame in 8 s from DXcam and RapidShot, while mss kept re-grabbing, is a
    source that stopped -- and only the source can tell that from a still desktop."""
    reason = mp.source_stall(presented, wall, fps, exit_code)
    if expected is None:
        assert reason is None
    else:
        assert expected in reason and "Rerun" in reason


class FakeSource:
    def __init__(self, logs, presents, exit_code=None, fps=100):
        self.logs, self.workload, self.width, self.height, self.fps = logs, "scroll", 900, 700, fps
        self._presents, self._exit, self.closed = list(presents), exit_code, False

    def presents(self):
        return self._presents.pop(0)

    def exit_code(self):
        return self._exit

    def close(self):
        self.closed = True


@pytest.mark.parametrize("library", ["mss", "dxcam"])
def test_a_row_measured_while_the_source_was_stalled_is_failed_with_why(
        logs, monkeypatch, library):
    """mss's numbers from that screen look fine, and are just as mislabelled."""
    monkeypatch.setattr(mp, "spawn_worker", lambda *a: {"library": library, "fps": 50.0})
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=9.0))
    row = mp.measure_case(FakeSource(logs, [100, 101]), library, "scroll", 8.0, logs)
    assert "presented 1 frames" in row["error"]


def test_a_row_measured_while_the_source_ran_is_left_alone(logs, monkeypatch):
    monkeypatch.setattr(mp, "spawn_worker", lambda *a: {"library": "dxcam", "fps": 99.0})
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=9.0))
    row = mp.measure_case(FakeSource(logs, [100, 990]), "dxcam", "scroll", 8.0, logs)
    assert "error" not in row


def test_a_source_asked_for_more_than_the_panel_shows_is_not_a_stall(logs, monkeypatch):
    """--source-fps 240 on a 100 Hz panel presents 100 a second under vsync."""
    monkeypatch.setattr(mp, "spawn_worker", lambda *a: {"library": "dxcam", "fps": 99.0})
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=9.0))
    row = mp.measure_case(FakeSource(logs, [0, 890], fps=240), "dxcam", "scroll", 8.0, logs,
                          rate=min(240, 100))
    assert "error" not in row


def test_a_worker_error_is_not_overwritten_by_the_stall(logs, monkeypatch):
    monkeypatch.setattr(mp, "spawn_worker", lambda *a: {"library": "dxcam", "error": "worker timeout"})
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=9.0))
    row = mp.measure_case(FakeSource(logs, [0, 0]), "dxcam", "scroll", 8.0, logs)
    assert row["error"] == "worker timeout"


def test_a_source_that_exited_is_replaced_before_the_next_case(logs, monkeypatch):
    started = []
    monkeypatch.setattr(mp.WorkloadSource, "start", lambda self: started.append(self))
    dead = FakeSource(logs, [], exit_code=1)
    fresh = mp.running_source(dead)
    assert dead.closed and started == [fresh]
    assert (fresh.workload, fresh.width, fresh.height, fresh.fps) == ("scroll", 900, 700, 100)
    assert '"exit_code": 1' in (logs.directory / "parent.log").read_text()
    alive = FakeSource(logs, [])
    assert mp.running_source(alive) is alive


def test_a_missing_source_names_the_command_that_builds_it(logs, monkeypatch):
    """The binary is a Rust target, so 'not found' is not actionable on its own
    -- the message has to carry the cargo line."""
    src = source(logs, monkeypatch, binary=False)
    with pytest.raises(RuntimeError, match="cargo build"):
        src.start()


# -- worker results -------------------------------------------------------


def worker_env(monkeypatch, stdout_text, *, times_out=False, returncode=0):
    """Fake the worker subprocess. Closure state, not module state: a global
    here would leak the flag between tests and make the order matter.

    `wait` keeps its `timeout` keyword because `spawn_worker` passes it by
    name, so the flag is called something else rather than shadowing it.
    """
    class Proc:
        def __init__(self, *a, **kw):
            kw["stdout"].write(stdout_text.encode())
            self.returncode = returncode
            self.killed = False

        def wait(self, timeout=None):
            if times_out:
                raise mp.subprocess.TimeoutExpired("cmd", timeout or 1)
            return returncode

        def kill(self):
            self.killed = True

    monkeypatch.setattr(mp.subprocess, "Popen", Proc)


def test_a_worker_result_is_read_from_its_last_json_line(logs, monkeypatch):
    """Adapters log to stdout as well, so the result is the last JSON line and
    not the first thing that happens to look like one."""
    worker_env(monkeypatch, 'noise\n{"library":"x","fps":1}\n{"library":"rapidshot","fps":9}\n')
    assert mp.spawn_worker("rapidshot", "static", 1.0, logs)["fps"] == 9


def test_a_worker_that_printed_nothing_usable_is_an_error(logs, monkeypatch):
    """Silently returning an empty row would put a blank line in the table
    where a failure belongs."""
    worker_env(monkeypatch, "traceback, no json\n", returncode=1)
    result = mp.spawn_worker("rapidshot", "static", 1.0, logs)
    assert "no worker result" in result["error"]
    assert "exit 1" in result["error"]


def test_malformed_json_does_not_stop_the_search(logs, monkeypatch):
    worker_env(monkeypatch, '{"library":"rapidshot","fps":5}\n{"truncated"\n')
    assert mp.spawn_worker("rapidshot", "static", 1.0, logs)["fps"] == 5


def test_a_hung_worker_is_killed_and_reported(logs, monkeypatch):
    worker_env(monkeypatch, "", times_out=True)
    assert mp.spawn_worker("rapidshot", "static", 1.0, logs)["error"] == "worker timeout"


# -- the table ------------------------------------------------------------


def test_the_table_prints_failures_rather_than_dropping_them(capsys):
    mp.print_table([{"workload": "static", "library": "dxcam",
                     "error": "ModuleNotFoundError: dxcam"}])
    assert "ERROR" in capsys.readouterr().out


def test_a_row_without_a_slope_does_not_break_the_table(capsys):
    """`None` is what a run too short to have a slope reports, and it reaches
    the formatter."""
    mp.print_table([{"workload": "static", "library": "rapidshot", "fps": 1.0,
                     "working_set_mb": 1.0, "private_mb": 1.0, "commit_mb": 1.0,
                     "working_set_over_baseline_mb": 1.0,
                     "working_set_peak_mb": 1.0,
                     "working_set_growth_mb_per_s": None}])
    assert "n/a" in capsys.readouterr().out


# -- one source for the run, switched over stdin --------------------------------

class SwitchingProc:
    """A running source whose stdin either acknowledges a workload switch, as
    rapidshot-native 0.2.2's does, or ends the process, as 0.2.1's did."""

    def __init__(self, stdout_path, acknowledges=True):
        self.stdout_path, self.acknowledges, self.exited = stdout_path, acknowledges, None
        self.written = []
        outer = self

        class Stdin:
            def write(self, data):
                outer.written.append(data)
                if outer.acknowledges:
                    name = data.decode().strip()
                    with outer.stdout_path.open("ab") as out:
                        out.write(b'{"event":"workload","name":"' + name.encode() + b'"}\n')
                else:
                    outer.exited = 0

            def flush(self):
                pass

        self.stdin = Stdin()

    def poll(self):
        return self.exited


def switching_source(logs, monkeypatch, acknowledges=True, steady=True):
    monkeypatch.setattr(mp.time, "monotonic", Clock(step=0.25))
    monkeypatch.setattr(mp.time, "sleep", lambda _s: None)
    monkeypatch.setattr(mp, "presenting_steadily", lambda path, fps, since=0: (steady, 30))
    src = mp.WorkloadSource(logs, "static", 900, 700, 60)
    src.stdout_path.write_bytes(b'{"event":"ready"}\n')
    src.proc = SwitchingProc(src.stdout_path, acknowledges)
    return src


def test_a_source_switches_workload_in_place(logs, monkeypatch):
    src = switching_source(logs, monkeypatch)
    assert src.switch("scroll") and src.workload == "scroll"
    assert src.proc.written == [b"scroll\n"]
    assert '"event": "source-switched"' in (logs.directory / "parent.log").read_text()


def test_an_older_source_that_cannot_switch_is_replaced(logs, monkeypatch):
    """0.2.1's source stops on any stdin line; the run then relaunches, as
    every run did before."""
    src = switching_source(logs, monkeypatch, acknowledges=False)
    assert not src.switch("scroll") and src.workload == "static"
    launched = []
    monkeypatch.setattr(mp.WorkloadSource, "start", lambda self: launched.append(self.workload))
    monkeypatch.setattr(mp.WorkloadSource, "close", lambda self: None)
    fresh = mp.next_source(src, logs, "motion", 900, 700, 60)
    assert fresh is not src and launched == ["motion"]


def test_a_source_that_stalls_after_switching_is_replaced(logs, monkeypatch):
    src = switching_source(logs, monkeypatch, steady=False)
    assert not src.switch("scroll")
    assert '"after": "switch"' in (logs.directory / "parent.log").read_text()


def test_next_source_keeps_a_source_that_switched(logs, monkeypatch):
    src = switching_source(logs, monkeypatch)
    assert mp.next_source(src, logs, "motion", 900, 700, 60) is src
