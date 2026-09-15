"""make_badges.py and compare_recordings.py: JSON in, text out, no measuring.

Both were at 0%. `make_badges.py --check` gates CI -- the README badges must
match `benchmarks/baseline.json` -- and `compare_recordings.py` produces the
drift-adjusted "what did this version buy" ratios quoted from stored
recordings. Neither touches the GPU.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import compare_recordings  # noqa: E402
import make_badges  # noqa: E402


# --------------------------------------------------------------------------
# make_badges
# --------------------------------------------------------------------------

def recording(rows, **machine):
    return {"machine": dict({"gpu": "Intel(R) Graphics", "frame": "1920x1080"}, **machine),
            "results": [{"name": name, "min_ms": ms} for name, ms in rows.items()]}


@pytest.fixture
def badge_env(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline.json"
    badges = tmp_path / "badges"
    monkeypatch.setattr(make_badges, "BASELINE", baseline)
    monkeypatch.setattr(make_badges, "BADGE_DIR", badges)
    baseline.write_text(json.dumps(recording(
        {"convert.RGB": 0.62, "convert.GRAY": 9.5, "shot.RGB": 17.0})), encoding="utf-8")
    return baseline, badges


@pytest.mark.parametrize("ms,expected", [
    (0.99, "brightgreen"), (1.0, "green"), (7.99, "green"),
    (8.0, "yellow"), (16.69, "yellow"), (16.7, "orange"),
])
def test_badge_colour_follows_the_60hz_budget(ms, expected):
    assert make_badges.colour(ms) == expected


def test_badges_are_built_from_the_baseline_minimums(badge_env):
    badges = make_badges.build()

    assert badges["convert-rgb.json"] == {
        "schemaVersion": 1, "label": "BGRA->RGB", "message": "0.62 ms/frame", "color": "brightgreen"}
    assert badges["convert-gray.json"]["color"] == "yellow"
    assert badges["shot.json"]["color"] == "orange"
    assert badges["measured-on.json"]["message"] == "Intel(R) Graphics @ 1920x1080"


def test_a_missing_row_refuses_to_build(badge_env):
    baseline, _ = badge_env
    baseline.write_text(json.dumps(recording({"convert.RGB": 1.0})), encoding="utf-8")

    with pytest.raises(SystemExit, match="convert.GRAY missing"):
        make_badges.build()


def test_check_passes_only_when_the_committed_badges_match(badge_env, monkeypatch, capsys):
    baseline, badges = badge_env

    monkeypatch.setattr(sys, "argv", ["make_badges.py", "--check"])
    assert make_badges.main() == 1, "no badges written yet"
    assert "convert-rgb.json" in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", ["make_badges.py"])
    assert make_badges.main() == 0
    written = json.loads((badges / "shot.json").read_text(encoding="utf-8"))
    assert written["message"] == "17.00 ms/frame"

    monkeypatch.setattr(sys, "argv", ["make_badges.py", "--check"])
    assert make_badges.main() == 0

    baseline.write_text(json.dumps(recording(
        {"convert.RGB": 0.70, "convert.GRAY": 9.5, "shot.RGB": 17.0})), encoding="utf-8")
    assert make_badges.main() == 1, "a re-recorded baseline makes the badges stale"


def test_the_committed_badges_match_the_committed_baseline(monkeypatch, capsys):
    """What CI's `make_badges.py --check` step asserts, runnable locally."""
    monkeypatch.setattr(sys, "argv", ["make_badges.py", "--check"])
    assert make_badges.main() == 0, capsys.readouterr().out


# --------------------------------------------------------------------------
# compare_recordings
# --------------------------------------------------------------------------

def write(tmp_path, name, timestamp, rows, version="2.5.0"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({
        "machine": {"timestamp": timestamp, "rapidshot": version},
        "results": [{"name": n, "min_ms": ms} for n, ms in rows.items()],
    }), encoding="utf-8")
    return path


def run(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["compare_recordings.py", *map(str, argv)])
    code = compare_recordings.main()
    return code, capsys.readouterr().out


def gain_line(out, name):
    """The row in the drift-adjusted table, which follows the raw table."""
    gains = out[out.index("DRIFT-ADJUSTED GAINS"):]
    return next(line for line in gains.splitlines() if line.startswith(name))


def test_gains_are_adjusted_by_each_recordings_control(tmp_path, monkeypatch, capsys):
    """Recorded on a machine twice as slow, the same 2x speed-up must still read 2x."""
    old = write(tmp_path, "old", "2026-08-01", {"control.memcopy": 1.0, "convert.RGB": 4.0})
    new = write(tmp_path, "new", "2026-09-01", {"control.memcopy": 2.0, "convert.RGB": 4.0})

    code, out = run(monkeypatch, capsys, new, old, "--base", old)

    assert code == 0
    assert "control drift 2.00x  (machine slower then)" in out
    # raw 4.0 -> 4.0 is 1.00x; the new machine was 2x slower, so the code is 2x faster.
    assert gain_line(out, "convert.RGB").rstrip().endswith("2.00x")


def test_the_second_oldest_is_the_default_reference(tmp_path, monkeypatch, capsys):
    files = [
        write(tmp_path, "baseline-a", "2026-07-01", {"control.memcopy": 1.0, "convert.RGB": 8.0}),
        write(tmp_path, "baseline-b", "2026-07-15", {"control.memcopy": 1.0, "convert.RGB": 4.0}),
        write(tmp_path, "baseline-c", "2026-08-01", {"control.memcopy": 1.0, "convert.RGB": 2.0}),
    ]

    code, out = run(monkeypatch, capsys, *reversed(files))

    assert code == 0
    assert "DRIFT-ADJUSTED GAINS vs baseline-b" in out
    line = gain_line(out, "convert.RGB")
    assert "0.50x" in line and "2.00x" in line, "a is half as fast as b, c twice as fast"


def test_rows_missing_or_zero_are_dashes_not_ratios(tmp_path, monkeypatch, capsys):
    a = write(tmp_path, "a", "2026-07-01", {"convert.RGB": 2.0, "live.grab_frame_gpu": 0.4})
    b = write(tmp_path, "b", "2026-08-01", {"convert.RGB": 0.0, "live.grab_frame_gpu": 0.2})

    code, out = run(monkeypatch, capsys, a, b, "--base", a)

    assert code == 0
    assert "control drift 1.00x  (machine comparable)" in out, "no control: no adjustment"
    assert gain_line(out, "convert.RGB").split()[-1] == "-"
    assert "live: informational" in gain_line(out, "live.grab_frame_gpu")


@pytest.mark.parametrize("argv_builder,message", [
    (lambda p: [p("only", "2026-08-01")], "need at least two recordings"),
    (lambda p: [p("a", "2026-08-01"), "missing.json"], "missing:"),
    (lambda p: [p("a", "2026-08-01"), p("b", "2026-08-02"), "--base", "zzz.json"],
     "--base zzz is not among the files given"),
])
def test_bad_invocations_explain_themselves(tmp_path, monkeypatch, capsys, argv_builder, message):
    def make(name, stamp):
        return write(tmp_path, name, stamp, {"convert.RGB": 1.0})

    code, out = run(monkeypatch, capsys, *argv_builder(make))

    assert code == 1 and message in out


def test_with_no_arguments_it_reads_the_committed_baselines(monkeypatch, capsys):
    code, out = run(monkeypatch, capsys)
    assert code == 0 and "DRIFT-ADJUSTED GAINS" in out
