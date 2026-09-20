"""Whether the source was animating the thing that was being captured.

This had happened and nothing noticed. `motion_source.py` animated a hardcoded
900x700 window while `memory_profile.py` captured the whole screen, so on a
2560x1600 panel **15.4% of the captured area was moving**. Desktop Duplication
reports only what changed, so every library did a fraction of the work a real
workload asks of it -- and the size of the discount depended on the monitor,
which means two machines running the identical command were not running the
same benchmark.

The fix is that the source covers the screen. The guard is that the fraction is
computed and recorded, so a future change that shrinks it again shows up as a
contaminated case rather than as a pleasing number.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import motion_source  # noqa: E402
import result_validation as rv  # noqa: E402

SCREEN = (0, 0, 2560, 1600)


# ---------------------------------------------------------------------------
# The arithmetic
# ---------------------------------------------------------------------------

def test_a_full_screen_source_covers_everything():
    assert rv.animated_fraction(SCREEN, SCREEN) == 1.0
    assert rv.coverage_reasons(SCREEN, SCREEN) == []


def test_the_old_window_reproduces_the_bug_it_was():
    """900x700 at +200+120 against a 2560x1600 capture."""
    old = (200, 120, 1100, 820)
    assert rv.animated_fraction(old, SCREEN) == pytest.approx(0.154, abs=0.001)
    reasons = rv.coverage_reasons(old, SCREEN)
    assert reasons and "15.4% of the captured area" in reasons[0]
    assert "mostly-still screen" in reasons[0]


def test_the_discount_depended_on_the_monitor():
    """The same command measured different things on different displays, which
    is what makes a cross-machine comparison meaningless."""
    old = (200, 120, 1100, 820)
    # The same 900x700 window was 30% of a 1080p capture and 15% of a
    # 2560x1600 one, so the two machines' numbers were never comparable.
    assert rv.animated_fraction(old, (0, 0, 1920, 1080)) == pytest.approx(0.304,
                                                                         abs=0.001)
    assert rv.animated_fraction(old, SCREEN) == pytest.approx(0.154, abs=0.001)


def test_a_source_larger_than_the_capture_still_counts_as_full():
    # What matters is whether the measured area was moving, not whether the
    # source wasted effort outside it.
    assert rv.animated_fraction((0, 0, 3840, 2160), SCREEN) == 1.0


def test_a_region_inside_a_full_screen_source_is_fully_covered():
    assert rv.coverage_reasons(SCREEN, (1080, 600, 1480, 1000)) == []


def test_a_region_outside_the_animated_area_is_flagged():
    assert rv.coverage_reasons((0, 0, 400, 400), (1080, 600, 1480, 1000))


def test_unrecorded_geometry_is_not_treated_as_fine():
    """Not knowing and being fine are different, and only one is reportable."""
    assert "cannot be shown" in rv.coverage_reasons(None, SCREEN)[0]
    assert "cannot be shown" in rv.coverage_reasons(SCREEN, None)[0]


def test_a_degenerate_capture_is_zero_not_a_crash():
    assert rv.animated_fraction(SCREEN, (10, 10, 10, 10)) == 0.0


# ---------------------------------------------------------------------------
# The source
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("900x700+200+120", (900, 700, 200, 120)),
    ("640x480-100+50", (640, 480, -100, 50)),
    ("640x480+-100+50", (640, 480, -100, 50)),
    ("2560x1600+0+0", (2560, 1600, 0, 0)),
])
def test_a_window_specification_parses(text, expected):
    assert motion_source.parse_window(text) == expected


@pytest.mark.parametrize("bad", ["900x700", "900x700+1", "nonsense", "900X700+0+0",
                                 "", "x+0+0"])
def test_a_half_specified_window_is_refused(bad):
    """Three of four numbers is a mistake nobody wants to debug from a screenshot."""
    with pytest.raises(ValueError):
        motion_source.parse_window(bad)


def fake_tk(monkeypatch, state):
    class Root:
        def title(self, value): pass
        def overrideredirect(self, value): pass
        def winfo_screenwidth(self): return 2560
        def winfo_screenheight(self): return 1600
        def geometry(self, value): state.geometry = value
        def attributes(self, *args): pass
        def protocol(self, *args): pass
        def update_idletasks(self): pass
        def update(self): state.frames += 1
        def destroy(self): pass

    class Canvas:
        def __init__(self, *args, **kwargs):
            state.canvas = (kwargs.get("width"), kwargs.get("height"))

        def pack(self): pass

        def create_rectangle(self, *args, **kwargs):
            state.bars += 1
            return state.bars

        def coords(self, *args): pass

        def itemconfig(self, *args, **kwargs): pass

    monkeypatch.setitem(sys.modules, "tkinter",
                        SimpleNamespace(Tk=Root, Canvas=Canvas))
    monkeypatch.setattr(motion_source, "emit",
                        lambda event, **fields: state.emitted.append((event, fields)))


def run_source(monkeypatch, window=None):
    state = SimpleNamespace(geometry=None, canvas=None, bars=0, frames=0, emitted=[])
    fake_tk(monkeypatch, state)
    import threading
    stopped = threading.Event()

    original = motion_source.time.perf_counter
    ticks = iter([0.0] + [0.001 * index for index in range(1, 400)])
    monkeypatch.setattr(motion_source.time, "perf_counter",
                        lambda: next(ticks, 999.0))
    motion_source.animate(0.05, 0.0, False, stopped, window)
    monkeypatch.setattr(motion_source.time, "perf_counter", original)
    return state


def test_the_source_covers_the_whole_screen_by_default(monkeypatch):
    """The correction. A capture benchmark driven by a corner of the display is
    measuring a still desktop, whatever the results file says."""
    state = run_source(monkeypatch)
    assert state.geometry == "2560x1600+0+0"
    assert state.canvas == (2560, 1600)


def test_a_window_is_still_available_for_tests(monkeypatch):
    state = run_source(monkeypatch, window=(900, 700, 200, 120))
    assert state.geometry == "900x700+200+120"
    assert state.canvas == (900, 700)


def test_the_ready_event_reports_what_is_being_animated(monkeypatch):
    """A consumer must be able to record the animated area rather than assume it."""
    state = run_source(monkeypatch)
    ready = [fields for event, fields in state.emitted if event == "ready"]
    assert ready and ready[0]["rect"] == [0, 0, 2560, 1600]
    assert ready[0]["screen"] == [2560, 1600]
    assert ready[0]["fullscreen"] is True


def test_a_windowed_source_says_it_is_not_full_screen(monkeypatch):
    state = run_source(monkeypatch, window=(900, 700, 200, 120))
    ready = [fields for event, fields in state.emitted if event == "ready"]
    assert ready[0]["rect"] == [200, 120, 1100, 820]
    assert ready[0]["fullscreen"] is False


def test_the_bar_count_scales_with_the_canvas(monkeypatch):
    """28 bars fixed would be slivers on a 2560px display."""
    wide = run_source(monkeypatch)
    narrow = run_source(monkeypatch, window=(900, 700, 0, 0))
    assert wide.bars > narrow.bars >= 10


# ---------------------------------------------------------------------------
# The harnesses
# ---------------------------------------------------------------------------

def test_the_capture_region_is_centred_on_the_actual_display(monkeypatch):
    """It was `(760, 340, 1160, 740)`, commented "centred on a 1080p display" --
    which it was, and on nothing else."""
    import compare_libraries

    class User32:
        @staticmethod
        def GetSystemMetrics(index):
            return 2560 if index == 0 else 1600

    monkeypatch.setattr(compare_libraries.ctypes, "windll",
                        SimpleNamespace(user32=User32), raising=False)
    left, top, right, bottom = compare_libraries.capture_region(400)
    assert (right - left, bottom - top) == (400, 400)
    assert (left + right) // 2 == 1280 and (top + bottom) // 2 == 800


def test_the_region_shrinks_to_fit_a_small_display(monkeypatch):
    import compare_libraries

    class Tiny:
        @staticmethod
        def GetSystemMetrics(index):
            return 320 if index == 0 else 240

    monkeypatch.setattr(compare_libraries.ctypes, "windll",
                        SimpleNamespace(user32=Tiny), raising=False)
    left, top, right, bottom = compare_libraries.capture_region(400)
    assert right - left == 240 and bottom - top == 240
    assert left >= 0 and top >= 0


def test_memory_profile_follows_the_display_rather_than_a_window():
    """It defaulted to 900x700 while capturing the whole screen, and the memory
    table in ROADMAP section 7.0 was recorded that way.

    Asserts the parsed defaults rather than grepping the source: a rename or a
    reflow of the argparse call is not a behaviour change, and a test that
    breaks on one is a test nobody trusts.
    """
    import memory_profile

    source = Path(memory_profile.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    defaults = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        name = node.args[0].value
        if name in ("--width", "--height"):
            for keyword in node.keywords:
                if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                    defaults[name] = keyword.value.value
    assert defaults == {"--width": 0, "--height": 0}, (
        "0 is the sentinel meaning 'follow the display'; a literal size here is "
        "the bug that measured a mostly-still screen")
