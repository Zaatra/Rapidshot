"""Releasing once, from one thread at a time.

Two faults, both about handing something back more times than it was taken.

`Device.release()`, `StageSurface.release()` and `Duplicator.release()` each
called `.Release()` on a comtypes COM pointer and then dropped it. comtypes
issues `Release` itself when the pointer goes away, so that decremented the
refcount twice for one reference -- measured on real DXGI objects, an explicit
call plus the drop took the count down by two where the drop alone took it down
by one. Over-releasing frees the object while other holders still have valid
pointers. This file's fakes count the calls that must no longer happen; the
refcount itself is checked against real DXGI at the bottom, where there is one.

And `Frame.release()` and the factory's lazy setup both tested a flag and then
acted on it, with nothing in between to stop a second thread doing the same.
"""

import gc
import threading
import time

import pytest

pytest.importorskip("comtypes")

from rapidshot.core.device import Device  # noqa: E402
from rapidshot.core.duplicator import Duplicator  # noqa: E402
from rapidshot.core.stagesurf import StageSurface  # noqa: E402
from rapidshot.frame import Frame  # noqa: E402


class CountingPointer:
    """Stands in for a comtypes COM pointer and counts the calls it must not get."""

    def __init__(self):
        self.release_calls = 0

    def Release(self):
        self.release_calls += 1

    def __bool__(self):
        # A real comtypes pointer is truthy; release() used to test it.
        return True


# --------------------------------------------------------------------------
# Nothing calls Release() by hand any more
# --------------------------------------------------------------------------

def test_device_release_drops_its_pointers_without_releasing_them():
    device = Device.__new__(Device)
    pointers = {name: CountingPointer()
                for name in ("im_context", "context", "device", "adapter")}
    for name, pointer in pointers.items():
        setattr(device, name, pointer)

    device.release()

    for name, pointer in pointers.items():
        assert getattr(device, name) is None, f"{name} was not dropped"
        assert pointer.release_calls == 0, (
            f"{name}.Release() on top of dropping the pointer over-releases it")


def test_stage_surface_release_drops_its_texture_without_releasing_it():
    surface = StageSurface.__new__(StageSurface)
    texture = CountingPointer()
    surface.texture = texture
    surface.interface = object()
    surface.width, surface.height = 1920, 1080

    surface.release()

    assert surface.texture is None
    assert surface.interface is None
    assert (surface.width, surface.height) == (0, 0)
    assert texture.release_calls == 0


def test_duplicator_release_drops_the_interface_without_releasing_it():
    dup = Duplicator.__new__(Duplicator)
    interface = CountingPointer()
    dup.duplicator = interface
    dup._frame_acquired = False
    dup.texture = None

    dup.release()

    assert dup.duplicator is None
    assert dup._frame_acquired is False
    assert interface.release_calls == 0


def test_releasing_a_duplicator_twice_is_still_safe():
    dup = Duplicator.__new__(Duplicator)
    dup.duplicator = CountingPointer()
    dup._frame_acquired = False
    dup.texture = None

    dup.release()
    dup.release()   # must not raise

    assert dup.duplicator is None


# --------------------------------------------------------------------------
# Frame.release() under concurrency
# --------------------------------------------------------------------------

def _frame(on_release=None):
    return Frame(
        texture=object(),
        on_release=on_release,
        region=(0, 0, 16, 16),
    )


def test_concurrent_release_hands_the_surface_back_once():
    """on_release is ReleaseFrame. Calling it twice for one acquire is the bug."""
    calls = []
    frame = _frame(on_release=lambda: calls.append(1))

    # A slow drain holds the window open, so every thread would have been
    # inside release() at once before the lock.
    frame.defer_release_until(lambda: time.sleep(0.05))

    threads = [threading.Thread(target=frame.release) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(calls) == 1, f"the DXGI frame was handed back {len(calls)} times"


def test_concurrent_release_runs_each_drain_once():
    drained = []
    frame = _frame()
    frame.defer_release_until(lambda: (time.sleep(0.05), drained.append(1))[0])

    threads = [threading.Thread(target=frame.release) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(drained) == 1, (
        f"the drain ran {len(drained)} times; a GPU wait is not free to repeat")


def test_release_is_still_idempotent_on_one_thread():
    calls = []
    frame = _frame(on_release=lambda: calls.append(1))

    frame.release()
    frame.release()

    assert calls == [1]
    assert frame.released is True


def test_release_still_runs_drains_before_the_handback():
    order = []
    frame = _frame(on_release=lambda: order.append("handback"))
    frame.defer_release_until(lambda: order.append("drain"))

    frame.release()

    assert order == ["drain", "handback"]


def test_a_frame_can_be_used_as_a_context_manager_still():
    calls = []
    with _frame(on_release=lambda: calls.append(1)) as frame:
        assert frame.released is False
    assert calls == [1]


# --------------------------------------------------------------------------
# The factory is built once, however many threads ask at once
# --------------------------------------------------------------------------

def test_the_singleton_metaclass_builds_one_instance_under_contention():
    """Building the real factory opens a D3D11 device per adapter, so a second
    one is not just wasted work -- its devices stay open."""
    from rapidshot import Singleton

    built = []

    class Slow(metaclass=Singleton):
        def __init__(self):
            time.sleep(0.05)     # widen the window between check and act
            built.append(1)

    results = []
    threads = [threading.Thread(target=lambda: results.append(Slow()))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    try:
        assert len(built) == 1, f"the factory was constructed {len(built)} times"
        assert len({id(r) for r in results}) == 1, "threads got different instances"
        assert len(results) == 8
    finally:
        Singleton._instances.pop(Slow, None)


def test_get_factory_returns_one_instance_under_contention(monkeypatch):
    import rapidshot

    built = []

    class FakeFactory:
        def __init__(self):
            time.sleep(0.05)
            built.append(1)

    monkeypatch.setattr(rapidshot, "RapidshotFactory", FakeFactory)
    # The module global is spelled "__factory": no name mangling applies at
    # module level. monkeypatch restores whatever was there, so a real factory
    # built by an earlier test survives this one.
    assert "__factory" in vars(rapidshot), "the factory global was renamed"
    monkeypatch.setattr(rapidshot, "__factory", None)

    results = []
    threads = [threading.Thread(target=lambda: results.append(rapidshot.get_factory()))
               for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert len(built) == 1, f"the factory was built {len(built)} times"
    assert len({id(r) for r in results}) == 1
    assert all(isinstance(r, FakeFactory) for r in results)


# --------------------------------------------------------------------------
# The refcount itself, against real DXGI
# --------------------------------------------------------------------------

def _real_adapter():
    try:
        from rapidshot.util.io import enum_dxgi_adapters
        adapters = enum_dxgi_adapters()
    except Exception:
        return None
    return adapters[0] if adapters else None


@pytest.mark.skipif(_real_adapter() is None, reason="no DXGI adapter available")
def test_dropping_a_com_pointer_releases_it_exactly_once():
    """The measurement the fix rests on, run against a real D3D11 device.

    Headroom is AddRef'd first so no count can reach zero and leave the probe
    pointing at freed memory.
    """
    import comtypes

    headroom = 20
    device = Device(_real_adapter())
    probe = device.device.QueryInterface(comtypes.IUnknown)
    for _ in range(headroom):
        probe.AddRef()

    def count():
        n = probe.AddRef()
        probe.Release()
        return n - 1

    try:
        pointer = device.device
        before = count()
        device.device = None
        pointer = None
        del pointer
        gc.collect()
        after = count()

        assert before - after == 1, (
            f"dropping one reference moved the refcount by {before - after}; "
            "1 is correct, 2 means something released it a second time")
    finally:
        for _ in range(headroom):
            probe.Release()
        try:
            device.release()
        except Exception:
            pass
