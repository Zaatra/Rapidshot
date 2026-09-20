"""Whether the snapshot describes the machine, or describes the probe.

The failure this file exists to prevent is subtle and was real: `machine_id`
was built partly from the CIM query, so a cheaper re-check that skipped CIM
computed a *different* id for the same box and the run reported that the
hardware had been swapped mid-measurement. Anything that can be unavailable
must stay out of an identifier, and anything that was not looked at must not
read as agreement.

The live probes are exercised for consistency rather than for values -- a CI
runner has no panel and no NVIDIA driver, and "unavailable, with a reason" is a
correct answer here, not a skip.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))

import machine_inventory as mi  # noqa: E402


# ---------------------------------------------------------------------------
# The three states
# ---------------------------------------------------------------------------

def test_a_failing_probe_records_why_rather_than_vanishing():
    def broken():
        raise RuntimeError("no such counter")

    result = mi._probe(broken)
    assert result["available"] is False
    assert "RuntimeError: no such counter" in result["reason"]


def test_a_probe_returning_nothing_is_not_silently_available():
    assert mi._probe(lambda: None)["available"] is False


def test_unavailable_is_never_an_empty_dict():
    # An empty dict reads as "nothing was wrong and nothing was found", which
    # is the one thing this must never be mistaken for.
    assert mi._unavailable("gone") == {"available": False, "reason": "gone"}


# ---------------------------------------------------------------------------
# machine_id
# ---------------------------------------------------------------------------

def snapshot(**overrides):
    base = {
        "cpu": {"processor": "Intel64 Family 6", "physical_cores": 24,
                "logical_processors": 32, "topology": "hybrid",
                "caches": {"L3_unified": {"instances": 1}},
                "affinity": {"process_mask": 0xFFFFFFFF}},
        "memory": {"total_physical_mb": 34049.4, "available_physical_mb": 14531.0},
        "os": {"machine": "AMD64"},
        "power": {"ac_line_status": "ac", "battery_percent": 53},
        "hardware": {"gpus": [{"name": "RTX 4060", "pnp_device_id": "PCI\\VEN_10DE"}]},
    }
    base.update(overrides)
    return base


def test_machine_id_ignores_sections_that_can_be_unavailable():
    """The bug: an id that moved when a cheaper snapshot skipped the CIM query."""
    with_cim = snapshot()
    without_cim = snapshot(hardware=mi._unavailable("not collected for this snapshot"))
    assert mi.machine_id(with_cim) == mi.machine_id(without_cim)


@pytest.mark.parametrize("field,value", [
    ("power", {"ac_line_status": "battery", "battery_percent": 4}),
    ("memory", {"total_physical_mb": 34049.4, "available_physical_mb": 900.0}),
])
def test_machine_id_ignores_what_changes_between_runs_on_one_machine(field, value):
    # Two runs on one machine must share an id, or per-machine history cannot
    # be kept apart the way ROADMAP section 3 requires.
    assert mi.machine_id(snapshot(**{field: value})) == mi.machine_id(snapshot())


def test_machine_id_ignores_the_affinity_the_run_chose():
    pinned = snapshot()
    pinned["cpu"] = dict(pinned["cpu"], affinity={"process_mask": 0xFFFF})
    assert mi.machine_id(pinned) == mi.machine_id(snapshot())


@pytest.mark.parametrize("cpu_change", [
    {"physical_cores": 8}, {"logical_processors": 16}, {"topology": "uniform"},
    {"processor": "AMD64 Family 25"},
])
def test_machine_id_separates_actually_different_machines(cpu_change):
    other = snapshot()
    other["cpu"] = dict(other["cpu"], **cpu_change)
    assert mi.machine_id(other) != mi.machine_id(snapshot())


def test_machine_id_carries_no_hostname_or_username():
    # It is published with results, so it must not smuggle anything personal.
    identifier = mi.machine_id(snapshot())
    assert identifier.startswith("m-") and len(identifier) == 18


# ---------------------------------------------------------------------------
# CPU policy
# ---------------------------------------------------------------------------

def uniform_cpu():
    return {"available": True, "topology": "uniform", "processor_groups": 1,
            "efficiency_classes": [{"efficiency_class": 0, "mask": 0xFF}]}


def hybrid_cpu():
    return {"available": True, "topology": "hybrid", "processor_groups": 1,
            "efficiency_classes": [{"efficiency_class": 0, "mask": 0xFFFF0000},
                                   {"efficiency_class": 1, "mask": 0x0000FFFF}]}


def test_the_fastest_class_is_the_highest_efficiency_class():
    mask, topology = mi.performance_core_mask(hybrid_cpu())
    assert (mask, topology) == (0x0000FFFF, "hybrid")


def test_a_uniform_cpu_needs_no_mask_and_says_so():
    assert mi.performance_core_mask(uniform_cpu()) == (None, "uniform")


def test_an_unreadable_topology_is_unknown_not_uniform():
    """Claiming uniformity nobody observed leaves a hybrid CPU silently unpinned."""
    assert mi.performance_core_mask(mi._unavailable("nope")) == (None, "unknown")


def test_no_pin_is_a_recorded_policy_not_an_absence_of_one():
    policy = mi.apply_cpu_policy("none", cpu=hybrid_cpu())
    assert policy.policy == "none" and policy.requested_mask is None
    assert policy.verified is False
    assert any("recorded as its own configuration" in reason
               for reason in policy.reasons)


def test_an_unknown_topology_is_reported_rather_than_pinned_blindly():
    policy = mi.apply_cpu_policy("performance", cpu=mi._unavailable("denied"))
    assert policy.topology == "unknown" and policy.requested_mask is None
    assert any("rather than claiming the machine was uniform" in reason
               for reason in policy.reasons)


def test_multiple_processor_groups_refuse_a_partial_pin():
    """One affinity mask covers group 0 only, so pinning here would be a half-truth."""
    cpu = dict(hybrid_cpu(), processor_groups=2,
               multi_group_warning="2 processor groups; a single affinity mask "
                                   "covers only group 0")
    policy = mi.apply_cpu_policy("performance", cpu=cpu)
    assert policy.requested_mask is None
    assert "processor groups" in policy.reasons[0]


def test_an_inherited_restriction_is_narrowed_never_widened(monkeypatch):
    # Widening would hand the process cores the caller deliberately withheld.
    monkeypatch.setattr(mi, "current_affinity",
                        lambda: {"available": True, "process_mask": 0x00FF,
                                 "system_mask": 0xFFFFFFFF,
                                 "process_mask_hex": "0xff",
                                 "restricted_before_policy": True})
    applied = {}
    monkeypatch.setattr(mi, "_effective_mask", lambda: applied.get("mask"))

    class FakeKernel:
        def __getattr__(self, name):
            return self

        def __call__(self, *args):
            if len(args) == 2:
                applied["mask"] = args[1]
                return 1
            return 1

        restype = None
        argtypes = None

    monkeypatch.setattr(mi.ctypes, "WinDLL", lambda *a, **kw: FakeKernel(), raising=False)
    policy = mi.apply_cpu_policy("performance", cpu=hybrid_cpu())
    assert policy.requested_mask == 0x00FF          # 0xFFFF & 0x00FF
    assert any("intersected rather than widened" in reason
               for reason in policy.reasons)


def test_an_inherited_restriction_with_no_fast_cores_leaves_affinity_alone(monkeypatch):
    monkeypatch.setattr(mi, "current_affinity",
                        lambda: {"available": True, "process_mask": 0xFFFF0000,
                                 "system_mask": 0xFFFFFFFF,
                                 "process_mask_hex": "0xffff0000",
                                 "restricted_before_policy": True})
    policy = mi.apply_cpu_policy("performance", cpu=hybrid_cpu())
    assert policy.applied is False
    assert any("excludes every performance core" in reason
               for reason in policy.reasons)


def test_a_policy_that_was_not_granted_is_not_recorded_as_in_force(monkeypatch):
    """Asking is not getting, so the mask is read back rather than trusted."""
    monkeypatch.setattr(mi, "current_affinity",
                        lambda: {"available": True, "process_mask": 0xFFFFFFFF,
                                 "system_mask": 0xFFFFFFFF,
                                 "process_mask_hex": "0xffffffff",
                                 "restricted_before_policy": False})

    class Refuses:
        def __getattr__(self, name):
            return self

        def __call__(self, *args):
            return 1 if len(args) != 2 else 1

        restype = None
        argtypes = None

    monkeypatch.setattr(mi.ctypes, "WinDLL", lambda *a, **kw: Refuses(), raising=False)
    policy = mi.apply_cpu_policy("performance", cpu=hybrid_cpu())
    assert policy.applied is True
    assert policy.verified is False                 # requested 0xFFFF, got 0xFFFFFFFF
    assert any("but the OS reports" in reason for reason in policy.reasons)


# ---------------------------------------------------------------------------
# CPU policy against the real OS
#
# The tests above stub kernel32, so they check the policy's arithmetic and
# bookkeeping, not that Windows grants and reports the mask the way the code
# assumes. These run the same paths through the real SetProcessAffinityMask
# and GetProcessAffinityMask, on a synthetic topology built from the cores
# this process is actually allowed, and always put the original mask back.
# ---------------------------------------------------------------------------

def _set_affinity(mask):
    import ctypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    if not k32.SetProcessAffinityMask(k32.GetCurrentProcess(), mask):
        raise ctypes.WinError(ctypes.get_last_error())


@pytest.fixture
def real_affinity():
    """The process's real mask, split into a "fast" lowest core and the rest."""
    if sys.platform != "win32":
        pytest.skip("process affinity is read through kernel32")
    before = mi.current_affinity()
    mask = before["process_mask"]
    fast = mask & -mask
    if fast == mask:
        pytest.skip("only one logical processor is available to this process")
    cpu = {"available": True, "topology": "hybrid", "processor_groups": 1,
           "efficiency_classes": [{"efficiency_class": 0, "mask": mask & ~fast},
                                  {"efficiency_class": 1, "mask": fast}]}
    try:
        yield SimpleNamespace(mask=mask, fast=fast, cpu=cpu)
    finally:
        _set_affinity(mask)
        assert mi.current_affinity()["process_mask"] == mask


def test_a_granted_policy_is_what_the_os_then_reports(real_affinity):
    policy = mi.apply_cpu_policy("performance", cpu=real_affinity.cpu)

    assert policy.applied is True and policy.verified is True, policy.reasons
    assert policy.requested_mask == real_affinity.fast
    assert mi.current_affinity()["process_mask"] == real_affinity.fast
    assert mi.verify_affinity(real_affinity.fast)["matches_expected"] is True


def test_a_real_inherited_restriction_is_narrowed_never_widened(real_affinity):
    # Restrict for real first, then ask for more than that: the OS must end up
    # at the intersection, and the recording must say so.
    fast = real_affinity.fast
    rest = real_affinity.mask & ~fast
    second = rest & -rest
    _set_affinity(fast | second)
    wide = dict(real_affinity.cpu, efficiency_classes=[
        {"efficiency_class": 0, "mask": 0},
        {"efficiency_class": 1, "mask": real_affinity.mask}])

    policy = mi.apply_cpu_policy("performance", cpu=wide)

    assert policy.requested_mask == fast | second
    assert policy.verified is True, policy.reasons
    assert mi.current_affinity()["process_mask"] == fast | second
    if real_affinity.mask != fast | second:
        assert any("intersected rather than widened" in reason
                   for reason in policy.reasons)


# ---------------------------------------------------------------------------
# Environment drift
# ---------------------------------------------------------------------------

def full(**overrides):
    base = {"machine_id": "m-aaaa", "display_fingerprint": "dddd",
            "collected": ["cpu", "memory", "displays", "power", "os",
                          "hardware", "source", "runtime"],
            "cpu": {"topology": "hybrid", "logical_processors": 32},
            "memory": {"available_physical_mb": 14000.0, "memory_load_percent": 57},
            "power": {"ac_line_status": "ac", "battery_percent": 53},
            "source": {"commit": "abc123", "package_fingerprint": {"sha256": "ffff"}}}
    base.update(overrides)
    return base


def test_an_unchanged_machine_is_comparable():
    check = mi.verify_environment(full(), full())
    assert check.ok and check.comparable and not check.not_checked


def test_a_changed_display_configuration_blocks_a_verdict():
    check = mi.verify_environment(full(), full(display_fingerprint="eeee"))
    assert check.comparable is False
    assert check.blocking[0]["field"] == "display_fingerprint"


def test_coming_off_mains_blocks_a_verdict():
    """A laptop on battery throttles; the halves of the run measure different machines."""
    later = full()
    later["power"] = {"ac_line_status": "battery", "battery_percent": 40}
    check = mi.verify_environment(full(), later)
    assert check.comparable is False
    assert [entry["field"] for entry in check.blocking] == ["power.ac_line_status"]


def test_drifting_battery_charge_is_recorded_but_does_not_block():
    later = full()
    later["power"] = dict(later["power"], battery_percent=41)
    check = mi.verify_environment(full(), later)
    assert check.comparable is True
    assert [entry["field"] for entry in check.changed] == ["power.battery_percent"]


def test_a_section_that_was_not_gathered_is_not_treated_as_agreement():
    """The whole point: "not looked at" must not read as "did not change"."""
    cheap = full(collected=["cpu", "memory", "displays", "power", "os"])
    cheap["source"] = mi._unavailable("not collected for this snapshot")
    check = mi.verify_environment(full(), cheap)
    assert check.comparable is True
    assert set(check.not_checked) == {"source.commit", "source.package_fingerprint"}


def test_a_rebuilt_package_blocks_a_verdict_when_it_is_actually_checked():
    later = full()
    later["source"] = dict(later["source"], package_fingerprint={"sha256": "0000"})
    check = mi.verify_environment(full(), later)
    assert check.comparable is False
    assert check.blocking[0]["field"] == "source.package_fingerprint"


# ---------------------------------------------------------------------------
# Affinity verification
# ---------------------------------------------------------------------------

def test_a_worker_that_cannot_read_its_affinity_says_so(monkeypatch):
    monkeypatch.setattr(mi, "_probe", lambda *a, **kw: mi._unavailable("denied"))
    assert mi.verify_affinity(0xFFFF)["available"] is False


def test_a_worker_reports_whether_it_got_what_the_parent_asked_for(monkeypatch):
    monkeypatch.setattr(mi, "_probe", lambda *a, **kw: {
        "available": True, "process_mask": 0xFFFF, "process_mask_hex": "0xffff"})
    assert mi.verify_affinity(0xFFFF)["matches_expected"] is True
    assert mi.verify_affinity(0x00FF)["matches_expected"] is False
    # No expectation means nothing to contradict, not a silent pass/fail.
    assert mi.verify_affinity(None)["matches_expected"] is True


# ---------------------------------------------------------------------------
# Display fingerprint
# ---------------------------------------------------------------------------

def displays(**overrides):
    output = {"monitor_device_path": "\\\\?\\DISPLAY#AUOCDAB", "adapter_luid": 78122,
              "source_mode": {"width": 2560, "height": 1600},
              "rotation_degrees": 0,
              "refresh_rate": {"numerator": 77733000, "denominator": 471104},
              "scale_percent": 150, "primary": True}
    output.update(overrides)
    return {"available": True, "outputs": [output]}


def test_the_display_fingerprint_is_stable_for_an_unchanged_setup():
    assert mi.display_fingerprint(displays()) == mi.display_fingerprint(displays())


@pytest.mark.parametrize("change", [
    {"source_mode": {"width": 1920, "height": 1080}},
    {"rotation_degrees": 90},
    {"refresh_rate": {"numerator": 60, "denominator": 1}},
    {"scale_percent": 100},
    {"adapter_luid": 999},
    {"monitor_device_path": "\\\\?\\DISPLAY#OTHER"},
])
def test_the_display_fingerprint_moves_when_the_run_conditions_move(change):
    assert mi.display_fingerprint(displays(**change)) != mi.display_fingerprint(displays())


def test_the_display_fingerprint_does_not_depend_on_enumeration_order():
    one = displays()
    second = dict(one["outputs"][0], monitor_device_path="B", primary=False)
    forward = {"available": True, "outputs": [one["outputs"][0], second]}
    reverse = {"available": True, "outputs": [second, one["outputs"][0]]}
    assert mi.display_fingerprint(forward) == mi.display_fingerprint(reverse)


def test_unavailable_displays_fingerprint_as_unavailable():
    assert mi.display_fingerprint(mi._unavailable("headless")) == "unavailable"


# ---------------------------------------------------------------------------
# Live probes: shape and internal consistency only
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not mi.IS_WINDOWS, reason="Windows-only probes")
def test_the_live_cpu_probe_is_internally_consistent():
    cpu = mi._probe(mi.discover_cpu)
    if not cpu.get("available"):
        pytest.skip(f"CPU topology unreadable here: {cpu['reason']}")
    assert cpu["topology"] in ("hybrid", "uniform", "unknown")
    # SMT means more logical processors than physical cores, never fewer.
    assert cpu["logical_processors"] >= cpu["physical_cores"] > 0
    assert cpu["logical_processors"] == sum(
        entry["logical_processors"] for entry in cpu["efficiency_classes"])
    assert cpu["processor_groups"] >= 1
    assert cpu["affinity"]["process_mask"] > 0


@pytest.mark.skipif(not mi.IS_WINDOWS, reason="Windows-only probes")
def test_a_full_snapshot_names_every_section_it_gathered():
    snap = mi.discover_machine(include_cim=False)
    assert snap["schema_version"] == mi.SCHEMA_VERSION
    assert "hardware" not in snap["collected"]
    assert snap["hardware"]["available"] is False
    for section in ("cpu", "memory", "displays", "power", "os"):
        assert section in snap["collected"] and section in snap
    assert snap["machine_id"].startswith("m-")


@pytest.mark.skipif(not mi.IS_WINDOWS, reason="Windows-only probes")
def test_the_snapshot_records_source_identity_with_or_without_git():
    source = mi._probe(mi.discover_source)
    # A fingerprint is computed from files on disk, so it exists even where git
    # does not -- which is the copied-folder case Phase 3 depends on.
    assert source["package_fingerprint"]["files"] > 0
    assert len(source["package_fingerprint"]["sha256"]) == 64
    assert "commit" in source or not source.get("git", {}).get("available", True)


@pytest.mark.skipif(not mi.IS_WINDOWS, reason="Windows-only probes")
def test_prepare_run_applies_the_policy_before_taking_the_snapshot(monkeypatch):
    order = []
    monkeypatch.setattr(mi, "apply_cpu_policy",
                        lambda policy: order.append("policy") or mi.CpuPolicy(
                            policy=policy, topology="hybrid"))
    monkeypatch.setattr(mi, "discover_machine",
                        lambda **kw: order.append("snapshot") or {
                            "machine_id": "m-x", "display_fingerprint": "d"})
    mi.prepare_run(SimpleNamespace(no_pin=False), announce=False)
    # The snapshot has to record the affinity the run actually had, which means
    # it cannot be taken first.
    assert order == ["policy", "snapshot"]
