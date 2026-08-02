"""Tests for GPU/display topology detection (ROADMAP.md § 6.1, § 6.2).

The classification is pure data, so the interesting cases — headless VMs and
hybrid Optimus laptops — are tested by describing those machines rather than by
owning them. The one live test asserts the real machine agrees with DXGI.
"""

import pytest

from rapidshot.util.topology import (
    HEADLESS,
    HYBRID,
    MULTI_ADAPTER,
    SINGLE,
    AdapterInfo,
    GpuTopology,
    classify,
)


def igpu(outputs=1, index=0):
    return AdapterInfo(
        index=index,
        description="Intel(R) UHD Graphics",
        vendor_id=0x8086,
        dedicated_video_memory=128 * 1048576,
        output_count=outputs,
    )


def dgpu(outputs=0, index=1):
    return AdapterInfo(
        index=index,
        description="NVIDIA GeForce RTX 4070 Laptop GPU",
        vendor_id=0x10DE,
        dedicated_video_memory=8192 * 1048576,
        output_count=outputs,
    )


def warp(index=1):
    return AdapterInfo(
        index=index,
        description="Microsoft Basic Render Driver",
        vendor_id=0x1414,
        is_software=True,
        output_count=0,
    )


class TestClassification:
    def test_single_gpu_with_a_monitor(self):
        assert classify([igpu()]).kind == SINGLE

    def test_warp_alongside_one_gpu_is_still_single(self):
        # The Microsoft Basic Render Driver has no outputs, exactly like the
        # dGPU on an Optimus laptop. Reporting it as a hybrid system would make
        # every ordinary desktop look like one.
        topology = classify([igpu(), warp()])
        assert topology.kind == SINGLE
        assert not topology.is_hybrid
        assert topology.software_adapters == (warp(),)
        assert topology.render_only_adapters == ()

    def test_optimus_laptop_is_hybrid(self):
        topology = classify([igpu(), dgpu()])
        assert topology.kind == HYBRID
        assert topology.is_hybrid
        assert [a.description for a in topology.capture_adapters] == [
            "Intel(R) UHD Graphics"
        ]
        assert [a.description for a in topology.render_only_adapters] == [
            "NVIDIA GeForce RTX 4070 Laptop GPU"
        ]

    def test_dgpu_driving_a_monitor_is_not_hybrid(self):
        # Muxed laptop set to discrete-only, or a desktop with two cards: both
        # adapters drive displays, so capture can run on either.
        topology = classify([igpu(), dgpu(outputs=1)])
        assert topology.kind == MULTI_ADAPTER
        assert len(topology.capture_adapters) == 2

    def test_no_outputs_anywhere_is_headless(self):
        topology = classify([igpu(outputs=0), warp()])
        assert topology.kind == HEADLESS
        assert topology.is_headless

    def test_no_adapters_at_all_is_headless(self):
        assert GpuTopology().kind == HEADLESS

    def test_adapter_that_failed_to_open_cannot_capture(self):
        broken = AdapterInfo(
            index=0, description="Intel(R) UHD Graphics", output_count=1,
            error="D3D11CreateDevice failed",
        )
        topology = classify([broken])
        assert topology.capture_adapters == ()
        assert topology.kind == HEADLESS


class TestMessages:
    def test_headless_help_is_actionable(self):
        help_text = classify([igpu(outputs=0)]).help_text()
        assert "No display available to duplicate" in help_text
        assert "Indirect Display Driver" in help_text or "IDD" in help_text
        # The refresh-rate caveat: a virtual display at 500 Hz does not make
        # capture run at 500 fps, and people assume it does.
        assert "refresh" in help_text.lower()
        assert "presents" in help_text.lower()
        # It should say what it did find, so "no display" is not confused with
        # "no GPU".
        assert "Intel(R) UHD Graphics" in help_text

    def test_no_adapters_help_does_not_blame_the_display(self):
        help_text = GpuTopology().help_text()
        assert "No DXGI adapters found" in help_text
        assert "session 0" in help_text

    def test_capturable_topology_has_no_help_text(self):
        assert classify([igpu()]).help_text() == ""

    def test_hybrid_description_names_the_consequence(self):
        described = classify([igpu(), dgpu()]).describe()
        assert "Hybrid GPU system detected" in described
        assert "DXGI_ERROR_UNSUPPORTED" in described
        # Which adapter capture landed on is the part a caller needs in order
        # to know whether their inference device matches.
        assert "Capture runs on Intel(R) UHD Graphics" in described
        assert "NVIDIA GeForce RTX 4070 Laptop GPU" in described

    def test_messages_are_ascii(self):
        # These are printed to consoles that still default to cp1252, where a
        # non-ASCII dash arrives as a replacement character.
        for topology in (
            classify([igpu(outputs=0)]),
            classify([igpu(), dgpu()]),
            GpuTopology(),
        ):
            (topology.help_text() + topology.describe()).encode("ascii")

    def test_single_description_stays_quiet(self):
        described = classify([igpu()]).describe()
        assert described.startswith("Topology: single")
        assert "Hybrid" not in described


class TestFactoryReporting:
    """The factory must translate a headless probe into an actionable error."""

    def _factory_with(self, monkeypatch, topology, adapters):
        rapidshot = pytest.importorskip("rapidshot")
        monkeypatch.setattr(rapidshot, "probe_topology", lambda: topology)
        monkeypatch.setattr(rapidshot, "enum_dxgi_adapters", lambda: adapters)
        # Bypass the Singleton metaclass so this never touches the real
        # process-wide factory instance.
        factory = object.__new__(rapidshot.RapidshotFactory)
        factory.__init__()
        return factory

    def test_headless_machine_raises_actionable_error(self, monkeypatch):
        rapidshot = pytest.importorskip("rapidshot")
        topology = classify([igpu(outputs=0)])

        with pytest.raises(rapidshot.HeadlessError) as excinfo:
            self._factory_with(monkeypatch, topology, adapters=[object()])

        message = str(excinfo.value)
        assert "No display available to duplicate" in message
        assert "IDD" in message
        assert excinfo.value.topology is topology
        # It must stay catchable as the DeviceError callers already handle.
        assert isinstance(excinfo.value, rapidshot.DeviceError)

    def test_no_adapters_raises_before_touching_devices(self, monkeypatch):
        rapidshot = pytest.importorskip("rapidshot")

        with pytest.raises(rapidshot.HeadlessError) as excinfo:
            self._factory_with(monkeypatch, GpuTopology(), adapters=[])

        assert "No DXGI adapters found" in str(excinfo.value)

    def test_device_creation_failures_are_reported_not_swallowed(self, monkeypatch):
        rapidshot = pytest.importorskip("rapidshot")
        # Displays exist, so this is not a headless machine — every device
        # simply refused to open. The message must say so.
        topology = classify([igpu(outputs=1)])

        with pytest.raises(rapidshot.HeadlessError) as excinfo:
            self._factory_with(monkeypatch, topology, adapters=[object()])

        message = str(excinfo.value)
        assert "No display available" not in message
        assert "Device creation errors:" in message


def test_live_probe_matches_dxgi():
    """On real hardware, the probe agrees with the raw enumeration."""
    pytest.importorskip("comtypes", reason="COM is Windows-only")
    from rapidshot.util.io import enum_dxgi_adapters
    from rapidshot.util.topology import probe_topology

    try:
        expected = len(enum_dxgi_adapters())
    except Exception as exc:  # CI runners without a graphics stack
        pytest.skip(f"DXGI unavailable: {exc}")

    topology = probe_topology()
    assert len(topology.adapters) == expected
    assert topology.kind in {HEADLESS, SINGLE, HYBRID, MULTI_ADAPTER}

    # The three roles partition the adapters: an adapter drives a display, or
    # is a render-only GPU, or is software. Overlap would mean a WARP adapter
    # could be counted as a dGPU.
    roles = (
        topology.capture_adapters
        + topology.render_only_adapters
        + topology.software_adapters
    )
    indices = [a.index for a in roles]
    assert len(indices) == len(set(indices)), "an adapter was counted twice"
    unclassified = {a.index for a in topology.adapters} - set(indices)
    assert all(topology.adapters[i].error for i in unclassified)
