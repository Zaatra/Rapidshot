"""Tests for GPU/display topology detection (ROADMAP.md § 6.1, § 6.2).

The classification is pure data, so the interesting cases — headless VMs and
hybrid Optimus laptops — are tested by describing those machines rather than by
owning them. The one live test asserts the real machine agrees with DXGI.
"""

import sys

import pytest

from rapidshot.util.topology import (
    HEADLESS,
    HYBRID,
    MULTI_ADAPTER,
    SINGLE,
    AdapterInfo,
    GpuTopology,
    RenderOffload,
    classify,
    find_render_offloads,
    pci_ids,
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
        # Which adapter owns the display is the part a caller needs in order
        # to know whether their inference device matches.
        assert "Intel(R) UHD Graphics owns the display" in described
        assert "NVIDIA GeForce RTX 4070 Laptop GPU" in described

    def test_hybrid_description_does_not_rule_out_render_only_adapters(self):
        """The note describes topology; the candidate search decides capability.

        It used to say a render-only adapter "cannot run [Desktop Duplication]
        against it at all", which is a verdict rather than a description --
        and it contradicted `_build_duplicator`, which tries every adapter and
        uses whichever is granted duplication. A user reading the old wording
        could conclude they needed a cross-adapter copy that capture had in
        fact just avoided.
        """
        described = classify([igpu(), dgpu()]).describe()
        assert "cannot run against it at all" not in described
        assert "every adapter is tried" in described

    def test_hybrid_description_does_not_promise_capture_works(self):
        """Owning the display does not mean duplication will be granted.

        This used to read "Capture runs on <adapter>" -- a prediction stated as
        fact. Measured 2026-08-21 on a hybrid machine whose display-owning
        adapter refused DuplicateOutput with DXGI_ERROR_UNSUPPORTED, as did
        every other adapter: the sentence was false, and it was the only thing
        the caller was told.
        """
        described = classify([igpu(), dgpu()]).describe()
        assert "Capture runs on" not in described
        assert "only known once it has been tried" in described

    def test_duplication_failure_help_names_hybrid_causes(self):
        """Every adapter refusing is a different failure from having no display.

        help_text() answers "there is nothing to capture"; this answers "there
        is a display and nothing would duplicate it", which has different
        causes and a different fix. Previously it surfaced as a raw COM string
        naming neither.
        """
        help_text = classify([igpu(), dgpu()]).duplication_failure_help()
        assert "hybrid" in help_text.lower()
        assert "GPU mode" in help_text
        assert "driver is current" in help_text
        # It must show what was actually found, not just advise in the abstract.
        assert "Intel(R) UHD Graphics" in help_text
        assert "NVIDIA GeForce RTX 4070 Laptop GPU" in help_text

    def test_duplication_failure_help_is_generic_off_hybrid(self):
        help_text = classify([igpu()]).duplication_failure_help()
        assert "No adapter could duplicate the display" in help_text
        assert "MUX" not in help_text  # hybrid-only advice must not leak

    def test_messages_are_ascii(self):
        # These are printed to consoles that still default to cp1252, where a
        # non-ASCII dash arrives as a replacement character.
        for topology in (
            classify([igpu(outputs=0)]),
            classify([igpu(), dgpu()]),
            GpuTopology(),
        ):
            (topology.help_text() + topology.describe()
             + topology.duplication_failure_help()).encode("ascii")

    def test_single_description_stays_quiet(self):
        described = classify([igpu()]).describe()
        assert described.startswith("Topology: single")
        assert "Hybrid" not in described


# The ThinkPad E580 measured 2026-11-11: a Radeon 500 and a UHD 620, with the
# panel and HDMI both wired to Intel. With python.exe on the Radeon (the
# Windows default there), DXGI lists the panel under the Radeon and Intel shows
# no outputs -- the same picture an Optimus laptop with the display on the
# dGPU would give -- and every duplication is refused.
PANEL = "\\\\.\\DISPLAY1"
UHD_620 = "PCI\\VEN_8086&DEV_5917&SUBSYS_506F17AA&REV_07"


def radeon(outputs=1, index=0):
    return AdapterInfo(index=index, description="Radeon 500 Series", vendor_id=0x1002,
                       device_id=0x699F, dedicated_video_memory=2048 * 1048576,
                       output_count=outputs)


def uhd620(outputs=0, index=1):
    return AdapterInfo(index=index, description="Intel(R) UHD Graphics 620",
                       vendor_id=0x8086, device_id=0x5917,
                       dedicated_video_memory=128 * 1048576, output_count=outputs)


def e580():
    adapters = [radeon(), uhd620(), warp(index=2)]
    offloads = find_render_offloads(
        [(PANEL, adapters[0])], {PANEL: ("Intel(R) UHD Graphics 620", UHD_620)})
    return classify(adapters, offloads)


class TestRenderOffload:
    def test_pci_ids_are_read_from_a_gdi_device_id(self):
        assert pci_ids(UHD_620) == (0x8086, 0x5917)
        assert pci_ids("pci\\ven_10de&dev_2860") == (0x10DE, 0x2860)

    def test_a_device_id_that_is_not_pci_is_not_comparable(self):
        # An indirect display driver, a remote session: no PCI id, so no
        # claim either way.
        assert pci_ids("ROOT\\DISPLAY\\0000") is None
        assert pci_ids("") is None

    def test_the_e580_is_found(self):
        assert e580().render_offloads == (RenderOffload(
            output=PANEL, wired_to="Intel(R) UHD Graphics 620",
            listed_under="Radeon 500 Series"),)

    def test_the_same_gpu_on_both_sides_is_not_an_offload(self):
        # This PC's real ids, read 2026-11-11: GDI and DXGI agree.
        intel = AdapterInfo(index=0, description="Intel(R) Graphics", vendor_id=0x8086,
                            device_id=0x7D67, output_count=2)
        wired = {name: ("Intel(R) Graphics", "PCI\\VEN_8086&DEV_7D67&SUBSYS_8D13103C&REV_06")
                 for name in ("\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2")}
        assert find_render_offloads([(n, intel) for n in wired], wired) == ()

    def test_unknown_or_uncomparable_displays_are_left_out(self):
        listed = [(PANEL, radeon())]
        assert find_render_offloads(listed, {}) == ()
        assert find_render_offloads(listed, {PANEL: ("Remote Display", "ROOT\\RDP")}) == ()
        assert find_render_offloads([(PANEL, warp())], {PANEL: ("x", UHD_620)}) == ()

    def test_the_help_names_both_gpus_and_the_setting_not_the_firmware(self):
        help_text = e580().duplication_failure_help()
        assert "connected to Intel(R) UHD Graphics 620" in help_text
        assert "runs on Radeon 500 Series" in help_text
        assert "Settings > System > Display > Graphics" in help_text
        assert sys.executable in help_text or getattr(sys, "_base_executable", "") in help_text
        # The advice this case used to get, and which could not have helped.
        assert "MUX" not in help_text
        assert "GPU mode" not in help_text

    def test_the_description_does_not_say_the_rendering_gpu_owns_the_display(self):
        described = e580().describe()
        assert "Radeon 500 Series owns the display" not in described
        assert "connected to Intel(R) UHD Graphics 620" in described

    def test_a_hybrid_refusal_without_proof_checks_the_gpu_setting_first(self):
        help_text = classify([igpu(), dgpu()]).duplication_failure_help()
        assert help_text.index("Display > Graphics") < help_text.index("MUX")

    def test_offload_messages_are_ascii(self):
        topology = e580()
        (topology.describe() + topology.duplication_failure_help()).encode("ascii")

    def test_diagnose_says_where_the_display_is_connected(self, monkeypatch):
        rapidshot = pytest.importorskip("rapidshot")
        import rapidshot.util.topology as topology_module

        monkeypatch.setattr(topology_module, "probe_topology", e580)
        text = rapidshot.diagnose()
        assert "Radeon 500 Series (AMD, 1 output, listed for this process, not connected)" in text
        assert f"{PANEL}: connected to Intel(R) UHD Graphics 620" in text
        # The old note, which told this machine its capture ran on the Radeon.
        assert "Capture runs on the display adapter" not in text
        assert rapidshot.capabilities()["capture"]["render_offloads"] == [
            {"output": PANEL, "wired_to": "Intel(R) UHD Graphics 620",
             "listed_under": "Radeon 500 Series"}]


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
