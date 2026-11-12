"""GPU/display topology detection.

Desktop Duplication only works against an adapter that actually drives a
display. Two topologies break that assumption, and both currently fail with an
error that says nothing useful:

* **Headless** — no adapter has any output, so ``DuplicateOutput`` has nothing
  to duplicate. Every cloud VM and monitor-less server lands here.
* **Hybrid (Optimus/switchable)** — the iGPU drives the displays and the dGPU
  has no outputs at all. Duplication against the dGPU fails with
  ``DXGI_ERROR_UNSUPPORTED``, so capture happens on the iGPU while the caller's
  inference device is usually the dGPU. Nothing is broken, but a GPU-resident
  frame produced on the capture adapter cannot be consumed on the other one
  without a cross-adapter copy.

The classification here is pure data — it takes already-read adapter
descriptions — so it can be tested without a GPU. ``probe_topology()`` is the
thin COM layer that fills it in.

See ROADMAP.md § 6.1 and § 6.2.
"""

import ctypes
import logging
import re
import sys
import textwrap
from dataclasses import dataclass, field
from typing import List, Mapping, Optional, Sequence, Tuple

logger = logging.getLogger("rapidshot.util.topology")

# Topology kinds, as reported by GpuTopology.kind
HEADLESS = "headless"
SINGLE = "single"
HYBRID = "hybrid"
MULTI_ADAPTER = "multi-adapter"

# PCI vendor IDs, for human-readable reporting only. Never branch capture
# behaviour on these: vendor is not a capability.
_VENDOR_NAMES = {
    0x1002: "AMD",
    0x1022: "AMD",
    0x10DE: "NVIDIA",
    0x1414: "Microsoft",
    0x8086: "Intel",
}

# User-facing strings stay ASCII: they are printed to consoles that are still
# cp1252 by default, where a non-ASCII dash comes out as a replacement char.
_HEADLESS_HELP = (
    "No display available to duplicate.\n"
    "\n"
    "Desktop Duplication captures an attached monitor, so a machine with no "
    "display connected has nothing to capture. On a headless VM, a server, or "
    "a machine running with the lid closed and no external monitor, install a "
    "virtual display driver (an Indirect Display Driver / IDD) so Windows has "
    "a desktop to compose, then retry.\n"
    "\n"
    "Note: a virtual display's advertised refresh rate does not raise the "
    "capture rate. Desktop Duplication is driven by presents, not by refresh. "
    "A 500 Hz virtual display does not make an application render 500 fps."
)

_NO_ADAPTER_HELP = (
    "No DXGI adapters found.\n"
    "\n"
    "Windows reported zero graphics adapters, which normally means the "
    "graphics stack is unavailable to this process rather than that the "
    "machine has no GPU: a service running in session 0, or a container "
    "without GPU access, both look like this."
)

_HYBRID_NOTE = (
    "Hybrid GPU system detected. {capture} owns the display, so duplication is "
    "attempted there first. {others} owns no output, which usually means "
    "Desktop Duplication refuses it with DXGI_ERROR_UNSUPPORTED -- but that "
    "is where the attempt starts, "
    "not a verdict: every adapter is tried and the first one granted "
    "duplication is used. If your inference device is an adapter that is not "
    "capturing, a GPU-resident frame needs a cross-adapter copy to reach it; a "
    "CPU frame from grab() is unaffected.\n"
    "\n"
    "Owning an output is necessary but not sufficient. On some hybrid "
    "configurations the display-owning adapter still refuses to duplicate, so "
    "whether capture works is only known once it has been tried -- this note "
    "describes the arrangement, not a guarantee."
)

# Every adapter refused DuplicateOutput on a hybrid machine, and the display
# could not be shown to be rendered on one GPU and scanned out by another (when
# it can, _RENDER_OFFLOAD_HELP is said instead). Measured 2026-08-21 on an Acer
# PHN16-72: NVIDIA (1 output), Intel (0 outputs) and WARP all returned
# DXGI_ERROR_UNSUPPORTED. This text used to blame the OS hybrid path and the
# MUX alone; on 2026-11-11 a ThinkPad E580 gave the identical signature and the
# cause was the process's GPU preference, so that is checked first now.
_HYBRID_NO_TARGET_HELP = (
    "No adapter on this hybrid system can duplicate the display.\n"
    "\n"
    "Every adapter was tried and each refused with DXGI_ERROR_UNSUPPORTED. "
    "Desktop Duplication only works through the GPU that sends the picture to "
    "the screen, so this usually means the process is running on the other "
    "one. Check, in this order:\n"
    "  - Settings > System > Display > Graphics: {program} must not be set to "
    "the GPU the display is not connected to (on a laptop, usually 'High "
    "performance'), here or in the NVIDIA / AMD control panel. Choose the one "
    "the display is connected to ('Power saving' for the integrated GPU);\n"
    "  - on a laptop with a MUX switch, the GPU mode (Optimus / Hybrid / "
    "Discrete) is set consistently in the vendor utility and in firmware, and "
    "the machine was rebooted after it was changed;\n"
    "  - the integrated GPU driver is current. A stale iGPU driver can leave "
    "the hybrid path uninitialised while the GPU still enumerates normally."
)

# The display is connected to one GPU (GDI says which) while DXGI lists it
# under another for this process: Windows renders on the second and copies to
# the first, and Desktop Duplication through the rendering GPU is refused with
# DXGI_ERROR_UNSUPPORTED (Microsoft KB3019314). Measured 2026-11-11 on a
# ThinkPad E580 (Radeon 500 + UHD 620, both outputs wired to Intel): with
# python.exe on the Radeon every output and every device refused; set to Power
# saving, DXGI listed the display under Intel and duplication worked.
_RENDER_OFFLOAD_HELP = (
    "Desktop Duplication is refused because this process runs on a different "
    "GPU from the one the display is connected to.\n"
    "\n"
    "{offloads}\n"
    "\n"
    "Windows draws this process's view of the desktop on the GPU it runs on and "
    "copies it to the GPU wired to the screen, and Desktop Duplication only "
    "works through the GPU wired to the screen. Nothing in firmware or a driver "
    "needs to change. In Settings > System > Display > Graphics, add {program} "
    "and set it to the GPU the display is connected to ('Power saving' for an "
    "integrated GPU, 'High performance' for a discrete one), then start it "
    "again. In a virtual environment the setting belongs on the base "
    "interpreter, which the environment's python.exe starts; that is the path "
    "named here."
)

_NO_TARGET_HELP = (
    "No adapter could duplicate the display.\n"
    "\n"
    "Every adapter was tried and each refused. Desktop Duplication needs an "
    "adapter that is composing an attached desktop; a remote session, a "
    "process outside the interactive session, or a display driver in a failed "
    "state all produce this."
)


@dataclass(frozen=True)
class AdapterInfo:
    """One DXGI adapter, as far as capture is concerned."""

    index: int
    description: str
    vendor_id: int = 0
    device_id: int = 0
    dedicated_video_memory: int = 0
    is_software: bool = False
    output_count: int = 0
    # Set when the adapter was enumerated but could not be opened as a D3D11
    # device; such an adapter cannot be used for capture whatever its outputs.
    error: Optional[str] = None

    @property
    def vendor(self) -> str:
        return _VENDOR_NAMES.get(self.vendor_id, f"vendor {self.vendor_id:#06x}")

    @property
    def drives_display(self) -> bool:
        """True if this adapter has at least one output attached."""
        return self.output_count > 0

    @property
    def can_capture(self) -> bool:
        """True if Desktop Duplication can run against this adapter."""
        return self.drives_display and self.error is None

    @property
    def is_render_only(self) -> bool:
        """A hardware adapter with no outputs — the dGPU half of a hybrid rig.

        Software adapters (WARP / Microsoft Basic Render Driver) also have no
        outputs, but they are not a second GPU and must not be reported as one.
        """
        return not self.drives_display and not self.is_software

    def __str__(self) -> str:
        vram = self.dedicated_video_memory // 1048576
        bits = [f"{self.description}", f"{self.vendor}", f"{vram}MB VRAM"]
        if self.is_software:
            bits.append("software")
        bits.append(
            f"{self.output_count} output{'s' if self.output_count != 1 else ''}"
        )
        if self.error:
            bits.append(f"unusable: {self.error}")
        return f"Adapter[{self.index}] " + " ".join(f"({b})" for b in bits)


@dataclass(frozen=True)
class RenderOffload:
    """A display connected to one GPU that this process sees through another."""

    output: str
    #: The adapter GDI says the display is connected to.
    wired_to: str
    #: The adapter DXGI lists the display under for this process.
    listed_under: str

    def __str__(self) -> str:
        return (f"{self.output} is connected to {self.wired_to}, but this "
                f"process runs on {self.listed_under}, which only lists it.")


_PCI_ID = re.compile(r"VEN_([0-9A-F]{4})&DEV_([0-9A-F]{4})", re.IGNORECASE)


def pci_ids(device_id: str) -> Optional[Tuple[int, int]]:
    """(vendor, device) from a GDI DeviceID such as ``PCI\\VEN_8086&DEV_5917&...``.

    ``None`` for anything that is not a PCI device -- an indirect display
    driver's ``ROOT\\...`` id, a remote session's -- which cannot be compared
    with DXGI's ids and so must never be called a mismatch.
    """
    match = _PCI_ID.search(device_id or "")
    if not match:
        return None
    return int(match.group(1), 16), int(match.group(2), 16)


def find_render_offloads(
    listed: Sequence[Tuple[str, AdapterInfo]],
    wired: Mapping[str, Tuple[str, str]],
) -> Tuple[RenderOffload, ...]:
    """Displays DXGI lists under an adapter that is not the one driving them.

    Args:
        listed: (output device name, adapter it is listed under) from DXGI.
        wired: output device name -> (adapter description, GDI DeviceID), for
            the GDI adapters attached to the desktop.

    Pure, so the hybrid laptops it exists for are tested by describing them.
    A display GDI does not name, or names with no PCI id, is left out: this
    reports a mismatch only when both sides are known and disagree.
    """
    found = []
    for name, adapter in listed:
        if adapter.is_software or name not in wired:
            continue
        description, device_id = wired[name]
        ids = pci_ids(device_id)
        if ids is None or ids == (adapter.vendor_id, adapter.device_id):
            continue
        found.append(RenderOffload(output=name, wired_to=description,
                                   listed_under=adapter.description))
    return tuple(found)


def _program() -> str:
    """The executable Windows applies a per-app GPU preference to."""
    return getattr(sys, "_base_executable", None) or sys.executable or "python.exe"


@dataclass(frozen=True)
class GpuTopology:
    """How this machine's adapters and displays are arranged."""

    adapters: Tuple[AdapterInfo, ...] = field(default_factory=tuple)
    #: Displays this process sees through a GPU other than the one they are
    #: connected to. Empty when there are none, or GDI could not be asked.
    render_offloads: Tuple[RenderOffload, ...] = field(default_factory=tuple)

    @property
    def kind(self) -> str:
        if not self.capture_adapters:
            return HEADLESS
        if self.render_only_adapters:
            return HYBRID
        if len(self.capture_adapters) > 1:
            return MULTI_ADAPTER
        return SINGLE

    @property
    def capture_adapters(self) -> Tuple[AdapterInfo, ...]:
        """Adapters Desktop Duplication can actually run against."""
        return tuple(a for a in self.adapters if a.can_capture)

    @property
    def render_only_adapters(self) -> Tuple[AdapterInfo, ...]:
        """Hardware adapters with no outputs — dGPUs on a hybrid system."""
        return tuple(a for a in self.adapters if a.is_render_only)

    @property
    def software_adapters(self) -> Tuple[AdapterInfo, ...]:
        """WARP and friends. Not usable for capture, but a real second D3D12
        device — which makes it the only way to exercise the cross-adapter path
        on a single-GPU machine."""
        return tuple(a for a in self.adapters if a.is_software)

    @property
    def is_headless(self) -> bool:
        return self.kind == HEADLESS

    @property
    def is_hybrid(self) -> bool:
        return self.kind == HYBRID

    def help_text(self) -> str:
        """The actionable message for a topology that cannot capture."""
        if not self.adapters:
            return _NO_ADAPTER_HELP
        if self.is_headless:
            adapters = "\n".join(f"  {a}" for a in self.adapters)
            return f"{_HEADLESS_HELP}\n\nAdapters found, none with an output:\n{adapters}"
        return ""

    def duplication_failure_help(self) -> str:
        """Actionable text for "every adapter refused to duplicate".

        Separate from :meth:`help_text`, which answers "there is nothing here
        to capture". This one answers "there is a display and no adapter would
        duplicate it" -- a different cause with a different fix, and previously
        surfaced as a raw COM string that named neither.
        """
        adapters = "\n".join(f"  {a}" for a in self.adapters)
        if self.render_offloads:
            offloads = "\n".join(f"  {o}" for o in self.render_offloads)
            base = _RENDER_OFFLOAD_HELP.format(offloads=offloads, program=_program())
        elif self.is_hybrid:
            base = _HYBRID_NO_TARGET_HELP.format(program=_program())
        else:
            base = _NO_TARGET_HELP
        return f"{base}\n\nAdapters tried:\n{adapters}"

    def describe(self) -> str:
        """Multi-line human summary — what device_info() appends."""
        lines = [f"Topology: {self.kind}"]
        lines += [f"  {a}" for a in self.adapters]

        if self.render_offloads:
            # The hybrid note would name the GPU this process runs on as the
            # one that "owns the display", which is exactly backwards here.
            lines.append("")
            lines += [f"  {o}" for o in self.render_offloads]
            lines += textwrap.wrap(
                "Desktop Duplication through that GPU is refused; set "
                f"{_program()} to the GPU the display is connected to in "
                "Settings > System > Display > Graphics.",
                width=76, initial_indent="  ", subsequent_indent="  ")
        elif self.is_hybrid:
            note = _HYBRID_NOTE.format(
                capture=self.capture_adapters[0].description,
                others=", ".join(a.description for a in self.render_only_adapters),
            )
            lines.append("")
            # Wrap per paragraph: a single wrap() call folds the blank lines
            # out of the note and runs the paragraphs together.
            for i, para in enumerate(note.split("\n\n")):
                if i:
                    lines.append("")
                lines += textwrap.wrap(para, width=76, initial_indent="  ",
                                       subsequent_indent="  ")
        return "\n".join(lines)


def classify(adapters: Sequence[AdapterInfo],
             render_offloads: Sequence[RenderOffload] = ()) -> GpuTopology:
    """Build a GpuTopology from already-read adapter descriptions."""
    return GpuTopology(adapters=tuple(adapters), render_offloads=tuple(render_offloads))


def _wired_displays() -> dict:
    """GDI's view: display device name -> (adapter description, DeviceID), for
    the adapters attached to the desktop. Empty if GDI cannot be asked."""
    from rapidshot._libs.user32 import DISPLAY_DEVICE, DISPLAY_DEVICE_ACTIVE
    from rapidshot.util.io import _user32

    wired = {}
    if _user32 is None:
        return wired
    i = 0
    while True:
        device = DISPLAY_DEVICE()
        device.cb = ctypes.sizeof(device)
        if not _user32.EnumDisplayDevicesW(None, i, ctypes.byref(device), 0):
            break
        if device.StateFlags & DISPLAY_DEVICE_ACTIVE:
            wired[device.DeviceName] = (device.DeviceString, device.DeviceID)
        i += 1
    return wired


def probe_topology() -> GpuTopology:
    """Enumerate adapters and outputs, and classify the result.

    Returns:
        GpuTopology. An empty one if DXGI could not be reached at all.
    """
    from rapidshot._libs.dxgi import (
        DXGI_ADAPTER_DESC1,
        DXGI_ADAPTER_FLAG_SOFTWARE,
        DXGI_OUTPUT_DESC,
    )
    from rapidshot.util.io import enum_dxgi_adapters, enum_dxgi_outputs

    adapters: List[AdapterInfo] = []
    listed: List[Tuple[str, AdapterInfo]] = []
    try:
        p_adapters = enum_dxgi_adapters()
    except Exception as exc:  # DXGI unreachable — report it as "no adapters"
        logger.warning(f"Could not enumerate DXGI adapters: {exc}")
        return GpuTopology()

    for index, p_adapter in enumerate(p_adapters):
        desc = DXGI_ADAPTER_DESC1()
        error = None
        try:
            p_adapter.GetDesc1(ctypes.byref(desc))
        except Exception as exc:
            logger.warning(f"Could not describe adapter {index}: {exc}")
            adapters.append(AdapterInfo(index=index, description="<unreadable>", error=str(exc)))
            continue

        names: List[str] = []
        try:
            outputs = enum_dxgi_outputs(p_adapter)
            output_count = len(outputs)
            for p_output in outputs:
                out_desc = DXGI_OUTPUT_DESC()
                try:
                    p_output.GetDesc(ctypes.byref(out_desc))
                    names.append(out_desc.DeviceName)
                except Exception:  # a name is only needed for the comparison
                    pass
        except Exception as exc:
            logger.warning(f"Could not enumerate outputs on adapter {index}: {exc}")
            output_count = 0
            error = str(exc)

        info = AdapterInfo(
            index=index,
            description=desc.Description,
            vendor_id=desc.VendorId,
            device_id=desc.DeviceId,
            dedicated_video_memory=desc.DedicatedVideoMemory,
            is_software=bool(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
            output_count=output_count,
            error=error,
        )
        adapters.append(info)
        listed += [(name, info) for name in names]

    try:
        offloads = find_render_offloads(listed, _wired_displays())
    except Exception as exc:  # a diagnostic must not stop the probe
        logger.debug(f"Could not compare GDI and DXGI display owners: {exc}")
        offloads = ()
    return classify(adapters, offloads)
