"""What this machine can do, as distinct from how fast it does it. (ROADMAP.md § 7.5)

Two probes ``rapidshot benchmark`` records beside its timings:

    capture        the DXGI format a RapidShot capture actually receives on this
                   desktop -- BGRA8 normally, FP16 scRGB with HDR on -- which is
                   the evidence the display's advanced-colour state (recorded in
                   every environment) is only a claim about
    cross_adapter  whether a frame can cross to a second adapter through a shared
                   heap, and whether that adapter is real hardware or WARP

It runs as a worker, like every other harness step, so a driver fault in either
probe costs the report one section rather than the run.

Usage::

    python -m rapidshot._bench.capabilities --out capabilities.json
"""

import argparse
import ctypes
import json
from pathlib import Path
import time

#: DXGI_FORMAT values DuplicateOutput1 can hand back (see core.duplicator).
FORMATS = {87: "B8G8R8A8_UNORM", 28: "R8G8B8A8_UNORM", 24: "R10G10B10A2_UNORM",
           10: "R16G16B16A16_FLOAT"}


def _error(exc):
    return {"error": f"{type(exc).__name__}: {exc}"}


def captured_format(timeout_s: float = 3.0) -> dict:
    """Grab one GPU frame and read the texture's format."""
    import rapidshot
    from rapidshot._libs.d3d11 import D3D11_TEXTURE2D_DESC

    camera = rapidshot.create()
    try:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = camera.grab_frame()
            if frame is None:
                time.sleep(0.02)
                continue
            with frame:
                desc = D3D11_TEXTURE2D_DESC()
                frame.d3d11_texture.GetDesc(ctypes.byref(desc))
                return {"dxgi_format": desc.Format,
                        "format": FORMATS.get(desc.Format, str(desc.Format)),
                        "hdr": desc.Format == 10,
                        "width": desc.Width, "height": desc.Height}
        return {"error": f"no frame within {timeout_s:g} s"}
    finally:
        camera.release()


def displays() -> dict:
    """Advanced-colour state per output, in the shape every recording's environment has.

    Recorded here as well so the report can say whether HDR was on even when
    no timing pass produced an environment to read it from.
    """
    from .machine_inventory import discover_displays

    return {"outputs": [{"primary": o.get("primary"), "advanced_color": o.get("advanced_color")}
                        for o in discover_displays()["outputs"]]}


def cross_adapter() -> dict:
    from rapidshot import native

    if not native.is_available():
        return {"supported": None, "reason": "rapidshot-native is not installed"}
    # With one real GPU the far side is WARP: the mechanism is proven and
    # `representative` is False, so the timing says nothing about a real move.
    probe = native.probe_cross_adapter()
    keep = ("supported", "representative", "reason", "source", "destination",
            "source_row_major_texture", "destination_row_major_texture",
            "copy_ms_min", "copy_ms_median", "throughput_mb_s")
    return {k: probe[k] for k in keep if k in probe}


def probe() -> dict:
    out = {}
    for name, fn in (("displays", displays), ("capture", captured_format),
                     ("cross_adapter", cross_adapter)):
        try:
            out[name] = fn()
        except Exception as exc:  # noqa: BLE001 -- recorded, not raised
            out[name] = _error(exc)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    print("[capabilities] capture format, cross-adapter", flush=True)
    result = probe()
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
