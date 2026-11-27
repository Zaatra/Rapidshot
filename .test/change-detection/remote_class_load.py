"""A remote-desktop-class Desktop Duplication load, for contention measurement.

RustDesk, TeamViewer and AnyDesk are applications, not importable capture
libraries, so they cannot be driven as arms of `cost_benchmark.py`. What they
*do* to a machine, though, is well defined and reproducible: a second process
opens DXGI Desktop Duplication on the same output and pulls frames off it in a
loop at a target rate, copying each one to the CPU so it can be encoded.

This script is that load, and nothing else. It deliberately does not encode or
transmit: the question is what a competing duplication consumer on the same
output costs RapidShot, not how fast an encoder is.

It uses `dxcam` rather than `rapidshot` so the competing consumer is a
third-party DDA client, and a reader cannot object that RapidShot was measured
against a privileged copy of itself.

Usage:
    python remote_class_load.py --seconds 260 --fps 30 --output load.json

Honest scope: this is a RustDesk-*class* load, not RustDesk. It reproduces the
duplication contention and the CPU readback of a 30 fps remote-desktop sender.
It does not reproduce RustDesk's encoder, its network stack, or its cursor and
clipboard handling.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--display", type=int, default=0)
    args = parser.parse_args()

    import dxcam

    interval = 1.0 / args.fps
    camera = dxcam.create(device_idx=args.device, output_idx=args.display,
                          output_color="BGRA")
    stats = {
        "pid": os.getpid(),
        "requested_seconds": args.seconds,
        "target_fps": args.fps,
        "device_idx": args.device,
        "output_idx": args.display,
        "library": "dxcam",
        "frames": 0,
        "empty": 0,
        "bytes_copied": 0,
    }
    start = time.perf_counter()
    deadline = start + args.seconds
    next_due = start
    try:
        while True:
            now = time.perf_counter()
            if now >= deadline:
                break
            if now < next_due:
                time.sleep(min(next_due - now, 0.002))
                continue
            next_due += interval
            # copy=True so the pixels land in CPU memory, as an encoder needs
            frame = camera.grab()
            if frame is None:
                stats["empty"] += 1
                continue
            stats["frames"] += 1
            stats["bytes_copied"] += int(frame.nbytes)
    finally:
        try:
            camera.release()
        except Exception:
            pass
        stats["elapsed_seconds"] = time.perf_counter() - start
        stats["achieved_fps"] = (
            stats["frames"] / stats["elapsed_seconds"]
            if stats["elapsed_seconds"] else 0.0
        )
        stats["MiB_copied"] = round(stats["bytes_copied"] / 2 ** 20, 2)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
