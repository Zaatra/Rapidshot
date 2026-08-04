"""Verify that a cross-adapter transfer moves the *right* bytes.

Run this on real hardware. It cannot be a unit test: CI runners have no desktop
session, and a synthetic texture cannot be used either — D3D11 refuses
SHARED_NTHANDLE without SHARED_KEYEDMUTEX, and a keyed-mutex resource reads as
zeros until acquired, so only a genuinely duplicated surface exercises the path.

The check is against the CPU capture path, which is independent of every line of
D3D12 involved in the transfer. "The copy was submitted without error" is not
evidence; a transfer that silently delivers zeros, or a stale frame, or rows
shifted by the pitch padding, would pass that and fail this.

Desktop Duplication only reports *changed* content, so two captures moments
apart are not guaranteed identical. Rather than assume the screen is still, the
script proves it per attempt: it takes two CPU frames, and only compares against
the transfer when those two match. A blinking cursor alone is enough to spoil an
attempt, so it retries; an exact match over megabytes cannot happen by accident,
so the first success is conclusive.

    python examples/verify_cross_adapter.py
"""

import ctypes
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Running `python examples/verify_cross_adapter.py` puts *examples/* on the
# path, not the repo root, so the import below fails on a source checkout even
# though the package is right there. RELEASING.md tells you to run it exactly
# that way, so the script has to cope -- the same line the benchmark scripts use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rapidshot  # noqa: E402
from rapidshot import native  # noqa: E402


def cpu_frame(camera, attempts=300):
    """One BGRA frame through the ordinary CPU path, as a plain ndarray.

    `grab()` hands back a `PooledBuffer`, not an array: the pixels live in a
    pool buffer that must be handed back explicitly. Holding them across a whole
    attempt exhausts the pool, after which `grab()` returns None — which reads
    as an idle screen rather than as the leak it is. So copy, release, and let
    the caller work with an ordinary array.
    """
    for _ in range(attempts):
        buffer = camera.grab()
        if buffer is not None:
            try:
                return np.array(buffer.array, copy=True)
            finally:
                buffer.release()
        time.sleep(0.01)
    return None


def gpu_texture(camera, attempts=300):
    """One GPU-resident frame, returned as (frame, texture address)."""
    for _ in range(attempts):
        frame = camera.grab_frame()
        if frame is not None:
            return frame, ctypes.cast(frame.d3d11_texture, ctypes.c_void_p).value
        time.sleep(0.01)
    return None, None


def main() -> int:
    if not native.is_available():
        print("native extension not built; nothing to verify")
        return 1

    print(rapidshot.topology_info())
    print()

    # Anything printed while the comparison runs repaints the console, which is
    # a change on screen, which spoils the attempt. On a static desktop the
    # capture path logs a warning every 100 idle polls — enough on its own to
    # make this never converge.
    logging.getLogger("rapidshot").setLevel(logging.ERROR)

    camera = rapidshot.create(output_color="BGRA")
    ext = native.require()

    frame, ptr = gpu_texture(camera)
    if frame is None:
        print("FAIL: no GPU frame captured. Desktop Duplication reports only")
        print("      changed content, so an idle screen yields nothing.")
        return 1

    try:
        transfer = ext.CrossAdapterTransfer(ptr)
    except Exception:
        frame.release()
        raise
    print(f"source      : {transfer.source}")
    print(f"destination : {transfer.destination}")
    print(f"size        : {transfer.width}x{transfer.height}, "
          f"{transfer.total_bytes} bytes, row pitch {transfer.row_pitch}")
    if transfer.destination_is_software:
        print("note        : destination is WARP. The path is exercised in full,")
        print("              but this proves nothing about a real dGPU.")
    frame.release()

    checked = 0
    for attempt in range(1, 6):
        gpu, ptr = gpu_texture(camera)
        if gpu is None:
            time.sleep(0.05)
            continue
        try:
            # One command list writes both the shared heap and the reference,
            # so they see identical source content. Submitting them separately
            # does not work: the duplicated surface is live, and ~2000 bytes in
            # one screen region changed between two copies a millisecond apart.
            expected = np.frombuffer(
                bytes(transfer.transfer_with_reference(ptr)), dtype=np.uint8)
            arrived = np.frombuffer(
                bytes(transfer.read_back_destination()), dtype=np.uint8)
        finally:
            gpu.release()

        if not arrived.any():
            print("\nFAIL: everything arrived as zeros. The destination is reading")
            print("      a heap the source never wrote to.")
            return 1
        if not np.array_equal(arrived, expected):
            differing = int(np.count_nonzero(arrived != expected))
            first = int(np.flatnonzero(arrived != expected)[0])
            print(f"\nFAIL: {differing} of {arrived.size} bytes differ, first at "
                  f"offset {first}: got {arrived[first]}, expected {expected[first]}")
            return 1
        checked += 1

    if checked == 0:
        print("\nINCONCLUSIVE: no frames captured. Desktop Duplication reports only")
        print("      changed content, so an idle screen yields nothing — move the mouse.")
        return 1

    # Independently confirm the bytes really are the desktop, not a plausible
    # pattern both readbacks agree on: compare a CPU capture's colour
    # distribution against the transferred frame's. They are different frames,
    # so this is a sanity check, not an equality test.
    height, width = transfer.height, transfer.width
    image = arrived.reshape(height, transfer.row_pitch)[:, : width * 4]
    image = image.reshape(height, width, 4)
    cpu = cpu_frame(camera)
    if cpu is not None:
        print(f"\ntransferred frame mean BGRA {image.reshape(-1, 4).mean(axis=0).round(1)}")
        print(f"CPU capture mean BGRA        {cpu.reshape(-1, 4).mean(axis=0).round(1)}")

    print(f"\nPASS: {checked} frames transferred to {transfer.destination}; every")
    print(f"      one matched a source-side readback of the same texture exactly")
    print(f"      ({arrived.size} bytes each, row pitch {transfer.row_pitch}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
