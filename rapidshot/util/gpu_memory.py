"""This process's GPU memory, as DXGI accounts it.

Task Manager's per-process "Shared/Dedicated GPU memory" and the
``\\GPU Process Memory(*)`` counters charge a process a shared resource's full
size every time it opens one and never take the charge back (README, Known
limitations), so they report memory a capture program does not hold.
``IDXGIAdapter3::QueryVideoMemoryInfo`` follows the references the process
actually has, which is what this reads, summed over every adapter.
"""
from __future__ import annotations

import ctypes
from typing import Dict

__all__ = ["process_gpu_memory_mb"]


class _VideoMemoryInfo(ctypes.Structure):
    _fields_ = [("Budget", ctypes.c_uint64), ("CurrentUsage", ctypes.c_uint64),
                ("AvailableForReservation", ctypes.c_uint64),
                ("CurrentReservation", ctypes.c_uint64)]


# IUnknown 3 + IDXGIObject 4 + IDXGIAdapter 3 + GetDesc1 + GetDesc2, then
# IDXGIAdapter3's two content-protection methods: QueryVideoMemoryInfo is 14.
# rapidshot/_libs/dxgi.py declares it without argument types, so it is called
# through the vtable.
_QUERY_VIDEO_MEMORY_INFO = 14
_QueryVideoMemoryInfo = ctypes.WINFUNCTYPE(
    ctypes.c_long, ctypes.c_void_p, ctypes.c_uint, ctypes.c_int,
    ctypes.POINTER(_VideoMemoryInfo))


def process_gpu_memory_mb() -> Dict[str, float]:
    """``{"local": MiB, "non_local": MiB}`` this process uses, over all adapters.

    On an integrated GPU "local" is system memory -- what Task Manager calls
    shared -- and on a discrete one it is VRAM.
    """
    from rapidshot._libs.dxgi import IDXGIAdapter3
    from rapidshot.util.io import enum_dxgi_adapters

    totals = [0, 0]
    for adapter in enum_dxgi_adapters():
        adapter3 = adapter.QueryInterface(IDXGIAdapter3)
        raw = ctypes.cast(adapter3, ctypes.c_void_p).value
        vtable = ctypes.cast(raw, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)))[0]
        query = _QueryVideoMemoryInfo(vtable[_QUERY_VIDEO_MEMORY_INFO])
        for group in (0, 1):  # DXGI_MEMORY_SEGMENT_GROUP_LOCAL, _NON_LOCAL
            info = _VideoMemoryInfo()
            if query(raw, 0, group, ctypes.byref(info)) < 0:
                raise OSError(f"QueryVideoMemoryInfo failed on segment group {group}")
            totals[group] += info.CurrentUsage
    return {"local": round(totals[0] / 2**20, 2), "non_local": round(totals[1] / 2**20, 2)}
