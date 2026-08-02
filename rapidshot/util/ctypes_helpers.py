"""Utility helpers for working with ctypes values."""

from __future__ import annotations

import ctypes
from typing import Any, Optional, Tuple, Union


def describe_destination(dest: Any) -> Tuple[Optional[int], Optional[int]]:
    """Return ``(address, size_in_bytes)`` for a writable destination buffer.

    Accepts NumPy arrays, ctypes arrays, objects supporting the writable buffer
    protocol (``bytearray``, ``memoryview``), and bare pointers/addresses.

    The size is ``None`` only for bare pointers, which carry no length
    information. Callers must treat that case as "unverifiable" rather than
    "unlimited" -- writing to an undersized raw pointer corrupts the heap with
    no Python-level error.

    Parameters
    ----------
    dest:
        The destination buffer or a pointer to it.

    Returns
    -------
    Tuple[Optional[int], Optional[int]]
        The integer address (or ``None`` if it cannot be resolved) and the
        buffer size in bytes (or ``None`` if the size is unknowable).
    """

    if dest is None:
        return None, None

    # NumPy (and CuPy-on-host) arrays expose both address and exact byte count.
    ctypes_attr = getattr(dest, "ctypes", None)
    nbytes = getattr(dest, "nbytes", None)
    if ctypes_attr is not None and nbytes is not None:
        flags = getattr(dest, "flags", None)
        if flags is not None and not flags.c_contiguous:
            raise ValueError(
                "Destination array must be C-contiguous; pass "
                "numpy.ascontiguousarray(dest) instead."
            )
        if flags is not None and not flags.writeable:
            raise ValueError("Destination array is read-only.")
        return int(ctypes_attr.data), int(nbytes)

    # ctypes arrays and structures.
    if isinstance(dest, ctypes.Array) or isinstance(dest, ctypes.Structure):
        return ctypes.addressof(dest), ctypes.sizeof(dest)

    # Writable buffer-protocol objects (bytearray, memoryview over one).
    if isinstance(dest, (bytearray, memoryview)):
        view = memoryview(dest)
        if view.readonly:
            raise ValueError("Destination buffer is read-only.")
        backing = (ctypes.c_char * view.nbytes).from_buffer(dest)
        return ctypes.addressof(backing), view.nbytes

    # Anything else is treated as a bare pointer: address only, size unknown.
    return pointer_to_address(dest), None


def pointer_to_address(ptr: Union[int, ctypes.c_void_p, ctypes._Pointer]) -> Optional[int]:
    """Return the integer address represented by *ptr*.

    Parameters
    ----------
    ptr:
        A ctypes pointer-like object or an integer address.

    Returns
    -------
    Optional[int]
        The integer address, or ``None`` if *ptr* does not represent a
        valid address.
    """

    if ptr is None:
        return None

    if isinstance(ptr, int):
        return ptr

    if isinstance(ptr, ctypes.c_void_p):
        return ptr.value

    try:
        value = ctypes.cast(ptr, ctypes.c_void_p).value
        if value is not None:
            return value
    except (TypeError, ValueError):
        pass

    if hasattr(ptr, "value") and ptr.value is not None:
        return ptr.value

    if hasattr(ptr, "contents"):
        try:
            return ctypes.addressof(ptr.contents)
        except (TypeError, ValueError):
            return None

    return None
