"""What to capture, separate from how it is captured (ROADMAP § 7.3).

A *source* names the thing on screen -- a monitor, a window, a rectangle of
the desktop. A *backend* (``backend="dxgi"`` or ``"wgc"``) is the mechanism
that delivers its pixels. Keeping the two apart is what lets the same
``Window(...)`` resolve to Windows.Graphics.Capture here and, later, to
PipeWire on Linux or ScreenCaptureKit on macOS (§ 9), so nothing in these types
may assume Windows beyond an opaque handle.

    rapidshot.create(source=rapidshot.Monitor(1))
    rapidshot.create(source=rapidshot.Window.find("Notepad"))
    rapidshot.create(source=rapidshot.Region(0, 0, 1280, 720))

Plain values: hashable, comparable, and free of any OS resource, so a source
can be built before a camera exists and used as a cache key.
"""

from dataclasses import dataclass
from typing import Optional, Union

__all__ = ["Monitor", "Window", "Region", "Source"]


@dataclass(frozen=True)
class Monitor:
    """A whole display.

    Args:
        index: Position among the displays RapidShot enumerates, counted across
            every adapter in enumeration order (``rapidshot.output_info()``
            lists them). ``None`` is the primary display.
    """

    index: Optional[int] = None

    def __post_init__(self):
        if self.index is not None and (
                not isinstance(self.index, int) or isinstance(self.index, bool)
                or self.index < 0):
            raise ValueError(f"Monitor index must be a non-negative int or None, "
                             f"got {self.index!r}")

    def __str__(self) -> str:
        return "Monitor(primary)" if self.index is None else f"Monitor({self.index})"


@dataclass(frozen=True)
class Window:
    """One top-level window.

    Args:
        handle: The platform's window handle as an int -- an ``HWND`` on
            Windows. Opaque to everything but the backend that captures it.
    """

    handle: int

    def __post_init__(self):
        if not isinstance(self.handle, int) or isinstance(self.handle, bool) \
                or self.handle <= 0:
            raise ValueError(f"Window handle must be a positive int, got {self.handle!r}")

    @classmethod
    def find(cls, title: str, *, exact: bool = False) -> "Window":
        """The first visible top-level window whose title matches.

        Args:
            title: Text to look for in the window title, case-insensitively.
            exact: Require the whole title to match rather than contain it.

        Raises:
            LookupError: No visible top-level window matches.
        """
        from rapidshot.util.windows import find_window

        handle = find_window(title, exact=exact)
        if handle is None:
            raise LookupError(f"No visible window whose title "
                              f"{'is' if exact else 'contains'} {title!r}")
        return cls(handle)

    def __str__(self) -> str:
        return f"Window(0x{self.handle:x})"


@dataclass(frozen=True)
class Region:
    """A rectangle of the desktop, in virtual-desktop coordinates.

    Resolved to the one display that contains it; a rectangle spanning two
    displays is refused, because one capture delivers one display's pixels.
    ``(left, top)`` is inclusive and ``(right, bottom)`` exclusive, as
    everywhere else in RapidShot.
    """

    left: int
    top: int
    right: int
    bottom: int

    def __post_init__(self):
        for name in ("left", "top", "right", "bottom"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Region {name} must be an int, got {value!r}")
        if self.right <= self.left or self.bottom <= self.top:
            raise ValueError(f"Region must have positive size, got "
                             f"({self.left}, {self.top}, {self.right}, {self.bottom})")

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def __str__(self) -> str:
        return f"Region({self.left}, {self.top}, {self.right}, {self.bottom})"


Source = Union[Monitor, Window, Region]
