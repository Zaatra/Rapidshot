import pytest

pytest.importorskip("comtypes")

from rapidshot.capture import ScreenCapture


class DummyOutput:
    def __init__(self, surface_size, rotation_angle):
        self._surface_size = surface_size
        self._rotation_angle = rotation_angle

    @property
    def surface_size(self):
        return self._surface_size

    @property
    def rotation_angle(self):
        return self._rotation_angle


class DummyBox:
    def __init__(self):
        self.left = self.top = self.right = self.bottom = 0


def make_capture(width, height):
    capture = object.__new__(ScreenCapture)
    capture.width = width
    capture.height = height
    capture._sourceRegion = None
    capture.shot_w = 0
    capture.shot_h = 0
    return capture


@pytest.mark.parametrize(
    "rotation, surface_size, region, expected",
    [
        (0, (8, 6), (1, 2, 4, 5), (1, 2, 4, 5)),
        (90, (6, 8), (1, 2, 4, 5), (2, 4, 5, 7)),
        (180, (8, 6), (1, 2, 4, 5), (4, 1, 7, 4)),
        (270, (6, 8), (1, 2, 4, 5), (1, 1, 4, 4)),
    ],
)
def test_region_to_memory_region(rotation, surface_size, region, expected):
    capture = make_capture(8, 6)
    output = DummyOutput(surface_size, rotation)
    assert (
        capture.region_to_memory_region(region, rotation, output)
        == expected
    )


# Desktop is 8x6 at every rotation below; the texture is 8x6 at 0/180 and the
# transposed 6x8 at 90/270, as Output.surface_size reports it.
DESKTOP = (8, 6)
SURFACES = {0: (8, 6), 90: (6, 8), 180: (8, 6), 270: (6, 8)}


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_full_screen_region_maps_to_the_whole_texture(rotation):
    """The case that was broken: a full-screen region on a non-square panel.

    The old 90/270 formulas subtracted the wrong dimension and produced boxes
    with negative or out-of-range edges, so the copy was silently skipped.
    """
    width, height = SURFACES[rotation]
    capture = make_capture(*DESKTOP)
    output = DummyOutput((width, height), rotation)
    assert capture.region_to_memory_region(
        (0, 0, *DESKTOP), rotation, output) == (0, 0, width, height)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_every_valid_region_maps_inside_the_texture(rotation):
    width, height = SURFACES[rotation]
    capture = make_capture(*DESKTOP)
    output = DummyOutput((width, height), rotation)
    dw, dh = DESKTOP
    for left in range(dw):
        for right in range(left + 1, dw + 1):
            for top in range(dh):
                for bottom in range(top + 1, dh + 1):
                    l, t, r, b = capture.region_to_memory_region(
                        (left, top, right, bottom), rotation, output)
                    assert 0 <= l < r <= width and 0 <= t < b <= height, (
                        (left, top, right, bottom), (l, t, r, b))
                    assert (r - l) * (b - t) == (right - left) * (bottom - top)


def _texture_to_desktop(rect, rotation, desktop):
    """Where a texture rect appears on the desktop.

    An independent reference, not an inversion of the code under test: this is
    the direction Microsoft's Desktop Duplication sample computes when it draws
    dirty rects (DisplayManager::SetDirtyVert), with Width/Height being the
    desktop's dimensions.
    """
    left, top, right, bottom = rect
    width, height = desktop
    if rotation == 0:
        return rect
    if rotation == 90:
        return (width - bottom, left, width - top, right)
    if rotation == 180:
        return (width - right, height - bottom, width - left, height - top)
    return (top, height - right, bottom, height - left)


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_each_texture_pixel_round_trips(rotation):
    width, height = SURFACES[rotation]
    capture = make_capture(*DESKTOP)
    output = DummyOutput((width, height), rotation)
    for u in range(width):
        for v in range(height):
            texel = (u, v, u + 1, v + 1)
            on_desktop = _texture_to_desktop(texel, rotation, DESKTOP)
            assert capture.region_to_memory_region(on_desktop, rotation, output) == texel


def test_region_to_memory_region_rotation_mismatch():
    capture = make_capture(10, 10)
    output = DummyOutput((10, 10), 90)
    with pytest.raises(AssertionError):
        capture.region_to_memory_region((0, 0, 5, 5), 0, output)


def test_normalize_region_validates_without_side_effects():
    capture = make_capture(12, 8)
    normalized = capture._normalize_region((1, 2, 6, 7))
    assert normalized == (1, 2, 6, 7)
    # Calling normalize should not set capture.region
    assert not hasattr(capture, "region")


@pytest.mark.parametrize(
    "region",
    [
        (-1, 0, 2, 2),
        (0, -1, 2, 2),
        (0, 0, 13, 1),
        (0, 0, 1, 9),
        (4, 4, 3, 5),
    ],
)
def test_normalize_region_invalid(region):
    capture = make_capture(12, 8)
    with pytest.raises(ValueError):
        capture._normalize_region(region)


def test_validate_region_updates_state():
    capture = make_capture(20, 10)
    capture._sourceRegion = DummyBox()
    capture._validate_region((2, 3, 10, 9))
    assert capture.region == (2, 3, 10, 9)
    assert capture.shot_w == 8
    assert capture.shot_h == 6
    assert capture._sourceRegion.left == 2
    assert capture._sourceRegion.top == 3
    assert capture._sourceRegion.right == 10
    assert capture._sourceRegion.bottom == 9
