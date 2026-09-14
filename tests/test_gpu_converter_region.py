"""`GpuConverter` must convert a frame's region, not the monitor under it.

A camera created with ``region=`` hands back frames whose ``width`` and
``height`` are the region's, over a texture that is still the whole output.
Before crop existed the converter sized its sampling from the texture, so it
resized the **entire monitor** into a correctly shaped output — found
2026-09-14 while designing crop, with no test that could have caught it.

Its own module because it needs its own camera: a second camera on the same
output with different settings is refused.

The reference is read through the native converter directly, which knows
nothing about regions and so sees the whole surface.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

# Odd offsets, so a region applied off by one cannot hide behind an even stride.
REGION = (101, 51, 421, 291)
WIDTH, HEIGHT = REGION[2] - REGION[0], REGION[3] - REGION[1]


@pytest.fixture(scope="module")
def region_frame():
    camera = rapidshot.create(output_color="BGRA", region=REGION)
    frame = None
    for _ in range(600):
        frame = camera.grab_frame()
        if frame is not None:
            break
    if frame is None:
        camera.release()
        pytest.skip("no frame captured — the screen must be changing")
    yield frame
    frame.release()
    camera.release()


@pytest.fixture(scope="module")
def surface(region_frame):
    """(H, W, 4) BGRA of the whole texture, bypassing region handling."""
    info = native.describe_texture(region_frame)
    w, h = int(info["width"]), int(info["height"])
    impl = native.require().GpuConverter12(
        native._texture_address(region_frame), w, h, sampling="nearest", dtype="bgra8"
    )
    impl.process(native._texture_address(region_frame), source_id=region_frame.source_id)
    return np.frombuffer(impl.read_back(), dtype=np.uint8).reshape(h, w, 4)


def converted(frame, size, crop=None):
    return rapidshot.GpuConverter(
        frame, size, dtype="uint8", layout="nhwc", sampling="nearest"
    ).process(frame, crop=crop).numpy()[0]


def test_frame_is_smaller_than_its_texture(region_frame, surface):
    """The premise. If this stops holding, the bug below cannot occur."""
    assert (region_frame.width, region_frame.height) == (WIDTH, HEIGHT)
    assert surface.shape[1] > WIDTH and surface.shape[0] > HEIGHT


def test_region_frame_converts_only_the_region(region_frame, surface):
    left, top, right, bottom = REGION
    expected = surface[top:bottom, left:right]

    whole_monitor = surface[
        np.ix_(
            np.arange(HEIGHT) * surface.shape[0] // HEIGHT,
            np.arange(WIDTH) * surface.shape[1] // WIDTH,
        )
    ]
    if np.array_equal(expected, whole_monitor):
        pytest.skip("region and resized monitor are indistinguishable on this screen")

    np.testing.assert_array_equal(converted(region_frame, (WIDTH, HEIGHT)), expected)


def test_crop_is_relative_to_the_region(region_frame, surface):
    crop = (11, 21, 111, 101)
    left, top = REGION[0] + crop[0], REGION[1] + crop[1]
    expected = surface[top:top + 80, left:left + 100]
    np.testing.assert_array_equal(converted(region_frame, (100, 80), crop), expected)


def test_batched_regions_are_relative_to_the_frame_region(region_frame, surface):
    a, b = (0, 0, 64, 48), (201, 151, 265, 199)
    got = rapidshot.GpuConverter(
        region_frame, (64, 48), dtype="uint8", layout="nhwc", sampling="nearest", batch=2
    ).process(region_frame, regions=[a, b]).numpy()
    for slot, (left, top, right, bottom) in enumerate((a, b)):
        x, y = REGION[0] + left, REGION[1] + top
        np.testing.assert_array_equal(got[slot], surface[y:y + 48, x:x + 64])


# --------------------------------------------------------------------------
# GpuPreprocessor12 — the same bug, shipped since the path existed
# --------------------------------------------------------------------------


def preprocessor_reference(pixels):
    """(1, 3, H, W) float32 RGB in 0..1 from BGRA bytes — what the preprocessor
    computes at scale 1.0, bias 0.0. UNORM-to-float is exact, so exact
    equality is the right bar."""
    rgb = pixels[..., [2, 1, 0]].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def test_preprocessor12_converts_only_the_region(region_frame, surface):
    """Verified broken 2026-09-14: the output was bit-identical to the whole
    1920x1080 monitor resized, and matched the region not at all."""
    left, top, right, bottom = REGION
    expected = preprocessor_reference(surface[top:bottom, left:right])
    whole = preprocessor_reference(surface[
        np.ix_(
            np.arange(HEIGHT) * surface.shape[0] // HEIGHT,
            np.arange(WIDTH) * surface.shape[1] // WIDTH,
        )
    ])
    if np.array_equal(expected, whole):
        pytest.skip("region and resized monitor are indistinguishable on this screen")

    pre = native.GpuPreprocessor12(region_frame, WIDTH, HEIGHT)
    pre.process(region_frame, 1.0, 0.0, False)
    np.testing.assert_array_equal(pre.read_back(), expected)


def test_preprocessor12_downscaled_region_matches_decimation(region_frame, surface):
    """Nearest decimation of the *region*, with the region's own stride."""
    out_w, out_h = 64, 48
    left, top, right, bottom = REGION
    ys = top + np.arange(out_h) * (bottom - top) // out_h
    xs = left + np.arange(out_w) * (right - left) // out_w

    pre = native.GpuPreprocessor12(region_frame, out_w, out_h)
    pre.process(region_frame, 1.0, 0.0, False)
    np.testing.assert_array_equal(
        pre.read_back(), preprocessor_reference(surface[np.ix_(ys, xs)])
    )


def test_preprocessor12_matches_the_converter_on_a_region(region_frame):
    """The two nearest paths agree on full frames (test_gpu_converter.py); they
    must agree on region frames too, or one of them is wrong."""
    pre = native.GpuPreprocessor12(region_frame, 64, 48)
    pre.process(region_frame, 1.0, 0.0, False)
    converted = rapidshot.GpuConverter(
        region_frame, (64, 48), dtype="float32", sampling="nearest"
    ).process(region_frame).numpy()
    np.testing.assert_array_equal(pre.read_back(), converted)


def test_d3d11_preprocessor_converts_only_the_region(region_frame, surface):
    """The D3D11 `native.GpuPreprocessor` — verified broken 2026-09-14, output
    bit-identical to the whole monitor on this region camera. It reads on the
    capture device's own queue, so it needs no capture ordering."""
    left, top, right, bottom = REGION
    expected = preprocessor_reference(surface[top:bottom, left:right])

    pre = native.GpuPreprocessor(region_frame, WIDTH, HEIGHT)
    pre.process(region_frame, 1.0, 0.0, False)
    np.testing.assert_array_equal(pre.read_back(), expected)


def test_d3d11_and_d3d12_preprocessors_agree_on_a_region(region_frame):
    """Same shader, two APIs: on a region frame they must produce one tensor."""
    d3d11 = native.GpuPreprocessor(region_frame, 64, 48)
    d3d11.process(region_frame, 1.0, 0.0, False)
    d3d12 = native.GpuPreprocessor12(region_frame, 64, 48)
    d3d12.process(region_frame, 1.0, 0.0, False)
    np.testing.assert_array_equal(d3d11.read_back(), d3d12.read_back())


def test_raw_preprocessor12_without_crop_still_converts_the_whole_surface(region_frame, surface):
    """The extension works in texels and knows nothing of regions; translating
    one is the Python wrapper's job. A raw caller that omits `crop` keeps the
    old behaviour, which is what the benchmarks calling it directly rely on."""
    h, w = surface.shape[:2]
    impl = native.require().GpuPreprocessor12(native._texture_address(region_frame), w, h)
    impl.process(native._texture_address(region_frame), 1.0, 0.0, False,
                 source_id=region_frame.source_id)
    got = np.frombuffer(impl.read_back(), dtype=np.float32).reshape(1, 3, h, w)
    np.testing.assert_array_equal(got, preprocessor_reference(surface))


def test_raw_preprocessor12_refuses_a_crop_outside_the_surface(region_frame, surface):
    h, w = surface.shape[:2]
    impl = native.require().GpuPreprocessor12(native._texture_address(region_frame), 16, 16)
    with pytest.raises(RuntimeError, match="does not fit"):
        impl.process(native._texture_address(region_frame), 1.0, 0.0, False,
                     source_id=region_frame.source_id, crop=(w - 8, 0, 16, 16))


def test_crop_outside_the_region_is_refused_even_inside_the_texture(region_frame):
    """The texture has room, but the frame does not: frame coordinates win."""
    with pytest.raises(ValueError, match="crop"):
        converted(region_frame, (16, 16), crop=(0, 0, WIDTH + 1, 16))
