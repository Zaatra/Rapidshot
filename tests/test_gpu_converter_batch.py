"""Multi-ROI: several regions, one dispatch, one `(N, ...)` result (ROADMAP § 7.2).

**The reference is the single-region path.** Each slot of a batched call must
equal, bit for bit, what a batch-of-one converter produces for that region
alone. The single-region path is itself pinned against captured bytes by
``test_gpu_converter_crop.py``, so exact equality here carries that through —
and exact is the right bar, because a batched kernel doing the same
arithmetic per invocation has no excuse for a different number.

**What exact equality cannot see is slot order when regions look alike.** A
kernel writing every region into slot 0, or reading ``Regions[0]`` for all of
them, passes an equality test over identical crops. So the regions here are
chosen to differ, the test checks they do, and reversing them must reverse the
slots.

What is *not* verified: that it is one dispatch. That is a property of the
code (``Dispatch(x, y, count)``), not something a readback can observe.
"""

import numpy as np
import pytest

import rapidshot
from rapidshot import native

from test_gpu_converter_crop import busy_crop, texels

pytestmark = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

OUT = (64, 48)


@pytest.fixture(scope="module")
def live_frame():
    camera = rapidshot.create(output_color="BGRA")
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
def surface(live_frame):
    return rapidshot.GpuConverter(
        live_frame, (live_frame.width, live_frame.height),
        dtype="uint8", layout="nhwc", sampling="nearest",
    ).process(live_frame).numpy()[0]


@pytest.fixture(scope="module")
def regions(live_frame, surface):
    """Four regions of different sizes and places, checked to differ."""
    w, h = live_frame.width, live_frame.height
    chosen = [
        busy_crop(surface, 200, 150),
        (0, 0, 320, 240),
        (w - 401, h - 301, w - 1, h - 1),
        busy_crop(surface, 48, 36),
    ]
    probe = rapidshot.GpuConverter(
        live_frame, OUT, dtype="uint8", layout="nhwc", sampling="nearest"
    )
    images = [probe.process(live_frame, crop=r).numpy()[0].copy() for r in chosen]
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            if np.array_equal(images[i], images[j]):
                pytest.skip("two chosen regions look identical on this screen")
    return chosen


def single(frame, region, **kwargs):
    return rapidshot.GpuConverter(frame, OUT, **kwargs).process(
        frame, crop=region
    ).numpy()[0]


# --------------------------------------------------------------------------
# every slot is the single-region result
# --------------------------------------------------------------------------


FORMATS = [
    dict(dtype="float32"),
    dict(dtype="float16"),
    dict(dtype="uint8", layout="nhwc"),
    dict(dtype="float32", bgr=True, normalize=False),
]


@pytest.mark.parametrize("sampling", ["nearest", "bilinear"])
@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: "-".join(map(str, f.values())))
def test_each_slot_equals_that_region_alone(live_frame, regions, sampling, fmt):
    converter = rapidshot.GpuConverter(
        live_frame, OUT, sampling=sampling, batch=len(regions), **fmt
    )
    got = converter.process(live_frame, regions=regions).numpy()

    assert got.shape[0] == len(regions)
    for slot, region in enumerate(regions):
        np.testing.assert_array_equal(
            got[slot], single(live_frame, region, sampling=sampling, **fmt),
            err_msg=f"slot {slot}, region {region}",
        )


def test_identity_size_slots_are_the_captured_bytes(live_frame, surface):
    """Ground truth that does not route through another converter."""
    a = busy_crop(surface, 64, 48)
    b = (3, 5, 67, 53)
    got = rapidshot.GpuConverter(
        live_frame, (64, 48), dtype="uint8", layout="nhwc", sampling="nearest", batch=2
    ).process(live_frame, regions=[a, b]).numpy()
    np.testing.assert_array_equal(got[0], texels(surface, a))
    np.testing.assert_array_equal(got[1], texels(surface, b))


def test_reversing_regions_reverses_slots(live_frame, regions):
    converter = rapidshot.GpuConverter(
        live_frame, OUT, sampling="nearest", batch=len(regions)
    )
    forward = converter.process(live_frame, regions=regions).numpy().copy()
    backward = converter.process(live_frame, regions=regions[::-1]).numpy()
    np.testing.assert_array_equal(forward, backward[::-1])


def test_the_same_region_twice_fills_two_identical_slots(live_frame, regions):
    got = rapidshot.GpuConverter(
        live_frame, OUT, sampling="nearest", batch=2
    ).process(live_frame, regions=[regions[0], regions[0]]).numpy()
    np.testing.assert_array_equal(got[0], got[1])


# --------------------------------------------------------------------------
# a varying region count against a fixed buffer
# --------------------------------------------------------------------------


def test_fewer_regions_than_batch(live_frame, regions):
    converter = rapidshot.GpuConverter(
        live_frame, OUT, dtype="float16", sampling="nearest", batch=8
    )
    assert converter.batch == 8
    assert converter.output_byte_size == 8 * 3 * OUT[0] * OUT[1] * 2

    tensor = converter.process(live_frame, regions=regions[:3])
    assert tensor.shape == (3, 3, OUT[1], OUT[0])
    assert tensor.nbytes == 3 * 3 * OUT[0] * OUT[1] * 2
    got = tensor.numpy()
    assert got.shape == tensor.shape
    for slot in range(3):
        np.testing.assert_array_equal(
            got[slot], single(live_frame, regions[slot], dtype="float16", sampling="nearest")
        )


def test_a_smaller_call_after_a_larger_one_reports_only_its_own_slots(live_frame, regions):
    """Slot 1 still holds the earlier call's region. It must not appear."""
    converter = rapidshot.GpuConverter(live_frame, OUT, sampling="nearest", batch=4)
    converter.process(live_frame, regions=regions)
    got = converter.process(live_frame, regions=[regions[3]]).numpy()
    assert got.shape[0] == 1
    np.testing.assert_array_equal(got[0], single(live_frame, regions[3], sampling="nearest"))


def test_plain_process_on_a_batch_converter_is_one_slot(live_frame, regions):
    converter = rapidshot.GpuConverter(
        live_frame, OUT, sampling="nearest", batch=4, crop=regions[2]
    )
    got = converter.process(live_frame).numpy()
    assert got.shape[0] == 1
    np.testing.assert_array_equal(got[0], single(live_frame, regions[2], sampling="nearest"))


def test_a_refused_call_leaves_the_shape_alone(live_frame, regions):
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=4)
    converter.process(live_frame, regions=regions[:3])
    with pytest.raises(ValueError):
        converter.process(live_frame, regions=[regions[0], (0, 0, 0, 0)])
    assert converter.shape[0] == 3


def test_batch_crosses_adapters_byte_exact(live_frame, regions):
    """The transfer moves the whole buffer; the readback stops at N."""
    converter = rapidshot.GpuConverter(
        live_frame, OUT, dtype="float16", batch=len(regions)
    )
    try:
        transfer = rapidshot.TensorTransfer(converter)
    except RuntimeError as error:
        if "only one adapter" in str(error):
            pytest.skip("single-adapter machine; nothing to transfer to")
        raise
    produced = converter.process(live_frame, regions=regions).numpy()
    transfer.transfer()
    assert transfer.total_bytes == converter.output_byte_size
    np.testing.assert_array_equal(transfer.read_back_destination(), produced)


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


def test_more_regions_than_batch_is_refused(live_frame, regions):
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=2)
    with pytest.raises(ValueError, match="batch=2"):
        converter.process(live_frame, regions=regions[:3])


def test_empty_regions_is_refused(live_frame):
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=2)
    with pytest.raises(ValueError, match="empty"):
        converter.process(live_frame, regions=[])


def test_crop_and_regions_together_is_refused(live_frame, regions):
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=2)
    with pytest.raises(ValueError, match="not both"):
        converter.process(live_frame, crop=regions[0], regions=regions[:2])


def test_a_bad_region_names_its_index(live_frame, regions):
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=4)
    bad = (0, 0, live_frame.width + 1, 10)
    with pytest.raises(ValueError, match=r"regions\[2\]"):
        converter.process(live_frame, regions=[regions[0], regions[1], bad])


@pytest.mark.parametrize("batch", [0, -1, 1.5, True, "2"])
def test_bad_batch_is_refused(live_frame, batch):
    with pytest.raises((ValueError, TypeError)):
        rapidshot.GpuConverter(live_frame, OUT, batch=batch)


def test_yuv_refuses_a_batch(live_frame):
    with pytest.raises(ValueError, match="not a batch"):
        rapidshot.GpuConverter(live_frame, (64, 48), pixel_format="nv12", batch=2)


def test_native_layer_refuses_more_regions_than_slots(live_frame):
    """Beyond the Python check: the shader would read past the rect buffer."""
    converter = rapidshot.GpuConverter(live_frame, OUT, batch=2)
    with pytest.raises(RuntimeError, match="at most 2"):
        converter._impl.process(
            native._texture_address(live_frame), 1.0, 0.0, False,
            source_id=live_frame.source_id,
            regions=[(0, 0, 16, 16)] * 3,
        )


def test_output_past_32_bit_offsets_is_refused_before_allocating(live_frame):
    """Offsets past 4 GB wrap *inside* the buffer and overwrite earlier slots,
    which would never fault. 1000 x 640² x FP32 is ~4.9 GB."""
    with pytest.raises(RuntimeError, match="4 GB"):
        rapidshot.GpuConverter(live_frame, (640, 640), dtype="float32", batch=1000)
