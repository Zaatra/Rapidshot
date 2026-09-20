"""The README's code blocks, run against the real API.

CI already checks that every ```python block in the README *parses*. Parsing is
a weak claim: `tensor.output_resource_address` parses perfectly and does not
exist — it is a `GpuConverter` property, and the 2.6 README shipped that error
until this file caught it.

So these run the examples instead. They are deliberately written to mirror the
README's shape rather than to be good tests: if a block here needs changing, the
README needs the same change, and that is the point.

What this cannot check is whether the README describes the *published* package.
A source checkout has features PyPI may not have yet — which is exactly how a
README ends up documenting a wheel nobody can install. `test_release_safety.py`
covers the packaging side; the final answer is a fresh-environment install of
the published artifacts, which RELEASING.md makes a release step.

Live blocks skip on an idle desktop, for the reason `test_gpu_converter.py`
gives: a red suite that means "nothing moved on screen" trains people to ignore
red suites.
"""
import re
from pathlib import Path

import numpy as np
import pytest

import rapidshot
from rapidshot import native

README = Path(__file__).resolve().parents[1] / "README.md"

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _frame(camera, tries=600):
    for _ in range(tries):
        frame = camera.grab_frame()
        if frame is not None:
            return frame
    pytest.skip("no frame captured — the screen must be changing")


def _array(camera, tries=600, **kwargs):
    for _ in range(tries):
        frame = camera.grab(**kwargs)
        if frame is not None:
            return frame
    pytest.skip("no frame captured — the screen must be changing")


needs_native = pytest.mark.skipif(
    not native.is_available(), reason="native extension not built"
)

def _skip_if_cross_adapter(exc):
    """A hybrid laptop cannot export to CUDA without a transfer, by design.

    Capture runs on the integrated GPU and CUDA only sees the discrete one, so
    `to_cupy()` raises `CrossAdapterRequired` -- documented behaviour, not a
    failure. The README's GPU quick start therefore does not run as printed on
    an Optimus machine, which is why the README says so and points at
    `TensorTransfer`.
    """
    pytest.skip(f"hybrid GPU: {type(exc).__name__} -- capture and CUDA are on "
                "different adapters, see the README's hybrid section")


# -- the document itself ---------------------------------------------------

def test_every_python_block_parses():
    """The check CI runs, kept here so a local run catches it first."""
    import ast

    blocks = re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert blocks, "no python blocks found — has the README moved?"
    for index, block in enumerate(blocks, 1):
        try:
            ast.parse(block)
        except SyntaxError as exc:  # pragma: no cover - the failure message is the point
            pytest.fail(f"README python block {index} does not parse: {exc}")


def test_the_native_floor_is_stated_consistently():
    """The README, pyproject and the feature table must name one version.

    Three places tell a user which native version they need, and a reader who
    finds two different answers cannot know which is wrong.
    """
    text = README.read_text(encoding="utf-8")
    required = sorted(set(native._FEATURE_SINCE.values()))
    assert len(required) == 1, f"more than one floor in _FEATURE_SINCE: {required}"
    floor = required[0]

    assert f"rapidshot-native >= {floor}" in text or f"rapidshot-native>={floor}" in text, (
        f"README does not state the native floor {floor}"
    )

    pyproject = (README.parent / "pyproject.toml").read_text(encoding="utf-8")
    assert f'"rapidshot-native>={floor}' in pyproject, (
        f"pyproject does not floor the native extra at {floor}"
    )


def test_no_python_block_uses_a_name_the_package_does_not_export():
    """Catches the class of error that parsing cannot: a plausible attribute.

    Only checks `rapidshot.<Name>` at the top level, which is where the README
    points readers first, and where a rename would be silently wrong.
    """
    text = README.read_text(encoding="utf-8")
    referenced = set(re.findall(r"\brapidshot\.([A-Za-z_][A-Za-z0-9_]*)", text))
    missing = sorted(name for name in referenced if not hasattr(rapidshot, name))
    assert not missing, f"README references rapidshot.{{{', '.join(missing)}}}, which do not exist"


# -- quick start -----------------------------------------------------------

def test_quickstart_numpy_block():
    camera = rapidshot.create()
    try:
        frame = _array(camera)
        assert frame.shape[2] == 3
        frame.release()
    finally:
        camera.release()


def test_region_and_colour_block():
    camera = rapidshot.create(output_color="BGR")
    try:
        frame = _array(camera, region=(0, 0, 1920, 1080))
        assert frame.shape == (1080, 1920, 3)
        frame.release()
    finally:
        camera.release()


def test_continuous_capture_block():
    camera = rapidshot.create()
    try:
        camera.start(target_fps=60, video_mode=True)
        latest = None
        for _ in range(200):
            latest = camera.get_latest_frame()
            if latest is not None:
                break
        camera.stop()
        if latest is None:
            pytest.skip("no frame captured — the screen must be changing")
    finally:
        camera.release()


def test_pool_output_false_block():
    """The README says this hands back plain ndarrays."""
    camera = rapidshot.create(pool_output=False, pool_size_frames=4)
    try:
        frame = _array(camera)
        assert isinstance(frame, np.ndarray)
    finally:
        camera.release()


# -- GPU-resident ----------------------------------------------------------

def test_grab_frame_attributes_block():
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            assert frame.d3d11_texture is not None
            assert frame.width and frame.height
            assert frame.timestamp_qpc
            _ = frame.dirty_rects, frame.changed_fraction, frame.cursor
    finally:
        camera.release()


# -- model-ready tensors ---------------------------------------------------

@needs_native
def test_explicit_converter_block():
    """The README's `GpuConverter(frame, (640, 640), ...)` example.

    `frame` is positional and required: the constructor needs the source
    surface to size its resources. A sketch that omits it will not run, which
    is why this test exists rather than a prose note.
    """
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            converter = rapidshot.GpuConverter(
                frame, (640, 640), dtype="float16", layout="NCHW",
            )
            tensor = converter.process(frame)
            assert tensor.shape == (1, 3, 640, 640)
    finally:
        camera.release()


@needs_native
def test_layout_accepts_the_uppercase_spelling_the_readme_uses():
    """The README writes layout="NCHW"; the API lowercases it. If that stopped
    being true, every GPU example in the document would fail."""
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            upper = rapidshot.GpuConverter(frame, (64, 64), layout="NCHW").process(frame)
            lower = rapidshot.GpuConverter(frame, (64, 64), layout="nchw").process(frame)
            assert upper.shape == lower.shape
    finally:
        camera.release()


@needs_native
def test_multi_roi_block_returns_the_documented_shape():
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            converter = rapidshot.GpuConverter(frame, (224, 224), batch=4)
            tensor = converter.process(frame, regions=[
                (0, 0, 400, 300), (800, 40, 1000, 240),
            ])
            assert tensor.numpy().shape == (2, 3, 224, 224)
    finally:
        camera.release()


@needs_native
def test_payload_sizes_in_the_readme_table():
    """The README publishes three byte counts at 640². They come from here.

    An earlier draft quoted 1.23 MB for FP16, which is the FP32 figure divided
    by four -- FP16 is half of FP32, not a quarter. These assertions are why
    that cannot be reintroduced quietly.
    """
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            expected = {
                ("uint8", "nhwc"): 640 * 640 * 4,        # 1.64 MB, resized BGRA
                ("float16", "nchw"): 640 * 640 * 3 * 2,  # 2.46 MB
                ("float32", "nchw"): 640 * 640 * 3 * 4,  # 4.92 MB
            }
            for (dtype, layout), size in expected.items():
                converter = rapidshot.GpuConverter(
                    frame, (640, 640), dtype=dtype, layout=layout,
                )
                assert converter.output_byte_size == size, (
                    f"{dtype}/{layout}: README says {size / 1e6:.2f} MB, "
                    f"got {converter.output_byte_size / 1e6:.2f} MB"
                )
    finally:
        camera.release()


@needs_native
def test_converter_exposes_the_advanced_section_handles():
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            converter = rapidshot.GpuConverter(frame, (64, 64))
            # On the converter, not the tensor. The 2.6 README said otherwise.
            assert isinstance(converter.output_resource_address, int)
            assert isinstance(converter.output_byte_size, int)
    finally:
        camera.release()


@needs_native
def test_tensor_transfer_block_constructs_and_runs():
    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            converter = rapidshot.GpuConverter(frame, (640, 640), dtype="float16")
            transfer = rapidshot.TensorTransfer(converter)
            converter.process(frame)
            transfer.transfer()
    finally:
        camera.release()


# -- diagnostics -----------------------------------------------------------

def test_diagnose_block():
    assert isinstance(rapidshot.diagnose(), str)
    assert rapidshot.diagnose()
    assert isinstance(rapidshot.diagnose(probe_gpu=True), str)


def test_advanced_section_names_resolve():
    if not native.is_available():
        pytest.skip("native extension not built")
    for name in ("probe_d3d12_sharing", "probe_shareable_buffers",
                 "texture_sharing_info", "CrossAdapterTransfer"):
        assert hasattr(native, name), f"README names native.{name}, which is absent"


@needs_native
def test_tensor_stream_quickstart_block():
    """The README's primary GPU example, and the one most people will copy.

    Written to match the document exactly -- construct, iterate, call
    `to_torch()` -- because that is what a reader will run. The loop breaks
    after a couple of tensors rather than running forever, which is the only
    difference from the block as printed.
    """
    torch = pytest.importorskip("torch")

    camera = rapidshot.create()
    try:
        stream = rapidshot.TensorStream(
            camera, size=(640, 640), dtype="float16", layout="NCHW",
        )
        seen = 0
        for tensor in stream:
            try:
                x = tensor.to_torch()
            except rapidshot.CrossAdapterRequired as exc:
                _skip_if_cross_adapter(exc)
            assert tuple(x.shape) == (1, 3, 640, 640)
            assert x.dtype is torch.float16
            assert x.is_cuda
            seen += 1
            if seen >= 2:
                break
        if seen == 0:
            pytest.skip("no frame captured -- the screen must be changing")
    finally:
        camera.release()


@needs_native
def test_to_cupy_and_to_dlpack_alternatives_block():
    """`array = tensor.to_cupy()` and `capsule = tensor.to_dlpack()`."""
    pytest.importorskip("cupy")

    camera = rapidshot.create()
    try:
        with _frame(camera) as frame:
            converter = rapidshot.GpuConverter(frame, (640, 640), dtype="float16")
            tensor = converter.process(frame)

            try:
                array = tensor.to_cupy()
            except rapidshot.CrossAdapterRequired as exc:
                _skip_if_cross_adapter(exc)
            assert array.shape == (1, 3, 640, 640)

            capsule = tensor.to_dlpack()
            assert type(capsule).__name__ == "PyCapsule"
    finally:
        camera.release()
