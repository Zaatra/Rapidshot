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


def _camera(**kwargs):
    """`rapidshot.create()`, or a skip when this machine cannot capture.

    A CI runner has no desktop session, and on some it has no usable Direct3D
    device at all, so `create()` raises rather than returning a camera that
    yields nothing. Every live block here goes through this: without it the
    README tests are the only ones in the suite that turn "no display" into a
    red run, which is the failure mode `test_gpu_converter.py` warns about.
    """
    try:
        return rapidshot.create(**kwargs)
    except Exception as exc:                      # HeadlessError and friends
        if type(exc).__name__ in ("HeadlessError", "DeviceError", "OutputError"):
            pytest.skip(f"capture unavailable: {type(exc).__name__}: {exc}")
        raise


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
    """The README and pyproject must floor at the newest gated feature.

    Three places tell a user which native version they need. The feature
    table records when each symbol arrived, so it can name several versions;
    the floor everyone installs is the newest of them, and a README or extra
    naming an older one would install an extension missing a documented
    feature.
    """
    text = README.read_text(encoding="utf-8")
    floor = max([*native._FEATURE_SINCE.values(), *native._BEHAVIOUR_SINCE.values()],
                key=lambda v: tuple(int(part) for part in v.split(".")))

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
    camera = _camera()
    try:
        frame = _array(camera)
        assert frame.shape[2] == 3
        frame.release()
    finally:
        camera.release()


def test_region_and_colour_block():
    """The README shows `region=(0, 0, 1920, 1080)`.

    The size is taken from the camera rather than hardcoded: a CI runner's
    virtual display is smaller than 1080p, and a region outside the output is
    refused — correctly — which made this the one README test that failed on
    CI while passing on every developer machine. What the block demonstrates is
    that a region crops and that `output_color` picks the channel order, and
    both survive using a region this display actually has.
    """
    camera = _camera(output_color="BGR")
    try:
        width = min(1920, camera.width)
        height = min(1080, camera.height)
        frame = _array(camera, region=(0, 0, width, height))
        assert frame.shape == (height, width, 3)
        frame.release()
    finally:
        camera.release()


def test_continuous_capture_block():
    camera = _camera()
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


def test_freshness_contract_block():
    """`start(target_fps=240, policy="latest", max_age_ms=12)`, then
    `get_latest_frame()` and `stream_stats`, as the README prints them."""
    camera = _camera()
    try:
        camera.start(target_fps=240, policy="latest", max_age_ms=12)
        frame = None
        for _ in range(20):
            frame = camera.get_latest_frame()
            if frame is not None:
                break
        stats = camera.stream_stats
        camera.stop()
        if frame is None:
            pytest.skip("no frame captured — the screen must be changing")
        for key in ("stale_frames_replaced", "consumer_drops", "frames_expired",
                    "capture_drops", "age_ms_last", "age_ms_mean"):
            assert key in stats, f"the README names stream_stats[{key!r}]"
        assert stats["policy"] == "latest" and stats["max_age_ms"] == 12
        assert stats["frames_consumed"] >= 1
    finally:
        camera.release()


def test_policy_table_names_every_policy_and_reader():
    """The README's policy table and the code agree on names."""
    from rapidshot.capture import QUEUE_POLICIES, ScreenCapture

    text = README.read_text(encoding="utf-8")
    for policy in QUEUE_POLICIES:
        assert f'`"{policy}"`' in text, f"README's policy table omits {policy!r}"
    for reader in ("get_latest_frame", "get_next_frame"):
        assert hasattr(ScreenCapture, reader)
        assert f"`{reader}()`" in text


def test_pool_output_false_block():
    """The README says this hands back plain ndarrays."""
    camera = _camera(pool_output=False, pool_size_frames=4)
    try:
        frame = _array(camera)
        assert isinstance(frame, np.ndarray)
    finally:
        camera.release()


# -- GPU-resident ----------------------------------------------------------

def test_grab_frame_attributes_block():
    camera = _camera()
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
    camera = _camera()
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
    camera = _camera()
    try:
        with _frame(camera) as frame:
            upper = rapidshot.GpuConverter(frame, (64, 64), layout="NCHW").process(frame)
            lower = rapidshot.GpuConverter(frame, (64, 64), layout="nchw").process(frame)
            assert upper.shape == lower.shape
    finally:
        camera.release()


@needs_native
def test_multi_roi_block_returns_the_documented_shape():
    camera = _camera()
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
    camera = _camera()
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
    camera = _camera()
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
    camera = _camera()
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

    camera = _camera()
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

    camera = _camera()
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


# -- exact changed tiles (2.9) ---------------------------------------------

def _exact_camera():
    """The camera the README's exact-tiles block creates, verbatim."""
    return _camera(output_color="BGRA", nvidia_gpu=False)


def _tiled(camera, tries=4000, **kwargs):
    for _ in range(tries):
        frame = camera.grab_tiles(**kwargs)
        if frame is not None:
            return frame
    pytest.skip("no frame captured — the screen must be changing")


def test_exact_tiles_block_runs_as_printed():
    """`grab_tiles(tight_rectangles=True, audit_every=120)` + `materialize(rect)`.

    The README's one exact-change example, run as written. `materialize()` is
    documented as returning an owned, writable, C-contiguous array, and the
    rectangles it is fed come straight out of `changes.tight_rectangles` — so
    if the rectangle convention ever stops being half-open frame coordinates
    this fails here rather than in a reader's loop.
    """
    camera = _exact_camera()
    try:
        frame = _tiled(camera, tight_rectangles=True, audit_every=120)
        with frame:
            rects = frame.changes.tight_rectangles
            assert rects is not None, "tight_rectangles=True must populate the tuple"
            for rect in rects:
                patch = frame.materialize(rect)
                left, top, right, bottom = rect
                assert patch.shape == (bottom - top, right - left, 4)
                assert patch.flags["C_CONTIGUOUS"] and patch.flags["WRITEABLE"]
            assert len(list(frame.tiles)) == len(frame.changes.indices)
    finally:
        camera.release()


def test_exact_tiles_lease_raises_after_release():
    """The README promises `BufferReleasedError`, and the map staying readable."""
    camera = _exact_camera()
    try:
        frame = _tiled(camera, tight_rectangles=True)
        reason = frame.changes.reason
        frame.release()
        with pytest.raises(rapidshot.BufferReleasedError):
            frame.materialize()
        assert frame.changes.reason == reason, "the README says the map survives release"
    finally:
        camera.release()


def test_grab_with_changes_block_runs_as_printed():
    """The README's `grab_with_changes()` fields and `pixels.copy()`."""
    camera = _exact_camera()
    try:
        for _ in range(4000):
            frame = camera.grab_with_changes()
            if frame is not None:
                break
        else:
            pytest.skip("no frame captured — the screen must be changing")
        with frame:
            changes = frame.changes
            assert changes.reason in ("initial", "reset", "changed", "identical")
            assert changes.size == (frame.pixels.shape[1], frame.pixels.shape[0])
            assert len(changes.rectangles) == len(changes.indices)
            for left, top, right, bottom in changes.rectangles:
                assert frame.pixels[top:bottom, left:right].shape[2] == 4
            owned = frame.pixels.copy()
        assert owned.shape[2] == 4, "the README offers copy() as the owned route"
        with pytest.raises(rapidshot.BufferReleasedError):
            frame.pixels
    finally:
        camera.release()


def test_readme_names_the_change_types_the_package_exports():
    """Every name the exact-tiles section tells a reader to import or catch."""
    text = README.read_text(encoding="utf-8")
    for name in ("ChangedFrame", "TiledFrame", "ChangeSet", "ChangeAuditStats",
                 "BufferReleasedError", "PoolExhaustedError"):
        assert name in text, f"the README no longer mentions {name}"
        assert hasattr(rapidshot, name), f"README names {name}; rapidshot does not export it"
        assert name in rapidshot.__all__, f"{name} is not in rapidshot.__all__"


def test_change_audit_stats_block():
    """`camera.change_audit_stats()`, as the README prints it."""
    camera = _exact_camera()
    try:
        stats = camera.change_audit_stats()
        for field in ("audited_frames", "miss_frames", "missed_tiles"):
            assert hasattr(stats, field), f"the README names {field}"
            assert getattr(stats, field) >= 0
    finally:
        camera.release()


def test_the_exact_paths_refuse_what_the_readme_says_they_refuse():
    """Known limitations: DXGI, CPU, BGRA8, SDR, one-shot — and no live stream."""
    camera = _exact_camera()
    try:
        camera.start(video_mode=False)
        try:
            with pytest.raises(ValueError):
                camera.grab_with_changes()
            with pytest.raises(ValueError):
                camera.grab_tiles()
        finally:
            camera.stop()
    finally:
        camera.release()


def test_tile_size_matches_the_readme():
    """The README states 128x128 tiles as a fixed number, not a setting."""
    from rapidshot.changes import TILE_SIZE
    assert TILE_SIZE == 128
    assert "128×128" in README.read_text(encoding="utf-8")
