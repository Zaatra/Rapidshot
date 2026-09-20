"""Is an incremental capture engine worth building? (ROADMAP.md 6.3, 7.4)

Section 6.3 applied dirty rects to the *last* stage of the loop and got 1.5x,
against a projection of 12x. The reason is in ROADMAP.md:946 -- acquire, the
`CopySubresourceRegion` into the staging surface, and the map are untouched, so
Amdahl caps whatever the conversion does:

    AcquireNextFrame
      -> CopySubresourceRegion   full frame, every frame
      -> Map                     blocks until that copy finishes
      -> process()               dirty-only since 6.3
      -> out

At 2560x1600 that copy moves 16.4 MB per frame. At the 0.8% median dirty
fraction section 6.3 measured for an animated window, about 130 KB of it is new
information. This measures whether copying only the changed part is actually
cheaper, and what shape that copy should take.

Six questions, one per sub-benchmark:

    copy-scaling  Does the GPU->staging copy cost scale with area, or is it
                  fixed? Everything else here depends on the answer.
    tiles         Exact rects or a fixed tile grid? Quantising to tiles copies
                  more pixels but fewer times, and gives downstream a stable
                  index. Where is the knee?
    vram          For a consumer that ends in VRAM, what does the same
                  incremental update cost with no readback at all?
    live          What dirty-tile occupancy does a real desktop produce? The
                  design's load-bearing input, and section 6.3's own lesson is
                  to quote the distribution rather than a number.
    overreport    DWM reports what was *redrawn*, not what *changed*. Of the
                  pixels it marks dirty, how many actually differ? This decides
                  whether content hashing is worth anything.
    mask          Per-consumer change tracking: union of tile masks versus
                  union of rect lists.

Two methodology notes, both from mistakes this repo has already made:

  * `dirty_rect_pipeline.py` used a NumPy array as a proxy for the mapped
    staging surface, and ROADMAP.md:948 records what that cost -- a real mapped
    surface is uncached and far slower, so the component that shrinks least was
    modelled most optimistically, and the 12x figure was measuring the wrong
    thing. Everything here runs against a real D3D11 staging surface on the real
    capture device.
  * GPU waits are bimodal: when the GPU has already finished, the sample is ~0.
    ROADMAP.md:2985 records a probe that reported `wait = 0.0 us` by quoting
    minima. Every row here reports p50, with p90 beside it -- the first run of
    `copy-scaling` put a 50%-dirty row *slower* than a full frame, which did not
    reproduce across three repeats. It was interference landing in the median
    of a short run, and a visible p90 is what makes that legible.

    python benchmarks/incremental_capture.py copy-scaling
    python benchmarks/incremental_capture.py all
"""

import argparse
import ctypes
import statistics
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

import rapidshot  # noqa: E402
from rapidshot._libs.d3d11 import (  # noqa: E402
    D3D11_BOX,
    D3D11_CPU_ACCESS_READ,
    D3D11_TEXTURE2D_DESC,
    D3D11_USAGE_DEFAULT,
    D3D11_USAGE_STAGING,
    DXGI_FORMAT_B8G8R8A8_UNORM,
    ID3D11Texture2D,
)
from rapidshot._libs.dxgi import DXGI_MAPPED_RECT, IDXGISurface  # noqa: E402

REPS = 80
WARMUP = 5
#: Sleep between reps. A capture loop is not back-to-back, and section 2's
#: duty-cycle note applies to GPU work as much as to the kernels.
PACE_S = 0.004


class D3D11_SUBRESOURCE_DATA(ctypes.Structure):
    """Not in `_libs/d3d11.py` because the library never seeds a texture."""
    _fields_ = [
        ("pSysMem", ctypes.c_void_p),
        ("SysMemPitch", ctypes.c_uint32),
        ("SysMemSlicePitch", ctypes.c_uint32),
    ]


def make_texture(device, width, height, *, usage, cpu_access=0, data=None):
    """A BGRA 2D texture, optionally seeded from a host array."""
    desc = D3D11_TEXTURE2D_DESC()
    desc.Width, desc.Height = width, height
    desc.MipLevels = desc.ArraySize = 1
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM
    desc.SampleDesc.Count, desc.SampleDesc.Quality = 1, 0
    desc.Usage = usage
    desc.CPUAccessFlags = cpu_access
    desc.BindFlags = desc.MiscFlags = 0

    initial = None
    seed = None
    if data is not None:
        seed = np.ascontiguousarray(data)
        sub = D3D11_SUBRESOURCE_DATA(seed.ctypes.data, seed.strides[0], 0)
        # Keep `sub` alive across the call; ctypes does not own what byref sees.
        initial = ctypes.cast(ctypes.byref(sub), ctypes.c_void_p)

    texture = ctypes.POINTER(ID3D11Texture2D)()
    device.CreateTexture2D(ctypes.byref(desc), initial, ctypes.byref(texture))
    del seed
    return texture


class Staging:
    """A staging surface owned by this benchmark rather than by the camera.

    Separate from `core.stagesurf.StageSurface` on purpose: these experiments
    need two of them, need to map without the camera's live-frame guard, and
    must not disturb the state the camera is holding.
    """

    def __init__(self, device, width, height):
        self.width, self.height = width, height
        self.texture = make_texture(device, width, height,
                                    usage=D3D11_USAGE_STAGING,
                                    cpu_access=D3D11_CPU_ACCESS_READ)
        self.surface = self.texture.QueryInterface(IDXGISurface)

    def map(self) -> DXGI_MAPPED_RECT:
        rect = DXGI_MAPPED_RECT()
        self.surface.Map(ctypes.byref(rect), 1)
        return rect

    def unmap(self) -> None:
        self.surface.Unmap()

    def view(self, rect: DXGI_MAPPED_RECT) -> np.ndarray:
        """The mapped surface as (height, pitch) uint8. Uncached memory."""
        buffer = (ctypes.c_ubyte * (rect.Pitch * self.height)).from_address(rect.pBits)
        return np.frombuffer(buffer, dtype=np.uint8).reshape(self.height, rect.Pitch)

    def release(self) -> None:
        self.surface = None
        self.texture = None


def copy_rects(context, destination, source, rects) -> None:
    """Copy each rect from `source` into the same position of `destination`."""
    box = D3D11_BOX()
    box.front, box.back = 0, 1
    for left, top, right, bottom in rects:
        box.left, box.top, box.right, box.bottom = left, top, right, bottom
        context.CopySubresourceRegion(destination, 0, left, top, 0,
                                      source, 0, ctypes.byref(box))


# --------------------------------------------------------------------------
# rect shapes
# --------------------------------------------------------------------------

def band_rects(count, fraction, width, height):
    """`count` full-width bands covering `fraction` of the frame.

    The friendly shape: whole rows, so the staging read stays contiguous.
    """
    rows = max(count, int(round(height * fraction)))
    per = max(1, rows // count)
    stride = height // count
    rects = []
    for i in range(count):
        top = min(i * stride, height - per)
        rects.append((0, top, width, top + per))
    return rects


def grid_rects(count, fraction, width, height):
    """`count` square-ish rects spread over the frame."""
    side = max(1, int(round(((width * height * fraction) / count) ** 0.5)))
    side = min(side, width, height)
    columns = max(1, int(round(count ** 0.5)))
    rows = (count + columns - 1) // columns
    step_x = max(side, width // columns)
    step_y = max(side, height // rows)
    rects = []
    for i in range(count):
        x = min((i % columns) * step_x, width - side)
        y = min((i // columns) * step_y, height - side)
        rects.append((x, y, x + side, y + side))
    return rects


def column_rects(count, fraction, width, height):
    """`count` tall narrow rects -- 6.3's worst shape for a row-limited read."""
    columns = max(1, int(round(width * fraction / count)))
    stride = width // count
    rects = []
    for i in range(count):
        left = min(i * stride, width - columns)
        rects.append((left, 0, left + columns, height))
    return rects


SHAPES = {"bands": band_rects, "grid": grid_rects, "columns": column_rects}


def area_of(rects):
    return sum((r - l) * (b - t) for l, t, r, b in rects)


def rows_touched(rects, height):
    touched = np.zeros(height, dtype=bool)
    for _, top, _, bottom in rects:
        touched[top:bottom] = True
    return int(touched.sum())


# --------------------------------------------------------------------------
# tiles
# --------------------------------------------------------------------------

def tile_grid(width, height, tile):
    return (width + tile - 1) // tile, (height + tile - 1) // tile


def tiles_for(rects, tile, width, height):
    """The set of tile indices any rect touches, as a boolean grid."""
    across, down = tile_grid(width, height, tile)
    mask = np.zeros((down, across), dtype=bool)
    for left, top, right, bottom in rects:
        mask[top // tile:(bottom - 1) // tile + 1,
             left // tile:(right - 1) // tile + 1] = True
    return mask


def tile_runs(mask, tile, width, height):
    """Coalesce each tile row's dirty tiles into runs, then into pixel rects.

    A run is one `CopySubresourceRegion` instead of one per tile, which is the
    whole reason a fixed grid can beat exact rects on call count.
    """
    rects = []
    down, across = mask.shape
    for ty in range(down):
        row = mask[ty]
        tx = 0
        while tx < across:
            if not row[tx]:
                tx += 1
                continue
            start = tx
            while tx < across and row[tx]:
                tx += 1
            rects.append((start * tile, ty * tile,
                          min(tx * tile, width), min((ty + 1) * tile, height)))
    return rects


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------

def summarise(samples):
    """p50 and p90, in ms.

    p50 because section 10's rule for a GPU wait is to read the median -- a
    minimum reports ~0 whenever the GPU happened to have finished already. p90
    because the first run of this benchmark produced a 50%-dirty row *slower*
    than a full frame, which did not reproduce: it was desktop interference
    landing in the median of a short run. A visible p90 makes that obvious
    instead of leaving it to look like a property of the copy.
    """
    ordered = sorted(samples)
    p90 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.9))]
    return statistics.median(ordered) * 1000.0, p90 * 1000.0


def measure_copy(harness, rects, *, read=True):
    """One submit/map/read cycle per rep, against the real staging surface.

    Returns (submit, map, read) samples in seconds. The map is where the copy's
    real cost lands: `CopySubresourceRegion` returns once submitted, and
    ROADMAP.md:2996 shows the map collapsing 24x given 2 ms of slack, which
    identifies it as a wait on the GPU rather than work on the CPU.
    """
    context, stage, source = harness.context, harness.stage, harness.source_a
    submits, maps, reads = [], [], []

    for rep in range(WARMUP + REPS):
        start = time.perf_counter()
        copy_rects(context, stage.texture, source, rects)
        submitted = time.perf_counter()
        mapped = stage.map()
        available = time.perf_counter()
        if read:
            view = stage.view(mapped)
            for left, top, right, bottom in rects:
                # Whole rows: ROADMAP.md:954 measured column reads as no better.
                np.copyto(harness.sink[top:bottom, :mapped.Pitch], view[top:bottom])
        finished = time.perf_counter()
        stage.unmap()

        if rep >= WARMUP:
            submits.append(submitted - start)
            maps.append(available - submitted)
            reads.append(finished - available)
        time.sleep(PACE_S)

    return submits, maps, reads


class Harness:
    """A real capture device, two seeded VRAM textures, and a staging surface.

    The source textures are seeded from live desktop frames so the content is
    representative, but every experiment then runs against those fixed textures
    rather than against the desktop. ROADMAP.md:958 is explicit about why:
    timing a live `grab()` measures what happened to change on screen at that
    instant, and gave 2.26x, 1.56x and 0.87x for the same comparison on
    consecutive runs.
    """

    def __init__(self, want_second=False):
        self.camera = rapidshot.create(output_color="BGRA", pool_output=False)
        self.width, self.height = self.camera.width, self.camera.height

        first = self._grab_settled()
        self.source_a = make_texture(self.device, self.width, self.height,
                                     usage=D3D11_USAGE_DEFAULT, data=first)
        self.frame_a = first

        self.frame_b = None
        self.source_b = None
        if want_second:
            # Every pixel differs, deterministically. A second *live* frame
            # would differ only where the desktop happened to change, which
            # makes a persistence test that passes prove nothing -- the pixels
            # it checked were identical in both frames anyway.
            second = np.ascontiguousarray(~first)
            self.source_b = make_texture(self.device, self.width, self.height,
                                         usage=D3D11_USAGE_DEFAULT, data=second)
            self.frame_b = second

        self.stage = Staging(self.device, self.width, self.height)
        self.stage_b = None
        # Host destination for the read phase, sized to the widest pitch a
        # driver might hand back.
        self.sink = np.empty((self.height, self.width * 4 + 4096), np.uint8)

    @property
    def device(self):
        return self.camera._device.device

    @property
    def context(self):
        return self.camera._device.im_context

    def _grab_settled(self, different_from=None, tries=200):
        """A frame, and if asked, one whose pixels actually differ."""
        for _ in range(tries):
            frame = self.camera.grab()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = np.ascontiguousarray(frame)
            if different_from is None or not np.array_equal(frame, different_from):
                return frame
            time.sleep(0.01)
        raise RuntimeError(
            "no differing frame after {} tries -- is anything moving on screen? "
            "Run benchmarks/motion_source.py in another window.".format(tries))

    def second_stage(self):
        if self.stage_b is None:
            self.stage_b = Staging(self.device, self.width, self.height)
        return self.stage_b

    def release(self):
        for attribute in ("stage", "stage_b"):
            surface = getattr(self, attribute, None)
            if surface is not None:
                surface.release()
        self.source_a = self.source_b = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def header(title, subtitle=""):
    print()
    print(title)
    print("=" * len(title))
    if subtitle:
        print(subtitle)
    print()


# --------------------------------------------------------------------------
# 0. correctness gate
# --------------------------------------------------------------------------

def bench_persistence(harness) -> bool:
    """Does a staging surface keep undirtied pixels across partial copies?

    Everything else here assumes it does. If a partial `CopySubresourceRegion`
    into a staging texture left the untouched pixels undefined -- or if `Map`
    discarded them -- an incremental engine would produce silently wrong
    frames, which section 11 rates as worthless however fast it is.

    Source B is the bitwise complement of source A, so every pixel differs and
    a stale pixel cannot pass by coincidence.
    """
    header("0. Staging persistence",
           "Partial copies must leave the rest of the surface intact.")

    stage, context = harness.stage, harness.context
    width, height = harness.width, harness.height
    patch = (400, 300, 1200, 900)

    copy_rects(context, stage.texture, harness.source_a, [(0, 0, width, height)])
    copy_rects(context, stage.texture, harness.source_b, [patch])

    mapped = stage.map()
    try:
        view = stage.view(mapped)[:, :width * 4].reshape(height, width, 4)
        left, top, right, bottom = patch
        inside = np.array_equal(view[top:bottom, left:right],
                                harness.frame_b[top:bottom, left:right])
        above = np.array_equal(view[:top], harness.frame_a[:top])
        below = np.array_equal(view[bottom:], harness.frame_a[bottom:])
        beside = np.array_equal(view[top:bottom, :left],
                                harness.frame_a[top:bottom, :left])
        after_remap = None
        stage.unmap()
        # Map/unmap must not disturb it either: the accumulator is read every
        # frame and written only where something changed.
        remapped = stage.map()
        view = stage.view(remapped)[:, :width * 4].reshape(height, width, 4)
        after_remap = np.array_equal(view[:top], harness.frame_a[:top])
    finally:
        stage.unmap()

    for label, passed in (("patched region holds the new pixels", inside),
                          ("region above the patch is preserved", above),
                          ("region below the patch is preserved", below),
                          ("region beside the patch is preserved", beside),
                          ("preserved across a map/unmap cycle", after_remap)):
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    ok = all((inside, above, below, beside, after_remap))
    print()
    print("  A persistent staging surface is a valid accumulator." if ok else
          "  A persistent staging surface is NOT safe. Everything below is moot.")
    return ok


# --------------------------------------------------------------------------
# 1. does the copy scale with area?
# --------------------------------------------------------------------------

FRACTIONS = (0.008, 0.03, 0.10, 0.25, 0.50, 0.75, 1.00)


def bench_copy_scaling(harness) -> None:
    """The premise: is GPU->staging copy cost proportional to area, or fixed?

    If it is fixed, ideas built on copying less are dead on arrival and only
    the per-consumer change tracking survives. If it scales, the 16.4 MB the
    loop moves every frame is the largest remaining target in the capture path.
    """
    header("1. Does the staging copy scale with copied area?",
           f"{harness.width}x{harness.height} BGRA, "
           f"{harness.width * harness.height * 4 / 1e6:.1f} MB full frame. "
           "p50 (p90) in ms.")

    full = [(0, 0, harness.width, harness.height)]
    submits, maps, reads = measure_copy(harness, full)
    base_submit, base_submit_p90 = summarise(submits)
    base_map, base_map_p90 = summarise(maps)
    base_read, base_read_p90 = summarise(reads)
    base_total = base_submit + base_map + base_read

    print(f"  Full-frame baseline: submit {base_submit:.3f} ({base_submit_p90:.3f})  "
          f"map {base_map:.3f} ({base_map_p90:.3f})  "
          f"read {base_read:.3f} ({base_read_p90:.3f})  "
          f"total {base_total:.3f} ms")
    print()
    print(f"  {'dirty':>7} {'rects':>6} {'MB':>7} {'rows':>6} "
          f"{'submit':>14} {'map':>16} {'read':>15} {'total':>8} {'vs full':>8}")
    print("  " + "-" * 92)

    rows = []
    for fraction in FRACTIONS:
        for count in (1, 8, 64):
            if fraction == 1.00 and count != 1:
                continue
            rects = band_rects(count, fraction, harness.width, harness.height)
            submits, maps, reads = measure_copy(harness, rects)
            submit, submit_p90 = summarise(submits)
            map_ms, map_p90 = summarise(maps)
            read_ms, read_p90 = summarise(reads)
            total = submit + map_ms + read_ms
            megabytes = area_of(rects) * 4 / 1e6
            touched = rows_touched(rects, harness.height)
            print(f"  {fraction * 100:6.1f}% {count:6d} {megabytes:7.2f} "
                  f"{touched * 100 / harness.height:5.1f}% "
                  f"{submit:8.3f} ({submit_p90:5.3f}) "
                  f"{map_ms:8.3f} ({map_p90:5.3f}) "
                  f"{read_ms:7.3f} ({read_p90:5.3f}) "
                  f"{total:7.3f} {base_total / total:7.2f}x")
            rows.append((fraction, count, megabytes, submit, map_ms, read_ms, total))

    print()
    print("  Shape at 10% dirty, 8 rects -- does rect geometry matter?")
    print(f"    {'shape':>8} {'MB':>7} {'rows':>6} {'submit':>8} {'map':>8} "
          f"{'read':>8} {'total':>8}")
    for name, builder in SHAPES.items():
        rects = builder(8, 0.10, harness.width, harness.height)
        submits, maps, reads = measure_copy(harness, rects)
        submit = summarise(submits)[0]
        map_ms = summarise(maps)[0]
        read_ms = summarise(reads)[0]
        print(f"    {name:>8} {area_of(rects) * 4 / 1e6:7.2f} "
              f"{rows_touched(rects, harness.height) * 100 / harness.height:5.1f}% "
              f"{submit:8.3f} {map_ms:8.3f} {read_ms:8.3f} "
              f"{submit + map_ms + read_ms:8.3f}")

    # Re-measure the baseline last. A live desktop competes for the same GPU,
    # and a sweep that takes a minute can drift underneath itself: an early run
    # of this benchmark reported a 50%-dirty row slower than a full frame purely
    # because the machine got busier part-way through. Quoting the drift means
    # a reader can tell a real effect from that.
    submits, maps, reads = measure_copy(harness, full)
    again = summarise(submits)[0] + summarise(maps)[0] + summarise(reads)[0]
    drift = again / base_total
    print()
    print(f"  Baseline re-measured after the sweep: {again:.3f} ms vs "
          f"{base_total:.3f} ms before ({drift:.2f}x).")
    if drift > 1.25 or drift < 0.8:
        print("  The machine drifted by more than 25% during the sweep. Treat "
              "the high-dirty rows")
        print("  as unreliable and repeat the run -- section 3's rule is that "
              "one recording is not")
        print("  a measurement.")

    small = [r for r in rows if r[0] == 0.008 and r[1] == 1][0]
    print()
    print(f"  At the 0.8% median section 6.3 measured for an animated window: "
          f"{base_total / small[6]:.1f}x the full-frame path, "
          f"{small[2]:.2f} MB instead of "
          f"{harness.width * harness.height * 4 / 1e6:.1f} MB.")


# --------------------------------------------------------------------------
# 2. exact rects, or a fixed tile grid?
# --------------------------------------------------------------------------

TILE_SIZES = (32, 64, 128, 256, 512)


def bench_tiles(harness) -> None:
    """Quantising to a fixed grid copies more pixels but fewer times.

    The case for a grid is not raw speed. It is that tile count is bounded (a
    2560x1600 frame is 260 tiles at 128px, whatever the desktop does), tile
    positions never move so downstream can cache per-tile state by index, and a
    run of tiles in one row coalesces into a single copy. The case against is
    inflation: a one-pixel caret dirties a whole tile.

    This measures what that inflation actually costs.
    """
    header("2. Exact rects versus a fixed tile grid",
           "Inflation is the price; bounded call count and a stable index are "
           "what it buys.")

    width, height = harness.width, harness.height
    print(f"  {'dirty':>7} {'exact':>24} "
          f"{'tile':>5} {'tiles':>9} {'copies':>7} {'MB':>6} {'infl':>6} "
          f"{'total':>8} {'vs exact':>9}")
    print("  " + "-" * 98)

    for fraction, count in ((0.008, 8), (0.03, 16), (0.10, 16), (0.30, 32)):
        rects = grid_rects(count, fraction, width, height)
        submits, maps, reads = measure_copy(harness, rects)
        exact_total = (summarise(submits)[0] + summarise(maps)[0]
                       + summarise(reads)[0])
        exact_mb = area_of(rects) * 4 / 1e6
        exact_label = f"{len(rects)}r {exact_mb:5.2f}MB {exact_total:6.3f}ms"

        for tile in TILE_SIZES:
            mask = tiles_for(rects, tile, width, height)
            runs = tile_runs(mask, tile, width, height)
            submits, maps, reads = measure_copy(harness, runs)
            total = (summarise(submits)[0] + summarise(maps)[0]
                     + summarise(reads)[0])
            megabytes = area_of(runs) * 4 / 1e6
            across, down = tile_grid(width, height, tile)
            occupancy = f"{int(mask.sum())}/{across * down}"
            print(f"  {fraction * 100:6.1f}% {exact_label:>24} "
                  f"{tile:5d} {occupancy:>9} "
                  f"{len(runs):7d} {megabytes:6.2f} "
                  f"{megabytes / max(exact_mb, 1e-9):5.2f}x "
                  f"{total:7.3f} {exact_total / total:8.2f}x")
        print()

    print("  Grid size is a fixed cost per frame, whatever is on screen:")
    for tile in TILE_SIZES:
        across, down = tile_grid(width, height, tile)
        total_tiles = across * down
        print(f"    {tile:3d}px  {across:3d}x{down:<3d} = {total_tiles:5d} tiles  "
              f"mask {((total_tiles + 63) // 64) * 8:4d} bytes  "
              f"tile payload {tile * tile * 4 / 1024:7.1f} KB")


# --------------------------------------------------------------------------
# 3. the same update with no readback at all
# --------------------------------------------------------------------------

def bench_vram(harness) -> None:
    """A consumer that ends in VRAM never needs the staging surface.

    NVENC, a CUDA model, and this repo's own `GpuConverter` all consume a
    texture. For them the incremental update is texture->texture inside VRAM,
    and the readback -- which is the entire PCIe cost -- does not happen.

    Timing GPU-only work needs a sync point. Rather than a fence, each rep
    issues the copies BATCH times and then maps a 64x64 staging probe, which
    cannot complete until the queue ahead of it has drained. The probe's own
    cost is measured with an empty rect list and subtracted.
    """
    header("3. Incremental update with no readback (VRAM-resident)",
           "texture->texture inside VRAM, per update, in ms.")

    batch = 16
    width, height = harness.width, harness.height
    context = harness.context
    destination = make_texture(harness.device, width, height,
                               usage=D3D11_USAGE_DEFAULT)
    probe = Staging(harness.device, 64, 64)
    probe_box = D3D11_BOX()
    probe_box.left, probe_box.top, probe_box.front = 0, 0, 0
    probe_box.right, probe_box.bottom, probe_box.back = 64, 64, 1

    def run(rects):
        samples = []
        for rep in range(WARMUP + REPS):
            start = time.perf_counter()
            for _ in range(batch):
                copy_rects(context, destination, harness.source_a, rects)
            context.CopySubresourceRegion(probe.texture, 0, 0, 0, 0,
                                          harness.source_a, 0,
                                          ctypes.byref(probe_box))
            probe.map()
            probe.unmap()
            if rep >= WARMUP:
                samples.append(time.perf_counter() - start)
            time.sleep(PACE_S)
        return samples

    floor_p50 = summarise(run([]))[0]
    full = [(0, 0, width, height)]
    full_each = (summarise(run(full))[0] - floor_p50) / batch
    staging_full = summarise(measure_copy(harness, full)[1])[0]

    print(f"  sync-probe floor {floor_p50:.3f} ms per rep, subtracted below")
    print(f"  full-frame VRAM copy {full_each:.4f} ms "
          f"({width * height * 4 / 1e6:.1f} MB)")
    print(f"  full-frame staging map {staging_full:.3f} ms "
          f"(what the loop pays today)")
    print()
    print(f"  {'dirty':>7} {'MB':>7} {'per update':>12} {'vs full VRAM':>13} "
          f"{'vs staging map':>15}")
    print("  " + "-" * 60)

    for fraction in (0.008, 0.03, 0.10, 0.30, 1.00):
        rects = band_rects(1, fraction, width, height)
        each = max((summarise(run(rects))[0] - floor_p50) / batch, 1e-6)
        megabytes = area_of(rects) * 4 / 1e6
        print(f"  {fraction * 100:6.1f}% {megabytes:7.2f} "
              f"{each:9.4f} ms {full_each / each:12.2f}x "
              f"{staging_full / each:14.1f}x")

    probe.release()
    print()
    print("  The last column is against the full-frame staging map the loop "
          "pays today,")
    print("  which is the comparison a GPU-terminating consumer actually faces.")


# --------------------------------------------------------------------------
# 4. what does a real desktop dirty?
# --------------------------------------------------------------------------

def percentiles(values, points=(50, 90, 99)):
    if not values:
        return {p: float("nan") for p in points}
    ordered = sorted(values)
    return {p: ordered[min(len(ordered) - 1, int(len(ordered) * p / 100))]
            for p in points}


def bench_live(harness, seconds=20.0) -> None:
    """Dirty-tile occupancy on the real desktop, as a distribution.

    Section 6.3's own lesson, at ROADMAP.md:952: the 0.7-0.8% headline was a
    small animated window on an otherwise still desktop, and a dragged window
    measured median 0.68 -- about 85x that. A single number here would mislead
    whoever reads it next, so this reports the distribution, and the caller is
    expected to say what was on screen while it ran.
    """
    header(f"4. Live dirty-tile occupancy ({seconds:.0f} s of real capture)",
           "Whatever is happening on this desktop right now. Run "
           "benchmarks/motion_source.py alongside for a defined workload.")

    camera = harness.camera
    width, height = harness.width, harness.height
    frame_pixels = width * height
    tiles_seen = {tile: [] for tile in TILE_SIZES}
    areas, counts = [], []
    coalesced = moved = frames = empty = 0

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pixels = camera.grab()
        if pixels is None:
            continue
        rects = camera._duplicator.dirty_rects
        if rects is None:
            continue
        if not rects:
            empty += 1
            continue
        frames += 1
        counts.append(len(rects))
        areas.append(area_of(rects) / frame_pixels)
        coalesced += bool(camera._duplicator.rects_coalesced)
        moved += bool(camera._duplicator.move_rects)
        for tile in TILE_SIZES:
            mask = tiles_for(rects, tile, width, height)
            tiles_seen[tile].append(mask.sum() / mask.size)

    if not frames:
        print("  No frames carried dirty rects. Is anything moving on screen?")
        return

    print(f"  {frames} frames with dirty rects, {empty} with none, "
          f"{frames / seconds:.1f} fps")
    print(f"  {coalesced} frames reported coalesced rects, "
          f"{moved} carried move rects")
    print()
    area_p = percentiles(areas)
    count_p = percentiles(counts)
    print(f"  dirty area    p50 {area_p[50] * 100:6.2f}%  "
          f"p90 {area_p[90] * 100:6.2f}%  p99 {area_p[99] * 100:6.2f}%")
    print(f"  rects/frame   p50 {count_p[50]:6.0f}   "
          f"p90 {count_p[90]:6.0f}   p99 {count_p[99]:6.0f}")
    print()
    print(f"  {'tile':>7} {'grid':>10} {'occupancy p50':>14} {'p90':>8} "
          f"{'p99':>8} {'inflation p50':>14}")
    print("  " + "-" * 67)
    for tile in TILE_SIZES:
        occupancy = percentiles(tiles_seen[tile])
        across, down = tile_grid(width, height, tile)
        inflation = (occupancy[50] / area_p[50]) if area_p[50] > 0 else float("nan")
        print(f"  {tile:5d}px {across:4d}x{down:<4d} "
              f"{occupancy[50] * 100:13.2f}% {occupancy[90] * 100:7.2f}% "
              f"{occupancy[99] * 100:7.2f}% {inflation:13.2f}x")

    print()
    print("  Occupancy is the fraction of the frame a tile-grid engine copies;")
    print("  inflation is how much more than the exact dirty area that is.")


# --------------------------------------------------------------------------
# 5. is DWM telling the truth about what changed?
# --------------------------------------------------------------------------

def changed_tiles(difference, tile):
    """Reduce a per-pixel difference mask to per-tile booleans.

    `reduceat` rather than a reshape because 1600 is not a multiple of 128:
    the bottom row of tiles is short, and a reshape would either refuse or
    silently drop it.
    """
    rows = np.add.reduceat(difference, np.arange(0, difference.shape[0], tile), axis=0)
    both = np.add.reduceat(rows, np.arange(0, difference.shape[1], tile), axis=1)
    return both > 0


def bench_overreport(harness, seconds=15.0, tile=128) -> None:
    """DWM reports what was *redrawn*. How much of that actually changed?

    A caret blink repaints a whole text control; a HUD repaints identically
    every frame. If DWM's rects are loose, hashing tile content and suppressing
    the unchanged ones is worth building. If they are already tight, it is not,
    and the idea dies here rather than after someone writes a compute shader.

    Note what this cannot see: a tile DWM did *not* mark dirty is never
    compared, so this measures over-reporting only. Under-reporting would be a
    correctness bug in DDA, not an optimisation opportunity.
    """
    header(f"5. Does DWM over-report? ({seconds:.0f} s, {tile}px tiles)",
           "Of the pixels and tiles DWM marks dirty, how many actually differ "
           "from the previous frame?")

    camera = harness.camera
    width, height = harness.width, harness.height
    previous = None
    tile_ratios, pixel_ratios = [], []
    frames = wholly_false = 0
    hash_samples = []
    # The decisive pair: what DWM claims, against what a perfect detector would
    # find. The second is the floor for *any* change-detection scheme, so it
    # bounds the whole idea rather than just this implementation of it.
    claimed_area, true_area = [], []
    claimed_occupancy = {size: [] for size in TILE_SIZES}
    true_occupancy = {size: [] for size in TILE_SIZES}

    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pixels = camera.grab()
        if pixels is None:
            continue
        rects = camera._duplicator.dirty_rects
        current = np.ascontiguousarray(pixels)
        if previous is None or not rects:
            previous = current
            continue

        difference = np.any(current != previous, axis=2)
        claimed = tiles_for(rects, tile, width, height)
        actual = changed_tiles(difference, tile)
        actually_dirty = claimed & actual

        claimed_count = int(claimed.sum())
        if claimed_count:
            frames += 1
            tile_ratios.append(int(actually_dirty.sum()) / claimed_count)

        claimed_pixels = 0
        changed_pixels = 0
        for left, top, right, bottom in rects:
            claimed_pixels += (right - left) * (bottom - top)
            changed_pixels += int(difference[top:bottom, left:right].sum())
        if claimed_pixels:
            pixel_ratios.append(changed_pixels / claimed_pixels)
            if changed_pixels == 0:
                wholly_false += 1

        claimed_area.append(claimed_pixels / (width * height))
        true_area.append(float(difference.mean()))
        for size in TILE_SIZES:
            claim = tiles_for(rects, size, width, height)
            truth = changed_tiles(difference, size)
            claimed_occupancy[size].append(claim.sum() / claim.size)
            true_occupancy[size].append((claim & truth).sum() / truth.size)

        # What a per-tile digest costs on the CPU, as an upper bound on the
        # detection side. A GPU shader would not pay the readback this does.
        start = time.perf_counter()
        changed_tiles(difference, tile)
        hash_samples.append(time.perf_counter() - start)

        previous = current

    if not frames:
        print("  No frames carried dirty rects. Is anything moving on screen?")
        return

    tile_p = percentiles(tile_ratios)
    pixel_p = percentiles(pixel_ratios, (1, 50, 90))
    print(f"  {frames} frame pairs compared")
    print()
    print(f"  Of the TILES DWM marked dirty, the fraction that really changed:")
    print(f"    p50 {tile_p[50] * 100:6.2f}%   p90 {tile_p[90] * 100:6.2f}%   "
          f"p99 {tile_p[99] * 100:6.2f}%")
    print(f"  Of the PIXELS inside the dirty rects, the fraction that really "
          f"changed:")
    print(f"    p1  {pixel_p[1] * 100:6.2f}%   p50 {pixel_p[50] * 100:6.2f}%   "
          f"p90 {pixel_p[90] * 100:6.2f}%")
    print(f"  Frames reported dirty where NOTHING changed: "
          f"{wholly_false} of {len(pixel_ratios)}")
    print()
    print(f"  CPU per-tile difference pass: {summarise(hash_samples)[0]:.3f} ms "
          f"(upper bound; a GPU digest avoids the readback)")
    print()
    claimed_p = percentiles(claimed_area)
    true_p = percentiles(true_area)
    print("  How much of the frame would each scheme copy? (p50)")
    print(f"    everything, as today                      100.00%")
    print(f"    DWM dirty rects                        {claimed_p[50] * 100:9.2f}%")
    print(f"    pixels that actually changed           {true_p[50] * 100:9.2f}%"
          f"   <- floor for any detector")
    print()
    print(f"  {'tile':>7} {'DWM claims':>12} {'+ digest':>10} {'saved':>8}")
    print("  " + "-" * 40)
    for size in TILE_SIZES:
        claim = percentiles(claimed_occupancy[size])[50]
        truth = percentiles(true_occupancy[size])[50]
        saved = (1 - truth / claim) * 100 if claim else float("nan")
        print(f"  {size:5d}px {claim * 100:11.2f}% {truth * 100:9.2f}% "
              f"{saved:7.1f}%")
    print()
    if tile_p[50] > 0.9:
        print("  DWM's rects are tight at tile granularity. Content hashing "
              "has little to suppress:")
        print("  the detector would cost more than it saves.")
    else:
        print("  DWM marks substantially more than changes. A content digest "
              "could suppress the rest,")
        print("  at the cost of making correctness probabilistic -- see "
              "section 11.")


# --------------------------------------------------------------------------
# 6. per-consumer change tracking
# --------------------------------------------------------------------------

def pack_bits(mask_grid):
    """A tile grid as one Python integer, bit `i` for tile `i`.

    A Python int, not a NumPy array: 260 tiles is 5 machine words, and NumPy
    charges ~1.5 us of call overhead to operate on 40 bytes. The first version
    of this benchmark used NumPy and made the bitmask look *slower* than a rect
    scan, which measured the wrapper rather than the representation.
    """
    value = 0
    for index in np.flatnonzero(mask_grid.ravel()):
        value |= 1 << int(index)
    return value


def bench_mask(harness) -> None:
    """Per-consumer change tracking: tile bitmask versus rect list.

    A consumer polling at 10 Hz off a 165 Hz capture needs the union of what
    changed across ~16 frames, not the last frame's rects. That state is what
    makes section 7.4's ROI scheduling correct rather than merely cheap.

    Both queries are measured twice. A rect scan exits early when it finds an
    overlap, so a hit is its best case and a miss is its worst -- and a miss is
    the answer an idle consumer gets most of the time, which is exactly when it
    wanted to do no work at all.
    """
    header("6. Per-consumer change tracking: bitmask versus rect list",
           "The union a slow consumer needs, and the 'did my ROI change?' "
           "query. Microseconds per operation.")

    width, height = harness.width, harness.height
    rng = np.random.default_rng(20260920)
    frames = 16

    for tile in (64, 128, 256):
        across, down = tile_grid(width, height, tile)
        words = (across * down + 63) // 64

        frame_masks, frame_rects = [], []
        for _ in range(frames):
            count = int(rng.integers(1, 12))
            rects = [tuple(int(v) for v in r)
                     for r in grid_rects(count, 0.02, width, height)]
            frame_rects.append(rects)
            frame_masks.append(pack_bits(tiles_for(rects, tile, width, height)))

        # An ROI in the top-left quadrant, as a mask and as a rect.
        roi_grid = np.zeros((down, across), dtype=bool)
        roi_grid[: max(1, down // 2), : max(1, across // 2)] = True
        roi_bits = pack_bits(roi_grid)
        roi_rect = (0, 0, width // 2, height // 2)
        # One far from anything the frames touch, for the miss case.
        away_grid = np.zeros((down, across), dtype=bool)
        away_grid[-1:, -1:] = True
        away_bits = pack_bits(away_grid)
        away_rect = (width - tile, height - tile, width, height)

        def union_bits():
            accumulated = 0
            for value in frame_masks:
                accumulated |= value
            return accumulated

        def union_rects():
            accumulated = set()
            for rects in frame_rects:
                accumulated.update(rects)
            return accumulated

        bits_state = union_bits()
        rect_state = union_rects()

        def query_bits(probe):
            return bits_state & probe != 0

        def query_rects(probe):
            left, top, right, bottom = probe
            for rl, rt, rr, rb in rect_state:
                if rl < right and rr > left and rt < bottom and rb > top:
                    return True
            return False

        def timed(function, reps=20000):
            for _ in range(200):
                function()
            start = time.perf_counter()
            for _ in range(reps):
                function()
            return (time.perf_counter() - start) / reps * 1e6

        assert query_bits(roi_bits) == query_rects(roi_rect)
        assert query_bits(away_bits) == query_rects(away_rect)

        print(f"  {tile}px grid: {across}x{down} = {across * down} tiles, "
              f"{words} x u64 = {words * 8} bytes of state per consumer")
        print(f"    union of {frames} frames        "
              f"bitmask {timed(union_bits):7.2f}   "
              f"rects {timed(union_rects):7.2f}")
        print(f"    ROI query, changed (hit)   "
              f"bitmask {timed(lambda: query_bits(roi_bits)):7.2f}   "
              f"rects {timed(lambda: query_rects(roi_rect)):7.2f}")
        print(f"    ROI query, quiet (miss)    "
              f"bitmask {timed(lambda: query_bits(away_bits)):7.2f}   "
              f"rects {timed(lambda: query_rects(away_rect)):7.2f}")
        print(f"    state after {frames} frames     "
              f"bitmask {words * 8:4d} bytes    "
              f"rects {len(rect_state):3d} rects "
              f"= {len(rect_state) * 4 * 8:5d} bytes")
        print()

    print("  Rect state grows with how much happened; mask state cannot. That")
    print("  bound, not the microseconds, is the argument -- a consumer that")
    print("  polls rarely accumulates rects without limit until it looks.")


# --------------------------------------------------------------------------

BENCHMARKS = {
    "persistence": (bench_persistence, True),
    "copy-scaling": (bench_copy_scaling, False),
    "tiles": (bench_tiles, False),
    "vram": (bench_vram, False),
    "live": (bench_live, False),
    "overreport": (bench_overreport, False),
    "mask": (bench_mask, False),
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bench", nargs="?", default="all",
                        choices=sorted(BENCHMARKS) + ["all"])
    parser.add_argument("--seconds", type=float, default=20.0,
                        help="duration for the live sub-benchmarks")
    parser.add_argument("--reps", type=int, default=REPS)
    args = parser.parse_args(argv)

    globals()["REPS"] = args.reps
    chosen = sorted(BENCHMARKS) if args.bench == "all" else [args.bench]
    needs_second = any(BENCHMARKS[name][1] for name in chosen)

    print(rapidshot.device_info().strip())
    with Harness(want_second=needs_second) as harness:
        print(f"Output: {harness.width}x{harness.height}, "
              f"{harness.width * harness.height * 4 / 1e6:.1f} MB per BGRA frame")
        for name in chosen:
            function = BENCHMARKS[name][0]
            if name in ("live", "overreport"):
                function(harness, seconds=args.seconds)
            else:
                function(harness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
