"""Screen to encoded packet: how long, at what cost, and is the video right.

    python benchmarks/encode_latency.py                     # every encoder, 5 shuffled runs
    python benchmarks/encode_latency.py --encoder h264_nvenc --path gpu-nv12
    python benchmarks/encode_latency.py --seconds 5 --repeats 5 --json out.json

ROADMAP § 7.3a's first step: measure before building ``rapidshot.Encoder``.
A real RapidShot camera captures the primary display under the frame-ID test
source (``latency_source.exe``, as ``rapidshot benchmark`` uses it), and every
frame goes through one of two paths into an FFmpeg encoder through PyAV:

- ``gpu-nv12``: ``grab_frame()`` -> ``GpuConverter(pixel_format="nv12")`` on
  the capture GPU -> read back to system memory -> encoder. RapidShot's own
  route today: the colour conversion never touches the CPU.
- ``cpu-bgra``: ``grab()`` BGRA -> FFmpeg's swscale to NV12 on the CPU ->
  encoder. What a program without RapidShot's converter writes.

Every encoder is opened with a low-latency configuration (no B-frames, CBR,
the encoder's own low-delay switches) and **refused if FFmpeg left any of
those options unused** -- PyAV accepts an unknown option silently, so a
misspelled ``zerolatency`` would otherwise measure the default.

What it measures, per frame, on one QPC clock:

- **latency**: from the test source's ``Present()`` of the pixels to the
  encoded packet in hand. The frame is identified by decoding the *encoded
  stream* afterwards and reading the frame-ID marker out of the decoded
  picture, then looking that ID up in the source's own present log. A frame
  that is not in the stream, or is not the frame we think, cannot count.
- the stages: present -> captured (Windows' ``LastPresentTime``), captured ->
  NV12 in system memory, and encoder in -> packet out.
- **fidelity**: luma PSNR of the decoded stream against exactly the NV12 the
  encoder was given, on every 10th frame. A fast encoder that produces the
  wrong picture loses here, not in a footnote.
- frames encoded a second, process CPU per frame, bitrate.

Software decoders do the verification (h264, hevc, libdav1d), outside the
timed loop.

Licensing note, checked rather than assumed: PyAV 19.0.2's wheel bundles an
FFmpeg that reports LGPL-3.0 but ships ``libx264`` and ``libx265``, which are
GPL. Fine for running this benchmark; not a dependency RapidShot can take on
without a licence review.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import statistics
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

# Before anything imports comtypes. `import rapidshot` imports comtypes, which
# initialises COM on this thread as a single-threaded apartment, and FFmpeg's
# Media Foundation encoder refuses one ("COM must not be in STA mode"):
# h264_mf's first open fails, and its error path uninitialises COM on the
# thread, so later opens succeed and the camera's COM state is gone. Measured:
# STA 1 of 3 opens fail; MTA (this flag) 3 of 3 succeed; an encoder on its own
# thread, which is MTA, succeeds. rapidshot.Encoder must give Media Foundation
# an MTA thread of its own; this harness runs everything MTA instead.
sys.coinit_flags = 0  # COINIT_MULTITHREADED

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

#: Low-latency configuration per encoder. Every key is checked against what
#: FFmpeg actually consumed when the encoder opened.
ENCODERS = {
    "h264_nvenc": {"preset": "p1", "tune": "ull", "zerolatency": "1",
                   "delay": "0", "rc": "cbr"},
    "hevc_nvenc": {"preset": "p1", "tune": "ull", "zerolatency": "1",
                   "delay": "0", "rc": "cbr"},
    "av1_nvenc": {"preset": "p1", "tune": "ull", "zerolatency": "1",
                  "delay": "0", "rc": "cbr"},
    "h264_qsv": {"preset": "veryfast", "async_depth": "1",
                 "look_ahead_depth": "0", "low_delay_brc": "1"},
    "hevc_qsv": {"preset": "veryfast", "async_depth": "1",
                 "look_ahead_depth": "0", "low_delay_brc": "1"},
    "h264_mf": {"hw_encoding": "1", "scenario": "display_remoting",
                "rate_control": "cbr"},
    "libx264": {"preset": "ultrafast", "tune": "zerolatency"},
}
DECODERS = {"h264": "h264", "hevc": "hevc", "av1": "libdav1d", "libx264": "h264"}
PATHS = ("gpu-nv12", "cpu-bgra")
MODES = ("sync", "reuse", "pipelined")
BITRATE = 20_000_000
PSNR_EVERY = 10


def decode_marker_luma(luma):
    """The frame ID from a *decoded* picture's luma plane, or None.

    ``pixel_age.decode_marker`` wants each cell above 240 or below 15 in all
    three channels, which suits an uncompressed capture. Measured on this
    harness: every marker was readable in what the encoders were given, and
    AV1 and HEVC then softened up to a third of them past those thresholds.
    So read the cell centres at the midpoint instead -- the cells are only
    ever black or white -- and keep what makes a misread impossible to pass:
    the magic byte and the checksum over the ID.
    """
    from rapidshot.util.pixel_age import CELL, MAGIC, MARKER_BITS, checksum

    if luma.shape[0] <= 8 or luma.shape[1] < MARKER_BITS * CELL:
        return None
    cells = luma[8, np.arange(MARKER_BITS) * CELL + CELL // 2]
    bits = sum(int(v > 128) << i for i, v in enumerate(cells))
    frame_id = (bits >> 8) & 0xFFFFFFFF
    if bits & 255 != MAGIC or bits >> 40 != checksum(frame_id):
        return None
    return frame_id


def _process_cpu_seconds() -> float:
    return time.process_time()


def _decoder_for(encoder: str) -> str:
    return DECODERS["libx264" if encoder == "libx264" else encoder.split("_")[0]]


def _open_encoder(name, width, height, hz):
    import av

    ctx = av.CodecContext.create(name, "w")
    ctx.width, ctx.height = width, height
    ctx.pix_fmt = "nv12"
    ctx.time_base = Fraction(1, hz)
    ctx.framerate = Fraction(hz, 1)
    ctx.bit_rate = BITRATE
    ctx.max_b_frames = 0
    ctx.gop_size = hz * 2
    ctx.options = dict(ENCODERS[name])
    ctx.open()
    unused = dict(ctx.options)
    if unused:
        raise RuntimeError(f"{name} did not use {unused}: refusing to measure "
                           "a configuration other than the one named")
    return ctx


#: Pre-allocated encoder frames for the ``reuse`` and ``pipelined`` modes. A
#: frame goes back to the pool one encode after it was sent, and the decode
#: check would show it if an encoder still read one being refilled.
POOL = 6


def _frame_pool(width, height):
    """PyAV NV12 frames allocated once, with NumPy views of their planes.

    Measured on this harness: ``VideoFrame.from_ndarray`` for a 2560x1600 NV12
    frame is 2.32 ms p50, copying the same bytes into a frame allocated once
    is 0.21 ms -- the cost is first-touching fresh memory, not copying.
    """
    import av

    pool = []
    for _ in range(POOL):
        frame = av.VideoFrame(width, height, "nv12")
        y, uv = frame.planes[0], frame.planes[1]
        pool.append((frame,
                     np.frombuffer(y, np.uint8).reshape(height, y.line_size)[:, :width],
                     np.frombuffer(uv, np.uint8).reshape(height // 2, uv.line_size)[:, :width]))
    return pool


def _fill(slot, nv12, height):
    _frame, y, uv = slot
    np.copyto(y, nv12[:height])
    np.copyto(uv, nv12[height:])


def _run_pipelined(camera, pool, width, height, deadline, send, now, nv12_output="buffer"):
    """Capture, convert and read back on one thread; encode on this one.

    One slot between them, newest wins -- what ``policy="latest"`` does for a
    consumer -- so a slow encoder costs dropped frames, never a queue of stale
    ones. A pool slot returns one encode after it was sent. Returns how many
    frames were replaced unencoded.
    """
    import queue
    import threading

    import rapidshot

    handoff = queue.Queue(maxsize=1)
    free = queue.Queue()
    for index in range(len(pool)):
        free.put(index)
    stop = threading.Event()
    state = {"drops": 0, "error": None}

    def produce():
        converter = None
        try:
            while not stop.is_set():
                frame = camera.grab_frame()
                if frame is None:
                    continue
                captured = now()
                with frame:
                    present = frame.timestamp_qpc
                    if converter is None:
                        converter = rapidshot.GpuConverter(
                            frame, (width, height), pixel_format="nv12",
                            output=nv12_output)
                    nv12 = converter.process(frame).numpy()
                nv12 = nv12.reshape(height * 3 // 2, width)
                try:
                    index = free.get(timeout=0.5)
                except queue.Empty:
                    continue
                _fill(pool[index], nv12, height)
                item = (index, present, captured, now())
                try:
                    handoff.put_nowait(item)
                except queue.Full:
                    try:
                        replaced = handoff.get_nowait()
                        free.put(replaced[0])
                        state["drops"] += 1
                    except queue.Empty:
                        pass
                    handoff.put_nowait(item)
        except Exception as exc:  # noqa: BLE001 - surfaced below
            state["error"] = exc

    producer = threading.Thread(target=produce, name="capture-convert", daemon=True)
    producer.start()
    held = None
    try:
        while time.monotonic() < deadline and state["error"] is None:
            try:
                index, present, captured, ready = handoff.get(timeout=0.5)
            except queue.Empty:
                continue
            send(pool[index][0], present, captured, ready)
            if held is not None:
                free.put(held)
            held = index
    finally:
        stop.set()
        producer.join(5)
    if state["error"] is not None:
        raise state["error"]
    return state["drops"]


def run_case(path, encoder, seconds, log, hz, mode="sync", nv12_output="buffer"):
    import av
    import rapidshot
    from rapidshot.frame import _qpc_freq, _qpc_now

    if mode != "sync" and path != "gpu-nv12":
        raise ValueError(f"mode {mode!r} is measured on the gpu-nv12 path only")
    freq = _qpc_freq()
    camera = rapidshot.create(output_color="BGRA")
    width, height = camera.width, camera.height
    ctx = _open_encoder(encoder, width, height, hz)
    converter = None
    sent = {}            # pts -> (present_qpc, captured_qpc, ready_qpc, send_qpc)
    out_qpc = {}         # pts -> packet-out qpc
    packets = []
    references = {}      # pts -> luma the encoder was given (every Nth)
    flushed = 0
    pipeline_drops = 0
    pool = _frame_pool(width, height) if mode != "sync" else None
    try:
        for _ in range(10):                       # warm-up: pools, first frames
            warm = camera.grab()
            if warm is not None and hasattr(warm, "release"):
                warm.release()
        log.refresh()
        first_present = max(log.frames, default=0)
        pts = 0
        cpu0 = _process_cpu_seconds()
        start = time.monotonic()
        deadline = start + seconds

        def send(vframe, present, captured, ready):
            nonlocal pts
            vframe.pts = pts
            if pts % PSNR_EVERY == 0:
                references[pts] = vframe.to_ndarray()[:height].copy()
            sent_at = _qpc_now()
            emitted = ctx.encode(vframe)
            done = _qpc_now()
            for packet in emitted:
                out_qpc[packet.pts] = done
                packets.append(packet)
            sent[pts] = (present, captured, ready, sent_at)
            pts += 1

        if mode == "pipelined":
            pipeline_drops = _run_pipelined(camera, pool, width, height, deadline,
                                            send, _qpc_now, nv12_output)
        while mode != "pipelined" and time.monotonic() < deadline:
            if path == "gpu-nv12":
                frame = camera.grab_frame()
                if frame is None:
                    continue
                captured = _qpc_now()
                with frame:
                    present = frame.timestamp_qpc
                    if converter is None:
                        converter = rapidshot.GpuConverter(
                            frame, (width, height), pixel_format="nv12",
                            output=nv12_output)
                    nv12 = converter.process(frame).numpy()
                nv12 = nv12.reshape(height * 3 // 2, width)
                if mode == "reuse":
                    slot = pool[pts % POOL]
                    _fill(slot, nv12, height)
                    vframe = slot[0]
                else:
                    vframe = av.VideoFrame.from_ndarray(nv12, format="nv12")
                ready = _qpc_now()
            else:
                buffer = camera.grab()
                if buffer is None:
                    continue
                captured = _qpc_now()
                present = camera.last_present_time
                bgra = np.asarray(getattr(buffer, "array", buffer))
                vframe = av.VideoFrame.from_ndarray(bgra, format="bgra")
                if hasattr(buffer, "release"):
                    buffer.release()
                vframe = vframe.reformat(format="nv12")     # swscale, on the CPU
                ready = _qpc_now()
            send(vframe, present, captured, ready)
        wall = time.monotonic() - start
        cpu = _process_cpu_seconds() - cpu0
        for packet in ctx.encode(None):                 # flush: not timed
            packets.append(packet)
            flushed += 1
        log.refresh()
        source_fps = (max(log.frames, default=0) - first_present) / wall
    finally:
        converter = None
        camera.release()

    # ---- verification, outside the timed loop -------------------------------
    decoder = av.CodecContext.create(_decoder_for(encoder), "r")
    decoded = []
    for packet in packets:
        decoded.extend(decoder.decode(packet))
    decoded.extend(decoder.decode(None))
    by_pts = {}
    for index, picture in enumerate(decoded):
        key = picture.pts if picture.pts is not None else index
        by_pts[key] = picture

    latencies, psnrs, marker_ids = [], [], []
    undecodable = order_errors = identity_errors = identity_checked = 0
    for p in sorted(sent):
        picture = by_pts.get(p)
        if picture is None:
            continue
        luma = picture.to_ndarray(format="nv12")[:height]
        if p in references:
            mse = float(np.mean((luma.astype(np.float64) - references[p]) ** 2))
            psnrs.append(99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse))
            # The decoded picture must be the frame that went in as this pts.
            given, got = decode_marker_luma(references[p]), decode_marker_luma(luma)
            if given is not None and got is not None:
                identity_checked += 1
                identity_errors += given != got
        if p not in out_qpc:
            continue                                     # came out in the flush
        frame_id = decode_marker_luma(luma)
        if frame_id is None:
            undecodable += 1
            continue
        if marker_ids and frame_id < marker_ids[-1]:
            order_errors += 1
        marker_ids.append(frame_id)
        row = log.frames.get(frame_id)
        if row is not None:
            latencies.append((out_qpc[p] - row["qpc_before"]) * 1000.0 / freq)

    def ms(a, b):
        return [(b_ - a_) * 1000.0 / freq for a_, b_ in zip(a, b)]

    stages = [sent[p] for p in sorted(sent) if p in out_qpc]
    # Windows' LastPresentTime; 0 where the backend reported none.
    present_to_capture = [(s[1] - s[0]) * 1000.0 / freq for s in stages if s[0]]
    capture_to_ready = ms([s[1] for s in stages], [s[2] for s in stages])
    encode = [(out_qpc[p] - sent[p][3]) * 1000.0 / freq for p in sorted(sent) if p in out_qpc]
    frames = len(sent)
    total_bytes = sum(packet.size for packet in packets)

    def pct(values, q):
        if not values:
            return None
        values = sorted(values)
        return values[min(len(values) - 1, int(q * len(values)))]

    return {
        "path": path, "encoder": encoder, "mode": mode, "nv12_output": nv12_output,
        "pipeline_drops": pipeline_drops,
        "width": width, "height": height,
        "frames": frames, "fps": frames / wall,
        "latency_ms_p50": pct(latencies, .5), "latency_ms_p95": pct(latencies, .95),
        "latency_ms_max": max(latencies) if latencies else None,
        "present_to_capture_ms_p50": pct(present_to_capture, .5),
        "capture_to_nv12_ms_p50": pct(capture_to_ready, .5),
        "encode_ms_p50": pct(encode, .5), "encode_ms_p95": pct(encode, .95),
        "cpu_ms_per_frame": cpu * 1000.0 / frames if frames else None,
        "mbps": total_bytes * 8 / wall / 1e6,
        "psnr_db_median": statistics.median(psnrs) if psnrs else None,
        "psnr_db_min": min(psnrs) if psnrs else None,
        "decoded_frames": len(decoded), "timed_frames_identified": len(latencies),
        "undecodable_markers": undecodable, "order_errors": order_errors,
        "identity_checked": identity_checked, "identity_errors": identity_errors,
        "flushed_packets": flushed, "source_fps": source_fps,
    }


# --------------------------------------------------------------------------
# Runner and report
# --------------------------------------------------------------------------

METRICS = ("fps", "latency_ms_p50", "latency_ms_p95", "latency_ms_max",
           "present_to_capture_ms_p50", "capture_to_nv12_ms_p50",
           "encode_ms_p50", "encode_ms_p95", "cpu_ms_per_frame", "mbps",
           "psnr_db_median", "psnr_db_min", "undecodable_markers",
           "order_errors", "source_fps", "timed_frames_identified",
           "identity_checked", "identity_errors", "pipeline_drops")


def aggregate(runs):
    out = {"path": runs[0]["path"], "encoder": runs[0]["encoder"], "mode": runs[0]["mode"],
           "nv12_output": runs[0].get("nv12_output", "buffer"),
           "runs": len(runs), "range": {}}
    for key in METRICS:
        values = [r[key] for r in runs if r.get(key) is not None]
        out[key] = statistics.median(values) if values else None
        out["range"][key] = (min(values), max(values)) if values else None
    return out


def _cell(row, key, digits=1):
    value, bounds = row[key], row["range"].get(key)
    if value is None:
        return "-"
    text = f"{value:.{digits}f}" if isinstance(value, float) else str(value)
    if bounds and bounds[0] != bounds[1]:
        lo, hi = bounds
        text += f" ({lo:.{digits}f}-{hi:.{digits}f})"
    return text


def report(rows, meta):
    lines = [
        f"# Screen to encoded packet, {meta['width']}x{meta['height']} @ {meta['hz']} Hz",
        "",
        f"Each case ran {meta['repeats']} times for {meta['seconds']:g} s, all runs in one "
        f"shuffled order (seed {meta['seed']}). Medians, with the min-max range in "
        "brackets where runs differed. Latency is from the test source's Present() "
        "to the packet in hand, for frames identified by decoding the stream. "
        f"{BITRATE // 1_000_000} Mbps CBR, no B-frames.",
        "",
        "| path | mode | encoder | fps | latency p50 ms | latency p95 ms | present->capture ms "
        "| capture->NV12 ms | encode p50 / p95 ms | CPU ms/frame | Mbps | PSNR median / min dB | frames replaced unencoded |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(rows, key=lambda r: (r["latency_ms_p50"] is None, r["latency_ms_p50"] or 0)):
        lines.append(
            f"| {r['path']}{'/' + r['nv12_output'] if r['path'] == 'gpu-nv12' else ''} | {r['mode']} | `{r['encoder']}` | {_cell(r, 'fps')} "
            f"| {_cell(r, 'latency_ms_p50')} | {_cell(r, 'latency_ms_p95')} "
            f"| {_cell(r, 'present_to_capture_ms_p50')} | {_cell(r, 'capture_to_nv12_ms_p50')} "
            f"| {_cell(r, 'encode_ms_p50')} / {_cell(r, 'encode_ms_p95')} "
            f"| {_cell(r, 'cpu_ms_per_frame')} | {_cell(r, 'mbps')} "
            f"| {_cell(r, 'psnr_db_median')} / {_cell(r, 'psnr_db_min')} "
            f"| {_cell(r, 'pipeline_drops', 0) if r['mode'] == 'pipelined' else '-'} |")
    problems = [f"- {r['path']} {r['mode']} `{r['encoder']}`: {r['undecodable_markers']} undecodable "
                f"markers, {r['order_errors']} out of order, {r['identity_errors']} of "
                f"{r['identity_checked']} checked frames not the frame sent"
                for r in rows if (r["undecodable_markers"] or 0) or (r["order_errors"] or 0)
                or (r["identity_errors"] or 0)]
    checked = sum(int(r["identity_checked"] or 0) for r in rows)
    lines += ["", (f"Verification: every timed frame's ID was read back from the decoded "
                   f"stream, in order, and {checked} sampled frames (medians, summed "
                   "over cases) were each the frame sent.") if not problems else
              "Verification problems (medians):\n" + "\n".join(problems)]
    if meta["unhealthy"]:
        lines += ["", "**Not trustworthy: the test source fell below 90% of its rate in:**",
                  *[f"- {u}" for u in meta["unhealthy"]]]
    if meta["failed"]:
        lines += ["", "Cases that could not run:", *[f"- {f}" for f in meta["failed"]]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoder", choices=sorted(ENCODERS), action="append")
    parser.add_argument("--path", choices=PATHS, action="append")
    parser.add_argument("--nv12-output", choices=("buffer", "texture"), action="append",
                        help="GpuConverter output for gpu-nv12; default buffer")
    parser.add_argument("--mode", choices=MODES, action="append",
                        help="gpu-nv12 handoff: sync (step 1), reuse, pipelined; default sync")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pause", type=float, default=1.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    from rapidshot.planner import _Source
    from rapidshot.util.pixel_age import PresentLog
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from freshness_policies import _primary_mode

    width, height, hz = _primary_mode()
    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
    modes = args.mode or ["sync"]
    outputs = args.nv12_output or ["buffer"]
    cases = [(p, e, m, o) for p in (args.path or PATHS) for e in (args.encoder or ENCODERS)
             for m in modes if m == "sync" or p == "gpu-nv12"
             for o in (outputs if p == "gpu-nv12" else ["buffer"])]
    jobs = [(p, e, m, o, r) for r in range(args.repeats) for p, e, m, o in cases]
    random.Random(seed).shuffle(jobs)
    print(f"{len(jobs)} runs, {width}x{height} @ {hz} Hz, seed {seed}",
          file=sys.stderr, flush=True)

    import logging

    import av.logging
    libav = []

    class _Keep(logging.Handler):
        def emit(self, record):
            libav.append(record.getMessage())

    av.logging.set_level(av.logging.VERBOSE)
    logging.getLogger("libav").addHandler(_Keep())
    logging.getLogger("libav").setLevel(logging.DEBUG)
    logging.getLogger("libav").propagate = False

    runs, failed, unhealthy = {}, [], []
    source = _Source(width, height, hz)
    log = PresentLog(source.log_path, source.frequency)
    try:
        for index, (path, encoder, mode, output, repeat) in enumerate(jobs, 1):
            print(f"[{index}/{len(jobs)}] {path}/{output} {mode} {encoder} (run {repeat + 1})",
                  file=sys.stderr, flush=True)
            libav.clear()
            try:
                result = run_case(path, encoder, args.seconds, log, hz, mode, output)
            except Exception as exc:  # noqa: BLE001 - a refusing encoder is a result
                # Keep FFmpeg's own account of it: h264_mf failed its first
                # open in a process 2 times in about 30 while this was
                # written, and did not reproduce in 5 fresh processes.
                evidence = [m for m in libav if "MFT" in m or "activate" in m
                            or "rror" in m or "ail" in m][-12:]
                failed.append(f"run {index} {path} {mode} {encoder}: {type(exc).__name__}: {exc}"
                              + "".join(f"\n  - libav: {m.strip()}" for m in evidence))
                print("FAILED " + failed[-1], file=sys.stderr, flush=True)
                continue
            result.update(run=repeat + 1, order=index)
            if result["source_fps"] < 0.9 * hz:
                unhealthy.append(f"run {index} ({path} {mode} {encoder}): source "
                                 f"{result['source_fps']:.1f} fps of {hz}")
                print("WARNING " + unhealthy[-1], file=sys.stderr, flush=True)
            runs.setdefault((path, encoder, mode, output), []).append(result)
            gc.collect()
            time.sleep(args.pause)
    finally:
        log.close()
        source.close()

    rows = [aggregate(rs) for rs in runs.values()]
    meta = {"width": width, "height": height, "hz": hz, "seconds": args.seconds,
            "repeats": args.repeats, "seed": seed, "unhealthy": unhealthy,
            "failed": failed}
    print(report(rows, meta))
    if args.json:
        args.json.write_text(json.dumps({
            **meta, "encoders": ENCODERS, "bitrate": BITRATE, "medians": rows,
            "runs": {f"{p}|{e}|{m}|{o}": r for (p, e, m, o), r in runs.items()},
        }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
