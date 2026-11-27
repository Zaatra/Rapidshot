"""Summarize the tiled release gates from completed, immutable raw trials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

MIB = 1024 ** 2
ARMS = {
    "rapidshot_plain": "RapidShot plain",
    "rapidshot_exact": "RapidShot exact full frame",
    "rapidshot_tiled": "RapidShot candidate tiles + ROI",
    "rapidshot_tight": "RapidShot candidate tiles + tight ROI",
    "rapidshot_adaptive_wait": "RapidShot experimental adaptive wait",
    "dxcam_view": "DXcam borrowed view",
    "windows_capture_wgc": "Windows Capture WGC",
}


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def middle(values):
    observed = [value for value in values if value is not None]
    return median(observed) if observed else None


def fmt(value, scale=1):
    return "—" if value is None else f"{value / scale:.2f}"


def load_trials(directory):
    grouped = {}
    for path in sorted((directory / "trials").glob("*.json")):
        trial = read(path)
        if "summary" in trial and "process" in trial:
            grouped.setdefault(trial["arm"], {})[trial["round"]] = trial
    return grouped


def gpu_usage(trial, segment):
    samples = (trial["worker"]["video_memory"] or {}).get("samples", [])
    amounts = [item["segments"][segment]["current_usage_bytes"] for item in samples
               if item.get("segments", {}).get(segment) is not None]
    return max(amounts) if amounts else None


def gpu_duty(trial, engine):
    entries = trial["process"]["gpu_engines"].get("engines", [])
    busy = [row["mean_busy_percent"] for row in entries if row["engine_type"].casefold() == engine.casefold()]
    return max(busy) if busy else None


def paired_values(grouped, arm, metric):
    def measure(row):
        if metric == "Idle CPU":
            return row["process"]["cpu"]["percent_of_one_core"]
        key = "steady_private_bytes" if metric == "Steady commit" else "os_peak_private_bytes"
        value = row["process"]["memory"].get(key)
        return value / MIB if value is not None else None

    plain = grouped.get("rapidshot_plain", {})
    variant = grouped.get(arm, {})
    values = [measure(variant[i]) - measure(plain[i]) for i in sorted(variant.keys() & plain.keys())
              if measure(variant[i]) is not None and measure(plain[i]) is not None]
    return values


def phase_table(grouped):
    rows = ["| Arm | States delivered | Correct on time | Wrong results | Pixel age to delivery p50/p95 ms | "
            "Pixel age to result p50/p95 ms | CPU % of one core | Mcycles / state | Private steady / OS peak MiB | "
            "RSS steady / OS peak MiB | Local / non-local GPU peak MiB | 3D / Copy engine busy % |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for arm, name in ARMS.items():
        arms = list(grouped.get(arm, {}).values())
        if not arms:
            continue
        summaries = [row["summary"] for row in arms]
        cpu = [row["process"]["cpu"] for row in arms]
        memory = [row["process"]["memory"] for row in arms]

        def quantile(field):
            return "/".join(fmt(middle(row[field][key] for row in summaries)) for key in ("p50", "p95"))

        offered = sum(row["eligible_markers"] for row in summaries)
        delivered = sum(row["new_delivered_frames"] for row in summaries)
        deadline_hits = [row.get("correct_on_time") for row in summaries]
        correct = "—" if not offered or any(value is None for value in deadline_hits) else f"{sum(deadline_hits)}/{offered}"
        private = "/".join(fmt(middle(row.get(key) for row in memory), MIB)
                           for key in ("steady_private_bytes", "os_peak_private_bytes"))
        rss = "/".join(fmt(middle(row.get(key) for row in memory), MIB)
                       for key in ("steady_working_set_bytes", "os_peak_working_set_bytes"))
        gpu = "/".join(fmt(middle(gpu_usage(row, key) for row in arms), MIB) for key in ("local", "non_local"))
        duty = "/".join(fmt(middle(gpu_duty(row, key) for row in arms)) for key in ("3D", "Copy"))
        wrong = str(sum(row["fixture_result_mismatches"] for row in summaries)) if correct != "—" else "—"
        coverage = f"{delivered}/{offered}" if offered else f"{sum(row['grab_returns'] for row in summaries)} idle returns"
        rows.append(f"| {name} | {coverage} | {correct} | {wrong} | "
                    f"{quantile('desktop_to_delivery_ms')} | {quantile('desktop_to_downstream_ms')} | "
                    f"{fmt(middle(row['percent_of_one_core'] for row in cpu))} | "
                    f"{fmt(middle(row['cycles_per_delivered_frame'] for row in cpu), 1e6)} | {private} | {rss} | {gpu} | {duty} |")
    return rows


def copy_table(grouped):
    rows = ["| Arm | GPU staging MiB / state | CPU mapped writes | Private history writes | Probe crop writes | "
            "Search crop writes | Change gate p50 ms | Search p50 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    fields = ("gpu_staging_payload_bytes", "mapped_staging_to_cpu_bytes", "private_history_bytes_written",
              "fixture_probe_materialization_bytes", "task_crop_materialization_bytes")
    for arm, name in ARMS.items():
        trials = list(grouped.get(arm, {}).values())
        if not trials:
            continue
        metrics = [trial["summary"] for trial in trials]
        cols = [fmt(middle(row["payload_per_delivered_frame"].get(key) for row in metrics), MIB) for key in fields]
        cols += [fmt(middle(row[key]["p50"] for row in metrics)) for key in ("gate_ms", "search_ms")]
        rows.append(f"| {name} | {' | '.join(cols)} |")
    return rows


def quiet_table(output):
    path = output / "quiet-idle/quiet-tiled.json"
    if not path.is_file():
        return []
    data = read(path)
    rows = ["", "### Confirmed-quiet follow-up", "",
            "Some nominal idle trials received desktop frames. Their CPU deltas remain in the frozen gate table. "
            "This follow-up requires zero delivered frames during measurement and a seeded baseline in the exact arms. "
            "A quiet comparison can explain the idle component while the original nominal-idle failures remain visible.", "",
            "| Round | Variant | Captures during window | Qualified | CPU % of one core | Mcycles/s | Delta vs plain in same round |",
            "| ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for round_row in data["rounds"]:
        arms = round_row["arms"]
        plain = arms.get("plain")
        for arm, row in arms.items():
            cpu = row["telemetry"]["cpu"]["percent_of_one_core"]
            delta = (cpu - plain["telemetry"]["cpu"]["percent_of_one_core"]
                     if plain is not None and plain["qualifying"] and row["qualifying"] else None)
            rows.append(f"| {round_row['round']} | {arm} | {row['deliveries_in_window']} | "
                        f"{'yes' if row['qualifying'] else 'no'} | {fmt(cpu)} | "
                        f"{fmt(row['telemetry']['cpu']['cycles_per_second'], 1e6)} | {fmt(delta)} |")
    rows.extend(["", "[Quiet window eligibility, CPU cycles and memory](quiet-idle/quiet-tiled.json)"])
    return rows


def report(output):
    root = output / "measurements"
    run = read(root / "campaign.json")
    live = read(output / "exact-live.json")
    soak = read(output / "soak/soak.json")
    timer = read(output / "soak/soak-run.json")
    phases = {name: load_trials(root / name) for name in ("idle-30", "sparse-100", "matched-30")}
    heading = ["# RapidShot tiled capture release checks", "",
               f"Measured on the HP 324pe at Intel 1920×1080 and 100 Hz. The live comparison took "
               f"**{run['elapsed_seconds']:.1f} seconds** across two shuffled rounds. The one-process soak ran within "
               f"**{timer['elapsed_seconds']:.1f} seconds** wall clock against its **{timer['cap_seconds']} second** cap.", "",
               f"The live oracle matched **{len(live['tiled']['frames'])}** tiled deliveries against a separate full "
               f"readback of the same acquired texture and **{len(live['frames'])}** existing exact deliveries against "
               "a full pixel comparison. Both real lease pools recovered after forced exhaustion without consuming "
               "an update while full. [Live oracle and pool evidence](exact-live.json)", "",
               "The targeted regression suite exercised Python and native DXGI, validation, recovery, and buffer "
               "ownership. [Test log](pytest.log)", "",
               f"The soak status was **{soak['status']}**. It checked **{soak['verified']}** delivered pixel maps and "
               f"**{soak['retained_checks']}** retained snapshots. It found "
               f"**{soak.get('audits', {}).get('missed_tiles', '—')}** missed tiles in "
               f"**{soak.get('audits', {}).get('audited_frames', '—')}** full scheduled audits. The fixture offered "
               f"**{soak['offered']}** updates, and **{soak['matched_presentations']}** deliveries had the expected "
               "marker and entire scripted BGR image. Capture coverage and mixed presentations are separate from "
               "the tile-map oracle over delivered pixels. The soak includes fixture and oracle overhead and supplies "
               "no uninterrupted 30-minute result. [Soak raw evidence](soak/soak.json)", "",
               "## Frozen paired gates", "",
               "The original thresholds are +12 MiB steady private commit, +20 MiB OS peak private commit and "
               "+0.5 points of idle CPU against matched plain RapidShot. Both fresh pairs must pass. The historical "
               "30/s idle CPU pair was +0.613 points and the historical 100/s sparse commit pair was +15.53 MiB. "
               "Both historical failures remain in the record. [Earlier evidence](../results-cost-matrix/report.md)", "",
               "| Workload | Variant | Metric | Round deltas | Median | Worst | Within threshold | Gate |",
               "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for workload in ("idle-30", "sparse-100"):
        metrics = (("Steady commit", 12), ("OS peak commit", 20))
        if workload == "idle-30":
            metrics += (("Idle CPU", .5),)
        for arm in ("rapidshot_exact", "rapidshot_tiled", "rapidshot_tight"):
            for metric, threshold in metrics:
                values = paired_values(phases[workload], arm, metric)
                passed = sum(value <= threshold for value in values)
                outcome = "open" if len(values) < 2 else "pass" if passed == len(values) else "fail"
                heading.append(f"| {workload} | {ARMS[arm]} | {metric} | "
                               f"{' / '.join(f'{v:+.2f}' for v in values)} | {fmt(middle(values))} | "
                               f"{fmt(max(values) if values else None)} | {passed}/{len(values)} | {outcome} |")

    heading.extend(["", "## Live 30/s with the same template task", "",
                    "RapidShot existing and DXcam view run the same regional template search with complete captured "
                    "frames. Candidate tiles materialize only required crops. The tight arm also narrows the template "
                    "origin windows from measured tight bounds. A result counts as on time only if it is correct "
                    "and completes within the offered 33.33 ms period, measured from submission. Pixel age uses "
                    "validated DXGI present timestamps. Quantiles and resource entries are medians of the two trial "
                    "values. Counts sum both trials. An em dash means unavailable or inapplicable.", "",
                    *phase_table(phases["matched-30"]), "",
                    "The tile-only capture change can lower delivery age while leaving the OpenCV work largely "
                    "unchanged. The tight search arm is the combined test of capture and downstream region size.", "",
                    "Logical copy writes below are charged by path. Driver allocation and OS process/GPU usage "
                    "are in the table above. DXcam view pays for keeping a private previous-frame snapshot.", "",
                    *copy_table(phases["matched-30"]), "",
                    "[Matched trials and telemetry](measurements/matched-30/report.md)", "",
                    "## Cost controls", "", "### Nominal idle", "", *phase_table(phases["idle-30"]),
                    *quiet_table(output), "",
                    "### Sparse 100/s", "", *phase_table(phases["sparse-100"]), "",
                    "[Idle trials](measurements/idle-30/report.md) · [Sparse 100/s trials](measurements/sparse-100/report.md)", "",
                    "## Scope", "",
                    "The release report covers this Intel 1080p setup. Native 4K, NVIDIA and AMD remain "
                    "hardware-dependent. [Campaign timing and completeness](measurements/campaign.json)", "",
                    "### Ideas held or dropped after the prototype sweep", "",
                    "| Idea | Observed cost or limitation |",
                    "| --- | --- |",
                    "| Adaptive search planner | Missed the 2% maximum-loss target in 10/52 pairs. |",
                    "| Two-slot staging | Similar throughput with 12–13 ms pixel age, versus 4–5 ms synchronous. |",
                    "| GPU exact comparison on Intel | 0.77 ms for sparse GPU comparison versus 0.48 ms for CPU candidates. |",
                    "| Drop dense history and reseed | 277 ms per sequence versus 105–130 ms with retained history. |",
                    "| Lazy history without a separate baseline | Exact comparison still needs a complete previous image. |",
                    "| Settled frames | Two equal acquisitions delivered 0/128 states in the busy fixture. |",
                    "| Mixed-frame risk flags | Accumulated frames flagged 40/63 mixed results and 1,737/2,955 correct controls. |",
                    "| Scroll reuse | Faster in sampled browser searches, slower in Word, and no usable Excel shift. |",
                    "| Priority ranking and repeat cache | Useful in limited first-hit and repeating-content workloads. |", "",
                    "[Full prototype scorecard, offline 4K and model tests](../results-idea-sweep/report.md)", ""])
    (output / "report.md").write_text("\n".join(heading), encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    report(parser.parse_args().output)
