use crate::tile_cache::CacheReport;
use crate::AnyResult;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};
use windows::Win32::System::Threading::GetCurrentThread;
use windows::Win32::System::WindowsProgramming::QueryThreadCycleTime;

const WARMUP: usize = 8;

#[derive(Default, Serialize)]
pub struct Sample {
    pub latency_us: f64,
    pub stages: BTreeMap<String, f64>,
    pub cpu_cycles: Option<u64>,
}

pub struct Arm<'a> {
    pub name: &'a str,
    pub run: Box<dyn FnMut(usize, bool) -> AnyResult<Sample> + 'a>,
}

#[derive(Serialize)]
pub struct Distribution {
    pub p50: f64,
    pub p95: f64,
    pub p50_min: f64,
    pub p50_max: f64,
    samples: usize,
}

#[derive(Serialize)]
struct Trial {
    round: usize,
    position: usize,
    elapsed_ms: f64,
    samples: Vec<Sample>,
}

#[derive(Serialize)]
pub struct ArmResult {
    pub name: String,
    pub latency_us: Distribution,
    pub incremental_us: Option<Distribution>,
    pub cpu_cycles: Option<Distribution>,
    pub stages_us: BTreeMap<String, Distribution>,
    trials: Vec<Trial>,
}

#[derive(Serialize)]
pub struct Group {
    name: String,
    endpoint: String,
    baseline: Option<String>,
    seed: u64,
    rounds: usize,
    reps: usize,
    hz: u32,
    warmup: usize,
    order: Vec<Vec<String>>,
    pub arms: Vec<ArmResult>,
}

pub fn thread_cycles() -> Option<u64> {
    let mut cycles = 0;
    unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) }.ok()?;
    Some(cycles)
}

pub(crate) fn random_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn draw_order(count: usize, seed: &mut u64) -> Vec<usize> {
    let mut order: Vec<_> = (0..count).collect();
    for index in (1..count).rev() {
        let other = random_u64(seed) as usize % (index + 1);
        order.swap(index, other);
    }
    order
}

fn percentile(values: &[f64], fraction: f64) -> f64 {
    let index = (values.len() - 1) as f64 * fraction;
    let first = index.floor() as usize;
    let last = index.ceil() as usize;
    values[first] + (values[last] - values[first]) * index.fract()
}

fn statistics(trials: impl IntoIterator<Item = Vec<f64>>) -> Option<Distribution> {
    let mut count = 0;
    let mut p50 = Vec::new();
    let mut p95 = Vec::new();
    for mut trial in trials.into_iter().filter(|trial| !trial.is_empty()) {
        count += trial.len();
        trial.sort_by(f64::total_cmp);
        p50.push(percentile(&trial, 0.50));
        p95.push(percentile(&trial, 0.95));
    }
    if p50.is_empty() {
        return None;
    }
    p50.sort_by(f64::total_cmp);
    p95.sort_by(f64::total_cmp);
    Some(Distribution {
        p50: percentile(&p50, 0.50),
        p95: percentile(&p95, 0.50),
        p50_min: p50[0],
        p50_max: p50[p50.len() - 1],
        samples: count,
    })
}

fn pace(start: Instant, period: Duration) {
    if let Some(remaining) = period.checked_sub(start.elapsed()) {
        thread::sleep(remaining);
    }
}

fn trial(
    arm: &mut Arm<'_>,
    round: usize,
    position: usize,
    reps: usize,
    period: Duration,
) -> AnyResult<Trial> {
    for i in 0..WARMUP {
        let start = Instant::now();
        (arm.run)(i, true).map_err(|error| format!("{} warmup sample {i}: {error}", arm.name))?;
        pace(start, period);
    }
    let mut samples = Vec::with_capacity(reps);
    let start = Instant::now();
    for i in 0..reps {
        let frame_start = Instant::now();
        let sample = (arm.run)(i, false)
            .map_err(|error| format!("{} round {round} sample {i}: {error}", arm.name))?;
        if !sample.latency_us.is_finite()
            || sample.latency_us < 0.0
            || sample.stages.values().any(|value| !value.is_finite())
        {
            return Err(format!("{} round {round} sample {i}: invalid timing", arm.name).into());
        }
        samples.push(sample);
        pace(frame_start, period);
    }
    Ok(Trial {
        round,
        position,
        elapsed_ms: start.elapsed().as_secs_f64() * 1_000.0,
        samples,
    })
}

fn latency(trials: &[Trial]) -> Distribution {
    statistics(trials.iter().map(|trial| {
        trial
            .samples
            .iter()
            .map(|sample| sample.latency_us)
            .collect()
    }))
    .unwrap()
}

fn incremental(trials: &[Trial], base: &[Trial]) -> Distribution {
    statistics(trials.iter().zip(base).map(|(trial, baseline)| {
        trial
            .samples
            .iter()
            .zip(&baseline.samples)
            .map(|(sample, baseline)| sample.latency_us - baseline.latency_us)
            .collect()
    }))
    .unwrap()
}

fn stage_timings(trials: &[Trial]) -> BTreeMap<String, Distribution> {
    let names: BTreeSet<_> = trials
        .iter()
        .flat_map(|trial| trial.samples.iter())
        .flat_map(|sample| sample.stages.keys())
        .collect();
    names
        .into_iter()
        .map(|name| {
            let measure = statistics(trials.iter().map(|trial| {
                trial
                    .samples
                    .iter()
                    .filter_map(|sample| sample.stages.get(name).copied())
                    .collect()
            }))
            .unwrap();
            (name.clone(), measure)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn run_group(
    name: &str,
    endpoint: &str,
    baseline: Option<&str>,
    arms: &mut [Arm<'_>],
    rounds: usize,
    seed: u64,
    reps: usize,
    hz: u32,
) -> AnyResult<Group> {
    if arms.is_empty() || rounds == 0 || reps == 0 || hz == 0 {
        return Err(format!("{name} needs arms, rounds, samples and a nonzero rate").into());
    }
    let names: BTreeSet<_> = arms.iter().map(|arm| arm.name).collect();
    if names.len() != arms.len() {
        return Err(format!("{name} has duplicate arm names").into());
    }
    let baseline_index = baseline
        .map(|reference| {
            arms.iter()
                .position(|arm| arm.name == reference)
                .ok_or_else(|| format!("{name} has no baseline arm named {reference}"))
        })
        .transpose()?;
    let period = Duration::from_secs_f64(1.0 / f64::from(hz));
    let mut state = seed;
    let mut order = Vec::with_capacity(rounds);
    let mut trials: Vec<_> = (0..arms.len())
        .map(|_| Vec::with_capacity(rounds))
        .collect();
    for round in 1..=rounds {
        let indices = draw_order(arms.len(), &mut state);
        order.push(
            indices
                .iter()
                .map(|&index| arms[index].name.to_owned())
                .collect(),
        );
        for (position, index) in indices.into_iter().enumerate() {
            let arm = &mut arms[index];
            eprintln!("{name} | round {round}/{rounds} | {}", arm.name);
            trials[index].push(trial(arm, round, position + 1, reps, period)?);
        }
    }
    let paired: Vec<_> = (0..arms.len())
        .map(|index| {
            baseline_index
                .filter(|&base| base != index)
                .map(|base| incremental(&trials[index], &trials[base]))
        })
        .collect();
    let arms = arms
        .iter()
        .zip(trials)
        .zip(paired)
        .map(|((arm, trials), incremental_us)| ArmResult {
            name: arm.name.to_owned(),
            latency_us: latency(&trials),
            incremental_us,
            cpu_cycles: statistics(trials.iter().map(|trial| {
                trial
                    .samples
                    .iter()
                    .filter_map(|sample| sample.cpu_cycles.map(|cycles| cycles as f64))
                    .collect()
            })),
            stages_us: stage_timings(&trials),
            trials,
        })
        .collect();
    Ok(Group {
        name: name.to_owned(),
        endpoint: endpoint.to_owned(),
        baseline: baseline.map(str::to_owned),
        seed,
        rounds,
        reps,
        hz,
        warmup: WARMUP,
        order,
        arms,
    })
}

fn range(distribution: &Distribution) -> String {
    format!("{:.1} to {:.1}", distribution.p50_min, distribution.p50_max)
}

fn markdown_group(out: &mut String, group: &Group) {
    writeln!(out, "### {}\n\n{}\n", group.name, group.endpoint).unwrap();
    if let Some(baseline) = &group.baseline {
        writeln!(out, "Paired deltas use **{baseline}** as the baseline.\n").unwrap();
    }
    out.push_str("| Arm | p50 (µs) | p95 (µs) | Trial p50 range (µs) | Paired delta p50 (µs) | Paired delta p95 (µs) | Thread cycles p50 |\n| --- | --- | --- | --- | --- | --- | --- |\n");
    for arm in &group.arms {
        let time = &arm.latency_us;
        let (delta_p50, delta_p95) = arm
            .incremental_us
            .as_ref()
            .map(|delta| (format!("{:+.1}", delta.p50), format!("{:+.1}", delta.p95)))
            .unwrap_or_else(|| ("—".into(), "—".into()));
        let cycles = arm
            .cpu_cycles
            .as_ref()
            .map_or_else(|| "—".into(), |cycles| format!("{:.0}", cycles.p50));
        writeln!(
            out,
            "| {} | {:.1} | {:.1} | {} | {} | {} | {} |",
            arm.name,
            time.p50,
            time.p95,
            range(time),
            delta_p50,
            delta_p95,
            cycles,
        )
        .unwrap();
    }
    out.push('\n');
    if group.arms.iter().any(|arm| !arm.stages_us.is_empty()) {
        out.push_str("| Arm | Stage | p50 (µs) | p95 (µs) | Observations |\n| --- | --- | --- | --- | --- |\n");
        for arm in &group.arms {
            for (name, stage) in &arm.stages_us {
                writeln!(
                    out,
                    "| {} | {} | {:.1} | {:.1} | {} |",
                    arm.name, name, stage.p50, stage.p95, stage.samples,
                )
                .unwrap();
            }
        }
        out.push('\n');
    }
}

pub fn write_report(
    output: &Path,
    metadata: serde_json::Value,
    groups: &[Group],
    cache: &CacheReport,
) -> AnyResult<()> {
    #[derive(Serialize)]
    struct Document<'a> {
        metadata: &'a serde_json::Value,
        groups: &'a [Group],
        cache: &'a CacheReport,
    }
    let data = Document {
        metadata: &metadata,
        groups,
        cache,
    };
    let mut out = String::from("# RapidShot change detection follow-up\n\n");
    if let Some(note) = metadata.get("summary").and_then(|value| value.as_str()) {
        writeln!(out, "{note}\n").unwrap();
    }
    out.push_str("Trials replayed the same input sequence. Each round shuffled the arms using a recorded seed. Warmup precedes each paced trial. Table values are medians of trial percentiles. Paired deltas subtract the baseline at the same round and replay index before calculating percentiles. CPU cycles are measured on the submitting thread. Samples, run order and environment details are in [the raw results](results.json).\n\n");
    for group in groups {
        markdown_group(&mut out, group);
    }
    cache.markdown(&mut out);
    for group in &cache.groups {
        markdown_group(&mut out, group);
    }

    let mut json = serde_json::to_vec_pretty(&data)?;
    json.push(b'\n');
    fs::create_dir_all(output)?;
    fs::write(output.join("results.json"), json)?;
    fs::write(output.join("report.md"), out)?;
    Ok(())
}
