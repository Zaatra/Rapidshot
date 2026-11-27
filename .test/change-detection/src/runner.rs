use crate::{AnyResult, Case, FramePair, Gpu};
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};
use windows::Win32::Foundation::FILETIME;
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

const FRAME_PERIOD: Duration = Duration::from_nanos(16_666_667);
const WARMUP_FRAMES: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum Algorithm {
    DirtyRects,
    Hybrid,
    FullScreen,
}

impl Algorithm {
    const ALL: [Self; 3] = [Self::DirtyRects, Self::Hybrid, Self::FullScreen];

    fn index(self) -> usize {
        self as usize
    }

    fn name(self) -> &'static str {
        match self {
            Self::DirtyRects => "Dirty rects",
            Self::Hybrid => "Hybrid",
            Self::FullScreen => "Full screen",
        }
    }

    fn candidates<'a>(self, pair: &'a FramePair, all_tiles: &'a [u32]) -> &'a [u32] {
        match self {
            Self::DirtyRects => &pair.metadata_tiles,
            Self::Hybrid => &pair.hybrid_tiles,
            Self::FullScreen => all_tiles,
        }
    }
}

#[derive(Clone, Copy, Default, Serialize)]
struct Counts {
    pairs: usize,
    candidate_tiles: usize,
    selected_tiles: usize,
    expected_tiles: usize,
    missed_tiles: usize,
    excess_tiles: usize,
    pairs_with_missed_tiles: usize,
    pairs_with_excess_tiles: usize,
    needed_decisions: usize,
    delivered_needed_decisions: usize,
    skipped_unchanged: usize,
    redundant_decisions: usize,
    missed_decisions: usize,
}

impl Counts {
    fn compare(candidates: usize, selected: &[u32], expected: &[u32]) -> Self {
        let missed = expected
            .iter()
            .filter(|tile| selected.binary_search(tile).is_err())
            .count();
        let excess = selected
            .iter()
            .filter(|tile| expected.binary_search(tile).is_err())
            .count();
        let needed = !expected.is_empty();
        let selected_any = !selected.is_empty();
        Self {
            pairs: 1,
            candidate_tiles: candidates,
            selected_tiles: selected.len(),
            expected_tiles: expected.len(),
            missed_tiles: missed,
            excess_tiles: excess,
            pairs_with_missed_tiles: usize::from(missed > 0),
            pairs_with_excess_tiles: usize::from(excess > 0),
            needed_decisions: usize::from(needed),
            delivered_needed_decisions: usize::from(needed && selected_any),
            skipped_unchanged: usize::from(!needed && !selected_any),
            redundant_decisions: usize::from(!needed && selected_any),
            missed_decisions: usize::from(needed && !selected_any),
        }
    }

    fn add(&mut self, counts: Self) {
        self.pairs += counts.pairs;
        self.candidate_tiles += counts.candidate_tiles;
        self.selected_tiles += counts.selected_tiles;
        self.expected_tiles += counts.expected_tiles;
        self.missed_tiles += counts.missed_tiles;
        self.excess_tiles += counts.excess_tiles;
        self.pairs_with_missed_tiles += counts.pairs_with_missed_tiles;
        self.pairs_with_excess_tiles += counts.pairs_with_excess_tiles;
        self.needed_decisions += counts.needed_decisions;
        self.delivered_needed_decisions += counts.delivered_needed_decisions;
        self.skipped_unchanged += counts.skipped_unchanged;
        self.redundant_decisions += counts.redundant_decisions;
        self.missed_decisions += counts.missed_decisions;
    }
}

#[derive(Serialize)]
struct PairReport {
    pair_index: usize,
    fallback: Option<&'static str>,
    by_algorithm: [Counts; 3],
}

#[derive(Serialize)]
struct Trial {
    round: usize,
    order_position: usize,
    detector_latency_us: Vec<f64>,
    p50_us: f64,
    p95_us: f64,
    process_cpu_ms: Option<f64>,
    paced_elapsed_ms: f64,
    counts: Counts,
}

#[derive(Serialize)]
struct Timing {
    p50_us: f64,
    p95_us: f64,
    p50_min_us: f64,
    p50_max_us: f64,
    process_cpu_us_per_frame: Option<f64>,
}

#[derive(Serialize)]
struct ArmReport {
    algorithm: Algorithm,
    fixture_counts: Counts,
    timing: Timing,
    trials: Vec<Trial>,
}

#[derive(Serialize)]
struct CaseReport {
    name: String,
    source: String,
    fixture_pairs: usize,
    round_order: Vec<[Algorithm; 3]>,
    fallback_pairs: BTreeMap<&'static str, usize>,
    pair_results: Vec<PairReport>,
    arms: Vec<ArmReport>,
}

#[derive(Serialize)]
pub struct Report {
    seed: u64,
    rounds: usize,
    reps_per_arm_per_round: usize,
    frame_period_ms: f64,
    warmup_frames_per_trial: usize,
    tiles_per_frame: usize,
    pair_result_algorithm_order: [Algorithm; 3],
    cases: Vec<CaseReport>,
}

#[derive(Serialize)]
pub struct ReportContext {
    pub commit: String,
    pub adapter: String,
    pub width: u32,
    pub height: u32,
    pub tile_size: u32,
    pub live: serde_json::Value,
}

fn random_u64(state: &mut u64) -> u64 {
    // SplitMix64 gives a stable schedule for any user-supplied seed.
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut value = *state;
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn randomized_order(rounds: usize, state: &mut u64) -> Vec<[Algorithm; 3]> {
    let [dirty, hybrid, full] = Algorithm::ALL;
    let mut orders = [
        [dirty, hybrid, full],
        [dirty, full, hybrid],
        [hybrid, dirty, full],
        [hybrid, full, dirty],
        [full, dirty, hybrid],
        [full, hybrid, dirty],
    ];
    let mut order = Vec::with_capacity(rounds);
    while order.len() < rounds {
        for index in (1..orders.len()).rev() {
            let other = (random_u64(state) % (index as u64 + 1)) as usize;
            orders.swap(index, other);
        }
        let take = (rounds - order.len()).min(orders.len());
        order.extend_from_slice(&orders[..take]);
    }
    order
}

fn detect(
    gpu: &Gpu,
    algorithm: Algorithm,
    pair: &FramePair,
    candidates: &[u32],
    result: &mut Vec<u32>,
) -> AnyResult<()> {
    result.clear();
    match algorithm {
        Algorithm::DirtyRects => result.extend_from_slice(candidates),
        _ => gpu.compare(&pair.previous, &pair.current, candidates, result)?,
    }
    Ok(())
}

fn validate_mask(mask: &[u32], tiles: usize, name: &str) -> AnyResult<()> {
    if mask.windows(2).any(|pair| pair[0] >= pair[1])
        || mask.last().is_some_and(|&tile| tile as usize >= tiles)
    {
        return Err(format!("{name} contains unordered, duplicate or invalid tile IDs").into());
    }
    Ok(())
}

fn exact_result(
    algorithm: Algorithm,
    case: &Case,
    index: usize,
    result: &[u32],
    expected: &[u32],
) -> AnyResult<()> {
    if algorithm == Algorithm::DirtyRects || result == expected {
        return Ok(());
    }
    let missed: Vec<_> = expected
        .iter()
        .filter(|tile| result.binary_search(tile).is_err())
        .take(8)
        .collect();
    let excess: Vec<_> = result
        .iter()
        .filter(|tile| expected.binary_search(tile).is_err())
        .take(8)
        .collect();
    Err(format!(
        "{} failed correctness for {} pair {} (fallback {:?}): expected {} tiles, got {}; first missing IDs {:?}, first excess IDs {:?}",
        algorithm.name(),
        case.name,
        index,
        case.pairs[index].fallback,
        expected.len(),
        result.len(),
        missed,
        excess,
    )
    .into())
}

fn check_case(gpu: &Gpu, case: &Case, all_tiles: &[u32]) -> AnyResult<Vec<PairReport>> {
    if case.pairs.is_empty() {
        return Err(format!("{} has no pairs to replay", case.name).into());
    }
    let mut pair_results = Vec::with_capacity(case.pairs.len());
    let mut result = Vec::with_capacity(all_tiles.len());
    for (index, pair) in case.pairs.iter().enumerate() {
        validate_mask(
            &pair.expected,
            all_tiles.len(),
            &format!("{} pair {index} CPU oracle", case.name),
        )?;
        let mut by_algorithm = [Counts::default(); 3];
        for algorithm in Algorithm::ALL {
            let candidates = algorithm.candidates(pair, all_tiles);
            let label = format!("{} pair {index} {}", case.name, algorithm.name());
            validate_mask(candidates, all_tiles.len(), &format!("{label} candidates"))?;
            detect(gpu, algorithm, pair, candidates, &mut result)
                .map_err(|error| format!("{label}: {error}"))?;
            validate_mask(&result, all_tiles.len(), &format!("{label} result"))?;
            exact_result(algorithm, case, index, &result, &pair.expected)?;
            by_algorithm[algorithm.index()] =
                Counts::compare(candidates.len(), &result, &pair.expected);
        }
        pair_results.push(PairReport {
            pair_index: index,
            fallback: pair.fallback,
            by_algorithm,
        });
    }
    Ok(pair_results)
}

fn percentile_sorted(sample: &[f64], fraction: f64) -> f64 {
    let position = (sample.len() - 1) as f64 * fraction;
    let left = position.floor() as usize;
    let right = position.ceil() as usize;
    sample[left] + (sample[right] - sample[left]) * (position - left as f64)
}

fn ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn pace(start: Instant) {
    if let Some(remaining) = FRAME_PERIOD.checked_sub(start.elapsed()) {
        thread::sleep(remaining);
    }
}

fn process_ticks() -> Option<u64> {
    fn ticks(time: FILETIME) -> u64 {
        (u64::from(time.dwHighDateTime) << 32) | u64::from(time.dwLowDateTime)
    }

    let mut creation = FILETIME::default();
    let mut exit = FILETIME::default();
    let mut kernel = FILETIME::default();
    let mut user = FILETIME::default();
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    }
    .ok()?;
    Some(ticks(kernel) + ticks(user))
}

struct Replay<'a> {
    gpu: &'a Gpu,
    case: &'a Case,
    all_tiles: &'a [u32],
    pair_results: &'a [PairReport],
    reps: usize,
}

impl Replay<'_> {
    fn trial(&self, algorithm: Algorithm, round: usize, order_position: usize) -> AnyResult<Trial> {
        let pairs = &self.case.pairs;
        let mut result = Vec::with_capacity(self.all_tiles.len());
        for repeat in 0..WARMUP_FRAMES {
            let start = Instant::now();
            let pair = &pairs[repeat % pairs.len()];
            let candidates = algorithm.candidates(pair, self.all_tiles);
            detect(self.gpu, algorithm, pair, candidates, &mut result)?;
            pace(start);
        }

        let mut detector_latency_us = Vec::with_capacity(self.reps);
        let cpu_start = process_ticks();
        let trial_start = Instant::now();
        for repeat in 0..self.reps {
            let start = Instant::now();
            let index = repeat % pairs.len();
            let pair = &pairs[index];
            let candidates = algorithm.candidates(pair, self.all_tiles);
            detect(self.gpu, algorithm, pair, candidates, &mut result).map_err(|error| {
                format!(
                    "{} {} round {round} pair {index}: {error}",
                    self.case.name,
                    algorithm.name()
                )
            })?;
            detector_latency_us.push(start.elapsed().as_secs_f64() * 1_000_000.0);
            exact_result(algorithm, self.case, index, &result, &pair.expected)?;
            pace(start);
        }
        let paced_elapsed_ms = ms(trial_start.elapsed());
        let process_cpu_ms = cpu_start
            .zip(process_ticks())
            .and_then(|(start, end)| end.checked_sub(start))
            .map(|ticks| ticks as f64 / 10_000.0);

        let mut sorted = detector_latency_us.clone();
        sorted.sort_by(f64::total_cmp);
        let mut counts = Counts::default();
        for sample in 0..self.reps {
            counts.add(self.pair_results[sample % pairs.len()].by_algorithm[algorithm.index()]);
        }
        Ok(Trial {
            round,
            order_position,
            detector_latency_us,
            p50_us: percentile_sorted(&sorted, 0.50),
            p95_us: percentile_sorted(&sorted, 0.95),
            process_cpu_ms,
            paced_elapsed_ms,
            counts,
        })
    }
}

fn timing(trials: &[Trial], reps: usize) -> Timing {
    let mut p50: Vec<_> = trials.iter().map(|trial| trial.p50_us).collect();
    let mut p95: Vec<_> = trials.iter().map(|trial| trial.p95_us).collect();
    let mut cpu: Vec<_> = trials
        .iter()
        .filter_map(|trial| trial.process_cpu_ms)
        .map(|value| value * 1_000.0 / reps as f64)
        .collect();
    p50.sort_by(f64::total_cmp);
    p95.sort_by(f64::total_cmp);
    cpu.sort_by(f64::total_cmp);
    Timing {
        p50_us: percentile_sorted(&p50, 0.50),
        p95_us: percentile_sorted(&p95, 0.50),
        p50_min_us: p50[0],
        p50_max_us: p50[p50.len() - 1],
        process_cpu_us_per_frame: (!cpu.is_empty()).then(|| percentile_sorted(&cpu, 0.50)),
    }
}

pub fn run(
    gpu: &Gpu,
    cases: &[Case],
    all_tiles: &[u32],
    rounds: usize,
    seed: u64,
    reps: usize,
) -> AnyResult<Report> {
    if cases.is_empty() || all_tiles.is_empty() || rounds == 0 || reps == 0 {
        return Err("a benchmark needs cases, tiles, rounds and samples".into());
    }
    let checks: Vec<_> = cases
        .iter()
        .map(|case| check_case(gpu, case, all_tiles))
        .collect::<AnyResult<_>>()?;
    eprintln!(
        "Correctness passed for both exact detectors on all {} workload(s)",
        cases.len()
    );

    let mut state = seed;
    let mut results = Vec::with_capacity(cases.len());
    for (case, pair_results) in cases.iter().zip(checks) {
        let round_order = randomized_order(rounds, &mut state);
        let replay = Replay {
            gpu,
            case,
            all_tiles,
            pair_results: &pair_results,
            reps,
        };
        let mut trials = Algorithm::ALL.map(|_| Vec::with_capacity(rounds));
        for (round, order) in round_order.iter().enumerate() {
            for (position, &algorithm) in order.iter().enumerate() {
                eprintln!(
                    "{} | round {}/{} | {}",
                    case.name,
                    round + 1,
                    rounds,
                    algorithm.name()
                );
                trials[algorithm.index()].push(replay.trial(algorithm, round + 1, position + 1)?);
            }
        }

        let mut fallback_pairs = BTreeMap::new();
        for pair in &pair_results {
            if let Some(reason) = pair.fallback {
                *fallback_pairs.entry(reason).or_insert(0) += 1;
            }
        }
        let arms = Algorithm::ALL
            .into_iter()
            .zip(trials)
            .map(|(algorithm, trials)| {
                let mut fixture_counts = Counts::default();
                for pair in &pair_results {
                    fixture_counts.add(pair.by_algorithm[algorithm.index()]);
                }
                ArmReport {
                    algorithm,
                    fixture_counts,
                    timing: timing(&trials, reps),
                    trials,
                }
            })
            .collect();
        results.push(CaseReport {
            name: case.name.clone(),
            source: case.source.clone(),
            fixture_pairs: case.pairs.len(),
            round_order,
            fallback_pairs,
            pair_results,
            arms,
        });
    }
    Ok(Report {
        seed,
        rounds,
        reps_per_arm_per_round: reps,
        frame_period_ms: ms(FRAME_PERIOD),
        warmup_frames_per_trial: WARMUP_FRAMES,
        tiles_per_frame: all_tiles.len(),
        pair_result_algorithm_order: Algorithm::ALL,
        cases: results,
    })
}

fn markdown(report: &Report, context: &ReportContext) -> String {
    let mut out = String::new();
    writeln!(out, "# RapidShot change detection benchmark\n").unwrap();
    writeln!(
        out,
        "RapidShot `{}` on {}. Each frame was {} × {} BGRA with {} × {} tiles and {} tiles per frame.\n",
        context.commit,
        context.adapter,
        context.width,
        context.height,
        context.tile_size,
        context.tile_size,
        report.tiles_per_frame,
    )
    .unwrap();
    writeln!(
        out,
        "Each workload used {} rounds with {} measured decisions per method per round. Each round ran all three methods in a shuffled order (seed `{}`). Orders were drawn without replacement from the six permutations and reshuffled if exhausted. Each trial used {} untimed warmup decisions and a target period of {:.3} ms per decision. The frame pairs and their candidate lists were prepared once and replayed on the same D3D11 device for all methods.\n",
        report.rounds,
        report.reps_per_arm_per_round,
        report.seed,
        report.warmup_frames_per_trial,
        report.frame_period_ms,
    )
    .unwrap();
    out.push_str("Latency covers detector calls through a CPU-readable result, including GPU completion and readback for the exact methods. The dirty-rects baseline emits the metadata tile list. Preparation, acquisition, previous-frame texture maintenance, and downstream actions sit outside the timing window. Process CPU time includes replay loop overhead and exact result checks, and its OS accounting granularity can obscure small differences. This benchmark measures replayed decisions. Idle replay timing does not measure a desktop's frame delivery rate.\n\n");
    out.push_str("## Detector latency\n\nThe table gives the median of trial p50 and p95 values. The range shows the lowest and highest trial p50.\n\n");
    out.push_str("| Workload | Method | p50 (µs) | Trial p50 range (µs) | p95 (µs) | Process CPU (µs/decision) |\n| --- | --- | --- | --- | --- | --- |\n");
    for case in &report.cases {
        for arm in &case.arms {
            let time = &arm.timing;
            let cpu = time
                .process_cpu_us_per_frame
                .map_or_else(|| "unavailable".to_owned(), |value| format!("{value:.1}"));
            writeln!(
                out,
                "| {} | {} | {:.3} | {:.3}–{:.3} | {:.3} | {} |",
                case.name,
                arm.algorithm.name(),
                time.p50_us,
                time.p50_min_us,
                time.p50_max_us,
                time.p95_us,
                cpu,
            )
            .unwrap();
        }
    }
    out.push_str("\n## Detection accuracy and work\n\nCounts cover each fixture pair once. Candidate coverage is the fraction of frame tiles examined or selected from metadata, averaged over pairs. Extra tiles are selected tiles whose pixels did not change. A redundant wake-up is a selected frame whose tiles were all unchanged. A missed wake-up is a changed frame with no selected tiles. Both exact methods matched the independent CPU tile oracle on every fixture before timing and on every timed sample.\n\n");
    out.push_str("| Workload | Method | Pairs | Candidate coverage | Changed tiles | Extra tiles | Missed tiles | Unchanged pairs skipped | Redundant wake-ups | Missed wake-ups |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for case in &report.cases {
        for arm in &case.arms {
            let counts = arm.fixture_counts;
            let coverage = 100.0 * counts.candidate_tiles as f64
                / (counts.pairs * report.tiles_per_frame) as f64;
            writeln!(
                out,
                "| {} | {} | {} | {:.2}% | {} | {} | {} | {} | {} | {} |",
                case.name,
                arm.algorithm.name(),
                counts.pairs,
                coverage,
                counts.expected_tiles,
                counts.excess_tiles,
                counts.missed_tiles,
                counts.skipped_unchanged,
                counts.redundant_decisions,
                counts.missed_decisions,
            )
            .unwrap();
        }
    }
    out.push_str("\n## Fixture sources\n\n");
    for case in &report.cases {
        writeln!(out, "- **{}**. {}", case.name, case.source).unwrap();
        for (reason, count) in &case.fallback_pairs {
            writeln!(out, "  - Fallback `{reason}` on {count} pair(s).").unwrap();
        }
    }
    out.push_str("\n[Raw results](results.json) include the live-capture context, the recorded order of every round, and latency samples in replay order. Sample index modulo fixture pair count gives the zero-based fixture pair index.\n");
    out
}

pub fn write_report(report: &Report, output: &Path, context: ReportContext) -> AnyResult<()> {
    #[derive(Serialize)]
    struct Document<'a> {
        context: &'a ReportContext,
        #[serde(flatten)]
        report: &'a Report,
    }

    fs::create_dir_all(output)?;
    let document = Document {
        context: &context,
        report,
    };
    let mut json = serde_json::to_vec_pretty(&document)?;
    json.push(b'\n');
    fs::write(output.join("results.json"), json)?;
    fs::write(output.join("report.md"), markdown(report, &context))?;
    Ok(())
}
