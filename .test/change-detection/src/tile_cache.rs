use crate::followup::{run_group, thread_cycles, Arm, Group, Sample};
use crate::AnyResult;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write as _;
use std::time::Instant;

const SIDE: usize = 128;
const CACHE_SIZES: [(usize, &str); 3] = [(0, "K=0"), (1, "K=1"), (4, "K=4")];

#[derive(Clone, Copy, PartialEq, Eq)]
struct Identity {
    source: u32,
    format: u32,
    width: u32,
    height: u32,
    tile_size: u32,
    task_config: u32,
}

const SOURCE: Identity = Identity {
    source: 0,
    format: 87, // DXGI_FORMAT_B8G8R8A8_UNORM
    width: SIDE as u32,
    height: SIDE as u32,
    tile_size: SIDE as u32,
    task_config: 0,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TaskOutput {
    histogram: [u32; 16],
    horizontal_gradient: u32,
    bright_pixels: u32,
}

fn classify_tile(pixels: &[u32], width: usize, config: u32) -> TaskOutput {
    let mut histogram = [0; 16];
    let mut horizontal_gradient = 0;
    let mut bright_pixels = 0;
    for row in pixels.chunks_exact(width) {
        let mut previous = 0u32;
        for (column, &pixel) in row.iter().enumerate() {
            let red = (pixel >> 16) & 255;
            let green = (pixel >> 8) & 255;
            let blue = pixel & 255;
            let luma = (77 * red + 150 * green + 29 * blue) >> 8;
            histogram[(luma >> 4) as usize] += 1;
            if column > 0 {
                horizontal_gradient += luma.abs_diff(previous);
            }
            bright_pixels += u32::from(luma >= 144 + config);
            previous = luma;
        }
    }
    TaskOutput {
        histogram,
        horizontal_gradient,
        bright_pixels,
    }
}

// The digest stands in for a fingerprint already delivered to the consumer.
// It indexes the cache. Equal digest values still require exact pixel equality.
fn digest(pixels: &[u32]) -> u64 {
    pixels.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &pixel| {
        (hash ^ u64::from(pixel)).wrapping_mul(0x0100_0000_01b3)
    })
}

struct Entry {
    fingerprint: u64,
    pixels: Box<[u32]>,
    output: TaskOutput,
}

struct Cache {
    capacity: usize,
    identity: Option<Identity>,
    tiles: Vec<VecDeque<Entry>>,
}

#[derive(Default)]
struct Decision {
    hit: bool,
    evicted: bool,
    invalidated: bool,
    collisions: usize,
    rejected_first_digest_output: Option<TaskOutput>,
}

impl Decision {
    fn hash_only_would_return_wrong_output(&self, expected: TaskOutput) -> bool {
        self.rejected_first_digest_output
            .is_some_and(|cached| cached != expected)
    }
}

impl Cache {
    fn new(capacity: usize, tiles: usize) -> Self {
        Self {
            capacity,
            identity: None,
            tiles: (0..tiles).map(|_| VecDeque::new()).collect(),
        }
    }

    fn clear(&mut self) {
        self.identity = None;
        self.tiles.iter_mut().for_each(VecDeque::clear);
    }

    fn retained_pixels(&self) -> usize {
        self.tiles
            .iter()
            .flatten()
            .map(|entry| std::mem::size_of_val(entry.pixels.as_ref()))
            .sum()
    }

    fn evaluate(
        &mut self,
        identity: Identity,
        tile: usize,
        width: usize,
        fingerprint: u64,
        pixels: &[u32],
    ) -> (TaskOutput, Decision) {
        let mut decision = Decision::default();
        if self.capacity == 0 {
            return (classify_tile(pixels, width, identity.task_config), decision);
        }
        if self.identity != Some(identity) {
            decision.invalidated = self.identity.is_some();
            self.clear();
            self.identity = Some(identity);
        }
        let history = &mut self.tiles[tile];
        for index in 0..history.len() {
            let entry = &history[index];
            if entry.fingerprint != fingerprint {
                continue;
            }
            if entry.pixels.as_ref() == pixels {
                let entry = history.remove(index).expect("cache entry was located");
                let output = entry.output;
                history.push_front(entry);
                decision.hit = true;
                return (output, decision);
            }
            decision.collisions += 1;
            decision
                .rejected_first_digest_output
                .get_or_insert(entry.output);
        }

        let output = classify_tile(pixels, width, identity.task_config);
        if history.len() == self.capacity {
            history.pop_back();
            decision.evicted = true;
        }
        history.push_front(Entry {
            fingerprint,
            pixels: pixels.into(),
            output,
        });
        (output, decision)
    }
}

struct StatePixels {
    pixels: Vec<u32>,
    fingerprint: u64,
    reference: TaskOutput,
}

fn tile(state: usize) -> StatePixels {
    let mut pixels = vec![0xff24_2c38; SIDE * SIDE];
    let caret_width = if state == 0 { 0 } else { state + 2 };
    for y in 19..65 {
        for x in 23..23 + caret_width {
            pixels[y * SIDE + x] = 0xfff3_dc90;
        }
    }
    let fingerprint = digest(&pixels);
    let reference = classify_tile(&pixels, SIDE, SOURCE.task_config);
    StatePixels {
        pixels,
        fingerprint,
        reference,
    }
}

#[derive(Default, Serialize)]
struct Counts {
    eligible_changed_tiles: usize,
    verified_hits: usize,
    task_runs: usize,
    evictions: usize,
    invalidations: usize,
    rejected_digest_candidates: usize,
    hash_only_wrong_reuses: usize,
    max_retained_pixel_bytes: usize,
}

impl Counts {
    fn add(&mut self, decision: Decision, retained: usize, expected: TaskOutput) {
        self.eligible_changed_tiles += 1;
        self.verified_hits += usize::from(decision.hit);
        self.task_runs += usize::from(!decision.hit);
        self.evictions += usize::from(decision.evicted);
        self.invalidations += usize::from(decision.invalidated);
        self.rejected_digest_candidates += decision.collisions;
        self.hash_only_wrong_reuses +=
            usize::from(decision.hash_only_would_return_wrong_output(expected));
        self.max_retained_pixel_bytes = self.max_retained_pixel_bytes.max(retained);
    }
}

struct Replay {
    cache: Cache,
    counts: Counts,
}

struct Case {
    name: &'static str,
    states: Vec<StatePixels>,
    source_after_steps: Option<usize>,
}

impl Case {
    fn expected_hits(&self, capacity: usize, rounds: usize, reps: usize) -> usize {
        if capacity < self.states.len() {
            return 0;
        }
        let segment = self.source_after_steps.unwrap_or(reps);
        let cold = (reps / segment) * self.states.len().min(segment)
            + self.states.len().min(reps % segment);
        rounds * (reps - cold)
    }

    fn sample(&self, state: &mut Replay, index: usize, warmup: bool) -> AnyResult<Sample> {
        if index == 0 {
            state.cache.clear();
        }
        let input = &self.states[index % self.states.len()];
        let mut identity = SOURCE;
        if let Some(interval) = self.source_after_steps {
            identity.source = (index / interval) as u32;
        }

        let cpu_start = thread_cycles();
        let start = Instant::now();
        let (output, decision) =
            state
                .cache
                .evaluate(identity, 0, SIDE, input.fingerprint, &input.pixels);
        let latency_us = start.elapsed().as_secs_f64() * 1_000_000.0;
        let cpu_cycles = cpu_start
            .zip(thread_cycles())
            .and_then(|(begin, end)| end.checked_sub(begin));
        if output != input.reference {
            return Err(format!(
                "{} returned a stale task result at sample {index}",
                self.name
            )
            .into());
        }
        let mut stages = BTreeMap::new();
        let stage = if decision.hit {
            "verified cache hit"
        } else {
            "lookup plus task on miss"
        };
        stages.insert(stage.into(), latency_us);
        if !warmup {
            state
                .counts
                .add(decision, state.cache.retained_pixels(), input.reference);
        }
        Ok(Sample {
            latency_us,
            cpu_cycles,
            stages,
        })
    }
}

#[derive(Serialize)]
struct Row {
    workload: &'static str,
    capacity_per_tile: usize,
    counts: Counts,
}

#[derive(Serialize)]
struct Correctness {
    injected_digest_candidates_rejected: usize,
    injected_hash_only_wrong_reuses: usize,
    byte_verified_return_hits: usize,
    reset_after_source_change: bool,
    reset_after_task_change: bool,
    independent_tile_slots: bool,
}

fn check_guards() -> AnyResult<Correctness> {
    let a = tile(0);
    let b = tile(1);
    if a.reference == b.reference {
        return Err("collision fixture needs distinct task outputs".into());
    }
    let mut cache = Cache::new(4, 2);
    let fingerprint = 0x5abc;
    let (first, first_action) = cache.evaluate(SOURCE, 0, SIDE, fingerprint, &a.pixels);
    let (miss, collision) = cache.evaluate(SOURCE, 0, SIDE, fingerprint, &b.pixels);
    let (hit, returning) = cache.evaluate(SOURCE, 0, SIDE, fingerprint, &a.pixels);
    let (other_tile, other) = cache.evaluate(SOURCE, 1, SIDE, fingerprint, &a.pixels);
    let next_source = Identity {
        source: 3,
        ..SOURCE
    };
    let (after_source, source) = cache.evaluate(next_source, 0, SIDE, fingerprint, &a.pixels);
    let changed_task = Identity {
        task_config: 100,
        ..next_source
    };
    let (after_task, task) = cache.evaluate(changed_task, 0, SIDE, fingerprint, &a.pixels);
    let collision_would_be_wrong = collision.hash_only_would_return_wrong_output(b.reference);
    let return_would_be_wrong = returning.hash_only_would_return_wrong_output(a.reference);
    if first != a.reference
        || first_action.hit
        || miss != b.reference
        || collision.hit
        || collision.collisions != 1
        || !collision_would_be_wrong
        || hit != a.reference
        || !returning.hit
        || returning.collisions != 1
        || !return_would_be_wrong
        || other_tile != a.reference
        || other.hit
        || after_source != a.reference
        || !source.invalidated
        || source.hit
        || after_task != classify_tile(&a.pixels, SIDE, changed_task.task_config)
        || !task.invalidated
        || task.hit
    {
        return Err("tile cache identity or injected digest collision correctness failed".into());
    }
    Ok(Correctness {
        injected_digest_candidates_rejected: collision.collisions + returning.collisions,
        injected_hash_only_wrong_reuses: usize::from(collision_would_be_wrong)
            + usize::from(return_would_be_wrong),
        byte_verified_return_hits: usize::from(returning.hit),
        reset_after_source_change: source.invalidated,
        reset_after_task_change: task.invalidated,
        independent_tile_slots: !other.hit,
    })
}

#[derive(Serialize)]
pub struct CacheReport {
    task: &'static str,
    tile_size: usize,
    correctness: Correctness,
    workloads: Vec<Row>,
    pub(crate) groups: Vec<Group>,
}

impl CacheReport {
    pub(crate) fn markdown(&self, out: &mut String) {
        out.push_str("## Bounded tile-result cache\n\n");
        writeln!(out, "The task was {}. Each sample supplied a confirmed changed tile and an already-computed fingerprint. The timer covers lookup, byte verification, and the analysis and insert on a miss. Each measured trial started with an empty cache. Retained bytes count stored pixels. Task outputs and container allocations add to that footprint.\n", self.task).unwrap();
        out.push_str("| Workload | Entries per tile | Changed tiles | Task runs | Task runs avoided | Evictions | Source resets | Peak retained pixel bytes |\n| --- | --- | --- | --- | --- | --- | --- | --- |\n");
        for row in &self.workloads {
            let counts = &row.counts;
            writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} |",
                row.workload,
                row.capacity_per_tile,
                counts.eligible_changed_tiles,
                counts.task_runs,
                counts.verified_hits,
                counts.evictions,
                counts.invalidations,
                counts.max_retained_pixel_bytes,
            )
            .unwrap();
        }
        writeln!(out, "\nForced digest collisions would have returned stale output on {} requests with hash-only reuse. Byte verification rejected both entries, recomputed on the real miss, and reused the original tile on its return. Reset and tile-identity checks also passed.\n", self.correctness.injected_hash_only_wrong_reuses).unwrap();
    }
}

pub fn measure(rounds: usize, seed: u64, reps: usize, hz: u32) -> AnyResult<CacheReport> {
    let correctness = check_guards()?;
    let cases = [
        Case {
            name: "Two-state caret",
            states: (0..2).map(tile).collect(),
            source_after_steps: None,
        },
        Case {
            name: "Five-state cache churn",
            states: (0..5).map(tile).collect(),
            source_after_steps: None,
        },
        Case {
            name: "Two-state caret with source resets",
            states: (0..2).map(tile).collect(),
            source_after_steps: Some(10),
        },
    ];
    let mut groups = Vec::with_capacity(cases.len());
    let mut workloads = Vec::with_capacity(cases.len() * CACHE_SIZES.len());
    for (index, case) in cases.iter().enumerate() {
        let mut replays = CACHE_SIZES.map(|(capacity, _)| Replay {
            cache: Cache::new(capacity, 1),
            counts: Counts::default(),
        });
        let mut arms: Vec<_> = replays
            .iter_mut()
            .zip(CACHE_SIZES)
            .map(|(replay, (_, name))| Arm {
                name,
                run: Box::new(move |index, warmup| case.sample(replay, index, warmup)),
            })
            .collect();
        groups.push(run_group(
            case.name,
            "CPU tile-local result available from an already-available fingerprint and crop.",
            Some("K=0"),
            &mut arms,
            rounds,
            seed.wrapping_add(index as u64),
            reps,
            hz,
        )?);
        drop(arms);
        for (replay, (capacity_per_tile, _)) in replays.into_iter().zip(CACHE_SIZES) {
            let expected = case.expected_hits(capacity_per_tile, rounds, reps);
            if replay.counts.verified_hits != expected {
                return Err(format!(
                    "{} K={capacity_per_tile}: expected {expected} verified hits, got {}",
                    case.name, replay.counts.verified_hits
                )
                .into());
            }
            workloads.push(Row {
                workload: case.name,
                capacity_per_tile,
                counts: replay.counts,
            });
        }
    }
    Ok(CacheReport {
        task: "a tile-local luminance histogram, gradient, and brightness analysis proxy",
        tile_size: SIDE,
        correctness,
        workloads,
        groups,
    })
}
