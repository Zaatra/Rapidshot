use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::path::Path;
use std::rc::Rc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Serialize;
use serde_json::{json, Value};
use windows::core::BOOL;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D11::*;

use crate::consumer::{query, query_data, Consumer};
use crate::delivery::{retained_history_bytes, Delivery, Mode};
use crate::followup::{self, thread_cycles, Arm, Sample};
use crate::{
    fixed_pair, solid_frame, AnyResult, Case, Display, Extent, FramePair, Gpu, GpuFrame, InputCase,
    InputPair, Pixels, BASELINE, TILE_SIZE,
};

const HZ: u32 = 60;

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn sample(start: Instant, cycles: Option<u64>) -> Sample {
    let elapsed = micros(start.elapsed());
    Sample {
        latency_us: elapsed,
        cpu_cycles: cycles
            .zip(thread_cycles())
            .map(|(before, after)| after.saturating_sub(before)),
        stages: BTreeMap::new(),
    }
}

fn title(case: &Case, subject: &str) -> String {
    format!("{} / {subject}", case.name)
}

pub(super) fn add_validation_cases(inputs: &mut Vec<InputCase>, extent: Extent) {
    let mut a = solid_frame(extent, 0xff32_5276).to_vec();
    a[0] = 0xff19_3257;
    a[1] = 0xffe4_c69f;
    let mut b = a.clone();
    b.swap(0, 1);
    let mut c = b.clone();
    *c.last_mut().unwrap() ^= 0x0100_0000;
    let a: Pixels = a.into();
    let b: Pixels = b.into();
    let c: Pixels = c.into();
    let tiny = RECT {
        right: 2,
        bottom: 1,
        ..RECT::default()
    };
    let corner = RECT {
        left: extent.width as i32 - 1,
        top: extent.height as i32 - 1,
        ..extent.screen_rect()
    };
    inputs.push(InputCase {
        name: "fingerprint_order_and_edges",
        source: "Synthetic same colors at new positions, an alpha-only edge change, and identical pixels",
        pairs: vec![fixed_pair(&a, &b, true, vec![tiny]), fixed_pair(&b, &c, true, vec![corner]),
            fixed_pair(&c, &c, true, vec![corner])],
    });
}

fn needs_seed(pairs: &[InputPair], index: usize) -> bool {
    let prior = (index + pairs.len() - 1) % pairs.len();
    index == 0 || !Rc::ptr_eq(&pairs[prior].current, &pairs[index % pairs.len()].previous)
}

fn require_tiles(selected: &[u32], pair: &FramePair, label: &str) -> AnyResult<()> {
    if selected != pair.expected {
        return Err(format!("{label}: got {selected:?}, expected {:?}", pair.expected).into());
    }
    Ok(())
}

fn output_pixels(engine: &Delivery, expected: &[u32]) -> AnyResult<()> {
    match engine.delivered() {
        Some(delivered) if delivered == expected => Ok(()),
        _ => Err("CPU delivery pixel bytes differ from the replay input".into()),
    }
}

fn deliver_mode(mode: Mode) -> bool {
    !matches!(mode, Mode::Fingerprint)
}

fn decides(mode: Mode) -> bool {
    !matches!(mode, Mode::HardwareDelivery | Mode::ShaderDelivery)
}

#[derive(Default, Serialize)]
struct DeliveryCounts {
    comparisons: usize,
    decision_mismatches: usize,
    second_map_probes: usize,
    second_map_busy: usize,
}

fn timed_delivery<'a>(
    name: &'a str,
    engine: &'a RefCell<Delivery>,
    mode: Mode,
    case: &'a Case,
    input: &'a InputCase,
    extent: Extent,
    counts: &'a RefCell<BTreeMap<String, DeliveryCounts>>,
) -> Box<dyn FnMut(usize, bool) -> AnyResult<Sample> + 'a> {
    let mut changed = Vec::with_capacity(extent.tile_count() as usize);
    let key = format!("{} / {name}", case.name);
    Box::new(move |index, warmup| {
        let i = index % case.pairs.len();
        let pair = &case.pairs[i];
        let pixels = &input.pairs[i];
        let mut engine = engine.borrow_mut();
        if needs_seed(&input.pairs, index) {
            engine.reset(mode, &pair.previous, Some(&pixels.previous))?;
        }
        let cycles = thread_cycles();
        let start = Instant::now();
        engine.run(mode, &pair.current, &pair.hybrid_tiles, &mut changed)?;
        let mut report = sample(start, cycles);
        if let Some(us) = engine.cpu_digest_update_us() {
            report
                .stages
                .insert("cpu_digest_compare_and_update".into(), us);
        }
        if matches!(mode, Mode::CpuExact | Mode::Verify) {
            require_tiles(&changed, pair, "CPU delivery exact")?;
        }
        if !warmup && decides(mode) {
            let mut counts = counts.borrow_mut();
            let stats = counts.entry(key.clone()).or_default();
            stats.comparisons += 1;
            stats.decision_mismatches += usize::from(changed != pair.expected);
            if let Some(busy) = engine.second_map_waited() {
                stats.second_map_probes += 1;
                stats.second_map_busy += usize::from(busy);
            }
        }
        Ok(report)
    })
}

#[derive(Clone, Copy)]
enum Detector {
    Metadata,
    Resident,
    Owned,
    Full,
}

fn detect_arm<'a>(
    name: &'a str,
    kind: Detector,
    case: &'a Case,
    input: &'a InputCase,
    gpu: &'a Gpu,
    all_tiles: &'a [u32],
) -> AnyResult<Arm<'a>> {
    let history = gpu.history()?;
    let staging = gpu.staging_buffer()?;
    let mut selected = Vec::with_capacity(all_tiles.len());
    Ok(Arm {
        name,
        run: Box::new(move |index, _warmup| {
            let i = index % case.pairs.len();
            let pair = &case.pairs[i];
            let uses_history = matches!(kind, Detector::Owned | Detector::Full);
            if uses_history && needs_seed(&input.pairs, index) {
                gpu.seed_history(&history, &pair.previous)?;
            }
            let cycles = thread_cycles();
            let start = Instant::now();
            match kind {
                Detector::Metadata => {
                    selected.clear();
                    selected.extend_from_slice(&pair.metadata_tiles);
                    std::hint::black_box(&selected);
                }
                Detector::Resident => gpu.compare(
                    &pair.previous,
                    &pair.current,
                    &pair.hybrid_tiles,
                    &mut selected,
                )?,
                Detector::Owned | Detector::Full => {
                    let candidates = if matches!(kind, Detector::Full) {
                        all_tiles
                    } else {
                        &pair.hybrid_tiles
                    };
                    gpu.submit(&history, &pair.current, candidates)?;
                    if !candidates.is_empty() {
                        gpu.retain(&history, &pair.current);
                        gpu.enqueue_readback(&staging, candidates.len());
                    }
                    gpu.read_flags(&staging, candidates, &mut selected, false)?;
                }
            }
            let report = sample(start, cycles);
            if !matches!(kind, Detector::Metadata) {
                require_tiles(&selected, pair, name)?;
            }
            Ok(report)
        }),
    })
}

struct Probe<'a> {
    context: &'a ID3D11DeviceContext,
    disjoint: ID3D11Query,
    marks: [ID3D11Query; 4],
    readback: ID3D11Buffer,
    history: GpuFrame,
    selected: Vec<u32>,
}

impl<'a> Probe<'a> {
    fn new(display: &'a Display, gpu: &Gpu, capacity: usize) -> AnyResult<Self> {
        let stamp = || query(&display.device, D3D11_QUERY_TIMESTAMP);
        Ok(Self {
            context: &display.context,
            disjoint: query(&display.device, D3D11_QUERY_TIMESTAMP_DISJOINT)?,
            marks: [stamp()?, stamp()?, stamp()?, stamp()?],
            readback: gpu.staging_buffer()?,
            history: gpu.history()?,
            selected: Vec::with_capacity(capacity),
        })
    }

    fn run(
        &mut self,
        gpu: &Gpu,
        pair: &FramePair,
        readback_only: bool,
        reseed: bool,
    ) -> AnyResult<Sample> {
        let tiles = &pair.hybrid_tiles;
        if tiles.is_empty() {
            return Err("GPU timing probe requires a presentation".into());
        }
        if !readback_only && reseed {
            gpu.seed_history(&self.history, &pair.previous)?;
        }
        let cycles = thread_cycles();
        let start = Instant::now();
        unsafe {
            self.context.Begin(&self.disjoint);
            self.context.End(&self.marks[0]);
        }
        if !readback_only {
            gpu.submit(&self.history, &pair.current, tiles)?;
        }
        let after_submit = Instant::now();
        unsafe {
            self.context.End(&self.marks[1]);
        }
        if !readback_only {
            gpu.retain(&self.history, &pair.current);
        }
        unsafe {
            self.context.End(&self.marks[2]);
        }
        gpu.enqueue_readback(&self.readback, tiles.len());
        unsafe {
            self.context.End(&self.marks[3]);
            self.context.End(&self.disjoint);
        }
        let before_map = Instant::now();
        gpu.read_flags(&self.readback, tiles, &mut self.selected, false)?;
        let after_map = Instant::now();
        let mut report = sample(start, cycles);
        report
            .stages
            .insert("cpu_submit_compare".into(), micros(after_submit - start));
        report.stages.insert(
            "blocking_map_and_decode".into(),
            micros(after_map - before_map),
        );
        if !readback_only {
            require_tiles(&self.selected, pair, "GPU instrumentation")?;
        }
        unsafe {
            self.context.Flush();
        }
        let clock: D3D11_QUERY_DATA_TIMESTAMP_DISJOINT = query_data(self.context, &self.disjoint)?;
        if !clock.Disjoint.as_bool() && clock.Frequency != 0 {
            let a: u64 = query_data(self.context, &self.marks[0])?;
            let b: u64 = query_data(self.context, &self.marks[1])?;
            let c: u64 = query_data(self.context, &self.marks[2])?;
            let d: u64 = query_data(self.context, &self.marks[3])?;
            let period = 1_000_000.0 / clock.Frequency as f64;
            report.stages.insert(
                "gpu_compare_interval".into(),
                (b.saturating_sub(a)) as f64 * period,
            );
            report.stages.insert(
                "gpu_history_copy_interval".into(),
                (c.saturating_sub(b)) as f64 * period,
            );
            report.stages.insert(
                "gpu_result_copy_interval".into(),
                (d.saturating_sub(c)) as f64 * period,
            );
        }
        Ok(report)
    }
}

#[derive(Clone, Copy)]
enum Scheduling {
    CpuGated,
    Indirect,
    Always,
}

#[allow(clippy::too_many_arguments)]
fn consumer_arm<'a>(
    name: &'a str,
    route: Scheduling,
    consumer: &'a Consumer,
    display: &'a Display,
    gpu: &'a Gpu,
    case: &'a Case,
    inputs: &'a InputCase,
    all: &'a [u32],
) -> AnyResult<Arm<'a>> {
    let done = query(&display.device, D3D11_QUERY_EVENT)?;
    let history = gpu.history()?;
    let staging = gpu.staging_buffer()?;
    let mut selected = Vec::with_capacity(all.len());
    Ok(Arm {
        name,
        run: Box::new(move |index, _warmup| {
            let pair = &case.pairs[index % case.pairs.len()];
            let candidates = &pair.hybrid_tiles;
            let used_history = !matches!(route, Scheduling::Always);
            if used_history && needs_seed(&inputs.pairs, index) {
                gpu.seed_history(&history, &pair.previous)?;
            }
            let cpu = thread_cycles();
            let start = Instant::now();
            match route {
                Scheduling::Always => consumer.direct(&pair.current, all),
                Scheduling::Indirect => {
                    gpu.submit(&history, &pair.current, candidates)?;
                    consumer.indirect(gpu, &pair.current, candidates.len());
                    if !candidates.is_empty() {
                        gpu.retain(&history, &pair.current);
                    }
                }
                Scheduling::CpuGated => {
                    gpu.submit(&history, &pair.current, candidates)?;
                    if !candidates.is_empty() {
                        gpu.enqueue_readback(&staging, candidates.len());
                        gpu.retain(&history, &pair.current);
                    }
                    gpu.read_flags(&staging, candidates, &mut selected, false)?;
                    consumer.direct(&pair.current, &selected);
                }
            }
            unsafe {
                display.context.End(&done);
                display.context.Flush();
            }
            let mut report = sample(start, cpu);
            let ready: BOOL = query_data(&display.context, &done)?;
            if !ready.as_bool() {
                return Err("GPU consumer completion event returned false".into());
            }
            let total = micros(start.elapsed());
            report
                .stages
                .insert("cpu_path_through_submit".into(), report.latency_us);
            report.stages.insert(
                "completion_observation_after_submit".into(),
                total - report.latency_us,
            );
            report.latency_us = total;
            if matches!(route, Scheduling::CpuGated) {
                require_tiles(&selected, pair, name)?;
            }
            Ok(report)
        }),
    })
}

struct Queued {
    slot: usize,
    frame_id: usize,
    case_index: usize,
    start: Instant,
    first_us: f64,
    measured: bool,
}

#[derive(Serialize)]
struct Confirmation {
    frame_id: usize,
    first_signal_us: f64,
    final_observed_us: f64,
    disposition: &'static str,
}

struct Tentative<'a> {
    gpu: &'a Gpu,
    context: &'a ID3D11DeviceContext,
    history: GpuFrame,
    slots: Vec<ID3D11Buffer>,
    free: Vec<usize>,
    pending: VecDeque<Queued>,
    selected: Vec<u32>,
    proposed: Vec<u32>,
    id: usize,
    full_queue_events: usize,
    events: Vec<Confirmation>,
}

impl<'a> Tentative<'a> {
    fn new(gpu: &'a Gpu, context: &'a ID3D11DeviceContext, tile_count: usize) -> AnyResult<Self> {
        let slots = (0..3)
            .map(|_| gpu.staging_buffer())
            .collect::<AnyResult<Vec<_>>>()?;
        Ok(Self {
            gpu,
            context,
            history: gpu.history()?,
            free: (0..slots.len()).collect(),
            slots,
            pending: VecDeque::new(),
            selected: Vec::with_capacity(tile_count),
            proposed: Vec::with_capacity(tile_count),
            id: 0,
            full_queue_events: 0,
            events: Vec::new(),
        })
    }

    fn poll(&mut self, case: &Case) -> AnyResult<()> {
        while let Some(queued) = self.pending.front() {
            let pair = &case.pairs[queued.case_index];
            if !self.gpu.read_flags(
                &self.slots[queued.slot],
                &pair.hybrid_tiles,
                &mut self.selected,
                true,
            )? {
                break;
            }
            let final_observed_us = micros(queued.start.elapsed());
            require_tiles(&self.selected, pair, "Tentative confirmation")?;
            let queued = self.pending.pop_front().unwrap();
            self.free.push(queued.slot);
            if queued.measured {
                let disposition = if self.selected == pair.metadata_tiles {
                    "confirmed"
                } else if self.selected.is_empty() {
                    "retracted"
                } else {
                    "corrected"
                };
                self.events.push(Confirmation {
                    frame_id: queued.frame_id,
                    first_signal_us: queued.first_us,
                    final_observed_us,
                    disposition,
                });
            }
        }
        Ok(())
    }

    fn drain(&mut self, case: &Case) -> AnyResult<()> {
        let deadline = Instant::now() + Duration::from_secs(2);
        while !self.pending.is_empty() {
            self.poll(case)?;
            if self.pending.is_empty() {
                break;
            }
            if Instant::now() >= deadline {
                return Err("Tentative queue failed to drain within 2 s".into());
            }
            thread::yield_now();
        }
        Ok(())
    }

    fn step(
        &mut self,
        case: &Case,
        inputs: &InputCase,
        index: usize,
        warmup: bool,
        reps: usize,
    ) -> AnyResult<Sample> {
        if index == 0 && !self.pending.is_empty() {
            self.drain(case)?;
        }
        self.poll(case)?;
        if self.free.is_empty() {
            if !warmup {
                self.full_queue_events += 1;
            }
            self.drain(case)?;
        }
        let i = index % case.pairs.len();
        let pair = &case.pairs[i];
        if needs_seed(&inputs.pairs, index) {
            self.gpu.seed_history(&self.history, &pair.previous)?;
        }
        let cpu = thread_cycles();
        let start = Instant::now();
        self.proposed.clear();
        self.proposed.extend_from_slice(&pair.metadata_tiles);
        std::hint::black_box(&self.proposed);
        let first_us = micros(start.elapsed());
        let slot = self.free.pop().ok_or("No tentative readback buffer")?;
        let candidates = &pair.hybrid_tiles;
        if !candidates.is_empty() {
            self.gpu.submit(&self.history, &pair.current, candidates)?;
            self.gpu.retain(&self.history, &pair.current);
            self.gpu
                .enqueue_readback(&self.slots[slot], candidates.len());
            unsafe {
                self.context.Flush();
            }
        }
        self.id += usize::from(!warmup);
        self.pending.push_back(Queued {
            slot,
            frame_id: self.id,
            case_index: i,
            start,
            first_us,
            measured: !warmup,
        });
        let mut report = sample(start, cpu);
        report.stages.insert("tentative_signal".into(), first_us);
        if !warmup && index + 1 == reps {
            // Give the final frame the same next-frame observation cadence.
            let period = Duration::from_secs_f64(1.0 / HZ as f64);
            thread::sleep(period.saturating_sub(start.elapsed()));
            self.drain(case)?;
        }
        Ok(report)
    }
}

fn summarize_tentative(events: &[Confirmation], full: usize) -> Value {
    fn p(values: impl Iterator<Item = f64>, fraction: f64) -> f64 {
        let mut values: Vec<_> = values.collect();
        values.sort_by(f64::total_cmp);
        let index = (values.len().saturating_sub(1)) as f64 * fraction;
        let x = values[index.floor() as usize];
        x + (values[index.ceil() as usize] - x) * index.fract()
    }
    json!({"frames": events.len(), "full_queue_events": full,
        "retractions": events.iter().filter(|event| event.disposition == "retracted").count(),
        "corrections": events.iter().filter(|event| event.disposition == "corrected").count(),
        "first_p50_us": p(events.iter().map(|e| e.first_signal_us), 0.50),
        "final_observed_p50_us": p(events.iter().map(|e| e.final_observed_us), 0.50),
        "final_observed_p95_us": p(events.iter().map(|e| e.final_observed_us), 0.95),
        "events": events})
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    display: &Display,
    extent: Extent,
    gpu: &Gpu,
    inputs: &[InputCase],
    cases: &[Case],
    all: &[u32],
    rounds: usize,
    reps: usize,
    seed: u64,
    output: &Path,
    live: Value,
) -> AnyResult<()> {
    let key = |case: &Case| {
        inputs
            .iter()
            .find(|input| input.name == case.name)
            .ok_or("missing CPU input for replay case")
    };
    let delivery = RefCell::new(Delivery::new(
        &display.device,
        &display.context,
        extent.width,
        extent.height,
        TILE_SIZE,
    )?);
    let consumer = Consumer::new(&display.device, &display.context, gpu, extent)?;
    let modes = [
        ("standalone fingerprint", Mode::Fingerprint),
        ("hardware delivery", Mode::HardwareDelivery),
        ("shader copy only", Mode::ShaderDelivery),
        ("hardware copy plus queued hash", Mode::Independent),
        ("fused fingerprint delivery", Mode::Fused),
        ("mapped CPU exact", Mode::CpuExact),
        ("verified fingerprint delivery", Mode::Verify),
    ];
    let mut deviations = BTreeMap::new();
    let delivery_counts = RefCell::new(BTreeMap::new());
    let mut checked = 0usize;
    let mut changed_tile_count = 0usize;
    let mut redundant_metadata = 0usize;
    for case in cases {
        for (pair, input) in case.pairs.iter().zip(&key(case)?.pairs) {
            checked += 1;
            changed_tile_count += pair.expected.len();
            redundant_metadata +=
                usize::from(pair.expected.is_empty() && !pair.metadata_tiles.is_empty());
            let mut selected = Vec::with_capacity(all.len());
            for (name, mode) in modes {
                let mut delivery = delivery.borrow_mut();
                delivery.reset(mode, &pair.previous, Some(&input.previous))?;
                delivery.run(mode, &pair.current, &pair.hybrid_tiles, &mut selected)?;
                if matches!(
                    mode,
                    Mode::Fingerprint | Mode::Independent | Mode::Fused | Mode::Verify
                ) && delivery.fingerprints().len() != all.len()
                {
                    return Err("GPU digest history has the wrong number of tiles".into());
                }
                if deliver_mode(mode) {
                    output_pixels(&delivery, &input.current)?;
                }
                if decides(mode) && selected != pair.expected {
                    if matches!(mode, Mode::Verify | Mode::CpuExact) {
                        require_tiles(&selected, pair, name)?;
                    }
                    *deviations.entry(name).or_insert(0usize) += 1;
                }
            }
            gpu.compare(
                &pair.previous,
                &pair.current,
                &pair.hybrid_tiles,
                &mut selected,
            )?;
            require_tiles(&selected, pair, "exact hybrid")?;
            gpu.compare(&pair.previous, &pair.current, all, &mut selected)?;
            require_tiles(&selected, pair, "exact full screen")?;
            consumer.clear_scores();
            gpu.submit(&pair.previous, &pair.current, &pair.hybrid_tiles)?;
            consumer.indirect(gpu, &pair.current, pair.hybrid_tiles.len());
            consumer.verify(&pair.expected, &input.current, true)?;
            consumer.clear_scores();
            consumer.direct(&pair.current, &pair.expected);
            consumer.verify(&pair.expected, &input.current, false)?;
        }
    }
    eprintln!("Follow-up correctness gate completed on {checked} distinct frame pairs");
    let mut groups = Vec::new();
    let mut tentative = BTreeMap::new();
    let mut skipped_timings = Vec::new();
    for case in cases {
        if case.name.starts_with("idle_")
            || case.name == "metadata_fallback_checks"
            || case.name == "fingerprint_order_and_edges"
        {
            skipped_timings.push(case.name.as_str());
            continue;
        }
        let input = key(case)?;
        let mut detector_arms = vec![
            detect_arm(
                "dirty rectangles",
                Detector::Metadata,
                case,
                input,
                gpu,
                all,
            )?,
            detect_arm(
                "exact hybrid resident pair",
                Detector::Resident,
                case,
                input,
                gpu,
                all,
            )?,
            detect_arm(
                "exact hybrid with history copy",
                Detector::Owned,
                case,
                input,
                gpu,
                all,
            )?,
            detect_arm(
                "exact full screen with history copy",
                Detector::Full,
                case,
                input,
                gpu,
                all,
            )?,
            Arm {
                name: "standalone fingerprint",
                run: timed_delivery(
                    "standalone fingerprint",
                    &delivery,
                    Mode::Fingerprint,
                    case,
                    input,
                    extent,
                    &delivery_counts,
                ),
            },
        ];
        groups.push(followup::run_group(
            &title(case, "detector"),
            "GPU frame ready to CPU tile decision",
            None,
            &mut detector_arms,
            rounds,
            seed,
            reps,
            HZ,
        )?);
        let mut delivery_arms: Vec<_> = modes
            .iter()
            .filter(|(_, mode)| deliver_mode(*mode))
            .map(|&(name, mode)| Arm {
                name,
                run: timed_delivery(name, &delivery, mode, case, input, extent, &delivery_counts),
            })
            .collect();
        groups.push(followup::run_group(
            &title(case, "CPU delivery"),
            "GPU frame ready to owned CPU pixel output, plus decision for detector arms",
            Some("hardware delivery"),
            &mut delivery_arms,
            rounds,
            seed.rotate_left(7),
            reps,
            HZ,
        )?);
        let mut consumer_arms = [
            consumer_arm(
                "CPU-gated exact",
                Scheduling::CpuGated,
                &consumer,
                display,
                gpu,
                case,
                input,
                all,
            )?,
            consumer_arm(
                "GPU-only indirect exact",
                Scheduling::Indirect,
                &consumer,
                display,
                gpu,
                case,
                input,
                all,
            )?,
            consumer_arm(
                "unconditional GPU task",
                Scheduling::Always,
                &consumer,
                display,
                gpu,
                case,
                input,
                all,
            )?,
        ];
        groups.push(followup::run_group(
            &title(case, "GPU task"),
            "GPU frame ready to observed completion of tile luminance task",
            Some("CPU-gated exact"),
            &mut consumer_arms,
            rounds,
            seed.rotate_left(19),
            reps,
            HZ,
        )?);
        let mut timeline = Tentative::new(gpu, &display.context, all.len())?;
        let mut event_arms = [
            detect_arm(
                "dirty-rectangle first signal",
                Detector::Metadata,
                case,
                input,
                gpu,
                all,
            )?,
            detect_arm(
                "blocking exact decision",
                Detector::Owned,
                case,
                input,
                gpu,
                all,
            )?,
            Arm {
                name: "tentative plus asynchronous exact",
                run: Box::new(|frame, warmup| timeline.step(case, input, frame, warmup, reps)),
            },
        ];
        groups.push(followup::run_group(&title(case, "tentative"), "Synchronous signal or async enqueue; final async events use separate next-frame observation", None, &mut event_arms, rounds, seed.rotate_left(31), reps, HZ)?);
        drop(event_arms);
        timeline.drain(case)?;
        if timeline.events.len() != reps * rounds {
            return Err("Missing tentative confirmations".into());
        }
        tentative.insert(
            case.name.as_str(),
            summarize_tentative(&timeline.events, timeline.full_queue_events),
        );
        if case.name.contains("sparse") || case.name.contains("full_motion") {
            let mut read_only = Probe::new(display, gpu, all.len())?;
            let mut exact = Probe::new(display, gpu, all.len())?;
            gpu.clear_output();
            let mut arms = [
                Arm {
                    name: "small readback only",
                    run: Box::new(|frame, _| {
                        read_only.run(gpu, &case.pairs[frame % case.pairs.len()], true, false)
                    }),
                },
                Arm {
                    name: "instrumented exact with history copy",
                    run: Box::new(|frame, _| {
                        exact.run(
                            gpu,
                            &case.pairs[frame % case.pairs.len()],
                            false,
                            needs_seed(&input.pairs, frame),
                        )
                    }),
                },
            ];
            groups.push(followup::run_group(
                &title(case, "GPU timing diagnostics"),
                "Instrumented GPU intervals and CPU blocking Map",
                Some("small readback only"),
                &mut arms,
                rounds,
                seed.rotate_left(43),
                reps,
                HZ,
            )?);
        }
    }
    let cache = crate::tile_cache::measure(rounds, seed, reps, HZ)?;
    let metadata = json!({"commit": BASELINE, "adapter": display.adapter, "width": extent.width, "height": extent.height,
        "tile_size": TILE_SIZE, "history_bytes": {"full_pixels": extent.width * extent.height * 4,
            "fingerprint_only": retained_history_bytes(Mode::Fingerprint, extent.width, extent.height, extent.tile_count()),
            "verified_fingerprints_and_pixels": retained_history_bytes(Mode::Verify, extent.width, extent.height, extent.tile_count())},
        "oracle_frame_pairs": checked, "oracle_changed_tiles": changed_tile_count, "metadata_redundant_frame_pairs": redundant_metadata,
        "fingerprint_oracle_deviations_by_method": deviations, "delivery_observations": delivery_counts.into_inner(), "validation_only_cases": skipped_timings,
        "live": live, "tentative": tentative, "frequency_hz": HZ,
        "notes": ["Replays use owned D3D11 R32_UINT textures with identical BGRA bytes. Timing starts after desktop acquisition and upload.",
            "History reseeding after fixture discontinuities is outside timed samples. Full BGRA history upkeep is inside relevant timed methods.",
            "Timing diagnostics add GPU queries and are a separate group. GPU intervals include queue activity within their markers.",
            "The GPU task is a tile luminance sum. GPU-ready wall time includes polling for a completion event; submission measurements end before polling.",
            "Tentative events deliver metadata first and poll exact results at the next replay frame; final latency reflects observation cadence.",
            "Fingerprint correctness uses finite fixtures. Digest equality alone is probabilistic."]});
    followup::write_report(output, metadata, &groups, &cache)?;
    eprintln!("Saved follow-up results in {}", output.display());
    Ok(())
}
