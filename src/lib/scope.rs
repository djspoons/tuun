//! The state behind the scope views: the latest output samples for Snapshot,
//! and the per-column ranges of the output for Scan.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::ids::{MarkId, WaveformId};
use crate::tracker;

/// The number of output samples Snapshot shows.
pub const SNAPSHOT_LEN: usize = 2048;

/// How close an output range comes to clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipLevel {
    None,
    Warn,
    Clip,
}

/// Returns how close the range from `min` to `max` comes to clipping.
pub fn clip_level(min: f32, max: f32) -> ClipLevel {
    let peak = min.abs().max(max.abs());
    if peak <= 0.95 {
        ClipLevel::None
    } else if peak <= 1.0 {
        ClipLevel::Warn
    } else {
        ClipLevel::Clip
    }
}

/// One column of Scan, as it is drawn.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScanColumn {
    Empty,
    /// The output's range during the write head's pass.
    Current {
        min: f32,
        max: f32,
        clip: ClipLevel,
    },
    /// The output's range during the pass before the write head's.
    Previous {
        min: f32,
        max: f32,
        clip: ClipLevel,
    },
    /// Samples dropped during the write head's pass.
    CurrentGap,
    /// Samples dropped during the pass before the write head's.
    PreviousGap,
    /// The write head's own column.
    Head,
}

/// Returns the start of the timekeeping mark that Scan's passes are aligned to,
/// or `None` when `status` has no such mark.
///
/// Any active or pending top-level mark of the odd Beats waveform will do: its
/// starts differ by whole multiples of two measures.
pub fn measure_origin(status: &tracker::Status<WaveformId, MarkId>) -> Option<Instant> {
    status
        .marks
        .iter()
        .find(|m| m.waveform_id == WaveformId::Beats(false) && m.mark_id == MarkId::TopLevel)
        .map(|m| m.start)
}

/// What one column of Scan holds.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Slot {
    Empty,
    Range { pass: i64, min: f32, max: f32 },
    Gap { pass: i64 },
}

impl Slot {
    fn pass(&self) -> Option<i64> {
        match self {
            Slot::Empty => None,
            Slot::Range { pass, .. } | Slot::Gap { pass } => Some(*pass),
        }
    }
}

/// Scan's columns and write head.
///
/// Positions are measured in columns since the measure origin, so a position's
/// pass is its floor divided by the number of columns, and its column is the
/// remainder.
struct Scan {
    window: Duration,
    slots: Vec<Slot>,
    // The position of the end of the newest sample, rounded down.
    head: Option<i64>,
}

impl Scan {
    /// Returns the position of `t` relative to `origin`, in columns.
    fn position(&self, t: Instant, origin: Instant) -> f64 {
        let offset = if t >= origin {
            (t - origin).as_secs_f64()
        } else {
            -(origin - t).as_secs_f64()
        };
        offset / self.window.as_secs_f64() * self.slots.len() as f64
    }

    /// Returns the pass and column of the position `x`.
    fn place(&self, x: f64) -> (i64, usize) {
        let columns = self.slots.len() as i64;
        let x = x.floor() as i64;
        (x.div_euclid(columns), x.rem_euclid(columns) as usize)
    }

    /// Records a sample `value` at position `x`.
    fn add_sample(&mut self, x: f64, value: f32) {
        let (pass, column) = self.place(x);
        let slot = &mut self.slots[column];
        if slot.pass().is_some_and(|p| p > pass) {
            return;
        }
        if let Slot::Range { pass: p, min, max } = slot
            && *p == pass
        {
            *min = min.min(value);
            *max = max.max(value);
        } else {
            *slot = Slot::Range {
                pass,
                min: value,
                max: value,
            };
        }
    }

    /// Records that samples were dropped at position `x`.
    fn add_gap(&mut self, x: f64) {
        let (pass, column) = self.place(x);
        let slot = &mut self.slots[column];
        if slot.pass().is_none_or(|p| p < pass) {
            *slot = Slot::Gap { pass };
        }
    }

    /// Records the samples of one callback placed by `header` against `origin`.
    fn add_chunk(
        &mut self,
        origin: Instant,
        sample_rate: u32,
        header: &tracker::SampleHeader,
        samples: impl Iterator<Item = f32>,
    ) {
        let columns_per_sample =
            self.slots.len() as f64 / (self.window.as_secs_f64() * sample_rate as f64);
        let start = self.position(header.buffer_start, origin);
        if header.dropped > 0 {
            // Older passes are overwritten anyway, so mark at most one window.
            let gap_start = (start - header.dropped as f64 * columns_per_sample)
                .max(start - self.slots.len() as f64);
            for x in gap_start.floor() as i64..start.ceil() as i64 {
                self.add_gap(x as f64);
            }
        }
        for (i, value) in samples.enumerate() {
            self.add_sample(start + i as f64 * columns_per_sample, value);
        }
        let end = start + header.written as f64 * columns_per_sample;
        self.head = Some(end.floor() as i64);
    }

    fn columns(&self) -> impl Iterator<Item = ScanColumn> + '_ {
        let head = self.head.map(|x| self.place(x as f64));
        self.slots.iter().enumerate().map(move |(column, slot)| {
            let Some((head_pass, head_column)) = head else {
                return ScanColumn::Empty;
            };
            let current = if column < head_column {
                head_pass
            } else if column > head_column {
                head_pass - 1
            } else {
                return ScanColumn::Head;
            };
            match *slot {
                Slot::Range { pass, min, max } if pass == current => {
                    let clip = clip_level(min, max);
                    if column < head_column {
                        ScanColumn::Current { min, max, clip }
                    } else {
                        ScanColumn::Previous { min, max, clip }
                    }
                }
                Slot::Gap { pass } if pass == current => {
                    if column < head_column {
                        ScanColumn::CurrentGap
                    } else {
                        ScanColumn::PreviousGap
                    }
                }
                _ => ScanColumn::Empty,
            }
        })
    }
}

/// The output samples that the scope views draw, read from the tracker's
/// sample rings.
pub struct Scope {
    consumer: tracker::SampleConsumer,
    sample_rate: u32,
    snapshot: VecDeque<f32>,
    scan: Scan,
}

impl Scope {
    /// Returns a scope that reads from `consumer`, whose Scan view divides each
    /// `scan_window` into `scan_columns` columns.
    pub fn new(
        consumer: tracker::SampleConsumer,
        sample_rate: u32,
        scan_window: Duration,
        scan_columns: usize,
    ) -> Scope {
        Scope {
            consumer,
            sample_rate,
            snapshot: VecDeque::with_capacity(SNAPSHOT_LEN),
            scan: Scan {
                window: scan_window,
                slots: vec![Slot::Empty; scan_columns],
                head: None,
            },
        }
    }

    /// Reads every sample the tracker has reported since the last drain.
    ///
    /// Scan's passes start at `measure_origin`; without one, only Snapshot
    /// changes.
    pub fn drain(&mut self, measure_origin: Option<Instant>) {
        while let Ok(header) = self.consumer.headers.pop() {
            let chunk = self.consumer.samples.read_chunk(header.written).expect(
                "a popped header's samples are in the sample ring, \
                 since the tracker pushes them before the header",
            );
            let (first, second) = chunk.as_slices();
            let samples = || first.iter().chain(second).copied();
            for sample in samples() {
                if self.snapshot.len() == SNAPSHOT_LEN {
                    self.snapshot.pop_front();
                }
                self.snapshot.push_back(sample);
            }
            if let Some(origin) = measure_origin {
                self.scan
                    .add_chunk(origin, self.sample_rate, &header, samples());
            }
            chunk.commit_all();
        }
    }

    /// Returns the latest samples, oldest first: up to [`SNAPSHOT_LEN`] of them.
    pub fn snapshot(&self) -> impl Iterator<Item = f32> + '_ {
        self.snapshot.iter().copied()
    }

    /// Returns Scan's columns, left to right.
    pub fn scan_columns(&self) -> impl Iterator<Item = ScanColumn> + '_ {
        self.scan.columns()
    }
}

#[cfg(test)]
mod tests {
    use crate::tracker::{Mark, SampleProducer, Status, sample_rings};

    use super::*;

    // One column is one second, or 1024 samples: whole numbers of samples and
    // columns have exact positions.
    const SAMPLE_RATE: u32 = 1024;
    const COLUMNS: usize = 8;
    const WINDOW: Duration = Duration::from_secs(8);
    const COLUMN: Duration = Duration::from_secs(1);
    const COLUMN_SAMPLES: usize = 1024;

    fn test_scope_with_headers(header_capacity: usize) -> (SampleProducer, Scope) {
        let (producer, consumer) = sample_rings(10_000, header_capacity);
        (producer, Scope::new(consumer, SAMPLE_RATE, WINDOW, COLUMNS))
    }

    fn test_scope() -> (SampleProducer, Scope) {
        test_scope_with_headers(16)
    }

    fn slots(scope: &Scope) -> Vec<Slot> {
        scope.scan.slots.clone()
    }

    #[test]
    fn drain_reads_exactly_each_headers_samples() {
        let (mut producer, mut scope) = test_scope();
        let now = Instant::now();
        producer.push(now, &[1.0, 2.0, 3.0]);
        producer.push(now, &[4.0, 5.0]);
        scope.drain(None);
        assert_eq!(
            scope.snapshot().collect::<Vec<_>>(),
            [1.0, 2.0, 3.0, 4.0, 5.0]
        );
        assert!(scope.consumer.samples.is_empty());
        assert!(scope.consumer.headers.is_empty());
    }

    #[test]
    fn snapshot_keeps_the_latest_samples() {
        let (mut producer, mut scope) = test_scope();
        let now = Instant::now();
        let ramp =
            |from: usize, len: usize| (from..from + len).map(|i| i as f32).collect::<Vec<_>>();

        producer.push(now, &ramp(0, 100));
        scope.drain(None);
        assert_eq!(scope.snapshot().collect::<Vec<_>>(), ramp(0, 100));

        // Longer than the snapshot.
        producer.push(now, &ramp(100, 3000));
        scope.drain(None);
        assert_eq!(
            scope.snapshot().collect::<Vec<_>>(),
            ramp(3100 - SNAPSHOT_LEN, SNAPSHOT_LEN)
        );

        // Shorter than the snapshot.
        producer.push(now, &ramp(3100, 10));
        scope.drain(None);
        assert_eq!(
            scope.snapshot().collect::<Vec<_>>(),
            ramp(3110 - SNAPSHOT_LEN, SNAPSHOT_LEN)
        );
    }

    #[test]
    fn samples_are_placed_by_pass_and_column() {
        let (mut producer, mut scope) = test_scope();
        let origin = Instant::now() + Duration::from_secs(10);
        let quarter = WINDOW / 4;

        producer.push(origin + quarter, &[0.5]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[COLUMNS / 4],
            Slot::Range {
                pass: 0,
                min: 0.5,
                max: 0.5
            }
        );

        producer.push(origin + WINDOW + quarter, &[0.25]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[COLUMNS / 4],
            Slot::Range {
                pass: 1,
                min: 0.25,
                max: 0.25
            }
        );

        let (mut producer, mut scope) = test_scope();
        producer.push(origin - quarter, &[0.5]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[COLUMNS * 3 / 4],
            Slot::Range {
                pass: -1,
                min: 0.5,
                max: 0.5
            }
        );
    }

    #[test]
    fn samples_widen_their_pass_and_replace_older_passes() {
        let (mut producer, mut scope) = test_scope();
        let origin = Instant::now();
        let column_start = origin + WINDOW + COLUMN;

        // Several samples within one column widen its range.
        producer.push(column_start, &[0.1, -0.3, 0.2]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[1],
            Slot::Range {
                pass: 1,
                min: -0.3,
                max: 0.2
            }
        );

        // An older pass is ignored.
        producer.push(column_start - WINDOW, &[0.9]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[1],
            Slot::Range {
                pass: 1,
                min: -0.3,
                max: 0.2
            }
        );

        // A newer pass replaces the slot.
        producer.push(column_start + WINDOW, &[0.7]);
        scope.drain(Some(origin));
        assert_eq!(
            slots(&scope)[1],
            Slot::Range {
                pass: 2,
                min: 0.7,
                max: 0.7
            }
        );
    }

    #[test]
    fn scan_columns_show_the_head_and_the_passes_around_it() {
        let (mut producer, mut scope) = test_scope();
        let origin = Instant::now();

        // Pass 0 fills columns 1, 5 and 6, and pass -1 column 7. Pass 1 fills
        // columns 0 to 2, and the head ends in column 3.
        producer.push(origin + COLUMN, &[0.5]);
        producer.push(origin + 5 * COLUMN, &[0.5]);
        producer.push(origin + 6 * COLUMN, &[0.5]);
        producer.push(origin - COLUMN, &[0.5]);
        producer.push(origin + WINDOW, &[-0.5; COLUMN_SAMPLES]);
        producer.push(origin + WINDOW + COLUMN, &[0.99; 2 * COLUMN_SAMPLES]);
        scope.drain(Some(origin));

        let current = |min, max| ScanColumn::Current {
            min,
            max,
            clip: clip_level(min, max),
        };
        let previous = ScanColumn::Previous {
            min: 0.5,
            max: 0.5,
            clip: ClipLevel::None,
        };
        assert_eq!(
            scope.scan_columns().collect::<Vec<_>>(),
            [
                current(-0.5, -0.5),
                current(0.99, 0.99),
                current(0.99, 0.99),
                ScanColumn::Head,
                ScanColumn::Empty,
                previous,
                previous,
                // Older than the previous pass.
                ScanColumn::Empty,
            ]
        );
    }

    #[test]
    fn dropped_samples_mark_gaps_before_their_header() {
        let (mut producer, mut scope) = test_scope_with_headers(1);
        let origin = Instant::now();

        // Fill column 1 of pass 0, then drop columns 2 to 4 while the header
        // ring is full.
        producer.push(origin + COLUMN, &[0.5; COLUMN_SAMPLES]);
        producer.push(origin + 2 * COLUMN, &[0.0; 3 * COLUMN_SAMPLES]);
        scope.drain(Some(origin));
        producer.push(origin + 5 * COLUMN, &[]);
        scope.drain(Some(origin));

        let gap = Slot::Gap { pass: 0 };
        assert_eq!(
            slots(&scope)[..6],
            [
                Slot::Empty,
                Slot::Range {
                    pass: 0,
                    min: 0.5,
                    max: 0.5
                },
                gap,
                gap,
                gap,
                Slot::Empty,
            ]
        );
    }

    #[test]
    fn without_a_measure_origin_only_snapshot_changes() {
        let (mut producer, mut scope) = test_scope();
        producer.push(Instant::now(), &[0.5; 10]);
        scope.drain(None);
        assert_eq!(scope.snapshot().count(), 10);
        assert!(slots(&scope).iter().all(|s| *s == Slot::Empty));
        assert!(scope.scan_columns().all(|c| c == ScanColumn::Empty));
    }

    #[test]
    fn clip_level_thresholds() {
        assert_eq!(clip_level(0.0, 0.95), ClipLevel::None);
        assert_eq!(clip_level(0.0, 0.96), ClipLevel::Warn);
        assert_eq!(clip_level(0.0, 1.0), ClipLevel::Warn);
        assert_eq!(clip_level(0.0, 1.01), ClipLevel::Clip);
        assert_eq!(clip_level(-1.01, 0.0), ClipLevel::Clip);
    }

    #[test]
    fn measure_origin_is_the_odd_beats_top_level_start() {
        let now = Instant::now();
        let mark = |waveform_id, mark_id, offset| Mark {
            waveform_id,
            mark_id,
            start: now + Duration::from_secs(offset),
            duration: None,
        };
        let status = |marks| Status {
            buffer_start: now,
            marks,
            tracker_load: None,
            allocations: None,
        };

        assert_eq!(measure_origin(&status(vec![])), None);
        assert_eq!(
            measure_origin(&status(vec![
                mark(WaveformId::Beats(true), MarkId::TopLevel, 1),
                mark(WaveformId::Beats(false), MarkId::Terminator, 2),
                mark(WaveformId::Program(0), MarkId::TopLevel, 3),
            ])),
            None
        );
        assert_eq!(
            measure_origin(&status(vec![
                mark(WaveformId::Beats(true), MarkId::TopLevel, 1),
                mark(WaveformId::Beats(false), MarkId::TopLevel, 4),
            ])),
            Some(now + Duration::from_secs(4))
        );
    }
}
