//! Sends waveforms to the tracker — all tracker I/O lives here.
//!
//! `Player` owns the two command routes: `precompute_sender` goes through
//! the precompute thread (used for playback scheduled at the next measure,
//! where latency is hidden), and `fast_sender` goes straight to the
//! tracker (used for immediate playback and note-on/off, where keystroke
//! latency matters). All methods take `&self`.

use std::sync::mpsc;
use std::time;

use crate::diagnostics::Source;
use crate::evaluator::Evaluator;
use crate::expr;
use crate::ids::{MarkId, WaveformId, WaveformSelector};
use crate::optimizer;
use crate::programs::{ProgramSet, ProgramSliders};
use crate::sequencer;
use crate::slider;
use crate::tracker;
use crate::waveform;

/// Sentinel sixteenth for a program's silent anchor step, played alongside
/// the real steps to carry the family's `TopLevel` mark.
pub const ANCHOR_SIXTEENTH: u8 = u8::MAX;

pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Substitutes each slider's current value into every `Marked { id:
/// Slider(label), … }` node in `waveform`.
///
/// Returns the per-slider `(label, value)` pairs so callers that need to seed
/// the slider worker's `last_slider_values` map (e.g., for a fresh
/// `WaveformId::Key`) can build their own keyed map without re-denormalizing.
pub fn substitute_current_slider_values(
    waveform: &mut waveform::Waveform<MarkId>,
    sliders: &ProgramSliders,
) -> Vec<(String, f32)> {
    let mut values = Vec::with_capacity(sliders.configs().len());
    for (config, &normalized) in sliders.configs().iter().zip(sliders.normalized_values()) {
        let value = slider::denormalize(&config.function, normalized).unwrap_or(0.0);
        values.push((config.label.clone(), value));
        waveform::substitute(
            waveform,
            &MarkId::Slider(config.label.clone()),
            &waveform::Waveform::Const(value),
        );
    }
    values
}

pub struct Player {
    tempo: u32,
    beats_per_measure: u32,
    precompute_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    fast_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
}

impl Player {
    pub fn new(
        tempo: u32,
        beats_per_measure: u32,
        precompute_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        fast_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    ) -> Player {
        Player {
            tempo,
            beats_per_measure,
            precompute_sender,
            fast_sender,
        }
    }

    /// Plays the program at `program_index` as a waveform, substituting its
    /// current slider values. Returns the user-visible message, or `None`
    /// when the program's text didn't evaluate to a waveform (or the index
    /// is out of range) and nothing was played.
    ///
    /// `start_at_next_measure` routes through the precompute thread and
    /// schedules the start at the next measure boundary; otherwise the
    /// waveform plays immediately via the fast route.
    pub fn play_program(
        &self,
        set: &ProgramSet,
        program_index: usize,
        status: &tracker::Status<WaveformId, MarkId>,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    ) -> Option<String> {
        let program = set.program(program_index)?;
        let display_name = set.display_name(program_index);
        let message;
        let repeat_every;
        if let Some(measures) = repeat_after_measures {
            let beats = (measures * self.beats_per_measure) as u64;
            message = format!("Looping waveform {} every {:?} beats", display_name, beats);
            repeat_every = Some(duration_from_beats(self.tempo, beats));
        } else {
            // Otherwise, play it once
            message = format!("Playing waveform {}", display_name);
            repeat_every = None;
        }
        let start = if start_at_next_measure {
            Some(next_measure_start(status))
        } else {
            None
        };
        let mut waveform = program.waveform().cloned()?;
        // Substitute the program's current slider positions before handing
        // the waveform to the tracker (since the cached ones may be old).
        substitute_current_slider_values(&mut waveform, program.sliders());
        if start_at_next_measure {
            &self.precompute_sender
        } else {
            &self.fast_sender
        }
        .send(tracker::Command::Play {
            // TODO maybe extend the top-level mark to the full measure?
            id: WaveformId::Program(program_index),
            waveform: build_top_level_waveform(waveform, program.level_db()),
            start,
            repeat_every,
        })
        .unwrap();
        Some(message)
    }

    /// Plays the sequenceable program at `program_index` decomposed: one
    /// scheduled, independently repeating `WaveformId::Step` per listed
    /// beat, plus a silent anchor step carrying the `TopLevel` mark.
    /// Returns the user-visible message, or `None` when the program's text
    /// didn't evaluate to a sequenceable waveform and nothing was played.
    ///
    /// `start_at_next_measure` routes through the precompute thread and
    /// anchors the cycle at the next measure boundary; otherwise the cycle
    /// is anchored at the current instant.
    pub fn play_program_steps(
        &self,
        set: &ProgramSet,
        program_index: usize,
        status: &tracker::Status<WaveformId, MarkId>,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    ) -> Option<String> {
        let program = set.program(program_index)?;
        let sequence = program.sequence()?;
        let display_name = set.display_name(program_index);
        let message;
        let repeat_every;
        if let Some(measures) = repeat_after_measures {
            let beats = (measures * self.beats_per_measure) as u64;
            message = format!("Looping sequence {} every {:?} beats", display_name, beats);
            repeat_every = Some(duration_from_beats(self.tempo, beats));
        } else {
            message = format!("Playing sequence {}", display_name);
            repeat_every = None;
        }
        let base = if start_at_next_measure {
            next_measure_start(status)
        } else {
            time::Instant::now()
        };
        let mut step = sequence.step_waveform.clone();
        substitute_current_slider_values(&mut step, program.sliders());
        let sender = if start_at_next_measure {
            &self.precompute_sender
        } else {
            &self.fast_sender
        };
        for &beat in &sequence.beats {
            sender
                .send(tracker::Command::Play {
                    id: WaveformId::Step {
                        program: program_index,
                        sixteenth: sequencer::sixteenth_for_beat(beat),
                    },
                    waveform: build_leveled_waveform(step.clone(), program.level_db()),
                    start: Some(base + duration_from_beats_f32(self.tempo, beat - 1.0)),
                    repeat_every,
                })
                .unwrap();
        }
        // The anchor spans the whole pattern so the family reads as playing
        // for as long as its steps do.
        let pattern_beats =
            sequence
                .beats
                .iter()
                .fold(self.beats_per_measure as f32, |acc, &beat| {
                    let measures = ((beat - 0.75) / self.beats_per_measure as f32).ceil();
                    acc.max(measures * self.beats_per_measure as f32)
                });
        sender
            .send(tracker::Command::Play {
                id: WaveformId::Step {
                    program: program_index,
                    sixteenth: ANCHOR_SIXTEENTH,
                },
                waveform: build_top_level_waveform(
                    silence_of_beats(pattern_beats, self.tempo),
                    program.level_db(),
                ),
                start: Some(base),
                repeat_every,
            })
            .unwrap();
        Some(message)
    }

    /// Schedules one step of a live sequenceable program at the next
    /// occurrence of its sixteenth — this cycle when the instant is still
    /// ahead, otherwise the next cycle.
    ///
    /// The cycle phase and period are read from the anchor step's marks in
    /// `status`; `default_repeat_measures` supplies the period when only a
    /// queued (pending) anchor is visible. Returns a user-visible warning
    /// when the step can't sound (no upcoming occurrence), and `None`
    /// otherwise.
    pub fn play_step(
        &self,
        set: &ProgramSet,
        program_index: usize,
        sixteenth: u8,
        status: &tracker::Status<WaveformId, MarkId>,
        default_repeat_measures: Option<u32>,
    ) -> Option<String> {
        let program = set.program(program_index)?;
        let sequence = program.sequence()?;
        let now = time::Instant::now();
        let anchor = WaveformId::Step {
            program: program_index,
            sixteenth: ANCHOR_SIXTEENTH,
        };
        // The cycle the family is in now, and the queued next cycle.
        let mut cycle_start: Option<time::Instant> = None;
        let mut next_cycle_start: Option<time::Instant> = None;
        for mark in &status.marks {
            if mark.waveform_id == anchor && mark.mark_id == MarkId::TopLevel {
                if mark.start <= now {
                    cycle_start = Some(cycle_start.map_or(mark.start, |s| s.max(mark.start)));
                } else {
                    next_cycle_start =
                        Some(next_cycle_start.map_or(mark.start, |s| s.min(mark.start)));
                }
            }
        }
        let offset =
            duration_from_beats_f32(self.tempo, sequencer::sixteenth_beat(sixteenth) - 1.0);
        let start = match (cycle_start, next_cycle_start) {
            (Some(cycle), _) if cycle + offset > now => cycle + offset,
            (_, Some(next_cycle)) => next_cycle + offset,
            (Some(_), None) => {
                return Some(format!(
                    "Beat {} has passed and the sequence doesn't repeat",
                    sequencer::sixteenth_beat(sixteenth)
                ));
            }
            (None, None) => return None,
        };
        let repeat_every = match (cycle_start, next_cycle_start) {
            (Some(cycle), Some(next_cycle)) => Some(next_cycle - cycle),
            _ => default_repeat_measures.map(|measures| {
                duration_from_beats(self.tempo, (measures * self.beats_per_measure) as u64)
            }),
        };
        let mut step = sequence.step_waveform.clone();
        substitute_current_slider_values(&mut step, program.sliders());
        let _ = self.fast_sender.send(tracker::Command::Play {
            id: WaveformId::Step {
                program: program_index,
                sixteenth,
            },
            waveform: build_leveled_waveform(step, program.level_db()),
            start: Some(start),
            repeat_every,
        });
        None
    }

    /// Plays a note waveform under `WaveformId::Key(key)` immediately via
    /// the fast route, wrapped with the top-level amplitude/terminator
    /// marks at `level_db`.
    pub fn play_note(&self, key: u8, waveform: waveform::Waveform<MarkId>, level_db: f32) {
        let _ = self.fast_sender.send(tracker::Command::Play {
            id: WaveformId::Key(key),
            waveform: build_top_level_waveform(waveform, level_db),
            start: None,
            repeat_every: None,
        });
    }

    /// Fades out the waveforms matched by the selector over a short ramp. A
    /// no-op if no matching waveform is playing.
    pub fn stop_waveform(&self, selector: WaveformSelector) {
        self.fast_sender
            .send(tracker::Command::Modify {
                selector,
                mark_id: MarkId::Terminator,
                waveform: stop_ramp(),
            })
            .unwrap();
    }

    /// Removes the pending (not-yet-started) waveforms matched by the selector.
    /// With `after`, only pending waveform starting at or after that instant
    /// are removed; earlier ones play out their current occurrence without
    /// repeating.
    pub fn remove_pending(&self, selector: WaveformSelector, after: Option<time::Instant>) {
        let _ = self
            .fast_sender
            .send(tracker::Command::RemovePending { selector, after });
    }

    /// Replaces the waveform under `mark_id` on every waveform matched by the
    /// selector.
    pub fn modify(
        &self,
        selector: WaveformSelector,
        mark_id: MarkId,
        waveform: waveform::Waveform<MarkId>,
    ) {
        let _ = self.fast_sender.send(tracker::Command::Modify {
            selector,
            mark_id,
            waveform,
        });
    }

    /// Starts the two alternating Beats waveforms that keep time for the
    /// rest of the runtime. Blocks on `status_receiver` until the first
    /// Beats waveform is scheduled so the second can start a measure later.
    pub fn start_beats(
        &self,
        evaluator: &Evaluator,
        status_receiver: &mpsc::Receiver<tracker::Status<WaveformId, MarkId>>,
    ) {
        // Play the odd Beats waveform starting immediately and repeating every two measures
        self.precompute_sender
            .send(tracker::Command::Play {
                id: WaveformId::Beats(false),
                waveform: self.beats_waveform(evaluator),
                start: None,
                repeat_every: Some(
                    duration_from_beats(self.tempo, self.beats_per_measure as u64) * 2,
                ),
            })
            .unwrap();
        // We need to wait to start the even Beats until we know when the odd Beats started
        'start_even_beats: loop {
            if let Ok(status) = status_receiver.recv() {
                for mark in status.marks {
                    if mark.waveform_id == WaveformId::Beats(false)
                        && mark.mark_id == MarkId::TopLevel
                    {
                        self.precompute_sender
                            .send(tracker::Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: self.beats_waveform(evaluator),
                                start: Some(mark.start + mark.duration),
                                repeat_every: Some(
                                    duration_from_beats(self.tempo, self.beats_per_measure as u64)
                                        * 2,
                                ),
                            })
                            .unwrap();
                        break 'start_even_beats;
                    }
                }
            }
        }
    }

    /// Builds the per-measure beats waveform — a sequence of `mark`-tagged
    /// short silences, one per beat — used to keep timing visible to the
    /// rest of the runtime.
    pub fn beats_waveform(&self, evaluator: &Evaluator) -> waveform::Waveform<MarkId> {
        let seconds_per_beat = duration_from_beats(self.tempo, 1);
        let mut ws = Vec::new();
        for i in 0..self.beats_per_measure {
            ws.push(format!(
                "0 | fin(time - {}) | seq(time - {}) | mark({})",
                seconds_per_beat.as_secs_f32(),
                seconds_per_beat.as_secs_f32(),
                i + 1
            ));
        }
        let source = format!("<[{}]>", ws.join(", "));
        let bindings: Vec<expr::SourceBinding<MarkId, Source>> =
            vec![expr::Binding::Open(vec!["__prelude".to_string()]).into()];
        match evaluator
            .evaluate_source(&source, &bindings)
            .map(|s| s.expr)
        {
            Ok(expr::Expr::Seq { waveform, .. }) => match waveform.expr {
                expr::Expr::Waveform(waveform) => waveform::Waveform::<MarkId>::Marked {
                    id: MarkId::TopLevel,
                    waveform: Box::new(optimizer::optimize(waveform)),
                },
                expr => panic!("Error creating beats waveform with seq, got {}", expr),
            },
            Ok(expr) => panic!("Error creating beats waveform, got {}", expr),
            Err(message) => panic!("Error evaluating beats waveform: {}", message),
        }
    }
}

/// Builds the short fade-out ramp to be substituted at a `Terminator` mark to
/// stop a waveform.
pub fn stop_ramp() -> waveform::Waveform<MarkId> {
    use waveform::{Operator, Waveform::*};
    const STOP_DURATION_SECS: f32 = 0.05;
    Fin {
        length: Box::new(BinaryPointOp(
            Operator::Subtract,
            Box::new(Time(())),
            Box::new(Const(STOP_DURATION_SECS)),
        )),
        waveform: Box::new(BinaryPointOp(
            Operator::Subtract,
            Box::new(Const(1.0)),
            Box::new(BinaryPointOp(
                Operator::Multiply,
                Box::new(Time(())),
                Box::new(Const(1.0 / STOP_DURATION_SECS)),
            )),
        )),
    }
}

/// Wraps `waveform` with the standard voice marks: an `Amplitude` mark at
/// the amplitude for `level_db` and a `Terminator` mark used to stop it.
fn build_leveled_waveform(
    waveform: waveform::Waveform<MarkId>,
    level_db: f32,
) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryPointOp, Const, Marked};
    BinaryPointOp(
        waveform::Operator::Multiply,
        Box::new(BinaryPointOp(
            waveform::Operator::Multiply,
            Box::new(waveform),
            Box::new(Marked {
                id: MarkId::Amplitude,
                waveform: Box::new(Const(db_to_amplitude(level_db))),
            }),
        )),
        Box::new(Marked {
            id: MarkId::Terminator,
            waveform: Box::new(Const(1.0)),
        }),
    )
}

/// Wraps `waveform` with the standard top-level marks: a `TopLevel` mark
/// around the amplitude/terminator wrap of `build_leveled_waveform`.
fn build_top_level_waveform(
    waveform: waveform::Waveform<MarkId>,
    level_db: f32,
) -> waveform::Waveform<MarkId> {
    waveform::Waveform::Marked {
        id: MarkId::TopLevel,
        waveform: Box::new(build_leveled_waveform(waveform, level_db)),
    }
}

/// Builds a silent waveform lasting the given number of beats.
fn silence_of_beats(beats: f32, tempo: u32) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryPointOp, Const, Fin, Time};
    Fin {
        length: Box::new(BinaryPointOp(
            waveform::Operator::Subtract,
            Box::new(Time(())),
            Box::new(Const(beats * 60.0 / tempo as f32)),
        )),
        waveform: Box::new(Const(0.0)),
    }
}

// Returns the start time of the next measure
fn next_measure_start(status: &tracker::Status<WaveformId, MarkId>) -> time::Instant {
    for mark in &status.marks {
        match mark.waveform_id {
            WaveformId::Beats(_)
                if mark.mark_id == MarkId::TopLevel && mark.start > time::Instant::now() =>
            {
                return mark.start;
            }
            _ => (),
        }
    }
    panic!("No next measure found in marks");
}

fn duration_from_beats(tempo: u32, beats: u64) -> time::Duration {
    time::Duration::from_secs_f32(beats as f32 * 60.0 / tempo as f32)
}

/// Fractional-beat sibling of `duration_from_beats`, for sixteenth offsets.
fn duration_from_beats_f32(tempo: u32, beats: f32) -> time::Duration {
    time::Duration::from_secs_f32(beats * 60.0 / tempo as f32)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn empty_status() -> tracker::Status<WaveformId, MarkId> {
        tracker::Status {
            buffer_start: Instant::now(),
            marks: vec![],
            buffer: None,
            tracker_load: None,
            allocations_per_sample: None,
        }
    }

    /// Builds a program set whose program 0 is sequenceable with beats
    /// [1, 2.5], evaluated so its sequence is available.
    fn sequenced_set() -> ProgramSet {
        let source = "on_beats = fn(w, bs) => w;\n\
                      #{level_db=0}\n\
                      _ = on_beats(1 | fin(time - 1), [1, 2.5]);\n";
        let (mut set, message) =
            ProgramSet::from_source(source.to_string(), std::path::PathBuf::new())
                .expect("test source should parse");
        assert!(message.is_empty(), "{}", message);
        let evaluator = Evaluator::new(8000, 90, std::path::PathBuf::new());
        set.evaluate_and_record(&evaluator, 0)
            .expect("test program should evaluate");
        assert!(set.program(0).unwrap().sequence().is_some());
        set
    }

    fn test_player() -> (
        Player,
        mpsc::Receiver<tracker::Command<WaveformId, MarkId>>,
        mpsc::Receiver<tracker::Command<WaveformId, MarkId>>,
    ) {
        let (precompute_sender, precompute_receiver) = mpsc::channel();
        let (fast_sender, fast_receiver) = mpsc::channel();
        let player = Player::new(90, 4, precompute_sender, fast_sender);
        (player, precompute_receiver, fast_receiver)
    }

    fn anchor_mark(program: usize, start: Instant) -> tracker::Mark<WaveformId, MarkId> {
        tracker::Mark {
            waveform_id: WaveformId::Step {
                program,
                sixteenth: ANCHOR_SIXTEENTH,
            },
            mark_id: MarkId::TopLevel,
            start,
            duration: Duration::ZERO,
        }
    }

    #[test]
    fn play_program_steps_sends_one_play_per_beat_plus_anchor() {
        let set = sequenced_set();
        let (player, _precompute_receiver, fast_receiver) = test_player();

        let message = player.play_program_steps(&set, 0, &empty_status(), false, Some(1));
        assert!(message.is_some());

        let commands: Vec<_> = fast_receiver.try_iter().collect();
        let plays: Vec<_> = commands
            .iter()
            .map(|c| match c {
                tracker::Command::Play {
                    id,
                    waveform,
                    start,
                    repeat_every,
                } => (id, waveform, start.unwrap(), repeat_every.unwrap()),
                _ => panic!("expected only Play commands"),
            })
            .collect();
        assert_eq!(plays.len(), 3);

        let one_measure = duration_from_beats(90, 4);
        let (beat_1, beat_2_5, anchor) = (&plays[0], &plays[1], &plays[2]);
        assert_eq!(
            *beat_1.0,
            WaveformId::Step {
                program: 0,
                sixteenth: 0
            }
        );
        assert_eq!(
            *beat_2_5.0,
            WaveformId::Step {
                program: 0,
                sixteenth: 6
            }
        );
        assert_eq!(
            *anchor.0,
            WaveformId::Step {
                program: 0,
                sixteenth: ANCHOR_SIXTEENTH
            }
        );
        // Steps are offset from the anchored cycle start by their beats.
        assert_eq!(beat_1.2, anchor.2);
        assert_eq!(beat_2_5.2 - beat_1.2, duration_from_beats_f32(90, 1.5));
        // All entries share the loop period.
        assert!(plays.iter().all(|p| p.3 == one_measure));
        // Only the anchor carries the TopLevel mark.
        assert!(!format!("{}", beat_1.1).contains("top-level"));
        assert!(!format!("{}", beat_2_5.1).contains("top-level"));
        assert!(format!("{}", anchor.1).contains("top-level"));
    }

    #[test]
    fn play_step_schedules_this_cycle_when_sixteenth_is_ahead() {
        let set = sequenced_set();
        let (player, _precompute_receiver, fast_receiver) = test_player();

        let now = Instant::now();
        let cycle = now - Duration::from_millis(100);
        let next_cycle = cycle + duration_from_beats(90, 4);
        let mut status = empty_status();
        status.marks = vec![anchor_mark(0, cycle), anchor_mark(0, next_cycle)];

        // Sixteenth 8 is beat 3, two beats into the cycle — still ahead.
        let message = player.play_step(&set, 0, 8, &status, None);
        assert!(message.is_none());

        let commands: Vec<_> = fast_receiver.try_iter().collect();
        assert_eq!(commands.len(), 1);
        let tracker::Command::Play {
            id,
            start,
            repeat_every,
            ..
        } = &commands[0]
        else {
            panic!("expected a Play command");
        };
        assert_eq!(
            *id,
            WaveformId::Step {
                program: 0,
                sixteenth: 8
            }
        );
        assert_eq!(start.unwrap(), cycle + duration_from_beats_f32(90, 2.0));
        assert_eq!(repeat_every.unwrap(), next_cycle - cycle);
    }

    #[test]
    fn play_step_rolls_to_next_cycle_when_sixteenth_passed() {
        let set = sequenced_set();
        let (player, _precompute_receiver, fast_receiver) = test_player();

        let now = Instant::now();
        let cycle = now - Duration::from_secs(1);
        let next_cycle = cycle + duration_from_beats(90, 4);
        let mut status = empty_status();
        status.marks = vec![anchor_mark(0, cycle), anchor_mark(0, next_cycle)];

        // Sixteenth 0 is beat 1, which started the cycle — already passed.
        let message = player.play_step(&set, 0, 0, &status, None);
        assert!(message.is_none());

        let commands: Vec<_> = fast_receiver.try_iter().collect();
        assert_eq!(commands.len(), 1);
        let tracker::Command::Play { start, .. } = &commands[0] else {
            panic!("expected a Play command");
        };
        assert_eq!(start.unwrap(), next_cycle);
    }

    #[test]
    fn play_step_warns_when_passed_and_not_repeating() {
        let set = sequenced_set();
        let (player, _precompute_receiver, fast_receiver) = test_player();

        let now = Instant::now();
        let mut status = empty_status();
        status.marks = vec![anchor_mark(0, now - Duration::from_secs(1))];

        let message = player.play_step(&set, 0, 0, &status, None);
        assert!(message.unwrap().contains("passed"));
        assert!(fast_receiver.try_iter().next().is_none());
    }
}
