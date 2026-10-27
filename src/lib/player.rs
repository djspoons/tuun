//! Sends waveforms to the tracker — all tracker I/O lives here.
//!
//! `Player` owns the sender that goes straight to the tracker. All methods take
//! `&self`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, mpsc};
use std::time;

use crate::diagnostics::Source;
use crate::environment::Environment;
use crate::expr;
use crate::generator;
use crate::ids::{ANCHOR_SIXTEENTH, MarkId, WaveformId, WaveformSelector};
use crate::optimizer;
use crate::programs::{Evaluated, Program, ProgramSet};
use crate::sequencer;
use crate::slider;
use crate::trace_walk::{self, WaveformMarks};
use crate::tracker;
use crate::waveform;

/// The marks of a waveform sent to play, and the program text they index into.
#[derive(Clone, Debug)]
pub struct PlayedMarks {
    /// The index of the program the waveform comes from.
    pub program: usize,
    /// The program text the marks' byte ranges index into.
    pub text: Arc<str>,
    /// The marks of the waveform as evaluated.
    pub marks: Arc<WaveformMarks>,
    /// Additional `marks`, such as a step's literal marks.
    pub extra: Vec<trace_walk::Mark>,
}

/// Returns the played marks of a waveform of the program at `program_index`,
/// whose marks as evaluated are `marks`.
fn played_marks(
    program_index: usize,
    program: &Program,
    marks: &Arc<WaveformMarks>,
    extra: Vec<trace_walk::Mark>,
) -> PlayedMarks {
    PlayedMarks {
        program: program_index,
        text: program.evaluated_text().clone(),
        marks: marks.clone(),
        extra,
    }
}

/// Re-records the marks in `waveform_marks` of every step of the program at
/// `program_index` from its current sequence in `set`, without playing anything.
///
/// A step whose sixteenth no longer holds a step of the sequence loses its
/// entry. Does nothing when the program has no sequence.
pub fn record_step_marks(
    set: &ProgramSet,
    program_index: usize,
    waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
) {
    let Some(program) = set.program(program_index) else {
        return;
    };
    let Some(sequence) = program.sequence() else {
        return;
    };
    let sixteenths: HashSet<u8> = sequence
        .step_beats()
        .map(sequencer::sixteenth_for_beat)
        .collect();
    waveform_marks.retain(|id, _| match id {
        WaveformId::Step { program, sixteenth } if *program == program_index => {
            sixteenths.contains(sixteenth)
        }
        _ => true,
    });
    for (id, marks) in waveform_marks.iter_mut() {
        if let WaveformId::Step {
            program: p,
            sixteenth,
        } = id
            && *p == program_index
        {
            let extra = sequence.step_literal_marks(*sixteenth);
            *marks = played_marks(program_index, program, &sequence.step_marks, extra);
        }
    }
}

pub fn db_to_amplitude(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// Substitutes every slider's current value into the matching `Marked { id:
/// Slider { .. }, … }` nodes of `waveform`.
///
/// Sliders of every program in `set` are applied, not just those of the program
/// the waveform came from.
pub fn substitute_current_slider_values(
    waveform: &mut waveform::Waveform<MarkId>,
    set: &ProgramSet,
) {
    for (program_index, program) in set.programs().iter().enumerate() {
        let sliders = program.sliders();
        for (config, &normalized) in sliders.configs().iter().zip(sliders.normalized_values()) {
            let value = slider::denormalize(&config.function, normalized).unwrap_or(0.0);
            let mark_id = MarkId::Slider {
                program: program_index,
                label: config.label.clone(),
            };
            let marked = waveform::Waveform::Marked {
                id: mark_id.clone(),
                waveform: Box::new(waveform::Waveform::Const(value)),
            };
            waveform::substitute(waveform, &mark_id, &marked);
        }
    }
}

/// When a played program starts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// Immediately.
    Now,
    /// At the next measure boundary.
    NextMeasure,
    /// At the given instant. An instant already past plays from where the
    /// waveform would be by now, dropping what was missed.
    At(time::Instant),
}

/// What a launch plays, as its status text names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Played {
    Waveform,
    Sequence,
}

/// Returns the status text for playing the `played` of the program named
/// `display_name`, repeating every `loop_after_beats` beats if given.
pub fn play_message(played: Played, display_name: &str, loop_after_beats: Option<u32>) -> String {
    let noun = match played {
        Played::Waveform => "waveform",
        Played::Sequence => "sequence",
    };
    match loop_after_beats {
        Some(beats) => format!("Looping {} {} every {} beats", noun, display_name, beats),
        None => format!("Playing {} {}", noun, display_name),
    }
}

pub struct Player {
    tempo: u32,
    beats_per_measure: u32,
    sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
}

impl Player {
    pub fn new(
        tempo: u32,
        beats_per_measure: u32,
        sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    ) -> Player {
        Player {
            tempo,
            beats_per_measure,
            sender,
        }
    }

    /// Returns the tempo in beats per minute.
    pub fn tempo(&self) -> u32 {
        self.tempo
    }

    /// Returns the number of beats in a measure.
    pub fn beats_per_measure(&self) -> u32 {
        self.beats_per_measure
    }

    /// Plays the program at `program_index`, substituting its current slider
    /// values. Returns the user-visible message, or `None` when the program's
    /// text didn't evaluate to a waveform or a sequence (or the index is out of
    /// range) and nothing was played.
    ///
    /// A waveform plays as one tracker entry under
    /// `WaveformId::Program(program_index)`. A sequence plays decomposed: one
    /// scheduled, independently repeating `WaveformId::Step` per listed step,
    /// plus a silent anchor step that spans each pass. Either starts where
    /// `start` says, repeating every `loop_after_beats` beats if given.
    ///
    /// Records the marks of each waveform played in `waveform_marks`, by the id
    /// it plays under.
    pub fn play_program(
        &self,
        set: &ProgramSet,
        program_index: usize,
        status: &tracker::Status<WaveformId>,
        start: Start,
        loop_after_beats: Option<u32>,
        waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
    ) -> Option<String> {
        // A sequence always plays decomposed, so the sequencer can edit its
        // steps live.
        if set.program(program_index)?.sequence().is_some() {
            self.play_sequence(
                set,
                program_index,
                status,
                start,
                loop_after_beats,
                waveform_marks,
            )
        } else {
            self.play_waveform(
                set,
                program_index,
                status,
                start,
                loop_after_beats,
                waveform_marks,
            )
        }
    }

    /// Plays the waveform of the program at `program_index` as described by
    /// [`Player::play_program`].
    fn play_waveform(
        &self,
        set: &ProgramSet,
        program_index: usize,
        status: &tracker::Status<WaveformId>,
        start: Start,
        loop_after_beats: Option<u32>,
        waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
    ) -> Option<String> {
        let program = set.program(program_index)?;
        let Some(Evaluated::Waveform(waveform, marks)) = program.evaluated() else {
            return None;
        };
        let message = play_message(
            Played::Waveform,
            &set.display_name(program_index),
            loop_after_beats,
        );
        let repeat_every =
            loop_after_beats.map(|beats| duration_from_beats(self.tempo, beats.into()));
        let now = time::Instant::now();
        let start_instant = match start {
            Start::Now => None,
            Start::NextMeasure => Some(next_measure_start(status, now).unwrap_or_else(|| {
                panic!(
                    "No next measure found at {:?}: {:?}",
                    now, status.start_times
                )
            })),
            Start::At(instant) => Some(instant),
        };
        let mut waveform = waveform.clone();
        // Substitute the program's current slider positions before handing
        // the waveform to the tracker (since the cached ones may be old).
        substitute_current_slider_values(&mut waveform, set);
        let id = WaveformId::Program(program_index);
        self.sender
            .send(tracker::Command::Play {
                id: id.clone(),
                waveform: build_leveled_waveform(waveform, program.level_db()),
                start: start_instant,
                repeat_every,
            })
            .unwrap();
        waveform_marks.insert(id, played_marks(program_index, program, marks, vec![]));
        Some(message)
    }

    /// Plays the sequence of the program at `program_index` as described by
    /// [`Player::play_program`].
    fn play_sequence(
        &self,
        set: &ProgramSet,
        program_index: usize,
        status: &tracker::Status<WaveformId>,
        start: Start,
        loop_after_beats: Option<u32>,
        waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
    ) -> Option<String> {
        let program = set.program(program_index)?;
        let sequence = program.sequence()?;
        let message = play_message(
            Played::Sequence,
            &set.display_name(program_index),
            loop_after_beats,
        );
        let repeat_every =
            loop_after_beats.map(|beats| duration_from_beats(self.tempo, beats.into()));
        let now = time::Instant::now();
        let base = match start {
            Start::Now => now,
            Start::NextMeasure => next_measure_start(status, now).unwrap_or_else(|| {
                panic!(
                    "No next measure found at {:?}: {:?}",
                    now, status.start_times
                )
            }),
            Start::At(instant) => instant,
        };
        let mut step = sequence.step_waveform.clone();
        substitute_current_slider_values(&mut step, set);
        for step_beat in sequence.step_beats() {
            let sixteenth = sequencer::sixteenth_for_beat(step_beat);
            let id = WaveformId::Step {
                program: program_index,
                sixteenth,
            };
            self.sender
                .send(tracker::Command::Play {
                    id: id.clone(),
                    waveform: build_leveled_waveform(step.clone(), program.level_db()),
                    start: Some(base + beat_offset(self.tempo, step_beat)),
                    repeat_every,
                })
                .unwrap();
            let extra = sequence.step_literal_marks(sixteenth);
            let marks = played_marks(program_index, program, &sequence.step_marks, extra);
            waveform_marks.insert(id, marks);
        }
        // The anchor spans the whole pattern so the family reads as playing
        // for as long as its steps do.
        let pattern_beats =
            sequence
                .step_beats()
                .fold(self.beats_per_measure as f32, |acc, step_beat| {
                    let measures = ((step_beat - 0.75) / self.beats_per_measure as f32).ceil();
                    acc.max(measures * self.beats_per_measure as f32)
                });
        self.sender
            .send(tracker::Command::Play {
                id: WaveformId::Step {
                    program: program_index,
                    sixteenth: ANCHOR_SIXTEENTH,
                },
                waveform: build_leveled_waveform(
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
    /// The cycle phase and period are read from the anchor step's starts in
    /// `status`; the program's `loop_after_beats` supplies the period when
    /// only a queued (pending) anchor is visible. Returns a user-visible warning
    /// when the step can't sound (no upcoming occurrence), and `None`
    /// otherwise. Records the step's marks in `waveform_marks` when it plays.
    pub fn play_step(
        &self,
        set: &ProgramSet,
        program_index: usize,
        sixteenth: u8,
        status: &tracker::Status<WaveformId>,
        waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
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
        for &(_, start) in status.start_times.iter().filter(|(id, _)| *id == anchor) {
            if start <= now {
                cycle_start = Some(cycle_start.map_or(start, |s| s.max(start)));
            } else {
                next_cycle_start = Some(next_cycle_start.map_or(start, |s| s.min(start)));
            }
        }
        let offset = beat_offset(self.tempo, sequencer::sixteenth_beat(sixteenth));
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
            // A running anchor with nothing queued behind it is a one-shot
            // family; the step must not outlive it.
            (Some(_), None) => None,
            _ => program
                .loop_after_beats()
                .map(|beats| duration_from_beats(self.tempo, beats.into())),
        };
        let mut step = sequence.step_waveform.clone();
        substitute_current_slider_values(&mut step, set);
        let id = WaveformId::Step {
            program: program_index,
            sixteenth,
        };
        let _ = self.sender.send(tracker::Command::Play {
            id: id.clone(),
            waveform: build_leveled_waveform(step, program.level_db()),
            start: Some(start),
            repeat_every,
        });
        let extra = sequence.step_literal_marks(sixteenth);
        let marks = played_marks(program_index, program, &sequence.step_marks, extra);
        waveform_marks.insert(id, marks);
        None
    }

    /// Plays a note waveform under `WaveformId::Key(key)` immediately,
    /// wrapped with the amplitude/terminator marks at `level_db`, and records its marks, `marks`, in `waveform_marks`.
    pub fn play_note(
        &self,
        key: u8,
        waveform: waveform::Waveform<MarkId>,
        level_db: f32,
        marks: PlayedMarks,
        waveform_marks: &mut HashMap<WaveformId, PlayedMarks>,
    ) {
        let id = WaveformId::Key(key);
        let _ = self.sender.send(tracker::Command::Play {
            id: id.clone(),
            waveform: build_leveled_waveform(waveform, level_db),
            start: None,
            repeat_every: None,
        });
        waveform_marks.insert(id, marks);
    }

    /// Fades out the waveforms matched by the selector over a short ramp. A
    /// no-op if no matching waveform is playing.
    pub fn stop_waveform(&self, selector: WaveformSelector) {
        self.sender
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
            .sender
            .send(tracker::Command::RemovePending { selector, after });
    }

    /// Replaces each `mark_id` mark with `waveform` on every waveform matched
    /// by the selector.
    pub fn modify(
        &self,
        selector: WaveformSelector,
        mark_id: MarkId,
        waveform: waveform::Waveform<MarkId>,
    ) {
        let _ = self.sender.send(tracker::Command::Modify {
            selector,
            mark_id,
            waveform,
        });
    }

    /// Starts the two alternating Beats waveforms that keep time for the
    /// rest of the runtime, with their finite parts replaced by their samples
    /// when `precompute` is set (see [`generator::Generator::precompute`]).
    /// Blocks on `status_receiver` until the first Beats waveform is
    /// scheduled so the second can start a measure later.
    pub fn start_beats(
        &self,
        environment: &Environment,
        status_receiver: &mpsc::Receiver<tracker::Status<WaveformId>>,
        precompute: bool,
    ) {
        let mut beats = self.beats_waveform(environment);
        if precompute {
            beats = generator::Generator::new(environment.sample_rate()).precompute(beats);
        }
        let measure = duration_from_beats(self.tempo, self.beats_per_measure as u64);
        // Play the odd Beats waveform starting immediately and repeating every two measures
        self.sender
            .send(tracker::Command::Play {
                id: WaveformId::Beats(false),
                waveform: beats.clone(),
                start: None,
                repeat_every: Some(measure * 2),
            })
            .unwrap();
        // We need to wait to start the even Beats until we know when the odd Beats started
        'start_even_beats: loop {
            if let Ok(status) = status_receiver.recv() {
                for (id, start) in status.start_times {
                    if id == WaveformId::Beats(false) {
                        self.sender
                            .send(tracker::Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: beats,
                                start: Some(start + measure),
                                repeat_every: Some(measure * 2),
                            })
                            .unwrap();
                        break 'start_even_beats;
                    }
                }
            }
        }
    }

    /// Builds the per-measure beats waveform: one measure of metronome clicks,
    /// one on each beat, whose level a `MetronomeAmplitude` mark sets.
    pub fn beats_waveform(&self, environment: &Environment) -> waveform::Waveform<MarkId> {
        let seconds_per_beat = duration_from_beats(self.tempo, 1).as_secs_f32();
        let mut ws = Vec::new();
        for i in 0..self.beats_per_measure {
            ws.push(format!(
                "({} * _metronome_amplitude) | fin(time - {}) | seq(time - {})",
                click_source(i == 0),
                seconds_per_beat,
                seconds_per_beat,
            ));
        }
        let source = format!("<[{}]>", ws.join(", "));
        let metronome_amplitude = metronome_amplitude(false);
        let bindings: Vec<expr::SourceBinding<MarkId, Source>> = vec![
            expr::Binding::Open(vec!["__prelude".to_string()]).into(),
            expr::SourceBinding::definition(
                expr::Pattern::Identifier("_metronome_amplitude".to_string()),
                expr::Expr::Waveform(metronome_amplitude).into(),
            ),
        ];
        match environment
            .evaluate_source(&source, &bindings)
            .map(|s| s.expr)
        {
            Ok(expr::Expr::Seq { waveform, .. }) => match waveform.expr {
                expr::Expr::Waveform(waveform) => optimizer::optimize(waveform),
                expr => panic!("Error creating beats waveform with seq, got {}", expr),
            },
            Ok(expr) => panic!("Error creating beats waveform, got {}", expr),
            Err(message) => panic!("Error evaluating beats waveform: {}", message),
        }
    }

    /// Turns the metronome click on both Beats waveforms on or off.
    pub fn set_metronome(&self, on: bool) {
        self.modify(
            WaveformSelector::AllBeats,
            MarkId::MetronomeAmplitude,
            metronome_amplitude(on),
        );
    }
}

/// Returns tuun source for a metronome click followed by silence, pitched
/// higher when `accent`.
fn click_source(accent: bool) -> String {
    const CLICK_SECS: f32 = 0.02;
    const Q: f32 = 4.0;
    // Measured peaks land around -12 dBFS.
    const GAIN: f32 = 1.4;
    let center_hz = if accent { 2000.0 } else { 1000.0 };
    // A constant-peak-gain band-pass filter (RBJ Audio EQ Cookbook, "BPF
    // (constant 0 dB peak gain)") applied to a decaying noise burst.
    format!(
        "let
           w0 = 2 * 3.14159265 * {center_hz} / sample_rate,
           alpha = sin(w0) / (2 * {Q}),
           a0 = 1 + alpha,
         in
           ((noise * (1 - time / {CLICK_SECS}) * {GAIN}
             | filter([alpha / a0, 0, -alpha / a0],
                      [-2 * sin(w0 + 3.14159265 / 2) / a0, (1 - alpha) / a0])
             | fin(time - {CLICK_SECS})
             | seq(time - {CLICK_SECS}))
            \\ 0)"
    )
}

/// Returns the `MetronomeAmplitude` mark holding the click's amplitude: 1
/// when `on`, 0 otherwise.
fn metronome_amplitude(on: bool) -> waveform::Waveform<MarkId> {
    waveform::Waveform::Marked {
        id: MarkId::MetronomeAmplitude,
        waveform: Box::new(waveform::Waveform::Const(if on { 1.0 } else { 0.0 })),
    }
}

/// Returns the `Amplitude` mark holding the amplitude for `level_db`.
pub fn amplitude(level_db: f32) -> waveform::Waveform<MarkId> {
    waveform::Waveform::Marked {
        id: MarkId::Amplitude,
        waveform: Box::new(waveform::Waveform::Const(db_to_amplitude(level_db))),
    }
}

/// Returns `waveform` scaled by a fresh `Terminator` mark: `waveform *
/// Marked(Terminator, 1)`.
pub fn terminate(waveform: waveform::Waveform<MarkId>) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryOp, Const, Marked};
    BinaryOp(
        waveform::BinaryOperator::Multiply,
        Box::new(waveform),
        Box::new(Marked {
            id: MarkId::Terminator,
            waveform: Box::new(Const(1.0)),
        }),
    )
}

/// Returns the replacement for a `Terminator` mark that fades a voice out over
/// a short ramp and then ends it.
pub fn stop_ramp() -> waveform::Waveform<MarkId> {
    use waveform::{BinaryOperator, Waveform::*};
    const STOP_DURATION_SECS: f32 = 0.05;
    terminate(Fin {
        length: Box::new(BinaryOp(
            BinaryOperator::Subtract,
            Box::new(Time(())),
            Box::new(Const(STOP_DURATION_SECS)),
        )),
        waveform: Box::new(BinaryOp(
            BinaryOperator::Subtract,
            Box::new(Const(1.0)),
            Box::new(BinaryOp(
                BinaryOperator::Multiply,
                Box::new(Time(())),
                Box::new(Const(1.0 / STOP_DURATION_SECS)),
            )),
        )),
        state: (),
    })
}

/// Wraps `waveform` with the standard voice marks: an `Amplitude` mark at
/// the amplitude for `level_db` and a `Terminator` mark used to stop it.
fn build_leveled_waveform(
    waveform: waveform::Waveform<MarkId>,
    level_db: f32,
) -> waveform::Waveform<MarkId> {
    terminate(waveform::Waveform::BinaryOp(
        waveform::BinaryOperator::Multiply,
        Box::new(waveform),
        Box::new(amplitude(level_db)),
    ))
}

/// Builds a silent waveform lasting the given number of beats.
fn silence_of_beats(beats: f32, tempo: u32) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryOp, Const, Fin, Time};
    Fin {
        length: Box::new(BinaryOp(
            waveform::BinaryOperator::Subtract,
            Box::new(Time(())),
            Box::new(Const(beats * 60.0 / tempo as f32)),
        )),
        waveform: Box::new(Const(0.0)),
        state: (),
    }
}

/// Returns the first measure boundary after `now`: the earliest start of a
/// Beats waveform, or `None` when `status` holds no Beats waveform starting
/// after `now`.
pub fn next_measure_start(
    status: &tracker::Status<WaveformId>,
    now: time::Instant,
) -> Option<time::Instant> {
    measure_starts(status).filter(|&start| start > now).min()
}

/// Returns the latest measure boundary at or before `now`, or `None` when
/// `status` holds no Beats waveform starting by then.
pub fn previous_measure_start(
    status: &tracker::Status<WaveformId>,
    now: time::Instant,
) -> Option<time::Instant> {
    measure_starts(status).filter(|&start| start <= now).max()
}

/// Returns the start of every Beats waveform in `status`.
fn measure_starts(
    status: &tracker::Status<WaveformId>,
) -> impl Iterator<Item = time::Instant> + '_ {
    status
        .start_times
        .iter()
        .filter(|(id, _)| id.is_beats())
        .map(|(_, start)| *start)
}

/// Returns how long `beats` beats last at `tempo` beats per minute.
pub fn duration_from_beats(tempo: u32, beats: u64) -> time::Duration {
    time::Duration::from_secs_f32(beats as f32 * 60.0 / tempo as f32)
}

/// Returns the time from the start of a sequence's pass to the start of its
/// step on `beat` (1-based), at `tempo` beats per minute.
pub fn beat_offset(tempo: u32, beat: f32) -> time::Duration {
    duration_from_beats_f32(tempo, beat - 1.0)
}

/// Fractional-beat sibling of `duration_from_beats`, for sixteenth offsets.
fn duration_from_beats_f32(tempo: u32, beats: f32) -> time::Duration {
    time::Duration::from_secs_f32(beats * 60.0 / tempo as f32)
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use super::*;

    fn empty_status() -> tracker::Status<WaveformId> {
        tracker::Status {
            buffer_start: Instant::now(),
            start_times: Vec::new(),
            tracker_load: None,
            allocations: None,
        }
    }

    #[test]
    fn measure_starts_bracket_now() {
        let t0 = Instant::now();
        let mut status = empty_status();
        // Out of order, as the odd and even Beats waveforms interleave.
        for (k, even) in [(1, true), (0, false), (2, false)] {
            status
                .start_times
                .push((WaveformId::Beats(even), t0 + Duration::from_secs(4 * k)));
        }
        let now = t0 + Duration::from_secs(5);
        assert_eq!(
            previous_measure_start(&status, now),
            Some(t0 + Duration::from_secs(4))
        );
        assert_eq!(
            next_measure_start(&status, now),
            Some(t0 + Duration::from_secs(8))
        );
        let later = t0 + Duration::from_secs(9);
        assert_eq!(next_measure_start(&status, later), None);
        assert_eq!(
            previous_measure_start(&status, t0 - Duration::from_secs(1)),
            None
        );
    }

    /// Collects the known length of each mark in `waveform`, by mark.
    fn collect_mark_lengths(
        generator: &generator::Generator,
        waveform: &waveform::Waveform<MarkId>,
        lengths: &mut Vec<(MarkId, Option<usize>)>,
    ) {
        use waveform::Waveform::*;
        match waveform {
            Marked { id, waveform } => {
                lengths.push((id.clone(), generator.known_length(waveform)));
                collect_mark_lengths(generator, waveform, lengths);
            }
            Fin { waveform, .. } => collect_mark_lengths(generator, waveform, lengths),
            Append(a, b, _) | BinaryOp(_, a, b) => {
                collect_mark_lengths(generator, a, lengths);
                collect_mark_lengths(generator, b, lengths);
            }
            _ => {}
        }
    }

    // The measure has a known length, and each beat keeps the mark the
    // metronome switch substitutes, with or without precompute.
    #[test]
    fn beats_last_a_measure_and_keep_their_metronome_marks() {
        let environment = Environment::new(8000, 90, std::path::PathBuf::new());
        let (player, _receiver) = test_player();
        let mut generator = generator::Generator::new(8000);
        let beats = player.beats_waveform(&environment);
        let precomputed = generator.precompute(beats.clone());
        for waveform in [beats, precomputed] {
            // A beat at 90 bpm is 2/3 s: 5334 samples at 8000 Hz, rounded up.
            assert_eq!(
                generator.known_length(&waveform),
                Some(4 * 5334),
                "for beats {}",
                waveform
            );
            let mut lengths = Vec::new();
            collect_mark_lengths(&generator, &waveform, &mut lengths);
            assert_eq!(
                lengths,
                vec![(MarkId::MetronomeAmplitude, Some(usize::MAX)); 4],
                "for beats {}",
                waveform
            );
        }
    }

    // The click is evaluated with only the prelude in scope (no std), so it
    // must not depend on library definitions.
    #[test]
    fn click_source_evaluates_with_the_prelude_alone() {
        let environment = Environment::new(44100, 90, std::path::PathBuf::new());
        let bindings: Vec<expr::SourceBinding<MarkId, Source>> =
            vec![expr::Binding::Open(vec!["__prelude".to_string()]).into()];
        for accent in [false, true] {
            let result = environment.evaluate_source(&click_source(accent), &bindings);
            assert!(result.is_ok(), "{:?}", result.err());
        }
    }

    /// Builds a program set whose program 0 is a sequence with steps on beats
    /// [1, 2.5], evaluated so its sequence is available.
    fn sequenced_set() -> ProgramSet {
        sequenced_set_of("level_db=0", "[1, 2.5]")
    }

    /// Builds a program set whose program 0 is a sequence with steps on
    /// `beats` and the given annotation, evaluated so its sequence is
    /// available.
    fn sequenced_set_of(annotation: &str, beats: &str) -> ProgramSet {
        let source = format!(
            "on_beats = fn(w) => fn(bs) => w;\n\
             #{{{}}}\n\
             _ = {} | on_beats(1 | fin(time - 1));\n",
            annotation, beats
        );
        let set = evaluated_set(source);
        assert!(set.program(0).unwrap().sequence().is_some());
        set
    }

    /// Builds a program set from `source` with program 0 evaluated.
    fn evaluated_set(source: String) -> ProgramSet {
        let (mut set, message) = ProgramSet::from_source(source, std::path::PathBuf::new())
            .expect("test source should parse");
        assert!(message.is_empty(), "{}", message);
        let environment = Environment::new(8000, 90, std::path::PathBuf::new());
        let input = set.evaluation_input(0);
        let evaluated = environment
            .evaluate_program(&input)
            .expect("test program should evaluate");
        set.record(0, input.text.into(), Ok((Some(evaluated), Vec::new())))
            .expect("a success records");
        set
    }

    /// A receiver for the commands a test player sends to the tracker.
    type CommandReceiver = mpsc::Receiver<tracker::Command<WaveformId, MarkId>>;

    fn test_player() -> (Player, CommandReceiver) {
        let (sender, receiver) = mpsc::channel();
        let player = Player::new(90, 4, sender);
        (player, receiver)
    }

    fn anchor_start(program: usize, start: Instant) -> (WaveformId, Instant) {
        (
            WaveformId::Step {
                program,
                sixteenth: ANCHOR_SIXTEENTH,
            },
            start,
        )
    }

    #[test]
    fn play_program_sends_one_play_per_step_of_a_sequence_plus_anchor() {
        let set = sequenced_set();
        let (player, receiver) = test_player();

        let mut waveform_marks = HashMap::new();
        let message = player.play_program(
            &set,
            0,
            &empty_status(),
            Start::Now,
            Some(4),
            &mut waveform_marks,
        );
        assert!(message.is_some());

        let commands: Vec<_> = receiver.try_iter().collect();
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
        let (step_on_1, step_on_2_5, anchor) = (&plays[0], &plays[1], &plays[2]);
        assert_eq!(
            *step_on_1.0,
            WaveformId::Step {
                program: 0,
                sixteenth: 0
            }
        );
        assert_eq!(
            *step_on_2_5.0,
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
        assert_eq!(step_on_1.2, anchor.2);
        assert_eq!(
            step_on_2_5.2 - step_on_1.2,
            duration_from_beats_f32(90, 1.5)
        );
        // All entries share the loop period.
        assert!(plays.iter().all(|p| p.3 == one_measure));
        // Each step's marks add its own literal; the anchor has none.
        let sequence = set.program(0).unwrap().sequence().unwrap();
        for sixteenth in [0, 6] {
            let marks = &waveform_marks[&WaveformId::Step {
                program: 0,
                sixteenth,
            }];
            assert_eq!(marks.program, 0);
            assert_eq!(marks.extra, sequence.step_literal_marks(sixteenth));
            assert_eq!(marks.text, *set.program(0).unwrap().evaluated_text());
        }
        assert_eq!(waveform_marks.len(), 2);
    }

    #[test]
    fn play_step_schedules_this_cycle_when_sixteenth_is_ahead() {
        let set = sequenced_set();
        let (player, receiver) = test_player();

        let now = Instant::now();
        let cycle = now - Duration::from_millis(100);
        let next_cycle = cycle + duration_from_beats(90, 4);
        let mut status = empty_status();
        status.start_times = vec![anchor_start(0, cycle), anchor_start(0, next_cycle)];

        // Sixteenth 8 is beat 3, two beats into the cycle — still ahead.
        let message = player.play_step(&set, 0, 8, &status, &mut HashMap::new());
        assert!(message.is_none());

        let commands: Vec<_> = receiver.try_iter().collect();
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
        let (player, receiver) = test_player();

        let now = Instant::now();
        let cycle = now - Duration::from_secs(1);
        let next_cycle = cycle + duration_from_beats(90, 4);
        let mut status = empty_status();
        status.start_times = vec![anchor_start(0, cycle), anchor_start(0, next_cycle)];

        // Sixteenth 0 is beat 1, which started the cycle — already passed.
        let message = player.play_step(&set, 0, 0, &status, &mut HashMap::new());
        assert!(message.is_none());

        let commands: Vec<_> = receiver.try_iter().collect();
        assert_eq!(commands.len(), 1);
        let tracker::Command::Play { start, .. } = &commands[0] else {
            panic!("expected a Play command");
        };
        assert_eq!(start.unwrap(), next_cycle);
    }

    #[test]
    fn play_step_warns_when_passed_and_not_repeating() {
        let set = sequenced_set();
        let (player, receiver) = test_player();

        let now = Instant::now();
        let mut status = empty_status();
        status.start_times = vec![anchor_start(0, now - Duration::from_secs(1))];

        let message = player.play_step(&set, 0, 0, &status, &mut HashMap::new());
        assert!(message.unwrap().contains("passed"));
        assert!(receiver.try_iter().next().is_none());
    }

    /// An anchor that is running with nothing queued behind it is a one-shot
    /// family, so the step must not outlive it — not even when the program has
    /// a loop.
    #[test]
    fn play_step_does_not_repeat_into_a_one_shot_family() {
        let set = sequenced_set_of("loop_after_beats=4", "[1, 2.5]");
        let (player, receiver) = test_player();

        let now = Instant::now();
        let cycle = now - Duration::from_millis(100);
        let mut status = empty_status();
        status.start_times = vec![anchor_start(0, cycle)];

        // Sixteenth 8 is beat 3, two beats into the cycle — still ahead, so
        // the step does sound; only its repeat is at issue.
        let message = player.play_step(&set, 0, 8, &status, &mut HashMap::new());
        assert!(message.is_none());

        let commands: Vec<_> = receiver.try_iter().collect();
        assert_eq!(commands.len(), 1);
        let tracker::Command::Play {
            start,
            repeat_every,
            ..
        } = &commands[0]
        else {
            panic!("expected a Play command");
        };
        assert_eq!(start.unwrap(), cycle + duration_from_beats_f32(90, 2.0));
        assert_eq!(
            *repeat_every, None,
            "the step outlived the one-shot family it was scheduled into"
        );
    }

    /// Before the first pass starts, a newly enabled step takes the period
    /// of the program's own loop.
    #[test]
    fn play_step_takes_the_programs_loop_before_the_first_pass() {
        let set = sequenced_set_of("loop_after_beats=3", "[1, 2.5]");
        let (player, receiver) = test_player();

        let mut status = empty_status();
        status.start_times = vec![anchor_start(0, Instant::now() + Duration::from_secs(1))];

        let message = player.play_step(&set, 0, 8, &status, &mut HashMap::new());
        assert!(message.is_none());

        let commands: Vec<_> = receiver.try_iter().collect();
        assert_eq!(commands.len(), 1);
        let tracker::Command::Play { repeat_every, .. } = &commands[0] else {
            panic!("expected a Play command");
        };
        assert_eq!(*repeat_every, Some(duration_from_beats(90, 3)));
    }

    /// Returns the period of every Play command in `receiver`.
    fn repeat_periods(receiver: &CommandReceiver) -> Vec<Option<Duration>> {
        receiver
            .try_iter()
            .map(|c| match c {
                tracker::Command::Play { repeat_every, .. } => repeat_every,
                _ => panic!("expected only Play commands"),
            })
            .collect()
    }

    /// A loop that isn't a whole number of measures isn't rounded to one, so
    /// in 4/4 a waveform with a 3-beat loop drifts across the measure start.
    #[test]
    fn play_program_loops_a_waveform_every_n_beats_off_the_measure() {
        let set = evaluated_set("#{loop_after_beats=3}\n_ = 1 | fin(time - 1);\n".to_string());
        let (player, receiver) = test_player();

        let message = player.play_program(
            &set,
            0,
            &empty_status(),
            Start::Now,
            Some(3),
            &mut HashMap::new(),
        );

        assert_eq!(message.unwrap(), "Looping waveform A:1 every 3 beats");
        assert_eq!(
            repeat_periods(&receiver),
            vec![Some(duration_from_beats(90, 3))]
        );
    }

    /// Every pass of a sequence plays all of its steps, even those past beat
    /// N, so passes overlap rather than dropping steps.
    #[test]
    fn play_program_plays_every_step_of_a_sequence_longer_than_its_loop() {
        let set = sequenced_set_of("loop_after_beats=3", "[1, 5]");
        let (player, receiver) = test_player();

        let message = player.play_program(
            &set,
            0,
            &empty_status(),
            Start::Now,
            Some(3),
            &mut HashMap::new(),
        );

        assert_eq!(message.unwrap(), "Looping sequence A:1 every 3 beats");
        // Two steps and the anchor, all repeating every 3 beats.
        assert_eq!(
            repeat_periods(&receiver),
            vec![Some(duration_from_beats(90, 3)); 3]
        );
    }
}
