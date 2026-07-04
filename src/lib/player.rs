//! Sends waveforms to the tracker — all tracker I/O lives here.
//!
//! `Player` owns the two command routes: `precompute_sender` goes through
//! the precompute thread (used for playback scheduled at the next measure,
//! where latency is hidden), and `fast_sender` goes straight to the
//! tracker (used for immediate playback and note-on/off, where keystroke
//! latency matters). All methods take `&self`.

use std::sync::mpsc;
use std::time;

use crate::evaluator::Evaluator;
use crate::ids::{MarkId, WaveformId};
use crate::optimizer;
use crate::parser;
use crate::programs::{ProgramSet, ProgramSliders};
use crate::slider;
use crate::tracker;
use crate::waveform;

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
    /// when the program has no cached waveform (or the index is out of
    /// range) and nothing was played.
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
        let waveform = program.waveform().cloned()?;
        let mut waveform = optimizer::optimize(waveform);
        println!("optimizer::optimize returned: {}", &waveform);
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

    /// Fades out the waveform with the given id over a short ramp. A no-op
    /// if no matching waveform is playing.
    pub fn stop_waveform(&self, id: WaveformId) {
        use waveform::{Operator, Waveform::*};
        const STOP_DURATION_SECS: f32 = 0.05;
        self.fast_sender
            .send(tracker::Command::Modify {
                id,
                mark_id: MarkId::Terminator,
                waveform: Fin {
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
                },
            })
            .unwrap();
    }

    /// Removes the pending (not-yet-started) waveform with the given id.
    pub fn remove_pending(&self, id: WaveformId) {
        let _ = self
            .fast_sender
            .send(tracker::Command::RemovePending { id });
    }

    /// Replaces the waveform under `mark_id` on the waveform with the given
    /// id.
    pub fn modify(&self, id: WaveformId, mark_id: MarkId, waveform: waveform::Waveform<MarkId>) {
        let _ = self.fast_sender.send(tracker::Command::Modify {
            id,
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
        let bindings: Vec<parser::SourceBinding<MarkId>> =
            vec![parser::Binding::Open(vec!["__prelude".to_string()]).into()];
        match evaluator
            .evaluate_source(&source, &bindings)
            .map(|s| s.expr)
        {
            Ok(parser::Expr::Seq { waveform, .. }) => match waveform.expr {
                parser::Expr::Waveform(waveform) => waveform::Waveform::<MarkId>::Marked {
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

/// Wraps `waveform` with the standard top-level marks: an `Amplitude` mark
/// at the amplitude for `level_db` and a `Terminator` mark used to stop it.
fn build_top_level_waveform(
    waveform: waveform::Waveform<MarkId>,
    level_db: f32,
) -> waveform::Waveform<MarkId> {
    use waveform::Waveform::{BinaryPointOp, Const, Marked};
    Marked {
        id: MarkId::TopLevel,
        waveform: Box::new(BinaryPointOp(
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
        )),
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
