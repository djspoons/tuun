use std::sync::mpsc;
use std::time;

use crate::evaluator::Evaluator;
use crate::optimizer;
use crate::parser;
use crate::programs::{Program, ProgramSet, ProgramSliders};
use crate::renderer::{MarkId, WaveformId};
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

pub struct PlayHelper {
    tempo: u32,
    beats_per_measure: u32,
    /// The evaluation environment (prelude + module cache).
    evaluator: Evaluator,
    precomputing_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    fast_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
}

impl PlayHelper {
    pub fn new(
        sample_rate: u32,
        tempo: u32,
        beats_per_measure: u32,
        library_root: std::path::PathBuf,
        precomputing_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        fast_command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    ) -> PlayHelper {
        PlayHelper {
            tempo,
            beats_per_measure,
            evaluator: Evaluator::new(sample_rate, tempo, library_root),
            precomputing_command_sender,
            fast_command_sender,
        }
    }

    /// Returns the evaluation environment.
    pub fn evaluator(&self) -> &Evaluator {
        &self.evaluator
    }

    /// Evaluates the program at `program_index` in preparation for playback
    /// or for installing as the keys instrument. Returns an expression if
    /// the program parses and evaluates successfully; otherwise returns a
    /// message.
    ///
    /// The program is evaluated in the context of the file-level bindings
    /// preceding it plus its sliders. Bindings after it are ignored.
    pub fn evaluate_program(
        &self,
        set: &ProgramSet,
        program_index: usize,
    ) -> Result<parser::SourceExpr<MarkId>, String> {
        self.evaluator.evaluate_source(
            set.programs()[program_index].text(),
            &set.evaluation_bindings(program_index),
        )
    }

    /// Plays a program as a waveform. Returns the user-visible message on
    /// success or failure. `program_index` is the 0-based position in
    /// `state.programs` and doubles as the tracker's waveform id.
    /// `display_name` is the user-facing label (see
    /// `ProgramSet::display_name`).
    pub fn play_program_as_waveform(
        &mut self,
        program_index: usize,
        program: &Program,
        display_name: &str,
        status: &tracker::Status<WaveformId, MarkId>,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    ) -> Option<String> {
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
        if let Some(waveform) = program.waveform().cloned() {
            let mut waveform = optimizer::optimize(waveform);
            println!("optimizer::optimize returned: {}", &waveform);
            // Substitute the program's current slider positions before handing
            // the waveform to the tracker (since the ones in cached_waveform
            // may be old).
            substitute_current_slider_values(&mut waveform, program.sliders());
            if start_at_next_measure {
                &mut self.precomputing_command_sender
            } else {
                &mut self.fast_command_sender
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
        } else {
            None
        }
    }

    pub fn stop_waveform(&mut self, id: WaveformId) {
        use waveform::{Operator, Waveform::*};
        const STOP_DURATION_SECS: f32 = 0.05;
        self.fast_command_sender
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

    pub fn start_beats(
        &self,
        status_receiver: &mpsc::Receiver<tracker::Status<WaveformId, MarkId>>,
    ) {
        // Play the odd Beats waveform starting immediately and repeating every two measures
        self.precomputing_command_sender
            .send(tracker::Command::Play {
                id: WaveformId::Beats(false),
                waveform: self.beats_waveform(),
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
                        self.precomputing_command_sender
                            .send(tracker::Command::Play {
                                id: WaveformId::Beats(true),
                                waveform: self.beats_waveform(),
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
    pub fn beats_waveform(&self) -> waveform::Waveform<MarkId> {
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
        match self
            .evaluator
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

pub fn build_top_level_waveform(
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
