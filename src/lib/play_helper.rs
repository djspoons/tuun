use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::optimizer;
use crate::parser;
use crate::renderer::{MarkId, Mode, Program, WaveformId, WaveformOrMode};
use crate::slider;
use crate::tracker;
use crate::waveform;

pub struct PlayHelper {
    tempo: u32,
    beats_per_measure: u32,

    command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
}

impl PlayHelper {
    pub fn new(
        tempo: u32,
        beats_per_measure: u32,
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    ) -> PlayHelper {
        PlayHelper {
            tempo,
            beats_per_measure,
            command_sender,
        }
    }

    pub fn play_waveform(
        &mut self,
        context: &[(String, parser::Expr<MarkId>)],
        cursor_position: usize,
        program: &Program,
        status: &tracker::Status<WaveformId, MarkId>,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    ) -> Mode {
        match prepare_waveform(context, cursor_position, program) {
            WaveformOrMode::Waveform(waveform) => {
                let message;
                let repeat_every;
                if let Some(measures) = repeat_after_measures {
                    let beats = (measures * self.beats_per_measure) as u64;
                    message = format!("Looping waveform {} every {:?} beats", program.id, beats);
                    repeat_every = Some(duration_from_beats(self.tempo, beats));
                } else {
                    // Otherwise, play it once
                    message = format!("Playing waveform {}", program.id);
                    repeat_every = None;
                }
                let start = if start_at_next_measure {
                    Some(next_measure_start(&status))
                } else {
                    None
                };
                self.command_sender
                    .send(tracker::Command::Play {
                        // TODO maybe extend the mark to the full measure?
                        id: WaveformId::Program(program.id),
                        waveform: waveform::Waveform::Marked {
                            id: MarkId::TopLevel,
                            waveform: Box::new(waveform),
                        },
                        start,
                        repeat_every,
                    })
                    .unwrap();
                Mode::Select { message }
            }
            WaveformOrMode::Mode(mode) => mode,
        }
    }

    pub fn stop_waveform(&mut self, id: WaveformId) {
        use waveform::{Operator, Waveform::*};
        const STOP_DURATION_SECS: f32 = 0.05;
        self.command_sender
            .send(tracker::Command::Modify {
                id,
                mark_id: MarkId::TopLevel,
                waveform: Fin {
                    length: Box::new(BinaryPointOp(
                        Operator::Subtract,
                        Box::new(Time(())),
                        Box::new(Const(STOP_DURATION_SECS)),
                    )),
                    waveform: Box::new(BinaryPointOp(
                        Operator::Multiply,
                        Box::new(BinaryPointOp(
                            Operator::Subtract,
                            Box::new(Const(1.0)),
                            Box::new(BinaryPointOp(
                                Operator::Multiply,
                                Box::new(Time(())),
                                Box::new(Const(1.0 / STOP_DURATION_SECS)),
                            )),
                        )),
                        Box::new(Prior),
                    )),
                },
            })
            .unwrap();
    }

    pub fn start_beats(
        &self,
        status_receiver: &mpsc::Receiver<tracker::Status<WaveformId, MarkId>>,
        context: &Vec<(String, parser::Expr<MarkId>)>,
    ) {
        // Play the odd Beats waveform starting immediately and repeating every two measures
        self.command_sender
            .send(tracker::Command::Play {
                id: WaveformId::Beats(false),
                waveform: beats_waveform(self.tempo, self.beats_per_measure, context),
                start: None,
                repeat_every: Some(
                    duration_from_beats(self.tempo, self.beats_per_measure as u64) * 2,
                ),
            })
            .unwrap();
        // We need to wait to start the even Beats until we know when the odd Beats started
        'start_even_beats: loop {
            match status_receiver.recv() {
                Ok(status) => {
                    for mark in status.marks {
                        if mark.waveform_id == WaveformId::Beats(false)
                            && mark.mark_id == MarkId::TopLevel
                        {
                            self.command_sender
                                .send(tracker::Command::Play {
                                    id: WaveformId::Beats(true),
                                    waveform: beats_waveform(
                                        self.tempo,
                                        self.beats_per_measure,
                                        context,
                                    ),
                                    start: Some(mark.start + mark.duration),
                                    repeat_every: Some(
                                        duration_from_beats(
                                            self.tempo,
                                            self.beats_per_measure as u64,
                                        ) * 2,
                                    ),
                                })
                                .unwrap();
                            break 'start_even_beats;
                        }
                    }
                }
                Err(_) => {}
            }
        }
    }
}

/// Prepares waveform for playing by parsing and evaluating the given program.
///
/// Returns a waveform if the program parses and evaluates successfully; otherwise
/// returns a mode that contains any errors.
pub fn prepare_waveform(
    context: &[(String, parser::Expr<MarkId>)],
    cursor_position: usize,
    program: &Program,
) -> WaveformOrMode {
    match parser::parse_program(&program.text) {
        Ok(expr) => {
            println!("Parser returned: {}", &expr);
            let expr = slider::prepend_slider_bindings(
                &program.sliders.configs,
                &program.sliders.normalized_values,
                MarkId::Slider,
                expr,
            );
            match parser::evaluate(context, expr) {
                Ok(expr) => {
                    println!("parser::evaluate returned: {}", &expr);
                    use parser::Expr;
                    let mut waveform = match expr {
                        Expr::Waveform(waveform) => waveform,
                        Expr::Seq { waveform, .. } => match *waveform {
                            Expr::Waveform(waveform) => waveform,
                            _ => panic!("Got non-Waveform in seq after evaluate"),
                        },
                        _ => {
                            println!("Expression is not a waveform, cannot play: {:#?}", expr);
                            return WaveformOrMode::Mode(Mode::Edit {
                                cursor_position,
                                errors: vec![parser::Error::new(
                                    "Expression is not a waveform".to_string(),
                                )],
                                message: format!("Not a waveform: {}", expr),
                            });
                        }
                    };
                    waveform = optimizer::optimize(waveform);
                    println!("optimizer::optimize returned: {}", &waveform);
                    return WaveformOrMode::Waveform(waveform);
                }
                Err(error) => {
                    // If there are errors, we stay in edit mode
                    println!("Errors while evaluating input: {:?}", error);
                    let message = format!("Error: {}", error.to_string());
                    return WaveformOrMode::Mode(Mode::Edit {
                        cursor_position,
                        errors: vec![error],
                        message: message,
                    });
                }
            }
        }
        Err(errors) => {
            // If there are errors, we stay in edit mode
            println!("Errors while parsing input: {:?}", errors);
            let message = format!("Error: {}", errors[0].to_string());
            return WaveformOrMode::Mode(Mode::Edit {
                cursor_position,
                errors,
                message,
            });
        }
    }
}

// Returns the start time of the next measure
fn next_measure_start(status: &tracker::Status<WaveformId, MarkId>) -> Instant {
    for mark in &status.marks {
        match mark.waveform_id {
            WaveformId::Beats(_)
                if mark.mark_id == MarkId::TopLevel && mark.start > Instant::now() =>
            {
                return mark.start;
            }
            _ => (),
        }
    }
    panic!("No next measure found in marks");
}

fn duration_from_beats(tempo: u32, beats: u64) -> Duration {
    Duration::from_secs_f32(beats as f32 * 60.0 / tempo as f32)
}

pub fn beats_waveform(
    tempo: u32,
    beats_per_measure: u32,
    context: &Vec<(String, parser::Expr<MarkId>)>,
) -> waveform::Waveform<MarkId> {
    let seconds_per_beat = duration_from_beats(tempo, 1);
    let mut ws = Vec::new();
    for i in 0..beats_per_measure {
        ws.push(format!(
            "0 | fin(time - {}) | seq(time - {}) | mark({})",
            seconds_per_beat.as_secs_f32(),
            seconds_per_beat.as_secs_f32(),
            i + 1
        ));
    }
    let expr = match parser::parse_program(&format!("<[{}]>", ws.join(", "))) {
        Ok(expr) => expr,
        Err(errors) => panic!("Error parsing beats waveform: {:?}", errors),
    };
    match parser::evaluate(&context, expr) {
        Ok(parser::Expr::Seq { waveform, .. }) => match *waveform {
            parser::Expr::Waveform(waveform) => waveform::Waveform::<MarkId>::Marked {
                id: MarkId::TopLevel,
                waveform: Box::new(optimizer::optimize(waveform)),
            },
            expr => panic!("Error creating beats waveform with seq, got {}", expr),
        },
        Ok(expr) => panic!("Error creating beats waveform, got {}", expr),
        Err(errors) => panic!("Error evaluating beats waveform: {:?}", errors),
    }
}
