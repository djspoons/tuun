use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use crate::launchkey;
use crate::parser;
use crate::play_helper;
use crate::renderer::{self, MarkId, PROGRAMS_PER_BANK, Program, WaveformId};
use crate::slider;
use crate::tracker;
use crate::waveform;

struct Keys {
    id: renderer::ProgramId,

    note_on_function: parser::Expr<MarkId>,
    note_off_function: parser::Expr<MarkId>,
}

pub struct InputHandler {
    launchkey: launchkey::Launchkey,
    repeat_after_measures: Option<u32>,
    keys: Option<Keys>,

    play_helper: play_helper::PlayHelper,
    command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    slider_sender: mpsc::Sender<renderer::SliderEvent>,
}

impl InputHandler {
    pub fn new(
        launchkey: launchkey::Launchkey,
        play_helper: play_helper::PlayHelper,
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        slider_sender: mpsc::Sender<renderer::SliderEvent>,
    ) -> InputHandler {
        InputHandler {
            launchkey,
            repeat_after_measures: None,
            keys: None,
            play_helper,
            command_sender,
            slider_sender,
        }
    }

    pub fn events(&self) -> &Receiver<launchkey::Event> {
        &self.launchkey.events
    }

    pub fn handle_event(
        &mut self,
        event: launchkey::Event,
        context: &[(String, parser::Expr<MarkId>)],
        mode: renderer::Mode,
        active_program_index: &mut usize,
        status: &tracker::Status<WaveformId, MarkId>,
        programs: &mut Vec<Program>,
    ) -> renderer::Mode {
        use launchkey::Event;
        use renderer::Mode;
        let bank_start = *active_program_index - (*active_program_index % PROGRAMS_PER_BANK);
        match (mode, event) {
            (mode, Event::NoteOn { key, velocity }) => {
                if let Some(Keys {
                    note_on_function, ..
                }) = &self.keys
                {
                    use parser::Expr;
                    let expr = Expr::Application {
                        function: Box::new(note_on_function.clone()),
                        argument: Box::new(Expr::Tuple(vec![
                            Expr::Float(key as f32),
                            Expr::Float(velocity as f32),
                        ])),
                    };
                    match parser::evaluate(&[], expr) {
                        Ok(expr) => match expr {
                            Expr::Waveform(waveform) => {
                                self.command_sender
                                    .send(tracker::Command::Play {
                                        id: WaveformId::Key(key),
                                        waveform: waveform::Waveform::Marked {
                                            id: MarkId::TopLevel,
                                            waveform: Box::new(waveform),
                                        },
                                        start: None,
                                        repeat_every: None,
                                    })
                                    .unwrap();
                            }
                            expr => {
                                return mode_with_message(
                                    mode,
                                    format!("Expected waveform for note-on, got: {}", expr),
                                );
                            }
                        },
                        Err(error) => {
                            return mode_with_message(
                                mode,
                                format!("Error evaluating note-on: {}", error.to_string()),
                            );
                        }
                    }
                }
                mode
            }
            (mode, Event::NoteOff { key }) => {
                // In the case of problems with handling this event, just default to stopping the
                // waveform.
                if let Some(Keys {
                    note_off_function, ..
                }) = &self.keys
                {
                    use parser::Expr;
                    let expr = Expr::Application {
                        function: Box::new(note_off_function.clone()),
                        argument: Box::new(Expr::Tuple(vec![Expr::Float(key as f32)])),
                    };
                    match parser::evaluate(&[], expr) {
                        Ok(expr) => match expr {
                            Expr::Waveform(waveform) => {
                                self.command_sender
                                    .send(tracker::Command::Modify {
                                        id: WaveformId::Key(key),
                                        mark_id: MarkId::TopLevel,
                                        waveform,
                                    })
                                    .unwrap();
                            }
                            expr => {
                                self.play_helper.stop_waveform(WaveformId::Key(key));
                                println!("Expected waveform for note-off, got: {}", expr);
                                return mode_with_message(
                                    mode,
                                    format!("Expected waveform for note-off, got: {}", expr),
                                );
                            }
                        },
                        Err(error) => {
                            self.play_helper.stop_waveform(WaveformId::Key(key));
                            println!("Error evaluating note-off: {}", error.to_string());
                            return mode_with_message(
                                mode,
                                format!("Error evaluating note-off: {}", error.to_string()),
                            );
                        }
                    }
                } else {
                    self.play_helper.stop_waveform(WaveformId::Key(key));
                }
                mode
            }

            (Mode::Select { .. }, Event::NextTrackDown) => {
                *active_program_index = (*active_program_index + 1) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::PreviousTrackDown) => {
                *active_program_index =
                    (*active_program_index + programs.len() - 1) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::NextTrackBankDown) => {
                *active_program_index =
                    (*active_program_index + PROGRAMS_PER_BANK) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::PreviousTrackBankDown) => {
                *active_program_index =
                    (*active_program_index + programs.len() - PROGRAMS_PER_BANK) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (mode, Event::PluginEncoderChange { index, value }) => {
                let index = index as usize;
                let program = &mut programs[*active_program_index];
                let ps = &mut program.sliders;
                let message;
                if index < ps.configs.len() {
                    let norm = &mut ps.normalized_values[index];
                    *norm = (value as f32 / 127.0).clamp(0.0, 1.0);
                    let config = &ps.configs[index];
                    let actual_value = slider::denormalize(&config.function, *norm).unwrap_or(0.0);
                    self.slider_sender
                        .send(renderer::SliderEvent::UpdateSlider {
                            id: WaveformId::Program(program.id),
                            slider: config.label.clone(),
                            value: actual_value,
                        })
                        .unwrap();
                    message = format!(
                        "{}",
                        renderer::SliderDisplay {
                            label: config.label.clone(),
                            axis: index.to_string(),
                            normalized_value: *norm,
                            actual_value,
                        }
                    );
                } else {
                    message = format!("No slider with index {}", index);
                }
                mode_with_message(mode, message)
            }

            (mode, Event::DAWTopPadDown { index }) => {
                let program = &programs[bank_start + index as usize];
                if status.has_active_mark(
                    Instant::now(),
                    WaveformId::Program(program.id),
                    MarkId::TopLevel,
                ) {
                    self.play_helper
                        .stop_waveform(WaveformId::Program(program.id));
                    mode_with_message(mode, format!("Stopped program {}", program.id))
                } else {
                    self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        false,
                        None,
                    )
                }
            }
            (mode, Event::DAWBottomPadDown { index }) => {
                let program = &programs[bank_start + index as usize];
                if status.has_pending_mark(
                    Instant::now(),
                    WaveformId::Program(program.id),
                    MarkId::TopLevel,
                ) {
                    self.command_sender
                        .send(tracker::Command::RemovePending {
                            id: WaveformId::Program(program.id),
                        })
                        .unwrap();
                    mode_with_message(
                        mode,
                        format!("Removed pending waveform for program {}", program.id),
                    )
                } else {
                    self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        true,
                        self.repeat_after_measures,
                    )
                }
            }
            (mode, Event::PadFunctionDown) => {
                match self.repeat_after_measures {
                    None => {
                        self.repeat_after_measures = Some(1);
                    }
                    Some(1) => {
                        self.repeat_after_measures = Some(2);
                    }
                    Some(_) => {
                        self.repeat_after_measures = None;
                    }
                }
                mode
            }

            (mode, Event::CaptureMIDIDown) => {
                if self.keys.is_some() {
                    self.keys = None;
                    self.launchkey.set_capture_midi_brightness(0);
                    return mode_with_message(mode, "Uninstalled keys".to_string());
                }
                // When installing a MIDI instrument, we expect the program text to evaluate to a pair
                // of functions:
                //   (note_on: (midi_note, velocity) -> waveform,
                //    note_off: (midi_note) -> waveform)
                // The first will be called whenever a NoteOn event is received, and the resulting
                // waveform will be played immediately. The second will be called when a NoteOff
                // event is received, and the resulting waveform will immediately replace the original
                // waveform. (Note that the note-off waveform may include the "Prior" waveform.)
                let program = &programs[*active_program_index];
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
                                match expr {
                                    Expr::Tuple(mut exprs) => {
                                        if exprs.len() != 2 {
                                            let message = format!(
                                                "Expected two elements for keys program, got: {}",
                                                &Expr::Tuple(exprs),
                                            );
                                            return mode_with_message(mode, message);
                                        }
                                        let note_on_function = exprs.remove(0);
                                        if !test_note_function(&note_on_function, true) {
                                            return mode_with_message(
                                                mode,
                                                "Note-on functional failed test".to_string(),
                                            );
                                        }
                                        let note_off_function = exprs.remove(0);
                                        if !test_note_function(&note_off_function, false) {
                                            return mode_with_message(
                                                mode,
                                                "Note-off functional failed test".to_string(),
                                            );
                                        }
                                        self.keys = Some(Keys {
                                            id: program.id,
                                            note_on_function,
                                            note_off_function,
                                        });
                                        self.launchkey.set_capture_midi_brightness(127);
                                        mode_with_message(
                                            mode,
                                            format!("Installed keys from program {}", program.id),
                                        )
                                    }
                                    _ => {
                                        let message = format!(
                                            "Expected tuple for keys program, got: {}",
                                            expr
                                        );
                                        mode_with_message(mode, message)
                                    }
                                }
                            }
                            Err(error) => {
                                println!("Errors while evaluating input: {:?}", error);
                                let message = format!("Error: {}", error.to_string());
                                mode_with_message(mode, message)
                            }
                        }
                    }
                    Err(errors) => {
                        // If there are errors, we stay in edit mode
                        println!("Errors while parsing input: {:?}", errors);
                        mode_with_message(mode, format!("Error: {}", errors[0].to_string()))
                    }
                }
            }

            (mode, event) => {
                println!("TODO handle mode / event: {:?} / {:?}", mode, event);
                mode
            }
        }
    }

    /// Updates the state of the MIDI device based on the sliders of the given program.
    pub fn update_slider_state(&mut self, program: &Program) {
        for (i, value) in program.sliders.normalized_values.iter().enumerate() {
            self.launchkey
                .update_encoder_state((i as u8).into(), ((127.0 * value) as u8).into());
        }
    }

    pub fn update_state(
        &mut self,
        programs: &[renderer::Program],
        status: &tracker::Status<WaveformId, MarkId>,
        _mode: &renderer::Mode,
        active_program_index: usize,
    ) {
        // TODO update slider state

        match self.repeat_after_measures {
            None => {
                self.launchkey
                    .set_pad_function_color(launchkey::Color::BrightGreen);
            }
            Some(1) => {
                self.launchkey
                    .set_pad_function_color(launchkey::Color::YellowGreen);
            }
            Some(2) => {
                self.launchkey
                    .set_pad_function_color(launchkey::Color::GoldenOrange);
            }
            i => {
                println!(
                    "midi_input::InputHandler: unexpected repeat_after_measures: {:?}",
                    i
                );
            }
        }

        let now = Instant::now();
        let (_current_beat, current_beat_start, current_beat_duration) =
            renderer::current_beat_info(now, status);
        let bank_start = active_program_index - (active_program_index % PROGRAMS_PER_BANK);
        for (i, program) in programs[bank_start..bank_start + renderer::PROGRAMS_PER_BANK]
            .iter()
            .enumerate()
        {
            // Top row is based on active waveforms
            if status.has_active_mark(now, WaveformId::Program(program.id), MarkId::TopLevel)
                || (if let Some(Keys { id, .. }) = self.keys
                    && id == program.id
                {
                    status
                        .marks
                        .iter()
                        .find(|m| {
                            if let WaveformId::Key(_) = m.waveform_id {
                                true
                            } else {
                                false
                            }
                        })
                        .is_some()
                } else {
                    false
                })
            {
                const U7_MAX: u8 = u8::MAX / 2;
                let intensity = (now
                    .duration_since(current_beat_start)
                    .div_duration_f32(current_beat_duration)
                    * U7_MAX as f32) as u8;
                self.launchkey.set_daw_top_pad_color(
                    i as u8,
                    0,
                    U7_MAX.saturating_sub(intensity),
                    0,
                );
            } else if !program.text.is_empty() {
                match program.color {
                    Some(color) => {
                        self.launchkey.set_daw_top_pad_color(
                            i as u8,
                            color.0 / 2,
                            color.1 / 2,
                            color.2 / 2,
                        );
                    }
                    None => {
                        self.launchkey.set_daw_top_pad_color(i as u8, 0, 127, 127);
                    }
                }
            } else {
                // empty
                self.launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
            }
            // Bottom row is based on pending waveforms
            if status.has_pending_mark(now, WaveformId::Program(program.id), MarkId::TopLevel) {
                self.launchkey.set_daw_bottom_pad_color(i as u8, 0, 127, 0);
            } else if let Some(Keys { id, .. }) = self.keys
                && id == program.id
            {
                self.launchkey
                    .set_daw_bottom_pad_color(i as u8, 127, 127, 0);
            } else if !program.text.is_empty() {
                match program.color {
                    Some(color) => {
                        self.launchkey.set_daw_bottom_pad_color(
                            i as u8,
                            color.0 / 2,
                            color.1 / 2,
                            color.2 / 2,
                        );
                    }
                    None => {
                        self.launchkey
                            .set_daw_bottom_pad_color(i as u8, 0, 127, 127);
                    }
                }
            } else {
                // empty
                self.launchkey.set_daw_bottom_pad_color(i as u8, 0, 0, 0);
            }
        }
    }
}

fn mode_with_message(mode: renderer::Mode, message: String) -> renderer::Mode {
    use renderer::Mode;
    match mode {
        Mode::Edit {
            cursor_position,
            errors,
            message,
        } => Mode::Edit {
            cursor_position,
            errors,
            message,
        },
        Mode::Select { .. } => Mode::Select { message },
        _ => mode,
    }
}

fn test_note_function(expr: &parser::Expr<MarkId>, is_note_on: bool) -> bool {
    true
}
