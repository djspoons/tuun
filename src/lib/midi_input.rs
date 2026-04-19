use std::collections::HashMap;
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
    context: Vec<(String, parser::Expr<MarkId>)>,
    note_on_function: parser::Expr<MarkId>,
    note_off_function: parser::Expr<MarkId>,
    sliders: renderer::ProgramSliders,
}

pub struct InputHandler {
    launchkey: launchkey::Launchkey,
    repeat_after_measures: Option<u32>,
    keys: Option<Keys>,

    play_helper: play_helper::PlayHelper,
    command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    slider_sender: mpsc::Sender<renderer::SliderEvent>,
}

/// Applies a note function (either "on" or "off"), expecting a Waveform result.
fn apply_note_function_as_waveform(
    context: &[(String, parser::Expr<MarkId>)],
    expr: &parser::Expr<MarkId>,
    args: Vec<parser::Expr<MarkId>>,
    sliders: &renderer::ProgramSliders,
    label: &str,
) -> Result<waveform::Waveform<MarkId>, String> {
    let expr = parser::Expr::Application {
        function: Box::new(expr.clone()),
        argument: Box::new(parser::Expr::Tuple(args)),
    };
    let expr = slider::prepend_slider_bindings(
        &sliders.configs,
        &sliders.normalized_values,
        MarkId::Slider,
        expr,
    );
    let expr = parser::evaluate(context, expr).map_err(|e| e.to_string());
    match expr {
        Ok(parser::Expr::Waveform(w)) => Ok(w),
        Ok(expr) => Err(format!("Expected waveform for {}, got: {}", label, expr)),
        Err(e) => Err(format!("Error evaluating {}: {}", label, e)),
    }
}

impl InputHandler {
    pub fn new(
        mut launchkey: launchkey::Launchkey,
        play_helper: play_helper::PlayHelper,
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        slider_sender: mpsc::Sender<renderer::SliderEvent>,
    ) -> InputHandler {
        // There's only one supported DAW mode at the moment.
        launchkey.set_daw_mode_display(&"Clip Launcher".to_string());
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
                    id,
                    context,
                    note_on_function,
                    sliders,
                    ..
                }) = &self.keys
                {
                    let args = vec![
                        parser::Expr::Float(key as f32),
                        // TODO use a marked waveform for velocity
                        parser::Expr::Float(velocity as f32 / 127.0),
                    ];
                    match apply_note_function_as_waveform(
                        context,
                        note_on_function,
                        args,
                        sliders,
                        "note-on",
                    ) {
                        Ok(waveform) => {
                            // We need to make sure the initial values for this key are set.
                            let ps = &programs[renderer::index_from_id(*id)].sliders;
                            let mut last_slider_values = HashMap::new();
                            for (j, config) in ps.configs.iter().enumerate() {
                                let value =
                                    slider::denormalize(&config.function, ps.normalized_values[j])
                                        .unwrap_or(0.0);
                                last_slider_values
                                    .insert((WaveformId::Key(key), config.label.clone()), value);
                            }
                            self.slider_sender
                                .send(renderer::SliderEvent::UpdateInitialValues(
                                    last_slider_values,
                                ))
                                .unwrap();
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
                        Err(message) => return mode_with_message(mode, message),
                    }
                }
                mode
            }
            (mode, Event::NoteOff { key }) => {
                // In the case of problems with handling this event, just default to stopping the
                // waveform.
                if let Some(Keys {
                    context,
                    note_off_function,
                    sliders,
                    ..
                }) = &self.keys
                {
                    let args = vec![parser::Expr::Float(key as f32)];
                    match apply_note_function_as_waveform(
                        context,
                        note_off_function,
                        args,
                        sliders,
                        "note-off",
                    ) {
                        Ok(waveform) => {
                            self.command_sender
                                .send(tracker::Command::Modify {
                                    id: WaveformId::Key(key),
                                    mark_id: MarkId::TopLevel,
                                    waveform,
                                })
                                .unwrap();
                        }
                        Err(message) => {
                            self.play_helper.stop_waveform(WaveformId::Key(key));
                            return mode_with_message(mode, message);
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
                if index < ps.configs.len() {
                    let norm = &mut ps.normalized_values[index];
                    *norm = (value as f32 / 127.0).clamp(0.0, 1.0);
                    let config = &ps.configs[index];
                    let actual_value = slider::denormalize(&config.function, *norm).unwrap_or(0.0);
                    // Now send to all the relevant ids: both clips and keys.
                    let mut waveform_ids = vec![];
                    if let Some(Keys { id, sliders, .. }) = &mut self.keys
                        && *id == program.id
                    {
                        for mark in &status.marks {
                            if let WaveformId::Key(_) = mark.waveform_id {
                                waveform_ids.push(mark.waveform_id.clone());
                            }
                        }
                        // Also update the value that we copied into the installed keys.
                        sliders.normalized_values[index] = *norm;
                    } else {
                        waveform_ids.push(WaveformId::Program(program.id));
                    }
                    for id in waveform_ids {
                        self.slider_sender
                            .send(renderer::SliderEvent::UpdateSlider {
                                id,
                                slider: config.label.clone(),
                                value: actual_value,
                            })
                            .unwrap();
                    }
                    mode_with_message(
                        mode,
                        format!(
                            "{}",
                            renderer::SliderDisplay {
                                label: config.label.clone(),
                                axis: index.to_string(),
                                normalized_value: *norm,
                                actual_value,
                            }
                        ),
                    )
                } else {
                    mode_with_message(mode, format!("No slider with index {}", index))
                }
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
                } else if let Some(Keys { id, .. }) = self.keys
                    && id == program.id
                {
                    // Do nothing if it's the installed keys
                    mode
                } else {
                    match self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        false,
                        None,
                    ) {
                        Mode::Edit { message, .. } => mode_with_message(mode, message),
                        _ => mode,
                    }
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
                } else if let Some(Keys { id, .. }) = self.keys
                    && id == program.id
                {
                    // Do nothing if it's the installed keys
                    mode
                } else {
                    match self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        true,
                        self.repeat_after_measures,
                    ) {
                        Mode::Edit { message, .. } => mode_with_message(mode, message),
                        _ => mode,
                    }
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
                let expr = match parser::parse_program(&program.text) {
                    Ok(expr) => expr,
                    Err(errors) => {
                        return mode_with_message(
                            mode,
                            format!("Error: {}", errors[0].to_string()),
                        );
                    }
                };
                let mut exprs = match expr {
                    parser::Expr::Tuple(exprs) if exprs.len() == 2 => exprs,
                    other => {
                        return mode_with_message(
                            mode,
                            format!("Expected (note_on, note_off) tuple, got: {}", other),
                        );
                    }
                };
                let note_on_function = exprs.remove(0);
                let note_off_function = exprs.remove(0);
                if let Err(message) =
                    test_note_function(context, &note_on_function, &program.sliders, true)
                {
                    return mode_with_message(mode, message);
                }
                if let Err(message) =
                    test_note_function(context, &note_off_function, &program.sliders, false)
                {
                    return mode_with_message(mode, message);
                }
                self.keys = Some(Keys {
                    id: program.id,
                    context: Vec::from(context),
                    note_on_function,
                    note_off_function,
                    sliders: program.sliders.clone(),
                });
                self.launchkey.set_capture_midi_brightness(127);
                mode_with_message(mode, format!("Installed keys from program {}", program.id))
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
            let config = &program.sliders.configs[i];
            let actual_value = slider::denormalize(&config.function, *value).unwrap_or(0.0);
            self.launchkey.set_encoder_display(
                i as u8,
                &config.label,
                &format!("{:.3}", actual_value),
            );
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
            const U7_MAX: u8 = u8::MAX / 2;
            // 7-bit color values for the current program.
            let (red, blue, green) = match program.color {
                Some(color) => (color.0 / 2, color.1 / 2, color.2 / 2),
                None => (0, 127, 127),
            };
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
                // Based on time since beginning of the current beat.
                let intensity = U7_MAX.saturating_sub(
                    (now.duration_since(current_beat_start)
                        .div_duration_f32(current_beat_duration)
                        * U7_MAX as f32) as u8,
                );
                self.launchkey
                    .set_daw_top_pad_color(i as u8, 0, intensity, 0);
            } else if let Some(Keys { id, .. }) = self.keys
                && id == program.id
            {
                // If it's the installed keys program, don't color the top pad (unless it's playing).
                self.launchkey.set_daw_top_pad_color(i as u8, 0, 0, 0);
            } else if !program.text.is_empty() {
                self.launchkey
                    .set_daw_top_pad_color(i as u8, red, blue, green);
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
                // If it's the installed keys program, pulse the configured color.
                let intensity = now
                    .duration_since(current_beat_start)
                    .div_duration_f32(current_beat_duration);
                self.launchkey.set_daw_bottom_pad_color(
                    i as u8,
                    red.saturating_sub((intensity * red as f32) as u8),
                    green.saturating_sub((intensity * green as f32) as u8),
                    blue.saturating_sub((intensity * blue as f32) as u8),
                );
            } else if !program.text.is_empty() {
                self.launchkey
                    .set_daw_bottom_pad_color(i as u8, red, blue, green);
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

fn test_note_function(
    context: &[(String, parser::Expr<MarkId>)],
    expr: &parser::Expr<MarkId>,
    sliders: &renderer::ProgramSliders,
    is_note_on: bool,
) -> Result<(), String> {
    let args = if is_note_on {
        // TODO use a waveform for velocity
        vec![parser::Expr::Float(60.0), parser::Expr::Float(0.7)]
    } else {
        vec![parser::Expr::Float(60.0)]
    };
    let label = if is_note_on { "note-on" } else { "note-off" };
    apply_note_function_as_waveform(context, expr, args, sliders, label)?;
    Ok(())
}
