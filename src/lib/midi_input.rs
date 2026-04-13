use std::sync::mpsc;
use std::sync::mpsc::Receiver;
use std::time::Instant;

use crate::launchkey;
use crate::parser;
use crate::play_helper;
use crate::renderer::{self, MarkId, PROGRAMS_PER_BANK, Program, WaveformId};
use crate::slider;
use crate::tracker;

pub struct InputHandler {
    launchkey: launchkey::Launchkey,

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
            (Mode::Select { .. }, Event::NextTrack) => {
                *active_program_index = (*active_program_index + 1) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::PreviousTrack) => {
                *active_program_index =
                    (*active_program_index + programs.len() - 1) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::NextTrackBank) => {
                *active_program_index =
                    (*active_program_index + PROGRAMS_PER_BANK) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (Mode::Select { .. }, Event::PreviousTrackBank) => {
                *active_program_index =
                    (*active_program_index + programs.len() - PROGRAMS_PER_BANK) % programs.len();
                self.update_slider_state(&programs[*active_program_index]);
                Mode::Select {
                    message: String::new(),
                }
            }
            (
                Mode::Select { .. } | Mode::Edit { .. },
                Event::PluginEncoderChange { index, value },
            ) => {
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
                Mode::Select { message }
            }

            (mode, Event::DAWTopPadPressed { index }) => {
                let program = &programs[bank_start + index as usize];
                if status.has_active_mark(
                    Instant::now(),
                    WaveformId::Program(program.id),
                    MarkId::TopLevel,
                ) {
                    self.command_sender
                        .send(tracker::Command::Stop {
                            id: WaveformId::Program(program.id),
                        })
                        .unwrap();
                } else {
                    self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        false,
                        None,
                    );
                    // TODO This drops the returned mode, which might contain an error
                }
                mode
            }
            (mode, Event::DAWBottomPadPressed { index }) => {
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
                } else {
                    self.play_helper.play_waveform(
                        context,
                        program.text.len(),
                        &program,
                        status,
                        true,
                        None,
                    );
                    // TODO This drops the returned mode, which might contain an error
                }
                mode
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
        let now = Instant::now();
        let (_current_beat, current_beat_start, current_beat_duration) =
            renderer::current_beat_info(now, status);
        let bank_start = active_program_index - (active_program_index % PROGRAMS_PER_BANK);
        for (i, program) in programs[bank_start..bank_start + renderer::PROGRAMS_PER_BANK]
            .iter()
            .enumerate()
        {
            // Top row is based on active waveforms
            if status.has_active_mark(now, WaveformId::Program(program.id), MarkId::TopLevel) {
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
