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
                println!("Got DAW top pad: {}, program_id = {}", index, program.id);
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
}
