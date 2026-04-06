use std::sync::mpsc;
use std::sync::mpsc::Receiver;

use crate::launchkey;
use crate::parser;
use crate::renderer::{self, MarkId, Program, WaveformId};
use crate::tracker;

pub struct InputHandler {
    launchkey: launchkey::Launchkey,

    slider_sender: mpsc::Sender<renderer::SliderEvent>,
}

impl InputHandler {
    pub fn new(
        launchkey: launchkey::Launchkey,
        slider_sender: mpsc::Sender<renderer::SliderEvent>,
    ) -> InputHandler {
        InputHandler {
            launchkey,
            slider_sender,
        }
    }

    pub fn events(&self) -> &Receiver<launchkey::Event> {
        &self.launchkey.events
    }

    pub fn handle_event(
        &mut self,
        event: launchkey::Event,
        _context: &Vec<(String, parser::Expr<MarkId>)>,
        mode: renderer::Mode,
        _status: &tracker::Status<WaveformId, MarkId>,
        programs: &mut Vec<Program>,
    ) -> renderer::Mode {
        use launchkey::Event;
        use renderer::Mode;
        match (mode, event) {
            (
                Mode::Select {
                    active_program_index,
                    ..
                },
                Event::NextTrack,
            ) => {
                let active_program_index = (active_program_index + 1) % programs.len();
                self.update_slider_state(&programs[active_program_index]);
                Mode::Select {
                    active_program_index,
                    message: String::new(),
                }
            }
            (
                Mode::Select {
                    active_program_index,
                    ..
                },
                Event::PreviousTrack,
            ) => {
                let active_program_index =
                    (active_program_index + programs.len() - 1) % programs.len();
                self.update_slider_state(&programs[active_program_index]);
                Mode::Select {
                    active_program_index,
                    message: String::new(),
                }
            }
            (
                Mode::Select {
                    active_program_index,
                    ..
                },
                Event::PluginEncoderChange { index, value },
            ) => {
                let index = index as usize;
                let program = &mut programs[active_program_index];
                let ps = &mut program.sliders;
                let message;
                if index < ps.configs.len() {
                    let norm = &mut ps.normalized_values[index];
                    *norm = (value as f32 / 127.0).clamp(0.0, 1.0);
                    let config = &ps.configs[index];
                    let actual_value = config.min + *norm * (config.max - config.min);
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
                Mode::Select {
                    active_program_index,
                    message,
                }
            }

            (mode, event) => {
                println!("TODO handle mode/event: {:?}/{:?}", mode, event);
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
