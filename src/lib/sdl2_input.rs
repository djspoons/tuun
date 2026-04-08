use std::fs;
use std::sync::mpsc;
use std::time::Instant;

use crate::parser;
use crate::play_helper;
use crate::renderer::{
    self, MarkId, Mode, PROGRAMS_PER_BANK, Program, SliderEvent, WaveformId, WaveformOrMode,
};
use crate::slider;
use crate::tracker;

pub struct InputHandler {
    handle_mouse_events: bool,
    display_width: u32,
    display_height: u32,

    play_helper: play_helper::PlayHelper,
    command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
    slider_sender: mpsc::Sender<SliderEvent>,
}

impl<'a> InputHandler {
    pub fn new(
        handle_mouse_events: bool,
        display_width: u32,
        display_height: u32,
        play_helper: play_helper::PlayHelper,
        command_sender: mpsc::Sender<tracker::Command<WaveformId, MarkId>>,
        slider_sender: mpsc::Sender<SliderEvent>,
    ) -> InputHandler {
        InputHandler {
            handle_mouse_events,
            display_width,
            display_height,
            play_helper,
            command_sender,
            slider_sender,
        }
    }

    pub fn handle_event(
        &mut self,
        event: sdl2::event::Event,
        context: &Vec<(String, parser::Expr<MarkId>)>,
        mode: renderer::Mode,
        active_program_index: &mut usize,
        status: &tracker::Status<WaveformId, MarkId>,
        programs: &mut Vec<Program>,
    ) -> renderer::Mode {
        use sdl2::event::Event;
        use sdl2::keyboard::Mod;
        use sdl2::keyboard::Scancode;
        match event {
            // XXX should we use Instant::now() or buffer_start?
            Event::Quit { .. } => return Mode::Exit,
            Event::KeyDown {
                scancode, keymod, ..
            } => {
                match (mode, scancode) {
                    // Exit on control-C
                    (mode, Some(Scancode::C)) => {
                        if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) {
                            return Mode::Exit;
                        } else {
                            return mode;
                        }
                    }
                    (Mode::Select { .. }, Some(Scancode::Return)) => {
                        if keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD) {
                            let repeat_after_measures = if keymod.contains(Mod::LSHIFTMOD)
                                || keymod.contains(Mod::RSHIFTMOD)
                            {
                                2
                            } else {
                                1
                            };
                            self.play_helper.play_waveform(
                                context,
                                programs[*active_program_index].text.len(),
                                &programs[*active_program_index],
                                status,
                                true,
                                Some(repeat_after_measures),
                            );
                        }
                        // Check to see whether or not the current index is in the tracker's
                        // pending waveforms
                        if status.has_pending_mark(
                            Instant::now(),
                            WaveformId::Program(programs[*active_program_index].id),
                            MarkId::TopLevel,
                        ) {
                            // If it is, send a command to remove it.
                            self.command_sender
                                .send(tracker::Command::RemovePending {
                                    id: WaveformId::Program(programs[*active_program_index].id),
                                })
                                .unwrap();
                        }
                        let mut mode = edit_mode_from_program(
                            programs[*active_program_index].text.len(),
                            &programs[*active_program_index].text,
                        );
                        mode = match mode {
                            Mode::Edit {
                                cursor_position,
                                message,
                                errors,
                            } => {
                                if !errors.is_empty() {
                                    Mode::Edit {
                                        cursor_position,
                                        message: format!("Error: {}", errors[0].to_string()),
                                        errors,
                                    }
                                } else if !programs[*active_program_index]
                                    .sliders
                                    .configs
                                    .is_empty()
                                {
                                    let ps = &programs[*active_program_index].sliders;
                                    Mode::Edit {
                                        cursor_position,
                                        message: format!(
                                            "{}",
                                            ps.slider_display()
                                                .iter()
                                                .map(|s| format!("{}", s))
                                                .collect::<Vec<_>>()
                                                .join(", ")
                                        ),
                                        errors,
                                    }
                                } else {
                                    Mode::Edit {
                                        cursor_position,
                                        errors,
                                        message,
                                    }
                                }
                            }
                            _ => mode,
                        };
                        mode
                    }
                    (Mode::Select { .. }, Some(Scancode::Escape)) => {
                        let mut message = String::new();

                        if keymod.contains(Mod::LGUIMOD)
                            || keymod.contains(Mod::RGUIMOD)
                                && status.has_active_mark(
                                    Instant::now(),
                                    WaveformId::Program(programs[*active_program_index].id),
                                    MarkId::TopLevel,
                                )
                        {
                            // If the program is active, stop it
                            self.command_sender
                                .send(tracker::Command::Stop {
                                    id: WaveformId::Program(programs[*active_program_index].id),
                                })
                                .unwrap();
                            message =
                                format!("Stopped program {}", programs[*active_program_index].id);
                        } else if !keymod.contains(Mod::LGUIMOD)
                            && !keymod.contains(Mod::RGUIMOD)
                            && status.has_pending_mark(
                                Instant::now(),
                                WaveformId::Program(programs[*active_program_index].id),
                                MarkId::TopLevel,
                            )
                        {
                            // If it is, send a command to remove it.
                            self.command_sender
                                .send(tracker::Command::RemovePending {
                                    id: WaveformId::Program(programs[*active_program_index].id),
                                })
                                .unwrap();
                            message = format!(
                                "Removed pending waveform for program {}",
                                programs[*active_program_index].id
                            );
                        }
                        Mode::Select { message }
                    }
                    (Mode::Select { .. }, Some(Scancode::Up)) => {
                        *active_program_index =
                            (*active_program_index + programs.len() - 1) % programs.len();
                        Mode::Select {
                            message: String::new(),
                        }
                    }
                    (Mode::Select { .. }, Some(Scancode::Down)) => {
                        *active_program_index = (*active_program_index + 1) % programs.len();
                        Mode::Select {
                            message: String::new(),
                        }
                    }
                    (Mode::Select { .. }, Some(Scancode::LAlt) | Some(Scancode::RAlt))
                        if self.handle_mouse_events =>
                    {
                        Mode::MoveSliders {}
                    }
                    (
                        Mode::Edit {
                            cursor_position, ..
                        },
                        Some(Scancode::Return),
                    ) => {
                        // If the alt key is down, play the waveform in a loop
                        let repeat_after_measures = if keymod.contains(Mod::LGUIMOD)
                            || keymod.contains(Mod::RGUIMOD)
                        {
                            if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                                Some(2)
                            } else {
                                Some(1)
                            }
                        } else {
                            None
                        };
                        self.play_helper.play_waveform(
                            context,
                            cursor_position,
                            &programs[*active_program_index],
                            status,
                            true,
                            repeat_after_measures,
                        )
                    }
                    (
                        Mode::Edit {
                            cursor_position, ..
                        },
                        Some(Scancode::Backspace),
                    ) => {
                        // If the option key is down, clear the last word
                        let mut new_cursor_position = cursor_position;
                        let mut text = programs[*active_program_index].text.clone();
                        if keymod.contains(Mod::LALTMOD) {
                            if let Some(char_index) =
                                text[..cursor_position].rfind(|e: char| !e.is_whitespace())
                            {
                                if let Some(space_index) =
                                    text[..char_index].rfind(char::is_whitespace)
                                {
                                    // Remove everything between that whitespace and the cursor
                                    let mut new_text = text[..=space_index].to_string();
                                    new_text.push_str(&text[cursor_position..]);
                                    text = new_text;
                                    new_cursor_position = space_index + 1;
                                } else {
                                    text = text[cursor_position..].to_string();
                                    new_cursor_position = 0;
                                }
                            } else {
                                // No non-whitespace characters, so clear everything before the cursor
                                text = text[cursor_position..].to_string();
                                new_cursor_position = 0;
                            }
                        } else {
                            if !text.is_empty() && cursor_position > 0 {
                                text.remove(cursor_position - 1);
                                new_cursor_position = cursor_position - 1;
                            }
                        }
                        programs[*active_program_index].text = text;
                        let mode = edit_mode_from_program(
                            new_cursor_position,
                            &programs[*active_program_index].text,
                        );
                        mode
                    }
                    (
                        Mode::Edit {
                            cursor_position,
                            errors,
                            ..
                        },
                        Some(Scancode::Escape),
                    ) => {
                        if keymod.contains(Mod::LGUIMOD)
                            || keymod.contains(Mod::RGUIMOD)
                                && status.has_active_mark(
                                    Instant::now(),
                                    WaveformId::Program(programs[*active_program_index].id),
                                    MarkId::TopLevel,
                                )
                        {
                            // If the program is active, stop it
                            self.command_sender
                                .send(tracker::Command::Stop {
                                    id: WaveformId::Program(programs[*active_program_index].id),
                                })
                                .unwrap();
                            let message =
                                format!("Stopped program {}", programs[*active_program_index].id);
                            return Mode::Edit {
                                cursor_position,
                                errors,
                                message,
                            };
                        }
                        // Otherwise, return to select mode
                        Mode::Select {
                            message: String::new(),
                        }
                    }
                    (
                        Mode::Edit {
                            cursor_position,
                            errors,
                            message,
                        },
                        Some(Scancode::Left),
                    ) => {
                        let new_cursor_position;
                        if keymod.contains(Mod::LALTMOD) {
                            let text = &programs[*active_program_index].text;
                            if let Some(char_index) =
                                text[..cursor_position].rfind(|e: char| !e.is_whitespace())
                            {
                                if let Some(space_index) =
                                    text[..char_index].rfind(char::is_whitespace)
                                {
                                    new_cursor_position = space_index + 1;
                                } else {
                                    new_cursor_position = 0;
                                }
                            } else {
                                new_cursor_position = 0;
                            }
                        } else {
                            new_cursor_position = cursor_position.saturating_sub(1);
                        }
                        Mode::Edit {
                            cursor_position: new_cursor_position,
                            errors,
                            message,
                        }
                    }
                    (
                        Mode::Edit {
                            cursor_position,
                            errors,
                            message,
                        },
                        Some(Scancode::Right),
                    ) => {
                        // TODO check for LALTMOD and move to next word
                        let cursor_position = programs[*active_program_index]
                            .text
                            .len()
                            .min(cursor_position + 1);
                        Mode::Edit {
                            cursor_position,
                            errors,
                            message,
                        }
                    }
                    (
                        Mode::Edit {
                            cursor_position: _,
                            errors,
                            message,
                        },
                        Some(Scancode::A),
                    ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => {
                        Mode::Edit {
                            cursor_position: 0,
                            errors,
                            message,
                        }
                    }
                    (
                        Mode::Edit {
                            cursor_position: _,
                            errors,
                            message,
                        },
                        Some(Scancode::E),
                    ) if keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD) => {
                        Mode::Edit {
                            cursor_position: programs[*active_program_index].text.len(),
                            errors,
                            message,
                        }
                    }
                    (mode, _) => mode,
                }
            }
            Event::KeyUp {
                scancode: Some(Scancode::LAlt) | Some(Scancode::RAlt),
                ..
            } => {
                // Exit move sliders mode when the left alt key is released
                match mode {
                    Mode::MoveSliders { .. } => {
                        // If we were in move sliders mode, return to select mode

                        Mode::Select {
                            message: String::new(),
                        }
                    }
                    _ => mode,
                }
            }
            Event::TextInput { text, .. } => {
                match mode {
                    Mode::Select { .. } => {
                        // If the text is a number less than the number of programs per bank, update the index
                        if let Ok(new_active_program_id) = text.parse::<renderer::ProgramId>() {
                            if new_active_program_id > 0
                                && new_active_program_id as usize <= renderer::PROGRAMS_PER_BANK
                            {
                                let bank_start = *active_program_index
                                    - (*active_program_index % PROGRAMS_PER_BANK);
                                *active_program_index = renderer::index_from_id(
                                    bank_start as i32 + new_active_program_id,
                                );
                                return Mode::Select {
                                    message: String::new(),
                                };
                            } else {
                                return Mode::Select {
                                    message: format!(
                                        "Invalid program id: {}",
                                        new_active_program_id
                                    ),
                                };
                            }
                        } else if text == "R" {
                            // Reload context
                            return Mode::LoadContext {};
                        } else if text == "L" {
                            return Mode::LoadPrograms {};
                        } else if text == "S" {
                            // Save programs
                            use std::io::Write;
                            let datetime = chrono::Local::now().format("%Y-%m-%d_%H-%M-%S");
                            let filename = format!("programs_{}.tuunp", datetime);
                            let mut file = fs::File::create(&filename).unwrap();
                            for program in programs.iter() {
                                if !program.text.is_empty() {
                                    let ps = &program.sliders;
                                    if !ps.configs.is_empty() {
                                        let slider_strings: Vec<String> = ps
                                            .configs
                                            .iter()
                                            .enumerate()
                                            .map(|(j, c)| match &c.function {
                                                parser::SliderFunction::Linear {
                                                    min, max, ..
                                                } => {
                                                    // Use the current value as the new initial value.
                                                    let actual =
                                                        min + ps.normalized_values[j] * (max - min);
                                                    format!(
                                                        "\"{}:{:.3}:{:.3}:{:.3}\"",
                                                        c.label, actual, min, max
                                                    )
                                                }
                                                parser::SliderFunction::UserDefined {
                                                    function_source,
                                                    ..
                                                } => {
                                                    format!(
                                                        "\"{}:{:.3}:{}\"",
                                                        c.label,
                                                        ps.normalized_values[j],
                                                        function_source
                                                    )
                                                }
                                            })
                                            .collect();
                                        writeln!(
                                            file,
                                            "//#{}{}{}",
                                            "{sliders=[",
                                            slider_strings.join(","),
                                            "]}"
                                        )
                                        .unwrap();
                                    }
                                    writeln!(file, "{}", program.text).unwrap();
                                }
                            }
                            return Mode::Select {
                                message: format!("Saved to {}", &filename),
                            };
                        } else if text == "D" {
                            // Dump the current waveform definition to the console
                            match play_helper::prepare_waveform(
                                &context,
                                programs[*active_program_index].text.len(),
                                &programs[*active_program_index],
                            ) {
                                WaveformOrMode::Waveform(waveform) => {
                                    println!(
                                        "Waveform definition for program {}:",
                                        programs[*active_program_index].id
                                    );
                                    println!("{:#?}", waveform);
                                }
                                _ => (),
                            }
                            return Mode::Select {
                                message: format!("Dumped waveform to console"),
                            };
                        } else {
                            return Mode::Select {
                                message: format!("Invalid command: {}", text),
                            };
                        }
                    }
                    Mode::Edit {
                        cursor_position, ..
                    } => {
                        let mut new_text =
                            programs[*active_program_index].text[..cursor_position].to_string();
                        new_text.push_str(&text);
                        new_text.push_str(&programs[*active_program_index].text[cursor_position..]);
                        programs[*active_program_index].text = new_text;
                        return edit_mode_from_program(
                            cursor_position + text.len(),
                            &programs[*active_program_index].text,
                        );
                    }
                    Mode::MoveSliders { .. }
                    | Mode::LoadContext { .. }
                    | Mode::LoadPrograms { .. }
                    | Mode::Exit => {
                        return mode;
                    }
                }
            }
            Event::MouseMotion { xrel, yrel, .. } if self.handle_mouse_events => match mode {
                Mode::MoveSliders { .. } => {
                    let program = &mut programs[*active_program_index];
                    let ps = &mut program.sliders;
                    // First slider maps to mouse X axis
                    if xrel != 0 && !ps.configs.is_empty() {
                        let norm = &mut ps.normalized_values[0];
                        *norm = (*norm + xrel as f32 / self.display_width as f32).clamp(0.0, 1.0);
                        let config = &ps.configs[0];
                        let actual_value =
                            slider::denormalize(&config.function, *norm).unwrap_or(0.0);
                        self.slider_sender
                            .send(SliderEvent::UpdateSlider {
                                id: WaveformId::Program(program.id),
                                slider: config.label.clone(),
                                value: actual_value,
                            })
                            .unwrap();
                    }
                    // Second slider maps to mouse Y axis
                    if yrel != 0 && ps.configs.len() >= 2 {
                        let norm = &mut ps.normalized_values[1];
                        *norm = (*norm - yrel as f32 / self.display_height as f32).clamp(0.0, 1.0);
                        let config = &ps.configs[1];
                        let actual_value =
                            slider::denormalize(&config.function, *norm).unwrap_or(0.0);
                        self.slider_sender
                            .send(SliderEvent::UpdateSlider {
                                id: WaveformId::Program(program.id),
                                slider: config.label.clone(),
                                value: actual_value,
                            })
                            .unwrap();
                    }
                    return Mode::MoveSliders {};
                }
                _ => {
                    return mode;
                }
            },
            _ => {
                return mode;
            }
        }
    }
}

fn edit_mode_from_program(cursor_position: usize, program: &str) -> Mode {
    Mode::Edit {
        cursor_position,
        errors: if program.is_empty() {
            Vec::new()
        } else {
            match parser::parse_program::<MarkId>(program) {
                Ok(_) => Vec::new(),
                Err(errors) => errors,
            }
        },
        // TODO could show the current sliders here
        message: String::new(),
    }
}
