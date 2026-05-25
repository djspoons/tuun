use std::time::Instant;

use crate::actions;
use crate::renderer::{MarkId, Mode, PROGRAMS_PER_BANK, Program, WaveformId};
use crate::tracker;

/// Configuration for the SDL2 classifier — display geometry (used by mouse
/// motion handling) and whether mouse events should be observed at all.
pub struct InputHandler {
    handle_mouse_events: bool,
    display_width: u32,
    display_height: u32,
}

impl InputHandler {
    pub fn new(handle_mouse_events: bool, display_width: u32, display_height: u32) -> InputHandler {
        InputHandler {
            handle_mouse_events,
            display_width,
            display_height,
        }
    }

    /// Classify an SDL event into a list of `Action`s. Returns `None` if
    /// this classifier doesn't recognize the event (the caller may log it
    /// as unhandled). Returns `Some(vec![])` when an event is recognized
    /// but produces no actions in the current state.
    pub fn classify(
        &self,
        event: &sdl2::event::Event,
        state: &crate::actions::AppState,
        status: &tracker::Status<WaveformId, MarkId>,
    ) -> Option<Vec<actions::Action>> {
        use actions::Action;
        use sdl2::event::Event;
        let mode: &Mode = &state.mode;
        match event {
            // XXX should we use Instant::now() or buffer_start?
            Event::Quit { .. } => Some(vec![Action::RequestExit]),
            Event::KeyDown {
                scancode, keymod, ..
            } => self.classify_keydown(*scancode, *keymod, state, status),
            Event::KeyUp { scancode, .. } => self.classify_keyup(*scancode, mode),
            Event::TextInput { text, .. } => self.classify_text_input(text, state),
            Event::MouseMotion { xrel, yrel, .. } => {
                if self.handle_mouse_events && matches!(mode, Mode::MoveSliders { .. }) {
                    let dx = *xrel as f32 / self.display_width as f32;
                    let dy = -(*yrel as f32) / self.display_height as f32;
                    Some(vec![
                        Action::AdjustMouseSlider { axis: 0, delta: dx },
                        Action::AdjustMouseSlider { axis: 1, delta: dy },
                    ])
                } else {
                    // Recognized but ignored: mouse motion outside
                    // MoveSliders mode (or when mouse handling is off).
                    Some(vec![])
                }
            }
            // Recognized but ignored: window resize / focus / show events
            // and audio device hotplug don't drive any app state today.
            Event::Window { .. } | Event::AudioDeviceAdded { .. } => Some(vec![]),
            _ => None,
        }
    }

    fn classify_keydown(
        &self,
        scancode: Option<sdl2::keyboard::Scancode>,
        keymod: sdl2::keyboard::Mod,
        state: &crate::actions::AppState,
        status: &tracker::Status<WaveformId, MarkId>,
    ) -> Option<Vec<actions::Action>> {
        let mode: &Mode = &state.mode;
        let active_program_index = state.active_program_index;
        let programs: &[Program] = &state.programs;
        use actions::Action;
        use sdl2::keyboard::{Mod, Scancode};
        let ctrl = keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD);
        let gui = keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD);
        match (mode, scancode) {
            (_, Some(Scancode::C)) if ctrl => Some(vec![Action::RequestExit]),
            (Mode::Select { .. }, Some(Scancode::Up)) => Some(vec![Action::AdvanceProgram(-1)]),
            (Mode::Select { .. }, Some(Scancode::Down)) => Some(vec![Action::AdvanceProgram(1)]),
            (Mode::Select { .. }, Some(Scancode::Right)) => {
                Some(vec![Action::AdvanceProgram(PROGRAMS_PER_BANK as i32)])
            }
            (Mode::Select { .. }, Some(Scancode::Left)) => {
                Some(vec![Action::AdvanceProgram(-(PROGRAMS_PER_BANK as i32))])
            }
            (Mode::Select { .. }, Some(Scancode::LAlt) | Some(Scancode::RAlt))
                if self.handle_mouse_events =>
            {
                Some(vec![Action::EnterMoveSlidersMode])
            }
            (Mode::Select { .. }, Some(Scancode::Escape)) => {
                let program = &programs[active_program_index];
                let id = WaveformId::Program(program.id);
                let now = Instant::now();
                if gui && status.has_active_mark(now, id.clone(), MarkId::TopLevel) {
                    Some(vec![Action::StopProgram(active_program_index)])
                } else if !gui && status.has_pending_mark(now, id, MarkId::TopLevel) {
                    Some(vec![Action::RemovePendingProgram(active_program_index)])
                } else {
                    Some(vec![])
                }
            }
            (Mode::Select { .. }, Some(Scancode::Return)) => {
                if gui {
                    // Cmd+Return: play with repeat (Shift=2, otherwise 1).
                    let measures =
                        if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                            2
                        } else {
                            1
                        };
                    let program = &programs[active_program_index];
                    Some(vec![Action::PlayProgram {
                        program_index: active_program_index,
                        cursor_position: program.text.len(),
                        start_at_next_measure: true,
                        repeat_after_measures: Some(measures),
                        return_to_select_on_success: false,
                    }])
                } else {
                    // Plain Return: remove any pending then enter Edit mode.
                    let program = &programs[active_program_index];
                    let id = WaveformId::Program(program.id);
                    let now = Instant::now();
                    let mut actions = Vec::new();
                    if status.has_pending_mark(now, id, MarkId::TopLevel) {
                        actions.push(Action::RemovePendingProgram(active_program_index));
                    }
                    actions.push(Action::EnterEditMode);
                    Some(actions)
                }
            }
            (Mode::Edit { .. }, Some(Scancode::Escape)) => {
                let program = &programs[active_program_index];
                let id = WaveformId::Program(program.id);
                let now = Instant::now();
                if gui && status.has_active_mark(now, id, MarkId::TopLevel) {
                    // Cmd+Escape stops the active waveform but stays in Edit
                    // mode.
                    Some(vec![Action::StopProgram(active_program_index)])
                } else {
                    // Otherwise, return to Select mode.
                    Some(vec![Action::EnterSelectMode])
                }
            }
            (
                Mode::Edit {
                    cursor_position, ..
                },
                Some(Scancode::Return),
            ) => {
                // Cmd+Return: play with repeat (Shift=2, otherwise 1).
                // Plain Return: play once, no repeat.
                // Either way, ask the runner to drop us into Select on
                // success (or stay in Edit with the error on parse
                // failure) — matches the pre-refactor behavior.
                let repeat = if gui {
                    if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                        Some(2)
                    } else {
                        Some(1)
                    }
                } else {
                    None
                };
                Some(vec![Action::PlayProgram {
                    program_index: active_program_index,
                    cursor_position: *cursor_position,
                    start_at_next_measure: true,
                    repeat_after_measures: repeat,
                    return_to_select_on_success: true,
                }])
            }
            (Mode::Edit { .. }, Some(Scancode::A)) if ctrl => Some(vec![Action::MoveCursorToStart]),
            (Mode::Edit { .. }, Some(Scancode::E)) if ctrl => Some(vec![Action::MoveCursorToEnd]),
            (Mode::Edit { .. }, Some(Scancode::Left))
                if keymod.contains(Mod::LALTMOD) || keymod.contains(Mod::RALTMOD) =>
            {
                Some(vec![Action::MoveCursorToPreviousWord])
            }
            (Mode::Edit { .. }, Some(Scancode::Left)) => Some(vec![Action::MoveCursorBy(-1)]),
            (Mode::Edit { .. }, Some(Scancode::Right)) => Some(vec![Action::MoveCursorBy(1)]),
            (Mode::Edit { .. }, Some(Scancode::Backspace))
                if keymod.contains(Mod::LALTMOD) || keymod.contains(Mod::RALTMOD) =>
            {
                Some(vec![Action::DeleteWordBeforeCursor])
            }
            (Mode::Edit { .. }, Some(Scancode::Backspace)) => {
                Some(vec![Action::DeleteCharBeforeCursor])
            }
            // Recognized keyboard event with no binding in the current
            // mode (e.g. a bare modifier, an F-key, a letter we haven't
            // wired up yet). Returning Some(vec![]) keeps these out of
            // the "Unhandled SDL event" log without enumerating every
            // possible scancode.
            _ => Some(vec![]),
        }
    }

    fn classify_keyup(
        &self,
        scancode: Option<sdl2::keyboard::Scancode>,
        mode: &Mode,
    ) -> Option<Vec<actions::Action>> {
        use actions::Action;
        use sdl2::keyboard::Scancode;
        match (mode, scancode) {
            (Mode::MoveSliders { .. }, Some(Scancode::LAlt) | Some(Scancode::RAlt)) => {
                Some(vec![Action::ExitMoveSlidersMode])
            }
            // Recognized but no binding — see classify_keydown above.
            _ => Some(vec![]),
        }
    }

    fn classify_text_input(
        &self,
        text: &str,
        state: &crate::actions::AppState,
    ) -> Option<Vec<actions::Action>> {
        use actions::Action;
        match state.mode {
            Mode::Select { .. } => match text {
                "R" => Some(vec![Action::RequestLoadContext]),
                "L" => Some(vec![Action::RequestLoadPrograms]),
                "S" => Some(vec![Action::SaveProgramsToFile]),
                "D" => Some(vec![Action::DumpActiveWaveform]),
                // Digits 1..=8 select a program in the active bank.
                t if t.len() == 1 => match t.parse::<usize>() {
                    Ok(n) if n >= 1 && n <= PROGRAMS_PER_BANK => {
                        Some(vec![Action::SelectProgram(state.bank_start() + n - 1)])
                    }
                    _ => Some(vec![]),
                },
                _ => Some(vec![]),
            },
            Mode::Edit { .. } => Some(vec![Action::InsertText(text.to_string())]),
            // Text input in modes that don't accept it (MoveSliders,
            // LoadContext, LoadPrograms, Exit) — recognized, ignored.
            _ => Some(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{Action, AppState};
    use crate::renderer::ProgramSliders;
    use sdl2::keyboard::{Mod, Scancode};

    fn test_state(mode: Mode) -> AppState {
        AppState {
            programs: vec![Program {
                text: "test".to_string(),
                id: 1,
                sliders: ProgramSliders::default(),
                color: None,
                level_db: 0.0,
            }],
            active_program_index: 0,
            mode,
            keys: None,
            repeat_after_measures: None,
        }
    }

    fn empty_status() -> tracker::Status<WaveformId, MarkId> {
        tracker::Status {
            buffer_start: Instant::now(),
            marks: vec![],
            buffer: None,
            tracker_load: None,
            allocations_per_sample: None,
        }
    }

    #[test]
    fn edit_mode_return_asks_runner_to_return_to_select_on_success() {
        // The mode transition out of Edit on success (and the stay-in-Edit
        // behavior on parse error) is driven by `return_to_select_on_success`
        // on the PlayProgram effect — the runner sets state.mode from
        // play_waveform's return. The classifier's job is just to set the
        // flag.
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
            message: String::new(),
        });
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::Return), Mod::NOMOD, &state, &status)
            .expect("Return in Edit mode should produce actions");
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(
                actions[0],
                Action::PlayProgram {
                    program_index: 0,
                    cursor_position: 4,
                    start_at_next_measure: true,
                    repeat_after_measures: None,
                    return_to_select_on_success: true,
                }
            ),
            "expected PlayProgram with return_to_select_on_success=true, got {:?}",
            actions[0]
        );
    }
}
