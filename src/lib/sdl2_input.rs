use std::time::Instant;

use crate::actions;
use crate::renderer::{MarkId, Mode, PROGRAMS_PER_BANK, Program, WaveformId};
use crate::tracker;

/// Maps a QWERTY scancode to a MIDI note number for Keys (piano) mode.
/// Lower row plays white keys starting at C4 (MIDI 60); the row above
/// plays sharps where they correspond to black keys on a piano (so `f`
/// and `k` — above the E and B positions, which have no black key —
/// return None).
///
///   Lower:  z  x  c  v  b  n  m  ,  .  /
///   Notes:  C  D  E  F  G  A  B  C  D  E
///   Upper:  s  d     g  h  j     l  ;
///   Sharps: C# D#    F# G# A#    C# D#
fn scancode_to_midi_note(scancode: sdl2::keyboard::Scancode) -> Option<u8> {
    use sdl2::keyboard::Scancode::*;
    Some(match scancode {
        Z => 60,         // C4
        S => 61,         // C#4
        X => 62,         // D4
        D => 63,         // D#4
        C => 64,         // E4
        V => 65,         // F4
        G => 66,         // F#4
        B => 67,         // G4
        H => 68,         // G#4
        N => 69,         // A4
        J => 70,         // A#4
        M => 71,         // B4
        Comma => 72,     // C5
        L => 73,         // C#5
        Period => 74,    // D5
        Semicolon => 75, // D#5
        Slash => 76,     // E5
        _ => return None,
    })
}

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
            Event::Quit { .. } => Some(vec![Action::Exit]),
            Event::KeyDown {
                scancode,
                keymod,
                repeat,
                ..
            } => self.classify_keydown(*scancode, *keymod, *repeat, state, status),
            Event::KeyUp { scancode, .. } => self.classify_keyup(*scancode, mode),
            Event::TextInput { text, .. } => self.classify_text_input(text, state),
            Event::MouseMotion { xrel, yrel, .. } => {
                if self.handle_mouse_events && matches!(mode, Mode::MoveSliders) {
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
            // and audio device hot-plug don't drive any app state today.
            Event::Window { .. } | Event::AudioDeviceAdded { .. } => Some(vec![]),
            _ => None,
        }
    }

    fn classify_keydown(
        &self,
        scancode: Option<sdl2::keyboard::Scancode>,
        keymod: sdl2::keyboard::Mod,
        repeat: bool,
        state: &crate::actions::AppState,
        status: &tracker::Status<WaveformId, MarkId>,
    ) -> Option<Vec<actions::Action>> {
        let mode: &Mode = &state.mode;
        let active_program_index = state.active_program_index;
        let programs: &[Program] = &state.programs;
        use actions::Action;
        use sdl2::keyboard::{Mod, Scancode};
        let ctrl = keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD);
        let gui_mod = keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD);
        // Keys mode: piano keystrokes → NoteOn (suppress on auto-repeat
        // so a held key doesn't retrigger), Escape exits. Ctrl+C still
        // exits the app so the user can quit without first backing out
        // to Select mode.
        if let Mode::Keys = mode {
            if let Some(Scancode::C) = scancode
                && ctrl
            {
                return Some(vec![Action::Exit]);
            }
            if let Some(Scancode::Escape) = scancode {
                return Some(vec![Action::EnterSelectMode]);
            }
            if repeat {
                return Some(vec![]);
            }
            if let Some(sc) = scancode
                && let Some(midi_note) = scancode_to_midi_note(sc)
            {
                return Some(vec![Action::NoteOn {
                    key: midi_note,
                    velocity: 64, // computer keyboard has no velocity; pick mf
                }]);
            }
            return Some(vec![]);
        }
        match (mode, scancode) {
            (_, Some(Scancode::C)) if ctrl => Some(vec![Action::Exit]),
            (Mode::Select, Some(Scancode::Up)) => Some(vec![Action::AdvanceProgram(-1)]),
            (Mode::Select, Some(Scancode::Down)) => Some(vec![Action::AdvanceProgram(1)]),
            (Mode::Select, Some(Scancode::Right)) => {
                Some(vec![Action::AdvanceProgram(PROGRAMS_PER_BANK as i32)])
            }
            (Mode::Select, Some(Scancode::Left)) => {
                Some(vec![Action::AdvanceProgram(-(PROGRAMS_PER_BANK as i32))])
            }
            (Mode::Select, Some(Scancode::LAlt) | Some(Scancode::RAlt))
                if self.handle_mouse_events =>
            {
                Some(vec![Action::EnterMoveSlidersMode])
            }
            (Mode::Select, Some(Scancode::Escape)) => {
                let program = &programs[active_program_index];
                let id = WaveformId::Program(program.id);
                let now = Instant::now();
                if gui_mod && status.has_active_mark(now, id.clone(), MarkId::TopLevel) {
                    Some(vec![Action::StopProgram(active_program_index)])
                } else if !gui_mod && status.has_pending_mark(now, id, MarkId::TopLevel) {
                    Some(vec![Action::RemovePendingProgram(active_program_index)])
                } else {
                    Some(vec![])
                }
            }
            (Mode::Select, Some(Scancode::Return)) => {
                if gui_mod {
                    // Cmd+Return: play with repeat (Shift=2, otherwise 1).
                    let measures =
                        if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                            2
                        } else {
                            1
                        };
                    if programs[active_program_index].cached_waveform.is_some() {
                        Some(vec![Action::PlayProgram {
                            program_index: active_program_index,
                            start_at_next_measure: true,
                            repeat_after_measures: Some(measures),
                        }])
                    } else {
                        Some(vec![Action::ShowMessage(format!(
                            "Program {} is not a valid waveform",
                            active_program_index
                        ))])
                    }
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
                if gui_mod && status.has_active_mark(now, id, MarkId::TopLevel) {
                    // Cmd+Escape stops the active waveform but stays in Edit
                    // mode.
                    Some(vec![Action::StopProgram(active_program_index)])
                } else {
                    // Otherwise, return to Select mode.
                    Some(vec![Action::EvaluateAndLeaveEditMode {
                        mode_on_failure: Mode::Select,
                    }])
                }
            }
            (mode @ Mode::Edit { .. }, Some(Scancode::Return)) => {
                // Cmd+Return: play with repeat (Shift=2, otherwise 1).
                // Plain Return: play once, no repeat.
                let repeat = if gui_mod {
                    if keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD) {
                        Some(2)
                    } else {
                        Some(1)
                    }
                } else {
                    None
                };
                // Either way, ask the runner to drop us into Select on
                // success or stay in Edit with the error on parse
                // failure.
                Some(vec![
                    Action::EvaluateAndLeaveEditMode {
                        mode_on_failure: mode.clone(),
                    },
                    Action::PlayProgram {
                        program_index: active_program_index,
                        start_at_next_measure: true,
                        repeat_after_measures: repeat,
                    },
                ])
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
        // NoteOff fires for piano scancodes in any mode, not just Keys.
        // This avoids stuck notes if the user exits Keys mode while
        // still holding a key (Escape, click away, etc.). The runner's
        // PlayNoteOff is a no-op when the key isn't currently playing.
        if let Some(sc) = scancode
            && let Some(midi_note) = scancode_to_midi_note(sc)
        {
            return Some(vec![Action::NoteOff { key: midi_note }]);
        }
        match (mode, scancode) {
            (Mode::MoveSliders, Some(Scancode::LAlt) | Some(Scancode::RAlt)) => {
                Some(vec![Action::EnterSelectMode])
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
            Mode::Select => match text {
                "D" => Some(vec![Action::DumpActiveWaveform]),
                // Uppercase K: install the active program as the keys
                // instrument (computer-keyboard analogue of pressing the
                // active program's pad in "Install Keys" mode).
                "K" => Some(vec![Action::InstallKeys(state.active_program_index)]),
                // Lowercase k: enter Keys (piano) mode — only if keys
                // are actually installed; otherwise silently no-op.
                "k" if state.keys.is_some() => Some(vec![Action::EnterKeysMode]),
                // Digits 1..=8 select a program in the active bank.
                t if t.len() == 1 => match t.parse::<usize>() {
                    Ok(n) if (1..=PROGRAMS_PER_BANK).contains(&n) => {
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
    use sdl2::keyboard::{Mod, Scancode};

    fn test_state(mode: Mode) -> AppState {
        AppState {
            programs: vec![Program::from_string("test", 0, 1)],
            bindings: Vec::new(),
            source: String::new(),
            input_path: std::path::PathBuf::new(),
            active_program_index: 0,
            mode,
            keys: None,
            repeat_after_measures: None,
            daw_pad_mode: actions::DawPadMode::ClipLauncher,
            should_exit: false,
            message: String::new(),
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
    fn keys_mode_lower_row_emits_note_on() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::NOMOD, false, &state, &status)
            .expect("Z in Keys mode should produce actions");
        assert!(
            matches!(
                actions[0],
                Action::NoteOn {
                    key: 60,
                    velocity: 64
                }
            ),
            "expected NoteOn(C4=60), got {:?}",
            actions[0]
        );
    }

    #[test]
    fn keys_mode_suppresses_note_on_for_held_key() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let status = empty_status();
        // repeat=true means SDL is auto-repeating a held key. We must
        // not retrigger NoteOn — the note is already playing.
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::NOMOD, true, &state, &status)
            .expect("Z (repeat) in Keys mode should be classified");
        assert!(
            actions.is_empty(),
            "expected no actions on key repeat, got {:?}",
            actions
        );
    }

    #[test]
    fn keys_mode_escape_returns_to_select() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::Escape), Mod::NOMOD, false, &state, &status)
            .expect("Escape in Keys mode should produce actions");
        assert!(matches!(actions[0], Action::EnterSelectMode));
    }

    #[test]
    fn keys_mode_ctrl_c_exits() {
        // Ctrl+C is a universal quit and must work even when in Keys
        // mode (where Escape is bound to exiting the mode, not the app).
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::C), Mod::LCTRLMOD, false, &state, &status)
            .expect("Ctrl+C in Keys mode should produce actions");
        assert!(
            matches!(actions[0], Action::Exit),
            "expected Exit, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn keys_mode_bare_c_still_emits_note_on() {
        // Sanity check that the Ctrl+C carve-out didn't break piano C.
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::C), Mod::NOMOD, false, &state, &status)
            .expect("C in Keys mode should produce actions");
        assert!(
            matches!(actions[0], Action::NoteOn { key: 64, .. }),
            "expected NoteOn(E4=64), got {:?}",
            actions[0]
        );
    }

    #[test]
    fn keyup_emits_note_off_even_outside_keys_mode() {
        // Ensures notes held while exiting Keys mode still get NoteOff
        // when the key is released back in Select mode.
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Select);
        let actions = handler
            .classify_keyup(Some(Scancode::Z), &state.mode)
            .expect("Z keyup should be classified");
        assert!(
            matches!(actions[0], Action::NoteOff { key: 60 }),
            "expected NoteOff(60), got {:?}",
            actions[0]
        );
    }

    #[test]
    fn edit_mode_return_asks_runner_to_return_to_select_on_success() {
        // Return in Edit mode emits two actions: `EvaluateAndLeaveEditMode`
        // (parse + evaluate + splice back, plus the mode transition on
        // success or stay-in-Edit on parse error) followed by `PlayProgram`.
        // The classifier's job is just to set up both.
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
        });
        let status = empty_status();
        let actions = handler
            .classify_keydown(Some(Scancode::Return), Mod::NOMOD, false, &state, &status)
            .expect("Return in Edit mode should produce actions");
        assert_eq!(actions.len(), 2);
        assert!(
            matches!(actions[0], Action::EvaluateAndLeaveEditMode { .. }),
            "expected EvaluateAndLeaveEditMode, got {:?}",
            actions[0]
        );
        assert!(
            matches!(
                actions[1],
                Action::PlayProgram {
                    program_index: 0,
                    start_at_next_measure: true,
                    repeat_after_measures: None,
                }
            ),
            "expected PlayProgram, got {:?}",
            actions[1]
        );
    }
}
