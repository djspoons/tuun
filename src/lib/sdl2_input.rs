use std::cell::Cell;
use std::time::Instant;

use crate::actions;
use crate::actions::{Launch, Mode, PrefixKey};
use crate::programs::{PROGRAMS_PER_BANK, Program};

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

/// Returns whether `scancode` is a modifier key, which neither completes nor
/// cancels a prefix key.
fn is_modifier(scancode: sdl2::keyboard::Scancode) -> bool {
    use sdl2::keyboard::Scancode;
    matches!(
        scancode,
        Scancode::LCtrl
            | Scancode::RCtrl
            | Scancode::LShift
            | Scancode::RShift
            | Scancode::LAlt
            | Scancode::RAlt
            | Scancode::LGui
            | Scancode::RGui
            | Scancode::CapsLock
            | Scancode::Mode
    )
}

/// Configuration for the SDL2 classifier — display geometry (used by mouse
/// motion handling) and whether mouse events should be observed at all.
pub struct InputHandler {
    handle_mouse_events: bool,
    display_width: u32,
    display_height: u32,
    /// Whether text input is dropped until the next key press: set when a key
    /// press completes or cancels a prefix key, since SDL follows a printable
    /// key's press with its text.
    swallow_text_input: Cell<bool>,
    /// Whether the next quit event is dropped: set by a Cmd-Q press, since the
    /// app menu follows that press with a quit event of its own.
    swallow_quit: Cell<bool>,
}

impl InputHandler {
    pub fn new(handle_mouse_events: bool, display_width: u32, display_height: u32) -> InputHandler {
        InputHandler {
            handle_mouse_events,
            display_width,
            display_height,
            swallow_text_input: Cell::new(false),
            swallow_quit: Cell::new(false),
        }
    }

    /// Classify an SDL event into a list of `Action`s.
    ///
    /// Returns `None` if this classifier doesn't recognize the event. Returns
    /// `Some(vec![])` when an event is recognized but produces no actions in
    /// the current state.
    pub fn classify(
        &self,
        event: &sdl2::event::Event,
        state: &actions::AppState,
    ) -> Option<Vec<actions::Action>> {
        use actions::Action;
        use sdl2::event::Event;
        let mode: &Mode = &state.mode;
        match event {
            // A quit event exits at once (the app menu's Quit, the window's
            // close button, Ctrl-C in the terminal), unless it follows a Cmd-Q
            // press, which is the quit prefix key.
            Event::Quit { .. } => Some(if self.swallow_quit.replace(false) {
                vec![]
            } else {
                vec![Action::Quit]
            }),
            Event::KeyDown {
                scancode,
                keymod,
                repeat,
                ..
            } => self.classify_keydown(*scancode, *keymod, *repeat, state),
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
        state: &actions::AppState,
    ) -> Option<Vec<actions::Action>> {
        let mode: &Mode = &state.mode;
        let active_program_index = state.active_program_index;
        let programs: &[Program] = state.programs.programs();
        use actions::Action;
        use sdl2::keyboard::{Mod, Scancode};
        let ctrl = keymod.contains(Mod::LCTRLMOD) || keymod.contains(Mod::RCTRLMOD);
        // gui_mod is "command" on a Mac
        let gui_mod = keymod.contains(Mod::LGUIMOD) || keymod.contains(Mod::RGUIMOD);
        let shift = keymod.contains(Mod::LSHIFTMOD) || keymod.contains(Mod::RSHIFTMOD);
        // Text input follows its own key press, so the text of a key press
        // that completed or cancelled a prefix is dropped up to here.
        self.swallow_text_input.set(false);
        // Likewise, a quit event follows its Cmd-Q press directly.
        self.swallow_quit.set(false);
        // Cmd-Q Cmd-Q quits, in every mode and whatever prefix is pending. A
        // held Cmd-Q's auto-repeat does nothing.
        if let Some(Scancode::Q) = scancode
            && gui_mod
        {
            self.swallow_quit.set(true);
            return Some(if repeat {
                vec![]
            } else if state.pending_prefix == Some(PrefixKey::Quit) {
                vec![Action::Quit]
            } else {
                vec![Action::StartPrefix(PrefixKey::Quit)]
            });
        }
        // A pending prefix key takes the next key press, whatever the mode:
        // the press completes a prefixed command or cancels the prefix. A
        // modifier press or a held key's auto-repeat does neither.
        if let Some(prefix) = state.pending_prefix {
            if repeat || scancode.is_some_and(is_modifier) {
                return Some(vec![]);
            }
            self.swallow_text_input.set(true);
            let command = match (prefix, scancode) {
                (PrefixKey::Annotation, Some(Scancode::K)) => {
                    Some(Action::ToggleKeysAnnotation(active_program_index))
                }
                (PrefixKey::Annotation, Some(Scancode::U)) => Some(Action::ChangeSkipSlots {
                    program_index: active_program_index,
                    delta: -1,
                }),
                (PrefixKey::Annotation, Some(Scancode::D)) => Some(Action::ChangeSkipSlots {
                    program_index: active_program_index,
                    delta: 1,
                }),
                _ => None,
            };
            return Some(std::iter::once(Action::EndPrefix).chain(command).collect());
        }
        // Cmd-N starts an annotation command in the modes that don't own the
        // active program's text.
        if let Some(Scancode::N) = scancode
            && gui_mod
        {
            return Some(match mode {
                Mode::Select | Mode::Keys if !repeat => {
                    vec![Action::StartPrefix(PrefixKey::Annotation)]
                }
                _ => vec![],
            });
        }
        // Ctrl+S switches the scope view in every mode, once per press: a held
        // chord's auto-repeat would flip it back and forth.
        if let Some(Scancode::S) = scancode
            && ctrl
        {
            return Some(if repeat {
                vec![]
            } else {
                vec![Action::ToggleScopeView]
            });
        }
        // Keys mode: piano keystrokes → NoteOn (suppress on auto-repeat so a
        // held key doesn't retrigger), Escape leaves keys mode.
        if let Mode::Keys = mode {
            if let Some(Scancode::Escape) = scancode {
                return Some(vec![Action::EnterSelectMode]);
            }
            if repeat {
                return Some(vec![]);
            }
            if let Some(sc) = scancode
                && let Some(midi_note) = scancode_to_midi_note(sc)
            {
                // Stamped at poll, so up to one main-loop pass late.
                return Some(vec![Action::NoteOn {
                    key: midi_note,
                    velocity: 64, // computer keyboard has no velocity; pick mf
                    stamp: Instant::now(),
                }]);
            }
            return Some(vec![]);
        }
        match (mode, scancode) {
            (Mode::Select, Some(Scancode::R)) if gui_mod => Some(vec![Action::ReloadSource]),
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
                if gui_mod {
                    Some(vec![
                        Action::RemovePendingProgram(active_program_index),
                        Action::StopProgram(active_program_index),
                    ])
                } else {
                    Some(vec![Action::RemovePendingProgram(active_program_index)])
                }
            }
            // TODO remove this?
            // Shift+Cmd+Return does nothing.
            (Mode::Select | Mode::Edit { .. }, Some(Scancode::Return)) if gui_mod && shift => {
                Some(vec![])
            }
            (Mode::Select, Some(Scancode::Return)) => {
                if gui_mod {
                    // Cmd+Return: play once, now.
                    if programs[active_program_index].is_playable() {
                        Some(vec![Action::PlayProgram {
                            program_index: active_program_index,
                            launch: Launch::Now,
                        }])
                    } else {
                        Some(vec![Action::ShowMessage(format!(
                            "Program {} is not a valid waveform",
                            state.programs.display_name(active_program_index)
                        ))])
                    }
                } else {
                    Some(vec![Action::EnterEditMode])
                }
            }
            (Mode::Edit { .. }, Some(Scancode::Escape)) => {
                if gui_mod {
                    // Cmd+Escape stops the active waveform (if playing)
                    // but stays in Edit mode.
                    Some(vec![Action::StopProgram(active_program_index)])
                } else {
                    // Plain Escape returns to Select mode.
                    Some(vec![Action::EvaluateAndLeaveEditMode {
                        mode_on_failure: Mode::Select,
                        play: None,
                    }])
                }
            }
            (mode @ Mode::Edit { .. }, Some(Scancode::Return)) => {
                // Cmd+Return: play once, now.
                // Plain Return: play from the next measure.
                let launch = if gui_mod {
                    Launch::Now
                } else {
                    Launch::Pending
                };
                // Either way, ask the runner to drop us into Select on
                // success or stay in Edit with the error on parse
                // failure.
                Some(vec![Action::EvaluateAndLeaveEditMode {
                    mode_on_failure: mode.clone(),
                    play: Some(launch),
                }])
            }
            // Char- and line-level ops on Ctrl, word-level ops on Cmd
            // (mirroring emacs's Ctrl/Meta split).
            (Mode::Edit { .. }, Some(Scancode::A)) if ctrl => Some(vec![Action::MoveCursorToStart]),
            (Mode::Edit { .. }, Some(Scancode::E)) if ctrl => Some(vec![Action::MoveCursorToEnd]),
            (Mode::Edit { .. }, Some(Scancode::F)) if ctrl => Some(vec![Action::MoveCursorBy(1)]),
            (Mode::Edit { .. }, Some(Scancode::B)) if ctrl => Some(vec![Action::MoveCursorBy(-1)]),
            (Mode::Edit { .. }, Some(Scancode::D)) if ctrl => {
                Some(vec![Action::DeleteCharAfterCursor])
            }
            (Mode::Edit { .. }, Some(Scancode::K)) if ctrl => Some(vec![Action::DeleteToEndOfLine]),
            (Mode::Edit { .. }, Some(Scancode::F)) if gui_mod => {
                Some(vec![Action::MoveCursorToNextWord])
            }
            (Mode::Edit { .. }, Some(Scancode::B)) if gui_mod => {
                Some(vec![Action::MoveCursorToPreviousWord])
            }
            (Mode::Edit { .. }, Some(Scancode::D)) if gui_mod => {
                Some(vec![Action::DeleteWordAfterCursor])
            }
            (Mode::Edit { .. }, Some(Scancode::Backspace)) if gui_mod => {
                Some(vec![Action::DeleteWordBeforeCursor])
            }
            // Completion on Cmd-/.
            (Mode::Edit { .. }, Some(Scancode::Slash)) if gui_mod => Some(vec![Action::Complete]),
            // Type of the identifier under the cursor on Ctrl-H.
            (Mode::Edit { .. }, Some(Scancode::H)) if ctrl => Some(vec![Action::ShowType]),
            // Undo / redo on Ctrl+Z or Cmd+Z, in Edit or Select mode; Shift
            // reverses. The redo arm comes first so the shifted chord isn't
            // captured by the undo arm.
            (Mode::Edit { .. } | Mode::Select, Some(Scancode::Z)) if (ctrl || gui_mod) && shift => {
                Some(vec![Action::Redo])
            }
            (Mode::Edit { .. } | Mode::Select, Some(Scancode::Z))
                if (ctrl || gui_mod) && !shift =>
            {
                Some(vec![Action::Undo])
            }
            (Mode::Edit { .. }, Some(Scancode::Left)) => Some(vec![Action::MoveCursorBy(-1)]),
            (Mode::Edit { .. }, Some(Scancode::Right)) => Some(vec![Action::MoveCursorBy(1)]),
            (Mode::Edit { .. }, Some(Scancode::Backspace)) => {
                Some(vec![Action::DeleteCharBeforeCursor])
            }
            (Mode::Edit { .. }, Some(Scancode::Delete)) => {
                Some(vec![Action::DeleteCharAfterCursor])
            }
            // Recognized keyboard event with no binding in the current mode.
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
        // NoteOff fires for piano scancodes in any mode, not just Keys. This
        // avoids stuck notes if the user exits Keys mode while still holding a
        // key. The PlayNoteOff is a no-op when the key isn't currently playing.
        if let Some(sc) = scancode
            && let Some(midi_note) = scancode_to_midi_note(sc)
        {
            return Some(vec![Action::NoteOff {
                key: midi_note,
                stamp: Instant::now(),
            }]);
        }
        match (mode, scancode) {
            (Mode::MoveSliders, Some(Scancode::LAlt) | Some(Scancode::RAlt)) => {
                Some(vec![Action::EnterSelectMode])
            }
            // Recognized but no binding.
            _ => Some(vec![]),
        }
    }

    fn classify_text_input(
        &self,
        text: &str,
        state: &actions::AppState,
    ) -> Option<Vec<actions::Action>> {
        use actions::Action;
        if self.swallow_text_input.replace(false) || state.pending_prefix.is_some() {
            return Some(vec![]);
        }
        match state.mode {
            Mode::Select => match text {
                "D" => Some(vec![Action::PrintEvaluatedProgram(
                    state.active_program_index,
                )]),
                // Install/uninstall the active program as the keys instrument
                "K" => Some(vec![Action::ToggleInstalledKeys(
                    state.active_program_index,
                )]),
                // Enter Keys (piano) mode
                "k" => Some(vec![Action::EnterKeysMode]),
                "m" => Some(vec![Action::ToggleMetronome]),
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
            // Text input in modes that don't accept it is ignored.
            _ => Some(vec![]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::{Action, AppState};
    use crate::recorder;
    use sdl2::keyboard::{Mod, Scancode};

    fn test_state(mode: Mode) -> AppState {
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = test;".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        state.mode = mode;
        state
    }

    #[test]
    fn record_mode_accepts_only_ctrl_s() {
        let handler = InputHandler::new(true, 800, 600);
        let now = Instant::now();
        let state = test_state(Mode::Record {
            take: recorder::Take::arm("kb".to_string(), 60, now, now),
            keys_program: 0,
        });
        let toggle = handler.classify_keydown(Some(Scancode::S), Mod::LCTRLMOD, false, &state);
        assert!(matches!(toggle.as_deref(), Some([Action::ToggleScopeView])));
        for (scancode, keymod) in [
            (Scancode::Return, Mod::NOMOD),
            (Scancode::Escape, Mod::NOMOD),
            (Scancode::Escape, Mod::LGUIMOD),
            (Scancode::Up, Mod::NOMOD),
            (Scancode::R, Mod::LGUIMOD),
            (Scancode::LAlt, Mod::NOMOD),
            (Scancode::Z, Mod::LGUIMOD),
            (Scancode::N, Mod::LGUIMOD),
        ] {
            let actions = handler.classify_keydown(Some(scancode), keymod, false, &state);
            assert!(
                matches!(actions.as_deref(), Some([])),
                "{:?} gave {:?}",
                scancode,
                actions
            );
        }
        for text in ["1", "k", "K", "D", "m"] {
            let actions = handler.classify_text_input(text, &state);
            assert!(matches!(actions.as_deref(), Some([])), "{:?}", text);
        }
    }

    #[test]
    fn keys_mode_lower_row_emits_note_on() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::NOMOD, false, &state)
            .expect("Z in Keys mode should produce actions");
        assert!(
            matches!(
                actions[0],
                Action::NoteOn {
                    key: 60,
                    velocity: 64,
                    ..
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
        // repeat=true means SDL is auto-repeating a held key. We must
        // not retrigger NoteOn — the note is already playing.
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::NOMOD, true, &state)
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
        let actions = handler
            .classify_keydown(Some(Scancode::Escape), Mod::NOMOD, false, &state)
            .expect("Escape in Keys mode should produce actions");
        assert!(matches!(actions[0], Action::EnterSelectMode));
    }

    /// Returns `state` with `prefix` pending.
    fn pending(mut state: AppState, prefix: PrefixKey) -> AppState {
        state.pending_prefix = Some(prefix);
        state
    }

    #[test]
    fn quitting_takes_two_cmd_q_presses() {
        let handler = InputHandler::new(false, 800, 600);
        let quit = sdl2::event::Event::Quit { timestamp: 0 };
        // Each Cmd-Q press is followed by the app menu's quit event.
        let state = test_state(Mode::Keys);
        let first = handler.classify_keydown(Some(Scancode::Q), Mod::LGUIMOD, false, &state);
        assert!(matches!(
            first.as_deref(),
            Some([Action::StartPrefix(PrefixKey::Quit)])
        ));
        assert!(matches!(
            handler.classify(&quit, &state).as_deref(),
            Some([])
        ));
        let state = pending(test_state(Mode::Keys), PrefixKey::Quit);
        let second = handler.classify_keydown(Some(Scancode::Q), Mod::LGUIMOD, false, &state);
        assert!(matches!(second.as_deref(), Some([Action::Quit])));
        assert!(matches!(
            handler.classify(&quit, &state).as_deref(),
            Some([])
        ));
        // Cmd-Q while another prefix waits starts the quit prefix.
        let state = pending(test_state(Mode::Select), PrefixKey::Annotation);
        let actions = handler.classify_keydown(Some(Scancode::Q), Mod::RGUIMOD, false, &state);
        assert!(matches!(
            actions.as_deref(),
            Some([Action::StartPrefix(PrefixKey::Quit)])
        ));
        // A held Cmd-Q does nothing more.
        let actions = handler.classify_keydown(Some(Scancode::Q), Mod::LGUIMOD, true, &state);
        assert!(matches!(actions.as_deref(), Some([])));
    }

    #[test]
    fn a_quit_event_without_cmd_q_exits_at_once() {
        let handler = InputHandler::new(false, 800, 600);
        let quit = sdl2::event::Event::Quit { timestamp: 0 };
        let state = test_state(Mode::Select);
        assert!(matches!(
            handler.classify(&quit, &state).as_deref(),
            Some([Action::Quit])
        ));
        // A Cmd-Q press drops only the quit event right after it.
        handler.classify_keydown(Some(Scancode::Q), Mod::LGUIMOD, false, &state);
        handler.classify_keydown(Some(Scancode::Escape), Mod::NOMOD, false, &state);
        assert!(matches!(
            handler.classify(&quit, &state).as_deref(),
            Some([Action::Quit])
        ));
    }

    #[test]
    fn a_key_press_cancels_a_pending_quit() {
        let handler = InputHandler::new(false, 800, 600);
        let state = pending(test_state(Mode::Select), PrefixKey::Quit);
        for (scancode, keymod) in [
            (Scancode::Escape, Mod::NOMOD),
            (Scancode::Return, Mod::NOMOD),
            (Scancode::S, Mod::LCTRLMOD),
            (Scancode::K, Mod::NOMOD),
        ] {
            let actions = handler.classify_keydown(Some(scancode), keymod, false, &state);
            assert!(
                matches!(actions.as_deref(), Some([Action::EndPrefix])),
                "{:?} gave {:?}",
                scancode,
                actions
            );
        }
        // The cancelling key's text does nothing either.
        let actions = handler.classify_text_input("k", &test_state(Mode::Select));
        assert!(matches!(actions.as_deref(), Some([])), "{:?}", actions);
    }

    #[test]
    fn modifiers_and_repeats_leave_a_prefix_pending() {
        let handler = InputHandler::new(false, 800, 600);
        let state = pending(test_state(Mode::Keys), PrefixKey::Quit);
        for (scancode, keymod, repeat) in [
            (Scancode::LGui, Mod::LGUIMOD, false),
            (Scancode::RShift, Mod::RSHIFTMOD, false),
            (Scancode::Z, Mod::NOMOD, true),
        ] {
            let actions = handler.classify_keydown(Some(scancode), keymod, repeat, &state);
            assert!(
                matches!(actions.as_deref(), Some([])),
                "{:?} gave {:?}",
                scancode,
                actions
            );
        }
    }

    #[test]
    fn cmd_n_starts_the_annotation_prefix_in_select_and_keys_modes() {
        let handler = InputHandler::new(false, 800, 600);
        for mode in [Mode::Select, Mode::Keys] {
            let state = test_state(mode);
            let actions = handler.classify_keydown(Some(Scancode::N), Mod::LGUIMOD, false, &state);
            assert!(
                matches!(
                    actions.as_deref(),
                    Some([Action::StartPrefix(PrefixKey::Annotation)])
                ),
                "{:?} gave {:?}",
                state.mode,
                actions
            );
        }
        let state = test_state(Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        });
        let actions = handler.classify_keydown(Some(Scancode::N), Mod::LGUIMOD, false, &state);
        assert!(matches!(actions.as_deref(), Some([])), "{:?}", actions);
    }

    #[test]
    fn annotation_prefix_completes_on_k_u_and_d() {
        let handler = InputHandler::new(false, 800, 600);
        let state = pending(test_state(Mode::Select), PrefixKey::Annotation);
        let classify = |scancode, keymod| {
            handler
                .classify_keydown(Some(scancode), keymod, false, &state)
                .expect("key press should be classified")
        };
        assert!(matches!(
            classify(Scancode::K, Mod::NOMOD)[..],
            [Action::EndPrefix, Action::ToggleKeysAnnotation(0)]
        ));
        // Cmd may still be held from the prefix key.
        assert!(matches!(
            classify(Scancode::U, Mod::LGUIMOD)[..],
            [
                Action::EndPrefix,
                Action::ChangeSkipSlots {
                    program_index: 0,
                    delta: -1
                }
            ]
        ));
        assert!(matches!(
            classify(Scancode::D, Mod::NOMOD)[..],
            [
                Action::EndPrefix,
                Action::ChangeSkipSlots {
                    program_index: 0,
                    delta: 1
                }
            ]
        ));
        assert!(matches!(
            classify(Scancode::X, Mod::NOMOD)[..],
            [Action::EndPrefix]
        ));
        // In Select mode `k` would enter Keys mode, but its text follows the
        // press that completed the prefix.
        let actions = handler.classify_text_input("k", &test_state(Mode::Select));
        assert!(matches!(actions.as_deref(), Some([])), "{:?}", actions);
        // Text after the next key press is read again.
        handler.classify_keydown(
            Some(Scancode::M),
            Mod::NOMOD,
            false,
            &test_state(Mode::Select),
        );
        let actions = handler.classify_text_input("m", &test_state(Mode::Select));
        assert!(matches!(
            actions.as_deref(),
            Some([Action::ToggleMetronome])
        ));
    }

    #[test]
    fn annotation_prefix_in_keys_mode_plays_no_note() {
        let handler = InputHandler::new(false, 800, 600);
        let state = pending(test_state(Mode::Keys), PrefixKey::Annotation);
        let actions = handler
            .classify_keydown(Some(Scancode::D), Mod::NOMOD, false, &state)
            .expect("D should be classified");
        assert!(
            !actions.iter().any(|a| matches!(a, Action::NoteOn { .. })),
            "got {:?}",
            actions
        );
    }

    #[test]
    fn ctrl_s_toggles_the_scope_view_in_every_mode() {
        let handler = InputHandler::new(true, 800, 600);
        let now = Instant::now();
        for mode in [
            Mode::Select,
            Mode::Edit {
                cursor_position: 0,
                errors: vec![],
                completion: None,
            },
            Mode::MoveSliders,
            Mode::Keys,
            Mode::Record {
                take: recorder::Take::arm("kb".to_string(), 60, now, now),
                keys_program: 0,
            },
        ] {
            let state = test_state(mode);
            for keymod in [Mod::LCTRLMOD, Mod::RCTRLMOD] {
                let actions = handler.classify_keydown(Some(Scancode::S), keymod, false, &state);
                assert!(
                    matches!(actions.as_deref(), Some([Action::ToggleScopeView])),
                    "{:?} with {:?} gave {:?}",
                    state.mode,
                    keymod,
                    actions
                );
            }
            // A held chord's auto-repeat doesn't toggle again.
            let actions = handler.classify_keydown(Some(Scancode::S), Mod::LCTRLMOD, true, &state);
            assert!(
                matches!(actions.as_deref(), Some([])),
                "{:?} on repeat gave {:?}",
                state.mode,
                actions
            );
        }
    }

    #[test]
    fn keys_mode_bare_s_still_emits_note_on() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Keys);
        let actions = handler
            .classify_keydown(Some(Scancode::S), Mod::NOMOD, false, &state)
            .expect("S in Keys mode should produce actions");
        assert!(
            matches!(actions[..], [Action::NoteOn { key: 61, .. }]),
            "expected NoteOn(C#4=61), got {:?}",
            actions
        );
    }

    #[test]
    fn select_mode_m_toggles_the_metronome() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Select);
        let actions = handler.classify_text_input("m", &state);
        assert!(matches!(
            actions.as_deref(),
            Some([Action::ToggleMetronome])
        ));
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
            matches!(actions[0], Action::NoteOff { key: 60, .. }),
            "expected NoteOff(60), got {:?}",
            actions[0]
        );
    }

    #[test]
    fn edit_mode_binds_char_ops_to_ctrl_and_word_ops_to_cmd() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
            completion: None,
        });
        let classify = |scancode, keymod| {
            let actions = handler
                .classify_keydown(Some(scancode), keymod, false, &state)
                .expect("Edit-mode key should be classified");
            assert_eq!(actions.len(), 1, "expected one action, got {:?}", actions);
            actions.into_iter().next().unwrap()
        };
        // Char- and line-level ops on Ctrl.
        assert!(matches!(
            classify(Scancode::F, Mod::LCTRLMOD),
            Action::MoveCursorBy(1)
        ));
        assert!(matches!(
            classify(Scancode::B, Mod::LCTRLMOD),
            Action::MoveCursorBy(-1)
        ));
        assert!(matches!(
            classify(Scancode::D, Mod::LCTRLMOD),
            Action::DeleteCharAfterCursor
        ));
        assert!(matches!(
            classify(Scancode::K, Mod::LCTRLMOD),
            Action::DeleteToEndOfLine
        ));
        // Word-level ops on Cmd (gui mod).
        assert!(matches!(
            classify(Scancode::F, Mod::LGUIMOD),
            Action::MoveCursorToNextWord
        ));
        assert!(matches!(
            classify(Scancode::B, Mod::LGUIMOD),
            Action::MoveCursorToPreviousWord
        ));
        assert!(matches!(
            classify(Scancode::D, Mod::LGUIMOD),
            Action::DeleteWordAfterCursor
        ));
        assert!(matches!(
            classify(Scancode::Backspace, Mod::LGUIMOD),
            Action::DeleteWordBeforeCursor
        ));
        // Completion on Cmd-/.
        assert!(matches!(
            classify(Scancode::Slash, Mod::LGUIMOD),
            Action::Complete
        ));
        // Type of the identifier under the cursor on Ctrl-H.
        assert!(matches!(
            classify(Scancode::H, Mod::LCTRLMOD),
            Action::ShowType
        ));
        // Forward delete works unmodified.
        assert!(matches!(
            classify(Scancode::Delete, Mod::NOMOD),
            Action::DeleteCharAfterCursor
        ));
    }

    #[test]
    fn select_mode_binds_reload_on_cmd_r() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Select);
        let actions = handler
            .classify_keydown(Some(Scancode::R), Mod::LGUIMOD, false, &state)
            .expect("Cmd+R in Select mode should be classified");
        assert!(matches!(actions[0], Action::ReloadSource));
        // Plain R is inert in Select mode.
        let actions = handler
            .classify_keydown(Some(Scancode::R), Mod::NOMOD, false, &state)
            .expect("plain R falls through");
        assert!(actions.is_empty());
    }

    #[test]
    fn select_mode_binds_undo_and_redo_on_z() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Select);
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::LGUIMOD, false, &state)
            .expect("Cmd+Z in Select mode should be classified");
        assert!(matches!(actions[..], [Action::Undo]));
        let actions = handler
            .classify_keydown(
                Some(Scancode::Z),
                Mod::LCTRLMOD | Mod::LSHIFTMOD,
                false,
                &state,
            )
            .expect("Ctrl+Shift+Z in Select mode should be classified");
        assert!(matches!(actions[..], [Action::Redo]));
    }

    #[test]
    fn edit_mode_binds_undo_and_redo_on_z() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
            completion: None,
        });
        let classify = |keymod| {
            let actions = handler
                .classify_keydown(Some(Scancode::Z), keymod, false, &state)
                .expect("Z chord in Edit mode should be classified");
            assert_eq!(actions.len(), 1, "expected one action, got {:?}", actions);
            actions.into_iter().next().unwrap()
        };
        assert!(matches!(classify(Mod::LCTRLMOD), Action::Undo));
        assert!(matches!(classify(Mod::LGUIMOD), Action::Undo));
        assert!(matches!(
            classify(Mod::LCTRLMOD | Mod::LSHIFTMOD),
            Action::Redo
        ));
        assert!(matches!(
            classify(Mod::LGUIMOD | Mod::LSHIFTMOD),
            Action::Redo
        ));
        // Plain Z is not a keydown binding — the character arrives through
        // a TextInput event instead.
        let actions = handler
            .classify_keydown(Some(Scancode::Z), Mod::NOMOD, false, &state)
            .expect("plain Z should be recognized");
        assert!(actions.is_empty(), "expected no actions, got {:?}", actions);
    }

    #[test]
    fn edit_mode_return_asks_runner_to_return_to_select_on_success() {
        // Return in Edit mode leaves Edit mode and makes a pending launch, in
        // a single action.
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
            completion: None,
        });
        let actions = handler
            .classify_keydown(Some(Scancode::Return), Mod::NOMOD, false, &state)
            .expect("Return in Edit mode should produce actions");
        assert_eq!(actions.len(), 1);
        assert!(
            matches!(
                actions[0],
                Action::EvaluateAndLeaveEditMode {
                    play: Some(Launch::Pending),
                    ..
                }
            ),
            "expected EvaluateAndLeaveEditMode with play, got {:?}",
            actions[0]
        );
    }

    #[test]
    fn edit_mode_cmd_return_plays_now() {
        let handler = InputHandler::new(false, 800, 600);
        let state = test_state(Mode::Edit {
            cursor_position: 4,
            errors: vec![],
            completion: None,
        });
        let actions = handler
            .classify_keydown(Some(Scancode::Return), Mod::LGUIMOD, false, &state)
            .expect("Cmd+Return in Edit mode should produce actions");
        assert!(
            matches!(
                &actions[..],
                [Action::EvaluateAndLeaveEditMode {
                    play: Some(Launch::Now),
                    ..
                }]
            ),
            "expected EvaluateAndLeaveEditMode playing now, got {:?}",
            actions
        );
    }
}
