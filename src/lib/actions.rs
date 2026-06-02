//! Action / Effect / reducer for app state.
//!
//! Handlers (sdl2_input, midi_input) classify raw input events into pure
//! `Action`s. `apply` then mutates `AppState` and returns `Effect`s, which
//! the runner in `effects.rs` executes against the outside world.

use crate::launchkey;
use crate::loader;
use crate::midi_input;
use crate::parser;
use crate::play_helper;
use crate::renderer::{self, MarkId, Mode, PROGRAMS_PER_BANK, WaveformId};
use crate::slider;
use crate::waveform;

/// Behavior of the pads when the controller is in DAW pad mode. Cycled by
/// `Action::CycleDawPadMode` on each `Event::PadModeChanged(DAW)` so the
/// user can repurpose the pads by re-pressing the DAW pad-mode button.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DawPadMode {
    /// Top row plays or stops, bottom row queues or cancels.
    ClipLauncher,
    /// Bottom row installs (or uninstalls) the program as the keys instrument.
    KeysInstaller,
}

impl DawPadMode {
    /// Human-readable label shown on the controller's display strip when
    /// this sub-mode is active.
    pub fn display_name(self) -> &'static str {
        match self {
            DawPadMode::ClipLauncher => "Clip Launcher",
            DawPadMode::KeysInstaller => "Keys Installer",
        }
    }
}

/// Internal state of the application. The reducer takes `&mut AppState` and mutates it in
/// place; `main` keeps a single instance for the lifetime of the program.
pub struct AppState {
    pub programs: Vec<renderer::Program>,
    pub active_program_index: usize,
    pub mode: Mode,
    pub keys: Option<midi_input::Keys>,
    pub repeat_after_measures: Option<u32>,
    /// Active behavior for the DAW pad.
    pub daw_pad_mode: DawPadMode,
    /// Parser context: prelude + bindings loaded from context files.
    pub context: Vec<(String, parser::Expr<MarkId>)>,
    /// Loader configuration: filenames, tempo, sample rate.
    pub config: loader::Config,
    /// Set by `Effect::Exit`; `main` checks at top of loop and breaks.
    pub should_exit: bool,
    /// Last user-visible status message. Set by `Effect::ShowMessage` (and
    /// a few direct writes from the reducer / runner). Persists across
    /// mode transitions; cleared explicitly by navigation actions.
    pub message: String,
}

impl AppState {
    pub fn bank_start(&self) -> usize {
        self.active_program_index - (self.active_program_index % PROGRAMS_PER_BANK)
    }

    pub fn active_program(&self) -> &renderer::Program {
        &self.programs[self.active_program_index]
    }
}

/// Things that can happen, emitted by handlers as pure data.
#[derive(Debug)]
pub enum Action {
    // --- playback ---
    /// Play the program at `program_index`.
    ///
    /// When `return_to_select_on_success` is true, the runner sets
    /// `state.mode` from `play_waveform`'s return: `Mode::Select` on
    /// success (so the caller drops back to Select), or `Mode::Edit` on
    /// a parse error (so the caller stays in Edit and can fix the
    /// syntax). Used by the SDL2 Edit+Return path; left false elsewhere.
    PlayProgram {
        program_index: usize,
        cursor_position: usize,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
        return_to_select_on_success: bool,
    },
    /// Stop playback of the program waveform immediately.
    StopProgram(usize),
    /// Remove any pending playback for the given program.
    RemovePendingProgram(usize),

    // --- MIDI keys ---
    /// Install the program at `program_index` as the keys instrument. If
    /// keys are already installed (regardless of which program), this
    /// uninstalls them. (That is, the action is a toggle.)
    InstallKeys(usize),
    NoteOn {
        key: u8,
        velocity: u8,
    },
    NoteOff {
        key: u8,
    },

    // --- mode transitions ---
    EnterEditMode,
    EnterSelectMode,
    EnterMoveSlidersMode,
    /// Enter computer-keyboard piano mode. Caller (the classifier) is
    /// responsible for only emitting this when `state.keys.is_some()` —
    /// the reducer accepts unconditionally.
    EnterKeysMode,

    // --- navigation ---
    /// Jump to a specific program index.
    SelectProgram(usize),
    /// Move the active index by `delta` (wraps).
    AdvanceProgram(i32),

    // --- text editing (Edit mode) ---
    InsertText(String),
    DeleteCharBeforeCursor,
    DeleteWordBeforeCursor,
    MoveCursorBy(i32),
    MoveCursorToStart,
    MoveCursorToEnd,
    MoveCursorToPreviousWord,

    // --- sliders / level ---
    /// Set a slider on a program by normalized value [0.0, 1.0].
    SetSliderNormalized {
        program: usize,
        slider_index: usize,
        normalized: f32,
    },
    /// Set the level (in dB) of a program.
    SetLevelDb {
        program: usize,
        level_db: f32,
    },

    // --- mouse-driven sliders ---
    /// Adjust the X- or Y-axis slider on the active program by a relative
    /// amount (normalized delta in [-1, 1] roughly).
    AdjustMouseSlider {
        axis: usize, // 0 = X, 1 = Y
        delta: f32,
    },

    // --- other MIDI controller state ---
    /// Encoder mode changed on the controller.
    SetEncoderMode(launchkey::EncoderMode),
    /// Pad mode changed on the controller.
    SetPadMode(launchkey::PadMode),
    /// Cycle the DAW-pad-mode sub-behavior.
    CycleDawPadMode,
    /// Toggle the default repeat (cycle None -> Some(1) -> Some(2) -> None).
    CycleRepeatAfterMeasures,

    // --- program-related I/O and other effects ---
    SaveProgramsToFile,
    DumpActiveWaveform,
    LoadContext,
    LoadPrograms,
    Exit,
}

/// Side effects to perform after the reducer runs. The runner translates
/// these into MPSC sends, MIDI controller updates, etc.
#[derive(Debug)]
pub enum Effect {
    // --- tracker commands ---
    /// Send a Play command for the program at `program_index`. The runner
    /// looks up the program text, parses, and dispatches via play_helper.
    /// See `Action::PlayProgram` for `return_to_select_on_success`.
    PlayProgram {
        program_index: usize,
        cursor_position: usize,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
        return_to_select_on_success: bool,
    },
    /// Send a "stop" ramp for the program waveform.
    StopProgram(usize),
    /// Remove a pending program from the tracker.
    RemovePendingProgram(usize),
    /// Send a Modify command to the tracker for one waveform.
    ModifyWaveform {
        id: WaveformId,
        mark_id: MarkId,
        waveform: waveform::Waveform<MarkId>,
    },

    /// Parse the program at `program_index` as a keys instrument function and
    /// install it as `state.keys`. The runner handles parsing and validation
    /// because they depend on parser + midi_input helpers; on success it
    /// mutates `state.keys` directly.
    InstallKeysFromActive(usize),

    /// Play the note-on waveform produced by the installed keys function.
    PlayNoteOn {
        key: u8,
        velocity: u8,
    },
    /// Modify the Level mark on the active key waveform with the note-off
    /// waveform that was stored at NoteOn time. Falls back to a stop-ramp if
    /// no waveform was stored.
    PlayNoteOff {
        key: u8,
    },

    /// Push a slider value change into the slider pipeline. The downstream
    /// worker thread coalesces these per quantum and sends `Command::Modify`
    /// to the tracker with a ramp from the previous value, so the running
    /// waveform actually picks up the new slider value. (The renderer also
    /// observes these events for visual feedback.)
    UpdateSlider {
        id: WaveformId,
        slider: String,
        value: f32,
    },
    /// Propagate a slider change to every currently active key waveform.
    /// The runner walks the tracker status to find active Key marks.
    UpdateActiveKeySliders {
        // TODO consider using UpdateSlider?
        slider: String,
        value: f32,
    },
    /// Modify the Amplitude mark on every currently active key waveform.
    //
    // XXX Should this go through the slider pipeline like UpdateActiveKeySliders
    // does (coalesce per quantum, ramp from prior value)? Currently each call
    // sends a Const directly to the tracker, which can cause a step change.
    ModifyActiveKeysAmplitude {
        amplitude: f32,
    },
    /// User-visible status message.
    ShowMessage(String),

    // --- launchkey hardware ---
    /// Update the controller's encoder display text.
    SetEncoderDisplay {
        index: u8,
        name: String,
        value: String,
    },
    /// Push the current bank/program's encoder values back to the controller.
    /// Only emitted when the source-of-truth (program index, encoder mode)
    /// actually changes.
    SyncEncoders,
    /// Update the cached encoder mode on the Launchkey controller and
    /// re-sync the encoders. Runner skips the sync if the cached value
    /// already matches.
    SetLaunchkeyEncoderMode(launchkey::EncoderMode),
    /// Update the pad mode on the Launchkey controller.
    SetLaunchkeyPadMode(launchkey::PadMode),
    /// Push a label onto the controller display strip indicating which DAW mode is active.
    SetDawModeDisplay(String),

    // --- I/O ---
    SaveProgramsToFile,
    DumpActiveWaveform,
    /// Re-read `state.config.context_files` into `state.context`.
    LoadContext,
    /// Re-read `state.config.programs_file` (+ additional programs) into
    /// `state.programs` and push the resulting slider initial values into
    /// the slider pipeline.
    LoadPrograms,
    /// Sets `state.should_exit = true`.
    Exit,
}

/// Pure reducer: applies an action to state, returns effects.
///
/// No I/O. No MPSC sends. No closures.
pub fn apply(state: &mut AppState, action: Action) -> Vec<Effect> {
    match action {
        Action::PlayProgram {
            program_index,
            cursor_position,
            start_at_next_measure,
            repeat_after_measures,
            return_to_select_on_success,
        } => vec![Effect::PlayProgram {
            program_index,
            cursor_position,
            start_at_next_measure,
            repeat_after_measures,
            return_to_select_on_success,
        }],
        Action::StopProgram(i) => vec![
            Effect::StopProgram(i),
            Effect::ShowMessage(format!("Stopped program {}", state.programs[i].id)),
        ],
        Action::RemovePendingProgram(i) => vec![
            Effect::RemovePendingProgram(i),
            Effect::ShowMessage(format!(
                "Removed pending waveform for program {}",
                state.programs[i].id
            )),
        ],

        Action::InstallKeys(i) => apply_install_keys(state, i),
        // The runner does the parser work and the actual state mutation
        // (storing note_off waveforms) for both of these.
        Action::NoteOn { key, velocity } => {
            if state.keys.is_some() {
                vec![Effect::PlayNoteOn { key, velocity }]
            } else {
                vec![]
            }
        }
        Action::NoteOff { key } => vec![Effect::PlayNoteOff { key }],

        Action::EnterEditMode => {
            let program = state.active_program();
            let cursor = program.text.len();
            let errors = parse_program_errors(&program.text);
            state.message = if !errors.is_empty() {
                format!("Error: {}", errors[0].to_string())
            } else if !program.sliders.configs.is_empty() {
                program
                    .sliders
                    .slider_display()
                    .iter()
                    .map(|s| format!("{}", s))
                    .collect::<Vec<_>>()
                    .join(", ")
            } else {
                String::new()
            };
            state.mode = Mode::Edit {
                cursor_position: cursor,
                errors,
            };
            vec![]
        }
        Action::EnterSelectMode => {
            state.mode = Mode::Select;
            state.message.clear();
            vec![]
        }
        Action::EnterMoveSlidersMode => {
            state.mode = Mode::MoveSliders;
            vec![]
        }
        Action::EnterKeysMode => {
            state.mode = Mode::Keys;
            vec![Effect::ShowMessage("Piano keys enabled".to_string())]
        }

        Action::SelectProgram(i) => apply_select_program(state, i),
        Action::AdvanceProgram(delta) => {
            let len = state.programs.len() as i32;
            if len == 0 {
                vec![]
            } else {
                let cur = state.active_program_index as i32;
                let new = ((cur + delta) % len + len) % len;
                apply_select_program(state, new as usize)
            }
        }

        Action::InsertText(text) => apply_insert_text(state, &text),
        Action::DeleteCharBeforeCursor => apply_delete_char(state),
        Action::DeleteWordBeforeCursor => apply_delete_word(state),
        Action::MoveCursorBy(delta) => apply_move_cursor_by(state, delta),
        Action::MoveCursorToStart => apply_move_cursor_to(state, 0),
        Action::MoveCursorToEnd => {
            let len = state.active_program().text.len();
            apply_move_cursor_to(state, len)
        }
        Action::MoveCursorToPreviousWord => apply_move_cursor_prev_word(state),

        Action::SetSliderNormalized {
            program,
            slider_index,
            normalized,
        } => apply_slider(state, program, slider_index, normalized),

        Action::SetLevelDb { program, level_db } => apply_level_db(state, program, level_db),

        Action::AdjustMouseSlider { axis, delta } => apply_mouse_slider(state, axis, delta),

        Action::SetEncoderMode(new_mode) => {
            // encoder_mode now lives on Launchkey; the runner updates that
            // cache and re-syncs only if it actually changed.
            vec![Effect::SetLaunchkeyEncoderMode(new_mode)]
        }
        Action::SetPadMode(new_mode) => vec![Effect::SetLaunchkeyPadMode(new_mode)],
        Action::CycleDawPadMode => {
            let next = match state.daw_pad_mode {
                DawPadMode::ClipLauncher => DawPadMode::KeysInstaller,
                DawPadMode::KeysInstaller => DawPadMode::ClipLauncher,
            };
            state.daw_pad_mode = next;
            let label = next.display_name().to_string();
            vec![
                Effect::SetDawModeDisplay(label.clone()),
                Effect::ShowMessage(label),
            ]
        }

        Action::CycleRepeatAfterMeasures => {
            let effect;
            (state.repeat_after_measures, effect) = match state.repeat_after_measures {
                None => (
                    Some(1),
                    Effect::ShowMessage("Repeat after 1 measure".to_string()),
                ),
                Some(1) => (
                    Some(2),
                    Effect::ShowMessage("Repeat after 2 measures".to_string()),
                ),
                Some(_) => (None, Effect::ShowMessage("No repeats".to_string())),
            };
            vec![effect]
        }

        Action::SaveProgramsToFile => vec![Effect::SaveProgramsToFile],
        Action::DumpActiveWaveform => vec![Effect::DumpActiveWaveform],
        Action::LoadContext => vec![Effect::LoadContext],
        Action::LoadPrograms => vec![Effect::LoadPrograms, Effect::SyncEncoders],
        Action::Exit => vec![Effect::Exit],
    }
}

fn apply_select_program(state: &mut AppState, i: usize) -> Vec<Effect> {
    if i >= state.programs.len() {
        return vec![];
    }
    let changed = state.active_program_index != i;
    state.active_program_index = i;
    // Always clear the status message: navigation represents a fresh
    // context, so any prior "Removed pending..." / "Playing..." etc.
    // shouldn't carry over.
    let mut effects = vec![Effect::ShowMessage(String::new())];
    if changed {
        effects.push(Effect::SyncEncoders);
    }
    effects
}

fn apply_install_keys(state: &mut AppState, program_index: usize) -> Vec<Effect> {
    // Applying with the currently-installed program uninstalls it; for any
    // other program, tries to install that one in its place. (The runner
    // leaves `state.keys` untouched if the new program fails to parse or
    // validate, so an invalid target is a no-op rather than a kick-out.)
    if let Some(keys) = &state.keys
        && renderer::index_from_id(keys.id) == program_index
    {
        state.keys = None;
        return vec![Effect::ShowMessage("Uninstalled keys".to_string())];
    }
    vec![Effect::InstallKeysFromActive(program_index)]
}

/// Re-parses `text` and returns the syntax errors. Empty `text` is treated
/// as a clean parse (renderer would have nothing to highlight anyway).
fn parse_program_errors(text: &str) -> Vec<crate::parser::Error> {
    use crate::parser;
    if text.is_empty() {
        Vec::new()
    } else {
        match parser::parse_program::<MarkId>(text) {
            Ok(_) => Vec::new(),
            Err(es) => es,
        }
    }
}

/// Refreshes `Mode::Edit.errors` from the active program's text. Called
/// after every keystroke in Edit mode so the renderer's per-character
/// syntax highlighting stays in sync. Leaves `cursor_position` and
/// `message` untouched.
fn refresh_edit_errors(state: &mut AppState) {
    let new_errors = parse_program_errors(&state.active_program().text);
    if let Mode::Edit { errors, .. } = &mut state.mode {
        *errors = new_errors;
    }
}

fn apply_insert_text(state: &mut AppState, text: &str) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    let program = &mut state.programs[state.active_program_index];
    let mut new_text = program.text.clone();
    new_text.insert_str(cursor, text);
    program.text = new_text;
    program.valid_keys_program.set(None);
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        *cursor_position = cursor + text.len();
    }
    refresh_edit_errors(state);
    vec![]
}

fn apply_delete_char(state: &mut AppState) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    if cursor == 0 {
        return vec![];
    }
    let program = &mut state.programs[state.active_program_index];
    let mut new_text = program.text.clone();
    new_text.remove(cursor - 1);
    program.text = new_text;
    program.valid_keys_program.set(None);
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        *cursor_position = cursor - 1;
    }
    refresh_edit_errors(state);
    vec![]
}

fn apply_delete_word(state: &mut AppState) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    if cursor == 0 {
        return vec![];
    }
    let program = &mut state.programs[state.active_program_index];
    let prefix = &program.text[..cursor];
    let new_cursor = match prefix.trim_end().rfind(char::is_whitespace) {
        Some(idx) => idx + 1,
        None => 0,
    };
    let mut new_text = program.text.clone();
    new_text.replace_range(new_cursor..cursor, "");
    program.text = new_text;
    program.valid_keys_program.set(None);
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        *cursor_position = new_cursor;
    }
    refresh_edit_errors(state);
    vec![]
}

fn apply_move_cursor_by(state: &mut AppState, delta: i32) -> Vec<Effect> {
    let len = state.active_program().text.len();
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        let new = (*cursor_position as i32 + delta).max(0).min(len as i32) as usize;
        *cursor_position = new;
    }
    vec![]
}

fn apply_move_cursor_to(state: &mut AppState, pos: usize) -> Vec<Effect> {
    let len = state.active_program().text.len();
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        *cursor_position = pos.min(len);
    }
    vec![]
}

fn apply_move_cursor_prev_word(state: &mut AppState) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    if cursor == 0 {
        return vec![];
    }
    let program = state.active_program();
    let prefix = &program.text[..cursor];
    let new = match prefix.trim_end().rfind(char::is_whitespace) {
        Some(idx) => idx + 1,
        None => 0,
    };
    if let Mode::Edit {
        cursor_position, ..
    } = &mut state.mode
    {
        *cursor_position = new;
    }
    vec![]
}

fn apply_mouse_slider(state: &mut AppState, axis: usize, delta: f32) -> Vec<Effect> {
    let program_index = state.active_program_index;
    let program = &state.programs[program_index];
    if axis >= program.sliders.configs.len() {
        return vec![];
    }
    let current = program.sliders.normalized_values[axis];
    let new = (current + delta).clamp(0.0, 1.0);
    apply_slider(state, program_index, axis, new)
}

fn apply_slider(
    state: &mut AppState,
    program_index: usize,
    slider_index: usize,
    normalized: f32,
) -> Vec<Effect> {
    let normalized = normalized.clamp(0.0, 1.0);
    let program = match state.programs.get_mut(program_index) {
        Some(p) => p,
        None => return vec![],
    };
    let program_id = program.id;
    let ps = &mut program.sliders;
    if slider_index >= ps.configs.len() {
        return vec![Effect::ShowMessage(format!(
            "No slider with index {}",
            slider_index
        ))];
    }
    ps.normalized_values[slider_index] = normalized;
    let config = &ps.configs[slider_index];
    let label = config.label.clone();
    let actual_value = slider::denormalize(&config.function, normalized).unwrap_or(0.0);

    let mut effects = vec![Effect::UpdateSlider {
        id: WaveformId::Program(program_id),
        slider: label.clone(),
        value: actual_value,
    }];

    // If the keys were installed from this program, also propagate the new
    // slider value to every active key waveform. The runner will look up
    // active Key marks from the tracker status when handling this effect.
    if let Some(keys) = state.keys.as_mut()
        && keys.id == program_id
    {
        keys.sliders.normalized_values[slider_index] = normalized;
        effects.push(Effect::UpdateActiveKeySliders {
            slider: label.clone(),
            value: actual_value,
        });
    }

    let formatted_value = renderer::format_sig_digits(actual_value, 3);

    // Refresh the controller's display for this encoder. In Plugin mode
    // the 8 encoders map 1:1 to the active program's sliders, so the
    // slider_index IS the encoder index. (Format matches sync_encoders.)
    effects.push(Effect::SetEncoderDisplay {
        index: slider_index as u8,
        name: label.clone(),
        value: formatted_value.clone(),
    });

    effects.push(Effect::ShowMessage(format!(
        "{}({}) = {}",
        label, slider_index, formatted_value,
    )));
    effects
}

fn apply_level_db(state: &mut AppState, program_index: usize, level_db: f32) -> Vec<Effect> {
    let program = match state.programs.get_mut(program_index) {
        Some(p) => p,
        None => return vec![],
    };
    program.level_db = level_db;
    let amplitude = play_helper::db_to_amplitude(level_db);
    let program_id = program.id;

    let mut effects = vec![Effect::ModifyWaveform {
        id: WaveformId::Program(program_id),
        mark_id: MarkId::Amplitude,
        waveform: waveform::Waveform::Const(amplitude),
    }];

    // Mirror onto installed keys.
    if let Some(keys) = state.keys.as_mut()
        && keys.id == program_id
    {
        keys.level_db = level_db;
        effects.push(Effect::ModifyActiveKeysAmplitude { amplitude });
    }

    // Bank-relative encoder index for the display update.
    let bank_start = program_index - (program_index % PROGRAMS_PER_BANK);
    let encoder_index = (program_index - bank_start) as u8;
    effects.push(Effect::SetEncoderDisplay {
        index: encoder_index,
        name: "level".to_string(),
        value: format!("{:.1} dB", level_db),
    });
    effects.push(Effect::ShowMessage(format!(
        "level({}) = {:.1} dB",
        program_id, level_db
    )));
    effects
}

#[cfg(test)]
mod tests {
    use super::*;
    use renderer::{Program, ProgramSliders};
    use std::cell;

    fn test_program() -> Program {
        Program {
            text: "test".to_string(),
            id: 1,
            sliders: ProgramSliders::default(),
            color: None,
            level_db: 0.0,
            valid_keys_program: cell::Cell::new(None),
        }
    }

    fn test_config() -> loader::Config {
        loader::Config {
            tempo: 90,
            sample_rate: 44100,
            context_files: vec![],
            programs_file: String::new(),
            additional_programs: vec![],
        }
    }

    fn test_state() -> AppState {
        AppState {
            programs: vec![test_program()],
            active_program_index: 0,
            mode: Mode::Select,
            keys: None,
            repeat_after_measures: None,
            daw_pad_mode: DawPadMode::ClipLauncher,
            context: vec![],
            config: test_config(),
            should_exit: false,
            message: String::new(),
        }
    }

    #[test]
    fn set_level_db_updates_state_and_emits_modify() {
        let mut state = test_state();
        let effects = apply(
            &mut state,
            Action::SetLevelDb {
                program: 0,
                level_db: -6.0,
            },
        );
        assert!((state.programs[0].level_db - -6.0).abs() < 1e-6);
        assert!(matches!(
            effects[0],
            Effect::ModifyWaveform {
                mark_id: MarkId::Amplitude,
                ..
            }
        ));
    }

    #[test]
    fn insert_text_refreshes_edit_mode_errors() {
        // Start in Edit mode with a clean program and a snapshot saying
        // "no errors" (as `Action::EnterEditMode` would have populated).
        // Typing a character that breaks the parse should be reflected
        // in `Mode::Edit.errors`, otherwise the renderer's per-character
        // highlighting goes stale.
        let mut state = AppState {
            programs: vec![Program {
                text: String::new(),
                id: 1,
                sliders: ProgramSliders::default(),
                color: None,
                level_db: 0.0,
                valid_keys_program: cell::Cell::new(None),
            }],
            active_program_index: 0,
            mode: Mode::Edit {
                cursor_position: 0,
                errors: vec![],
            },
            keys: None,
            repeat_after_measures: None,
            daw_pad_mode: DawPadMode::ClipLauncher,
            context: vec![],
            config: test_config(),
            should_exit: false,
            message: String::new(),
        };
        apply(&mut state, Action::InsertText("(".to_string()));
        let Mode::Edit { errors, .. } = &state.mode else {
            panic!("expected Edit mode after InsertText");
        };
        assert!(
            !errors.is_empty(),
            "expected parse errors after inserting an unbalanced '('"
        );
        // Now type the closing paren — errors should clear.
        apply(&mut state, Action::InsertText("1)".to_string()));
        let Mode::Edit { errors, .. } = &state.mode else {
            panic!("expected Edit mode after second InsertText");
        };
        assert!(
            errors.is_empty(),
            "expected no errors once parens are balanced, got {:?}",
            errors
        );
    }

    #[test]
    fn advance_program_emits_empty_show_message_to_clear_status() {
        // Two programs so AdvanceProgram(1) actually moves the index.
        let mut state = test_state();
        state.programs.push(Program {
            text: "second".to_string(),
            id: 2,
            sliders: ProgramSliders::default(),
            color: None,
            level_db: 0.0,
            valid_keys_program: cell::Cell::new(None),
        });
        let effects = apply(&mut state, Action::AdvanceProgram(1));
        assert_eq!(state.active_program_index, 1);
        // The empty ShowMessage is what gets folded into Mode::Select to
        // wipe any stale "Removed pending..." / "Playing..." text.
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ShowMessage(s) if s.is_empty())),
            "expected an empty ShowMessage to clear the status, got {:?}",
            effects
        );
    }

    #[test]
    fn set_encoder_mode_emits_set_launchkey_encoder_mode() {
        let mut state = test_state();
        let effects = apply(
            &mut state,
            Action::SetEncoderMode(launchkey::EncoderMode::Mixer),
        );
        // Reducer no longer tracks the mode (it now lives on Launchkey);
        // it just emits an effect that the runner uses to update the
        // controller cache and re-sync the encoders.
        assert!(matches!(
            effects[0],
            Effect::SetLaunchkeyEncoderMode(launchkey::EncoderMode::Mixer)
        ));
    }
}
