//! Action / Effect / reducer for app state.
//!
//! Handlers (sdl2_input, midi_input) classify raw input events into pure
//! `Action`s. `apply` then mutates `AppState` and returns `Effect`s, which
//! the runner in `effects.rs` executes against the outside world.

use std::collections::HashSet;
use std::time::Instant;

use crate::diagnostics::{self, Diagnostic, Source};
use crate::evaluator;
use crate::expr;
use crate::ids::{MarkId, WaveformId};
use crate::keys;
use crate::launchkey;
use crate::player;
use crate::programs::{self, PROGRAMS_PER_BANK, Program};
use crate::tracker;
use crate::waveform;

/// The app's interaction mode, driving which inputs the classifiers
/// accept and how the renderer draws the active program.
#[derive(Debug, Clone)]
pub enum Mode {
    Select,
    Edit {
        /// Byte offset into the program text; the cursor sits before the
        /// character starting here. Always lies on a `char` boundary — every
        /// cursor op moves over whole characters (see `prev_char_boundary` /
        /// `next_char_boundary`).
        cursor_position: usize,
        /// Diagnostics for the current text — parse errors while typing,
        /// or evaluation errors after a failed evaluate. The renderer
        /// highlights their `program_range`s.
        errors: Vec<Diagnostic>,
        /// In-progress `Action::Complete` cycle, if the last action was a
        /// completion. Cleared by every text or cursor change, so a stored
        /// cycle always describes the current text.
        completion: Option<Completion>,
    },
    MoveSliders,
    /// Computer-keyboard piano: lower QWERTY row plays white keys, row above
    /// plays sharps. Only reachable when `state.keys` is installed.
    Keys,
}

/// The state of an identifier-completion cycle (`Action::Complete`).
///
/// The text from `start` to the cursor holds the ring entry inserted by the
/// previous completion; the next `Action::Complete` replaces it with the
/// following entry. The ring is `candidates` followed by `original`, so cycling
/// past the last candidate restores what the user typed.
#[derive(Debug, Clone)]
pub struct Completion {
    /// Byte offset where the completed identifier starts.
    start: usize,
    /// The identifier fragment the user typed, restored when the cycle wraps.
    original: String,
    /// In-scope names starting with `original`, most recently bound first.
    candidates: Vec<String>,
    /// Ring index of the entry the next completion inserts;
    /// `candidates.len()` denotes `original`.
    next: usize,
}

/// Behavior of the pads when the controller is in DAW pad mode. Cycled by
/// `Action::PadModeChanged` when the user re-presses the DAW pad-mode
/// button while already in DAW mode, repurposing the pads.
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
    /// The programs and the source file backing them.
    pub programs: programs::ProgramSet,
    pub active_program_index: usize,
    pub mode: Mode,
    pub keys: Option<keys::Keys>,
    pub repeat_after_measures: Option<u32>,
    /// Active behavior for the DAW pad.
    pub daw_pad_mode: DawPadMode,
    /// Set by `Effect::Exit`; `main` checks at top of loop and breaks.
    pub should_exit: bool,
    /// Last user-visible status message. Set by `Effect::ShowMessage` (and
    /// a few direct writes from the reducer / runner). Persists across
    /// mode transitions; cleared explicitly by navigation actions.
    ///
    /// May be multi-line (e.g. an error followed by its source snippet):
    /// the first line is the status-line summary, and single-line display
    /// sites should show only that line.
    pub message: String,
}

impl AppState {
    /// Builds an `AppState` from the contents of a source file. `input_path`
    /// is the file the splice path writes back to (use an empty `PathBuf`
    /// to suppress the write, e.g. in tests).
    pub fn from_source(
        source: String,
        input_path: std::path::PathBuf,
    ) -> Result<AppState, Vec<expr::Error<Source>>> {
        let (programs, message) = programs::ProgramSet::from_source(source, input_path)?;
        Ok(AppState {
            programs,
            active_program_index: 0,
            mode: Mode::Select,
            keys: None,
            repeat_after_measures: None,
            daw_pad_mode: DawPadMode::ClipLauncher,
            should_exit: false,
            message,
        })
    }

    pub fn bank_start(&self) -> usize {
        self.active_program_index - (self.active_program_index % PROGRAMS_PER_BANK)
    }

    pub fn active_program(&self) -> &Program {
        &self.programs.programs()[self.active_program_index]
    }
}

/// Read-only snapshot of the world used by the reducer.
pub struct Context<'a> {
    pub status: &'a tracker::Status<WaveformId, MarkId>,
    pub now: Instant,
    /// Evaluation environment (prelude, module cache) — used by
    /// `Action::Complete` to find the names in scope.
    pub evaluator: &'a evaluator::Evaluator,
}

/// Things that can happen, emitted by handlers as pure data.
#[derive(Debug)]
pub enum Action {
    // --- playback ---
    /// Play the program at `program_index` if it evaluates to a waveform.
    /// Otherwise, do nothing.
    ///
    /// If `start_at_next_measure` is false, then play immediately.
    PlayProgram {
        program_index: usize,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
    },
    /// Immediately stop playback of the program waveform if it's playing.
    /// Otherwise, do nothing.
    StopProgram(usize),
    /// Remove any pending playback for the given program. Does nothing if no
    /// playback is pending.
    RemovePendingProgram(usize),
    /// Stop the program if it's playing; otherwise play it immediately. Does
    /// nothing if the program isn't a waveform.
    ToggleProgramPlayback(usize),
    /// Remove the program's pending playback if there is one; otherwise queue
    /// it to play at the beginning of next measure (repeating per the app-wide
    /// default). Does nothing if the program isn't a waveform.
    ToggleProgramPendingPlayback(usize),

    // --- MIDI keys ---
    /// Install the program at the given index as the keys instrument. If the
    /// program at the given index is already installed, uninstall it.
    ToggleInstalledKeys(usize),
    NoteOn {
        key: u8,
        velocity: u8,
    },
    NoteOff {
        key: u8,
    },

    // --- mode transitions ---
    /// Enter Edit mode on the active program, first removing any pending
    /// playback for it.
    EnterEditMode,
    /// Parse and evaluate the current program, updating its state and the
    /// source file. On success, return to Select mode; otherwise set the mode
    /// to `mode_on_failure`.
    EvaluateAndLeaveEditMode {
        mode_on_failure: Mode,
    },
    EnterSelectMode,
    EnterMoveSlidersMode,
    /// Enter computer-keyboard piano mode. Does nothing if a keys instrument is
    /// not installed.
    EnterKeysMode,

    // --- navigation ---
    /// Jump to a specific program index.
    SelectProgram(usize),
    /// Move the active index by `delta` (wraps).
    AdvanceProgram(i32),

    // --- text editing (Edit mode) ---
    InsertText(String),
    DeleteCharBeforeCursor,
    DeleteCharAfterCursor,
    DeleteWordBeforeCursor,
    DeleteWordAfterCursor,
    /// Delete from the cursor to the end of the line, or the newline itself
    /// when the cursor is already at the end of a line (emacs `kill-line`).
    DeleteToEndOfLine,
    MoveCursorBy(i32),
    MoveCursorToStart,
    MoveCursorToEnd,
    MoveCursorToPreviousWord,
    MoveCursorToNextWord,
    /// Program completion at the cursor.
    ///
    /// With an identifier fragment before the cursor, cycles it through the
    /// in-scope names that share the prefix, most recently bound first,
    /// wrapping back to the fragment itself. Right after a `(`, inserts a
    /// parameter hint for the function being called instead, leaving the
    /// cursor on the first parameter.
    Complete,
    /// Restore the active program's text and cursor from before the most recent
    /// edit unit, remembering the current state for `Redo`.
    Undo,
    /// Restore the active program's text and cursor from the most recently
    /// undone edit unit.
    Redo,

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
    ///
    /// Entering DAW mode announces the current DAW-pad sub-mode on the
    /// controller display and in the status message; re-selecting DAW while
    /// already in it cycles the sub-mode.
    PadModeChanged {
        previous: launchkey::PadMode,
        current: launchkey::PadMode,
    },
    /// Toggle the default repeat (cycle None -> Some(1) -> Some(2) -> None).
    CycleRepeatAfterMeasures,

    // --- program-related I/O and other effects ---
    ShowMessage(String),
    DumpActiveWaveform,
    Exit,
}

/// Side effects to perform after the reducer runs. The runner translates
/// these into MPSC sends, MIDI controller updates, etc.
#[derive(Debug)]
pub enum Effect {
    // --- tracker commands ---
    /// Send a Play command for the program at `program_index`.
    PlayProgram {
        program_index: usize,
        start_at_next_measure: bool,
        repeat_after_measures: Option<u32>,
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

    /// Parse and evaluate the current program, updating its state. On success,
    /// return to Select mode; otherwise set the mode to `mode_on_failure`.
    EvaluateProgram {
        program_index: usize,
        mode_on_failure: Mode,
    },

    /// Update the source file with the current text of the given program, then
    /// re-parse and refresh all bindings and program state.
    UpdateSource(usize),

    /// Interpret the program at the given index as a keys instrument function
    /// and install it as `state.keys`. If that program is already installed,
    /// uninstall it.
    InstallKeys(usize),

    /// Play the note-on waveform produced by the installed keys function.
    PlayNoteOn { key: u8, velocity: u8 },
    /// Modify the Level mark on the active key waveform with the note-off
    /// waveform that was stored at NoteOn time. Falls back to a stop-ramp if
    /// no waveform was stored.
    PlayNoteOff { key: u8 },

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
    ModifyActiveKeysAmplitude { amplitude: f32 },

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
    /// User-visible status message.
    ShowMessage(String),
    /// Print the waveform of the active program.
    DumpActiveWaveform,
    /// Sets `state.should_exit = true`.
    Exit,
}

/// Applies an action to state, returning effects for the runner to execute.
///
/// Performs only the state mutation that needs no I/O, reading nothing beyond
/// `state` and `ctx`. Effects whose outcome depends on I/O (evaluating a
/// program, splicing source, playing notes) mutate state in the runner instead.
pub fn apply(state: &mut AppState, ctx: &Context, action: Action) -> Vec<Effect> {
    match action {
        Action::PlayProgram {
            program_index,
            start_at_next_measure,
            repeat_after_measures,
        } => play_program_effects(program_index, start_at_next_measure, repeat_after_measures),
        Action::StopProgram(i) => stop_program_effects(state, ctx, i),
        Action::RemovePendingProgram(i) => remove_pending_effects(state, ctx, i),
        Action::ToggleProgramPlayback(i) => {
            if ctx
                .status
                .has_active_mark(ctx.now, WaveformId::Program(i), MarkId::TopLevel)
            {
                stop_program_effects(state, ctx, i)
            } else if state.keys.as_ref().is_some_and(|k| k.id == i) {
                vec![]
            } else {
                play_program_effects(i, false, None)
            }
        }
        Action::ToggleProgramPendingPlayback(i) => {
            if ctx
                .status
                .has_pending_mark(ctx.now, WaveformId::Program(i), MarkId::TopLevel)
            {
                remove_pending_effects(state, ctx, i)
            } else if state.keys.as_ref().is_some_and(|k| k.id == i) {
                vec![]
            } else {
                play_program_effects(i, true, state.repeat_after_measures)
            }
        }

        Action::ToggleInstalledKeys(i) => apply_install_keys(state, i),
        Action::NoteOn { key, velocity } => {
            if state.keys.is_some() {
                vec![Effect::PlayNoteOn { key, velocity }]
            } else {
                vec![]
            }
        }
        Action::NoteOff { key } => vec![Effect::PlayNoteOff { key }],

        Action::EnterEditMode => {
            // Editing a program whose playback is still queued would be
            // confusing (the stale waveform would start mid-edit), so cancel
            // any pending playback on the way in.
            let effects = remove_pending_effects(state, ctx, state.active_program_index);
            // Re-entering edit starts fresh typing: the first keystroke must
            // open a new undo unit, not coalesce with pre-exit typing.
            if let Some(program) = state.programs.program_mut(state.active_program_index) {
                program.close_insert_run();
            }
            let program = state.active_program();
            let cursor = program.text().len();
            let errors = parse_program_errors(program.text());
            state.message = if !errors.is_empty() {
                diagnostics::error_message(&errors)
            } else if !program.sliders().configs().is_empty() {
                program
                    .sliders()
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
                completion: None,
            };
            effects
        }
        Action::EvaluateAndLeaveEditMode { mode_on_failure } => {
            vec![
                Effect::EvaluateProgram {
                    program_index: state.active_program_index,
                    mode_on_failure,
                },
                Effect::UpdateSource(state.active_program_index),
            ]
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
            if state.keys.is_none() {
                return vec![Effect::ShowMessage(
                    "No keys instrument installed".to_string(),
                )];
            }
            state.mode = Mode::Keys;
            vec![Effect::ShowMessage("Piano keys enabled".to_string())]
        }

        Action::SelectProgram(i) => apply_select_program(state, i),
        Action::AdvanceProgram(delta) => {
            let len = state.programs.programs().len() as i32;
            if len == 0 {
                vec![]
            } else {
                let cur = state.active_program_index as i32;
                let new = ((cur + delta) % len + len) % len;
                apply_select_program(state, new as usize)
            }
        }

        Action::InsertText(text) => {
            edit_text_op(state, HistoryOp::Insert(&text), |current, cursor| {
                let mut new_text = current.to_string();
                new_text.insert_str(cursor, &text);
                Some((new_text, cursor + text.len()))
            })
        }
        Action::DeleteCharBeforeCursor => {
            edit_text_op(state, HistoryOp::Unit, |current, cursor| {
                if cursor == 0 {
                    return None;
                }
                let start = prev_char_boundary(current, cursor);
                let mut new_text = current.to_string();
                new_text.replace_range(start..cursor, "");
                Some((new_text, start))
            })
        }
        Action::DeleteCharAfterCursor => edit_text_op(state, HistoryOp::Unit, |current, cursor| {
            if cursor == current.len() {
                return None;
            }
            let end = next_char_boundary(current, cursor);
            let mut new_text = current.to_string();
            new_text.replace_range(cursor..end, "");
            Some((new_text, cursor))
        }),
        Action::DeleteWordBeforeCursor => {
            edit_text_op(state, HistoryOp::Unit, |current, cursor| {
                if cursor == 0 {
                    return None;
                }
                let new_cursor = prev_word_start(&current[..cursor]);
                let mut new_text = current.to_string();
                new_text.replace_range(new_cursor..cursor, "");
                Some((new_text, new_cursor))
            })
        }
        Action::DeleteWordAfterCursor => edit_text_op(state, HistoryOp::Unit, |current, cursor| {
            if cursor == current.len() {
                return None;
            }
            let end = cursor + next_word_end(&current[cursor..]);
            let mut new_text = current.to_string();
            new_text.replace_range(cursor..end, "");
            Some((new_text, cursor))
        }),
        Action::DeleteToEndOfLine => edit_text_op(state, HistoryOp::Unit, |current, cursor| {
            if cursor == current.len() {
                return None;
            }
            let end = match current[cursor..].find('\n') {
                // At the end of a line: delete the newline itself, joining
                // the next line onto this one.
                Some(0) => cursor + 1,
                Some(i) => cursor + i,
                None => current.len(),
            };
            let mut new_text = current.to_string();
            new_text.replace_range(cursor..end, "");
            Some((new_text, cursor))
        }),
        Action::MoveCursorBy(delta) => edit_cursor_op(state, |current, cursor| {
            let mut cursor = cursor;
            for _ in 0..delta.unsigned_abs() {
                cursor = if delta < 0 {
                    prev_char_boundary(current, cursor)
                } else {
                    next_char_boundary(current, cursor)
                };
            }
            cursor
        }),
        Action::MoveCursorToStart => edit_cursor_op(state, |_, _| 0),
        Action::MoveCursorToEnd => edit_cursor_op(state, |current, _| current.len()),
        Action::MoveCursorToPreviousWord => edit_cursor_op(state, |current, cursor| {
            if cursor == 0 {
                0
            } else {
                prev_word_start(&current[..cursor])
            }
        }),
        Action::MoveCursorToNextWord => edit_cursor_op(state, |current, cursor| {
            cursor + next_word_end(&current[cursor..])
        }),
        Action::Complete => apply_complete(state, ctx),
        Action::Undo => apply_history_restore(state, Program::undo, "Nothing to undo"),
        Action::Redo => apply_history_restore(state, Program::redo, "Nothing to redo"),

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
        Action::PadModeChanged { previous, current } => {
            let mut effects = vec![Effect::SetLaunchkeyPadMode(current)];
            if current == launchkey::PadMode::DAW {
                if previous == launchkey::PadMode::DAW {
                    state.daw_pad_mode = match state.daw_pad_mode {
                        DawPadMode::ClipLauncher => DawPadMode::KeysInstaller,
                        DawPadMode::KeysInstaller => DawPadMode::ClipLauncher,
                    };
                }
                let label = state.daw_pad_mode.display_name().to_string();
                effects.push(Effect::SetDawModeDisplay(label.clone()));
                effects.push(Effect::ShowMessage(label));
            }
            effects
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

        Action::ShowMessage(message) => vec![Effect::ShowMessage(message)],
        Action::DumpActiveWaveform => vec![Effect::DumpActiveWaveform],
        Action::Exit => vec![
            Effect::UpdateSource(state.active_program_index),
            Effect::Exit,
        ],
    }
}

/// Returns the effects that play the given program and persist its source.
fn play_program_effects(
    program_index: usize,
    start_at_next_measure: bool,
    repeat_after_measures: Option<u32>,
) -> Vec<Effect> {
    vec![
        Effect::PlayProgram {
            program_index,
            start_at_next_measure,
            repeat_after_measures,
        },
        Effect::UpdateSource(program_index),
    ]
}

/// Returns the effects that stop the given program, or nothing if it isn't
/// currently playing.
fn stop_program_effects(state: &AppState, ctx: &Context, i: usize) -> Vec<Effect> {
    if !ctx
        .status
        .has_active_mark(ctx.now, WaveformId::Program(i), MarkId::TopLevel)
    {
        return vec![];
    }
    vec![
        Effect::StopProgram(i),
        Effect::ShowMessage(format!(
            "Stopped program {}",
            state.programs.display_name(i)
        )),
    ]
}

/// Returns the effects that remove the given program's pending playback, or
/// nothing if no playback is pending.
fn remove_pending_effects(state: &AppState, ctx: &Context, i: usize) -> Vec<Effect> {
    if !ctx
        .status
        .has_pending_mark(ctx.now, WaveformId::Program(i), MarkId::TopLevel)
    {
        return vec![];
    }
    vec![
        Effect::RemovePendingProgram(i),
        Effect::ShowMessage(format!(
            "Removed pending waveform for program {}",
            state.programs.display_name(i)
        )),
    ]
}

fn apply_select_program(state: &mut AppState, i: usize) -> Vec<Effect> {
    if i >= state.programs.programs().len() {
        return vec![];
    }
    let changed = state.active_program_index != i;
    state.active_program_index = i;
    // Replace the previous status with the newly-selected program's name.
    // Navigation represents a fresh context, so any prior
    // "Removed pending..." / "Playing..." etc. shouldn't carry over.
    let mut effects = vec![Effect::ShowMessage(state.programs.name(i))];
    if changed {
        effects.push(Effect::SyncEncoders);
    }
    effects
}

fn apply_install_keys(state: &mut AppState, program_index: usize) -> Vec<Effect> {
    // Applying with the currently-installed program uninstalls it; for any
    // other program, tries to install that one in its place.
    if let Some(keys) = &state.keys
        && keys.id == program_index
    {
        state.keys = None;
        return vec![Effect::ShowMessage("Uninstalled keys".to_string())];
    }
    vec![Effect::InstallKeys(program_index)]
}

/// Re-parses `text` and returns its syntax errors as diagnostics. Empty
/// `text` is treated as a clean parse (renderer would have nothing to
/// highlight anyway).
fn parse_program_errors(text: &str) -> Vec<Diagnostic> {
    use crate::parser;
    // Empty (or whitespace-only) text is a pending deletion, not a parse error.
    if text.trim().is_empty() {
        Vec::new()
    } else {
        match parser::parse_program::<MarkId, _>(text, Source::Program) {
            Ok(_) => Vec::new(),
            Err(es) => es
                .into_iter()
                .map(|e| match e.range() {
                    Some(range) => Diagnostic::in_program(e.message().to_string(), range, text),
                    None => Diagnostic::message_only(e.message().to_string()),
                })
                .collect(),
        }
    }
}

/// Applies `Action::Complete` in Edit mode; does nothing in other modes.
///
/// Continues an in-progress cycle when one is stored; otherwise dispatches on
/// the text before the cursor: an identifier fragment starts a completion cycle
/// or a `(` inserts a parameter hint. See [`Action::Complete`] for the
/// user-facing behavior.
fn apply_complete(state: &mut AppState, ctx: &Context) -> Vec<Effect> {
    let Mode::Edit {
        cursor_position,
        completion,
        ..
    } = &state.mode
    else {
        return vec![];
    };
    let cursor = *cursor_position;

    // Continue a cycle: replace the previous insertion with the next ring
    // entry. The text from `start` to the cursor still holds the previous
    // insertion — every other text or cursor action clears `completion`.
    if let Some(cycle) = completion.clone() {
        let replacement = cycle
            .candidates
            .get(cycle.next)
            .unwrap_or(&cycle.original)
            .clone();
        let start = cycle.start;
        // Skip history: the whole cycle is one undo unit, recorded when the
        // cycle's first press replaced the fragment.
        edit_text_op(state, HistoryOp::Skip, |current, cursor| {
            let text = format!("{}{}{}", &current[..start], replacement, &current[cursor..]);
            Some((text, start + replacement.len()))
        });
        if let Mode::Edit { completion, .. } = &mut state.mode {
            *completion = Some(Completion {
                next: (cycle.next + 1) % (cycle.candidates.len() + 1),
                ..cycle
            });
        }
        return vec![];
    }

    let text = state.active_program().text();
    let before = &text[..cursor];
    let fragment_start = before.trim_end_matches(is_word_char).len();
    if fragment_start == cursor {
        if before.ends_with('(') {
            return apply_parameter_hint(state, ctx, cursor);
        }
        return vec![Effect::ShowMessage(
            "Nothing to complete (the cursor must follow an identifier or \"(\")".to_string(),
        )];
    }

    // An identifier fragment ends at the cursor: start a new cycle.
    let fragment = before[fragment_start..].to_string();
    let context = match ctx
        .evaluator
        .program_context(&state.programs, state.active_program_index)
    {
        Ok(context) => context,
        Err(error) => return vec![Effect::ShowMessage(format!("Can't complete: {}", error))],
    };
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();
    for (name, _) in context.iter().rev() {
        // Walking from the end, the first occurrence of a name is the live
        // binding; any earlier occurrence is shadowed.
        if seen.insert(name.clone()) && name.starts_with(&fragment) && *name != fragment {
            candidates.push(name.clone());
        }
    }
    if candidates.is_empty() {
        return vec![Effect::ShowMessage(format!(
            "No completions for \"{}\"",
            fragment
        ))];
    }

    let replacement = candidates[0].clone();
    edit_text_op(state, HistoryOp::Unit, |current, cursor| {
        let text = format!(
            "{}{}{}",
            &current[..fragment_start],
            replacement,
            &current[cursor..]
        );
        Some((text, fragment_start + replacement.len()))
    });
    if let Mode::Edit { completion, .. } = &mut state.mode {
        *completion = Some(Completion {
            start: fragment_start,
            original: fragment,
            candidates,
            next: 1,
        });
    }
    vec![]
}

/// Inserts a parameter hint for the call being typed: with the Edit-mode
/// `cursor` just after `(` and the identifier before it bound to a function,
/// inserts that function's positional parameters, its named parameters as
/// `name = <default value>` pairs, and a closing `)`.
///
/// The hint previews the call's shape rather than typing it for the user:
/// the cursor lands after the first parameter, ready to replace the
/// positional placeholders with real arguments (or after the `)` when the
/// function takes none).
fn apply_parameter_hint(state: &mut AppState, ctx: &Context, cursor: usize) -> Vec<Effect> {
    let text = state.active_program().text();
    let head = &text[..cursor - 1];
    let name_start = head.trim_end_matches(is_word_char).len();
    let name = head[name_start..].to_string();
    if name.is_empty() {
        return vec![Effect::ShowMessage("Nothing to complete".to_string())];
    }
    let context = match ctx
        .evaluator
        .program_context(&state.programs, state.active_program_index)
    {
        Ok(context) => context,
        Err(error) => return vec![Effect::ShowMessage(format!("Can't complete: {}", error))],
    };
    // The last occurrence of the name is the live binding.
    let Some((_, value)) = context.iter().rev().find(|(n, _)| *n == name) else {
        return vec![Effect::ShowMessage(format!("\"{}\" is not defined", name))];
    };
    match &value.expr {
        expr::Expr::Function {
            positional, named, ..
        } => {
            // Named defaults were evaluated when the function was defined,
            // so they print as values even when the source wrote an
            // expression (`y = 10 + 1` hints as `y = 11`).
            let mut parts: Vec<String> = positional.iter().map(|p| p.to_string()).collect();
            parts.extend(named.iter().map(|(n, v)| format!("{} = {}", n, v)));
            let hint = format!("{})", parts.join(", "));
            // Land the cursor after the first parameter, ready for a
            // delete-word to kill the placeholder and type the real
            // argument; with no parameters the call is complete, so land
            // after the ")".
            let advance = parts.first().map_or(hint.len(), |p| p.len());
            edit_text_op(state, HistoryOp::Unit, |current, cursor| {
                let mut new_text = current.to_string();
                new_text.insert_str(cursor, &hint);
                Some((new_text, cursor + advance))
            })
        }
        // TODO built-ins don't carry parameter names or arity; give BuiltIn
        // signature metadata so calls like `sine(` can be hinted too.
        expr::Expr::BuiltIn { name, .. } => vec![Effect::ShowMessage(format!(
            "No parameter hint for built-in \"{}\"",
            name
        ))],
        _ => vec![Effect::ShowMessage(format!(
            "\"{}\" is not a function",
            name
        ))],
    }
}

/// Applies `Action::Undo` or `Action::Redo` in Edit mode: `restore` moves
/// the active program's state between its undo and redo stacks and returns
/// the restored cursor. On success the Edit-mode cursor, parse errors,
/// status message, and completion cycle are refreshed, mirroring
/// `edit_text_op`; when there is nothing to restore, shows `empty_message`.
/// Does nothing outside Edit mode.
fn apply_history_restore(
    state: &mut AppState,
    restore: impl FnOnce(&mut Program, usize) -> Option<usize>,
    empty_message: &str,
) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    let program = state
        .programs
        .program_mut(state.active_program_index)
        .unwrap();
    let Some(new_cursor) = restore(program, cursor) else {
        return vec![Effect::ShowMessage(empty_message.to_string())];
    };
    if let Mode::Edit {
        cursor_position,
        completion,
        ..
    } = &mut state.mode
    {
        *cursor_position = new_cursor;
        *completion = None;
    }
    refresh_edit_errors(state);
    state.message.clear();
    vec![]
}

/// Refreshes `Mode::Edit.errors` from the active program's text. Called
/// after every keystroke in Edit mode so the renderer's per-character
/// syntax highlighting stays in sync. Leaves `cursor_position` and
/// `message` untouched.
fn refresh_edit_errors(state: &mut AppState) {
    let new_errors = parse_program_errors(state.active_program().text());
    if let Mode::Edit { errors, .. } = &mut state.mode {
        *errors = new_errors;
    }
}

/// How a text edit participates in the active program's undo history.
enum HistoryOp<'a> {
    /// Typed text: extends an open insert run, starting a new undo unit only at
    /// a word boundary.
    Insert(&'a str),
    /// A standalone edit that always forms its own undo unit and closes any
    /// open insert run.
    Unit,
    /// An edit that must leave history untouched (e.g. one that itself restores
    /// or replays a recorded state).
    Skip,
}

/// Applies a text edit to the active program's Edit-mode text.
///
/// Calls `f` with the current text and cursor position; when `f` returns a new
/// (text, cursor) pair, records the pre-edit state in the program's undo
/// history per `history`, writes both back, refreshes the Edit-mode parse
/// errors, and clears the status message and any completion cycle (both
/// describe text that just changed). Does nothing outside Edit mode or when `f`
/// returns `None` — except that a `HistoryOp::Unit` edit still closes any open
/// insert run, so e.g. a no-op delete stops the next keystroke from coalescing
/// with earlier typing.
fn edit_text_op(
    state: &mut AppState,
    history: HistoryOp,
    f: impl FnOnce(&str, usize) -> Option<(String, usize)>,
) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    let program = state
        .programs
        .program_mut(state.active_program_index)
        .unwrap();
    if let Some((new_text, new_cursor)) = f(program.text(), cursor) {
        // Record before set_text: the snapshot is the pre-edit text.
        match history {
            HistoryOp::Insert(text) => {
                if let (Some(first), Some(last)) = (text.chars().next(), text.chars().next_back()) {
                    // A new undo unit starts at each word start (a word char
                    // typed right after a non-word char); anything else extends
                    // the open run.
                    let new_unit = match program.last_inserted() {
                        None => true,
                        Some(prev) => is_word_char(first) && !is_word_char(prev),
                    };
                    program.record_insert(new_unit, last, cursor);
                }
            }
            HistoryOp::Unit => program.record_edit(cursor),
            HistoryOp::Skip => {}
        }
        program.set_text(new_text);
        if let Mode::Edit {
            cursor_position,
            completion,
            ..
        } = &mut state.mode
        {
            *cursor_position = new_cursor;
            *completion = None;
        }
        refresh_edit_errors(state);
        state.message.clear();
    } else if let HistoryOp::Unit = history {
        program.close_insert_run();
    }
    vec![]
}

/// Moves the Edit-mode cursor: `f` maps (text, cursor) to the new position,
/// which is clamped to the text's length. Clears any completion cycle (the
/// cycle's insertion ends at the cursor, so moving it breaks the cycle) and
/// closes any insert-coalescing run (typing resumed elsewhere is a new undo
/// unit). Does nothing outside Edit mode.
fn edit_cursor_op(state: &mut AppState, f: impl FnOnce(&str, usize) -> usize) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        _ => return vec![],
    };
    let text = state.programs.programs()[state.active_program_index].text();
    let new_cursor = f(text, cursor).min(text.len());
    if let Some(program) = state.programs.program_mut(state.active_program_index) {
        program.close_insert_run();
    }
    if let Mode::Edit {
        cursor_position,
        completion,
        ..
    } = &mut state.mode
    {
        *cursor_position = new_cursor;
        *completion = None;
    }
    vec![]
}

/// Returns the byte offset of the start of the character before `cursor`, or 0
/// when the cursor is at the start of `text`.
///
/// `cursor` must lie on a char boundary of `text`.
fn prev_char_boundary(text: &str, cursor: usize) -> usize {
    text[..cursor]
        .chars()
        .next_back()
        .map_or(0, |c| cursor - c.len_utf8())
}

/// Returns the byte offset just past the character at `cursor`, or `text.len()`
/// when the cursor is already at the end of `text`.
///
/// `cursor` must lie on a char boundary of `text`.
fn next_char_boundary(text: &str, cursor: usize) -> usize {
    text[cursor..]
        .chars()
        .next()
        .map_or(text.len(), |c| cursor + c.len_utf8())
}

/// Returns true if `c` can appear in a word for cursor word operations.
///
/// Word characters are the characters that can appear in identifiers
/// (alphanumerics, `_`, and `#` — as in the note name `c#4`); everything
/// else (whitespace, operators, punctuation) separates words.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '#'
}

/// Returns the byte offset where the word preceding the end of `prefix`
/// starts: skips any trailing non-word characters, then scans back over one
/// run of word characters (emacs `backward-word`). Returns 0 when `prefix`
/// contains no word characters.
fn prev_word_start(prefix: &str) -> usize {
    prefix
        .trim_end_matches(|c: char| !is_word_char(c))
        .trim_end_matches(is_word_char)
        .len()
}

/// Returns the byte offset just past the word at the start of `suffix`:
/// skips any leading non-word characters, then scans over one run of word
/// characters (emacs `forward-word`). Returns `suffix.len()` when `suffix`
/// contains no word characters.
fn next_word_end(suffix: &str) -> usize {
    let rest = suffix
        .trim_start_matches(|c: char| !is_word_char(c))
        .trim_start_matches(is_word_char);
    suffix.len() - rest.len()
}

fn apply_mouse_slider(state: &mut AppState, axis: usize, delta: f32) -> Vec<Effect> {
    let program_index = state.active_program_index;
    let program = &state.programs.programs()[program_index];
    if axis >= program.sliders().configs().len() {
        return vec![];
    }
    let current = program.sliders().normalized_values()[axis];
    let new = (current + delta).clamp(0.0, 1.0);
    apply_slider(state, program_index, axis, new)
}

fn apply_slider(
    state: &mut AppState,
    program_index: usize,
    slider_index: usize,
    normalized: f32,
) -> Vec<Effect> {
    let program = match state.programs.program_mut(program_index) {
        Some(p) => p,
        None => return vec![],
    };
    let Some(change) = program.set_slider_normalized(slider_index, normalized) else {
        return vec![Effect::ShowMessage(format!(
            "No slider with index {}",
            slider_index
        ))];
    };
    let label = change.label;
    let actual_value = change.value;

    let mut effects = vec![Effect::UpdateSlider {
        id: WaveformId::Program(program_index),
        slider: label.clone(),
        value: actual_value,
    }];

    // If the keys were installed from this program, also propagate the new
    // slider value to every active key waveform. The runner will look up
    // active Key marks from the tracker status when handling this effect.
    if let Some(keys) = state.keys.as_mut()
        && keys.id == program_index
    {
        effects.push(Effect::UpdateActiveKeySliders {
            slider: label.clone(),
            value: actual_value,
        });
    }

    let formatted_value = programs::format_sig_digits(actual_value, 3);

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
    let program = match state.programs.program_mut(program_index) {
        Some(p) => p,
        None => return vec![],
    };
    program.set_level_db(level_db);
    let amplitude = player::db_to_amplitude(level_db);

    let mut effects = vec![Effect::ModifyWaveform {
        id: WaveformId::Program(program_index),
        mark_id: MarkId::Amplitude,
        waveform: waveform::Waveform::Const(amplitude),
    }];

    // Mirror onto installed keys.
    if let Some(keys) = state.keys.as_mut()
        && keys.id == program_index
    {
        effects.push(Effect::ModifyActiveKeysAmplitude { amplitude });
    }

    // Bank-relative encoder index for the display update.
    let bank_start = program_index - (program_index % PROGRAMS_PER_BANK);
    let encoder_index = (program_index - bank_start) as u8;
    let formatted_level = programs::format_level_db(level_db);
    effects.push(Effect::SetEncoderDisplay {
        index: encoder_index,
        name: "level".to_string(),
        value: formatted_level.clone(),
    });
    effects.push(Effect::ShowMessage(format!(
        "level({}) = {}",
        state.programs.display_name(program_index),
        formatted_level
    )));
    effects
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// A tracker status with no marks at all — nothing active, nothing pending.
    fn empty_status() -> tracker::Status<WaveformId, MarkId> {
        tracker::Status {
            buffer_start: Instant::now(),
            marks: vec![],
            buffer: None,
            tracker_load: None,
            allocations_per_sample: None,
        }
    }

    /// A status with a single TopLevel mark for program 0 starting at `start`
    /// (before `now` = active, after `now` = pending).
    fn status_with_mark(start: Instant) -> tracker::Status<WaveformId, MarkId> {
        let mut status = empty_status();
        status.marks.push(tracker::Mark {
            waveform_id: WaveformId::Program(0),
            mark_id: MarkId::TopLevel,
            start,
            duration: Duration::from_secs(1),
        });
        status
    }

    /// Applies `action` against `status`, timestamped `now`. The evaluator
    /// has the standard prelude but no library root, so `open`s of anything
    /// but the prelude fail to resolve.
    fn apply_with_status(
        state: &mut AppState,
        status: &tracker::Status<WaveformId, MarkId>,
        now: Instant,
        action: Action,
    ) -> Vec<Effect> {
        let evaluator = evaluator::Evaluator::new(48000, 120, std::path::PathBuf::new());
        let ctx = Context {
            status,
            now,
            evaluator: &evaluator,
        };
        apply(state, &ctx, action)
    }

    /// Applies `action` against an empty tracker status.
    fn apply_with_empty_status(state: &mut AppState, action: Action) -> Vec<Effect> {
        // XXX should we use Instant::now() or buffer_start?
        apply_with_status(state, &empty_status(), Instant::now(), action)
    }

    /// Builds a state whose slot 1 holds `_ = test;` (a program named by the
    /// anonymous pattern, so navigation shows no name). `input_path` is empty
    /// so splice never touches disk.
    fn test_state() -> AppState {
        AppState::from_source(
            "#{level_db=0}\n_ = test;".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse")
    }

    #[test]
    fn set_level_db_updates_state_and_emits_modify() {
        let mut state = test_state();
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetLevelDb {
                program: 0,
                level_db: -6.0,
            },
        );
        assert!((state.programs.programs()[0].level_db() - -6.0).abs() < 1e-6);
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
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("(".to_string()));
        let Mode::Edit { errors, .. } = &state.mode else {
            panic!("expected Edit mode after InsertText");
        };
        assert!(
            !errors.is_empty(),
            "expected parse errors after inserting an unbalanced '('"
        );
        // Now type the closing paren — errors should clear.
        apply_with_empty_status(&mut state, Action::InsertText("1)".to_string()));
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
    fn cursor_ops_respect_multibyte_char_boundaries() {
        // Arrow keys must move over whole characters, not bytes:
        // stepping left over a 2-byte char and typing used to panic
        // in `insert_str` with a mid-char cursor.
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("π".to_string()));
        apply_with_empty_status(&mut state, Action::MoveCursorBy(-1));
        let Mode::Edit {
            cursor_position, ..
        } = state.mode
        else {
            panic!("expected Edit mode");
        };
        assert_eq!(cursor_position, 0);
        apply_with_empty_status(&mut state, Action::InsertText("x".to_string()));
        assert_eq!(state.active_program().text(), "xπ");
    }

    #[test]
    fn backspace_removes_whole_multibyte_char() {
        // Backspace after a 2-byte char must remove the whole char
        // (removing at `cursor - 1` bytes used to panic mid-char).
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("aπ".to_string()));
        apply_with_empty_status(&mut state, Action::DeleteCharBeforeCursor);
        assert_eq!(state.active_program().text(), "a");
        let Mode::Edit {
            cursor_position, ..
        } = state.mode
        else {
            panic!("expected Edit mode");
        };
        assert_eq!(cursor_position, 1);
    }

    #[test]
    fn delete_word_handles_multibyte_whitespace() {
        // `prev_word_start` must skip past the full whitespace char
        // even when it's multi-byte (e.g. a non-breaking space).
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("a\u{a0}bc".to_string()));
        apply_with_empty_status(&mut state, Action::DeleteWordBeforeCursor);
        assert_eq!(state.active_program().text(), "a\u{a0}");
    }

    #[test]
    fn word_boundaries_follow_identifier_chars() {
        // Backward: skip non-word chars (operators, whitespace), then one
        // run of word chars.
        assert_eq!(prev_word_start("sine(440)"), 5); // before "440"
        assert_eq!(prev_word_start("sine("), 0);
        assert_eq!(prev_word_start("a + b"), 4);
        assert_eq!(prev_word_start("a + "), 0);
        assert_eq!(prev_word_start("c#4"), 0); // '#' is a word char
        assert_eq!(prev_word_start("x_1"), 0); // '_' is a word char
        assert_eq!(prev_word_start("+-*/"), 0); // no word: to the start
        assert_eq!(prev_word_start(""), 0);
        // Forward mirror.
        assert_eq!(next_word_end("sine(440)"), 4); // after "sine"
        assert_eq!(next_word_end("(440)"), 4); // after "440"
        assert_eq!(next_word_end(" + b"), 4);
        assert_eq!(next_word_end("+-*/"), 4); // no word: to the end
        assert_eq!(next_word_end(""), 0);
    }

    #[test]
    fn move_cursor_by_word_stops_at_identifier_boundaries() {
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(
            &mut state,
            Action::InsertText("sine(440) | env".to_string()),
        );
        let cursor = |state: &AppState| match state.mode {
            Mode::Edit {
                cursor_position, ..
            } => cursor_position,
            _ => panic!("expected Edit mode"),
        };
        apply_with_empty_status(&mut state, Action::MoveCursorToPreviousWord);
        assert_eq!(cursor(&state), 12); // before "env"
        apply_with_empty_status(&mut state, Action::MoveCursorToPreviousWord);
        assert_eq!(cursor(&state), 5); // before "440"
        apply_with_empty_status(&mut state, Action::MoveCursorToPreviousWord);
        assert_eq!(cursor(&state), 0); // before "sine"
        apply_with_empty_status(&mut state, Action::MoveCursorToNextWord);
        assert_eq!(cursor(&state), 4); // after "sine"
        apply_with_empty_status(&mut state, Action::MoveCursorToNextWord);
        assert_eq!(cursor(&state), 8); // after "440"
        apply_with_empty_status(&mut state, Action::MoveCursorToNextWord);
        assert_eq!(cursor(&state), 15); // after "env" (end of text)
        apply_with_empty_status(&mut state, Action::MoveCursorToNextWord);
        assert_eq!(cursor(&state), 15); // no-op at the end
    }

    #[test]
    fn delete_word_after_cursor_deletes_through_next_word() {
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("sine(440)".to_string()));
        apply_with_empty_status(&mut state, Action::MoveCursorToStart);
        apply_with_empty_status(&mut state, Action::DeleteWordAfterCursor);
        assert_eq!(state.active_program().text(), "(440)");
        apply_with_empty_status(&mut state, Action::DeleteWordAfterCursor);
        assert_eq!(state.active_program().text(), ")");
        // No word after the cursor: delete to the end of the text.
        apply_with_empty_status(&mut state, Action::DeleteWordAfterCursor);
        assert_eq!(state.active_program().text(), "");
        // No-op on empty text.
        apply_with_empty_status(&mut state, Action::DeleteWordAfterCursor);
        assert_eq!(state.active_program().text(), "");
    }

    #[test]
    fn delete_char_after_cursor_removes_whole_multibyte_char() {
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("πa".to_string()));
        apply_with_empty_status(&mut state, Action::MoveCursorToStart);
        apply_with_empty_status(&mut state, Action::DeleteCharAfterCursor);
        assert_eq!(state.active_program().text(), "a");
        // No-op at the end of the text.
        apply_with_empty_status(&mut state, Action::MoveCursorToEnd);
        apply_with_empty_status(&mut state, Action::DeleteCharAfterCursor);
        assert_eq!(state.active_program().text(), "a");
    }

    #[test]
    fn delete_to_end_of_line_deletes_rest_then_newline() {
        let mut state =
            AppState::from_source(String::new(), std::path::PathBuf::new()).expect("empty source");
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        apply_with_empty_status(&mut state, Action::InsertText("ab\ncd".to_string()));
        apply_with_empty_status(&mut state, Action::MoveCursorToStart);
        apply_with_empty_status(&mut state, Action::MoveCursorBy(1));
        apply_with_empty_status(&mut state, Action::DeleteToEndOfLine);
        assert_eq!(state.active_program().text(), "a\ncd");
        // At the end of a line: the newline itself is deleted.
        apply_with_empty_status(&mut state, Action::DeleteToEndOfLine);
        assert_eq!(state.active_program().text(), "acd");
        apply_with_empty_status(&mut state, Action::DeleteToEndOfLine);
        assert_eq!(state.active_program().text(), "a");
        // No-op at the end of the text.
        apply_with_empty_status(&mut state, Action::DeleteToEndOfLine);
        assert_eq!(state.active_program().text(), "a");
    }

    /// Builds a state from `source` whose slot-0 program text is replaced by
    /// `text`, in Edit mode with the cursor at `cursor`.
    fn edit_state(source: &str, text: &str, cursor: usize) -> AppState {
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");
        state
            .programs
            .program_mut(0)
            .unwrap()
            .set_text(text.to_string());
        state.mode = Mode::Edit {
            cursor_position: cursor,
            errors: vec![],
            completion: None,
        };
        state
    }

    /// The active program's (text, cursor) pair; panics outside Edit mode.
    fn edit_text_and_cursor(state: &AppState) -> (String, usize) {
        let Mode::Edit {
            cursor_position, ..
        } = state.mode
        else {
            panic!("expected Edit mode, got {:?}", state.mode);
        };
        (state.active_program().text().to_string(), cursor_position)
    }

    #[test]
    fn complete_cycles_through_matching_names_most_recent_first() {
        let mut state = edit_state(
            "za = 1;\nzb = 2;\nzab = 3;\n#{level_db=0}\n_ = test;",
            "z",
            1,
        );
        for (text, cursor) in [
            ("zab", 3), // the most recently bound match comes first
            ("zb", 2),
            ("za", 2),
            ("z", 1), // the ring wraps through the original fragment
            ("zab", 3),
        ] {
            apply_with_empty_status(&mut state, Action::Complete);
            assert_eq!(edit_text_and_cursor(&state), (text.to_string(), cursor));
        }
    }

    #[test]
    fn complete_skips_shadowed_names() {
        let mut state = edit_state("za = 1;\nza = 2;\n#{level_db=0}\n_ = test;", "z", 1);
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("za".to_string(), 2));
        // The shadowed za was skipped, so the next stop is the original.
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("z".to_string(), 1));
    }

    #[test]
    fn complete_includes_prelude_names() {
        let mut state = edit_state("#{level_db=0}\n_ = test;", "sin", 3);
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("sine".to_string(), 4));
    }

    #[test]
    fn completion_cycle_resets_after_another_action() {
        let mut state = edit_state("za = 1;\nzab = 2;\n#{level_db=0}\n_ = test;", "z", 1);
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("zab".to_string(), 3));
        // Any cursor motion ends the cycle: the next completion starts fresh
        // from the (now fully typed) fragment "zab", which matches nothing.
        apply_with_empty_status(&mut state, Action::MoveCursorBy(-1));
        apply_with_empty_status(&mut state, Action::MoveCursorBy(1));
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("zab".to_string(), 3));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("No completions")),
            "expected a no-completions message, got {:?}",
            effects
        );
    }

    #[test]
    fn complete_after_open_paren_inserts_parameter_hint() {
        let mut state = edit_state(
            "foo = fn(x, y = 10 + 1) => x * y;\n#{level_db=0}\n_ = test;",
            "foo(",
            4,
        );
        apply_with_empty_status(&mut state, Action::Complete);
        // The named default was evaluated at definition (10 + 1 → 11), and
        // the cursor lands after the first parameter, ready to replace it.
        assert_eq!(
            edit_text_and_cursor(&state),
            ("foo(x, y = 11)".to_string(), 5)
        );
    }

    #[test]
    fn complete_after_open_paren_of_builtin_reports_no_hint() {
        let mut state = edit_state("#{level_db=0}\n_ = test;", "sine(", 5);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("sine(".to_string(), 5));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("built-in")),
            "expected a built-in message, got {:?}",
            effects
        );
    }

    #[test]
    fn undo_reverts_last_unit_and_redo_restores_it() {
        let mut state = edit_state("", "", 0);
        // Two units: "440" starts a new one ('4' typed right after '(').
        apply_with_empty_status(&mut state, Action::InsertText("sine(".to_string()));
        apply_with_empty_status(&mut state, Action::InsertText("440".to_string()));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("sine(".to_string(), 5));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("".to_string(), 0));
        apply_with_empty_status(&mut state, Action::Redo);
        assert_eq!(edit_text_and_cursor(&state), ("sine(".to_string(), 5));
        apply_with_empty_status(&mut state, Action::Redo);
        assert_eq!(edit_text_and_cursor(&state), ("sine(440".to_string(), 8));
    }

    #[test]
    fn typing_coalesces_units_at_word_starts() {
        let mut state = edit_state("", "", 0);
        for c in "sine(440) | env".chars() {
            apply_with_empty_status(&mut state, Action::InsertText(c.to_string()));
        }
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(
            edit_text_and_cursor(&state),
            ("sine(440) | ".to_string(), 12)
        );
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("sine(".to_string(), 5));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("".to_string(), 0));
        let effects = apply_with_empty_status(&mut state, Action::Undo);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Nothing to undo"),
            "expected nothing-to-undo, got {:?}",
            effects
        );
    }

    #[test]
    fn undo_and_redo_show_message_when_history_is_empty() {
        let mut state = edit_state("", "", 0);
        let effects = apply_with_empty_status(&mut state, Action::Undo);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Nothing to undo"),
            "expected nothing-to-undo, got {:?}",
            effects
        );
        let effects = apply_with_empty_status(&mut state, Action::Redo);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Nothing to redo"),
            "expected nothing-to-redo, got {:?}",
            effects
        );
        assert_eq!(edit_text_and_cursor(&state), ("".to_string(), 0));
    }

    #[test]
    fn fresh_edit_clears_redo_stack() {
        let mut state = edit_state("", "", 0);
        apply_with_empty_status(&mut state, Action::InsertText("ab".to_string()));
        apply_with_empty_status(&mut state, Action::Undo);
        apply_with_empty_status(&mut state, Action::InsertText("cd".to_string()));
        let effects = apply_with_empty_status(&mut state, Action::Redo);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Nothing to redo"),
            "expected nothing-to-redo, got {:?}",
            effects
        );
    }

    #[test]
    fn delete_ops_record_units_and_close_the_run() {
        let mut state = edit_state("", "", 0);
        apply_with_empty_status(&mut state, Action::InsertText("ab".to_string()));
        apply_with_empty_status(&mut state, Action::DeleteCharBeforeCursor);
        // 'c' after 'b' would coalesce if the delete hadn't closed the run.
        apply_with_empty_status(&mut state, Action::InsertText("c".to_string()));
        assert_eq!(edit_text_and_cursor(&state), ("ac".to_string(), 2));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("a".to_string(), 1));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("ab".to_string(), 2));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("".to_string(), 0));
    }

    #[test]
    fn completion_cycle_undoes_as_one_unit() {
        let mut state = edit_state(
            "za = 1;\nzb = 2;\nzab = 3;\n#{level_db=0}\n_ = test;",
            "z",
            1,
        );
        apply_with_empty_status(&mut state, Action::Complete);
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("zb".to_string(), 2));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("z".to_string(), 1));
        apply_with_empty_status(&mut state, Action::Redo);
        assert_eq!(edit_text_and_cursor(&state), ("zb".to_string(), 2));
    }

    #[test]
    fn undo_refreshes_edit_errors_and_clears_completion() {
        let mut state = edit_state("", "", 0);
        apply_with_empty_status(&mut state, Action::InsertText("(".to_string()));
        let Mode::Edit { errors, .. } = &state.mode else {
            panic!("expected Edit mode");
        };
        assert!(!errors.is_empty(), "expected parse errors for \"(\"");
        apply_with_empty_status(&mut state, Action::Undo);
        let Mode::Edit {
            errors, completion, ..
        } = &state.mode
        else {
            panic!("expected Edit mode");
        };
        assert!(errors.is_empty(), "expected no errors after undo");
        assert!(completion.is_none());
        apply_with_empty_status(&mut state, Action::Redo);
        let Mode::Edit { errors, .. } = &state.mode else {
            panic!("expected Edit mode");
        };
        assert!(!errors.is_empty(), "expected parse errors again after redo");
    }

    #[test]
    fn noop_edit_records_no_history() {
        let mut state = edit_state("", "", 0);
        apply_with_empty_status(&mut state, Action::DeleteCharBeforeCursor);
        let effects = apply_with_empty_status(&mut state, Action::Undo);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Nothing to undo"),
            "expected nothing-to-undo, got {:?}",
            effects
        );
    }

    #[test]
    fn history_survives_mode_round_trip() {
        let mut state = edit_state("", "", 0);
        apply_with_empty_status(&mut state, Action::InsertText("ab".to_string()));
        apply_with_empty_status(&mut state, Action::EnterSelectMode);
        apply_with_empty_status(&mut state, Action::EnterEditMode);
        // A new unit only because EnterEditMode closed the insert run —
        // 'c' right after 'b' would otherwise coalesce.
        apply_with_empty_status(&mut state, Action::InsertText("cd".to_string()));
        assert_eq!(edit_text_and_cursor(&state), ("abcd".to_string(), 4));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("ab".to_string(), 2));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("".to_string(), 0));
    }

    #[test]
    fn undo_outside_edit_mode_is_a_no_op() {
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::Undo);
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        assert!(matches!(state.mode, Mode::Select));
    }

    #[test]
    fn advance_program_emits_empty_show_message_to_clear_status() {
        // Two programs so AdvanceProgram(1) actually moves the index.
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = test;\n#{level_db=0}\n_ = second;".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let effects = apply_with_empty_status(&mut state, Action::AdvanceProgram(1));
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
    fn select_program_shows_binding_name_on_navigate() {
        // Named binding → status shows the identifier; anonymous `_`
        // binding → status is left blank; a slot with no source binding
        // (padding) → also blank.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
_ = saw(220);";
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");

        // Slot 1: named `kick`.
        let effects = apply_with_empty_status(&mut state, Action::SelectProgram(0));
        let msg = effects
            .iter()
            .find_map(|e| match e {
                Effect::ShowMessage(s) => Some(s.clone()),
                _ => None,
            })
            .expect("expected a ShowMessage");
        assert_eq!(msg, "kick");

        // Slot 2: anonymous `_` — status stays empty.
        let effects = apply_with_empty_status(&mut state, Action::SelectProgram(1));
        let msg = effects
            .iter()
            .find_map(|e| match e {
                Effect::ShowMessage(s) => Some(s.clone()),
                _ => None,
            })
            .expect("expected a ShowMessage");
        assert_eq!(msg, "");

        // Slot 3: padding (no binding for this slot).
        let effects = apply_with_empty_status(&mut state, Action::SelectProgram(2));
        let msg = effects
            .iter()
            .find_map(|e| match e {
                Effect::ShowMessage(s) => Some(s.clone()),
                _ => None,
            })
            .expect("expected a ShowMessage");
        assert_eq!(msg, "");
    }

    #[test]
    fn set_encoder_mode_emits_set_launchkey_encoder_mode() {
        let mut state = test_state();
        let effects = apply_with_empty_status(
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

    /// Installs program 0 as the keys instrument with a dummy function.
    fn install_test_keys(state: &mut AppState) {
        state.keys = Some(keys::Keys {
            id: 0,
            function: expr::SourceExpr::float(0.0),
            note_off_waveforms: Default::default(),
        });
    }

    #[test]
    fn slider_change_on_keys_program_propagates_to_active_keys() {
        let mut state = AppState::from_source(
            "#{sliders=[\"vol:0.5:0:1\"]}\nk = fn(note, vel) => (vol, vol);".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        install_test_keys(&mut state);
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetSliderNormalized {
                program: 0,
                slider_index: 0,
                normalized: 0.8,
            },
        );
        // The program's own waveform gets the update...
        assert!(
            effects.iter().any(|e| matches!(
                e,
                Effect::UpdateSlider {
                    id: WaveformId::Program(0),
                    ..
                }
            )),
            "expected UpdateSlider, got {:?}",
            effects
        );
        // ...and so does every active key waveform of the installed instrument.
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::UpdateActiveKeySliders { .. })),
            "expected UpdateActiveKeySliders, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_program_playback_stops_when_active() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_mark(now - Duration::from_secs(1));
        let effects = apply_with_status(&mut state, &status, now, Action::ToggleProgramPlayback(0));
        assert!(
            matches!(effects[0], Effect::StopProgram(0)),
            "expected StopProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_program_playback_plays_immediately_when_idle() {
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::ToggleProgramPlayback(0));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start_at_next_measure: false,
                    repeat_after_measures: None,
                }
            ),
            "expected an immediate PlayProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_program_playback_ignores_installed_keys_program() {
        let mut state = test_state();
        install_test_keys(&mut state);
        let effects = apply_with_empty_status(&mut state, Action::ToggleProgramPlayback(0));
        assert!(
            effects.is_empty(),
            "expected no effects for the installed keys program, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_queued_program_removes_pending_when_queued() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_mark(now + Duration::from_secs(1));
        let effects = apply_with_status(
            &mut state,
            &status,
            now,
            Action::ToggleProgramPendingPlayback(0),
        );
        assert!(
            matches!(effects[0], Effect::RemovePendingProgram(0)),
            "expected RemovePendingProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_queued_program_queues_with_default_repeat_when_idle() {
        let mut state = test_state();
        state.repeat_after_measures = Some(2);
        let effects = apply_with_empty_status(&mut state, Action::ToggleProgramPendingPlayback(0));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start_at_next_measure: true,
                    repeat_after_measures: Some(2),
                }
            ),
            "expected a queued PlayProgram with the default repeat, got {:?}",
            effects
        );
    }

    #[test]
    fn stop_and_remove_pending_are_no_ops_when_nothing_is_playing() {
        // Their doc comments promise "otherwise, do nothing" — in
        // particular no stale "Stopped program …" status message.
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::StopProgram(0));
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        let effects = apply_with_empty_status(&mut state, Action::RemovePendingProgram(0));
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
    }

    #[test]
    fn enter_edit_mode_removes_pending_playback() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_mark(now + Duration::from_secs(1));
        let effects = apply_with_status(&mut state, &status, now, Action::EnterEditMode);
        assert!(
            matches!(state.mode, Mode::Edit { .. }),
            "expected Edit mode, got {:?}",
            state.mode
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RemovePendingProgram(0))),
            "expected the pending playback to be removed, got {:?}",
            effects
        );
    }
}
