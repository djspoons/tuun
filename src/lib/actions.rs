//! Action / Effect / reducer for app state.
//!
//! Handlers (sdl2_input, midi_input) classify raw input events into pure
//! `Action`s. `apply` then mutates `AppState` and returns `Effect`s, which
//! the runner in `effects.rs` executes against the outside world.

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use crate::diagnostics::{self, Diagnostic, Source};
use crate::environment;
use crate::expr;
use crate::ids::{ANCHOR_SIXTEENTH, MarkId, WaveformId, WaveformSelector};
use crate::keys;
use crate::launchkey;
use crate::player;
use crate::programs::{self, BoundTo, PROGRAMS_PER_BANK, Program};
use crate::recorder::{self, Closed, Finish, Phase, Response, Take, Tick};
use crate::scope;
use crate::sequencer;
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
    /// Recording a phrase from the keys into the active program. Entered from
    /// Select with the Record transport press and left, always to Select, when
    /// the take is written or dropped. The checks made at arm (the keys
    /// binding's name, the helper's scope) must hold until the write, so no
    /// gesture reachable in this mode may change any program's text.
    Record {
        take: Take,
        /// The installed keys program when the take was armed.
        keys_program: usize,
    },
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
    /// Names starting with `original` (in-scope names or a module's bindings
    /// when the fragment is qualified) most recently bound first.
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
    /// Both rows toggle the active program's `on_beats` steps as a 16-pad
    /// sixteenth grid.
    Sequencer,
}

impl DawPadMode {
    /// Human-readable label shown on the controller's display strip when
    /// this sub-mode is active.
    pub fn display_name(self) -> &'static str {
        match self {
            DawPadMode::ClipLauncher => "Clip Launcher",
            DawPadMode::KeysInstaller => "Keys Installer",
            DawPadMode::Sequencer => "Sequencer",
        }
    }
}

/// How a press of a clip-launcher top pad relates to the program's playback.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LaunchMode {
    /// A press starts the program when it is silent and stops it when it is
    /// sounding.
    Toggle,
    /// A press always plays the program again, once, and never stops it, so
    /// repeated presses layer.
    Trigger,
}

impl LaunchMode {
    /// Human-readable label for this mode.
    pub fn display_name(self) -> &'static str {
        match self {
            LaunchMode::Toggle => "Toggle",
            LaunchMode::Trigger => "Trigger",
        }
    }
}

/// A key that does nothing by itself and waits for the next key to pick a
/// command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefixKey {
    /// Quits on a second press.
    Quit,
    /// Changes the active program's annotation.
    Annotation,
    /// Changes the active program's Loop.
    Loop,
}

impl PrefixKey {
    /// Returns the prompt shown while this prefix key waits on the program
    /// named `program_name`.
    pub fn prompt(self, program_name: &str) -> String {
        match self {
            PrefixKey::Quit => "Quit?".to_string(),
            PrefixKey::Annotation => "Annotation? (k)eys (u)p (d)own (l)oop".to_string(),
            PrefixKey::Loop => format!("Loop {}? (1-9) beats (0) once", program_name),
        }
    }
}

/// The status message shown while a program's result is on its way back from
/// the processing thread.
pub const PROCESSING_MESSAGE: &str = "Processing…";

/// A job sent to the processing thread whose reply hasn't been applied yet.
#[derive(Debug)]
pub enum PendingJob {
    /// The evaluation of one program (`Effect::EvaluateProgram`).
    Program {
        /// The program's index.
        index: usize,
        /// The program's generation when the job was sent.
        generation: u64,
        /// The program's text when the job was sent: the text evaluated.
        text: Arc<str>,
        /// The effects to run once a successful result is recorded.
        on_success: Vec<Effect>,
        /// The effects to run once a failed result is recorded.
        on_failure: Vec<Effect>,
        /// The status message replaced by [`PROCESSING_MESSAGE`] when the job
        /// was sent.
        message: String,
    },
    /// The load of a whole program set (`Effect::ReloadSource`, or startup).
    Load {
        /// Whether this is the load at startup, which reports differently from
        /// a reload.
        startup: bool,
        /// The slot to select once the set is installed, if any.
        select: Option<usize>,
    },
}

/// Internal state of the application. The reducer takes `&mut AppState` and mutates it in
/// place; `main` keeps a single instance for the lifetime of the program.
pub struct AppState {
    /// The programs and the source file backing them.
    pub programs: programs::ProgramSet,
    pub active_program_index: usize,
    pub mode: Mode,
    pub keys: Option<keys::Keys>,
    /// How a clip-launcher top pad press relates to playback. Kept across
    /// changes of `daw_pad_mode`.
    pub launch_mode: LaunchMode,
    /// Active behavior for the DAW pad.
    pub daw_pad_mode: DawPadMode,
    /// The measure the sequencer pad grid is showing: page 0 = beats [1, 5),
    /// page 1 = beats [5, 9), and so on.
    pub sequencer_page: u8,
    /// Whether the metronome clicks on every beat.
    pub metronome: bool,
    /// Which scope view the screen draws.
    pub scope_view: scope::ScopeView,
    /// The prefix key waiting for the next key, if any.
    pub pending_prefix: Option<PrefixKey>,
    /// Set by `Effect::Quit`; `main` checks at top of loop and breaks.
    pub should_quit: bool,
    /// Last user-visible status message. Set by `Effect::ShowMessage` (and
    /// a few direct writes from the reducer / runner). Persists across
    /// mode transitions; cleared explicitly by navigation actions.
    ///
    /// May be multi-line (e.g. an error followed by its source snippet):
    /// the first line is the status-line summary, and single-line display
    /// sites should show only that line.
    pub message: String,
    /// The jobs sent to the processing thread whose replies haven't been
    /// applied, in the order they were sent (the order replies arrive in).
    pub processing: VecDeque<PendingJob>,
    /// The marks of the waveform last sent to play under each id.
    pub waveform_marks: HashMap<WaveformId, player::PlayedMarks>,
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
            launch_mode: LaunchMode::Toggle,
            daw_pad_mode: DawPadMode::ClipLauncher,
            sequencer_page: 0,
            metronome: false,
            scope_view: scope::ScopeView::Snapshot,
            pending_prefix: None,
            should_quit: false,
            message,
            processing: VecDeque::new(),
            waveform_marks: HashMap::new(),
        })
    }

    pub fn bank_start(&self) -> usize {
        self.active_program_index - (self.active_program_index % PROGRAMS_PER_BANK)
    }

    pub fn active_program(&self) -> &Program {
        &self.programs.programs()[self.active_program_index]
    }

    /// Returns when the take being recorded next needs a tick, if it does.
    pub fn next_take_tick(&self) -> Option<Instant> {
        match &self.mode {
            Mode::Record { take, .. } => take.next_tick(),
            _ => None,
        }
    }

    /// Returns whether a take is waiting on a tick that is due at `now`.
    pub fn take_tick_due(&self, now: Instant) -> bool {
        self.next_take_tick().is_some_and(|due| due <= now)
    }

    /// Returns whether an evaluation of the program at `index` is waiting on
    /// the processing thread.
    pub fn is_processing(&self, index: usize) -> bool {
        self.processing
            .iter()
            .any(|job| matches!(job, PendingJob::Program { index: i, .. } if *i == index))
    }

    /// Returns whether a load of a whole program set is waiting on the
    /// processing thread.
    pub fn is_loading(&self) -> bool {
        self.processing
            .iter()
            .any(|job| matches!(job, PendingJob::Load { .. }))
    }
}

/// Read-only snapshot of the world used by the reducer.
pub struct Context<'a> {
    pub status: &'a tracker::Status<WaveformId>,
    pub now: Instant,
    /// The environment programs are checked and evaluated in (prelude,
    /// module cache) — used by `Action::Complete` to find the names in
    /// scope.
    pub environment: &'a environment::Environment,
    /// True when the backing file has changed on disk since the program set
    /// last read or wrote it. Editing gestures refuse while stale so in-app
    /// changes can't pile up against external ones.
    pub source_stale: bool,
    /// The tempo in beats per minute.
    pub tempo: u32,
}

/// A transport buttons on the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    Record,
    Play,
    Stop,
}

/// When a launch starts, which also decides whether it repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Launch {
    /// At the beginning of the next measure, repeating as the program's
    /// `loop_after_beats` says.
    Pending,
    /// Immediately, playing once.
    Now,
}

/// Things that can happen, emitted by handlers as pure data.
#[derive(Debug)]
pub enum Action {
    // --- playback ---
    /// Play the program at `program_index` as `launch` says, whether or not
    /// it is already sounding. Does nothing if the program is the installed
    /// keys instrument or isn't a waveform.
    PlayProgram {
        program_index: usize,
        launch: Launch,
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
    /// it to play at the beginning of next measure (repeating as the program's
    /// `loop_after_beats` says). Does nothing if the program isn't a waveform.
    ToggleProgramPendingPlayback(usize),
    /// Press a transport button.
    ///
    /// In Record mode the press goes to the take: Record and Play finish it
    /// at the press's measure boundary (or disarm it before it starts), and
    /// Stop discards it. Otherwise, Record arms a take on the active program
    /// if every precondition for recording holds, Play queues the active
    /// program to play at the beginning of the next measure (repeating as its
    /// `loop_after_beats` says) unless its playback is already pending, and Stop
    /// removes the active program's pending playback.
    Transport(Transport),
    /// Advance the take in Record mode to now: start recording at its start
    /// boundary, or write it at its close. Does nothing in other modes.
    TakeTick,
    /// Toggle the given sixteenth of the active program's `on_beats` step
    /// list, in the source text and — when the program is live — on the
    /// tracker.
    ToggleSequencerStep {
        sixteenth: u8,
    },
    /// Shift the sequencer pad grid's page by the given number of (4 beat)
    /// measures, clamped to the valid page range.
    ChangeSequencerPage(i8),
    /// Stop every playing program and rebuild the program set from the backing
    /// file's current contents.
    ReloadSource,

    // --- MIDI keys ---
    /// Install the program at the given index as the keys instrument. If the
    /// program at the given index is already installed, uninstall it.
    ToggleInstalledKeys(usize),
    /// Press a key on the keys instrument. `stamp` is when the press was
    /// received.
    NoteOn {
        key: u8,
        velocity: u8,
        stamp: Instant,
    },
    /// Release a key on the keys instrument. `stamp` is when the release was
    /// received.
    NoteOff {
        key: u8,
        stamp: Instant,
    },

    // --- mode transitions ---
    /// Enter Edit mode on the active program, first removing any pending
    /// playback for it.
    EnterEditMode,
    /// Parse and evaluate the current program, updating its state and the
    /// source file. On success, return to Select mode; otherwise set the mode
    /// to `mode_on_failure`. With `play`, then play the program as that
    /// launch says.
    EvaluateAndLeaveEditMode {
        mode_on_failure: Mode,
        play: Option<Launch>,
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
    /// wrapping back to the fragment itself. A fragment qualified by a module
    /// projection (`pm.pia`) completes among that module's bindings instead,
    /// and a bare `pm.` cycles through all of them. Right after a `(`, inserts
    /// a parameter hint for the function being called instead, leaving the
    /// cursor on the first parameter.
    Complete,
    /// Report the type of the identifier under the cursor.
    ///
    /// A diagnostic aid: shows what the checker makes of a name where it
    /// stands, so a parameter's sort or an overloaded function's table can be
    /// read without provoking an error. The cursor may sit anywhere in the
    /// identifier, including just past its last character.
    ShowType,
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
    /// Switch the clip launcher's top pads between `LaunchMode::Toggle` and
    /// `LaunchMode::Trigger`.
    ToggleLaunchMode,
    /// Turn the metronome click on or off.
    ToggleMetronome,
    /// Switch the screen between `scope::ScopeView::Snapshot` and
    /// `scope::ScopeView::Scan`.
    ToggleScopeView,

    // --- annotations ---
    /// Add the `keys` annotation to the program, or remove it, saving the
    /// change and evaluating the program again. A program that loses `keys`
    /// while installed as the keys instrument is uninstalled.
    ToggleKeysAnnotation(usize),
    /// Set the program's `loop_after_beats` annotation to `beats`, or remove it
    /// when `beats` is `None`, saving the change. Whatever is already playing
    /// keeps its period.
    SetLoopAfterBeats {
        program_index: usize,
        beats: Option<u32>,
    },
    /// Change the program's `skip_slots` by `delta`, which moves it and every
    /// later program by `delta` slots, then reload the file. The selection
    /// follows the program. Refused when it would move the program onto the
    /// slot before it or past the last slot.
    ChangeSkipSlots {
        program_index: usize,
        delta: i32,
    },

    // --- prefix keys ---
    /// Wait for the next key to complete the prefix key, showing its prompt.
    /// `PrefixKey::Loop` is refused, with a message, when the active
    /// program's Loop can't be changed.
    StartPrefix(PrefixKey),
    /// Stop waiting on the pending prefix key, clearing its prompt.
    EndPrefix,

    // --- program-related I/O and other effects ---
    ShowMessage(String),
    /// Print the evaluated waveform of the program at the given index.
    ///
    /// For keys instruments, use an arbitrary note and velocity to generate an
    /// example waveform.
    PrintEvaluatedProgram(usize),
    Quit,
}

/// Side effects to perform after the reducer runs. The runner translates
/// these into MPSC sends, MIDI controller updates, etc.
#[derive(Debug)]
pub enum Effect {
    // --- tracker commands ---
    /// Send a Play command for the program at `program_index`, starting where
    /// `start` says, and post the play text unless the start is an instant.
    PlayProgram {
        program_index: usize,
        start: player::Start,
        loop_after_beats: Option<u32>,
    },
    /// Send a "stop" ramp for the program waveform.
    StopProgram(usize),
    /// Cancel the program's queued playback from the next cycle boundary
    /// on; anything already scheduled within the current cycle plays out.
    RemovePendingProgram(usize),
    /// Schedule one step of a live sequenceable program at the next
    /// occurrence of its sixteenth.
    PlaySequencerStep { program_index: usize, sixteenth: u8 },
    /// Remove one sequencer step's pending playback from the tracker.
    RemovePendingSequencerStep { program_index: usize, sixteenth: u8 },
    /// Re-record the marks of the program's steps from its current sequence,
    /// without playing them.
    RecordSequencerStepMarks(usize),
    /// Turn the metronome click on the Beats waveforms on or off.
    SetMetronome(bool),
    /// Send a Modify command to the tracker for the selected waveforms.
    ModifyWaveform {
        selector: WaveformSelector,
        mark_id: MarkId,
        waveform: waveform::Waveform<MarkId>,
    },

    /// Send the given program to the processing thread to be parsed and
    /// evaluated. Once the result is recorded, run `on_success`, or
    /// `on_failure` when the evaluation failed. A failure's diagnostics go to
    /// Edit mode when that is the mode `on_failure` leaves.
    EvaluateProgram {
        program_index: usize,
        on_success: Vec<Effect>,
        on_failure: Vec<Effect>,
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
    /// worker thread coalesces these per quantum and sends `Command::Modify` to
    /// the tracker with a ramp from the previous value, so every sounding voice
    /// that embeds `mark` picks up the new value.
    UpdateSlider { mark: MarkId, value: f32 },

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
    /// Write the backing file with the program's binding given `skip_slots`,
    /// and send the new source to the processing thread as `ReloadSource` does,
    /// selecting slot `select` once the set is installed. A failed write
    /// changes nothing.
    MoveProgram {
        program_index: usize,
        skip_slots: u32,
        select: usize,
    },
    /// Re-read the backing file and send it to the processing thread to be
    /// rebuilt into a program set. Once the set is built, stop every voice on
    /// the tracker and install it. A file that can't be read or parsed leaves
    /// playback and state untouched.
    ReloadSource,
    /// Switch to the given mode. Only for a switch that waits on another
    /// effect's outcome (see `EvaluateProgram`); otherwise the reducer sets
    /// `state.mode` itself.
    SetMode(Mode),
    /// User-visible status message.
    ShowMessage(String),
    /// Print the evaluated waveform of the program at the given index to
    /// stdout.
    ///
    /// For keys instruments, use an arbitrary note and velocity to generate an
    /// example waveform.
    PrintEvaluatedProgram(usize),
    /// Sets `state.should_quit = true`.
    Quit,
}

/// Applies an action to state, returning effects for the runner to execute.
///
/// Performs only the state mutation that needs no I/O, reading nothing beyond
/// `state` and `ctx`. Effects whose outcome depends on I/O (evaluating a
/// program, splicing source, playing notes) mutate state in the runner instead.
pub fn apply(state: &mut AppState, ctx: &Context, action: Action) -> Vec<Effect> {
    // The take owns the transport while it lives.
    if let (Action::Transport(transport), Mode::Record { .. }) = (&action, &state.mode) {
        return apply_take_transport(state, ctx, *transport);
    }
    if refused_by_ownership(state, ctx.status, ctx.now, &action) {
        return vec![];
    }
    if let Some(effects) = refused_by_processing(state, &action) {
        return effects;
    }
    match action {
        Action::PlayProgram {
            program_index,
            launch,
        } => {
            if state.keys.as_ref().is_some_and(|k| k.id == program_index) {
                vec![]
            } else {
                launch_effects(state, program_index, launch)
            }
        }
        Action::StopProgram(i) => stop_program_effects(state, ctx, i),
        Action::RemovePendingProgram(i) => remove_pending_effects(state, ctx, i),
        Action::ToggleProgramPlayback(i) => {
            if ctx
                .status
                .has_active(ctx.now, &WaveformSelector::ProgramPlayback(i))
            {
                stop_program_effects(state, ctx, i)
            } else if state.keys.as_ref().is_some_and(|k| k.id == i) {
                vec![]
            } else {
                launch_effects(state, i, Launch::Now)
            }
        }
        Action::ToggleProgramPendingPlayback(i) => {
            if ctx
                .status
                .has_pending(ctx.now, &WaveformSelector::ProgramPlayback(i))
            {
                remove_pending_effects(state, ctx, i)
            } else if state.keys.as_ref().is_some_and(|k| k.id == i) {
                vec![]
            } else {
                launch_effects(state, i, Launch::Pending)
            }
        }

        Action::Transport(Transport::Play) => transport_play_effects(state, ctx),
        Action::Transport(Transport::Stop) => {
            remove_pending_effects(state, ctx, state.active_program_index)
        }
        Action::Transport(Transport::Record) => apply_arm(state, ctx),
        Action::TakeTick => apply_take_tick(state, ctx),

        Action::ToggleSequencerStep { sixteenth } => {
            apply_toggle_sequencer_step(state, ctx, sixteenth)
        }
        Action::ReloadSource => {
            // Reloading under an active edit session would discard it silently;
            // only Select mode may reload.
            if matches!(state.mode, Mode::Select) {
                vec![Effect::ReloadSource]
            } else {
                vec![]
            }
        }

        Action::ChangeSequencerPage(delta) => {
            state.sequencer_page = (state.sequencer_page as i16 + delta as i16)
                .clamp(0, sequencer::MAX_PAGE as i16) as u8;
            let first_beat = state.sequencer_page as u32 * 4 + 1;
            vec![Effect::ShowMessage(format!(
                "Sequencer beats {}–{}",
                first_beat,
                first_beat + 3
            ))]
        }

        Action::ToggleInstalledKeys(i) => apply_install_keys(state, i),
        Action::NoteOn {
            key,
            velocity,
            stamp,
        } => {
            if let Mode::Record { take, .. } = &mut state.mode {
                take.note_on(key, velocity, stamp);
            }
            if state.keys.is_some() {
                vec![Effect::PlayNoteOn { key, velocity }]
            } else {
                vec![]
            }
        }
        Action::NoteOff { key, stamp } => {
            if let Mode::Record { take, .. } = &mut state.mode {
                take.note_off(key, stamp);
            }
            vec![Effect::PlayNoteOff { key }]
        }

        Action::EnterEditMode => {
            // Editing on top of external changes would set up a conflict the
            // splice guard then has to refuse; reload first instead.
            if ctx.source_stale {
                return vec![Effect::ShowMessage(
                    "File changed on disk (reload before editing)".to_string(),
                )];
            }
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
        Action::EvaluateAndLeaveEditMode {
            mode_on_failure,
            play,
        } => {
            let i = state.active_program_index;
            // The play reads the result, so it waits for it; the save does not,
            // so it runs now.
            let mut on_success = vec![Effect::SetMode(Mode::Select)];
            if let Some(launch) = play {
                on_success.push(launch_effect(state, i, launch));
            }
            vec![
                Effect::EvaluateProgram {
                    program_index: i,
                    on_success,
                    on_failure: vec![Effect::SetMode(mode_on_failure)],
                },
                Effect::UpdateSource(i),
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
        Action::ShowType => apply_show_type(state, ctx),
        Action::Undo => {
            apply_history_restore(state, ctx, Program::undo, "Undid", "Nothing to undo")
        }
        Action::Redo => {
            apply_history_restore(state, ctx, Program::redo, "Redid", "Nothing to redo")
        }

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
                        DawPadMode::KeysInstaller => DawPadMode::Sequencer,
                        DawPadMode::Sequencer => DawPadMode::ClipLauncher,
                    };
                }
                let label = state.daw_pad_mode.display_name().to_string();
                effects.push(Effect::SetDawModeDisplay(label.clone()));
                let message = if state.daw_pad_mode == DawPadMode::Sequencer
                    && sequencer::analyze(state.active_program().text()).is_none()
                {
                    format!(
                        "{}: {} is not [steps] | on_beats(w)",
                        label,
                        state.programs.display_name(state.active_program_index)
                    )
                } else {
                    label
                };
                effects.push(Effect::ShowMessage(message));
            }
            effects
        }

        Action::ToggleLaunchMode => {
            state.launch_mode = match state.launch_mode {
                LaunchMode::Toggle => LaunchMode::Trigger,
                LaunchMode::Trigger => LaunchMode::Toggle,
            };
            vec![Effect::ShowMessage(format!(
                "Launch mode: {}",
                state.launch_mode.display_name()
            ))]
        }

        Action::ToggleScopeView => {
            state.scope_view = state.scope_view.toggled();
            vec![Effect::ShowMessage(format!(
                "Scope view: {}",
                state.scope_view.display_name()
            ))]
        }

        Action::ToggleMetronome => {
            state.metronome = !state.metronome;
            vec![
                Effect::SetMetronome(state.metronome),
                Effect::ShowMessage(format!(
                    "Metronome {}",
                    if state.metronome { "on" } else { "off" }
                )),
            ]
        }

        Action::ToggleKeysAnnotation(i) => apply_toggle_keys_annotation(state, ctx, i),
        Action::SetLoopAfterBeats {
            program_index,
            beats,
        } => apply_set_loop_after_beats(state, ctx, program_index, beats),
        Action::ChangeSkipSlots {
            program_index,
            delta,
        } => apply_change_skip_slots(state, ctx, program_index, delta),

        Action::StartPrefix(prefix) => {
            let i = state.active_program_index;
            if prefix == PrefixKey::Loop
                && let Some(refusal) = loop_refusal(state, ctx, i)
            {
                return vec![Effect::ShowMessage(refusal)];
            }
            state.pending_prefix = Some(prefix);
            state.message = prefix.prompt(&state.programs.display_name(i));
            vec![]
        }
        Action::EndPrefix => {
            state.pending_prefix = None;
            state.message.clear();
            vec![]
        }

        Action::ShowMessage(message) => vec![Effect::ShowMessage(message)],
        Action::PrintEvaluatedProgram(program_index) => {
            vec![Effect::PrintEvaluatedProgram(program_index)]
        }
        Action::Quit => vec![
            Effect::UpdateSource(state.active_program_index),
            Effect::Quit,
        ],
    }
}

/// Returns whether the editor mode owns the active program.
///
/// While it does, changes that would affect or depend on the active
/// program's text are refused.
fn owns_active_program(state: &AppState) -> bool {
    matches!(state.mode, Mode::Edit { .. } | Mode::Record { .. })
}

/// Returns whether `action` is refused, silently, because the editor mode owns
/// the active program.
///
/// Stopping the owned program's playback, removing its pending playback, or
/// uninstalling it as the keys instrument are allowed: they act on an
/// already-evaluated waveform or instrument.
pub fn refused_by_ownership(
    state: &AppState,
    status: &tracker::Status<WaveformId>,
    now: Instant,
    action: &Action,
) -> bool {
    if !owns_active_program(state) {
        return false;
    }
    let owned = state.active_program_index;
    let playback = WaveformSelector::ProgramPlayback(owned);
    match *action {
        Action::SelectProgram(_)
        | Action::AdvanceProgram(_)
        | Action::EnterEditMode
        | Action::ToggleSequencerStep { .. }
        | Action::ToggleKeysAnnotation(_)
        | Action::SetLoopAfterBeats { .. }
        | Action::ChangeSkipSlots { .. } => true,
        // Refused only when it would install; uninstalling drops an
        // already-evaluated instrument.
        Action::ToggleInstalledKeys(i) => {
            i == owned && !state.keys.as_ref().is_some_and(|k| k.id == owned)
        }
        Action::PlayProgram { program_index, .. } => program_index == owned,
        Action::Transport(Transport::Play) => true,
        // The toggles are refused only when they would start or queue.
        Action::ToggleProgramPlayback(i) => i == owned && !status.has_active(now, &playback),
        Action::ToggleProgramPendingPlayback(i) => {
            i == owned && !status.has_pending(now, &playback)
        }
        // Actions that check the editor mode themselves, such as reload and
        // undo, refuse there.
        _ => false,
    }
}

/// Returns the effects of refusing `action` because a job is waiting on the
/// processing thread, or `None` when it isn't refused.
///
/// While a program set is loading, the actions that would change the source
/// (entering Edit mode, toggling a sequencer step, arming a take, undo and
/// redo, changing an annotation, and another reload) are refused with a message, since the loaded set
/// replaces the source they would change.
///
/// While an evaluation of the active program is waiting, entering Edit mode is
/// refused with a message. In Edit mode, the keyboard actions that would change
/// the text, move the cursor, or evaluate again are dropped silently, so the
/// text the result answers is the text on screen. Showing a type only reads, so
/// it stays allowed.
fn refused_by_processing(state: &AppState, action: &Action) -> Option<Vec<Effect>> {
    if state.is_loading()
        && matches!(
            action,
            Action::EnterEditMode
                | Action::ToggleSequencerStep { .. }
                | Action::Transport(Transport::Record)
                | Action::Undo
                | Action::Redo
                | Action::ToggleKeysAnnotation(_)
                | Action::SetLoopAfterBeats { .. }
                | Action::ChangeSkipSlots { .. }
                | Action::ReloadSource
        )
    {
        return Some(vec![Effect::ShowMessage(PROCESSING_MESSAGE.to_string())]);
    }
    if !state.is_processing(state.active_program_index) {
        return None;
    }
    match action {
        Action::EnterEditMode => Some(vec![Effect::ShowMessage(PROCESSING_MESSAGE.to_string())]),
        Action::InsertText(_)
        | Action::DeleteCharBeforeCursor
        | Action::DeleteCharAfterCursor
        | Action::DeleteWordBeforeCursor
        | Action::DeleteWordAfterCursor
        | Action::DeleteToEndOfLine
        | Action::MoveCursorBy(_)
        | Action::MoveCursorToStart
        | Action::MoveCursorToEnd
        | Action::MoveCursorToPreviousWord
        | Action::MoveCursorToNextWord
        | Action::Complete
        | Action::Undo
        | Action::Redo
        | Action::EvaluateAndLeaveEditMode { .. }
            if matches!(state.mode, Mode::Edit { .. }) =>
        {
            Some(vec![])
        }
        _ => None,
    }
}

/// Returns the effects of the Play transport press outside Record mode: queue
/// the active program at the next measure, unless its playback is already
/// pending or it is the installed keys program.
fn transport_play_effects(state: &AppState, ctx: &Context) -> Vec<Effect> {
    let i = state.active_program_index;
    if ctx
        .status
        .has_pending(ctx.now, &WaveformSelector::ProgramPlayback(i))
        || state.keys.as_ref().is_some_and(|k| k.id == i)
    {
        vec![]
    } else {
        launch_effects(state, i, Launch::Pending)
    }
}

/// Returns the measure boundary a transport press at `ctx.now` refers to (see
/// [`recorder::boundary_for_press`]), or `None` when no boundary after now is
/// known.
fn press_boundary(ctx: &Context) -> Option<Instant> {
    // Unlike notes, transport presses carry no stamp from the MIDI callback,
    // so a press is timed when the reducer runs, up to one main-loop pass late.
    let next = player::next_measure_start(ctx.status, ctx.now)?;
    let previous = player::previous_measure_start(ctx.status, ctx.now);
    Some(recorder::boundary_for_press(
        ctx.now, previous, next, ctx.tempo,
    ))
}

/// Arms a take on the active program from Select mode, or tells why it can't.
///
/// Refuses silently outside Select mode. The take starts at the press's
/// measure boundary; the active program's pending playback is removed.
fn apply_arm(state: &mut AppState, ctx: &Context) -> Vec<Effect> {
    if !matches!(state.mode, Mode::Select) {
        return vec![];
    }
    let refuse = |message: String| vec![Effect::ShowMessage(message)];
    let i = state.active_program_index;
    let Some(keys_program) = state.keys.as_ref().map(|k| k.id) else {
        return refuse("No keys instrument installed".to_string());
    };
    if i == keys_program {
        return refuse("Can't record into the keys program".to_string());
    }
    let Some(keys_name) = state.programs.binding_name(keys_program) else {
        return refuse("Keys program has no name".to_string());
    };
    let keys_name = keys_name.to_string();
    let display_name = state.programs.display_name(i);
    if state.programs.resolve_name(ctx.environment, i, &keys_name) != BoundTo::Program(keys_program)
    {
        return refuse(format!("{} isn't in scope at {}", keys_name, display_name));
    }
    if state
        .programs
        .resolve_name(ctx.environment, i, recorder::PHRASE_HELPER)
        == BoundTo::Unbound
    {
        return refuse(format!(
            "{} not in scope (open std)",
            recorder::PHRASE_HELPER
        ));
    }
    if ctx.source_stale {
        return refuse("File changed on disk (reload before recording)".to_string());
    }
    let Some(start) = press_boundary(ctx) else {
        return vec![];
    };
    let take = Take::arm(keys_name.clone(), ctx.tempo, ctx.now, start);
    let message = match take.phase() {
        Phase::Armed => format!("Armed: recording {} into {}", keys_name, display_name),
        _ => format!("Recording {} into {}", keys_name, display_name),
    };
    // A queued playback would keep playing the old text through the take and
    // after the write.
    let mut effects = remove_pending_effects(state, ctx, i);
    effects.push(Effect::ShowMessage(message));
    state.mode = Mode::Record { take, keys_program };
    effects
}

/// Hands a transport press in Record mode to the take.
fn apply_take_transport(state: &mut AppState, ctx: &Context, transport: Transport) -> Vec<Effect> {
    let Mode::Record { take, keys_program } = &mut state.mode else {
        return vec![];
    };
    let keys_program = *keys_program;
    let finish = if transport == Transport::Play {
        Finish::Play
    } else {
        Finish::Silent
    };
    let response = match transport {
        Transport::Stop => take.stop(),
        Transport::Record | Transport::Play => {
            let Some(boundary) = press_boundary(ctx) else {
                return vec![];
            };
            take.finish(finish, ctx.now, boundary)
        }
    };
    let display_name = state.programs.display_name(state.active_program_index);
    match response {
        Response::Disarmed { play } => {
            state.mode = Mode::Select;
            if play {
                transport_play_effects(state, ctx)
            } else {
                vec![Effect::ShowMessage("Disarmed".to_string())]
            }
        }
        // The phrase isn't queued until the write, so the text a queue or an
        // un-queue would post is posted here.
        Response::Finishing { previous } => match (finish, previous) {
            (Finish::Play, _) => vec![Effect::ShowMessage(player::play_message(
                player::Played::Waveform,
                &display_name,
                state
                    .programs
                    .program(state.active_program_index)
                    .and_then(|p| p.loop_after_beats()),
            ))],
            (Finish::Silent, Some(Finish::Play)) => vec![Effect::ShowMessage(format!(
                "Removed pending waveform for program {}",
                display_name
            ))],
            (Finish::Silent, _) => vec![],
        },
        Response::Closed(closed) => write_take(state, ctx, keys_program, closed),
        Response::Discarded { notes } => {
            state.mode = Mode::Select;
            vec![Effect::ShowMessage(format!(
                "Discarded take ({})",
                note_count(notes)
            ))]
        }
    }
}

/// Advances the take in Record mode to `ctx.now`.
fn apply_take_tick(state: &mut AppState, ctx: &Context) -> Vec<Effect> {
    let Mode::Record { take, keys_program } = &mut state.mode else {
        return vec![];
    };
    let keys_program = *keys_program;
    match take.tick(ctx.now) {
        Tick::Waiting => vec![],
        Tick::Started => vec![Effect::ShowMessage(format!(
            "Recording {} into {}",
            take.keys_name(),
            state.programs.display_name(state.active_program_index)
        ))],
        Tick::Closed(closed) => write_take(state, ctx, keys_program, closed),
    }
}

/// Writes a closed take into the active program and returns to Select mode.
///
/// An empty take leaves the program untouched. Otherwise the phrase replaces
/// the program's text as one undo unit, and an empty slot takes the level of
/// the program at `keys_program`. With the play outcome, the phrase starts
/// repeating from the take's end boundary; an empty take queues the program
/// as the Play press does.
fn write_take(
    state: &mut AppState,
    ctx: &Context,
    keys_program: usize,
    closed: Closed,
) -> Vec<Effect> {
    // The take closes one margin before its end boundary, so a transport
    // press in that last margin lands in Select mode and acts on the written
    // program as it would outside a take.
    state.mode = Mode::Select;
    if closed.notes == 0 {
        let mut effects = vec![Effect::ShowMessage("Nothing recorded".to_string())];
        if closed.finish == Finish::Play {
            effects.extend(transport_play_effects(state, ctx));
        }
        return effects;
    }
    let i = state.active_program_index;
    let display_name = state.programs.display_name(i);
    let keys_level_db = state.programs.program(keys_program).map(|p| p.level_db());
    let Some(program) = state.programs.program_mut(i) else {
        return vec![];
    };
    if program.is_empty()
        && let Some(level_db) = keys_level_db
    {
        program.set_level_db(level_db);
    }
    let cursor = program.text().len();
    program.record_edit(cursor);
    program.set_text(closed.text);
    let mut on_success = Vec::new();
    if closed.finish == Finish::Play {
        on_success.push(Effect::PlayProgram {
            program_index: i,
            start: player::Start::At(closed.end),
            loop_after_beats: state.programs.program(i).and_then(|p| p.loop_after_beats()),
        });
    }
    vec![
        // Posted first so that an evaluation error or a refused save, which
        // set their own message, stay on screen. A play from an instant
        // posts nothing, so this is still the last message on success.
        Effect::ShowMessage(format!(
            "Recorded {} into {}",
            note_count(closed.notes),
            display_name
        )),
        Effect::EvaluateProgram {
            program_index: i,
            on_success,
            on_failure: vec![],
        },
        Effect::UpdateSource(i),
    ]
}

/// Returns `n` notes as text, e.g. "1 note" or "7 notes".
fn note_count(n: usize) -> String {
    if n == 1 {
        "1 note".to_string()
    } else {
        format!("{} notes", n)
    }
}

/// Returns the effect that plays the given program as `launch` says.
fn launch_effect(state: &AppState, program_index: usize, launch: Launch) -> Effect {
    let (start, loop_after_beats) = match launch {
        Launch::Pending => (
            player::Start::NextMeasure,
            state
                .programs
                .program(program_index)
                .and_then(|p| p.loop_after_beats()),
        ),
        Launch::Now => (player::Start::Now, None),
    };
    Effect::PlayProgram {
        program_index,
        start,
        loop_after_beats,
    }
}

/// Returns the effects that play the given program as `launch` says and
/// persist its source.
fn launch_effects(state: &AppState, program_index: usize, launch: Launch) -> Vec<Effect> {
    vec![
        launch_effect(state, program_index, launch),
        Effect::UpdateSource(program_index),
    ]
}

/// Returns the effects that stop the given program and persist its source,
/// or nothing if it isn't currently playing.
///
/// `UpdateSource` runs last so that a failed save's error message isn't
/// overwritten by the "Stopped program" status.
fn stop_program_effects(state: &AppState, ctx: &Context, i: usize) -> Vec<Effect> {
    if !ctx
        .status
        .has_active(ctx.now, &WaveformSelector::ProgramPlayback(i))
    {
        return vec![];
    }
    vec![
        Effect::StopProgram(i),
        Effect::ShowMessage(format!(
            "Stopped program {}",
            state.programs.display_name(i)
        )),
        Effect::UpdateSource(i),
    ]
}

/// Returns the effects that remove the given program's pending playback and
/// persist its source, or nothing if no playback is pending.
///
/// `UpdateSource` runs last so that a failed save's error message isn't
/// overwritten by the "Removed pending" status.
fn remove_pending_effects(state: &AppState, ctx: &Context, i: usize) -> Vec<Effect> {
    if !ctx
        .status
        .has_pending(ctx.now, &WaveformSelector::ProgramPlayback(i))
    {
        return vec![];
    }
    vec![
        Effect::RemovePendingProgram(i),
        Effect::ShowMessage(format!(
            "Removed pending waveform for program {}",
            state.programs.display_name(i)
        )),
        Effect::UpdateSource(i),
    ]
}

fn apply_select_program(state: &mut AppState, i: usize) -> Vec<Effect> {
    if i >= state.programs.programs().len() {
        return vec![];
    }
    let changed = state.active_program_index != i;
    state.active_program_index = i;
    // Replace the previous status with the newly-selected program's
    // selection message. Navigation represents a fresh context, so any prior
    // "Removed pending..." / "Playing..." etc. shouldn't carry over.
    let mut effects = vec![Effect::ShowMessage(state.programs.selection_message(i))];
    if changed {
        effects.push(Effect::SyncEncoders);
    }
    effects
}

/// Toggles a sixteenth of the active program's step list: rewrites the list
/// literal in the program's text, re-evaluates and persists it, and — when the
/// program is playing — schedules or removes that one step so the change sounds
/// as soon as possible.
fn apply_toggle_sequencer_step(state: &mut AppState, ctx: &Context, sixteenth: u8) -> Vec<Effect> {
    if sixteenth >= (sequencer::MAX_PAGE + 1) * sequencer::SIXTEENTHS_PER_PAGE {
        return vec![];
    }
    // A toggle both edits and persists; refuse on top of external changes.
    if ctx.source_stale {
        return vec![Effect::ShowMessage(
            "File changed on disk (reload before editing)".to_string(),
        )];
    }
    let i = state.active_program_index;
    let display_name = state.programs.display_name(i);
    let Some(program) = state.programs.program(i) else {
        return vec![];
    };
    let Some(shape) = sequencer::analyze(program.text()) else {
        return vec![Effect::ShowMessage(format!(
            "{} is not sequenceable (expected [steps] | on_beats(w))",
            display_name
        ))];
    };
    let edit = sequencer::toggle_sixteenth_text(program.text(), &shape, sixteenth);

    let program = state
        .programs
        .program_mut(i)
        .expect("program existed above");
    program.record_edit(program.text().len());
    program.set_text(edit.new_text);

    let live = sequence_is_live(ctx, i);

    // The message waits for success so that a failure's error stays on screen,
    // and the play because it reads the result. A removal reads nothing, so it
    // runs either way.
    let mut on_success = vec![Effect::ShowMessage(format!(
        "{} beat {} {}",
        display_name,
        edit.beat,
        if edit.turned_on { "on" } else { "off" }
    ))];
    let mut on_failure = Vec::new();
    if live {
        // Steps already playing keep repeating without another Play, so their
        // marks must follow the new text here.
        on_success.push(Effect::RecordSequencerStepMarks(i));
        if edit.turned_on {
            on_success.push(Effect::PlaySequencerStep {
                program_index: i,
                sixteenth,
            });
        } else {
            for effects in [&mut on_success, &mut on_failure] {
                effects.push(Effect::RemovePendingSequencerStep {
                    program_index: i,
                    sixteenth,
                });
            }
        }
    }
    vec![
        Effect::EvaluateProgram {
            program_index: i,
            on_success,
            on_failure,
        },
        Effect::UpdateSource(i),
    ]
}

/// Returns whether the program at `program_index` is playing as a sequence:
/// its anchor step is sounding or queued.
fn sequence_is_live(ctx: &Context, program_index: usize) -> bool {
    let anchor = WaveformSelector::Only(WaveformId::Step {
        program: program_index,
        sixteenth: ANCHOR_SIXTEENTH,
    });
    ctx.status.has_active(ctx.now, &anchor) || ctx.status.has_pending(ctx.now, &anchor)
}

/// Returns the sixteenths holding a step of `text`, or none when `text` isn't
/// sequenceable.
fn step_sixteenths(text: &str) -> BTreeSet<u8> {
    sequencer::analyze(text)
        .map(|shape| {
            shape
                .steps
                .iter()
                .map(|&(beat, _)| sequencer::sixteenth_for_beat(beat))
                .collect()
        })
        .unwrap_or_default()
}

fn apply_install_keys(state: &mut AppState, program_index: usize) -> Vec<Effect> {
    // Applying with the currently-installed program uninstalls it; for any
    // other program, tries to install that one in its place.
    if let Some(keys) = &state.keys
        && keys.id == program_index
    {
        state.keys = None;
        return vec![
            // Stop every held key; their stored note-off waveforms are gone
            // with state.keys.
            Effect::ModifyWaveform {
                selector: WaveformSelector::AllKeys,
                mark_id: MarkId::Terminator,
                waveform: player::stop_ramp(),
            },
            Effect::ShowMessage("Uninstalled keys".to_string()),
        ];
    }
    vec![Effect::InstallKeys(program_index)]
}

/// Applies `Action::ToggleKeysAnnotation`.
fn apply_toggle_keys_annotation(
    state: &mut AppState,
    ctx: &Context,
    program_index: usize,
) -> Vec<Effect> {
    let name = state.programs.display_name(program_index);
    if state.programs.skip_slots(program_index).is_none() {
        return vec![Effect::ShowMessage(format!("{} is empty", name))];
    }
    if ctx.source_stale {
        return vec![Effect::ShowMessage(state.programs.stale_message())];
    }
    let Some(program) = state.programs.program_mut(program_index) else {
        return vec![];
    };
    if program.loop_after_beats().is_some() {
        return vec![Effect::ShowMessage(format!("Remove {}'s Loop first", name))];
    }
    let mut effects = vec![];
    let message = match program.kind() {
        programs::ProgramKind::Waveform => {
            program.set_kind(programs::ProgramKind::Keys);
            format!("Added keys to {}", name)
        }
        programs::ProgramKind::Keys => {
            program.set_kind(programs::ProgramKind::Waveform);
            if state.keys.as_ref().is_some_and(|k| k.id == program_index) {
                effects = apply_install_keys(state, program_index);
                // Keys mode plays the instrument just uninstalled.
                if matches!(state.mode, Mode::Keys) {
                    state.mode = Mode::Select;
                }
                format!("Removed keys from {} and uninstalled it", name)
            } else {
                format!("Removed keys from {}", name)
            }
        }
    };
    effects.extend([
        Effect::ShowMessage(message),
        Effect::EvaluateProgram {
            program_index,
            on_success: vec![],
            on_failure: vec![],
        },
        Effect::UpdateSource(program_index),
    ]);
    effects
}

/// Returns why the Loop of the program at `program_index` can't be changed:
/// the slot is empty, the backing file is stale, or the program plays keys.
/// Returns `None` when it can be changed.
pub fn loop_refusal(state: &AppState, ctx: &Context, program_index: usize) -> Option<String> {
    let name = state.programs.display_name(program_index);
    if state.programs.skip_slots(program_index).is_none() {
        return Some(format!("{} is empty", name));
    }
    if ctx.source_stale {
        return Some(state.programs.stale_message());
    }
    let program = state.programs.program(program_index)?;
    (program.kind() == programs::ProgramKind::Keys)
        .then(|| format!("{} plays keys, so it can't loop", name))
}

/// Returns the message for setting the Loop of the program named
/// `program_name` to `beats`, or to play once when `beats` is `None`.
///
/// While the program is `playing`, the message says the change waits for its
/// next launch.
///
/// # Example
///
/// ```
/// use tuun::actions::loop_message;
/// assert_eq!(loop_message("kick", Some(4), false), "kick loops every 4 beats");
/// assert_eq!(
///     loop_message("kick", None, true),
///     "kick plays once from its next launch"
/// );
/// ```
pub fn loop_message(program_name: &str, beats: Option<u32>, playing: bool) -> String {
    let setting = match beats {
        Some(beats) => format!("loops every {} beats", beats),
        None => "plays once".to_string(),
    };
    let when = if playing { " from its next launch" } else { "" };
    format!("{} {}{}", program_name, setting, when)
}

/// Applies `Action::SetLoopAfterBeats`.
fn apply_set_loop_after_beats(
    state: &mut AppState,
    ctx: &Context,
    program_index: usize,
    beats: Option<u32>,
) -> Vec<Effect> {
    if let Some(refusal) = loop_refusal(state, ctx, program_index) {
        return vec![Effect::ShowMessage(refusal)];
    }
    let name = state.programs.display_name(program_index);
    let Some(program) = state.programs.program_mut(program_index) else {
        return vec![];
    };
    program.set_loop_after_beats(beats);
    let playback = WaveformSelector::ProgramPlayback(program_index);
    let playing =
        ctx.status.has_active(ctx.now, &playback) || ctx.status.has_pending(ctx.now, &playback);
    vec![
        Effect::ShowMessage(loop_message(&name, beats, playing)),
        Effect::UpdateSource(program_index),
    ]
}

/// Applies `Action::ChangeSkipSlots`.
fn apply_change_skip_slots(
    state: &mut AppState,
    ctx: &Context,
    program_index: usize,
    delta: i32,
) -> Vec<Effect> {
    let name = state.programs.display_name(program_index);
    let Some(skip_slots) = state.programs.skip_slots(program_index) else {
        return vec![Effect::ShowMessage(format!("{} is empty", name))];
    };
    if ctx.source_stale {
        return vec![Effect::ShowMessage(state.programs.stale_message())];
    }
    let Some(new_skip_slots) = skip_slots.checked_add_signed(delta) else {
        return vec![Effect::ShowMessage(format!(
            "Can't move {} up: no empty slot above it",
            name
        ))];
    };
    let last_slot = state.programs.last_slot().unwrap_or(program_index);
    if last_slot as i64 + delta as i64 >= state.programs.programs().len() as i64 {
        return vec![Effect::ShowMessage(format!(
            "Can't move {} down: a program would pass the last slot",
            name
        ))];
    }
    vec![Effect::MoveProgram {
        program_index,
        skip_slots: new_skip_slots,
        select: (program_index as i64 + delta as i64) as usize,
    }]
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
/// the text before the cursor: an identifier fragment (optionally qualified by
/// a `module.` projection) starts a completion cycle or a `(` inserts a
/// parameter hint. See [`Action::Complete`] for the user-facing behavior.
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
    let qualifier = projection_qualifier(before, fragment_start);
    if fragment_start == cursor && qualifier.is_none() {
        if before.ends_with('(') {
            return apply_parameter_hint(state, ctx, cursor);
        }
        return vec![Effect::ShowMessage(
            "Nothing to complete (the cursor must follow an identifier or \"(\")".to_string(),
        )];
    }

    // A fragment (possibly empty when after a `.`) ends at the cursor: start a
    // new cycle.
    let fragment = before[fragment_start..].to_string();
    let context = match ctx
        .environment
        .program_context(&state.programs, state.active_program_index)
    {
        Ok(context) => context,
        Err(error) => return vec![Effect::ShowMessage(format!("Can't complete: {}", error))],
    };
    // A fragment after a `.` is a projection name and completes only among
    // the qualifying module's bindings — when the qualifier doesn't resolve
    // to a module value there is nothing the projection could mean, so
    // nothing to offer (`time.f`, `.sq`, or the `0.` inside a float). A
    // fragment without a `.` completes among the in-scope names.
    let candidates: Vec<String> = match &qualifier {
        Some(path) => match resolve_module_path(&context, path) {
            Some(entries) => entries
                .iter()
                .rev()
                .filter(|(name, _)| name.starts_with(&fragment) && *name != fragment)
                .map(|(name, _)| name.clone())
                .collect(),
            None if path.is_empty() => {
                return vec![Effect::ShowMessage(
                    "Nothing to complete (no module name before '.')".to_string(),
                )];
            }
            None => {
                return vec![Effect::ShowMessage(format!(
                    "\"{}\" is not a module",
                    path.join(".")
                ))];
            }
        },
        None => {
            let mut seen = HashSet::new();
            let mut candidates = Vec::new();
            for (name, _) in context.iter().rev() {
                // Walking from the end, the first occurrence of a name is the
                // live binding; any earlier occurrence is shadowed.
                if seen.insert(name.clone()) && name.starts_with(&fragment) && *name != fragment {
                    candidates.push(name.clone());
                }
            }
            candidates
        }
    };
    if candidates.is_empty() {
        let message = match &qualifier {
            Some(path) => format!(
                "No completions for \"{}\" in module \"{}\"",
                fragment,
                path.join(".")
            ),
            None => format!("No completions for \"{}\"", fragment),
        };
        return vec![Effect::ShowMessage(message)];
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

/// The range of the token around `cursor`: the run of like characters the
/// cursor sits in or just past, or `None` where it touches neither kind.
///
/// Names and operators are separate runs, so a cursor between the two names
/// whichever it just left — `x*|y` is the operator, `x|*y` the name. Both
/// are variables to the checker, so both have a type worth reporting.
fn token_at(text: &str, cursor: usize) -> Option<Range<usize>> {
    let before = text[..cursor].chars().next_back();
    let at = text[cursor..].chars().next();
    let class: fn(char) -> bool = match (before, at) {
        (Some(c), _) if is_word_char(c) => is_word_char,
        (Some(c), _) if is_operator_char(c) => is_operator_char,
        (_, Some(c)) if is_word_char(c) => is_word_char,
        (_, Some(c)) if is_operator_char(c) => is_operator_char,
        _ => return None,
    };
    let start = text[..cursor].trim_end_matches(class).len();
    let after = &text[cursor..];
    let end = cursor + after.len() - after.trim_start_matches(class).len();
    (start < end).then_some(start..end)
}

/// Applies `Action::ShowType` in Edit mode; does nothing in other modes.
///
/// Reports the type of the name or operator the cursor is in, or just past.
/// See [`Action::ShowType`] for the user-facing behavior.
fn apply_show_type(state: &mut AppState, ctx: &Context) -> Vec<Effect> {
    let Mode::Edit {
        cursor_position, ..
    } = &state.mode
    else {
        return vec![];
    };
    let cursor = *cursor_position;
    let text = state.active_program().text();
    let Some(token) = token_at(text, cursor) else {
        return vec![Effect::ShowMessage(
            "No name or operator under the cursor".to_string(),
        )];
    };
    let name = text[token.clone()].to_string();
    match ctx
        .environment
        .type_at(&state.programs, state.active_program_index, token.start)
    {
        Some(ty) => vec![Effect::ShowMessage(format!("{} : {}", name, ty))],
        None => vec![Effect::ShowMessage(format!("No type for \"{}\"", name))],
    }
}

/// Inserts a parameter hint for the call being typed: with the Edit-mode
/// `cursor` just after `(` and the identifier before it (possibly a module
/// projection) bound to a function, inserts that function's positional
/// parameters, its named parameters as `name = <default value>` pairs, and a
/// closing `)`.
fn apply_parameter_hint(state: &mut AppState, ctx: &Context, cursor: usize) -> Vec<Effect> {
    let text = state.active_program().text();
    let head = &text[..cursor - 1];
    let name_start = head.trim_end_matches(is_word_char).len();
    let name = head[name_start..].to_string();
    if name.is_empty() {
        return vec![Effect::ShowMessage("Nothing to complete".to_string())];
    }
    let qualifier = projection_qualifier(head, name_start);
    let context = match ctx
        .environment
        .program_context(&state.programs, state.active_program_index)
    {
        Ok(context) => context,
        Err(error) => return vec![Effect::ShowMessage(format!("Can't complete: {}", error))],
    };
    // A name qualified by a projection (`pm.osc(`) is looked up in that
    // module's bindings — a `.` whose qualifier doesn't resolve to a module
    // value can't be hinted; otherwise the last occurrence of the name in
    // the context is the live binding.
    let (display, value) = match &qualifier {
        Some(path) => match resolve_module_path(&context, path) {
            Some(entries) => {
                let display = format!("{}.{}", path.join("."), name);
                match entries.iter().find(|(n, _)| *n == name) {
                    Some((_, value)) => (display, value),
                    None => {
                        return vec![Effect::ShowMessage(format!(
                            "\"{}\" is not defined",
                            display
                        ))];
                    }
                }
            }
            None if path.is_empty() => {
                return vec![Effect::ShowMessage(
                    "Nothing to complete (no module name before '.')".to_string(),
                )];
            }
            None => {
                return vec![Effect::ShowMessage(format!(
                    "\"{}\" is not a module",
                    path.join(".")
                ))];
            }
        },
        None => match context.iter().rev().find(|(n, _)| *n == name) {
            Some((_, value)) => (name.clone(), value),
            None => return vec![Effect::ShowMessage(format!("\"{}\" is not defined", name))],
        },
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
        // signature metadata so calls like `phase(` can be hinted too.
        expr::Expr::BuiltIn { name, .. } => vec![Effect::ShowMessage(format!(
            "No parameter hint for built-in \"{}\"",
            name
        ))],
        _ => vec![Effect::ShowMessage(format!(
            "\"{}\" is not a function",
            display
        ))],
    }
}

/// Applies `Action::Undo` or `Action::Redo`: `restore` moves the active
/// program's state between its undo and redo stacks and returns the restored
/// cursor.
///
/// In Edit mode, the Edit-mode cursor, parse errors, status message, and
/// completion cycle are refreshed, mirroring `edit_text_op`. In Record mode it
/// does nothing. In every other mode the restored text is re-evaluated and
/// persisted, and `verb` names the change in the status message ("Undid change
/// to kick"). When there is nothing to restore, shows `empty_message`.
fn apply_history_restore(
    state: &mut AppState,
    ctx: &Context,
    restore: impl FnOnce(&mut Program, usize) -> Option<usize>,
    verb: &str,
    empty_message: &str,
) -> Vec<Effect> {
    let cursor = match &state.mode {
        Mode::Edit {
            cursor_position, ..
        } => *cursor_position,
        Mode::Record { .. } => return vec![],
        _ => return apply_committed_history_restore(state, ctx, restore, verb, empty_message),
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

/// Applies `restore` to the active program outside Edit mode, then re-evaluates
/// and persists the restored text.
///
/// When the program is playing as a sequence, steps the restore added are
/// scheduled and steps it removed have their queued repeats cancelled, as for
/// `Action::ToggleSequencerStep`. Refuses when the source file has changed on
/// disk.
fn apply_committed_history_restore(
    state: &mut AppState,
    ctx: &Context,
    restore: impl FnOnce(&mut Program, usize) -> Option<usize>,
    verb: &str,
    empty_message: &str,
) -> Vec<Effect> {
    // A restore both edits and persists; refuse on top of external changes.
    if ctx.source_stale {
        return vec![Effect::ShowMessage(
            "File changed on disk (reload before editing)".to_string(),
        )];
    }
    let i = state.active_program_index;
    let display_name = state.programs.display_name(i);
    let Some(program) = state.programs.program_mut(i) else {
        return vec![];
    };
    // There is no Edit-mode cursor to save, so the redo snapshot's cursor is
    // the end of the text, as for a recorded take.
    let cursor = program.text().len();
    let old_text = program.text().to_string();
    if restore(program, cursor).is_none() {
        return vec![Effect::ShowMessage(empty_message.to_string())];
    }
    // Playing steps repeat independently of the text, so bring them in line
    // with the restored step list. A play reads the result, so it waits for
    // success; a removal reads nothing, so it runs either way.
    let mut on_success = Vec::new();
    let mut on_failure = Vec::new();
    if sequence_is_live(ctx, i) {
        let old_steps = step_sixteenths(&old_text);
        let new_steps = step_sixteenths(state.active_program().text());
        // Steps already playing keep repeating without another Play, so their
        // marks must follow the restored text here.
        on_success.push(Effect::RecordSequencerStepMarks(i));
        on_success.extend(new_steps.difference(&old_steps).map(|&sixteenth| {
            Effect::PlaySequencerStep {
                program_index: i,
                sixteenth,
            }
        }));
        for effects in [&mut on_success, &mut on_failure] {
            effects.extend(old_steps.difference(&new_steps).map(|&sixteenth| {
                Effect::RemovePendingSequencerStep {
                    program_index: i,
                    sixteenth,
                }
            }));
        }
    }
    vec![
        // Posted first so that an evaluation error or a refused save, which
        // set their own message, stay on screen.
        Effect::ShowMessage(format!("{} change to {}", verb, display_name)),
        Effect::EvaluateProgram {
            program_index: i,
            on_success,
            on_failure,
        },
        Effect::UpdateSource(i),
    ]
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

/// Whether `c` can appear in an operator.
///
/// The union of the characters the parser's binary and unary operator tags
/// are spelled from, so a run of them covers the longest operator the cursor
/// could be sitting in (`<=`, `~*`, `!=`).
fn is_operator_char(c: char) -> bool {
    matches!(
        c,
        '*' | '/' | '~' | '+' | '-' | '&' | '=' | '!' | '<' | '>' | '|' | '\\' | '@' | '$' | '%'
    ) || c == '?'
}

/// Returns the dotted path qualifying the word starting at `start` in `text`
/// (the `["a", "b"]` in `a.b.osc`) outermost component first.
///
/// `None` when the word doesn't follow a `.` at all; `Some(vec![])` when it
/// does but no module name precedes the dot (`.x`, or `f(x).y` where the
/// module is not a name), so the projection has no resolvable path.
fn projection_qualifier(text: &str, start: usize) -> Option<Vec<String>> {
    if !text[..start].ends_with('.') {
        return None;
    }
    let mut path = Vec::new();
    let mut end = start;
    while let Some(head) = text[..end].strip_suffix('.') {
        let word_start = head.trim_end_matches(is_word_char).len();
        if word_start == head.len() {
            break;
        }
        path.insert(0, head[word_start..].to_string());
        end = word_start;
    }
    Some(path)
}

/// Returns the bindings of the module value that `path` names in `context`: the
/// last binding of the path's first component, projected through nested module
/// values for the remaining components. `None` when `path` is empty or any step
/// is unbound or not a module value.
fn resolve_module_path<'a, M, S>(
    context: &'a [expr::NamedExpr<M, S>],
    path: &[String],
) -> Option<&'a [expr::NamedExpr<M, S>]> {
    let (first, rest) = path.split_first()?;
    // The last occurrence of the name is the live binding.
    let (_, value) = context.iter().rev().find(|(n, _)| n == first)?;
    let mut entries = match &value.expr {
        expr::Expr::BoundModule(entries) => entries.as_slice(),
        _ => return None,
    };
    for component in rest {
        // Module entries have unique names, so any occurrence is the binding.
        let (_, value) = entries.iter().find(|(n, _)| n == component)?;
        entries = match &value.expr {
            expr::Expr::BoundModule(entries) => entries.as_slice(),
            _ => return None,
        };
    }
    Some(entries)
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
        mark: MarkId::Slider {
            program: program_index,
            label: label.clone(),
        },
        value: actual_value,
    }];

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
    let amplitude = player::amplitude(level_db);

    let mut effects = vec![Effect::ModifyWaveform {
        selector: WaveformSelector::ProgramVoices(program_index),
        mark_id: MarkId::Amplitude,
        waveform: amplitude.clone(),
    }];

    // Mirror onto installed keys.
    //
    // XXX Should this go through the slider pipeline like slider updates do
    // (coalesce per quantum, ramp from prior value)? Currently each call
    // sends a Const directly to the tracker, which can cause a step change.
    if let Some(keys) = state.keys.as_mut()
        && keys.id == program_index
    {
        effects.push(Effect::ModifyWaveform {
            selector: WaveformSelector::AllKeys,
            mark_id: MarkId::Amplitude,
            waveform: amplitude,
        });
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

    /// A tracker status with no waveforms at all — nothing active, nothing pending.
    fn empty_status() -> tracker::Status<WaveformId> {
        tracker::Status {
            buffer_start: Instant::now(),
            start_times: Vec::new(),
            tracker_load: None,
            allocations: None,
        }
    }

    /// A status with program 0's waveform starting at `start` (before `now` =
    /// active, after `now` = pending).
    fn status_with_start(start: Instant) -> tracker::Status<WaveformId> {
        let mut status = empty_status();
        status.start_times.push((WaveformId::Program(0), start));
        status
    }

    /// Applies `action` against `status`, timestamped `now`. The environment
    /// has the standard prelude but no library root, so `open`s of anything
    /// but the prelude fail to resolve.
    fn apply_with_status(
        state: &mut AppState,
        status: &tracker::Status<WaveformId>,
        now: Instant,
        action: Action,
    ) -> Vec<Effect> {
        let environment = environment::Environment::new(48000, 120, std::path::PathBuf::new());
        let ctx = Context {
            status,
            now,
            environment: &environment,
            source_stale: false,
            tempo: 60,
        };
        apply(state, &ctx, action)
    }

    /// Applies `action` against an empty tracker status.
    fn apply_with_empty_status(state: &mut AppState, action: Action) -> Vec<Effect> {
        // XXX should we use Instant::now() or buffer_start?
        apply_with_status(state, &empty_status(), Instant::now(), action)
    }

    /// Applies `action` with the backing file flagged as changed on disk.
    fn apply_with_stale_source(state: &mut AppState, action: Action) -> Vec<Effect> {
        let environment = environment::Environment::new(48000, 120, std::path::PathBuf::new());
        let status = empty_status();
        let ctx = Context {
            status: &status,
            now: Instant::now(),
            environment: &environment,
            source_stale: true,
            tempo: 60,
        };
        apply(state, &ctx, action)
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

    /// Returns a state whose one program loops every 8 beats.
    fn looping_state() -> AppState {
        AppState::from_source(
            "#{loop_after_beats=8}\n_ = test;".to_string(),
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
                selector: WaveformSelector::ProgramVoices(0),
                mark_id: MarkId::Amplitude,
                ..
            }
        ));
    }

    #[test]
    fn level_change_on_keys_program_emits_all_keys_amplitude_modify() {
        let mut state = test_state();
        install_test_keys(&mut state);
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetLevelDb {
                program: 0,
                level_db: -6.0,
            },
        );
        assert!(
            effects.iter().any(|e| matches!(
                e,
                Effect::ModifyWaveform {
                    selector: WaveformSelector::AllKeys,
                    mark_id: MarkId::Amplitude,
                    ..
                }
            )),
            "expected an AllKeys Amplitude ModifyWaveform, got {:?}",
            effects
        );
    }

    #[test]
    fn uninstall_keys_stops_all_held_keys() {
        let mut state = test_state();
        install_test_keys(&mut state);
        let effects = apply_with_empty_status(&mut state, Action::ToggleInstalledKeys(0));
        assert!(state.keys.is_none());
        assert!(
            matches!(
                effects[0],
                Effect::ModifyWaveform {
                    selector: WaveformSelector::AllKeys,
                    mark_id: MarkId::Terminator,
                    ..
                }
            ),
            "expected an AllKeys Terminator ModifyWaveform first, got {:?}",
            effects
        );
        assert!(
            matches!(&effects[1], Effect::ShowMessage(m) if m == "Uninstalled keys"),
            "expected the uninstall message second, got {:?}",
            effects
        );
    }

    #[test]
    fn install_toggle_on_uninstalled_program_emits_install_keys() {
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::ToggleInstalledKeys(0));
        assert!(
            matches!(effects[0], Effect::InstallKeys(0)),
            "expected InstallKeys, got {:?}",
            effects
        );
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
            "not_fo = 1;\nnot_fu = 2;\nnot_foo = 3;\n#{level_db=0}\n_ = test;",
            "not_f",
            5,
        );
        for (text, cursor) in [
            ("not_foo", 7), // the most recently bound match comes first
            ("not_fu", 6),
            ("not_fo", 6),
            ("not_f", 5), // the ring wraps through the original fragment
            ("not_foo", 7),
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
        // cSpell:disable
        let mut state = edit_state("#{level_db=0}\n_ = test;", "appe", 4);
        // cSpell:enable
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("append".to_string(), 6));
    }

    /// The message a `ShowType` produced; panics if it produced none.
    #[track_caller]
    fn show_type_message(source: &str, text: &str, cursor: usize) -> String {
        let mut state = edit_state(source, text, cursor);
        let effects = apply_with_empty_status(&mut state, Action::ShowType);
        effects
            .into_iter()
            .find_map(|effect| match effect {
                Effect::ShowMessage(message) => Some(message),
                _ => None,
            })
            .expect("ShowType should report something")
    }

    #[test]
    fn show_type_reports_the_identifier_under_the_cursor() {
        let source = "#{level_db=0}\n_ = test;";
        // A prelude name, with the cursor inside it and just past it.
        assert_eq!(show_type_message(source, "time", 1), "time : waveform");
        assert_eq!(show_type_message(source, "time", 4), "time : waveform");
        // A parameter reads as what the body settled it to, not as an
        // unknown — the point of reporting after inference finishes.
        assert_eq!(
            show_type_message(source, "fn(dur) => time | fin(dur)", 23),
            "dur : waveform"
        );
        // A let-bound name, and an overloaded one, both from inside the
        // expression rather than from the program's bindings.
        assert_eq!(
            show_type_message(source, "let n = 1 in n + 2", 13),
            "n : int"
        );
    }

    #[test]
    fn show_type_reports_operators_too() {
        let source = "#{level_db=0}\n_ = test;";
        // An operator is a variable to the checker, so it has a type worth
        // reporting — the table its overloads make.
        assert_eq!(
            show_type_message(source, "1 <= 2", 2),
            "<= : (float, float) -> bool"
        );
        // A run of operator characters is one token, so the cursor inside
        // `<=` names the whole operator rather than `<`.
        assert_eq!(
            show_type_message(source, "1 <= 2", 3),
            "<= : (float, float) -> bool"
        );
        // Unary and binary uses share one name and one table.
        assert!(show_type_message(source, "-time", 0).starts_with("- : (int) -> int"));
        // Names and operators are separate tokens, so a cursor between them
        // names whichever it just left.
        assert_eq!(show_type_message(source, "time+1", 4), "time : waveform");
        assert!(show_type_message(source, "time+1", 5).starts_with("+ : "));
        // `\\` is a variable like the rest, which needs its span from the
        // parser.
        assert_eq!(
            show_type_message(source, "seq(0)(1) \\ 1", 10),
            "\\ : (seq, seq) -> seq ∧ (seq, waveform) -> waveform"
        );
    }

    #[test]
    fn show_type_reports_when_there_is_nothing_to_report() {
        let source = "#{level_db=0}\n_ = test;";
        // Touching neither a name nor an operator.
        assert_eq!(
            show_type_message(source, "1  + 2", 2),
            "No name or operator under the cursor"
        );
        // A name the checker has no type for.
        assert_eq!(
            show_type_message(source, "nope + 1", 2),
            "No type for \"nope\""
        );
        // `|` is reverse application, not a function: the parser applies the
        // right-hand side directly, so there is no variable to report on.
        assert_eq!(
            show_type_message(source, "time | fin(1)", 5),
            "No type for \"|\""
        );
    }

    #[test]
    fn completion_cycle_resets_after_another_action() {
        let mut state = edit_state(
            "not_ba = 1;\nnot_baz = 2;\n#{level_db=0}\n_ = test;",
            "not_b",
            5,
        );
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("not_baz".to_string(), 7));
        // Any cursor motion ends the cycle: the next completion starts fresh
        // from the (now fully typed) fragment "not_baz", which matches nothing.
        apply_with_empty_status(&mut state, Action::MoveCursorBy(-1));
        apply_with_empty_status(&mut state, Action::MoveCursorBy(1));
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("not_baz".to_string(), 7));
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
        let mut state = edit_state("#{level_db=0}\n_ = test;", "log(", 4);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("log(".to_string(), 4));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("built-in")),
            "expected a built-in message, got {:?}",
            effects
        );
    }

    /// Writes a library root holding modules for completion tests — `pm`
    /// with two `piano_*` bindings and `outer` re-exporting `pm` as
    /// `alias` — and returns that root.
    ///
    /// The files are written once per process: tests sharing the root run
    /// in parallel, and a rewrite is not atomic, so a concurrent reader
    /// could otherwise see a truncated module.
    fn completion_library_root() -> std::path::PathBuf {
        static ROOT: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();
        ROOT.get_or_init(|| {
            let dir = std::env::temp_dir().join("tuun_test_completion_modules");
            std::fs::create_dir_all(&dir).expect("temp module dir");
            std::fs::write(
                dir.join("pm.tuun"),
                "piano_keys = fn(freq, gain = 1 + 1) => freq;\npiano_bass = 2;\n",
            )
            .expect("write pm module");
            std::fs::write(dir.join("outer.tuun"), "use pm;\nalias = pm;\n")
                .expect("write outer module");
            dir
        })
        .clone()
    }

    /// Applies `action` with an environment resolving modules under `root`.
    fn apply_with_library_root(
        state: &mut AppState,
        root: std::path::PathBuf,
        action: Action,
    ) -> Vec<Effect> {
        let environment = environment::Environment::new(48000, 120, root);
        let status = empty_status();
        let ctx = Context {
            status: &status,
            now: Instant::now(),
            environment: &environment,
            source_stale: false,
            tempo: 60,
        };
        apply(state, &ctx, action)
    }

    #[test]
    fn complete_qualified_fragment_uses_module_bindings() {
        let mut state = edit_state("use pm;\n#{level_db=0}\n_ = test;", "pm.pia", 6);
        let root = completion_library_root();
        for (text, cursor) in [
            ("pm.piano_bass", 13), // the module's most recent match comes first
            ("pm.piano_keys", 13),
            ("pm.pia", 6), // the ring wraps through the original fragment
        ] {
            apply_with_library_root(&mut state, root.clone(), Action::Complete);
            assert_eq!(edit_text_and_cursor(&state), (text.to_string(), cursor));
        }
    }

    #[test]
    fn complete_after_module_dot_cycles_all_bindings() {
        let mut state = edit_state("use pm;\n#{level_db=0}\n_ = test;", "pm.", 3);
        let root = completion_library_root();
        apply_with_library_root(&mut state, root.clone(), Action::Complete);
        assert_eq!(
            edit_text_and_cursor(&state),
            ("pm.piano_bass".to_string(), 13)
        );
        apply_with_library_root(&mut state, root, Action::Complete);
        assert_eq!(
            edit_text_and_cursor(&state),
            ("pm.piano_keys".to_string(), 13)
        );
    }

    #[test]
    fn complete_through_nested_module_values() {
        let mut state = edit_state(
            "use outer;\n#{level_db=0}\n_ = test;",
            "outer.alias.piano_k",
            19,
        );
        let root = completion_library_root();
        apply_with_library_root(&mut state, root, Action::Complete);
        assert_eq!(
            edit_text_and_cursor(&state),
            ("outer.alias.piano_keys".to_string(), 22)
        );
    }

    #[test]
    fn complete_qualified_fragment_without_matches_names_the_module() {
        let mut state = edit_state("use pm;\n#{level_db=0}\n_ = test;", "pm.zzz", 6);
        let root = completion_library_root();
        let effects = apply_with_library_root(&mut state, root, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("pm.zzz".to_string(), 6));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m)
                if m == "No completions for \"zzz\" in module \"pm\""),
            "expected a no-completions message, got {:?}",
            effects
        );
    }

    #[test]
    fn complete_after_non_module_qualifier_reports_not_a_module() {
        // `time` is bound, but not to a module value: no projection of it
        // can evaluate, so the fragment must not complete against the
        // context (`time.f` must not become `time.filter`).
        let mut state = edit_state("#{level_db=0}\n_ = test;", "time.f", 6);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("time.f".to_string(), 6));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "\"time\" is not a module"),
            "expected a not-a-module message, got {:?}",
            effects
        );

        // Same for an unbound qualifier.
        let mut state = edit_state("#{level_db=0}\n_ = test;", "x.sin", 5);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("x.sin".to_string(), 5));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "\"x\" is not a module"),
            "expected a not-a-module message, got {:?}",
            effects
        );
    }

    #[test]
    fn complete_after_number_dot_reports_not_a_module() {
        // Mid-float (`0.`) the dot looks like a projection of `0`; there
        // is nothing to complete there, only a (truthful) message.
        let mut state = edit_state("#{level_db=0}\n_ = test;", "_ = 0.", 6);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("_ = 0.".to_string(), 6));
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "\"0\" is not a module"),
            "expected a not-a-module message, got {:?}",
            effects
        );
    }

    #[test]
    fn complete_after_dot_without_module_name_refuses() {
        // A fragment after a `.` is a projection name; with no module name
        // before the dot there is nothing it could complete to (`.s` must
        // not become `.square`). Same for the bare dot and for a dot after
        // a call result, which can't be resolved statically.
        for (text, cursor) in [(".s", 2), (".", 1), ("fin(1).s", 8)] {
            let mut state = edit_state("#{level_db=0}\n_ = test;", text, cursor);
            let effects = apply_with_empty_status(&mut state, Action::Complete);
            assert_eq!(edit_text_and_cursor(&state), (text.to_string(), cursor));
            assert!(
                matches!(&effects[0], Effect::ShowMessage(m)
                    if m == "Nothing to complete (no module name before '.')"),
                "for {:?}: expected a no-module-name message, got {:?}",
                text,
                effects
            );
        }
    }

    #[test]
    fn parameter_hint_for_non_module_qualifier_reports_not_a_module() {
        // Without this, `time.filter(` would hint the parameters of the
        // unrelated bare `filter` binding.
        let mut state = edit_state("#{level_db=0}\n_ = test;", "time.filter(", 12);
        let effects = apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(
            edit_text_and_cursor(&state),
            ("time.filter(".to_string(), 12)
        );
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "\"time\" is not a module"),
            "expected a not-a-module message, got {:?}",
            effects
        );
    }

    #[test]
    fn complete_after_open_paren_hints_module_binding_parameters() {
        let mut state = edit_state("use pm;\n#{level_db=0}\n_ = test;", "pm.piano_keys(", 14);
        let root = completion_library_root();
        apply_with_library_root(&mut state, root, Action::Complete);
        // The named default was evaluated when the module loaded (1 + 1 → 2).
        assert_eq!(
            edit_text_and_cursor(&state),
            ("pm.piano_keys(freq, gain = 2)".to_string(), 18)
        );
    }

    #[test]
    fn parameter_hint_for_missing_module_binding_names_the_projection() {
        let mut state = edit_state("use pm;\n#{level_db=0}\n_ = test;", "pm.nope(", 8);
        let root = completion_library_root();
        let effects = apply_with_library_root(&mut state, root, Action::Complete);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "\"pm.nope\" is not defined"),
            "expected a not-defined message, got {:?}",
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
            "not_fo = 1;\nnot_fu = 2;\nnot_foo = 3;\n#{level_db=0}\n_ = test;",
            "not_f",
            5,
        );
        apply_with_empty_status(&mut state, Action::Complete);
        apply_with_empty_status(&mut state, Action::Complete);
        assert_eq!(edit_text_and_cursor(&state), ("not_fu".to_string(), 6));
        apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(edit_text_and_cursor(&state), ("not_f".to_string(), 5));
        apply_with_empty_status(&mut state, Action::Redo);
        assert_eq!(edit_text_and_cursor(&state), ("not_fu".to_string(), 6));
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
    fn undo_in_select_mode_with_empty_history_shows_message() {
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::Undo);
        assert_eq!(messages(&effects), vec!["Nothing to undo"]);
        let effects = apply_with_empty_status(&mut state, Action::Redo);
        assert_eq!(messages(&effects), vec!["Nothing to redo"]);
        assert!(matches!(state.mode, Mode::Select));
    }

    #[test]
    fn advance_program_replaces_the_status() {
        // Two programs so AdvanceProgram(1) actually moves the index.
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = test;\n#{level_db=0}\n_ = second;".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let effects = apply_with_empty_status(&mut state, Action::AdvanceProgram(1));
        assert_eq!(state.active_program_index, 1);
        // The ShowMessage replaces any stale "Removed pending..." /
        // "Playing..." text.
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ShowMessage(s) if s == "A:2")),
            "expected a ShowMessage replacing the status, got {:?}",
            effects
        );
    }

    #[test]
    fn select_program_shows_selection_message_on_navigate() {
        // Named binding → status shows the address and identifier; anonymous
        // `_` binding → only the address; a slot with no source binding
        // (padding) → blank.
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
        assert_eq!(msg, "A:1 kick");

        // Slot 2: anonymous `_` — only the address.
        let effects = apply_with_empty_status(&mut state, Action::SelectProgram(1));
        let msg = effects
            .iter()
            .find_map(|e| match e {
                Effect::ShowMessage(s) => Some(s.clone()),
                _ => None,
            })
            .expect("expected a ShowMessage");
        assert_eq!(msg, "A:2");

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
            text: Arc::default(),
            note_off_waveforms: Default::default(),
        });
    }

    /// A slider move emits exactly one update, addressed by a mark that
    /// names the declaring program. The single update is what reaches key
    /// voices too — no separate mirror onto the keys family.
    #[test]
    fn slider_change_emits_one_update_keyed_by_the_declaring_program() {
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
        let updates: Vec<_> = effects
            .iter()
            .filter(|e| matches!(e, Effect::UpdateSlider { .. }))
            .collect();
        assert_eq!(updates.len(), 1, "got {:?}", effects);
        assert!(
            matches!(
                updates[0],
                Effect::UpdateSlider {
                    mark: MarkId::Slider { program: 0, label },
                    ..
                } if label == "vol"
            ),
            "got {:?}",
            updates[0]
        );
    }

    #[test]
    fn toggle_program_playback_stops_when_active() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_start(now - Duration::from_secs(1));
        let effects = apply_with_status(&mut state, &status, now, Action::ToggleProgramPlayback(0));
        assert!(
            matches!(effects[0], Effect::StopProgram(0)),
            "expected StopProgram, got {:?}",
            effects
        );
    }

    /// Where `ToggleProgramPlayback` would stop, playing now layers another
    /// play, once, without stopping the one already sounding.
    #[test]
    fn play_program_now_layers_when_already_active() {
        let mut state = looping_state();
        let now = Instant::now();
        let status = status_with_start(now - Duration::from_secs(1));
        let effects = apply_with_status(
            &mut state,
            &status,
            now,
            Action::PlayProgram {
                program_index: 0,
                launch: Launch::Now,
            },
        );
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::PlayProgram {
                        program_index: 0,
                        start: player::Start::Now,
                        loop_after_beats: None,
                    },
                    Effect::UpdateSource(0),
                ]
            ),
            "expected one immediate PlayProgram, got {:?}",
            effects
        );
    }

    /// A pending launch loops as the program's `loop_after_beats` says.
    #[test]
    fn play_program_pending_loops_per_the_annotation() {
        let mut state = looping_state();
        let effects = apply_with_empty_status(
            &mut state,
            Action::PlayProgram {
                program_index: 0,
                launch: Launch::Pending,
            },
        );
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::NextMeasure,
                    loop_after_beats: Some(8),
                }
            ),
            "expected a pending PlayProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn play_program_leaves_the_keys_instrument_alone() {
        let mut state = test_state();
        install_test_keys(&mut state);
        for launch in [Launch::Now, Launch::Pending] {
            let effects = apply_with_empty_status(
                &mut state,
                Action::PlayProgram {
                    program_index: 0,
                    launch,
                },
            );
            assert!(
                effects.is_empty(),
                "expected no effects for the keys instrument, got {:?}",
                effects
            );
        }
    }

    #[test]
    fn toggle_launch_mode_flips_between_toggle_and_trigger() {
        let mut state = test_state();
        assert_eq!(state.launch_mode, LaunchMode::Toggle);
        apply_with_empty_status(&mut state, Action::ToggleLaunchMode);
        assert_eq!(state.launch_mode, LaunchMode::Trigger);
        apply_with_empty_status(&mut state, Action::ToggleLaunchMode);
        assert_eq!(state.launch_mode, LaunchMode::Toggle);
    }

    #[test]
    fn toggle_scope_view_flips_the_view_and_says_so() {
        let mut state = test_state();
        assert_eq!(state.scope_view, scope::ScopeView::Snapshot);
        let effects = apply_with_empty_status(&mut state, Action::ToggleScopeView);
        assert_eq!(state.scope_view, scope::ScopeView::Scan);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m == "Scope view: Scan"),
            "expected the Scan message, got {:?}",
            effects
        );
        let effects = apply_with_empty_status(&mut state, Action::ToggleScopeView);
        assert_eq!(state.scope_view, scope::ScopeView::Snapshot);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m == "Scope view: Snapshot"),
            "expected the Snapshot message, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_metronome_flips_the_click_on_and_off() {
        let mut state = test_state();
        assert!(!state.metronome);
        let effects = apply_with_empty_status(&mut state, Action::ToggleMetronome);
        assert!(state.metronome);
        assert!(matches!(effects[0], Effect::SetMetronome(true)));
        assert!(matches!(&effects[1], Effect::ShowMessage(m) if m == "Metronome on"));
        let effects = apply_with_empty_status(&mut state, Action::ToggleMetronome);
        assert!(!state.metronome);
        assert!(matches!(effects[0], Effect::SetMetronome(false)));
        assert!(matches!(&effects[1], Effect::ShowMessage(m) if m == "Metronome off"));
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
                    start: player::Start::Now,
                    loop_after_beats: None,
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
        let status = status_with_start(now + Duration::from_secs(1));
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
    fn toggle_queued_program_queues_with_the_programs_loop_when_idle() {
        let mut state = looping_state();
        let effects = apply_with_empty_status(&mut state, Action::ToggleProgramPendingPlayback(0));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::NextMeasure,
                    loop_after_beats: Some(8),
                }
            ),
            "expected a queued PlayProgram with the program's Loop, got {:?}",
            effects
        );
    }

    #[test]
    fn stop_and_remove_pending_persist_source() {
        let mut state = test_state();
        let now = Instant::now();
        // A mark starting in the past is active: StopProgram applies.
        let status = status_with_start(now - Duration::from_secs(1));
        let effects = apply_with_status(&mut state, &status, now, Action::StopProgram(0));
        assert!(
            effects.iter().any(|e| matches!(e, Effect::UpdateSource(0))),
            "expected UpdateSource, got {:?}",
            effects
        );
        // A mark starting in the future is pending: RemovePendingProgram applies.
        let status = status_with_start(now + Duration::from_secs(1));
        let effects = apply_with_status(&mut state, &status, now, Action::RemovePendingProgram(0));
        assert!(
            effects.iter().any(|e| matches!(e, Effect::UpdateSource(0))),
            "expected UpdateSource, got {:?}",
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
        let status = status_with_start(now + Duration::from_secs(1));
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

    /// Builds a state whose program 0 is sequenceable with steps on beats [1,
    /// 2.5], with on_beats stubbed so evaluation succeeds with just the
    /// prelude.
    fn sequencer_state() -> AppState {
        AppState::from_source(
            "on_beats = fn(w) => fn(bs) => w;\n\
             #{level_db=0}\n\
             _ = [1, 2.5] | on_beats(1 | fin(time - 1));\n"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse")
    }

    /// A status with program 0's anchor step starting at `start` (before
    /// `now` = active, after `now` = pending).
    fn status_with_anchor(start: Instant) -> tracker::Status<WaveformId> {
        let mut status = empty_status();
        status.start_times.push((
            WaveformId::Step {
                program: 0,
                sixteenth: ANCHOR_SIXTEENTH,
            },
            start,
        ));
        status
    }

    #[test]
    fn toggle_sequencer_step_inserts_step_and_emits_evaluate_then_update_source() {
        let mut state = sequencer_state();
        let effects =
            apply_with_empty_status(&mut state, Action::ToggleSequencerStep { sixteenth: 8 });
        assert_eq!(
            state.active_program().text(),
            "[1, 2.5, 3] | on_beats(1 | fin(time - 1))"
        );
        // The message waits for a successful result, so that a failure's error
        // stays on screen; the save does not wait.
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::EvaluateProgram {
                        program_index: 0,
                        on_success,
                        on_failure,
                    },
                    Effect::UpdateSource(0),
                    // Not playing: the toggle is text-only, no tracker effect.
                ] if matches!(&on_success[..], [Effect::ShowMessage(m)] if m.contains("beat 3 on"))
                    && on_failure.is_empty()
            ),
            "expected an evaluation followed by a beat-on message, got {:?}",
            effects
        );
    }

    #[test]
    fn undo_and_redo_of_sequencer_toggle_reevaluate_and_persist() {
        let mut state = sequencer_state();
        let original = state.active_program().text().to_string();
        apply_with_empty_status(&mut state, Action::ToggleSequencerStep { sixteenth: 8 });
        let toggled = state.active_program().text().to_string();
        assert_ne!(original, toggled);

        for (action, verb, expected) in [
            (Action::Undo, "Undid", &original),
            (Action::Redo, "Redid", &toggled),
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert_eq!(state.active_program().text(), expected);
            assert!(
                matches!(&effects[0], Effect::ShowMessage(m) if m.starts_with(verb)),
                "expected a {} message, got {:?}",
                verb,
                effects
            );
            assert!(matches!(
                &effects[1],
                Effect::EvaluateProgram {
                    program_index: 0,
                    on_success,
                    on_failure,
                } if on_success.is_empty() && on_failure.is_empty()
            ));
            assert!(matches!(effects[2], Effect::UpdateSource(0)));
            assert_eq!(effects.len(), 3);
        }
    }

    #[test]
    fn undo_outside_edit_mode_refuses_when_source_stale() {
        let mut state = sequencer_state();
        apply_with_empty_status(&mut state, Action::ToggleSequencerStep { sixteenth: 8 });
        let toggled = state.active_program().text().to_string();
        let effects = apply_with_stale_source(&mut state, Action::Undo);
        assert_eq!(
            messages(&effects),
            vec!["File changed on disk (reload before editing)"]
        );
        assert_eq!(state.active_program().text(), toggled);
    }

    #[test]
    fn toggle_sequencer_step_off_removes_off_grid_steps_in_window() {
        let mut state = AppState::from_source(
            "on_beats = fn(w) => fn(bs) => w;\n\
             #{level_db=0}\n\
             _ = [1, 2.1, 2.2, 3] | on_beats(1 | fin(time - 1));\n"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let effects = in_run_order(apply_with_empty_status(
            &mut state,
            Action::ToggleSequencerStep { sixteenth: 4 },
        ));
        assert_eq!(
            state.active_program().text(),
            "[1, 3] | on_beats(1 | fin(time - 1))"
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::ShowMessage(m) if m.contains("beat 2 off"))),
            "expected a beat-off message, got {:?}",
            effects
        );
    }

    #[test]
    fn undo_and_redo_while_playing_sync_the_playing_steps() {
        let mut state = sequencer_state();
        let now = Instant::now();
        let status = status_with_anchor(now - Duration::from_secs(1));
        apply_with_status(
            &mut state,
            &status,
            now,
            Action::ToggleSequencerStep { sixteenth: 4 },
        );

        let effects = in_run_order(apply_with_status(&mut state, &status, now, Action::Undo));
        assert!(
            matches!(
                effects.last(),
                Some(Effect::RemovePendingSequencerStep {
                    program_index: 0,
                    sixteenth: 4,
                })
            ),
            "expected RemovePendingSequencerStep last, got {:?}",
            effects
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RecordSequencerStepMarks(0))),
            "expected RecordSequencerStepMarks, got {:?}",
            effects
        );
        assert_eq!(effects.len(), 5);

        let effects = in_run_order(apply_with_status(&mut state, &status, now, Action::Redo));
        assert!(
            matches!(
                effects.last(),
                Some(Effect::PlaySequencerStep {
                    program_index: 0,
                    sixteenth: 4,
                })
            ),
            "expected PlaySequencerStep last, got {:?}",
            effects
        );
        assert!(
            effects
                .iter()
                .any(|e| matches!(e, Effect::RecordSequencerStepMarks(0))),
            "expected RecordSequencerStepMarks, got {:?}",
            effects
        );
        assert_eq!(effects.len(), 5);
    }

    #[test]
    fn toggle_on_while_playing_emits_play_sequencer_step() {
        let mut state = sequencer_state();
        let now = Instant::now();
        let status = status_with_anchor(now - Duration::from_secs(1));
        let effects = in_run_order(apply_with_status(
            &mut state,
            &status,
            now,
            Action::ToggleSequencerStep { sixteenth: 4 },
        ));
        assert!(
            matches!(
                effects.last(),
                Some(Effect::PlaySequencerStep {
                    program_index: 0,
                    sixteenth: 4,
                })
            ),
            "expected PlaySequencerStep last, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_off_while_playing_emits_remove_pending_step() {
        let mut state = sequencer_state();
        let now = Instant::now();
        let status = status_with_anchor(now - Duration::from_secs(1));
        let effects = in_run_order(apply_with_status(
            &mut state,
            &status,
            now,
            Action::ToggleSequencerStep { sixteenth: 0 },
        ));
        assert_eq!(
            state.active_program().text(),
            "[2.5] | on_beats(1 | fin(time - 1))"
        );
        // The removal reads no result, so it runs on a failure too.
        assert!(
            matches!(
                &effects[0],
                Effect::EvaluateProgram { on_failure, .. } if matches!(
                    &on_failure[..],
                    [Effect::RemovePendingSequencerStep {
                        program_index: 0,
                        sixteenth: 0,
                    }]
                )
            ),
            "got {:?}",
            effects
        );
        assert!(
            matches!(
                effects.last(),
                Some(Effect::RemovePendingSequencerStep {
                    program_index: 0,
                    sixteenth: 0,
                })
            ),
            "expected RemovePendingSequencerStep last, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_with_pending_anchor_still_emits_play() {
        let mut state = sequencer_state();
        let now = Instant::now();
        let status = status_with_anchor(now + Duration::from_secs(1));
        let effects = in_run_order(apply_with_status(
            &mut state,
            &status,
            now,
            Action::ToggleSequencerStep { sixteenth: 4 },
        ));
        assert!(
            matches!(effects.last(), Some(Effect::PlaySequencerStep { .. })),
            "expected PlaySequencerStep for a queued family, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_on_non_sequenceable_program_only_shows_message() {
        let mut state = AppState::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        let text_before = state.active_program().text().to_string();
        let effects =
            apply_with_empty_status(&mut state, Action::ToggleSequencerStep { sixteenth: 0 });
        assert_eq!(state.active_program().text(), text_before);
        assert_eq!(effects.len(), 1);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("not sequenceable")),
            "expected a not-sequenceable message, got {:?}",
            effects
        );
    }

    #[test]
    fn transport_play_queues_active_program_at_next_measure() {
        let mut state = looping_state();
        let effects = apply_with_empty_status(&mut state, Action::Transport(Transport::Play));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::NextMeasure,
                    loop_after_beats: Some(8),
                }
            ),
            "expected a queued PlayProgram with the program's Loop, got {:?}",
            effects
        );
    }

    #[test]
    fn transport_play_noops_when_already_pending() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_start(now + Duration::from_secs(1));
        let effects =
            apply_with_status(&mut state, &status, now, Action::Transport(Transport::Play));
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
    }

    #[test]
    fn transport_play_queues_a_playing_sequence_despite_its_pending_steps() {
        let mut state = sequencer_state();
        let now = Instant::now();
        // A one-shot pass: its anchor started, and a later step waits.
        let mut status = status_with_anchor(now - Duration::from_secs(1));
        status.start_times.push((
            WaveformId::Step {
                program: 0,
                sixteenth: 8,
            },
            now + Duration::from_secs(1),
        ));
        let effects =
            apply_with_status(&mut state, &status, now, Action::Transport(Transport::Play));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 0,
                    start: player::Start::NextMeasure,
                    ..
                }
            ),
            "expected a queued PlayProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn transport_stop_removes_active_program_pending_playback() {
        let mut state = test_state();
        let now = Instant::now();
        let status = status_with_start(now + Duration::from_secs(1));
        let effects =
            apply_with_status(&mut state, &status, now, Action::Transport(Transport::Stop));
        assert!(
            matches!(effects[0], Effect::RemovePendingProgram(0)),
            "expected RemovePendingProgram, got {:?}",
            effects
        );
    }

    #[test]
    fn enter_edit_mode_refuses_when_source_stale() {
        let mut state = test_state();
        let effects = apply_with_stale_source(&mut state, Action::EnterEditMode);
        assert!(
            matches!(state.mode, Mode::Select),
            "expected to stay in Select, got {:?}",
            state.mode
        );
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("changed on disk")),
            "expected the stale-source message, got {:?}",
            effects
        );
    }

    #[test]
    fn toggle_sequencer_step_refuses_when_source_stale() {
        let mut state = sequencer_state();
        let text_before = state.active_program().text().to_string();
        let effects =
            apply_with_stale_source(&mut state, Action::ToggleSequencerStep { sixteenth: 0 });
        assert_eq!(state.active_program().text(), text_before);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("changed on disk")),
            "expected the stale-source message, got {:?}",
            effects
        );
    }

    #[test]
    fn reload_source_only_applies_in_select_mode() {
        let mut state = test_state();
        let effects = apply_with_empty_status(&mut state, Action::ReloadSource);
        assert!(matches!(effects[0], Effect::ReloadSource));

        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        let effects = apply_with_empty_status(&mut state, Action::ReloadSource);
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
    }

    #[test]
    fn change_sequencer_page_clamps_and_reports() {
        let mut state = sequencer_state();
        // Already at the first page: down stays put.
        apply_with_empty_status(&mut state, Action::ChangeSequencerPage(-1));
        assert_eq!(state.sequencer_page, 0);

        let effects = apply_with_empty_status(&mut state, Action::ChangeSequencerPage(1));
        assert_eq!(state.sequencer_page, 1);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m.contains("beats 5–8")),
            "expected the page's beat range, got {:?}",
            effects
        );

        state.sequencer_page = sequencer::MAX_PAGE;
        apply_with_empty_status(&mut state, Action::ChangeSequencerPage(1));
        assert_eq!(state.sequencer_page, sequencer::MAX_PAGE);
    }

    #[test]
    fn toggle_on_a_later_page_edits_later_steps() {
        let mut state = sequencer_state();
        // Sixteenth 16 is beat 5, the first pad of page 1.
        apply_with_empty_status(&mut state, Action::ToggleSequencerStep { sixteenth: 16 });
        assert_eq!(
            state.active_program().text(),
            "[1, 2.5, 5] | on_beats(1 | fin(time - 1))"
        );
    }

    #[test]
    fn pad_mode_cycles_through_three_sub_modes() {
        let mut state = sequencer_state();
        assert_eq!(state.daw_pad_mode, DawPadMode::ClipLauncher);
        let cycle = || Action::PadModeChanged {
            previous: launchkey::PadMode::DAW,
            current: launchkey::PadMode::DAW,
        };
        apply_with_empty_status(&mut state, cycle());
        assert_eq!(state.daw_pad_mode, DawPadMode::KeysInstaller);
        apply_with_empty_status(&mut state, cycle());
        assert_eq!(state.daw_pad_mode, DawPadMode::Sequencer);
        apply_with_empty_status(&mut state, cycle());
        assert_eq!(state.daw_pad_mode, DawPadMode::ClipLauncher);
    }

    /// Builds a state with two programs, `a` in slot 1 and `b` in slot 2,
    /// editing slot 1.
    fn editing_state() -> AppState {
        editing_state_of("#{level_db=0}\na = sine(440);\n#{level_db=0}\nb = sine(220);\n")
    }

    /// Returns an Edit-mode state over `source`, with the cursor at the end of
    /// program 0.
    fn editing_state_of(source: &str) -> AppState {
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");
        state.mode = Mode::Edit {
            cursor_position: state.active_program().text().len(),
            errors: vec![],
            completion: None,
        };
        state
    }

    #[test]
    fn owned_program_stays_active_while_editing() {
        let mut state = editing_state();
        for action in [Action::SelectProgram(1), Action::AdvanceProgram(1)] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
            assert_eq!(state.active_program_index, 0);
        }
    }

    #[test]
    fn owned_program_cannot_start_or_queue_while_editing() {
        let mut state = editing_state();
        for action in [
            Action::PlayProgram {
                program_index: 0,
                launch: Launch::Pending,
            },
            Action::PlayProgram {
                program_index: 0,
                launch: Launch::Now,
            },
            Action::Transport(Transport::Play),
            Action::ToggleProgramPlayback(0),
            Action::ToggleProgramPendingPlayback(0),
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        }
    }

    #[test]
    fn owned_program_can_stop_and_dequeue_while_editing() {
        let mut state = editing_state();
        let now = Instant::now();
        let active = status_with_start(now - Duration::from_millis(100));
        for action in [Action::StopProgram(0), Action::ToggleProgramPlayback(0)] {
            let effects = apply_with_status(&mut state, &active, now, action);
            assert!(
                matches!(effects[0], Effect::StopProgram(0)),
                "expected StopProgram, got {:?}",
                effects
            );
        }
        let pending = status_with_start(now + Duration::from_millis(100));
        for action in [
            Action::RemovePendingProgram(0),
            Action::Transport(Transport::Stop),
            Action::ToggleProgramPendingPlayback(0),
        ] {
            let effects = apply_with_status(&mut state, &pending, now, action);
            assert!(
                matches!(effects[0], Effect::RemovePendingProgram(0)),
                "expected RemovePendingProgram, got {:?}",
                effects
            );
        }
    }

    #[test]
    fn other_programs_play_while_editing() {
        let mut state = editing_state();
        for action in [
            Action::PlayProgram {
                program_index: 1,
                launch: Launch::Pending,
            },
            Action::PlayProgram {
                program_index: 1,
                launch: Launch::Now,
            },
            Action::ToggleProgramPlayback(1),
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(
                matches!(
                    effects[0],
                    Effect::PlayProgram {
                        program_index: 1,
                        ..
                    }
                ),
                "expected PlayProgram, got {:?}",
                effects
            );
        }
    }

    #[test]
    fn owned_program_text_and_keys_are_refused_while_editing() {
        let mut state = editing_state();
        state.mode = Mode::Edit {
            cursor_position: 0,
            errors: vec![],
            completion: None,
        };
        let text_before = state.active_program().text().to_string();
        for action in [
            Action::EnterEditMode,
            Action::ToggleSequencerStep { sixteenth: 0 },
            Action::ToggleInstalledKeys(0),
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        }
        assert_eq!(state.active_program().text(), text_before);
        assert!(matches!(
            state.mode,
            Mode::Edit {
                cursor_position: 0,
                ..
            }
        ));

        let effects = apply_with_empty_status(&mut state, Action::ToggleInstalledKeys(1));
        assert!(
            matches!(effects[0], Effect::InstallKeys(1)),
            "expected InstallKeys, got {:?}",
            effects
        );
    }

    #[test]
    fn leaving_edit_mode_can_play_the_owned_program() {
        let mut state = editing_state_of("#{loop_after_beats=8}\na = sine(440);\n");
        let effects = apply_with_empty_status(
            &mut state,
            Action::EvaluateAndLeaveEditMode {
                mode_on_failure: Mode::Select,
                play: Some(Launch::Pending),
            },
        );
        // The mode stays Edit until the result arrives, and the play waits
        // for it; the save does not.
        assert!(matches!(state.mode, Mode::Edit { .. }));
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::EvaluateProgram {
                        program_index: 0,
                        on_success,
                        on_failure,
                    },
                    Effect::UpdateSource(0),
                ] if matches!(
                    &on_success[..],
                    [
                        Effect::SetMode(Mode::Select),
                        Effect::PlayProgram {
                            program_index: 0,
                            start: player::Start::NextMeasure,
                            loop_after_beats: Some(8),
                        },
                    ]
                ) && matches!(&on_failure[..], [Effect::SetMode(Mode::Select)])
            ),
            "expected EvaluateProgram then PlayProgram, got {:?}",
            effects
        );
    }

    /// Marks an evaluation of the program at `index` as waiting on the
    /// processing thread.
    fn pend_evaluation(state: &mut AppState, index: usize) {
        let program = &state.programs.programs()[index];
        let generation = program.generation();
        let text = program.text().into();
        state.processing.push_back(PendingJob::Program {
            index,
            generation,
            text,
            on_success: vec![Effect::SetMode(Mode::Select)],
            on_failure: vec![],
            message: String::new(),
        });
    }

    #[test]
    fn edit_keys_are_dropped_while_the_result_is_pending() {
        let mut state = editing_state();
        apply_with_empty_status(&mut state, Action::InsertText(" ".to_string()));
        pend_evaluation(&mut state, 0);
        let text_before = state.active_program().text().to_string();
        for action in [
            Action::InsertText("x".to_string()),
            Action::DeleteCharBeforeCursor,
            Action::DeleteCharAfterCursor,
            Action::DeleteWordBeforeCursor,
            Action::DeleteWordAfterCursor,
            Action::DeleteToEndOfLine,
            Action::MoveCursorBy(-1),
            Action::MoveCursorToStart,
            Action::MoveCursorToEnd,
            Action::MoveCursorToPreviousWord,
            Action::MoveCursorToNextWord,
            Action::Complete,
            Action::Undo,
            Action::Redo,
            Action::EvaluateAndLeaveEditMode {
                mode_on_failure: Mode::Select,
                play: Some(Launch::Now),
            },
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        }
        assert_eq!(state.active_program().text(), text_before);
        assert!(matches!(
            state.mode,
            Mode::Edit { cursor_position, .. } if cursor_position == text_before.len()
        ));

        // Reading a type, and pads for other programs, still work.
        let effects = apply_with_empty_status(&mut state, Action::ShowType);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(_)]),
            "got {:?}",
            effects
        );
        let effects = apply_with_empty_status(&mut state, Action::ToggleInstalledKeys(1));
        assert!(
            matches!(&effects[..], [Effect::InstallKeys(1)]),
            "got {:?}",
            effects
        );

        // Once the result is in, editing resumes.
        state.processing.clear();
        apply_with_empty_status(&mut state, Action::InsertText("x".to_string()));
        assert_eq!(state.active_program().text(), format!("{}x", text_before));
    }

    #[test]
    fn entering_edit_mode_is_refused_while_the_result_is_pending() {
        let mut state = editing_state();
        state.mode = Mode::Select;
        pend_evaluation(&mut state, 0);
        let effects = apply_with_empty_status(&mut state, Action::EnterEditMode);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m == PROCESSING_MESSAGE),
            "got {:?}",
            effects
        );
        assert!(matches!(state.mode, Mode::Select));

        // A result pending for another program doesn't hold this one.
        state.processing.clear();
        pend_evaluation(&mut state, 1);
        apply_with_empty_status(&mut state, Action::EnterEditMode);
        assert!(matches!(state.mode, Mode::Edit { .. }));
    }

    /// The actions that change the source are refused while a program set
    /// loads, and playback is not.
    #[test]
    fn source_changes_are_refused_while_a_set_loads() {
        let mut state = test_state();
        state.processing.push_back(PendingJob::Load {
            startup: false,
            select: None,
        });
        for action in [
            Action::EnterEditMode,
            Action::ToggleSequencerStep { sixteenth: 0 },
            Action::Transport(Transport::Record),
            Action::Undo,
            Action::Redo,
            Action::ToggleKeysAnnotation(0),
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: 1,
            },
            Action::ReloadSource,
        ] {
            let description = format!("{:?}", action);
            let effects = apply_with_empty_status(&mut state, action);
            assert!(
                matches!(&effects[..], [Effect::ShowMessage(m)] if m == PROCESSING_MESSAGE),
                "{} got {:?}",
                description,
                effects
            );
        }
        assert!(matches!(state.mode, Mode::Select));

        let effects = apply_with_empty_status(
            &mut state,
            Action::PlayProgram {
                program_index: 0,
                launch: Launch::Now,
            },
        );
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::PlayProgram {
                        program_index: 0,
                        ..
                    },
                    ..
                ]
            ),
            "got {:?}",
            effects
        );

        // Once the set is in, the same actions go through.
        state.processing.clear();
        let effects = apply_with_empty_status(&mut state, Action::ReloadSource);
        assert!(
            matches!(&effects[..], [Effect::ReloadSource]),
            "got {:?}",
            effects
        );
    }

    #[test]
    fn owned_program_keys_can_be_uninstalled_while_editing() {
        let mut state = editing_state();
        install_test_keys(&mut state);
        let effects = apply_with_empty_status(&mut state, Action::ToggleInstalledKeys(0));
        assert!(state.keys.is_none());
        assert!(
            matches!(&effects[1], Effect::ShowMessage(m) if m == "Uninstalled keys"),
            "expected the uninstall message, got {:?}",
            effects
        );
    }

    /// A file whose slot 1 (`kb`) is a named keys program at -6 dB, followed by
    /// the phrase helper and an unnamed program in slot 2; slot 3 is empty.
    const RECORD_SOURCE: &str = "#{keys, level_db=-6}\nkb = fn(k, v) => (0, 0);\n\
        as_midi_phrase = fn(inst) => 0;\n#{level_db=0}\n_ = 1;\n";

    /// Returns a Select-mode state over `source` with slot 1 installed as the
    /// keys instrument and `active` as the active program.
    fn record_state(source: &str, active: usize) -> AppState {
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");
        install_test_keys(&mut state);
        state.active_program_index = active;
        state
    }

    /// A status whose measure boundaries (4 beats at 60 bpm) fall at `t0`
    /// plus 0, 4, 8 and 12 seconds.
    fn measures(t0: Instant) -> tracker::Status<WaveformId> {
        let mut status = empty_status();
        for k in 0..4 {
            status.start_times.push((
                WaveformId::Beats(k % 2 == 1),
                t0 + Duration::from_secs(4 * k),
            ));
        }
        status
    }

    /// Applies `action` at `seconds` past `t0` against [`measures`]`(t0)`.
    fn apply_at(state: &mut AppState, t0: Instant, seconds: f64, action: Action) -> Vec<Effect> {
        let now = t0 + Duration::from_secs_f64(seconds);
        apply_with_status(state, &measures(t0), now, action)
    }

    /// Returns `effects` in the order they run when every evaluation among
    /// them succeeds and answers after the rest: each `EvaluateProgram`'s
    /// `on_success` effects move to the end, leaving its `on_success` empty.
    fn in_run_order(effects: Vec<Effect>) -> Vec<Effect> {
        let mut now = Vec::new();
        let mut later = Vec::new();
        for effect in effects {
            match effect {
                Effect::EvaluateProgram {
                    program_index,
                    on_success,
                    on_failure,
                } => {
                    later.extend(on_success);
                    now.push(Effect::EvaluateProgram {
                        program_index,
                        on_success: vec![],
                        on_failure,
                    });
                }
                effect => now.push(effect),
            }
        }
        now.extend(later);
        now
    }

    /// Returns the messages among `effects`, in order.
    fn messages(effects: &[Effect]) -> Vec<&str> {
        effects
            .iter()
            .filter_map(|e| match e {
                Effect::ShowMessage(m) => Some(m.as_str()),
                _ => None,
            })
            .collect()
    }

    /// Returns the take's phase; panics outside Record mode.
    fn take_phase(state: &AppState) -> Phase {
        let Mode::Record { take, .. } = &state.mode else {
            panic!("expected Record mode, got {:?}", state.mode);
        };
        take.phase()
    }

    fn record() -> Action {
        Action::Transport(Transport::Record)
    }

    #[test]
    fn scope_view_toggles_while_editing_or_recording() {
        let mut state = editing_state();
        assert!(matches!(state.mode, Mode::Edit { .. }));
        let effects = apply_with_empty_status(&mut state, Action::ToggleScopeView);
        assert_eq!(state.scope_view, scope::ScopeView::Scan);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(_)]),
            "expected a message in Edit mode, got {:?}",
            effects
        );

        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 1.0, record());
        assert!(matches!(state.mode, Mode::Record { .. }));
        let effects = apply_at(&mut state, t0, 1.5, Action::ToggleScopeView);
        assert_eq!(state.scope_view, scope::ScopeView::Scan);
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(_)]),
            "expected a message in Record mode, got {:?}",
            effects
        );
    }

    #[test]
    fn undo_during_a_take_is_a_no_op() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        let program = state.programs.program_mut(1).unwrap();
        program.record_edit(0);
        program.set_text("2".to_string());
        apply_at(&mut state, t0, 1.0, record());
        assert!(matches!(state.mode, Mode::Record { .. }));
        let effects = apply_at(&mut state, t0, 1.5, Action::Undo);
        assert!(effects.is_empty(), "expected no effects, got {:?}", effects);
        assert_eq!(state.programs.program(1).unwrap().text(), "2");
    }

    fn note_on(t0: Instant, seconds: f64) -> Action {
        Action::NoteOn {
            key: 60,
            velocity: 127,
            stamp: t0 + Duration::from_secs_f64(seconds),
        }
    }

    fn note_off(t0: Instant, seconds: f64) -> Action {
        Action::NoteOff {
            key: 60,
            stamp: t0 + Duration::from_secs_f64(seconds),
        }
    }

    #[test]
    fn record_refuses_with_the_failing_precondition() {
        let t0 = Instant::now();
        let cases = [
            // The keys program itself is active.
            (RECORD_SOURCE, 0, "Can't record into the keys program"),
            // The keys program is bound to `_`.
            (
                "#{keys}\n_ = fn(k, v) => (0, 0);\n#{level_db=0}\n_ = 1;\n",
                1,
                "Keys program has no name",
            ),
            // A later binding shadows `kb` before slot 2.
            (
                "#{keys}\nkb = fn(k, v) => (0, 0);\nkb = 2;\nas_midi_phrase = 0;\n\
                 #{level_db=0}\n_ = 1;\n",
                1,
                "kb isn't in scope at A:2",
            ),
            // The helper isn't bound.
            (
                "#{keys}\nkb = fn(k, v) => (0, 0);\n#{level_db=0}\n_ = 1;\n",
                1,
                "as_midi_phrase not in scope (open std)",
            ),
        ];
        for (source, active, expected) in cases {
            let mut state = record_state(source, active);
            let effects = apply_at(&mut state, t0, 1.0, record());
            assert_eq!(messages(&effects), vec![expected], "for {:?}", source);
            assert!(matches!(state.mode, Mode::Select));
        }

        let mut state = record_state(RECORD_SOURCE, 1);
        state.keys = None;
        let effects = apply_at(&mut state, t0, 1.0, record());
        assert_eq!(messages(&effects), vec!["No keys instrument installed"]);

        let mut state = record_state(RECORD_SOURCE, 1);
        let effects = apply_with_stale_source(&mut state, record());
        assert_eq!(
            messages(&effects),
            vec!["File changed on disk (reload before recording)"]
        );
        assert!(matches!(state.mode, Mode::Select));
    }

    #[test]
    fn record_outside_select_mode_is_silent() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        state.mode = Mode::Keys;
        let effects = apply_at(&mut state, t0, 1.0, record());
        assert!(effects.is_empty(), "got {:?}", effects);
        assert!(matches!(state.mode, Mode::Keys));
    }

    #[test]
    fn record_into_a_slot_before_the_keys_program_is_refused() {
        let t0 = Instant::now();
        let source = "as_midi_phrase = 0;\n#{level_db=0}\n_ = 1;\n#{keys}\n\
                      kb = fn(k, v) => (0, 0);\n";
        let mut state = record_state(source, 0);
        state.keys.as_mut().unwrap().id = 1;
        let effects = apply_at(&mut state, t0, 1.0, record());
        assert_eq!(messages(&effects), vec!["kb isn't in scope at A:1"]);
    }

    #[test]
    fn record_arms_until_the_next_boundary_and_removes_pending_playback() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        let mut status = measures(t0);
        status
            .start_times
            .push((WaveformId::Program(1), t0 + Duration::from_secs(4)));
        let now = t0 + Duration::from_secs(1);
        let effects = apply_with_status(&mut state, &status, now, record());
        assert_eq!(take_phase(&state), Phase::Armed);
        assert!(
            matches!(effects[0], Effect::RemovePendingProgram(1)),
            "got {:?}",
            effects
        );
        assert_eq!(
            messages(&effects).last(),
            Some(&"Armed: recording kb into A:2")
        );
        assert!(!state.take_tick_due(t0 + Duration::from_secs_f64(3.9)));
        assert!(state.take_tick_due(t0 + Duration::from_secs(4)));

        let effects = apply_at(&mut state, t0, 4.0, Action::TakeTick);
        assert_eq!(take_phase(&state), Phase::Recording);
        assert_eq!(messages(&effects), vec!["Recording kb into A:2"]);
        assert!(!state.take_tick_due(t0 + Duration::from_secs(100)));
    }

    #[test]
    fn record_just_after_a_boundary_records_from_it() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        let effects = apply_at(&mut state, t0, 4.05, record());
        assert_eq!(take_phase(&state), Phase::Recording);
        assert_eq!(messages(&effects), vec!["Recording kb into A:2"]);
    }

    #[test]
    fn take_is_written_at_the_close() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 1.0, record());
        apply_at(&mut state, t0, 4.0, Action::TakeTick);
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        apply_at(&mut state, t0, 5.0, note_off(t0, 5.0));
        let effects = apply_at(&mut state, t0, 6.0, record());
        assert!(effects.is_empty(), "got {:?}", effects);
        assert_eq!(take_phase(&state), Phase::Finishing(Finish::Silent));
        // The close is one margin (0.125 s at 60 bpm) before the end boundary.
        assert!(!state.take_tick_due(t0 + Duration::from_secs_f64(7.8)));
        assert!(state.take_tick_due(t0 + Duration::from_secs_f64(7.875)));

        let effects = apply_at(&mut state, t0, 7.9, Action::TakeTick);
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(
            state.active_program().text(),
            "[(1.500, 60, 1.000, 0.500)] | as_midi_phrase(kb)"
        );
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:2"]);
        assert!(matches!(
            &effects[1..],
            [
                Effect::EvaluateProgram {
                    program_index: 1,
                    on_success,
                    on_failure,
                },
                Effect::UpdateSource(1),
            ] if on_success.is_empty() && on_failure.is_empty()
        ));
        // The write is one undo unit.
        let program = state.programs.program_mut(1).unwrap();
        program.undo(0);
        assert_eq!(program.text(), "1");
    }

    #[test]
    fn stop_press_at_the_close_writes_at_once() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        // Within one margin before the end boundary at 8 s.
        let effects = apply_at(&mut state, t0, 7.95, record());
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:2"]);
        // Still held at the close, so released at the end boundary.
        assert_eq!(
            state.active_program().text(),
            "[(1.500, 60, 1.000, 3.500)] | as_midi_phrase(kb)"
        );
    }

    /// Returns the start of the `PlayProgram` effects among `effects`,
    /// including those waiting on an evaluation's success, in order.
    fn play_starts(effects: &[Effect]) -> Vec<player::Start> {
        effects
            .iter()
            .flat_map(|e| match e {
                Effect::PlayProgram { start, .. } => vec![*start],
                Effect::EvaluateProgram { on_success, .. } => play_starts(on_success),
                _ => vec![],
            })
            .collect()
    }

    #[test]
    fn play_while_recording_plays_the_phrase_from_the_end_boundary() {
        let t0 = Instant::now();
        let source = RECORD_SOURCE.replace("level_db=0", "loop_after_beats=6");
        let mut state = record_state(&source, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        let effects = apply_at(&mut state, t0, 6.0, Action::Transport(Transport::Play));
        assert_eq!(take_phase(&state), Phase::Finishing(Finish::Play));
        assert_eq!(
            messages(&effects),
            vec!["Looping waveform A:2 every 6 beats"]
        );
        assert!(play_starts(&effects).is_empty());

        let effects = apply_at(&mut state, t0, 7.9, Action::TakeTick);
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:2"]);
        // The play waits for the evaluation; the save does not.
        assert!(
            matches!(
                &effects[1..],
                [
                    Effect::EvaluateProgram {
                        program_index: 1,
                        on_success,
                        on_failure,
                    },
                    Effect::UpdateSource(1),
                ] if on_failure.is_empty() && matches!(
                    &on_success[..],
                    [Effect::PlayProgram {
                        program_index: 1,
                        loop_after_beats: Some(6),
                        ..
                    }]
                )
            ),
            "got {:?}",
            effects
        );
        assert_eq!(
            play_starts(&effects),
            vec![player::Start::At(t0 + Duration::from_secs(8))]
        );
    }

    #[test]
    fn late_play_press_plays_from_the_boundary_just_passed() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        // Within one margin after the boundary at 8 s.
        let effects = apply_at(&mut state, t0, 8.05, Action::Transport(Transport::Play));
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:2"]);
        assert_eq!(
            play_starts(&effects),
            vec![player::Start::At(t0 + Duration::from_secs(8))]
        );
    }

    #[test]
    fn record_after_play_while_finishing_writes_silently() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        let effects = apply_at(&mut state, t0, 6.0, Action::Transport(Transport::Play));
        assert_eq!(messages(&effects), vec!["Playing waveform A:2"]);
        let effects = apply_at(&mut state, t0, 6.5, record());
        assert_eq!(take_phase(&state), Phase::Finishing(Finish::Silent));
        assert_eq!(
            messages(&effects),
            vec!["Removed pending waveform for program A:2"]
        );
        // Play again while Finishing re-posts the play text.
        let effects = apply_at(&mut state, t0, 6.7, Action::Transport(Transport::Play));
        assert_eq!(messages(&effects), vec!["Playing waveform A:2"]);
        let effects = apply_at(&mut state, t0, 6.9, record());
        assert_eq!(
            messages(&effects),
            vec!["Removed pending waveform for program A:2"]
        );
        // Record again while Finishing silent says nothing.
        let effects = apply_at(&mut state, t0, 7.0, record());
        assert!(effects.is_empty(), "got {:?}", effects);

        let effects = apply_at(&mut state, t0, 7.9, Action::TakeTick);
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:2"]);
        assert!(play_starts(&effects).is_empty());
    }

    #[test]
    fn empty_take_with_play_queues_the_program() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 6.0, Action::Transport(Transport::Play));
        let effects = apply_at(&mut state, t0, 7.9, Action::TakeTick);
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Nothing recorded"]);
        assert_eq!(play_starts(&effects), vec![player::Start::NextMeasure]);
        assert_eq!(state.active_program().text(), "1");
    }

    #[test]
    fn empty_take_leaves_the_program_untouched() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 6.0, record());
        let effects = apply_at(&mut state, t0, 7.9, Action::TakeTick);
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Nothing recorded"]);
        assert_eq!(effects.len(), 1);
        assert_eq!(state.active_program().text(), "1");
    }

    #[test]
    fn take_into_an_empty_slot_takes_the_keys_level() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 2);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        let effects = apply_at(&mut state, t0, 7.95, record());
        assert_eq!(messages(&effects), vec!["Recorded 1 note into A:3"]);
        assert_eq!(state.active_program().level_db(), -6.0);
    }

    #[test]
    fn stop_disarms_or_discards_the_take() {
        let t0 = Instant::now();
        let stop = || Action::Transport(Transport::Stop);

        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 1.0, record());
        let effects = apply_at(&mut state, t0, 2.0, stop());
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Disarmed"]);

        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 4.0, record());
        apply_at(&mut state, t0, 4.5, note_on(t0, 4.5));
        let effects = apply_at(&mut state, t0, 5.0, stop());
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Discarded take (1 note)"]);
        assert_eq!(state.active_program().text(), "1");
    }

    #[test]
    fn record_or_play_while_armed_disarms() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 1.0, record());
        let effects = apply_at(&mut state, t0, 2.0, record());
        assert!(matches!(state.mode, Mode::Select));
        assert_eq!(messages(&effects), vec!["Disarmed"]);

        apply_at(&mut state, t0, 2.5, record());
        let effects = apply_at(&mut state, t0, 3.0, Action::Transport(Transport::Play));
        assert!(matches!(state.mode, Mode::Select));
        assert!(
            matches!(
                effects[0],
                Effect::PlayProgram {
                    program_index: 1,
                    start: player::Start::NextMeasure,
                    ..
                }
            ),
            "expected today's Play, got {:?}",
            effects
        );
    }

    #[test]
    fn record_mode_owns_the_active_program() {
        let t0 = Instant::now();
        let mut state = record_state(RECORD_SOURCE, 1);
        apply_at(&mut state, t0, 1.0, record());
        for action in [
            Action::SelectProgram(0),
            Action::AdvanceProgram(1),
            Action::EnterEditMode,
            Action::PlayProgram {
                program_index: 1,
                launch: Launch::Now,
            },
        ] {
            let effects = apply_at(&mut state, t0, 2.0, action);
            assert!(effects.is_empty(), "got {:?}", effects);
        }
        assert_eq!(state.active_program_index, 1);
        assert_eq!(take_phase(&state), Phase::Armed);
    }

    /// Returns a Select-mode state over `source` with `active` as the active
    /// program.
    fn annotation_state(source: &str, active: usize) -> AppState {
        let mut state = AppState::from_source(source.to_string(), std::path::PathBuf::new())
            .expect("test source should parse");
        state.active_program_index = active;
        state
    }

    #[test]
    fn prefix_shows_its_prompt_until_it_ends() {
        let mut state = test_state();
        let effects =
            apply_with_empty_status(&mut state, Action::StartPrefix(PrefixKey::Annotation));
        assert!(effects.is_empty());
        assert_eq!(state.pending_prefix, Some(PrefixKey::Annotation));
        assert_eq!(state.message, "Annotation? (k)eys (u)p (d)own (l)oop");
        apply_with_empty_status(&mut state, Action::EndPrefix);
        assert_eq!(state.pending_prefix, None);
        assert!(state.message.is_empty());
    }

    #[test]
    fn toggling_keys_changes_the_kind_saves_and_evaluates() {
        let mut state = annotation_state("#{level_db=0}\nk = 1;", 0);
        let effects = apply_with_empty_status(&mut state, Action::ToggleKeysAnnotation(0));
        assert_eq!(state.active_program().kind(), programs::ProgramKind::Keys);
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::ShowMessage(m),
                    Effect::EvaluateProgram {
                        program_index: 0,
                        ..
                    },
                    Effect::UpdateSource(0),
                ] if m == "Added keys to A:1 (k)"
            ),
            "got {:?}",
            effects
        );
        let effects = apply_with_empty_status(&mut state, Action::ToggleKeysAnnotation(0));
        assert_eq!(
            state.active_program().kind(),
            programs::ProgramKind::Waveform
        );
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "Removed keys from A:1 (k)"),
            "got {:?}",
            effects
        );
    }

    #[test]
    fn the_loop_prefix_names_the_program() {
        let mut state = annotation_state("#{level_db=0}\nkick = 1;", 0);
        let effects = apply_with_empty_status(&mut state, Action::StartPrefix(PrefixKey::Loop));
        assert!(effects.is_empty(), "got {:?}", effects);
        assert_eq!(state.pending_prefix, Some(PrefixKey::Loop));
        assert_eq!(state.message, "Loop A:1 (kick)? (1-9) beats (0) once");
    }

    #[test]
    fn the_loop_prefix_is_refused_on_an_empty_slot_a_stale_file_or_keys() {
        let mut state = annotation_state("#{keys}\nlead = fn(n, v) => (0, 0);", 1);
        let effects = apply_with_empty_status(&mut state, Action::StartPrefix(PrefixKey::Loop));
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m == "A:2 is empty"),
            "got {:?}",
            effects
        );
        state.active_program_index = 0;
        let effects = apply_with_stale_source(&mut state, Action::StartPrefix(PrefixKey::Loop));
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m.contains("changed on disk")),
            "got {:?}",
            effects
        );
        let effects = apply_with_empty_status(&mut state, Action::StartPrefix(PrefixKey::Loop));
        assert!(
            matches!(
                &effects[..],
                [Effect::ShowMessage(m)] if m == "A:1 (lead) plays keys, so it can't loop"
            ),
            "got {:?}",
            effects
        );
        assert_eq!(state.pending_prefix, None);
    }

    #[test]
    fn setting_a_loop_saves_it_and_says_so() {
        let mut state = annotation_state("#{level_db=0}\nkick = 1;", 0);
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetLoopAfterBeats {
                program_index: 0,
                beats: Some(3),
            },
        );
        assert_eq!(state.active_program().loop_after_beats(), Some(3));
        assert!(
            matches!(
                &effects[..],
                [Effect::ShowMessage(m), Effect::UpdateSource(0)]
                    if m == "A:1 (kick) loops every 3 beats"
            ),
            "got {:?}",
            effects
        );
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetLoopAfterBeats {
                program_index: 0,
                beats: None,
            },
        );
        assert_eq!(state.active_program().loop_after_beats(), None);
        assert!(
            matches!(&effects[0], Effect::ShowMessage(m) if m == "A:1 (kick) plays once"),
            "got {:?}",
            effects
        );
    }

    #[test]
    fn setting_the_loop_of_a_playing_program_waits_for_its_next_launch() {
        let mut state = annotation_state("#{level_db=0}\nkick = 1;", 0);
        let now = Instant::now();
        for start in [now - Duration::from_secs(1), now + Duration::from_secs(1)] {
            let effects = apply_with_status(
                &mut state,
                &status_with_start(start),
                now,
                Action::SetLoopAfterBeats {
                    program_index: 0,
                    beats: Some(3),
                },
            );
            assert!(
                matches!(
                    &effects[0],
                    Effect::ShowMessage(m)
                        if m == "A:1 (kick) loops every 3 beats from its next launch"
                ),
                "got {:?}",
                effects
            );
        }
    }

    #[test]
    fn setting_a_loop_is_refused_on_a_keys_program() {
        let mut state = annotation_state("#{keys}\nlead = fn(n, v) => (0, 0);", 0);
        let effects = apply_with_empty_status(
            &mut state,
            Action::SetLoopAfterBeats {
                program_index: 0,
                beats: Some(3),
            },
        );
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m.ends_with("so it can't loop")),
            "got {:?}",
            effects
        );
        assert_eq!(state.active_program().loop_after_beats(), None);
    }

    #[test]
    fn adding_keys_is_refused_on_a_program_that_loops() {
        let mut state = annotation_state("#{loop_after_beats=4}\nkick = 1;", 0);
        let effects = apply_with_empty_status(&mut state, Action::ToggleKeysAnnotation(0));
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m == "Remove A:1 (kick)'s Loop first"),
            "got {:?}",
            effects
        );
        assert_eq!(
            state.active_program().kind(),
            programs::ProgramKind::Waveform
        );
    }

    #[test]
    fn losing_keys_uninstalls_the_installed_instrument() {
        let mut state = annotation_state("#{keys}\nk = fn(n, v) => (0, 0);", 0);
        install_test_keys(&mut state);
        state.mode = Mode::Keys;
        let effects = apply_with_empty_status(&mut state, Action::ToggleKeysAnnotation(0));
        assert!(state.keys.is_none());
        assert!(matches!(state.mode, Mode::Select));
        assert!(
            matches!(
                &effects[..],
                [
                    Effect::ModifyWaveform {
                        selector: WaveformSelector::AllKeys,
                        mark_id: MarkId::Terminator,
                        ..
                    },
                    ..,
                    Effect::EvaluateProgram { .. },
                    Effect::UpdateSource(0),
                ]
            ),
            "got {:?}",
            effects
        );
    }

    #[test]
    fn annotation_changes_refuse_an_empty_slot_and_a_stale_file() {
        let mut state = annotation_state("#{level_db=0}\nk = 1;", 1);
        for action in [
            Action::ToggleKeysAnnotation(1),
            Action::ChangeSkipSlots {
                program_index: 1,
                delta: 1,
            },
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(
                matches!(&effects[..], [Effect::ShowMessage(m)] if m == "A:2 is empty"),
                "got {:?}",
                effects
            );
        }
        state.active_program_index = 0;
        for action in [
            Action::ToggleKeysAnnotation(0),
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: 1,
            },
        ] {
            let effects = apply_with_stale_source(&mut state, action);
            assert!(
                matches!(&effects[..], [Effect::ShowMessage(m)] if m.contains("changed on disk")),
                "got {:?}",
                effects
            );
        }
        assert_eq!(
            state.active_program().kind(),
            programs::ProgramKind::Waveform
        );
    }

    #[test]
    fn annotation_changes_are_refused_while_the_mode_owns_the_program() {
        let mut state = editing_state();
        for action in [
            Action::ToggleKeysAnnotation(0),
            Action::SetLoopAfterBeats {
                program_index: 0,
                beats: Some(4),
            },
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: 1,
            },
        ] {
            let effects = apply_with_empty_status(&mut state, action);
            assert!(effects.is_empty(), "got {:?}", effects);
        }
        assert_eq!(
            state.active_program().kind(),
            programs::ProgramKind::Waveform
        );
    }

    #[test]
    fn changing_skip_slots_moves_the_program_and_the_selection() {
        let mut state = annotation_state(
            "#{level_db=0}\na = 1;\n#{skip_slots=1, level_db=0}\nb = 2;",
            2,
        );
        let effects = apply_with_empty_status(
            &mut state,
            Action::ChangeSkipSlots {
                program_index: 2,
                delta: -1,
            },
        );
        assert!(
            matches!(
                &effects[..],
                [Effect::MoveProgram {
                    program_index: 2,
                    skip_slots: 0,
                    select: 1
                }]
            ),
            "got {:?}",
            effects
        );
        let effects = apply_with_empty_status(
            &mut state,
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: 1,
            },
        );
        assert!(
            matches!(
                &effects[..],
                [Effect::MoveProgram {
                    program_index: 0,
                    skip_slots: 1,
                    select: 1
                }]
            ),
            "got {:?}",
            effects
        );
    }

    #[test]
    fn changing_skip_slots_refuses_at_zero_and_past_the_last_slot() {
        let mut state = annotation_state(
            "#{level_db=0}\na = 1;\n#{skip_slots=62, level_db=0}\nb = 2;",
            0,
        );
        let effects = apply_with_empty_status(
            &mut state,
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: -1,
            },
        );
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m.starts_with("Can't move A:1 (a) up")),
            "got {:?}",
            effects
        );
        // `b` is in the last slot, so moving `a` down would push it out.
        let effects = apply_with_empty_status(
            &mut state,
            Action::ChangeSkipSlots {
                program_index: 0,
                delta: 1,
            },
        );
        assert!(
            matches!(&effects[..], [Effect::ShowMessage(m)] if m.starts_with("Can't move A:1 (a) down")),
            "got {:?}",
            effects
        );
    }
}
