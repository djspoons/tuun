//! Programs and their state: text, sliders, level, and evaluation results.

use std::cmp;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::ops::Range;
use std::path;
use std::sync::Arc;
use std::time;

use crate::diagnostics::{Diagnostic, Source};
use crate::environment::Environment;
use crate::expr;
use crate::ids::MarkId;
use crate::parser;
use crate::sequencer;
use crate::slider;
use crate::slider_analysis;
use crate::slider_providers;
use crate::trace_walk::WaveformMarks;
use crate::waveform;

pub const PROGRAMS_PER_BANK: usize = 8;
pub const NUM_PROGRAM_BANKS: usize = 8;

/// Returns the bank-relative address of the slot at `index`, such as `"B:3"`.
///
/// The letter is the bank (A..H) and the digit is the 1-based position within
/// that bank — the digit the performer types to select the slot.
///
/// # Example
///
/// ```
/// use tuun::programs::bank_address;
/// assert_eq!(bank_address(9), "B:2");
/// ```
pub fn bank_address(index: usize) -> String {
    let bank = index / PROGRAMS_PER_BANK;
    let slot_in_bank = (index % PROGRAMS_PER_BANK) + 1;
    format!("{}:{}", (b'A' + bank as u8) as char, slot_in_bank)
}

/// A slider's spec, from the code that calls it, and its current value in the
/// slider's own units.
///
/// Invariant: `value` lies in the spec's range.
#[derive(Debug, Clone, PartialEq)]
pub struct SliderState {
    pub spec: slider::SliderSpec,
    pub value: f32,
}

/// Renders a `level_db` value the way the UI shows it: one decimal place
/// followed by the unit, e.g. `-6.0 dB`.
pub fn format_level_db(level_db: f32) -> String {
    format!("{:.1} dB", level_db)
}

pub fn format_sig_digits(val: f32, sig_figs: usize) -> String {
    if val == 0.0 || !val.is_finite() {
        return format!("{val:.precision$}", precision = sig_figs - 1);
    }

    // Calculate the position of the first significant digit
    let digits_before_decimal = val.abs().log10().floor() + 1.0;

    // Determine required fractional precision
    let precision = (sig_figs as f32 - digits_before_decimal) as isize;

    if precision >= 0 {
        format!("{val:.precision$}", precision = precision as usize)
    } else {
        // If the number is large, round to the nearest tens/hundreds/etc.
        let scale = 10.0f32.powi(precision as i32);
        let rounded = (val * scale).round() / scale;
        format!("{rounded:.0}")
    }
}

/// A slider mutation's user-visible result: the slider's label and its new
/// value, in the slider's own units (not its position).
pub struct SliderChange {
    pub label: String,
    pub value: f32,
}

/// The declared role of a program slot, from its annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProgramKind {
    /// The text must evaluate to a playable waveform.
    #[default]
    Waveform,
    /// Marked `#{keys}`: the text must evaluate to a keys-instrument note
    /// function.
    Keys,
}

/// What a program plays.
#[derive(Debug, Clone)]
pub enum Plays {
    /// The program evaluated to a playable waveform that isn't sequenceable,
    /// with its marks. The waveform is optimized; slider values and the
    /// level wrap are applied per play.
    Waveform(waveform::Waveform<MarkId>, Arc<WaveformMarks>),
    /// The program evaluated to a playable waveform and its text has the
    /// sequenceable `on_beats` shape, so it plays as the decomposed sequence.
    Sequence(sequencer::Sequence),
    /// The program evaluated to a function usable as a keys instrument.
    KeysInstrument(KeysInstrument),
}

/// A function usable as a keys instrument, with the listing bindings its code
/// was written in.
#[derive(Debug, Clone)]
pub struct KeysInstrument {
    /// A function which takes a pair (MIDI note, velocity) and returns a pair
    /// of waveforms, for note-on and note-off. A closed value except for
    /// references to sliders.
    pub function: expr::SourceExpr<MarkId, Source>,
    /// The listing bindings the function's code was written in, found with
    /// the slots' text as it was when the function was evaluated, which its
    /// spans index into.
    pub listings: slider::Listings<Source>,
}

/// The result of successfully evaluating a program's text: what it plays, and
/// the sliders it can reach, on any branch and any note of a keys instrument,
/// in the order the text reaches their first calls (see
/// [`crate::slider_analysis::Analysis::reached`]).
#[derive(Debug, Clone)]
pub struct Evaluated {
    pub plays: Plays,
    pub sliders: Vec<slider_analysis::Reached>,
}

/// What evaluating one program reads, copied out of its [`ProgramSet`] so
/// that the program can be evaluated without the set.
#[derive(Debug, Clone)]
pub struct EvaluationInput {
    /// The program's text.
    pub text: String,
    /// The program's declared kind.
    pub kind: ProgramKind,
    /// The bindings the program is evaluated under (see
    /// [`ProgramSet::evaluation_bindings`]).
    pub bindings: Vec<expr::SourceBinding<MarkId, Source>>,
    /// The backing file's text, which errors located in the file resolve
    /// against.
    pub source: String,
    /// The program's index among its set's programs.
    pub index: usize,
    /// The file's program slots, which can provide sliders (see
    /// [`ProgramSet::slider_providers`]).
    pub providers: slider_providers::Providers,
}

impl EvaluationInput {
    /// Returns the listing bindings code is written in when the program is
    /// played (see [`slider_providers::Providers::listings`]).
    pub fn listings(&self) -> slider::Listings<Source> {
        self.providers.listings(self.index)
    }
}

/// The result of checking and evaluating one program.
#[derive(Debug)]
pub enum Outcome {
    /// The text is blank, as when the program was deleted.
    Empty,
    /// The program evaluated, with the checker's warnings.
    Evaluated(Evaluated, Vec<Diagnostic>),
    /// The program failed to check or to evaluate, with the diagnostics
    /// (any checker warnings first).
    Failed(Vec<Diagnostic>),
}

/// Maximum number of undo units kept per program; recording past the cap drops
/// the oldest unit.
const MAX_UNDO_UNITS: usize = 100;

/// A snapshot of a program's text and Edit-mode cursor, restored by undo/redo.
///
/// `cursor` is a byte offset lying on a char boundary of `text` (both are
/// recorded together from a live Edit-mode state).
#[derive(Debug, Clone)]
struct EditSnapshot {
    text: String,
    cursor: usize,
}

/// Undo/redo history for Edit-mode changes to a program's text.
#[derive(Debug, Clone, Default)]
struct EditHistory {
    /// Snapshots from before each undo unit, oldest first.
    undo: Vec<EditSnapshot>,
    /// Snapshots undone since the last fresh edit, most recently undone
    /// last.
    redo: Vec<EditSnapshot>,
    /// The final character of the last coalesced insertion, or `None` when
    /// no insert run is open.
    last_inserted: Option<char>,
}

impl EditHistory {
    /// Pushes an undo unit, dropping the oldest when the stack is at
    /// `MAX_UNDO_UNITS`.
    fn push_undo_capped(&mut self, snapshot: EditSnapshot) {
        if self.undo.len() == MAX_UNDO_UNITS {
            self.undo.remove(0);
        }
        self.undo.push(snapshot);
    }
}

/// One program slot: its source text, level, color, and the result of its
/// last evaluation.
#[derive(Debug, Clone)]
pub struct Program {
    /// The program's source expression text. Kept in sync with `evaluated`:
    /// any change drops it (see `set_text` / `realign`).
    text: String,
    /// The span in the source file from which this program came. `0..0`
    /// marks a brand-new or padding slot.
    span: Range<usize>,
    /// Index of the originating binding in the file's `Vec<SourceBinding>`.
    binding_index: usize,
    color: Option<(u8, u8, u8)>,
    level_db: f32,
    /// The declared kind, from a `#{keys}` annotation (or its absence).
    kind: ProgramKind,
    /// From a `#{loop_after_beats=N}` annotation.
    loop_after_beats: Option<u32>,
    /// Result of the last evaluation recorded, which may be of an earlier text;
    /// `None` when never evaluated, when the last evaluation failed, or when it
    /// was of an empty text.
    evaluated: Option<Evaluated>,
    /// The text of the last evaluation recorded, which `evaluated`'s marks
    /// index into.
    evaluated_text: Arc<str>,
    /// Undo/redo history for Edit-mode text changes.
    history: EditHistory,
    /// Counts the replacements of `text` by [`Program::set_text`].
    generation: u64,
}

impl Program {
    /// Builds a program from a string without attempting to parse it.
    fn from_string(text: &str, binding_index: usize) -> Program {
        Program {
            text: text.to_string(),
            span: 0..0,
            binding_index,
            color: None,
            level_db: 0.0,
            kind: ProgramKind::Waveform,
            loop_after_beats: None,
            evaluated: None,
            evaluated_text: Arc::default(),
            history: EditHistory::default(),
            generation: 0,
        }
    }

    /// Builds a `Program` from a `SourceBinding` plus its position in the
    /// file's bindings vec.
    ///
    /// Only `Definition`s with at least one annotation become programs.
    /// `Definition`s with no annotations are not returned.
    fn from_source_binding(
        sb: &expr::SourceBinding<MarkId, Source>,
        binding_index: usize,
        source: &str,
    ) -> Option<Program> {
        if sb.annotations.is_empty() {
            return None;
        }
        let mut color: Option<(u8, u8, u8)> = None;
        let mut level_db: f32 = 0.0;
        let mut kind = ProgramKind::Waveform;
        let mut loop_after_beats: Option<u32> = None;
        for sa in &sb.annotations {
            match &sa.annotation {
                expr::Annotation::Sliders(_) => {
                    // Read from the binding (see `ProgramSet::listed`).
                }
                expr::Annotation::Color(r, g, b) => {
                    color = Some((*r, *g, *b));
                }
                expr::Annotation::Level(v) => {
                    level_db = *v;
                }
                expr::Annotation::SkipSlots(_) => {
                    // Consumed by the position walk, not stored on Program.
                }
                expr::Annotation::Keys => {
                    kind = ProgramKind::Keys;
                }
                expr::Annotation::LoopAfterBeats(n) => {
                    loop_after_beats = Some(*n);
                }
            }
        }
        if let expr::Binding::Definition(_, expr) = &sb.binding {
            if let Some(s) = &expr.span
                && s.range.end <= source.len()
            {
                let span = s.range.clone();
                let text = source[s.range.clone()].to_string();
                Some(Program {
                    text,
                    span,
                    binding_index,
                    color,
                    level_db,
                    kind,
                    loop_after_beats,
                    evaluated: None,
                    evaluated_text: Arc::default(),
                    history: EditHistory::default(),
                    generation: 0,
                })
            } else {
                log::warn!(
                    "Found source expression without span or invalid span: {:?}",
                    sb
                );
                None
            }
        } else {
            // Nothing to do for other binding types.
            None
        }
    }

    /// Returns the program's source expression text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns true when the program's text is empty (a padding slot).
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Returns the program's display color from its `color=` annotation.
    pub fn color(&self) -> Option<(u8, u8, u8)> {
        self.color
    }

    /// Returns the program's declared kind.
    pub fn kind(&self) -> ProgramKind {
        self.kind
    }

    /// Returns the number of beats after which a pending launch plays the
    /// program again, or `None` when it plays once.
    pub fn loop_after_beats(&self) -> Option<u32> {
        self.loop_after_beats
    }

    /// Returns the program's output level in dB.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Returns the result of the program's last recorded evaluation, if that
    /// evaluation succeeded.
    ///
    /// The result reflects the text and the evaluation context (sibling
    /// bindings, opened modules) as of the evaluation: a text replaced since,
    /// or a dependency edited since, is not reflected until the program is
    /// evaluated again.
    pub fn evaluated(&self) -> Option<&Evaluated> {
        self.evaluated.as_ref()
    }

    /// Returns the text of the last evaluation recorded: the text the byte
    /// ranges of [`Program::evaluated`]'s marks index into.
    pub fn evaluated_text(&self) -> &Arc<str> {
        &self.evaluated_text
    }

    /// Returns the waveform if the last recorded evaluation produced one that
    /// isn't sequenceable.
    pub fn waveform(&self) -> Option<&waveform::Waveform<MarkId>> {
        match &self.evaluated {
            Some(Evaluated {
                plays: Plays::Waveform(waveform, _),
                ..
            }) => Some(waveform),
            _ => None,
        }
    }

    /// Returns whether the last recorded evaluation produced something to
    /// play: a waveform or a sequence.
    pub fn is_playable(&self) -> bool {
        matches!(
            self.evaluated,
            Some(Evaluated {
                plays: Plays::Waveform(..) | Plays::Sequence(_),
                ..
            })
        )
    }

    /// Returns the keys instrument if the last recorded evaluation produced
    /// one.
    pub fn keys_instrument(&self) -> Option<&KeysInstrument> {
        match &self.evaluated {
            Some(Evaluated {
                plays: Plays::KeysInstrument(instrument),
                ..
            }) => Some(instrument),
            _ => None,
        }
    }

    /// Returns the decomposed sequence if the last recorded evaluation produced
    /// a sequenceable waveform.
    pub fn sequence(&self) -> Option<&sequencer::Sequence> {
        match &self.evaluated {
            Some(Evaluated {
                plays: Plays::Sequence(sequence),
                ..
            }) => Some(sequence),
            _ => None,
        }
    }

    /// Replaces the program's `text` and advances its generation.
    ///
    /// The last evaluation's result stays until the next one is recorded.
    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.generation += 1;
    }

    /// Returns the program's generation: how many times its text has been
    /// replaced by [`Program::set_text`].
    ///
    /// Comparing two generations tells whether the text changed in between.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns the final character of the last coalesced insertion, or
    /// `None` when no insert run is open.
    pub fn last_inserted(&self) -> Option<char> {
        self.history.last_inserted
    }

    /// Records an undo point for typed text about to be inserted with the
    /// Edit-mode cursor at `cursor`.
    ///
    /// When `new_unit` is true, the current text and `cursor` are pushed as a
    /// fresh undo unit; otherwise the insertion extends the open insert run.
    /// Either way `last` becomes the run's last-inserted character and the redo
    /// stack is cleared.
    pub fn record_insert(&mut self, new_unit: bool, last: char, cursor: usize) {
        if new_unit {
            self.history.push_undo_capped(EditSnapshot {
                text: self.text.clone(),
                cursor,
            });
        }
        self.history.last_inserted = Some(last);
        self.history.redo.clear();
    }

    /// Records an undo point for an edit about to replace the program's text,
    /// with the Edit-mode cursor at `cursor`. The edit forms its own undo unit;
    /// any open insert run is closed and the redo stack cleared.
    pub fn record_edit(&mut self, cursor: usize) {
        self.history.push_undo_capped(EditSnapshot {
            text: self.text.clone(),
            cursor,
        });
        self.history.last_inserted = None;
        self.history.redo.clear();
    }

    /// Closes any open insert run so the next insertion starts a new undo unit.
    pub fn close_insert_run(&mut self) {
        self.history.last_inserted = None;
    }

    /// Restores the text and cursor from before the most recent undo unit,
    /// remembering the current text and `cursor` for `redo`. Returns the
    /// restored cursor position, or `None` when there is nothing to undo.
    pub fn undo(&mut self, cursor: usize) -> Option<usize> {
        let snapshot = self.history.undo.pop()?;
        let current = EditSnapshot {
            text: self.text().to_string(),
            cursor,
        };
        self.history.redo.push(current);
        self.set_text(snapshot.text);
        self.history.last_inserted = None;
        Some(snapshot.cursor)
    }

    /// Restores the most recently undone text and cursor, remembering the
    /// current text and `cursor` for `undo`. Returns the restored cursor
    /// position, or `None` when there is nothing to redo.
    pub fn redo(&mut self, cursor: usize) -> Option<usize> {
        let snapshot = self.history.redo.pop()?;
        let current = EditSnapshot {
            text: self.text().to_string(),
            cursor,
        };
        // Re-push without the cap or a redo-clear: this entry came from an undo
        // pop, so undo + redo together never exceed the cap, and redoing must
        // not discard the rest of the redo stack.
        self.history.undo.push(current);
        self.set_text(snapshot.text);
        self.history.last_inserted = None;
        Some(snapshot.cursor)
    }

    /// Sets the program's output level in dB.
    pub fn set_level_db(&mut self, level_db: f32) {
        self.level_db = level_db;
    }

    /// Sets the program's declared kind.
    ///
    /// The previous evaluation result is kept, though it may no longer match
    /// the kind until the program is evaluated again.
    pub fn set_kind(&mut self, kind: ProgramKind) {
        self.kind = kind;
    }

    /// Sets the program's loop: the number of beats after which a pending
    /// launch plays it again, or `None` to play once.
    pub fn set_loop_after_beats(&mut self, loop_after_beats: Option<u32>) {
        self.loop_after_beats = loop_after_beats;
    }

    /// Realigns the program with its binding after the source file was
    /// re-parsed: updates the binding index and span.
    ///
    /// Leaves `text` and the evaluation result alone, so `text` may still
    /// differ from `source[span]` when it holds unsaved edits.
    fn realign(&mut self, binding_index: usize, span: Range<usize>) {
        self.binding_index = binding_index;
        self.span = span;
    }

    /// Replaces `text` with the program's binding in `source`. However, the
    /// text must be unchanged semantically. This does *not* update the
    /// evaluation results or the generation.
    fn resync_text(&mut self, source: &str) {
        self.text = source[self.span.clone()].to_string();
    }

    /// Marks the program as a padding slot with no binding: the binding
    /// index points one past the end of the bindings and the span is empty.
    fn mark_padding(&mut self, binding_count: usize) {
        self.binding_index = binding_count;
        self.span = 0..0;
    }
}

/// Returns the last `skip_slots=N` value on `sb`, or 0 if none is present.
fn read_skip_slots(sb: &expr::SourceBinding<MarkId, Source>) -> u32 {
    // The reverse walk mirrors the "last wins" semantics used elsewhere for
    // repeated annotations of the same kind.
    sb.annotations
        .iter()
        .rev()
        .find_map(|sa| match &sa.annotation {
            expr::Annotation::SkipSlots(n) => Some(*n),
            _ => None,
        })
        .unwrap_or(0)
}

/// Walks `bindings` in source order and returns one `(position, binding_index,
/// expr_span)` triple per UI-program binding.
///
/// A UI-program binding is any annotated `Definition` whose expression span is
/// valid. `position` starts at 0 and advances by `skip_slots + 1` for each such
/// binding — the same walk `ProgramSet::from_source` uses to assign grid slots.
fn walk_ui_positions(
    bindings: &[expr::SourceBinding<MarkId, Source>],
    source_len: usize,
) -> Vec<(usize, usize, Range<usize>)> {
    let mut out = Vec::new();
    let mut position: usize = 0;
    for (i, sb) in bindings.iter().enumerate() {
        if sb.annotations.is_empty() {
            continue;
        }
        if let expr::Binding::Definition(_, expr) = &sb.binding
            && let Some(s) = &expr.span
            && s.range.end <= source_len
        {
            position += read_skip_slots(sb) as usize;
            out.push((position, i, s.range.clone()));
            position += 1;
        }
    }
    out
}

/// What a name resolves to in a program's scope (see
/// [`ProgramSet::resolve_name`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundTo {
    /// The binding of the program at this index.
    Program(usize),
    /// Some other binding, a module's definition, or the prelude.
    Other,
    /// Nothing: the name is unbound.
    Unbound,
}

/// The set of programs backed by a source file: the file's contents, its parsed
/// bindings, and one `Program` per UI slot.
///
/// Owns the program ↔ source alignment: programs are created from annotated
/// bindings (any binding with `#{...}`) in source order, offset by any
/// `skip_slots=N`, and keep their `binding_index`/`span` pointing back into
/// `bindings`/`source`.
pub struct ProgramSet {
    programs: Vec<Program>,
    bindings: Vec<expr::SourceBinding<MarkId, Source>>,
    source: String,
    input_path: path::PathBuf,
    /// The backing file's modification time as of our last read or write, used
    /// to detect external edits before overwriting them. `None` when there is
    /// no backing file.
    synced_mtime: Option<time::SystemTime>,
    /// The state of every slider a program can reach, by the slot that
    /// provides it and its label.
    sliders: HashMap<(usize, String), SliderState>,
}

/// Returns the current modification time of `path`, or `None` when the path is
/// empty or can't be stat-ed.
fn file_mtime(path: &path::Path) -> Option<time::SystemTime> {
    if path.as_os_str().is_empty() {
        return None;
    }
    fs::metadata(path).and_then(|m| m.modified()).ok()
}

impl ProgramSet {
    /// Builds a `ProgramSet` from the contents of a source file, along with a
    /// warning message for recoverable parse errors (empty when the parse was
    /// clean). Fills every UI slot with an empty padding program, then walks
    /// annotated bindings in source order, assigning each to the next slot
    /// after any `#{skip_slots=N}` gap.
    ///
    /// `input_path` is the file the splice path writes back to (use an empty
    /// `PathBuf` to suppress the write, e.g. in tests).
    pub fn from_source(
        source: String,
        input_path: path::PathBuf,
    ) -> Result<(ProgramSet, String), Vec<expr::Error<Source>>> {
        let mut message = String::new();
        let (bindings, errors) = parser::parse_module::<MarkId, _>(&source, Source::File)?;
        // TODO sort of a bummer that we don't know which binding this error was
        // in... some opportunity here to improve the type of parse_module.
        if !errors.is_empty() {
            message = match errors.len() {
                1 => format!("Parse error: {}", errors[0].display_with_source(&source)),
                n => format!(
                    "Parse error: {} (+{} more)",
                    errors[0].display_with_source(&source),
                    n - 1
                ),
            };
        }
        let total_slots = NUM_PROGRAM_BANKS * PROGRAMS_PER_BANK;
        let mut programs: Vec<Program> = (0..total_slots)
            .map(|_| Program::from_string("", bindings.len()))
            .collect();
        let mut position: usize = 0;
        for (binding_index, sb) in bindings.iter().enumerate() {
            if let Some(program) = Program::from_source_binding(sb, binding_index, &source) {
                position += read_skip_slots(sb) as usize;
                if position < programs.len() {
                    programs[position] = program;
                } else {
                    log::warn!(
                        "Ignoring program with out-of-range slot {} (max {})",
                        position + 1,
                        programs.len()
                    );
                }
                position += 1;
            }
        }
        let synced_mtime = file_mtime(&input_path);
        Ok((
            ProgramSet {
                programs,
                bindings,
                source,
                input_path,
                synced_mtime,
                sliders: HashMap::new(),
            },
            message,
        ))
    }

    /// Returns true when the backing file has been modified (or removed) since
    /// this set last read or wrote it.
    pub fn disk_changed(&self) -> bool {
        match self.synced_mtime {
            None => false,
            Some(synced) => file_mtime(&self.input_path) != Some(synced),
        }
    }

    /// Returns the message that refuses a write because the backing file has
    /// changed on disk (see [`ProgramSet::disk_changed`]).
    pub fn stale_message(&self) -> String {
        format!(
            "Not saved: {} changed on disk (reload first)",
            self.display_path()
        )
    }

    /// Returns the `skip_slots` of the binding of the program at `index`, or
    /// `None` when the slot has no binding.
    pub fn skip_slots(&self, index: usize) -> Option<u32> {
        let program = self.programs.get(index)?;
        self.bindings
            .get(program.binding_index)
            .map(read_skip_slots)
    }

    /// Returns the slot of the last program in the file, counting a program
    /// pushed past the last slot (which isn't loaded), or `None` when the file
    /// has no programs.
    pub fn last_slot(&self) -> Option<usize> {
        walk_ui_positions(&self.bindings, self.source.len())
            .last()
            .map(|(position, _, _)| *position)
    }

    /// Writes the backing file with the binding of the program at `index` given
    /// `skip_slots`, and returns the new source.
    ///
    /// The set itself is left as it was, so it reads as changed on disk until
    /// the new source is loaded. Refuses, with a user-visible message, when the
    /// slot has no binding or the backing file has changed on disk.
    pub fn write_skip_slots(&mut self, index: usize, skip_slots: u32) -> Result<String, String> {
        if self.disk_changed() {
            return Err(self.stale_message());
        }
        let program = &self.programs[index];
        let Some(binding) = self.bindings.get(program.binding_index) else {
            return Err(format!("{} is empty", self.display_name(index)));
        };
        let mut new_source = self.source.clone();
        if let Some((span, replacement)) =
            skip_slots_edit(binding, skip_slots, program.level_db(), &self.source)
        {
            new_source.replace_range(span, &replacement);
        }
        // Skip the file write when no path is set, as `splice` does.
        if !self.input_path.as_os_str().is_empty()
            && let Err(e) = fs::write(&self.input_path, &new_source)
        {
            return Err(format!(
                "Warning: failed to write {}: {}",
                self.input_path.display(),
                e
            ));
        }
        Ok(new_source)
    }

    /// Returns the current contents of the backing file, or the error message
    /// when there is no backing file or it can't be read.
    pub fn read_backing_file(&self) -> Result<String, String> {
        if self.input_path.as_os_str().is_empty() {
            return Err("No backing file to reload".to_string());
        }
        fs::read_to_string(&self.input_path).map_err(|e| {
            format!(
                "Failed to read {}: {}",
                self.input_path.to_string_lossy(),
                e
            )
        })
    }

    /// Returns the path of the backing file (empty when there is none).
    pub fn input_path(&self) -> &path::Path {
        &self.input_path
    }

    /// Returns the backing file's name for user-visible messages.
    pub fn display_path(&self) -> String {
        self.input_path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.input_path.to_string_lossy().into_owned())
    }

    /// Returns all program slots in slot order.
    pub fn programs(&self) -> &[Program] {
        &self.programs
    }

    /// Returns the program at `index`, or `None` if the index is out of
    /// range.
    pub fn program(&self, index: usize) -> Option<&Program> {
        self.programs.get(index)
    }

    /// Returns a mutable reference to the program at `index`, or `None` if
    /// the index is out of range.
    ///
    /// Safe to hand out because `Program`'s fields are private — callers
    /// can only mutate through `Program`'s invariant-preserving methods.
    pub fn program_mut(&mut self, index: usize) -> Option<&mut Program> {
        self.programs.get_mut(index)
    }

    /// Returns the name to show in the status line for the program at
    /// `index`, derived from its binding's pattern:
    /// - `Definition` with an `Identifier("_")` → empty (anonymous, don't
    ///   clutter the status line).
    /// - `Definition` with any other identifier or a tuple pattern → the
    ///   pattern's `Display` form.
    /// - No binding (padding slot) or an `Open`/`Empty` binding → empty.
    pub fn name(&self, index: usize) -> String {
        let Some(program) = self.programs.get(index) else {
            return String::new();
        };
        let Some(binding) = self.bindings.get(program.binding_index) else {
            return String::new();
        };
        match &binding.binding {
            expr::Binding::Definition(pattern, _) => match pattern {
                expr::Pattern::Identifier(name) if name == "_" => String::new(),
                _ => format!("{}", pattern),
            },
            _ => String::new(),
        }
    }

    /// Returns the name the program at `index` is bound to, or `None` when its
    /// binding is not a single identifier other than `_` (a tuple pattern, an
    /// anonymous or missing binding).
    pub fn binding_name(&self, index: usize) -> Option<&str> {
        let program = self.programs.get(index)?;
        match &self.bindings.get(program.binding_index)?.binding {
            expr::Binding::Definition(expr::Pattern::Identifier(name), _) if name != "_" => {
                Some(name)
            }
            _ => None,
        }
    }

    /// Returns what `name` resolves to in the scope of the program at `index`:
    /// the bindings before it in source order (for an empty slot, those before
    /// the binding of the next program, where its binding would be inserted),
    /// then the prelude.
    pub fn resolve_name(&self, environment: &Environment, index: usize, name: &str) -> BoundTo {
        let program = &self.programs[index];
        let end = if program.binding_index < self.bindings.len() {
            program.binding_index
        } else {
            walk_ui_positions(&self.bindings, self.source.len())
                .into_iter()
                .find(|(position, _, _)| *position > index)
                .map_or(self.bindings.len(), |(_, binding_index, _)| binding_index)
        };
        for (binding_index, source_binding) in self.bindings[..end].iter().enumerate().rev() {
            match &source_binding.binding {
                expr::Binding::Definition(pattern, _) if pattern.binds(name) => {
                    return self
                        .programs
                        .iter()
                        .position(|p| p.binding_index == binding_index)
                        .map_or(BoundTo::Other, BoundTo::Program);
                }
                expr::Binding::Use(path) if path.last().is_some_and(|last| last == name) => {
                    return BoundTo::Other;
                }
                expr::Binding::Open(path) if environment.module_defines(path, name) => {
                    return BoundTo::Other;
                }
                _ => {}
            }
        }
        if environment.module_defines(&["__prelude".to_string()], name) {
            BoundTo::Other
        } else {
            BoundTo::Unbound
        }
    }

    /// Returns a user-facing label for the program at `index`.
    ///
    /// Prefers the binding's name (the identifier it was bound to in
    /// source). Falls back to a bank-relative address like `"B:3"` where
    /// the letter is the bank (A..H) and the digit is a 1-based position
    /// within the bank — the same digit the user would type to select it.
    ///
    /// **Convention:** user-visible strings that refer to a program must go
    /// through this helper. Do NOT interpolate a raw program index (or
    /// `index + 1`) — the grid has 8 banks of 8 slots, so a raw index
    /// doesn't match the keystroke the user would use, and program indices
    /// don't survive future layout changes.
    pub fn display_name(&self, index: usize) -> String {
        if self.programs.get(index).is_none() {
            return String::new();
        }
        let address = bank_address(index);
        let name = self.name(index);
        if name.is_empty() {
            address
        } else {
            format!("{} ({})", address, name)
        }
    }

    /// Returns the status message for selecting the program at `index`.
    ///
    /// The message is the bank-relative address, then the binding's name (if
    /// any), then the program's loop or keys setting. An empty slot gives an
    /// empty message.
    pub fn selection_message(&self, index: usize) -> String {
        let Some(program) = self.programs.get(index) else {
            return String::new();
        };
        if program.is_empty() {
            return String::new();
        }
        let mut message = bank_address(index);
        let name = self.name(index);
        if !name.is_empty() {
            message = format!("{} {}", message, name);
        }
        if let Some(beats) = program.loop_after_beats() {
            message = format!("{}, loops every {} beats", message, beats);
        } else if program.kind() == ProgramKind::Keys {
            message = format!("{}, plays keys", message);
        }
        message
    }

    /// Returns the bindings to evaluate the program at `index` under: the
    /// file-level bindings preceding it (bindings after it are ignored), with
    /// anonymous `_` definitions filtered out.
    ///
    /// These are the first `counts[index]` of the bindings
    /// [`ProgramSet::used_evaluation_bindings`] returns.
    pub fn evaluation_bindings(&self, index: usize) -> Vec<expr::SourceBinding<MarkId, Source>> {
        (0..self.programs[index].binding_index)
            .filter_map(|binding_index| self.evaluation_binding(binding_index))
            .collect()
    }

    /// Returns the file-level bindings programs are evaluated under, in
    /// source order, and for each program the number of them it reads.
    ///
    /// The program at `index` is evaluated under the first `counts[index]`
    /// bindings, which is what [`ProgramSet::evaluation_bindings`] returns.
    pub fn used_evaluation_bindings(
        &self,
    ) -> (Vec<expr::SourceBinding<MarkId, Source>>, Vec<usize>) {
        let mut bindings = Vec::new();
        // The number of bindings that precede each binding index.
        let mut preceding = Vec::with_capacity(self.bindings.len() + 1);
        for binding_index in 0..self.bindings.len() {
            preceding.push(bindings.len());
            bindings.extend(self.evaluation_binding(binding_index));
        }
        // Padding slots sit past every binding.
        preceding.push(bindings.len());
        let counts = self
            .programs
            .iter()
            .map(|program| preceding[program.binding_index])
            .collect();
        (bindings, counts)
    }

    /// Returns the binding at `binding_index` as later programs read it, or
    /// `None` for an anonymous `_` definition.
    fn evaluation_binding(
        &self,
        binding_index: usize,
    ) -> Option<expr::SourceBinding<MarkId, Source>> {
        let source_binding = &self.bindings[binding_index];
        // Anonymous definitions are not referenceable, so evaluating them
        // would be wasted work — and a broken one would poison every later
        // program's context.
        match &source_binding.binding {
            expr::Binding::Definition(expr::Pattern::Identifier(v), _) if v == "_" => None,
            _ => Some(source_binding.clone()),
        }
    }

    /// Returns a copy of what evaluating the program at `index` reads.
    pub fn evaluation_input(&self, index: usize) -> EvaluationInput {
        let program = &self.programs[index];
        EvaluationInput {
            text: program.text.clone(),
            kind: program.kind,
            bindings: self.evaluation_bindings(index),
            source: self.source.clone(),
            index,
            providers: self.slider_providers(),
        }
    }

    /// Returns the program slots of the file as providers of sliders, by
    /// index.
    pub fn slider_providers(&self) -> slider_providers::Providers {
        let slot = |index: usize| {
            let program = &self.programs[index];
            // The last one wins, as in `listed`.
            let listing = self
                .bindings
                .get(program.binding_index)
                .and_then(|binding| {
                    binding
                        .annotations
                        .iter()
                        .rev()
                        .find(|a| matches!(a.annotation, expr::Annotation::Sliders(_)))
                })
                .and_then(|annotation| annotation.span.clone());
            slider_providers::ProviderSlot {
                name: self.display_name(index),
                text: program.span.clone(),
                listed: self.listed(index).to_vec(),
                listing,
            }
        };
        slider_providers::Providers {
            slots: (0..self.programs.len()).map(slot).collect(),
        }
    }

    /// Returns the backing source text the program set was loaded from.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Records `outcome` as the result of evaluating `text` as the program at
    /// `index`, and returns its diagnostics: the warnings on success (none for
    /// an empty program), the errors (and any warnings) on failure.
    ///
    /// When the program evaluated, the sliders it can reach are reached (see
    /// [`ProgramSet::reach`]). Otherwise any previous result is dropped.
    pub fn record(
        &mut self,
        index: usize,
        text: Arc<str>,
        outcome: Outcome,
    ) -> Result<Vec<Diagnostic>, Vec<Diagnostic>> {
        self.programs[index].evaluated_text = text;
        match outcome {
            Outcome::Empty => {
                self.programs[index].evaluated = None;
                Ok(Vec::new())
            }
            Outcome::Evaluated(evaluated, warnings) => {
                self.reach(&evaluated.sliders);
                self.programs[index].evaluated = Some(evaluated);
                Ok(warnings)
            }
            Outcome::Failed(diagnostics) => {
                self.programs[index].evaluated = None;
                Err(diagnostics)
            }
        }
    }

    /// Returns the entries of the `sliders` annotation of the program at
    /// `index`, in annotation order; empty when it has none.
    pub fn listed(&self, index: usize) -> &[expr::Slider] {
        self.programs
            .get(index)
            .and_then(|program| self.bindings.get(program.binding_index))
            .and_then(|binding| {
                // The last one wins, as the parser reads repeated annotations.
                binding.annotations.iter().rev().find_map(|annotation| {
                    match &annotation.annotation {
                        expr::Annotation::Sliders(entries) => Some(entries.as_slice()),
                        _ => None,
                    }
                })
            })
            .unwrap_or(&[])
    }

    /// Returns the state of the slider `label` the slot at `provider`
    /// provides, once a program can reach it.
    pub fn slider(&self, provider: usize, label: &str) -> Option<&SliderState> {
        self.sliders.get(&(provider, label.to_string()))
    }

    /// Returns the mark and current value of every slider a program's
    /// waveform has reached.
    pub fn slider_values(&self) -> impl Iterator<Item = (MarkId, f32)> + '_ {
        self.sliders.iter().map(|((program, label), state)| {
            (
                MarkId::Slider {
                    program: *program,
                    label: label.clone(),
                },
                state.value,
            )
        })
    }

    /// Adds each slider in `reached` that is new at the value its provider's
    /// annotation saves (clamped to its range) or, when that saves none, at
    /// its default.
    ///
    /// A slider already reached keeps its position: its value is unchanged
    /// when its spec is, and otherwise moves to the same position along the
    /// new spec.
    pub fn reach(&mut self, reached: &[slider_analysis::Reached]) {
        for slider in reached {
            let key = (slider.provider, slider.label.clone());
            let value = match self.sliders.get(&key) {
                Some(state) if state.spec == slider.spec => continue,
                Some(state) => slider.spec.denormalize(state.spec.normalize(state.value)),
                None => {
                    let saved = self
                        .listed(slider.provider)
                        .iter()
                        .find(|entry| entry.label == slider.label)
                        .and_then(|entry| entry.value);
                    saved.map_or(slider.spec.default, |value| slider.spec.clamp(value))
                }
            };
            self.sliders.insert(
                key,
                SliderState {
                    spec: slider.spec.clone(),
                    value,
                },
            );
        }
    }

    /// Returns the labels of the sliders on the encoders of the program at
    /// `index`, in encoder order.
    ///
    /// These are the sliders it lists, in annotation order, then the sliders it
    /// provides but doesn't list ("auto-shown"), in the order its last
    /// successful evaluation found them (see [`Evaluated::sliders`]).
    pub fn shown(&self, index: usize) -> Vec<&str> {
        let listed = self.listed(index);
        let mut shown: Vec<&str> = listed.iter().map(|entry| entry.label.as_str()).collect();
        // TODO maybe also show sliders provided by earlier bindings this
        // program is built through, labeled with their provider? turning one
        // would also
        // move other programs that share it.
        let reached = self
            .programs
            .get(index)
            .and_then(Program::evaluated)
            .map_or(&[][..], |evaluated| &evaluated.sliders);
        for slider in reached {
            if slider.provider == index && !shown.contains(&slider.label.as_str()) {
                shown.push(&slider.label);
            }
        }
        shown
    }

    /// Returns the label and state of the slider at `slider_index` among
    /// those [`ProgramSet::shown`] returns, or `None` when there are fewer or
    /// no program's waveform has reached that slider.
    pub fn shown_slider(&self, index: usize, slider_index: usize) -> Option<(&str, &SliderState)> {
        let label = *self.shown(index).get(slider_index)?;
        // TODO a listed slider nothing has reached has no spec, so its encoder
        // does nothing and shows nothing; consider showing it as unreached.
        Some((label, self.slider(index, label)?))
    }

    /// Returns the position, from 0 to 1, of the slider
    /// [`ProgramSet::shown_slider`] returns.
    pub fn slider_normalized(&self, index: usize, slider_index: usize) -> Option<f32> {
        let (_, state) = self.shown_slider(index, slider_index)?;
        Some(state.spec.normalize(state.value))
    }

    /// Moves the slider [`ProgramSet::shown_slider`] returns to position
    /// `normalized` (see [`slider::SliderSpec::denormalize`]) and returns its
    /// label and new value, or `None` when there is no such slider.
    pub fn set_slider_normalized(
        &mut self,
        index: usize,
        slider_index: usize,
        normalized: f32,
    ) -> Option<SliderChange> {
        let label = self.shown(index).get(slider_index)?.to_string();
        let state = self.sliders.get_mut(&(index, label.clone()))?;
        state.value = state.spec.denormalize(normalized);
        Some(SliderChange {
            label,
            value: state.value,
        })
    }

    /// Returns how to show each slider [`ProgramSet::shown`] returns for the
    /// program at `index` that a program's waveform has reached.
    // TODO this method only works for mouse-based sliders
    pub fn slider_display(&self, index: usize) -> Vec<SliderDisplay> {
        (0..self.shown(index).len())
            .filter_map(|j| {
                let (label, state) = self.shown_slider(index, j)?;
                Some(SliderDisplay {
                    label: label.to_string(),
                    // TODO this is wrong for encoders
                    axis: match j {
                        0 => "X".to_string(),
                        1 => "Y".to_string(),
                        // Only encoders reach the rest.
                        _ => (j + 1).to_string(),
                    },
                    normalized_value: state.spec.normalize(state.value),
                    actual_value: state.value,
                })
            })
            .collect()
    }
}

/// Float tolerance for "this hasn't moved" comparisons against parsed
/// annotation values. Tighter than the source format's precision would
/// matter (which would be confusing) and looser than encoder/MIDI noise.
const ANNOTATION_EPSILON: f32 = 1e-4;

/// Returns annotation-persistence edits for `program`, the program at `index`,
/// against its parsed `binding`.
///
/// This includes one each for `keys` and `loop_after_beats` if the program's
/// kind or loop has diverged from the binding's annotations, one for `level_db`
/// if the runtime level has diverged from what the binding's last `Level`
/// annotation encodes, and one for `sliders` as [`sliders_edit`] returns it.
/// Returns an empty list when nothing has changed.
fn annotation_edits(
    program: &Program,
    index: usize,
    binding: &expr::SourceBinding<MarkId, Source>,
    source: &str,
    sliders: &HashMap<(usize, String), SliderState>,
) -> Vec<(Range<usize>, String)> {
    let keys = (program.kind() == ProgramKind::Keys).then_some(expr::Annotation::Keys);
    let loop_after_beats = program
        .loop_after_beats()
        .map(expr::Annotation::LoopAfterBeats);
    let presence_edits = [
        (
            keys.is_none(),
            annotation_edit(
                binding,
                |a| matches!(a, expr::Annotation::Keys),
                keys.as_ref(),
                program.level_db(),
                source,
            ),
        ),
        (
            loop_after_beats.is_none(),
            annotation_edit(
                binding,
                |a| matches!(a, expr::Annotation::LoopAfterBeats(_)),
                loop_after_beats.as_ref(),
                program.level_db(),
                source,
            ),
        ),
    ];
    // Removing a lone annotation writes `level_db=…` in its place, so a level
    // edit would write it twice.
    let replaces_lone_annotation = binding.annotations.len() == 1
        && presence_edits
            .iter()
            .any(|(removes, edit)| *removes && edit.is_some());
    let mut edits = Vec::new();
    if !replaces_lone_annotation && let Some(edit) = level_edit(program, binding, source) {
        edits.push(edit);
    }
    if let Some(edit) = sliders_edit(program, index, binding, source, sliders) {
        edits.push(edit);
    }
    // Pushed last so that an annotation inserted at the start of another
    // annotation's span is applied after that span's replacement.
    edits.extend(presence_edits.into_iter().filter_map(|(_, edit)| edit));
    edits
}

/// Returns the edit for the binding's `level_db` annotation, or `None` if
/// the runtime level matches what the binding currently encodes.
fn level_edit(
    program: &Program,
    binding: &expr::SourceBinding<MarkId, Source>,
    source: &str,
) -> Option<(Range<usize>, String)> {
    let (parsed_value, parsed_span) = match last_annotation_of(binding, |a| match a {
        expr::Annotation::Level(v) => Some(*v),
        _ => None,
    }) {
        Some((v, span)) => (v, span),
        None => (0.0, None),
    };
    if (program.level_db() - parsed_value).abs() < ANNOTATION_EPSILON {
        return None;
    }
    let annotation = expr::Annotation::Level(program.level_db());
    let body = format!("{}", annotation);
    match parsed_span {
        Some(span) => Some((span, body)),
        None => {
            // Fall back to inserting a fresh `level_db=…` annotation when no
            // `Level` annotation exists in source yet.
            Some(insert_annotation_edit(binding, &body, source))
        }
    }
}

/// Returns the edit that makes the binding's last annotation matched by
/// `matches` read `wanted`, or `None` if it already does.
///
/// A `wanted` of `None` removes the annotation. Removing the binding's only
/// annotation writes `level_db=…` (at `level_db`) in its place, so the binding
/// stays a UI program.
fn annotation_edit(
    binding: &expr::SourceBinding<MarkId, Source>,
    matches: impl Fn(&expr::Annotation) -> bool,
    wanted: Option<&expr::Annotation>,
    level_db: f32,
    source: &str,
) -> Option<(Range<usize>, String)> {
    let parsed = last_annotation_of(binding, |a| matches(a).then(|| a.to_string()));
    match (wanted, parsed) {
        (Some(wanted), None) => Some(insert_annotation_edit(binding, &wanted.to_string(), source)),
        (Some(wanted), Some((parsed, Some(span)))) if parsed != wanted.to_string() => {
            Some((span, wanted.to_string()))
        }
        (None, Some((_, Some(span)))) if binding.annotations.len() == 1 => {
            Some((span, expr::Annotation::Level(level_db).to_string()))
        }
        // TODO a repeated annotation keeps all but the last.
        (None, Some((_, Some(span)))) => Some(remove_annotation_edit(span, source)),
        _ => None,
    }
}

/// Builds an `(range, replacement)` edit that inserts a fresh `#{…}`
/// annotation line at `pos`, prepending or appending `\n` only when
/// needed so the inserted line doesn't collapse onto a neighbor.
fn insert_annotation_line(pos: usize, body: &str, source: &str) -> (Range<usize>, String) {
    let bytes = source.as_bytes();
    let prefix = if pos == 0 || bytes.get(pos - 1) == Some(&b'\n') {
        ""
    } else {
        "\n"
    };
    let suffix = if bytes.get(pos) == Some(&b'\n') {
        ""
    } else {
        "\n"
    };
    (pos..pos, format!("{}#{{{}}}{}", prefix, body, suffix))
}

/// Builds the edit that adds the annotation `body` to `binding`.
///
/// The annotation is prepended to the binding's first `#{…}` set, or, when the
/// binding has no annotations, written as a fresh `#{…}` line just above its
/// pattern (below any leading comments).
fn insert_annotation_edit(
    binding: &expr::SourceBinding<MarkId, Source>,
    body: &str,
    source: &str,
) -> (Range<usize>, String) {
    if let Some(first) = binding.annotations.first().and_then(|a| a.span.as_ref()) {
        let pos = first.range.start;
        return (pos..pos, format!("{}, ", body));
    }
    // No annotations: skip the leading trivia to the pattern.
    let mut pos = binding
        .span
        .as_ref()
        .expect("parsed binding has span")
        .range
        .start;
    loop {
        let rest = source[pos..].trim_start();
        pos = source.len() - rest.len();
        if !rest.starts_with("//") {
            return insert_annotation_line(pos, body, source);
        }
        pos += rest.find('\n').unwrap_or(rest.len());
    }
}

/// Returns the offset of the newline that ends the line containing `pos`, when
/// the rest of that line is blank or a `//` comment.
///
/// Returns `None` when other code follows `pos` on its line or the line has no
/// newline.
fn trivia_line_end(pos: usize, source: &str) -> Option<usize> {
    let rest = &source[pos..];
    let len = rest.find('\n')?;
    let line = rest[..len].trim_start();
    (line.is_empty() || line.starts_with("//")).then_some(pos + len)
}

/// Returns the offset where a binding inserted after `pred` begins: just past
/// `pred`'s `;`, or past the end of its line when only blank space or a
/// comment follows it there.
///
/// Returns 0 when `pred` is `None`.
fn insertion_point(pred: Option<&expr::SourceBinding<MarkId, Source>>, source: &str) -> usize {
    let Some(pred) = pred else {
        return 0;
    };
    let end = pred
        .span
        .as_ref()
        .expect("parsed binding has span")
        .range
        .end;
    trivia_line_end(end, source).map_or(end, |newline| newline + 1)
}

/// Builds the edit that sets `binding`'s `skip_slots` annotation to `new_skip`,
/// or `None` when the source already encodes that value.
///
/// Replaces the existing annotation in place, adds `skip_slots=…` when none
/// exists, or removes the annotation entirely when `new_skip` is
/// 0 — swapping in `level_db=…` instead when `skip_slots` is the binding's only
/// annotation, so the binding stays a UI program.
fn skip_slots_edit(
    binding: &expr::SourceBinding<MarkId, Source>,
    new_skip: u32,
    level_db: f32,
    source: &str,
) -> Option<(Range<usize>, String)> {
    if read_skip_slots(binding) == new_skip {
        return None;
    }
    let existing = last_annotation_of(binding, |a| match a {
        expr::Annotation::SkipSlots(_) => Some(()),
        _ => None,
    })
    .and_then(|(_, span)| span);
    let body = format!("{}", expr::Annotation::SkipSlots(new_skip));
    match existing {
        Some(span) if new_skip > 0 => Some((span, body)),
        Some(span) if binding.annotations.len() == 1 => {
            Some((span, format!("{}", expr::Annotation::Level(level_db))))
        }
        Some(span) => Some(remove_annotation_edit(span, source)),
        None if new_skip > 0 => Some(insert_annotation_edit(binding, &body, source)),
        // No annotation and no skip needed.
        None => None,
    }
}

/// Builds the edit that removes the annotation at `span` from its `#{…}` set,
/// consuming an adjacent comma (and surrounding spacing) so the remaining
/// annotations stay well-formed. When the annotation is alone in its set,
/// removes the whole `#{…}` group and any trailing newline.
fn remove_annotation_edit(span: Range<usize>, source: &str) -> (Range<usize>, String) {
    let bytes = source.as_bytes();
    // A following comma: the annotation is first or interior in its set.
    let mut end = span.end;
    while bytes.get(end).is_some_and(u8::is_ascii_whitespace) {
        end += 1;
    }
    if bytes.get(end) == Some(&b',') {
        end += 1;
        while bytes.get(end) == Some(&b' ') || bytes.get(end) == Some(&b'\t') {
            end += 1;
        }
        return (span.start..end, String::new());
    }
    // A preceding comma: the annotation is last in its set.
    let mut start = span.start;
    while start > 0 && bytes[start - 1].is_ascii_whitespace() {
        start -= 1;
    }
    if start > 0 && bytes[start - 1] == b',' {
        return (start - 1..span.end, String::new());
    }
    // Alone in its set: remove the whole `#{…}` group. `start - 1` is the
    // set's `{`; scan back over the optional whitespace to its `#`.
    debug_assert_eq!(bytes.get(start - 1), Some(&b'{'));
    let mut set_start = start - 1;
    while set_start > 0 && bytes[set_start - 1].is_ascii_whitespace() {
        set_start -= 1;
    }
    debug_assert_eq!(bytes.get(set_start - 1), Some(&b'#'));
    set_start -= 1;
    // `end` sits on the set's `}`; also consume a trailing newline so the
    // removal doesn't leave a blank line behind.
    let mut set_end = end + 1;
    if bytes.get(set_end) == Some(&b'\n') {
        set_end += 1;
    }
    (set_start..set_end, String::new())
}

/// Returns the edit that saves, in `binding`'s `sliders` annotation, the value
/// in `sliders` of each slider owned by `program`, the program at `index`, or
/// `None` when nothing needs saving.
///
/// A listed slider is saved when its value differs from what its entry saves
/// (or its default, for an entry that saves none). A slider `program` reaches
/// and owns but doesn't list is appended as a new entry once its value
/// differs from its default, after a bare entry for each such slider shown
/// before it, so that saving keeps the order of [`ProgramSet::shown`]. The
/// annotation is added if the binding has none.
fn sliders_edit(
    program: &Program,
    index: usize,
    binding: &expr::SourceBinding<MarkId, Source>,
    source: &str,
    sliders: &HashMap<(usize, String), SliderState>,
) -> Option<(Range<usize>, String)> {
    let moved = |label: &str, from: Option<f32>| {
        let state = sliders.get(&(index, label.to_string()))?;
        ((state.value - from.unwrap_or(state.spec.default)).abs() > ANNOTATION_EPSILON)
            .then_some(state.value)
    };
    let entries = last_annotation_of(binding, |a| match a {
        expr::Annotation::Sliders(entries) => Some(entries.clone()),
        _ => None,
    })
    .map_or(Vec::new(), |(entries, _)| entries);
    let mut changed = false;
    let mut updated: Vec<expr::Slider> = entries
        .into_iter()
        .map(|entry| match moved(&entry.label, entry.value) {
            Some(value) => {
                changed = true;
                expr::Slider {
                    label: entry.label,
                    value: Some(value),
                }
            }
            None => entry,
        })
        .collect();
    let reached = program.evaluated().map_or(&[][..], |e| &e.sliders);
    let mut shown: Vec<expr::Slider> = Vec::new();
    for slider in reached {
        if slider.provider == index
            && !updated
                .iter()
                .chain(&shown)
                .any(|entry| entry.label == slider.label)
        {
            shown.push(expr::Slider {
                label: slider.label.clone(),
                value: moved(&slider.label, None),
            });
        }
    }
    // Unmoved sliders shown after the last moved one aren't written.
    if let Some(last) = shown.iter().rposition(|entry| entry.value.is_some()) {
        changed = true;
        updated.extend(shown.into_iter().take(last + 1));
    }
    if !changed {
        return None;
    }
    annotation_edit(
        binding,
        |a| matches!(a, expr::Annotation::Sliders(_)),
        Some(&expr::Annotation::Sliders(updated)),
        program.level_db(),
        source,
    )
}

/// Returns the (value, span) of the last annotation on `binding` that `pick`
/// matches.
///
/// The reverse walk matches the parser's "last wins" semantics for repeated
/// annotations of the same kind, so persisting onto the same span keeps the
/// source authoritative.
fn last_annotation_of<T, F>(
    binding: &expr::SourceBinding<MarkId, Source>,
    mut pick: F,
) -> Option<(T, Option<Range<usize>>)>
where
    F: FnMut(&expr::Annotation) -> Option<T>,
{
    binding
        .annotations
        .iter()
        .rev()
        .find_map(|a| pick(&a.annotation).map(|v| (v, a.span.as_ref().map(|s| s.range.clone()))))
}

impl ProgramSet {
    /// Splices the given program's edited `text` back into `self.source`,
    /// persists every program's changed state as annotation edits, re-parses
    /// the file to refresh `self.bindings`, realigns each program's
    /// `span`/`binding_index` via the position walk, and writes the new source
    /// to `self.input_path`.
    ///
    /// Programs with no source binding (padding slots) are treated as
    /// brand-new: a fresh `Definition` is inserted between its source-order
    /// neighbors with a `#{skip_slots=…, level_db=…}` annotation, and the
    /// following program's `skip_slots` is decremented to keep its absolute
    /// slot stable. The new binding goes just past its predecessor's line (so
    /// comments above the next binding stay with it), with a blank line on each
    /// side and a newline after it. Existing programs are spliced in place —
    /// except when the edited text is empty (or whitespace-only), which deletes
    /// the binding (annotations included) from source and grows the following
    /// program's `skip_slots` to compensate.
    ///
    /// Returns a user-visible warning message if any step failed; in that case
    /// `self.bindings`, `self.source`, and the file on disk are all left
    /// unchanged.
    pub fn splice(&mut self, program_index: usize) -> Result<(), String> {
        // Never overwrite external edits: a backing file newer than our last
        // read or write means someone else wrote it. Refuse before touching any
        // state; in-memory edits keep working and can be persisted after a
        // reload.
        // TODO double check this... seems like we can get into some weird states
        if self.disk_changed() {
            return Err(self.stale_message());
        }
        let edited_span = self.programs[program_index].span.clone();
        let mut edited_text = self.programs[program_index].text().to_string();
        // Remove any semicolons since these aren't valid within an expression and
        // can defeat parsing error recovery.
        edited_text.retain(|c| c != ';');
        // A padding slot has no binding to splice into (its binding index
        // points one past the end — see `mark_padding`). Span emptiness would
        // misclassify an existing binding whose expression is empty (e.g. `_ =
        // ;` loaded from a file).
        let binding_index = self.programs[program_index].binding_index;
        let is_new = binding_index >= self.bindings.len();
        let is_deletion = !is_new && edited_text.trim().is_empty();

        // Collect every edit we want to apply to `self.source` in one pass: the
        // active program's expression splice (or a fresh-binding insert or
        // whole-binding deletion) plus, for every program, annotation edits
        // that bring level/slider state on disk in line with the runtime.
        // Applying in reverse-position order keeps each remaining edit's span
        // valid.
        let mut edits: Vec<(Range<usize>, String)> = Vec::new();
        let mut new_binding_edit: Option<(Range<usize>, String)> = None;
        if is_new {
            if edited_text.trim().is_empty() {
                // Padding slot still empty after edit — nothing to do.
                // TODO is this true? what about sliders/levels?
                return Ok(());
            }
            // Find the source-order neighbors of the new program in the
            // pre-splice bindings so we can insert the new binding between them
            // and adjust the following program's `skip_slots`.
            let positions = walk_ui_positions(&self.bindings, self.source.len());
            let prev_pos: Option<usize> = positions
                .iter()
                .rev()
                .find(|(pos, _, _)| *pos < program_index)
                .map(|(pos, _, _)| *pos);
            let next: Option<(usize, usize)> = positions
                .iter()
                .find(|(pos, _, _)| *pos > program_index)
                .map(|(pos, i, _)| (*pos, *i));

            // Build the new binding: `skip_slots` fills the gap since the
            // previous UI program (or the top of the file); `level_db` is
            // always emitted so a new binding always carries at least one
            // annotation (the "any annotation → UI program" invariant).
            let new_skip = match prev_pos {
                Some(p) => program_index - p - 1,
                None => program_index,
            };
            let mut annos: Vec<expr::Annotation> = Vec::new();
            if new_skip > 0 {
                annos.push(expr::Annotation::SkipSlots(new_skip as u32));
            }
            annos.push(expr::Annotation::Level(
                self.programs[program_index].level_db(),
            ));
            let anno_body = annos
                .iter()
                .map(|a| format!("{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            // The new binding follows the binding just before its successor
            // (or the last non-trivia binding), sitting with the code rather
            // than the comments, with a blank line on each side.
            let pred_index = match next {
                Some((_, next_binding_index)) => next_binding_index.checked_sub(1),
                None => self
                    .bindings
                    .iter()
                    .rposition(|b| !matches!(b.binding, expr::Binding::Empty)),
            };
            let anchor = insertion_point(pred_index.map(|i| &self.bindings[i]), &self.source);
            let prefix = match anchor.checked_sub(1).map(|i| self.source.as_bytes()[i]) {
                None => "",
                Some(b'\n') => "\n",
                Some(_) => "\n\n",
            };
            let rest = &self.source[anchor..];
            let suffix = if rest.is_empty() || rest.starts_with('\n') {
                ""
            } else {
                "\n"
            };
            new_binding_edit = Some((
                anchor..anchor,
                format!("{prefix}#{{{anno_body}}}\n_ = {edited_text};\n{suffix}"),
            ));

            // If there's a UI program after the insertion point, its
            // `skip_slots` needs to compensate for the new program's slot
            // so that the follower's absolute grid position stays stable.
            if let Some((q, next_binding_index)) = next {
                let next_new_skip = (q - program_index - 1) as u32;
                let level = self.programs.get(q).map_or(0.0, |p| p.level_db());
                if let Some(edit) = skip_slots_edit(
                    &self.bindings[next_binding_index],
                    next_new_skip,
                    level,
                    &self.source,
                ) {
                    edits.push(edit);
                }
            }
        } else if is_deletion {
            // Clearing an existing program deletes its whole binding —
            // leading trivia, annotations, definition, and terminating
            // `;` — and grows the following UI program's `skip_slots` so
            // every later program keeps its slot.
            //
            // When the predecessor's line ends in a comment, that comment is
            // part of this binding's leading trivia but belongs to the
            // predecessor, so the deletion starts at its line's newline.
            let binding_span = self.bindings[binding_index]
                .span
                .clone()
                .expect("parsed binding has span")
                .range;
            let start = if binding_index > 0 {
                trivia_line_end(binding_span.start, &self.source).unwrap_or(binding_span.start)
            } else {
                binding_span.start
            };
            edits.push((start..binding_span.end, String::new()));

            let positions = walk_ui_positions(&self.bindings, self.source.len());
            let prev_pos: Option<usize> = positions
                .iter()
                .rev()
                .find(|(pos, _, _)| *pos < program_index)
                .map(|(pos, _, _)| *pos);
            let next: Option<(usize, usize)> = positions
                .iter()
                .find(|(pos, _, _)| *pos > program_index)
                .map(|(pos, i, _)| (*pos, *i));
            if let Some((q, next_binding_index)) = next {
                let next_new_skip = match prev_pos {
                    Some(p) => q - p - 1,
                    None => q,
                } as u32;
                let level = self.programs.get(q).map_or(0.0, |p| p.level_db());
                if let Some(edit) = skip_slots_edit(
                    &self.bindings[next_binding_index],
                    next_new_skip,
                    level,
                    &self.source,
                ) {
                    edits.push(edit);
                }
            }
        } else {
            edits.push((edited_span, edited_text));
        }

        // Walk every program (active and not) and append annotation edits
        // for level_db / sliders that have diverged from the parsed source.
        // We use the *pre-splice* `self.bindings` so all spans line up with
        // `self.source` before any of these edits land.
        for (i, program) in self.programs.iter().enumerate() {
            // The deleted program's binding is being removed wholesale;
            // annotation edits inside that span would corrupt the deletion.
            if is_deletion && i == program_index {
                continue;
            }
            if let Some(binding) = self.bindings.get(program.binding_index) {
                edits.extend(annotation_edits(
                    program,
                    i,
                    binding,
                    &self.source,
                    &self.sliders,
                ));
            }
        }

        // Pushed last so that, among edits at the same offset (an annotation
        // line inserted above the successor), it is applied last by the stable
        // sort and so lands first in the text.
        edits.extend(new_binding_edit);

        let mut new_source = self.source.clone();
        edits.sort_by_key(|(span, _)| cmp::Reverse(span.start));
        for (span, replacement) in edits {
            new_source.replace_range(span, &replacement);
        }

        // It's ok to drop the errors here: if parse_module succeeded then we know
        // that the resulting bindings span the entire input (even if there are
        // errors).
        let (new_bindings, _errors) = parser::parse_module::<MarkId, _>(&new_source, Source::File)
            .map_err(|errs| format!("Warning: source re-parse failed: {:?}", errs))?;

        // Realign each program to its position in the re-parsed source by
        // running the same position walk used at load time.
        let slot_lookup: Vec<Option<(usize, Range<usize>)>> = {
            let mut lookup: Vec<Option<(usize, Range<usize>)>> = vec![None; self.programs.len()];
            for (pos, i, span) in walk_ui_positions(&new_bindings, new_source.len()) {
                if pos < lookup.len() {
                    lookup[pos] = Some((i, span));
                }
            }
            lookup
        };

        for (i, program) in self.programs.iter_mut().enumerate() {
            match &slot_lookup[i] {
                Some((binding_index, span)) => {
                    program.realign(*binding_index, span.clone());
                    // Only the spliced program's text is re-read, picking up
                    // the stripped semicolons. Any other program's text may
                    // hold unsaved edits the file doesn't have.
                    if i == program_index {
                        program.resync_text(&new_source);
                    }
                }
                None => {
                    // No binding for this slot — keep it as a padding slot.
                    program.mark_padding(new_bindings.len());
                }
            }
        }
        if is_deletion {
            // The deleted slot starts over as a fresh padding program; don't
            // let level/slider/color state from the removed binding linger on
            // it.
            // It keeps its generation, so a result for the deleting text
            // still matches it.
            let generation = self.programs[program_index].generation;
            self.programs[program_index] = Program::from_string("", new_bindings.len());
            self.programs[program_index].generation = generation;
            self.sliders
                .retain(|(provider, _), _| *provider != program_index);
        }

        // Skip the file write when no path is set — tests use an empty
        // `PathBuf` so they can exercise the splice without touching disk.
        if !self.input_path.as_os_str().is_empty() {
            if let Err(e) = fs::write(&self.input_path, &new_source) {
                return Err(format!(
                    "Warning: failed to write {}: {}",
                    self.input_path.display(),
                    e
                ));
            }
            self.synced_mtime = file_mtime(&self.input_path);
        }

        self.source = new_source;
        self.bindings = new_bindings;
        Ok(())
    }
}

/// A slider's user-facing state, formatted for the status line and slider
/// overlays.
pub struct SliderDisplay {
    pub label: String,
    pub axis: String, // "X" or "Y" or an index
    pub normalized_value: f32,
    pub actual_value: f32,
}

impl fmt::Display for SliderDisplay {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}({}) = {}",
            self.label,
            self.axis,
            format_sig_digits(self.actual_value, 3),
        )
    }
}

#[cfg(test)]
mod tests {
    //! Exercises the open-file → edit-program → splice → reparse loop
    //! without touching disk: `splice` skips the file write when
    //! `input_path` is empty.
    use super::*;
    use std::env::temp_dir;
    use std::path::PathBuf;

    /// Builds a `ProgramSet` from inline source, with `input_path` empty
    /// so `splice` is a no-op against disk.
    fn state_from(source: &str) -> ProgramSet {
        ProgramSet::from_source(source.to_string(), PathBuf::new())
            .expect("test source should parse")
            .0
    }

    /// Returns the slider `label` the slot at `provider` provides, reached
    /// with a linear spec of `default` over `min` to `max`.
    fn reached(
        provider: usize,
        label: &str,
        default: f32,
        min: f32,
        max: f32,
    ) -> slider_analysis::Reached {
        slider_analysis::Reached {
            provider,
            label: label.to_string(),
            spec: slider::SliderSpec::new(default, min, max, slider::Curve::Linear).unwrap(),
        }
    }

    #[test]
    fn a_listed_slider_moves_by_position_once_reached() {
        let mut set = state_from("#{sliders=[\"vol\"]}\ntone = saw(220);");
        // Listed but not yet reached: there is no range to move along.
        assert!(set.set_slider_normalized(0, 0, 0.5).is_none());
        set.reach(&[reached(0, "vol", 5.0, 0.0, 10.0)]);
        assert_eq!(set.slider_normalized(0, 0), Some(0.5));
        // No slider at index 1.
        assert!(set.set_slider_normalized(0, 1, 0.5).is_none());
        // Positions past the end clamp to it.
        let change = set.set_slider_normalized(0, 0, 1.5).unwrap();
        assert_eq!((change.label.as_str(), change.value), ("vol", 10.0));
        assert_eq!(set.slider(0, "vol").unwrap().value, 10.0);
    }

    #[test]
    fn a_reached_slider_starts_at_its_saved_value_or_its_default() {
        let mut set = state_from(
            "#{sliders=[\"vol:0.25\", \"pan\", \"far:7\"]}\ntone = saw(220);\n\
             #{level_db=0}\n_ = tone;",
        );
        set.reach(&[
            reached(0, "vol", 0.5, 0.0, 1.0),
            reached(0, "pan", 0.5, 0.0, 1.0),
            reached(0, "far", 0.5, 0.0, 1.0),
            // Slot 1 saves nothing for its own `vol`.
            reached(1, "vol", 0.5, 0.0, 1.0),
        ]);
        let value = |provider, label| set.slider(provider, label).unwrap().value;
        assert_eq!(value(0, "vol"), 0.25);
        assert_eq!(value(0, "pan"), 0.5);
        // A saved value outside the range is clamped.
        assert_eq!(value(0, "far"), 1.0);
        assert_eq!(value(1, "vol"), 0.5);
    }

    #[test]
    fn a_slider_reached_again_keeps_its_position() {
        let mut set = state_from("#{sliders=[\"vol\"]}\ntone = saw(220);");
        set.reach(&[reached(0, "vol", 0.5, 0.0, 1.0)]);
        set.set_slider_normalized(0, 0, 0.75);
        set.reach(&[reached(0, "vol", 0.5, 0.0, 1.0)]);
        assert_eq!(set.slider(0, "vol").unwrap().value, 0.75);
        // A new default and range: three quarters of the way along it.
        set.reach(&[reached(0, "vol", 2.0, 0.0, 4.0)]);
        assert_eq!(set.slider(0, "vol").unwrap().value, 3.0);
    }

    #[test]
    fn a_slider_reached_for_the_first_time_starts_at_its_saved_value() {
        let mut set = state_from("#{sliders=[\"vol:0.25\"]}\ntone = saw(220);");
        set.reach(&[reached(0, "vol", 0.5, 0.0, 1.0)]);
        assert_eq!(set.slider(0, "vol").unwrap().value, 0.25);
    }

    /// Records an evaluation of the program at `index` that can reach
    /// `sliders`.
    fn record_reaching(set: &mut ProgramSet, index: usize, sliders: Vec<slider_analysis::Reached>) {
        let evaluated = Evaluated {
            plays: Plays::Waveform(waveform::Waveform::Const(1.0), Arc::default()),
            sliders,
        };
        let text = set.programs[index].text().into();
        set.record(index, text, Outcome::Evaluated(evaluated, Vec::new()))
            .expect("the evaluation is recorded");
    }

    #[test]
    fn encoders_show_the_listed_sliders_then_the_owned_ones_in_reach_order() {
        let mut set = state_from(
            "#{sliders=[\"b\", \"a\"]}\npad = saw(220);\n\
             #{sliders=[\"q\"]}\n_ = pad;",
        );
        record_reaching(
            &mut set,
            1,
            vec![
                reached(1, "z", 0.5, 0.0, 1.0),
                // Owned by the binding that lists it, so not slot 1's to show.
                reached(0, "a", 0.5, 0.0, 1.0),
                reached(1, "q", 0.5, 0.0, 1.0),
                reached(1, "y", 0.5, 0.0, 1.0),
            ],
        );
        assert_eq!(set.shown(1), ["q", "z", "y"]);
        // A listed slider keeps its place before anything reaches it.
        assert_eq!(set.shown(0), ["b", "a"]);
        assert!(set.shown_slider(0, 0).is_none());
        let change = set.set_slider_normalized(1, 1, 1.0).unwrap();
        assert_eq!((change.label.as_str(), change.value), ("z", 1.0));
        assert_eq!(set.slider(1, "y").unwrap().value, 0.5);
    }

    #[test]
    fn a_shown_slider_is_saved_in_its_place_once_moved_off_its_default() {
        let source = "#{level_db=0}\n_ = saw(220);";
        let mut set = state_from(source);
        record_reaching(
            &mut set,
            0,
            vec![
                reached(0, "cutoff", 0.5, 0.0, 1.0),
                reached(0, "q", 0.5, 0.0, 1.0),
                reached(0, "pan", 0.5, 0.0, 1.0),
            ],
        );
        set.splice(0).unwrap();
        assert_eq!(set.source, source);
        set.set_slider_normalized(0, 1, 0.75);
        set.splice(0).unwrap();
        // `cutoff` is written bare to keep its encoder ahead of `q`'s; `pan`,
        // unmoved and after it, isn't written.
        assert_eq!(
            set.source,
            "#{sliders=[\"cutoff\",\"q:0.75\"], level_db=0}\n_ = saw(220);"
        );
        assert_eq!(set.shown(0), ["cutoff", "q", "pan"]);
    }

    #[test]
    fn a_moved_shown_slider_is_appended_to_the_listed_ones() {
        let mut set = state_from("#{sliders=[\"a\"]}\n_ = saw(220);");
        record_reaching(
            &mut set,
            0,
            vec![
                reached(0, "b", 0.5, 0.0, 1.0),
                reached(0, "a", 0.5, 0.0, 1.0),
            ],
        );
        set.set_slider_normalized(0, 1, 0.25);
        set.splice(0).unwrap();
        assert_eq!(set.source, "#{sliders=[\"a\",\"b:0.25\"]}\n_ = saw(220);");
        assert_eq!(set.shown(0), ["a", "b"]);
    }

    #[test]
    fn a_clamped_saved_value_keeps_its_encoder_and_is_saved_clamped() {
        let mut set = state_from("#{sliders=[\"far:7\", \"gone\"]}\n_ = saw(220);");
        record_reaching(&mut set, 0, vec![reached(0, "far", 0.5, 0.0, 1.0)]);
        // The unreached entry keeps its encoder and its place in the file.
        assert_eq!(set.shown(0), ["far", "gone"]);
        set.splice(0).unwrap();
        assert_eq!(set.source, "#{sliders=[\"far:1\",\"gone\"]}\n_ = saw(220);");
    }

    #[test]
    fn splice_refuses_when_disk_changed() {
        let path = temp_dir().join(format!("tuun_splice_guard_{}.tuun", std::process::id()));
        fs::write(&path, "#{level_db=0}\ntone = 1 | fin(time - 1);\n").unwrap();
        let (mut set, _) =
            ProgramSet::from_source(fs::read_to_string(&path).unwrap(), path.clone())
                .expect("test source should parse");

        // An external edit lands after our read.
        let external = "#{level_db=0}\nexternal = 2 | fin(time - 1);\n";
        fs::write(&path, external).unwrap();
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(time::SystemTime::UNIX_EPOCH + time::Duration::from_secs(1))
            .unwrap();

        // An in-app edit then tries to persist: refused, file untouched.
        set.program_mut(0).unwrap().set_text("2".to_string());
        let error = set.splice(0).expect_err("stale disk should refuse");
        assert!(error.contains("changed on disk"), "got: {}", error);
        assert_eq!(fs::read_to_string(&path).unwrap(), external);

        // Reloading adopts the external content and clears the staleness.
        let source = set.read_backing_file().expect("the file reads");
        let (mut new_set, _) =
            ProgramSet::from_source(source, path.clone()).expect("the file parses");
        assert_eq!(new_set.display_name(0), "A:1 (external)");
        new_set.program_mut(0).unwrap().set_text("3".to_string());
        new_set.splice(0).expect("fresh set should splice");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn read_backing_file_rejects_empty_input_path() {
        let (set, _) = ProgramSet::from_source(
            "#{level_db=0}\n_ = 1 | fin(time - 1);\n".to_string(),
            PathBuf::new(),
        )
        .expect("test source should parse");
        assert!(set.read_backing_file().is_err());
    }

    #[test]
    fn read_backing_file_picks_up_external_edits_and_fails_when_missing() {
        let path = temp_dir().join(format!("tuun_read_backing_{}.tuun", std::process::id()));
        fs::write(&path, "#{level_db=0}\ntone = 1 | fin(time - 1);\n").unwrap();
        let (set, _) = ProgramSet::from_source(fs::read_to_string(&path).unwrap(), path.clone())
            .expect("test source should parse");

        let external = "#{level_db=0, keys}\nk = fn(note, vel) => (1, 1);\n";
        fs::write(&path, external).unwrap();
        assert_eq!(set.read_backing_file(), Ok(external.to_string()));

        fs::remove_file(&path).ok();
        let error = set
            .read_backing_file()
            .expect_err("missing file should fail");
        assert!(error.contains("Failed to read"), "got: {}", error);
    }

    #[test]
    fn keys_annotation_sets_program_kind() {
        let (set, _) = ProgramSet::from_source(
            "#{level_db=0, keys}\nk = fn(note, vel) => (1, 1);\n\
             #{level_db=0}\ntone = 1 | fin(time - 1);\n"
                .to_string(),
            PathBuf::new(),
        )
        .expect("test source should parse");
        assert_eq!(set.programs()[0].kind(), ProgramKind::Keys);
        assert_eq!(set.programs()[1].kind(), ProgramKind::Waveform);
    }

    #[test]
    fn evaluated_accessors_follow_the_cached_variant() {
        let mut program = Program::from_string("x", 0);
        let evaluated = |plays| {
            Some(Evaluated {
                plays,
                sliders: Vec::new(),
            })
        };
        program.evaluated = evaluated(Plays::Waveform(
            waveform::Waveform::Const(1.0),
            Arc::default(),
        ));
        assert!(program.waveform().is_some());
        assert!(program.is_playable());
        assert!(program.keys_instrument().is_none());
        assert!(program.sequence().is_none());

        program.evaluated = evaluated(Plays::Sequence(sequencer::Sequence {
            steps: vec![(1.0, 0..1)],
            step_waveform: waveform::Waveform::Const(1.0),
            step_marks: Arc::default(),
        }));
        assert!(program.waveform().is_none());
        assert!(program.is_playable());
        assert!(program.keys_instrument().is_none());
        assert!(program.sequence().is_some());

        program.evaluated = evaluated(Plays::KeysInstrument(KeysInstrument {
            function: expr::SourceExpr::float(1.0),
            listings: slider::Listings::new(|_| None),
        }));
        assert!(program.waveform().is_none());
        assert!(!program.is_playable());
        assert!(program.keys_instrument().is_some());
        assert!(program.sequence().is_none());

        // A text change keeps the result until the next one is recorded.
        program.set_text("y".to_string());
        assert!(program.keys_instrument().is_some());
    }

    #[test]
    fn undo_history_caps_at_max_units() {
        let mut program = Program::from_string("", 0);
        for i in 0..150 {
            program.record_edit(0);
            program.set_text(format!("t{}", i));
        }
        // Only the most recent 100 units survive; the oldest 50 were
        // dropped as the cap was exceeded.
        for _ in 0..100 {
            assert!(program.undo(0).is_some());
        }
        assert_eq!(program.text(), "t49");
        assert!(program.undo(0).is_none());
    }

    /// Returns the name each binding of `bindings` binds, for the `Definition`s
    /// among them.
    fn binding_names(bindings: &[expr::SourceBinding<MarkId, Source>]) -> Vec<String> {
        bindings
            .iter()
            .filter_map(|b| match &b.binding {
                expr::Binding::Definition(expr::Pattern::Identifier(name), _) => Some(name.clone()),
                _ => None,
            })
            .collect()
    }

    /// A `sliders` annotation lists labels; it binds no names.
    #[test]
    fn a_listed_label_is_not_a_name_in_scope() {
        let set = state_from("#{sliders=[\"mix\"]}\nm = mix;");
        let environment = Environment::new(8000, 90, PathBuf::new());
        let diagnostics = environment.check_program(&set.evaluation_input(0)).findings;
        assert_eq!(diagnostics.len(), 1, "got {:?}", diagnostics);
        assert_eq!(diagnostics[0].to_string(), "1:1: unbound variable 'mix'");
    }

    #[test]
    fn slider_providers_are_the_files_program_slots() {
        let source = "#{sliders=[\"q:0.5\"]}\nf = fn(x) => x;\n\
                      #{level_db=0}\n_ = f(1);\n\
                      #{level_db=0}\n_ = f(2);\n";
        let (set, _) = ProgramSet::from_source(source.to_string(), PathBuf::new()).unwrap();
        let providers = set.slider_providers();
        // A slot for each program, then the empty ones after them.
        assert_eq!(providers.slots.len(), set.programs().len());
        let [first, second, ..] = &providers.slots[..] else {
            panic!("expected a slot per program, got {:?}", providers.slots);
        };
        assert_eq!(&source[second.text.clone()], "f(1)");
        assert!(second.listed.is_empty());
        assert_eq!(second.listing, None);
        assert_eq!(first.name, set.display_name(0));
        assert_eq!(&source[first.text.clone()], "fn(x) => x");
        assert_eq!(
            first.listed,
            [expr::Slider {
                label: "q".to_string(),
                value: Some(0.5),
            }]
        );
        let listing = first.listing.clone().expect("slot 0 lists a slider");
        assert_eq!(listing.source, Source::File);
        assert!(source[listing.range].contains("q:0.5"));
    }

    #[test]
    fn evaluation_bindings_are_a_prefix_of_used_bindings() {
        let source = "\
pi = 3.14;
#{level_db=0}
_ = pulse(60);
#{sliders=[\"vol\"]}
tone = saw(220);
open more;
#{sliders=[\"pan\"]}
_ = tone;";
        let set = state_from(source);
        let (used, counts) = set.used_evaluation_bindings();
        assert_eq!(binding_names(&used), vec!["pi", "tone"]);
        for index in 0..set.programs().len() {
            let expected = used[..counts[index]].to_vec();
            let actual = set.evaluation_bindings(index);
            assert_eq!(
                format!("{:?}", actual),
                format!("{:?}", expected),
                "program {}",
                index
            );
        }
        // The last program reads `pi`, `tone` and `open more`.
        assert_eq!(counts[2], 3);
        assert_eq!(
            binding_names(&set.evaluation_bindings(2)),
            vec!["pi", "tone"]
        );
    }

    #[test]
    fn evaluation_bindings_filters_anonymous() {
        let source = "\
pi = 3.14;
#{level_db=0}
_ = pulse(60);
#{level_db=0}
tone = saw(220);";
        let set = state_from(source);
        let names = binding_names(&set.evaluation_bindings(1));
        // `pi` precedes the program and survives; the anonymous `_` slot-1
        // definition is filtered.
        assert_eq!(names, vec!["pi".to_string()]);
    }

    #[test]
    fn editing_existing_program_splices_into_its_span() {
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);

        // Two consecutive UI programs → slots 0 and 1. Sanity-check the initial
        // extraction.
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        assert_eq!(state.programs()[1].text(), "saw(220)");

        // Simulate the Edit-mode flow: select slot 2, change the text.
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(440)".to_string());

        state.splice(1).expect("splice should succeed");

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
synth = saw(440);"
        );
        // Slot 1 untouched, slot 2's program now reflects the new text and
        // span.
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        assert_eq!(state.programs()[1].text(), "saw(440)");
    }

    #[test]
    fn splice_keeps_unsaved_text_of_other_programs() {
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);

        // Slot 2 holds unsaved edits while slot 1 is spliced with a longer
        // text, shifting slot 2's span.
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(440) * 0.5".to_string());
        state
            .program_mut(0)
            .unwrap()
            .set_text("pulse(120)".to_string());
        state.splice(0).expect("splice should succeed");

        assert_eq!(state.programs()[1].text(), "saw(440) * 0.5");
        assert!(state.source.ends_with("synth = saw(220);"));

        // Splicing slot 2 later lands its edits in its shifted span.
        state.splice(1).expect("splice should succeed");
        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(120);
#{level_db=0}
synth = saw(440) * 0.5;"
        );
    }

    /// The generation counts text replacements: the splice's re-read of the
    /// text it saved is not one, and a deleted slot keeps its count.
    #[test]
    fn generation_counts_text_replacements() {
        let mut state = state_from("#{level_db=0}\nkick = pulse(120);\n#{level_db=0}\nhat = 1;");
        assert_eq!(state.programs()[0].generation(), 0);

        // The trailing semicolon is stripped by the splice's re-read.
        state
            .program_mut(0)
            .unwrap()
            .set_text("pulse(60);".to_string());
        assert_eq!(state.programs()[0].generation(), 1);
        state.splice(0).expect("splice should succeed");
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        assert_eq!(state.programs()[0].generation(), 1);

        state.program_mut(1).unwrap().set_text(String::new());
        state.splice(1).expect("splice should succeed");
        assert!(state.programs()[1].is_empty());
        assert_eq!(state.programs()[1].generation(), 1);
    }

    #[test]
    fn editing_program_does_not_disturb_library_bindings() {
        // A `Definition` with no annotations is a library binding, not a UI
        // program. Editing a UI program must leave library bindings (and `Open`
        // directives) in place.
        let source = "\
open util.synths;
pi = 3.14159;
#{level_db=0}
tone = saw(440);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "saw(440)");

        state
            .program_mut(0)
            .unwrap()
            .set_text("saw(220)".to_string());
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
open util.synths;
pi = 3.14159;
#{level_db=0}
tone = saw(220);"
        );
    }

    #[test]
    fn editing_program_does_not_disturb_use_bindings() {
        // A `use` binding must ride through splices verbatim, like `open`.
        let source = "\
use util.synths;
#{level_db=0}
tone = synths.saw(440);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "synths.saw(440)");

        state
            .program_mut(0)
            .unwrap()
            .set_text("synths.saw(220)".to_string());
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
use util.synths;
#{level_db=0}
tone = synths.saw(220);"
        );
    }

    #[test]
    fn new_program_is_appended_at_end_when_no_next_ui_program() {
        // Editing a previously-empty padding slot (span 0..0) with no UI
        // program after it should append a fresh `#{skip_slots=…,
        // level_db=…}\n_ = <text>` Definition at the end of source rather than
        // splicing at offset 0.
        let source = "\
#{level_db=0}
kick = pulse(60);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        // Slot 5 (index 4) is padding initially.
        assert_eq!(state.programs()[4].text(), "");
        assert_eq!(state.programs()[4].span.clone(), 0..0);

        // Edit slot 5.
        state
            .program_mut(4)
            .unwrap()
            .set_text("saw(440)".to_string());

        state.splice(4).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{skip_slots=3, level_db=0}
_ = saw(440);
"
        );
        // The new binding is now part of the bindings vec, and the
        // slot 5 program has a real span/text.
        assert_eq!(state.programs()[4].text(), "saw(440)");
        assert!(state.programs()[4].span.clone().start < state.programs()[4].span.clone().end);
    }

    #[test]
    fn editing_padding_slot_with_empty_text_is_a_no_op() {
        // The user enters Edit on an empty slot, types nothing, hits
        // Return. We must not append a phantom `_ = ` definition.
        let source = "\
#{level_db=0}
kick = pulse(60);";
        let mut state = state_from(source);
        let before = state.source.clone();
        // text is already empty; just splice.
        state.splice(4).unwrap();
        assert_eq!(state.source, before);
    }

    #[test]
    fn changed_level_db_replaces_existing_annotation_in_place() {
        // The existing `level_db=-6` annotation has its own span; the
        // splice should land exactly on those bytes, leaving the rest
        // of the binding (including its leading comment) untouched.
        let source = "\
#{level_db=-6}
// keep this comment
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs()[0].level_db() - -6.0).abs() < 1e-6);

        state.program_mut(0).unwrap().set_level_db(-3.5);
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=-3.5}
// keep this comment
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_level_db_inserts_fresh_annotation_when_none_exists() {
        // No `level_db=…` in source but the binding is still a UI program (has
        // `color=…`). Prepend `level_db=…` to the existing annotation set,
        // leaving the pre-binding comment alone.
        let source = "\
// header comment
#{color=rgb(255,0,0)}
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs()[0].level_db() - 0.0).abs() < 1e-6);

        state.program_mut(0).unwrap().set_level_db(-3.0);
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
// header comment
#{level_db=-3, color=rgb(255,0,0)}
kick = pulse(60);"
        );
    }

    #[test]
    fn gaining_keys_prepends_the_annotation() {
        let mut state = state_from("#{level_db=-6}\nk = 1;");
        state.program_mut(0).unwrap().set_kind(ProgramKind::Keys);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{keys, level_db=-6}\nk = 1;");
    }

    #[test]
    fn losing_keys_removes_the_annotation() {
        let mut state = state_from("#{level_db=-6, keys}\nk = 1;");
        state
            .program_mut(0)
            .unwrap()
            .set_kind(ProgramKind::Waveform);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{level_db=-6}\nk = 1;");
    }

    #[test]
    fn losing_a_lone_keys_annotation_swaps_in_level_db() {
        // A binding with no annotation would no longer be a program.
        let mut state = state_from("#{keys}\nk = 1;");
        let program = state.program_mut(0).unwrap();
        program.set_kind(ProgramKind::Waveform);
        program.set_level_db(-2.0);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{level_db=-2}\nk = 1;");
        assert_eq!(state.display_name(0), "A:1 (k)");
    }

    #[test]
    fn gaining_a_loop_prepends_the_annotation() {
        let mut state = state_from("#{level_db=-6}\nk = 1;");
        state.program_mut(0).unwrap().set_loop_after_beats(Some(4));
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{loop_after_beats=4, level_db=-6}\nk = 1;");
        assert_eq!(state.programs()[0].loop_after_beats(), Some(4));
    }

    #[test]
    fn gaining_a_loop_and_a_level_writes_both() {
        let mut state = state_from("#{level_db=-6}\nk = 1;");
        let program = state.program_mut(0).unwrap();
        program.set_loop_after_beats(Some(4));
        program.set_level_db(-3.0);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{loop_after_beats=4, level_db=-3}\nk = 1;");
    }

    #[test]
    fn changing_a_loop_rewrites_the_annotation() {
        let mut state = state_from("#{level_db=-6, loop_after_beats=4}\nk = 1;");
        state.program_mut(0).unwrap().set_loop_after_beats(Some(12));
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{level_db=-6, loop_after_beats=12}\nk = 1;");
    }

    #[test]
    fn losing_a_loop_removes_the_annotation() {
        let mut state = state_from("#{level_db=-6, loop_after_beats=4}\nk = 1;");
        state.program_mut(0).unwrap().set_loop_after_beats(None);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{level_db=-6}\nk = 1;");
    }

    #[test]
    fn losing_a_lone_loop_annotation_swaps_in_level_db() {
        let mut state = state_from("#{loop_after_beats=4}\nk = 1;");
        let program = state.program_mut(0).unwrap();
        program.set_loop_after_beats(None);
        program.set_level_db(-2.0);
        state.splice(0).unwrap();
        assert_eq!(state.source, "#{level_db=-2}\nk = 1;");
        assert_eq!(state.programs()[0].loop_after_beats(), None);
    }

    #[test]
    fn write_skip_slots_returns_the_moved_source() {
        let source =
            "#{level_db=0}\na = 1;\n#{skip_slots=2, level_db=0}\nb = 2;\n#{level_db=0}\nc = 3;";
        let mut state = state_from(source);
        assert_eq!(state.skip_slots(3), Some(2));
        assert_eq!(state.skip_slots(1), None);
        assert_eq!(state.last_slot(), Some(4));

        let moved = state.write_skip_slots(3, 3).unwrap();
        assert_eq!(
            moved,
            "#{level_db=0}\na = 1;\n#{skip_slots=3, level_db=0}\nb = 2;\n#{level_db=0}\nc = 3;"
        );
        // The set itself is unchanged until the new source is loaded.
        assert_eq!(state.source, source);
        let (moved_set, _) = ProgramSet::from_source(moved, PathBuf::new()).unwrap();
        assert_eq!(moved_set.display_name(4), "A:5 (b)");
        assert_eq!(moved_set.display_name(5), "A:6 (c)");

        let moved = state.write_skip_slots(3, 0).unwrap();
        assert_eq!(
            moved,
            "#{level_db=0}\na = 1;\n#{level_db=0}\nb = 2;\n#{level_db=0}\nc = 3;"
        );
        assert!(state.write_skip_slots(1, 1).is_err());
    }

    #[test]
    fn a_moved_slider_saves_its_value_in_the_annotation() {
        // Only the `sliders=…` annotation's bytes change; the binding body
        // stays verbatim.
        let source = "\
#{sliders=[\"vol\", \"pan:0.5\", \"far\"]}
tone = saw(220);";
        let mut state = state_from(source);
        state.reach(&[
            reached(0, "vol", 0.5, 0.0, 1.0),
            reached(0, "pan", 0.25, 0.0, 1.0),
        ]);
        // Unmoved, nothing is written, though `pan` differs from its default.
        state.splice(0).unwrap();
        assert_eq!(state.source, source);
        state.set_slider_normalized(0, 0, 0.75);
        state.set_slider_normalized(0, 1, 0.25);
        state.splice(0).unwrap();
        // An unreached entry is kept as it is.
        assert_eq!(
            state.source,
            "\
#{sliders=[\"vol:0.75\",\"pan:0.25\",\"far\"]}
tone = saw(220);"
        );
    }

    #[test]
    fn no_divergence_means_no_annotation_edits() {
        // Splicing without any runtime divergence should round-trip
        // the source exactly — even though splice_program walks every
        // program looking for level/slider drift.
        let source = "\
#{level_db=-6, sliders=[\"vol:0.25\"]}
kick = pulse(60);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        state.reach(&[reached(0, "vol", 0.5, 0.0, 1.0)]);
        // No runtime mutations — same edit-mode-exit pattern but with
        // the program's text unchanged from what parse_module produced.
        state.splice(0).unwrap();
        assert_eq!(state.source, source);
    }

    #[test]
    fn non_active_program_divergence_persists_on_any_save() {
        // The user adjusts level on program 2 via encoder, then edits
        // program 1's text and hits Return. Both writes need to land.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);

        // Program 2: level changed (encoder).
        state.program_mut(1).unwrap().set_level_db(-9.0);
        // Program 1: text edited (active save target).
        state
            .program_mut(0)
            .unwrap()
            .set_text("pulse(80)".to_string());

        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(80);
#{level_db=-9}
synth = saw(220);"
        );
    }

    #[test]
    fn new_program_at_end_uses_runtime_level_db_when_set() {
        // A brand-new program with a non-default runtime level should emit
        // `level_db` alongside `skip_slots` in its inserted `#{…}` header so
        // the file picks it up on the next parse.
        let source = "\
#{level_db=0}
kick = pulse(60);";
        let mut state = state_from(source);
        state
            .program_mut(4)
            .unwrap()
            .set_text("saw(440)".to_string());
        state.program_mut(4).unwrap().set_level_db(-2.5);

        state.splice(4).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{skip_slots=3, level_db=-2.5}
_ = saw(440);
"
        );
    }

    // ------------------------------------------------------------
    // Position walker: `from_source` semantics.
    // ------------------------------------------------------------

    #[test]
    fn library_bindings_without_annotations_do_not_take_slots() {
        // Two annotated bindings sandwich a bare library binding; the library
        // binding neither occupies a UI slot nor advances the position counter.
        let source = "\
#{level_db=0}
kick = pulse(60);
pi = 3.14;
#{level_db=0}
synth = saw(220);";
        let set = state_from(source);
        assert_eq!(set.programs()[0].text(), "pulse(60)");
        assert_eq!(set.programs()[1].text(), "saw(220)");
        // Slot 3 (index 2) is still padding.
        assert!(set.programs()[2].is_empty());
    }

    #[test]
    fn level_db_zero_alone_still_makes_a_ui_program() {
        // `level_db=0` is a valid annotation on its own; the binding shows up
        // at slot 0.
        let source = "\
#{level_db=0}
tone = saw(220);";
        let set = state_from(source);
        assert_eq!(set.programs()[0].text(), "saw(220)");
        assert!(set.programs()[1].is_empty());
    }

    #[test]
    fn loop_after_beats_is_read_from_the_annotation() {
        let source = "\
#{loop_after_beats=3}
kick = saw(60);
#{level_db=0}
tone = saw(220);";
        let set = state_from(source);
        assert_eq!(set.programs()[0].loop_after_beats(), Some(3));
        assert_eq!(set.programs()[1].loop_after_beats(), None);
    }

    #[test]
    fn selection_message_shows_address_name_and_setting() {
        let source = "\
#{loop_after_beats=4}
kick = saw(60);
#{keys}
lead = 1;
#{level_db=0}
tone = saw(220);
#{level_db=0}
_ = saw(110);
#{skip_slots=1, level_db=0}
last = saw(55);";
        let set = state_from(source);
        assert_eq!(set.selection_message(0), "A:1 kick, loops every 4 beats");
        assert_eq!(set.selection_message(1), "A:2 lead, plays keys");
        assert_eq!(set.selection_message(2), "A:3 tone");
        assert_eq!(set.selection_message(3), "A:4");
        assert!(set.programs()[4].is_empty());
        assert_eq!(set.selection_message(4), "");
    }

    #[test]
    fn skip_slots_leaves_a_gap_before_the_program() {
        // A single binding with `skip_slots=3` lands at slot 3; slots 0..2 stay
        // as padding.
        let source = "\
#{skip_slots=3, level_db=0}
tone = saw(220);";
        let set = state_from(source);
        for i in 0..3 {
            assert!(set.programs()[i].is_empty(), "slot {} should be padding", i);
        }
        assert_eq!(set.programs()[3].text(), "saw(220)");
    }

    #[test]
    fn consecutive_annotated_bindings_take_consecutive_slots() {
        // No `skip_slots` between them — each binding claims the next slot
        // after the previous one.
        let source = "\
#{level_db=0}
a = saw(110);
#{level_db=0}
b = saw(220);
#{level_db=0}
c = saw(440);";
        let set = state_from(source);
        assert_eq!(set.programs()[0].text(), "saw(110)");
        assert_eq!(set.programs()[1].text(), "saw(220)");
        assert_eq!(set.programs()[2].text(), "saw(440)");
        assert!(set.programs()[3].is_empty());
    }

    #[test]
    fn overflow_positions_are_dropped_not_panicked() {
        // A `skip_slots` value pushes the program past slot 63 (the last UI
        // slot). It should log and continue rather than panic; earlier programs
        // still get placed.
        let source = "\
#{level_db=0}
a = saw(110);
#{skip_slots=100, level_db=0}
b = saw(220);";
        let set = state_from(source);
        assert_eq!(set.programs()[0].text(), "saw(110)");
        // All remaining slots stay as padding.
        for i in 1..set.programs().len() {
            assert!(set.programs()[i].is_empty(), "slot {} should be padding", i);
        }
    }

    // ------------------------------------------------------------
    // Splice: inserting a new program between existing neighbors.
    // ------------------------------------------------------------

    #[test]
    fn new_program_inserts_between_neighbors_and_adjusts_next_skip_slots() {
        // prev at slot 0, next at slot 4 (`skip_slots=3`). Insert at slot 2.
        // Result: prev unchanged, new binding at slot 2 with `skip_slots=1`,
        // next binding's `skip_slots` becomes 1.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{skip_slots=3, level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        assert_eq!(state.programs()[4].text(), "saw(220)");

        state
            .program_mut(2)
            .unwrap()
            .set_text("saw(330)".to_string());
        state.splice(2).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{skip_slots=1, level_db=0}
_ = saw(330);

#{skip_slots=1, level_db=0}
synth = saw(220);"
        );
        // Slot 4 stayed at slot 4; slot 2 is now the new program.
        assert_eq!(state.programs()[2].text(), "saw(330)");
        assert_eq!(state.programs()[4].text(), "saw(220)");
    }

    #[test]
    fn insert_immediately_before_next_removes_its_skip_slots() {
        // prev at slot 0, next at slot 2 (`skip_slots=1`). Insert at slot 1.
        // next's gap collapses, so its `skip_slots` annotation is removed
        // entirely (never written as `skip_slots=0`).
        let source = "\
#{level_db=0}
kick = pulse(60);
#{skip_slots=1, level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(330)".to_string());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{level_db=0}
_ = saw(330);

#{level_db=0}
synth = saw(220);"
        );
        assert_eq!(state.programs()[2].text(), "saw(220)");
    }

    #[test]
    fn collapsing_a_trailing_skip_slots_consumes_the_preceding_comma() {
        // `skip_slots` is the last annotation in its set; removal has to take
        // the preceding comma with it.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0, skip_slots=1}
synth = saw(220);";
        let mut state = state_from(source);
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(330)".to_string());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{level_db=0}
_ = saw(330);

#{level_db=0}
synth = saw(220);"
        );
    }

    #[test]
    fn collapsing_the_only_annotation_swaps_in_level_db() {
        // `skip_slots` is the binding's only annotation. Deleting it would
        // demote the binding to library code, so swap in `level_db=…` instead
        // to keep it a UI program.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{skip_slots=1}
synth = saw(220);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[2].text(), "saw(220)");
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(330)".to_string());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);

#{level_db=0}
_ = saw(330);

#{level_db=0}
synth = saw(220);"
        );
        assert_eq!(state.programs()[2].text(), "saw(220)");
    }

    // ------------------------------------------------------------
    // Splice: deleting an existing program by clearing its text.
    // ------------------------------------------------------------

    #[test]
    fn deleting_a_program_removes_its_binding_and_keeps_next_position() {
        // Clearing the middle program removes its binding (annotations
        // included) and gives the following program a `skip_slots` so it stays
        // at slot 2.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
mid = saw(100);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        state.program_mut(1).unwrap().set_text(String::new());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);
#{skip_slots=1, level_db=0}
synth = saw(220);"
        );
        assert!(state.programs()[1].is_empty());
        assert_eq!(state.programs()[2].text(), "saw(220)");
    }

    #[test]
    fn deleting_a_program_grows_next_existing_skip_slots() {
        // kick at 0, mid at 1, synth at 4 (`skip_slots=2`). Deleting mid (with
        // whitespace-only text) grows synth's skip to 3.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
mid = saw(100);
#{skip_slots=2, level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        state.program_mut(1).unwrap().set_text("  ".to_string());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);
#{skip_slots=3, level_db=0}
synth = saw(220);"
        );
        assert!(state.programs()[1].is_empty());
        assert_eq!(state.programs()[4].text(), "saw(220)");
    }

    #[test]
    fn deleting_the_last_program_just_removes_its_binding() {
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        state.program_mut(1).unwrap().set_text(String::new());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);"
        );
        assert!(state.programs()[1].is_empty());
        assert_eq!(state.programs()[0].text(), "pulse(60)");
    }

    #[test]
    fn deleting_a_program_resets_its_slot_state() {
        // The deleted slot must not keep the removed binding's level or sliders
        // — a program later created there starts fresh.
        let source = "\
#{level_db=-6, sliders=[\"vol\"]}
kick = pulse(60);";
        let mut state = state_from(source);
        state.reach(&[reached(0, "vol", 0.5, 0.0, 1.0)]);
        state.program_mut(0).unwrap().set_text(String::new());
        state.splice(0).unwrap();

        assert_eq!(state.source, "");
        assert!(state.programs()[0].is_empty());
        assert!((state.programs()[0].level_db() - 0.0).abs() < 1e-6);
        assert!(state.listed(0).is_empty());
        assert_eq!(state.slider(0, "vol"), None);
    }

    #[test]
    fn empty_expression_binding_can_be_edited_in_place() {
        // A `_ = ;` binding (e.g. left over in a file) has a real binding with
        // an empty expression span. Typing into it must splice into that
        // binding, not append a duplicate.
        let source = "\
#{level_db=0}
_ = ;";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "");
        state
            .program_mut(0)
            .unwrap()
            .set_text("saw(440)".to_string());
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
_ = saw(440);"
        );
    }

    #[test]
    fn empty_expression_binding_can_be_deleted() {
        // Clearing a `_ = ;` binding (its text is already empty) removes it
        // from source entirely.
        let source = "\
#{level_db=0}
kick = pulse(60);
#{level_db=0}
_ = ;";
        let mut state = state_from(source);
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=0}
kick = pulse(60);"
        );
    }

    #[test]
    fn insert_before_first_ui_program_writes_skip_slots_on_new_binding() {
        // No prev; next at slot 3 (`skip_slots=3`). Insert at slot 1. New
        // binding gets `skip_slots=1`; next's `skip_slots` becomes 1 (3 - 1 -
        // 1).
        let source = "\
#{skip_slots=3, level_db=0}
synth = saw(220);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[3].text(), "saw(220)");
        state
            .program_mut(1)
            .unwrap()
            .set_text("saw(110)".to_string());
        state.splice(1).unwrap();

        assert_eq!(
            state.source,
            "\
#{skip_slots=1, level_db=0}
_ = saw(110);

#{skip_slots=1, level_db=0}
synth = saw(220);"
        );
        assert_eq!(state.programs()[1].text(), "saw(110)");
        assert_eq!(state.programs()[3].text(), "saw(220)");
    }

    // ------------------------------------------------------------
    // Splice: whitespace around an inserted binding.
    // ------------------------------------------------------------

    /// A file with a gap of two empty slots (A:3, A:4) between `kick` and
    /// `bass`, with comments above `kb` and `bass`.
    const GAPPED: &str = "\
// The keyboard.
#{level_db=-6,keys}
kb = saw(110);

#{level_db=0}
kick = pulse(60);

// Bass line, two slots down.
#{skip_slots=2, level_db=-3}
bass = saw(55);
";

    /// Records `text` into the padding slot at `index` with an undo point, as
    /// a recorded phrase would be, and returns the source after the insert.
    fn insert_with_undo_point(state: &mut ProgramSet, index: usize, text: &str) -> String {
        let program = state.program_mut(index).unwrap();
        program.record_edit(0);
        program.set_text(text.to_string());
        state.splice(index).unwrap();
        state.source.clone()
    }

    /// Undoes the last edit to the program at `index`, commits it, and returns
    /// the source.
    fn undo_and_commit(state: &mut ProgramSet, index: usize) -> String {
        state.program_mut(index).unwrap().undo(0).unwrap();
        state.splice(index).unwrap();
        state.source.clone()
    }

    #[test]
    fn inserted_binding_sits_with_the_code_and_undo_restores_the_file() {
        let mut state = state_from(GAPPED);
        state.program_mut(2).unwrap().set_level_db(-6.0);
        let inserted = insert_with_undo_point(&mut state, 2, "saw(220)");
        assert_eq!(
            inserted,
            "\
// The keyboard.
#{level_db=-6,keys}
kb = saw(110);

#{level_db=0}
kick = pulse(60);

#{level_db=-6}
_ = saw(220);

// Bass line, two slots down.
#{skip_slots=1, level_db=-3}
bass = saw(55);
"
        );
        assert_eq!(state.programs()[2].text(), "saw(220)");
        assert_eq!(state.programs()[4].text(), "saw(55)");
        assert_eq!(undo_and_commit(&mut state, 2), GAPPED);
    }

    #[test]
    fn undo_at_the_end_of_a_gap_restores_the_file() {
        // At the end of a gap the successor loses its `skip_slots`; undo puts
        // it back at the front of the successor's annotation set.
        let mut state = state_from(GAPPED);
        let inserted = insert_with_undo_point(&mut state, 3, "saw(220)");
        assert_eq!(
            inserted,
            "\
// The keyboard.
#{level_db=-6,keys}
kb = saw(110);

#{level_db=0}
kick = pulse(60);

#{skip_slots=1, level_db=0}
_ = saw(220);

// Bass line, two slots down.
#{level_db=-3}
bass = saw(55);
"
        );
        assert_eq!(undo_and_commit(&mut state, 3), GAPPED);
        assert_eq!(state.programs()[4].text(), "saw(55)");
    }

    #[test]
    fn inserted_binding_goes_before_a_trailing_comment() {
        let source = "\
#{level_db=0}
kick = pulse(60);

// TODO more programs here.
";
        let mut state = state_from(source);
        let inserted = insert_with_undo_point(&mut state, 1, "saw(220)");
        assert_eq!(
            inserted,
            "\
#{level_db=0}
kick = pulse(60);

#{level_db=0}
_ = saw(220);

// TODO more programs here.
"
        );
        assert_eq!(undo_and_commit(&mut state, 1), source);
    }

    #[test]
    fn inserted_binding_leaves_a_same_line_comment_on_its_line() {
        let source = "\
#{level_db=0}
kick = pulse(60); // four on the floor
#{skip_slots=1, level_db=0}
bass = saw(55);";
        let mut state = state_from(source);
        let inserted = insert_with_undo_point(&mut state, 1, "saw(220)");
        assert_eq!(
            inserted,
            "\
#{level_db=0}
kick = pulse(60); // four on the floor

#{level_db=0}
_ = saw(220);

#{level_db=0}
bass = saw(55);"
        );
        assert_eq!(state.programs()[2].text(), "saw(55)");
        // The inserted blank lines stay: the original had none to restore.
        assert_eq!(
            undo_and_commit(&mut state, 1),
            "\
#{level_db=0}
kick = pulse(60); // four on the floor

#{skip_slots=1, level_db=0}
bass = saw(55);"
        );
    }

    #[test]
    fn inserted_binding_into_a_file_without_bindings_starts_it() {
        let mut state = state_from("");
        assert_eq!(
            insert_with_undo_point(&mut state, 0, "saw(220)"),
            "#{level_db=0}\n_ = saw(220);\n"
        );

        let mut state = state_from("// Notes.\n");
        assert_eq!(
            insert_with_undo_point(&mut state, 0, "saw(220)"),
            "#{level_db=0}\n_ = saw(220);\n\n// Notes.\n"
        );
    }
}
