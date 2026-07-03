//! Programs and their state: text, sliders, level, and evaluation caches.
//!
//! `Program` exposes a narrow API — fields are private so the rest of the
//! UI (renderer, input handlers, reducer, effect runner) can only observe
//! programs through getters and mutate them through methods that preserve
//! the internal invariants.

use std::fmt;
use std::ops::Range;

use crate::parser;
use crate::renderer::{MarkId, format_sig_digits};
use crate::slider;
use crate::waveform;

pub const PROGRAMS_PER_BANK: usize = 8;
pub const NUM_PROGRAM_BANKS: usize = 8;

/// The sliders attached to a program: source-level configs plus the
/// current normalized position of each.
///
/// Invariant: `normalized_values` is parallel to `configs` and every value
/// lies in 0.0..=1.0.
#[derive(Debug, Clone, Default)]
pub struct ProgramSliders {
    configs: Vec<parser::Slider>,
    normalized_values: Vec<f32>,
}

impl ProgramSliders {
    /// Builds a `ProgramSliders` from a list of source-level slider configs.
    pub fn from_slider_configs(configs: Vec<parser::Slider>) -> Self {
        use parser::SliderFunction;
        let normalized_values = configs
            .iter()
            .map(|c| match &c.function {
                SliderFunction::Linear {
                    initial_value,
                    min,
                    max,
                } => ((initial_value - min) / (max - min)).clamp(0.0, 1.0),
                SliderFunction::UserDefined {
                    normalized_initial_value,
                    ..
                } => normalized_initial_value.clamp(0.0, 1.0),
            })
            .collect();
        ProgramSliders {
            configs,
            normalized_values,
        }
    }

    /// Returns the source-level slider configs.
    pub fn configs(&self) -> &[parser::Slider] {
        &self.configs
    }

    /// Returns the current normalized values, parallel to `configs`.
    pub fn normalized_values(&self) -> &[f32] {
        &self.normalized_values
    }

    // TODO this method only works for mouse-based sliders
    pub fn slider_display(&self) -> Vec<SliderDisplay> {
        self.configs
            .iter()
            .enumerate()
            .map(|(j, config)| {
                let norm = self.normalized_values[j];
                SliderDisplay {
                    label: config.label.clone(),
                    // TODO this is wrong for encoders
                    axis: if j == 0 {
                        "X".to_string()
                    } else {
                        "Y".to_string()
                    },
                    normalized_value: norm,
                    actual_value: slider::denormalize(&config.function, norm).unwrap_or(0.0),
                }
            })
            .collect()
    }

    /// Sets the slider at `index` to `normalized` (clamped to 0.0..=1.0)
    /// and returns the slider's label and new denormalized value, or `None`
    /// if there is no slider at `index`.
    fn set_normalized(&mut self, index: usize, normalized: f32) -> Option<SliderChange> {
        let config = self.configs.get(index)?;
        let normalized = normalized.clamp(0.0, 1.0);
        self.normalized_values[index] = normalized;
        Some(SliderChange {
            label: config.label.clone(),
            value: slider::denormalize(&config.function, normalized).unwrap_or(0.0),
        })
    }
}

/// A slider mutation's user-visible result: the slider's label and its new
/// denormalized value.
pub struct SliderChange {
    pub label: String,
    pub value: f32,
}

/// The result of evaluating a program's text.
pub enum Evaluation {
    /// The program evaluated to a playable waveform.
    Waveform(waveform::Waveform<MarkId>),
    /// The program evaluated to a function usable as a keys instrument.
    KeysInstrument(parser::SourceExpr<MarkId>),
    /// The program failed to parse or evaluate; holds the user-visible
    /// message.
    Invalid(String),
}

/// One program slot: its source text, sliders, level, color, and the
/// cached results of its last evaluation.
#[derive(Debug, Clone)]
pub struct Program {
    /// The program's source expression text. Kept in sync with the caches
    /// below: any change invalidates them (see `set_text` / `realign`).
    text: String,
    /// The span in the source file from which this program came. `0..0`
    /// marks a brand-new or padding slot.
    span: Range<usize>,
    /// Index of the originating binding in the file's `Vec<SourceBinding>`.
    binding_index: usize,
    sliders: ProgramSliders,
    color: Option<(u8, u8, u8)>,
    level_db: f32,
    /// Set if the current text evaluates to a valid waveform.
    cached_waveform: Option<waveform::Waveform<MarkId>>,
    /// Set if the current text evaluates to a valid keys instrument.
    cached_keys_instrument: Option<parser::SourceExpr<MarkId>>,
}

impl Program {
    /// Builds a program from a string without attempting to parse it.
    pub(crate) fn from_string(text: &str, binding_index: usize) -> Program {
        Program {
            text: text.to_string(),
            span: 0..0,
            binding_index,
            sliders: ProgramSliders::default(),
            color: None,
            level_db: 0.0,
            cached_waveform: None,
            cached_keys_instrument: None,
        }
    }

    /// Builds a `Program` from a `SourceBinding` plus its position in the
    /// file's bindings vec. Returns the 0-based program index (derived from
    /// the `#{slot=N}` annotation) alongside the constructed `Program`.
    ///
    /// Only `Definition`s carrying a `#{slot=N}` annotation become programs
    /// (the slot determines the UI position); other `Definition`s are treated
    /// as library bindings and are not shown.
    pub(crate) fn from_source_binding(
        sb: &parser::SourceBinding<MarkId>,
        binding_index: usize,
        source: &str,
    ) -> Option<(usize, Program)> {
        // TODO NextBank is ignored!
        let mut sliders = ProgramSliders::default();
        let mut color: Option<(u8, u8, u8)> = None;
        let mut level_db: f32 = 0.0;
        let mut slot: Option<u32> = None;
        for sa in &sb.annotations {
            match &sa.annotation {
                parser::Annotation::Sliders(configs) => {
                    sliders = ProgramSliders::from_slider_configs(configs.clone());
                }
                parser::Annotation::Color(r, g, b) => {
                    color = Some((*r, *g, *b));
                }
                parser::Annotation::Level(v) => {
                    level_db = *v;
                }
                parser::Annotation::Slot(n) => {
                    slot = Some(*n);
                }
            }
        }
        // No slot → not a UI program.
        let slot = slot?;
        if let parser::Binding::Definition(_, expr) = &sb.binding {
            if let Some(s) = &expr.span
                && s.end <= source.len()
            {
                let span = s.clone();
                let text = source[s.clone()].to_string();
                let program_index = (slot as usize).saturating_sub(1);
                Some((
                    program_index,
                    Program {
                        text,
                        span,
                        binding_index,
                        sliders,
                        color,
                        level_db,
                        cached_waveform: None,
                        cached_keys_instrument: None,
                    },
                ))
            } else {
                println!(
                    "Found source expression without span or invalid span: {:?}",
                    &sb
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

    /// Returns the program's output level in dB.
    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    /// Returns the program's sliders.
    pub fn sliders(&self) -> &ProgramSliders {
        &self.sliders
    }

    /// Returns the cached waveform if the current text evaluated to one.
    pub fn waveform(&self) -> Option<&waveform::Waveform<MarkId>> {
        self.cached_waveform.as_ref()
    }

    /// Returns the cached keys-instrument function if the current text
    /// evaluated to one.
    pub fn keys_instrument(&self) -> Option<&parser::SourceExpr<MarkId>> {
        self.cached_keys_instrument.as_ref()
    }

    /// Replaces the program's `text` and invalidates both cached
    /// evaluations.
    pub fn set_text(&mut self, text: String) {
        self.text = text;
        self.cached_waveform = None;
        self.cached_keys_instrument = None;
    }

    /// Sets the program's output level in dB.
    pub fn set_level_db(&mut self, level_db: f32) {
        self.level_db = level_db;
    }

    /// Sets the slider at `slider_index` to `normalized` (clamped to
    /// 0.0..=1.0) and returns its label and new denormalized value, or
    /// `None` if there is no slider at that index.
    pub fn set_slider_normalized(
        &mut self,
        slider_index: usize,
        normalized: f32,
    ) -> Option<SliderChange> {
        self.sliders.set_normalized(slider_index, normalized)
    }

    /// Records the result of evaluating the program's current text,
    /// replacing both caches. Returns the user-visible message as an error
    /// when the evaluation was invalid.
    ///
    /// An `Invalid` evaluation still clears both caches: even though
    /// editing already clears them, the failure may have come from a
    /// changed dependency rather than this program's own text.
    pub fn record_evaluation(&mut self, evaluation: Evaluation) -> Result<(), String> {
        match evaluation {
            Evaluation::Waveform(w) => {
                self.cached_waveform = Some(w);
                self.cached_keys_instrument = None;
                Ok(())
            }
            Evaluation::KeysInstrument(expr) => {
                self.cached_waveform = None;
                self.cached_keys_instrument = Some(expr);
                Ok(())
            }
            Evaluation::Invalid(message) => {
                self.cached_waveform = None;
                self.cached_keys_instrument = None;
                Err(message)
            }
        }
    }

    /// Realigns the program with its binding after the source file was
    /// re-parsed: updates the binding index and span, and re-slices `text`
    /// from `source`.
    ///
    /// Deliberately does NOT invalidate the evaluation caches — the text is
    /// unchanged semantically, only its location in the file moved. This is
    /// the one sanctioned way to rewrite `text` without clearing caches.
    pub(crate) fn realign(&mut self, binding_index: usize, span: Range<usize>, source: &str) {
        self.binding_index = binding_index;
        self.text = source[span.clone()].to_string();
        self.span = span;
    }

    /// Marks the program as a padding slot with no binding: the binding
    /// index points one past the end of the bindings and the span is empty.
    pub(crate) fn mark_padding(&mut self, binding_count: usize) {
        self.binding_index = binding_count;
        self.span = 0..0;
    }
}

/// The set of programs backed by a source file: the file's contents, its
/// parsed bindings, and one `Program` per UI slot.
///
/// Owns the program ↔ source alignment: programs are created from
/// `#{slot=N}` bindings and keep their `binding_index`/`span` pointing back
/// into `bindings`/`source`.
pub struct ProgramSet {
    programs: Vec<Program>,
    bindings: Vec<parser::SourceBinding<MarkId>>,
    source: String,
    input_path: std::path::PathBuf,
}

impl ProgramSet {
    /// Builds a `ProgramSet` from the contents of a source file, along with
    /// a warning message for recoverable parse errors (empty when the parse
    /// was clean). Fills every UI slot with an empty padding program, then
    /// overwrites slots whose `#{slot=N}` `Definition` exists in source.
    ///
    /// `input_path` is the file the splice path writes back to (use an
    /// empty `PathBuf` to suppress the write, e.g. in tests).
    pub fn from_source(
        source: String,
        input_path: std::path::PathBuf,
    ) -> Result<(ProgramSet, String), Vec<parser::Error>> {
        let mut message = String::new();
        let (bindings, errors) = parser::parse_module::<MarkId>(&source)?;
        // TODO sort of a bummer that we don't know which binding this error was
        // in... some opportunity here to improve the type of parse_module.
        if !errors.is_empty() {
            message = format!("Parse errors: {}", &errors[0]);
        }
        let total_slots = NUM_PROGRAM_BANKS * PROGRAMS_PER_BANK;
        let mut programs: Vec<Program> = (0..total_slots)
            .map(|_| Program::from_string("", bindings.len()))
            .collect();
        for (binding_index, sb) in bindings.iter().enumerate() {
            if let Some((program_index, program)) =
                Program::from_source_binding(sb, binding_index, &source)
            {
                if program_index < programs.len() {
                    programs[program_index] = program;
                } else {
                    println!(
                        "Ignoring program with out-of-range slot {} (max {})",
                        program_index + 1,
                        programs.len()
                    );
                }
            }
        }
        Ok((
            ProgramSet {
                programs,
                bindings,
                source,
                input_path,
            },
            message,
        ))
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
            parser::Binding::Definition(pattern, _) => match pattern {
                parser::Pattern::Identifier(name) if name == "_" => String::new(),
                _ => format!("{}", pattern),
            },
            _ => String::new(),
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
        let bank = index / PROGRAMS_PER_BANK;
        let slot_in_bank = (index % PROGRAMS_PER_BANK) + 1;
        let bank_letter = (b'A' + bank as u8) as char;
        let name = self.name(index);
        if name.is_empty() {
            format!("{}:{}", bank_letter, slot_in_bank)
        } else {
            format!("{}:{} ({})", bank_letter, slot_in_bank, name)
        }
    }

    /// Returns the bindings to evaluate the program at `index` under: the
    /// file-level bindings preceding it (bindings after it are ignored),
    /// with anonymous `_` definitions filtered out, plus a binding for each
    /// of the program's sliders at its current value.
    pub fn evaluation_bindings(&self, index: usize) -> Vec<parser::SourceBinding<MarkId>> {
        let program = &self.programs[index];
        let mut bindings: Vec<parser::SourceBinding<MarkId>> =
            self.bindings[..program.binding_index].to_vec();
        // TODO this is a pretty big hack but there's an interesting question
        // about what sliders in *other* bindings mean. To avoid answering that
        // for the moment, just assume that only "_" bindings have sliders and
        // that they can't be used so we can safely filter them out here. I
        // think the right answer is that we should bind sliders uniquely in
        // each binding... or at least those that have slots? (Otherwise, how
        // can you modify the slider value?)
        bindings.retain(|b| match &b.binding {
            parser::Binding::Definition(p, _) => {
                !matches!(p, parser::Pattern::Identifier(v) if v == "_")
            }
            _ => true,
        });
        slider::append_slider_bindings(
            program.sliders.configs(),
            program.sliders.normalized_values(),
            MarkId::Slider,
            &mut bindings,
        );
        bindings
    }
}

/// Float tolerance for "this hasn't moved" comparisons against parsed
/// annotation values. Tighter than the source format's precision would
/// matter (which would be confusing) and looser than encoder/MIDI noise.
const ANNOTATION_EPSILON: f32 = 1e-4;

/// Returns annotation-persistence edits for `program` against its parsed
/// `binding`.
///
/// This includes one for `level_db` if the runtime level has diverged from what
/// the binding's last `Level` annotation encodes, and one for `sliders` if any
/// slider's current normalized value has diverged from its parsed initial
/// value. Returns an empty list when nothing has changed.
fn annotation_edits(
    program: &Program,
    binding: &parser::SourceBinding<MarkId>,
    source: &str,
) -> Vec<(std::ops::Range<usize>, String)> {
    let mut edits = Vec::new();
    if let Some(edit) = level_edit(program, binding, source) {
        edits.push(edit);
    }
    if let Some(edit) = sliders_edit(program, binding) {
        edits.push(edit);
    }
    edits
}

/// Returns the edit for the binding's `level_db` annotation, or `None` if
/// the runtime level matches what the binding currently encodes.
fn level_edit(
    program: &Program,
    binding: &parser::SourceBinding<MarkId>,
    source: &str,
) -> Option<(std::ops::Range<usize>, String)> {
    let (parsed_value, parsed_span) = match last_annotation_of(binding, |a| match a {
        parser::Annotation::Level(v) => Some(*v),
        _ => None,
    }) {
        Some((v, span)) => (v, span),
        None => (0.0, None),
    };
    if (program.level_db() - parsed_value).abs() < ANNOTATION_EPSILON {
        return None;
    }
    let annotation = parser::Annotation::Level(program.level_db());
    let body = format!("{}", annotation);
    match parsed_span {
        Some(span) => Some((span, body)),
        None => {
            // Fall back to inserting a fresh `#{level_db=…}` line at the
            // binding's start when no `Level` annotation exists in source yet.
            let pos = binding
                .span
                .as_ref()
                .expect("parsed binding has span")
                .start;
            Some(insert_annotation_line(pos, &body, source))
        }
    }
}

/// Builds an `(range, replacement)` edit that inserts a fresh `#{…}`
/// annotation line at `pos`, prepending or appending `\n` only when
/// needed so the inserted line doesn't collapse onto a neighbor.
fn insert_annotation_line(
    pos: usize,
    body: &str,
    source: &str,
) -> (std::ops::Range<usize>, String) {
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

/// Returns the edit for the binding's `sliders` annotation, or `None` if
/// every slider's current normalized value matches its parsed initial.
fn sliders_edit(
    program: &Program,
    binding: &parser::SourceBinding<MarkId>,
) -> Option<(std::ops::Range<usize>, String)> {
    if program.sliders().configs().is_empty() {
        return None;
    }
    let diverged = program
        .sliders()
        .configs()
        .iter()
        .zip(program.sliders().normalized_values())
        .any(|(config, &current)| {
            (current - parsed_normalized_value(&config.function)).abs() > ANNOTATION_EPSILON
        });
    if !diverged {
        return None;
    }
    let (_, span) = last_annotation_of(binding, |a| match a {
        parser::Annotation::Sliders(_) => Some(()),
        _ => None,
    })?;
    let span = span?;
    let updated: Vec<parser::Slider> = program
        .sliders()
        .configs()
        .iter()
        .zip(program.sliders().normalized_values())
        .map(|(config, &normalized)| {
            let function = match &config.function {
                parser::SliderFunction::Linear { min, max, .. } => parser::SliderFunction::Linear {
                    initial_value: min + normalized * (max - min),
                    min: *min,
                    max: *max,
                },
                parser::SliderFunction::UserDefined {
                    function_source, ..
                } => parser::SliderFunction::UserDefined {
                    normalized_initial_value: normalized,
                    function_source: function_source.clone(),
                },
            };
            parser::Slider {
                label: config.label.clone(),
                function,
            }
        })
        .collect();
    Some((span, format!("{}", parser::Annotation::Sliders(updated))))
}

/// Returns the (value, span) of the last annotation on `binding` that `pick`
/// matches.
///
/// The reverse walk matches the parser's "last wins" semantics for repeated
/// annotations of the same kind, so persisting onto the same span keeps the
/// source authoritative.
fn last_annotation_of<T, F>(
    binding: &parser::SourceBinding<MarkId>,
    mut pick: F,
) -> Option<(T, Option<std::ops::Range<usize>>)>
where
    F: FnMut(&parser::Annotation) -> Option<T>,
{
    binding
        .annotations
        .iter()
        .rev()
        .find_map(|a| pick(&a.annotation).map(|v| (v, a.span.clone())))
}

/// Returns the normalized 0..1 position implied by a parsed slider's
/// initial value, so it can be compared against the runtime's current
/// `normalized_value` directly.
fn parsed_normalized_value(function: &parser::SliderFunction) -> f32 {
    match function {
        parser::SliderFunction::Linear {
            initial_value,
            min,
            max,
        } => (initial_value - min) / (max - min),
        parser::SliderFunction::UserDefined {
            normalized_initial_value,
            ..
        } => *normalized_initial_value,
    }
}

impl ProgramSet {
    /// Splices the given program's edited `text` back into `self.source`, persists
    /// every program's changed state as annotation edits, re-parses the file to
    /// refresh `self.bindings`, realigns each program's `span`/`binding_index`
    /// (matching by `slot=N` annotation), and writes the new source to
    /// `self.input_path`.
    ///
    /// Programs whose `span` is empty (`0..0`) are treated as brand-new: a fresh
    /// `Definition` is appended at the end of source with a generated name and a
    /// `#{slot=N}` annotation so the file picks it up on re-parse. Existing
    /// programs are spliced in place.
    ///
    /// Returns a user-visible warning message if any step failed; in that case
    /// `self.bindings`, `self.source`, and the file on disk are all left
    /// unchanged.
    pub fn splice(&mut self, program_index: usize) -> Result<(), String> {
        let slot = program_index + 1;
        let edited_span = self.programs[program_index].span.clone();
        let mut edited_text = self.programs[program_index].text().to_string();
        // Remove any semicolons since these aren't valid within an expression and
        // can defeat parsing error recovery.
        edited_text.retain(|c| c != ';');
        let is_new = edited_span.start == edited_span.end;

        // Collect every edit we want to apply to `self.source` in one pass:
        // the active program's expression splice (or a fresh-binding append)
        // plus, for every program, annotation edits that bring level/slider
        // state on disk in line with the runtime. Applying in reverse-position
        // order keeps each remaining edit's span valid.
        let mut edits: Vec<(std::ops::Range<usize>, String)> = Vec::new();
        let mut append: Option<String> = None;
        if is_new {
            if edited_text.trim().is_empty() {
                // Padding slot still empty after edit — nothing to do.
                // TODO is this true? what about sliders/levels?
                return Ok(());
            }
            // Build the appended binding inline: slot is mandatory; level_db
            // gets added when the runtime has a non-default level (new
            // programs have no sliders since their configs come from a
            // parsed `sliders=…` annotation).
            let mut annos: Vec<parser::Annotation> = vec![parser::Annotation::Slot(slot as u32)];
            let level = self.programs[program_index].level_db();
            if level.abs() > ANNOTATION_EPSILON {
                annos.push(parser::Annotation::Level(level));
            }
            let anno_body = annos
                .iter()
                .map(|a| format!("{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            append = Some(format!("\n#{{{}}}\n_ = {};", anno_body, edited_text));
        } else {
            edits.push((edited_span, edited_text));
        }

        // Walk every program (active and not) and append annotation edits
        // for level_db / sliders that have diverged from the parsed source.
        // We use the *pre-splice* `self.bindings` so all spans line up with
        // `self.source` before any of these edits land.
        for program in &self.programs {
            if let Some(binding) = self.bindings.get(program.binding_index) {
                edits.extend(annotation_edits(program, binding, &self.source));
            }
        }

        let mut new_source = self.source.clone();
        edits.sort_by_key(|(span, _)| std::cmp::Reverse(span.start));
        for (span, replacement) in edits {
            new_source.replace_range(span, &replacement);
        }
        if let Some(text) = append {
            new_source.push_str(&text);
        }

        // It's ok to drop the errors here: if parse_module succeeded then we know
        // that the resulting bindings span the entire input (even if there are
        // errors).
        let (new_bindings, _errors) = parser::parse_module::<MarkId>(&new_source)
            .map_err(|errs| format!("Warning: source re-parse failed: {:?}", errs))?;

        // Build a slot -> (binding_index, expr_span) map from the new bindings,
        // so we can realign each `programs[slot-1]` independently of any
        // shifts in source order.
        let slot_lookup: Vec<Option<(usize, std::ops::Range<usize>)>> = {
            let mut by_slot: std::collections::HashMap<u32, (usize, std::ops::Range<usize>)> =
                std::collections::HashMap::new();
            for (i, sb) in new_bindings.iter().enumerate() {
                let slot = sb.annotations.iter().find_map(|sa| match &sa.annotation {
                    parser::Annotation::Slot(n) => Some(*n),
                    _ => None,
                });
                if let Some(slot) = slot
                    && let parser::Binding::Definition(_, expr) = &sb.binding
                    && let Some(s) = &expr.span
                    && s.end <= new_source.len()
                {
                    by_slot.insert(slot, (i, s.clone()));
                }
            }
            (1..=self.programs.len() as u32)
                .map(|slot| by_slot.remove(&slot))
                .collect()
        };

        for (i, program) in self.programs.iter_mut().enumerate() {
            match &slot_lookup[i] {
                Some((binding_index, span)) => {
                    program.realign(*binding_index, span.clone(), &new_source);
                }
                None => {
                    // No binding for this slot — keep it as a padding slot.
                    program.mark_padding(new_bindings.len());
                }
            }
        }

        // Skip the file write when no path is set — tests use an empty
        // `PathBuf` so they can exercise the splice without touching disk.
        if !self.input_path.as_os_str().is_empty()
            && let Err(e) = std::fs::write(&self.input_path, &new_source)
        {
            return Err(format!(
                "Warning: failed to write {}: {}",
                self.input_path.display(),
                e
            ));
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
    use std::path::PathBuf;

    /// Builds a `ProgramSet` from inline source, with `input_path` empty
    /// so `splice` is a no-op against disk.
    fn state_from(source: &str) -> ProgramSet {
        ProgramSet::from_source(source.to_string(), PathBuf::new())
            .expect("test source should parse")
            .0
    }

    #[test]
    fn set_slider_normalized_clamps_and_rejects_out_of_range() {
        let source = "#{slot=1, sliders=[\"vol:0.5:0:1\"]}\ntone = saw(220);";
        let mut set = state_from(source);
        // No slider at index 1 → None, nothing changed.
        assert!(
            set.program_mut(0)
                .unwrap()
                .set_slider_normalized(1, 0.5)
                .is_none()
        );
        // Values above the range clamp to 1.0.
        let change = set
            .program_mut(0)
            .unwrap()
            .set_slider_normalized(0, 1.5)
            .unwrap();
        assert_eq!(change.label, "vol");
        assert!((change.value - 1.0).abs() < 1e-6);
        assert!((set.programs()[0].sliders().normalized_values()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn record_evaluation_replaces_both_caches() {
        let mut program = Program::from_string("x", 0);
        assert!(
            program
                .record_evaluation(Evaluation::Waveform(waveform::Waveform::Const(1.0)))
                .is_ok()
        );
        assert!(program.waveform().is_some());
        assert!(program.keys_instrument().is_none());

        assert!(
            program
                .record_evaluation(Evaluation::KeysInstrument(parser::SourceExpr::float(1.0)))
                .is_ok()
        );
        assert!(program.waveform().is_none());
        assert!(program.keys_instrument().is_some());

        // Invalid clears both caches and hands back the message.
        assert_eq!(
            program.record_evaluation(Evaluation::Invalid("nope".to_string())),
            Err("nope".to_string())
        );
        assert!(program.waveform().is_none());
        assert!(program.keys_instrument().is_none());
    }

    #[test]
    fn evaluation_bindings_filters_anonymous_and_appends_sliders() {
        let source = "\
pi = 3.14;
#{slot=1}
_ = pulse(60);
#{slot=2, sliders=[\"vol:0.5:0:1\"]}
tone = saw(220);";
        let set = state_from(source);
        let names: Vec<String> = set
            .evaluation_bindings(1)
            .iter()
            .filter_map(|b| match &b.binding {
                parser::Binding::Definition(parser::Pattern::Identifier(name), _) => {
                    Some(name.clone())
                }
                _ => None,
            })
            .collect();
        // `pi` precedes the program and survives; the anonymous `_` slot-1
        // definition is filtered; the program's own slider is appended.
        assert_eq!(names, vec!["pi".to_string(), "vol".to_string()]);
    }

    #[test]
    fn editing_existing_program_splices_into_its_span() {
        let source = "\
#{slot=1}
kick = pulse(60);
#{slot=2}
synth = saw(220);";
        let mut state = state_from(source);

        // Slot 2 → index 1. Sanity-check the initial extraction.
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
#{slot=1}
kick = pulse(60);
#{slot=2}
synth = saw(440);"
        );
        // Slot 1 untouched, slot 2's program now reflects the new text and
        // span.
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        assert_eq!(state.programs()[1].text(), "saw(440)");
    }

    #[test]
    fn editing_program_does_not_disturb_library_bindings() {
        // A `Definition` without a `slot=N` annotation is a library
        // binding, not a UI program. Editing a UI program must leave
        // library bindings (and `Open` directives) in place.
        let source = "\
open util.synths;
pi = 3.14159;
#{slot=1}
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
#{slot=1}
tone = saw(220);"
        );
    }

    #[test]
    fn new_program_is_appended_with_slot_annotation() {
        // Editing a previously-empty padding slot (span 0..0) should
        // append a fresh `#{slot=N}\n_ = <text>` Definition at the
        // end of source rather than splicing at offset 0.
        let source = "\
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].text(), "pulse(60)");
        // Slot 5 is padding initially.
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
#{slot=1}
kick = pulse(60);
#{slot=5}
_ = saw(440);"
        );
        // The new binding is now part of the bindings vec, and the slot 5
        // program has a real span/text.
        assert_eq!(state.programs()[4].text(), "saw(440)");
        assert!(state.programs()[4].span.clone().start < state.programs()[4].span.clone().end);
    }

    // TODO add a test where a new program is adding in the middle

    #[test]
    fn editing_padding_slot_with_empty_text_is_a_no_op() {
        // The user enters Edit on an empty slot, types nothing, hits
        // Return. We must not append a phantom `_ = ` definition.
        let source = "\
#{slot=1}
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
#{slot=1, level_db=-6}
// keep this comment
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs()[0].level_db() - -6.0).abs() < 1e-6);

        state.program_mut(0).unwrap().set_level_db(-3.5);
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
#{slot=1, level_db=-3.5}
// keep this comment
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_level_db_inserts_fresh_annotation_when_none_exists() {
        // No `level_db=…` in source. Add a new `#{level_db=…}` line just before
        // the binding, preserving the existing annotation block and the
        // pre-binding comment.
        let source = "\
// header comment
#{slot=1}
kick = pulse(60);";
        let mut state = state_from(source);
        assert!((state.programs()[0].level_db() - 0.0).abs() < 1e-6);

        state.program_mut(0).unwrap().set_level_db(-3.0);
        state.splice(0).unwrap();

        assert_eq!(
            state.source,
            "\
#{level_db=-3}
// header comment
#{slot=1}
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_slider_value_rewrites_sliders_annotation() {
        // Move the slider from its parsed initial 0.5 to 0.75. Only
        // the `sliders=…` annotation's bytes change; the surrounding
        // `slot=…` annotation and binding body stay verbatim.
        let source = "\
#{slot=1, sliders=[\"vol:0.5:0:1\"]}
tone = saw(220);";
        let mut state = state_from(source);
        assert_eq!(state.programs()[0].sliders().configs().len(), 1);
        assert!((state.programs()[0].sliders().normalized_values()[0] - 0.5).abs() < 1e-6);

        let _ = state.program_mut(0).unwrap().set_slider_normalized(0, 0.75);
        state.splice(0).unwrap();

        // Linear slider: actual value = min + normalized * (max - min)
        // = 0 + 0.75 * 1 = 0.75.
        assert_eq!(
            state.source,
            "\
#{slot=1, sliders=[\"vol:0.75:0:1\"]}
tone = saw(220);"
        );
    }

    #[test]
    fn no_divergence_means_no_annotation_edits() {
        // Splicing without any runtime divergence should round-trip
        // the source exactly — even though splice_program walks every
        // program looking for level/slider drift.
        let source = "\
#{slot=1, level_db=-6, sliders=[\"vol:0.25:0:1\"]}
kick = pulse(60);
#{slot=2}
synth = saw(220);";
        let mut state = state_from(source);
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
#{slot=1}
kick = pulse(60);
#{slot=2}
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
#{slot=1}
kick = pulse(80);
#{level_db=-9}
#{slot=2}
synth = saw(220);"
        );
    }

    #[test]
    fn new_program_appends_level_db_when_set() {
        // A brand-new program with a non-default runtime level should emit
        // `level_db` alongside `slot` in its appended `#{…}` header so the file
        // picks it up on the next parse.
        let source = "\
#{slot=1}
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
#{slot=1}
kick = pulse(60);
#{slot=5, level_db=-2.5}
_ = saw(440);"
        );
    }
}
