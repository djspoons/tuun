//! Programs and their state: text, sliders, level, and evaluation caches.

use std::fmt;
use std::ops::Range;

use crate::ids::MarkId;
use crate::parser;
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
//
// `Program` exposes a narrow API — fields are private so the rest of the
// UI (renderer, input handlers, reducer, effect runner) can only observe
// programs through getters and mutate them through methods that preserve
// the internal invariants.
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
    /// file's bindings vec.
    ///
    /// Only `Definition`s with at least one annotation become programs.
    /// `Definition`s with no annotations are not returned.
    pub(crate) fn from_source_binding(
        sb: &parser::SourceBinding<MarkId>,
        binding_index: usize,
        source: &str,
    ) -> Option<Program> {
        if sb.annotations.is_empty() {
            return None;
        }
        let mut sliders = ProgramSliders::default();
        let mut color: Option<(u8, u8, u8)> = None;
        let mut level_db: f32 = 0.0;
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
                parser::Annotation::SkipSlots(_) => {
                    // Consumed by the position walk, not stored on Program.
                }
            }
        }
        if let parser::Binding::Definition(_, expr) = &sb.binding {
            if let Some(s) = &expr.span
                && s.end <= source.len()
            {
                let span = s.clone();
                let text = source[s.clone()].to_string();
                Some(Program {
                    text,
                    span,
                    binding_index,
                    sliders,
                    color,
                    level_db,
                    cached_waveform: None,
                    cached_keys_instrument: None,
                })
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

    /// Records the result of evaluating the program's current text, replacing
    /// both caches. Returns the user-visible message as an error when the
    /// evaluation was invalid.
    ///
    /// An `Invalid` evaluation still clears both caches: even though editing
    /// already clears them, the failure may have come from a changed dependency
    /// rather than this program's own text.
    fn record_evaluation(&mut self, evaluation: Evaluation) -> Result<(), String> {
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
    fn realign(&mut self, binding_index: usize, span: Range<usize>, source: &str) {
        self.binding_index = binding_index;
        self.text = source[span.clone()].to_string();
        self.span = span;
    }

    /// Marks the program as a padding slot with no binding: the binding
    /// index points one past the end of the bindings and the span is empty.
    fn mark_padding(&mut self, binding_count: usize) {
        self.binding_index = binding_count;
        self.span = 0..0;
    }
}

/// Returns the last `skip_slots=N` value on `sb`, or 0 if none is present.
fn read_skip_slots(sb: &parser::SourceBinding<MarkId>) -> u32 {
    // The reverse walk mirrors the "last wins" semantics used elsewhere for
    // repeated annotations of the same kind.
    sb.annotations
        .iter()
        .rev()
        .find_map(|sa| match &sa.annotation {
            parser::Annotation::SkipSlots(n) => Some(*n),
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
    bindings: &[parser::SourceBinding<MarkId>],
    source_len: usize,
) -> Vec<(usize, usize, std::ops::Range<usize>)> {
    let mut out = Vec::new();
    let mut position: usize = 0;
    for (i, sb) in bindings.iter().enumerate() {
        if sb.annotations.is_empty() {
            continue;
        }
        if let parser::Binding::Definition(_, expr) = &sb.binding
            && let Some(s) = &expr.span
            && s.end <= source_len
        {
            position += read_skip_slots(sb) as usize;
            out.push((position, i, s.clone()));
            position += 1;
        }
    }
    out
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
    bindings: Vec<parser::SourceBinding<MarkId>>,
    source: String,
    input_path: std::path::PathBuf,
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
        let mut position: usize = 0;
        for (binding_index, sb) in bindings.iter().enumerate() {
            if let Some(program) = Program::from_source_binding(sb, binding_index, &source) {
                position += read_skip_slots(sb) as usize;
                if position < programs.len() {
                    programs[position] = program;
                } else {
                    println!(
                        "Ignoring program with out-of-range slot {} (max {})",
                        position + 1,
                        programs.len()
                    );
                }
                position += 1;
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

    /// Evaluates the program at `index` and records the result in its
    /// evaluation caches. Returns the user-visible message as an error when the
    /// evaluation was invalid (the caches are still cleared in that case).
    pub fn evaluate_and_record(
        &mut self,
        evaluator: &crate::evaluator::Evaluator,
        index: usize,
    ) -> Result<(), String> {
        // An empty program is a deletion, not a parse error: clear both caches
        // and succeed so the editor can leave Edit mode (the splice that
        // follows removes the binding from source).
        if self.programs[index].text().trim().is_empty() {
            let program = &mut self.programs[index];
            program.cached_waveform = None;
            program.cached_keys_instrument = None;
            return Ok(());
        }
        let evaluation = evaluator.evaluate_program(self, index);
        self.programs[index].record_evaluation(evaluation)
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

/// Builds the edit that sets `binding`'s `skip_slots` annotation to `new_skip`,
/// or `None` when the source already encodes that value.
///
/// Replaces the existing annotation in place, inserts a fresh `#{skip_slots=…}`
/// line when none exists, or removes the annotation entirely when `new_skip` is
/// 0 — swapping in `level_db=…` instead when `skip_slots` is the binding's only
/// annotation, so the binding stays a UI program.
fn skip_slots_edit(
    binding: &parser::SourceBinding<MarkId>,
    new_skip: u32,
    level_db: f32,
    source: &str,
) -> Option<(std::ops::Range<usize>, String)> {
    if read_skip_slots(binding) == new_skip {
        return None;
    }
    let existing = last_annotation_of(binding, |a| match a {
        parser::Annotation::SkipSlots(_) => Some(()),
        _ => None,
    })
    .and_then(|(_, span)| span);
    let body = format!("{}", parser::Annotation::SkipSlots(new_skip));
    match existing {
        Some(span) if new_skip > 0 => Some((span, body)),
        Some(span) if binding.annotations.len() == 1 => {
            Some((span, format!("{}", parser::Annotation::Level(level_db))))
        }
        Some(span) => Some(remove_annotation_edit(span, source)),
        None if new_skip > 0 => {
            let pos = binding
                .span
                .as_ref()
                .expect("parsed binding has span")
                .start;
            Some(insert_annotation_line(pos, &body, source))
        }
        // No annotation and no skip needed.
        None => None,
    }
}

/// Builds the edit that removes the annotation at `span` from its `#{…}` set,
/// consuming an adjacent comma (and surrounding spacing) so the remaining
/// annotations stay well-formed. When the annotation is alone in its set,
/// removes the whole `#{…}` group and any trailing newline.
fn remove_annotation_edit(
    span: std::ops::Range<usize>,
    source: &str,
) -> (std::ops::Range<usize>, String) {
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
    /// via the position walk, and writes the new source to `self.input_path`.
    ///
    /// Programs with no source binding (padding slots) are treated as
    /// brand-new: a fresh `Definition` is inserted between its source-order
    /// neighbors with a `#{skip_slots=…, level_db=…}` annotation, and the
    /// following program's `skip_slots` is decremented to keep its absolute
    /// slot stable. Existing programs are spliced in place — except when the
    /// edited text is empty (or whitespace-only), which deletes the binding
    /// (annotations included) from source and grows the following program's
    /// `skip_slots` to compensate.
    ///
    /// Returns a user-visible warning message if any step failed; in that case
    /// `self.bindings`, `self.source`, and the file on disk are all left
    /// unchanged.
    pub fn splice(&mut self, program_index: usize) -> Result<(), String> {
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
        let mut edits: Vec<(std::ops::Range<usize>, String)> = Vec::new();
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
            let mut annos: Vec<parser::Annotation> = Vec::new();
            if new_skip > 0 {
                annos.push(parser::Annotation::SkipSlots(new_skip as u32));
            }
            annos.push(parser::Annotation::Level(
                self.programs[program_index].level_db(),
            ));
            let anno_body = annos
                .iter()
                .map(|a| format!("{}", a))
                .collect::<Vec<_>>()
                .join(", ");
            let anchor = match next {
                Some((_, next_binding_index)) => {
                    self.bindings[next_binding_index]
                        .span
                        .as_ref()
                        .expect("parsed binding has span")
                        .start
                }
                None => self.source.len(),
            };
            let bytes = self.source.as_bytes();
            let prefix = if anchor == 0 || bytes.get(anchor - 1) == Some(&b'\n') {
                ""
            } else {
                "\n"
            };
            let suffix = if anchor == self.source.len() || bytes.get(anchor) == Some(&b'\n') {
                ""
            } else {
                "\n"
            };
            let new_binding = format!(
                "{}#{{{}}}\n_ = {};{}",
                prefix, anno_body, edited_text, suffix
            );

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

            edits.push((anchor..anchor, new_binding));
        } else if is_deletion {
            // Clearing an existing program deletes its whole binding —
            // leading trivia, annotations, definition, and terminating
            // `;` — and grows the following UI program's `skip_slots` so
            // every later program keeps its slot.
            let binding_span = self.bindings[binding_index]
                .span
                .clone()
                .expect("parsed binding has span");
            edits.push((binding_span, String::new()));

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
                edits.extend(annotation_edits(program, binding, &self.source));
            }
        }

        let mut new_source = self.source.clone();
        edits.sort_by_key(|(span, _)| std::cmp::Reverse(span.start));
        for (span, replacement) in edits {
            new_source.replace_range(span, &replacement);
        }

        // It's ok to drop the errors here: if parse_module succeeded then we know
        // that the resulting bindings span the entire input (even if there are
        // errors).
        let (new_bindings, _errors) = parser::parse_module::<MarkId>(&new_source)
            .map_err(|errs| format!("Warning: source re-parse failed: {:?}", errs))?;

        // Realign each program to its position in the re-parsed source by
        // running the same position walk used at load time.
        let slot_lookup: Vec<Option<(usize, std::ops::Range<usize>)>> = {
            let mut lookup: Vec<Option<(usize, std::ops::Range<usize>)>> =
                vec![None; self.programs.len()];
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
                    program.realign(*binding_index, span.clone(), &new_source);
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
            self.programs[program_index] = Program::from_string("", new_bindings.len());
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
        let source = "#{sliders=[\"vol:0.5:0:1\"]}\ntone = saw(220);";
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
#{level_db=0}
_ = pulse(60);
#{sliders=[\"vol:0.5:0:1\"]}
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
_ = saw(440);"
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
        // `color=…`). Add a new `#{level_db=…}` line just before the binding,
        // preserving the existing annotation block and the pre-binding comment.
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
#{level_db=-3}
// header comment
#{color=rgb(255,0,0)}
kick = pulse(60);"
        );
    }

    #[test]
    fn changed_slider_value_rewrites_sliders_annotation() {
        // Move the slider from its parsed initial 0.5 to 0.75. Only the
        // `sliders=…` annotation's bytes change; the binding body stays
        // verbatim.
        let source = "\
#{sliders=[\"vol:0.5:0:1\"]}
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
#{sliders=[\"vol:0.75:0:1\"]}
tone = saw(220);"
        );
    }

    #[test]
    fn no_divergence_means_no_annotation_edits() {
        // Splicing without any runtime divergence should round-trip
        // the source exactly — even though splice_program walks every
        // program looking for level/slider drift.
        let source = "\
#{level_db=-6, sliders=[\"vol:0.25:0:1\"]}
kick = pulse(60);
#{level_db=0}
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
_ = saw(440);"
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
#{skip_slots=1}
#{level_db=0}
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
#{level_db=-6, sliders=[\"vol:0.5:0:1\"]}
kick = pulse(60);";
        let mut state = state_from(source);
        state.program_mut(0).unwrap().set_text(String::new());
        state.splice(0).unwrap();

        assert_eq!(state.source, "");
        assert!(state.programs()[0].is_empty());
        assert!((state.programs()[0].level_db() - 0.0).abs() < 1e-6);
        assert!(state.programs()[0].sliders().configs().is_empty());
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
    fn evaluating_an_empty_program_succeeds_and_clears_caches() {
        // Clearing a program's text must not surface a parse error from the
        // evaluator — an empty program is a deletion in progress.
        let mut state = state_from("#{level_db=0}\nkick = pulse(60);");
        state.program_mut(0).unwrap().set_text("  ".to_string());
        let evaluator = crate::evaluator::Evaluator::new(44100, 90, std::path::PathBuf::new());
        assert!(state.evaluate_and_record(&evaluator, 0).is_ok());
        assert!(state.programs()[0].waveform().is_none());
        assert!(state.programs()[0].keys_instrument().is_none());
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
}
