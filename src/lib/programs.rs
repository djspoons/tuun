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
    // TODO make private to this module once the Keys sliders mirror is gone.
    pub(crate) fn set_normalized(&mut self, index: usize, normalized: f32) -> Option<SliderChange> {
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

    /// Returns the span of this program's expression in the source file.
    // TODO remove once splice lives in this module.
    pub(crate) fn span(&self) -> Range<usize> {
        self.span.clone()
    }

    /// Returns the index of the originating binding in the file's bindings.
    // TODO remove once splice and evaluation_bindings live in this module.
    pub(crate) fn binding_index(&self) -> usize {
        self.binding_index
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
