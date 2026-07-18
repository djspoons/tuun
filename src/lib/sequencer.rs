//! Analysis and text editing of step-sequenceable programs.
//!
//! A program is sequenceable when its body is a single top-level call
//! `on_beats(w, [b1, b2, ...])` whose beat list holds only float literals.
//! The pad grid addresses the measure in sixteenths (quarter-beats):
//! sixteenth `k` is beat `1 + k/4`, and covers the window `[b, b + 0.25)`.

use std::ops::Range;

use crate::diagnostics::Source;
use crate::expr::{Expr, SourceExpr};
use crate::ids::MarkId;
use crate::parser;
use crate::waveform;

/// Beat-comparison tolerance for classifying a literal as on-grid.
const EPS: f32 = 1e-3;

/// The decomposed form of a sequenceable program: what the sequencer plays
/// one tracker entry per beat from.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// The listed beats, in source order.
    pub beats: Vec<f32>,
    /// The waveform argument `w`, evaluated in the program's context.
    /// Unoptimized; slider values and the level wrap are applied per play.
    pub step_waveform: waveform::Waveform<MarkId>,
}

/// The syntactic shape of a sequenceable program.
pub struct SequenceShape {
    /// The waveform argument `w`, unevaluated.
    pub waveform_expr: SourceExpr<MarkId, Source>,
    /// The byte range of the beat-list literal, brackets included.
    pub list_range: Range<usize>,
    /// Each beat literal with its byte range, in source order.
    pub beats: Vec<(f32, Range<usize>)>,
}

/// The display state of one sixteenth of the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SixteenthState {
    /// No beat in this sixteenth's window.
    Empty,
    /// A beat sits exactly on this sixteenth.
    OnGrid,
    /// Only off-grid beats sit inside this sixteenth's window.
    OffGridOnly,
}

/// The result of toggling a sixteenth in a sequenceable program's text.
pub struct ToggleEdit {
    pub new_text: String,
    /// True when the toggle added a beat; false when it removed the
    /// sixteenth's beats.
    pub turned_on: bool,
    /// The sixteenth's grid beat, for user-facing messages.
    pub beat: f32,
    /// Start of the edited region, for shifting a cursor sitting after it.
    pub edit_start: usize,
    /// Signed byte-length change of the text.
    pub delta: isize,
}

/// Returns the grid beat addressed by a sixteenth (sixteenth 0 = beat 1).
pub fn sixteenth_beat(sixteenth: u8) -> f32 {
    1.0 + sixteenth as f32 * 0.25
}

/// Returns the sixteenth whose window contains `beat`.
///
/// Clamped to `u8::MAX - 1` so a listed beat can never collide with the
/// anchor sentinel.
pub fn sixteenth_for_beat(beat: f32) -> u8 {
    ((beat - 1.0) * 4.0)
        .floor()
        .clamp(0.0, (u8::MAX - 1) as f32) as u8
}

/// Returns whether `beat` falls in the given sixteenth's window.
fn in_window(beat: f32, sixteenth: u8) -> bool {
    let b = sixteenth_beat(sixteenth);
    beat >= b && beat < b + 0.25
}

/// Returns the display state of a sixteenth given a program's beat list.
pub fn sixteenth_state(beats: &[f32], sixteenth: u8) -> SixteenthState {
    let b = sixteenth_beat(sixteenth);
    let mut in_range = false;
    for &beat in beats {
        if (beat - b).abs() <= EPS {
            return SixteenthState::OnGrid;
        }
        in_range |= in_window(beat, sixteenth);
    }
    if in_range {
        SixteenthState::OffGridOnly
    } else {
        SixteenthState::Empty
    }
}

/// Returns the sequenceable shape of a program's text, or `None` when the
/// text doesn't parse or isn't a single top-level `on_beats` call with a
/// literal beat list.
pub fn analyze(text: &str) -> Option<SequenceShape> {
    let expr = parser::parse_program::<MarkId, _>(text, Source::Program).ok()?;
    let Expr::Application {
        function,
        positional,
        named,
    } = expr.expr
    else {
        return None;
    };
    if !named.is_empty() || positional.len() != 2 {
        return None;
    }
    match &function.expr {
        Expr::Variable(name) if name == "on_beats" => {}
        _ => return None,
    }
    let mut positional = positional;
    let list = positional.pop().expect("length checked above");
    let waveform_expr = positional.pop().expect("length checked above");
    let list_range = list.span.as_ref()?.range.clone();
    let Expr::List(elements) = list.expr else {
        return None;
    };
    let mut beats = Vec::with_capacity(elements.len());
    for element in elements {
        let Expr::Float(value) = element.expr else {
            return None;
        };
        beats.push((value, element.span.as_ref()?.range.clone()));
    }
    Some(SequenceShape {
        waveform_expr,
        list_range,
        beats,
    })
}

/// Toggles a sixteenth in the beat-list text: removes every beat in its
/// window when any is present, otherwise inserts the sixteenth's grid beat
/// at its sorted position.
///
/// Only the affected elements and separators change; the rest of the text
/// is preserved byte for byte.
pub fn toggle_sixteenth_text(text: &str, shape: &SequenceShape, sixteenth: u8) -> ToggleEdit {
    let beat = sixteenth_beat(sixteenth);
    let matched: Vec<usize> = (0..shape.beats.len())
        .filter(|&i| in_window(shape.beats[i].0, sixteenth))
        .collect();

    let mut new_text = text.to_string();
    if matched.is_empty() {
        // Insert at the sorted position, formatted minimally ("3", "2.5").
        let formatted = format!("{}", beat);
        let (insert_at, insertion) = match shape.beats.iter().find(|(value, _)| *value > beat) {
            Some((_, range)) => (range.start, format!("{}, ", formatted)),
            None => match shape.beats.last() {
                Some((_, range)) => (range.end, format!(", {}", formatted)),
                None => (shape.list_range.start + 1, formatted),
            },
        };
        new_text.insert_str(insert_at, &insertion);
        return ToggleEdit {
            new_text,
            turned_on: true,
            beat,
            edit_start: insert_at,
            delta: insertion.len() as isize,
        };
    }

    // Remove maximal runs of consecutive matched elements, each with one
    // adjoining separator, so ranges never overlap even when a window holds
    // several beats.
    let mut removals: Vec<Range<usize>> = Vec::new();
    let mut run_start = 0;
    while run_start < matched.len() {
        let mut run_end = run_start;
        while run_end + 1 < matched.len() && matched[run_end + 1] == matched[run_end] + 1 {
            run_end += 1;
        }
        let (first, last) = (matched[run_start], matched[run_end]);
        let range = if last + 1 < shape.beats.len() {
            // A kept successor follows: consume the run's trailing separator.
            shape.beats[first].1.start..shape.beats[last + 1].1.start
        } else if first > 0 {
            // The run ends the list: consume the leading separator instead.
            shape.beats[first - 1].1.end..shape.beats[last].1.end
        } else {
            // The run is the whole list.
            shape.beats[first].1.start..shape.beats[last].1.end
        };
        removals.push(range);
        run_start = run_end + 1;
    }
    let edit_start = removals[0].start;
    let mut delta = 0isize;
    for range in removals.into_iter().rev() {
        delta -= range.len() as isize;
        new_text.replace_range(range, "");
    }
    ToggleEdit {
        new_text,
        turned_on: false,
        beat,
        edit_start,
        delta,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(text: &str) -> SequenceShape {
        analyze(text).expect("text should be sequenceable")
    }

    fn beats(text: &str) -> Vec<f32> {
        shape(text).beats.iter().map(|(b, _)| *b).collect()
    }

    #[test]
    fn analyze_accepts_on_beats_with_literal_list() {
        let text = "on_beats(0.5 * b | unseq(), [1, 2.5, 4.75])";
        let shape = shape(text);
        assert_eq!(beats(text), vec![1.0, 2.5, 4.75]);
        assert_eq!(&text[shape.list_range.clone()], "[1, 2.5, 4.75]");
        assert_eq!(&text[shape.beats[1].1.clone()], "2.5");
    }

    #[test]
    fn analyze_rejects_piped_or_wrapped_body() {
        assert!(analyze("on_beats(b, [1]) | mark(1)").is_none());
        assert!(analyze("1 + on_beats(b, [1])").is_none());
    }

    #[test]
    fn analyze_rejects_non_literal_beats() {
        assert!(analyze("on_beats(b, [1, x])").is_none());
        assert!(analyze("on_beats(b, [1, 2 + 1])").is_none());
        assert!(analyze("on_beats(b, bs)").is_none());
    }

    #[test]
    fn analyze_rejects_other_functions() {
        assert!(analyze("off_beats(b, [1])").is_none());
        assert!(analyze("b | fin(1)").is_none());
        assert!(analyze("(").is_none());
    }

    #[test]
    fn sixteenth_arithmetic_round_trips() {
        assert_eq!(sixteenth_beat(0), 1.0);
        assert_eq!(sixteenth_beat(6), 2.5);
        assert_eq!(sixteenth_beat(15), 4.75);
        assert_eq!(sixteenth_for_beat(1.0), 0);
        assert_eq!(sixteenth_for_beat(2.5), 6);
        assert_eq!(sixteenth_for_beat(2.6), 6);
        // Beat 8 lives in the second measure.
        assert_eq!(sixteenth_for_beat(8.0), 28);
        // Clamped clear of the anchor sentinel.
        assert_eq!(sixteenth_for_beat(1e9), u8::MAX - 1);
    }

    #[test]
    fn toggle_inserts_at_sorted_position() {
        let text = "on_beats(b, [1, 3])";
        let edit = toggle_sixteenth_text(text, &shape(text), 6);
        assert_eq!(edit.new_text, "on_beats(b, [1, 2.5, 3])");
        assert!(edit.turned_on);
        assert_eq!(edit.beat, 2.5);
        assert_eq!(edit.delta, 5);
    }

    #[test]
    fn toggle_appends_after_last() {
        let text = "on_beats(b, [1, 3])";
        let edit = toggle_sixteenth_text(text, &shape(text), 12);
        assert_eq!(edit.new_text, "on_beats(b, [1, 3, 4])");
    }

    #[test]
    fn toggle_into_empty_list() {
        let text = "on_beats(b, [])";
        let edit = toggle_sixteenth_text(text, &shape(text), 0);
        assert_eq!(edit.new_text, "on_beats(b, [1])");
    }

    #[test]
    fn toggle_removes_single_matching_beat() {
        let text = "on_beats(b, [1, 2.5, 3])";
        let edit = toggle_sixteenth_text(text, &shape(text), 6);
        assert_eq!(edit.new_text, "on_beats(b, [1, 3])");
        assert!(!edit.turned_on);
        assert_eq!(edit.delta, -5);
    }

    #[test]
    fn toggle_removes_first_and_last_beats_cleanly() {
        let text = "on_beats(b, [1, 3])";
        let first = toggle_sixteenth_text(text, &shape(text), 0);
        assert_eq!(first.new_text, "on_beats(b, [3])");
        let last = toggle_sixteenth_text(text, &shape(text), 8);
        assert_eq!(last.new_text, "on_beats(b, [1])");
        let text = "on_beats(b, [1])";
        let only = toggle_sixteenth_text(text, &shape(text), 0);
        assert_eq!(only.new_text, "on_beats(b, [])");
    }

    #[test]
    fn toggle_removes_every_beat_in_sixteenth_window() {
        // 2, 2.1, and 2.2 all live in sixteenth 4's window [2, 2.25).
        let text = "on_beats(b, [1, 2, 2.1, 2.2, 3])";
        let edit = toggle_sixteenth_text(text, &shape(text), 4);
        assert_eq!(edit.new_text, "on_beats(b, [1, 3])");
    }

    #[test]
    fn toggle_removes_a_whole_trailing_run() {
        let text = "on_beats(b, [1, 2, 2.1])";
        let edit = toggle_sixteenth_text(text, &shape(text), 4);
        assert_eq!(edit.new_text, "on_beats(b, [1])");
    }

    #[test]
    fn toggle_preserves_out_of_measure_beats() {
        let text = "on_beats(b, [1, 5.5])";
        let edit = toggle_sixteenth_text(text, &shape(text), 0);
        assert_eq!(edit.new_text, "on_beats(b, [5.5])");
    }

    #[test]
    fn beat_formatting_is_minimal() {
        let text = "on_beats(b, [])";
        assert_eq!(
            toggle_sixteenth_text(text, &shape(text), 8).new_text,
            "on_beats(b, [3])"
        );
        assert_eq!(
            toggle_sixteenth_text(text, &shape(text), 15).new_text,
            "on_beats(b, [4.75])"
        );
    }

    #[test]
    fn sixteenth_state_reports_off_grid_dim() {
        let beats = [1.0, 2.1];
        assert_eq!(sixteenth_state(&beats, 0), SixteenthState::OnGrid);
        assert_eq!(sixteenth_state(&beats, 4), SixteenthState::OffGridOnly);
        assert_eq!(sixteenth_state(&beats, 8), SixteenthState::Empty);
    }
}
