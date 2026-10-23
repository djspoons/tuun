//! Reading the trace marks off an evaluated waveform: the segments it sounds
//! in, and the program text each mark lights while one of those segments
//! sounds.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use crate::expr::{Expr, SourceExpr, Span};
use crate::generator::Generator;
use crate::optimizer;
use crate::trace::{self, CarriesTrace, Step, Trace, Via};
use crate::waveform::Waveform;

/// The smallest finite range of samples around one or more source waveforms
/// (`phase`, `noise` or `fixed`), or just the start of those samples, measured
/// from the waveform's start.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Segment {
    pub offset: usize,
    /// `None` for sources outside any finite node: they sound as long as the
    /// waveform does.
    pub duration: Option<usize>,
}

/// A range of program text and the segments during which it is current.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Mark {
    /// A byte range in the program's text.
    pub text_range: Range<usize>,
    /// Indices into the waveform's [segments](WaveformMarks::segments).
    pub segments: Range<usize>,
}

/// A waveform's marks: when it sounds and which program text is current then.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct WaveformMarks {
    pub segments: Vec<Segment>,
    pub marks: Vec<Mark>,
}

/// Returns the trace of the evaluation result `result` as a waveform: for a
/// `Seq` whose waveform part has a span or via of its own, the part's; for
/// any other value, its own.
pub fn result_trace<M: CarriesTrace<S>, S: Clone>(result: &SourceExpr<M, S>) -> Arc<Trace<S>> {
    let own = Arc::new(Trace::of(result));
    match &result.expr {
        Expr::Seq { waveform, .. } if waveform.span.is_some() || waveform.via.is_some() => {
            Arc::new(Trace::of(waveform))
        }
        _ => own,
    }
}

/// Returns `waveform` without its trace marks, and its marks read off it at
/// `sample_rate`.
///
/// `trace` is the waveform's own trace (see [`result_trace`]): it marks the
/// root, so the code that made the waveform lights. `instrument`, for a keys
/// note, is the instrument's trace, lit during any segment that nothing else
/// lights.
pub fn walk<M, S>(
    waveform: Waveform<M>,
    trace: &Arc<Trace<S>>,
    instrument: Option<&Trace<S>>,
    sample_rate: u32,
) -> (Waveform<M>, WaveformMarks)
where
    M: CarriesTrace<S> + Clone + PartialEq,
{
    let root = match M::from_trace(trace) {
        Some(id) if trace.reaches_program && trace::trace_marking() => Waveform::Marked {
            id,
            waveform: Box::new(waveform),
        },
        _ => waveform,
    };
    let mut walk = Walk {
        generator: Generator::new(sample_rate),
        segments: Vec::new(),
        marks: Vec::new(),
        sources: 0,
        unclaimed: 0,
        unscoped: 0,
    };
    let out = walk.walk(root, Some(0), false);
    if walk.unclaimed > 0 {
        walk.segments.push(Segment {
            offset: 0,
            duration: None,
        });
    }
    let all_segments = 0..walk.segments.len();
    let mut lit = Lit::default();
    for mark in &walk.marks {
        let segments = mark.scope.clone().unwrap_or(all_segments.clone());
        for text_range in lights::<M, S>(&mark.trace, mark.folded) {
            lit.add(Mark {
                text_range,
                segments: segments.clone(),
            });
        }
    }
    let instrument_ranges = instrument.map(place::<M, S>).unwrap_or_default();
    for k in 0..walk.segments.len() {
        if !lit.marks.iter().any(|m| m.segments.contains(&k)) {
            for text_range in &instrument_ranges {
                lit.add(Mark {
                    text_range: text_range.clone(),
                    segments: k..k + 1,
                });
            }
        }
    }
    let marks = WaveformMarks {
        segments: walk.segments,
        marks: lit.marks,
    };
    (out, marks)
}

/// The marks lit so far, without duplicates.
#[derive(Default)]
struct Lit {
    marks: Vec<Mark>,
    seen: HashSet<Mark>,
}

impl Lit {
    fn add(&mut self, mark: Mark) {
        if self.seen.insert(mark.clone()) {
            self.marks.push(mark);
        }
    }
}

/// A trace mark as the walk found it.
struct WalkMark<S> {
    trace: Arc<Trace<S>>,
    /// Whether no trace mark sits beneath it, so its value's contributors
    /// were consumed during evaluation.
    folded: bool,
    /// The segments of the smallest finite node, at or above it, that holds a
    /// source waveform; `None` until one is found, which means the whole
    /// waveform.
    scope: Option<Range<usize>>,
}

/// The walk's state.
struct Walk<S> {
    generator: Generator<'static>,
    segments: Vec<Segment>,
    marks: Vec<WalkMark<S>>,
    /// Source waveforms seen so far.
    sources: usize,
    /// Source waveforms not yet inside a segment.
    unclaimed: usize,
    /// Marks with no scope yet.
    unscoped: usize,
}

impl<S> Walk<S> {
    /// Returns `w` without trace marks, recording its segments and marks.
    ///
    /// `offset` is when `w` starts, in samples from the start of the whole waveform, or
    /// `None` if that isn't known. Inside an `opaque` waveform (a `Reset`'s),
    /// no node is finite and offsets don't advance.
    fn walk<M>(&mut self, w: Waveform<M>, offset: Option<usize>, opaque: bool) -> Waveform<M>
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        use Waveform::*;
        let (m0, s0, sources0, unclaimed0, unscoped0) = (
            self.marks.len(),
            self.segments.len(),
            self.sources,
            self.unclaimed,
            self.unscoped,
        );
        let out = match w {
            Marked { id, waveform } => match id.as_trace().cloned() {
                Some(trace) => {
                    let inner = *self.child(waveform, offset, opaque);
                    self.marks.push(WalkMark {
                        trace,
                        folded: self.marks.len() == m0,
                        scope: None,
                    });
                    self.unscoped += 1;
                    inner
                }
                None => Marked {
                    id,
                    waveform: self.child(waveform, offset, opaque),
                },
            },
            // Lengths are dark: they only time a segment.
            Fin {
                length,
                waveform,
                state,
            } => Fin {
                length: Box::new(self.measurable(trace::strip_trace_marks(*length))),
                waveform: Box::new(self.walk(*waveform, offset, opaque)),
                state,
            },
            Append(a, b, state) => {
                let a = self.walk(*a, offset, opaque);
                let b_offset = match opaque {
                    true => offset,
                    false => offset
                        .zip(self.finite_length(&a))
                        .map(|(offset, length)| offset + length),
                };
                Append(
                    Box::new(a),
                    Box::new(self.walk(*b, b_offset, opaque)),
                    state,
                )
            }
            // Resets aren't tracked: the inner waveform's sources belong to
            // the segment around the whole `Reset`.
            Reset {
                trigger,
                waveform,
                state,
            } => Reset {
                trigger: self.child(trigger, offset, opaque),
                waveform: Box::new(self.walk(*waveform, offset, true)),
                state,
            },
            Phase {
                frequency,
                offset: phase_offset,
                state,
            } => {
                let w = Phase {
                    frequency: self.child(frequency, offset, opaque),
                    offset: self.child(phase_offset, offset, opaque),
                    state,
                };
                self.add_source();
                w
            }
            Noise => {
                self.add_source();
                Noise
            }
            Fixed(samples, state) => {
                self.add_source();
                Fixed(samples, state)
            }
            leaf @ (Const(_) | Time(_)) => leaf,
            Filter {
                waveform,
                feed_forward,
                feedback,
                state,
            } => Filter {
                waveform: self.child(waveform, offset, opaque),
                feed_forward: feed_forward
                    .into_iter()
                    .map(|w| self.walk(w, offset, opaque))
                    .collect(),
                feedback: feedback
                    .into_iter()
                    .map(|w| self.walk(w, offset, opaque))
                    .collect(),
                state,
            },
            BinaryOp(op, a, b) => BinaryOp(
                op,
                self.child(a, offset, opaque),
                self.child(b, offset, opaque),
            ),
            UnaryOp(op, a) => UnaryOp(op, self.child(a, offset, opaque)),
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => Alt {
                trigger: self.child(trigger, offset, opaque),
                positive_waveform: self.child(positive_waveform, offset, opaque),
                negative_waveform: self.child(negative_waveform, offset, opaque),
            },
            Captured {
                file_stem,
                waveform,
            } => Captured {
                file_stem,
                waveform: self.child(waveform, offset, opaque),
            },
        };
        // A node's length is only read when it could claim a source or scope
        // a mark beneath it.
        let claims = self.unclaimed > unclaimed0;
        let scopes = self.sources > sources0 && self.unscoped > unscoped0;
        if !opaque
            && (claims || scopes)
            && let Some(length) = self.finite_length(&out)
        {
            if claims {
                if let Some(offset) = offset {
                    self.segments.push(Segment {
                        offset,
                        duration: Some(length),
                    });
                }
                self.unclaimed = unclaimed0;
            }
            if scopes {
                let scope = s0..self.segments.len();
                for mark in &mut self.marks[m0..] {
                    if mark.scope.is_none() {
                        mark.scope = Some(scope.clone());
                        self.unscoped -= 1;
                    }
                }
            }
        }
        out
    }

    /// Returns the child `w` walked as [`Walk::walk`] does, in the same box.
    fn child<M>(
        &mut self,
        mut w: Box<Waveform<M>>,
        offset: Option<usize>,
        opaque: bool,
    ) -> Box<Waveform<M>>
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        let inner = std::mem::replace(&mut *w, Waveform::Noise);
        *w = self.walk(inner, offset, opaque);
        w
    }

    fn add_source(&mut self) {
        self.sources += 1;
        self.unclaimed += 1;
    }

    /// Returns the length of `w` in samples when it is known and finite.
    fn finite_length<M>(&self, w: &Waveform<M>) -> Option<usize> {
        self.generator
            .known_length(w)
            .filter(|&length| length != usize::MAX)
    }

    /// Returns the `Fin` length `length` in a form whose cutoff the generator
    /// can read, if optimizing it gives one.
    fn measurable<M: Clone + PartialEq>(&self, length: Waveform<M>) -> Waveform<M> {
        match self.generator.fin_cutoff(&length) {
            Some(_) => length,
            None => optimizer::optimize(length),
        }
    }
}

/// Returns the range of `span` if it is program text.
fn program_range<M: CarriesTrace<S>, S>(span: &Span<S>) -> Option<Range<usize>> {
    M::in_program(span).then(|| span.range.clone())
}

/// Returns the steps of `via`, newest first.
fn steps<S>(via: &Option<Arc<Via<S>>>) -> impl Iterator<Item = &Step<S>> {
    std::iter::successors(via.as_deref(), |v| v.prev.as_deref()).map(|v| &v.step)
}

/// Returns the oldest step of `via`.
fn oldest<S>(via: &Option<Arc<Via<S>>>) -> Option<&Step<S>> {
    steps(via).last()
}

/// Returns the program text a value with `trace` lights: where it was written,
/// and every use site and call it went through, newest first.
///
/// Where a value was written is its span, unless a built-in made it. A new
/// value's span is the built-in's call, which never lights: where it was
/// written is the place of the head of a built-in that computed it, and nothing
/// for one that assembled it.
fn place<M: CarriesTrace<S>, S>(trace: &Trace<S>) -> Vec<Range<usize>> {
    if !trace.reaches_program {
        return vec![];
    }
    let mut out = match oldest(&trace.via) {
        Some(Step::Computed { head, .. }) => place::<M, S>(head),
        Some(Step::Assembled) => vec![],
        _ => trace
            .span
            .as_ref()
            .and_then(program_range::<M, S>)
            .into_iter()
            .collect(),
    };
    // A built-in's step is only ever the oldest, so the rest are use sites and
    // calls.
    out.extend(steps(&trace.via).filter_map(|step| match step {
        Step::UseSite(span) | Step::Call(span) => program_range::<M, S>(span),
        Step::Computed { .. } | Step::Assembled => None,
    }));
    out
}

/// Returns the parents of `trace`'s oldest step when a built-in computed it,
/// wherever its head was written.
fn lit_parents<S>(trace: &Trace<S>) -> &[Arc<Trace<S>>] {
    match oldest(&trace.via) {
        Some(Step::Computed { parents, .. }) => parents,
        _ => &[],
    }
}

/// Returns the program text a mark with `trace` lights. A `folded` mark, one
/// with no trace mark beneath it, also lights the arguments its value consumed,
/// through built-ins that computed it (see [`lit_parents`]).
fn lights<M: CarriesTrace<S>, S>(trace: &Arc<Trace<S>>, folded: bool) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = place::<M, S>(trace);
    if folded {
        // The parents graph is shared and can be deep, so this walks it with
        // a stack and visits each trace once.
        let mut stack: Vec<&Arc<Trace<S>>> = lit_parents(trace).iter().collect();
        let mut seen = HashSet::new();
        while let Some(parent) = stack.pop() {
            if seen.insert(Arc::as_ptr(parent)) && parent.reaches_program {
                out.extend(place::<M, S>(parent));
                stack.extend(lit_parents(parent));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::fmt;

    use super::*;
    use crate::waveform::BinaryOperator;

    /// Which text a test span indexes.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Text {
        Program,
        Library,
    }

    /// A mark id that carries a trace, or is some other kind of mark.
    #[derive(Clone, Debug)]
    enum Id {
        Traced(Arc<Trace<Text>>),
        Other,
    }

    impl PartialEq for Id {
        fn eq(&self, other: &Self) -> bool {
            match (self, other) {
                (Id::Traced(a), Id::Traced(b)) => Arc::ptr_eq(a, b),
                (Id::Other, Id::Other) => true,
                _ => false,
            }
        }
    }

    impl fmt::Display for Id {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{:?}", self)
        }
    }

    impl CarriesTrace<Text> for Id {
        fn from_trace(trace: &Arc<Trace<Text>>) -> Option<Self> {
            Some(Id::Traced(trace.clone()))
        }
        fn as_trace(&self) -> Option<&Arc<Trace<Text>>> {
            match self {
                Id::Traced(trace) => Some(trace),
                Id::Other => None,
            }
        }
        fn in_program(span: &Span<Text>) -> bool {
            span.source == Text::Program
        }
    }

    type W = Waveform<Id>;

    /// Samples per second, so one second is 10 samples.
    const RATE: u32 = 10;

    fn span(range: Range<usize>) -> Span<Text> {
        Span::new(Text::Program, range)
    }

    fn library() -> Span<Text> {
        Span::new(Text::Library, 0..1)
    }

    fn written(range: Range<usize>) -> Arc<Trace<Text>> {
        Arc::new(Trace::new::<Id>(Some(span(range)), None))
    }

    fn trace(span: Option<Span<Text>>, steps: Vec<Step<Text>>) -> Arc<Trace<Text>> {
        let via = steps
            .into_iter()
            .fold(None, |prev, step| Some(Via::push::<Id>(step, prev)));
        Arc::new(Trace::new::<Id>(span, via))
    }

    fn mark(trace: &Arc<Trace<Text>>, w: W) -> W {
        Waveform::Marked {
            id: Id::Traced(trace.clone()),
            waveform: Box::new(w),
        }
    }

    /// A `Fin` length of `seconds`.
    fn seconds(seconds: f32) -> Box<W> {
        Box::new(Waveform::BinaryOp(
            BinaryOperator::Subtract,
            Box::new(Waveform::Time(())),
            Box::new(Waveform::Const(seconds)),
        ))
    }

    /// A source waveform lasting `seconds`.
    fn note(seconds_long: f32) -> W {
        Waveform::Fin {
            length: seconds(seconds_long),
            waveform: Box::new(Waveform::Noise),
            state: (),
        }
    }

    fn op(a: W, b: W) -> W {
        Waveform::BinaryOp(BinaryOperator::Multiply, Box::new(a), Box::new(b))
    }

    fn append(a: W, b: W) -> W {
        Waveform::Append(Box::new(a), Box::new(b), ())
    }

    /// A trace that lights nothing.
    fn dark() -> Arc<Trace<Text>> {
        trace(Some(library()), vec![])
    }

    fn walked(w: W, trace: &Arc<Trace<Text>>) -> WaveformMarks {
        walk(w, trace, None, RATE).1
    }

    fn segment(offset: usize, duration: usize) -> Segment {
        Segment {
            offset,
            duration: Some(duration),
        }
    }

    #[test]
    fn appended_parts_are_offset_by_what_comes_before() {
        let marks = walked(append(note(1.0), append(note(0.5), note(2.0))), &dark());
        assert_eq!(
            marks.segments,
            [segment(0, 10), segment(10, 5), segment(15, 20)]
        );
    }

    #[test]
    fn several_sources_in_one_finite_node_are_one_segment() {
        let w = Waveform::Fin {
            length: seconds(1.0),
            waveform: Box::new(op(Waveform::Noise, Waveform::Noise)),
            state: (),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
    }

    #[test]
    fn sources_in_no_finite_node_sound_for_the_whole_waveform() {
        let marks = walked(Waveform::Noise, &dark());
        assert_eq!(
            marks.segments,
            [Segment {
                offset: 0,
                duration: None
            }]
        );
    }

    #[test]
    fn a_reset_inner_waveform_belongs_to_the_segment_around_the_reset() {
        let w = Waveform::Fin {
            length: seconds(3.0),
            waveform: Box::new(Waveform::Reset {
                trigger: Box::new(Waveform::Time(())),
                waveform: Box::new(append(note(1.0), note(1.0))),
                state: (),
            }),
            state: (),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 30)]);
    }

    #[test]
    fn a_mark_is_scoped_to_the_smallest_finite_node_holding_a_source() {
        // `scale` sits beside two notes, so it is current during both.
        let scale = written(0..1);
        let first = written(2..3);
        let w = op(
            append(mark(&first, note(1.0)), note(1.0)),
            mark(&scale, Waveform::Const(0.5)),
        );
        let marks = walked(w, &dark());
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 2..3,
                    segments: 0..1
                },
                Mark {
                    text_range: 0..1,
                    segments: 0..2
                }
            ]
        );
    }

    #[test]
    fn walking_strips_trace_marks_and_keeps_others() {
        let other = Waveform::Marked {
            id: Id::Other,
            waveform: Box::new(Waveform::Noise),
        };
        let w = mark(&written(0..1), other.clone());
        let (out, _) = walk(w, &written(1..2), None, RATE);
        assert_eq!(out, other);
    }

    #[test]
    fn marks_in_fin_lengths_are_dark_but_lengths_are_still_measured() {
        let marked_second = || {
            Box::new(Waveform::BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Waveform::Time(())),
                Box::new(mark(&written(0..1), Waveform::Const(1.0))),
            ))
        };
        let w = Waveform::Fin {
            length: marked_second(),
            waveform: Box::new(Waveform::Noise),
            state: (),
        };
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 10)]);
        assert!(marks.marks.is_empty());
        // A length the generator can only read once optimized.
        let w = Waveform::Fin {
            length: Box::new(Waveform::BinaryOp(
                BinaryOperator::Add,
                marked_second(),
                Box::new(Waveform::Const(0.0)),
            )),
            waveform: Box::new(Waveform::Noise),
            state: (),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
    }

    #[test]
    fn the_waveform_trace_lights_the_whole_waveform() {
        let marks = walked(append(note(1.0), note(1.0)), &written(4..5));
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 4..5,
                segments: 0..2
            }]
        );
    }

    #[test]
    fn a_built_value_lights_its_head_not_its_span() {
        let head = written(3..4);
        let built = trace(
            Some(span(0..9)),
            vec![Step::Computed {
                head,
                parents: vec![],
            }],
        );
        assert_eq!(place::<Id, Text>(&built), vec![Range { start: 3, end: 4 }]);
    }

    #[test]
    fn a_library_value_lights_each_program_text_it_went_through() {
        let used = trace(
            Some(library()),
            vec![
                Step::UseSite(library()),
                Step::Call(span(5..8)),
                Step::UseSite(span(1..2)),
            ],
        );
        assert_eq!(place::<Id, Text>(&used), [1..2, 5..8]);
    }

    #[test]
    fn a_folded_constant_lights_its_consumed_arguments() {
        let computed = |head, parents| trace(None, vec![Step::Computed { head, parents }]);
        // `(1 + 2)`, then consumed by another computation.
        let sum = computed(written(2..3), vec![written(1..2), written(3..4)]);
        let folded = computed(written(5..6), vec![sum.clone()]);
        let w = mark(&folded, Waveform::Const(3.0));
        let ranges: HashSet<Range<usize>> = walked(op(w, note(1.0)), &dark())
            .marks
            .into_iter()
            .map(|m| m.text_range)
            .collect();
        assert_eq!(ranges, HashSet::from([5..6, 2..3, 1..2, 3..4]));
        // Not folded: something marked sits beneath.
        let w = mark(&folded, mark(&written(7..8), Waveform::Const(3.0)));
        let ranges: Vec<Range<usize>> = walked(op(w, note(1.0)), &dark())
            .marks
            .into_iter()
            .map(|m| m.text_range)
            .collect();
        assert_eq!(ranges, [7..8, 5..6]);
    }

    #[test]
    fn the_instrument_lights_segments_nothing_else_lights() {
        let lit = written(0..1);
        let w = append(mark(&lit, note(1.0)), note(1.0));
        let (_, marks) = walk(w, &dark(), Some(&written(6..9)), RATE);
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 0..1,
                    segments: 0..1
                },
                Mark {
                    text_range: 6..9,
                    segments: 1..2
                }
            ]
        );
    }

    #[test]
    fn equal_marks_are_kept_once() {
        let w = mark(&written(0..1), mark(&written(0..1), note(1.0)));
        assert_eq!(walked(w, &dark()).marks.len(), 1);
    }
}
