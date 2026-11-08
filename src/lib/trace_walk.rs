//! Reading the trace marks off an evaluated waveform: the segments it sounds
//! in, and the program text each mark lights while one of those segments
//! sounds.

use std::collections::{HashMap, HashSet};
use std::ops::Range;
use std::sync::Arc;

use crate::expr::{Expr, SourceExpr, Span};
use crate::generator::Generator;
use crate::optimizer;
use crate::trace::{self, CarriesTrace, Step, Trace, Via};
use crate::waveform::{ShareId, Waveform};

/// The smallest finite range of samples around one or more source waveforms
/// (`phase`, `noise`, `fixed`, or a `Ref` to a shared waveform holding one), or
/// just the start of those samples, measured from the waveform's start.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Segment {
    pub offset: usize,
    /// `None` for sources that sound until the waveform ends: those outside
    /// any finite node, and those in a `Fin` of unknown length that nothing
    /// outlasts.
    pub duration: Option<usize>,
}

impl Segment {
    /// Returns the part of the segment from `start` on, with no duration when
    /// the segment ends by `start`.
    fn after(&self, start: usize) -> Segment {
        let offset = self.offset.max(start);
        Segment {
            offset,
            duration: self
                .duration
                .map(|duration| (self.offset + duration).saturating_sub(offset)),
        }
    }
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
    let mut walk = Walk::new(sample_rate);
    let out = walk.walk(root, Some(0), false, Bound::Whole);
    if let Some(offset) = walk.first_unclaimed(0) {
        walk.segments.push(Segment {
            offset,
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
    /// source waveform or that its parent may outlast; `None` until one is
    /// found, which means the whole waveform.
    scope: Option<Range<usize>>,
}

/// What the walk holds from a shared waveform for each `Ref` to it.
struct Held<S> {
    /// The segments of the shared waveform's marks, measured from the start
    /// of the whole waveform.
    segments: Vec<Segment>,
    marks: Vec<HeldMark<S>>,
    /// Whether a source waveform is in the shared waveform.
    sounds: bool,
    /// The sources in the shared waveform that it doesn't put in a segment:
    /// when each starts, as in [`Walk::unclaimed`].
    unclaimed: Vec<Option<usize>>,
}

/// A trace mark in a shared waveform.
struct HeldMark<S> {
    trace: Arc<Trace<S>>,
    folded: bool,
    /// For a mark the shared waveform scopes itself, the segments it is
    /// scoped to, as indices into [`Held::segments`]; `None` for one it
    /// doesn't scope.
    scope: Option<Range<usize>>,
}

/// The walk's state.
struct Walk<S> {
    generator: Generator<'static>,
    segments: Vec<Segment>,
    marks: Vec<WalkMark<S>>,
    /// Source waveforms seen so far.
    sources: usize,
    /// Source waveforms not yet inside a segment: when each starts, in samples
    /// from the start of the whole waveform, or `None` if that isn't known.
    unclaimed: Vec<Option<usize>>,
    /// Marks with no scope yet.
    unscoped: usize,
    /// The `Shared` nodes whose body holds the node being walked, outermost
    /// first, each with its id and whether its shared waveform is infinite.
    shared: Vec<(ShareId, bool)>,
    /// For each of those `Shared` nodes, what is held from its shared waveform.
    held: Vec<Held<S>>,
}

/// What a node's end bounds.
#[derive(Clone, Copy, PartialEq)]
enum Bound {
    /// The whole waveform ends no later than the node does.
    Whole,
    /// The node's parent ends no later than it does, but the waveform may
    /// outlast it.
    Parent,
    /// The node's parent may outlast it.
    Outlasted,
}

impl Bound {
    /// Returns the bound of a child that its parent, bounded by `self`, ends
    /// no later than.
    fn inner(self) -> Bound {
        match self {
            Bound::Whole => Bound::Whole,
            Bound::Parent | Bound::Outlasted => Bound::Parent,
        }
    }
}

impl<S> Walk<S> {
    /// Returns a walk that has recorded nothing yet.
    fn new(sample_rate: u32) -> Self {
        Walk {
            generator: Generator::new(sample_rate),
            segments: Vec::new(),
            marks: Vec::new(),
            sources: 0,
            unclaimed: Vec::new(),
            unscoped: 0,
            shared: Vec::new(),
            held: Vec::new(),
        }
    }

    /// Returns `w` without trace marks, recording its segments and marks.
    ///
    /// `offset` is when `w` starts, in samples from the start of the whole
    /// waveform, or `None` if that isn't known. Inside an `opaque` waveform (a
    /// `Reset`'s), no node is finite and offsets don't advance. `bound` is what
    /// `w`'s end bounds.
    fn walk<M>(
        &mut self,
        w: Waveform<M>,
        offset: Option<usize>,
        opaque: bool,
        bound: Bound,
    ) -> Waveform<M>
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        use Waveform::*;
        let (m0, s0, sources0, unclaimed0, unscoped0) = (
            self.marks.len(),
            self.segments.len(),
            self.sources,
            self.unclaimed.len(),
            self.unscoped,
        );
        let within = bound.inner();
        let out = match w {
            Marked { id, waveform } => match id.as_trace().cloned() {
                Some(trace) => {
                    let inner = *self.child(waveform, offset, opaque, within);
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
                    waveform: self.child(waveform, offset, opaque, within),
                },
            },
            // Lengths are dark: they only time a segment.
            Fin {
                length,
                waveform,
                state,
            } => Fin {
                length: Box::new(self.measurable(trace::strip_trace_marks(*length))),
                waveform: Box::new(self.walk(*waveform, offset, opaque, within)),
                state,
            },
            // After a first part of unknown length, the second part's offset
            // is unknown too, so its sources get no segment and stay dark.
            Append(a, b, state) => {
                let a = self.walk(*a, offset, opaque, Bound::Outlasted);
                let b_offset = match opaque {
                    true => offset,
                    false => offset
                        .zip(self.finite_length(&a))
                        .map(|(offset, length)| offset + length),
                };
                Append(
                    Box::new(a),
                    Box::new(self.walk(*b, b_offset, opaque, within)),
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
                trigger: self.child(trigger, offset, opaque, within),
                waveform: Box::new(self.walk(*waveform, offset, true, Bound::Outlasted)),
                state,
            },
            Phase {
                frequency,
                offset: phase_offset,
                state,
            } => {
                let w = Phase {
                    frequency: self.child(frequency, offset, opaque, within),
                    offset: self.child(phase_offset, offset, opaque, within),
                    state,
                };
                self.add_source(offset);
                w
            }
            Noise => {
                self.add_source(offset);
                Noise
            }
            Fixed(samples, state) => {
                self.add_source(offset);
                Fixed(samples, state)
            }
            leaf @ (Const(_) | Time(_)) => leaf,
            Filter {
                waveform,
                feed_forward,
                feedback,
                state,
            } => Filter {
                waveform: self.child(waveform, offset, opaque, within),
                feed_forward: feed_forward
                    .into_iter()
                    .map(|w| self.walk(w, offset, opaque, Bound::Outlasted))
                    .collect(),
                feedback: feedback
                    .into_iter()
                    .map(|w| self.walk(w, offset, opaque, Bound::Outlasted))
                    .collect(),
                state,
            },
            // An operator that stops with its shorter operand ends no later
            // than either of them.
            BinaryOp(op, a, b) => {
                let bound = match op.extends_to_longer() {
                    true => Bound::Outlasted,
                    false => within,
                };
                BinaryOp(
                    op,
                    self.child(a, offset, opaque, bound),
                    self.child(b, offset, opaque, bound),
                )
            }
            UnaryOp(op, a) => UnaryOp(op, self.child(a, offset, opaque, within)),
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => Alt {
                trigger: self.child(trigger, offset, opaque, within),
                positive_waveform: self.child(positive_waveform, offset, opaque, Bound::Outlasted),
                negative_waveform: self.child(negative_waveform, offset, opaque, Bound::Outlasted),
            },
            Captured {
                file_stem,
                waveform,
            } => Captured {
                file_stem,
                waveform: self.child(waveform, offset, opaque, within),
            },
            // The shared waveform is generated whether or not it is heard, so
            // its marks are held rather than lit here. A `Ref` is heard in its
            // place: it lights the held marks, along with the mark above it
            // (see `builtins::share`), and is a source when its shared waveform
            // holds one, as a copy would be.
            Shared { id, waveform, body } => {
                let infinite = self.known_length(&waveform) == Some(usize::MAX);
                let (waveform, held) = self.walk_shared(*waveform, offset, opaque);
                self.shared.push((id, infinite));
                self.held.push(held);
                let body = self.child(body, offset, opaque, within);
                self.shared.pop();
                self.held.pop();
                Shared {
                    id,
                    waveform: Box::new(waveform),
                    body,
                }
            }
            Ref(id) => {
                if let Some(i) = self.shared.iter().rposition(|(shared, _)| *shared == id) {
                    self.add_held(i, offset.filter(|_| !opaque));
                    if self.held[i].sounds {
                        self.sources += 1;
                        // Inside a `Reset`, or after an unknown offset, the
                        // `Ref` is one source where it starts.
                        match offset.filter(|_| !opaque) {
                            Some(start) => self.unclaimed.extend(
                                self.held[i]
                                    .unclaimed
                                    .iter()
                                    .map(|source| source.map(|source| source.max(start))),
                            ),
                            None => self.unclaimed.push(offset),
                        }
                    }
                }
                Ref(id)
            }
        };
        // A node's length is only read when it could claim a source or scope a
        // mark beneath it. A `Fin` whose length can't be measured still ends,
        // so it claims and scopes too. Its segment runs from its offset until
        // the waveform ends when nothing outlasts it; otherwise it has none,
        // and what it holds stays dark. A node its parent may outlast scopes
        // marks even with no source beneath: above it, they'd light while it
        // no longer plays. With no source beneath, a node of unknown length
        // may end at any time, so it scopes them as such a `Fin` does.
        let claims = self.unclaimed.len() > unclaimed0;
        let scopes =
            self.unscoped > unscoped0 && (self.sources > sources0 || bound == Bound::Outlasted);
        let length = match !opaque && (claims || scopes) {
            true => self.finite_length(&out).map(Some).or_else(|| {
                let ends = matches!(out, Fin { .. })
                    || (self.sources == sources0 && self.known_length(&out).is_none());
                ends.then_some(None)
            }),
            false => None,
        };
        if let Some(length) = length {
            if claims {
                // The segment starts when the first source it claims does,
                // and ends with this node. A segment cut before it starts
                // keeps its place, with no duration left.
                if let Some(offset) = offset
                    && (length.is_some() || bound == Bound::Whole)
                    && let Some(start) = self.first_unclaimed(unclaimed0)
                {
                    let start = start.max(offset);
                    self.segments.push(Segment {
                        offset: start,
                        duration: length.map(|length| (offset + length).saturating_sub(start)),
                    });
                }
                self.unclaimed.truncate(unclaimed0);
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
        // Nothing beneath a node sounds after the node ends, so a node that can
        // stop before its children cuts their segments at its end. A segment
        // cut before it starts keeps its place, with no duration left. A `Fin`
        // ends by its cutoff even when what it holds has no known length.
        if !opaque
            && self.segments.len() > s0
            && stops_children(&out)
            && let Some(offset) = offset
            && let Some(length) = self.finite_length(&out).or(match &out {
                Fin { length, .. } => self.generator.fin_cutoff(length),
                _ => None,
            })
        {
            let end = offset + length;
            for segment in &mut self.segments[s0..] {
                let left = end.saturating_sub(segment.offset);
                segment.duration = Some(segment.duration.map_or(left, |d| d.min(left)));
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
        bound: Bound,
    ) -> Box<Waveform<M>>
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        let inner = std::mem::replace(&mut *w, Waveform::Noise);
        *w = self.walk(inner, offset, opaque, bound);
        w
    }

    /// Returns the shared waveform `w` without trace marks, and what to hold
    /// from it for each `Ref` to it.
    ///
    /// `offset` and `opaque` are those of the `Shared` node. Nothing in `w` is
    /// recorded: only a `Ref` to it is heard.
    fn walk_shared<M>(
        &mut self,
        w: Waveform<M>,
        offset: Option<usize>,
        opaque: bool,
    ) -> (Waveform<M>, Held<S>)
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        let segments = std::mem::take(&mut self.segments);
        let marks = std::mem::take(&mut self.marks);
        let unclaimed = std::mem::take(&mut self.unclaimed);
        let counts = (self.sources, self.unscoped);
        // Walked at the `Shared` node's offset, `w`'s segments are measured on
        // the clock of the whole waveform, as a `Ref`'s are. A `Ref` to `w`
        // ends no later than `w` does.
        let w = self.walk(w, offset, opaque, Bound::Parent);
        let w_segments = std::mem::replace(&mut self.segments, segments);
        // Each `Ref` in `w` adds the marks and segments held for it, the same
        // ones when the `Ref`s start together: kept, they would double at each
        // level of nested sharing. A mark lights while any of its segments
        // sounds, so it keeps each segment once.
        let mut seen = HashSet::new();
        let marks: Vec<_> = std::mem::replace(&mut self.marks, marks)
            .into_iter()
            .map(|mark| {
                let segments = mark.scope.map(|scope| {
                    let mut distinct = HashSet::new();
                    w_segments[scope]
                        .iter()
                        .filter(|segment| distinct.insert(*segment))
                        .cloned()
                        .collect::<Vec<_>>()
                });
                (mark.trace, mark.folded, segments)
            })
            .filter(|(trace, folded, segments)| {
                seen.insert((Arc::as_ptr(trace), *folded, segments.clone()))
            })
            .collect();
        let (segments, runs) = lay_out(marks.iter().filter_map(|(_, _, s)| s.as_ref()));
        let held = Held {
            segments,
            marks: marks
                .into_iter()
                .map(|(trace, folded, segments)| HeldMark {
                    trace,
                    folded,
                    scope: segments.map(|segments| runs[&segments].clone()),
                })
                .collect(),
            sounds: self.sources > counts.0,
            unclaimed: std::mem::replace(&mut self.unclaimed, unclaimed),
        };
        (self.sources, self.unscoped) = counts;
        (w, held)
    }

    /// Records the marks held for the `i`th enclosing `Shared` node, as a `Ref`
    /// that starts at `start` hears them.
    ///
    /// A mark the shared waveform scopes itself lights during its segments from
    /// `start` on, or stays dark when `start` is `None`. The rest are scoped
    /// where the `Ref` is, as a copy's marks would be.
    fn add_held(&mut self, i: usize, start: Option<usize>) {
        let held = &self.held[i];
        let first = self.segments.len();
        if let Some(start) = start {
            self.segments
                .extend(held.segments.iter().map(|segment| segment.after(start)));
        }
        for mark in &held.marks {
            let scope = match (&mark.scope, start) {
                (None, _) => {
                    self.unscoped += 1;
                    None
                }
                (Some(scope), Some(_)) => Some(first + scope.start..first + scope.end),
                (Some(_), None) => Some(first..first),
            };
            self.marks.push(WalkMark {
                trace: mark.trace.clone(),
                folded: mark.folded,
                scope,
            });
        }
    }

    /// Records a source waveform that starts at `offset`.
    fn add_source(&mut self, offset: Option<usize>) {
        self.sources += 1;
        self.unclaimed.push(offset);
    }

    /// Returns the earliest start of the unclaimed sources past the first
    /// `from`, if there are any.
    ///
    /// A source whose start isn't known is taken to start at 0, as early as
    /// any could.
    fn first_unclaimed(&self, from: usize) -> Option<usize> {
        // TODO a source whose start isn't known may start later than the
        // segment does, so it could light too early.
        self.unclaimed[from..]
            .iter()
            .map(|start| start.unwrap_or(0))
            .min()
    }

    /// Returns the length of `w` in samples when it is known and finite.
    fn finite_length<M>(&mut self, w: &Waveform<M>) -> Option<usize> {
        self.known_length(w).filter(|&length| length != usize::MAX)
    }

    /// Returns the length of `w` in samples when it is known, where `w` sits
    /// at the node being walked.
    fn known_length<M>(&mut self, w: &Waveform<M>) -> Option<usize> {
        self.generator.known_length_within(w, &mut self.shared)
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

/// Returns segments holding each of `runs` contiguously, and where each run
/// is in them.
///
/// A run that is part of a longer one shares its segments.
fn lay_out<'a>(
    runs: impl Iterator<Item = &'a Vec<Segment>>,
) -> (Vec<Segment>, HashMap<Vec<Segment>, Range<usize>>) {
    let mut runs: Vec<_> = runs.collect();
    runs.sort_by_key(|run| std::cmp::Reverse(run.len()));
    let mut segments: Vec<Segment> = Vec::new();
    let mut placed = HashMap::new();
    for run in runs {
        if placed.contains_key(run) {
            continue;
        }
        let at = match run.is_empty() {
            true => Some(0),
            false => segments
                .windows(run.len())
                .position(|w| w == run.as_slice()),
        };
        let at = at.unwrap_or_else(|| {
            segments.extend(run.iter().cloned());
            segments.len() - run.len()
        });
        placed.insert(run.clone(), at..at + run.len());
    }
    (segments, placed)
}

/// Returns whether `w` can end before one of its children does.
fn stops_children<M>(w: &Waveform<M>) -> bool {
    match w {
        Waveform::Fin { .. }
        | Waveform::Alt { .. }
        | Waveform::Phase { .. }
        | Waveform::Filter { .. } => true,
        Waveform::BinaryOp(op, _, _) => !op.extends_to_longer(),
        _ => false,
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
    use crate::waveform::{BinaryOperator, ShareId};

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
    fn a_source_after_a_rest_starts_its_segment_when_it_starts() {
        // With no finite node around it, it sounds from then on.
        let delayed = || append(rest(1.0), Waveform::Noise);
        assert_eq!(
            walked(delayed(), &dark()).segments,
            [Segment {
                offset: 10,
                duration: None
            }]
        );
        // A finite node around it ends the segment, but doesn't start it.
        assert_eq!(
            walked(op(delayed(), rest(2.0)), &dark()).segments,
            [segment(10, 10)]
        );
        // Sources that start at different times share a segment from the first.
        let w = Waveform::BinaryOp(
            BinaryOperator::Merge,
            Box::new(append(rest(1.5), Waveform::Noise)),
            Box::new(delayed()),
        );
        assert_eq!(
            walked(op(w, rest(2.0)), &dark()).segments,
            [segment(10, 10)]
        );
    }

    /// A marked source in a `Fin` whose length reads a mark the generator can't
    /// see through, as a slider's does.
    fn unmeasurable_note(trace: &Arc<Trace<Text>>) -> W {
        Waveform::Fin {
            length: Box::new(Waveform::BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Waveform::Time(())),
                Box::new(Waveform::Marked {
                    id: Id::Other,
                    waveform: Box::new(Waveform::Const(1.0)),
                }),
            )),
            waveform: Box::new(mark(trace, Waveform::Noise)),
            state: (),
        }
    }

    #[test]
    fn a_rest_of_unmeasurable_length_beside_a_note_stays_dark() {
        // As in `0 * rest & note`, with the rest's length read from a slider
        // and `0 * rest` at 0..1: the rest may end at any time, so it doesn't
        // light while the note plays.
        let rest = Waveform::Fin {
            length: Box::new(Waveform::BinaryOp(
                BinaryOperator::Subtract,
                Box::new(Waveform::Time(())),
                Box::new(Waveform::Marked {
                    id: Id::Other,
                    waveform: Box::new(Waveform::Const(1.0)),
                }),
            )),
            waveform: Box::new(Waveform::Const(0.0)),
            state: (),
        };
        let w = Waveform::BinaryOp(
            BinaryOperator::Merge,
            Box::new(mark(&written(0..1), op(Waveform::Const(0.0), rest))),
            Box::new(note(2.0)),
        );
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 20)]);
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 0..1,
                segments: 0..0
            }]
        );
    }

    #[test]
    fn a_fin_of_unmeasurable_length_that_ends_the_waveform_sounds_until_it_ends() {
        let w = append(note(1.0), unmeasurable_note(&written(0..1)));
        let marks = walked(w, &dark());
        let open = Segment {
            offset: 10,
            duration: None,
        };
        assert_eq!(marks.segments, [segment(0, 10), open.clone()]);
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 0..1,
                segments: 1..2
            }]
        );
        // `+` stops with its shorter operand, so the note can't outlast it.
        let w = Waveform::BinaryOp(
            BinaryOperator::Add,
            Box::new(unmeasurable_note(&written(0..1))),
            Box::new(note(1.0)),
        );
        let open = Segment {
            offset: 0,
            duration: None,
        };
        assert_eq!(walked(w, &dark()).segments, [open, segment(0, 10)]);
    }

    #[test]
    fn a_fin_of_unmeasurable_length_that_may_be_outlasted_stays_dark() {
        // `&` runs as long as its longer operand.
        let w = Waveform::BinaryOp(
            BinaryOperator::Merge,
            Box::new(unmeasurable_note(&written(0..1))),
            Box::new(note(1.0)),
        );
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 10)]);
        assert!(marks.marks.iter().all(|m| m.segments.is_empty()));
        // Whatever comes after a part outlasts it.
        let w = append(unmeasurable_note(&written(0..1)), note(1.0));
        let marks = walked(w, &dark());
        assert!(marks.marks.iter().all(|m| m.segments.is_empty()));
    }

    #[test]
    fn a_segment_is_cut_where_a_node_above_it_ends() {
        // `*` stops with its shorter operand.
        let w = op(note(2.0), rest(1.0));
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
        // An `Alt` stops with its trigger.
        let w = Waveform::Alt {
            trigger: Box::new(rest(1.0)),
            positive_waveform: Box::new(note(2.0)),
            negative_waveform: Box::new(Waveform::Const(0.0)),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
        // A part that starts after the `Fin` around it ends never sounds.
        let w = Waveform::Fin {
            length: seconds(1.5),
            waveform: Box::new(append(note(1.0), note(1.0))),
            state: (),
        };
        assert_eq!(
            walked(w, &dark()).segments,
            [segment(0, 10), segment(10, 5)]
        );
        let w = Waveform::Fin {
            length: seconds(0.5),
            waveform: Box::new(append(note(1.0), note(1.0))),
            state: (),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 5), segment(10, 0)]);
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

    /// A constant lasting `seconds`.
    fn rest(seconds_long: f32) -> W {
        Waveform::Fin {
            length: seconds(seconds_long),
            waveform: Box::new(Waveform::Const(0.0)),
            state: (),
        }
    }

    #[test]
    fn a_mark_with_no_source_stops_at_a_node_its_parent_may_outlast() {
        // A rest beside a note, or before one, has stopped by the time the
        // note plays.
        let merged = Waveform::BinaryOp(
            BinaryOperator::Merge,
            Box::new(mark(&written(0..1), rest(1.0))),
            Box::new(note(2.0)),
        );
        let appended = append(mark(&written(0..1), rest(1.0)), note(1.0));
        for w in [merged, appended] {
            let marks = walked(w, &dark());
            assert_eq!(marks.segments.len(), 1);
            assert!(marks.marks.iter().all(|m| m.segments.is_empty()));
        }
        // A constant that multiplies a note is current while the note plays.
        let w = op(note(1.0), mark(&written(0..1), rest(1.0)));
        assert_eq!(
            walked(w, &dark()).marks,
            [Mark {
                text_range: 0..1,
                segments: 0..1
            }]
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
    fn a_shared_waveform_lights_where_a_ref_to_it_is_heard() {
        // As in `share(noise, fn(x) => rest \ x)`, with `noise` at 0..1 and
        // `x` at 5..6: the shared waveform is generated from the start, but
        // only heard after the rest.
        let used = trace(Some(span(0..1)), vec![Step::UseSite(span(5..6))]);
        let heard = Waveform::Fin {
            length: seconds(1.0),
            waveform: Box::new(mark(&used, Waveform::Ref(ShareId(0)))),
            state: (),
        };
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(mark(&written(0..1), Waveform::Noise)),
            body: Box::new(append(rest(1.0), heard)),
        };
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(10, 10)]);
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 0..1,
                    segments: 0..1
                },
                Mark {
                    text_range: 5..6,
                    segments: 0..1
                }
            ]
        );
    }

    #[test]
    fn marks_a_shared_waveform_scopes_itself_light_during_their_own_segments() {
        // As in `share(a \\ b & noise, fn(x) => x)` with `a`, `b`, the whole
        // shared waveform and `x` at 0..1, 2..3, 0..4 and 5..6: `a` and `b`
        // each light while they play, the rest while `x` is heard.
        let shared = mark(
            &written(0..4),
            Waveform::BinaryOp(
                BinaryOperator::Merge,
                Box::new(append(
                    mark(&written(0..1), note(1.0)),
                    mark(&written(2..3), note(1.0)),
                )),
                Box::new(Waveform::Noise),
            ),
        );
        let used = trace(None, vec![Step::UseSite(span(5..6))]);
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(shared),
            body: Box::new(mark(&used, Waveform::Ref(ShareId(0)))),
        };
        let marks = walked(w, &dark());
        assert_eq!(
            marks.segments,
            [
                segment(0, 10),
                segment(10, 10),
                Segment {
                    offset: 0,
                    duration: None
                }
            ]
        );
        let mark = |text_range, segments| Mark {
            text_range,
            segments,
        };
        assert_eq!(
            marks.marks,
            [
                mark(0..1, 0..1),
                mark(2..3, 1..2),
                mark(0..4, 0..3),
                mark(5..6, 0..3)
            ]
        );
    }

    #[test]
    fn a_fin_cuts_what_it_holds_at_its_cutoff_when_its_length_is_unknown() {
        // As in `share(a, fn(x) => x | fin(time - 1))`, with `a` a 2 s note at
        // 0..1: a `Ref` to a finite waveform has no known length, but the
        // `Fin` still ends at 1 s.
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(mark(&written(0..1), note(2.0))),
            body: Box::new(Waveform::Fin {
                length: seconds(1.0),
                waveform: Box::new(Waveform::Ref(ShareId(0))),
                state: (),
            }),
        };
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 10)]);
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 0..1,
                segments: 0..1
            }]
        );
    }

    #[test]
    fn a_ref_lights_the_parts_of_a_shared_waveform_from_when_it_starts() {
        // As in `share(a \\ b, fn(x) => rest \\ x)`, with `a` and `b` at 0..1
        // and 2..3: `x` starts as `b` does, so `a` has no time left.
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(append(
                mark(&written(0..1), note(1.0)),
                mark(&written(2..3), note(1.0)),
            )),
            body: Box::new(append(rest(1.0), Waveform::Ref(ShareId(0)))),
        };
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(10, 0), segment(10, 10)]);
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 0..1,
                    segments: 0..1
                },
                Mark {
                    text_range: 2..3,
                    segments: 1..2
                }
            ]
        );
    }

    #[test]
    fn a_ref_to_a_shared_waveform_with_no_source_is_not_a_source() {
        // As in `share(a, fn(x) => x * rest)` with `a` at 0..1: like a copy of
        // `a`, it gets no segment of its own...
        let shared = |body| Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(mark(&written(0..1), Waveform::Time(()))),
            body: Box::new(body),
        };
        let copy = walked(
            op(mark(&written(0..1), Waveform::Time(())), rest(1.0)),
            &dark(),
        );
        let marks = walked(shared(op(Waveform::Ref(ShareId(0)), rest(1.0))), &dark());
        assert_eq!(marks.segments, []);
        assert_eq!(marks, copy);
        // ...but lights with a note it controls.
        let marks = walked(shared(op(Waveform::Ref(ShareId(0)), note(1.0))), &dark());
        assert_eq!(marks.segments, [segment(0, 10)]);
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 0..1,
                segments: 0..1
            }]
        );
    }

    #[test]
    fn marks_held_through_several_refs_are_held_once() {
        // As in `share(share(a, fn(y) => y & y), fn(x) => x)`, with `a` at 0..1:
        // both uses of `y` add `a`, but `x` holds it once.
        let w = Waveform::Shared {
            id: ShareId(1),
            waveform: Box::new(mark(&written(0..1), Waveform::Noise)),
            body: Box::new(Waveform::BinaryOp(
                BinaryOperator::Merge,
                Box::new(Waveform::Ref(ShareId(1))),
                Box::new(Waveform::Ref(ShareId(1))),
            )),
        };
        let (_, held) = Walk::new(RATE).walk_shared(w, Some(0), false);
        assert_eq!(held.marks.len(), 1);
    }

    #[test]
    fn equal_marks_are_kept_once() {
        let w = mark(&written(0..1), mark(&written(0..1), note(1.0)));
        assert_eq!(walked(w, &dark()).marks.len(), 1);
    }
}
