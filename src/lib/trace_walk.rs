//! Reading the trace marks off an evaluated waveform: the segments it sounds
//! in, and the program text each mark lights while one of those segments
//! sounds.

use std::collections::HashSet;
use std::ops::Range;
use std::sync::Arc;

use crate::expr::{Expr, SourceExpr, Span};
use crate::generator::{Generator, SharedExtent};
use crate::optimizer;
use crate::trace::{self, CarriesTrace, Step, Trace, Via};
use crate::waveform::Waveform;

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
    let mut root = match M::from_trace(trace) {
        Some(id) if trace.reaches_program && trace::trace_marking() => Waveform::Marked {
            id,
            waveform: Box::new(waveform),
        },
        _ => waveform,
    };
    let walked = Walk::new(sample_rate).walk(&mut root, Some(0), false, Bound::Whole);
    let mut segments = walked.segments;
    if let Some(offset) = walked.unclaimed {
        segments.push(Segment {
            offset,
            duration: None,
        });
    }
    let all_segments = 0..segments.len();
    let mut lit = Lit::default();
    for mark in &walked.marks {
        let segments = mark.scope.clone().unwrap_or(all_segments.clone());
        for text_range in lights::<M, S>(&mark.trace, mark.folded) {
            lit.add(Mark {
                text_range,
                segments: segments.clone(),
            });
        }
    }
    let instrument_ranges = instrument.map(place::<M, S>).unwrap_or_default();
    for k in 0..segments.len() {
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
        segments,
        marks: lit.marks,
    };
    (trace::strip_trace_marks(root), marks)
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
    /// The segments of the smallest node, at or above it, that scopes it, as
    /// indices into the segments of the [`Walked`] that holds the mark; `None`
    /// until one does, which means the whole waveform.
    scope: Option<Range<usize>>,
}

/// What the walk found in a waveform.
struct Walked<S> {
    /// The segments of the sources in it.
    segments: Vec<Segment>,
    /// The trace marks in it, each after those beneath it.
    marks: Vec<WalkMark<S>>,
    /// Whether a source waveform is in it.
    sounds: bool,
    /// When the earliest source in it that is not yet inside a segment starts,
    /// in samples from the start of the whole waveform, if there is one. A
    /// source whose start isn't known counts as starting at 0, as early as any
    /// could.
    unclaimed: Option<usize>,
}

impl<S> Default for Walked<S> {
    fn default() -> Self {
        Walked {
            segments: Vec::new(),
            marks: Vec::new(),
            sounds: false,
            unclaimed: None,
        }
    }
}

impl<S> Walked<S> {
    /// Adds what the walk found in a waveform after this one.
    fn add(&mut self, other: Walked<S>) {
        let first = self.segments.len();
        self.segments.extend(other.segments);
        self.marks.extend(other.marks.into_iter().map(|mark| {
            WalkMark {
                scope: mark
                    .scope
                    .map(|scope| first + scope.start..first + scope.end),
                ..mark
            }
        }));
        self.sounds |= other.sounds;
        self.unclaimed = self.unclaimed.into_iter().chain(other.unclaimed).min();
    }

    /// Adds a source waveform that starts at `start`.
    fn add_source(&mut self, start: Option<usize>) {
        // TODO a source whose start isn't known may start later than the
        // segment that claims it does, so it could light too early.
        self.add(Walked {
            sounds: true,
            unclaimed: Some(start.unwrap_or(0)),
            ..Walked::default()
        });
    }

    /// Returns this, what the walk found in a shared waveform, with duplicate
    /// marks removed.
    ///
    /// Two marks are duplicates when they have the same trace, are both folded
    /// or both not, and are scoped to equal segments.
    fn held(mut self) -> Walked<S> {
        // Duplicates come from `Ref`s in the shared waveform to an enclosing
        // one: those that start together each add the same marks. Kept, they
        // would double at each level of nested sharing.
        let mut seen = HashSet::new();
        let segments = &self.segments;
        self.marks.retain(|mark| {
            let heard = mark.scope.clone().map(|scope| segments[scope].to_vec());
            seen.insert((Arc::as_ptr(&mark.trace), mark.folded, heard))
        });
        self
    }

    /// Returns what a `Ref` that starts at `offset` hears of the shared
    /// waveform held as this.
    ///
    /// What plays before the `Ref` starts is cut. With no `offset`, nothing
    /// held has a segment: the marks the shared waveform scopes itself stay
    /// dark, and the `Ref` is one source where it starts. The rest of the
    /// marks are left unscoped, as a copy's would be.
    fn heard(&self, offset: Option<usize>) -> Walked<S> {
        let marks = self.marks.iter().map(|mark| WalkMark {
            trace: mark.trace.clone(),
            folded: mark.folded,
            scope: mark.scope.clone().map(|scope| match offset {
                Some(_) => scope,
                None => 0..0,
            }),
        });
        let (segments, unclaimed) = match offset {
            Some(start) => (
                self.segments.iter().map(|s| s.after(start)).collect(),
                self.unclaimed.map(|source| source.max(start)),
            ),
            None => (vec![], self.sounds.then_some(0)),
        };
        Walked {
            segments,
            marks: marks.collect(),
            sounds: self.sounds,
            unclaimed,
        }
    }
}

/// The walk's state.
struct Walk<S> {
    generator: Generator<'static>,
    /// The `Shared` nodes whose body holds the node being walked, outermost
    /// first, each with when its shared waveform starts and its length.
    // Kept apart from `held` because the generator measures lengths with
    // these alone, adding its own as it measures through nested `Shared`
    // nodes.
    shared: Vec<SharedExtent>,
    /// For each of those `Shared` nodes, what the walk found in its shared
    /// waveform, which each `Ref` to it hears from (see [`Walked::heard`]).
    held: Vec<Walked<S>>,
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
    /// Returns a walk outside any `Shared` node.
    fn new(sample_rate: u32) -> Self {
        Walk {
            generator: Generator::new(sample_rate),
            shared: Vec::new(),
            held: Vec::new(),
        }
    }

    /// Returns what the walk finds in `w`: its segments and trace marks.
    ///
    /// `offset` is when `w` starts, in samples from the start of the whole
    /// waveform, or `None` if that isn't known. Inside an `opaque` waveform (a
    /// `Reset`'s), no node is finite and offsets don't advance. `bound` is what
    /// `w`'s end bounds.
    ///
    /// Each `Fin` length in `w` is left without trace marks, in a form whose
    /// cutoff the generator can read if optimizing it gives one, so that the
    /// lengths of `w` and its parts can be measured with its other marks in
    /// place.
    fn walk<M>(
        &mut self,
        w: &mut Waveform<M>,
        offset: Option<usize>,
        opaque: bool,
        bound: Bound,
    ) -> Walked<S>
    where
        M: CarriesTrace<S> + Clone + PartialEq,
    {
        // Notes on the implementation of `walk`:
        //
        // 1. `offset` should be passed through to children who start at the
        //    same time as their parent; otherwise use a new offset (as in
        //    `Append`, including `None` if the offset is unknown) or set
        //    `opaque` if the generator may restart the child (as in `Reset`).
        //
        // 2. Each child gets a `bound` that indicates how that child's length
        //    relates to the overall waveform and to its parent (the current
        //    node). This is used to know whether or not that child can safely
        //    pass any marks up to its parent or if it should scope them itself
        //    (though a child with a source beneath it scopes its marks whatever
        //    its bound). Use `within` if the node ends no later than the
        //    child, Bound::Outlasted if the node may play on after it.
        //
        // 3. The results of walking a child are added to the parent's result
        //    unless 1) that child is the `length` of a `Fin` (which are
        //    stripped) or 2) that child is the `waveform` of a `Shared` node
        //    (in which case the results are held until a `Ref` node adds what
        //    is heard).
        //
        // 4. A node that makes possibly-audible samples (currently a Phase,
        //    Noise, or Fixed node) that don't come from a child adds a source
        //    where it starts.
        //
        // 5. When a node ends:
        //    - It claims the sources beneath it that aren't yet in a segment
        //      into one segment, from when the first starts until the node
        //      ends. If that end isn't known, the segment runs until the
        //      waveform ends when the node's bound is `Bound::Whole`; otherwise
        //      there is none and they stay dark.
        //    - When a source is beneath the node or its parent may outlast it,
        //      it scopes the marks beneath it that aren't yet scoped to its
        //      segments.
        //    - It cuts every segment beneath it at its end, since nothing
        //      beneath it sounds after. A segment cut before it starts keeps
        //      its place, with no duration left.
        //
        // 6. A non-`Fin` node doesn't end if it is known to be infinite or its
        //    length isn't known (not even as a bound) and a source beneath it
        //    isn't yet in a segment. In either case, it passes that source and
        //    the marks beneath it to its parent. (`Fin` nodes always end.)
        //
        // 7. Some imprecision is allowed and expected: when highlighting
        //    depends on sample *values* (as in `Alt` or `Reset`), a child
        //    lights whenever it *may* play; an end may be an upper bound, so a
        //    node may still light after it stops; a source whose start isn't
        //    known may stay dark or may light too early.
        use Waveform::*;
        let within = bound.inner();
        let mut here = Walked::default();
        match w {
            Marked { id, waveform } => {
                here = self.walk(waveform, offset, opaque, within);
                if let Some(trace) = id.as_trace() {
                    let folded = here.marks.is_empty();
                    here.marks.push(WalkMark {
                        trace: trace.clone(),
                        folded,
                        scope: None,
                    });
                }
            }
            // Lengths are dark: they only time a segment.
            Fin {
                length, waveform, ..
            } => {
                let marked = std::mem::replace(&mut **length, Noise);
                **length = self.measurable(trace::strip_trace_marks(marked));
                here = self.walk(waveform, offset, opaque, within);
            }
            // After a first part of unknown length, the second part's offset
            // is unknown too, so its sources get no segment of their own: they
            // stay dark unless a node of known end above the `Append` claims
            // them.
            Append(a, b, _) => {
                here = self.walk(a, offset, opaque, Bound::Outlasted);
                let b_offset = match opaque {
                    true => offset,
                    false => offset
                        .zip(self.known_length(a, offset))
                        .filter(|&(_, length)| length != usize::MAX)
                        .map(|(offset, length)| offset + length),
                };
                here.add(self.walk(b, b_offset, opaque, within));
            }
            // Resets aren't tracked: the inner waveform's sources belong to
            // the segment around the whole `Reset`.
            Reset {
                trigger, waveform, ..
            } => {
                here = self.walk(trigger, offset, opaque, within);
                here.add(self.walk(waveform, offset, true, Bound::Outlasted));
            }
            Phase {
                frequency,
                offset: phase_offset,
                ..
            } => {
                here = self.walk(frequency, offset, opaque, within);
                here.add(self.walk(phase_offset, offset, opaque, within));
                here.add_source(offset);
            }
            Noise | Fixed(..) => here.add_source(offset),
            Const(_) | Time(_) => {}
            Filter {
                waveform,
                feed_forward,
                feedback,
                ..
            } => {
                here = self.walk(waveform, offset, opaque, within);
                for w in feed_forward.iter_mut().chain(feedback) {
                    here.add(self.walk(w, offset, opaque, Bound::Outlasted));
                }
            }
            // An operator that stops with its shorter operand ends no later
            // than either of them.
            BinaryOp(op, a, b) => {
                let bound = match op.extends_to_longer() {
                    true => Bound::Outlasted,
                    false => within,
                };
                here = self.walk(a, offset, opaque, bound);
                here.add(self.walk(b, offset, opaque, bound));
            }
            UnaryOp(_, waveform) | Captured { waveform, .. } => {
                here = self.walk(waveform, offset, opaque, within);
            }
            Alt {
                trigger,
                positive_waveform,
                negative_waveform,
            } => {
                here = self.walk(trigger, offset, opaque, within);
                here.add(self.walk(positive_waveform, offset, opaque, Bound::Outlasted));
                here.add(self.walk(negative_waveform, offset, opaque, Bound::Outlasted));
            }
            // The shared waveform is generated whether or not it is heard, so
            // what the walk finds in it is held rather than added here. A
            // `Ref` is heard in its place (see `Walked::heard`), along with
            // the mark above it (see `builtins::share`).
            Shared { id, waveform, body } => {
                // Walked at the `Shared` node's offset, the shared waveform's
                // segments are measured on the clock of the whole waveform,
                // as a `Ref`'s are. A `Ref` to it ends no later than it does.
                let held = self.walk(waveform, offset, opaque, Bound::Parent).held();
                // Inside a `Reset`, offsets don't advance, so where the
                // shared waveform starts isn't known. Its length is measured
                // once the walk has left its `Fin` lengths measurable.
                let extent = SharedExtent {
                    id: *id,
                    start: offset.filter(|_| !opaque),
                    length: self.known_length(waveform, offset),
                };
                self.shared.push(extent);
                self.held.push(held);
                here = self.walk(body, offset, opaque, within);
                self.shared.pop();
                self.held.pop();
            }
            // Inside a `Reset`, the shared waveform isn't reset: a `Ref` hears
            // it on its own clock from where the `Reset` starts.
            Ref(id) => {
                if let Some(i) = self.shared.iter().rposition(|extent| extent.id == *id) {
                    here = self.held[i].heard(offset);
                }
            }
        }
        // A node is measured as what is left once trace marks are stripped.
        self.end(strip_outer_trace_marks(w), offset, opaque, bound, here)
    }

    /// Returns `here`, what the walk found in `w`, as it is once `w` ends.
    ///
    /// `offset`, `opaque` and `bound` are as in [`Walk::walk`].
    fn end<M>(
        &mut self,
        w: &Waveform<M>,
        offset: Option<usize>,
        opaque: bool,
        bound: Bound,
        mut here: Walked<S>,
    ) -> Walked<S> {
        // See note in `walk` on what happens when a node ends. Only a node that
        // can stop before its children can cut a segment it neither claimed nor
        // scoped.
        let claims = here.unclaimed.is_some();
        let scopes = here.marks.iter().any(|mark| mark.scope.is_none())
            && (here.sounds || bound == Bound::Outlasted);
        let cuts = !here.segments.is_empty() && stops_children(w);
        if opaque || !(claims || scopes || cuts) {
            return here;
        }
        let length = self.known_length(w, offset);
        if length == Some(usize::MAX) {
            return here;
        }
        // Without its own length, a node ends at one it won't outlast.
        let length = length.or_else(|| self.length_at_most(w, offset));
        // Without either, it passes what it holds to its parent.
        if length.is_none() && here.unclaimed.is_some() && !matches!(w, Waveform::Fin { .. }) {
            return here;
        }
        let end = offset.zip(length).map(|(offset, length)| offset + length);
        if let Some(start) = here.unclaimed.take()
            && let Some(offset) = offset
            && (end.is_some() || bound == Bound::Whole)
        {
            here.segments.push(Segment {
                offset: start.max(offset),
                duration: None,
            });
        }
        if scopes {
            let scope = 0..here.segments.len();
            for mark in &mut here.marks {
                mark.scope.get_or_insert_with(|| scope.clone());
            }
        }
        if let Some(end) = end {
            for segment in &mut here.segments {
                let left = end.saturating_sub(segment.offset);
                segment.duration = Some(segment.duration.map_or(left, |d| d.min(left)));
            }
        }
        here
    }

    /// Returns the length of `w` in samples when it is known, where `w` sits
    /// at the node being walked and starts at `offset`.
    fn known_length<M>(&mut self, w: &Waveform<M>, offset: Option<usize>) -> Option<usize> {
        self.generator.known_length_at(w, offset, &mut self.shared)
    }

    /// Returns a length that `w`, starting at `offset`, won't outlast, if one
    /// is known though its own length isn't.
    fn length_at_most<M>(&mut self, w: &Waveform<M>, offset: Option<usize>) -> Option<usize> {
        match w {
            Waveform::Fin { length, .. } => self.generator.fin_cutoff(length),
            // An operator that stops with its shorter operand ends no later
            // than either operand whose length is known.
            Waveform::BinaryOp(op, a, b) if !op.extends_to_longer() => [a, b]
                .into_iter()
                .filter_map(|w| self.known_length(w, offset))
                .filter(|&length| length != usize::MAX)
                .min(),
            _ => None,
        }
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

/// Returns `w` without the trace marks around it.
fn strip_outer_trace_marks<M: CarriesTrace<S>, S>(w: &Waveform<M>) -> &Waveform<M> {
    match w {
        Waveform::Marked { id, waveform } if id.as_trace().is_some() => {
            strip_outer_trace_marks(waveform)
        }
        _ => w,
    }
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
        // `+` stops with its shorter operand, so neither can outlast it.
        let w = Waveform::BinaryOp(
            BinaryOperator::Add,
            Box::new(unmeasurable_note(&written(0..1))),
            Box::new(unmeasurable_note(&written(2..3))),
        );
        let open = Segment {
            offset: 0,
            duration: None,
        };
        assert_eq!(walked(w, &dark()).segments, [open.clone(), open]);
    }

    #[test]
    fn an_operator_that_stops_with_its_shorter_operand_ends_by_the_known_one() {
        // As in `(a | fin(time - slider)) + (b | fin(time - 1))`: the sum
        // ends within a second, and so does `a`.
        let w = Waveform::BinaryOp(
            BinaryOperator::Add,
            Box::new(unmeasurable_note(&written(0..1))),
            Box::new(note(1.0)),
        );
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 10), segment(0, 10)]);
        assert_eq!(
            marks.marks,
            [Mark {
                text_range: 0..1,
                segments: 0..1
            }]
        );
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
    fn a_source_after_a_part_of_unmeasurable_length_lights_from_what_claims_it() {
        // As in `append(a | fin(time - slider), b) * Hw`, with `b` endless:
        // `b` starts at an unknown time, and lights from when the product
        // starts.
        let w = op(
            append(
                unmeasurable_note(&written(0..1)),
                mark(&written(2..3), Waveform::Noise),
            ),
            note(2.0),
        );
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 20), segment(0, 20)]);
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 0..1,
                    segments: 0..0
                },
                Mark {
                    text_range: 2..3,
                    segments: 0..2
                },
            ]
        );
        // A `b` that ends where its start isn't known stays dark.
        let w = op(
            append(
                unmeasurable_note(&written(0..1)),
                mark(&written(2..3), note(1.0)),
            ),
            note(2.0),
        );
        assert!(
            walked(w, &dark())
                .marks
                .iter()
                .all(|m| m.segments.is_empty())
        );
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
        // As in `share(append(a, b) & noise, fn(x) => x)` with `a`, `b`, the
        // whole shared waveform and `x` at 0..1, 2..3, 0..4 and 5..6: `a` and
        // `b` each light while they play, the rest while `x` is heard.
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
        // What the `Fin` holds has no known length, but the `Fin` still ends
        // at 1 s.
        let w = Waveform::Fin {
            length: seconds(1.0),
            waveform: Box::new(unmeasurable_note(&written(0..1))),
            state: (),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
    }

    #[test]
    fn a_ref_lights_the_parts_of_a_shared_waveform_from_when_it_starts() {
        // As in `share(append(a, b), fn(x) => append(rest, x))`, with `a` and
        // `b` at 0..1 and 2..3: `x` starts as `b` does, so `a` has no time
        // left.
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
        let mut w = Waveform::Shared {
            id: ShareId(1),
            waveform: Box::new(mark(&written(0..1), Waveform::Noise)),
            body: Box::new(Waveform::BinaryOp(
                BinaryOperator::Merge,
                Box::new(Waveform::Ref(ShareId(1))),
                Box::new(Waveform::Ref(ShareId(1))),
            )),
        };
        let walked = Walk::new(RATE).walk(&mut w, Some(0), false, Bound::Parent);
        assert_eq!(walked.marks.len(), 2);
        assert_eq!(walked.held().marks.len(), 1);
    }

    #[test]
    fn a_mark_around_a_fin_of_unmeasurable_length_is_scoped_to_it() {
        let w = append(note(1.0), mark(&written(2..3), unmeasurable_note(&dark())));
        assert_eq!(
            walked(w, &dark()).marks,
            [Mark {
                text_range: 2..3,
                segments: 1..2
            }]
        );
    }

    #[test]
    fn a_shared_source_of_unknown_start_sounds_no_earlier_than_the_ref() {
        // The noise follows a part of unknown length, so its start isn't
        // known, but it can't be heard before `x` starts reading at 1 s.
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(append(
                unmeasurable_note(&written(0..1)),
                mark(&written(2..3), Waveform::Noise),
            )),
            body: Box::new(append(rest(1.0), Waveform::Ref(ShareId(0)))),
        };
        let marks = walked(w, &dark());
        assert_eq!(
            marks.segments,
            [Segment {
                offset: 10,
                duration: None
            }]
        );
        assert_eq!(
            marks.marks,
            [
                Mark {
                    text_range: 0..1,
                    segments: 0..0
                },
                Mark {
                    text_range: 2..3,
                    segments: 0..1
                }
            ]
        );
    }

    #[test]
    fn a_ref_in_a_reset_lights_the_shared_waveform_on_its_own_clock() {
        // As in `share(append(a, b), fn(x) => reset(time, x) * Hw)`, with `a`
        // and `b` at 0..1 and 2..3: the shared waveform isn't reset, so `a`
        // and `b` each light while they play.
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(append(
                mark(&written(0..1), note(1.0)),
                mark(&written(2..3), note(1.0)),
            )),
            body: Box::new(Waveform::Fin {
                length: seconds(2.0),
                waveform: Box::new(Waveform::Reset {
                    trigger: Box::new(Waveform::Time(())),
                    waveform: Box::new(Waveform::Ref(ShareId(0))),
                    state: (),
                }),
                state: (),
            }),
        };
        let marks = walked(w, &dark());
        assert_eq!(marks.segments, [segment(0, 10), segment(10, 10)]);
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
    fn a_shared_waveform_whose_length_is_marked_is_measured_for_its_refs() {
        // As in `share(a, fn(x) => x * (1 | fin(time - 1)))`, with `a`
        // lasting two seconds, a length that carries a trace mark: `a` is
        // heard for one.
        let w = Waveform::Shared {
            id: ShareId(0),
            waveform: Box::new(Waveform::Fin {
                length: Box::new(Waveform::BinaryOp(
                    BinaryOperator::Subtract,
                    Box::new(Waveform::Time(())),
                    Box::new(mark(&written(0..1), Waveform::Const(2.0))),
                )),
                waveform: Box::new(Waveform::Noise),
                state: (),
            }),
            body: Box::new(op(
                Waveform::Ref(ShareId(0)),
                Waveform::Fin {
                    length: seconds(1.0),
                    waveform: Box::new(Waveform::Const(1.0)),
                    state: (),
                },
            )),
        };
        assert_eq!(walked(w, &dark()).segments, [segment(0, 10)]);
    }

    #[test]
    fn equal_marks_are_kept_once() {
        let w = mark(&written(0..1), mark(&written(0..1), note(1.0)));
        assert_eq!(walked(w, &dark()).marks.len(), 1);
    }
}
