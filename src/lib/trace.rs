//! Provenance for evaluated values: where each value was written, the places
//! it went afterwards, and the trace marks that carry that record into the
//! waveforms built-ins build.

use std::cell::Cell;
use std::sync::Arc;

use crate::expr::{Expr, SourceExpr, Span};
use crate::waveform::Waveform;

/// A value's provenance: the payload of a trace mark id, and the head and
/// parents of a [`Step::Built`].
#[derive(Debug)]
pub struct Trace<S> {
    /// Where the value was written.
    pub span: Option<Span<S>>,
    /// Where the value went afterwards, newest step first; the path of this
    /// trace.
    pub via: Option<Arc<Via<S>>>,
    /// The container's trace, for a part tagged with a trace of its own.
    pub outer: Option<Arc<Trace<S>>>,
    /// Whether `span`, `via` or `outer` reaches program text.
    pub reaches_program: bool,
}

impl<S> Trace<S> {
    /// Builds a trace, deciding whether it reaches program text by `M`'s
    /// [`CarriesTrace::in_program`].
    pub fn new<M: CarriesTrace<S>>(
        span: Option<Span<S>>,
        via: Option<Arc<Via<S>>>,
        outer: Option<Arc<Trace<S>>>,
    ) -> Trace<S> {
        let reaches_program = span.as_ref().is_some_and(M::in_program)
            || via.as_ref().is_some_and(|v| v.reaches_program)
            || outer.as_ref().is_some_and(|o| o.reaches_program);
        Trace {
            span,
            via,
            outer,
            reaches_program,
        }
    }

    /// Returns the trace of `expr`: its span and via, with no container.
    pub fn of<M: CarriesTrace<S>>(expr: &SourceExpr<M, S>) -> Trace<S>
    where
        S: Clone,
    {
        Trace::new::<M>(expr.span.clone(), expr.via.clone(), None)
    }
}

/// One step of a `via` chain.
#[derive(Debug)]
pub struct Via<S> {
    pub step: Step<S>,
    /// The steps before this one.
    pub prev: Option<Arc<Via<S>>>,
    /// Whether this step or any before it reaches program text.
    pub reaches_program: bool,
}

impl<S> Via<S> {
    /// Returns the chain `prev` with `step` added as its newest step,
    /// deciding whether it reaches program text by `M`'s [`CarriesTrace::in_program`].
    pub fn push<M: CarriesTrace<S>>(step: Step<S>, prev: Option<Arc<Via<S>>>) -> Arc<Via<S>> {
        let here = match &step {
            Step::UseSite(span) | Step::Call(span) => M::in_program(span),
            Step::Built { head, parents } => {
                head.as_ref().is_some_and(|h| h.reaches_program)
                    || parents.iter().any(|p| p.reaches_program)
            }
        };
        let reaches_program = here || prev.as_ref().is_some_and(|p| p.reaches_program);
        Arc::new(Via {
            step,
            prev,
            reaches_program,
        })
    }
}

/// What happened to a value at one step of its `via` chain.
#[derive(Debug)]
pub enum Step<S> {
    /// A substituted variable or a module projection, at the place it was
    /// written.
    UseSite(Span<S>),
    /// A user function's result, at the place the call was written.
    Call(Span<S>),
    /// A built-in's own result: one that isn't an argument handed back.
    Built {
        /// Where the built-in was named, or `None` when the built-in
        /// [arranges](crate::expr::BuiltInKind::Arranges) its arguments.
        head: Option<Arc<Trace<S>>>,
        /// The arguments' traces.
        parents: Vec<Arc<Trace<S>>>,
    },
}

/// Mark id types that may carry a trace.
///
/// Every method has a default for a mark id type that carries none: it makes
/// no trace mark ids, reads none, and treats no text as program text.
pub trait CarriesTrace<S>: Sized {
    /// Returns a mark id carrying `trace`, or `None` if this type carries
    /// none.
    fn from_trace(_trace: &Arc<Trace<S>>) -> Option<Self> {
        None
    }

    /// Returns the trace this mark id carries, if it is of the trace kind.
    fn as_trace(&self) -> Option<&Arc<Trace<S>>> {
        None
    }

    /// Returns whether `span` is in program text: the text of the program
    /// being evaluated (as opposed to a library function).
    fn in_program(_span: &Span<S>) -> bool {
        false
    }
}

impl<S> CarriesTrace<S> for u32 {}
impl<S> CarriesTrace<S> for () {}

thread_local! {
    static TRACE_MARKING: Cell<bool> = const { Cell::new(true) };
}

/// Sets whether evaluation on the current thread makes trace marks.
///
/// It does until this turns it off. Off, evaluation still records each
/// value's `via`, but builds the same waveforms as a mark id type that
/// carries no trace.
pub fn set_trace_marking(on: bool) {
    TRACE_MARKING.with(|marking| marking.set(on));
}

/// Returns whether evaluation on the current thread makes trace marks (see
/// [`set_trace_marking`]).
pub fn trace_marking() -> bool {
    TRACE_MARKING.with(|marking| marking.get())
}

/// Returns the trace of the marked node at `w`'s root, if it is a trace mark.
pub fn root_trace<M: CarriesTrace<S>, S>(w: &Waveform<M>) -> Option<&Arc<Trace<S>>> {
    match w {
        Waveform::Marked { id, .. } => id.as_trace(),
        _ => None,
    }
}

/// Returns `w` looking through any trace marks at its root.
pub fn peel<M: CarriesTrace<S>, S>(mut w: &Waveform<M>) -> &Waveform<M> {
    while let Waveform::Marked { id, waveform } = w
        && id.as_trace().is_some()
    {
        w = waveform;
    }
    w
}

/// Returns `w` without the trace marks at its root.
pub fn strip_root_trace_marks<M: CarriesTrace<S>, S>(mut w: Waveform<M>) -> Waveform<M> {
    loop {
        match w {
            Waveform::Marked { id, waveform } if id.as_trace().is_some() => w = *waveform,
            w => return w,
        }
    }
}

/// Returns `w` without any of its trace marks; every other node is kept.
pub fn strip_trace_marks<M: CarriesTrace<S>, S>(w: Waveform<M>) -> Waveform<M> {
    use Waveform::*;
    let strip = |w: Box<Waveform<M>>| Box::new(strip_trace_marks(*w));
    match w {
        leaf @ (Const(_) | Time(_) | Noise | Fixed(_, _)) => leaf,
        Fin {
            length,
            waveform,
            state,
        } => Fin {
            length: strip(length),
            waveform: strip(waveform),
            state,
        },
        Append(a, b, state) => Append(strip(a), strip(b), state),
        Phase {
            frequency,
            offset,
            state,
        } => Phase {
            frequency: strip(frequency),
            offset: strip(offset),
            state,
        },
        Filter {
            waveform,
            feed_forward,
            feedback,
            state,
        } => Filter {
            waveform: strip(waveform),
            feed_forward: feed_forward.into_iter().map(strip_trace_marks).collect(),
            feedback: feedback.into_iter().map(strip_trace_marks).collect(),
            state,
        },
        BinaryOp(op, a, b) => BinaryOp(op, strip(a), strip(b)),
        UnaryOp(op, a) => UnaryOp(op, strip(a)),
        Reset {
            trigger,
            waveform,
            state,
        } => Reset {
            trigger: strip(trigger),
            waveform: strip(waveform),
            state,
        },
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => Alt {
            trigger: strip(trigger),
            positive_waveform: strip(positive_waveform),
            negative_waveform: strip(negative_waveform),
        },
        Marked { id, waveform } if id.as_trace().is_some() => strip_trace_marks(*waveform),
        Marked { id, waveform } => Marked {
            id,
            waveform: strip(waveform),
        },
        Captured {
            file_stem,
            waveform,
        } => Captured {
            file_stem,
            waveform: strip(waveform),
        },
    }
}

/// Returns `expr` without the trace marks in its waveforms, at any depth
/// through `Seq`, `List` and `Tuple`. Functions are left as they are.
pub fn strip_expr_trace_marks<M: CarriesTrace<S>, S>(expr: Expr<M, S>) -> Expr<M, S> {
    let strip = |part: SourceExpr<M, S>| SourceExpr {
        expr: strip_expr_trace_marks(part.expr),
        ..part
    };
    match expr {
        Expr::Waveform(w) => Expr::Waveform(strip_trace_marks(w)),
        Expr::Seq { offset, waveform } => Expr::Seq {
            offset: Box::new(strip(*offset)),
            waveform: Box::new(strip(*waveform)),
        },
        Expr::List(exprs) => Expr::List(exprs.into_iter().map(strip).collect()),
        Expr::Tuple(exprs) => Expr::Tuple(exprs.into_iter().map(strip).collect()),
        expr => expr,
    }
}
