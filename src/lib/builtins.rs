use std::fmt::{Debug, Display};

use crate::eval;
use crate::expr;
use crate::expr::{BuiltInFn, BuiltInKind, Error, Expr, SourceExpr, boxed};
use crate::optimizer;
use crate::trace::{self, CarriesTrace};
use crate::waveform::{BinaryOperator, UnaryOperator, Waveform};
use Expr::{Bool, BuiltIn, List, Seq};

type BuiltinFn<M, S> = fn(Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>;

/// Returns `argument` as a bare constant when it is a constant under trace
/// marks, and unchanged otherwise, for a built-in that consumes it.
fn peel_const<M: CarriesTrace<S>, S>(argument: Expr<M, S>) -> Expr<M, S> {
    match argument.as_const_float() {
        Some(value) => Expr::float(value),
        None => argument,
    }
}

// TODO maybe use Display instead of Debug for errors?
fn binary_op<M, S>(
    mut arguments: Vec<Expr<M, S>>,
    name: String,
    op: BinaryOperator,
) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    Ok({
        fn make_seq<M, S>(
            offset: Box<SourceExpr<M, S>>,
            op: BinaryOperator,
            a: Waveform<M>,
            b: Waveform<M>,
        ) -> Expr<M, S>
        where
            M: Debug,
        {
            Seq {
                offset,
                waveform: boxed(Expr::Waveform(Waveform::BinaryOp(
                    op,
                    Box::new(a),
                    Box::new(b),
                ))),
            }
        }

        if arguments.len() != 2 {
            return Err(Error::internal_here(format!(
                "Expected two arguments for {}, got {:?}",
                name, arguments
            )));
        }
        let (x, y) = (arguments.remove(0), arguments.remove(0));
        // Two constants fold eagerly, so arithmetic on constants stays constant.
        if let (Some(a), Some(b)) = (x.as_const_float(), y.as_const_float()) {
            return Ok(Expr::float(op.apply(a, b)));
        }
        match (x, y) {
            (Expr::Waveform(a), Expr::Waveform(b)) => {
                Expr::Waveform(Waveform::BinaryOp(op, Box::new(a), Box::new(b)))
            }
            (Seq { offset, waveform }, Expr::Waveform(b)) => match waveform.expr {
                Expr::Waveform(a) => make_seq(offset, op, a, b),
                expr => {
                    return Err(Error::internal_here(format!(
                        "Invalid argument to seq in {}: {:?}",
                        name, expr
                    )));
                }
            },
            (Expr::Waveform(a), Seq { offset, waveform }) => match waveform.expr {
                Expr::Waveform(b) => make_seq(offset, op, a, b),
                expr => {
                    return Err(Error::internal_here(format!(
                        "Invalid argument to seq in {}: {:?}",
                        name, expr
                    )));
                }
            },
            (a, b) => {
                return Err(Error::internal_here(format!(
                    "Invalid arguments for {}: {:?} and {:?}",
                    name, a, b
                )));
            }
        }
    })
}

pub fn plus<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    binary_op(arguments, "+".to_string(), BinaryOperator::Add)
}

pub fn minus<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    // Unary minus is multiplication by -1, so it folds and threads seqs the
    // same way.
    if arguments.len() == 1 {
        arguments.insert(0, Expr::float(-1.0));
        return binary_op(arguments, "-".to_string(), BinaryOperator::Multiply);
    }
    binary_op(arguments, "-".to_string(), BinaryOperator::Subtract)
}

pub fn times<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    binary_op(arguments, "*".to_string(), BinaryOperator::Multiply)
}

pub fn divide<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    binary_op(arguments, "/".to_string(), BinaryOperator::Divide)
}

pub fn merge<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    binary_op(arguments, "&".to_string(), BinaryOperator::Merge)
}

// Given waveforms that represent offsets (and assuming offset waveforms are of the form
// `Time ~+ w` or `Const(x)`) return a new waveform which represents the sum of those offsets.
fn add_offsets<M, S>(a: Waveform<M>, b: Waveform<M>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone + Debug + PartialEq + CarriesTrace<S>,
    S: Clone + Debug,
{
    // The roots are read through bare copies; `a` and `b` themselves keep
    // their trace marks, which the caller embeds.
    let (a_bare, b_bare) = (
        trace::strip_trace_marks(a.clone()),
        trace::strip_trace_marks(b.clone()),
    );
    match (
        optimizer::first_root(&a_bare),
        optimizer::first_root(&b_bare),
    ) {
        // The sum's root is the sum of the roots: `Time - (a + b)`.
        (Some(a_root), Some(b_root)) => Ok(Expr::Waveform(Waveform::BinaryOp(
            BinaryOperator::Subtract,
            Box::new(Waveform::Time(())),
            Box::new(Waveform::BinaryOp(
                BinaryOperator::Add,
                Box::new(a_root),
                Box::new(b_root),
            )),
        ))),
        (a_root, b_root) => Err(Error::internal_here(format!(
            "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
            a_root, b_root, a, b
        ))),
    }
}

/// Returns the merge of two arguments where the second argument delayed until
/// after the offset of the first (which therefore must be a seq).
pub fn followed_by<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Display + Clone + PartialEq + CarriesTrace<S>,
    S: Clone + Debug,
{
    Ok({
        if arguments.len() != 2 {
            return Err(Error::internal_here("Expected two arguments to \\"));
        }
        let (a_offset, a) = match arguments.remove(0) {
            Seq { offset, waveform } => {
                match (offset.expr, waveform.expr) {
                    (Expr::Waveform(offset), Expr::Waveform(waveform)) => (offset, waveform),
                    // We know that the arguments are values and Seq-as-a-value always has two waveforms.
                    (err @ Expr::Error { .. }, _) => {
                        return Ok(err);
                    }
                    (_, err @ Expr::Error { .. }) => {
                        return Ok(err);
                    }
                    _ => panic!("Found a non-Waveform element in a Seq value"),
                }
            }
            expr => {
                return Err(Error::internal_here(format!(
                    "Expected seq as first argument to \\, got {}",
                    expr
                )));
            }
        };

        match arguments.remove(0) {
            Expr::Waveform(b) => Expr::Waveform(Waveform::BinaryOp(
                BinaryOperator::Merge,
                Box::new(a),
                Box::new(Waveform::Append(
                    Box::new(Waveform::Fin {
                        length: Box::new(a_offset),
                        waveform: Box::new(Waveform::Const(0.0)),
                        state: (),
                    }),
                    Box::new(b),
                    (),
                )),
            )),
            Seq {
                offset: b_offset,
                waveform,
            } => {
                let (b_offset, b) = match (b_offset.expr, waveform.expr) {
                    (Expr::Waveform(b_offset), Expr::Waveform(b)) => (b_offset, b),
                    (err @ Expr::Error { .. }, _) => {
                        return Ok(err);
                    }
                    (_, err @ Expr::Error { .. }) => {
                        return Ok(err);
                    }
                    _ => panic!("Found a non-Waveform element in a Seq value"),
                };
                let total_offset = add_offsets(a_offset.clone(), b_offset)?;
                Seq {
                    offset: boxed(total_offset),
                    waveform: boxed(Expr::Waveform(Waveform::BinaryOp(
                        BinaryOperator::Merge,
                        Box::new(a),
                        Box::new(Waveform::Append(
                            Box::new(Waveform::Fin {
                                length: Box::new(a_offset),
                                waveform: Box::new(Waveform::Const(0.0)),
                                state: (),
                            }),
                            Box::new(b),
                            (),
                        )),
                    ))),
                }
            }
            expr => {
                return Err(Error::internal_here(format!(
                    "Expected second argument to \\ to be a float, waveform or seq, got {}",
                    expr
                )));
            }
        }
    })
}

pub fn power<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    binary_op(arguments, "pow".to_string(), BinaryOperator::Power)
}

pub fn log<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone + Debug,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [Expr::Waveform(a), Expr::Waveform(b)] => Expr::Waveform(Waveform::BinaryOp(
                BinaryOperator::Log,
                Box::new(a.clone()),
                Box::new(b.clone()),
            )),
            _ => return Err(Error::internal_here("Invalid arguments for log")),
        }
    })
}

pub fn sqrt<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [Expr::Waveform(Waveform::Const(value))] if value >= 0.0 => Expr::float(value.sqrt()),
            [Expr::Waveform(Waveform::Const(value))] => {
                return Err(Error::eval_here(format!(
                    "square root of a negative number: {}",
                    value
                )));
            }
            _ => return Err(Error::internal_here("Invalid argument for sqrt")),
        }
    })
}

/// Returns the result of applying `op` to each sample of a waveform or seq.
///
/// A seq keeps its offset, so `op` applies to its waveform only.
fn unary_op<M, S>(
    mut arguments: Vec<Expr<M, S>>,
    name: &str,
    op: UnaryOperator,
) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    if arguments.len() != 1 {
        return Err(Error::internal_here(format!(
            "Expected one argument for {}, got {:?}",
            name, arguments
        )));
    }
    Ok(match arguments.remove(0) {
        Expr::Waveform(a) => Expr::Waveform(Waveform::UnaryOp(op, Box::new(a))),
        Seq { offset, waveform } => match waveform.expr {
            Expr::Waveform(a) => Seq {
                offset,
                waveform: boxed(Expr::Waveform(Waveform::UnaryOp(op, Box::new(a)))),
            },
            expr => {
                return Err(Error::internal_here(format!(
                    "Invalid argument to seq in {}: {:?}",
                    name, expr
                )));
            }
        },
        expr => {
            return Err(Error::internal_here(format!(
                "Invalid argument for {}: {:?}",
                name, expr
            )));
        }
    })
}

/// Returns a waveform combinator that returns the exponential function of each sample.
pub fn exp<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    unary_op(arguments, "exp", UnaryOperator::Exp)
}

/// Returns the waveform combinator that computes the sine of each sample.
pub fn sin<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    unary_op(arguments, "sin", UnaryOperator::Sin)
}

pub fn phase<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone,
    S: Clone + Debug,
{
    Ok({
        // Like the waveform, Phase, the first argument is frequency in hertz
        match &arguments[..] {
            [Expr::Waveform(freq), Expr::Waveform(offset)] => Expr::Waveform(Waveform::Phase {
                frequency: Box::new(freq.clone()),
                offset: Box::new(offset.clone()),
                state: (),
            }),
            _ => return Err(Error::internal_here("Invalid arguments for phase")),
        }
    })
}

pub fn round<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [Expr::Waveform(Waveform::Const(value))] => Expr::float(value.round()),
            _ => return Err(Error::internal_here("Invalid argument for round")),
        }
    })
}

pub fn equals<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match &arguments[..] {
            [Bool(a), Bool(b)] => Expr::Bool(a == b),
            // TODO could consider a more general form of equality on waveforms but
            // this seems to be enough for now.
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a == b),
            [Expr::String(a), Expr::String(b)] => Expr::Bool(a == b),
            _ => return Err(Error::internal_here("Invalid arguments for ==")),
        }
    })
}

pub fn not_equals<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match &arguments[..] {
            [Bool(a), Bool(b)] => Expr::Bool(a != b),
            // TODO same as for ==, equality on Waveform?
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a != b),
            [Expr::String(a), Expr::String(b)] => Expr::Bool(a != b),
            _ => return Err(Error::internal_here("Invalid arguments for !=")),
        }
    })
}

pub fn less_than<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a < b),
            _ => return Err(Error::internal_here("Invalid arguments for <")),
        }
    })
}

pub fn less_than_equals<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a <= b),
            _ => return Err(Error::internal_here("Invalid arguments for <=")),
        }
    })
}

pub fn greater_than<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a > b),
            _ => return Err(Error::internal_here("Invalid arguments for >")),
        }
    })
}

pub fn greater_than_equals<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    let arguments: Vec<_> = arguments.into_iter().map(peel_const).collect();
    Ok({
        match arguments[..] {
            [
                Expr::Waveform(Waveform::Const(a)),
                Expr::Waveform(Waveform::Const(b)),
            ] => Expr::Bool(a >= b),
            _ => return Err(Error::internal_here("Invalid arguments for >=")),
        }
    })
}

pub fn map<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone + Debug + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [function, List(exprs)] => {
                let mut results: Vec<SourceExpr<M, S>> = Vec::new();
                for expr in exprs {
                    results.push(eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![expr.clone()], // can we avoid this clone?
                    ))?);
                }
                Expr::List(results)
            }
            _ => return Err(Error::internal_here("Invalid arguments for map")),
        }
    })
}

pub fn reduce<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [function, acc, List(exprs)] => {
                let mut acc: SourceExpr<M, S> = SourceExpr::from(acc.clone());
                for expr in exprs {
                    acc = eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![acc, expr.clone()],
                    ))?;
                }
                acc.expr
            }
            _ => return Err(Error::internal_here("Invalid arguments for reduce")),
        }
    })
}

pub fn unfold<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    if let Some(count) = arguments.pop() {
        arguments.push(peel_const(count));
    }
    Ok({
        match &arguments[..] {
            [function, seed, Expr::Waveform(Waveform::Const(n))]
                if *n >= 0.0 && n.fract() == 0.0 =>
            {
                let mut results: Vec<SourceExpr<M, S>> = Vec::new();
                let mut current: SourceExpr<M, S> = SourceExpr::from(seed.clone());
                for _ in 0..(*n as u32) {
                    results.push(current.clone());
                    current = eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![current.clone()],
                    ))?;
                }
                Expr::List(results)
            }
            // A negative or fractional count is a value the lattice cannot rule
            // out of `int`.
            [_, _, Expr::Waveform(Waveform::Const(n))] => {
                return Err(Error::eval_here(format!(
                    "unfold needs a non-negative whole count, got {}",
                    n
                )));
            }
            _ => return Err(Error::internal_here("Invalid arguments for unfold")),
        }
    })
}

// Appends exactly two waveforms end to end. Lists join with `concat`.
pub fn append<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [Expr::Waveform(a), Expr::Waveform(b)] => Expr::Waveform(Waveform::Append(
                Box::new(a.clone()),
                Box::new(b.clone()),
                (),
            )),
            _ => return Err(Error::internal_here("Expected two waveforms for append")),
        }
    })
}

fn concat<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [List(elmts)] => {
                let mut results = Vec::new();
                for elmt in elmts {
                    if let List(exprs) = &elmt.expr {
                        results.extend(exprs.clone());
                    } else {
                        return Err(Error::internal_here(
                            "Expected list of lists as argument for concat",
                        ));
                    }
                }
                Expr::List(results)
            }
            _ => return Err(Error::internal_here("Invalid arguments for concat")),
        }
    })
}

pub fn nth<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    if !arguments.is_empty() {
        let index = peel_const(arguments.remove(0));
        arguments.insert(0, index);
    }
    Ok({
        match &arguments[..] {
            [Expr::Waveform(Waveform::Const(a)), List(b)] if *a >= 0.0 && a.fract() == 0.0 => {
                if let Some(element) = b.get(*a as usize) {
                    element.expr.clone()
                } else {
                    return Err(Error::eval_here(format!("no element with index {}", a)));
                }
            }
            // A negative index is an int like any other, so the lattice cannot
            // rule it out; a fractional or non-numeric one it can.
            [Expr::Waveform(Waveform::Const(a)), List(_)] if a.fract() == 0.0 => {
                return Err(Error::eval_here(format!("no element with index {}", a)));
            }
            _ => return Err(Error::internal_here("Invalid arguments for nth")),
        }
    })
}

pub fn fixed<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    Ok({
        match &arguments[..] {
            [List(samples)] => {
                let mut fixed_samples = Vec::new();
                for sample in samples {
                    match sample.expr.as_const_float() {
                        Some(value) => fixed_samples.push(value),
                        None => {
                            return Err(Error::internal_here("Invalid sample in fixed waveform"));
                        }
                    }
                }
                Expr::Waveform(Waveform::Fixed(fixed_samples, ()))
            }
            _ => return Err(Error::internal_here("Invalid argument for fixed waveform")),
        }
    })
}

/// Returns a built-in of kind `kind` that applies `f` to a waveform argument or
/// to a seq argument's waveform.
pub fn curry<M, S>(
    kind: BuiltInKind,
    f: impl Fn(Box<Waveform<M>>) -> Waveform<M> + Send + Sync + 'static,
) -> BuiltInFn<M, S>
where
    M: Display,
{
    BuiltInFn::new(
        kind,
        move |mut arguments: Vec<Expr<M, S>>| -> Result<Expr<M, S>, Error<S>> {
            if arguments.len() != 1 {
                return Err(Error::internal_here("Expected waveform"));
            }
            let waveform = arguments.remove(0);
            Ok(match waveform {
                Expr::Waveform(a) => Expr::Waveform(f(Box::new(a))),
                Seq { offset, waveform } => match waveform.expr {
                    Expr::Waveform(waveform) => Seq {
                        offset,
                        waveform: boxed(Expr::Waveform(f(Box::new(waveform)))),
                    },
                    expr => {
                        return Err(Error::internal_here(format!(
                            "Expected waveform as argument to seq, got {}",
                            expr
                        )));
                    }
                },
                expr => {
                    return Err(Error::internal_here(format!(
                        "Expected waveform or seq, got {}",
                        expr
                    )));
                }
            })
        },
    )
}

pub fn fin<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static,
{
    Ok({
        if arguments.len() != 1 {
            return Err(Error::internal_here(format!(
                "Expected one argument for fin, got {}",
                arguments.len()
            )));
        }
        let arg = arguments.remove(0);
        match arg {
            Expr::Waveform(length) => {
                let length = length;
                BuiltIn {
                    name: format!("fin({})", length),
                    function: curry(BuiltInKind::Arranges, move |waveform: Box<Waveform<M>>| {
                        Waveform::Fin {
                            length: Box::new(length.clone()),
                            waveform,
                            state: (),
                        }
                    }),
                }
            }
            _ => return Err(Error::internal_here("Invalid arguments for fin")),
        }
    })
}

pub fn seq<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static,
{
    Ok({
        if arguments.len() != 1 {
            return Err(Error::internal_here(format!(
                "Expected one argument for seq, got {}",
                arguments.len()
            )));
        }
        let offset = match arguments.remove(0) {
            Expr::Waveform(offset) => offset,
            expr => {
                return Err(Error::internal_here(format!(
                    "Invalid argument for seq: {}",
                    expr
                )));
            }
        };
        let name = format!("seq({})", offset);
        BuiltIn {
            name,
            function: BuiltInFn::new(
                BuiltInKind::Arranges,
                move |mut arguments: Vec<Expr<M, S>>| -> Result<Expr<M, S>, Error<S>> {
                    let offset = offset.clone();
                    if arguments.len() != 1 {
                        return Err(Error::internal_here(format!(
                            "Expected one argument for seq({}), got {}",
                            offset,
                            arguments.len()
                        )));
                    }
                    Ok(match arguments.remove(0) {
                        Expr::Waveform(waveform) => Seq {
                            offset: boxed(Expr::Waveform(offset)),
                            waveform: boxed(Expr::Waveform(waveform)),
                        },
                        expr => {
                            return Err(Error::internal_here(format!(
                                "Expected argument to seq({}) to be a waveform or float, got {}",
                                offset, expr
                            )));
                        }
                    })
                },
            ),
        }
    })
}

pub fn unseq<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + 'static,
{
    Ok({
        if !arguments.is_empty() {
            return Err(Error::internal_here(format!(
                "Expected no arguments for unseq, got {}",
                arguments.len()
            )));
        }
        Expr::BuiltIn {
            name: "unseq()".to_string(),
            function: BuiltInFn::new(
                BuiltInKind::Arranges,
                |mut arguments: Vec<Expr<M, S>>| -> Result<Expr<M, S>, Error<S>> {
                    if arguments.len() != 1 {
                        return Err(Error::internal_here(format!(
                            "Expected argument for unseq(), got {}",
                            arguments.len()
                        )));
                    }
                    Ok(match arguments.remove(0) {
                        Seq { waveform, .. } => waveform.expr,
                        _ => return Err(Error::internal_here("Expected seq as argument to unseq")),
                    })
                },
            ),
        }
    })
}

/*
// TODO reconsider this: maybe this doesn't make sense any more... or the right argument
// needs to be a list of waveforms
pub fn waveform_convolution(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>> where M: Debug {
    Ok({
    if arguments.len() != 2 {
        return Err(Error::internal_here(format!("Expected two arguments for ~*")));
    }
    match (arguments.remove(0), arguments.remove(0)) {
        (a, b) => {
            let waveform = match (a, b) {
                (Expr::Waveform(a), Expr::Waveform(b)) => Waveform::Filter {
                    waveform: Box::new(a),
                    feed_forward: vec![b],
                    feedback: vec![],
                    state: (),
                },
                _ => return Err(Error::internal_here("Invalid arguments for ~*")),
            };
            Expr::Waveform(waveform)
        }
    }
    })
}
*/

pub fn waveform_filter<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static,
{
    Ok({
        if arguments.len() != 2 {
            return Err(Error::internal_here(
                "Expected two lists of waveforms for filter",
            ));
        }
        let feed_forward = match arguments.remove(0) {
            Expr::List(exprs) => {
                if exprs.is_empty() {
                    return Err(Error::eval_here(
                        "filter needs at least one feed-forward coefficient".to_string(),
                    ));
                }
                let mut feed_forward = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    match expr.expr {
                        Expr::Waveform(a) => feed_forward.push(a),
                        _ => {
                            return Err(Error::internal_here(
                                "Filter feed_forward argument must be a list",
                            ));
                        }
                    }
                }
                feed_forward
            }
            _ => {
                return Err(Error::internal_here(
                    "Feed-forward argument to filter must be a list",
                ));
            }
        };
        let feedback = match arguments.remove(0) {
            Expr::List(exprs) => {
                let mut feedback = Vec::with_capacity(exprs.len());
                for expr in exprs {
                    match expr.expr {
                        Expr::Waveform(a) => feedback.push(a),
                        _ => {
                            return Err(Error::internal_here(
                                "Filter feedback argument must be a list",
                            ));
                        }
                    }
                }
                feedback
            }
            _ => {
                return Err(Error::internal_here(
                    "Feedback argument to filter must be a list",
                ));
            }
        };

        BuiltIn {
            name: format!(
                "filter([{}], [{}])",
                feed_forward
                    .iter()
                    .map(|w| format!("{}", w))
                    .collect::<Vec<_>>()
                    .join(", "),
                feedback
                    .iter()
                    .map(|w| format!("{}", w))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            function: curry(BuiltInKind::Computes, move |waveform: Box<Waveform<M>>| {
                Waveform::Filter {
                    waveform,
                    feed_forward: feed_forward.clone(),
                    feedback: feedback.clone(),
                    state: (),
                }
            }),
        }
    })
}

pub fn reset<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    Ok({
        // TODO make it work in curried form?
        if arguments.len() != 2 {
            return Err(Error::internal_here("Expected two waveforms"));
        }
        let trigger = match arguments.remove(0) {
            Expr::Waveform(a) => a,
            _ => return Err(Error::internal_here("First argument must be a waveform")),
        };
        let waveform = match arguments.remove(0) {
            Expr::Waveform(a) => a,
            _ => return Err(Error::internal_here("Second argument must be a waveform")),
        };
        Expr::Waveform(Waveform::Reset {
            trigger: Box::new(trigger),
            waveform: Box::new(waveform),
            state: (),
        })
    })
}

pub fn alt<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Debug,
{
    Ok({
        // TODO make it work in curried form?
        if arguments.len() != 3 {
            return Err(Error::internal_here("Expected three waveforms"));
        }
        let trigger = match peel_const(arguments.remove(0)) {
            Expr::Waveform(a) => a,
            _ => return Err(Error::internal_here("First argument must be a waveform")),
        };
        let positive_waveform = match arguments.remove(0) {
            Expr::Waveform(a) => a,
            _ => return Err(Error::internal_here("Second argument must be a waveform")),
        };
        let negative_waveform = match arguments.remove(0) {
            Expr::Waveform(a) => a,
            _ => return Err(Error::internal_here("Third argument must be a waveform")),
        };
        match trigger {
            Waveform::Const(v) if v >= 0.0 => Expr::Waveform(positive_waveform),
            Waveform::Const(_) => Expr::Waveform(negative_waveform),
            trigger => Expr::Waveform(Waveform::Alt {
                trigger: Box::new(trigger),
                positive_waveform: Box::new(positive_waveform),
                negative_waveform: Box::new(negative_waveform),
            }),
        }
    })
}

// TODO move to main? (Because it only works in the native app anyway...?)
fn capture<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Display + 'static,
{
    Ok({
        if arguments.len() != 1 {
            return Err(Error::internal_here("Expected one argument for capture"));
        }
        let file_stem = match arguments.remove(0) {
            Expr::String(file_stem) => file_stem,
            _ => {
                return Err(Error::internal_here(
                    "Expected a string argument to capture",
                ));
            }
        };
        BuiltIn {
            name: format!("capture({})", file_stem),
            function: curry(BuiltInKind::Arranges, move |waveform: Box<Waveform<M>>| {
                Waveform::Captured {
                    file_stem: file_stem.clone(),
                    waveform,
                }
            }),
        }
    })
}

pub fn chord<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [List(exprs)] => {
                let mut result = Waveform::Fixed(vec![], ());
                for expr in exprs.iter().rev() {
                    let waveform: Box<Waveform<M>> = match &expr.expr {
                        Expr::Waveform(waveform) => Box::new(waveform.clone()),
                        _ => {
                            return Err(Error::internal_here(format!(
                                "Invalid element in chord: {}",
                                expr
                            )));
                        }
                    };
                    result = Waveform::BinaryOp(BinaryOperator::Merge, waveform, Box::new(result));
                }
                Expr::Waveform(result)
            }
            _ => return Err(Error::internal_here("Invalid argument for chord")),
        }
    })
}

pub fn sequence<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + PartialEq + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    Ok({
        if arguments.len() != 1 {
            return Err(Error::internal_here("Invalid argument for sequence"));
        }
        match &mut arguments[0] {
            List(exprs) => {
                // The empty sequence is the identity of `\`: no sound, no
                // time advance.
                if exprs.is_empty() {
                    return Ok(Seq {
                        offset: boxed(Expr::Waveform(Waveform::Const(0.0))),
                        waveform: boxed(Expr::Waveform(Waveform::Fixed(vec![], ()))),
                    });
                }
                if let Some(element) = exprs
                    .iter()
                    .find(|element| !matches!(element.expr, Expr::Seq { .. }))
                {
                    return Err(Error::internal_here(format!(
                        "Expected a seq in sequence, got {}",
                        element.expr
                    )));
                }
                let mut result = exprs.remove(exprs.len() - 1).expr;
                while !exprs.is_empty() {
                    result = followed_by(vec![exprs.remove(exprs.len() - 1).expr, result])?;
                }
                result
            }
            _ => return Err(Error::internal_here("Invalid argument for sequence")),
        }
    })
}

/// Builds the `debug` built-in, which renders its argument using the supplied
/// `print` function.
///
/// # Example
/// ```tuun
/// $(440 | debug("freq"))   // logs `[DEBUG] freq: 440`, plays $440
/// ```
pub fn debug<M, S>(print: impl Fn(&str) + Clone + Send + Sync + 'static) -> SourceExpr<M, S>
where
    M: Clone + Display + 'static + CarriesTrace<S>,
    S: Clone + 'static,
{
    SourceExpr::from(Expr::BuiltIn {
        name: "debug".to_string(),
        function: BuiltInFn::new(
            BuiltInKind::Arranges,
            move |mut arguments: Vec<Expr<M, S>>| {
                if arguments.len() != 1 {
                    return Err(Error::internal_here("Expected one argument for debug"));
                }
                let label = match arguments.remove(0) {
                    Expr::String(label) => label,
                    _ => {
                        return Err(Error::internal_here("Expected a string argument to debug"));
                    }
                };
                let print = print.clone();
                Ok(BuiltIn {
                    name: format!("debug({})", label),
                    function: BuiltInFn::new(
                        BuiltInKind::Arranges,
                        move |mut arguments: Vec<Expr<M, S>>| {
                            if arguments.len() != 1 {
                                return Err(Error::internal_here("Expected one argument"));
                            }
                            // The value prints as it would play, without the
                            // trace marks it carries.
                            let shown = trace::strip_expr_trace_marks(arguments[0].clone());
                            print(&format!("[DEBUG] {}: {}", label, shown));
                            Ok(arguments.pop().unwrap())
                        },
                    ),
                })
            },
        ),
    })
}

/// Adds all of the built-ins to `bindings`.
pub fn add_bindings<M, S>(bindings: &mut Vec<expr::SourceBinding<M, S>>)
where
    M: Debug + Clone + Display + PartialEq + Send + Sync + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    fn def<M, S>(id: &str, expr: expr::SourceExpr<M, S>) -> expr::SourceBinding<M, S> {
        use crate::expr::Binding;
        use crate::expr::Pattern;
        Binding::Definition(Pattern::Identifier(id.to_string()), expr).into()
    }
    bindings.push(def("true", SourceExpr::bool(true)));
    bindings.push(def("false", SourceExpr::bool(false)));
    bindings.push(def(
        "time",
        SourceExpr::from(Expr::Waveform(Waveform::Time(()))),
    ));
    bindings.push(def(
        "noise",
        SourceExpr::from(Expr::Waveform(Waveform::Noise)),
    ));

    use BuiltInKind::{Arranges, Computes};
    let builtins: Vec<(&str, BuiltInKind, BuiltinFn<M, S>)> = vec![
        ("+", Computes, plus),
        ("-", Computes, minus),
        ("*", Computes, times),
        ("/", Computes, divide),
        ("&", Computes, merge),
        ("\\", Arranges, followed_by),
        ("==", Arranges, equals),
        ("!=", Arranges, not_equals),
        ("<", Arranges, less_than),
        ("<=", Arranges, less_than_equals),
        (">", Arranges, greater_than),
        (">=", Arranges, greater_than_equals),
        ("pow", Computes, power),
        ("log", Computes, log),
        ("sqrt", Computes, sqrt),
        ("exp", Computes, exp),
        ("sin", Computes, sin),
        ("phase", Computes, phase),
        ("round", Computes, round),
        ("map", Arranges, map),
        ("reduce", Arranges, reduce),
        ("unfold", Arranges, unfold),
        ("append", Arranges, append),
        ("concat", Arranges, concat),
        ("nth", Arranges, nth),
        ("fixed", Computes, fixed),
        ("fin", Arranges, fin),
        ("seq", Arranges, seq),
        ("unseq", Arranges, unseq),
        ("filter", Computes, waveform_filter),
        //("~*", waveform_convolution),
        ("reset", Computes, reset),
        ("alt", Computes, alt),
        ("capture", Arranges, capture),
        ("__chord", Arranges, chord),
        ("__sequence", Arranges, sequence),
    ];
    for (name, kind, function) in builtins {
        bindings.push(def(
            name,
            SourceExpr::from(Expr::BuiltIn {
                name: name.to_string(),
                function: BuiltInFn::new(kind, function),
            }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::expr::ErrorKind;
    use Expr::BuiltIn;

    #[test]
    fn test_debug() {
        use std::sync::Mutex;
        let printed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&printed);
        let debug = debug::<u32, ()>(move |line| sink.lock().unwrap().push(line.to_string()));
        let Expr::BuiltIn { function, .. } = debug.expr else {
            panic!("debug should build a BuiltIn");
        };

        let func = function.0(vec![Expr::String("freq".to_string())]);
        let result = match func.unwrap() {
            BuiltIn { function, .. } => function.0(vec![Expr::float(440.0)]),
            _ => panic!("debug applied to label should be a BuiltIn"),
        };
        assert_eq!(format!("{}", result.clone().unwrap()), "440");
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] freq: 440"]);
    }

    #[test]
    fn test_nth_requires_a_non_negative_integer_index() {
        let list = || List(vec![SourceExpr::<u32>::float(1.0), SourceExpr::float(2.0)]);
        let result = nth(vec![Expr::float(1.0), list()]);
        assert_eq!(format!("{}", result.clone().unwrap()), "2");
        // A fractional index is a sort the checker rules out, so reaching
        // one here is its failure, not the caller's.
        let result = nth(vec![Expr::float(0.5), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Internal));
        // A negative index is an `int` like any other and the lattice has no
        // way to exclude it, so this one is the caller's.
        let result = nth(vec![Expr::float(-1.0), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Eval));
        assert_eq!(result.unwrap_err().message(), "no element with index -1");
        // As is an index past the end.
        let result = nth(vec![Expr::float(9.0), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Eval));
    }

    #[test]
    fn test_map() {
        let exprs: Vec<SourceExpr<u32>> = vec![
            SourceExpr::float(2.0),
            SourceExpr::float(3.0),
            SourceExpr::float(4.0),
        ];
        let result = map(vec![
            BuiltIn {
                name: "minus".to_string(),
                function: BuiltInFn::new(BuiltInKind::Computes, minus),
            },
            List(exprs),
        ]);
        assert_eq!(format!("{}", result.clone().unwrap()), "[-2, -3, -4]");
    }

    #[test]
    fn test_reduce() {
        let exprs: Vec<SourceExpr<u32>> = vec![
            SourceExpr::float(2.0),
            SourceExpr::float(3.0),
            SourceExpr::float(4.0),
        ];
        let result = reduce(vec![
            BuiltIn {
                name: "plus".to_string(),
                function: BuiltInFn::new(BuiltInKind::Computes, plus),
            },
            Expr::float(1.0),
            List(exprs),
        ]);
        assert_eq!(format!("{}", result.clone().unwrap()), "10");
    }

    #[test]
    fn test_followed_by_sums_seq_offsets() {
        fn seq(seconds: f32) -> Expr<u32, ()> {
            Seq {
                offset: boxed(Expr::Waveform(Waveform::BinaryOp(
                    BinaryOperator::Subtract,
                    Box::new(Waveform::Time(())),
                    Box::new(Waveform::Const(seconds)),
                ))),
                waveform: boxed(Expr::float(1.0)),
            }
        }
        fn offset_root(expr: Expr<u32, ()>) -> Option<Waveform<u32>> {
            let Seq { offset, .. } = expr else {
                panic!("seq \\ seq should be a seq");
            };
            let Expr::Waveform(offset) = offset.expr else {
                panic!("offset should be a waveform");
            };
            optimizer::first_root(&offset)
        }
        // The combined offset is left unoptimized; the next `\` still reads
        // its root off, so a chain stays flat.
        let two = followed_by(vec![seq(1.0), seq(2.0)]).unwrap();
        assert_eq!(offset_root(two.clone()), Some(Waveform::Const(3.0)));
        let three = followed_by(vec![two, seq(4.0)]).unwrap();
        assert_eq!(offset_root(three), Some(Waveform::Const(7.0)));
    }

    #[test]
    fn test_unary_op_keeps_seq_offset() {
        let seq = Seq {
            offset: boxed(Expr::float(2.0)),
            waveform: boxed(Expr::Waveform(Waveform::Time(()))),
        };
        let Seq { offset, waveform } = exp::<u32, ()>(vec![seq]).unwrap() else {
            panic!("exp of a seq should be a seq");
        };
        assert!(matches!(offset.expr, Expr::Waveform(Waveform::Const(v)) if v == 2.0));
        assert!(matches!(
            waveform.expr,
            Expr::Waveform(Waveform::UnaryOp(UnaryOperator::Exp, a)) if *a == Waveform::Time(())
        ));
    }

    #[test]
    fn test_alt_with_constant_trigger_selects_its_branch() {
        let alt_of = |trigger: Waveform<u32>| {
            alt::<u32, ()>(vec![
                Expr::Waveform(trigger),
                Expr::Waveform(Waveform::Const(1.0)),
                Expr::Waveform(Waveform::Time(())),
            ])
            .unwrap()
        };
        assert!(
            matches!(alt_of(Waveform::Const(0.0)), Expr::Waveform(Waveform::Const(v)) if v == 1.0)
        );
        assert!(matches!(
            alt_of(Waveform::Const(-0.5)),
            Expr::Waveform(Waveform::Time(()))
        ));
        assert!(matches!(
            alt_of(Waveform::Time(())),
            Expr::Waveform(Waveform::Alt { .. })
        ));
    }

    #[test]
    fn test_chord_of_nothing_is_the_empty_waveform() {
        let result = chord::<u32, ()>(vec![List(vec![])]).unwrap();
        assert!(matches!(result, Expr::Waveform(Waveform::Fixed(v, ())) if v.is_empty()));
    }

    /// A mark id that carries a trace and nothing else.
    #[derive(Clone, Debug)]
    struct Traced(Arc<trace::Trace<()>>);

    impl Display for Traced {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "traced")
        }
    }

    // Two mark ids are equal when they share one trace.
    impl PartialEq for Traced {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.0, &other.0)
        }
    }

    impl CarriesTrace<()> for Traced {
        fn from_trace(trace: &Arc<trace::Trace<()>>) -> Option<Self> {
            Some(Traced(trace.clone()))
        }
        fn as_trace(&self) -> Option<&Arc<trace::Trace<()>>> {
            Some(&self.0)
        }
    }

    /// Returns `w` under a trace mark.
    fn traced(w: Waveform<Traced>) -> Waveform<Traced> {
        Waveform::Marked {
            id: Traced(Arc::new(trace::Trace::new::<Traced>(None, None))),
            waveform: Box::new(w),
        }
    }

    fn traced_float(value: f32) -> Expr<Traced, ()> {
        Expr::Waveform(traced(Waveform::Const(value)))
    }

    #[test]
    fn consumed_constants_are_read_through_trace_marks() {
        let float = |result: Result<Expr<Traced, ()>, Error<()>>| result.unwrap().as_const_float();
        assert_eq!(float(sqrt(vec![traced_float(4.0)])), Some(2.0));
        assert_eq!(float(round(vec![traced_float(1.6)])), Some(2.0));
        assert_eq!(
            float(times(vec![traced_float(2.0), traced_float(3.0)])),
            Some(6.0)
        );
        let list = || List(vec![SourceExpr::float(1.0), SourceExpr::float(2.0)]);
        assert_eq!(float(nth(vec![traced_float(1.0), list()])), Some(2.0));
        let fixed = fixed(vec![List(vec![SourceExpr::from(traced_float(0.5))])]).unwrap();
        assert!(matches!(fixed, Expr::Waveform(Waveform::Fixed(v, ())) if v == [0.5]));
        for (compare, expected) in [
            (equals as BuiltinFn<Traced, ()>, false),
            (not_equals, true),
            (less_than, true),
            (less_than_equals, true),
            (greater_than, false),
            (greater_than_equals, false),
        ] {
            let result = compare(vec![traced_float(1.0), traced_float(2.0)]).unwrap();
            assert!(matches!(result, Bool(b) if b == expected));
        }
    }

    #[test]
    fn debug_prints_values_without_trace_marks() {
        use std::sync::Mutex;
        let printed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&printed);
        let debug = debug::<Traced, ()>(move |line| sink.lock().unwrap().push(line.to_string()));
        let Expr::BuiltIn { function, .. } = debug.expr else {
            panic!("debug should build a BuiltIn");
        };
        let Ok(BuiltIn { function, .. }) = function.0(vec![Expr::String("freq".to_string())])
        else {
            panic!("debug applied to label should be a BuiltIn");
        };
        // The value is handed back as it was, marks and all.
        let result = function.0(vec![traced_float(440.0)]).unwrap();
        assert!(matches!(result, Expr::Waveform(Waveform::Marked { .. })));
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] freq: 440"]);
    }

    #[test]
    fn a_traced_constant_count_unfolds() {
        let function = BuiltIn {
            name: "minus".to_string(),
            function: BuiltInFn::new(BuiltInKind::Computes, minus),
        };
        let result = unfold(vec![function, Expr::float(1.0), traced_float(2.0)]).unwrap();
        assert_eq!(format!("{}", result), "[1, -1]");
    }

    #[test]
    fn a_traced_constant_trigger_chooses_its_branch() {
        let result = alt::<Traced, ()>(vec![
            traced_float(1.0),
            Expr::float(2.0),
            Expr::Waveform(Waveform::Time(())),
        ])
        .unwrap();
        assert_eq!(result.as_const_float(), Some(2.0));
    }

    #[test]
    fn traced_offsets_still_add() {
        let seq = |seconds: f32| Seq {
            offset: boxed(Expr::Waveform(Waveform::BinaryOp(
                BinaryOperator::Subtract,
                Box::new(traced(Waveform::Time(()))),
                Box::new(traced(Waveform::Const(seconds))),
            ))),
            waveform: boxed(Expr::float(1.0)),
        };
        let Seq { offset, .. } = followed_by(vec![seq(1.0), seq(2.0)]).unwrap() else {
            panic!("seq \\ seq should be a seq");
        };
        let Expr::Waveform(offset) = offset.expr else {
            panic!("offset should be a waveform");
        };
        let root = optimizer::first_root(&trace::strip_trace_marks(offset));
        assert!(matches!(root, Some(Waveform::Const(v)) if v == 3.0));
    }
}
