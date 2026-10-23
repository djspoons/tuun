use std::fmt::{Debug, Display};

use crate::eval;
use crate::eval::BuiltInCall;
use crate::expr;
use crate::expr::{BuiltInFn, Error, Expr, SourceExpr};
use crate::optimizer;
use crate::trace::{self, CarriesTrace};
use crate::waveform::{BinaryOperator, UnaryOperator, Waveform};
use Expr::{Bool, BuiltIn, List, Seq};

type BuiltinFn<M, S> =
    fn(Vec<SourceExpr<M, S>>, &BuiltInCall<S>) -> Result<SourceExpr<M, S>, Error<S>>;

/// Returns `arguments` as an array of exactly `N`, or an internal error naming
/// the built-in `name`.
fn exactly<const N: usize, T, S>(arguments: Vec<T>, name: &str) -> Result<[T; N], Error<S>> {
    let count = arguments.len();
    arguments.try_into().map_err(|_| {
        Error::internal_here(format!(
            "Expected {} argument(s) for {}, got {}",
            N, name, count
        ))
    })
}

/// Returns the waveform `argument` marked for embedding (see [`eval::embed`]),
/// or an internal error saying `expected` when it isn't a waveform.
fn embed_waveform<M: CarriesTrace<S>, S: Clone>(
    argument: SourceExpr<M, S>,
    expected: &str,
) -> Result<Waveform<M>, Error<S>> {
    match eval::embed(argument) {
        Expr::Waveform(w) => Ok(w),
        _ => Err(Error::internal_here(expected.to_string())),
    }
}

/// An argument to a built-in that threads seqs: a waveform, marked for
/// embedding, or a seq's offset, unchanged, and its waveform, marked.
enum Operand<M, S> {
    Waveform(Waveform<M>),
    Seq(SourceExpr<M, S>, Waveform<M>),
}

/// Returns `argument` as an operand of the built-in `name`.
fn operand<M, S>(argument: SourceExpr<M, S>, name: &str) -> Result<Operand<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let SourceExpr { expr, span, via } = argument;
    match expr {
        Expr::Waveform(w) => Ok(Operand::Waveform(embed_waveform(
            SourceExpr {
                expr: Expr::Waveform(w),
                span,
                via,
            },
            name,
        )?)),
        Seq { offset, waveform } => {
            let waveform =
                embed_waveform(*waveform, &format!("Invalid argument to seq in {}", name))?;
            Ok(Operand::Seq(*offset, waveform))
        }
        expr => Err(Error::internal_here(format!(
            "Invalid argument for {}: {:?}",
            name, expr
        ))),
    }
}

// TODO maybe use Display instead of Debug for errors?
fn binary_op<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
    name: &str,
    op: BinaryOperator,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [x, y] = exactly(arguments, name)?;
    // Two constants fold eagerly, so arithmetic on constants stays constant.
    if let (Some(a), Some(b)) = (x.expr.as_const_float(), y.expr.as_const_float()) {
        return Ok(call.computed(Expr::float(op.apply(a, b))));
    }
    let apply = |a, b| Expr::Waveform(Waveform::BinaryOp(op, Box::new(a), Box::new(b)));
    Ok(call.computed(match (operand(x, name)?, operand(y, name)?) {
        (Operand::Waveform(a), Operand::Waveform(b)) => apply(a, b),
        (Operand::Seq(offset, a), Operand::Waveform(b))
        | (Operand::Waveform(a), Operand::Seq(offset, b)) => Seq {
            offset: Box::new(offset),
            waveform: Box::new(call.computed(apply(a, b))),
        },
        (Operand::Seq(..), Operand::Seq(..)) => {
            return Err(Error::internal_here(format!(
                "Invalid arguments for {}: two seqs",
                name
            )));
        }
    }))
}

pub fn plus<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    binary_op(arguments, call, "+", BinaryOperator::Add)
}

pub fn minus<M, S>(
    mut arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    // Unary minus is multiplication by -1, so it folds and threads seqs the
    // same way.
    if arguments.len() == 1 {
        arguments.insert(0, SourceExpr::float(-1.0));
        return binary_op(arguments, call, "-", BinaryOperator::Multiply);
    }
    binary_op(arguments, call, "-", BinaryOperator::Subtract)
}

pub fn times<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    binary_op(arguments, call, "*", BinaryOperator::Multiply)
}

pub fn divide<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    binary_op(arguments, call, "/", BinaryOperator::Divide)
}

pub fn merge<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    binary_op(arguments, call, "&", BinaryOperator::Merge)
}

// Given waveforms that represent offsets (and assuming offset waveforms are of the form
// `Time ~+ w` or `Const(x)`) return a new waveform which represents the sum of those offsets.
fn add_offsets<M, S>(a: Waveform<M>, b: Waveform<M>) -> Result<Waveform<M>, Error<S>>
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
        (Some(a_root), Some(b_root)) => Ok(Waveform::BinaryOp(
            BinaryOperator::Subtract,
            Box::new(Waveform::Time(())),
            Box::new(Waveform::BinaryOp(
                BinaryOperator::Add,
                Box::new(a_root),
                Box::new(b_root),
            )),
        )),
        (a_root, b_root) => Err(Error::internal_here(format!(
            "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
            a_root, b_root, a, b
        ))),
    }
}

/// Returns the merge of `a` with `b` delayed until `a_offset`, `a`'s offset.
/// (This is the waveform implementation of followed_by.)
fn after<M>(a_offset: Waveform<M>, a: Waveform<M>, b: Waveform<M>) -> Waveform<M> {
    Waveform::BinaryOp(
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
    )
}

/// Returns the seq `argument`'s offset and waveform, each marked for
/// embedding, or an internal error naming the built-in `name`.
fn embed_seq<M, S>(
    argument: SourceExpr<M, S>,
    name: &str,
) -> Result<(Waveform<M>, Waveform<M>), Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    match operand(argument, name)? {
        Operand::Seq(offset, waveform) => Ok((
            embed_waveform(offset, &format!("Invalid offset of seq in {}", name))?,
            waveform,
        )),
        Operand::Waveform(_) => Err(Error::internal_here(format!(
            "Expected seq as argument to {}",
            name
        ))),
    }
}

/// Returns the merge of two arguments where the second argument delayed until
/// after the offset of the first (which therefore must be a seq).
pub fn followed_by<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Display + Clone + PartialEq + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [a, b] = exactly(arguments, "\\")?;
    let (a_offset, a) = embed_seq(a, "\\")?;
    Ok(call.assembled(match operand(b, "\\")? {
        Operand::Waveform(b) => Expr::Waveform(after(a_offset, a, b)),
        Operand::Seq(b_offset, b) => {
            let b_offset = embed_waveform(b_offset, "Invalid offset of seq in \\")?;
            let total_offset = add_offsets(a_offset.clone(), b_offset)?;
            Seq {
                offset: Box::new(call.assembled(Expr::Waveform(total_offset))),
                waveform: Box::new(call.assembled(Expr::Waveform(after(a_offset, a, b)))),
            }
        }
    }))
}

pub fn power<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    binary_op(arguments, call, "pow", BinaryOperator::Power)
}

pub fn log<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [a, b] = exactly(arguments, "log")?;
    let a = embed_waveform(a, "Invalid arguments for log")?;
    let b = embed_waveform(b, "Invalid arguments for log")?;
    Ok(call.computed(Expr::Waveform(Waveform::BinaryOp(
        BinaryOperator::Log,
        Box::new(a),
        Box::new(b),
    ))))
}

pub fn sqrt<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [x] = exactly(arguments, "sqrt")?;
    match x.expr.as_const_float() {
        Some(value) if value >= 0.0 => Ok(call.computed(Expr::float(value.sqrt()))),
        Some(value) => Err(Error::eval_here(format!(
            "square root of a negative number: {}",
            value
        ))),
        None => Err(Error::internal_here("Invalid argument for sqrt")),
    }
}

/// Returns the result of applying `op` to each sample of a waveform or seq.
///
/// A seq keeps its offset, so `op` applies to its waveform only.
fn unary_op<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
    name: &str,
    op: UnaryOperator,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [x] = exactly(arguments, name)?;
    let apply = |a| Expr::Waveform(Waveform::UnaryOp(op, Box::new(a)));
    Ok(call.computed(match operand(x, name)? {
        Operand::Waveform(a) => apply(a),
        Operand::Seq(offset, a) => Seq {
            offset: Box::new(offset),
            waveform: Box::new(call.computed(apply(a))),
        },
    }))
}

/// Returns a waveform combinator that returns the exponential function of each sample.
pub fn exp<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    unary_op(arguments, call, "exp", UnaryOperator::Exp)
}

/// Returns the waveform combinator that computes the sine of each sample.
pub fn sin<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    unary_op(arguments, call, "sin", UnaryOperator::Sin)
}

pub fn phase<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    // Like the waveform, Phase, the first argument is frequency in hertz
    let [frequency, offset] = exactly(arguments, "phase")?;
    let frequency = embed_waveform(frequency, "Invalid arguments for phase")?;
    let offset = embed_waveform(offset, "Invalid arguments for phase")?;
    Ok(call.computed(Expr::Waveform(Waveform::Phase {
        frequency: Box::new(frequency),
        offset: Box::new(offset),
        state: (),
    })))
}

pub fn round<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [x] = exactly(arguments, "round")?;
    match x.expr.as_const_float() {
        Some(value) => Ok(call.computed(Expr::float(value.round()))),
        None => Err(Error::internal_here("Invalid argument for round")),
    }
}

/// Returns whether two booleans, strings, or constants are equal (when `equal`)
/// or unequal, for the built-in `name`.
fn equality<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
    name: &str,
    equal: bool,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [a, b] = exactly(arguments, name)?;
    let same = match (&a.expr, &b.expr) {
        (Bool(a), Bool(b)) => a == b,
        (Expr::String(a), Expr::String(b)) => a == b,
        // TODO could consider a more general form of equality on waveforms but
        // this seems to be enough for now.
        (a, b) => match (a.as_const_float(), b.as_const_float()) {
            (Some(a), Some(b)) => a == b,
            _ => {
                return Err(Error::internal_here(format!(
                    "Invalid arguments for {}",
                    name
                )));
            }
        },
    };
    Ok(call.assembled(Bool(same == equal)))
}

/// Returns whether `holds` holds of two constants, for the built-in `name`.
fn compare<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
    name: &str,
    holds: fn(f32, f32) -> bool,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [a, b] = exactly(arguments, name)?;
    match (a.expr.as_const_float(), b.expr.as_const_float()) {
        (Some(a), Some(b)) => Ok(call.assembled(Bool(holds(a, b)))),
        _ => Err(Error::internal_here(format!(
            "Invalid arguments for {}",
            name
        ))),
    }
}

pub fn equals<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    equality(arguments, call, "==", true)
}

pub fn not_equals<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    equality(arguments, call, "!=", false)
}

pub fn less_than<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    compare(arguments, call, "<", |a, b| a < b)
}

pub fn less_than_equals<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    compare(arguments, call, "<=", |a, b| a <= b)
}

pub fn greater_than<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    compare(arguments, call, ">", |a, b| a > b)
}

pub fn greater_than_equals<M: CarriesTrace<S>, S: Clone>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>> {
    compare(arguments, call, ">=", |a, b| a >= b)
}

pub fn map<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + Debug + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [function, list] = exactly(arguments, "map")?;
    let List(elements) = list.expr else {
        return Err(Error::internal_here("Invalid arguments for map"));
    };
    let mut results: Vec<SourceExpr<M, S>> = Vec::with_capacity(elements.len());
    for element in elements {
        // An error with no span of its own is reported at the element.
        let at = element.span.clone();
        // TODO this application has no span, so it records no `Call` step: in
        // `map(fn (x) => x, …)` the result lights `x` but not `fn (x) => x`,
        // which a direct application `(fn (x) => x)(…)` lights.
        results.push(
            eval::evaluate_closed(SourceExpr::application(function.clone(), vec![element]))
                .map_err(|e| e.or_at(at))?,
        );
    }
    Ok(call.assembled(List(results)))
}

pub fn reduce<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    _call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [function, seed, list] = exactly(arguments, "reduce")?;
    let List(elements) = list.expr else {
        return Err(Error::internal_here("Invalid arguments for reduce"));
    };
    let mut acc = seed;
    for element in elements {
        // As in `map`, an error with no span is reported at the element.
        let at = element.span.clone();
        // TODO as in `map`, this application records no `Call` step, so the
        // function's text doesn't light.
        acc = eval::evaluate_closed(SourceExpr::application(
            function.clone(),
            vec![acc, element],
        ))
        .map_err(|e| e.or_at(at))?;
    }
    Ok(acc)
}

pub fn unfold<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [function, seed, count] = exactly(arguments, "unfold")?;
    let n = match count.expr.as_const_float() {
        Some(n) if n >= 0.0 && n.fract() == 0.0 => n as u32,
        // A negative or fractional count is a value type checker cannot rule
        // out.
        Some(n) => {
            return Err(Error::eval_here(format!(
                "unfold needs a non-negative whole count, got {}",
                n
            )));
        }
        None => return Err(Error::internal_here("Invalid arguments for unfold")),
    };
    let mut results: Vec<SourceExpr<M, S>> = Vec::new();
    let mut current = seed;
    for _ in 0..n {
        results.push(current.clone());
        // As in `map`, an error with no span is reported at the value the
        // function was applied to.
        let at = current.span.clone();
        // TODO as in `map`, this application records no `Call` step, so the
        // function's text doesn't light.
        current = eval::evaluate_closed(SourceExpr::application(function.clone(), vec![current]))
            .map_err(|e| e.or_at(at))?;
    }
    Ok(call.assembled(List(results)))
}

// Appends exactly two waveforms end to end. Lists join with `concat`.
pub fn append<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [a, b] = exactly(arguments, "append")?;
    let a = embed_waveform(a, "Expected two waveforms for append")?;
    let b = embed_waveform(b, "Expected two waveforms for append")?;
    Ok(call.assembled(Expr::Waveform(Waveform::Append(
        Box::new(a),
        Box::new(b),
        (),
    ))))
}

fn concat<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + CarriesTrace<S>,
    S: Clone + Debug,
{
    let [lists] = exactly(arguments, "concat")?;
    let List(lists) = lists.expr else {
        return Err(Error::internal_here("Invalid arguments for concat"));
    };
    let mut results = Vec::new();
    for list in lists {
        let List(elements) = list.expr else {
            return Err(Error::internal_here(
                "Expected list of lists as argument for concat",
            ));
        };
        results.extend(elements);
    }
    Ok(call.assembled(List(results)))
}

pub fn nth<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    _call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Clone + CarriesTrace<S>,
    S: Clone,
{
    let [index, list] = exactly(arguments, "nth")?;
    let index = index.expr.as_const_float();
    let List(elements) = list.expr else {
        return Err(Error::internal_here("Invalid arguments for nth"));
    };
    match index {
        Some(a) if a >= 0.0 && a.fract() == 0.0 => match elements.into_iter().nth(a as usize) {
            Some(element) => Ok(element),
            None => Err(Error::eval_here(format!("no element with index {}", a))),
        },
        // A negative index can't be ruled out by the type checker, so it's an
        // ordinary error (not an internal one).
        Some(a) if a.fract() == 0.0 => {
            Err(Error::eval_here(format!("no element with index {}", a)))
        }
        _ => Err(Error::internal_here("Invalid arguments for nth")),
    }
}

pub fn fixed<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [samples] = exactly(arguments, "fixed")?;
    let List(samples) = samples.expr else {
        return Err(Error::internal_here("Invalid argument for fixed waveform"));
    };
    let mut fixed_samples = Vec::with_capacity(samples.len());
    for sample in samples {
        match sample.expr.as_const_float() {
            Some(value) => fixed_samples.push(value),
            None => return Err(Error::internal_here("Invalid sample in fixed waveform")),
        }
    }
    Ok(call.computed(Expr::Waveform(Waveform::Fixed(fixed_samples, ()))))
}

/// Builds a result of a call, as [`BuiltInCall::computed`] and
/// [`BuiltInCall::assembled`] do.
type Build<M, S> = fn(&BuiltInCall<S>, Expr<M, S>) -> SourceExpr<M, S>;

/// Returns a built-in that applies `f` to a waveform argument or to a seq
/// argument's waveform, building its result (and a seq's new `waveform`) with
/// `build`.
pub fn curry<M, S>(
    build: Build<M, S>,
    f: impl Fn(Box<Waveform<M>>) -> Waveform<M> + Send + Sync + 'static,
) -> BuiltInFn<M, S>
where
    M: Debug + CarriesTrace<S> + 'static,
    S: Clone + Debug + 'static,
{
    BuiltInFn::new(move |arguments, call| {
        let [x] = exactly(arguments, "a curried built-in")?;
        Ok(build(
            call,
            match operand(x, "a curried built-in")? {
                Operand::Waveform(a) => Expr::Waveform(f(Box::new(a))),
                Operand::Seq(offset, a) => Seq {
                    offset: Box::new(offset),
                    waveform: Box::new(build(call, Expr::Waveform(f(Box::new(a))))),
                },
            },
        ))
    })
}

pub fn fin<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    let [length] = exactly(arguments, "fin")?;
    let Expr::Waveform(w) = &length.expr else {
        return Err(Error::internal_here("Invalid arguments for fin"));
    };
    let name = format!("fin({})", w);
    let length = embed_waveform(length, "Invalid arguments for fin")?;
    Ok(call.assembled(BuiltIn {
        name,
        function: curry(BuiltInCall::assembled, move |waveform: Box<Waveform<M>>| {
            Waveform::Fin {
                length: Box::new(length.clone()),
                waveform,
                state: (),
            }
        }),
    }))
}

pub fn seq<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static + CarriesTrace<S>,
    S: Clone + Send + Sync + 'static,
{
    let [offset] = exactly(arguments, "seq")?;
    let Expr::Waveform(w) = &offset.expr else {
        return Err(Error::internal_here(format!(
            "Invalid argument for seq: {}",
            offset.expr
        )));
    };
    let name = format!("seq({})", w);
    Ok(call.assembled(BuiltIn {
        name: name.clone(),
        function: BuiltInFn::new(move |arguments, call| {
            let [waveform] = exactly(arguments, &name)?;
            match waveform.expr {
                Expr::Waveform(_) => Ok(call.assembled(Seq {
                    offset: Box::new(offset.clone()),
                    waveform: Box::new(waveform),
                })),
                expr => Err(Error::internal_here(format!(
                    "Expected argument to {} to be a waveform or float, got {}",
                    name, expr
                ))),
            }
        }),
    }))
}

pub fn unseq<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S> + 'static,
    S: Clone + 'static,
{
    let [] = exactly(arguments, "unseq")?;
    Ok(call.assembled(Expr::BuiltIn {
        name: "unseq()".to_string(),
        function: BuiltInFn::new(|arguments, _call| {
            let [seq] = exactly(arguments, "unseq()")?;
            let Seq { waveform, .. } = seq.expr else {
                return Err(Error::internal_here("Expected seq as argument to unseq"));
            };
            Ok(*waveform)
        }),
    }))
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

/// Returns the waveforms of `list`, each marked for embedding, or an internal
/// error saying `expected` when it isn't a list of waveforms.
fn embed_waveforms<M: CarriesTrace<S>, S: Clone>(
    list: SourceExpr<M, S>,
    expected: &str,
) -> Result<Vec<Waveform<M>>, Error<S>> {
    let List(elements) = eval::embed(list) else {
        return Err(Error::internal_here(expected.to_string()));
    };
    elements
        .into_iter()
        .map(|element| match element.expr {
            Expr::Waveform(w) => Ok(w),
            _ => Err(Error::internal_here(expected.to_string())),
        })
        .collect()
}

pub fn waveform_filter<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + Send + Sync + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    let [feed_forward, feedback] = exactly(arguments, "filter")?;
    if matches!(&feed_forward.expr, List(exprs) if exprs.is_empty()) {
        return Err(Error::eval_here(
            "filter needs at least one feed-forward coefficient".to_string(),
        ));
    }
    let feed_forward = embed_waveforms(
        feed_forward,
        "Feed-forward argument to filter must be a list of waveforms",
    )?;
    let feedback = embed_waveforms(
        feedback,
        "Feedback argument to filter must be a list of waveforms",
    )?;
    let shown = |coefficients: &[Waveform<M>]| {
        coefficients
            .iter()
            .map(|w| format!("{}", trace::strip_trace_marks(w.clone())))
            .collect::<Vec<_>>()
            .join(", ")
    };
    Ok(call.computed(BuiltIn {
        name: format!("filter([{}], [{}])", shown(&feed_forward), shown(&feedback)),
        function: curry(BuiltInCall::computed, move |waveform: Box<Waveform<M>>| {
            Waveform::Filter {
                waveform,
                feed_forward: feed_forward.clone(),
                feedback: feedback.clone(),
                state: (),
            }
        }),
    }))
}

pub fn reset<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    // TODO make it work in curried form?
    let [trigger, waveform] = exactly(arguments, "reset")?;
    let trigger = embed_waveform(trigger, "First argument must be a waveform")?;
    let waveform = embed_waveform(waveform, "Second argument must be a waveform")?;
    Ok(call.computed(Expr::Waveform(Waveform::Reset {
        trigger: Box::new(trigger),
        waveform: Box::new(waveform),
        state: (),
    })))
}

pub fn alt<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + CarriesTrace<S>,
    S: Clone + Debug,
{
    // TODO make it work in curried form?
    let [trigger, positive, negative] = exactly(arguments, "alt")?;
    if !matches!(trigger.expr, Expr::Waveform(_)) {
        return Err(Error::internal_here("First argument must be a waveform"));
    }
    if !matches!(positive.expr, Expr::Waveform(_)) {
        return Err(Error::internal_here("Second argument must be a waveform"));
    }
    if !matches!(negative.expr, Expr::Waveform(_)) {
        return Err(Error::internal_here("Third argument must be a waveform"));
    }
    match trigger.expr.as_const_float() {
        Some(v) if v >= 0.0 => Ok(positive),
        Some(_) => Ok(negative),
        None => match (
            eval::embed(trigger),
            eval::embed(positive),
            eval::embed(negative),
        ) {
            (Expr::Waveform(trigger), Expr::Waveform(positive), Expr::Waveform(negative)) => {
                Ok(call.computed(Expr::Waveform(Waveform::Alt {
                    trigger: Box::new(trigger),
                    positive_waveform: Box::new(positive),
                    negative_waveform: Box::new(negative),
                })))
            }
            _ => Err(Error::internal_here("Expected three waveforms for alt")),
        },
    }
}

// TODO move to main? (Because it only works in the native app anyway...?)
fn capture<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Display + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    let [file_stem] = exactly(arguments, "capture")?;
    let Expr::String(file_stem) = file_stem.expr else {
        return Err(Error::internal_here(
            "Expected a string argument to capture",
        ));
    };
    Ok(call.assembled(BuiltIn {
        name: format!("capture({})", file_stem),
        function: curry(BuiltInCall::assembled, move |waveform: Box<Waveform<M>>| {
            Waveform::Captured {
                file_stem: file_stem.clone(),
                waveform,
            }
        }),
    }))
}

pub fn chord<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: CarriesTrace<S>,
    S: Clone,
{
    let [notes] = exactly(arguments, "chord")?;
    let notes = embed_waveforms(notes, "Invalid argument for chord")?;
    let mut result = Waveform::Fixed(vec![], ());
    for note in notes.into_iter().rev() {
        result = Waveform::BinaryOp(BinaryOperator::Merge, Box::new(note), Box::new(result));
    }
    Ok(call.assembled(Expr::Waveform(result)))
}

pub fn sequence<M, S>(
    arguments: Vec<SourceExpr<M, S>>,
    call: &BuiltInCall<S>,
) -> Result<SourceExpr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + PartialEq + 'static + CarriesTrace<S>,
    S: Clone + Debug + 'static,
{
    let [list] = exactly(arguments, "sequence")?;
    let List(mut elements) = list.expr else {
        return Err(Error::internal_here("Invalid argument for sequence"));
    };
    let seq = |offset, waveform| Seq {
        offset: Box::new(call.assembled(Expr::Waveform(offset))),
        waveform: Box::new(call.assembled(Expr::Waveform(waveform))),
    };
    // The empty sequence is the identity of `\`: no sound, no time advance.
    let Some(last) = elements.pop() else {
        return Ok(call.assembled(seq(Waveform::Const(0.0), Waveform::Fixed(vec![], ()))));
    };
    if elements.is_empty() {
        return match last.expr {
            Seq { .. } => Ok(call.assembled(last.expr)),
            expr => Err(Error::internal_here(format!(
                "Expected a seq in sequence, got {}",
                expr
            ))),
        };
    }
    let (mut offset, mut waveform) = embed_seq(last, "sequence")?;
    while let Some(element) = elements.pop() {
        let (a_offset, a) = embed_seq(element, "sequence")?;
        offset = add_offsets(a_offset.clone(), offset)?;
        waveform = after(a_offset, a, waveform);
    }
    Ok(call.assembled(seq(offset, waveform)))
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
        function: BuiltInFn::new(move |arguments, call| {
            let [label] = exactly(arguments, "debug")?;
            let Expr::String(label) = label.expr else {
                return Err(Error::internal_here("Expected a string argument to debug"));
            };
            let print = print.clone();
            Ok(call.assembled(BuiltIn {
                name: format!("debug({})", label),
                function: BuiltInFn::new(move |arguments, _call| {
                    let [value] = exactly(arguments, "debug")?;
                    // The value prints as it would play, without the
                    // trace marks it carries.
                    let shown = trace::strip_expr_trace_marks(value.expr.clone());
                    print(&format!("[DEBUG] {}: {}", label, shown));
                    Ok(value)
                }),
            }))
        }),
    })
}

/// Adds all of the built-ins to `bindings`.
pub fn add_bindings<M, S>(bindings: &mut Vec<expr::SourceBinding<M, S>>)
where
    M: Debug + Clone + Display + PartialEq + Send + Sync + 'static + CarriesTrace<S>,
    S: Clone + Debug + Send + Sync + 'static,
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

    let builtins: Vec<(&str, BuiltinFn<M, S>)> = vec![
        ("+", plus),
        ("-", minus),
        ("*", times),
        ("/", divide),
        ("&", merge),
        ("\\", followed_by),
        ("==", equals),
        ("!=", not_equals),
        ("<", less_than),
        ("<=", less_than_equals),
        (">", greater_than),
        (">=", greater_than_equals),
        ("pow", power),
        ("log", log),
        ("sqrt", sqrt),
        ("exp", exp),
        ("sin", sin),
        ("phase", phase),
        ("round", round),
        ("map", map),
        ("reduce", reduce),
        ("unfold", unfold),
        ("append", append),
        ("concat", concat),
        ("nth", nth),
        ("fixed", fixed),
        ("fin", fin),
        ("seq", seq),
        ("unseq", unseq),
        ("filter", waveform_filter),
        //("~*", waveform_convolution),
        ("reset", reset),
        ("alt", alt),
        ("capture", capture),
        ("__chord", chord),
        ("__sequence", sequence),
    ];
    for (name, function) in builtins {
        bindings.push(def(
            name,
            SourceExpr::from(Expr::BuiltIn {
                name: name.to_string(),
                function: BuiltInFn::new(function),
            }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::expr::{ErrorKind, boxed};
    use crate::trace::Trace;
    use Expr::BuiltIn;

    /// Applies the built-in function `f` to `arguments`, written nowhere, in a
    /// call written nowhere, and returns its result's value.
    fn call<M: CarriesTrace<S>, S: Clone>(
        f: impl Fn(Vec<SourceExpr<M, S>>, &BuiltInCall<S>) -> Result<SourceExpr<M, S>, Error<S>>,
        arguments: Vec<Expr<M, S>>,
    ) -> Result<Expr<M, S>, Error<S>> {
        let arguments = arguments.into_iter().map(SourceExpr::from).collect();
        call_source(f, arguments).map(|result| result.expr)
    }

    /// Applies the built-in function `f` to `arguments` in a call written
    /// nowhere.
    fn call_source<M: CarriesTrace<S>, S: Clone>(
        f: impl Fn(Vec<SourceExpr<M, S>>, &BuiltInCall<S>) -> Result<SourceExpr<M, S>, Error<S>>,
        arguments: Vec<SourceExpr<M, S>>,
    ) -> Result<SourceExpr<M, S>, Error<S>> {
        let call = BuiltInCall::new(None, Trace::new::<M>(None, None), &arguments);
        f(arguments, &call)
    }

    /// Returns `expr` written at `range`.
    fn written(expr: Expr<u32, ()>, range: std::ops::Range<usize>) -> SourceExpr<u32, ()> {
        SourceExpr {
            span: Some(expr::Span::new((), range)),
            ..SourceExpr::from(expr)
        }
    }

    /// Returns where `value` was written.
    fn range(value: &SourceExpr<u32, ()>) -> std::ops::Range<usize> {
        value.span.as_ref().expect("a span").range.clone()
    }

    #[test]
    fn projections_hand_back_the_part_with_its_own_trace() {
        let list = written(
            List(vec![
                written(Expr::float(1.0), 1..2),
                written(Expr::float(2.0), 3..4),
            ]),
            0..6,
        );
        let element = call_source(nth, vec![SourceExpr::float(1.0), list]).unwrap();
        assert_eq!(range(&element), 3..4);

        let seq = written(
            Seq {
                offset: Box::new(written(Expr::float(0.0), 8..9)),
                waveform: Box::new(written(Expr::float(1.0), 10..11)),
            },
            7..12,
        );
        let Ok(BuiltIn { function, .. }) = call(unseq::<u32, ()>, vec![]) else {
            panic!("unseq() should be a BuiltIn");
        };
        let part = call_source(|a, c| function.0(a, c), vec![seq]).unwrap();
        assert_eq!(range(&part), 10..11);
    }

    #[test]
    fn reduce_hands_back_its_accumulator() {
        let function = SourceExpr::from(BuiltIn {
            name: "minus".to_string(),
            function: BuiltInFn::new(minus),
        });
        let seed = written(Expr::float(1.0), 3..4);
        let result = call_source(reduce, vec![function, seed, SourceExpr::from(List(vec![]))]);
        assert_eq!(range(&result.unwrap()), 3..4);
    }

    #[test]
    fn callback_errors_point_at_the_failing_element() {
        // Fails on 2, with no span of its own.
        let fails_on_two = SourceExpr::from(BuiltIn {
            name: "fails_on_two".to_string(),
            function: BuiltInFn::new(|arguments: Vec<SourceExpr<u32, ()>>, _call| match arguments
                .last()
                .and_then(|a| a.expr.as_const_float())
            {
                Some(2.0) => Err(Error::eval_here("two")),
                _ => Ok(arguments.into_iter().last().unwrap()),
            }),
        });
        let list = || {
            SourceExpr::from(List(vec![
                written(Expr::float(1.0), 1..2),
                written(Expr::float(2.0), 3..4),
            ]))
        };
        let error = call_source(map, vec![fails_on_two.clone(), list()]).unwrap_err();
        assert_eq!(error.range(), Some(3..4));
        let seed = SourceExpr::float(0.0);
        let error = call_source(reduce, vec![fails_on_two.clone(), seed, list()]).unwrap_err();
        assert_eq!(error.range(), Some(3..4));
        let seed = written(Expr::float(2.0), 5..6);
        let count = SourceExpr::float(1.0);
        let error = call_source(unfold, vec![fails_on_two, seed, count]).unwrap_err();
        assert_eq!(error.range(), Some(5..6));
    }

    /// Applies `function` to `arguments` as [`call`] does.
    fn apply<M: CarriesTrace<S>, S: Clone>(
        function: &BuiltInFn<M, S>,
        arguments: Vec<Expr<M, S>>,
    ) -> Result<Expr<M, S>, Error<S>> {
        call(|a, c| function.0(a, c), arguments)
    }

    #[test]
    fn test_debug() {
        use std::sync::Mutex;
        let printed: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&printed);
        let debug = debug::<u32, ()>(move |line| sink.lock().unwrap().push(line.to_string()));
        let Expr::BuiltIn { function, .. } = debug.expr else {
            panic!("debug should build a BuiltIn");
        };

        let func = apply(&function, vec![Expr::String("freq".to_string())]);
        let result = match func.unwrap() {
            BuiltIn { function, .. } => apply(&function, vec![Expr::float(440.0)]),
            _ => panic!("debug applied to label should be a BuiltIn"),
        };
        assert_eq!(format!("{}", result.clone().unwrap()), "440");
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] freq: 440"]);
    }

    #[test]
    fn test_nth_requires_a_non_negative_integer_index() {
        let list = || List(vec![SourceExpr::<u32>::float(1.0), SourceExpr::float(2.0)]);
        let result = call(nth, vec![Expr::float(1.0), list()]);
        assert_eq!(format!("{}", result.clone().unwrap()), "2");
        // A fractional index is a sort the checker rules out, so reaching
        // one here is its failure, not the caller's.
        let result = call(nth, vec![Expr::float(0.5), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Internal));
        // A negative index is an `int` like any other and the lattice has no
        // way to exclude it, so this one is the caller's.
        let result = call(nth, vec![Expr::float(-1.0), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Eval));
        assert_eq!(result.unwrap_err().message(), "no element with index -1");
        // As is an index past the end.
        let result = call(nth, vec![Expr::float(9.0), list()]);
        assert!(matches!(&result, Err(e) if e.kind() == ErrorKind::Eval));
    }

    #[test]
    fn test_map() {
        let exprs: Vec<SourceExpr<u32>> = vec![
            SourceExpr::float(2.0),
            SourceExpr::float(3.0),
            SourceExpr::float(4.0),
        ];
        let result = call(
            map,
            vec![
                BuiltIn {
                    name: "minus".to_string(),
                    function: BuiltInFn::new(minus),
                },
                List(exprs),
            ],
        );
        assert_eq!(format!("{}", result.clone().unwrap()), "[-2, -3, -4]");
    }

    #[test]
    fn test_reduce() {
        let exprs: Vec<SourceExpr<u32>> = vec![
            SourceExpr::float(2.0),
            SourceExpr::float(3.0),
            SourceExpr::float(4.0),
        ];
        let result = call(
            reduce,
            vec![
                BuiltIn {
                    name: "plus".to_string(),
                    function: BuiltInFn::new(plus),
                },
                Expr::float(1.0),
                List(exprs),
            ],
        );
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
        let two = call(followed_by, vec![seq(1.0), seq(2.0)]).unwrap();
        assert_eq!(offset_root(two.clone()), Some(Waveform::Const(3.0)));
        let three = call(followed_by, vec![two, seq(4.0)]).unwrap();
        assert_eq!(offset_root(three), Some(Waveform::Const(7.0)));
    }

    #[test]
    fn test_unary_op_keeps_seq_offset() {
        let seq = Seq {
            offset: boxed(Expr::float(2.0)),
            waveform: boxed(Expr::Waveform(Waveform::Time(()))),
        };
        let Seq { offset, waveform } = call(exp::<u32, ()>, vec![seq]).unwrap() else {
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
            call(
                alt::<u32, ()>,
                vec![
                    Expr::Waveform(trigger),
                    Expr::Waveform(Waveform::Const(1.0)),
                    Expr::Waveform(Waveform::Time(())),
                ],
            )
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
        let result = call(chord::<u32, ()>, vec![List(vec![])]).unwrap();
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
        assert_eq!(float(call(sqrt, vec![traced_float(4.0)])), Some(2.0));
        assert_eq!(float(call(round, vec![traced_float(1.6)])), Some(2.0));
        assert_eq!(
            float(call(times, vec![traced_float(2.0), traced_float(3.0)])),
            Some(6.0)
        );
        let list = || List(vec![SourceExpr::float(1.0), SourceExpr::float(2.0)]);
        assert_eq!(float(call(nth, vec![traced_float(1.0), list()])), Some(2.0));
        let fixed = call(fixed, vec![List(vec![SourceExpr::from(traced_float(0.5))])]).unwrap();
        assert!(matches!(fixed, Expr::Waveform(Waveform::Fixed(v, ())) if v == [0.5]));
        for (compare, expected) in [
            (equals as BuiltinFn<Traced, ()>, false),
            (not_equals, true),
            (less_than, true),
            (less_than_equals, true),
            (greater_than, false),
            (greater_than_equals, false),
        ] {
            let result = call(compare, vec![traced_float(1.0), traced_float(2.0)]).unwrap();
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
        let Ok(BuiltIn { function, .. }) = apply(&function, vec![Expr::String("freq".to_string())])
        else {
            panic!("debug applied to label should be a BuiltIn");
        };
        // The value is handed back as it was, marks and all.
        let result = apply(&function, vec![traced_float(440.0)]).unwrap();
        assert!(matches!(result, Expr::Waveform(Waveform::Marked { .. })));
        assert_eq!(printed.lock().unwrap().as_slice(), ["[DEBUG] freq: 440"]);
    }

    #[test]
    fn a_traced_constant_count_unfolds() {
        let function = BuiltIn {
            name: "minus".to_string(),
            function: BuiltInFn::new(minus),
        };
        let result = call(unfold, vec![function, Expr::float(1.0), traced_float(2.0)]).unwrap();
        assert_eq!(format!("{}", result), "[1, -1]");
    }

    #[test]
    fn a_traced_constant_trigger_chooses_its_branch() {
        let result = call(
            alt::<Traced, ()>,
            vec![
                traced_float(1.0),
                Expr::float(2.0),
                Expr::Waveform(Waveform::Time(())),
            ],
        )
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
        let Seq { offset, .. } = call(followed_by, vec![seq(1.0), seq(2.0)]).unwrap() else {
            panic!("seq \\ seq should be a seq");
        };
        let Expr::Waveform(offset) = offset.expr else {
            panic!("offset should be a waveform");
        };
        let root = optimizer::first_root(&trace::strip_trace_marks(offset));
        assert!(matches!(root, Some(Waveform::Const(v)) if v == 3.0));
    }
}
