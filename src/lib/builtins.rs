use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::eval;
use crate::expr;
use crate::expr::{BuiltInFn, Error, Expr, SourceExpr, boxed};
use crate::optimizer;
use crate::waveform::{Operator, Waveform};
use Expr::{Bool, BuiltIn, List, Seq};

type WaveformBinOp<M> = fn(Box<Waveform<M>>, Box<Waveform<M>>) -> Waveform<M>;

type BuiltinFn<M, S> = fn(Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>;

fn unary_op<M, S>(
    mut arguments: Vec<Expr<M, S>>,
    name: String,
    float_op: fn(f32) -> f32,
    waveform_op: fn(Box<Waveform<M>>) -> Waveform<M>,
) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    Ok({
        if arguments.len() != 1 {
            return Err(Error::internal_here(format!(
                "Expected one argument for {}",
                name
            )));
        }
        match arguments.remove(0) {
            // A constant folds eagerly, so arithmetic on constants stays
            // constant (see `Expr::as_const_float`).
            Expr::Waveform(Waveform::Const(a)) => Expr::float(float_op(a)),
            Expr::Waveform(a) => Expr::Waveform(waveform_op(Box::new(a))),
            a => {
                return Err(Error::internal_here(format!(
                    "Invalid argument for {}: {:?}",
                    name, a
                )));
            }
        }
    })
}

// TODO maybe use Display instead of Debug for errors?
fn binary_op<M, S>(
    mut arguments: Vec<Expr<M, S>>,
    name: String,
    fold: fn(f32, f32) -> f32,
    waveform_op: WaveformBinOp<M>,
) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    Ok({
        fn make_seq<M, S>(
            offset: Box<SourceExpr<M, S>>,
            waveform_op: WaveformBinOp<M>,
            a: Waveform<M>,
            b: Waveform<M>,
        ) -> Expr<M, S>
        where
            M: Debug,
        {
            Seq {
                offset,
                waveform: boxed(Expr::Waveform(waveform_op(Box::new(a), Box::new(b)))),
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
            return Ok(Expr::float(fold(a, b)));
        }
        match (x, y) {
            (Expr::Waveform(a), Expr::Waveform(b)) => {
                Expr::Waveform(waveform_op(Box::new(a), Box::new(b)))
            }
            (Seq { offset, waveform }, Expr::Waveform(b)) => match waveform.expr {
                Expr::Waveform(a) => make_seq(offset, waveform_op, a, b),
                expr => {
                    return Err(Error::internal_here(format!(
                        "Invalid argument to seq in {}: {:?}",
                        name, expr
                    )));
                }
            },
            (Expr::Waveform(a), Seq { offset, waveform }) => match waveform.expr {
                Expr::Waveform(b) => make_seq(offset, waveform_op, a, b),
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
    M: Debug,
    S: Debug,
{
    binary_op(arguments, "+".to_string(), std::ops::Add::add, |a, b| {
        Waveform::BinaryPointOp(Operator::Add, a, b)
    })
}

pub fn minus<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    if arguments.len() == 1 {
        return unary_op(
            arguments,
            "-".to_string(),
            |a| -a,
            |waveform| {
                Waveform::BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Waveform::Const(-1.0)),
                    waveform,
                )
            },
        );
    }
    binary_op(arguments, "-".to_string(), std::ops::Sub::sub, |a, b| {
        Waveform::BinaryPointOp(Operator::Subtract, a, b)
    })
}

pub fn times<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    binary_op(arguments, "*".to_string(), std::ops::Mul::mul, |a, b| {
        Waveform::BinaryPointOp(Operator::Multiply, a, b)
    })
}

pub fn divide<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    binary_op(arguments, "/".to_string(), std::ops::Div::div, |a, b| {
        Waveform::BinaryPointOp(Operator::Divide, a, b)
    })
}

pub fn merge<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    binary_op(arguments, "&".to_string(), std::ops::Add::add, |a, b| {
        Waveform::BinaryPointOp(Operator::Merge, a, b)
    })
}

// Given waveforms that represent offsets (and assuming offset waveforms are of the form
// `Time ~+ w` or `Const(x)`) return a new waveform which represents the sum of those offsets.
fn add_offsets<M, S>(a: Waveform<M>, b: Waveform<M>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone + Debug + PartialEq,
    S: Clone + Debug,
{
    match (optimizer::first_root(&a), optimizer::first_root(&b)) {
        (Some(a_root), Some(b_root)) => {
            let b = optimizer::optimize(Waveform::BinaryPointOp(
                Operator::Multiply,
                Box::new(Waveform::BinaryPointOp(
                    Operator::Add,
                    Box::new(a_root),
                    Box::new(b_root),
                )),
                Box::new(Waveform::Const(-1.0)),
            ));
            Ok(Expr::Waveform(Waveform::BinaryPointOp(
                Operator::Add,
                Box::new(Waveform::Time(())),
                Box::new(b),
            )))
        }
        (a_root, b_root) => Err(Error::internal_here(format!(
            "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
            a_root, b_root, a, b
        ))),
    }
}

pub fn followed_by<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Display + Clone + PartialEq,
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
            Expr::Waveform(b) => Expr::Waveform(Waveform::BinaryPointOp(
                Operator::Merge,
                Box::new(a),
                Box::new(Waveform::Append(
                    Box::new(Waveform::Fin {
                        length: Box::new(a_offset),
                        waveform: Box::new(Waveform::Const(0.0)),
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
                    waveform: boxed(Expr::Waveform(Waveform::BinaryPointOp(
                        Operator::Merge,
                        Box::new(a),
                        Box::new(Waveform::Append(
                            Box::new(Waveform::Fin {
                                length: Box::new(a_offset),
                                waveform: Box::new(Waveform::Const(0.0)),
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
    M: Debug,
    S: Debug,
{
    binary_op(arguments, "pow".to_string(), f32::powf, |a, b| {
        Waveform::BinaryPointOp(Operator::Power, a, b)
    })
}

pub fn log<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    Ok({
        match arguments[..] {
            [
                Expr::Waveform(Waveform::Const(value)),
                Expr::Waveform(Waveform::Const(base)),
            ] => Expr::float(value.log(base)),
            _ => return Err(Error::internal_here("Invalid arguments for log")),
        }
    })
}

pub fn sqrt<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
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

pub fn exp<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
    Ok({
        match arguments[..] {
            [Expr::Waveform(Waveform::Const(value))] => Expr::float(value.exp()),
            _ => return Err(Error::internal_here("Invalid argument for exp")),
        }
    })
}

pub fn sine<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone,
    S: Clone + Debug,
{
    Ok({
        // Like the waveform, Sine, the first argument is frequency in radians per
        // second, and the second is phase in radians.
        match &arguments[..] {
            // A zero-frequency sine of a constant phase folds to a constant.
            [
                Expr::Waveform(Waveform::Const(frequency)),
                Expr::Waveform(Waveform::Const(phase)),
            ] if *frequency == 0.0 => Expr::float(phase.sin()),
            [Expr::Waveform(freq), Expr::Waveform(phase)] => Expr::Waveform(Waveform::Sine {
                frequency: Box::new(freq.clone()),
                phase: Box::new(phase.clone()),
                state: (),
            }),
            _ => return Err(Error::internal_here("Invalid arguments for sine")),
        }
    })
}

// TODO: can this be moved to std?
pub fn cos<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [Expr::Waveform(Waveform::Const(value))] => Expr::float(value.cos()),
            [Expr::Waveform(a)] => Expr::Waveform(Waveform::Sine {
                frequency: Box::new(Waveform::Const(0.0)),
                phase: Box::new(Waveform::BinaryPointOp(
                    Operator::Add,
                    Box::new(a.clone()),
                    Box::new(Waveform::Const(std::f32::consts::FRAC_PI_2)),
                )),
                state: (),
            }),
            _ => return Err(Error::internal_here("Invalid argument for cos")),
        }
    })
}

pub fn equals<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug,
    S: Debug,
{
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
    M: Debug,
    S: Debug,
{
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
    M: Debug,
    S: Debug,
{
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
    M: Debug,
    S: Debug,
{
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
    M: Debug,
    S: Debug,
{
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
    M: Debug,
    S: Debug,
{
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
    M: Clone + Debug + Display,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [function, List(exprs)] => {
                let mut results: Vec<SourceExpr<M, S>> = Vec::new();
                for expr in exprs {
                    let result = eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![expr.clone()], // can we avoid this clone?
                    ));
                    match result {
                        Ok(expr) => results.push(expr),
                        Err(err) => return Err(Error::internal_here(err.to_string())),
                    }
                }
                Expr::List(results)
            }
            _ => return Err(Error::internal_here("Invalid arguments for map")),
        }
    })
}

pub fn reduce<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [function, acc, List(exprs)] => {
                let mut acc: SourceExpr<M, S> = SourceExpr::from(acc.clone());
                for expr in exprs {
                    let result = eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![acc, expr.clone()],
                    ));
                    acc = match result {
                        Ok(expr) => expr,
                        Err(err) => return Err(Error::internal_here(err.to_string())),
                    };
                }
                acc.expr
            }
            _ => return Err(Error::internal_here("Invalid arguments for reduce")),
        }
    })
}

pub fn unfold<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display,
    S: Clone + Debug,
{
    Ok({
        match &arguments[..] {
            [function, seed, Expr::Waveform(Waveform::Const(n))]
                if *n >= 0.0 && n.fract() == 0.0 =>
            {
                let mut results: Vec<SourceExpr<M, S>> = Vec::new();
                let mut current: SourceExpr<M, S> = SourceExpr::from(seed.clone());
                for _ in 0..(*n as u32) {
                    results.push(current.clone());
                    let result = eval::evaluate_closed(SourceExpr::application(
                        function.clone().into(),
                        vec![current.clone()],
                    ));
                    current = match result {
                        Ok(expr) => expr,
                        Err(err) => return Err(Error::internal_here(err.to_string())),
                    };
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

pub fn nth<M, S>(arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Clone,
    S: Clone,
{
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
    M: Debug,
    S: Debug,
{
    Ok({
        match &arguments[..] {
            [List(samples)] => {
                let mut fixed_samples = Vec::new();
                for sample in samples {
                    match &sample.expr {
                        Expr::Waveform(Waveform::Const(value)) => fixed_samples.push(*value),
                        _ => return Err(Error::internal_here("Invalid sample in fixed waveform")),
                    }
                }
                Expr::Waveform(Waveform::Fixed(fixed_samples, ()))
            }
            _ => return Err(Error::internal_here("Invalid argument for fixed waveform")),
        }
    })
}

pub fn curry<M, S>(f: impl Fn(Box<Waveform<M>>) -> Waveform<M> + 'static) -> BuiltInFn<M, S>
where
    M: Display,
{
    BuiltInFn(Rc::new(
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
    ))
}

pub fn fin<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + 'static,
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
                    function: curry(move |waveform: Box<Waveform<M>>| Waveform::Fin {
                        length: Box::new(length.clone()),
                        waveform,
                    }),
                }
            }
            _ => return Err(Error::internal_here("Invalid arguments for fin")),
        }
    })
}

pub fn seq<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + 'static,
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
            function: BuiltInFn(Rc::new(
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
            )),
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
            function: BuiltInFn(Rc::new(
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
            )),
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
    M: Debug + Clone + Display + 'static,
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
            function: curry(move |waveform: Box<Waveform<M>>| Waveform::Filter {
                waveform,
                feed_forward: feed_forward.clone(),
                feedback: feedback.clone(),
                state: (),
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
    M: Debug,
    S: Debug,
{
    Ok({
        // TODO make it work in curried form?
        if arguments.len() != 3 {
            return Err(Error::internal_here("Expected three waveforms"));
        }
        let trigger = match arguments.remove(0) {
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
        Expr::Waveform(Waveform::Alt {
            trigger: Box::new(trigger),
            positive_waveform: Box::new(positive_waveform),
            negative_waveform: Box::new(negative_waveform),
        })
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
            function: curry(move |waveform: Box<Waveform<M>>| Waveform::Captured {
                file_stem: file_stem.clone(),
                waveform,
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
                let mut result = Waveform::Fin {
                    length: Box::new(Waveform::Const(0.0)),
                    waveform: Box::new(Waveform::Const(0.0)),
                };
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
                    result = Waveform::BinaryPointOp(Operator::Merge, waveform, Box::new(result));
                }
                Expr::Waveform(result)
            }
            _ => return Err(Error::internal_here("Invalid argument for chord")),
        }
    })
}

pub fn sequence<M, S>(mut arguments: Vec<Expr<M, S>>) -> Result<Expr<M, S>, Error<S>>
where
    M: Debug + Clone + Display + PartialEq + 'static,
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

/// Builds the `debug` built-in: applied, it renders its arguments as a line
/// `debug: [a, b, ...]`, hands the line to `print`, and evaluates to its last
/// argument (or an empty list when given none).
///
/// Passing the last value through lets a call wrap any sub-expression without
/// changing what it evaluates to; earlier arguments can serve as labels.
/// `print` supplies the caller's logging sink (terminal, browser console, ...).
///
/// # Example
/// ```tuun
/// sine(debug("freq", freq), 0)   // logs `debug: [freq, 440]`, plays sine(freq, 0)
/// ```
pub fn debug<M, S>(print: impl Fn(&str) + 'static) -> SourceExpr<M, S>
where
    M: Display + 'static,
    S: 'static,
{
    SourceExpr::from(Expr::BuiltIn {
        name: "debug".to_string(),
        function: BuiltInFn(Rc::new(move |mut arguments: Vec<Expr<M, S>>| {
            let rendered = arguments
                .iter()
                .map(|argument| argument.to_string())
                .collect::<Vec<_>>()
                .join(", ");
            print(&format!("debug: [{}]", rendered));
            Ok(arguments.pop().unwrap_or(Expr::List(Vec::new())))
        })),
    })
}

/// Adds all of the built-ins to `bindings`.
pub fn add_bindings<M, S>(bindings: &mut Vec<expr::SourceBinding<M, S>>)
where
    M: Debug + Clone + Display + PartialEq + 'static,
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
        ("sine", sine),
        ("cos", cos),
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
                function: BuiltInFn(Rc::new(function)),
            }),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ErrorKind;
    use Expr::BuiltIn;

    #[test]
    fn test_debug() {
        use std::cell::RefCell;
        let printed: Rc<RefCell<Vec<String>>> = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&printed);
        let debug = debug::<u32, ()>(move |line| sink.borrow_mut().push(line.to_string()));
        let Expr::BuiltIn { function, .. } = debug.expr else {
            panic!("debug should build a BuiltIn");
        };

        // Logs all arguments, evaluates to the last.
        let result = function.0(vec![Expr::String("freq".to_string()), Expr::float(440.0)]);
        assert_eq!(format!("{}", result.clone().unwrap()), "440");
        assert_eq!(printed.borrow().as_slice(), ["debug: [freq, 440]"]);

        // No arguments: logs an empty list and evaluates to one.
        let result = function.0(vec![]);
        assert_eq!(format!("{}", result.clone().unwrap()), "[]");
        assert_eq!(printed.borrow().last().unwrap(), "debug: []");
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
                function: BuiltInFn(Rc::new(minus)),
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
                function: BuiltInFn(Rc::new(plus)),
            },
            Expr::float(1.0),
            List(exprs),
        ]);
        assert_eq!(format!("{}", result.clone().unwrap()), "10");
    }
}
