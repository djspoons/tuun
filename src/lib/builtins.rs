use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::optimizer;
use crate::parser;
use crate::parser::{BuiltInFn, Expr};
use crate::waveform::{Operator, Waveform};
use Expr::{Application, Bool, BuiltIn, Error, Float, List, Seq, Tuple};

fn unary_op<M>(
    mut arguments: Vec<Expr<M>>,
    name: String,
    float_op: fn(f32) -> f32,
    waveform_op: fn(Box<Waveform<M>>) -> Waveform<M>,
) -> Expr<M>
where
    M: Debug,
{
    if arguments.len() != 1 {
        return Error(format!("Expected one argument for {}", name));
    }
    match arguments.remove(0) {
        Float(a) => Expr::Float(float_op(a)),
        Expr::Waveform(a) => Expr::Waveform(waveform_op(Box::new(a))),
        a => Error(format!("Invalid argument for {}: {:?}", name, a)),
    }
}

fn binary_op<M>(
    mut arguments: Vec<Expr<M>>,
    name: String,
    float_op: fn(f32, f32) -> f32,
    waveform_op: fn(Box<Waveform<M>>, Box<Waveform<M>>) -> Waveform<M>,
) -> Expr<M>
where
    M: Debug,
{
    fn make_seq<M>(
        offset: Box<Expr<M>>,
        waveform_op: fn(Box<Waveform<M>>, Box<Waveform<M>>) -> Waveform<M>,
        a: Waveform<M>,
        b: Waveform<M>,
    ) -> Expr<M>
    where
        M: Debug,
    {
        Seq {
            offset,
            waveform: Box::new(Expr::Waveform(waveform_op(Box::new(a), Box::new(b)))),
        }
    }

    if arguments.len() != 2 {
        return Error(format!(
            "Expected two arguments for {}, got {:?}",
            name, arguments
        ));
    }
    match (arguments.remove(0), arguments.remove(0)) {
        (Float(a), Float(b)) => Expr::Float(float_op(a, b)),
        (Expr::Waveform(a), Expr::Waveform(b)) => {
            Expr::Waveform(waveform_op(Box::new(a), Box::new(b)))
        }
        (Expr::Waveform(a), Float(b)) => {
            Expr::Waveform(waveform_op(Box::new(a), Box::new(Waveform::Const(b))))
        }
        (Float(a), Expr::Waveform(b)) => {
            Expr::Waveform(waveform_op(Box::new(Waveform::Const(a)), Box::new(b)))
        }
        (Seq { offset, waveform }, Expr::Waveform(b)) => match *waveform {
            Expr::Waveform(a) => make_seq(offset, waveform_op, a, b),
            expr => Error(format!("Invalid argument to seq in {}: {:?}", name, expr)),
        },
        (Expr::Waveform(a), Seq { offset, waveform }) => match *waveform {
            Expr::Waveform(b) => make_seq(offset, waveform_op, a, b),
            expr => Error(format!("Invalid argument to seq in {}: {:?}", name, expr)),
        },
        (Seq { offset, waveform }, Float(b)) => match *waveform {
            Expr::Waveform(a) => make_seq(offset, waveform_op, a, Waveform::Const(b)),
            expr => Error(format!("Invalid argument to seq in {}: {:?}", name, expr)),
        },
        (Float(a), Seq { offset, waveform }) => match *waveform {
            Expr::Waveform(b) => make_seq(offset, waveform_op, Waveform::Const(a), b),
            expr => Error(format!("Invalid argument to seq in {}: {:?}", name, expr)),
        },
        (a, b) => Error(format!(
            "Invalid arguments for {}: {:?} and {:?}",
            name, a, b
        )),
    }
}

pub fn plus<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    return binary_op(arguments, "+".to_string(), std::ops::Add::add, |a, b| {
        Waveform::BinaryPointOp(Operator::Add, a, b)
    });
}

pub fn minus<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
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
    return binary_op(arguments, "-".to_string(), std::ops::Sub::sub, |a, b| {
        Waveform::BinaryPointOp(Operator::Subtract, a, b)
    });
}

pub fn times<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    return binary_op(arguments, "*".to_string(), std::ops::Mul::mul, |a, b| {
        Waveform::BinaryPointOp(Operator::Multiply, a, b)
    });
}

pub fn divide<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    binary_op(arguments, "/".to_string(), std::ops::Div::div, |a, b| {
        Waveform::BinaryPointOp(Operator::Divide, a, b)
    })
}

pub fn merge<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    // It seems strange to ever apply & to two floats, so just promote them
    // here.
    match &arguments[..] {
        [Float(a), Float(b)] => {
            return Expr::Waveform(Waveform::BinaryPointOp(
                Operator::Merge,
                Box::new(Waveform::Const(*a)),
                Box::new(Waveform::Const(*b)),
            ));
        }
        _ => (),
    }
    // Otherwise, just do the usual binary op thing.
    binary_op(
        arguments,
        "&".to_string(),
        |_, _| unreachable!("Should never reach the float/float case for merge"),
        |a, b| Waveform::BinaryPointOp(Operator::Merge, a, b),
    )
}

// Given waveforms that represent offsets (and assuming offset waveforms are of the form
// `Time ~+ w` or `Const(x)`) return a new waveform which represents the sum of those offsets.
fn add_offsets<M>(a: Waveform<M>, b: Waveform<M>) -> Expr<M>
where
    M: Clone + Debug + PartialEq,
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
            Expr::Waveform(Waveform::BinaryPointOp(
                Operator::Add,
                Box::new(Waveform::Time(())),
                Box::new(b),
            ))
        }
        (a_root, b_root) => Error(format!(
            "Cannot add offsets that are not linear functions of Time, got {:?} and {:?} for {:?} and {:?}",
            a_root, b_root, a, b
        )),
    }
}

pub fn followed_by<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Display + Clone + PartialEq,
{
    if arguments.len() != 2 {
        return Error("Expected two arguments to \\".to_string());
    }
    let (a_offset, a) = match arguments.remove(0) {
        Seq { offset, waveform } => {
            match (*offset, *waveform) {
                (Expr::Waveform(offset), Expr::Waveform(waveform)) => (offset, waveform),
                // We know that the arguments are values and Seq-as-a-value always has two waveforms.
                (err @ Error(_), _) => {
                    return err;
                }
                (_, err @ Error(_)) => {
                    return err;
                }
                _ => panic!("Found a non-Waveform element in a Seq value"),
            }
        }
        expr => {
            return Error(format!(
                "Expected seq as first argument to \\, got {}",
                expr
            ));
        }
    };

    match arguments.remove(0) {
        Float(b) => Expr::Waveform(Waveform::BinaryPointOp(
            Operator::Merge,
            Box::new(a),
            Box::new(Waveform::Append(
                Box::new(Waveform::Fin {
                    length: Box::new(a_offset),
                    waveform: Box::new(Waveform::Const(0.0)),
                }),
                Box::new(Waveform::Const(b)),
            )),
        )),
        Expr::Waveform(b) => Expr::Waveform(Waveform::BinaryPointOp(
            Operator::Merge,
            Box::new(a),
            Box::new(Waveform::Append(
                Box::new(Waveform::Fin {
                    length: Box::new(a_offset),
                    waveform: Box::new(Waveform::Const(0.0)),
                }),
                Box::new(b),
            )),
        )),
        Seq {
            offset: b_offset,
            waveform,
        } => {
            let (b_offset, b) = match (*b_offset, *waveform) {
                (Expr::Waveform(b_offset), Expr::Waveform(b)) => (b_offset, b),
                (err @ Error(_), _) => {
                    return err;
                }
                (_, err @ Error(_)) => {
                    return err;
                }
                _ => panic!("Found a non-Waveform element in a Seq value"),
            };
            let total_offset = add_offsets(a_offset.clone(), b_offset);
            Seq {
                offset: Box::new(total_offset),
                waveform: Box::new(Expr::Waveform(Waveform::BinaryPointOp(
                    Operator::Merge,
                    Box::new(a),
                    Box::new(Waveform::Append(
                        Box::new(Waveform::Fin {
                            length: Box::new(a_offset),
                            waveform: Box::new(Waveform::Const(0.0)),
                        }),
                        Box::new(b),
                    )),
                ))),
            }
        }
        expr => Error(format!(
            "Expected second argument to \\ to be a float, waveform or seq, got {}",
            expr
        )),
    }
}

pub fn power<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(base), Float(exponent)] => Expr::Float(base.powf(exponent)),
        _ => Error("Invalid arguments for power".to_string()),
    }
}

pub fn log<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(value), Float(base)] => Expr::Float(value.log(base)),
        _ => Error("Invalid arguments for log".to_string()),
    }
}

pub fn sqrt<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(value)] if value >= 0.0 => Expr::Float(value.sqrt()),
        _ => Error("Invalid argument for sqrt".to_string()),
    }
}

pub fn exp<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(value)] => Expr::Float(value.exp()),
        _ => Error("Invalid argument for exp".to_string()),
    }
}

pub fn sine<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone,
{
    // Like the waveform, Sine, the first argument is frequency in radians per
    // second, and the second is phase in radians.
    match &arguments[..] {
        [Float(frequency), Float(value)] if *frequency == 0.0 => Float(value.sin()),
        [Float(freq), Float(phase)] => Expr::Waveform(Waveform::Sine {
            frequency: Box::new(Waveform::Const(*freq)),
            phase: Box::new(Waveform::Const(*phase)),
            state: (),
        }),
        [Expr::Waveform(freq), Float(phase)] => Expr::Waveform(Waveform::Sine {
            frequency: Box::new(freq.clone()),
            phase: Box::new(Waveform::Const(*phase)),
            state: (),
        }),
        [Float(freq), Expr::Waveform(phase)] => Expr::Waveform(Waveform::Sine {
            frequency: Box::new(Waveform::Const(*freq)),
            phase: Box::new(phase.clone()),
            state: (),
        }),
        [Expr::Waveform(freq), Expr::Waveform(phase)] => Expr::Waveform(Waveform::Sine {
            frequency: Box::new(freq.clone()),
            phase: Box::new(phase.clone()),
            state: (),
        }),
        [_] => Error("Expected two arguments for sine".to_string()),
        _ => Error("Invalid arguments for sine".to_string()),
    }
}

// TODO: can this be moved to context?
pub fn cos<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone,
{
    match &arguments[..] {
        [Float(value)] => Expr::Float(value.cos()),
        [Expr::Waveform(a)] => Expr::Waveform(Waveform::Sine {
            frequency: Box::new(Waveform::Const(0.0)),
            phase: Box::new(Waveform::BinaryPointOp(
                Operator::Add,
                Box::new(a.clone()),
                Box::new(Waveform::Const(std::f32::consts::FRAC_PI_2)),
            )),
            state: (),
        }),
        // TODO handle other cases?
        _ => Error("Invalid argument for cos".to_string()),
    }
}

pub fn equals<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match &arguments[..] {
        [Bool(a), Bool(b)] => Expr::Bool(a == b),
        [Float(a), Float(b)] => Expr::Bool(a == b),
        [Expr::String(a), Expr::String(b)] => Expr::Bool(a == b),
        _ => Error("Invalid arguments for ==".to_string()),
    }
}

pub fn not_equals<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match &arguments[..] {
        [Bool(a), Bool(b)] => Expr::Bool(a != b),
        [Float(a), Float(b)] => Expr::Bool(a != b),
        [Expr::String(a), Expr::String(b)] => Expr::Bool(a != b),
        _ => Error("Invalid arguments for !=".to_string()),
    }
}

pub fn less_than<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a < b),
        _ => Error("Invalid arguments for <".to_string()),
    }
}

pub fn less_than_equals<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a <= b),
        _ => Error("Invalid arguments for <=".to_string()),
    }
}

pub fn greater_than<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a > b),
        _ => Error("Invalid arguments for >".to_string()),
    }
}

pub fn greater_than_equals<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a >= b),
        _ => Error("Invalid arguments for >=".to_string()),
    }
}

pub fn map<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Clone + Debug + Display,
{
    match &arguments[..] {
        [function, List(exprs)] => {
            let mut results = Vec::new();
            let context = vec![];
            for expr in exprs {
                let result = parser::evaluate(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        argument: Box::new(expr.clone()), // can we avoid this clone?
                    },
                );
                match result {
                    Ok(expr) => results.push(expr),
                    Err(err) => results.push(Error(err.to_string())),
                }
            }
            Expr::List(results)
        }
        _ => Error("Invalid arguments for map".to_string()),
    }
}

pub fn reduce<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display,
{
    match &arguments[..] {
        [function, acc, List(exprs)] => {
            let context = vec![];
            let mut acc = acc.clone();
            for expr in exprs {
                let result = parser::evaluate(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        argument: Box::new(Tuple(vec![acc, expr.clone()])), // can we avoid this clone?
                    },
                );
                acc = match result {
                    Ok(expr) => expr,
                    Err(err) => return Error(err.to_string()),
                };
            }
            acc
        }
        _ => Error("Invalid arguments for reduce".to_string()),
    }
}

pub fn unfold<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display,
{
    match &arguments[..] {
        [function, seed, Float(n)] if *n >= 0.0 && n.fract() == 0.0 => {
            let context = vec![];
            let mut results = Vec::new();
            let mut current = seed.clone();
            for _ in 0..(*n as u32) {
                results.push(current.clone());
                let result = parser::evaluate(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        argument: Box::new(current.clone()),
                    },
                );
                current = match result {
                    Ok(expr) => expr,
                    Err(err) => return Error(err.to_string()),
                };
            }
            Expr::List(results)
        }
        _ => Error("Invalid arguments for unfold".to_string()),
    }
}

// Appends two or more lists or waveforms together
pub fn append<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone,
{
    match &arguments[..] {
        [List(a), rest @ ..] => {
            let mut result = a.clone();
            for b in rest {
                if let List(exprs) = b {
                    result.extend(exprs.clone());
                } else {
                    return Error("Expected more lists as arguments for append".to_string());
                }
            }
            Expr::List(result)
        }
        [Expr::Waveform(a), rest @ ..] => {
            let mut result = a.clone();
            for b in rest {
                if let Expr::Waveform(b) = b {
                    result = Waveform::Append(Box::new(result), Box::new(b.clone()));
                } else {
                    return Error("Expected more waveforms as arguments for append".to_string());
                }
            }
            Expr::Waveform(result)
        }
        _ => Error("Invalid arguments for append".to_string()),
    }
}

pub fn fixed<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    match &arguments[..] {
        [List(samples)] => {
            let mut fixed_samples = Vec::new();
            for sample in samples {
                match sample {
                    Float(value) => fixed_samples.push(*value),
                    _ => return Error("Invalid sample in fixed waveform".to_string()),
                }
            }
            Expr::Waveform(Waveform::Fixed(fixed_samples, ()))
        }
        _ => Error("Invalid argument for fixed waveform".to_string()),
    }
}

pub fn curry<M>(f: impl Fn(Box<Waveform<M>>) -> Waveform<M> + 'static) -> BuiltInFn<M>
where
    M: Display,
{
    BuiltInFn(Rc::new(move |mut arguments: Vec<Expr<M>>| -> Expr<M> {
        if arguments.len() != 1 {
            return Error("Expected waveform".to_string());
        }
        let waveform = arguments.remove(0);
        match waveform {
            Expr::Waveform(a) => Expr::Waveform(f(Box::new(a))),
            Expr::Float(value) => Expr::Waveform(f(Box::new(Waveform::Const(value)))),
            Seq { offset, waveform } => match *waveform {
                Expr::Waveform(waveform) => Seq {
                    offset,
                    waveform: Box::new(Expr::Waveform(f(Box::new(waveform)))),
                },
                expr => Error(format!(
                    "Expected waveform as argument to seq, got {}",
                    expr
                )),
            },
            expr => Error(format!("Expected waveform, seq, or float, got {}", expr)),
        }
    }))
}

pub fn fin<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display + 'static,
{
    if arguments.len() != 1 {
        return Error(format!(
            "Expected one argument for fin, got {}",
            arguments.len()
        ));
    }
    let arg = arguments.remove(0);
    match arg {
        // Note that we are treating floats here as a const waveform, not a duration
        Expr::Float(f) => {
            let w = Waveform::Const(f);
            BuiltIn {
                name: format!("fin({})", w),
                function: curry(move |waveform: Box<Waveform<M>>| Waveform::Fin {
                    length: Box::new(w.clone()),
                    waveform,
                }),
            }
        }
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
        _ => Error("Invalid arguments for fin".to_string()),
    }
}

pub fn seq<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display + 'static,
{
    if arguments.len() != 1 {
        return Expr::Error(format!(
            "Expected one argument for seq, got {}",
            arguments.len()
        ));
    }
    let offset = match arguments.remove(0) {
        Expr::Waveform(offset) => offset,
        Expr::Float(value) => Waveform::Const(value),
        expr => {
            return Expr::Error(format!("Invalid argument for seq: {}", expr));
        }
    };
    let name = format!("seq({})", offset);
    BuiltIn {
        name,
        function: BuiltInFn(Rc::new(move |mut arguments: Vec<Expr<M>>| -> Expr<M> {
            let offset = offset.clone();
            if arguments.len() != 1 {
                return Error(format!(
                    "Expected one argument for seq({}), got {}",
                    offset,
                    arguments.len()
                ));
            }
            match arguments.remove(0) {
                Expr::Waveform(waveform) => Seq {
                    offset: Box::new(Expr::Waveform(offset)),
                    waveform: Box::new(Expr::Waveform(waveform)),
                },
                Expr::Float(f) => Seq {
                    offset: Box::new(Expr::Waveform(offset)),
                    waveform: Box::new(Expr::Waveform(Waveform::Const(f))),
                },
                expr => Error(format!(
                    "Expected argument to seq({}) to be a waveform or float, got {}",
                    offset, expr
                )),
            }
        })),
    }
}

pub fn unseq<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + 'static,
{
    if arguments.len() != 0 {
        return Error(format!(
            "Expected no arguments for unseq, got {}",
            arguments.len()
        ));
    }
    Expr::BuiltIn {
        name: format!("unseq()"),
        function: BuiltInFn(Rc::new(|mut arguments: Vec<Expr<M>>| -> Expr<M> {
            if arguments.len() != 1 {
                return Error(format!(
                    "Expected argument for unseq(), got {}",
                    arguments.len()
                ));
            }
            match arguments.remove(0) {
                Seq { waveform, .. } => *waveform,
                _ => Error("Expected seq as argument to unseq".to_string()),
            }
        })),
    }
}

/*
// TODO reconsider this: maybe this doesn't make sense any more... or the right argument
// needs to be a list of waveforms
pub fn waveform_convolution(mut arguments: Vec<Expr<M>>) -> Expr<M> where M: Debug {
    if arguments.len() != 2 {
        return Error(format!("Expected two arguments for ~*"));
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
                _ => return Error("Invalid arguments for ~*".to_string()),
            };
            Expr::Waveform(waveform)
        }
    }
}
*/

pub fn waveform_filter<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display + 'static,
{
    if arguments.len() != 2 {
        return Error("Expected two lists of waveforms for filter".to_string());
    }
    let feed_forward = match arguments.remove(0) {
        Expr::List(exprs) => {
            if exprs.len() == 0 {
                return Error("Filter requires at least one feed-forward coefficient".to_string());
            }
            let mut feed_forward = Vec::with_capacity(exprs.len());
            for expr in exprs {
                match expr {
                    Expr::Waveform(a) => feed_forward.push(a),
                    Float(value) => feed_forward.push(Waveform::Const(value)),
                    _ => {
                        return Error("Filter feed_forward argument must be a list".to_string());
                    }
                }
            }
            feed_forward
        }
        _ => return Error("Feed-forward argument to filter must be a list".to_string()),
    };
    let feedback = match arguments.remove(0) {
        Expr::List(exprs) => {
            let mut feedback = Vec::with_capacity(exprs.len());
            for expr in exprs {
                match expr {
                    Expr::Waveform(a) => feedback.push(a),
                    Float(value) => feedback.push(Waveform::Const(value)),
                    _ => return Error("Filter feedback argument must be a list".to_string()),
                }
            }
            feedback
        }
        _ => return Error("Feedback argument to filter must be a list".to_string()),
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
            waveform: waveform,
            feed_forward: feed_forward.clone(),
            feedback: feedback.clone(),
            state: (),
        }),
    }
}

pub fn reset<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    // TODO make it work in curried form?
    if arguments.len() != 2 {
        return Error("Expected two waveforms".to_string());
    }
    let trigger = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Error("First argument must be a waveform".to_string()),
    };
    let waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Error("Second argument must be a waveform or a float".to_string()),
    };
    Expr::Waveform(Waveform::Reset {
        trigger: Box::new(trigger),
        waveform: Box::new(waveform),
        state: (),
    })
}

pub fn alt<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug,
{
    // TODO make it work in curried form?
    if arguments.len() != 3 {
        return Error("Expected three waveforms".to_string());
    }
    let trigger = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Error("First argument must be a waveform".to_string()),
    };
    let positive_waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Error("Second argument must be a waveform or a float".to_string()),
    };
    let negative_waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Error("Third argument must be a waveform or a float".to_string()),
    };
    Expr::Waveform(Waveform::Alt {
        trigger: Box::new(trigger),
        positive_waveform: Box::new(positive_waveform),
        negative_waveform: Box::new(negative_waveform),
    })
}

// TODO move to main? (Because it only works in the native app anyway...?)
fn capture<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Display + 'static,
{
    if arguments.len() != 1 {
        return Error("Expected one argument for capture".to_string());
    }
    let file_stem = match arguments.remove(0) {
        Expr::String(file_stem) => file_stem,
        _ => return Error("Expected a string argument to capture".to_string()),
    };
    BuiltIn {
        name: format!("capture({})", file_stem),
        function: curry(move |waveform: Box<Waveform<M>>| Waveform::Captured {
            file_stem: file_stem.clone(),
            waveform,
        }),
    }
}

pub fn chord<M>(arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display,
{
    match &arguments[..] {
        [List(exprs)] => {
            let mut result = Waveform::Fin {
                length: Box::new(Waveform::Const(0.0)),
                waveform: Box::new(Waveform::Const(0.0)),
            };
            for expr in exprs.iter().rev() {
                let waveform: Box<Waveform<M>> = match expr {
                    Expr::Waveform(waveform) => Box::new(waveform.clone()),
                    Expr::Float(value) => Box::new(Waveform::Const(*value)),
                    _ => return Error(format!("Invalid element in chord: {}", expr)),
                };
                result = Waveform::BinaryPointOp(Operator::Merge, waveform, Box::new(result));
            }
            return Expr::Waveform(result);
        }
        _ => return Error("Invalid argument for chord".to_string()),
    }
}

pub fn sequence<M>(mut arguments: Vec<Expr<M>>) -> Expr<M>
where
    M: Debug + Clone + Display + PartialEq + 'static,
{
    if arguments.len() != 1 {
        return Error("Invalid argument for sequence".to_string());
    }
    match &mut arguments[0] {
        List(exprs) => {
            if exprs.len() == 0 {
                return Expr::Waveform(Waveform::Fixed(vec![], ()));
            } else if exprs.len() == 1 {
                return match &exprs[0] {
                    Expr::Waveform(waveform) => Expr::Waveform(waveform.clone()),
                    Expr::Float(value) => Expr::Waveform(Waveform::Const(*value)),
                    _ => return Error("Invalid argument for sequence".to_string()),
                };
            }
            let mut result = exprs.remove(exprs.len() - 1);
            while !exprs.is_empty() {
                result = followed_by(vec![exprs.remove(exprs.len() - 1), result]);
            }
            return result;
        }
        _ => return Error("Invalid argument for sequence".to_string()),
    }
}

pub fn add_prelude<M: Debug + Clone + Display + PartialEq + 'static>(
    context: &mut Vec<(String, Expr<M>)>,
) {
    context.push(("true".to_string(), Expr::Bool(true)));
    context.push(("false".to_string(), Expr::Bool(false)));
    context.push(("time".to_string(), Expr::Waveform(Waveform::Time(()))));
    context.push(("noise".to_string(), Expr::Waveform(Waveform::Noise)));

    let builtins: Vec<(&str, fn(Vec<Expr<M>>) -> Expr<M>)> = vec![
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
        ("fixed", fixed),
        ("fin", fin),
        ("seq", seq),
        ("unseq", unseq),
        ("filter", waveform_filter),
        //("~*", waveform_convolution),
        ("reset", reset),
        ("alt", alt),
        ("capture", capture),
        ("_chord", chord),
        ("_sequence", sequence),
    ];
    for (name, function) in builtins {
        context.push((
            name.to_string(),
            Expr::BuiltIn {
                name: name.to_string(),
                function: BuiltInFn(Rc::new(function)),
            },
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Expr::BuiltIn;

    #[test]
    fn test_map() {
        let exprs: Vec<Expr<u32>> = vec![Float(2.0), Float(3.0), Float(4.0)];
        let result = map(vec![
            BuiltIn {
                name: "minus".to_string(),
                function: BuiltInFn(Rc::new(minus)),
            },
            List(exprs),
        ]);
        assert_eq!(format!("{}", result), "[-2, -3, -4]");
    }

    #[test]
    fn test_reduce() {
        let exprs: Vec<Expr<u32>> = vec![Float(2.0), Float(3.0), Float(4.0)];
        let result = reduce(vec![
            BuiltIn {
                name: "plus".to_string(),
                function: BuiltInFn(Rc::new(plus)),
            },
            Float(1.0),
            List(exprs),
        ]);
        assert_eq!(format!("{}", result), "10");
    }
}
