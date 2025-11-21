use std::rc::Rc;

use crate::parser::{BuiltInFn, Expr, simplify};
use crate::waveform::{Operator, Slider, Waveform};
use Expr::{Application, Bool, BuiltIn, Error, Float, List, Tuple};

fn unary_op(
    mut arguments: Vec<Expr>,
    name: String,
    float_op: fn(f32) -> f32,
    waveform_op: fn(Box<Waveform>) -> Waveform,
) -> Expr {
    if arguments.len() != 1 {
        return Expr::Error(format!("Expected one argument for {}", name));
    }
    match arguments.remove(0) {
        Float(a) => Expr::Float(float_op(a)),
        Expr::Waveform(a) => Expr::Waveform(waveform_op(Box::new(a))),
        a => Expr::Error(format!("Invalid argument for {}: {:?}", name, a)),
    }
}

fn binary_op(
    mut arguments: Vec<Expr>,
    name: String,
    float_op: fn(f32, f32) -> f32,
    waveform_op: fn(Box<Waveform>, Box<Waveform>) -> Waveform,
) -> Expr {
    if arguments.len() != 2 {
        return Expr::Error(format!("Expected two arguments for {}", name));
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
        (a, b) => Expr::Error(format!(
            "Invalid arguments for {}: {:?} and {:?}",
            name, a, b
        )),
    }
}

pub fn plus(arguments: Vec<Expr>) -> Expr {
    return binary_op(arguments, "+".to_string(), std::ops::Add::add, |a, b| {
        Waveform::BinaryPointOp(Operator::Add, a, b)
    });
}

pub fn minus(arguments: Vec<Expr>) -> Expr {
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

pub fn times(arguments: Vec<Expr>) -> Expr {
    return binary_op(arguments, "*".to_string(), std::ops::Mul::mul, |a, b| {
        Waveform::BinaryPointOp(Operator::Multiply, a, b)
    });
}

pub fn divide(arguments: Vec<Expr>) -> Expr {
    binary_op(arguments, "/".to_string(), std::ops::Div::div, |a, b| {
        Waveform::BinaryPointOp(Operator::Divide, a, b)
    })
}

pub fn power(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(base), Float(exponent)] => Expr::Float(base.powf(exponent)),
        _ => Expr::Error("Invalid arguments for power".to_string()),
    }
}

pub fn sqrt(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(value)] if value >= 0.0 => Expr::Float(value.sqrt()),
        _ => Expr::Error("Invalid argument for sqrt".to_string()),
    }
}

pub fn exp(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(value)] => Expr::Float(value.exp()),
        _ => Expr::Error("Invalid argument for exp".to_string()),
    }
}

pub fn sin(arguments: Vec<Expr>) -> Expr {
    unary_op(arguments, "sin".to_string(), f32::sin, |waveform| {
        Waveform::Sin(waveform, ())
    })
}

pub fn cos(arguments: Vec<Expr>) -> Expr {
    // TODO use unary_op
    match &arguments[..] {
        [Float(value)] => Expr::Float(value.cos()),
        [Expr::Waveform(a)] => Expr::Waveform(Waveform::Sin(
            Box::new(Waveform::BinaryPointOp(
                Operator::Add,
                Box::new(a.clone()),
                Box::new(Waveform::Const(std::f32::consts::FRAC_PI_2)),
            )),
            (),
        )),
        _ => Expr::Error("Invalid argument for cos".to_string()),
    }
}

pub fn equals(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Bool(a), Bool(b)] => Expr::Bool(a == b),
        [Float(a), Float(b)] => Expr::Bool(a == b),
        [Expr::String(a), Expr::String(b)] => Expr::Bool(a == b),
        _ => Expr::Error("Invalid arguments for ==".to_string()),
    }
}

pub fn not_equals(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Bool(a), Bool(b)] => Expr::Bool(a != b),
        [Float(a), Float(b)] => Expr::Bool(a != b),
        [Expr::String(a), Expr::String(b)] => Expr::Bool(a != b),
        _ => Expr::Error("Invalid arguments for !=".to_string()),
    }
}

pub fn less_than(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a < b),
        _ => Expr::Error("Invalid arguments for <".to_string()),
    }
}

pub fn less_than_equals(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a <= b),
        _ => Expr::Error("Invalid arguments for <=".to_string()),
    }
}

pub fn greater_than(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a > b),
        _ => Expr::Error("Invalid arguments for >".to_string()),
    }
}

pub fn greater_than_equals(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Bool(a >= b),
        _ => Expr::Error("Invalid arguments for >=".to_string()),
    }
}

pub fn map(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [function, List(exprs)] => {
            let mut results = Vec::new();
            let context = vec![];
            for expr in exprs {
                let result = simplify(
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
        _ => Expr::Error("Invalid arguments for map".to_string()),
    }
}

pub fn reduce(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [function, acc, List(exprs)] => {
            let context = vec![];
            let mut acc = acc.clone();
            for expr in exprs {
                let result = simplify(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        argument: Box::new(Tuple(vec![acc, expr.clone()])), // can we avoid this clone?
                    },
                );
                acc = match result {
                    Ok(expr) => expr,
                    Err(err) => return Expr::Error(err.to_string()),
                };
            }
            acc
        }
        _ => Expr::Error("Invalid arguments for reduce".to_string()),
    }
}

pub fn unfold(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [function, seed, Float(n)] if *n >= 0.0 && n.fract() == 0.0 => {
            let context = vec![];
            let mut results = Vec::new();
            let mut current = seed.clone();
            for _ in 0..(*n as u32) {
                results.push(current.clone());
                let result = simplify(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        argument: Box::new(current.clone()),
                    },
                );
                current = match result {
                    Ok(expr) => expr,
                    Err(err) => return Expr::Error(err.to_string()),
                };
            }
            Expr::List(results)
        }
        _ => Expr::Error("Invalid arguments for unfold".to_string()),
    }
}

// Appends two or more lists or waveforms together
pub fn append(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(a), rest @ ..] => {
            let mut result = a.clone();
            for b in rest {
                if let List(exprs) = b {
                    result.extend(exprs.clone());
                } else {
                    return Expr::Error("Expected more lists as arguments for append".to_string());
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
                    return Expr::Error(
                        "Expected more waveforms as arguments for append".to_string(),
                    );
                }
            }
            Expr::Waveform(result)
        }
        _ => Expr::Error("Invalid arguments for append".to_string()),
    }
}

pub fn fixed(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(samples)] => {
            let mut fixed_samples = Vec::new();
            for sample in samples {
                match sample {
                    Float(value) => fixed_samples.push(*value),
                    _ => return Expr::Error("Invalid sample in fixed waveform".to_string()),
                }
            }
            Expr::Waveform(Waveform::Fixed(fixed_samples))
        }
        _ => Expr::Error("Invalid argument for fixed waveform".to_string()),
    }
}

fn curry(f: impl Fn(Box<Waveform>) -> Waveform + 'static) -> BuiltInFn {
    BuiltInFn(Rc::new(move |mut arguments: Vec<Expr>| -> Expr {
        if arguments.len() != 1 {
            return Expr::Error("Expected waveform".to_string());
        }
        let waveform = arguments.remove(0);
        match waveform {
            Expr::Waveform(a) => Expr::Waveform(f(Box::new(a))),
            Expr::Float(value) => Expr::Waveform(f(Box::new(Waveform::Const(value)))),
            _ => Expr::Error("Invalid waveform".to_string()),
        }
    }))
}

pub fn fin(mut arguments: Vec<Expr>) -> Expr {
    if arguments.len() != 1 {
        return Expr::Error(format!("Expected one argument for fin{}", arguments.len()));
    }
    let arg = arguments.remove(0);
    match arg {
        // Note that we are treating floats here as a const waveform, not a duration
        Expr::Float(f) => {
            let w = Waveform::Const(f);
            BuiltIn {
                name: format!("fin({})", w),
                function: curry(move |waveform: Box<Waveform>| Waveform::Fin {
                    length: Box::new(w.clone()),
                    waveform,
                }),
            }
        }
        Expr::Waveform(length) => {
            let length = length;
            BuiltIn {
                name: format!("fin({})", length),
                function: curry(move |waveform: Box<Waveform>| Waveform::Fin {
                    length: Box::new(length.clone()),
                    waveform,
                }),
            }
        }
        _ => Expr::Error("Invalid arguments for fin".to_string()),
    }
}

pub fn seq(mut arguments: Vec<Expr>) -> Expr {
    if arguments.len() != 1 {
        return Expr::Error(format!("Expected one argument for fin{}", arguments.len()));
    }
    let arg = arguments.remove(0);
    match arg {
        // Note that we are treating floats here as a const waveform, not a duration
        Expr::Float(f) => {
            let w = Waveform::Const(f);
            BuiltIn {
                name: format!("seq({})", w),
                function: curry(move |waveform: Box<Waveform>| Waveform::Seq {
                    offset: Box::new(w.clone()),
                    waveform,
                }),
            }
        }
        Expr::Waveform(offset) => {
            let offset = offset;
            BuiltIn {
                name: format!("seq({})", offset),
                function: curry(move |waveform: Box<Waveform>| Waveform::Seq {
                    offset: Box::new(offset.clone()),
                    waveform,
                }),
            }
        }
        _ => Expr::Error("Invalid arguments for seq".to_string()),
    }
}

pub fn waveform_convolution(mut arguments: Vec<Expr>) -> Expr {
    if arguments.len() != 2 {
        return Expr::Error(format!("Expected two arguments for ~*"));
    }
    match (arguments.remove(0), arguments.remove(0)) {
        (a, b) => {
            let waveform = match (a, b) {
                (Expr::Waveform(a), Expr::Waveform(b)) => Waveform::Filter {
                    waveform: Box::new(a),
                    feed_forward: Box::new(b),
                    feedback: Box::new(Waveform::Fixed(vec![])),
                    state: (),
                },
                _ => return Expr::Error("Invalid arguments for ~*".to_string()),
            };
            Expr::Waveform(waveform)
        }
    }
}

pub fn waveform_filter(mut arguments: Vec<Expr>) -> Expr {
    if arguments.len() != 2 {
        return Expr::Error("Expected two waveforms".to_string());
    }
    let feed_forward = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Expr::Error("First argument must be a waveform or float".to_string()),
    };
    let feedback = match arguments.remove(0) {
        Expr::Waveform(b) => b,
        Float(value) => Waveform::Const(value),
        _ => return Expr::Error("Second argument must be a waveform or float".to_string()),
    };

    BuiltIn {
        name: format!("filter({}, {})", feed_forward, feedback),
        function: curry(move |waveform: Box<Waveform>| Waveform::Filter {
            waveform: waveform,
            feed_forward: Box::new(feed_forward.clone()),
            feedback: Box::new(feedback.clone()),
            state: (),
        }),
    }
}

pub fn res(mut arguments: Vec<Expr>) -> Expr {
    // TODO make it work in curried form?
    if arguments.len() != 2 {
        return Expr::Error("Expected two waveforms".to_string());
    }
    let trigger = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Expr::Error("First argument must be a waveform".to_string()),
    };
    let waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Expr::Error("Second argument must be a waveform or a float".to_string()),
    };
    Expr::Waveform(Waveform::Res {
        trigger: Box::new(trigger),
        waveform: Box::new(waveform),
        state: (),
    })
}

pub fn alt(mut arguments: Vec<Expr>) -> Expr {
    // TODO make it work in curried form?
    if arguments.len() != 3 {
        return Expr::Error("Expected three waveforms".to_string());
    }
    let trigger = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Expr::Error("First argument must be a waveform".to_string()),
    };
    let positive_waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Expr::Error("Second argument must be a waveform or a float".to_string()),
    };
    let negative_waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        Float(value) => Waveform::Const(value),
        _ => return Expr::Error("Third argument must be a waveform or a float".to_string()),
    };
    Expr::Waveform(Waveform::Alt {
        trigger: Box::new(trigger),
        positive_waveform: Box::new(positive_waveform),
        negative_waveform: Box::new(negative_waveform),
    })
}

fn mark(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Float(id)] if *id >= 1.0 && id.fract() == 0.0 => {
            let id = id.round() as u32;
            BuiltIn {
                name: format!("mark({})", id),
                function: curry(move |waveform: Box<Waveform>| Waveform::Marked { id, waveform }),
            }
        }
        _ => Expr::Error("Invalid argument for mark".to_string()),
    }
}

fn capture(mut arguments: Vec<Expr>) -> Expr {
    if arguments.len() != 1 {
        return Expr::Error("Expected one argument for capture".to_string());
    }
    let file_stem = match arguments.remove(0) {
        Expr::String(file_stem) => file_stem,
        _ => return Expr::Error("Expected a string argument to capture".to_string()),
    };
    BuiltIn {
        name: format!("capture({})", file_stem),
        function: curry(move |waveform: Box<Waveform>| Waveform::Captured {
            file_stem: file_stem.clone(),
            waveform,
        }),
    }
}

pub fn chord(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(exprs)] => {
            let mut result = Waveform::Fin {
                length: Box::new(Waveform::Const(0.0)),
                waveform: Box::new(Waveform::Const(0.0)),
            };
            for expr in exprs.iter().rev() {
                let waveform = match expr {
                    Expr::Waveform(waveform) => Box::new(waveform.clone()),
                    &Expr::Float(value) => Box::new(Waveform::Const(value)),
                    _ => return Expr::Error(format!("Invalid element in chord: {}", expr)),
                };
                result = Waveform::BinaryPointOp(
                    Operator::Add,
                    Box::new(Waveform::Seq {
                        offset: Box::new(Waveform::Const(0.0)),
                        waveform,
                    }),
                    Box::new(result),
                );
            }
            return Expr::Waveform(result);
        }
        _ => return Expr::Error("Invalid argument for chord".to_string()),
    }
}

pub fn sequence(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(exprs)] => {
            let mut result = Waveform::Fin {
                length: Box::new(Waveform::Const(0.0)),
                waveform: Box::new(Waveform::Const(0.0)),
            };
            for expr in exprs.iter().rev() {
                let waveform = match expr {
                    Expr::Waveform(waveform) => Box::new(waveform.clone()),
                    &Expr::Float(value) => Box::new(Waveform::Const(value)),
                    _ => return Expr::Error(format!("Invalid element in sequence: {}", expr)),
                };
                result = Waveform::BinaryPointOp(Operator::Add, waveform, Box::new(result));
            }
            return Expr::Waveform(result);
        }
        [expr] => return Expr::Error(format!("Invalid argument for sequence: {}", expr)),
        _ => return Expr::Error("Invalid argument for sequence".to_string()),
    }
}

pub fn add_prelude(context: &mut Vec<(String, Expr)>) {
    context.push(("true".to_string(), Expr::Bool(true)));
    context.push(("false".to_string(), Expr::Bool(false)));
    context.push(("time".to_string(), Expr::Waveform(Waveform::Time)));
    context.push(("noise".to_string(), Expr::Waveform(Waveform::Noise)));
    context.push(("X".to_string(), Expr::Waveform(Waveform::Slider(Slider::X))));
    context.push(("Y".to_string(), Expr::Waveform(Waveform::Slider(Slider::Y))));

    let builtins: Vec<(&str, fn(Vec<Expr>) -> Expr)> = vec![
        ("+", plus),
        ("-", minus),
        ("*", times),
        ("/", divide),
        ("==", equals),
        ("!=", not_equals),
        ("<", less_than),
        ("<=", less_than_equals),
        (">", greater_than),
        (">=", greater_than_equals),
        ("pow", power),
        ("sqrt", sqrt),
        ("exp", exp),
        ("sin", sin),
        ("cos", cos),
        ("map", map),
        ("reduce", reduce),
        ("unfold", unfold),
        ("append", append),
        ("fixed", fixed),
        ("fin", fin),
        ("seq", seq),
        ("filter", waveform_filter),
        ("~*", waveform_convolution),
        ("res", res),
        ("alt", alt),
        ("mark", mark),
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
        let exprs = vec![Float(2.0), Float(3.0), Float(4.0)];
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
        let exprs = vec![Float(2.0), Float(3.0), Float(4.0)];
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
