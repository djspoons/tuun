use std::rc::Rc;
use std::time::Duration;

use crate::parser::{simplify, BuiltInFn, Expr};
use crate::tracker::{Slider, Waveform};
use Expr::{Application, Bool, BuiltIn, Error, Float, List, Tuple};

pub fn plus(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a + b),
        _ => Expr::Error("Invalid arguments for +".to_string()),
    }
}

pub fn minus(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a)] => Expr::Float(-a),
        [Float(a), Float(b)] => Expr::Float(a - b),
        _ => Expr::Error(format!("Invalid arguments for -: {:?}", arguments)),
    }
}

pub fn times(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a * b),
        _ => Expr::Error("Invalid arguments for *".to_string()),
    }
}

pub fn divide(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a / b),
        _ => Expr::Error("Invalid arguments for /".to_string()),
    }
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
                        arguments: Box::new(expr.clone()), // can we avoid this clone?
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
                        arguments: Box::new(Tuple(vec![acc, expr.clone()])), // can we avoid this clone?
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

pub fn append(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(a), List(b)] => {
            let mut result = a.clone();
            result.extend(b.clone());
            Expr::List(result)
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

pub fn fin(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(duration)] => BuiltIn {
            name: format!("fin({})", duration),
            function: curry(move |waveform: Box<Waveform>| Waveform::Fin {
                duration: Duration::from_secs_f32(duration),
                waveform,
            }),
        },
        _ => Expr::Error("Invalid arguments for fin".to_string()),
    }
}

pub fn seq(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Float(duration)] => {
            let duration = duration.clone();
            BuiltIn {
                name: format!("seq({})", duration),
                function: curry(move |waveform: Box<Waveform>| Waveform::Seq {
                    duration: Duration::from_secs_f32(duration),
                    waveform,
                }),
            }
        }
        [expr] => Expr::Error(format!("Invalid argument for seq: {}", expr)),
        _ => Expr::Error(format!(
            "Invalid number of arguments for seq: {}",
            arguments.len()
        )),
    }
}

pub fn waveform_sin(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Expr::Waveform(a)] => Expr::Waveform(Waveform::Sin(Box::new(a.clone()))),
        [Float(value)] => Expr::Waveform(Waveform::Sin(Box::new(Waveform::Const(*value)))),
        _ => Expr::Error("Invalid argument for sin".to_string()),
    }
}

fn waveform_binary_op(
    mut arguments: Vec<Expr>,
    op: fn(Box<Waveform>, Box<Waveform>) -> Waveform,
) -> Expr {
    if arguments.len() != 2 {
        return Expr::Error("Expected two waveforms".to_string());
    }
    let a = match arguments.remove(0) {
        Expr::Waveform(a) => Box::new(a),
        Float(value) => Box::new(Waveform::Const(value)),
        _ => return Expr::Error("First argument must be a waveform or float".to_string()),
    };
    let b = match arguments.remove(0) {
        Expr::Waveform(b) => Box::new(b),
        Float(value) => Box::new(Waveform::Const(value)),
        _ => return Expr::Error("Second argument must be a waveform or float".to_string()),
    };
    return Expr::Waveform(op(a, b));
}

pub fn waveform_convolution(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, |waveform, kernel| Waveform::Filter {
        waveform,
        feed_forward: kernel,
        feedback: Box::new(Waveform::Fixed(vec![])),
        state: (),
    });
}

pub fn waveform_sum(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, Waveform::Sum);
}

pub fn waveform_dot_product(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, Waveform::DotProduct);
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
        [Float(id)] if *id >= 1.0 => {
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
                duration: Duration::ZERO,
                waveform: Box::new(Waveform::Const(0.0)),
            };
            for expr in exprs.iter().rev() {
                let waveform = match expr {
                    Expr::Waveform(waveform) => Box::new(waveform.clone()),
                    &Expr::Float(value) => Box::new(Waveform::Const(value)),
                    _ => return Expr::Error(format!("Invalid element in chord: {}", expr)),
                };
                result = Waveform::Sum(
                    Box::new(Waveform::Seq {
                        duration: Duration::ZERO,
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
                duration: Duration::ZERO,
                waveform: Box::new(Waveform::Const(0.0)),
            };
            for expr in exprs.iter().rev() {
                let waveform = match expr {
                    Expr::Waveform(waveform) => Box::new(waveform.clone()),
                    &Expr::Float(value) => Box::new(Waveform::Const(value)),
                    _ => return Expr::Error(format!("Invalid element in sequence: {}", expr)),
                };
                result = Waveform::Sum(waveform, Box::new(result));
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
        ("pow", power),
        ("sqrt", sqrt),
        ("exp", exp),
        ("map", map),
        ("reduce", reduce),
        ("append", append),
        ("fixed", fixed),
        ("fin", fin),
        ("seq", seq),
        ("sin", waveform_sin),
        ("filter", waveform_filter),
        ("~*", waveform_convolution),
        ("~+", waveform_sum),
        ("~.", waveform_dot_product),
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
