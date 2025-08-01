use std::rc::Rc;

use crate::parser::{simplify, BuiltInFn, Expr};
use crate::tracker::{Dial, Waveform};
use Expr::{Application, BuiltIn, Error, Float, List, Tuple};

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

pub fn sine_waveform(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [Expr::Waveform(a)] => Expr::Waveform(Waveform::SineWave {
            frequency: Box::new(a.clone()),
        }),
        [Float(value)] => Expr::Waveform(Waveform::SineWave {
            frequency: Box::new(Waveform::Const(*value)),
        }),
        _ => Expr::Error("Invalid argument for $".to_string()),
    }
}

pub fn fixed_waveform(arguments: Vec<Expr>) -> Expr {
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

fn filter(f: impl Fn(Box<Waveform>) -> Waveform + 'static) -> BuiltInFn {
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
            function: filter(move |waveform: Box<Waveform>| Waveform::Fin { duration, waveform }),
        },
        _ => Expr::Error("Invalid arguments for fin".to_string()),
    }
}

pub fn rep(mut arguments: Vec<Expr>) -> Expr {
    // TODO make it work in curried form
    if arguments.len() != 2 {
        return Expr::Error("Expected a waveform".to_string());
    }
    let trigger = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Expr::Error("First argument must be a waveform".to_string()),
    };
    let waveform = match arguments.remove(0) {
        Expr::Waveform(a) => a,
        _ => return Expr::Error("Second argument must be a waveform".to_string()),
    };
    Expr::Waveform(Waveform::Rep {
        trigger: Box::new(trigger),
        waveform: Box::new(waveform),
    })
}

pub fn seq(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(duration)] => BuiltIn {
            name: format!("seq({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::Seq { duration, waveform }),
        },
        _ => Expr::Error("Invalid arguments for seq".to_string()),
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

pub fn waveform_sum(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, Waveform::Sum);
}

pub fn waveform_dot_product(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, Waveform::DotProduct);
}

pub fn waveform_convolution(arguments: Vec<Expr>) -> Expr {
    return waveform_binary_op(arguments, |waveform, kernel| Waveform::Convolution {
        waveform,
        kernel,
    });
}

pub fn chord(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(exprs)] => {
            let mut result = Waveform::Fin {
                duration: 0.0,
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
                        duration: 0.0,
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
                duration: 0.0,
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
    context.push(("time".to_string(), Expr::Waveform(Waveform::Time)));
    context.push(("noise".to_string(), Expr::Waveform(Waveform::Noise)));
    context.push(("X".to_string(), Expr::Waveform(Waveform::Dial(Dial::X))));
    context.push(("Y".to_string(), Expr::Waveform(Waveform::Dial(Dial::Y))));

    let builtins: Vec<(&str, fn(Vec<Expr>) -> Expr)> = vec![
        ("+", plus),
        ("-", minus),
        ("*", times),
        ("/", divide),
        ("pow", power),
        ("sqrt", sqrt),
        ("exp", exp),
        ("map", map),
        ("reduce", reduce),
        ("$", sine_waveform),
        ("fixed", fixed_waveform),
        ("fin", fin),
        ("rep", rep),
        ("seq", seq),
        ("~+", waveform_sum),
        ("~.", waveform_dot_product),
        ("~*", waveform_convolution),
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
