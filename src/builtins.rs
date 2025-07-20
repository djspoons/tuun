use std::rc::Rc;

use crate::parser::{simplify, BuiltInFn, Expr};
use crate::tracker::Waveform;
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
        _ => Expr::Error("Invalid arguments for -".to_string()),
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

pub fn linear_waveform(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(initial_value), Float(slope)] => Expr::Waveform(Waveform::Linear {
            initial_value,
            slope,
        }),
        _ => Expr::Error("Invalid argument for linear_waveform".to_string()),
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
    let builtins: Vec<(&str, fn(Vec<Expr>) -> Expr)> = vec![
        ("+", plus),
        ("-", minus),
        ("*", times),
        ("/", divide),
        ("pow", power),
        ("map", map),
        ("reduce", reduce),
        ("$", sine_waveform),
        ("linear", linear_waveform),
        ("fin", fin),
        ("seq", seq),
        ("~+", waveform_sum),
        ("~.", waveform_dot_product),
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
