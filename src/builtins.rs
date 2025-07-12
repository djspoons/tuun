use std::rc::Rc;

use crate::parser::{simplify, BuiltInFn, Expr};
use crate::tracker::Waveform;
use Expr::{Application, BuiltIn, Error, Float, List, Tuple};

pub fn plus(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a + b),
        _ => Expr::Error("Invalid arguments for plus".to_string()),
    }
}

pub fn minus(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a)] => Expr::Float(-a),
        [Float(a), Float(b)] => Expr::Float(a - b),
        _ => Expr::Error("Invalid arguments for minus".to_string()),
    }
}

pub fn times(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a * b),
        _ => Expr::Error("Invalid arguments for times".to_string()),
    }
}

pub fn divide(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a / b),
        _ => Expr::Error("Invalid arguments for divide".to_string()),
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
        [function, accum, List(exprs)] => {
            let context = vec![];
            let mut accum = accum.clone();
            for expr in exprs {
                let result = simplify(
                    &context,
                    Application {
                        function: Box::new(function.clone()),
                        arguments: Box::new(Tuple(vec![accum, expr.clone()])), // can we avoid this clone?
                    },
                );
                accum = match result {
                    Ok(expr) => expr,
                    Err(err) => return Expr::Error(err.to_string()),
                };
            }
            accum
        }
        _ => Expr::Error("Invalid arguments for reduce".to_string()),
    }
}

pub fn sine_wave(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(frequency)] => Expr::Waveform(Waveform::SineWave { frequency }),
        _ => Expr::Error("Invalid argument for sine_wave".to_string()),
    }
}

fn filter(f: impl Fn(Box<Waveform>) -> Waveform + 'static) -> BuiltInFn {
    BuiltInFn(Rc::new(move |mut arguments: Vec<Expr>| -> Expr {
        if arguments.len() != 1 {
            return Expr::Error("Expected waveform".to_string());
        }
        let waveform = arguments.remove(0);
        match waveform {
            Expr::Waveform(waveform) => Expr::Waveform(f(Box::new(waveform))),
            _ => Expr::Error("Invalid waveform".to_string()),
        }
    }))
}

pub fn amplify(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(level)] => BuiltIn {
            name: format!("amp({})", level),
            function: filter(move |waveform: Box<Waveform>| Waveform::Amplify { level, waveform }),
        },
        _ => Expr::Error("Invalid arguments for amplify".to_string()),
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

pub fn fin(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(duration)] => BuiltIn {
            name: format!("seq({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::Fin { duration, waveform }),
        },
        _ => Expr::Error("Invalid arguments for fin".to_string()),
    }
}

pub fn linear_ramp(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(initial_level), Float(duration), Float(final_level)] => BuiltIn {
            name: format!("linear_ramp({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::LinearRamp {
                initial_level,
                duration,
                final_level,
                waveform,
            }),
        },
        _ => Expr::Error("Invalid arguments for linear_ramp".to_string()),
    }
}

pub fn sustain(arguments: Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(level), Float(duration)] => BuiltIn {
            name: format!("S({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::Sustain {
                level,
                duration,
                waveform,
            }),
        },
        _ => Expr::Error("Invalid arguments for sustain".to_string()),
    }
}

pub fn chord(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(exprs)] => {
            let mut waveforms = Vec::<Waveform>::new();
            for expr in exprs {
                match expr {
                    Expr::Waveform(waveform) => waveforms.push(waveform.clone()),
                    _ => return Expr::Error("Invalid argument for chord".to_string()),
                }
            }
            Expr::Waveform(Waveform::Chord(waveforms))
        }
        _ => return Expr::Error("Invalid argument for chord".to_string()),
    }
}

pub fn sequence(arguments: Vec<Expr>) -> Expr {
    match &arguments[..] {
        [List(exprs)] => {
            let mut waveforms = Vec::<Waveform>::new();
            for expr in exprs {
                match expr {
                    Expr::Waveform(waveform) => waveforms.push(waveform.clone()),
                    _ => return Expr::Error(format!("Invalid element in sequence: {}", expr)),
                }
            }
            Expr::Waveform(Waveform::Sequence(waveforms))
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
        ("_chord", chord),
        ("_sequence", sequence),
        ("$", sine_wave),
        ("amp", amplify),
        ("seq", seq),
        ("fin", fin),
        ("linear_ramp", linear_ramp),
        ("S", sustain),
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
