use std::rc::Rc;

use crate::parser::{BuiltInFn, Expr};
use crate::tracker::Waveform;
use Expr::{BuiltIn, Float};

pub fn add(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a + b),
        _ => Expr::Error("Invalid arguments for add".to_string()),
    }
}

pub fn subtract(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a - b),
        _ => Expr::Error("Invalid arguments for subtract".to_string()),
    }
}

pub fn multiply(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a * b),
        _ => Expr::Error("Invalid arguments for multiply".to_string()),
    }
}

pub fn divide(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(a), Float(b)] => Expr::Float(a / b),
        _ => Expr::Error("Invalid arguments for divide".to_string()),
    }
}

pub fn power(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(base), Float(exponent)] => Expr::Float(base.powf(exponent)),
        _ => Expr::Error("Invalid arguments for power".to_string()),
    }
}

pub fn sine_wave(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(frequency)] => Expr::Waveform(Waveform::SineWave { frequency }),
        _ => Expr::Error("Invalid argument for sine_wave".to_string()),
    }
}

fn filter(f: impl Fn(Box<Waveform>) -> Waveform + 'static) -> BuiltInFn {
    Rc::new(move |arguments: &mut Vec<Expr>| -> Expr {
        if arguments.len() != 1 {
            return Expr::Error("Expected waveform".to_string());
        }
        let waveform = arguments.remove(0);
        match waveform {
            Expr::Waveform(waveform) => Expr::Waveform(f(Box::new(waveform))),
            _ => Expr::Error("Invalid waveform".to_string()),
        }
    })
}

pub fn amplify(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(level)] => BuiltIn {
            name: format!("amp({})", level),
            function: filter(move |waveform: Box<Waveform>| Waveform::Amplify { level, waveform }),
        },
        _ => Expr::Error("Invalid arguments for amplify".to_string()),
    }
}

pub fn seq(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(duration)] => BuiltIn {
            name: format!("seq({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::Seq { duration, waveform }),
        },
        _ => Expr::Error("Invalid arguments for seq".to_string()),
    }
}

pub fn fin(arguments: &mut Vec<Expr>) -> Expr {
    match arguments[..] {
        [Float(duration)] => BuiltIn {
            name: format!("seq({})", duration),
            function: filter(move |waveform: Box<Waveform>| Waveform::Fin { duration, waveform }),
        },
        _ => Expr::Error("Invalid arguments for fin".to_string()),
    }
}

pub fn linear_ramp(arguments: &mut Vec<Expr>) -> Expr {
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

pub fn sustain(arguments: &mut Vec<Expr>) -> Expr {
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

pub fn chord(arguments: &mut Vec<Expr>) -> Expr {
    let mut waveforms = Vec::<Waveform>::new();
    for argument in arguments.iter() {
        match argument {
            Expr::Waveform(waveform) => waveforms.push(waveform.clone()),
            _ => return Expr::Error("Invalid argument for chord".to_string()),
        }
    }
    return Expr::Waveform(Waveform::Chord(waveforms));
}

pub fn sequence(arguments: &mut Vec<Expr>) -> Expr {
    let mut waveforms = Vec::<Waveform>::new();
    for argument in arguments.iter() {
        match argument {
            Expr::Waveform(waveform) => waveforms.push(waveform.clone()),
            _ => return Expr::Error("Invalid argument for sequence".to_string()),
        }
    }
    return Expr::Waveform(Waveform::Sequence(waveforms));
}

pub fn add_standard_context(context: &mut Vec<(String, Expr)>) {
    let builtins: Vec<(&str, fn(&mut Vec<Expr>) -> Expr)> = vec![
        ("+", add),
        ("-", subtract),
        ("*", multiply),
        ("/", divide),
        ("pow", power),
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
                function: Rc::new(function),
            },
        ));
    }
}
