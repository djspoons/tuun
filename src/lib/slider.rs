use crate::parser;
use crate::waveform;

use waveform::Operator;
use waveform::Waveform::{Append, BinaryPointOp, Const, Fin, Time};

// This file defines types and functions related to sliders that are platform-independent.

pub fn prepend_slider_bindings<M, F>(
    configs: &Vec<parser::Slider>,
    normalized_values: &Vec<f32>,
    mark_id: F,
    expr: parser::Expr<M>,
) -> parser::Expr<M>
where
    F: Fn(String) -> M,
{
    if configs.is_empty() {
        return expr;
    }
    use parser::SliderFunction;
    let bindings = configs
        .iter()
        .zip(normalized_values)
        .map(|(config, normalized_value)| {
            let value = match config.function {
                SliderFunction::Linear { min, max, .. } => min + normalized_value * (max - min),
            };
            (
                parser::Pattern::Identifier(config.label.clone()),
                parser::Expr::Waveform(waveform::Waveform::Marked {
                    id: mark_id(config.label.clone()),
                    waveform: Box::new(waveform::Waveform::Const(value)),
                }),
            )
        })
        .collect::<Vec<_>>();
    parser::make_let(bindings, expr)
}

/// Build a waveform that ramps linearly from `last_value` to `new_value` over
/// `ramp_duration_secs`, then holds `new_value` forever.
pub fn make_ramp<M>(
    last_value: f32,
    new_value: f32,
    ramp_duration_secs: f32,
) -> waveform::Waveform<M> {
    Append(
        Box::new(Fin {
            length: Box::new(BinaryPointOp(
                Operator::Subtract,
                Box::new(Time(())),
                Box::new(Const(ramp_duration_secs)),
            )),
            waveform: Box::new(BinaryPointOp(
                Operator::Add,
                Box::new(BinaryPointOp(
                    Operator::Multiply,
                    Box::new(Time(())),
                    Box::new(Const((new_value - last_value) / ramp_duration_secs)),
                )),
                Box::new(Const(last_value)),
            )),
        }),
        Box::new(Const(new_value)),
    )
}
