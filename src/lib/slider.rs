use crate::waveform;
use crate::{builtins, parser};

use waveform::Operator;
use waveform::Waveform::{Append, BinaryPointOp, Const, Fin, Time};

// This file defines types and functions related to sliders that are platform-independent.

/// Dummy mark type used for evaluating slider function expressions (which don't use marks).
#[derive(Clone, Debug, PartialEq)]
struct NoMark;

impl std::fmt::Display for NoMark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoMark")
    }
}

/// Converts a normalized slider value (0–1) to an actual value using the slider's function.
///
/// For `Linear`, this is `min + normalized * (max - min)`.
/// For `UserDefined`, this evaluates the user-provided function expression.
pub fn denormalize(function: &parser::SliderFunction, normalized: f32) -> Result<f32, String> {
    use parser::SliderFunction;
    match function {
        SliderFunction::Linear { min, max, .. } => Ok(min + normalized * (max - min)),
        SliderFunction::UserDefined {
            function_source, ..
        } => {
            let source = format!("({})({})", function_source, normalized);
            let expr = parser::parse_program::<NoMark>(&source)
                .map_err(|errors| format!("slider function parse error: {:?}", errors))?;
            let mut context = Vec::new();
            builtins::add_prelude(&mut context);
            let result = parser::evaluate(&context, expr)
                .map_err(|e| format!("slider function eval error: {}", e.to_string()))?;
            match result {
                parser::Expr::Float(v) => Ok(v),
                other => Err(format!(
                    "slider function did not return a number, got: {:?}",
                    other
                )),
            }
        }
    }
}

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
    let bindings = configs
        .iter()
        .zip(normalized_values)
        .map(|(config, normalized_value)| {
            let value = denormalize(&config.function, *normalized_value).unwrap_or(0.0);
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
