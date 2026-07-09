use std::fmt::Display;

use crate::waveform;
use crate::{builtins, eval, expr, parser};

use waveform::Operator;
use waveform::Waveform::{Append, BinaryPointOp, Const, Fin, Time};

// This file defines types and functions related to sliders that are platform-independent.

/// Dummy mark type used for evaluating slider function expressions (which don't use marks).
#[derive(Clone, Debug, PartialEq)]
struct NoMark;

impl Display for NoMark {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NoMark")
    }
}

/// Converts a normalized slider value (0–1) to an actual value using the slider's function.
///
/// For `Linear`, this is `min + normalized * (max - min)`.
/// For `UserDefined`, this evaluates the user-provided function expression.
pub fn denormalize(function: &expr::SliderFunction, normalized: f32) -> Result<f32, String> {
    use expr::SliderFunction;
    match function {
        SliderFunction::Linear { min, max, .. } => Ok(min + normalized * (max - min)),
        SliderFunction::UserDefined {
            function_source, ..
        } => {
            let source = format!("({})({})", function_source, normalized);
            let expr = parser::parse_program::<NoMark>(&source)
                .map_err(|errors| format!("slider function parse error: {:?}", errors))?;
            let mut bindings = Vec::new();
            builtins::add_bindings(&mut bindings);
            let resolve = |_: &[String]| {
                Err(expr::Error::new(
                    "didn't expect to resolve inside of slider function".to_string(),
                ))
            };
            let result = eval::evaluate(resolve, &bindings, expr)
                .map_err(|e| format!("slider function eval error: {}", e))?;
            match result.expr {
                expr::Expr::Float(v) => Ok(v),
                other => Err(format!(
                    "slider function did not return a number, got: {:?}",
                    other
                )),
            }
        }
    }
}

pub fn append_slider_bindings<M, F>(
    configs: &[expr::Slider],
    normalized_values: &[f32],
    mark_id: F,
    bindings: &mut Vec<expr::SourceBinding<M>>,
) where
    F: Fn(String) -> M,
{
    bindings.append(
        &mut configs
            .iter()
            .zip(normalized_values)
            .map(|(config, normalized_value)| {
                let value = denormalize(&config.function, *normalized_value).unwrap_or(0.0);
                expr::SourceBinding::definition(
                    expr::Pattern::Identifier(config.label.clone()),
                    expr::SourceExpr::from(expr::Expr::Waveform(waveform::Waveform::Marked {
                        id: mark_id(config.label.clone()),
                        waveform: Box::new(waveform::Waveform::Const(value)),
                    })),
                )
            })
            .collect::<Vec<_>>(),
    );
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
        (),
    )
}
