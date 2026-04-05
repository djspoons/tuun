use crate::parser;
use crate::waveform;
use waveform::Operator;
use waveform::Waveform::{Append, BinaryPointOp, Const, Fin, Time};

// This file defines types and functions related to sliders that are platform-independent.

#[derive(Debug, Clone)]
pub struct SliderConfig {
    pub label: String,
    pub min: f32,
    pub max: f32,
    pub initial_value: f32,
}

pub fn parse_slider_configs(input: &str) -> Vec<SliderConfig> {
    if !input.starts_with('[') || !input.ends_with(']') {
        return vec![];
    }
    let list_inner = &input[1..input.len() - 1];

    let mut configs = Vec::new();
    for item in list_inner.split(',') {
        let item = item.trim().trim_matches('"');
        if item.is_empty() {
            continue;
        }
        let parts: Vec<&str> = item.split(':').collect();
        let label = parts[0].to_string();
        if label.is_empty() {
            continue;
        }
        let min = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
        const MIN_GAP: f32 = 0.01;
        let max = f32::max(
            parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(1.0),
            min + MIN_GAP,
        );
        let initial_value = parts
            .get(3)
            .and_then(|s| s.parse().ok())
            .unwrap_or((min + max) / 2.0)
            .clamp(min, max);
        configs.push(SliderConfig {
            label,
            min,
            max,
            initial_value,
        });
    }
    configs
}

pub fn parse_slider_pragma(line: &str) -> Option<Vec<SliderConfig>> {
    let trimmed = line.trim();
    if !trimmed.starts_with("//#{") || !trimmed.ends_with('}') {
        return None;
    }
    let inner = &trimmed[4..trimmed.len() - 1]; // strip "//#{" and "}"
    if !inner.starts_with("sliders=") {
        return None;
    }
    let sliders_value = inner["sliders=".len()..].trim();
    Some(parse_slider_configs(sliders_value))
}

pub fn prepend_slider_bindings<M, F>(
    configs: &Vec<SliderConfig>,
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
            let value = config.min + normalized_value * (config.max - config.min);
            (
                parser::Pattern::Identifier(config.label.to_string()),
                parser::Expr::Waveform(waveform::Waveform::Marked {
                    id: mark_id(config.label.to_string()),
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
