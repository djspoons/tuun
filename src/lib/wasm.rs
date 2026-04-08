//! WebAssembly interface for Tuun music synthesizer.
//!
//! This module provides JavaScript-friendly bindings for the core synthesis engine.
//! The `Wasm` struct acts as a simple single-waveform tracker: it owns the currently
//! playing waveform and handles slider updates via `Waveform::Marked` + `waveform::substitute`.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use wasm_bindgen::prelude::*;

use crate::{builtins, generator, optimizer, parser, slider, waveform};

#[derive(Clone, Debug, PartialEq)]
enum MarkId {
    // We currently only support marked waveforms as part of the slider implementation.
    Slider(String),
}

impl fmt::Display for MarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkId::Slider(name) => write!(f, "slider({:?})", name),
        }
    }
}

/// WebAssembly interface for the Tuun synthesizer.
///
/// Owns the currently-playing waveform.
#[wasm_bindgen(js_name = "Tuun")]
pub struct Wasm {
    sample_rate: i32,
    context: Vec<(String, parser::Expr<MarkId>)>,
    waveform: Option<generator::Waveform<MarkId>>,
    last_slider_values: HashMap<String, f32>,
    buffer_duration: Duration,
}

#[wasm_bindgen(js_class = "Tuun")]
impl Wasm {
    /// Creates a new Tuun instance with the specified sample rate and tempo.
    ///
    /// # Arguments
    /// * `sample_rate` - The audio sample rate in Hz (e.g., 44100)
    /// * `tempo` - The tempo in beats per minute (e.g., 120)
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: i32, tempo: f32) -> Result<Wasm, String> {
        use parser::Expr;
        // Set up better panic messages in the browser console
        console_error_panic_hook::set_once();

        // TODO this has a lot of repetition with main.rs

        // Initialize context with builtins
        let mut context = Vec::new();
        context.push(("sample_rate".to_string(), Expr::Float(sample_rate as f32)));
        builtins::add_prelude(&mut context);

        context.push(("tempo".to_string(), Expr::Float(tempo)));

        // Load context from embedded .tuun file
        // The file is embedded at compile time using include_str!
        // Default: context.tuun from the repository
        let context_content = include_str!("../../context.tuun");

        // Strip comments and parse the context file
        let context_content: String = context_content
            .lines()
            .map(|line| {
                if let Some(comment_index) = line.find("//") {
                    &line[..comment_index]
                } else {
                    line
                }
            })
            .collect::<Vec<&str>>()
            .join("\n");

        // Parse and add all context definitions
        match parser::parse_context(&context_content) {
            Ok(parsed_defs) => {
                for (pattern, expr) in parsed_defs {
                    match parser::evaluate(&context, expr) {
                        Ok(expr) => {
                            if let Err(e) = parser::extend_context(&mut context, &pattern, &expr) {
                                eprintln!("Warning: Failed to add context definition: {:?}", e);
                            }
                        }
                        Err(e) => eprintln!("Warning: Failed to evaluate context: {:?}", e),
                    }
                }
            }
            Err(e) => eprintln!("Warning: Failed to parse context file: {:?}", e),
        }

        Ok(Wasm {
            sample_rate,
            context,
            waveform: None,
            last_slider_values: HashMap::new(),
            buffer_duration: Duration::from_secs_f32(128.0 / sample_rate as f32),
        })
    }

    /// Parses an expression with slider bindings and prepares for playback.
    ///
    /// `slider_json` is a JSON object mapping slider names to initial values,
    /// for example, `{"volume": 0.5, "cutoff": 2000}`.
    /// Pass `"{}"` for no sliders.
    ///
    /// # Examples
    /// ```javascript
    /// const waveform = tuun.parse("sine(2764, 0)", "{}");
    /// ```
    pub fn parse(&mut self, expression: &str, slider_json: &str) -> Result<(), String> {
        let parsed_expr = parser::parse_program(expression)
            .map_err(|errors| format!("Parse errors: {:?}", errors))?;
        let sliders = parse_json(slider_json)?;

        // Wrap with slider bindings: let name = Marked(Slider(name), Const(value)) in ...
        let expr = if sliders.is_empty() {
            parsed_expr
        } else {
            let bindings = sliders
                .iter()
                .map(|(name, value)| {
                    (
                        parser::Pattern::Identifier(name.clone()),
                        parser::Expr::Waveform(waveform::Waveform::Marked {
                            id: MarkId::Slider(name.clone()),
                            waveform: Box::new(waveform::Waveform::Const(*value)),
                        }),
                    )
                })
                .collect();
            parser::make_let(bindings, parsed_expr)
        };

        let expr = parser::evaluate(&self.context, expr)
            .map_err(|e| format!("Evaluate error: {:?}", e))?;

        // TODO Do we want to precompute here?

        let waveform = match expr {
            parser::Expr::Waveform(w) => w,
            parser::Expr::Seq { waveform, .. } => match *waveform {
                parser::Expr::Waveform(w) => w,
                _ => return Err("Got non-Waveform in seq after evaluate".to_string()),
            },
            other => {
                return Err(format!(
                    "Expression did not evaluate to a waveform, got: {:?}",
                    other
                ));
            }
        };

        let waveform = optimizer::optimize(waveform);
        let waveform = generator::initialize_state(waveform);

        self.waveform = Some(waveform);
        self.last_slider_values = sliders;

        Ok(())
    }

    /// Drops the current waveform.
    pub fn stop(&mut self) {
        self.waveform = None;
        self.last_slider_values.clear();
    }

    /// Updates a slider value in the current waveform.
    ///
    /// Builds a linear ramp from the last value to the new value and
    /// substitutes it into the playing waveform.
    pub fn update_slider(&mut self, name: &str, value: f32) {
        let waveform = match &mut self.waveform {
            Some(w) => w,
            None => return,
        };

        let last_value = self.last_slider_values.get(name).copied().unwrap_or(value);

        let ramp = slider::make_ramp(last_value, value, self.buffer_duration.as_secs_f32());
        let ramp = generator::initialize_state(ramp);
        waveform::substitute(waveform, &MarkId::Slider(name.to_string()), &ramp);

        self.last_slider_values.insert(name.to_string(), value);
    }

    /// Generates audio samples from the current waveform. Updates the internal
    /// state of the waveform so that the next call to `generate()` will continue
    /// from the point at which this call left off.
    ///
    /// # Arguments
    /// * `out` - A buffer to fill with samples
    ///
    /// # Returns
    /// A boolean indicating whether or not the current waveform will generate any
    /// more samples
    ///
    /// # Examples
    /// ```javascript
    /// tuun.parse("$440", "{}");
    /// const done = tuun.process(output);
    /// ```
    pub fn process(&mut self, out: &mut [f32]) -> bool {
        self.buffer_duration = Duration::from_secs_f32(out.len() as f32 / self.sample_rate as f32);

        let waveform = match &mut self.waveform {
            Some(w) => w,
            None => return false,
        };

        let mut g = generator::Generator::new(self.sample_rate);
        let len = g.generate(waveform, out);
        // Web audio expects the whole buffer to be filled.
        out[len..].fill(0.0);
        len == out.len()
    }

    /// Returns whether a waveform is currently playing.
    pub fn is_playing(&self) -> bool {
        self.waveform.is_some()
    }

    /// Returns the current sample rate.
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }
}

/// Parses a simple JSON object like `{"name": 0.5, "other": 1.0}` into a HashMap.
fn parse_json(json: &str) -> Result<HashMap<String, f32>, String> {
    let trimmed = json.trim();
    if trimmed == "{}" {
        return Ok(HashMap::new());
    }
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err("JSON must be an object".to_string());
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut result = HashMap::new();
    for pair in inner.split(',') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let parts: Vec<&str> = pair.splitn(2, ':').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid entry: {}", pair));
        }
        let key = parts[0].trim().trim_matches('"');
        let value: f32 = parts[1]
            .trim()
            .parse()
            .map_err(|_| format!("Invalid value for '{}': {}", key, parts[1].trim()))?;
        result.insert(key.to_string(), value);
    }
    Ok(result)
}

/// Parses a slider config string like `["volume:0.5:0:1", "freq:0.5:fn(x) => 100 * pow(100, x)"]`
/// and returns a JSON array of slider objects.
///
/// Linear sliders: `{ type: "linear", label, initial_value, min, max }`
/// User-defined sliders: `{ type: "user-defined", label, normalized_initial_value, function_source, initial_value, value_at_0, value_at_1 }`
///
/// Returns an error string if parsing fails.
#[wasm_bindgen(js_name = "parseSliders")]
pub fn parse_sliders(input: &str) -> Result<String, String> {
    let sliders = parser::parse_sliders(input).map_err(|errors| {
        errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ")
    })?;

    // Manually build JSON since we don't have serde
    let entries: Vec<String> = sliders
        .iter()
        .map(|s| match &s.function {
            parser::SliderFunction::Linear {
                initial_value,
                min,
                max,
            } => {
                format!(
                    r#"{{"type":"linear","label":"{}","initial_value":{},"min":{},"max":{}}}"#,
                    s.label, initial_value, min, max
                )
            }
            parser::SliderFunction::UserDefined {
                normalized_initial_value,
                function_source,
            } => {
                let initial_value = slider::denormalize(&s.function, *normalized_initial_value)
                    .unwrap_or(0.0);
                let value_at_0 = slider::denormalize(&s.function, 0.0).unwrap_or(0.0);
                let value_at_1 = slider::denormalize(&s.function, 1.0).unwrap_or(0.0);
                format!(
                    r#"{{"type":"user-defined","label":"{}","normalized_initial_value":{},"function_source":"{}","initial_value":{},"value_at_0":{},"value_at_1":{}}}"#,
                    s.label, normalized_initial_value,
                    function_source.replace('\\', "\\\\").replace('"', "\\\""),
                    initial_value, value_at_0, value_at_1
                )
            }
        })
        .collect();
    Ok(format!("[{}]", entries.join(",")))
}

/// Evaluates a user-defined slider function at a given normalized value.
///
/// For example, `evaluateSlider("fn(x) => 100 * pow(100, x)", 0.5)` returns ~1000.
#[wasm_bindgen(js_name = "evaluateSlider")]
pub fn evaluate_slider(function_source: &str, normalized_value: f32) -> Result<f32, String> {
    let function = parser::SliderFunction::UserDefined {
        normalized_initial_value: 0.0, // unused for evaluation
        function_source: function_source.to_string(),
    };
    slider::denormalize(&function, normalized_value)
}

/// Initializes the WASM module.
/// This is called automatically when you import the module.
#[wasm_bindgen(start)]
pub fn main() {
    console_error_panic_hook::set_once();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test all examples from the web UI to ensure they parse and generate samples
    #[test]
    fn test_web_ui_examples() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let examples = vec![
            ("sine(2764, 0)", "Sine wave (440 Hz)"),
            ("noise * 0.1", "Noise"),
        ];

        for (expr, description) in examples {
            println!("Testing: {} - {}", description, expr);

            tuun.parse(expr, "{}")
                .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", expr, e));

            let mut out = vec![0.0; 100];
            let more = tuun.process(&mut out);
            assert!(more, "Expected at least 100 samples for '{}'", expr);

            for (i, &sample) in out.iter().enumerate() {
                assert!(
                    sample >= -1.0 && sample <= 1.0,
                    "Sample {} out of range for '{}': {}",
                    i,
                    expr,
                    sample
                );
            }

            tuun.stop();
            println!("  ✓ Parsed and generated successfully");
        }
    }

    /// Test that invalid expressions produce appropriate errors
    #[test]
    fn test_invalid_expressions() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let invalid_examples = vec!["undefined_function()", "sine(2764)", "1 + "];

        for expr in invalid_examples {
            println!("Testing invalid expression: {}", expr);
            let result = tuun.parse(expr, "{}");
            assert!(
                result.is_err(),
                "Expected error for invalid expression '{}', but got success",
                expr
            );
            println!("  ✓ Correctly rejected");
        }
    }

    #[test]
    fn test_context_functions() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let lpf_examples = vec![
            ("$440 * Qw | lpf(0.5, 1900)", "Low-pass filtered sine wave"),
            ("noise * 0.1 | lpf(0.7, 2000)", "Low-pass filtered noise"),
            (
                "sawtooth(220) * Qw | lpf(0.5, 1500)",
                "Low-pass filtered sawtooth",
            ),
        ];

        for (expr, description) in lpf_examples {
            println!("Testing lpf: {} - {}", description, expr);

            tuun.parse(expr, "{}")
                .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", expr, e));

            let mut out = vec![0.0; 100];
            let more = tuun.process(&mut out);
            assert!(more, "Expected at least 100 samples for '{}'", expr);

            for (i, &sample) in out.iter().enumerate() {
                assert!(
                    sample >= -1.0 && sample <= 1.0,
                    "Sample {} out of range for '{}': {}",
                    i,
                    expr,
                    sample
                );
            }

            tuun.stop();
            println!("  ✓ lpf filter works correctly");
        }
    }
}
