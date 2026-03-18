//! WebAssembly interface for Tuun music synthesizer.
//!
//! This module provides JavaScript-friendly bindings for the core synthesis engine.
//!
//! # Example (JavaScript)
//! ```javascript
//! import init, { Tuun } from './pkg/tuun.js';
//!
//! await init();
//! const tuun = new Tuun(44100);
//! const waveform = tuun.parse("sine(2764, 0)");
//! const samples = tuun.generate(waveform, 4096);
//! ```

use std::collections::HashMap;

use wasm_bindgen::prelude::*;

use crate::{builtins, generator, optimizer, parser};

/// WebAssembly interface for the Tuun synthesizer.
///
/// Provides parsing, optimization, and audio generation from Tuun expressions.
#[wasm_bindgen(js_name = "Tuun")]
pub struct Wasm {
    sample_rate: i32,
    context: Vec<(String, parser::Expr)>,
    slider_state: generator::SliderState,
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
                            if let Err(e) =
                                parser::extend_context(&mut context, &pattern, &expr)
                            {
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
            slider_state: generator::SliderState {
                last_values: HashMap::new(),
                values: HashMap::new(),
                buffer_length: 0,
                buffer_position: 0,
            },
        })
    }

    /// Parses a Tuun expression and returns a WasmWaveform.
    ///
    /// # Arguments
    /// * `expression` - The Tuun expression string to parse
    ///
    /// # Returns
    /// A WasmWaveform that can be used with `generate()`, or an error string
    ///
    /// # Example
    /// ```javascript
    /// const waveform = tuun.parse("sine(2764, 0)");
    /// ```
    pub fn parse(&self, expression: &str) -> Result<WasmWaveform, String> {
        // Parse the program
        let parsed_expr = parser::parse_program(expression)
            .map_err(|errors| format!("Parse errors: {:?}", errors))?;

        // Evaluate the expression
        let expr = parser::evaluate(&self.context, parsed_expr)
            .map_err(|e| format!("Evaluate error: {:?}", e))?;

        // Extract the waveform from the expression
        match expr {
            parser::Expr::Waveform(waveform) => {
                let waveform = optimizer::simplify(waveform);
                // TODO could precompute here as well

                // Initialize the waveform state for generation
                let waveform = generator::initialize_state(waveform);
                Ok(WasmWaveform { inner: waveform })
            }
            parser::Expr::Seq { waveform, .. } => {
                match *waveform {
                    parser::Expr::Waveform(waveform) => {
                        let waveform = optimizer::simplify(waveform);
                        // TODO could precompute here as well

                        // Initialize the waveform state for generation
                        let waveform = generator::initialize_state(waveform);
                        Ok(WasmWaveform { inner: waveform })
                    }
                    _ => panic!("Got non-Waveform in seq after evaluate"),
                }
            }
            other => Err(format!(
                "Expression did not evaluate to a waveform, got: {:?}",
                other
            )),
        }
    }

    pub fn set_slider_value(&mut self, name: &str, value: f32) {
        self.slider_state.values.insert(name.to_string(), value);
    }

    /// Generates audio samples from a waveform. Updates the internal state
    /// of the waveform so that the next call to `generate()` will continue
    /// from the point at which this call left off.
    ///
    /// # Arguments
    /// * `waveform` - The WasmWaveform to generate from
    /// * `desired` - The number of samples to generate
    ///
    /// # Returns
    /// A Float32Array of audio samples
    ///
    /// # Example
    /// ```javascript
    /// const samples = tuun.generate(waveform, 4096);
    /// // samples is a Float32Array that can be used with Web Audio API
    /// ```
    pub fn generate(&mut self, waveform: &mut WasmWaveform, desired: usize) -> Vec<f32> {
        self.slider_state.buffer_length = desired;
        self.slider_state.buffer_position = 0;

        let mut g = generator::Generator::new(self.sample_rate);
        g.slider_state = Some(&self.slider_state);

        let out = g.generate(&mut waveform.inner, desired);

        self.slider_state.last_values = self.slider_state.values.clone();
        return out;
    }

    /// Returns the current sample rate.
    #[wasm_bindgen(getter)]
    pub fn sample_rate(&self) -> i32 {
        self.sample_rate
    }
}

/// A waveform that can be used to generate audio samples.
///
/// This wraps the internal Waveform type and maintains state between
/// calls to generate().
#[wasm_bindgen]
pub struct WasmWaveform {
    inner: generator::Waveform,
}

#[wasm_bindgen]
impl WasmWaveform {
    /// Returns a string representation of the waveform (for debugging).
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("{}", self.inner)
    }
}

// Additional utility functions that don't need to be part of a struct

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

            // Test parsing
            let mut waveform = tuun
                .parse(expr)
                .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", expr, e));

            // Test generation (small sample to verify it doesn't crash)
            let samples = tuun.generate(&mut waveform, 100);

            assert_eq!(samples.len(), 100, "Expected 100 samples for '{}'", expr);

            for (i, &sample) in samples.iter().enumerate() {
                assert!(
                    sample >= -1.0 && sample <= 1.0,
                    "Sample {} out of range for '{}': {}",
                    i,
                    expr,
                    sample
                );
            }

            println!("  ✓ Parsed and generated successfully");
        }
    }

    /// Test that invalid expressions produce appropriate errors
    #[test]
    fn test_invalid_expressions() {
        let tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let invalid_examples = vec![
            "undefined_function()",
            "sine(2764)", // Wrong number of args
            "1 + ",       // Incomplete expression
        ];

        for expr in invalid_examples {
            println!("Testing invalid expression: {}", expr);

            let result = tuun.parse(expr);
            assert!(
                result.is_err(),
                "Expected error for invalid expression '{}', but got success",
                expr
            );

            println!("  ✓ Correctly rejected");
        }
    }

    /// Test context functions that require special definitions
    #[test]
    fn test_context_functions() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        // Test lpf with various expressions
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

            let mut waveform = tuun
                .parse(expr)
                .unwrap_or_else(|e| panic!("Failed to parse '{}': {}", expr, e));

            let samples = tuun.generate(&mut waveform, 100);
            assert_eq!(samples.len(), 100, "Expected 100 samples for '{}'", expr);

            // Verify samples are in valid range [-1.0, 1.0]
            for (i, &sample) in samples.iter().enumerate() {
                assert!(
                    sample >= -1.0 && sample <= 1.0,
                    "Sample {} out of range for '{}': {}",
                    i,
                    expr,
                    sample
                );
            }

            println!("  ✓ lpf filter works correctly");
        }
    }
}
