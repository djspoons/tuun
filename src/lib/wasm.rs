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

use wasm_bindgen::prelude::*;

use crate::{builtins, generator, optimizer, parser};

/// WebAssembly interface for the Tuun synthesizer.
///
/// Provides parsing, optimization, and audio generation from Tuun expressions.
#[wasm_bindgen(js_name = "Tuun")]
pub struct Wasm {
    sample_rate: i32,
    context: Vec<(String, parser::Expr)>,
}

#[wasm_bindgen(js_class = "Tuun")]
impl Wasm {
    /// Creates a new Tuun instance with the specified sample rate.
    ///
    /// # Arguments
    /// * `sample_rate` - The audio sample rate in Hz (e.g., 44100)
    #[wasm_bindgen(constructor)]
    pub fn new(sample_rate: i32) -> Result<Wasm, String> {
        use parser::Expr;
        // Set up better panic messages in the browser console
        console_error_panic_hook::set_once();

        // TODO this has a lot of repetition with main.rs

        // Initialize context with builtins
        let mut context = Vec::new();
        // sampling_frequency must be a Float, not a Waveform, for use in calculations (e.g., lpf)
        context.push((
            "sampling_frequency".to_string(),
            Expr::Float(sample_rate as f32),
        ));
        builtins::add_prelude(&mut context);

        // TODO make this configurable
        let tempo = 120.0;
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
                    match parser::simplify(&context, expr) {
                        Ok(simplified) => {
                            if let Err(e) =
                                parser::extend_context(&mut context, &pattern, &simplified)
                            {
                                eprintln!("Warning: Failed to add context definition: {:?}", e);
                            }
                        }
                        Err(e) => eprintln!("Warning: Failed to simplify context: {:?}", e),
                    }
                }
            }
            Err(e) => eprintln!("Warning: Failed to parse context file: {:?}", e),
        }

        Ok(Wasm {
            sample_rate,
            context,
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

        // Simplify the expression
        let simplified = parser::simplify(&self.context, parsed_expr)
            .map_err(|e| format!("Simplify error: {:?}", e))?;

        // Extract the waveform from the expression
        match simplified {
            parser::Expr::Waveform(waveform) => {
                let (_, waveform) = optimizer::replace_seq(waveform);
                let waveform = optimizer::simplify(waveform);

                // Initialize the waveform state for generation
                let waveform = generator::initialize_state(waveform);
                Ok(WasmWaveform { inner: waveform })
            }
            other => Err(format!(
                "Expression did not evaluate to a waveform, got: {:?}",
                other
            )),
        }
    }

    /// Generates audio samples from a waveform.
    ///
    /// # Arguments
    /// * `waveform` - The WasmWaveform to generate from
    /// * `num_samples` - The number of samples to generate
    ///
    /// # Returns
    /// A Float32Array of audio samples in the range [-1.0, 1.0]
    ///
    /// # Example
    /// ```javascript
    /// const samples = tuun.generate(waveform, 4096);
    /// // samples is a Float32Array that can be used with Web Audio API
    /// ```
    pub fn generate(&self, waveform: &mut WasmWaveform, num_samples: usize) -> Vec<f32> {
        let mut audio_gen = generator::Generator::new(self.sample_rate);
        audio_gen.slider_state = None;
        audio_gen.capture_state = None;

        let (updated_waveform, samples) = audio_gen.generate(waveform.inner.clone(), num_samples);
        waveform.inner = updated_waveform;

        samples
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
/// generation calls for time-dependent waveforms.
#[wasm_bindgen]
pub struct WasmWaveform {
    inner: generator::Waveform,
}

#[wasm_bindgen]
impl WasmWaveform {
    /// Returns a string representation of the waveform (for debugging).
    #[wasm_bindgen(js_name = toString)]
    pub fn to_string(&self) -> String {
        format!("{:?}", self.inner)
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
        let tuun = Wasm::new(44100).expect("Failed to create Tuun instance");

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
        let tuun = Wasm::new(44100).expect("Failed to create Tuun instance");

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
        let tuun = Wasm::new(44100).expect("Failed to create Tuun instance");

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
