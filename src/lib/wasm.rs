//! WebAssembly interface for Tuun music synthesizer.
//!
//! This module provides JavaScript-friendly bindings for the core synthesis engine.
//! The `Wasm` struct acts as a simple single-waveform tracker: it owns the currently
//! playing waveform and handles slider updates via `Waveform::Marked` + `waveform::substitute`.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

use wasm_bindgen::prelude::*;
use web_sys::console;

use crate::{builtins, eval, expr, generator, modules, optimizer, parser, slider, waveform};

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

/// Identifies which text a span's byte range indexes in the wasm runtime.
#[derive(Copy, Clone, Debug, PartialEq)]
enum Source {
    /// The user expression passed to [`Wasm::install`].
    Expression,
    /// The embedded module at this index of
    /// [`modules::EMBEDDED_MODULES`].
    Module(usize),
}

/// Renders `error` against the text its span indexes: `expression` for
/// expression errors, or the embedded module the error came from,
/// prefixed with that module's name. Falls back to the bare message for
/// errors with no span.
///
/// TODO include a source snippet (like `diagnostics::render_snippet`,
/// which is native-only); the widget's status area would need
/// preformatted rendering to keep the caret line aligned.
fn display_error(error: &expr::Error<Source>, expression: &str) -> String {
    match error.source() {
        Some(Source::Expression) => error.display_with_source(expression),
        Some(Source::Module(index)) => match modules::EMBEDDED_MODULES.get(index) {
            Some((name, content)) => format!("{}:{}", name, error.display_with_source(content)),
            None => error.to_string(),
        },
        None => error.to_string(),
    }
}

/// WebAssembly interface for the Tuun synthesizer.
///
/// Owns the currently-playing waveform.
#[wasm_bindgen(js_name = "Tuun")]
pub struct Wasm {
    sample_rate: i32,
    /// Per-instance prelude (built-ins + `sample_rate` + `tempo`) prepended
    /// to every install.
    prelude: Vec<expr::SourceBinding<MarkId, Source>>,
    /// Parsed embedded modules keyed by dotted path (e.g. `"std"`).
    /// Looked up by the `evaluate` resolve callback when an `Open`
    /// binding is processed.
    modules: HashMap<String, Vec<expr::SourceBinding<MarkId, Source>>>,
    waveform: Option<generator::Waveform<MarkId>>,
    last_slider_values: HashMap<String, f32>,
    buffer_duration: Duration,
}

/// Builds a `Definition` binding for `id = expr`.
fn def_binding(
    id: &str,
    expr: expr::SourceExpr<MarkId, Source>,
) -> expr::SourceBinding<MarkId, Source> {
    expr::Binding::Definition(expr::Pattern::Identifier(id.to_string()), expr).into()
}

/// Joins parse errors into a single user-visible string of messages,
/// without positions.
fn join_errors<S>(errors: &[expr::Error<S>]) -> String {
    errors
        .iter()
        .map(|e| e.to_string())
        .collect::<Vec<_>>()
        .join("; ")
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
        use expr::SourceExpr;
        // Set up better panic messages in the browser console
        console_error_panic_hook::set_once();

        // TODO this has a lot of repetition with the native app
        // Prelude: sample_rate + tempo + built-ins. Cloned into the
        // bindings vec on every install.
        let mut prelude: Vec<expr::SourceBinding<MarkId, Source>> = Vec::new();
        prelude.push(def_binding(
            "sample_rate",
            SourceExpr::float(sample_rate as f32),
        ));
        prelude.push(def_binding("tempo", SourceExpr::float(tempo)));
        builtins::add_bindings(&mut prelude);
        prelude.push(def_binding(
            "debug",
            builtins::debug(|message| console::log_1(&message.into())),
        ));

        // Parse every embedded module once. Anything that fails to parse
        // surfaces as a constructor error rather than a later evaluate
        // error, since modules are fixed at build time.
        //
        // Each module gets an implicit `open __prelude` prepended so its
        // bindings can reference prelude names (`sample_rate`, `tempo`,
        // built-ins) without depending on the caller having opened the
        // prelude first. Mirrors `evaluator::Evaluator::resolve` in
        // the native runtime.
        let mut modules: HashMap<String, Vec<expr::SourceBinding<MarkId, Source>>> = HashMap::new();
        for (index, (name, content)) in modules::EMBEDDED_MODULES.iter().enumerate() {
            let render = |errors: Vec<expr::Error<Source>>| {
                format!(
                    "Failed to parse module '{}': {}",
                    name,
                    errors
                        .iter()
                        .map(|e| e.display_with_source(content))
                        .collect::<Vec<_>>()
                        .join("; ")
                )
            };
            let (mut bindings, errors) =
                parser::parse_module::<MarkId, _>(content, Source::Module(index))
                    .map_err(&render)?;
            // Embedded modules are fixed at build time, so recoverable parse
            // errors mean the build itself is broken — fail loudly.
            if !errors.is_empty() {
                return Err(render(errors));
            }
            bindings.insert(0, expr::Binding::Open(vec!["__prelude".to_string()]).into());
            modules.insert((*name).to_string(), bindings);
        }

        Ok(Wasm {
            sample_rate,
            prelude,
            modules,
            waveform: None,
            last_slider_values: HashMap::new(),
            buffer_duration: Duration::from_secs_f32(128.0 / sample_rate as f32),
        })
    }

    /// Installs an expression as the current waveform: parses it, evaluates
    /// it under the slider bindings and opened modules, and stores the
    /// resulting waveform for [`Wasm::process`] to render.
    ///
    /// `slider_json` is a JSON object mapping slider names to initial values,
    /// for example, `{"volume": 0.5, "cutoff": 2000}`. Pass `"{}"` for no
    /// sliders.
    ///
    /// `open_json` is a JSON array of dotted module paths to bring into scope
    /// before evaluating, e.g. `["std", "foo.bar"]`. Each entry behaves like an
    /// `open` binding at the top of the expression. Pass `"[]"` for no opens.
    ///
    /// # Examples
    /// ```javascript
    /// tuun.install("sine(2764, 0)", "{}", "[]");
    /// tuun.install("$440 | lpf(0.5, 1900)", "{}", '["std"]');
    /// ```
    pub fn install(
        &mut self,
        expression: &str,
        slider_json: &str,
        open_json: &str,
    ) -> Result<(), String> {
        let parsed_expr = parser::parse_program::<MarkId, _>(expression, Source::Expression)
            .map_err(|errors| {
                let rendered: Vec<String> = errors
                    .iter()
                    .map(|e| e.display_with_source(expression))
                    .collect();
                format!("Parse errors: {}", rendered.join("; "))
            })?;
        let sliders = parse_json(slider_json)?;
        let opens = modules::parse_open_json(open_json)?;

        // Build the bindings vec passed to `evaluate`. Implicit
        // `open __prelude` first so the user expression can reference
        // prelude names directly (same prefix each embedded module
        // gets). Then user-requested opens, then slider bindings.
        let mut bindings: Vec<expr::SourceBinding<MarkId, Source>> = Vec::new();
        bindings.push(expr::Binding::Open(vec!["__prelude".to_string()]).into());
        for path in opens {
            bindings.push(expr::Binding::Open(path).into());
        }
        for (name, value) in &sliders {
            bindings.push(def_binding(
                name,
                expr::SourceExpr::from(expr::Expr::Waveform(waveform::Waveform::Marked {
                    id: MarkId::Slider(name.clone()),
                    waveform: Box::new(waveform::Waveform::Const(*value)),
                })),
            ));
        }

        // Resolve `Open(path)` either to the in-memory prelude (for the
        // special `__prelude` path used by modules) or to a dotted entry
        // in the embedded module table. Borrowed for the duration of
        // `evaluate`.
        let prelude = &self.prelude;
        let modules = &self.modules;
        let resolve = |path: &[String]| -> Result<
            &[expr::SourceBinding<MarkId, Source>],
            expr::Error<Source>,
        > {
            if path.len() == 1 && path[0] == "__prelude" {
                return Ok(prelude.as_slice());
            }
            let key = path.join(".");
            modules
                .get(&key)
                .map(|v| v.as_slice())
                .ok_or_else(|| expr::Error::new(format!("Module not found: {}", key)))
        };

        let expr = eval::evaluate(resolve, &bindings, parsed_expr)
            .map_err(|e| format!("Evaluate error: {}", display_error(&e, expression)))?;

        // TODO Do we want to precompute here?

        let waveform = match expr.expr {
            expr::Expr::Waveform(w) => w,
            expr::Expr::Seq { waveform, .. } => match waveform.expr {
                expr::Expr::Waveform(w) => w,
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
    /// tuun.install("$440", "{}", "[]");
    /// const done = tuun.process(output);
    /// ```
    pub fn process(&mut self, out: &mut [f32]) -> bool {
        self.buffer_duration = Duration::from_secs_f32(out.len() as f32 / self.sample_rate as f32);

        let waveform = match &mut self.waveform {
            Some(w) => w,
            None => return false,
        };

        let mut g = generator::Generator::new(self.sample_rate as u32);
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
    let sliders = parser::parse_sliders(&format!("sliders={}", input))
        .map_err(|errors| join_errors(&errors))?;

    // Manually build JSON since we don't have serde
    let entries: Vec<String> = sliders
        .iter()
        .map(|s| match &s.function {
            expr::SliderFunction::Linear {
                initial_value,
                min,
                max,
            } => {
                format!(
                    r#"{{"type":"linear","label":"{}","initial_value":{},"min":{},"max":{}}}"#,
                    s.label, initial_value, min, max
                )
            }
            expr::SliderFunction::UserDefined {
                normalized_initial_value,
                function_source,
            } => {
                let initial_value =
                    slider::denormalize(&s.function, *normalized_initial_value).unwrap_or(0.0);
                let value_at_0 = slider::denormalize(&s.function, 0.0).unwrap_or(0.0);
                let value_at_1 = slider::denormalize(&s.function, 1.0).unwrap_or(0.0);
                format!(
                    r#"{{"type":"user-defined","label":"{}","normalized_initial_value":{},"function_source":"{}","initial_value":{},"value_at_0":{},"value_at_1":{}}}"#,
                    s.label,
                    normalized_initial_value,
                    function_source.replace('\\', "\\\\").replace('"', "\\\""),
                    initial_value,
                    value_at_0,
                    value_at_1
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
    let function = expr::SliderFunction::UserDefined {
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

            tuun.install(expr, "{}", "[]")
                .unwrap_or_else(|e| panic!("Failed to install '{}': {}", expr, e));

            let mut out = vec![0.0; 100];
            let more = tuun.process(&mut out);
            assert!(more, "Expected at least 100 samples for '{}'", expr);

            for (i, &sample) in out.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&sample),
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
            let result = tuun.install(expr, "{}", "[]");
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
        // These all rely on names from the embedded `std` module (`Qw`,
        // `lpf`, `sawtooth`), so each install opens it explicitly.
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

            tuun.install(expr, "{}", r#"["std"]"#)
                .unwrap_or_else(|e| panic!("Failed to install '{}': {}", expr, e));

            let mut out = vec![0.0; 100];
            let more = tuun.process(&mut out);
            assert!(more, "Expected at least 100 samples for '{}'", expr);

            for (i, &sample) in out.iter().enumerate() {
                assert!(
                    (-1.0..=1.0).contains(&sample),
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

    #[test]
    fn test_open_unknown_module_errors() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");
        let result = tuun.install("sine(2764, 0)", "{}", r#"["does_not_exist"]"#);
        let message = result.expect_err("opening an unknown module should fail");
        assert!(
            message.contains("does_not_exist"),
            "expected error to name the missing module, got: {}",
            message
        );
    }
}
