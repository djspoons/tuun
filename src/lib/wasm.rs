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

use crate::{
    builtins, eval, expr, generator, infer, modules, optimizer, parser, signatures, slider,
    slider_analysis, trace, waveform,
};

#[derive(Clone, Debug, PartialEq)]
enum MarkId {
    /// A slider, named by its label alone, since the browser plays one
    /// expression, which lists none.
    Slider(String),
}

impl fmt::Display for MarkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MarkId::Slider(name) => write!(f, "slider({:?})", name),
        }
    }
}

// The browser doesn't light program text, so its marks carry no trace and
// nothing it evaluates has trace marks to strip.
impl trace::CarriesTrace<Source> for MarkId {
    fn from_slider(label: &str, _provider: slider::Provider) -> Option<Self> {
        Some(MarkId::Slider(label.to_string()))
    }
}

/// A slider of the playing waveform: its spec and current value.
struct PlayingSlider {
    spec: slider::SliderSpec,
    value: f32,
}

/// Returns `text` as a JSON string literal.
fn json_string(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Returns `reached` as a JSON array of `{label, default, min, max, curve}`
/// objects, in order.
fn sliders_json(reached: &[slider_analysis::Reached]) -> String {
    let entries: Vec<String> = reached
        .iter()
        .map(|slider| {
            format!(
                r#"{{"label":{},"default":{},"min":{},"max":{},"curve":"{}"}}"#,
                json_string(&slider.label),
                slider.spec.default,
                slider.spec.min,
                slider.spec.max,
                slider.spec.curve
            )
        })
        .collect();
    format!("[{}]", entries.join(","))
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
    /// The current value of each slider the playing waveform reaches, by label,
    /// with its spec.
    sliders: HashMap<String, PlayingSlider>,
    buffer_duration: Duration,
    /// Module export types and findings, carried between installs. The
    /// embedded modules never change within an instance, so entries stay
    /// valid for its whole life.
    module_types: infer::ModuleCache<Source>,
}

/// Builds a `Definition` binding for `id = expr`.
fn def_binding(
    id: &str,
    expr: expr::SourceExpr<MarkId, Source>,
) -> expr::SourceBinding<MarkId, Source> {
    expr::Binding::Definition(expr::Pattern::Identifier(id.to_string()), expr).into()
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
        // prelude first. Mirrors `environment::Environment::resolve` in
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
            sliders: HashMap::new(),
            buffer_duration: Duration::from_secs_f32(128.0 / sample_rate as f32),
            module_types: infer::ModuleCache::default(),
        })
    }

    /// Returns the sliders `expression` reaches, as a JSON array of `{label,
    /// default, min, max, curve}` objects in the order the expression first
    /// reaches their calls, without installing it.
    ///
    /// `open_json` and `use_json` are as for [`Wasm::install`]. Returns the
    /// error [`Wasm::install`] would.
    ///
    /// # Examples
    /// ```javascript
    /// tuun.sliders('$220 * slider("level", 0.5)', '["std"]', "[]");
    /// // '[{"label":"level","default":0.5,"min":0,"max":1,"curve":"linear"}]'
    /// ```
    pub fn sliders(
        &mut self,
        expression: &str,
        open_json: &str,
        use_json: &str,
    ) -> Result<String, String> {
        let (_, reached) = self.evaluate(expression, open_json, use_json)?;
        Ok(sliders_json(&reached))
    }

    /// Installs an expression as the current waveform: parses it, evaluates
    /// it under the opened modules, and stores the resulting waveform for
    /// [`Wasm::process`] to render. Returns the sliders it reaches, as
    /// [`Wasm::sliders`] does.
    ///
    /// `value_json` is a JSON object mapping slider labels to the values they
    /// start at, for example `{"volume": 0.5, "cutoff": 2000}`. A value
    /// outside its slider's range is clamped; a slider with no value starts
    /// at its default, and a label the expression doesn't reach is ignored.
    /// Pass `"{}"` for every slider's default.
    ///
    /// `open_json` is a JSON array of dotted module paths to bring into scope
    /// before evaluating, e.g. `["std", "foo.bar"]`. Each entry behaves like an
    /// `open` binding at the top of the expression. Pass `"[]"` for no opens.
    ///
    /// `use_json` is a JSON array of dotted module paths to bind as module
    /// values, e.g. `["synth.pm"]`. Each entry behaves like a `use` binding
    /// at the top of the expression: the module is bound to its last path
    /// component and its bindings are reached by `.name` projection. Pass
    /// `"[]"` for no uses.
    ///
    /// # Examples
    /// ```javascript
    /// tuun.install("$440", "{}", '["std"]', "[]");
    /// tuun.install("std.square(220) * 0.3", "{}", "[]", '["std"]');
    /// ```
    pub fn install(
        &mut self,
        expression: &str,
        value_json: &str,
        open_json: &str,
        use_json: &str,
    ) -> Result<String, String> {
        let values = parse_json(value_json)?;
        let (mut waveform, reached) = self.evaluate(expression, open_json, use_json)?;
        let mut sliders = HashMap::new();
        for slider in &reached {
            let value = values
                .get(&slider.label)
                .map_or(slider.spec.default, |&value| slider.spec.clamp(value));
            let id = MarkId::Slider(slider.label.clone());
            let marked = waveform::Waveform::Marked {
                id: id.clone(),
                waveform: Box::new(waveform::Waveform::Const(value)),
            };
            waveform::substitute(&mut waveform, &id, &marked);
            sliders.insert(
                slider.label.clone(),
                PlayingSlider {
                    spec: slider.spec.clone(),
                    value,
                },
            );
        }
        self.waveform = Some(generator::initialize_state(waveform));
        self.sliders = sliders;
        Ok(sliders_json(&reached))
    }

    /// Returns the optimized waveform `expression` evaluates to, under the
    /// modules `open_json` and `use_json` name, and the sliders it reaches.
    fn evaluate(
        &mut self,
        expression: &str,
        open_json: &str,
        use_json: &str,
    ) -> Result<(waveform::Waveform<MarkId>, Vec<slider_analysis::Reached>), String> {
        let parsed_expr = parser::parse_program::<MarkId, _>(expression, Source::Expression)
            .map_err(|errors| {
                let rendered: Vec<String> = errors
                    .iter()
                    .map(|e| e.display_with_source(expression))
                    .collect();
                format!("Parse errors: {}", rendered.join("; "))
            })?;
        let opens = modules::parse_module_paths_json(open_json)?;
        let uses = modules::parse_module_paths_json(use_json)?;

        // Build the bindings vec passed to `evaluate`. Implicit
        // `open __prelude` first so the user expression can reference
        // prelude names directly (same prefix each embedded module
        // gets). Then user-requested opens and uses.
        let mut bindings: Vec<expr::SourceBinding<MarkId, Source>> = Vec::new();
        bindings.push(expr::Binding::Open(vec!["__prelude".to_string()]).into());
        for path in opens {
            bindings.push(expr::Binding::Open(path).into());
        }
        for path in uses {
            bindings.push(expr::Binding::Use(path).into());
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
                .ok_or_else(|| expr::Error::eval_here(format!("Module not found: {}", key)))
        };

        // The checker runs first and error findings gate the install,
        // mirroring the native `processing::process`. The expression must
        // be playable — produce a waveform or a seq — matching the
        // waveform check below.
        let (findings, _) = infer::check_program(
            resolve,
            &bindings,
            signatures::signature,
            &parsed_expr,
            Some(infer::Expectation::Playable),
            &mut self.module_types,
        );
        if !findings.is_empty() {
            let rendered: Vec<String> = findings
                .iter()
                .map(|finding| display_error(finding, expression))
                .collect();
            return Err(format!("Type errors: {}", rendered.join("; ")));
        }
        // The expression lists no sliders, so it provides all it reaches.
        // A closure of its own, so the analysis borrows the expression only
        // for as long as it runs.
        let analysis = slider_analysis::analyze(
            |path| resolve(path),
            &slider_analysis::Program {
                bindings: &bindings,
                expr: &parsed_expr,
                keys: false,
                listings: slider::Listings::new(|_| None),
                name: &|_| "the expression".to_string(),
                annotation: &|_| None,
            },
        );
        if !analysis.errors.is_empty() {
            let rendered: Vec<String> = analysis
                .errors
                .iter()
                .map(|error| display_error(error, expression))
                .collect();
            return Err(format!("Slider errors: {}", rendered.join("; ")));
        }

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

        Ok((optimizer::optimize(waveform), analysis.reached))
    }

    /// Drops the current waveform.
    pub fn stop(&mut self) {
        self.waveform = None;
        self.sliders.clear();
    }

    /// Moves the slider `name` of the playing waveform to `value`, clamped to
    /// its range, ramping from its last value. Does nothing if nothing is
    /// playing or the waveform doesn't reach that slider.
    pub fn update_slider(&mut self, name: &str, value: f32) {
        let (Some(waveform), Some(state)) = (&mut self.waveform, self.sliders.get_mut(name)) else {
            return;
        };
        let value = state.spec.clamp(value);
        let last_value = state.value;
        state.value = value;

        let mark_id = MarkId::Slider(name.to_string());
        let ramp = slider::make_ramp(
            mark_id.clone(),
            last_value,
            value,
            self.buffer_duration.as_secs_f32(),
        );
        let ramp = generator::initialize_state(ramp);
        waveform::substitute(waveform, &mark_id, &ramp);
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
    /// tuun.install("$440", "{}", '["std"]', "[]");
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

/// Returns the spec of a slider over `min` to `max` with the curve named
/// `curve`, or why there is none.
fn range_spec(min: f32, max: f32, curve: &str) -> Result<slider::SliderSpec, String> {
    let curve = slider::Curve::named(curve)
        .ok_or_else(|| format!("the curve must be \"linear\" or \"exp\", not {:?}", curve))?;
    slider::SliderSpec::new(min, min, max, curve)
}

/// Returns the value at `position`, from 0 at `min` to 1 at `max`, along a
/// slider with the curve named `curve` (see
/// [`slider::SliderSpec::denormalize`]).
#[wasm_bindgen(js_name = "sliderValue")]
pub fn slider_value(min: f32, max: f32, curve: &str, position: f32) -> Result<f32, String> {
    Ok(range_spec(min, max, curve)?.denormalize(position))
}

/// Returns the position of `value` along a slider over `min` to `max` with
/// the curve named `curve`, the inverse of [`slider_value`].
#[wasm_bindgen(js_name = "sliderPosition")]
pub fn slider_position(min: f32, max: f32, curve: &str, value: f32) -> Result<f32, String> {
    Ok(range_spec(min, max, curve)?.normalize(value))
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
            ("sin(6.2831853 * phase(440, 0))", "Sine wave (440 Hz)"),
            ("noise * 0.1", "Noise"),
        ];

        for (expr, description) in examples {
            println!("Testing: {} - {}", description, expr);

            tuun.install(expr, "{}", "[]", "[]")
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

    /// Type errors gate the install, before evaluation, with rendered
    /// positions; sliders type as waveforms; clean expressions with
    /// sliders, opens, and uses still install.
    #[test]
    fn test_type_errors_gate_install() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let error = tuun
            .install("time | fin(\"a\")", "{}", "[]", "[]")
            .expect_err("a string length should gate the install");
        assert!(
            error.starts_with("Type errors: 1:12: expected waveform, found string"),
            "unexpected message: {}",
            error
        );
        let error = tuun
            .install("seq(0)(1) + seq(0)(2)", "{}", "[]", "[]")
            .expect_err("two seqs should gate the install");
        assert!(
            error.contains("cannot combine two seqs with +"),
            "unexpected message: {}",
            error
        );
        // A slider is a marked waveform: usable anywhere a waveform is,
        // gated where a waveform is not.
        tuun.install("$220 * slider(\"volume\", 0.5)", "{}", "[\"std\"]", "[]")
            .expect("a slider-scaled tone should install");
        let error = tuun
            .install("nth(slider(\"volume\", 0.5), [1, 2, 3])", "{}", "[]", "[]")
            .expect_err("a slider index should gate the install");
        assert!(
            error.contains("expected int, found waveform"),
            "unexpected message: {}",
            error
        );
    }

    /// A `slider` call in the expression is the slider of that label, which
    /// `update_slider` moves.
    #[test]
    fn test_slider_builtin_is_live() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");
        tuun.install("slider(\"level\", 0.25)", "{}", "[]", "[]")
            .expect("a slider should install");
        let mut out = vec![0.0; 4];
        tuun.process(&mut out);
        assert_eq!(out, [0.25; 4]);
        // The move ramps from the value the slider started at, over one
        // buffer, then holds.
        tuun.update_slider("level", 1.0);
        tuun.process(&mut out);
        assert_eq!(out[0], 0.25);
        tuun.process(&mut out);
        assert!(out.iter().all(|x| (x - 1.0).abs() < 1e-5), "{:?}", out);
    }

    /// The sliders an expression reaches are listed once each, in the order
    /// its text reaches their calls (here, not the order the waveform holds
    /// them), without installing anything.
    #[test]
    fn test_sliders_lists_each_reached_slider_once_in_order() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");
        let sliders = tuun
            .sliders(
                "let level = slider(\"level\", 0.5) in \
                 $(slider(\"freq\", 200, 50, 1000, \"exp\")) * level * level",
                "[\"std\"]",
                "[]",
            )
            .expect("the expression evaluates");
        assert_eq!(
            sliders,
            r#"[{"label":"level","default":0.5,"min":0,"max":1,"curve":"linear"},{"label":"freq","default":200,"min":50,"max":1000,"curve":"exp"}]"#
        );
        assert!(!tuun.is_playing());
        let error = tuun
            .sliders("slider(\"a\", 0.5) + slider(\"a\", 2, 0, 4)", "[]", "[]")
            .expect_err("calls that disagree are an error");
        assert!(
            error.contains("slider \"a\" of the expression is")
                && error.contains("calls of one slider must agree"),
            "unexpected message: {}",
            error
        );
    }

    /// An install starts each slider at its value in `value_json`, clamped to
    /// its range, or at its default, and returns the sliders it reaches.
    #[test]
    fn test_install_starts_sliders_at_the_given_values() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");
        let sliders = tuun
            .install(
                "slider(\"a\", 0.25) + slider(\"b\", 0.5, 0, 2) + slider(\"c\", 0.125)",
                r#"{"a": 0.5, "b": 3, "gone": 1}"#,
                "[]",
                "[]",
            )
            .expect("the expression installs");
        assert!(sliders.starts_with(r#"[{"label":"a","#), "{}", sliders);
        let mut out = vec![0.0; 4];
        tuun.process(&mut out);
        // 0.5 as given, 2 clamped from 3, and c's default.
        assert_eq!(out, [2.625; 4]);
        // A move ramps from the value the slider started at.
        tuun.update_slider("b", 0.0);
        tuun.process(&mut out);
        assert!(out[0] > 0.625 && out[3] < out[0], "{:?}", out);
        // A move is clamped to the slider's range, and a label the waveform
        // doesn't reach moves nothing.
        tuun.update_slider("a", 5.0);
        tuun.update_slider("gone", 5.0);
        tuun.process(&mut out);
        tuun.process(&mut out);
        assert!((out[3] - 1.125).abs() < 1e-5, "{:?}", out);
    }

    /// The exported mapping follows the slider's curve both ways.
    #[test]
    fn test_slider_value_and_position_follow_the_curve() {
        assert_eq!(slider_value(0.0, 10.0, "linear", 0.5), Ok(5.0));
        let value = slider_value(100.0, 10000.0, "exp", 0.5).unwrap();
        assert!((value - 1000.0).abs() < 0.1, "{}", value);
        let position = slider_position(100.0, 10000.0, "exp", 1000.0).unwrap();
        assert!((position - 0.5).abs() < 1e-4, "{}", position);
        assert!(slider_value(0.0, 1.0, "log", 0.5).is_err());
    }

    /// Test that invalid expressions produce appropriate errors
    #[test]
    fn test_invalid_expressions() {
        let mut tuun = Wasm::new(44100, 120.0).expect("Failed to create Tuun instance");

        let invalid_examples = vec!["undefined_function()", "log(2764)", "1 + "];

        for expr in invalid_examples {
            println!("Testing invalid expression: {}", expr);
            let result = tuun.install(expr, "{}", "[]", "[]");
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
        // These all rely on names from the embedded `std` and `filters.rbj`
        // modules (`Qw`, `lpf`, `sawtooth`), so each install opens it
        // explicitly.
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

            tuun.install(expr, "{}", r#"["std", "filters.rbj"]"#, "[]")
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
        let result = tuun.install("time", "{}", r#"["does_not_exist"]"#, "[]");
        let message = result.expect_err("opening an unknown module should fail");
        assert!(
            message.contains("does_not_exist"),
            "expected error to name the missing module, got: {}",
            message
        );
    }
}
