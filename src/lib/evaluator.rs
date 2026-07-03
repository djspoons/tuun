//! Parses and evaluates program text — compute only, no tracker I/O.
//!
//! `Evaluator` owns the evaluation environment: the prelude (built-ins plus
//! environment-derived definitions) and the module cache backing `open`
//! directives. Everything takes `&self`; the module cache hides behind a
//! `RefCell`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fs;
use std::path;
use std::time;

use crate::builtins;
use crate::parser;
use crate::programs::{Evaluation, ProgramSet, ProgramSliders};
use crate::renderer::{self, MarkId};
use crate::slider;
use crate::waveform;

/// One slot in [`Evaluator`]'s module cache: the file's mtime at the time
/// we parsed it, plus a leaked `&'static` slice of the parsed bindings.
// See the field doc on `Evaluator::modules` for the leak strategy.
struct ModuleCacheEntry {
    mtime: time::SystemTime,
    bindings: &'static [parser::SourceBinding<MarkId>],
}

pub struct Evaluator {
    /// Built-ins + environment-derived definitions; implicitly opened at
    /// the top of every other loaded module — see [`Evaluator::resolve`].
    prelude: Vec<parser::SourceBinding<MarkId>>,
    /// Filesystem root for module resolution. A module path
    /// `["foo", "bar"]` is looked up as `<library_root>/foo/bar.tuun`.
    library_root: path::PathBuf,
    /// Cache of loaded modules, keyed by module path.
    ///
    /// Each entry pairs the file's last-seen mtime with a leaked `&'static`
    /// slice of bindings. On `resolve`, we stat the file and re-read+re-parse
    /// if the mtime has changed; otherwise we hand back the cached slice.
    ///
    /// Because previously-returned borrows must stay valid for the duration of
    /// an evaluate session, invalidation does NOT free the old allocation — the
    /// stale leak stays in memory until process exit. That's bounded by `(files
    /// loaded) * (edits per file)`, which is fine for tuun's interactive
    /// workflow.
    // TODO consider other ways of caching that don't depend on leak
    modules: RefCell<HashMap<Vec<String>, ModuleCacheEntry>>,
}

impl Evaluator {
    /// Builds an evaluator whose prelude defines `tempo`, `sample_rate`,
    /// and `mark` alongside the built-ins.
    pub fn new(sample_rate: u32, tempo: u32, library_root: path::PathBuf) -> Evaluator {
        let mut prelude = Vec::new();
        builtins::add_bindings(&mut prelude);

        use parser::SourceExpr;
        fn def<M>(id: &str, expr: SourceExpr<M>) -> parser::SourceBinding<M> {
            use parser::Binding;
            use parser::Pattern;
            Binding::Definition(Pattern::Identifier(id.to_string()), expr).into()
        }
        prelude.push(def("tempo", SourceExpr::float(tempo as f32)));
        prelude.push(def("sample_rate", SourceExpr::float(sample_rate as f32)));

        prelude.push(def(
            "mark",
            SourceExpr::from(parser::Expr::BuiltIn {
                name: "mark".to_string(),
                function: parser::BuiltInFn(std::rc::Rc::new(renderer::mark)),
            }),
        ));

        Evaluator {
            prelude,
            library_root,
            modules: RefCell::new(HashMap::new()),
        }
    }

    /// Resolves a module path to its (parsed) bindings.
    ///
    /// The special path `["__prelude"]` returns the in-memory prelude built by
    /// [`Evaluator::new`]. Other paths are mapped to a file under
    /// [`library_root`](Self::library_root): `["foo", "bar"]` →
    /// `<library_root>/foo/bar.tuun`.
    ///
    /// Each call stats the file and compares its mtime with the cached entry's.
    /// On a match we hand back the existing slice; on a mismatch or miss, we
    /// re-read, re-parse, prepend an implicit `open __prelude`, leak the new
    /// bindings, and replace the cache entry. The previous leaked allocation
    /// stays in memory — any borrows handed out before invalidation remain
    /// valid (essential for recursive resolve calls during one `evaluate`
    /// session).
    fn resolve(&self, path: &[String]) -> Result<&[parser::SourceBinding<MarkId>], parser::Error> {
        if path.len() == 1 && path[0] == "__prelude" {
            return Ok(&self.prelude);
        }

        // Build the file path: <library_root>/<path components>.tuun.
        let mut file_path = self.library_root.clone();
        for part in path {
            file_path.push(part);
        }
        file_path.set_extension("tuun");

        // Stat the file once. If the cache has a matching mtime, return the
        // cached slice without re-reading or re-parsing.
        let current_mtime = fs::metadata(&file_path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                parser::Error::new(format!(
                    "Failed to stat module {}: {}",
                    file_path.display(),
                    e
                ))
            })?;
        if let Some(entry) = self.modules.borrow().get(path)
            && entry.mtime == current_mtime
        {
            return Ok(entry.bindings);
        }

        // Cache miss or stale — reload.
        let contents = fs::read_to_string(&file_path).map_err(|e| {
            parser::Error::new(format!(
                "Failed to read module {}: {}",
                file_path.display(),
                e
            ))
        })?;
        // TODO these two error cases are a little hard to unravel, but :shrug:
        let (mut bindings, errors) =
            parser::parse_module::<MarkId>(&contents).map_err(|errors| {
                errors.into_iter().next().unwrap_or_else(|| {
                    parser::Error::new(format!("Parse failed for {}", file_path.display()))
                })
            })?;
        if !errors.is_empty() {
            errors.into_iter().next().unwrap_or_else(|| {
                parser::Error::new(format!("Parse failed for {}", file_path.display()))
            });
        }

        // Every loaded module implicitly opens the prelude as its first binding
        // so it can reference prelude names without an explicit `open __prelude`
        // line.
        bindings.insert(
            0,
            parser::Binding::Open(vec!["__prelude".to_string()]).into(),
        );

        // Leak the parsed bindings so we can hand out a stable
        // `&[SourceBinding]` reference. On a stale-cache reload, the previous
        // leak isn't freed — any borrows from before the reload (e.g. earlier
        // in the same evaluate session) remain valid.
        let leaked: &'static [parser::SourceBinding<MarkId>] =
            Box::leak(bindings.into_boxed_slice());
        self.modules.borrow_mut().insert(
            path.to_vec(),
            ModuleCacheEntry {
                mtime: current_mtime,
                bindings: leaked,
            },
        );
        Ok(leaked)
    }

    /// Parses and evaluates `text` under `bindings`, resolving `open`
    /// directives through the module cache. Returns the evaluated expression
    /// or a user-visible message.
    pub fn evaluate_source(
        &self,
        text: &str,
        bindings: &[parser::SourceBinding<MarkId>],
    ) -> Result<parser::SourceExpr<MarkId>, String> {
        let expr = match parser::parse_program(text) {
            Err(errors) => {
                println!("Errors while parsing input: {:?}", errors);
                return Err(format!("Error: {}", errors[0]));
            }
            Ok(expr) => expr,
        };
        println!("parser::parse_program returned: {}", &expr);

        match parser::evaluate(|path| self.resolve(path), bindings, expr) {
            Err(error) => {
                println!("Errors while evaluating input: {:?}", error);
                Err(format!("Error: {}", error))
            }
            Ok(expr) => {
                println!("parser::evaluate returned: {}", &expr);
                Ok(expr)
            }
        }
    }

    /// Evaluates the program at `index` and classifies the result as a
    /// playable waveform, a keys instrument, or invalid.
    ///
    /// Keys-instrument candidates (functions) are sanity-checked by
    /// actually invoking them with dummy note/velocity arguments.
    pub fn evaluate_program(&self, set: &ProgramSet, index: usize) -> Evaluation {
        const NOT_A_PROGRAM: &str = "Program is not a waveform or keys instrument";
        let bindings = set.evaluation_bindings(index);
        let expr = match self.evaluate_source(set.programs()[index].text(), &bindings) {
            Err(message) => return Evaluation::Invalid(message),
            Ok(expr) => expr,
        };
        match expr.expr {
            parser::Expr::Waveform(w) => Evaluation::Waveform(w),
            parser::Expr::Seq { waveform, .. } => {
                if let parser::Expr::Waveform(w) = waveform.expr {
                    Evaluation::Waveform(w)
                } else {
                    Evaluation::Invalid(NOT_A_PROGRAM.to_string())
                }
            }
            parser::Expr::Function { .. } | parser::Expr::BuiltIn { .. } => {
                // Sanity check: actually invoke with dummy args.
                // TODO use a waveform for velocity
                match self.apply_note_function(
                    &expr,
                    vec![
                        parser::SourceExpr::float(60.0),
                        parser::SourceExpr::float(0.7),
                    ],
                    set.programs()[index].sliders(),
                ) {
                    Ok(_) => Evaluation::KeysInstrument(expr),
                    Err(message) => Evaluation::Invalid(message),
                }
            }
            _ => Evaluation::Invalid(NOT_A_PROGRAM.to_string()),
        }
    }

    /// Applies a note function `expr` to the given `args`, expecting a pair
    /// of (note-on, note-off) waveforms as a result.
    ///
    /// The expressions `expr` and `args` should be closed except for
    /// references to `sliders`, which are bound at their current values.
    pub fn apply_note_function(
        &self,
        expr: &parser::SourceExpr<MarkId>,
        args: Vec<parser::SourceExpr<MarkId>>,
        sliders: &ProgramSliders,
    ) -> Result<(waveform::Waveform<MarkId>, waveform::Waveform<MarkId>), String> {
        use parser::Expr::{Tuple, Waveform};
        let expr = parser::SourceExpr::from(parser::Expr::Application {
            function: Box::new(expr.clone()),
            argument: Box::new(parser::SourceExpr::from(Tuple(args))),
        });
        let mut bindings = vec![];
        slider::append_slider_bindings(
            sliders.configs(),
            sliders.normalized_values(),
            MarkId::Slider,
            &mut bindings,
        );
        let resolve = |_: &[String]| {
            Err(parser::Error::new(
                "Didn't expect to resolve in apply_note_function".to_string(),
            ))
        };
        let expr = parser::evaluate(resolve, &bindings, expr).map_err(|e| e.to_string());
        match expr.map(|s| s.expr) {
            Ok(Tuple(mut exprs)) => {
                if exprs.len() != 2 {
                    return Err(format!(
                        "Expected 2 waveforms for note, got {} elements",
                        exprs.len()
                    ));
                }
                match (exprs.remove(0).expr, exprs.remove(0).expr) {
                    (Waveform(note_on), Waveform(note_off)) => Ok((note_on, note_off)),
                    (expr, Waveform(_)) => {
                        Err(format!("Expected waveform for note-on, got: {}", expr))
                    }
                    (_, expr) => Err(format!("Expected waveform for note-off, got: {}", expr)),
                }
            }
            Ok(expr) => Err(format!("Expected 2 waveforms for note, got: {}", expr)),
            Err(e) => Err(format!("Error evaluating note: {}", e)),
        }
    }
}
