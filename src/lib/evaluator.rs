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
use crate::ids::MarkId;
use crate::parser;
use crate::programs::{Evaluation, ProgramSet, ProgramSliders};
use crate::slider;
use crate::waveform;

/// The `mark(N)` built-in: wraps a waveform in a `MarkId::UserDefined`
/// mark.
fn mark(arguments: Vec<parser::Expr<MarkId>>) -> parser::Expr<MarkId> {
    match &arguments[..] {
        [parser::Expr::Float(id)] if *id >= 1.0 && id.fract() == 0.0 => {
            let id = id.round() as u32;
            parser::Expr::BuiltIn {
                name: format!("mark({})", id),
                function: builtins::curry(move |waveform: Box<waveform::Waveform<MarkId>>| {
                    waveform::Waveform::Marked {
                        id: MarkId::UserDefined(id),
                        waveform,
                    }
                }),
            }
        }
        _ => parser::Expr::Error("Invalid argument for mark".to_string()),
    }
}

/// Returns a user-visible error for a failed module parse, naming the file
/// and including the first parse error when there is one.
fn module_parse_error(file_path: &path::Path, errors: Vec<parser::Error>) -> parser::Error {
    match errors.into_iter().next() {
        Some(error) => parser::Error::new(format!("{}: {}", file_path.display(), error)),
        None => parser::Error::new(format!("Parse failed for {}", file_path.display())),
    }
}

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
                function: parser::BuiltInFn(std::rc::Rc::new(mark)),
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
        // A module that parses with recoverable errors is still broken —
        // report it rather than evaluating with error placeholders.
        let (mut bindings, errors) = parser::parse_module::<MarkId>(&contents)
            .map_err(|errors| module_parse_error(&file_path, errors))?;
        if !errors.is_empty() {
            return Err(module_parse_error(&file_path, errors));
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
            Err(errors) => return Err(format!("Error: {}", errors[0])),
            Ok(expr) => expr,
        };

        parser::evaluate(|path| self.resolve(path), bindings, expr)
            .map_err(|error| format!("Error: {}", error))
    }

    /// Evaluates the program at `index` and classifies the result as a
    /// playable waveform, a keys instrument, or invalid.
    ///
    /// Keys-instrument candidates (functions) are sanity-checked by
    /// actually invoking them with dummy note/velocity arguments.
    pub fn evaluate_program(&self, set: &ProgramSet, index: usize) -> Evaluation {
        // TODO could improve error messages here
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::player::substitute_current_slider_values;
    use std::path::PathBuf;

    /// Walks `waveform` and collects the `Const` value under every
    /// `Marked(Slider(label), …)` node.
    fn slider_mark_values(waveform: &waveform::Waveform<MarkId>, found: &mut Vec<(String, f32)>) {
        use waveform::Waveform;
        match waveform {
            Waveform::Marked { id, waveform } => {
                if let MarkId::Slider(label) = id
                    && let Waveform::Const(v) = **waveform
                {
                    found.push((label.clone(), v));
                }
                slider_mark_values(waveform, found);
            }
            Waveform::BinaryPointOp(_, a, b) => {
                slider_mark_values(a, found);
                slider_mark_values(b, found);
            }
            Waveform::Fin { length, waveform } => {
                slider_mark_values(length, found);
                slider_mark_values(waveform, found);
            }
            _ => {}
        }
    }

    #[test]
    fn keys_note_on_reflects_current_slider_value() {
        // The function body needs no built-ins: the slider binding `vol` is
        // itself a (marked) waveform, so `(vol, vol)` is a valid
        // (note-on, note-off) pair.
        let (mut set, _) = ProgramSet::from_source(
            "#{sliders=[\"vol:0.5:0:1\"]}\nk = fn(note, vel) => (vol, vol);".to_string(),
            PathBuf::new(),
        )
        .expect("test source should parse");
        let evaluator = Evaluator::new(44100, 90, PathBuf::new());

        // The program classifies as a keys instrument.
        let Evaluation::KeysInstrument(function) = evaluator.evaluate_program(&set, 0) else {
            panic!("expected a keys instrument");
        };

        // A note played at the initial slider position carries vol = 0.5.
        let args = vec![
            parser::SourceExpr::float(60.0),
            parser::SourceExpr::float(0.5),
        ];
        let (note_on, _note_off) = evaluator
            .apply_note_function(&function, args.clone(), set.programs()[0].sliders())
            .expect("note function should apply");
        // Optimize first, exactly as Effect::PlayNoteOn does — the marks
        // must survive optimization for substitution and live updates.
        let mut note_on = crate::optimizer::optimize(note_on);
        let seeded = substitute_current_slider_values(&mut note_on, set.programs()[0].sliders());
        assert_eq!(seeded, vec![("vol".to_string(), 0.5)]);
        let mut marks = Vec::new();
        slider_mark_values(&note_on, &mut marks);
        assert_eq!(marks, vec![("vol".to_string(), 0.5)]);

        // Move the slider; the next note carries the new value.
        set.program_mut(0)
            .unwrap()
            .set_slider_normalized(0, 1.0)
            .expect("program has a vol slider");
        let (note_on, _note_off) = evaluator
            .apply_note_function(&function, args, set.programs()[0].sliders())
            .expect("note function should apply");
        let mut note_on = crate::optimizer::optimize(note_on);
        let seeded = substitute_current_slider_values(&mut note_on, set.programs()[0].sliders());
        assert_eq!(seeded, vec![("vol".to_string(), 1.0)]);
        let mut marks = Vec::new();
        slider_mark_values(&note_on, &mut marks);
        assert_eq!(marks, vec![("vol".to_string(), 1.0)]);
    }

    #[test]
    fn keys_note_on_slider_marks_survive_optimizer_for_realistic_instrument() {
        // Uses the real std library so the instrument shape (let-bindings,
        // filters, envelopes, seq/fin) matches live usage.
        let (mut set, warning) = ProgramSet::from_source(
            "open std;\n#{sliders=[\"vol:0.5:0:1\"]}\nk = fn(note, vel) => ((harmonica(H, @note) | unseq()) * vol, (harmonica(H, @note) | unseq()) * vol);"
                .to_string(),
            std::path::PathBuf::new(),
        )
        .expect("test source should parse");
        assert_eq!(warning, "");
        let evaluator = Evaluator::new(44100, 90, std::path::PathBuf::from("./lib/v0"));

        let function = match evaluator.evaluate_program(&set, 0) {
            Evaluation::KeysInstrument(function) => function,
            Evaluation::Invalid(message) => panic!("invalid: {}", message),
            Evaluation::Waveform(_) => panic!("classified as waveform"),
        };
        set.program_mut(0)
            .unwrap()
            .set_slider_normalized(0, 1.0)
            .expect("program has a vol slider");

        let args = vec![
            parser::SourceExpr::float(60.0),
            parser::SourceExpr::float(0.5),
        ];
        let (note_on, _note_off) = evaluator
            .apply_note_function(&function, args, set.programs()[0].sliders())
            .expect("note function should apply");
        let mut note_on = crate::optimizer::optimize(note_on);
        let seeded = substitute_current_slider_values(&mut note_on, set.programs()[0].sliders());
        assert_eq!(seeded, vec![("vol".to_string(), 1.0)]);
        let mut marks = Vec::new();
        slider_mark_values(&note_on, &mut marks);
        assert!(
            marks.contains(&("vol".to_string(), 1.0)),
            "expected a surviving vol mark at 1.0, got {:?}",
            marks
        );
    }
}
