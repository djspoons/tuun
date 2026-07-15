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
use crate::diagnostics::{Diagnostic, Source, render_snippet};
use crate::eval;
use crate::expr;
use crate::ids::MarkId;
use crate::parser;
use crate::programs::{Evaluation, ProgramSet, ProgramSliders};
use crate::slider;
use crate::waveform;

/// The `mark(N)` built-in: wraps a waveform in a `MarkId::UserDefined`
/// mark.
fn mark<S: 'static>(arguments: Vec<expr::Expr<MarkId, S>>) -> expr::Expr<MarkId, S> {
    match &arguments[..] {
        [expr::Expr::Float(id)] if *id >= 1.0 && id.fract() == 0.0 => {
            let id = id.round() as u32;
            expr::Expr::BuiltIn {
                name: format!("mark({})", id),
                function: builtins::curry(move |waveform: Box<waveform::Waveform<MarkId>>| {
                    waveform::Waveform::Marked {
                        id: MarkId::UserDefined(id),
                        waveform,
                    }
                }),
            }
        }
        _ => expr::Expr::Error("Invalid argument for mark".to_string()),
    }
}

/// Returns the error to report for a module that failed to parse: the
/// first of `errors`, whose span already locates it in the module.
fn first_module_error(errors: Vec<expr::Error<Source>>) -> expr::Error<Source> {
    errors
        .into_iter()
        .next()
        .unwrap_or_else(|| expr::Error::new("Parse failed".to_string()))
}

/// Returns the module's display path relative to the library root:
/// `foo/bar.tuun` for `["foo", "bar"]`. Used in user-visible messages,
/// which omit the library root.
fn module_display_path(path: &[String]) -> path::PathBuf {
    let mut display_path = path::PathBuf::new();
    for part in path {
        display_path.push(part);
    }
    display_path.set_extension("tuun");
    display_path
}

/// One slot in [`Evaluator`]'s module cache: the file's mtime at the time we
/// parsed it, plus a leaked `&'static` slice of the parsed bindings.
// See the field doc on `Evaluator::modules` for the leak strategy.
struct ModuleCacheEntry {
    mtime: time::SystemTime,
    bindings: &'static [expr::SourceBinding<MarkId, Source>],
}

/// The identity behind a module id: the module's path and the most
/// recently read source text (kept even when that text failed to parse,
/// so parse errors can be located).
struct ModuleInfo {
    path: Vec<String>,
    source: String,
}

pub struct Evaluator {
    /// Built-ins + environment-derived definitions; implicitly opened at
    /// the top of every other loaded module — see [`Evaluator::resolve`].
    prelude: Vec<expr::SourceBinding<MarkId, Source>>,
    /// Filesystem root for module resolution. A module path
    /// `["foo", "bar"]` is looked up as `<library_root>/foo/bar.tuun`.
    library_root: path::PathBuf,
    /// Cache of successfully parsed modules, keyed by module path.
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
    /// Path and latest source for each assigned module id (index = id).
    /// Ids are stable across mtime reloads of the same module.
    module_info: RefCell<Vec<ModuleInfo>>,
}

impl Evaluator {
    /// Builds an evaluator whose prelude defines `tempo`, `sample_rate`,
    /// `mark`, and `debug` alongside the built-ins.
    pub fn new(sample_rate: u32, tempo: u32, library_root: path::PathBuf) -> Evaluator {
        let mut prelude = Vec::new();
        builtins::add_bindings(&mut prelude);

        use expr::SourceExpr;
        fn def<M, S>(id: &str, expr: SourceExpr<M, S>) -> expr::SourceBinding<M, S> {
            use expr::Binding;
            use expr::Pattern;
            Binding::Definition(Pattern::Identifier(id.to_string()), expr).into()
        }
        prelude.push(def("tempo", SourceExpr::float(tempo as f32)));
        prelude.push(def("sample_rate", SourceExpr::float(sample_rate as f32)));

        prelude.push(def(
            "mark",
            SourceExpr::from(expr::Expr::BuiltIn {
                name: "mark".to_string(),
                function: expr::BuiltInFn(std::rc::Rc::new(mark)),
            }),
        ));
        prelude.push(def(
            "debug",
            builtins::debug(|message| println!("{}", message)),
        ));

        Evaluator {
            prelude,
            library_root,
            modules: RefCell::new(HashMap::new()),
            module_info: RefCell::new(Vec::new()),
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
    fn resolve(
        &self,
        path: &[String],
    ) -> Result<&[expr::SourceBinding<MarkId, Source>], expr::Error<Source>> {
        if path.len() == 1 && path[0] == "__prelude" {
            return Ok(&self.prelude);
        }

        let file_path = self.module_file_path(path);
        // User-visible messages name the module relative to the library root.
        let display_path = module_display_path(path);

        // Stat the file once. If the cache has a matching mtime, return the
        // cached slice without re-reading or re-parsing.
        let current_mtime = fs::metadata(&file_path)
            .and_then(|m| m.modified())
            .map_err(|e| {
                expr::Error::new(format!(
                    "Failed to stat module {}: {}",
                    display_path.display(),
                    e
                ))
            })?;
        if let Some(entry) = self.modules.borrow().get(path)
            && entry.mtime == current_mtime
        {
            return Ok(entry.bindings);
        }

        // Cache miss or stale — reload. The module's id (stable across
        // reloads of the same path) is assigned before parsing so both the
        // bindings' spans and any parse error's span carry it; the source
        // text is recorded even on a failed parse so those errors can be
        // located.
        let contents = fs::read_to_string(&file_path).map_err(|e| {
            expr::Error::new(format!(
                "Failed to read module {}: {}",
                display_path.display(),
                e
            ))
        })?;
        let id = self.record_module_info(path, contents);
        let module_info = self.module_info.borrow();
        let contents = &module_info[id as usize].source;

        // A module that parses with recoverable errors is still broken —
        // report it rather than evaluating with error placeholders.
        let (mut bindings, errors) =
            parser::parse_module::<MarkId, _>(contents, Source::Module(id))
                .map_err(first_module_error)?;
        if !errors.is_empty() {
            return Err(first_module_error(errors));
        }

        // Every loaded module implicitly opens the prelude as its first binding
        // so it can reference prelude names without an explicit `open __prelude`
        // line.
        bindings.insert(0, expr::Binding::Open(vec!["__prelude".to_string()]).into());

        // Leak the parsed bindings so we can hand out a stable
        // `&[SourceBinding]` reference. On a stale-cache reload, the previous
        // leak isn't freed — any borrows from before the reload (e.g. earlier
        // in the same evaluate session) remain valid.
        let leaked: &'static [expr::SourceBinding<MarkId, Source>] =
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

    /// Records `source` as the latest text of the module at `path` and
    /// returns the module's id, assigning the next free one on first sight.
    fn record_module_info(&self, path: &[String], source: String) -> u32 {
        let mut module_info = self.module_info.borrow_mut();
        match module_info.iter().position(|info| info.path == path) {
            Some(id) => {
                module_info[id].source = source;
                id as u32
            }
            None => {
                module_info.push(ModuleInfo {
                    path: path.to_vec(),
                    source,
                });
                (module_info.len() - 1) as u32
            }
        }
    }

    /// Returns the file backing the module at `path`:
    /// `<library_root>/<path components>.tuun`.
    fn module_file_path(&self, path: &[String]) -> path::PathBuf {
        self.library_root.join(module_display_path(path))
    }

    /// Resolves `error` into a [`Diagnostic`] for program `index` of `set`,
    /// according to the error's span source: module errors carry their module's
    /// display file and position; program-local errors carry a bare position
    /// relative to the program's own text (matching the editor's display);
    /// source-file errors (sibling bindings) a whole-file position. Each
    /// carries a snippet of the text its position resolves against.
    pub(crate) fn diagnose(
        &self,
        error: &expr::Error<Source>,
        set: &ProgramSet,
        index: usize,
    ) -> Diagnostic {
        let message = error.message().to_string();
        match (error.source(), error.range()) {
            (Some(Source::Program), Some(range)) => {
                Diagnostic::in_program(message, range, set.programs()[index].text())
            }
            (Some(Source::File), Some(range)) => match set.source_position(range.start) {
                Some(position) => Diagnostic {
                    file: None,
                    position: Some(position),
                    program_range: None,
                    snippet: Some(render_snippet(set.source(), &range)),
                    message,
                },
                None => Diagnostic::message_only(message),
            },
            (Some(Source::Module(id)), Some(range)) => {
                // After an mtime reload, positions resolve against the newest
                // source text, which can drift from the leaked bindings an
                // in-flight error came from — the same staleness the binding
                // cache already accepts.
                let module_info = self.module_info.borrow();
                match module_info.get(id as usize) {
                    Some(info) => Diagnostic {
                        file: Some(module_display_path(&info.path)),
                        position: Some(expr::line_col(&info.source, range.start)),
                        program_range: None,
                        snippet: Some(render_snippet(&info.source, &range)),
                        message,
                    },
                    None => Diagnostic::message_only(message),
                }
            }
            _ => Diagnostic::message_only(message),
        }
    }

    /// Parses and evaluates `text` under `bindings`, resolving `open`
    /// directives through the module cache. Returns the evaluated expression
    /// or a user-visible message.
    pub fn evaluate_source(
        &self,
        text: &str,
        bindings: &[expr::SourceBinding<MarkId, Source>],
    ) -> Result<expr::SourceExpr<MarkId, Source>, String> {
        let expr = match parser::parse_program(text, Source::Program) {
            Err(errors) => return Err(errors[0].display_with_source(text)),
            Ok(expr) => expr,
        };

        eval::evaluate(|path| self.resolve(path), bindings, expr).map_err(|error| error.to_string())
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
        let text = set.programs()[index].text();

        let expr = match parser::parse_program(text, Source::Program) {
            Err(errors) => {
                return Evaluation::Invalid(
                    errors
                        .iter()
                        .map(|error| self.diagnose(error, set, index))
                        .collect(),
                );
            }
            Ok(expr) => expr,
        };
        let expr = match eval::evaluate(|path| self.resolve(path), &bindings, expr) {
            Err(error) => {
                return Evaluation::Invalid(vec![self.diagnose(&error, set, index)]);
            }
            Ok(expr) => expr,
        };
        match expr.expr {
            expr::Expr::Waveform(w) => Evaluation::Waveform(w),
            expr::Expr::Seq { waveform, .. } => {
                if let expr::Expr::Waveform(w) = waveform.expr {
                    Evaluation::Waveform(w)
                } else {
                    Evaluation::Invalid(vec![Diagnostic::message_only(NOT_A_PROGRAM.to_string())])
                }
            }
            expr::Expr::Function { .. } | expr::Expr::BuiltIn { .. } => {
                // Sanity check: actually invoke with dummy args.
                // TODO use a waveform for velocity
                match self.apply_note_function(
                    &expr,
                    vec![expr::SourceExpr::float(60.0), expr::SourceExpr::float(0.7)],
                    set.programs()[index].sliders(),
                ) {
                    Ok(_) => Evaluation::KeysInstrument(expr),
                    Err(error) => Evaluation::Invalid(vec![self.diagnose(&error, set, index)]),
                }
            }
            _ => Evaluation::Invalid(vec![Diagnostic::message_only(NOT_A_PROGRAM.to_string())]),
        }
    }

    /// Applies a note function `expr` to the given `args`, expecting a pair
    /// of (note-on, note-off) waveforms as a result.
    ///
    /// The expressions `expr` and `args` should be closed except for
    /// references to `sliders`, which are bound at their current values.
    ///
    /// Errors keep any span the evaluation produced (e.g. an unbound
    /// variable in the function's body), so they can be diagnosed against
    /// the program the function came from.
    pub fn apply_note_function(
        &self,
        expr: &expr::SourceExpr<MarkId, Source>,
        args: Vec<expr::SourceExpr<MarkId, Source>>,
        sliders: &ProgramSliders,
    ) -> Result<(waveform::Waveform<MarkId>, waveform::Waveform<MarkId>), expr::Error<Source>> {
        use expr::Expr::{Tuple, Waveform};
        let expr = expr::SourceExpr::from(expr::Expr::Application {
            function: Box::new(expr.clone()),
            argument: Box::new(expr::SourceExpr::from(Tuple(args))),
        });
        let mut bindings = vec![];
        slider::append_slider_bindings(
            sliders.configs(),
            sliders.normalized_values(),
            MarkId::Slider,
            &mut bindings,
        );
        let resolve = |_: &[String]| {
            Err(expr::Error::new(
                "Didn't expect to resolve in apply_note_function".to_string(),
            ))
        };
        let expr = eval::evaluate(resolve, &bindings, expr)?;
        match expr.expr {
            Tuple(mut exprs) => {
                if exprs.len() != 2 {
                    return Err(expr::Error::new(format!(
                        "Expected 2 waveforms for note, got {} elements",
                        exprs.len()
                    )));
                }
                match (exprs.remove(0).expr, exprs.remove(0).expr) {
                    (Waveform(note_on), Waveform(note_off)) => Ok((note_on, note_off)),
                    (expr, Waveform(_)) => Err(expr::Error::new(format!(
                        "Expected waveform for note-on, got: {}",
                        expr
                    ))),
                    (_, expr) => Err(expr::Error::new(format!(
                        "Expected waveform for note-off, got: {}",
                        expr
                    ))),
                }
            }
            expr => Err(expr::Error::new(format!(
                "Expected 2 waveforms for note, got: {}",
                expr
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer;
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
        let args = vec![expr::SourceExpr::float(60.0), expr::SourceExpr::float(0.5)];
        let (note_on, _note_off) = evaluator
            .apply_note_function(&function, args.clone(), set.programs()[0].sliders())
            .expect("note function should apply");
        // Optimize first, exactly as Effect::PlayNoteOn does — the marks
        // must survive optimization for substitution and live updates.
        let mut note_on = optimizer::optimize(note_on);
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
        let mut note_on = optimizer::optimize(note_on);
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
            Evaluation::Invalid(diagnostics) => panic!("invalid: {:?}", diagnostics),
            Evaluation::Waveform(_) => panic!("classified as waveform"),
        };
        set.program_mut(0)
            .unwrap()
            .set_slider_normalized(0, 1.0)
            .expect("program has a vol slider");

        let args = vec![expr::SourceExpr::float(60.0), expr::SourceExpr::float(0.5)];
        let (note_on, _note_off) = evaluator
            .apply_note_function(&function, args, set.programs()[0].sliders())
            .expect("note function should apply");
        let mut note_on = optimizer::optimize(note_on);
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

    #[test]
    fn diagnose_locates_module_and_program_errors() {
        let (set, warning) = ProgramSet::from_source(
            "open std;\n#{level_db=0}\nbad = nope;\n".to_string(),
            PathBuf::from("song.tuun"),
        )
        .expect("test source should parse");
        assert_eq!(warning, "");
        let evaluator = Evaluator::new(44100, 90, PathBuf::from("./lib/v0"));

        // A program-local error shows a bare position relative to the
        // program's own text (matching the editor's display), no file.
        let error = expr::Error::with_span(
            "boom".to_string(),
            Some(expr::Span::new(Source::Program, 0..4)),
        );
        let diagnostic = evaluator.diagnose(&error, &set, 0);
        assert_eq!(diagnostic.to_string(), "1:1: boom");
        assert_eq!(diagnostic.program_range, Some(0..4));
        assert!(diagnostic.file.is_none());

        // A source-file error (e.g. from a sibling binding) locates into the
        // whole file: offset 24 is `bad` on line 3.
        let error = expr::Error::with_span(
            "boom".to_string(),
            Some(expr::Span {
                source: Source::File,
                range: 24..27,
            }),
        );
        let diagnostic = evaluator.diagnose(&error, &set, 0);
        assert_eq!(diagnostic.to_string(), "3:1: boom");
        assert!(diagnostic.program_range.is_none());

        // A module error names the module file relative to the library root
        // and locates into its cached source. `std` is the first module
        // resolved, so it holds id 0.
        evaluator
            .resolve(&["std".to_string()])
            .expect("std resolves");
        let error = expr::Error::with_span(
            "boom".to_string(),
            Some(expr::Span {
                source: Source::Module(0),
                range: 0..1,
            }),
        );
        let diagnostic = evaluator.diagnose(&error, &set, 0);
        assert_eq!(diagnostic.to_string(), "std.tuun:1:1: boom");
        assert!(diagnostic.program_range.is_none());

        // A module id that was never assigned degrades to the bare message.
        let error = expr::Error::with_span(
            "boom".to_string(),
            Some(expr::Span {
                source: Source::Module(99),
                range: 0..1,
            }),
        );
        let diagnostic = evaluator.diagnose(&error, &set, 0);
        assert!(diagnostic.file.is_none());
        assert!(diagnostic.position.is_none());
        assert_eq!(diagnostic.message, "boom");
    }

    #[test]
    fn module_parse_errors_are_located_in_the_module() {
        // A module that fails to parse resolves to an error whose span
        // carries the module's id, so its diagnostic names the module file
        // and a position within it — same as module evaluation errors.
        let dir = std::env::temp_dir().join("tuun_test_bad_module");
        std::fs::create_dir_all(&dir).expect("temp module dir");
        std::fs::write(dir.join("bad_mod.tuun"), "broken(;\n").expect("write bad module");
        let (set, _) = ProgramSet::from_source(
            "#{level_db=0}\nbad = 1;\n".to_string(),
            PathBuf::from("song.tuun"),
        )
        .expect("test source should parse");
        let evaluator = Evaluator::new(44100, 90, dir);

        let error = evaluator
            .resolve(&["bad_mod".to_string()])
            .expect_err("bad module should not resolve");
        let diagnostic = evaluator.diagnose(&error, &set, 0);
        assert_eq!(diagnostic.file, Some(PathBuf::from("bad_mod.tuun")));
        assert!(
            diagnostic.position.is_some(),
            "expected a position in the module, got {:?}",
            diagnostic
        );
        assert!(diagnostic.program_range.is_none());
    }

    #[test]
    fn evaluate_program_reports_position_for_unbound_variable() {
        let (set, _) = ProgramSet::from_source(
            "#{level_db=0}\nbad = (1, undefined_name);\n".to_string(),
            PathBuf::from("song.tuun"),
        )
        .expect("test source should parse");
        let evaluator = Evaluator::new(44100, 90, PathBuf::new());
        let Evaluation::Invalid(diagnostics) = evaluator.evaluate_program(&set, 0) else {
            panic!("expected an invalid evaluation");
        };
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:5: Variable 'undefined_name' not found in context"
        );
        // The range is available for editor highlighting: `undefined_name`
        // spans bytes 4..18 of the program text `(1, undefined_name)`.
        assert_eq!(diagnostics[0].program_range, Some(4..18));
    }
}
