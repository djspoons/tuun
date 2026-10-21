//! Checks that trace marks never change what plays: every program in the library
//! and the top-level `test*.tuun` files evaluates, with trace marks made and
//! then stripped at the evaluation boundary, to the same tree as with no trace
//! marks made at all.
#![cfg(feature = "native")]

use std::fs;
use std::path::{Path, PathBuf};

use tuun::environment::Environment;
use tuun::expr::{Binding, Expr, SourceExpr};
use tuun::optimizer;
use tuun::programs::{Evaluated, EvaluationInput, ProgramSet};
use tuun::trace;

const SAMPLE_RATE: u32 = 8000;

/// Returns every `.tuun` file under `dir`, at any depth.
fn tuun_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            tuun_files(&path, found);
        } else if path.extension().is_some_and(|e| e == "tuun") {
            found.push(path);
        }
    }
}

/// Returns a printable form of `evaluated`, playing a keys instrument's note
/// to get waveforms, with trace marks made or not as `trace_marking` says.
fn played(environment: &Environment, evaluated: Evaluated, trace_marking: bool) -> String {
    match evaluated {
        Evaluated::Waveform(w, _) => format!("Waveform:{:?}", w),
        Evaluated::Sequence(sequence) => {
            format!("{:?} {:?}", sequence.steps, sequence.step_waveform)
        }
        Evaluated::KeysInstrument(function) => {
            trace::set_trace_marking(trace_marking);
            let note = environment.apply_note_function(
                &function,
                vec![SourceExpr::float(60.0), SourceExpr::float(0.7)],
            );
            trace::set_trace_marking(true);
            // The marks differ by design; what plays must not.
            format!(
                "{:?}",
                note.map(|(note_on, note_off, _)| (note_on, note_off))
            )
        }
    }
}

/// Returns the program `input` describes as it plays, with trace marks made
/// or not as `trace_marking` says.
fn evaluate(environment: &Environment, input: &EvaluationInput, trace_marking: bool) -> String {
    trace::set_trace_marking(trace_marking);
    let evaluated = environment.evaluate_program(input);
    trace::set_trace_marking(true);
    match evaluated {
        Ok(evaluated) => played(environment, evaluated, trace_marking),
        Err(diagnostics) => format!("error: {:?}", diagnostics),
    }
}

/// Returns the waveform program `input` describes with its trace marks
/// stripped rather than walked, optimized, or `None` if it isn't one.
fn stripped(environment: &Environment, input: &EvaluationInput) -> Option<String> {
    let mut bindings = input.bindings.clone();
    bindings.insert(0, Binding::Open(vec!["__prelude".to_string()]).into());
    let value = environment.evaluate_source(&input.text, &bindings).ok()?;
    let waveform = match value.expr {
        Expr::Waveform(w) => w,
        Expr::Seq { waveform, .. } => match waveform.expr {
            Expr::Waveform(w) => w,
            _ => return None,
        },
        _ => return None,
    };
    Some(format!("{:?}", optimizer::optimize(waveform)))
}

/// Trace marking on, then stripped, plays the same optimized tree as trace
/// marking off, for every waveform, step, and keys program. Reading marks off
/// a waveform program plays the same tree as stripping them.
#[test]
fn trace_marked_programs_play_the_same_tree_as_unmarked() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let library_root = root.join("lib/v0");
    let mut files = Vec::new();
    tuun_files(&root.join("lib"), &mut files);
    for entry in fs::read_dir(&root).expect("read the crate root").flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with("test") && name.ends_with(".tuun") {
            files.push(entry.path());
        }
    }
    files.sort();

    let environment = Environment::new(SAMPLE_RATE, 90, library_root);
    let mut compared = 0;
    let mut mismatches = Vec::new();
    for path in &files {
        let source = fs::read_to_string(path).expect("read a program file");
        let Ok((set, _)) = ProgramSet::from_source(source, path.clone()) else {
            continue;
        };
        for (index, program) in set.programs().iter().enumerate() {
            if program.text().trim().is_empty() {
                continue;
            }
            let input = set.evaluation_input(index);
            let traced = evaluate(&environment, &input, true);
            let untraced = evaluate(&environment, &input, false);
            if traced.starts_with("error") && untraced.starts_with("error") {
                continue;
            }
            compared += 1;
            // `stripped` also evaluates steps' and instruments' programs, which
            // `traced` doesn't print as one waveform.
            let walked_differs = traced.starts_with("Waveform:")
                && stripped(&environment, &input)
                    .is_none_or(|s| format!("Waveform:{}", s) != traced);
            if traced != untraced || walked_differs {
                mismatches.push(format!(
                    "{} {}: {}",
                    path.display(),
                    set.display_name(index),
                    program.text()
                ));
            }
        }
    }
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    // Guards against a library move silently emptying the check.
    assert!(compared >= 30, "compared only {} programs", compared);
}
