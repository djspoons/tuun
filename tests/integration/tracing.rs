//! Checks that trace marks never change what plays: every program in the library
//! evaluates, with trace marks made and then stripped at the evaluation
//! boundary, to the same tree as with no trace marks made at all, and plays a
//! tree whose only marks are substitution marks.
//!
//! Both tests evaluate every program in the library, so they are slow and
//! ignored by default. Run them with `--ignored`:
//!
//! ```text
//! cargo test --test integration tracing:: -- --ignored
//! ```
#![cfg(feature = "native")]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use tuun::environment::Environment;
use tuun::expr::{Binding, Expr, SourceExpr};
use tuun::ids::MarkId;
use tuun::optimizer;
use tuun::player::{Player, Start};
use tuun::processing;
use tuun::programs::{Evaluated, EvaluationInput, ProgramSet};
use tuun::trace;
use tuun::tracker::{self, Status};
use tuun::waveform::Waveform;

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

/// Returns every `.tuun` file in the library except the untracked `private`
/// modules, sorted.
fn program_files(root: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    tuun_files(&root.join("lib"), &mut files);
    let private = root.join("lib/v0/private");
    files.retain(|path| !path.starts_with(&private));
    files.sort();
    files
}

/// Appends the id of every mark in `waveform` to `found`.
fn mark_ids(waveform: &Waveform<MarkId>, found: &mut Vec<MarkId>) {
    use Waveform::*;
    match waveform {
        Const(_) | Time(_) | Noise | Fixed(_, _) => {}
        Fin {
            length: a,
            waveform: b,
            ..
        }
        | Append(a, b, _)
        | Phase {
            frequency: a,
            offset: b,
            ..
        }
        | BinaryOp(_, a, b)
        | Reset {
            trigger: a,
            waveform: b,
            ..
        } => {
            mark_ids(a, found);
            mark_ids(b, found);
        }
        UnaryOp(_, a) | Captured { waveform: a, .. } => mark_ids(a, found),
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => {
            mark_ids(trigger, found);
            mark_ids(positive_waveform, found);
            mark_ids(negative_waveform, found);
        }
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => {
            mark_ids(waveform, found);
            for coefficient in feed_forward.iter().chain(feedback) {
                mark_ids(coefficient, found);
            }
        }
        Marked { id, waveform } => {
            found.push(id.clone());
            mark_ids(waveform, found);
        }
    }
}

/// Returns whether `id` marks a place that is substituted while a waveform
/// plays.
fn substitutes(id: &MarkId) -> bool {
    matches!(
        id,
        MarkId::Terminator | MarkId::Amplitude | MarkId::MetronomeAmplitude | MarkId::Slider { .. }
    )
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
#[ignore]
fn trace_marked_programs_play_the_same_tree_as_unmarked() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let library_root = root.join("lib/v0");
    let files = program_files(&root);

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
    assert!(compared >= 50, "compared only {} programs", compared);
}

/// Every tree played, whether a program's waveform, a sequence's steps and
/// anchor, a keys note or the Beats waveform, holds no marks but the places
/// substituted while it plays.
#[test]
#[ignore]
fn played_trees_carry_only_substitution_marks() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let environment = Environment::new(SAMPLE_RATE, 90, root.join("lib/v0"));
    let (sender, receiver) = mpsc::channel();
    let player = Player::new(90, 4, sender);
    let status = Status {
        buffer_start: Instant::now(),
        start_times: Vec::new(),
        tracker_load: None,
        allocations: None,
    };
    let mut trees = vec![("Beats".to_string(), player.beats_waveform(&environment))];
    for path in &program_files(&root) {
        let source = fs::read_to_string(path).expect("read a program file");
        let Ok((mut set, _)) = ProgramSet::from_source(source, path.clone()) else {
            continue;
        };
        for index in 0..set.programs().len() {
            let input = set.evaluation_input(index);
            let outcome = processing::process(&environment, &input, false);
            if set.record(index, input.text.into(), outcome).is_err() {
                continue;
            }
            let name = format!("{} {}", path.display(), set.display_name(index));
            if let Some(Evaluated::KeysInstrument(function)) = set.programs()[index].evaluated() {
                let note = environment.apply_note_function(
                    function,
                    vec![SourceExpr::float(60.0), SourceExpr::float(0.7)],
                );
                if let Ok((note_on, note_off, _)) = note {
                    trees.push((format!("{} note-on", name), note_on));
                    trees.push((format!("{} note-off", name), note_off));
                }
            } else if set.programs()[index].is_playable() {
                player.play_program(&set, index, &status, Start::Now, None, &mut HashMap::new());
                for command in receiver.try_iter() {
                    if let tracker::Command::Play { id, waveform, .. } = command {
                        trees.push((format!("{} {:?}", name, id), waveform));
                    }
                }
            }
        }
    }

    let mut unexpected = Vec::new();
    for (name, tree) in &trees {
        let mut found = Vec::new();
        mark_ids(tree, &mut found);
        found.retain(|id| !substitutes(id));
        if !found.is_empty() {
            unexpected.push(format!("{}: {:?}", name, found));
        }
    }
    assert!(unexpected.is_empty(), "{}", unexpected.join("\n"));
    // Guards against a library move silently emptying the check.
    assert!(trees.len() >= 50, "checked only {} trees", trees.len());
}
