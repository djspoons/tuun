//! Program processing — checking and evaluating programs, and loading a
//! source file's whole program set — expressed over copies of what it reads,
//! so it can run away from the [`ProgramSet`] it was copied from.

use std::path;

use crate::diagnostics::Severity;
use crate::environment::Environment;
use crate::programs::{EvaluationInput, Outcome, ProgramSet};

/// A program set built from a source file, with every program evaluated.
pub struct Loaded {
    /// The new set, each program's result recorded on it.
    pub set: ProgramSet,
    /// The warning for recoverable parse errors (empty when the parse was
    /// clean).
    pub warning: String,
    /// The indexes of the programs whose evaluation failed, in slot order.
    pub failed: Vec<usize>,
}

/// Checks and evaluates the program `input` describes.
///
/// An empty (or blank) program succeeds with no result: it is a deletion,
/// not a parse error. Otherwise the type checker runs first. Error-severity
/// findings fail without evaluating; warning-severity findings are the
/// warnings on success and lead the evaluation's diagnostics on failure, so
/// both views of a problem show side by side.
pub fn process(environment: &Environment, input: &EvaluationInput) -> Outcome {
    if input.text.trim().is_empty() {
        return Ok((None, Vec::new()));
    }
    let findings = environment.check_program(input);
    if findings
        .iter()
        .any(|finding| finding.severity == Severity::Error)
    {
        return Err(findings);
    }
    match environment.evaluate_program(input) {
        Ok(evaluated) => Ok((Some(evaluated), findings)),
        Err(diagnostics) => {
            let mut all = findings;
            all.extend(diagnostics);
            Err(all)
        }
    }
}

/// Builds the program set for `source`, the contents of the file at
/// `input_path`, and processes each of its programs.
///
/// Returns the error message when `source` doesn't parse.
pub fn load(
    environment: &Environment,
    source: String,
    input_path: path::PathBuf,
) -> Result<Loaded, String> {
    let (mut set, warning) =
        ProgramSet::from_source(source, input_path).map_err(|errors| match errors.first() {
            Some(error) => format!("Parse error: {}", error),
            None => "Parse error".to_string(),
        })?;
    let mut failed = Vec::new();
    for index in 0..set.programs().len() {
        if set.programs()[index].is_empty() {
            continue;
        }
        let outcome = process(environment, &set.evaluation_input(index));
        if set.record(index, outcome).is_err() {
            failed.push(index);
        }
    }
    Ok(Loaded {
        set,
        warning,
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::CheckMode;
    use crate::programs::Evaluated;
    use std::fs;
    use std::path::PathBuf;

    /// The values that cross to and from a processing thread.
    #[test]
    fn processing_values_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<EvaluationInput>();
        assert_send::<Outcome>();
        assert_send::<Loaded>();
    }

    /// Returns the set for `source`, which must parse cleanly.
    fn set_from(source: &str) -> ProgramSet {
        let (set, warning) =
            ProgramSet::from_source(source.to_string(), PathBuf::from("song.tuun"))
                .expect("test source should parse");
        assert_eq!(warning, "");
        set
    }

    #[test]
    fn processing_an_empty_program_succeeds_with_no_result() {
        let mut set = set_from("#{level_db=0}\nkick = 1 | fin(time - 1);\n");
        let environment = Environment::new(8000, 90, PathBuf::new());
        let outcome = process(&environment, &set.evaluation_input(0));
        assert!(set.record(0, outcome).is_ok());
        assert!(set.programs()[0].evaluated().is_some());

        // Clearing the text must not surface a parse error — an empty
        // program is a deletion in progress — and the old result goes.
        set.program_mut(0).unwrap().set_text("  ".to_string());
        let outcome = process(&environment, &set.evaluation_input(0));
        assert!(matches!(outcome, Ok((None, ref warnings)) if warnings.is_empty()));
        assert!(set.record(0, outcome).is_ok());
        assert!(set.programs()[0].evaluated().is_none());
    }

    #[test]
    fn findings_gate_or_warn() {
        let mut set = set_from("#{level_db=0}\nw = time | fin(\"a\");\n");
        let mut environment = Environment::new(44100, 90, PathBuf::new());

        // Default mode: findings are errors, positioned in the program's
        // own text (`"a"` starts at column 12), and they gate — evaluation
        // is never attempted, so the checker's finding stands alone.
        let diagnostics = process(&environment, &set.evaluation_input(0))
            .expect_err("error findings gate evaluation");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Severity::Error);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:12: expected waveform, found string"
        );
        assert_eq!(diagnostics[0].program_range, Some(11..14));

        // Warnings mode: the same finding reports as a warning, first, and
        // evaluation proceeds to its own error — side by side.
        environment.set_check_mode(CheckMode::Warnings);
        let diagnostics = process(&environment, &set.evaluation_input(0))
            .expect_err("fin(\"a\") fails at evaluation");
        assert_eq!(diagnostics.len(), 2);
        assert_eq!(diagnostics[0].severity, Severity::Warning);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:12: expected waveform, found string"
        );
        assert_eq!(diagnostics[1].severity, Severity::Error);

        // Clean programs evaluate with no findings in either mode,
        // including computed indexes.
        environment.set_check_mode(CheckMode::Errors);
        for text in ["time", "fin(nth(1 + 2, [1, 2, 4, 8]))(time)"] {
            set.program_mut(0).unwrap().set_text(text.to_string());
            match process(&environment, &set.evaluation_input(0)) {
                Ok((Some(Evaluated::Waveform { .. }), warnings)) => assert!(warnings.is_empty()),
                outcome => panic!("{} should evaluate cleanly, got {:?}", text, outcome),
            }
        }

        // A fractional index gates before evaluation can fail; column 9
        // points at `2.5` — the mismatched argument itself.
        set.program_mut(0)
            .unwrap()
            .set_text("fin(nth(2.5, [1, 2, 4, 8]))(time)".to_string());
        let diagnostics = process(&environment, &set.evaluation_input(0))
            .expect_err("the fractional index gates");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Severity::Error);
        assert_eq!(diagnostics[0].to_string(), "1:9: expected int, found float");

        // The gate is conservative where evaluation is lazy: a mistake
        // inside a never-called function still gates (evaluation would
        // have succeeded); warnings mode lets it through, reported.
        set.program_mut(0)
            .unwrap()
            .set_text("let f = fn(x) => nth(2.5, [1, 2]) in time".to_string());
        let diagnostics = process(&environment, &set.evaluation_input(0))
            .expect_err("the unused function's finding gates");
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:22: expected int, found float"
        );
        environment.set_check_mode(CheckMode::Warnings);
        let Ok((Some(_), warnings)) = process(&environment, &set.evaluation_input(0)) else {
            panic!("the unused function body never evaluates");
        };
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, Severity::Warning);
        assert_eq!(warnings[0].to_string(), "1:22: expected int, found float");
    }

    /// A program that uses a broken module definition gets its own
    /// diagnostic at the use site and gates — evaluation is never
    /// attempted, so the module's mistake cannot surface as a runtime
    /// error.
    #[test]
    fn uses_of_erroneous_module_definitions_gate() {
        let root =
            std::env::temp_dir().join(format!("tuun-module-gate-test-{}", std::process::id()));
        fs::create_dir_all(&root).expect("create temp library root");
        // The mistake hides inside a function body (like an instrument
        // definition), so the module itself still evaluates — only calls
        // would fail.
        fs::write(
            root.join("m.tuun"),
            "broken = fn(dur) => seq(0)(1) + seq(0)(dur);\nv = 440;\n",
        )
        .expect("write module");
        // The healthy program comes first: program bindings are in scope
        // for the programs after them, so a later broken sibling would
        // (correctly) report on this one's check as a file finding.
        let set = set_from(
            "open m;\n#{level_db=0}\nu = time | fin(v);\n#{level_db=0}\nw = time | fin(broken);\n",
        );
        let environment = Environment::new(44100, 90, root.clone());

        // A program using only the module's healthy exports is untouched.
        let Ok((Some(_), warnings)) = process(&environment, &set.evaluation_input(0)) else {
            panic!("healthy exports evaluate");
        };
        assert!(warnings.is_empty(), "got {:?}", warnings);

        // The use site carries the diagnostic and the gate fires.
        let diagnostics = process(&environment, &set.evaluation_input(1))
            .expect_err("a use of a broken definition gates evaluation");
        assert_eq!(diagnostics.len(), 1, "got {:?}", diagnostics);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:12: 'broken' has a type error"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn load_processes_every_program_and_lists_failures() {
        let environment = Environment::new(8000, 90, PathBuf::new());
        // The broken program comes last: a broken binding also fails the
        // checks of the programs after it.
        let loaded = load(
            &environment,
            "#{level_db=0}\ngood = 1 | fin(time - 1);\n\
             #{keys}\nk = fn(note, vel) => (1, 1);\n\
             #{skip_slots=1}\nbad = time | fin(\"a\");\n"
                .to_string(),
            PathBuf::from("song.tuun"),
        )
        .expect("the source parses");
        assert_eq!(loaded.warning, "");
        // `bad` sits in slot 3, after the skipped slot.
        assert_eq!(loaded.failed, vec![3]);
        let programs = loaded.set.programs();
        assert!(programs[0].waveform().is_some());
        assert!(programs[1].keys_instrument().is_some());
        assert!(programs[3].evaluated().is_none());
        assert_eq!(loaded.set.input_path(), path::Path::new("song.tuun"));
    }

    #[test]
    fn load_surfaces_parse_warnings_and_errors() {
        let environment = Environment::new(8000, 90, PathBuf::new());

        // A garbled binding is recoverable, exactly as at startup: the load
        // succeeds with a parse warning and the binding skipped.
        let loaded = load(
            &environment,
            "#{level_db=0}\ntone = ((;\n".to_string(),
            PathBuf::new(),
        )
        .expect("recoverable garble loads");
        assert!(
            loaded.warning.contains("Parse error"),
            "got: {}",
            loaded.warning
        );

        // A source that doesn't parse at all is an error.
        let Err(error) = load(&environment, "#{".to_string(), PathBuf::new()) else {
            panic!("an unparseable source should fail");
        };
        assert!(error.starts_with("Parse error"), "got: {}", error);
    }
}
