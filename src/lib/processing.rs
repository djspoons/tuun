//! Program processing — checking, evaluating and precomputing programs, and
//! loading a source file's whole program set — expressed over copies of what it
//! reads, so it can run away from the [`ProgramSet`] it was copied from, on the
//! thread [`spawn`] starts.

use std::path;
use std::sync::mpsc;
use std::thread;

use crate::diagnostics::{Diagnostic, Severity};
use crate::environment::{CheckMode, Environment, ModuleTable};
use crate::generator::Generator;
use crate::programs::{Evaluated, EvaluationInput, Outcome, Plays, ProgramSet};
use crate::sequencer;

/// Work for the processing thread.
pub enum Job {
    /// Check and evaluate one program (see [`process`]).
    Program(EvaluationInput),
    /// Build the program set for `source`, the contents of the file at
    /// `input_path`, and process each of its programs (see [`load`]).
    Load {
        source: String,
        input_path: path::PathBuf,
    },
}

/// The processing thread's answer to one [`Job`].
pub enum Reply {
    /// The outcome of a [`Job::Program`].
    Program(Outcome),
    /// The outcome of a [`Job::Load`].
    Loaded(Result<Loaded, String>),
}

/// Starts the processing thread and returns the sender its jobs go to and the
/// receiver its replies come from.
///
/// The thread evaluates in its own environment, built from the arguments as
/// [`Environment::with_module_table`] builds one, with `check_mode` set, and
/// pre-computes what it evaluates (see [`process`]). It handles jobs one at a
/// time in the order they are sent, sends one reply per job in the same order,
/// and calls `wake` after each reply. It exits once the job sender or the reply
/// receiver is dropped.
pub fn spawn(
    sample_rate: u32,
    tempo: u32,
    library_root: path::PathBuf,
    check_mode: CheckMode,
    module_table: ModuleTable,
    wake: impl Fn() + Send + 'static,
) -> (mpsc::Sender<Job>, mpsc::Receiver<Reply>) {
    let (job_sender, jobs) = mpsc::channel();
    let (reply_sender, replies) = mpsc::channel();
    thread::Builder::new()
        .name("processing".to_string())
        .spawn(move || {
            let mut environment =
                Environment::with_module_table(sample_rate, tempo, library_root, module_table);
            environment.set_check_mode(check_mode);
            for job in jobs {
                let reply = match job {
                    Job::Program(input) => Reply::Program(process(&environment, &input)),
                    Job::Load { source, input_path } => {
                        Reply::Loaded(load(&environment, source, input_path))
                    }
                };
                if reply_sender.send(reply).is_err() {
                    return;
                }
                wake();
            }
        })
        .expect("failed to spawn the processing thread");
    (job_sender, replies)
}

/// A program set built from a source file, with every program evaluated.
pub struct Loaded {
    /// The new set, each program's result recorded on it.
    pub set: ProgramSet,
    /// The warning for recoverable parse errors (empty when the parse was
    /// clean).
    pub warning: String,
    /// The indexes of the programs whose evaluation failed, with their
    /// diagnostics, in slot order.
    pub failed: Vec<(usize, Vec<Diagnostic>)>,
}

/// Checks and evaluates the program `input` describes.
///
/// An empty (or blank) program is [`Outcome::Empty`]: it is a deletion, not a
/// parse error. Otherwise the type checker runs first. Error-severity findings
/// fail without evaluating; warning-severity findings are the warnings on
/// success and lead the evaluation's diagnostics on failure, so both views of a
/// problem show side by side.
pub fn process(environment: &Environment, input: &EvaluationInput) -> Outcome {
    process_with(environment, input, || environment.evaluate_program(input))
}

/// Checks the program `input` describes and evaluates it with `evaluate`,
/// as [`process`] does.
fn process_with(
    environment: &Environment,
    input: &EvaluationInput,
    evaluate: impl FnOnce() -> Result<Evaluated, Vec<Diagnostic>>,
) -> Outcome {
    if input.text.trim().is_empty() {
        return Outcome::Empty;
    }
    let findings = environment.check_program(input);
    if findings
        .iter()
        .any(|finding| finding.severity == Severity::Error)
    {
        return Outcome::Failed(findings);
    }
    match evaluate() {
        Ok(evaluated) => Outcome::Evaluated(
            Evaluated {
                plays: precomputed(evaluated.plays, environment.sample_rate()),
                ..evaluated
            },
            findings,
        ),
        Err(diagnostics) => {
            let mut all = findings;
            all.extend(diagnostics);
            Outcome::Failed(all)
        }
    }
}

/// Returns `plays` with the finite parts of the waveform it plays replaced by
/// their samples generated at `sample_rate`.
fn precomputed(plays: Plays, sample_rate: u32) -> Plays {
    let mut generator = Generator::new(sample_rate);
    match plays {
        Plays::Waveform(waveform, marks) => Plays::Waveform(generator.precompute(waveform), marks),
        Plays::Sequence(sequence) => Plays::Sequence(sequencer::Sequence {
            step_waveform: generator.precompute(sequence.step_waveform),
            ..sequence
        }),
        Plays::KeysInstrument(function) => Plays::KeysInstrument(function),
    }
}

/// Builds the program set for `source`, the contents of the file at
/// `input_path`, and processes each of its programs (see [`process`]).
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
    // Each program reads a prefix of the same file bindings, so evaluate
    // them once, in source order, and each program in the context of its
    // prefix rather than evaluating the prefix again for every program.
    let (used, counts) = set.used_evaluation_bindings();
    let mut order: Vec<usize> = (0..set.programs().len())
        .filter(|&index| !set.programs()[index].is_empty())
        .collect();
    order.sort_by_key(|&index| counts[index]);
    let mut context = environment.base_context();
    let mut evaluated = 0;
    let mut failed = Vec::new();
    for index in order {
        if let Ok(entries) = &mut context
            && let Err(error) = environment.extend_context(entries, &used[evaluated..counts[index]])
        {
            context = Err(error);
        }
        evaluated = counts[index];
        let input = set.evaluation_input(index);
        let outcome = process_with(environment, &input, || match &mut context {
            Ok(entries) => environment.evaluate_program_in_context(&input, entries),
            // A binding this program uses failed: evaluating from scratch reports
            // it as it would for this program alone.
            Err(_) => environment.evaluate_program(&input),
        });
        // TODO the warnings of a program that evaluates are dropped, including
        // slider warnings (a clamped saved value, an unreached listing), so
        // they only show once the program is evaluated again.
        if let Err(diagnostics) = set.record(index, input.text.into(), outcome) {
            failed.push((index, diagnostics));
        }
    }
    failed.sort_by_key(|(index, _)| *index);
    Ok(Loaded {
        set,
        warning,
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// The values that cross to and from a processing thread.
    #[test]
    fn processing_values_are_send() {
        fn assert_send<T: Send>() {}
        assert_send::<EvaluationInput>();
        assert_send::<Outcome>();
        assert_send::<Loaded>();
        assert_send::<Job>();
        assert_send::<Reply>();
    }

    /// The thread evaluates what it is sent, in order, and wakes the receiver
    /// after each reply.
    #[test]
    fn the_thread_replies_to_each_job_in_order() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::Duration;

        let mut set = set_from("#{level_db=0}\ngood = 1 | fin(time - 1);\n");
        let wakes = Arc::new(AtomicUsize::new(0));
        let counted = wakes.clone();
        let (jobs, replies) = spawn(
            8000,
            90,
            PathBuf::new(),
            CheckMode::Errors,
            ModuleTable::default(),
            move || {
                counted.fetch_add(1, Ordering::SeqCst);
            },
        );

        jobs.send(Job::Program(set.evaluation_input(0))).unwrap();
        set.program_mut(0)
            .unwrap()
            .set_text("time | fin(\"a\")".to_string());
        jobs.send(Job::Program(set.evaluation_input(0))).unwrap();
        jobs.send(Job::Load {
            source: "#{level_db=0}\nother = 1 | fin(time - 1);\n".to_string(),
            input_path: PathBuf::from("song.tuun"),
        })
        .unwrap();

        let timeout = Duration::from_secs(30);
        let Ok(Reply::Program(first)) = replies.recv_timeout(timeout) else {
            panic!("expected the first program's reply");
        };
        assert!(matches!(
            first,
            Outcome::Evaluated(
                Evaluated {
                    plays: Plays::Waveform(..),
                    ..
                },
                _
            )
        ));
        let Ok(Reply::Program(second)) = replies.recv_timeout(timeout) else {
            panic!("expected the second program's reply");
        };
        assert!(matches!(second, Outcome::Failed(_)), "got {:?}", second);
        let Ok(Reply::Loaded(Ok(loaded))) = replies.recv_timeout(timeout) else {
            panic!("expected the load's reply");
        };
        assert!(loaded.set.programs()[0].waveform().is_some());
        // The wake follows the send, so it may land just after the reply.
        let deadline = std::time::Instant::now() + timeout;
        while wakes.load(Ordering::SeqCst) < 3 && std::time::Instant::now() < deadline {
            thread::yield_now();
        }
        assert_eq!(wakes.load(Ordering::SeqCst), 3);
    }

    /// Returns the diagnostics of `outcome`, which must have failed for the
    /// reason `why`.
    fn failed(outcome: Outcome, why: &str) -> Vec<Diagnostic> {
        match outcome {
            Outcome::Failed(diagnostics) => diagnostics,
            other => panic!("{why}: got {other:?}"),
        }
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
        let input = set.evaluation_input(0);
        let outcome = process(&environment, &input);
        assert!(set.record(0, input.text.into(), outcome).is_ok());
        assert!(set.programs()[0].evaluated().is_some());

        // Clearing the text must not surface a parse error — an empty
        // program is a deletion in progress — and the old result goes.
        set.program_mut(0).unwrap().set_text("  ".to_string());
        let input = set.evaluation_input(0);
        let outcome = process(&environment, &input);
        assert!(matches!(outcome, Outcome::Empty));
        assert!(set.record(0, input.text.into(), outcome).is_ok());
        assert!(set.programs()[0].evaluated().is_none());
    }

    #[test]
    fn findings_gate_or_warn() {
        let mut set = set_from("#{level_db=0}\nw = time | fin(\"a\");\n");
        let mut environment = Environment::new(44100, 90, PathBuf::new());

        // Default mode: findings are errors, positioned in the program's
        // own text (`"a"` starts at column 12), and they gate — evaluation
        // is never attempted, so the checker's finding stands alone.
        let diagnostics = failed(
            process(&environment, &set.evaluation_input(0)),
            "error findings gate evaluation",
        );
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
        let diagnostics = failed(
            process(&environment, &set.evaluation_input(0)),
            "fin(\"a\") fails at evaluation",
        );
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
                Outcome::Evaluated(
                    Evaluated {
                        plays: Plays::Waveform(..),
                        ..
                    },
                    warnings,
                ) => assert!(warnings.is_empty()),
                outcome => panic!("{} should evaluate cleanly, got {:?}", text, outcome),
            }
        }

        // A fractional index gates before evaluation can fail; column 9
        // points at `2.5` — the mismatched argument itself.
        set.program_mut(0)
            .unwrap()
            .set_text("fin(nth(2.5, [1, 2, 4, 8]))(time)".to_string());
        let diagnostics = failed(
            process(&environment, &set.evaluation_input(0)),
            "the fractional index gates",
        );
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].severity, Severity::Error);
        assert_eq!(diagnostics[0].to_string(), "1:9: expected int, found float");

        // The gate is conservative where evaluation is lazy: a mistake
        // inside a never-called function still gates (evaluation would
        // have succeeded); warnings mode lets it through, reported.
        set.program_mut(0)
            .unwrap()
            .set_text("let f = fn(x) => nth(2.5, [1, 2]) in time".to_string());
        let diagnostics = failed(
            process(&environment, &set.evaluation_input(0)),
            "the unused function's finding gates",
        );
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(
            diagnostics[0].to_string(),
            "1:22: expected int, found float"
        );
        environment.set_check_mode(CheckMode::Warnings);
        let Outcome::Evaluated(_, warnings) = process(&environment, &set.evaluation_input(0))
        else {
            panic!("the unused function body never evaluates");
        };
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, Severity::Warning);
        assert_eq!(warnings[0].to_string(), "1:22: expected int, found float");
    }

    /// Precompute replaces the finite waveform a program plays with its
    /// samples.
    #[test]
    fn precompute_fixes_the_finite_waveform_a_program_plays() {
        use crate::waveform::Waveform;

        let environment = Environment::new(8000, 90, PathBuf::new());
        let evaluated = |source: &str| {
            let set = set_from(source);
            match process(&environment, &set.evaluation_input(0)) {
                Outcome::Evaluated(evaluated, _) => evaluated.plays,
                outcome => panic!("{} should evaluate, got {:?}", source, outcome),
            }
        };

        let source = "#{level_db=0}\n_ = 1 | fin(time - 1);\n";
        let Plays::Waveform(Waveform::Fixed(samples, ()), _) = evaluated(source) else {
            panic!("expected samples for {}", source);
        };
        assert_eq!(samples, vec![1.0; 8000]);

        let source = "on_beats = fn(w) => fn(bs) => w;\n\
                      #{level_db=0}\n\
                      _ = [1, 2.5] | on_beats(1 | fin(time - 1));\n";
        let Plays::Sequence(sequence) = evaluated(source) else {
            panic!("expected a sequence for {}", source);
        };
        assert_eq!(sequence.step_beats().collect::<Vec<_>>(), vec![1.0, 2.5]);
        assert!(
            matches!(sequence.step_waveform, Waveform::Fixed(..)),
            "got {}",
            sequence.step_waveform
        );
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
        let Outcome::Evaluated(_, warnings) = process(&environment, &set.evaluation_input(0))
        else {
            panic!("healthy exports evaluate");
        };
        assert!(warnings.is_empty(), "got {:?}", warnings);

        // The use site carries the diagnostic and the gate fires.
        let diagnostics = failed(
            process(&environment, &set.evaluation_input(1)),
            "a use of a broken definition gates evaluation",
        );
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
        let failed: Vec<usize> = loaded.failed.iter().map(|(index, _)| *index).collect();
        assert_eq!(failed, vec![3]);
        assert!(!loaded.failed[0].1.is_empty());
        let programs = loaded.set.programs();
        assert!(programs[0].waveform().is_some());
        assert!(programs[1].keys_instrument().is_some());
        assert!(programs[3].evaluated().is_none());
        assert_eq!(loaded.set.input_path(), path::Path::new("song.tuun"));
    }

    #[test]
    fn load_evaluates_each_program_as_it_would_be_alone() {
        let environment = Environment::new(8000, 90, PathBuf::new());
        // Used bindings with sliders, anonymous and named programs, a
        // sequenceable program, a keys instrument, and a broken binding that
        // only the programs after it read.
        let source = "\
#{sliders=[\"gain\"]}
base = time * slider(\"gain\", 0.5) | fin(time - 1);
on_beats = fn(w) => fn(beats) => w;
#{level_db=0}
_ = base;
#{level_db=0, sliders=[\"depth\"]}
twice = base * slider(\"depth\", 0.25);
#{level_db=0}
_ = [1, 3] | on_beats(twice);
#{keys}
k = fn(note, vel) => (twice * vel, base);
broken = time | fin(\"a\");
#{level_db=0}
_ = twice;
";
        let loaded =
            load(&environment, source.to_string(), PathBuf::new()).expect("the source parses");
        let (set, _) = ProgramSet::from_source(source.to_string(), PathBuf::new()).unwrap();
        let mut live = 0;
        for (index, program) in loaded.set.programs().iter().enumerate() {
            if program.is_empty() {
                continue;
            }
            live += 1;
            let alone = process(&environment, &set.evaluation_input(index));
            let failure = loaded.failed.iter().find(|(i, _)| *i == index);
            match (alone, failure) {
                (Outcome::Evaluated(evaluated, _), None) => assert_eq!(
                    format!("{:?}", program.evaluated()),
                    format!("{:?}", Some(evaluated)),
                    "program {}",
                    index
                ),
                (Outcome::Failed(alone), Some((_, loaded))) => {
                    assert_eq!(loaded, &alone, "program {}", index)
                }
                (alone, failure) => panic!(
                    "program {}: alone {:?}, loaded failure {:?}",
                    index, alone, failure
                ),
            }
        }
        assert_eq!(live, 6);
        assert!(matches!(
            loaded.set.programs()[3].evaluated(),
            Some(Evaluated {
                plays: Plays::Sequence(_),
                ..
            })
        ));
        assert_eq!(loaded.failed.len(), 1);
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
