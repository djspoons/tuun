//! Checks of the `slider` built-in, from source text through the checker and
//! evaluation to the owners its sliders resolve to.
#![cfg(feature = "native")]

use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use tuun::diagnostics::Diagnostic;
use tuun::environment::Environment;
use tuun::expr::SourceExpr;
use tuun::generator::{self, Generator};
use tuun::ids::MarkId;
use tuun::player;
use tuun::processing;
use tuun::programs::{Evaluated, Outcome, Plays, ProgramSet, bank_address};
use tuun::waveform::Waveform;

const SAMPLE_RATE: u32 = 8000;

/// A library module whose function calls `slider` inside an operator.
const LIBRARY: &str =
    "pad = fn(freq) => (time * freq) * slider(\"cutoff\", 800, 100, 11000, \"exp\");\n";

/// An annotation that lists `cutoff`.
const LISTS: &str = "sliders=[\"cutoff\"]";

/// An annotation that lists nothing.
const PLAIN: &str = "level_db=0";

/// Returns an environment whose library root holds `LIBRARY` as `slider_lib`.
fn environment() -> Environment {
    // Written once: tests run in parallel, and one rewriting the file can
    // leave it empty while another reads it.
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    let root = ROOT.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("slider_lib");
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("slider_lib.tuun"), LIBRARY).unwrap();
        root
    });
    Environment::new(SAMPLE_RATE, 90, root.clone())
}

/// Returns the set for `source`, which must parse cleanly.
fn set(source: &str) -> ProgramSet {
    let (set, warning) = ProgramSet::from_source(source.to_string(), PathBuf::new())
        .expect("test source should parse");
    assert_eq!(warning, "");
    set
}

/// Checks and evaluates `_ = program`, returning the diagnostics on failure.
fn process(program: &str) -> Result<Evaluated, Vec<Diagnostic>> {
    let set = set(&format!("#{{{PLAIN}}}\n_ = {};\n", program));
    match processing::process(&environment(), &set.evaluation_input(0)) {
        Outcome::Evaluated(evaluated, _warnings) => Ok(evaluated),
        Outcome::Failed(diagnostics) => Err(diagnostics),
        Outcome::Empty => panic!("a non-empty program evaluates to something"),
    }
}

/// Returns the waveform the program at `index` of `set` evaluates to.
fn waveform(set: &ProgramSet, index: usize) -> Result<Waveform<MarkId>, Vec<Diagnostic>> {
    match environment()
        .evaluate_program(&set.evaluation_input(index))?
        .plays
    {
        Plays::Waveform(waveform, _) => Ok(waveform),
        Plays::Sequence(sequence) => Ok(sequence.step_waveform),
        Plays::KeysInstrument(_) => panic!("expected a waveform"),
    }
}

/// Returns each slider mark in `w` as `"label of A:1 = value"`, in pre-order.
fn sliders(w: &Waveform<MarkId>) -> Vec<String> {
    fn walk(w: &Waveform<MarkId>, out: &mut Vec<String>) {
        match w {
            Waveform::Marked { id, waveform } => {
                match (id, &**waveform) {
                    (MarkId::Slider { program, label }, Waveform::Const(value)) => {
                        out.push(format!("{label} of {} = {value}", bank_address(*program)));
                    }
                    (MarkId::SliderCall(call), _) => {
                        out.push(format!("{} unresolved", call.0.label))
                    }
                    _ => {}
                }
                walk(waveform, out);
            }
            Waveform::BinaryOp(_, a, b) | Waveform::Append(a, b, _) => {
                walk(a, out);
                walk(b, out);
            }
            Waveform::Fin {
                length, waveform, ..
            } => {
                walk(length, out);
                walk(waveform, out);
            }
            Waveform::UnaryOp(_, a) => walk(a, out),
            other => {
                let shown = format!("{other:?}");
                assert!(!shown.contains("Slider"), "missed a node kind: {shown}");
            }
        }
    }
    let mut out = Vec::new();
    walk(w, &mut out);
    out
}

/// Returns the messages of `diagnostics`, each with its line.
fn messages(diagnostics: &[Diagnostic]) -> Vec<String> {
    diagnostics.iter().map(|d| d.to_string()).collect()
}

#[test]
fn the_checker_accepts_each_arity() {
    for program in [
        "time * slider(\"mix\", 0.5)",
        "time * slider(\"Q\", 0.707, 0.1, 1.1)",
        "time * slider(\"cutoff\", 800, 100, 11000, \"exp\")",
        // Any scalar expression will do, not only a literal.
        "let lo = 100 in time * slider(\"cutoff\", lo * 8, lo, lo * 110, \"exp\")",
    ] {
        if let Err(diagnostics) = process(program) {
            panic!("{program}: {diagnostics:?}");
        }
    }
}

#[test]
fn the_checker_rejects_the_wrong_sorts_and_arities() {
    for program in [
        "slider(\"mix\", time)",
        "slider(1, 0.5)",
        "slider(\"mix\", 0.5, 0)",
        "slider(\"cutoff\", 800, 100, 11000, 2)",
        // A slider is never a plain number.
        "slider(\"outer\", slider(\"inner\", 0.5))",
    ] {
        assert!(process(program).is_err(), "{program} should not check");
    }
}

#[test]
fn an_invalid_spec_is_an_error_at_the_call() {
    let diagnostics = process("time * slider(\"q\", 2, 0, 1)").unwrap_err();
    let shown = format!("{diagnostics:?}");
    assert!(
        shown.contains("slider \\\"q\\\": default 2 is outside its range 0 to 1"),
        "{shown}"
    );
}

#[test]
fn an_unlisted_slider_is_the_played_programs_and_plays_its_default() {
    let set = set(&format!("#{{{PLAIN}}}\n_ = slider(\"level\", 0.25);\n"));
    let waveform = waveform(&set, 0).unwrap();
    assert_eq!(sliders(&waveform), ["level of A:1 = 0.25"]);
    let mut out = vec![1.0; 8];
    let len =
        Generator::new(SAMPLE_RATE).generate(&mut generator::initialize_state(waveform), &mut out);
    assert_eq!(len, 8);
    assert!(out.iter().all(|&x| x == 0.25), "{out:?}");
}

/// Returns the source of the design's ownership example: `A:1` defines
/// `delayed_pad`, which calls the library's `pad`, and `A:2` and `A:3` call
/// it, with the given annotations.
fn example(a1: &str, a2: &str) -> String {
    format!(
        "open slider_lib;\n\
         #{{{a1}}}\n\
         delayed_pad = fn(freq) => pad(freq);\n\
         #{{{a2}}}\n\
         _ = delayed_pad(220);\n\
         #{{{PLAIN}}}\n\
         _ = delayed_pad(330);\n"
    )
}

/// Returns who owns the `cutoff` that the program at `index` of `source`
/// reaches, or its errors.
fn owner(source: &str, index: usize) -> Result<Vec<String>, Vec<String>> {
    let set = set(source);
    waveform(&set, index)
        .map(|w| sliders(&w))
        .map_err(|diagnostics| messages(&diagnostics))
}

#[test]
fn without_a_listing_each_played_program_owns_its_own() {
    let source = example(PLAIN, PLAIN);
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:2 = 800".to_string()])
    );
    assert_eq!(
        owner(&source, 2),
        Ok(vec!["cutoff of A:3 = 800".to_string()])
    );
}

#[test]
fn a_listing_on_a_shared_binding_shares_its_slider() {
    let source = example(LISTS, PLAIN);
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:1 = 800".to_string()])
    );
    assert_eq!(
        owner(&source, 2),
        Ok(vec!["cutoff of A:1 = 800".to_string()])
    );
}

#[test]
fn a_listing_on_a_played_program_is_its_own() {
    let source = example(PLAIN, LISTS);
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:2 = 800".to_string()])
    );
    assert_eq!(
        owner(&source, 2),
        Ok(vec!["cutoff of A:3 = 800".to_string()])
    );
}

#[test]
fn two_listings_on_one_path_are_an_error_at_both() {
    let source = example(LISTS, LISTS);
    let message = "slider \"cutoff\" is listed by A:1 (delayed_pad) and A:2, and this \
                   waveform is built through each of them; keep only one listing";
    // The annotations are on lines 2 and 4 of the file.
    assert_eq!(
        owner(&source, 1),
        Err(vec![format!("2:3: {message}"), format!("4:3: {message}")])
    );
    // A program that reaches only one listing is unaffected.
    assert_eq!(
        owner(&source, 2),
        Ok(vec!["cutoff of A:1 = 800".to_string()])
    );
}

#[test]
fn a_slider_passed_into_a_listing_binding_is_the_callers() {
    let source = format!(
        "#{{{LISTS}}}\n\
         wrap = fn(w) => w * 2;\n\
         #{{{PLAIN}}}\n\
         _ = wrap(slider(\"cutoff\", 800, 100, 11000));\n"
    );
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:2 = 800".to_string()])
    );
}

#[test]
fn a_function_a_listing_binding_applies_is_its_own_however_it_applies_it() {
    // Directly, and through a built-in, whose own application of the function
    // is written nowhere.
    for apply in ["f(220)", "{map(f, [220])}"] {
        let source = format!(
            "#{{{LISTS}}}\n\
             play = fn(f) => {apply};\n\
             #{{{PLAIN}}}\n\
             _ = play(fn(freq) => (time * freq) * slider(\"cutoff\", 800, 100, 11000));\n"
        );
        assert_eq!(
            owner(&source, 1),
            Ok(vec!["cutoff of A:1 = 800".to_string()]),
            "{apply}"
        );
    }
}

#[test]
fn an_alias_binding_is_not_on_the_path() {
    let source = format!(
        "open slider_lib;\n\
         #{{{LISTS}}}\n\
         alias = pad;\n\
         #{{{PLAIN}}}\n\
         _ = alias(220);\n"
    );
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:2 = 800".to_string()])
    );
}

#[test]
fn a_listing_binding_whose_value_is_built_owns_it() {
    let source = format!(
        "open slider_lib;\n\
         #{{{LISTS}}}\n\
         built = pad(220);\n\
         #{{{PLAIN}}}\n\
         _ = built;\n"
    );
    assert_eq!(
        owner(&source, 1),
        Ok(vec!["cutoff of A:1 = 800".to_string()])
    );
}

#[test]
fn calls_that_agree_share_a_slider() {
    let set = set(&format!(
        "#{{{PLAIN}}}\n_ = slider(\"x\", 0.5) + slider(\"x\", 0.5);\n"
    ));
    assert_eq!(
        sliders(&waveform(&set, 0).unwrap()),
        ["x of A:1 = 0.5", "x of A:1 = 0.5"]
    );
}

#[test]
fn calls_that_disagree_are_an_error_naming_both_values() {
    let set = set(&format!(
        "#{{{PLAIN}}}\n_ = slider(\"x\", 0.5) + slider(\"x\", 0.25, 0, 2);\n"
    ));
    let diagnostics = waveform(&set, 0).unwrap_err();
    assert_eq!(
        messages(&diagnostics),
        [
            "1:20: slider \"x\" of A:1 is 0.25 (0 to 2, linear) here but 0.5 (0 to 1, linear) \
          at another call; calls of one slider must agree"
        ]
    );
    // The error points at the later call in the program's own text.
    assert_eq!(diagnostics[0].program_range, Some(19..42));
}

#[test]
fn a_sequence_steps_sliders_are_resolved() {
    let set = set(&format!(
        "on_beats = fn(w) => fn(bs) => w;\n\
         #{{{PLAIN}}}\n\
         _ = [1, 2.5] | on_beats(slider(\"x\", 0.5) | fin(time - 1));\n"
    ));
    assert!(matches!(
        environment().evaluate_program(&set.evaluation_input(0)),
        Ok(Evaluated {
            plays: Plays::Sequence(_),
            ..
        })
    ));
    assert_eq!(sliders(&waveform(&set, 0).unwrap()), ["x of A:1 = 0.5"]);
}

/// Returns the set for `source` with each program evaluated, as when the file
/// is loaded. A program that fails, such as one defining a function, is left
/// unevaluated.
fn loaded(source: &str) -> ProgramSet {
    processing::load(&environment(), source.to_string(), PathBuf::new())
        .expect("test source should parse")
        .set
}

/// Returns the waveform the program at `index` of `set` plays, with its sliders
/// at their current values.
fn played(set: &ProgramSet, index: usize) -> Waveform<MarkId> {
    let mut waveform = waveform(set, index).unwrap();
    player::substitute_current_slider_values(&mut waveform, set);
    waveform
}

#[test]
fn a_listed_slider_moves_along_its_curve() {
    let mut set = loaded(&example(LISTS, PLAIN));
    assert_eq!(sliders(&played(&set, 1)), ["cutoff of A:1 = 800"]);
    set.set_slider_normalized(0, 0, 1.0)
        .expect("A:1 lists a slider its programs reach");
    assert_eq!(sliders(&played(&set, 1)), ["cutoff of A:1 = 11000"]);
    // Halfway along an exp curve is the range's geometric mean.
    set.set_slider_normalized(0, 0, 0.5).unwrap();
    let value = set.slider(0, "cutoff").unwrap().value;
    assert!(
        (value - (100.0f32 * 11000.0).sqrt()).abs() < 0.01,
        "{value}"
    );
}

#[test]
fn a_slider_starts_at_the_value_its_owner_saves() {
    let set = loaded(&example("sliders=[\"cutoff:1200\"]", PLAIN));
    assert_eq!(sliders(&played(&set, 1)), ["cutoff of A:1 = 1200"]);
    assert_eq!(sliders(&played(&set, 2)), ["cutoff of A:1 = 1200"]);
}

#[test]
fn an_unlisted_slider_is_on_its_owners_encoders() {
    let mut set = loaded(&example(PLAIN, PLAIN));
    assert_eq!(set.slider(1, "cutoff").unwrap().value, 800.0);
    assert_eq!(set.slider(2, "cutoff").unwrap().value, 800.0);
    // Each playing program owns its own, on its own first encoder.
    assert_eq!(set.shown(1), ["cutoff"]);
    set.set_slider_normalized(1, 0, 1.0).unwrap();
    assert_eq!(set.slider(1, "cutoff").unwrap().value, 11000.0);
    assert_eq!(set.slider(2, "cutoff").unwrap().value, 800.0);
    // The function binding plays nothing, so it owns nothing to show.
    assert!(set.shown(0).is_empty());
}

#[test]
fn programs_owning_one_label_separately_need_not_agree() {
    let set = loaded(&format!(
        "#{{{PLAIN}}}
_ = slider(\"x\", 0.5);
#{{{PLAIN}}}
_ = slider(\"x\", 2, 0, 4);
"
    ));
    assert_eq!(set.slider(0, "x").unwrap().spec.max, 1.0);
    assert_eq!(set.slider(1, "x").unwrap().spec.max, 4.0);
}

#[test]
fn a_keys_note_resolves_its_sliders() {
    let mut set = set(&format!(
        "open slider_lib;\n\
         #{{{LISTS}}}\n\
         delayed_pad = fn(freq) => pad(freq);\n\
         #{{keys}}\n\
         k = fn(note, vel) => (delayed_pad(note), slider(\"release\", 0.5));\n"
    ));
    let environment = environment();
    let Ok(Evaluated {
        plays: Plays::KeysInstrument(function),
        sliders: reached,
    }) = environment.evaluate_program(&set.evaluation_input(1))
    else {
        panic!("expected a keys instrument");
    };
    let arguments = || vec![SourceExpr::float(60.0), SourceExpr::float(0.5)];
    let (note_on, note_off, _, note_reached) = environment
        .apply_note_function(&function, arguments(), &set.slider_owners(1))
        .expect("a note should play");
    assert_eq!(sliders(&note_on), ["cutoff of A:1 = 800"]);
    assert_eq!(sliders(&note_off), ["release of A:2 = 0.5"]);
    // The instrument reaches what its sample note reaches, the note-on's
    // first.
    let owned: Vec<_> = reached
        .iter()
        .map(|r| (r.owner, r.label.as_str()))
        .collect();
    assert_eq!(owned, [(0, "cutoff"), (1, "release")]);
    assert_eq!(note_reached, reached);
    set.reach(&reached);

    // A note played after the shared slider moves carries its new value.
    set.set_slider_normalized(0, 0, 1.0)
        .expect("A:1 lists a slider");
    let (mut note_on, _, _, _) = environment
        .apply_note_function(&function, arguments(), &set.slider_owners(1))
        .expect("a note should play");
    player::substitute_current_slider_values(&mut note_on, &set);
    assert_eq!(sliders(&note_on), ["cutoff of A:1 = 11000"]);
}

/// The slider demo loads with every program evaluating, and each program's
/// listed sliders are the ones its waveform reaches.
// TODO maybe remove this as tests should be more self-contained
#[test]
fn the_slider_demo_loads_and_reaches_what_it_lists() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
    let path = root.join("demos/sliders.tuun");
    let source = fs::read_to_string(&path).unwrap();
    let environment = Environment::new(SAMPLE_RATE, 90, root);
    let loaded = processing::load(&environment, source, path).expect("the demo parses");
    assert_eq!(loaded.warning, "");
    let failed: Vec<_> = loaded
        .failed
        .iter()
        .map(|(index, diagnostics)| (*index, messages(diagnostics)))
        .collect();
    assert_eq!(failed, []);
    let set = loaded.set;
    for index in 0..set.programs().len() {
        for entry in set.listed(index) {
            assert!(
                set.slider(index, &entry.label).is_some(),
                "{} lists {:?} but doesn't reach it",
                set.display_name(index),
                entry.label
            );
        }
    }
}

/// Returns the messages of the checker's findings for the program at `index`
/// of `source`.
fn check(source: &str, index: usize) -> Vec<String> {
    messages(&environment().check_program(&set(source).evaluation_input(index)))
}

/// Returns each slider the program at `index` of `source` can reach, as
/// `"label of A:1 = spec"`, in order.
fn reachable(source: &str, index: usize) -> Vec<String> {
    let (reached, _) = environment()
        .reachable_sliders(&set(source).evaluation_input(index))
        .expect("test source should parse");
    reached
        .iter()
        .map(|r| format!("{} of {} = {}", r.label, bank_address(r.owner), r.spec))
        .collect()
}

#[test]
fn a_label_must_be_written_in_its_call() {
    let source = format!("#{{{PLAIN}}}\n_ = let l = \"q\" in time * slider(l, 0.5);\n");
    assert_eq!(
        check(&source, 0),
        ["1:30: a slider's label must be a string written in its `slider` call"]
    );
}

#[test]
fn two_listings_a_call_is_reached_through_are_an_error_before_it_plays() {
    let source = example(LISTS, LISTS);
    let message = "slider \"cutoff\" is listed by A:1 (delayed_pad) and A:2, and this \
                   waveform is built through each of them; keep only one listing";
    assert_eq!(
        check(&source, 1),
        [format!("2:3: {message}"), format!("4:3: {message}")]
    );
    assert_eq!(check(&source, 2), Vec::<String>::new());
    assert_eq!(
        reachable(&source, 2),
        ["cutoff of A:1 = 800 (100 to 11000, exp)"]
    );
}

#[test]
fn a_listing_binding_provides_what_is_reached_through_it() {
    let source = example(LISTS, PLAIN);
    for index in [1, 2] {
        assert_eq!(check(&source, index), Vec::<String>::new());
        assert_eq!(
            reachable(&source, index),
            ["cutoff of A:1 = 800 (100 to 11000, exp)"]
        );
    }
    let source = example(PLAIN, PLAIN);
    assert_eq!(
        reachable(&source, 2),
        ["cutoff of A:3 = 800 (100 to 11000, exp)"]
    );
}

#[test]
fn a_listing_function_whose_spec_its_callers_give_is_an_error_at_the_listing() {
    for (slider, part) in [
        ("slider(\"q\", d / 4)", "default"),
        ("slider(\"q\", 0.5, 0, d)", "range"),
    ] {
        let source = format!(
            "#{{sliders=[\"q\"]}}\n\
             pad = fn(d) => time * {slider};\n\
             #{{{PLAIN}}}\n\
             _ = pad(2);\n"
        );
        let message = format!(
            "1:3: A:1 (pad) lists slider \"q\", but its {part} depends on `d`, which its \
             callers give; list it where `d` is given instead"
        );
        assert_eq!(check(&source, 1), [message], "{slider}");
    }
}

#[test]
fn a_named_parameter_a_caller_may_give_is_one_its_callers_give() {
    let source = format!(
        "#{{sliders=[\"q\"]}}\n\
         pad = fn(top = 1) => time * slider(\"q\", 0.5, 0, top);\n\
         #{{{PLAIN}}}\n\
         _ = pad();\n"
    );
    assert_eq!(
        check(&source, 1),
        [
            "1:3: A:1 (pad) lists slider \"q\", but its range depends on `top`, which its \
             callers give; list it where `top` is given instead"
        ]
    );
}

#[test]
fn a_function_returned_by_a_listing_binding_is_checked_alone_too() {
    let source = format!(
        "#{{sliders=[\"q\"]}}\n\
         pad = fn(a) => fn(b) => time * slider(\"q\", b);\n\
         #{{{PLAIN}}}\n\
         _ = pad(1)(0.5);\n"
    );
    assert_eq!(
        check(&source, 1),
        [
            "1:3: A:1 (pad) lists slider \"q\", but its default depends on `b`, which its \
             callers give; list it where `b` is given instead"
        ]
    );
}

#[test]
fn a_spec_its_callers_give_is_the_played_programs_when_nothing_lists_it() {
    let source = format!(
        "#{{{PLAIN}}}\n\
         pad = fn(d) => time * slider(\"q\", d * 2);\n\
         #{{{PLAIN}}}\n\
         _ = pad(0.2);\n"
    );
    assert_eq!(check(&source, 1), Vec::<String>::new());
    assert_eq!(reachable(&source, 1), ["q of A:2 = 0.4 (0 to 1, linear)"]);
    assert_eq!(owner(&source, 1), Ok(vec!["q of A:2 = 0.4".to_string()]));
}

#[test]
fn a_listing_binding_may_not_apply_a_function_its_callers_give() {
    for (apply, column) in [("f(220)", 17), ("{map(f, [220])}", 18)] {
        let source = format!(
            "#{{{LISTS}}}\n\
             play = fn(f) => {apply};\n\
             #{{{PLAIN}}}\n\
             _ = play(fn(freq) => (time * freq) * slider(\"cutoff\", 800, 100, 11000));\n"
        );
        assert_eq!(
            check(&source, 1),
            [format!(
                "2:{column}: A:1 (play) provides slider \"cutoff\" here, so it can't apply \
                 `f`, a function its callers give"
            )],
            "{apply}"
        );
    }
    // It may pass one along to be applied outside it.
    let source = format!(
        "#{{{LISTS}}}\n\
         play = fn(f) => (f, slider(\"cutoff\", 800, 100, 11000));\n\
         #{{{PLAIN}}}\n\
         _ = let (g, c) = play(fn(x) => x * time) in g(c);\n"
    );
    assert_eq!(check(&source, 1), Vec::<String>::new());
}

#[test]
fn branches_a_note_chooses_between_must_agree() {
    let source = "#{keys}\n\
                  k = fn(note, vel) => \
                  (if note > 60 then slider(\"q\", 0.5) else slider(\"q\", 0.25), 1);\n";
    assert_eq!(
        check(source, 0),
        [
            "1:59: slider \"q\" of A:1 (k) is 0.25 (0 to 1, linear) here but 0.5 (0 to 1, \
             linear) at another call; calls of one slider must agree"
        ]
    );
}

#[test]
fn a_keys_slider_whose_spec_depends_on_the_note_is_an_error() {
    for (program, what) in [
        ("slider(\"q\", note / 128)", "the key a note plays"),
        (
            "slider(\"q\", 0.5, 0, vel + 1)",
            "the velocity a note plays with",
        ),
    ] {
        let source = format!("#{{keys}}\nk = fn(note, vel) => ({program}, 1);\n");
        let part = if program.contains("vel") {
            "range"
        } else {
            "default"
        };
        assert_eq!(
            check(&source, 0),
            [format!(
                "1:19: slider \"q\"'s {part} depends on {what}, which isn't known until it plays"
            )]
        );
    }
}

#[test]
fn a_keys_instrument_reaches_the_sliders_of_every_note() {
    let source = "#{keys}\n\
                  k = fn(note, vel) => \
                  (if note > 60 then slider(\"hi\", 0.5) else slider(\"lo\", 0.5), 1);\n";
    assert_eq!(check(source, 0), Vec::<String>::new());
    assert_eq!(
        reachable(source, 0),
        [
            "hi of A:1 = 0.5 (0 to 1, linear)",
            "lo of A:1 = 0.5 (0 to 1, linear)"
        ]
    );
}

#[test]
fn an_invalid_spec_on_a_note_not_played_at_load_is_an_error() {
    let source = "#{keys}\n\
                  k = fn(note, vel) => \
                  (if note > 100 then slider(\"q\", 2) else 0, 1);\n";
    assert_eq!(
        check(source, 0),
        ["1:38: slider \"q\": default 2 is outside its range 0 to 1"]
    );
}

#[test]
fn the_same_spec_under_map_is_one_slider() {
    let source =
        format!("#{{{PLAIN}}}\n_ = {{map(fn(f) => (time * f) * slider(\"q\", 0.5), [1, 2])}};\n");
    assert_eq!(check(&source, 0), Vec::<String>::new());
    assert_eq!(reachable(&source, 0), ["q of A:1 = 0.5 (0 to 1, linear)"]);
}

#[test]
fn the_slider_demo_checks_clean_and_reaches_what_it_lists() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0/demos/sliders.tuun");
    let source = fs::read_to_string(&path).unwrap();
    let set = set(&source);
    let environment = Environment::new(
        SAMPLE_RATE,
        90,
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0"),
    );
    for index in 0..set.programs().len() {
        if set.programs()[index].text().trim().is_empty() {
            continue;
        }
        let input = set.evaluation_input(index);
        assert_eq!(
            messages(&environment.check_program(&input)),
            Vec::<String>::new(),
            "{}",
            set.display_name(index)
        );
        let (reached, _) = environment.reachable_sliders(&input).unwrap();
        let mut labels: Vec<&str> = reached.iter().map(|r| r.label.as_str()).collect();
        labels.sort();
        let mut listed: Vec<&str> = set.listed(index).iter().map(|s| s.label.as_str()).collect();
        listed.sort();
        assert_eq!(labels, listed, "{}", set.display_name(index));
        assert!(reached.iter().all(|r| r.owner == index));
    }
}

#[test]
fn a_listing_binding_may_compute_with_what_its_callers_give() {
    // `seq(time - dur)` is a built-in computed from `dur`, not `dur` itself.
    let source = format!(
        "#{{{LISTS}}}\n\
         play = fn(dur) => (time * slider(\"cutoff\", 800, 100, 11000)) | seq(time - dur);\n\
         #{{{PLAIN}}}\n\
         _ = play(1);\n"
    );
    assert_eq!(check(&source, 1), Vec::<String>::new());
    assert_eq!(
        reachable(&source, 1),
        ["cutoff of A:1 = 800 (100 to 11000, linear)"]
    );
}

#[test]
fn a_function_a_listing_binding_hands_its_callers_function_is_checked_alone() {
    let source = "#{level_db=0}\n\
                  apply_to = fn(f, g) => fn() => f(g);\n\
                  #{sliders=[\"q\"]}\n\
                  pad = fn(f) => apply_to(f, fn(x) => time * slider(\"q\", x));\n\
                  #{level_db=0}\n\
                  _ = pad(fn(g) => g(0.5))();\n";
    assert_eq!(
        check(source, 2),
        [
            "3:3: A:2 (pad) lists slider \"q\", but its default depends on `x`, which its \
             callers give; list it where `x` is given instead"
        ]
    );
}

#[test]
fn the_built_ins_passing_functions_through_are_those_with_polymorphic_signatures() {
    use tuun::builtins;
    use tuun::expr::{Binding, Expr};
    use tuun::signatures::signature;
    use tuun::slider_analysis::{PASSING, SPECIAL};
    use tuun::types::Type;

    let mut bindings = Vec::new();
    builtins::add_bindings::<u32, u32>(&mut bindings);
    let mut polymorphic: Vec<String> = bindings
        .iter()
        .filter_map(|binding| match &binding.binding {
            Binding::Definition(_, value) => match &value.expr {
                Expr::BuiltIn { name, .. } => Some(name.clone()),
                _ => None,
            },
            _ => None,
        })
        .filter(|name| matches!(signature(name), Some(Type::Forall(..))))
        .filter(|name| !SPECIAL.contains(&name.as_str()))
        .collect();
    polymorphic.sort();
    let mut passing: Vec<&str> = PASSING.to_vec();
    passing.sort();
    assert_eq!(polymorphic, passing);
}

#[test]
fn a_function_handed_through_debug_is_still_the_callers() {
    let source = format!(
        "#{{{LISTS}}}\n\
         play = fn(f) => debug(\"f\")(f)(220);\n\
         #{{{PLAIN}}}\n\
         _ = play(fn(freq) => (time * freq) * slider(\"cutoff\", 800, 100, 11000));\n"
    );
    assert_eq!(
        check(&source, 1),
        [
            "2:17: A:1 (play) provides slider \"cutoff\" here, so it can't apply `f`, a \
          function its callers give"
        ]
    );
}
