//! Checks which program text each voice's marks light, and when, from source
//! text through evaluation and the walk at the evaluation boundary, using the
//! real std library.
#![cfg(feature = "native")]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::sync::OnceLock;

use tuun::environment::Environment;
use tuun::expr::SourceExpr;
use tuun::programs::{Evaluated, ProgramSet};
use tuun::trace_walk::{self, WaveformMarks};

/// Samples per second: with `TEMPO`, a quarter note (`Q`) is 500 ms.
const SAMPLE_RATE: u32 = 8000;
const TEMPO: u32 = 120;

/// Modules standing in for code written outside the program: a drum whose
/// sound lasts 110 ms, and two 250 ms hits a quarter note apart.
const FIXTURES: [(&str, &str); 2] = [
    (
        "kit.tuun",
        "open std;\n\
         clap = fn(dur) => noise * (square(32) * 0.6) | fin(time - 0.11) | seq(time - dur);\n",
    ),
    (
        "hits.tuun",
        "open std;\n\
         two = <[$440 | fin(time - E) | seq(time - Q), $330 | fin(time - E) | seq(time - Q)]>;\n",
    ),
];

/// The bindings every program below sees: std, the fixtures, a drum voice
/// `c`, and a keys instrument `keys`.
const PREAMBLE: &str = "open std;\n\
                        use env.minmax;\n\
                        use synth.pm;\n\
                        use kit;\n\
                        use hits;\n\
                        #{level_db=0}\n\
                        c = kit.clap(E) | unseq();\n\
                        #{level_db=0, keys}\n\
                        keys = pm.brass_keys;\n";

/// Returns a library root holding the real library and the fixtures.
fn library_root() -> &'static PathBuf {
    static ROOT: OnceLock<PathBuf> = OnceLock::new();
    ROOT.get_or_init(|| {
        let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("lighting-library");
        fs::create_dir_all(&root).expect("create the library root");
        let library = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
        for entry in fs::read_dir(&library).expect("read the library").flatten() {
            let link = root.join(entry.file_name());
            if fs::symlink_metadata(&link).is_err() {
                std::os::unix::fs::symlink(entry.path(), link).expect("link a library entry");
            }
        }
        for (name, text) in FIXTURES {
            fs::write(root.join(name), text).expect("write a fixture module");
        }
        root
    })
}

/// The marks of one voice, with the program text they index.
struct Voice {
    text: String,
    marks: WaveformMarks,
    /// Marks a step voice adds to `marks`.
    extra: Vec<trace_walk::Mark>,
}

/// Evaluates the program `_ = program` annotated with `annotations`, after
/// the preamble, and returns the marks of the voice it plays: its own, its
/// step voice at `sixteenth` when it is sequenced, or one note when it is a
/// keys instrument.
fn voice_with(annotations: &str, program: &str, sixteenth: u8) -> Voice {
    voice_after(PREAMBLE, annotations, program, sixteenth)
}

/// Returns the marks of the program `_ = program` as [`voice_with`] does,
/// after `preamble` instead of the shared one.
fn voice_after(preamble: &str, annotations: &str, program: &str, sixteenth: u8) -> Voice {
    let source = format!("{}#{{{}}}\n_ = {};\n", preamble, annotations, program);
    let (set, _) = ProgramSet::from_source(source, PathBuf::new()).expect("source should parse");
    let index = set
        .programs()
        .iter()
        .rposition(|p| !p.text().trim().is_empty())
        .expect("a program");
    let environment = Environment::new(SAMPLE_RATE, TEMPO, library_root().clone());
    let text = set.programs()[index].text().to_string();
    let (marks, extra) = match environment.evaluate_program(&set.evaluation_input(index)) {
        Ok(Evaluated::Waveform(_, marks)) => ((*marks).clone(), vec![]),
        Ok(Evaluated::Sequence(sequence)) => (
            (*sequence.step_marks).clone(),
            sequence.step_literal_marks(sixteenth),
        ),
        Ok(Evaluated::KeysInstrument(function)) => {
            let arguments = vec![SourceExpr::float(60.0), SourceExpr::float(0.7)];
            let (_, _, marks) = environment
                .apply_note_function(&function, arguments)
                .expect("a note should play");
            ((*marks).clone(), vec![])
        }
        Err(diagnostics) => panic!("{}: {:?}", program, diagnostics),
    };
    Voice { text, marks, extra }
}

/// Returns each lit range's text and when it is current, in milliseconds from
/// the voice's start, in text order: e.g. `"$x: 0..500 750..1250"`. An
/// interval with no end is current while the voice sounds.
fn lit(voice: &Voice) -> Vec<String> {
    let segments = &voice.marks.segments;
    // Each range's intervals, as (start, end) in samples.
    type Intervals = Vec<(usize, Option<usize>)>;
    let mut times: BTreeMap<(usize, usize), Intervals> = BTreeMap::new();
    for mark in voice.marks.marks.iter().chain(&voice.extra) {
        let intervals = times
            .entry((mark.text_range.start, mark.text_range.end))
            .or_default();
        for segment in &segments[mark.segments.clone()] {
            let end = segment.duration.map(|d| segment.offset + d);
            intervals.push((segment.offset, end));
        }
    }
    let ms = |samples: usize| samples * 1000 / SAMPLE_RATE as usize;
    times
        .into_iter()
        .map(|((start, end), mut intervals)| {
            intervals.sort();
            let mut merged: Vec<(usize, Option<usize>)> = Vec::new();
            for (from, to) in intervals {
                match merged.last_mut() {
                    Some((_, last_end)) if last_end.is_none_or(|e| from <= e) => {
                        *last_end = match (*last_end, to) {
                            (Some(a), Some(b)) => Some(a.max(b)),
                            _ => None,
                        };
                    }
                    _ => merged.push((from, to)),
                }
            }
            let merged: Vec<String> = merged
                .into_iter()
                .map(|(from, to)| match to {
                    Some(to) => format!("{}..{}", ms(from), ms(to)),
                    None => format!("{}..", ms(from)),
                })
                .collect();
            format!("{}: {}", &voice.text[start..end], merged.join(" "))
        })
        .collect()
}

/// Returns the marks of the waveform program `_ = program` after the
/// preamble.
fn voice(program: &str) -> Voice {
    voice_with("level_db=0", program, 0)
}

/// Returns the voice's segments as `offset+duration` in milliseconds, or
/// `offset+` for one that sounds as long as the voice.
fn segments(voice: &Voice) -> Vec<String> {
    let ms = |samples: usize| samples * 1000 / SAMPLE_RATE as usize;
    voice
        .marks
        .segments
        .iter()
        .map(|s| match s.duration {
            Some(d) => format!("{}+{}", ms(s.offset), ms(d)),
            None => format!("{}+", ms(s.offset)),
        })
        .collect()
}

#[test]
fn t1_each_mapped_note_lights_its_own_element_and_the_shared_code() {
    let v = voice("<map(fn (x) => $x * Qw | unseq() | seq(time - Q*1.5), [220, 330])>");
    assert_eq!(segments(&v), ["0+500", "750+500"]);
    assert_eq!(
        lit(&v),
        [
            "$x: 0..500 750..1250",
            "x: 0..500 750..1250",
            "*: 0..500 750..1250",
            "Qw: 0..500 750..1250",
            "220: 0..500",
            "330: 750..1250",
        ]
    );
}

#[test]
fn t2_a_product_lights_its_operator_through_the_root_mark() {
    let v = voice("$440 * Qw");
    assert_eq!(segments(&v), ["0+500"]);
    assert_eq!(
        lit(&v),
        ["$440: 0..500", "440: 0..500", "*: 0..500", "Qw: 0..500"]
    );
}

#[test]
fn t3_a_library_envelope_lights_its_call_but_not_the_whole_pipe() {
    let v = voice("$440 | minmax.ADSR(0.5, 0.01, 0.1, 0.7, 0.1)");
    assert_eq!(segments(&v), ["0+500"]);
    // The arguments light too, inside the call.
    assert_eq!(
        lit(&v),
        [
            "$440: 0..500",
            "440: 0..500",
            "minmax.ADSR(0.5, 0.01, 0.1, 0.7, 0.1): 0..500",
            "0.5: 0..500",
            "0.01: 0..500",
            "0.1: 0..500",
            "0.7: 0..500",
            "0.1: 0..500",
        ]
    );
}

#[test]
fn t4_a_voice_built_in_a_module_lights_its_call() {
    let v = voice("kit.clap(E) | unseq()");
    assert_eq!(segments(&v), ["0+110"]);
    assert_eq!(lit(&v), ["kit.clap(E): 0..110"]);
}

#[test]
fn t5_a_step_lights_its_waveform_and_its_own_literal() {
    for (sixteenth, literal) in [(0, "1"), (16, "5")] {
        let v = voice_with("level_db=0", "[1, 5] | on_beats(c)", sixteenth);
        assert_eq!(segments(&v), ["0+110"]);
        assert_eq!(
            lit(&v),
            [format!("{}: 0..110", literal), "c: 0..110".to_string()]
        );
    }
}

#[test]
fn t6_a_keys_note_lights_its_instrument_while_it_is_held() {
    let v = voice_with("level_db=0, keys", "pm.brass_keys", 0);
    assert_eq!(segments(&v), ["0+"]);
    assert_eq!(lit(&v), ["pm.brass_keys: 0.."]);
}

#[test]
fn t7_a_module_voice_lights_its_name_during_each_hit() {
    let v = voice("hits.two");
    assert_eq!(segments(&v), ["0+250", "500+250"]);
    assert_eq!(lit(&v), ["hits.two: 0..250 500..750"]);
}

#[test]
fn t8_a_scale_beside_two_hits_is_dark_in_the_gap() {
    let v = voice("hits.two * 0.5");
    assert_eq!(segments(&v), ["0+250", "500+250"]);
    assert_eq!(
        lit(&v),
        [
            "hits.two: 0..250 500..750",
            "*: 0..250 500..750",
            "0.5: 0..250 500..750",
        ]
    );
}

#[test]
fn t9_a_nested_note_lights_for_its_own_length() {
    // The spec's `($220 * Qw + $330) * Hw` multiplies two seqs, which tuun
    // rejects; `&` keeps the outer node as long as `Hw`.
    let v = voice("(($220 * Qw | unseq()) & $330) * Hw");
    assert_eq!(segments(&v), ["0+500", "0+1000"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "&: 0..1000",
            "$330: 0..1000",
            "330: 0..1000",
            "*: 0..1000",
            "Hw: 0..1000",
        ]
    );
}

#[test]
fn t10_code_above_two_segments_lights_during_both() {
    // The spec's `$220 * Qw + $277 * Qw` adds two seqs, which tuun rejects.
    let v = voice("($220 * Qw | unseq()) + ($277 * Qw | unseq())");
    assert_eq!(segments(&v), ["0+500", "0+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "+: 0..500",
            "$277: 0..500",
            "277: 0..500",
            "*: 0..500",
            "Qw: 0..500",
        ]
    );
}

#[test]
fn t11_a_folded_constant_lights_what_it_was_computed_from() {
    let v = voice("$(220 * 2) * Qw");
    assert_eq!(
        lit(&v),
        [
            "$(220 * 2): 0..500",
            "220: 0..500",
            "*: 0..500",
            "2: 0..500",
            "*: 0..500",
            "Qw: 0..500",
        ]
    );
}

#[test]
fn t12_both_branches_of_an_alt_light_for_the_whole_alt() {
    let v = voice("alt(time - 1, $330, $220) * Hw");
    assert_eq!(segments(&v), ["0+1000"]);
    assert_eq!(
        lit(&v),
        [
            "alt: 0..1000",
            "time: 0..1000",
            "-: 0..1000",
            "1: 0..1000",
            "$330: 0..1000",
            "330: 0..1000",
            "$220: 0..1000",
            "220: 0..1000",
            "*: 0..1000",
            "Hw: 0..1000",
        ]
    );
}

#[test]
fn t13_code_that_only_times_a_note_stays_dark() {
    let v = voice("$440 | fin(time - 0.5) | seq(time - 0.5)");
    assert_eq!(segments(&v), ["0+500"]);
    assert_eq!(lit(&v), ["$440: 0..500", "440: 0..500"]);
}

#[test]
fn t14_each_phrase_note_lights_its_key_and_velocity() {
    let v = voice("[(1, 60, 0.465, 0.5), (2, 64, 0.8, 0.5)] | as_midi_phrase(keys)");
    // Each note holds for half a beat, then releases for 100 ms.
    assert_eq!(segments(&v), ["0+350", "500+350"]);
    assert_eq!(
        lit(&v),
        [
            "60: 0..350",
            "0.465: 0..350",
            "64: 500..850",
            "0.8: 500..850",
            "as_midi_phrase(keys): 0..350 500..850",
        ]
    );
}

#[test]
fn left_nested_seqs_light_each_note_while_it_sounds() {
    // `(a \ b) \ c` delays `c` until a length that is negative until `a`'s
    // offset reaches 0, then is `b`'s offset, reaches 0.
    let v = voice(
        "(($220 | fin(time - Q) | seq(time - Q)) \\ ($330 | fin(time - Q) | seq(time - Q))) \
         \\ ($440 | fin(time - Q))",
    );
    assert_eq!(segments(&v), ["0+500", "500+500", "1000+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "$330: 500..1000",
            "330: 500..1000",
            "$440: 1000..1500",
            "440: 1000..1500",
        ]
    );
}

#[test]
fn a_mapped_element_lights_the_lambdas_use_of_it() {
    let v = voice("<map(fn (x) => x, [$220 * Qw | unseq() | seq(time - Q)])>");
    assert_eq!(
        lit(&v),
        [
            "x: 0..500",
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500"
        ]
    );
}

#[test]
fn a_reduced_accumulator_lights_the_lambdas_use_of_it() {
    let v = voice("reduce(fn (acc, y) => acc, $220 * Qw | unseq(), [1])");
    assert_eq!(
        lit(&v),
        [
            "acc: 0..500",
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500"
        ]
    );
}

#[test]
fn an_accumulator_built_in_the_lambda_lights_its_seed_and_operator() {
    let v = voice("reduce(fn (acc, y) => $acc * Qw | unseq(), 220, [1])");
    assert_eq!(
        lit(&v),
        [
            "$acc: 0..500",
            "acc: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "220: 0..500"
        ]
    );
}

#[test]
fn note_names_light_during_their_own_notes() {
    // As the counterpoint lessons write melodies: each name is a constant
    // the note's frequency was folded from.
    let preamble = "open std;\n\
                    use midi.notes;\n\
                    use synth.pm;\n\
                    brass_notes = fn(dur) => fn(notes) => <map(fn(n) => pm.brass(dur, @n), notes)>;\n\
                    (C, C#, D, D#, E, F, F#, G, G#, A, A#, B) = notes.midi_octave(4);\n";
    let v = voice_after(preamble, "level_db=0", "[D, F] | brass_notes(H)", 0);
    assert_eq!(
        lit(&v),
        [
            "D: 0..1000",
            "F: 1000..2000",
            "brass_notes(H): 0..2000",
            "H: 0..2000",
        ]
    );
}

#[test]
fn a_value_bound_outside_a_nested_min_lights() {
    // `max(…)` reaches `min` as a shared waveform, so `0.7` lights through
    // the marks held from it.
    let v = voice("let s = 0.7 in $440 * min(time, max(time * 2, s)) | fin(time - 1)");
    assert_eq!(segments(&v), ["0+1000"]);
    assert_eq!(
        lit(&v),
        [
            "0.7: 0..1000",
            "$440: 0..1000",
            "440: 0..1000",
            "*: 0..1000",
            "min(time, max(time * 2, s)): 0..1000",
            "time: 0..1000",
            "max(time * 2, s): 0..1000",
            "time: 0..1000",
            "*: 0..1000",
            "2: 0..1000",
            "s: 0..1000",
        ]
    );
}

#[test]
fn a_shared_waveform_lights_where_its_parameter_is_heard() {
    // `$440` is generated from the start, but heard only through `x`, after
    // the rest.
    let v = voice("share($440, fn(x) => 0 * Qw \\ x * Qw | unseq())");
    assert_eq!(segments(&v), ["500+500"]);
    assert_eq!(
        lit(&v),
        [
            "$440: 500..1000",
            "440: 500..1000",
            "0: ",
            "*: ",
            "Qw: ",
            "x: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
        ]
    );
}

#[test]
fn notes_a_shared_waveform_times_itself_light_as_a_copy_s_do() {
    // Each note lights during its own half, while `x` reads it.
    let v = voice("share($440 * Qw \\ $220 * Qw | unseq(), fn(x) => x)");
    let copy = voice("let x = $440 * Qw \\ $220 * Qw | unseq() in x");
    assert_eq!(segments(&v), ["0+500", "500+500"]);
    assert_eq!(segments(&v), segments(&copy));
    assert_eq!(lit(&v), lit(&copy));
}

#[test]
fn a_delayed_ref_lights_only_the_part_it_reads() {
    // `x` starts reading halfway through, so it hears only `$330`: `$220`'s
    // segment is cut before it starts.
    let v = voice("share(($220 * Qw \\ $330 * Qw) | unseq(), fn(x) => 0 * Qw \\ x)");
    assert_eq!(segments(&v), ["500+0", "500+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 500..500",
            "220: 500..500",
            "*: 500..500",
            "Qw: 500..500",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "0: ",
            "*: ",
            "Qw: ",
            "x: 500..1000",
        ]
    );
}

#[test]
fn a_fin_around_a_ref_cuts_the_parts_it_reads() {
    let v = voice(
        "share(($220 * Qw \\ $330 * Qw) | unseq(), fn(x) => (0 * Qw \\ x) | fin(time - 0.75))",
    );
    assert_eq!(segments(&v), ["500+0", "500+250"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 500..500",
            "220: 500..500",
            "*: 500..500",
            "Qw: 500..500",
            "$330: 500..750",
            "330: 500..750",
            "*: 500..750",
            "Qw: 500..750",
            "0: ",
            "*: ",
            "Qw: ",
            "x: 500..750",
        ]
    );
}

#[test]
fn a_note_after_a_rest_lights_when_it_starts() {
    let v = voice("0 * Qw \\ $440");
    assert_eq!(segments(&v), ["500+"]);
    assert_eq!(lit(&v), ["0: ", "*: ", "Qw: ", "$440: 500..", "440: 500.."]);
    // A finite waveform around it ends it, but doesn't start it.
    let v = voice("(0 * Qw \\ $440) * Hw");
    assert_eq!(segments(&v), ["500+500"]);
    assert_eq!(
        lit(&v),
        [
            "0: ",
            "*: ",
            "Qw: ",
            "$440: 500..1000",
            "440: 500..1000",
            "*: 500..1000",
            "Hw: 500..1000",
        ]
    );
}

#[test]
fn a_shared_control_has_no_segment_of_its_own() {
    // As with `time * Qw`, nothing sounds, so nothing lights.
    let v = voice("share(time, fn(x) => x) * Qw");
    assert_eq!(segments(&v), Vec::<String>::new());
    assert_eq!(lit(&v), ["time: ", "x: ", "*: ", "Qw: "]);
}

#[test]
fn a_rest_of_no_length_stays_dark() {
    // `Qrw` is `fin(0)`: like `0 * Qw`, `0 * Qrw` doesn't light while the
    // note after it plays.
    let v = voice("0 * Qrw \\ $440 * Hw");
    assert_eq!(segments(&v), ["500+1000"]);
    assert_eq!(
        lit(&v),
        [
            "0: ",
            "*: ",
            "Qrw: ",
            "$440: 500..1500",
            "440: 500..1500",
            "*: 500..1500",
            "Hw: 500..1500",
        ]
    );
}

#[test]
fn a_ref_to_a_finite_shared_waveform_is_measured_as_a_copy_is() {
    // `x` reads all of a quarter note, so the half note around it ends the
    // segment, as it does around a copy.
    let v = voice("share(time * Qw | unseq(), fn(x) => (x & $440) * Hw | unseq())");
    let copy = voice("let x = time * Qw | unseq() in (x & $440) * Hw | unseq()");
    assert_eq!(segments(&v), ["0+1000"]);
    assert_eq!(lit(&v), lit(&copy));
}

#[test]
fn notes_of_a_shared_waveform_heard_twice_light_when_they_play() {
    // The shared waveform plays once, on its own clock: the second `x` reads
    // its second half.
    let v = voice("share($220 * Qw \\ $330 * Qw | unseq(), fn(x) => x * Qw \\ x * Qw | unseq())");
    assert_eq!(segments(&v), ["0+500", "500+0", "500+0", "500+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "x: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "x: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
        ]
    );
}

#[test]
fn nested_shares_light_the_parts_each_ref_reads() {
    // `b` reads `a` after a rest, so `$220` is never heard; the first `b`
    // hears `$330` and the second, `$440`.
    let v = voice(
        "share($220 * Qw \\ $330 * Qw \\ $440 * Qw | unseq(), \
         fn(a) => share(0 * Qw \\ a, fn(b) => (b * Hw \\ b * Qw) | unseq()))",
    );
    assert_eq!(
        segments(&v),
        ["500+0", "500+500", "1000+0", "1000+0", "1000+0", "1000+500"]
    );
    assert_eq!(
        lit(&v),
        [
            "$220: 500..500 1000..1000",
            "220: 500..500 1000..1000",
            "*: 500..500 1000..1000",
            "Qw: 500..500 1000..1000",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "$440: 1000..1500",
            "440: 1000..1500",
            "*: 1000..1500",
            "Qw: 1000..1500",
            "0: ",
            "*: ",
            "Qw: ",
            "a: 500..1500",
            "b: 500..1000",
            "*: 500..1000",
            "Hw: 500..1000",
            "b: 1000..1500",
            "*: 1000..1500",
            "Qw: 1000..1500",
        ]
    );
}

#[test]
fn a_shared_waveform_heard_in_a_reset_is_not_reset() {
    // The `reset` restarts `x`'s reading, not the shared waveform, so each
    // note lights while it plays. A copy is restarted, and holds `$220`.
    let v = voice(
        "share($220 * Qw \\ $330 * Qw | unseq(), fn(x) => reset(time - 0.25, x) * Hw | unseq())",
    );
    assert_eq!(segments(&v), ["0+500", "500+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "reset: 0..1000",
            "time: 0..1000",
            "-: 0..1000",
            "0.25: 0..1000",
            "x: 0..1000",
            "*: 0..1000",
            "Hw: 0..1000",
        ]
    );
}

#[test]
fn refs_in_both_branches_of_an_alt_light_while_they_read() {
    let v = voice(
        "share($220 * Qw \\ $330 * Qw | unseq(), \
         fn(x) => alt(0.25 - time, x * Ew | unseq(), 0 * Qw \\ x) * Hw | unseq())",
    );
    assert_eq!(segments(&v), ["0+250", "500+0", "500+0", "500+500"]);
    // `alt` plays `x * Ew` until 250 and `0 * Qw \\ x` after. The first
    // `x * Ew` ends at 250, cutting what it hears of `$220`, and nothing
    // sounds again until the second `x` reads `$330` from 500.
    assert_eq!(
        lit(&v),
        [
            "$220: 0..250 500..500",
            "220: 0..250 500..500",
            "*: 0..250 500..500",
            "Qw: 0..250 500..500",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "alt: 0..250 500..1000",
            "0.25: 0..250 500..1000",
            "-: 0..250 500..1000",
            "time: 0..250 500..1000",
            "x: 0..250 500..500",
            "*: 0..250 500..500",
            "Ew: 0..250 500..500",
            "0: ",
            "*: ",
            "Qw: ",
            "x: 500..1000",
            "*: 0..250 500..1000",
            "Hw: 0..250 500..1000",
        ]
    );
}

#[test]
fn a_part_of_a_shared_waveform_cut_before_it_plays_stays_dark() {
    // `(x & y) * Qw` ends when `$330` would start, and `y` has ended by the
    // time the second `y` reads it.
    let v = voice(
        "share($220 * Qw \\ $330 * Qw | unseq(), \
         fn(x) => share($440 * Ew | unseq(), fn(y) => (x & y) * Qw \\ y))",
    );
    assert_eq!(segments(&v), ["0+500", "500+0", "0+250", "500+0"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "$330: 500..500",
            "330: 500..500",
            "*: 500..500",
            "Qw: 500..500",
            "$440: 0..250 500..500",
            "440: 0..250 500..500",
            "*: 0..250 500..500",
            "Ew: 0..250 500..500",
            "x: 0..500",
            "&: 0..500",
            "y: 0..250",
            "*: 0..500",
            "Qw: 0..500",
            "y: 500..500",
        ]
    );
}

#[test]
fn a_fin_around_overlapping_refs_cuts_what_both_read() {
    let v = voice(
        "share($220 * Qw \\ $330 * Qw | unseq(), fn(x) => (x & (Ew \\ x)) | fin(time - 0.625))",
    );
    assert_eq!(segments(&v), ["0+500", "500+125", "250+250", "500+125"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "$330: 500..625",
            "330: 500..625",
            "*: 500..625",
            "Qw: 500..625",
            "x: 0..625",
            "&: 0..625",
            "Ew: ",
            "x: 250..625",
        ]
    );
}

#[test]
fn a_note_of_unknown_length_lights_no_longer_than_what_stops_it() {
    // `time * time - 1` stands in for a length that can't be measured, as a
    // slider's can't. The product ends by the end of `Qw`.
    let v = voice("($220 | fin(time * time - 1)) * Qw");
    assert_eq!(segments(&v), ["0+500"]);
    assert_eq!(
        lit(&v),
        ["$220: 0..500", "220: 0..500", "*: 0..500", "Qw: 0..500"]
    );
}

#[test]
fn a_ref_to_a_shared_waveform_of_unknown_length_lights_no_longer_than_what_stops_it() {
    let v = voice(
        "share(($220 * Qw \\ $330 * Qw | unseq()) * (1 | fin(time * time - 1)), \
         fn(x) => x * Qw \\ x * Qw | unseq())",
    );
    assert_eq!(segments(&v), ["0+500", "500+0", "500+0", "500+500"]);
    assert_eq!(
        lit(&v),
        [
            "$220: 0..500",
            "220: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "$330: 500..1000",
            "330: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
            "*: 0..1000",
            "1: 0..1000",
            "x: 0..500",
            "*: 0..500",
            "Qw: 0..500",
            "x: 500..1000",
            "*: 500..1000",
            "Qw: 500..1000",
        ]
    );
}
