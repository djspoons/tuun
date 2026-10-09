use criterion::{Criterion, criterion_group, criterion_main};

use std::sync::mpsc;

use tuun::builtins;
use tuun::diagnostics;
use tuun::environment;
use tuun::eval;
use tuun::expr;
use tuun::generator;
use tuun::ids::MarkId;
use tuun::optimizer;
use tuun::parser;
use tuun::player;
use tuun::waveform;

fn bench_filter(c: &mut Criterion) {
    use waveform::BinaryOperator::{Add, Multiply};
    use waveform::Waveform::{BinaryOp, Const, Filter, Time};
    type Waveform = waveform::Waveform<u32>;

    c.bench_function("filter_1_1", |b| {
        b.iter(|| {
            let mut generator = generator::Generator::new(44100);
            let w1: Waveform = Filter {
                waveform: Box::new(Time(())),
                feed_forward: vec![Const(0.5)],
                feedback: vec![Const(-0.5)],
                state: (),
            };
            let mut w1 = generator::initialize_state(w1);
            let mut out = vec![0.0; 1024];
            for _ in 0..43 {
                let _ = generator.generate(&mut w1, &mut out);
            }
        });
    });

    c.bench_function("filter_1_1_linear", |b| {
        b.iter(|| {
            let mut generator = generator::Generator::new(44100);
            let w1: Waveform = Filter {
                waveform: Box::new(Time(())),
                feed_forward: vec![BinaryOp(
                    Add,
                    Box::new(BinaryOp(
                        Multiply,
                        Box::new(Time(())),
                        Box::new(Const(-0.5)),
                    )),
                    Box::new(Const(0.5)),
                )],
                feedback: vec![BinaryOp(
                    Add,
                    Box::new(BinaryOp(Multiply, Box::new(Time(())), Box::new(Const(0.5)))),
                    Box::new(Const(-0.5)),
                )],
                state: (),
            };
            let mut w1 = generator::initialize_state(w1);
            let mut out = vec![0.0; 1024];
            for _ in 0..43 {
                let _ = generator.generate(&mut w1, &mut out);
            }
        });
    });

    c.bench_function("filter_4_3", |b| {
        b.iter(|| {
            let mut generator = generator::Generator::new(44100);
            let w2: Waveform = Filter {
                waveform: Box::new(Time(())),
                feed_forward: vec![
                    Const(0.00107949),
                    Const(0.00323847),
                    Const(0.00323847),
                    Const(0.00107949),
                ],
                feedback: vec![Const(-2.561_031_6), Const(2.2132402), Const(-0.643_572_7)],
                state: (),
            };
            let mut w2 = generator::initialize_state(w2);
            let mut out = vec![0.0; 1024];
            for _ in 0..43 {
                let _ = generator.generate(&mut w2, &mut out);
            }
        });
    });
}

fn bench_marks(c: &mut Criterion) {
    c.bench_function("marks_4_40", |b| {
        b.iter(|| {
            const SAMPLE_RATE: u32 = 44100;
            let (tx, _rx) = mpsc::channel();
            let environment =
                environment::Environment::new(SAMPLE_RATE, 120, std::path::PathBuf::from("./lib"));
            let player = player::Player::new(120, 4, tx.clone(), tx);
            let mut generator = generator::Generator::new(SAMPLE_RATE);
            let mut ws = Vec::new();
            for _ in 0..40 {
                ws.push(player.beats_waveform(&environment));
            }
            let w = ws
                .into_iter()
                .reduce(|result, w| waveform::Waveform::Append(Box::new(result), Box::new(w), ()))
                .unwrap();
            let mut w = generator::initialize_state(w);
            let mut out = vec![0.0; 1024];
            for _ in 0..3438 {
                // Approx. the length of the waveform
                let _ = generator.generate(&mut w, &mut out);
            }
        });
    });
}

fn bench_large(c: &mut Criterion) {
    let mut bindings: Vec<expr::SourceBinding<u32>> = Vec::new();
    builtins::add_bindings(&mut bindings);
    match parser::parse_module::<u32, _>(
        r#"
    pi = 3.14159265;
    $ = fn(freq_hz) => sin(2*pi * phase(freq_hz, 0));
    triangle = fn(freq_hz) => let t = $freq_hz, slope = 4 * freq_hz, a = time * slope - 1, b = time * -slope + 3 in alt(t, reset(t, a), reset(t, b));
    linear = fn(initial, slope) => initial + (time * slope);
    Rw = fn(dur, level) => linear(level, -level / dur) | fin(time - dur);
    R = fn(dur, level) => fn(w) => w * Rw(dur, level);"#,
        (),
    ) {
        Ok((parsed, _errors)) => bindings.extend(parsed),
        Err(e) => panic!("Failed to parse context: {:?}", e),
    }

    let resolve = |_: &[String]| {
        Err(expr::Error::eval_here(
            "didn't expect to resolve in bench_large".to_string(),
        ))
    };

    c.bench_function("large_440", |b| {
        b.iter(|| {
            let program = "triangle(55) + (noise * 0.2) | R(1.0, 1.0)";
            match parser::parse_program(program, ()) {
                Ok(expr) => match eval::evaluate(resolve, &bindings, expr) {
                    Ok(expr) => {
                        if let expr::Expr::Waveform(waveform) = expr.expr {
                            let mut generator = generator::Generator::new(44100);
                            let mut w = generator::initialize_state(waveform);
                            let mut out = vec![0.0; 1024];

                            for _ in 0..43 {
                                let _ = generator.generate(&mut w, &mut out);
                            }
                        } else {
                            panic!("Expected waveform");
                        }
                    }
                    Err(e) => panic!("Evaluate failed: {:?}", e),
                },
                _ => panic!("Parse failed"),
            }
        });
    });
}

/// Returns the number of `Fin` nodes in `waveform`.
fn count_fins<M, S>(waveform: &waveform::Waveform<M, S>) -> usize {
    use waveform::Waveform::*;
    match waveform {
        Const(_) | Time(_) | Noise | Fixed(_, _) => 0,
        Fin { length, waveform } => 1 + count_fins(length) + count_fins(waveform),
        Append(a, b, _) | BinaryOp(_, a, b) => count_fins(a) + count_fins(b),
        Phase {
            frequency, offset, ..
        } => count_fins(frequency) + count_fins(offset),
        Filter {
            waveform,
            feed_forward,
            feedback,
            ..
        } => {
            count_fins(waveform)
                + feed_forward.iter().map(count_fins).sum::<usize>()
                + feedback.iter().map(count_fins).sum::<usize>()
        }
        UnaryOp(_, a) => count_fins(a),
        Reset {
            trigger, waveform, ..
        } => count_fins(trigger) + count_fins(waveform),
        Alt {
            trigger,
            positive_waveform,
            negative_waveform,
        } => count_fins(trigger) + count_fins(positive_waveform) + count_fins(negative_waveform),
        Marked { waveform, .. } | Captured { waveform, .. } => count_fins(waveform),
    }
}

/// Bindings for `bench_fins`: a few copied from std, plus instruments whose
/// envelopes end with `fin`.
const FINS_MODULE: &str = r#"
// Copied from std.
min = fn(x, y) => alt(x - y, y, x);
max = fn(x, y) => alt(x - y, x, y);
pi = 3.14159265;
sine = fn(ang_freq, phase_offset) =>
  sin(2 * pi * phase(ang_freq / (2 * pi), phase_offset / (2 * pi)));
$ = fn(freq_hz) => sin(2 * pi * phase(freq_hz, 0));
linear = fn(initial, slope) => initial + (time * slope);

// An ADSR envelope shaped with min and max and ended by a single fin.
min_max_adsr = fn(total_dur, attack_dur, decay_dur, sustain_level, release_dur) =>
  let
    attack = linear(0.0, 1 / attack_dur),
    decay_slope = (sustain_level - 1) / decay_dur,
    decay = linear(1 - attack_dur * decay_slope, decay_slope),
    release = min(1, (total_dur - time) / release_dur),
  in
    fn(w) => w * min(attack, max(decay, sustain_level)) * release | fin(time - total_dur);

// An ADSR envelope sequenced from four finite segments.
fin_adsr = fn(total_dur, attack_dur, decay_dur, sustain_level, release_dur) =>
  let
    decay_slope = (sustain_level - 1) / decay_dur,
    sustain_dur = total_dur - attack_dur - decay_dur - release_dur,
  in
    fn(w) => w * ((linear(0.0, 1 / attack_dur) | fin(time - attack_dur) | seq(time - attack_dur))
                  \ (linear(1.0, decay_slope) | fin(time - decay_dur) | seq(time - decay_dur))
                  \ (sustain_level | fin(time - sustain_dur) | seq(time - sustain_dur))
                  \ (linear(sustain_level, -sustain_level / release_dur) | fin(time - release_dur)));

// A phase-modulation oscillator with modulation index I and frequency ratio D.
pm = fn(I, D, freq_hz) =>
  let
    w_c = 2 * pi * freq_hz,
  in
    sine(w_c, I * sine(D / 2 * w_c, 0));

// A mix of two enveloped PM oscillators whose envelopes also drive their
// modulation indexes.
two_operator_pm = fn(dur, freq_hz) =>
  let
    envelope1 = min_max_adsr(dur, 0.01, 0.55, 0.65, 0.2),
    envelope2 = min_max_adsr(dur, 0.01, 0.33, 0.2, 0.2),
    osc1 = pm(6 | envelope1, 2, freq_hz) | envelope1,
    osc2 = pm(10 | envelope2, 5.96, freq_hz * 1.01) | envelope2,
  in
    0.73 * osc1 & (0.27 * osc2) | seq(time - dur);
"#;

/// Benchmarks a sequence of notes from each instrument, generated in
/// 1024-sample buffers until the sequence ends.
fn bench_fins(c: &mut Criterion) {
    const SAMPLE_RATE: u32 = 44100;
    const NOTES: usize = 16;
    let environment = environment::Environment::new(SAMPLE_RATE, 120, std::path::PathBuf::new());
    let mut bindings: Vec<expr::SourceBinding<MarkId, diagnostics::Source>> =
        vec![expr::Binding::Open(vec!["__prelude".to_string()]).into()];
    match parser::parse_module(FINS_MODULE, diagnostics::Source::Program) {
        Ok((parsed, errors)) => {
            assert!(
                errors.is_empty(),
                "Errors parsing FINS_MODULE: {:?}",
                errors
            );
            bindings.extend(parsed);
        }
        Err(errors) => panic!("Failed to parse FINS_MODULE: {:?}", errors),
    }
    let instruments = [
        ("fins_two_operator_pm", "two_operator_pm(0.25, {f})"),
        (
            "fins_min_max_adsr",
            "$({f}) | min_max_adsr(0.25, 0.01, 0.1, 0.6, 0.1) | seq(time - 0.25)",
        ),
        (
            "fins_fin_adsr",
            "$({f}) | fin_adsr(0.25, 0.01, 0.1, 0.6, 0.1) | seq(time - 0.25)",
        ),
    ];
    for (name, note) in instruments {
        let notes: Vec<String> = (0..NOTES)
            .map(|i| note.replace("{f}", &(220.0 * (1.0 + i as f32 / 8.0)).to_string()))
            .collect();
        let source = format!("<[{}]>", notes.join(", "));
        let waveform = match environment.evaluate_source(&source, &bindings) {
            Ok(expr) => match expr.expr {
                expr::Expr::Seq { waveform, .. } => match waveform.expr {
                    expr::Expr::Waveform(waveform) => optimizer::optimize(waveform),
                    expr => panic!("Expected waveform, got {}", expr),
                },
                expr::Expr::Waveform(waveform) => optimizer::optimize(waveform),
                expr => panic!("Expected waveform, got {}", expr),
            },
            Err(message) => panic!("Error evaluating {}: {}", name, message),
        };
        println!(
            "{}: {} Fin nodes for {} notes",
            name,
            count_fins(&waveform),
            NOTES
        );
        c.bench_function(name, |b| {
            b.iter(|| {
                let mut generator = generator::Generator::new(SAMPLE_RATE);
                let mut w = generator::initialize_state(waveform.clone());
                let mut out = vec![0.0; 1024];
                while generator.generate(&mut w, &mut out) == out.len() {}
            });
        });
    }
}

criterion_group!(benches, bench_filter, bench_marks, bench_large, bench_fins);
criterion_main!(benches);
