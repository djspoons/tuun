use criterion::{Criterion, criterion_group, criterion_main};

use std::sync::mpsc;

use tuun::builtins;
use tuun::eval;
use tuun::evaluator;
use tuun::expr;
use tuun::generator;
use tuun::parser;
use tuun::player;
use tuun::waveform;

fn bench_filter(c: &mut Criterion) {
    use waveform::Operator::{Add, Multiply};
    use waveform::Waveform::{BinaryPointOp, Const, Filter, Time};
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
                feed_forward: vec![BinaryPointOp(
                    Add,
                    Box::new(BinaryPointOp(
                        Multiply,
                        Box::new(Time(())),
                        Box::new(Const(-0.5)),
                    )),
                    Box::new(Const(0.5)),
                )],
                feedback: vec![BinaryPointOp(
                    Add,
                    Box::new(BinaryPointOp(
                        Multiply,
                        Box::new(Time(())),
                        Box::new(Const(0.5)),
                    )),
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
            let evaluator =
                evaluator::Evaluator::new(SAMPLE_RATE, 120, std::path::PathBuf::from("./lib"));
            let player = player::Player::new(120, 4, tx.clone(), tx);
            let mut generator = generator::Generator::new(SAMPLE_RATE);
            let mut ws = Vec::new();
            for _ in 0..40 {
                ws.push(player.beats_waveform(&evaluator));
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
    match parser::parse_module::<u32>(
        r#"
    pi = 3.14159265;
    $ = fn(freq_hz) => sine(2*pi * freq_hz, 0);
    triangle = fn(freq_hz) => let t = $freq_hz, slope = 4 * freq_hz, a = time * slope - 1, b = time * -slope + 3 in alt(t, reset(t, a), reset(t, b));
    linear = fn(initial, slope) => initial + (time * slope);
    Rw = fn(dur, level) => linear(level, -level / dur) | fin(time - dur);
    R = fn(dur, level) => fn(w) => w * Rw(dur, level);"#,
    ) {
        Ok((parsed, _errors)) => bindings.extend(parsed),
        Err(e) => panic!("Failed to parse context: {:?}", e),
    }

    let resolve = |_: &[String]| {
        Err(expr::Error::new(
            "didn't expect to resolve in bench_large".to_string(),
        ))
    };

    c.bench_function("large_440", |b| {
        b.iter(|| {
            let program = "triangle(55) + (noise * 0.2) | R(1.0, 1.0)";
            match parser::parse_program(program) {
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

criterion_group!(benches, bench_filter, bench_marks, bench_large);
criterion_main!(benches);
