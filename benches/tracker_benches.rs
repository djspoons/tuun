use criterion::{Criterion, criterion_group, criterion_main};

use tuun::builtins;
use tuun::generator;
use tuun::parser;
use tuun::play_helper;
use tuun::renderer;
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
                feedback: vec![Const(-2.56103158), Const(2.2132402), Const(-0.64357271)],
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
            let mut context = vec![];
            builtins::add_prelude(&mut context);
            context.push((
                "mark".to_string(),
                parser::Expr::BuiltIn {
                    name: "mark".to_string(),
                    function: parser::BuiltInFn(std::rc::Rc::new(renderer::mark)),
                },
            ));
            const SAMPLE_RATE: i32 = 44100;
            let mut generator = generator::Generator::new(SAMPLE_RATE);
            let mut ws = Vec::new();
            for _ in 0..40 {
                ws.push(play_helper::beats_waveform(120, 4, &context));
            }
            let w = ws
                .into_iter()
                .reduce(|result, w| waveform::Waveform::Append(Box::new(result), Box::new(w)))
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
    type Expr = parser::Expr<u32>;
    let mut context: Vec<(String, Expr)> = Vec::new();
    builtins::add_prelude(&mut context);
    match parser::parse_context(
        r#"
    pi = 3.14159265,
    $ = fn(freq_hz) => sine(2*pi * freq_hz, 0),
    triangle = fn(freq_hz) => let t = $freq_hz, slope = 4 * freq_hz, a = time * slope - 1, b = time * -slope + 3 in alt(t, reset(t, a), reset(t, b)),
    linear = fn(initial, slope) => initial + (time * slope),
    Rw = fn(dur, level) => linear(level, -level / dur) | fin(time - dur),
    R = fn(dur, level) => fn(w) => w * Rw(dur, level),"#,
    ) {
        Ok(bindings) => {
            for (pattern, expr) in bindings {
                //println!("Parsed binding: {:?} = {:}", &pattern, &expr);
                match parser::evaluate(&context, expr) {
                    Ok(expr) => {
                        //println!("Evaluated to: {:}", &expr);
                        match parser::extend_context(&mut context, &pattern, &expr) {
                            Ok(()) => {}
                            Err(e) => panic!("Failed to extend context: {:?}", e),
                        }
                    }
                    Err(e) => panic!("Evaluate failed: {:?}", e),
                }
            }
        }
        Err(e) => panic!("Failed to parse context: {:?}", e),
    }

    c.bench_function("large_440", |b| {
        b.iter(|| {
            let program = "triangle(55) + (noise * 0.2) | R(1.0, 1.0)";
            match parser::parse_program(program) {
                Ok(expr) => {
                    //println!("Parser returned: {:}", &expr);
                    match parser::evaluate(&context, expr) {
                        Ok(expr) => {
                            //println!("Evaluate returned: {:}", &expr);
                            if let parser::Expr::Waveform(waveform) = expr {
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
                    }
                }
                _ => panic!("Parse failed"),
            }
        });
    });
}

criterion_group!(benches, bench_filter, bench_marks, bench_large);
criterion_main!(benches);
