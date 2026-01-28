use criterion::{Criterion, criterion_group, criterion_main};
use tuun::builtins;
use tuun::parser;

use tuun::generator;
use tuun::renderer;
use tuun::waveform;

fn bench_filter(c: &mut Criterion) {
    use waveform::Waveform::{Filter, Fixed, Time};

    c.bench_function("filter_1_1", |b| {
        b.iter(|| {
            let generator = generator::Generator::new(44100);
            let w1 = Filter {
                waveform: Box::new(Time(())),
                feed_forward: Box::new(Fixed(vec![0.5], ())),
                feedback: Box::new(Fixed(vec![-0.5], ())),
                state: (),
            };
            let mut w1 = generator::initialize_state(w1);
            for _ in 0..43 {
                let (tmp, _) = generator.generate(w1, 1024);
                w1 = tmp;
            }
        });
    });

    c.bench_function("filter_4_3", |b| {
        b.iter(|| {
            let generator = generator::Generator::new(44100);
            let w2 = Filter {
                waveform: Box::new(Time(())),
                feed_forward: Box::new(Fixed(
                    vec![0.00107949, 0.00323847, 0.00323847, 0.00107949],
                    (),
                )),
                feedback: Box::new(Fixed(vec![-2.56103158, 2.2132402, -0.64357271], ())),
                state: (),
            };
            let mut w2 = generator::initialize_state(w2);
            for _ in 0..43 {
                let (tmp, _) = generator.generate(w2, 1024);
                w2 = tmp;
            }
        });
    });
}

fn bench_marks(c: &mut Criterion) {
    c.bench_function("marks_4_40", |b| {
        b.iter(|| {
            let generator = generator::Generator::new(44100);
            let mut ws = Vec::new();
            for _ in 0..40 {
                ws.push(parser::Expr::Waveform(renderer::beats_waveform(120, 4)));
            }
            let w = match builtins::sequence(vec![parser::Expr::List(ws)]) {
                parser::Expr::Waveform(w) => w,
                _ => panic!("Expected waveform"),
            };
            let mut w = generator::initialize_state(w);
            for _ in 0..3438 {
                // Approx. the length of the waveform
                let (tmp, _) = generator.generate(w, 1024);
                w = tmp;
            }
        });
    });
}

fn bench_large(c: &mut Criterion) {
    let mut context = Vec::new();
    builtins::add_prelude(&mut context);
    match parser::parse_context(
        r#"
    pi = 3.14159265,
    $ = fn(freq) => sin((2 * pi) ~. freq ~. time),
    triangle = fn(freq) => let t = $freq, slope = 4*freq, a = time ~. slope ~+ -1, b = time ~. -slope ~+ 3 in alt(t, res(t, a), res(t, b)),
    linear = fn(initial, slope) => initial ~+ (time ~. slope),
    Rw = fn(dur, level) => linear(level, -level / dur) | fin(time ~+ -dur) | seq(dur),
    R = fn(dur, level) => fn(w) => w ~. Rw(dur, level),"#,
    ) {
        Ok(bindings) => {
            for (pattern, expr) in bindings {
                //println!("Parsed binding: {:?} = {:}", &pattern, &expr);
                match parser::simplify(&context, expr) {
                    Ok(expr) => {
                        //println!("Simplified to: {:}", &expr);
                        match parser::extend_context(&mut context, &pattern, &expr) {
                            Ok(()) => {}
                            Err(e) => panic!("Failed to extend context: {:?}", e),
                        }
                    }
                    Err(e) => panic!("Simplify failed: {:?}", e),
                }
            }
        }
        Err(e) => panic!("Failed to parse context: {:?}", e),
    }

    c.bench_function("large_440", |b| {
        b.iter(|| {
            let program = "triangle(55) ~+ (noise ~. 0.2) | R(1.0, 1.0) | seq(1) | mark(2)";
            match parser::parse_program(program) {
                Ok(expr) => {
                    println!("Parser returned: {:}", &expr);
                    match parser::simplify(&context, expr) {
                        Ok(expr) => {
                            println!("Simplify returned: {:}", &expr);
                            if let parser::Expr::Waveform(waveform) = expr {
                                let generator = generator::Generator::new(44100);
                                let mut w = generator::initialize_state(waveform);

                                for _ in 0..43 {
                                    let (tmp, _) = generator.generate(w, 1024);
                                    w = tmp;
                                }
                            } else {
                                panic!("Expected waveform");
                            }
                        }
                        Err(e) => panic!("Simplify failed: {:?}", e),
                    }
                }
                _ => panic!("Parse failed"),
            }
        });
    });
}

criterion_group!(benches, bench_filter, bench_marks, bench_large);
criterion_main!(benches);
