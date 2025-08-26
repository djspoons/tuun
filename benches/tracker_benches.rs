use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use tuun::builtins;
use tuun::parser;

use tuun::renderer;
use tuun::tracker;

fn bench_filter(c: &mut Criterion) {
    use tracker::Waveform::{Filter, Fixed, Time};

    let generator = tracker::Generator::new(44100);
    let w1 = Filter {
        waveform: Box::new(Time),
        feed_forward: Box::new(Fixed(vec![0.5])),
        feedback: Box::new(Fixed(vec![-0.5])),
        state: (),
    };
    let w1 = tracker::initialize_state(w1);
    c.bench_function("filter_1_1", |b| {
        b.iter(|| {
            for i in 0..43 {
                generator.generate(&w1, black_box(i * 1024), 1024);
            }
        });
    });

    let w2 = Filter {
        waveform: Box::new(Time),
        feed_forward: Box::new(Fixed(vec![0.00107949, 0.00323847, 0.00323847, 0.00107949])),
        feedback: Box::new(Fixed(vec![-2.56103158, 2.2132402, -0.64357271])),
        state: (),
    };
    let w2 = tracker::initialize_state(w2);
    c.bench_function("filter_4_3", |b| {
        b.iter(|| {
            for i in 0..43 {
                generator.generate(&w2, black_box(i * 1024), 1024);
            }
        });
    });
}

fn bench_marks(c: &mut Criterion) {
    let generator = tracker::Generator::new(44100);
    let mut ws = Vec::new();
    for _ in 0..40 {
        ws.push(parser::Expr::Waveform(renderer::beats_waveform(120, 4)));
    }
    let w = match builtins::sequence(vec![parser::Expr::List(ws)]) {
        parser::Expr::Waveform(w) => w,
        _ => panic!("Expected waveform"),
    };
    let w = tracker::initialize_state(w);
    c.bench_function("marks_4_40", |b| {
        b.iter(|| {
            for i in 0..3438 {
                // Approx. the length of the waveform
                generator.generate(&w, black_box(i * 1024), 1024);
            }
        });
    });
}

criterion_group!(benches, bench_filter, bench_marks);
criterion_main!(benches);
