//! Checks of `share`, from source text to generated samples, using the real
//! std library.
#![cfg(feature = "native")]

use std::path::PathBuf;

use tuun::environment::Environment;
use tuun::generator::{self, Generator};
use tuun::programs::{Evaluated, ProgramSet};

const SAMPLE_RATE: u32 = 8000;

/// Returns the samples of the program `_ = program`, after `open std;` and
/// `bindings`, up to `seconds` long.
fn render(bindings: &str, program: &str, seconds: f32) -> Vec<f32> {
    let source = format!(
        "open std;\n{}\n#{{level_db=0}}\n_ = {};\n",
        bindings, program
    );
    let (set, _) = ProgramSet::from_source(source, PathBuf::new()).expect("source should parse");
    let index = set
        .programs()
        .iter()
        .rposition(|p| !p.text().trim().is_empty())
        .expect("a program");
    let library_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
    let environment = Environment::new(SAMPLE_RATE, 90, library_root);
    let waveform = match environment.evaluate_program(&set.evaluation_input(index)) {
        Ok(Evaluated::Waveform(waveform, _)) => waveform,
        Err(diagnostics) => panic!("{}: {:?}", program, diagnostics),
        Ok(_) => panic!("{}: not a waveform", program),
    };
    let mut out = vec![0.0; (seconds * SAMPLE_RATE as f32) as usize];
    let len =
        Generator::new(SAMPLE_RATE).generate(&mut generator::initialize_state(waveform), &mut out);
    out.truncate(len);
    out
}

#[test]
fn every_use_of_a_shared_noise_reads_the_same_stream() {
    let copied = render("", "let n = noise in n - n", 0.5);
    assert!(copied.iter().any(|&x| x != 0.0));
    let shared = render("", "share(noise, fn(n) => n - n)", 0.5);
    assert_eq!(shared.len(), 4000);
    assert!(shared.iter().all(|&x| x == 0.0));
}

#[test]
fn a_shared_lfo_keeps_running_inside_a_reset() {
    // A 4 s LFO chooses between two ramps that restart every 0.5 s. Shared,
    // the LFO isn't restarted with the ramps, as when the `reset` is moved
    // below the `alt` by hand.
    let bindings = "lfo = phase(0.25, 0.5) - 0.5;\n\
                    t = phase(2, 0.5) - 0.5;\n\
                    up = 0.5 + 0.1*time;\n\
                    down = -0.5 - 0.1*time;";
    let shared = render(
        bindings,
        "share(lfo, fn(l) => reset(t, alt(l, up, down)))",
        6.5,
    );
    let distributed = render(bindings, "alt(lfo, reset(t, up), reset(t, down))", 6.5);
    assert_eq!(shared.len(), 52000);
    assert_eq!(shared, distributed);
    // Copied, the LFO restarts with the ramps and never turns negative.
    let copied = render(bindings, "reset(t, alt(lfo, up, down))", 6.5);
    assert!(copied.iter().all(|&x| x > 0.0));
}
