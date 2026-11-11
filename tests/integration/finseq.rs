//! Checks of the real std library's `finseq` envelopes, from source text to
//! generated samples.
#![cfg(feature = "native")]

use std::path::PathBuf;

use tuun::environment::Environment;
use tuun::generator::{self, Generator};
use tuun::programs::{Evaluated, ProgramSet};

#[test]
fn adsr_segments_meet_without_gaps_or_overlaps() {
    // This tests the rounding in \ implied by its associativity. When
    // right-associated, the second argument's own time starts at that offset.
    // When it is a seq, the result is a seq whose offset is the sum of the two
    // offsets, measured from the start of the first argument. That sum can
    // round to a different sample than the second argument's own offset does
    // from its own start, so a waveform that ends at its offset (e.g. `w |
    // fin(time - d) | seq(time - d)`) is followed sample-exactly by `a \ (b \
    // c)` but not always by `(a \ b) \ c`. The property depends on how the
    // parser nests `\`, how `\` delays each segment, and how the generator ends
    // each `fin`.
    let sample_rate = 44100;
    // (arguments, length of each segment in samples)
    let cases = [
        // Each duration is a whole number of samples.
        ("0.11, 0.01, 0.05, 0.65, 0.02", [441, 2205, 1323, 882]),
        // Each duration is 445.41 samples.
        ("0.0404, 0.0101, 0.0101, 0.65, 0.0101", [446; 4]),
    ];
    let library_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
    for (arguments, segments) in cases {
        let source = format!(
            "open std;\nuse env.finseq;\n#{{level_db=0}}\n_ = 1 | finseq.ADSR({});\n",
            arguments
        );
        let (set, warning) =
            ProgramSet::from_source(source, PathBuf::new()).expect("test source should parse");
        assert_eq!(warning, "");
        let environment = Environment::new(sample_rate, 90, library_root.clone());
        let waveform = match environment.evaluate_program(&set.evaluation_input(0)) {
            Ok(Evaluated::Waveform(waveform, _)) => waveform,
            Err(diagnostics) => panic!("invalid: {:?}", diagnostics),
            Ok(Evaluated::Sequence(_)) => panic!("classified as a sequence"),
            Ok(Evaluated::KeysInstrument(_)) => panic!("classified as keys"),
        };
        let mut out = vec![0.0; 10000];
        let len = Generator::new(sample_rate)
            .generate(&mut generator::initialize_state(waveform), &mut out);
        assert_eq!(len, segments.iter().sum::<usize>(), "ADSR({})", arguments);
        // Only the attack's first sample is silent, and since every segment
        // stays within [0, 1], a sample above 1 is an overlap.
        for (i, &x) in out[..len].iter().enumerate().skip(1) {
            assert!(x > 0.0 && x <= 1.0, "ADSR({}) [{}] = {}", arguments, i, x);
        }
    }
}

/// Returns the samples of the program `_ = program` after `open std;`, at
/// `sample_rate`, or the messages of the diagnostics evaluating it reports.
fn render(sample_rate: u32, program: &str) -> Result<Vec<f32>, Vec<String>> {
    let source = format!("open std;\n#{{level_db=0}}\n_ = {};\n", program);
    let (set, _) = ProgramSet::from_source(source, PathBuf::new()).expect("source should parse");
    let library_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
    let environment = Environment::new(sample_rate, 90, library_root);
    let waveform = match environment.evaluate_program(&set.evaluation_input(0)) {
        Ok(Evaluated::Waveform(waveform, _)) => waveform,
        Err(diagnostics) => return Err(diagnostics.into_iter().map(|d| d.message).collect()),
        Ok(_) => panic!("{}: not a waveform", program),
    };
    let mut out = vec![0.0; 10000];
    let len =
        Generator::new(sample_rate).generate(&mut generator::initialize_state(waveform), &mut out);
    out.truncate(len);
    Ok(out)
}

#[test]
fn left_nested_seqs_with_linear_offsets_meet_without_gaps_or_overlaps() {
    // Each offset reaches 0 at 0.25 s, though only the last is written
    // `time - 0.25`. Each segment lasts 0.25 s.
    let expected: Vec<f32> = [1.0, 2.0, 3.0]
        .iter()
        .flat_map(|&x| std::iter::repeat_n(x, 2000))
        .collect();
    for program in [
        "((1 | fin(time - 0.25) | seq(time * 2 - 0.5)) \\ \
          (2 | fin(time - 0.25) | seq(time - 0.25))) \\ (3 | fin(time - 0.25))",
        "((1 | fin(time - 0.25) | seq(time - 0.25)) \\ \
          (2 | fin(time - 0.25) | seq(time * 2 - 0.5))) \\ (3 | fin(time - 0.25))",
        "((1 | fin(time - 0.25) | seq((time - 0.5) + 0.25)) \\ \
          (2 | fin(time - 0.25) | seq(time - 0.25))) \\ (3 | fin(time - 0.25))",
    ] {
        assert_eq!(render(8000, program), Ok(expected.clone()), "{}", program);
    }
}

#[test]
fn left_nested_seqs_with_noisy_offsets_are_evaluation_errors() {
    // A noisy offset has no fixed root to sum, so the combined seq could start
    // its successor at a different time than the first seq's own offset ends.
    for program in [
        "((1 | fin(time - 0.25) | seq(time - 0.25 + 0.01 * noise)) \\ \
          (2 | fin(time - 0.25) | seq(time - 0.25))) \\ (3 | fin(time - 0.25))",
        "((1 | fin(time - 0.25) | seq(time + (0.01 * noise - 0.25))) \\ \
          (2 | fin(time - 0.25) | seq(time - 0.25))) \\ (3 | fin(time - 0.25))",
    ] {
        let messages = render(8000, program).expect_err(program);
        assert!(
            messages
                .iter()
                .any(|m| m.contains("Cannot combine seqs") && !m.contains("internal error")),
            "{}: {:?}",
            program,
            messages
        );
    }
}
