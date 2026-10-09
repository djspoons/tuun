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
        let waveform = match environment.evaluate_program(&set, 0) {
            Ok(Evaluated::Waveform { waveform, .. }) => waveform,
            Err(diagnostics) => panic!("invalid: {:?}", diagnostics),
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
