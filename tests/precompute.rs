//! Checks that precomputing a program once, when it is evaluated, plays the
//! same waveform as precomputing each play of it.
#![cfg(feature = "native")]

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Instant;

use tuun::environment::Environment;
use tuun::generator::Generator;
use tuun::ids::{MarkId, WaveformId};
use tuun::player::{Player, Start};
use tuun::processing;
use tuun::programs::ProgramSet;
use tuun::tracker;
use tuun::waveform::Waveform;

const SAMPLE_RATE: u32 = 8000;

/// Returns the waveform `player` sends for playing program 0 of `set`.
fn played(
    player: &Player,
    commands: &mpsc::Receiver<tracker::Command<WaveformId, MarkId>>,
    set: &ProgramSet,
) -> Waveform<MarkId> {
    let status = tracker::Status {
        buffer_start: Instant::now(),
        marks: Vec::new(),
        tracker_load: None,
        allocations: None,
    };
    player
        .play_program(set, 0, &status, Start::Now, None)
        .expect("the program plays");
    match commands.try_recv() {
        Ok(tracker::Command::Play { waveform, .. }) => waveform,
        _ => panic!("expected a Play command"),
    }
}

/// Returns the set for `source` with program 0 processed, precomputed when
/// `precompute` is set.
fn processed(environment: &Environment, source: &str, precompute: bool) -> ProgramSet {
    let (mut set, warning) = ProgramSet::from_source(source.to_string(), PathBuf::new())
        .expect("test source should parse");
    assert_eq!(warning, "");
    let outcome = processing::process(environment, &set.evaluation_input(0), precompute);
    set.record(0, outcome).expect("the program evaluates");
    set
}

/// The slider and level marks that a play adds to a precomputed program act on
/// the same nodes as when the played form is precomputed, so the two give the
/// same tree, before and after the slider moves.
#[test]
fn precomputing_at_evaluation_matches_precomputing_each_play() {
    let environment = Environment::new(SAMPLE_RATE, 90, PathBuf::new());
    // A finite part that precomputes to samples, then one the slider mark
    // keeps dynamic.
    let source = "#{sliders=[\"vol:0.5:0:1\"], level_db=-6}\n\
                  _ = (time | fin(time - 0.25) | seq(time - 0.25)) \\ (vol | fin(time - 0.25));\n";
    let mut at_evaluation = processed(&environment, source, true);
    let mut per_play = processed(&environment, source, false);
    let (sender, commands) = mpsc::channel();
    let player = Player::new(90, 4, sender);

    for (normalized, value) in [(0.5, 0.5), (1.0, 1.0)] {
        for set in [&mut at_evaluation, &mut per_play] {
            set.program_mut(0)
                .unwrap()
                .set_slider_normalized(0, normalized)
                .expect("the program has a slider");
        }
        let precomputed_once = played(&player, &commands, &at_evaluation);
        let precomputed_per_play =
            Generator::new(SAMPLE_RATE).precompute(played(&player, &commands, &per_play));
        assert_eq!(precomputed_once, precomputed_per_play);

        let tree = format!("{:?}", precomputed_once);
        assert!(tree.contains("Fixed("), "nothing was precomputed: {}", tree);
        // The slider still reaches the tracker as a mark holding its
        // current value.
        let slider = Waveform::<MarkId>::Marked {
            id: MarkId::Slider {
                program: 0,
                label: "vol".to_string(),
            },
            waveform: Box::new(Waveform::Const(value)),
        };
        assert!(
            tree.contains(&format!("{:?}", slider)),
            "no {:?} in {}",
            slider,
            tree
        );
    }
}
