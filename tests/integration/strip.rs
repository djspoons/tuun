//! Checks the measure strip's marks, from source text through evaluation, the
//! player and the tracker's waveform starts, using the real std library.
#![cfg(feature = "native")]

use std::collections::HashMap;

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use sdl2::audio::AudioCallback;

use tuun::environment::Environment;
use tuun::ids::{MarkId, WaveformId};
use tuun::player::{PlayedMarks, Player, Start};
use tuun::processing;
use tuun::programs::ProgramSet;
use tuun::renderer::{self, StripMark};
use tuun::tracker::{self, Status, Tracker};

/// Samples per second: with `TEMPO`, a quarter note (`Q`) is 500 ms.
const SAMPLE_RATE: u32 = 8000;
const TEMPO: u32 = 120;

/// Evaluates every program in `source`, plays each playable one at `start`,
/// and lets the tracker call back once.
///
/// Returns the evaluated programs, the tracker's status, and the played marks
/// of each waveform played.
fn play(
    source: &str,
    start: Instant,
) -> (
    ProgramSet,
    Status<WaveformId>,
    HashMap<WaveformId, PlayedMarks>,
) {
    let library_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lib/v0");
    let environment = Environment::new(SAMPLE_RATE, TEMPO, library_root);
    let (mut set, _) =
        ProgramSet::from_source(source.to_string(), PathBuf::new()).expect("source should parse");
    let (command_sender, command_receiver) = mpsc::channel();
    let (status_sender, status_receiver) = mpsc::channel();
    let (sample_producer, _samples) = tracker::sample_rings(1024, 4);
    let mut tracker: Tracker<WaveformId, MarkId> = Tracker::new(
        SAMPLE_RATE,
        PathBuf::new(),
        String::new(),
        command_receiver,
        status_sender,
        sample_producer,
        Duration::ZERO,
    );
    let player = Player::new(TEMPO, 4, command_sender);
    let no_status = Status {
        buffer_start: Instant::now(),
        start_times: Vec::new(),
        tracker_load: None,
        allocations: None,
    };
    let mut waveform_marks = HashMap::new();
    for index in 0..set.programs().len() {
        if set.programs()[index].is_empty() {
            continue;
        }
        let input = set.evaluation_input(index);
        let outcome = processing::process(&environment, &input, false);
        if set.record(index, input.text.into(), outcome).is_ok()
            && set.programs()[index].is_playable()
        {
            player.play_program(
                &set,
                index,
                &no_status,
                Start::At(start),
                None,
                &mut waveform_marks,
            );
        }
    }
    tracker.callback(&mut [0.0; 64]);
    let status = status_receiver
        .try_recv()
        .expect("the tracker sends a status");
    (set, status, waveform_marks)
}

/// Evaluates every program in `source`, plays each playable one at `start`, and
/// lets the tracker call back once.
///
/// Returns the strip's marks as the renderer draws them.
fn play_all(source: &str, start: Instant) -> Vec<StripMark> {
    let (set, status, waveform_marks) = play(source, start);
    renderer::strip_marks(
        &status.start_times,
        &waveform_marks,
        &set,
        TEMPO,
        SAMPLE_RATE,
    )
}

/// Returns the strip's marks with their starts measured from `start`, sorted.
fn from(start: Instant, strip: &[StripMark]) -> Vec<(usize, Duration, Duration)> {
    let mut marks: Vec<_> = strip
        .iter()
        .map(|mark| (mark.program, mark.start - start, mark.duration))
        .collect();
    marks.sort();
    marks
}

/// The strip draws two marks, a quarter note long, at 0 and 1.5 quarter notes.
#[test]
fn t1_draws_two_marks() {
    let start = Instant::now() + Duration::from_secs(1);
    let source = "open std;\n\
                  #{level_db=0}\n\
                  _ = <map(fn (x) => $x * Qw | unseq() | seq(time - Q*1.5), [220, 330])>;\n";
    let strip = play_all(source, start);
    let q = Duration::from_millis(500);
    assert_eq!(
        from(start, &strip),
        vec![(0, Duration::ZERO, q), (0, q * 3 / 2, q)]
    );
}

/// Each note lights `$x`, `*`, `Qw` and its own frequency while it sounds, and
/// nothing lights between the notes.
#[test]
fn t1_lights_each_note_while_it_sounds() {
    let start = Instant::now() + Duration::from_secs(1);
    let source = "open std;\n\
                  #{level_db=0}\n\
                  _ = <map(fn (x) => $x * Qw | unseq() | seq(time - Q*1.5), [220, 330])>;\n";
    let (set, status, waveform_marks) = play(source, start);
    let text = set.programs()[0].text();
    let lit = |at: Duration| -> Vec<&str> {
        renderer::lit_ranges(
            0,
            text,
            &status.start_times,
            &waveform_marks,
            SAMPLE_RATE,
            start + at,
        )
        .into_iter()
        .map(|range| &text[range])
        .collect()
    };
    let q = Duration::from_millis(500);
    assert_eq!(lit(q / 2), vec!["$x", "*", "Qw", "220"]);
    assert_eq!(lit(q * 5 / 4), Vec::<&str>::new());
    assert_eq!(lit(q * 2), vec!["$x", "*", "Qw", "330"]);
    assert_eq!(lit(q * 3), Vec::<&str>::new());
}
