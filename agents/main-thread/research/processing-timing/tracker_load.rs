// Times the tracker while it plays one program started at once (as a pad
// press does), with the program precomputed at evaluation and without.
//
// Usage: tracker_load <file.tuun> <library_root> <name>...
// where each <name> is a program's display name (`A:6`, `harm_line`).
//
// Each run evaluates the program, plays it with `Start::Now` into a fresh
// tracker and calls the tracker's callback until the program's terminator
// mark is gone (at most 30 s of audio), recording the tracker load of each
// buffer: generate time over the buffer's duration. Callbacks are paced at
// the buffer period, as an audio device calls them; faster calls make the
// tracker resynchronize its clock on every buffer.
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use sdl2::audio::AudioCallback;

use tuun::actions::AppState;
use tuun::environment::Environment;
use tuun::ids::{MarkId, WaveformId};
use tuun::player::{Player, Start};
use tuun::processing;
use tuun::tracker::Tracker;

const SAMPLE_RATE: u32 = 44100;
const BUFFER: usize = 1024;
const TEMPO: u32 = 90;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = PathBuf::from(&args[1]);
    let library_root = PathBuf::from(&args[2]);
    let source = fs::read_to_string(&path).unwrap();
    let mut state = AppState::from_source(source, PathBuf::new()).unwrap();
    let environment = Environment::new(SAMPLE_RATE, TEMPO, library_root);

    println!("program | precompute | process ms | buffers | load max | load p95 | load mean");
    for name in &args[3..] {
        let index = (0..state.programs.programs().len())
            .find(|&i| state.programs.display_name(i) == *name)
            .unwrap_or_else(|| {
                let names: Vec<String> = (0..state.programs.programs().len())
                    .map(|i| state.programs.display_name(i))
                    .collect();
                panic!("no program {name} in {names:?}")
            });
        for precompute in [false, true] {
            let input = state.programs.evaluation_input(index);
            let started = Instant::now();
            let outcome = processing::process(&environment, &input, precompute);
            let process_ms = started.elapsed().as_secs_f64() * 1000.0;
            state.programs.record(index, outcome).unwrap();

            let (command_sender, command_receiver) = mpsc::channel();
            let (status_sender, status_receiver) = mpsc::channel();
            let mut tracker = Tracker::<WaveformId, MarkId>::new(
                SAMPLE_RATE,
                PathBuf::from("."),
                String::from("tracker_load"),
                command_receiver,
                status_sender,
                Duration::from_secs_f32(BUFFER as f32 / SAMPLE_RATE as f32),
            );
            let mut out = vec![0.0f32; BUFFER];
            tracker.callback(&mut out);
            let status = status_receiver.recv().unwrap();
            let player = Player::new(TEMPO, 4, command_sender);
            player.play_program(&state.programs, index, &status, Start::Now, None);

            let mut loads = Vec::new();
            let max_buffers = 30 * SAMPLE_RATE as usize / BUFFER;
            let period = Duration::from_secs_f64(BUFFER as f64 / SAMPLE_RATE as f64);
            let mut due = Instant::now();
            for _ in 0..max_buffers {
                due += period;
                thread::sleep(due.saturating_duration_since(Instant::now()));
                tracker.callback(&mut out);
                let status = status_receiver.recv().unwrap();
                loads.push(status.tracker_load.unwrap());
                if !status.marks.iter().any(|m| m.mark_id == MarkId::Terminator) {
                    break;
                }
            }
            let mut sorted = loads.clone();
            sorted.sort_by(f32::total_cmp);
            let max = sorted.last().copied().unwrap_or_default();
            let p95 = sorted[(sorted.len() * 95 / 100).min(sorted.len() - 1)];
            let mean = loads.iter().sum::<f32>() / loads.len() as f32;
            println!(
                "{name} | {precompute} | {process_ms:.1} | {} | {max:.3} | {p95:.3} | {mean:.4}",
                loads.len()
            );
        }
    }
}
