extern crate sdl2;

use std::f32::consts::PI;
use std::time::Duration;
use std::sync::mpsc::Sender;

use sdl2::audio::AudioCallback;


// offset and out
// TODO maybe better to do everything as samples down here? (rather than durations?)
// returns true if the generator will continue to generate non-silence
// maybe also return how many were filled? (with out.len() + 1 meaning more to come?)
pub type Generator = Box<dyn Fn(f32, Duration, &mut [f32]) -> bool + Send>;

// Sinusoidal wave generator with the given frequency
pub fn wave_from_frequency(frequency: f32) -> Generator {
    Box::new(move |sample_frequency: f32, offset: Duration, out: &mut [f32]| {
        for (i, f) in out.iter_mut().enumerate() {
            let t_secs = i as f32 / sample_frequency + offset.as_secs_f32();
            *f = (2.0 * PI * frequency * t_secs).sin();
        }
        return true;
    })
}

pub fn wave_from_midi_number(note: u8) -> Generator {
    println!("Note: {} {}", note, 440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0));
    return wave_from_frequency(
        440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0),
    );
}

pub fn truncate(generator: Generator, duration: Duration) -> Generator {
    return Box::new(move |sample_frequency: f32, offset: Duration, out: &mut [f32]| {
        if offset >= duration {
            return false;
        }
        return generator(sample_frequency, offset, out);
    });
}

pub fn chord(generators: Vec<Generator>) -> Generator {
    return Box::new(move |sample_frequency: f32, offset: Duration, out: &mut [f32]| {
        let mut more = false;
        let n = generators.len() as f32;
        for (z, generator) in generators.iter().enumerate() {
            let mut tmp = vec![0.0; out.len()];
            more = generator(sample_frequency, offset, &mut tmp) || more;
            for (i, x) in tmp.iter().enumerate() {
                out[i] += x / n;
            }
        }
        return more;
    });
}

/*
struct Envelope {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32
}
 */

 pub struct Sequence {
    sample_frequency: f32,
    current_offset: Duration,
    generators: Vec<Generator>,
    sender: Sender<Vec<f32>>,
    send_counter: u32
}

pub fn new_sequence(
    sample_frequency: f32, 
    generators: Vec<Generator>,
    sender: Sender<Vec<f32>>) -> Sequence {
    return Sequence {
        sample_frequency: sample_frequency,
        current_offset: Duration::from_secs(0),
        generators: generators,
        sender: sender,
        send_counter: 0
    };
}

// TODO add metrics

impl AudioCallback for Sequence {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        match self.generators.first() {
            None => {
            },
            Some(generator) => {
                let more = generator(self.sample_frequency, self.current_offset, out);
                if more {
                    self.current_offset += Duration::from_secs_f32(
                        out.len() as f32 / self.sample_frequency);
                } else {
                    drop(self.generators.remove(0));
                    self.current_offset = Duration::from_secs(0);
                }
            }
        }

        if self.send_counter == 0 {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            self.sender.send(copy).unwrap();
            self.send_counter = 5
            ;
        } else {
            self.send_counter -= 1;
        }
    }

    // TODO need some sort of way to signal done
}

