extern crate sdl2;

use std::f32::consts::PI;
use std::time::Duration;
use std::sync::mpsc::Sender;

use sdl2::audio::AudioCallback;


// offset and out
// TODO maybe better to do everything as samples down here? (rather than durations?)
// returns true if the generator will continue to generate non-silence
pub type Generator = Box<dyn Fn(f32, Duration, &mut [f32]) -> bool + Send>;

// Just generates a sine wave forever
/*
pub struct WaveGenerator {
    pub frequency: f32,
}

impl Generator for WaveGenerator {
    fn generate(&self, offset: Duration, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = (offset.as_secs_f32() * self.frequency * 2.0 * std::f32::consts::PI).sin() * self.volume;
        }
    }
}
*/

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
    offsets: Vec<Duration>,
    current_offset: Duration,
    generators: Vec<Generator>,
    sender: Sender<Vec<f32>>
}

pub fn new_sequence(
    sample_frequency: f32, 
    generators: Vec<Generator>,
    sender: Sender<Vec<f32>>) -> Sequence {
    let offsets = (0..generators.len()).map(|i| {
        Duration::from_secs(i as u64)
    }).collect();
    return Sequence {
        sample_frequency: sample_frequency,
        offsets: offsets,
        current_offset: Duration::from_secs(0),
        generators: generators,
        sender: sender
    };
}

// TODO add metrics

impl AudioCallback for Sequence {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        // TODO handle overlapping generators use temp out for each generator, then add?
        // TODO maybe need to zero out too
        let mut more = Vec::new();
        for ((_i, offset), generator) in 
            self.offsets.iter().enumerate().zip(self.generators.iter()) {
                // This is not quite right if the offset was in the middle of the buffer... but maybe we're getting rid of offsets anyway
            if self.current_offset >= *offset {
                more.push(generator(self.sample_frequency,
                    self.current_offset - *offset, out));
            } else {
                more.push(true);
            }
        }

        self.current_offset += 
            Duration::from_secs_f32(out.len() as f32 / self.sample_frequency);
        let mut tmp_generators = std::mem::take(&mut self.generators);
        let mut tmp_offsets = std::mem::take(&mut self.offsets);

        let mut index = 0;
        for has_more in more.iter() {
            if *has_more {
                self.generators.push(tmp_generators.remove(index));
                self.offsets.push(tmp_offsets.remove(index));
            } else {
                index += 1;
            }
        }

        let mut copy: Vec<f32> = Vec::with_capacity(out.len());
        out.clone_into(&mut copy);
        self.sender.send(copy).unwrap();
    }

    // TODO need some sort of way to signal done
}

