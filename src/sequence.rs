extern crate sdl2;

use std::collections::VecDeque;
use std::f32::consts::PI;
use std::time::Duration;
use std::sync::mpsc::Sender;

use sdl2::audio::AudioCallback;

pub enum GeneratorResult {
    Finished(usize), // number of samples filled
    More
}

pub trait Generator: Send {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult;
}

// Sinusoidal wave generator with the given frequency
pub struct SineWaveGenerator {
    offset: usize, // in samples
    sample_frequency: i32,
    tone_frequency: f32
}

pub fn wave_from_frequency(sample_frequency: i32, tone_frequency: f32) -> SineWaveGenerator {
    return SineWaveGenerator {
        offset: 0,
        sample_frequency: sample_frequency,
        tone_frequency: tone_frequency
    };
}

impl Generator for SineWaveGenerator {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult {
        for (i, f) in out.iter_mut().enumerate() {
            let t_secs = (i + self.offset) as f32 / self.sample_frequency as f32;
            *f = (2.0 * PI * self.tone_frequency * t_secs).sin();
        }
        self.offset += out.len();
        return GeneratorResult::More;
    }
}

pub fn _wave_from_midi_number(sample_frequency: i32, note: u8) -> SineWaveGenerator {
    println!("Note: {} {}", note, 440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0));
    return wave_from_frequency(sample_frequency, 
        440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0),
    );
}

pub struct FiniteGenerator<'a> {
    remaining: u64,
    generator: Box<dyn Generator + 'a>
}

pub fn truncate<'a>(sample_frequency: i32, duration: Duration, generator: impl Generator + 'a) -> 
    FiniteGenerator<'a> {
    return FiniteGenerator {
        remaining: (duration.as_secs() as u64) * (sample_frequency as u64),
        generator: Box::new(generator)
    };
}

impl <'a> Generator for FiniteGenerator<'a> {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult {
        use GeneratorResult::*;
        if self.remaining <= out.len() as u64 {
            match self.generator.generate(&mut out[..self.remaining as usize]) {
                Finished(size) => {
                    // It's possible that size is smaller than remaining, but if it is that 
                    // means that the underlying generator finished (before we truncated
                    // it). Which means there's no more to do in any case.
                    self.remaining = 0;
                    return Finished(size);
                },
                More => {
                    let size = self.remaining as usize;
                    self.remaining = 0;
                    return Finished(size);
                }
            }
        } else { // else remaining > out.len()
            match self.generator.generate(out) {
                Finished(size) => {
                    // Again, if the underlying generator is finished, we're done too.
                    self.remaining = 0;
                    return Finished(size);
                },
                More => {
                    self.remaining -= out.len() as u64;
                    return More;
                }
            }
        }
    }
}

pub struct ChordGenerator<'a> {
    generators: Vec<Box<dyn Generator + 'a>>
}

pub fn chord<'a>(generators: Vec<impl Generator + 'a>) -> ChordGenerator<'a> {
    return ChordGenerator {
        // TODO why the 'as _'?!?
        generators: generators.into_iter().map(|g| Box::new(g) as _).collect()
    };
}

impl <'a> Generator for ChordGenerator<'a> {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult {
        use GeneratorResult::*;
        let n = self.generators.len() as f32;
        let mut result = Finished(0);
        for generator in self.generators.iter_mut() {
            let mut tmp = vec![0.0; out.len()];
            match generator.generate(&mut tmp) {
                Finished(s) => {
                    match result {
                        Finished(size) => {
                            result = Finished(size.max(s));
                        },
                        More => {}
                    }
                },
                More => {
                    result = More;
                }
            }
            for (i, x) in tmp.iter().enumerate() {
                out[i] += x / n;
            }
        }
        return result;
    }
}

/*
struct Envelope {
    attack: f32,
    decay: f32,
    sustain: f32,
    release: f32
}
 */

 pub struct Sequence<'a> {
    generators: VecDeque<Box<dyn Generator + Send + 'a>>,
    sender: Sender<Vec<f32>>,
    send_counter: u32
}

pub fn new_sequence<'a>(generators: Vec<impl Generator + Send + 'a>,
    sender: Sender<Vec<f32>>) -> Sequence<'a> {
    return Sequence {
        generators: generators.into_iter().map(|g| Box::new(g) as _).collect(),
        sender: sender,
        send_counter: 0
    };
}

// TODO add metrics

impl <'a> AudioCallback for Sequence<'a> {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        use GeneratorResult::*;
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut remaining = out.len();
        while remaining > 0 {
            match self.generators.pop_front() {
                None => {
                    remaining = 0;
                },
                Some(mut generator) => {
                    let offset = out.len() - remaining;
                    match generator.generate(&mut out[offset..]) {
                        Finished(size) => {
                            remaining -= size;
                        },
                        More => {
                            self.generators.push_front(generator);
                            remaining = 0;
                        }
                    }
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

