use std::collections::VecDeque;
use std::f32::consts::PI;
use std::time::Duration;
use std::sync::mpsc::{Receiver,Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

use crate::Command;
use crate::parser::{BuiltInFn, Expr};

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
            *f = (2.0 * PI * self.tone_frequency * t_secs).cos();
        }
        self.offset += out.len();
        return GeneratorResult::More;
    }
}

fn _wave_from_midi_number(sample_frequency: i32, note: u8) -> SineWaveGenerator {
    println!("Note: {} {}", note, 440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0));
    return wave_from_frequency(sample_frequency,
        // 12-TET
        440.0 * 2.0f32.powf((note as f32 - 69.0) / 12.0),
    );
}

struct AmplifyingGenerator {
    gain: f32,
    generator: Box<dyn Generator>,
}

fn amplify(gain: f32, generator: Box<dyn Generator>) ->
    AmplifyingGenerator {
    return AmplifyingGenerator {
        gain: gain,
        generator: generator,
};
}

impl <'a> Generator for AmplifyingGenerator {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult {
        use GeneratorResult::*;
        match self.generator.generate(out) {
            Finished(size) => {
                for x in out.iter_mut() {
                    *x *= self.gain;
                }
                return Finished(size);
            },
            More => {
                for x in out.iter_mut() {
                    *x *= self.gain;
                }
                return More;
            }
        }
    }
}

pub struct FiniteGenerator {
    remaining: u64,
    generator: Box<dyn Generator>,
}

pub fn truncate(sample_frequency: i32, duration: Duration, generator: Box<dyn Generator>) ->
    FiniteGenerator {
    return FiniteGenerator {
        remaining: (duration.as_secs() as u64) * (sample_frequency as u64),
        generator: generator,
};
}

impl <'a> Generator for FiniteGenerator {
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

struct ChordGenerator {
    generators: Vec<Box<dyn Generator>>
}

fn chord(generators: Vec<Box<dyn Generator>>) -> ChordGenerator {
    return ChordGenerator {
        // TODO why the 'as _'?!?
        generators: generators,
    };
}

impl Generator for ChordGenerator {
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

struct SequenceGenerator {
    generators: VecDeque<Box<dyn Generator>>,
}

fn sequence(generators: Vec<Box<dyn Generator>>) -> SequenceGenerator {
    return SequenceGenerator {
        generators: generators.into_iter().map(|g| g as _).collect()
    };
}

impl <'a> Generator for SequenceGenerator {
    fn generate(&mut self, out: &mut [f32]) -> GeneratorResult {
        use GeneratorResult::*;
        let mut remaining = out.len();
        while remaining > 0 {
            match self.generators.pop_front() {
                None => {
                    return Finished(out.len() - remaining);
                },
                Some(mut generator) => {
                    let offset = out.len() - remaining;
                    match generator.generate(&mut out[offset..]) {
                        Finished(size) => {
                            remaining -= size;
                        },
                        More => {
                            self.generators.push_front(generator);
                            return More;
                        }
                    }
                }
            }
        }
        return Finished(out.len());
    }
}

fn from_expr<'a>(sample_frequency: i32, expr: Expr) -> Option<Box<dyn Generator + 'a>> {
    use Expr::{Float, Tuple, Application, BuiltIn, SineWave, Truncated, Sequence, Chord};
    use BuiltInFn::Amplify;
    println!("from_expr called with expr {:?}", expr);
    match expr {
        Application {function, argument } => {
            match (*function, *argument) {
                (BuiltIn(Amplify), Tuple(arguments)) if arguments.len() == 2 => {
                    if let (Float(gain), waveform) = (&arguments[0], &arguments[1]) {
                        println!("Amplifying with gain {} and expr {:?}", gain, waveform);
                        return Some(Box::new(amplify(*gain,
                            from_expr(sample_frequency, waveform.clone())?)));
                    }
                },
                _ => {
                    return None;
                }
            }
            return None;
        }
        SineWave { frequency } => {
            if let Expr::Float(value) = *frequency {
                return Some(Box::new(wave_from_frequency(sample_frequency, value)));
            }
            return None;
        },
        Truncated { duration, waveform } => {
            return Some(Box::new(truncate(sample_frequency, duration,
                 from_expr(sample_frequency, *waveform)?)));
        },
        Sequence (exprs) => {
            let mut generators = Vec::new();
            for expr in exprs {
                generators.push(from_expr(sample_frequency, expr)?);
            }
            return Some(Box::new(sequence(generators)));
        },
        Chord (nodes) => {
            let mut generators = Vec::new();
            for node in nodes {
                generators.push(from_expr(sample_frequency, node)?);
            }
            return Some(Box::new(chord(generators)));
        },
        _ => {
            None
        }
    }
}

 pub struct Sequencer {
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    generator: Option<Box<dyn Generator>>,
    sample_sender: Sender<Vec<f32>>,
    send_counter: u32
}

pub fn new_sequencer(
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    sample_sender: Sender<Vec<f32>>) -> Sequencer {
    return Sequencer {
        sample_frequency: sample_frequency,
        beats_per_minute: beats_per_minute,
        command_receiver: command_receiver,
        generator: None,
        sample_sender: sample_sender,
        send_counter: 0
    };
}

// TODO add metrics

impl <'a> AudioCallback for Sequencer {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut generated = 0;
        let mut recv_allowed = true;
        while generated < out.len() && recv_allowed{
            if self.generator.is_none() && recv_allowed {
                recv_allowed = false;
                match self.command_receiver.try_recv() {
                    Ok(Command::PlayOnce{ expr, beat }) => {
                        self.generator = from_expr(self.sample_frequency, expr);
                        if let None = self.generator {
                            println!("Failed to convert expression to generator");
                        }
                        recv_allowed = true;
                    },
                    Err(_) => {
                    }
                }
            }
            if self.generator.is_some() {
                match self.generator.as_mut().unwrap().generate(out) {
                    GeneratorResult::Finished(size) => {
                        println!("Finished generating {} samples out of {}", size, out.len());
                        self.generator = None;
                        generated += size;
                    },
                    GeneratorResult::More => {
                        generated = out.len();
                    }
                }
            }
        }

        if self.send_counter == 0 {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            self.sample_sender.send(copy).unwrap();
            self.send_counter = 5
            ;
        } else {
            self.send_counter -= 1;
        }
    }

    // TODO need some sort of way to signal done
}

