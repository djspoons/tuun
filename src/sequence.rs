use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

use crate::parser::{BuiltInFn, Expr};
use crate::Command;

pub trait Generator: Send {
    fn generate(&mut self, out: &mut [f32]);
}

// Sinusoidal wave generator with the given frequency
struct SineWave {
    offset: usize, // in samples
    sample_frequency: i32,
    tone_frequency: f32,
}

impl Generator for SineWave {
    fn generate(&mut self, out: &mut [f32]) {
        for (i, f) in out.iter_mut().enumerate() {
            let t_secs = (i + self.offset) as f32 / self.sample_frequency as f32;
            *f = (2.0 * PI * self.tone_frequency * t_secs).sin();
        }
        self.offset += out.len();
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

struct Amplifier {
    gain: f32,
    generator: Box<dyn Generator>,
}

impl<'a> Generator for Amplifier {
    fn generate(&mut self, out: &mut [f32]) {
        self.generator.generate(out);
        for x in out.iter_mut() {
            *x *= self.gain;
        }
    }
}

/*
 * Like a generator, but may also have a known stopping point as a point at which another
 * generate can start after it.
 */
struct BoundedGenerator {
    generator: Box<dyn Generator>,
    total_samples: Option<usize>,
    samples_before_next: Option<usize>,
}

struct Chord {
    bounded: Vec<BoundedGenerator>,
    // TODO move all of these offsets into the Sequencer?
    offset: usize,
}

impl Generator for Chord {
    fn generate(&mut self, out: &mut [f32]) {
        for BoundedGenerator {
            generator,
            total_samples,
            ..
        } in &mut self.bounded
        {
            match total_samples {
                None => {
                    let mut tmp = vec![0.0; out.len()];
                    generator.generate(&mut tmp);
                    for (i, x) in tmp.iter().enumerate() {
                        out[i] += x;
                    }
                }

                Some(total_samples) => {
                    if self.offset < *total_samples as usize {
                        let len = *total_samples as usize - self.offset;
                        let mut tmp = vec![0.0; out.len().min(len)];
                        generator.generate(&mut tmp);
                        for (i, x) in tmp.iter().enumerate() {
                            out[i] += x;
                        }
                    }
                }
            }
        }
        self.offset += out.len(); // for the next call to generate
    }
}

/*
 * Play the generators in sequence, starting each after samples_before_next of the previous generator.
 * If a given generator doesn't provide samples_before_next then no subsequent generators will be played.
 */
struct Sequence {
    bounded: Vec<BoundedGenerator>,
    offset: usize,
}

impl<'a> Generator for Sequence {
    fn generate(&mut self, mut out: &mut [f32]) {
        // local_offset is the offset within the current generator, starting with the first
        let mut local_offset = self.offset;
        self.offset += out.len(); // for the next call to generate
        for BoundedGenerator {
            generator,
            total_samples,
            samples_before_next,
        } in &mut self.bounded
        {
            match total_samples {
                None => {
                    generator.generate(out);
                }
                Some(total_samples) => {
                    if local_offset < *total_samples {
                        let len = *total_samples - local_offset;
                        let mut tmp = vec![0.0; out.len().min(len)];
                        generator.generate(&mut tmp);
                        for (i, x) in tmp.iter().enumerate() {
                            out[i] += x;
                        }
                    }
                }
            }
            match samples_before_next {
                None => {
                    return;
                }
                Some(samples_before_next) => {
                    if local_offset + out.len() > *samples_before_next {
                        if local_offset < *samples_before_next {
                            out = &mut out[(*samples_before_next - local_offset)..];
                            local_offset = 0;
                        } else {
                            local_offset -= *samples_before_next;
                        }
                    } else {
                        return;
                    }
                }
            }
        }
    }
}

fn from_expr(sample_frequency: i32, expr: Expr) -> Option<BoundedGenerator> {
    use Expr::{Application, BuiltIn, Float, Truncated, Tuple};
    println!("from_expr called with expr {:?}", expr);
    match expr {
        Application { function, argument } => {
            match (*function, *argument) {
                (BuiltIn(BuiltInFn::Amplify), Tuple(arguments)) if arguments.len() == 2 => {
                    if let (Float(gain), waveform) = (&arguments[0], &arguments[1]) {
                        println!("Amplifying with gain {} and expr {:?}", gain, waveform);
                        let BoundedGenerator {
                            generator,
                            total_samples,
                            samples_before_next,
                        } = from_expr(sample_frequency, waveform.clone())?;
                        return Some(BoundedGenerator {
                            generator: Box::new(Amplifier {
                                gain: *gain,
                                generator,
                            }),
                            total_samples,
                            samples_before_next,
                        });
                    }
                }
                _ => {
                    return None;
                }
            }
            return None;
        }
        Expr::SineWave { frequency } => {
            if let Expr::Float(tone_frequency) = *frequency {
                return Some(BoundedGenerator {
                    generator: Box::new(SineWave {
                        sample_frequency,
                        tone_frequency,
                        offset: 0,
                    }),
                    total_samples: None,
                    samples_before_next: None,
                });
            }
            return None;
        }
        Truncated { duration, waveform } => {
            let BoundedGenerator { generator, .. } =
                from_expr(sample_frequency, *waveform.clone())?;
            return Some(BoundedGenerator {
                generator,
                total_samples: Some(duration.as_secs() as usize * sample_frequency as usize),
                samples_before_next: Some(duration.as_secs() as usize * sample_frequency as usize),
            });
        }
        Expr::Chord(exprs) => {
            let mut bounded = Vec::new();
            let mut total_samples = Some(0usize);
            let mut samples_before_next = Some(0usize);
            for expr in exprs {
                let generator = from_expr(sample_frequency, expr)?;
                match (total_samples, generator.total_samples) {
                    (Some(s), Some(t)) => {
                        total_samples = Some(s.max(t));
                    }
                    (_, _) => {
                        total_samples = None;
                    }
                }
                match (samples_before_next, generator.samples_before_next) {
                    (Some(s), Some(t)) => {
                        samples_before_next = Some(s.max(t));
                    }
                    (_, _) => {
                        samples_before_next = None;
                    }
                }
                bounded.push(generator);
            }
            return Some(BoundedGenerator {
                generator: Box::new(Chord { bounded, offset: 0 }),
                total_samples: total_samples,
                samples_before_next: samples_before_next,
            });
        }
        Expr::Sequence(exprs) => {
            let mut bounded: Vec<BoundedGenerator> = Vec::new();
            let mut offset = 0usize;
            let mut total_samples = 0usize;
            for expr in exprs {
                let generator = from_expr(sample_frequency, expr)?;
                match (generator.total_samples, generator.samples_before_next) {
                    (Some(s), Some(t)) => {
                        total_samples = offset + s;
                        offset += t;
                        bounded.push(generator);
                    }
                    (None, Some(t)) => {
                        offset += t;
                        bounded.push(generator);
                    }
                    (_, None) => {
                        println!("Warning: sequence generator used with unbounded component");
                        bounded.push(generator);
                        return Some(BoundedGenerator {
                            generator: Box::new(Sequence { bounded, offset: 0 }),
                            total_samples: None,
                            samples_before_next: None,
                        });
                    }
                }
            }
            return Some(BoundedGenerator {
                generator: Box::new(Sequence { bounded, offset: 0 }),
                total_samples: Some(total_samples),
                samples_before_next: Some(offset),
            });
        }
        _ => {
            println!("Unsupported expression in from_expr: {:?}", expr);
            None
        }
    }
}

pub struct Tracker {
    sample_frequency: i32,
    _beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    generator: Option<BoundedGenerator>,
    sample_sender: Sender<Vec<f32>>,
    send_counter: u32,
}

pub fn new_tracker(
    sample_frequency: i32,
    beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    sample_sender: Sender<Vec<f32>>,
) -> Tracker {
    return Tracker {
        sample_frequency: sample_frequency,
        _beats_per_minute: beats_per_minute,
        command_receiver: command_receiver,
        generator: None,
        sample_sender: sample_sender,
        send_counter: 0,
    };
}

// TODO add metrics

impl<'a> AudioCallback for Tracker {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        for x in out.iter_mut() {
            *x = 0.0;
        }
        let mut generated = 0;
        let mut recv_allowed = true;
        while generated < out.len() && recv_allowed {
            if self.generator.is_none() && recv_allowed {
                recv_allowed = false;
                match self.command_receiver.try_recv() {
                    Ok(Command::PlayOnce { expr, beat: _beat }) => {
                        self.generator = from_expr(self.sample_frequency, expr);
                        if let None = self.generator {
                            println!("Failed to convert expression to generator");
                        }
                        recv_allowed = true;
                    }
                    Err(_) => {}
                }
            }
            match &mut self.generator {
                Some(BoundedGenerator {
                    generator,
                    total_samples: None,
                    ..
                }) => {
                    generator.generate(out);
                    generated += out.len();
                }
                Some(BoundedGenerator {
                    generator,
                    total_samples: Some(total_samples),
                    ..
                }) => {
                    if *total_samples <= out.len() {
                        generator.generate(&mut out[..*total_samples]);
                        generated += *total_samples;
                        self.generator = None;
                    } else {
                        generator.generate(out);
                        generated += out.len();
                        self.generator.as_mut().unwrap().total_samples =
                            Some(*total_samples - generated);
                    }
                }
                None => (),
            }
        }

        if self.send_counter == 0 {
            let mut copy: Vec<f32> = Vec::with_capacity(out.len());
            out.clone_into(&mut copy);
            self.sample_sender.send(copy).unwrap();
            self.send_counter = 5;
        } else {
            self.send_counter -= 1;
        }
    }

    // TODO need some sort of way to signal done
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ConstGenerator {
        value: f32,
    }

    impl Generator for ConstGenerator {
        fn generate(&mut self, out: &mut [f32]) {
            for x in out.iter_mut() {
                *x = self.value;
            }
        }
    }
    fn const_gen(f: f32) -> Box<dyn Generator> {
        return Box::new(ConstGenerator { value: f });
    }

    fn generate_samples(generator: &mut dyn Generator, size: usize, out: &mut [f32]) {
        for n in 0..out.len() / size {
            generator.generate(&mut out[n * size..(n + 1) * size]);
        }
    }

    #[test]
    fn test_chord() {
        let desired = vec![
            1.875, 1.875, 1.875, 1.875, 1.875, 0.875, 0.875, 0.875, 0.75, 0.75, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut chord_gen = Chord {
                bounded: vec![
                    BoundedGenerator {
                        generator: const_gen(1.0),
                        total_samples: Some(5),
                        samples_before_next: None,
                    },
                    BoundedGenerator {
                        generator: const_gen(0.5),
                        total_samples: Some(10),
                        samples_before_next: None,
                    },
                    BoundedGenerator {
                        generator: const_gen(0.25),
                        total_samples: Some(15),
                        samples_before_next: None,
                    },
                    BoundedGenerator {
                        generator: const_gen(0.125),
                        total_samples: Some(8),
                        samples_before_next: None,
                    },
                ],
                offset: 0,
            };
            let mut out = vec![0.0; 16];
            println!("Generating chord with buffer size {}", size);
            generate_samples(&mut chord_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_sequence() {
        let desired = vec![
            1.0, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.25, 0.25, 0.25, 0.0, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut seq_gen = Sequence {
                bounded: vec![
                    BoundedGenerator {
                        generator: const_gen(1.0),
                        total_samples: Some(5),
                        samples_before_next: Some(1),
                    },
                    BoundedGenerator {
                        generator: const_gen(0.5),
                        total_samples: Some(10),
                        samples_before_next: Some(8),
                    },
                    BoundedGenerator {
                        generator: const_gen(0.25),
                        total_samples: Some(5),
                        samples_before_next: Some(5),
                    },
                ],
                offset: 0,
            };
            let mut out = vec![0.0; 16];
            println!("Generating sequence with buffer size {}", size);
            generate_samples(&mut seq_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }
}
