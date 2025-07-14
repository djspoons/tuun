use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

use crate::Command;

// Length is a possibly infinite size
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Length {
    Finite(usize), // finite length in samples
    Infinite,      // infinite length
}

impl std::ops::Add for Length {
    type Output = Length;

    fn add(self, other: Length) -> Length {
        match (self, other) {
            (Length::Finite(a), Length::Finite(b)) => Length::Finite(a + b),
            (Length::Infinite, _) | (_, Length::Infinite) => Length::Infinite,
        }
    }
}

impl Length {
    pub fn min(&self, other: &Length) -> Length {
        match (self, other) {
            (Length::Finite(a), Length::Finite(b)) => Length::Finite(*a.min(b)),
            (Length::Infinite, _) => other.clone(),
            (_, Length::Infinite) => self.clone(),
        }
    }

    pub fn max(&self, other: &Length) -> Length {
        match (self, other) {
            (Length::Finite(a), Length::Finite(b)) => Length::Finite(*a.max(b)),
            (Length::Infinite, _) | (_, Length::Infinite) => Length::Infinite,
        }
    }
}

impl From<usize> for Length {
    fn from(n: usize) -> Self {
        Length::Finite(n)
    }
}

/* XXX TODO duration in beats */

#[derive(Debug, Clone)]
pub enum Waveform {
    /*
     * SineWave is a sinusoidal wave generator with the given frequency.
     */
    SineWave {
        frequency: f32,
    },
    Const(f32),
    Linear {
        initial_value: f32,
        slope: f32, // slope in value per second
    },
    Fin {
        duration: f32, // duration in seconds
        waveform: Box<Waveform>,
    },
    /*
     * Seq sets the offset to the given value (ignoring offset of the underlying waveform).
     */
    Seq {
        duration: f32, // duration in seconds
        waveform: Box<Waveform>,
    },
    Sum(Box<Waveform>, Box<Waveform>),
    DotProduct(Box<Waveform>, Box<Waveform>),
}

impl Waveform {
    fn generate(&self, sample_frequency: i32, offset: usize, desired: usize) -> Vec<f32> {
        match self {
            Waveform::SineWave { frequency } => {
                let mut out = vec![0.0; desired];
                for (i, f) in out.iter_mut().enumerate() {
                    let t_secs = (i + offset) as f32 / sample_frequency as f32;
                    *f = (2.0 * PI * frequency * t_secs).sin();
                }
                return out;
            }
            Waveform::Const(value) => {
                return vec![*value; desired];
            }
            Waveform::Linear {
                initial_value,
                slope,
            } => {
                let mut out = vec![0.0; desired];
                for (i, x) in out.iter_mut().enumerate() {
                    *x = initial_value + slope * ((i + offset) as f32 / sample_frequency as f32);
                }
                return out;
            }
            Waveform::Fin { duration, waveform } => {
                let length = (duration * sample_frequency as f32) as usize;
                if offset >= length {
                    return Vec::new(); // No samples to generate
                }
                return waveform.generate(sample_frequency, offset, desired.min(length - offset));
            }
            Waveform::Seq { waveform, .. } => {
                return waveform.generate(sample_frequency, offset, desired);
            }
            Waveform::Sum(a, b) => {
                generate_binary_op(|x, y| x + y, a, b, sample_frequency, offset, desired)
            }
            Waveform::DotProduct(a, b) => {
                // Like sum, but we need to make sure we generate a length based on
                // the shorter waveform.
                let desired2 = match self.length(sample_frequency) {
                    Length::Finite(length) => {
                        if length > offset {
                            desired.min(length - offset)
                        } else {
                            0
                        }
                    }
                    Length::Infinite => desired,
                };
                generate_binary_op(|x, y| x * y, a, b, sample_frequency, offset, desired2)
            }
        }
    }

    fn length(&self, sample_frequency: i32) -> Length {
        match self {
            Waveform::SineWave { .. } => Length::Infinite,
            Waveform::Const { .. } => Length::Infinite,
            Waveform::Linear { .. } => Length::Infinite,
            Waveform::Fin { duration, .. } => {
                ((duration * sample_frequency as f32) as usize).into()
            }
            Waveform::Seq { waveform, .. } => waveform.length(sample_frequency),
            Waveform::Sum(a, b) => {
                let length =
                    Length::Finite(a.offset(sample_frequency)) + b.length(sample_frequency);
                a.length(sample_frequency).max(&length)
            }
            Waveform::DotProduct(a, b) => {
                let length =
                    Length::Finite(a.offset(sample_frequency)) + b.length(sample_frequency);
                a.length(sample_frequency).min(&length)
            }
        }
    }

    fn offset(&self, sample_frequency: i32) -> usize {
        match self {
            Waveform::SineWave { .. } => 0,
            Waveform::Const { .. } => 0,
            Waveform::Linear { .. } => 0,
            Waveform::Fin { waveform, .. } => waveform.offset(sample_frequency),
            Waveform::Seq { duration, .. } => (duration * sample_frequency as f32) as usize,
            Waveform::Sum(a, b) | Waveform::DotProduct(a, b) => {
                a.offset(sample_frequency) + b.offset(sample_frequency)
            }
        }
    }
}

fn generate_binary_op(
    op: fn(f32, f32) -> f32,
    a: &Waveform,
    b: &Waveform,
    sample_frequency: i32,
    offset: usize,
    desired: usize,
) -> Vec<f32> {
    let mut left = a.generate(sample_frequency, offset, desired);
    let next_offset = a.offset(sample_frequency);
    if next_offset < offset + desired {
        // There is an overlap between the desired portion and the right waveform...
        //    1) ... and the right waveform starts after the offset
        // or 2) ... and the right waveform starts before the offset

        if offset + left.len() < next_offset {
            // Either way, if the left side is shorter than the next offset, than extend it.
            left.resize(next_offset - offset, 0.0);
        }

        if offset < next_offset {
            // ... and the right waveform starts after the offset:
            let right = b.generate(sample_frequency, 0, desired - (next_offset - offset));
            // Merge the overlapping portion
            for (i, x) in left[next_offset - offset..].iter_mut().enumerate() {
                if i >= right.len() {
                    break;
                }
                *x = op(*x, right[i]);
            }
            // If the left side is shorter than the right, than append.
            if right.len() + next_offset > left.len() + offset {
                left.extend_from_slice(&right[(left.len() + offset - next_offset)..]);
            }
        } else {
            // ... and the right waveform starts before the offset
            let right = b.generate(sample_frequency, offset - next_offset, desired);
            // Merge the overlapping portion
            for (i, x) in left.iter_mut().enumerate() {
                if i >= right.len() {
                    break;
                }
                *x = op(*x, right[i]);
            }
            // If the left side is shorter than the right, than append.
            if right.len() > left.len() {
                left.extend_from_slice(&right[left.len()..]);
            }
        }
    }
    return left;
}

pub struct Tracker {
    sample_frequency: i32,
    _beats_per_minute: i32,
    command_receiver: Receiver<Command>,
    waveform: Option<(Waveform, usize)>,
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
        waveform: None,
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
            if self.waveform.is_none() && recv_allowed {
                recv_allowed = false;
                match self.command_receiver.try_recv() {
                    Ok(Command::PlayOnce { waveform, beat }) => {
                        println!(
                            "Received command to play once at {} with waveform {:?}",
                            beat, waveform
                        );
                        self.waveform = Some((waveform, 0));
                        recv_allowed = true;
                    }
                    Err(_) => {}
                }
            }
            match &self.waveform {
                Some((waveform, offset)) => {
                    let tmp = waveform.generate(self.sample_frequency, *offset, out.len());
                    (out[..tmp.len()]).copy_from_slice(&tmp);
                    generated += tmp.len();
                    if tmp.len() < out.len() {
                        self.waveform = None; // Finished generating this waveform
                        recv_allowed = true; // Allow receiving new commands
                    } else {
                        self.waveform.as_mut().unwrap().1 = offset + tmp.len();
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
    use Waveform::{Const, DotProduct, Fin, Linear, Seq, Sum};

    fn finite_const_gen(value: f32, fin_duration: f32, seq_duration: f32) -> Waveform {
        return Seq {
            duration: seq_duration,
            waveform: Box::new(Fin {
                duration: fin_duration,
                waveform: Box::new(Const(value)),
            }),
        };
    }

    fn run_tests(waveform: &Waveform, desired: Vec<f32>) {
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            for n in 0..out.len() / size {
                let tmp = waveform.generate(1, n * size, size);
                (&mut out[n * size..(n * size + tmp.len())]).copy_from_slice(&tmp);
            }
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_linear() {
        let w1 = Linear {
            initial_value: 10.0,
            slope: -1.0,
        };
        run_tests(&w1, vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0]);

        let w2 = Fin {
            duration: 4.0,
            waveform: Box::new(DotProduct(
                Box::new(Const(3.0)),
                Box::new(Sum(
                    Box::new(Seq {
                        duration: 4.0,
                        waveform: Box::new(Fin {
                            duration: 4.0,
                            waveform: Box::new(Linear {
                                initial_value: 0.0,
                                slope: 0.5,
                            }),
                        }),
                    }),
                    Box::new(Const(1.0)),
                )),
            )),
        };
        run_tests(&w2, vec![0.0, 1.5, 3.0, 4.5, 0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sum() {
        let w1 = Sum(
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
        );
        assert_eq!(w1.offset(1), 4);
        assert_eq!(w1.length(1), Length::Finite(7));
        run_tests(&w1, vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0]);

        let w2 = Fin {
            duration: 8.0,
            waveform: Box::new(Sum(
                Box::new(Seq {
                    duration: 0.0,
                    waveform: Box::new(Const(1.0)),
                }),
                Box::new(Sum(
                    Box::new(Seq {
                        duration: 0.0,
                        waveform: Box::new(Const(2.0)),
                    }),
                    Box::new(Fin {
                        duration: 0.0,
                        waveform: Box::new(Const(0.0)),
                    }),
                )),
            )),
        };
        run_tests(&w2, vec![3.0; 8]);

        let w5 = Sum(
            Box::new(finite_const_gen(3.0, 1.0, 3.0)),
            Box::new(finite_const_gen(2.0, 2.0, 2.0)),
        );
        run_tests(&w5, vec![3.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_dot_product() {
        let w1 = DotProduct(
            Box::new(finite_const_gen(3.0, 8.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        assert_eq!(w1.offset(1), 4);
        assert_eq!(w1.length(1), Length::Finite(7));
        run_tests(&w1, vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0]);

        let w2 = DotProduct(
            Box::new(finite_const_gen(3.0, 5.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        run_tests(&w2, vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0]);

        let w3 = Fin {
            duration: 8.0,
            waveform: Box::new(DotProduct(Box::new(Const(3.0)), Box::new(Const(2.0)))),
        };
        run_tests(&w3, vec![6.0; 8]);

        let w4 = DotProduct(
            Box::new(Seq {
                duration: 1.0,
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_gen(2.0, 5.0, 5.0)),
        );
        run_tests(&w4, vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0]);
    }
}
