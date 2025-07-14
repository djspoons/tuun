use std::f32::consts::PI;
use std::sync::mpsc::{Receiver, Sender};

extern crate sdl2;
use sdl2::audio::AudioCallback;

use crate::Command;

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
     * Seq sets the next_offset to the given value (ignoring next_offset of the underlying waveform).
     */
    Seq {
        duration: f32, // duration in seconds
        waveform: Box<Waveform>,
    },
    Sum(Box<Waveform>, Box<Waveform>),
    DotProduct(Box<Waveform>, Box<Waveform>),
}

fn half_binary_op(
    sample_frequency: i32,
    offset: usize,
    out: &mut [f32],
    w: &Waveform,
    op: fn(f32, f32) -> f32,
) {
    match w.samples(sample_frequency) {
        None => {
            let mut tmp = vec![0.0; out.len()];
            w.generate(sample_frequency, offset, &mut tmp);
            for (i, x) in tmp.iter().enumerate() {
                out[i] = op(out[i], *x);
            }
        }
        Some(samples) => {
            if offset < samples {
                let len = samples - offset;
                let mut tmp = vec![0.0; out.len().min(len)];
                w.generate(sample_frequency, offset, &mut tmp);
                for (i, x) in tmp.iter().enumerate() {
                    out[i] = op(out[i], *x);
                }
            }
        }
    }
}

impl Waveform {
    // TODO better to return the buffer?
    fn generate(&self, sample_frequency: i32, mut offset: usize, mut out: &mut [f32]) {
        match self {
            Waveform::SineWave { frequency } => {
                for (i, f) in out.iter_mut().enumerate() {
                    let t_secs = (i + offset) as f32 / sample_frequency as f32;
                    *f = (2.0 * PI * frequency * t_secs).sin();
                }
            }
            Waveform::Const(value) => {
                for x in out.iter_mut() {
                    *x = *value;
                }
            }
            Waveform::Linear {
                initial_value,
                slope,
            } => {
                for (i, x) in out.iter_mut().enumerate() {
                    *x = initial_value + slope * ((i + offset) as f32 / sample_frequency as f32);
                }
            }
            Waveform::Fin { duration, waveform } => {
                let samples = (duration * sample_frequency as f32) as usize;
                if offset + out.len() < samples {
                    waveform.generate(sample_frequency, offset, out);
                } else if offset < samples {
                    waveform.generate(sample_frequency, offset, &mut out[..samples - offset]);
                }
            }
            Waveform::Seq { waveform, .. } => {
                waveform.generate(sample_frequency, offset, out);
            }
            Waveform::Sum(a, b) => {
                half_binary_op(sample_frequency, offset, out, a, |x, y| x + y);
                let next_offset = a.next_offset(sample_frequency);
                if next_offset < offset + out.len() {
                    if offset < next_offset {
                        out = &mut out[(next_offset - offset)..];
                        offset = 0;
                    } else {
                        offset -= next_offset;
                    }
                } else {
                    return;
                }
                half_binary_op(sample_frequency, offset, out, b, |x, y| x + y);
            }

            Waveform::DotProduct(a, b) => {
                let samples = self.samples(sample_frequency);
                match samples {
                    Some(samples) => {
                        if offset >= samples {
                            return;
                        }
                        if offset + out.len() > samples {
                            out = &mut out[..samples - offset];
                        }
                    }
                    None => {}
                }
                half_binary_op(sample_frequency, offset, out, a, |x, y| x + y);
                let next_offset = a.next_offset(sample_frequency);
                if next_offset < offset + out.len() {
                    if offset < next_offset {
                        out = &mut out[(next_offset - offset)..];
                        offset = 0;
                    } else {
                        offset -= next_offset;
                    }
                } else {
                    return;
                }
                half_binary_op(sample_frequency, offset, out, b, |x, y| x * y);
            }
        }
    }

    fn samples(&self, sample_frequency: i32) -> Option<usize> {
        match self {
            Waveform::SineWave { .. } => None,
            Waveform::Const { .. } => None,
            Waveform::Linear { .. } => None,
            Waveform::Fin { duration, .. } => Some((duration * sample_frequency as f32) as usize),
            Waveform::Seq { waveform, .. } => waveform.samples(sample_frequency),
            Waveform::Sum(a, b) => {
                match (a.samples(sample_frequency), b.samples(sample_frequency)) {
                    (None, None) => None,
                    (Some(_), None) => None,
                    (None, Some(_)) => None,
                    (Some(a_samples), Some(b_samples)) => {
                        Some(a_samples.max(a.next_offset(sample_frequency) + b_samples))
                    }
                }
            }
            Waveform::DotProduct(a, b) => {
                match (a.samples(sample_frequency), b.samples(sample_frequency)) {
                    (None, None) => None,
                    (Some(a_samples), None) => Some(a_samples),
                    (None, Some(b_samples)) => Some(b_samples + a.next_offset(sample_frequency)),
                    (Some(a_samples), Some(b_samples)) => {
                        Some(a_samples.min(a.next_offset(sample_frequency) + b_samples))
                    }
                }
            }
        }
    }

    fn next_offset(&self, sample_frequency: i32) -> usize {
        match self {
            Waveform::SineWave { .. } => 0,
            Waveform::Const { .. } => 0,
            Waveform::Linear { .. } => 0,
            Waveform::Fin { waveform, .. } => waveform.next_offset(sample_frequency),
            Waveform::Seq { duration, .. } => (duration * sample_frequency as f32) as usize,
            Waveform::Sum(a, b) | Waveform::DotProduct(a, b) => {
                a.next_offset(sample_frequency) + b.next_offset(sample_frequency)
            }
        }
    }
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
                    match waveform.samples(self.sample_frequency) {
                        Some(samples) if samples - offset < out.len() => {
                            let tmp = &mut out[generated..samples - offset];
                            waveform.generate(self.sample_frequency, *offset, tmp);
                            generated += tmp.len();
                            self.waveform = None;
                            recv_allowed = true;
                        }
                        _ => {
                            // Unbounded waveform or enough samples to fill the buffer
                            let tmp = &mut out[generated..];
                            waveform.generate(self.sample_frequency, *offset, tmp);
                            generated += tmp.len();
                            self.waveform.as_mut().unwrap().1 = offset + tmp.len();
                        }
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

    fn generate_samples(waveform: &Waveform, size: usize, out: &mut [f32]) {
        for n in 0..out.len() / size {
            waveform.generate(1, n * size, &mut out[n * size..(n + 1) * size]);
        }
    }

    #[test]
    fn test_linear() {
        let w1 = Linear {
            initial_value: 10.0,
            slope: -1.0,
        };
        let desired = vec![10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&w1, size, &mut out);
            assert_eq!(out, desired);
        }

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
        let desired = vec![0.0, 1.5, 3.0, 4.5, 0.0, 0.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&w2, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_sum() {
        let w1 = Sum(
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
            Box::new(finite_const_gen(1.0, 5.0, 2.0)),
        );
        assert_eq!(w1.next_offset(1), 4);
        assert_eq!(w1.samples(1), Some(7));
        let desired = vec![1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&w1, size, &mut out);
            assert_eq!(out, desired);
        }

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
        let desired = vec![3.0; 8];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&w2, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_dot_product() {
        let w1 = DotProduct(
            Box::new(finite_const_gen(3.0, 8.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        assert_eq!(w1.next_offset(1), 4);
        assert_eq!(w1.samples(1), Some(7));
        let desired = vec![3.0, 3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&w1, size, &mut out);
            assert_eq!(out, desired);
        }

        let dot2 = DotProduct(
            Box::new(finite_const_gen(3.0, 5.0, 2.0)),
            Box::new(finite_const_gen(2.0, 5.0, 2.0)),
        );
        let desired = vec![3.0, 3.0, 6.0, 6.0, 6.0, 0.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&dot2, size, &mut out);
            assert_eq!(out, desired);
        }

        let dot3 = Fin {
            duration: 8.0,
            waveform: Box::new(DotProduct(Box::new(Const(3.0)), Box::new(Const(2.0)))),
        };
        let desired = vec![6.0; 8];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&dot3, size, &mut out);
            assert_eq!(out, desired);
        }

        let dot4 = DotProduct(
            Box::new(Seq {
                duration: 1.0,
                waveform: Box::new(Const(3.0)),
            }),
            Box::new(finite_const_gen(2.0, 5.0, 5.0)),
        );
        let desired = vec![3.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            generate_samples(&dot4, size, &mut out);
            assert_eq!(out, desired);
        }
    }
}
