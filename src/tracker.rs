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
    Amplify {
        level: f32,
        waveform: Box<Waveform>,
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
    /*
     * LinearRamp is a time-dependent filter that modifies the given waveform, starting after the
     * next_offset of that waveform. The output is multiplied by a linear ramp that starts at
     * initial_level and ends at final_level. If the waveform has no next_offset then the ramp
     * starts at the beginning of the output.
     */
    LinearRamp {
        initial_level: f32,
        duration: f32, // duration in seconds
        final_level: f32,
        waveform: Box<Waveform>,
    },
    /*
     * Sustain is a time-dependent filter that amplifies the portion of a waveform starting at the
     * next_offset of the underlying waveform and continuing for the given length. If the waveform
     * has no next_offset then the sustain starts at the beginning of the output.
     */
    Sustain {
        level: f32,
        duration: f32, // duration in seconds
        waveform: Box<Waveform>,
    },
    Chord(Vec<Waveform>),
    /*
     * Play the waveforms in sequence, starting each after next_offset of the previous generator.
     * If a given waveform doesn't provide next_offset then no subsequent waveforms will be played.
     */
    Sequence(Vec<Waveform>),
    // Used for testing at the moment
    #[allow(dead_code)]
    Const(f32),
}

impl Waveform {
    fn generate(&self, sample_frequency: i32, offset: usize, out: &mut [f32]) {
        match self {
            Waveform::SineWave { frequency } => {
                for (i, f) in out.iter_mut().enumerate() {
                    let t_secs = (i + offset) as f32 / sample_frequency as f32;
                    *f = (2.0 * PI * frequency * t_secs).sin();
                }
            }
            Waveform::Amplify { level, waveform } => {
                waveform.generate(sample_frequency, offset, out);
                for x in out.iter_mut() {
                    *x *= level;
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
            Waveform::LinearRamp {
                initial_level,
                duration,
                final_level,
                waveform,
            } => {
                waveform.generate(sample_frequency, offset, out);
                let length = (duration * sample_frequency as f32) as usize;
                let inner_offset = waveform.next_offset(sample_frequency).unwrap_or(0);
                let level_diff = final_level - initial_level;
                let mut offset = offset;
                for x in out.iter_mut() {
                    if offset >= inner_offset && offset < inner_offset + length {
                        let progress: f32 = (offset - inner_offset) as f32 / length as f32;
                        *x *= initial_level + (progress * level_diff);
                    }
                    offset += 1;
                }
            }
            Waveform::Sustain {
                level,
                duration,
                waveform,
            } => {
                waveform.generate(sample_frequency, offset, out);
                let length = (duration * sample_frequency as f32) as usize;
                let inner_offset = waveform.next_offset(sample_frequency).unwrap_or(0);
                for (i, x) in out.iter_mut().enumerate() {
                    if i + offset >= inner_offset && i + offset < inner_offset + length {
                        *x *= level;
                    }
                }
            }
            Waveform::Chord(waveforms) => {
                let n = waveforms.len() as f32;
                for waveform in waveforms {
                    match waveform.samples(sample_frequency) {
                        None => {
                            let mut tmp = vec![0.0; out.len()];
                            waveform.generate(sample_frequency, offset, &mut tmp);
                            for (i, x) in tmp.iter().enumerate() {
                                out[i] += x / n;
                            }
                        }
                        Some(samples) => {
                            if offset < samples {
                                let len = samples - offset;
                                let mut tmp = vec![0.0; out.len().min(len)];
                                waveform.generate(sample_frequency, offset, &mut tmp);
                                for (i, x) in tmp.iter().enumerate() {
                                    out[i] += x / n;
                                }
                            }
                        }
                    }
                }
            }
            Waveform::Sequence(waveforms) => {
                let mut out = out;
                // local_offset is the offset within the current waveform, starting with the first
                let mut local_offset = offset;
                for waveform in waveforms {
                    match waveform.samples(sample_frequency) {
                        None => {
                            waveform.generate(sample_frequency, offset, out);
                        }
                        Some(samples) => {
                            if local_offset < samples {
                                let len = samples - local_offset;
                                // We could optimize the case local_offset + next_offset is greater than
                                // out.len() in which case we don't need to allocate a tmp.
                                let mut tmp = vec![0.0; out.len().min(len)];
                                waveform.generate(sample_frequency, local_offset, &mut tmp);
                                for (i, x) in tmp.iter().enumerate() {
                                    out[i] += x;
                                }
                            }
                        }
                    }
                    match waveform.next_offset(sample_frequency) {
                        None => {
                            return;
                        }
                        Some(next_offset) => {
                            if local_offset + out.len() > next_offset {
                                if local_offset < next_offset {
                                    out = &mut out[(next_offset - local_offset)..];
                                    local_offset = 0;
                                } else {
                                    local_offset -= next_offset;
                                }
                            } else {
                                return;
                            }
                        }
                    }
                }
            }
            Waveform::Const(value) => {
                for x in out.iter_mut() {
                    *x = *value;
                }
            }
        }
    }

    fn samples(&self, sample_frequency: i32) -> Option<usize> {
        match self {
            Waveform::SineWave { .. } => None,
            Waveform::Amplify { waveform, .. } => waveform.samples(sample_frequency),
            Waveform::Fin { duration, .. } => Some((duration * sample_frequency as f32) as usize),
            Waveform::Seq { waveform, .. } => waveform.samples(sample_frequency),
            Waveform::LinearRamp { waveform, .. } => waveform.samples(sample_frequency),
            Waveform::Sustain { waveform, .. } => waveform.samples(sample_frequency),
            Waveform::Chord(waveforms) => {
                let mut samples = Some(0usize);
                for waveform in waveforms {
                    match (samples, waveform.samples(sample_frequency)) {
                        (Some(s), Some(t)) => {
                            samples = Some(s.max(t));
                        }
                        (_, _) => {
                            samples = None;
                        }
                    }
                }
                return samples;
            }
            Waveform::Sequence(waveforms) => {
                let (samples, _) =
                    samples_and_next_offset_for_sequence(sample_frequency, waveforms);
                return samples;
            }
            Waveform::Const(_) => None,
        }
    }

    fn next_offset(&self, sample_frequency: i32) -> Option<usize> {
        match self {
            Waveform::SineWave { .. } => None,
            Waveform::Amplify { waveform, .. } => waveform.next_offset(sample_frequency),
            Waveform::Fin { waveform, .. } => waveform.next_offset(sample_frequency),
            Waveform::Seq { duration, .. } => Some((duration * sample_frequency as f32) as usize),
            Waveform::LinearRamp {
                duration, waveform, ..
            } => Some(
                (duration * sample_frequency as f32) as usize
                    + waveform.next_offset(sample_frequency).unwrap_or(0),
            ),
            Waveform::Sustain {
                duration, waveform, ..
            } => Some(
                (duration * sample_frequency as f32) as usize
                    + waveform.next_offset(sample_frequency).unwrap_or(0),
            ),
            Waveform::Chord(waveforms) => {
                let mut next_offset = Some(0usize);
                for waveform in waveforms {
                    match (next_offset, waveform.next_offset(sample_frequency)) {
                        (Some(s), Some(t)) => {
                            next_offset = Some(s.max(t));
                        }
                        (_, _) => {
                            next_offset = None;
                        }
                    }
                }
                return next_offset;
            }
            Waveform::Sequence(waveforms) => {
                let (_, next_offset) =
                    samples_and_next_offset_for_sequence(sample_frequency, waveforms);
                return next_offset;
            }
            Waveform::Const(_) => None,
        }
    }
}

fn samples_and_next_offset_for_sequence(
    sample_frequency: i32,
    waveforms: &[Waveform],
) -> (Option<usize>, Option<usize>) {
    let mut samples = Some(0usize);
    let mut next_offset = Some(0usize);
    for waveform in waveforms {
        let new_samples = waveform.samples(sample_frequency);
        let new_next_offset = waveform.next_offset(sample_frequency);
        match (samples, new_samples, next_offset) {
            (Some(old_samples), Some(s), Some(old_next_offset)) => {
                samples = Some(old_samples.max(old_next_offset + s));
            }
            (_, None, _) => {
                samples = None;
            }
            (None, _, _) => {} // Samples is already None
            (_, _, None) => {
                panic!("Unexpected case in sequence construction");
            }
        }
        match (next_offset, new_next_offset) {
            (Some(old_next_offset), Some(t)) => {
                next_offset = Some(old_next_offset + t);
            }
            (_, None) => {
                return (samples, None);
            }
            (None, _) => {
                panic!("Unexpected case in sequence construction");
            }
        }
    }
    return (samples, next_offset);
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

    fn finite_const_gen(value: f32, fin_duration: f32, seq_duration: f32) -> Waveform {
        return Waveform::Seq {
            duration: seq_duration,
            waveform: Box::new(Waveform::Fin {
                duration: fin_duration,
                waveform: Box::new(Waveform::Const(value)),
            }),
        };
    }

    fn generate_samples(waveform: &Waveform, size: usize, out: &mut [f32]) {
        for n in 0..out.len() / size {
            waveform.generate(1, n * size, &mut out[n * size..(n + 1) * size]);
        }
    }

    #[test]
    fn test_amplify() {
        let amp_gen = Waveform::Amplify {
            level: 2.0,
            waveform: Box::new(finite_const_gen(1.0, 5.0, 5.0)),
        };
        let desired = vec![2.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            println!("Generating amplify with buffer size {}", size);
            generate_samples(&amp_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_linear_ramp() {
        let ramp_gen = Waveform::LinearRamp {
            initial_level: 0.0,
            duration: 4.0,
            final_level: 0.5,
            waveform: Box::new(finite_const_gen(1.0, 6.0, 0.0)),
        };
        let desired = vec![0.0, 0.125, 0.25, 0.375, 1.0, 1.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            println!("Generating linear ramp with buffer size {}", size);
            generate_samples(&ramp_gen, size, &mut out);
            assert_eq!(out, desired);
        }

        let ramp_gen = Waveform::LinearRamp {
            initial_level: 1.0,
            duration: 4.0,
            final_level: 0.5,
            waveform: Box::new(finite_const_gen(1.0, 5.0, 0.0)),
        };
        let desired = vec![1.0, 0.875, 0.75, 0.625, 1.0, 0.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            println!("Generating linear ramp with buffer size {}", size);
            generate_samples(&ramp_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_sustain() {
        let sustain_gen = Waveform::Sustain {
            level: 0.5,
            duration: 4.0,
            waveform: Box::new(finite_const_gen(1.0, 5.0, 3.0)),
        };
        let desired = vec![1.0, 1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0];
        for size in [1, 2, 4, 8] {
            let mut out = vec![0.0; 8];
            println!("Generating sustain with buffer size {}", size);
            generate_samples(&sustain_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_chord() {
        let chord_gen = Waveform::Chord(vec![
            finite_const_gen(4.0, 5.0, 5.0),
            finite_const_gen(2.0, 10.0, 10.0),
            finite_const_gen(1.0, 15.0, 15.0),
            finite_const_gen(0.5, 8.0, 8.0),
        ]);
        let desired = vec![
            1.875, 1.875, 1.875, 1.875, 1.875, 0.875, 0.875, 0.875, 0.75, 0.75, 0.25, 0.25, 0.25,
            0.25, 0.25, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut out = vec![0.0; 16];
            println!("Generating chord with buffer size {}", size);
            generate_samples(&chord_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }

    #[test]
    fn test_sequence() {
        let seq_gen = Waveform::Sequence(vec![
            finite_const_gen(1.0, 5.0, 1.0),
            finite_const_gen(0.5, 10.0, 8.0),
            finite_const_gen(0.25, 5.0, 5.0),
        ]);
        let desired = vec![
            1.0, 1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.25, 0.25, 0.25, 0.0, 0.0,
        ];
        for size in [1, 2, 4, 8, 16] {
            let mut out = vec![0.0; 16];
            println!("Generating sequence with buffer size {}", size);
            generate_samples(&seq_gen, size, &mut out);
            assert_eq!(out, desired);
        }
    }
}
